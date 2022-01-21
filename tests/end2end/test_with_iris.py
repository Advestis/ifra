from time import sleep

from ifra import CentralServer, Node
from multiprocessing import Process
import os

max_cpus = min(4, os.cpu_count())


def make_node(path_public, path_data):
    node = Node(path_public_configs=path_public, path_data=path_data)
    node.watch(timeout=60)


def test_iris(clean):

    nodes_public_config = [
        "tests/data/node_0/public_configs.json",
        "tests/data/node_1/public_configs.json",
        "tests/data/node_2/public_configs.json",
    ]

    nodes_data_config = [
        "tests/data/node_0/data_configs.json",
        "tests/data/node_1/data_configs.json",
        "tests/data/node_2/data_configs.json",
    ]

    processes = []

    for public, data in zip(nodes_public_config, nodes_data_config):
        processes.append(Process(target=make_node, args=(public, data)))
        processes[-1].start()
    sleep(3)

    central_config_path = "tests/data/central_configs.json"
    server = CentralServer(nodes_configs_paths=nodes_public_config, central_configs_path=central_config_path)
    processes.append(Process(target=server.watch, args=(40,)))
    processes[-1].start()

    success = []
    while len(processes) > 0:
        for proc in processes:
            if proc.exitcode is not None:
                processes.remove(proc)
                success.append(proc.exitcode == 0)
    assert all(success)

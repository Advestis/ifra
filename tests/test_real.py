from ifra import CentralServer, Node
from multiprocessing import Process
import os

max_cpus = min(4, os.cpu_count())


def make_node(path_public, path_data):
    node = Node(path_public_configs=path_public, path_data=path_data)
    node.watch(timeout=30)


def test_iris(clean_real):

    nodes_public_config = [
        "tests/data/real/node_0/public_configs.json",
        "tests/data/real/node_1/public_configs.json",
        "tests/data/real/node_2/public_configs.json",
        "tests/data/real/node_3/public_configs.json",
    ]

    nodes_data_config = [
        "tests/data/real/node_0/path_configs.json",
        "tests/data/real/node_1/path_configs.json",
        "tests/data/real/node_2/path_configs.json",
        "tests/data/real/node_3/path_configs.json",
    ]

    processes = []
    for public, data in zip(nodes_public_config, nodes_data_config):
        processes.append(Process(target=make_node, args=(public, data)))
        processes[-1].start()

    central_config_path = "tests/data/real/central_config.json"
    server = CentralServer(nodes_configs_paths=nodes_public_config, central_configs_path=central_config_path)
    server.watch(timeout=30)

    [p.join() for p in processes]

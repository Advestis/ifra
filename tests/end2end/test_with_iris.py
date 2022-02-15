from time import sleep

from ifra import CentralServer, Node, Aggregator, CentralConfig, NodeDataConfig, NodeLearningConfig, AggregatorConfig
from multiprocessing import Process
import os

max_cpus = min(4, os.cpu_count())

nodes_learning_config = [
    NodeLearningConfig("tests/data/node_0/learning_configs.json"),
    NodeLearningConfig("tests/data/node_1/learning_configs.json"),
    NodeLearningConfig("tests/data/node_2/learning_configs.json"),
]
aggregator_config = AggregatorConfig("tests/data/aggregator_configs.json")
central_config = CentralConfig("tests/data/central_configs.json")


def make_node(path_learning, path_data):
    node = Node(learning_configs=path_learning, data=path_data)
    node.run(timeout=60)


def make_nodes(processes):

    nodes_data_config = [
        NodeDataConfig("tests/data/node_0/data_configs.json"),
        NodeDataConfig("tests/data/node_1/data_configs.json"),
        NodeDataConfig("tests/data/node_2/data_configs.json"),
    ]

    for learning, data in zip(nodes_learning_config, nodes_data_config):
        processes.append(Process(target=make_node, args=(learning, data)))
        processes[-1].start()
    sleep(3)
    print("")
    print("")


def make_server(processes):
    server = CentralServer(central_configs=central_config)
    processes.append(Process(target=server.run, args=(50,)))
    processes[-1].start()
    sleep(2)
    print("")
    print("")


def make_aggregator(processes):
    aggregator = Aggregator(aggregator_configs=aggregator_config)
    processes.append(Process(target=aggregator.run, args=(40,)))
    processes[-1].start()
    sleep(2)
    print("")
    print("")


def test_iris_server_aggr_node(clean):
    processes = []

    make_server(processes)
    make_aggregator(processes)
    make_nodes(processes)

    success = []
    while len(processes) > 0:
        for proc in processes:
            if proc.exitcode is not None:
                processes.remove(proc)
                success.append(proc.exitcode == 0)
    assert all(success)

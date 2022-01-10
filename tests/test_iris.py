from infra import CentralServer, Node


def test_iris_one_iteration():
    nodes = [Node("tests/data/learning.json", f"tests/data/node_{i}/path_configs.json") for i in range(4)]
    cs = CentralServer(nodes=nodes)
    cs.fit(1)

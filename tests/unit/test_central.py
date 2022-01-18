from ifra.central_server import CentralServer, NodeGate
from ifra.configs import Config, CentralConfig, NodePublicConfig
from ifra.messenger import NodeMessenger


def one_node(node):
    assert isinstance(node, NodeGate)
    assert isinstance(node.public_configs, (Config, NodePublicConfig))
    assert isinstance(node.messenger, NodeMessenger)
    assert node.messenger.running is False
    assert node.messenger.fitting is False
    assert node.messenger.stop is False
    assert node.messenger.error is None
    assert node.messenger.central_error is None
    assert node.ruleset is None
    assert node.last_fetch is None
    assert node.new_data is False
    assert node.checked is False
    assert node.checked_once is False
    assert node.ok is True


def test_simple_init_and_watch():
    server = CentralServer(
        nodes_configs_paths=[
            "tests/data/node_0/public_configs.json",
            "tests/data/node_1/public_configs.json",
            "tests/data/node_2/public_configs.json",
            "tests/data/node_3/public_configs.json",
        ],
        central_configs_path="tests/data/central_configs.json",
    )
    assert isinstance(server.central_configs, (Config, CentralConfig))
    assert len(server.nodes) == 4
    for node in server.nodes:
        one_node(node)

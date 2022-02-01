from ifra.aggregator import Aggregator, NodeGate
from ifra.configs import AggregatorConfig, Config, NodePublicConfig


def one_node(node):
    assert isinstance(node, NodeGate)
    assert isinstance(node.public_configs, (Config, NodePublicConfig))
    assert node.ruleset is None
    assert node.last_fetch is None
    assert node.ok is True


def test_simple_init_and_watch():
    aggr = Aggregator(
        nodes_configs=[
            NodePublicConfig("tests/data/node_0/public_configs.json"),
            NodePublicConfig("tests/data/node_1/public_configs.json"),
            NodePublicConfig("tests/data/node_2/public_configs.json"),
        ],
        aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json")
    )
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 3

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    aggr = Aggregator(
        nodes_configs=[
            NodePublicConfig("tests/data/node_0/public_configs.json"),
            NodePublicConfig("tests/data/node_1/public_configs.json"),
            NodePublicConfig("tests/data/node_2/public_configs.json"),
        ],
        aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json")
    )
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 3
    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)

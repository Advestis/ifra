from ifra.aggregator import Aggregator, NodeGate
from ifra.configs import AggregatorConfig, Config


def one_node(node):
    assert isinstance(node, NodeGate)
    assert node.model is None
    assert node.last_fetch is None
    assert node.ok is True


def test_simple_init_and_watch():
    aggr = Aggregator(
        aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json")
    )
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 0  # Empty until aggr.run() is called

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    aggr = Aggregator(
        aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json")
    )
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 0
    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)

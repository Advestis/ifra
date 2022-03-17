from copy import deepcopy

import pytest

from ifra.aggregator import Aggregator, NodeGate
from ifra.configs import AggregatorConfig, Config
from ruleskit import RuleSet, ClassificationRule, HyperrectangleCondition, RegressionRule
from ifra.duplicated_rules_updater import update_duplicated_rules


predictions_float = [1, 1, 4, 2, 2, 2.1, 2.1, 3]
criterions = [1, 1, 2, 1, 1, 2, 3, 2]
predictions_str = ["a", "a", "aa", "b", "b", "bb", "bb", "c"]

classif_ruleset_float = RuleSet(
    [
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

for i in range(len(classif_ruleset_float)):
    # noinspection PyProtectedMember
    classif_ruleset_float[i]._prediction = predictions_float[i]

regress_ruleset = RuleSet(
    [
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])),
    ]
)

for i in range(len(regress_ruleset)):
    # noinspection PyProtectedMember
    regress_ruleset[i]._prediction = predictions_float[i]
    # noinspection PyProtectedMember
    regress_ruleset[i]._criterion = criterions[i]

classif_ruleset_str = RuleSet(
    [
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

for i in range(len(classif_ruleset_str)):
    # noinspection PyProtectedMember
    classif_ruleset_str[i]._prediction = predictions_str[i]

classif_expected_ruleset_float = RuleSet(
    [
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

regress_expected_ruleset = RuleSet(
    [
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])),
    ]
)

regress_expected_ruleset_weighted = RuleSet(
    [
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])),
    ]
)

regress_expected_ruleset_weighted_reversed = RuleSet(
    [
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 2])),
        RegressionRule(condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])),
    ]
)

classif_expected_ruleset_str = RuleSet(
    [
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 1])
        ),
        ClassificationRule(
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

# noinspection PyProtectedMember
classif_expected_ruleset_float[0]._prediction = 1
# noinspection PyProtectedMember
classif_expected_ruleset_float[1]._prediction = 3

# noinspection PyProtectedMember
regress_expected_ruleset[0]._prediction = 2
# noinspection PyProtectedMember
regress_expected_ruleset[1]._prediction = 2.05
# noinspection PyProtectedMember
regress_expected_ruleset[2]._prediction = 3

# noinspection PyProtectedMember
regress_expected_ruleset_weighted[0]._prediction = 2.5
# noinspection PyProtectedMember
regress_expected_ruleset_weighted[1]._prediction = 14.5 / 7.
# noinspection PyProtectedMember
regress_expected_ruleset_weighted[2]._prediction = 3

# noinspection PyProtectedMember
regress_expected_ruleset_weighted_reversed[0]._prediction = 1.75
# noinspection PyProtectedMember
regress_expected_ruleset_weighted_reversed[1]._prediction = 2.042857143
# noinspection PyProtectedMember
regress_expected_ruleset_weighted_reversed[2]._prediction = 3

# noinspection PyProtectedMember
classif_expected_ruleset_str[0]._prediction = "a"
# noinspection PyProtectedMember
classif_expected_ruleset_str[1]._prediction = "c"


def one_node(node):
    assert isinstance(node, NodeGate)
    assert node.model is None
    assert node.last_fetch is None
    assert node.ok is True


def test_simple_init_and_watch():
    aggr = Aggregator(aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json"))
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 0  # Empty until aggr.run() is called

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)

    assert aggr.emitter.doing is None
    assert aggr.emitter.error is None

    aggr = Aggregator(aggregator_configs=AggregatorConfig("tests/data/aggregator_configs.json"))
    assert isinstance(aggr.aggregator_configs, (Config, AggregatorConfig))
    assert len(aggr.nodes) == 0
    for node in aggr.nodes:
        one_node(node)

    aggr.run(timeout=1, sleeptime=0.5)


@pytest.mark.parametrize(
    "ruleset, expected_ruleset, weight, best",
    (
        (classif_ruleset_str, classif_expected_ruleset_str, "equi", "max"),
        (classif_ruleset_float, classif_expected_ruleset_float, "equi", "max"),
        (regress_ruleset, regress_expected_ruleset, "equi", "max"),
        (regress_ruleset, regress_expected_ruleset_weighted, "criterion", "max"),
        (regress_ruleset, regress_expected_ruleset_weighted_reversed, "criterion", "min"),
        (regress_ruleset, regress_expected_ruleset_weighted_reversed, "1 * rule.criterion", "min"),
        (regress_ruleset, regress_expected_ruleset_weighted_reversed, "1 * (rule.criterion)", "min"),
    ),
)
def test_classif_pred_aggr(ruleset, expected_ruleset, weight, best):
    ruleset = deepcopy(ruleset)
    update_duplicated_rules(ruleset, weight, best)

    assert len(ruleset) == len(expected_ruleset)
    for rule, exprule in zip(ruleset, expected_ruleset):
        assert rule == exprule
        assert rule.condition == exprule.condition
        if isinstance(rule.prediction, str):
            assert rule.prediction == exprule.prediction
        else:
            assert round(rule.prediction, 6) == round(exprule.prediction, 6)

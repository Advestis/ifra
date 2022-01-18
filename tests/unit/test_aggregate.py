from ifra.aggregations import aggregate_classif_preds
from ruleskit import RuleSet, ClassificationRule, HyperrectangleCondition
import pytest


predictions_float = [1, 1, 1.1, 2, 2, 2.1, 2.1, 3]
predictions_str = ["a", "a", "aa", "b", "b", "bb", "bb", "c"]

ruleset_float = RuleSet(
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

for i in range(len(ruleset_float)):
    # noinspection PyProtectedMember
    ruleset_float[i]._prediction = predictions_float[i]

ruleset_str = RuleSet(
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

for i in range(len(ruleset_str)):
    # noinspection PyProtectedMember
    ruleset_str[i]._prediction = predictions_str[i]

expected_ruleset_float = RuleSet(
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
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

expected_ruleset_str = RuleSet(
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
            condition=HyperrectangleCondition(features_indexes=[0, 1, 2], bmins=[0, 0, 0], bmaxs=[1, 1, 3])
        ),
    ]
)

# noinspection PyProtectedMember
expected_ruleset_float[0]._prediction = 1
# noinspection PyProtectedMember
expected_ruleset_float[1]._prediction = 1
# noinspection PyProtectedMember
expected_ruleset_float[2]._prediction = 1
# noinspection PyProtectedMember
expected_ruleset_float[3]._prediction = 3
# noinspection PyProtectedMember

expected_ruleset_str[0]._prediction = "a"
# noinspection PyProtectedMember
expected_ruleset_str[1]._prediction = "a"
# noinspection PyProtectedMember
expected_ruleset_str[2]._prediction = "a"
# noinspection PyProtectedMember
expected_ruleset_str[3]._prediction = "c"
# noinspection PyProtectedMember


@pytest.mark.parametrize(
    "ruleset, expected_ruleset", ((ruleset_str, expected_ruleset_str), (ruleset_float, expected_ruleset_float))
)
def test_classif_pred_aggr(ruleset, expected_ruleset):

    aggregate_classif_preds(ruleset)

    assert len(ruleset) == len(expected_ruleset)
    for rule, exprule in zip(ruleset, expected_ruleset):
        assert rule == exprule
        assert rule.condition == exprule.condition
        assert rule.prediction == exprule.prediction

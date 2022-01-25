import numpy as np
import pytest

from ifra.predictor import EquallyWeightedClassificator, CriterionWeightedClassificator, EquallyWeightedRegressor, \
    CriterionWeightedRegressor
from ruleskit import RuleSet, ClassificationRule, HyperrectangleCondition, RegressionRule
import pandas as pd

x_data = pd.DataFrame(
    index=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    columns=["X1", "X2", "X3"],
    data=[
        [0, 20, 100],
        [0, 20, 100],
        [3, 20, 120],
        [5, 10, 100],
        [5, 10, 100],
        [5, 20, 100],
        [3, 20, 150],
        [3, 20, 150],
        [0, 20, 150],
        [0, 20, 150],
    ]
)

classif_rule_x1 = ClassificationRule(
    condition=HyperrectangleCondition(
        features_indexes=[0],
        features_names=["X1"],
        bmins=[0],
        bmaxs=[4]
    )
)
classif_rule_x1._criterion = 0.6

regr_rule_x1 = RegressionRule(
    condition=HyperrectangleCondition(
        features_indexes=[0],
        features_names=["X1"],
        bmins=[0],
        bmaxs=[4]
    )
)
regr_rule_x1._criterion = 0.6

classif_rule_x3 = ClassificationRule(
    condition=HyperrectangleCondition(
        features_indexes=[2],
        features_names=["X3"],
        bmins=[130],
        bmaxs=[160]
    )
)
classif_rule_x3._criterion = 0.9

regr_rule_x3 = RegressionRule(
    condition=HyperrectangleCondition(
        features_indexes=[2],
        features_names=["X3"],
        bmins=[130],
        bmaxs=[160]
    )
)
regr_rule_x3._criterion = 0.9

classif_rule_x1_x3 = ClassificationRule(
    condition=HyperrectangleCondition(
        features_indexes=[0, 2],
        features_names=["X1", "X3"],
        bmins=[2, 110],
        bmaxs=[4, 160]
    )
)
classif_rule_x1_x3._criterion = 0.5

regr_rule_x1_x3 = RegressionRule(
    condition=HyperrectangleCondition(
        features_indexes=[0, 2],
        features_names=["X1", "X3"],
        bmins=[2, 110],
        bmaxs=[4, 160]
    )
)
regr_rule_x1_x3._criterion = 0.5

classif_rule_x1_x2 = ClassificationRule(
    condition=HyperrectangleCondition(
        features_indexes=[0, 2],
        features_names=["X1", "X2"],
        bmins=[0, 10],
        bmaxs=[4, 30]
    )
)
classif_rule_x1_x2._criterion = 0.7

regr_rule_x1_x2 = RegressionRule(
    condition=HyperrectangleCondition(
        features_indexes=[0, 2],
        features_names=["X1", "X2"],
        bmins=[0, 10],
        bmaxs=[4, 30]
    )
)
regr_rule_x1_x2._criterion = 0.7

classif_ruleset = RuleSet([classif_rule_x1, classif_rule_x3, classif_rule_x1_x3, classif_rule_x1_x2])
regr_ruleset = RuleSet([regr_rule_x1, regr_rule_x3, regr_rule_x1_x3, regr_rule_x1_x2])

predictions_int_equally_weighted = pd.DataFrame(
    data=[
        [2, np.nan, np.nan, 1, np.nan],
        [2, np.nan, np.nan, 1, np.nan],
        [2, np.nan, 2, 1, 2],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2, 3, 2, 1, 2],
        [2, 3, 2, 1, 2],
        [2, 3, np.nan, 1, np.nan],
        [2, 3, np.nan, 1, np.nan],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_regr_equally_weighted = pd.DataFrame(
    data=[
        [2, np.nan, np.nan, 1, 1.5],
        [2, np.nan, np.nan, 1, 1.5],
        [2, np.nan, 2, 1, 1.6666666666666],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2, 3, 2, 1, 2],
        [2, 3, 2, 1, 2],
        [2, 3, np.nan, 1, 2],
        [2, 3, np.nan, 1, 2],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_int_crit_weighted = pd.DataFrame(
    data=[
        [2, np.nan, np.nan, 1, 1],
        [2, np.nan, np.nan, 1, 1],
        [2, np.nan, 2, 1, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2, 3, 2, 1, 3],
        [2, 3, 2, 1, 3],
        [2, 3, np.nan, 1, 3],
        [2, 3, np.nan, 1, 3],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_regr_crit_weighted = pd.DataFrame(
    data=[
        [2, np.nan, np.nan, 1, 1.461538426],
        [2, np.nan, np.nan, 1, 1.461538426],
        [2, np.nan, 2, 1, 1.61111111111111],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2, 3, 2, 1, 2.074074074],
        [2, 3, 2, 1, 2.074074074],
        [2, 3, np.nan, 1, 2.090909091],
        [2, 3, np.nan, 1, 2.090909091],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_str_equally_weighted = pd.DataFrame(
    data=[
        ["a", "nan", "nan", "c", "nan"],
        ["a", "nan", "nan", "c", "nan"],
        ["a", "nan", "a", "c", "a"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["a", "b", "a", "c", "a"],
        ["a", "b", "a", "c", "a"],
        ["a", "b", "nan", "c", "nan"],
        ["a", "b", "nan", "c", "nan"],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_str_crit_weighted = pd.DataFrame(
    data=[
        ["a", "nan", "nan", "c", "c"],
        ["a", "nan", "nan", "c", "c"],
        ["a", "nan", "a", "c", "c"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["nan", "nan", "nan", "nan", "nan"],
        ["a", "b", "a", "c", "b"],
        ["a", "b", "a", "c", "b"],
        ["a", "b", "nan", "c", "b"],
        ["a", "b", "nan", "c", "b"],
     ],
    columns=["pred_X1", "pred_X3", "pred_X1_X3", "pred_X1_X2", "pred_ruleset"],
    index=x_data.index
)

predictions_int = [2, 3, 2, 1]
predictions_str = ["a", "b", "a", "c"]
predictions_float = [2.2, 3.3, 2.0, 1.5]


@pytest.mark.parametrize(
    "rules_predictions, preds_expected",
    (
            (predictions_int, predictions_int_equally_weighted),
            (predictions_str, predictions_str_equally_weighted)
    )
)
def test_equally_weighted_classif_predictor(rules_predictions, preds_expected):
    for rule, pred in zip(classif_ruleset, rules_predictions):
        rule._prediction = pred
    predictor = EquallyWeightedClassificator(classif_ruleset)
    predictions = predictor.predict(x_data)
    for ir in range(len(classif_ruleset)):
        rule_pred = classif_ruleset[ir].predict(x_data)
        expected_pred = preds_expected.iloc[:, ir]
        pd.testing.assert_series_equal(rule_pred, expected_pred, check_names=False)
    pd.testing.assert_series_equal(predictions, preds_expected.iloc[:, -1], check_names=False)


@pytest.mark.parametrize(
    "rules_predictions, preds_expected",
    (
            (predictions_int, predictions_int_crit_weighted),
            (predictions_str, predictions_str_crit_weighted)
    )
)
def test_criterion_weighted_classif_predictor(rules_predictions, preds_expected):
    for rule, pred in zip(classif_ruleset, rules_predictions):
        rule._prediction = pred
    predictor = CriterionWeightedClassificator(classif_ruleset)
    predictions = predictor.predict(x_data)
    for ir in range(len(classif_ruleset)):
        rule_pred = classif_ruleset[ir].predict(x_data)
        expected_pred = preds_expected.iloc[:, ir]
        pd.testing.assert_series_equal(rule_pred, expected_pred, check_names=False)
    pd.testing.assert_series_equal(predictions, preds_expected.iloc[:, -1], check_names=False)


@pytest.mark.parametrize(
    "rules_predictions, preds_expected",
    (
            (predictions_int, predictions_regr_equally_weighted),
    )
)
def test_equally_weighted_regr_predictor(rules_predictions, preds_expected):
    for rule, pred in zip(regr_ruleset, rules_predictions):
        rule._prediction = pred
    predictor = EquallyWeightedRegressor(regr_ruleset)
    predictions = predictor.predict(x_data)
    for ir in range(len(regr_ruleset)):
        rule_pred = regr_ruleset[ir].predict(x_data)
        expected_pred = preds_expected.iloc[:, ir]
        pd.testing.assert_series_equal(rule_pred, expected_pred, check_names=False)
    pd.testing.assert_series_equal(predictions, preds_expected.iloc[:, -1], check_names=False)


@pytest.mark.parametrize(
    "rules_predictions, preds_expected",
    (
            (predictions_int, predictions_regr_crit_weighted),
    )
)
def test_criterion_weighted_regr_predictor(rules_predictions, preds_expected):
    for rule, pred in zip(regr_ruleset, rules_predictions):
        rule._prediction = pred
    predictor = CriterionWeightedRegressor(regr_ruleset)
    predictions = predictor.predict(x_data)
    for ir in range(len(regr_ruleset)):
        rule_pred = regr_ruleset[ir].predict(x_data)
        expected_pred = preds_expected.iloc[:, ir]
        pd.testing.assert_series_equal(rule_pred, expected_pred, check_names=False)
    pd.testing.assert_series_equal(predictions, preds_expected.iloc[:, -1], check_names=False)

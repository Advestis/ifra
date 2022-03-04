from typing import Optional

import numpy as np
from ruleskit import RuleSet, ClassificationRule, RegressionRule


def lambda_function(delta_p, delta_v, n, cov):
    return (delta_p ** 2 / delta_v) * np.log((n - 1) * cov / (1 - cov))


# noinspection PyProtectedMember
def apply_diff_privacy_classif(ruleset: RuleSet, y: np.ndarray, c_min: Optional[float] = None):
    pass


# noinspection PyProtectedMember
def apply_diff_privacy_regression(ruleset: RuleSet, y: np.ndarray, c_min: Optional[float] = None):
    n = len(y)

    for r in ruleset:
        if r.prediction is None or np.isnan(r.prediction):
            continue
        c = r.coverage if c_min is None else c_min
        min_pts = int(n * c)
        delta_p = max(
            abs(min(y) - (min(y) * (min_pts - 1) + max(y)) / min_pts),
            abs(max(y) - (max(y) * (min_pts - 1) + min(y)) / min_pts)
        )
        delta_v = max(y) - min(y)
        lambda_value = lambda_function(delta_p=delta_p, delta_v=delta_v, n=n, cov=r.coverage)
        privacy_pred = r.prediction + np.random.laplace(0, lambda_value)
        r._prediction = privacy_pred


# noinspection PyProtectedMember
def apply_diff_privacy(ruleset: RuleSet, y: np.ndarray, c_min: Optional[float] = None):
    """
    Modifies rules predictions to make the learning private.

    The amount the prediction if modified respects :

    P(pred(Y1) in S) <= exp(epsilon * P(pred(Y2) in S))

    with Y1 and Y2 being two sets of targets different of only one point, pred(Y) the prediction of the rule when fitted
    using Y and P(pred is S) the probability of the prediction to be in a set S.

    Parameters
    ----------
    ruleset: RuleSet
    y: np.ndarray
    c_min: Optional[float]


    Returns
    -------

    """
    if len(ruleset) == 0:
        return
    if ruleset._activation is not None and ruleset._activation.length == 0:
        return
    if ruleset.rule_type is None:
        return
    if issubclass(ruleset.rule_type, ClassificationRule):
        apply_diff_privacy_classif(ruleset=ruleset, y=y, c_min=c_min)
    elif issubclass(ruleset.rule_type, RegressionRule):
        apply_diff_privacy_regression(ruleset=ruleset, y=y, c_min=c_min)
    else:
        raise TypeError(f"Invalid rule type {ruleset.rule_type}. Must derive from ClassificationRule or RegressionRule")

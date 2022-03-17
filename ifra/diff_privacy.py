from typing import Optional
import logging
import numpy as np
from ruleskit import RuleSet, ClassificationRule, RegressionRule

logger = logging.getLogger(__name__)


def lambda_function(delta_p, delta_v, n, p):
    return (delta_p ** 2 / delta_v) * np.log((n - 1) * p / (1 - p))


# noinspection PyProtectedMember
def apply_diff_privacy_classif(ruleset: RuleSet, y: np.ndarray, c_min: Optional[float] = None):
    pass


# noinspection PyProtectedMember
def apply_diff_privacy_regression(ruleset: RuleSet, y: np.ndarray, p: float, c_min: Optional[float] = None):
    n = len(y)

    good_rules = []

    for r in ruleset:
        if r.prediction is None or np.isnan(r.prediction):
            continue
        c = r.coverage if c_min is None else c_min
        min_pts = int(n * c)
        delta_pred_min = max(
            abs(min(y) - (min(y) * (min_pts - 1) + max(y)) / min_pts),
            abs(max(y) - (max(y) * (min_pts - 1) + min(y)) / min_pts)
        )
        delta_pred_max = max(y) - min(y)
        delta_activated_min = 1  # max variation of number of activated points when changing one point is... well... 1 !
        delta_activated_max = n
        lambda_value_pred = lambda_function(delta_p=delta_pred_min, delta_v=delta_pred_max, n=n, p=p)
        lambda_value_activated = lambda_function(
            delta_p=delta_activated_min, delta_v=delta_activated_max, n=n, p=r.coverage
        )
        if lambda_value_pred < 0:
            logging.warning("Got a negative lambda : not enough points to ensure privacy. Discarding rule.")
            continue
        if lambda_value_activated < 0:
            logging.warning("got a negative lambda : not enough points to ensure privacy. Discarding rule.")
            continue
        privacy_pred = r.prediction + np.random.laplace(0, lambda_value_pred)
        privacy_set_size = r.train_set_size + int(np.random.laplace(0, lambda_value_activated))
        logger.info(f"Privatised prediction : {r.prediction} -> {privacy_pred}")
        logger.info(f"Privatised train set size : {r.train_set_size} -> {privacy_set_size}")
        r._prediction = privacy_pred
        r._train_set_size = privacy_set_size
        good_rules.append(r)
    ruleset.rules = good_rules


# noinspection PyProtectedMember
def apply_diff_privacy(ruleset: RuleSet, y: np.ndarray, p: float, c_min: Optional[float] = None):
    """
    Modifies rules predictions and train set size to make the learning private up to a probability 'p' of identifying
    their real values.

    Parameters
    ----------
    ruleset: RuleSet
    y: np.ndarray
    c_min: Optional[float]
    p: float

    Returns
    -------
    None
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
        apply_diff_privacy_regression(ruleset=ruleset, y=y, c_min=c_min, p=p)
    else:
        raise TypeError(f"Invalid rule type {ruleset.rule_type}. Must derive from ClassificationRule or RegressionRule")

import numpy as np
import pytest
from ruleskit import RuleSet, ClassificationRule, RegressionRule, HyperrectangleCondition, Activation
from ifra.diff_privacy import apply_diff_privacy

c_min = 0.10

np.random.seed(42)


# @pytest.mark.parametrize(
#     "ruleset, y, exp_preds",
#     [
#         (
#             RuleSet(
#                 [
#                     RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
#                 ],
#             ),
#             np.random.randint(0, 2, 100),
#             [0.09225888254859654]
#         )
#     ]
# )
# def test_privacy(ruleset, y, exp_preds):
#
#     for r, perd in zip(ruleset, exp_preds):
#         act = np.random.uniform(0, 1, 100)
#         act = act > (1 - c_min)
#         r._prediction = np.mean(np.extract(y, act))
#         r._coverage = sum(act) / len(act)
#         r._activation = Activation(act)
#         assert r._prediction == 0.05357142857142857
#
#     apply_diff_privacy(ruleset=ruleset, y=y, c_min=c_min)
#
#     for r, perd in zip(ruleset, exp_preds):
#         assert r._prediction == exp_preds

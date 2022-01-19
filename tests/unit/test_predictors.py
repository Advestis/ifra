# from ifra.predictor import ClassificationPredictor
# from ruleskit import RuleSet, ClassificationRule, HyperrectangleCondition
# import pandas as pd
#
# x_data = pd.DataFrame(
#     index=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#     columns=["X1", "X2", "X3", "X4"],
#     data=[
#         [0, 10, 100, 1000],
#         [1, 10, 100, 1000],
#         [0, 10, 100, 1000],
#         [5, 10, 100, 1000],
#         [6, 10, 100, 1000],
#         [5, 10, 100, 1000],
#         [0, 10, 160, 1000],
#         [0, 10, 150, 1000],
#         [6, 10, 150, 1000],
#         [5, 10, 160, 1000],
#     ]
# )
#
# condition_0 = HyperrectangleCondition(features_names=["X1"], features_indexes=[0], bmins=[0], bmaxs=[4])
# prediction_0 = np.ndarray()

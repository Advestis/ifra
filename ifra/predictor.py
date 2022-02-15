import numpy as np
from ruleskit import RuleSet, ClassificationRule, RegressionRule
import pandas as pd


class Predictor:
    """Class to overload. Implements the notion of model's prediction."""

    def __init__(self, model: RuleSet, **kwargs):
        """

        Parameters
        ----------
        model: RuleSet
        kwargs:
            Any additionnal keyword argument that the overleading class accepts. Those arguments will become attributes.
        """
        self.model = model
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """To be implemented in daughter class"""
        pass


class ClassificationPredictor(Predictor):
    def __init__(self, model: RuleSet):
        if model.rule_type != ClassificationRule:
            raise TypeError("ClassificationPredictor can only be used on classification rules")
        super().__init__(model)


class RegressionPredictor(Predictor):
    def __init__(self, model: RuleSet):
        if model.rule_type != RegressionRule:
            raise TypeError("RegressionPredictor can only be used on regression rules")
        super().__init__(model)


class EquallyWeightedClassificator(ClassificationPredictor):
    """
    Predictor that can be used with classification rules. Assumes that for a given observation, all activated rules
    weight the same, and sends back the most frequent prediction. If two predictions are equally frequent, puts NaN in
    the prediction vector for this observation.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if len(self.model) == 0:
            return pd.Series(index=x.index)
        self.model.calc_activation(x)
        if isinstance(self.model[0].prediction, str):
            most_freq_pred = pd.concat(
                [r.predict(x) for r in self.model], axis=1
            ).fillna(value=np.nan).replace("nan", np.nan).T.mode()
        else:
            most_freq_pred = pd.concat([r.predict(x) for r in self.model], axis=1).T.mode()
        most_freq_pred = most_freq_pred.loc[:, most_freq_pred.count() == 1].dropna().T.squeeze()
        if isinstance(self.model[0].prediction, str):
            return most_freq_pred.reindex(x.index).fillna("nan")
        else:
            return most_freq_pred.reindex(x.index)


class CriterionWeightedClassificator(ClassificationPredictor):
    """
    Predictor that can be used with classification rules. For a given observation, chooses the prediction having the
    most "weight", the weight being the greatest criterion among all rules giving this prediction. If two predictions
    have the same weight,  puts NaN in the prediction vector for this observation.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if len(self.model) == 0:
            return pd.Series(index=x.index)
        self.model.calc_activation(x)
        if isinstance(self.model[0].prediction, str):
            predictions = pd.concat(
                [r.predict(x) for r in self.model], axis=1
            ).fillna(value=np.nan).replace("nan", np.nan)
        else:
            predictions = pd.concat([r.predict(x) for r in self.model], axis=1)

        predictions.columns = pd.RangeIndex(0, len(predictions.columns))
        criterions = pd.DataFrame(
            [[r.criterion for r in self.model] for _ in x.index], index=x.index
        ).fillna(value=np.nan).dropna(how="all")
        if criterions.empty:
            raise ValueError("No rules had evaluated criterion : can not use CriterionWeightedRegressor")
        # Put NaN where prediction is NaN so that not activated rules do not count in the weighting average
        predictions = predictions.reindex(criterions.index)
        criterions = (~predictions.isna() * 1).replace(0, np.nan) * criterions
        # noinspection PyUnresolvedReferences
        mask = (criterions.T == criterions.max(axis=1)).T
        predictions = predictions[mask]
        most_freq_pred = predictions.T.mode()
        most_freq_pred = most_freq_pred.loc[:, most_freq_pred.count() == 1].dropna().T.squeeze()
        if isinstance(self.model[0].prediction, str):
            return most_freq_pred.reindex(x.index).fillna("nan")
        else:
            return most_freq_pred.reindex(x.index)


class EquallyWeightedRegressor(RegressionPredictor):
    """
    Predictor that can be used with regression rules. Assumes that for a given observation, all activated rules
    weight the same, and sends back the average prediction. The returned prediction vector contains NaNs only where
    no rules are activated.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if len(self.model) == 0:
            return pd.Series(index=x.index)
        return pd.concat([r.predict(x) for r in self.model], axis=1).mean(axis=1)


class CriterionWeightedRegressor(RegressionPredictor):
    """
    Predictor that can be used with regression rules. Assumes that for a given observation, rules wit higher criterion
    weight more, and sends back the average prediction. Rules without predictions are discarded, observations where no
    rules bear criterions are discarded. If no criterion is present at all, raises ValueError.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if len(self.model) == 0:
            return pd.Series(index=x.index)
        self.model.calc_activation(x)
        predictions = pd.concat([r.predict(x) for r in self.model], axis=1)
        predictions.columns = pd.RangeIndex(0, len(predictions.columns))
        criterions = pd.DataFrame(
            [[r.criterion for r in self.model] for _ in x.index], index=x.index
        ).fillna(value=np.nan).dropna(how="all")
        if criterions.empty:
            raise ValueError("No rules had evaluated criterion : can not use CriterionWeightedRegressor")
        # Put NaN where prediction is NaN so that not activated rules do not count in the weighting average
        predictions = predictions.reindex(criterions.index)
        criterions = (~predictions.isna() * 1).replace(0, np.nan) * criterions
        return ((predictions * criterions).sum(axis=1) / criterions.sum(axis=1)).reindex(x.index)

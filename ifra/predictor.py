from ruleskit import RuleSet, ClassificationRule, RegressionRule
import pandas as pd


class Predictor:
    """Class to overload. Implements the notion of ruleset's prediction."""

    def __init__(self, ruleset: RuleSet):
        self.ruleset = ruleset

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """To be implemented in daughter class"""
        pass


class ClassificationPredictor(Predictor):
    def __init__(self, ruleset: RuleSet):
        if not isinstance(ruleset.rule_type, ClassificationRule):
            raise TypeError("ClassificationPredictor can only be used on classification rules")
        super().__init__(ruleset)


class RegressionPredictor(Predictor):
    def __init__(self, ruleset: RuleSet):
        if not isinstance(ruleset.rule_type, RegressionRule):
            raise TypeError("RegressionPredictor can only be used on regression rules")
        super().__init__(ruleset)


class EquallyWeightedClassificator(ClassificationPredictor):
    """
    Predictor that can be used with classification rules. Assumes that for a given observation, all activated rules
    weight the same, and sends back the most frequent prediction. If two predictions are equally frequent, puts NaN in
    the prediction vector for this observation.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        self.ruleset.calc_activation(x)
        most_freq_pred = pd.concat([r.predict(x) for r in self.ruleset], axis=1).T.mode()
        most_freq_pred = most_freq_pred.loc[:, most_freq_pred.count() == 1].dropna().T.squeeze()
        return most_freq_pred.reindex(x.index)


class EquallyWeightedRegressor(RegressionPredictor):
    """
    Predictor that can be used with regression rules. Assumes that for a given observation, all activated rules
    weight the same, and sends back the average prediction. The returned prediction vector contains NaNs only where
    no rules are activated.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return pd.concat([r.predict(x) for r in self.ruleset], axis=1).mean()


class CriterionWeightedRegressor(RegressionPredictor):
    """
    Predictor that can be used with regression rules. Assumes that for a given observation, rules wit higher criterion
    weight more, and sends back the average prediction. Rules without predictions are discarded, observations where no
    rules bear criterions are discarded. If no criterion is present at all, raises ValueError.
    """

    def predict(self, x: pd.DataFrame) -> pd.Series:
        self.ruleset.calc_activation(x)
        predictions = pd.concat([r.predict(x) for r in self.ruleset], axis=1)
        criterions = pd.concat([r.criterion for r in self.ruleset], axis=1).dropna(how="all")
        if criterions.empty:
            raise ValueError("No rules had evaluated criterion : can not use CriterionWeightedRegressor")
        predictions = predictions.reindex(criterions.index)
        return (predictions * criterions).sum() / criterions.sum().reindex(x.index)

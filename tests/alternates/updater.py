import pandas as pd
from ruleskit import RuleSet
from typing import Tuple
from ifra.updaters import Updater


class AlternateUpdater(Updater):
    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.DataFrame, ruleset: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ruleset.remember_activation = True
        ruleset.calc_activation(x.values)
        x = x[ruleset.activation == 1]
        y = y[ruleset.activation == 1]
        return x, y

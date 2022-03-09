import pandas as pd
from ruleskit import RuleSet
from typing import Tuple
from ifra.node_model_updaters import NodeModelUpdater


class AlternateUpdater(NodeModelUpdater):
    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.Series, ruleset: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ruleset.calc_activation(x.values)
        if ruleset.activation is not None:
            x = x.loc[ruleset.activation == 1]
            y = y[ruleset.activation == 1]
        return x, y

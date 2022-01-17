from typing import Tuple
import pandas as pd
from ruleskit import RuleSet

from .configs import NodeDataConfig


class Updater:
    """Abstract class implementing how nodes should take central model into account.

    Attributes
    ----------
    data: NodeDataConfig
        `ifra.node.Node` *data*
    """
    def __init__(self, data: NodeDataConfig):
        self.data = data

    def update(self, ruleset: RuleSet) -> None:
        """Reads x and y data, calls `ifra.updaters.make_update` and writes the updated data back to where they were
        read."""
        x = self.data.x.read(**self.data.x_read_kwargs)
        y = self.data.y.read(**self.data.y_read_kwargs)
        self.make_update(x, y, ruleset)
        self.data.x.write(x)
        self.data.y.write(y)

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.DataFrame, ruleset: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implement in daughter class

        Parameters
        ----------
        x: pd.DataFrame
        y: pd.DataFrame
        ruleset: RuleSet

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Modified x and y
        """
        pass


class AdaBoostUpdater(Updater):
    """Ignores points activated by the central server ruleset in order to find other relevant rules in the next
    iterations.

    Can be used by giving *adaboost_updater* as *updater* configuration when creating a `ifra.node.Node`
    """

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.DataFrame, ruleset: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ruleset.remember_activation = True
        ruleset.calc_activation(x.values)
        x = x[ruleset.activation == 0]
        y = y[ruleset.activation == 0]
        return x, y

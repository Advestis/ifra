from typing import Tuple
import pandas as pd
from ruleskit import RuleSet

from .configs import NodeDataConfig


class NodeModelUpdater:
    """Abstract class implementing how nodes should take central model into account.

    Attributes
    ----------
    data: NodeDataConfig
        `ifra.node.Node` *data*
    """
    def __init__(self, data: NodeDataConfig, **kwargs):
        """

        Parameters
        ----------
        data: NodeDataConfig
            `ifra.node.Node`'s *data*
        kwargs:
            Any additionnal keyword argument that the overleading class accepts. Those arguments will become attributes.
        """
        self.data = data
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def update(self, model: RuleSet) -> None:
        """Reads x and y train data, calls `ifra.updaters.make_update` and writes the updated data back to where they
        were read."""
        x = self.data.x_train_path.read(**self.data.x_read_kwargs)
        y = self.data.y_train_path.read(**self.data.y_read_kwargs)
        self.make_update(x, y, model)
        self.data.x_train_path.write(x)
        self.data.y_train_path.write(y)

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.DataFrame, model: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implement in daughter class

        Parameters
        ----------
        x: pd.DataFrame
        y: pd.DataFrame
        model: RuleSet

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Modified x and y
        """
        pass


class AdaBoostNodeModelUpdater(NodeModelUpdater):
    """Ignores points activated by the central model's model in order to find other relevant rules in the next
    iterations.

    Can be used by giving *adaboost* as *updater* configuration when creating a `ifra.node.Node`
    """

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.DataFrame, model: RuleSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        model.remember_activation = True
        model.calc_activation(x.values)
        x = x[model.activation == 0]
        y = y[model.activation == 0]
        return x, y

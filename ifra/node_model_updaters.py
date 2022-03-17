from typing import Tuple
import pandas as pd
from ruleskit import RuleSet

from .configs import NodeDataConfig
from .loader import load_y


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
        y = load_y(self.data.y_train_path, **self.data.y_read_kwargs)
        x, y = self.make_update(x, y, model)
        self.data.x_train_path.write(x)
        self.data.y_train_path.write(y)

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.Series, model: RuleSet) -> Tuple[pd.DataFrame, pd.Series]:
        """Implement in daughter class

        Parameters
        ----------
        x: pd.DataFrame
        y: pd.Series
        model: RuleSet

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Modified x and y
        """
        return x, y


class AdaBoostNodeModelUpdater(NodeModelUpdater):
    """Ignores points activated by the central model's model in order to find other relevant rules in the next
    iterations.

    Can be used by giving *adaboost* as *updater* configuration when creating a `ifra.node.Node`
    """

    @staticmethod
    def make_update(x: pd.DataFrame, y: pd.Series, model: RuleSet) -> Tuple[pd.DataFrame, pd.Series]:
        model.calc_activation(x.values)
        if model.activation is not None:
            x = x.loc[model.activation == 0]
            y = y[model.activation == 0]
        return x, y

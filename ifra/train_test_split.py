from typing import Tuple, Union

import numpy as np
import pandas as pd

from .configs import NodeDataConfig
import logging

logger = logging.getLogger(__name__)


class TrainTestSplit:
    """Abstract class for splitting data into train and test datasets.

     Can be used directly to not split anything by specifying *None* as *train_test_split* configuration when creating a
    `ifra.node.Node`
    """

    def __init__(self, data: NodeDataConfig):
        """
        Parameters
        ----------
        data: NodeDataConfig
            `ifra.node.Node`'s *data*
        """
        self.data = data

    def split(self):
        """Splits x and y into train and test sets by calling `ifra.train_test_split.TrainTestSplit.split_method`.
        Will write the train datasets in ifra.node.Node`'s *data.x_path* and `ifra.node.Node`'s *data.y_path*
        parent directories by appending *_train* to the files names.
        Sets `ifra.node.Node`'s *data.x_train_path* and `ifra.node.Node`'s *data.y_train_path* to point to those files.
        If test datasets are not None, will do the same, but appending *_test* instead of *_train*.
        Else, will set `ifra.node.Node`'s *data.x_test_path* and `ifra.node.Node`'s *data.y_test_path* to
        `ifra.node.Node`'s *data.x_train_path* and `ifra.node.Node`'s *data.y_train_path*.
        """
        if hasattr(self.data, "x_datapreped_path"):
            x_train, x_test, y_train, y_test = self.split_method(
                self.data.x_datapreped_path.read(**self.data.x_read_kwargs),
                self.data.y_datapreped_path.read(**self.data.y_read_kwargs)
            )
        else:
            x_train, x_test, y_train, y_test = self.split_method(
                self.data.x_path.read(**self.data.x_read_kwargs), self.data.y_path.read(**self.data.y_read_kwargs)
            )
        x_suffix = self.data.x_path.suffix
        y_suffix = self.data.y_path.suffix
        x_train_path = self.data.x_path.with_suffix("").append("_train").with_suffix(x_suffix)
        y_train_path = self.data.y_path.with_suffix("").append("_train").with_suffix(y_suffix)

        x_train_path.write(x_train)
        y_train_path.write(y_train)
        self.data.x_train_path = x_train_path
        self.data.y_train_path = y_train_path

        if x_test is not None and y_test is not None:
            x_test_path = self.data.x_path.with_suffix("").append("_test").with_suffix(x_suffix)
            y_test_path = self.data.y_path.with_suffix("").append("_test").with_suffix(y_suffix)
            x_test_path.write(x_test)
            y_test_path.write(y_test)
            self.data.x_test_path = x_test_path
            self.data.y_test_path = y_test_path
        else:
            self.data.x_test_path = x_train_path
            self.data.y_test_path = y_train_path

    # noinspection PyMethodMayBeStatic
    def split_method(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[
        Union[pd.DataFrame, pd.Series, np.ndarray],
        Union[pd.DataFrame, pd.Series, np.ndarray, None],
        Union[pd.DataFrame, pd.Series, np.ndarray],
        Union[pd.DataFrame, pd.Series, np.ndarray, None],
    ]:
        """To be implemented in daughter class. If not, will return x, None, y, None.

        Returns
        -------
        Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], Union[pd.DataFrame, pd.Series, np.ndarray, None],
              Union[pd.DataFrame, pd.Series, np.ndarray], Union[pd.DataFrame, pd.Series, np.ndarray, None]]:
            x_train, x_test, y_train, y_test. Tests dataset can be None.
        """
        logger.info("No splitting required")
        return x, None, y, None

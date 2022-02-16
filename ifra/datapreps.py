from bisect import bisect
from typing import Tuple

import numpy as np
import pandas as pd

from .configs import NodeDataConfig


class DataPrep:
    """Abstract class for DataPrep."""

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

    def dataprep(self):
        """Writes the output of the dataprep in `ifra.node.Node`'s *data.x_path* and `ifra.node.Node`'s *data.y_path*
        parent directories by appending *_datapreped* to the files names. Modifies `ifra.node.Node`'s
        *data.x_datapreped_path* and `ifra.node.Node`'s *data.y_datapreped_path* to point to those files.
        """
        x, y = self.dataprep_method(
            self.data.x_path.read(**self.data.x_read_kwargs),
            self.data.y_path.read(**self.data.y_read_kwargs)
        )
        x_suffix = self.data.x_path.suffix
        y_suffix = self.data.y_path.suffix
        self.data.x_datapreped_path = self.data.x_path.with_suffix("").append("_datapreped").with_suffix(x_suffix)
        self.data.y_datapreped_path = self.data.y_path.with_suffix("").append("_datapreped").with_suffix(y_suffix)

        self.data.x_datapreped_path.write(x)
        self.data.y_datapreped_path.write(y)

    def dataprep_method(self, x: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """To be implemented in daughter class.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Datapreped x and y
        """
        pass


class BinFeaturesDataPrep(DataPrep):
    """Overloads DataPrep class. Bins each feature columns in 'nbins' modalities. Does not modify y.

    Can be used by giving *binfeatures* as *dataprep* configuration when creating a `ifra.node.Node`

    Attributes
    ----------
    nbins: int
        Number of bins to use
    """

    def __init__(self, data: NodeDataConfig, nbins):
        super().__init__(data, nbins=nbins)

    def dataprep_method(self, x, y):

        def find_bins(xx, nbins: int):
            q_list = np.arange(100.0 / nbins, 100.0, 100.0 / nbins)
            bins = np.array([np.nanpercentile(xx, i) for i in q_list])
            return bins

        def get_bins(xx, nb_bucket: int):
            bins = find_bins(xx, nb_bucket)
            while len(set(bins.round(5))) != len(bins):
                nb_bucket -= 1
                bins = find_bins(xx, nb_bucket)
            if len(bins) != nb_bucket - 1:
                raise ValueError(f"Error in get_bins : {len(bins) + 1} bins where found but {nb_bucket} were asked.")
            return bins

        def dicretize(x_series):
            # noinspection PyUnresolvedReferences
            bins = get_bins(x_series, self.nbins)
            mask = np.isnan(x_series)
            discrete_x = x_series.apply(lambda var: bisect(bins, var))
            discrete_x[mask] = np.nan
            return discrete_x

        x = x.apply(lambda xx: dicretize(xx), axis=0)
        return x, y

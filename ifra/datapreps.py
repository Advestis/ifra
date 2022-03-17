from bisect import bisect
from typing import Tuple, Union, List, Optional
from transparentpath import TransparentPath

import numpy as np
import pandas as pd

from .configs import NodeDataConfig
from .loader import load_y
import logging
logger = logging.getLogger(__name__)


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
            load_y(self.data.y_path, **self.data.y_read_kwargs)
        )
        x_suffix = self.data.x_path.suffix
        y_suffix = self.data.y_path.suffix
        self.data.x_datapreped_path = self.data.x_path.with_suffix("").append("_datapreped").with_suffix(x_suffix)
        self.data.y_datapreped_path = self.data.y_path.with_suffix("").append("_datapreped").with_suffix(y_suffix)

        self.data.x_datapreped_path.write(x)
        self.data.y_datapreped_path.write(y)

    def dataprep_method(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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

    def __init__(self, data: NodeDataConfig, nbins: int, bins: dict, save_bins: Optional[TransparentPath]):
        if save_bins is not None:
            save_bins = TransparentPath(save_bins)
        super().__init__(data, nbins=nbins, bins=bins, save_bins=save_bins)

    def dataprep_method(self, x, y):

        # noinspection PyUnresolvedReferences
        def to_apply(xx: pd.Series):
            if xx.name in self.bins:
                bins = self.bins[xx.name]
            else:
                bins = get_bins(xx, self.nbins)
                self.bins[xx.name] = bins

            to_ret = discretize(xx, bins)

            if self.save_bins is not None:
                if not self.save_bins.isfile():
                    self.save_bins.write({xx.name: bins})
                else:
                    allbins = self.save_bins.read()
                    allbins[xx.name] = bins
                    self.save_bins.write(allbins)
            return to_ret

        def find_bins(xx: pd.Series, nbins: int) -> np.ndarray:
            """
            Function used to find the bins to discretize xcol in nbins modalities

            Parameters
            ----------
            xx : pd.Series
               Series to discretize

            nbins: int
                number of modalities

            Return
            ------
            bins: np.ndarray
               the bins for disretization (result from numpy percentile function)
            """
            q_list = np.arange(100.0 / nbins, 100.0, 100.0 / nbins)
            bins = np.array([np.nanpercentile(xx, i) for i in q_list])
            return bins

        def get_bins(xx: pd.Series, nb_bucket: int) -> np.ndarray:
            logger.debug(f"Getting bins for {xx.name}...")
            if nb_bucket == 0:
                raise ValueError("nb_bucket must be greater than 0")
            if nb_bucket == 1:
                return np.array([])
            if len(xx) == 0 or len(xx) == 1:
                return np.array([])
            if len(np.unique(xx)) <= nb_bucket:
                return np.ediff1d(np.unique(xx)) / 2. + np.unique(xx)[:-1]
            bins = find_bins(xx, nb_bucket)
            while len(set(bins.round(5))) != len(bins):
                nb_bucket -= 1
                bins = find_bins(xx, nb_bucket)
            if len(bins) != nb_bucket - 1:
                raise ValueError(f"Error in get_bins : {len(bins) + 1} bins where found but {nb_bucket} were asked.")
            logger.debug(f"... got bins for {xx.name}")
            return bins

        def discretize(xx: pd.Series, bins: Union[np.ndarray, List[float]]) -> pd.Series:
            """
            Transform a Series of float to a Series if int, where each float is replaced by the bin it matches.

            Parameters
            ----------
            xx : pd.Series
                Series to discretize
            bins : Union[np.ndarray, List[float]]
                The list of bins in the form e.g. [0.1, 0.3, ... 0.8] :
                    // -inf ---> 0.1 is bin 0
                    // 0.1 ---> 0.3 is bin 1, etc.
                    // 0.8 ---> +inf is the last bin

            Return
            ------
            discrete_x : pd.Series
                The discretization of x

            """
            logger.debug(f"Discretizing {xx.name}...")
            mask = np.isnan(xx)
            discrete_x = xx.apply(lambda var: bisect(bins, var))
            discrete_x[mask] = np.nan
            logger.debug(f"... discretized {xx.name}")
            return discrete_x

        # noinspection PyUnresolvedReferences
        x = x.apply(lambda xx: to_apply(xx), axis=0)
        return x, y

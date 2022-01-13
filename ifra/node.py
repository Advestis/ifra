from datetime import datetime
from pathlib import Path
from time import time, sleep

from typing import Union

from ruleskit import RuleSet
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, Paths
from .fitters import DecisionTreeFitter
from .plot import plot_histogram

logger = logging.getLogger(__name__)


class Node:
    """One node of federated learning. This class should be used by each machine that is supposed to produce a local
    model.
    """

    possible_fitters = {
        "decisiontree": DecisionTreeFitter
    }

    def __init__(
        self,
        path_learning_configs: TransparentPath,
        path_data: Union[str, Path, TransparentPath],
        fitter: str
    ):

        if fitter not in self.possible_fitters:
            s = f"Fitter '{fitter}' is not known. Can be one of:"
            "\n".join([s] + list(self.possible_fitters.keys()))
            raise ValueError(s)
        self.fitter = fitter

        self.learning_configs = NodePublicConfig(path_learning_configs)
        self.path_learning_configs = self.learning_configs.path  # Will be a TransparentPath

        if not len(list(self.learning_configs.local_model_path.parent.ls(""))) == 0:
            raise ValueError(
                "Path were Node model should be saved is not empty. Make sure you cleaned all previous learning output"
            )

        self.__data = Paths(path_data)
        self.__fitter = self.possible_fitters[self.fitter](self.learning_configs, self.__data)

        self.dataprep_method = None
        self.datapreped = False
        self.copied = False
        self.ruleset = None
        self.last_fetch = None
        if self.learning_configs.dataprep_method is not None:
            function = self.learning_configs.dataprep_method.split(".")[-1]
            module = self.learning_configs.dataprep_method.replace(f".{function}", "")
            self.dataprep_method = __import__(module, globals(), locals(), [function], 0)

    def fit(self) -> RuleSet:
        if self.learning_configs.plot_data:
            self.plot_data_histogram(self.__data.x.parent / "plots")
        if self.dataprep_method is not None and not self.datapreped:
            logger.info(f"Datapreping data of node {self.learning_configs.id}...")
            x, y = self.dataprep_method(
                self.__data.x.read(**self.__data.x_read_kwargs),
                self.__data.y.read(**self.__data.y_read_kwargs)
            )
            x_suffix = self.__data.x.suffix
            y_suffix = self.__data.y.suffix
            x_datapreped_path = self.__data.x.with_suffix("").append("_datapreped").with_suffix(x_suffix)
            y_datapreped_path = self.__data.y.with_suffix("").append("_datapreped").with_suffix(y_suffix)

            x_datapreped_path.write(x)
            y_datapreped_path.write(y)

            self.__fitter.data.x = x_datapreped_path
            self.__fitter.data.y = y_datapreped_path
            if self.learning_configs.plot_data:
                self.plot_data_histogram(self.__data.x.parent / "plots_datapreped")
            self.datapreped = True
            logger.info(f"...datapreping done for node {self.learning_configs.id}")

        if not self.copied:
            # Copy x and y in a different file, for it will be changed after each iterations by the update from central
            x_suffix = self.__data.x.suffix
            self.__data.x.cp(self.__data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix))
            self.__data.x = self.__data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix)
            y_suffix = self.__data.y.suffix
            self.__data.y.cp(self.__data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix))
            self.__data.y = self.__data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix)
            self.copied = True

        logger.info(f"Fitting node {self.learning_configs.id} using {self.fitter}...")
        self.ruleset = self.__fitter.fit()
        self.ruleset_to_file()
        logger.info(f"... node {self.learning_configs.id} fitted, results saved in"
                    f" {self.learning_configs.local_model_path.parent}.")
        return self.ruleset

    def update_from_central(self, ruleset):
        """Ignore points activated by the central server ruleset in order to find other relevant rules in the next
        iterations"""
        logger.info(f"Updating node {self.learning_configs.id}...")
        # Compute activation of the selected rules
        x = self.__data.x.read(**self.__data.x_read_kwargs)
        y = self.__data.y.read(**self.__data.y_read_kwargs)
        ruleset.remember_activation = True
        ruleset.calc_activation(x.values)
        x = x[ruleset.activation == 0]
        y = y[ruleset.activation == 0]
        self.__data.x.write(x)
        self.__data.y.write(y)
        logger.info(f"... node {self.learning_configs.id} updated.")

    def ruleset_to_file(
        self
    ):
        """Saves self.ruleset to 2 .csv files. Does not do anything if self.ruleset is None
        """

        ruleset = self.ruleset

        iteration = 0
        name = self.learning_configs.local_model_path.stem
        path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.csv"
        while path.isfile():
            iteration += 1
            path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.learning_configs.local_model_path)

    def plot_data_histogram(self, path):
        x = self.__data.x.read(**self.__data.x_read_kwargs)
        for col, name in zip(x.columns, self.learning_configs.features_names):
            fig = plot_histogram(
                data=x[col],
                xlabel=name,
                figsize=(10, 7)
            )

            iteration = 0
            name = name.replace(' ', '_')
            path_x = path.parent / f"{name}_{iteration}.pdf"
            while path_x.isfile():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.pdf"
            fig.savefig(path_x)

        y = self.__data.y.read(**self.__data.y_read_kwargs)
        fig = plot_histogram(
            data=y.squeeze(),
            xlabel="Class",
            figsize=(10, 7)
        )

        iteration = 0
        path_y = path.parent / f"classes_{iteration}.pdf"
        while path_y.isfile():
            iteration += 1
            path_y = path_y.parent / f"classes_{iteration}.pdf"

        fig.savefig(path_y)

    def watch(self, timeout: int = 60, sleeptime: int = 5):

        def get_ruleset():
            central_ruleset = RuleSet()
            central_ruleset.load(self.learning_configs.central_model_path)
            self.last_fetch = datetime.now()
            self.update_from_central(central_ruleset)
            logger.info(f"Fetched new central ruleset in node {self.learning_configs.id} at {self.last_fetch}")
            self.new_data = True

        t = time()
        new_central_model = True  # start at true to trigger fit even if no central model is here at first iteration
        while time() - t < timeout:
            if self.learning_configs.id is None:
                self.learning_configs = NodePublicConfig(self.path_learning_configs)
            if self.learning_configs.id is None:
                logger.warning(
                    "Node id is not set, meaning central server is not running. Waiting for central server to start."
                )
                continue

            if self.last_fetch is None:
                if self.learning_configs.central_model_path.isfile():
                    get_ruleset()
                    new_central_model = True
            else:
                if (
                    self.learning_configs.central_model_path.isfile()
                    and self.learning_configs.central_model_path.info()["mtime"] > self.last_fetch.timestamp()
                ):
                    get_ruleset()
                    new_central_model = True

            if new_central_model:
                self.fit()
                new_central_model = False
            sleep(sleeptime)

        logger.info(f"Timeout of {timeout} seconds reached, stopping learning in node {self.learning_configs.id}.")

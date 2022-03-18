import random
from datetime import datetime
from time import time, sleep

from typing import Union

import numpy as np
import pandas as pd
from ruleskit import RuleSet, RegressionRule, ClassificationRule
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .actor import Actor
from .configs import NodeLearningConfig, NodeDataConfig
from .diff_privacy import apply_diff_privacy
from .fitters import DecisionTreeClassificationFitter, Fitter, DecisionTreeRegressionFitter
from .loader import load_y
from .node_model_updaters import AdaBoostNodeModelUpdater, NodeModelUpdater
from .datapreps import BinFeaturesDataPrep, DataPrep
from .train_test_split import TrainTestSplit
from .plot import plot_histogram
from .decorator import emit

logger = logging.getLogger(__name__)


# noinspection PyAttributeOutsideInit
class Node(Actor):
    # noinspection PyUnresolvedReferences
    """One node of federated learning.
    This class should be used by each machine that is supposed to produce a local
    model. It monitors changes in a directory where the central server is supposed to push its model, and triggers fit
    when a new central model is available. It will also trigger a first fit before any central model is available. It
    also monitors changes in the input data file, to automatically re-fit the model when new data are available.
    Any rules produced by the node and already present in the central model are ignored.

    Attributes
    ----------
    learning_configs: `ifra.configs.NodeLearningConfig`
        The learning configuration of the node. Will be accessible by the central server.
        see `ifra.configs.NodeLearningConfig`
    path_learning_configs: TransparentPath
        Path to the json file containing learning configuration of the node. Needs to be kept in memory to potentially
        update the node's id once set by the central server.
    emitter: Emitter
        `ifra.messenger.Emitter`
    dataprep: Union[None, Callable]
        The dataprep method to apply to the data before fitting. Optional, specified in the node's learning
        configuration.
    datapreped: bool
        False if the dataprep has not been done yet, True after
    splitted: bool
        This flag is True if the features and targets have been splitted into train/test yet, False otherwise.
    model: Union[None, RuleSet]
        Fitted model
    last_fetch: Union[None, datetime]
        Date and time when the central model was last fetched.
    last_x: Union[None, datetime]
        Date and time when the x data were last modified.
    last_y: Union[None, datetime]
        Date and time when the y data were last modified.
    data: `ifra.configs.NodeDataConfig`
        Configuration of the paths to the node's features and target data files
    fitter: `ifra.fitters.Fitter`
        Configuration of the paths to the node's features and target data files
    updater: `ifra.node_model_updaters.NodeModelUpdater`
        Configuration of the paths to the node's features and target data files
    """

    possible_fitters = {
        "decisiontreeclassification": DecisionTreeClassificationFitter,
        "decisiontreeregression": DecisionTreeRegressionFitter,
    }
    """Possible string values and corresponding fitter object for *fitter* attribute of
    `ifra.configs.NodeLearningConfig`"""

    possible_updaters = {"adaboost": AdaBoostNodeModelUpdater}
    """Possible string values and corresponding updaters object for *updater* attribute of
    `ifra.configs.NodeLearningConfig`"""

    possible_datapreps = {"binfeatures": BinFeaturesDataPrep}
    """Possible string values and corresponding updaters object for *updater* attribute of
    `ifra.configs.NodeLearningConfig`"""

    possible_splitters = {}
    """Possible string values and corresponding splitters object for *train_test_split* attribute of
    `ifra.configs.NodeLearningConfig`"""

    timeout_central = 600
    """How long the node is supposed to wait for the central server to start up"""

    def __init__(
        self,
        learning_configs: NodeLearningConfig,
        data: NodeDataConfig,
    ):
        self.learning_configs = None
        self.data = None
        self.last_x = None
        self.last_y = None
        self.filenumber = None
        RegressionRule.rule_index += ["train_test_size"]
        ClassificationRule.rule_index += ["train_test_size"]
        super().__init__(learning_configs=learning_configs, data=data)

    @emit
    def create(self):

        self.datapreped = False
        self.copied = False
        self.splitted = False
        self.model = None
        self.last_fetch = None
        if not self.learning_configs.node_models_path.is_dir():
            self.learning_configs.node_models_path.mkdir()

        self.data.dataprep_kwargs = self.learning_configs.dataprep_kwargs

        """Set fitter"""

        if self.learning_configs.fitter not in self.possible_fitters:
            function = self.learning_configs.fitter.split(".")[-1]
            module = self.learning_configs.fitter.replace(f".{function}", "")
            self.fitter = getattr(__import__(module, globals(), locals(), [function], 0), function)(
                self.learning_configs
            )
            if not isinstance(self.fitter, Fitter):
                raise TypeError("Node fitter should inherite from Fitter class")
        else:
            self.fitter = self.possible_fitters[self.learning_configs.fitter](self.learning_configs)

        """Set splitter"""

        if self.learning_configs.train_test_split is not None:
            if self.learning_configs.train_test_split not in self.possible_splitters:
                function = self.learning_configs.train_test_split.split(".")[-1]
                module = self.learning_configs.train_test_split.replace(f".{function}", "")
                self.train_test_split = getattr(__import__(module, globals(), locals(), [function], 0), function)(
                    self.learning_configs
                )
                if not isinstance(self.train_test_split, TrainTestSplit):
                    raise TypeError("Node splitter should inherite from TrainTestSplit class")
            else:
                self.train_test_split = self.possible_splitters[self.learning_configs.train_test_split](self.data)
        else:
            self.train_test_split = TrainTestSplit(self.data)

        """Set updater"""

        if self.learning_configs.updater not in self.possible_updaters:
            function = self.learning_configs.updater.split(".")[-1]
            module = self.learning_configs.updater.replace(f".{function}", "")
            self.updater = getattr(__import__(module, globals(), locals(), [function], 0), function)(self.data)
            if not isinstance(self.updater, NodeModelUpdater):
                raise TypeError("Node updater should inherite from NodeModelUpdater class")
        else:
            self.updater = self.possible_updaters[self.learning_configs.updater](self.data)

        """Set dataprep"""

        if self.learning_configs.dataprep is not None:
            if self.learning_configs.dataprep not in self.possible_datapreps:
                function = self.learning_configs.dataprep.split(".")[-1]
                module = self.learning_configs.dataprep.replace(f".{function}", "")
                self.dataprep = getattr(__import__(module, globals(), locals(), [function], 0), function)(self.data)
                if not isinstance(self.dataprep, DataPrep):
                    raise TypeError("Node dataprep should inherite from DataPrep class")
            else:
                self.dataprep = self.possible_datapreps[self.learning_configs.dataprep](
                    self.data, **self.learning_configs.dataprep_kwargs
                )
        else:
            self.dataprep = None

    @emit
    def plot_dataprep_and_split(self) -> None:
        self._plot(self.data.x_path)
        self._dataprep()
        self._split()

    @emit
    def _plot(self, which_x, name="plots"):
        """Plots the features and classes distribution in `ifra.node.Node`'s *data.x_path* and
        `ifra.node.Node`'s *data.y_path* parent directories if `ifra.configs.NodeLearningConfig` *plot_data*
        is True."""
        if self.learning_configs.plot_data:
            if not (which_x.parent / name).is_dir():
                (which_x.parent / name).mkdir()
            self.plot_data_histogram(which_x, which_x.parent / name)

    @emit
    def _dataprep(self):
        """
        Triggers `ifra.node.Node`'s *dataprep* on the features and targets if a dataprep method was
        specified, sets `ifra.node.Node` datapreped to True, plots the distributions of the datapreped data.
        """
        self.last_x = self.last_y = datetime.now()

        if self.dataprep is not None and not self.datapreped:
            logger.info(f"{self.learning_configs.id} - Datapreping...")
            self.dataprep.dataprep()
            self.datapreped = True
            self._plot(self.data.x_path, "plots_datapreped")
            logger.info(f"{self.learning_configs.id} - ...datapreping done")

    @emit
    def _split(self):
        """
        Triggers `ifra.node.Node`'s *train_test_split* on the features and targets,
        sets `ifra.node.Node` splitted to True, plots the distributions of the splitted data.
        """
        if not self.splitted:
            logger.info(f"{self.learning_configs.id} - Splitting data in train/test...")
            self.train_test_split.split(self.iterations)
            self.splitted = True
            self._plot(self.data.x_train_path, f"plots_train_{self.iterations}")
            if self.data.x_train_path != self.data.x_test_path:
                self._plot(self.data.x_test_path, "plots_test")
            logger.info(f"{self.learning_configs.id} - ...splitting done")

    @emit
    def fit(self) -> None:
        """
        Calls `ifra.node.Node.plot_dataprep_and_copy`.
        Calls the fitter corresponding to `ifra.node.Node` *learning_configs.fitter* on the node's features and
        targets.
        Filters out rules that are already present in the central model, if it exists.
        Saves the resulting model in `ifra.node.Node` *learning_configs.node_models_path* if new rules were found.
        """

        self.plot_dataprep_and_split()

        x = self.data.x_train_path.read(**self.data.x_read_kwargs).values
        y = load_y(self.data.y_train_path, **self.data.y_read_kwargs).values
        if self.data.x_test_path != self.data.x_train_path:
            x_test = self.data.x_test_path.read(**self.data.x_read_kwargs).values
        else:
            x_test = None
        if self.data.y_test_path != self.data.y_train_path:
            y_test = load_y(self.data.y_test_path, **self.data.y_read_kwargs).values
        else:
            y_test = y

        logger.info(f"{self.learning_configs.id} - Fitting with {self.learning_configs.fitter}...")
        if len(x) == 0:
            logger.warning(f"{self.learning_configs.id} - No data in node's features")
            return
        self.model = self.fitter.fit(x=x, y=y)
        if self.model is None:
            logger.warning(f"{self.learning_configs.id} - Fitter could not produce a model")
            return
        central_model_path = self.learning_configs.central_model_path
        if central_model_path.isfile():
            central_model = RuleSet()
            central_model.load(central_model_path)
            rules = [r for r in self.model if r not in central_model]
            self.model = RuleSet(rules)

        if len(self.model) > 0:
            self.coverage = self.model.ruleset_coverage
            self.model.eval(
                xs=x_test, y=y_test, keep_new_activations=x_test is not None, **self.learning_configs.eval_kwargs
            )
            # noinspection PyProtectedMember
            if self.model.criterion is None:
                self.model.criterion = np.nan
            logger.info(
                f"{self.learning_configs.id} - Found {len(self.model)} new good rules."
                f" Criterion is {round(self.model.criterion, 3)}, coverage is {round(self.coverage, 3)}"
            )
            self.model_to_file()
            self.fitter.save(self.learning_configs.node_models_path / f"model_{self.filenumber}_{self.iterations}")
        else:
            logger.info(f"{self.learning_configs.id} - Did not find rules")
        if self.model is not None and self.learning_configs.privacy_proba is not None:
            apply_diff_privacy(
                ruleset=self.model, y=y, p=self.learning_configs.privacy_proba, name=self.learning_configs.id
            )
            if len(self.model) == 0:
                logger.warning(
                    f"{self.learning_configs.id} - No good rule left in model after applying differential privacy"
                )
                self.model = None

    @emit
    def update_from_central(self, model: RuleSet) -> None:
        """Modifies the files pointed by `ifra.node.Node`'s *data.x_path* and `ifra.node.Node`'s *data.y_path* by
        calling `ifra.node.Node` *learning_configs.updater*.

        Parameters
        ----------
        model: RuleSet
            The central model
        """
        logger.info(f"{self.learning_configs.id} - Updating node...")
        # Compute activation of the selected rules
        self.updater.update(model)
        logger.info(f"{self.learning_configs.id} - ... node updated.")

    @emit
    def model_to_file(self) -> None:
        """Saves `ifra.node.Node` *model* to `ifra.configs.NodeLearningConfig` *node_models_path*,
        under 'model_nnode_niteration.csv' and 'model_main_nnode', where 'nnode' is determined by counting how many
        files named 'model_*_0.csv' already exist, and where 'niteration' is incremented at each call of this method,
        starting at 0.
        Will also produce a .pdf of the model using TableWriter if LaTeX is available on the system.
        Does not do anything if `ifra.node.Node` *model* is None
        """

        model = self.model
        if model is None:
            return

        if not self.learning_configs.node_models_path.isdir():
            self.learning_configs.node_models_path.mkdir(parents=True)
        existing_files = list(self.learning_configs.node_models_path.glob("model_main_*.csv"))
        if self.filenumber is None:
            self.filenumber = random.randint(0, int(1e6))
            path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"
            while path_main in existing_files:
                self.filenumber += random.randint(int(2e6), int(3e6))
                path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"
            path_main.touch()  # To lock file for possible other nodes trying to write here
        else:
            path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"

        path_iteration = self.learning_configs.node_models_path / f"model_{self.filenumber}_{self.iterations}.csv"
        self.iterations += 1

        if not path_main.parent.isdir():
            path_main.parent.mkdir(parents=True)
        model.save(path_main)
        model.save(path_iteration)

        logger.info(f"{self.learning_configs.id} - Node's model saved in {path_iteration} and {path_main}.")

        try:
            path_table = path_main.with_suffix(".pdf")
            df = path_main.read(index_col=0)
            if not df.empty:
                df = df.apply(lambda x: x.round(3) if x.dtype == float else x)
            else:
                df = pd.DataFrame(data=[["empty"]])
            TableWriter(path_table, df, paperwidth=30).compile(clean_tex=True)
        except ValueError:
            logger.warning(f"{self.learning_configs.id} - Failed to produce tablewriter. Is LaTeX installed ?")

    @emit
    def plot_data_histogram(self, x_path: TransparentPath, output_path: TransparentPath) -> None:
        """Plots the distribution of the data located in `ifra.node.Node`'s *data.x_path* and
        `ifra.node.Node`'s *data.y_path* and save them in unique files.

        Parameters
        ----------
        x_path: TransparentPath
            Path of the x data to plot. Must be a file.
        output_path: TransparentPath
            Path to save the data. Should be a directory.
        """
        x = x_path.read(**self.data.x_read_kwargs)
        for col, name in zip(x.columns, self.learning_configs.features_names):
            fig = plot_histogram(data=x[col], xlabel=name, figsize=(10, 7))

            iteration = 0
            name = name.replace(" ", "_")
            path_x = output_path / f"{name}_{iteration}.pdf"
            while path_x.is_file():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.pdf"
            fig.savefig(path_x)

        y = load_y(self.data.y_path, **self.data.y_read_kwargs)
        fig = plot_histogram(data=y.squeeze(), xlabel="Class", figsize=(10, 7))

        iteration = 0
        path_y = output_path / f"classes_{iteration}.pdf"
        while path_y.is_file():
            iteration += 1
            path_y = path_y.parent / f"classes_{iteration}.pdf"

        fig.savefig(path_y)

    @emit
    def run(self, timeout: Union[int, float] = 0, sleeptime: Union[int, float] = 5):
        """Monitors new changes in the central server, every *sleeptime* seconds for *timeout* seconds, triggering
        node fit when a new model is found, if new data are available or if the function just started. Sets
        `ifra.configs.NodeLearningConfig` *id* and *central_server_path* by re-reading the configuration file.

        If central model is not present, waits until it is or until new data are available.
        This is the only method the user should call.

        Parameters
        ----------
        timeout: Union[int, float]
            How many seconds the run last. If <= 0>, will last until killed by the user. Default value = 0
        sleeptime: Union[int, float]
            How many seconds between each checks for new central model. Default value = 5
        """
        # noinspection PyUnusedLocal
        @emit  # Let self_ here for proper use of 'emit'
        def get_model(self_) -> None:
            """Fetch the central server's latest model.
            `ifra.node.Node.last_fetch` will be set to now."""
            central_model = RuleSet()
            central_model.load(self.learning_configs.central_model_path)
            self.last_fetch = datetime.now()
            self.update_from_central(central_model)
            logger.info(f"{self.learning_configs.id} - Fetched central model at {self.last_fetch}")

        # start at true to trigger fit even if no central model is here at first iteration
        do_fit = self.iterations == 0
        self.plot_dataprep_and_split()  # make dataprep at run start
        started = False  # To force at least one loop of the while to trigger

        if timeout <= 0:
            logger.warning(
                f"{self.learning_configs.id} -"
                " You did not specify a timeout for your run. It will last until manually stopped."
            )

        logger.info(f"{self.learning_configs.id} - Starting run")
        logger.info(
            f"Starting node {self.learning_configs.id}. Monitoring changes in "
            f"{self.learning_configs.central_model_path}, {self.data.x_path} and {self.data.y_path}."
        )

        t = time()
        while time() - t < timeout or timeout <= 0 or started is False:
            started = True
            if (
                self.last_x is not None
                and self.last_y is not None
                and (
                    self.data.x_path.info()["mtime"] > self.last_x.timestamp()
                    or self.data.y_path.info()["mtime"] > self.last_y.timestamp()
                )
            ):
                # New data arrived for the node to use : redo dataprep and copy and force update from the central model
                logger.info(f"{self.learning_configs.id} - New data available")
                self.datapreped = False
                self.splitted = False
                self.last_fetch = None
                self.plot_dataprep_and_split()

            self.learning_configs = NodeLearningConfig(self.learning_configs.path)  # In case NodeGate changed something

            if self.last_fetch is None:
                if self.learning_configs.central_model_path.is_file():
                    get_model(self)
                    do_fit = True
            else:
                if (
                    self.learning_configs.central_model_path.is_file()
                    and self.learning_configs.central_model_path.info()["mtime"] > self.last_fetch.timestamp()
                ):
                    get_model(self)
                    do_fit = True

            if do_fit:
                self.fit()
                do_fit = False
            sleep(sleeptime)

        logger.info(f"{self.learning_configs.id} - Timeout of {timeout} seconds reached, stopping learning.")

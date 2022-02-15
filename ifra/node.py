import random
from datetime import datetime
from time import time, sleep

from typing import Union

import pandas as pd
from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .actor import Actor
from .configs import NodeLearningConfig, NodeDataConfig
from .fitters import DecisionTreeFitter, Fitter
from .node_model_updaters import AdaBoostNodeModelUpdater, NodeModelUpdater
from .datapreps import BinFeaturesDataPrep, DataPrep
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
    copied: bool
        In order not to alter the data given to the node, they are copied in new files used for learning. This flag is
        False if this copy has not been done yet, False otherwise.
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

    possible_fitters = {"decisiontree": DecisionTreeFitter}
    """Possible string values and corresponding fitter object for *fitter* attribute of
    `ifra.configs.NodeLearningConfig`"""

    possible_updaters = {"adaboost": AdaBoostNodeModelUpdater}
    """Possible string values and corresponding updaters object for *updater* attribute of
    `ifra.configs.NodeLearningConfig`"""

    possible_datapreps = {"binfeatures": BinFeaturesDataPrep}
    """Possible string values and corresponding updaters object for *updater* attribute of
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
        self.iteration = 0
        super().__init__(learning_configs=learning_configs, data=data)

    @emit
    def create(self):

        self.datapreped = False
        self.copied = False
        self.model = None
        self.last_fetch = None
        if not self.learning_configs.node_models_path.is_dir():
            self.learning_configs.node_models_path.mkdir()

        self.data.dataprep_kwargs = self.learning_configs.dataprep_kwargs

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

        if self.learning_configs.updater not in self.possible_updaters:
            function = self.learning_configs.updater.split(".")[-1]
            module = self.learning_configs.updater.replace(f".{function}", "")
            self.updater = getattr(__import__(module, globals(), locals(), [function], 0), function)(self.data)
            if not isinstance(self.updater, NodeModelUpdater):
                raise TypeError("Node updater should inherite from NodeModelUpdater class")
        else:
            self.updater = self.possible_updaters[self.learning_configs.updater](self.data)

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
    def plot_dataprep_and_copy(self) -> None:
        """Plots the features and classes distribution in `ifra.node.Node`'s *data.x_path* and
        `ifra.node.Node`'s *data.y_path* parent directories if `ifra.configs.NodeLearningConfig` *plot_data*
        is True.

        Triggers `ifra.node.Node`'s *dataprep* on the features and classes if a dataprep method was
        specified, sets `ifra.node.Node` datapreped to True, plots the distributions of the datapreped data.

        If `ifra.node.Node` *copied* is False, copies the files pointed by `ifra.node.Node`'s *data.x_path*
        and `ifra.node.Node`'s *data.y_path* in new files in the same directories by appending
        *_to_use* to their names. Sets `ifra.node.Node` *copied* to True.
        """

        if self.learning_configs.plot_data:
            if not (self.data.x_path.parent / "plots").is_dir():
                (self.data.x_path.parent / "plots").mkdir()
            self.plot_data_histogram(self.data.x_path.parent / "plots")

        self.last_x = self.last_y = datetime.now()

        if self.dataprep is not None and not self.datapreped:
            logger.info(f"Datapreping data of node {self.learning_configs.id}...")
            self.dataprep.dataprep()
            if self.learning_configs.plot_data:
                if not (self.data.x_path.parent / "plots_datapreped").is_dir():
                    (self.data.x_path.parent / "plots_datapreped").mkdir()
                self.plot_data_histogram(self.data.x_path.parent / "plots_datapreped")
            self.datapreped = True
            self.copied = True
            logger.info(f"...datapreping done for node {self.learning_configs.id}")

        if not self.copied:
            # Copy x and y in a different file, for it will be changed after each iterations by the update from
            # central
            x_suffix = self.data.x_path.suffix
            self.data.x_path_to_use.cp(self.data.x_path.with_suffix("").append("_to_use").with_suffix(x_suffix))
            self.data.x_path_to_use = self.data.x_path.with_suffix("").append("_to_use").with_suffix(x_suffix)
            y_suffix = self.data.y_path.suffix
            self.data.y_path_to_use.cp(self.data.y_path.with_suffix("").append("_to_use").with_suffix(y_suffix))
            self.data.y_path_to_use = self.data.y_path.with_suffix("").append("_to_use").with_suffix(y_suffix)
            self.copied = True

    @emit
    def fit(self) -> None:
        """
        Calls `ifra.node.Node.plot_dataprep_and_copy`.
        Calls the fitter corresponding to `ifra.node.Node` *learning_configs.fitter* on the node's features and
        targets.
        Filters out rules that are already present in the central model, if it exists.
        Saves the resulting model in `ifra.node.Node` *learning_configs.node_models_path* if new rules were found.
        """

        self.plot_dataprep_and_copy()
        logger.info(f"Fitting node {self.learning_configs.id} using {self.learning_configs.fitter}...")
        model = self.fitter.fit(
            self.data.x_path_to_use.read(**self.data.x_read_kwargs).values,
            self.data.y_path_to_use.read(**self.data.y_read_kwargs).values,
        )
        good_rules = [r for r in model if r.good]
        central_model_path = self.learning_configs.central_model_path
        if central_model_path.isfile():
            central_model = RuleSet()
            central_model.load(central_model_path)
            rules = [r for r in good_rules if r not in central_model]
            if len(rules) > 0:
                self.model = RuleSet(
                    remember_activation=model.remember_activation,
                    stack_activation=model.stack_activation,
                    rules_list=rules
                )
                logger.info(f"Found {len(self.model)} new good rules in node {self.learning_configs.id}")
                self.model_to_file()
                self.fitter.save(
                    self.learning_configs.node_models_path / f"model_{self.filenumber}_{self.iteration}"
                )
            else:
                logger.info(f"Did not find new rules in node {self.learning_configs.id}")
        else:
            self.model = RuleSet(
                    remember_activation=model.remember_activation,
                    stack_activation=model.stack_activation,
                    rules_list=good_rules
                )
            if len(self.model) == 0:
                logger.info(f"No good rules found in node {self.learning_configs.id}")
            else:
                logger.info(f"Found {len(self.model)} rules in node {self.learning_configs.id}")
                self.model_to_file()
                self.fitter.save(
                    self.learning_configs.node_models_path / f"model_{self.filenumber}_{self.iteration}"
                )

    @emit
    def update_from_central(self, model: RuleSet) -> None:
        """Modifies the files pointed by `ifra.node.Node`'s *data.x_path* and `ifra.node.Node`'s *data.y_path* by
        calling `ifra.node.Node` *learning_configs.updater*.

        Parameters
        ----------
        model: RuleSet
            The central model
        """
        logger.info(f"Updating node {self.learning_configs.id}...")
        # Compute activation of the selected rules
        self.updater.update(model)
        logger.info(f"... node {self.learning_configs.id} updated.")

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

        existing_files = list(self.learning_configs.node_models_path.glob('model_main_*.csv'))
        if self.filenumber is None:
            self.filenumber = random.randint(0, int(1e6))
            path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"
            while path_main in existing_files:
                self.filenumber += random.randint(int(2e6), int(3e6))
                path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"
            path_main.touch()  # To lock file for possible other nodes trying to write here
        else:
            path_main = self.learning_configs.node_models_path / f"model_main_{self.filenumber}.csv"

        path_iteration = self.learning_configs.node_models_path / f"model_{self.filenumber}_{self.iteration}.csv"
        self.iteration += 1

        model.save(path_main)
        model.save(path_iteration)

        logger.info(f"Node {self.learning_configs.id}'s model saved in {path_iteration} and {path_main}.")

        try:
            path_table = path_main.with_suffix(".pdf")
            df = path_main.read(index_col=0)
            if not df.empty:
                df = df.apply(lambda x: x.round(3) if x.dtype == float else x)
            else:
                df = pd.DataFrame(data=[["empty"]])
            TableWriter(path_table, df, paperwidth=30).compile(clean_tex=True)
        except ValueError:
            logger.warning("Failed to produce tablewriter. Is LaTeX installed ?")

    @emit
    def plot_data_histogram(self, path: TransparentPath) -> None:
        """Plots the distribution of the data located in `ifra.node.Node`'s *data.x_path* and
        `ifra.node.Node`'s *data.y_path* and save them in unique files.

        Parameters
        ----------
        path: TransparentPath
            Path to save the data. Should be a directory.
        """
        x = self.data.x_path.read(**self.data.x_read_kwargs)
        for col, name in zip(x.columns, self.learning_configs.features_names):
            fig = plot_histogram(data=x[col], xlabel=name, figsize=(10, 7))

            iteration = 0
            name = name.replace(" ", "_")
            path_x = path / f"{name}_{iteration}.pdf"
            while path_x.is_file():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.pdf"
            fig.savefig(path_x)

        y = self.data.y_path.read(**self.data.y_read_kwargs)
        fig = plot_histogram(data=y.squeeze(), xlabel="Class", figsize=(10, 7))

        iteration = 0
        path_y = path / f"classes_{iteration}.pdf"
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
            logger.info(f"Fetched central model in node {self.learning_configs.id} at {self.last_fetch}")

        t = time()
        do_fit = True  # start at true to trigger fit even if no central model is here at first iteration

        if timeout <= 0:
            logger.warning("You did not specify a timeout for your run. It will last until manually stopped.")

        logger.info(f"Starting node {self.learning_configs.id}")
        logger.info(
            f"Starting node {self.learning_configs.id}. Monitoring changes in "
            f"{self.learning_configs.central_model_path}, {self.data.x_path} and {self.data.y_path}."
        )

        while time() - t < timeout or timeout <= 0:

            if (
                self.last_x is not None
                and self.last_y is not None
                and (
                    self.data.x_path.info()["mtime"] > self.last_x.timestamp()
                    or self.data.y_path.info()["mtime"] > self.last_y.timestamp()
                )
            ):
                # New data arrived for the node to use : redo dataprep and copy and force update from the central model
                logger.info(f"New data available for node {self.learning_configs.id}")
                self.datapreped = False
                self.copied = False
                self.last_fetch = None
                self.plot_dataprep_and_copy()

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

        logger.info(f"Timeout of {timeout} seconds reached, stopping learning in node {self.learning_configs.id}.")

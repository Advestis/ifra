import traceback
from datetime import datetime
from pathlib import Path
from time import time, sleep

from typing import Union, Optional

from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, NodeDataConfig
from .messenger import NodeMessenger
from .fitters import DecisionTreeFitter
from .updaters import AdaBoostUpdater
from .datapreps import BinFeaturesDataPrep
from .plot import plot_histogram

logger = logging.getLogger(__name__)


class Node:
    """One node of federated learning. This class should be used by each machine that is supposed to produce a local
    model. It monitors changes in a directory where the central server is supposed to push its model, and triggers fit
    when a new central model is available. It will also trigger a first fit before any central model is available.
    
    Attributes
    ----------
    public_configs: `ifra.configs.NodePublicConfig`
        The public configuration of the node. Will be accessible by the central server.
        see `ifra.configs.NodePublicConfig`
    path_public_configs: TransparentPath
        Path to the json file containing public configuration of the node. Needs to be kept in memory to potentially
        update the node's id once set by the central server.
    messenger: NodeMessages
        Interface for exchanging messages with central server. See `ifra.messenger.NodeMessages`
    dataprep: Union[None, Callable]
        The dataprep method to apply to the data before fitting. Optional, specified in the node's public configuration.
    datapreped: bool
        False if the dataprep has not been done yet, True after
    copied: bool
        In order not to alter the data given to the node, they are copied in new files used for learning. This flag is
        False if this copy has not been done yet, False otherwise.
    ruleset: Union[None, RuleSet]
        Fitted ruleset
    last_fetch: Union[None, datetime]
        Date and time when the central model was last fetched.
    data: `ifra.configs.NodeDataConfig`
        Configuration of the paths to the node's features and target data files
    fitter: `ifra.fitters.Fitter`
        Configuration of the paths to the node's features and target data files
    updater: `ifra.updaters.Updater`
        Configuration of the paths to the node's features and target data files
    """

    possible_fitters = {
        "decisiontree_fitter": DecisionTreeFitter
    }
    """Possible string values and corresponding fitter object for *fitter* attribute of
    `ifra.configs.NodePublicConfig`"""

    possible_updaters = {
        "adaboost_updater": AdaBoostUpdater
    }
    """Possible string values and corresponding updaters object for *updater* attribute of
    `ifra.configs.NodePublicConfig`"""

    possible_datapreps = {
        "binfeatures_dataprep": BinFeaturesDataPrep
    }
    """Possible string values and corresponding updaters object for *updater* attribute of
    `ifra.configs.NodePublicConfig`"""

    timeout_central = 600
    """How long the node is supposed to wait for the central server to start up"""

    def __init__(
        self,
        path_public_configs: Union[str, TransparentPath],
        path_data: Union[str, Path, TransparentPath],
    ):

        self.public_configs = NodePublicConfig(path_public_configs)
        self.path_public_configs = self.public_configs.path  # Will be a TransparentPath

        if self.public_configs.local_model_path.parent.is_dir():
            if not len(list(self.public_configs.local_model_path.parent.ls(""))) == 0:
                raise ValueError(
                    "Path where Node model should be saved is not empty. Make sure you cleaned all previous learning"
                    " output"
                )
        else:
            self.public_configs.local_model_path.parent.mkdir()

        self.path_messages = self.path_public_configs.parent / "messages.json"

        first_message = True
        t = time()
        while not self.path_messages.is_file():
            if first_message:
                logger.info("Waiting for central server to start before creating node...")
                first_message = False
            sleep(1)
            if time() - t > self.timeout_central:
                raise TimeoutError(f"Central server did not start after {self.timeout_central} seconds.")
        if first_message is False:
            logger.info("...central server started. Resuming node creation.")

        self.messenger = NodeMessenger(self.path_messages)
        self.data = NodeDataConfig(path_data)
        self.data.dataprep_kwargs = self.public_configs.dataprep_kwargs

        if self.public_configs.fitter not in self.possible_fitters:
            function = self.public_configs.fitter.split(".")[-1]
            module = self.public_configs.fitter.replace(f".{function}", "")
            self.fitter = getattr(
                __import__(module, globals(), locals(), [function], 0), function
            )(self.public_configs, self.data)
        else:
            self.fitter = self.possible_fitters[self.public_configs.fitter](self.public_configs, self.data)

        if self.public_configs.updater not in self.possible_updaters:
            function = self.public_configs.updater.split(".")[-1]
            module = self.public_configs.updater.replace(f".{function}", "")
            self.updater = getattr(__import__(module, globals(), locals(), [function], 0), function)(self.data)
        else:
            self.updater = self.possible_updaters[self.public_configs.updater](self.data)

        if self.public_configs.dataprep is not None:
            if self.public_configs.dataprep not in self.possible_datapreps:
                function = self.public_configs.dataprep.split(".")[-1]
                module = self.public_configs.dataprep.replace(f".{function}", "")
                self.dataprep = getattr(__import__(module, globals(), locals(), [function], 0), function)(self.data)
            else:
                self.dataprep = self.possible_datapreps[self.public_configs.dataprep](self.data)
        else:
            self.dataprep = None

        self.datapreped = False
        self.copied = False
        self.ruleset = None
        self.last_fetch = None

    def fit(self) -> None:
        """Plots the features and classes distribution in `ifra.node.Node`'s *data.x* and
        `ifra.node.Node`'s *data.y* parent directories if `ifra.configs.NodePublicConfig` *plot_data*
        is True.

        Triggers `ifra.node.Node`'s *dataprep* on the features and classes if a dataprep method was
        specified, sets `ifra.node.Node` datapreped to True, plots the distributions of the datapreped data.

        If `ifra.node.Node` *copied* is False, copies the files pointed by `ifra.node.Node`'s *data.x*
        and `ifra.node.Node`'s *data.y* in new files in the same directories by appending
        *_copy_for_learning* to their names. Sets `ifra.node.Node` *copied* to True.
        Calls the the fitter corresponding to `ifra.node.Node` *public_configs.fitter* on the node's features and
        targets and save the resulting ruleset in `ifra.node.Node` *public_configs.local_model_path*.
        """

        self.messenger.fitting = True

        try:
            if self.public_configs.plot_data:
                if not (self.data.x.parent / "plots").is_dir():
                    (self.data.x.parent / "plots").mkdir()
                self.plot_data_histogram(self.data.x.parent / "plots")

            if self.dataprep is not None and not self.datapreped:
                logger.info(f"Datapreping data of node {self.public_configs.id}...")
                self.dataprep.dataprep()
                if self.public_configs.plot_data:
                    if not (self.data.x.parent / "plots_datapreped").is_dir():
                        (self.data.x.parent / "plots_datapreped").mkdir()
                    self.plot_data_histogram(self.data.x.parent / "plots_datapreped")
                self.datapreped = True
                logger.info(f"...datapreping done for node {self.public_configs.id}")

            if not self.copied:
                # Copy x and y in a different file, for it will be changed after each iterations by the update from
                # central
                x_suffix = self.data.x.suffix
                self.data.x.cp(self.data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix))
                self.data.x = self.data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix)
                y_suffix = self.data.y.suffix
                self.data.y.cp(self.data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix))
                self.data.y = self.data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix)
                self.copied = True

            logger.info(f"Fitting node {self.public_configs.id} using {self.public_configs.fitter}...")
            self.ruleset = self.fitter.fit()
            logger.info(f"Found {len(self.ruleset)} rules in node {self.public_configs.id}")
            self.ruleset_to_file()

            self.messenger.fitting = False
            logger.info(f"... node {self.public_configs.id} fitted, results saved in"
                    f" {self.public_configs.local_model_path.parent}.")
        except Exception as e:
            self.messenger.fitting = False
            # Do not set messenger's error, done in self.watch
            raise e

    def update_from_central(self, ruleset: RuleSet) -> None:
        """Modifies the files pointed by `ifra.node.Node`'s *data.x* and `ifra.node.Node`'s *data.y* by
        calling `ifra.node.Node` *public_configs.updater*.

        Parameters
        ----------
        ruleset: RuleSet
            The central model's ruleset
        """
        logger.info(f"Updating node {self.public_configs.id}...")
        # Compute activation of the selected rules
        self.updater.update(ruleset)
        logger.info(f"... node {self.public_configs.id} updated.")

    def ruleset_to_file(self) -> None:
        """Saves `ifra.node.Node` *ruleset* to `ifra.configs.NodePublicConfig` *local_model_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the ruleset using TableWriter.
        Does not do anything if `ifra.node.Node` *ruleset* is None
        """

        ruleset = self.ruleset

        iteration = 0
        name = self.public_configs.local_model_path.stem
        path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.public_configs.local_model_path)

        try:
            path_table = path.with_suffix(".pdf")
            TableWriter(
                path_table,
                path.read(index_col=0).apply(
                    lambda x: x.round(3) if x.dtype == float else x
                ),
                paperwidth=30
            ).compile(clean_tex=True)
        except ValueError:
            logger.warning("Failed to produce tablewriter. Is LaTeX installed ?")

    def plot_data_histogram(self, path: TransparentPath) -> None:
        """Plots the distribution of the data located in `ifra.node.Node`'s *data.x* and
        `ifra.node.Node`'s *data.y* and save them in unique files.

        Parameters
        ----------
        path: TransparentPath
            Path to save the data. Should be a directory.
        """
        x = self.data.x.read(**self.data.x_read_kwargs)
        for col, name in zip(x.columns, self.public_configs.features_names):
            fig = plot_histogram(
                data=x[col],
                xlabel=name,
                figsize=(10, 7)
            )

            iteration = 0
            name = name.replace(' ', '_')
            path_x = path / f"{name}_{iteration}.pdf"
            while path_x.is_file():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.pdf"
            fig.savefig(path_x)

        y = self.data.y.read(**self.data.y_read_kwargs)
        fig = plot_histogram(
            data=y.squeeze(),
            xlabel="Class",
            figsize=(10, 7)
        )

        iteration = 0
        path_y = path / f"classes_{iteration}.pdf"
        while path_y.is_file():
            iteration += 1
            path_y = path_y.parent / f"classes_{iteration}.pdf"

        fig.savefig(path_y)

    def watch(self, timeout: int = 0, sleeptime: int = 1) -> None:
        """Monitors new changes in the central server, every *sleeptime* seconds for *timeout* seconds, triggering
        node fit when a new model is found, or if the function just started. Sets
       ` ifra.configs.NodePublicConfig` *id* by re-reading the configuration file if it is None.
        Stops if `ifra.configs.NodePublicConfig` *stop* is True (set by `ifra.central_server` *CentralServer*)
        This is the only method the user should call.

        Parameters
        ----------
        timeout: Optional[int]
            How many seconds the watch last. If <= 0>, will last until the central server sets
            `ifra.configs.NodePublicConfig` *stop* to True. Default value = 0
        sleeptime: int
            How many seconds between each checks for new central model. Default value = 5
        """

        try:
            def get_ruleset() -> None:
                """Fetch the central server's latest model's RuleSet.
                `ifra.node.Node.last_fetch` will be set to now and `ifra.node.Node.new_data` to True."""
                central_ruleset = RuleSet()
                central_ruleset.load(self.public_configs.central_model_path)
                self.last_fetch = datetime.now()
                self.update_from_central(central_ruleset)
                logger.info(f"Fetched new central ruleset in node {self.public_configs.id} at {self.last_fetch}")
                self.new_data = True

            t = time()
            stop_message = True
            new_central_model = True  # start at true to trigger fit even if no central model is here at first iteration
            logger.info(f"Starting node {self.public_configs.id}")
            self.messenger.running = True
            while time() - t < timeout or timeout <= 0:

                self.public_configs = NodePublicConfig(self.path_public_configs)
                self.messenger.get_latest_messages()
                if self.public_configs.id is None:
                    raise ValueError(
                        "Node id is not set, meaning central server is not running. Always start central server first."
                    )
                if self.messenger.central_error is not None:
                    logger.info(f"Central server crashed. Stopping node {self.public_configs.id}.")
                    stop_message = False
                    break
                if self.messenger.stop:
                    logger.info(f"'stop' flag is True. Stopping learning in node {self.public_configs.id}")
                    stop_message = False
                    break

                if self.last_fetch is None:
                    if self.public_configs.central_model_path.is_file():
                        get_ruleset()
                        new_central_model = True
                else:
                    if (
                        self.public_configs.central_model_path.is_file()
                        and self.public_configs.central_model_path.info()["mtime"] > self.last_fetch.timestamp()
                    ):
                        get_ruleset()
                        new_central_model = True

                if new_central_model:
                    self.fit()
                    new_central_model = False
                sleep(sleeptime)

            if stop_message:
                logger.info(f"Timeout of {timeout} seconds reached, stopping learning in node"
                            f" {self.public_configs.id}.")

            self.messenger.running = False
        except Exception as e:
            self.messenger.running = False
            self.messenger.error = traceback.format_exc()
            raise e

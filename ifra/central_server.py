from datetime import datetime
from time import time, sleep
from typing import Union

from ruleskit import RuleSet
from tablewriter import TableWriter
import logging

from .configs import CentralConfig
from .actor import Actor
from .decorator import emit

logger = logging.getLogger(__name__)


class CentralServer(Actor):
    """Implementation of the notion of central server in federated learning.

    It monitors changes in a remote GCP directory, were aggregator is expected to write its model.
    Upon changes of the model file, the central server downloads is, updates its using
    `ifra.central_model_updaters.central_model_update` model and saves it to a directory.
    This directory is read by the nodes to update their own models.

    Attributes
    ----------
    central_configs: CentralConfig
        see `ifra.configs.CentralConfig`
    model: RuleSet
        Central server model's model
    last_fetch: datetime
        last time the aggregated model was fetched
    """

    def __init__(
        self,
        central_configs: CentralConfig,
    ):
        """
        Parameters
        ----------
        central_configs: CentralConfig
            This central server configurations. see `ifra.configs.CentralConfig`.
        """
        self.central_configs = None
        self.model = None
        self.last_fetch = None
        super().__init__(central_configs=central_configs)

    @emit
    def create(self):
        self.model = RuleSet()
        # s = "\n".join([f"{key}: {self.central_configs.configs[key]}" for key in self.central_configs.configs])
        # logger.info(f"Central Server configuration:\n{s}")

    @emit
    def update(self):
        """Add new rules in aggregated_model to central model's models.
        If new rules were found, save the updated model to `ifra.central_server.CentralServer`'s 
        *central_configs.central_model_path*.
        
        Returns
        -------
        bool
            True if new rules were added, False if aggregated_model contained no new rules
        """
        aggregated_model = RuleSet()
        aggregated_model.load(self.central_configs.aggregated_model_path)
        self.last_fetch = datetime.now()
        if self.model is None:
            self.model = aggregated_model
        else:
            for r in aggregated_model:
                if r in self.model:
                    logger.warning(
                        f"Rule '{r}' was aggregated despite being already in the central server. It is ignored, its"
                        " prediction is discarded."
                    )
                else:
                    self.model.append(r, update_activation=False)
        logger.info(f"Fetched new model from aggregator at {self.last_fetch}")
        self.model_to_file()

    @emit
    def model_to_file(self) -> None:
        """Saves `ifra.node.Node` *model* to `ifra.configs.CentralConfig` *central_model_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the model using TableWriter.
        Does not do anything if `ifra.node.Node` *model* is None
        """
        model = self.model

        iteration = 0
        name = self.central_configs.central_model_path.stem
        path = self.central_configs.central_model_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.central_configs.central_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        if not path.parent.isdir():
            path.parent.mkdir(parents=True)
        model.save(path)
        model.save(self.central_configs.central_model_path)
        logger.info(f"Saved central model in '{self.central_configs.central_model_path}'")

        try:
            path_table = path.with_suffix(".pdf")
            TableWriter(
                path_table, path.read(index_col=0).apply(lambda x: x.round(3) if x.dtype == float else x), paperwidth=30
            ).compile(clean_tex=True)
        except ValueError:
            logger.warning("Failed to produce tablewriter. Is LaTeX installed ?")

    @emit
    def run(self, timeout: Union[int, float] = 0, sleeptime: Union[int, float] = 5):
        """Monitors new changes in the nodes, every ''sleeptime'' seconds for ''timeout'' seconds, triggering
        aggregation and pushing central model to nodes when enough new node models are available.

        Stops if all nodes gave a model but no new rules could be found from them. Tells the nodes to stop too.

        Parameters
        ----------
        timeout: Union[int, float]
            How many seconds should the run last. If <= 0, will last until killed by the user. Default value = 0.
        sleeptime: Union[int, float]
            How many seconds between each checks for new nodes models. Default value = 5.
        """

        started = self.iterations != 0  # To force at least one loop of the while to trigger

        if timeout <= 0:
            logger.warning("You did not specify a timeout for your run. It will last until manually stopped.")
        logger.info(f"Starting central server. Monitoring changes in {self.central_configs.aggregated_model_path}.")

        t = time()
        while time() - t < timeout or timeout <= 0 or started is False:
            started = True
            if len(self.model) == 0:
                if self.central_configs.aggregated_model_path.is_file():
                    self.update()
                    self.iterations += 1
            else:
                if (
                        self.central_configs.aggregated_model_path.is_file()
                        and self.central_configs.aggregated_model_path.info()["mtime"] > self.last_fetch.timestamp()
                ):
                    self.update()
                    self.iterations += 1

            sleep(sleeptime)

        logger.info(f"Timeout of {timeout} seconds reached, stopping central server.")
        if self.model is None:
            logger.warning("Learning failed to produce a central model. No output generated.")
        logger.info(f"Made {self.iterations} complete iterations between central server and aggregator.")
        logger.info(f"Results saved in {self.central_configs.central_model_path}")

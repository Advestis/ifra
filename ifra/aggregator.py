from datetime import datetime
from time import time, sleep
from typing import List, Union, Tuple

from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .configs import AggregatorConfig
from .actor import Actor
from .decorator import emit
from .aggregations import AdaBoostAggregation, Aggregation, ReverseAdaBoostAggregation, AggregateAll

logger = logging.getLogger(__name__)


class NodeGate:

    """This class is the gate used by the aggregator to interact with the remote nodes.
    It only knows the path to the main model of a given node, and nothing else, thus making each node an anonymous
    contributor to the model.
    It implements a method, `ifra.aggregator.NodeGate.interact`, that checks whether the node produced a new model.

    Attributes
    ----------
    ok: bool
        True if the object initiated correctly
    model_path: TransparentPath
        The file containing one node model
    id: int
        Unique NodeGate id number, corresponding to the number of existing NodeGate at object creation.
    model: RuleSet
        Latest node model, same as `ifra.node.Node` *model*. None at initialisation, set by
        `ifra.aggregator.NodeGate.interact`
    last_fetch: datetime
        Last time the node produced a new model. None at initialisation, set by
        `ifra.aggregator.NodeGate.interact`
    """

    instances = 0
    """Counts the number of existing objects"""

    paths = []
    """Remembers all paths known by all instances of the class. This is done to avoid creating several nodes with the
    same model path."""

    def __init__(self, model_main_path: TransparentPath):
        """
        Parameters
        ----------
        model_main_path: TransparentPath
            Path to the node's model file.
        """

        if model_main_path in self.paths:
            raise ValueError(f"You can not recreate already existing node with model path {model_main_path}")
        self.ok = False
        self.model_path = model_main_path
        self.id = self.instances
        self.__class__.instances += 1
        self.paths.append(model_main_path)

        self.model = None
        self.last_fetch = None
        self.id_set = False
        self.ok = True
        self.reference_node_config = None

    def interact(self) -> bool:
        """Fetch one anonymous node's latest model.

        Returns
        -------
        True if successfully fetched new model, else False. Can be False if the model's file disapeared, or if has not
        been modified since last check.
        """

        def get_model() -> None:
            """Fetch the node's model.
            `ifra.aggregator.NodeGate` *last_fetch* will be set to now.
            """
            self.model = RuleSet()
            self.model.load(self.model_path)
            self.last_fetch = datetime.now()
            self.new_model_found = True
            logger.info(f"Aggregator - Fetched new model from node {self.id} at {self.last_fetch}")

        if self.model is None:
            # The node has not produced any model yet if self.model is None. No need to bother with self.last_fetch
            # then, just get the model.
            if self.model_path.is_file():
                get_model()
                return True
            else:
                logger.warning(f"Aggregator - model located at {self.model_path} disapeared.")
                return False
        else:
            # The node has already produced a model. So we only get its model if it is new. We know that by
            # checking the modification time of the node's model file.
            if self.model_path.is_file():
                if self.model_path.info()["mtime"] > self.last_fetch.timestamp():
                    get_model()
                    return True
                else:
                    logger.debug(f"Aggregator - Node {self.id} has no new model. Skipping for now.")
                    return False
            else:
                logger.warning(f"Aggregator - Model located at {self.model_path} disapeared.")
                return False


class Aggregator(Actor):
    """Implementation of the notion of aggregator in federated learning.

    It monitors changes in a given remote GCP directory, were nodes are expected to write their models.
    It periodically checks for new or modified model files that is downloads. When enough model files are downloaded,
    the aggregator aggregates them to produce the aggregated model, which is saved to a directory.
    This directory is then read by the central server to update the central model
    (see `ifra.central_server.CentralServer`).
    "Enough" model is defined in the aggregator configuration (see `ifra.configs.AggregatorConfig`)

    Attributes
    ----------
    aggregator_configs: AggregatorConfig
        see `ifra.configs.AggregatorConfig`
    nodes: List[NodeGate]
        List of all gates to the nodes the aggregator found.
    model: RuleSet
        Aggregated model
    aggregation: Aggregation
        Instance of one of the `ifra.aggregations.Aggregation` daughter classes.
    """

    possible_aggregations = {
        "adaboost": AdaBoostAggregation, "reverseadaboost": ReverseAdaBoostAggregation, "keepall": AggregateAll
    }
    """Possible string values and corresponding aggregation methods for *aggregation* attribute of
    `ifra.aggregator.Aggregator`"""

    def __init__(
        self,
        aggregator_configs: AggregatorConfig,
    ):
        """
        Parameters
        ----------
        aggregator_configs: AggregatorConfig
            Aggregator configuration. See `ifra.configs.AggregatorConfig`
        """
        self.aggregator_configs = None
        self.nodes = []
        self.model = None
        self.aggregation = None
        super().__init__(aggregator_configs=aggregator_configs)

    @emit
    def create(self):

        self.nodes = []

        if self.aggregator_configs.min_number_of_new_models < 1:
            raise ValueError("Minimum number of new nodes to trigger aggregation must be 1 or more")

        self.model = None

        if self.aggregator_configs.aggregation not in self.possible_aggregations:
            function = self.aggregator_configs.aggregation.split(".")[-1]
            module = self.aggregator_configs.aggregation.replace(f".{function}", "")
            self.aggregation = getattr(__import__(module, globals(), locals(), [function], 0), function)(self)
            if not isinstance(self.aggregation, Aggregation):
                raise TypeError("Aggregator aggregation should inherite from Aggregation class")
        else:
            self.aggregation = self.possible_aggregations[self.aggregator_configs.aggregation](self)

    @emit
    def aggregate(self, models: List[RuleSet]) -> Tuple[str, Union[RuleSet, None]]:
        """Aggregates models in `ifra.aggregator.Aggregator` *model* using
        `ifra.aggregator.Aggregator` *aggregation*

        Parameters
        ----------
        models: List[RuleSet]
            New models provided by the nodes.

        Returns
        -------
        Tuple[str, Union[RuleSet, None]]
            First item of the tuple can be :
              * "updated" if aggregated model was updated
              * "pass" if no new rules were found
            The second item is the aggregated model if the first item is "updated", None otherwise
        """
        return self.aggregation.aggregate(models)

    @emit
    def model_to_file(self) -> None:
        """Saves `ifra.node.Node` *model* to `ifra.configs.AggregatorConfig` *aggregated_model_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the model using TableWriter.
        Does not do anything if `ifra.node.Node` *model* is None
        """
        model = self.model
        # Aggregated model's criterion and coverage are meaningless
        model._coverage = None
        model.criterion = None

        iteration = 0
        name = self.aggregator_configs.aggregated_model_path.stem
        path = self.aggregator_configs.aggregated_model_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.aggregator_configs.aggregated_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        if not path.parent.isdir():
            path.parent.mkdir(parents=True)
        model.save(path)
        model.save(self.aggregator_configs.aggregated_model_path)
        logger.info(f"Aggregator - Saved aggregated model in '{self.aggregator_configs.aggregated_model_path}'")

        try:
            path_table = path.with_suffix(".pdf")
            TableWriter(
                path_table, path.read(index_col=0).apply(lambda x: x.round(3) if x.dtype == float else x), paperwidth=30
            ).compile(clean_tex=True)
        except ValueError:
            logger.warning("Aggregator - Failed to produce tablewriter. Is LaTeX installed ?")

    @emit
    def run(self, timeout: Union[int, float] = 0, sleeptime: Union[int, float] = 5):
        """Monitors new changes in the nodes, every ''sleeptime'' seconds for ''timeout'' seconds, triggering
        aggregation and pushing aggregated model to nodes when enough new node models are available.

        Stops if all nodes gave a model but no new rules could be found from them. Tells the nodes to stop too.

        Parameters
        ----------
        timeout: Union[int, float]
            How many seconds should the run last. If <= 0, will last until killed by the user. Default value = 0.
        sleeptime: Union[int, float]
            How many seconds between each checks for new nodes models. Default value = 5.
        """
        updated_nodes = []

        if timeout <= 0:
            logger.warning(
                "Aggregator - You did not specify a timeout for your run. It will last until manually stopped."
            )
        logger.info(
            "Starting aggregator. Monitoring changes in nodes' models directories"
            f" {self.aggregator_configs.node_models_path}."
        )
        started = self.iterations != 0  # To force at least one loop of the while to trigger

        t = time()
        while time() - t < timeout or timeout <= 0 or started is False:
            started = True
            new_models = False
            new_nodes = 0
            if not self.aggregator_configs.node_models_path.isdir():
                self.aggregator_configs.node_models_path.mkdir(parents=True)
            for path in self.aggregator_configs.node_models_path.glob("model_main_*.csv"):
                if path in NodeGate.paths:
                    # Already found this node in a previous check
                    continue
                node = NodeGate(path)
                if node.ok:
                    self.nodes.append(node)
                    new_nodes += 1
                else:
                    logger.warning("Aggregator - One node could not be instantiated. Ignoring it.")

            if new_nodes > 0:
                logger.info(f"Aggregator - Found {new_nodes} new nodes. Aggregator now knows {len(self.nodes)} nodes.")

            for node in self.nodes:
                # Node fetches its latest model from GCP. Returns True if a new model was found.
                if node.interact() is False:
                    continue

                if node not in updated_nodes:
                    updated_nodes.append(node)
                if len(updated_nodes) >= self.aggregator_configs.min_number_of_new_models:
                    new_models = True

            if new_models:
                logger.info(f"Aggregator - Found enough ({len(updated_nodes)}) new nodes models.")
                what_now = self.aggregate([node.model for node in set(updated_nodes)])
                if what_now == "updated":
                    # Aggregation successfully updated aggregated model: clean the list of updated nodes.
                    updated_nodes = []

                    if self.model is None:
                        raise ValueError("Should never happen !")
                    self.iterations += 1
                    self.model_to_file()
                else:
                    logger.info("Aggregator - New models did not produce anything new yet.")

            sleep(sleeptime)

        logger.info(f"Aggregator - Timeout of {timeout} seconds reached, stopping aggregator.")
        if self.model is None:
            logger.warning("Learning failed to produce an aggregatored model. No output generated.")
        logger.info(f"Aggregator - Made {self.iterations} complete iterations between aggregator and nodes.")
        logger.info(f"Aggregator - Results saved in {self.aggregator_configs.aggregated_model_path}")

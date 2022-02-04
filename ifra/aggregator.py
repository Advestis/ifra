from datetime import datetime
from time import time, sleep
from typing import List, Union, Tuple

from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, AggregatorConfig
from .actor import Actor
from .decorator import emit
from .aggregations import AdaBoostAggregation, Aggregation

logger = logging.getLogger(__name__)


class NodeGate:

    """This class is the gate used by the aggregator to interact with the remote nodes.
    It knows the public configuration of one given node, to which must correspond an instance of the
    `ifra.node.Node` class. This configuration holds among other things, the places where the nodes
    are supposed to save their models. It implements a method, `ifra.aggregator.NodeGate.interact`,
    that check whether the node produced a new model.

    Attributes
    ----------
    ok: bool
        True if the object initiated correctly
    id: int
        Unique NodeGate id number. If the corresponding `ifra.node.Node` instance has a custom
        id, then it will use it instead.
    public_configs: NodePublicConfig
        Node's public configuration. See `ifra.configs.NodePublicConfig`. None at initialisation, set by
        `ifra.aggregator.NodeGate.interact`
    ruleset: RuleSet
        Latest node model's ruleset, same as `ifra.node.Node` *ruleset*. None at initialisation, set by
        `ifra.aggregator.NodeGate.interact`
    last_fetch: datetime
        Last time the node produced a new model. None at initialisation, set by
        `ifra.aggregator.NodeGate.interact`
    """

    instances = []

    def __init__(self, public_config: NodePublicConfig):
        """
        Parameters
        ----------
        public_config: Union[str, TransparentPath]
            Directory were node's public configuration file can be found
        """

        self.ok = False
        self.public_configs = public_config
        self.id = self.public_configs.id
        i = 2
        while self.id in self.instances:
            self.id = f"{self.public_configs.id}_{i}"
            i += 1
        self.instances.append(self.id)

        self.ruleset = None
        self.last_fetch = None
        self.id_set = False
        self.ok = True
        self.reference_node_config = None

    def interact(self, reference_node_config: NodePublicConfig) -> bool:
        """Fetch the node's latest i any model.

        Parameters
        ----------
        reference_node_config: NodePublicConfig
            Configuration reference. If the node's public configuration do not match it, the node is ignored.
            If reference_node_config is ont set yet, then this node's public configuration becomes the reference.

        Returns
        -------
        True if successfully fetched new model, else False
        """

        def get_ruleset():
            """Fetch the node's model's RuleSet.
            `ifra.aggregator.NodeGate` *last_fetch* will be set to now.
            """
            self.ruleset = RuleSet()
            self.ruleset.load(self.public_configs.node_model_path)
            self.last_fetch = datetime.now()
            self.new_model_found = True
            logger.info(f"Fetched new ruleset from node {self.id} at {self.last_fetch}")

        if len(reference_node_config.configs) > 0 and self.public_configs != reference_node_config:
            s = "\n"
            for key in self.public_configs.configs:
                if self.public_configs.configs[key] != reference_node_config.configs[key]:
                    s = (
                        f"{s}{key}: {self.public_configs.configs[key]}"
                        f" != {reference_node_config.configs[key]}\n"
                    )
            logger.warning(
                f"Node {self.id}'s configuration is not compatible with previous"
                f" configuration. Skipping for now. Differences are: {s}"
            )
            return False

        if self.ruleset is None:
            # The node has not produced any model yet if self.ruleset is None. No need to bother with self.last fetch
            # then, just get the ruleset.
            if self.public_configs.node_model_path.is_file():
                get_ruleset()
                return True
            return False
        else:
            # The node has already produced a model. So we only get its ruleset if the model is new. We know that by
            # checking the modification time of the node's ruleset file.
            if (
                self.public_configs.node_model_path.is_file()
                and self.public_configs.node_model_path.info()["mtime"] > self.last_fetch.timestamp()
            ):
                get_ruleset()
                return True
            logger.debug(f"Node {self.id} has no new model. Skipping for now.")
            return False


class Aggregator(Actor):
    """Implementation of the notion of aggregator in federated learning.

    It monitors changes in a given list of remote GCP directories, were nodes are expected to write their models.
    Upon changes of the model files, the aggregator downloads them. When enough model files are downloaded, the
    aggregator aggregates them to produce the aggregated model, which is saved to a directory. This directory is then
    read by the central server to update the central model (see `ifra.central_server.CentralServer`).
    "Enough" model is defined in the aggregator configuration (see `ifra.configs.AggregatorConfig`)

    Attributes
    ----------
    reference_node_config: NodePublicConfig
        Nodes linked to a given aggregator should have the same public configuration (except for their specific
        paths). This attribute remembers the said configuration, to ignore nodes with a wrong configuration.
    aggregator_configs: AggregatorConfig
        see `ifra.configs.AggregatorConfig`
    nodes: List[NodeGate]
        List of all gates to the nodes the aggregator should monitor.
    ruleset: RuleSet
        Aggregated model's ruleset
    aggregation: Aggregation
        Instance of one of the `ifra.aggregations.Aggregation` daughter classes.
    """

    possible_aggregations = {"adaboost": AdaBoostAggregation}
    """Possible string values and corresponding aggregation methods for *aggregation* attribute of
    `ifra.aggregator.Aggregator`"""

    def __init__(
        self,
        nodes_configs: List[NodePublicConfig],
        aggregator_configs: AggregatorConfig,
    ):
        """
        Parameters
        ----------
        nodes_configs: List[NodePublicConfig]
            List of each node's public configuration file
        aggregator_configs: AggregatorConfig
            Aggregator configuration. See `ifra.configs.AggregatorConfig`
        """
        self.nodes_configs = None
        self.aggregator_configs = None
        self.reference_node_config = None
        self.nodes = []
        self.ruleset = None
        self.aggregation = None
        super().__init__(nodes_configs=nodes_configs, aggregator_configs=aggregator_configs)

    @emit
    def create(self):

        self.nodes = []
        for config in self.nodes_configs:  # cast into set in order not to create twice the same node
            node = NodeGate(config)
            if node.ok:
                self.nodes.append(node)
            else:
                logger.warning("One node could not be instantiated. Ignoring it.")

        if self.aggregator_configs.min_number_of_new_models < 2:
            raise ValueError("Minimum number of new nodes to trigger aggregation must be 2 or more")

        if len(self.nodes) < self.aggregator_configs.min_number_of_new_models:
            raise ValueError(
                f"The minimum number of nodes to trigger aggregation "
                f"({self.aggregator_configs.min_number_of_new_models}) is greater than the number of"
                f" available nodes ({len(self.nodes)}): that can not work !"
            )

        self.ruleset = None

        if self.aggregator_configs.aggregation not in self.possible_aggregations:
            function = self.aggregator_configs.aggregation.split(".")[-1]
            module = self.aggregator_configs.aggregation.replace(f".{function}", "")
            self.aggregation = getattr(__import__(module, globals(), locals(), [function], 0), function)(self)
            if not isinstance(self.aggregation, Aggregation):
                raise TypeError("Aggregator aggregation should inherite from Aggregation class")
        else:
            self.aggregation = self.possible_aggregations[self.aggregator_configs.aggregation](self)

        # s = "\n".join([f"{key}: {self.aggregator_configs.configs[key]}" for key in self.aggregator_configs.configs])
        # logger.info(f"Aggregator configuration:\n{s}")

    @emit
    def aggregate(self, rulesets: List[RuleSet]) -> Tuple[str, Union[RuleSet, None]]:
        """Aggregates rulesets in `ifra.aggregator.Aggregator` *ruleset* using
        `ifra.aggregator.Aggregator` *aggregation*

        Parameters
        ----------
        rulesets: List[RuleSet]
            New rulesets provided by the nodes.

        Returns
        -------
        Tuple[str, Union[RuleSet, None]]
            First item of the tuple can be :
              * "updated" if aggregated model was updated
              * "pass" if no new rules were found
            The second item is the aggregated ruleset if the first item is "updated", None otherwise
        """
        return self.aggregation.aggregate(rulesets)

    @emit
    def ruleset_to_file(self) -> None:
        """Saves `ifra.node.Node` *ruleset* to `ifra.configs.AggregatorConfig` *aggregated_model_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the ruleset using TableWriter.
        Does not do anything if `ifra.node.Node` *ruleset* is None
        """
        ruleset = self.ruleset

        iteration = 0
        name = self.aggregator_configs.aggregated_model_path.stem
        path = self.aggregator_configs.aggregated_model_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.aggregator_configs.aggregated_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.aggregator_configs.aggregated_model_path)
        logger.info(f"Saved aggregated model in '{self.aggregator_configs.aggregated_model_path}'")

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
        aggregation and pushing aggregated model to nodes when enough new node models are available.

        Stops if all nodes gave a model but no new rules could be found from them. Tells the nodes to stop too.

        Parameters
        ----------
        timeout: Union[int, float]
            How many seconds should the run last. If <= 0, will last until killed by the user. Default value = 0.
        sleeptime: Union[int, float]
            How many seconds between each checks for new nodes models. Default value = 5.
        """
        t = time()
        updated_nodes = []
        iterations = 0

        if timeout <= 0:
            logger.warning("You did not specify a timeout for your run. It will last until manually stopped.")
        logger.info("Starting aggregator. Monitoring changes in nodes' models directories.")

        if len(self.nodes) == 0:
            raise ValueError("No nodes to learn on !")

        while time() - t < timeout or timeout <= 0:

            new_models = False

            for node in self.nodes:
                if self.reference_node_config is None:
                    self.reference_node_config = node.public_configs
                # Node fetches its latest model from GCP. Returns True if a new model was found.
                if node.interact(self.reference_node_config) is False:
                    continue

                updated_nodes.append(node)
                if len(updated_nodes) >= self.aggregator_configs.min_number_of_new_models:
                    new_models = True

            if new_models:
                logger.info(f"Found enough ({len(updated_nodes)}) new nodes models. Triggering aggregation.")
                what_now = self.aggregate([node.ruleset for node in updated_nodes])
                if what_now == "updated":
                    # Aggregation successfully updated aggregated model.
                    updated_nodes = []

                    if self.ruleset is None:
                        raise ValueError("Should never happen !")
                    iterations += 1
                    self.ruleset_to_file()

            sleep(sleeptime)

        logger.info(f"Timeout of {timeout} seconds reached, stopping aggregator.")
        if self.ruleset is None:
            logger.warning("Learning failed to produce an aggregatored model. No output generated.")
        logger.info(f"Made {iterations} complete iterations between aggregator and nodes.")
        logger.info(f"Results saved in {self.aggregator_configs.aggregated_model_path}")

from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import List, Union, Tuple, Optional

from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, CentralConfig
from .actor import Actor
from .decorator import emit
from .messenger import Receiver
from .aggregations import AdaBoostAggregation

logger = logging.getLogger(__name__)


class NodeGate:

    """This class is the gate used by the central server to interact with the remote nodes. Mainly, it has in memory
    the public configuration of one given node, to which must correspond an instance of the
    `ifra.node.Node` class. This configuration holds among other things, the places where the nodes
    are supposed to save their models and where they will be looking for the central model. It implements a method,
    `ifra.central_server.NodeGate.interact`, that check whether the node produced a new model.

    Attributes
    ----------
    ok: bool
        True if the object initiated correctly
    id: int
        Unique NodeGate id number. If the corresponding `ifra.node.Node` instance has not custom
        id, then it will use this one.
    public_configs: NodePublicConfig
        Node's public configuration. See `ifra.configs.NodePublicConfig`. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    path_receiver: TransparentPath
        Path to the node's emitter json file, to read with `ifra.messenger.Receiver`
    ruleset: RuleSet
        Latest node model's ruleset, same as `ifra.node.Node` *ruleset*. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    last_fetch: datetime
        Last time the node produced a new model. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    id_set: bool
        True if the Node's ID was set by this object, or if it did not need to set the Node's ID.
    """

    instances = 0

    def __init__(self, public_config: NodePublicConfig):
        """
        Parameters
        ----------
        public_config: Union[str, TransparentPath]
            Directory were node's public configuration file can be found
        """

        self.ok = False
        self.id = NodeGate.instances
        self.public_configs = public_config
        self.path_receiver = self.public_configs.path_emitter
        self.receiver = Receiver(self.path_receiver, wait=False)

        self.ruleset = None
        self.last_fetch = None
        self.id_set = False
        self.ok = True

        NodeGate.instances += 1

    def interact(self, reference_node_config: NodePublicConfig, central_model_path: TransparentPath) -> bool:
        """Fetch the node's configuration if not done yet and the node's latest model if it is new compared to central
        server model. Sets `ifra.central_server.NodeGate` *public_configs*,
        `ifra.central_server.NodeGate` *ruleset*, `ifra.central_server.NodeGate` *last_fetch* to now
        and `ifra.central_server.NodeGate` *new_data* to True

        Parameters
        ----------
        reference_node_config: NodePublicConfig
            Configuration reference. If the node's public configuration do not match it, the node is ignored.
            If reference_node_config is ont set yet, then this node's public configuration becomes the reference
            if it produced a ew model.
        central_model_path: TransparentPath
            Path to where the central model is, to give to the node's configurations.

        Returns
        -------
        True if successfully fetched new model, else False
        """

        def get_ruleset():
            """Fetch the node's latest model's RuleSet.
            `ifra.central_server.NodeGate` *last_fetch* will be set to now and
            `ifra.central_server.NodeGate` *new_data* to True."""
            self.ruleset = RuleSet()
            self.ruleset.load(self.public_configs.local_model_path)
            self.last_fetch = datetime.now()
            self.new_model_found = True
            logger.info(f"Fetched new ruleset from node {self.public_configs.id} at {self.last_fetch}")

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

        if self.id_set is False:
            self.id_set = True
            # First time the node is interacted with
            if self.public_configs.id is None:
                # If the node was not given an id by its local user, then the central server provides one, being
                # this NodeGate's id number
                self.public_configs.id = self.id
                self.public_configs.save()

        if not self.receiver.get_latest_messages(wait=False):
            # If emitter file does not exist, skip this node
            logger.info(f"Node {self.public_configs.id} emitter not reachable. Skipping for now.")
            return False

        if self.receiver.error is not None:
            # Handled by 'watch'
            logger.info(f"Node {self.public_configs.id} crashed. Skipping for now.")
            return False

        if self.receiver.doing is None:
            logger.debug(f"Node {self.public_configs.id} is not running. Skipping for now.")
            return False

        if self.receiver.doing != "run":
            logger.debug(f"Node {self.public_configs.id} is busy doing something. Skipping for now.")
            return False

        if self.ruleset is None:
            # The node has not produced any model yet if self.ruleset is None. No need to bother with self.last fetch
            # then, just get the ruleset.
            if self.public_configs.local_model_path.is_file():
                get_ruleset()
        else:
            # The node has already produced a model. So we only get its ruleset if the model is new. We know that by
            # checking the modification time of the node's ruleset file.
            if (
                self.public_configs.local_model_path.is_file()
                and self.public_configs.local_model_path.info()["mtime"] > self.last_fetch.timestamp()
            ):
                get_ruleset()
            else:
                logger.debug(f"Node {self.public_configs.id} has no new model. Skipping for now.")
                return False

        if len(reference_node_config.configs) == 0:
            reference_node_config.configs = deepcopy(self.public_configs.configs)

        if self.public_configs.central_model_path is None:
            # First time the node's model is catched : provide it the information about where it can read the central
            # model
            self.public_configs.central_model_path = central_model_path
            self.public_configs.save()

        return True


class CentralServer(Actor):
    """Implementation of the notion of central server in federated learning.

    It monitors changes in a given list of remote GCP directories, were nodes are expected to write their models.
    Upon changes of the model files, the central server downloads them. When enough model files are downloaded, the
    central server aggregates them to produce/update the central model, which is saved to a directory.
    "Enough" model is defined in the central server configuration (see `ifra.configs.CentralConfig`)

    Attributes
    ----------
    reference_node_config: NodePublicConfig
        Nodes linked to a given central server should have the same public configuration (except for their specific
        paths). This attribute remembers the said configuration, to ignore nodes with a wrong configuration.
    central_configs: CentralConfig
        see :class:~ifra.configs.CentralConfig
    nodes: List[NodeGate]
        List of all gates to the nodes the central server should monitor.
    ruleset: RuleSet
        Central server model's ruleset
    aggregation: Aggregation
        Instance of one of the `ifra.aggregations.Aggregation` daughter classes.
    """

    possible_aggregations = {"adaboost_aggregation": AdaBoostAggregation}
    """Possible string values and corresponding aggregation methods for *aggregation* attribute of
    `ifra.central_server.CentralServer`"""

    def __init__(
        self,
        nodes_configs: List[NodePublicConfig],
        central_configs: CentralConfig,
    ):
        """
        Parameters
        ----------
        nodes_configs: List[NodePublicConfig]
            List of each node's public configuration file
        central_configs: CentralConfig
            Central server configuration. See `ifra.configs.CentralConfig`
        """
        self.nodes_configs = None
        self.central_configs = None
        self.reference_node_config = None
        self.nodes = []
        self.ruleset = None
        self.aggregation = None
        super().__init__(nodes_configs=nodes_configs, central_configs=central_configs)

    @emit
    def create(self):

        self.nodes = []
        for config in self.nodes_configs:  # cast into set in order not to create twice the same node
            node = NodeGate(config)
            if node.ok:
                self.nodes.append(node)
            else:
                logger.warning("One node could not be instantiated. Ignoring it.")

        if self.central_configs.min_number_of_new_models < 2:
            raise ValueError("Minimum number of new nodes to trigger aggregation must be 2 or more")

        if len(self.nodes) < self.central_configs.min_number_of_new_models:
            raise ValueError(
                f"The minimum number of nodes to trigger aggregation "
                f"({self.central_configs.min_number_of_new_models}) is greater than the number of"
                f" available nodes ({len(self.nodes)}): that can not work !"
            )

        self.ruleset = None

        if self.central_configs.aggregation not in self.possible_aggregations:
            function = self.central_configs.aggregation.split(".")[-1]
            module = self.central_configs.aggregation.replace(f".{function}", "")
            self.aggregation = getattr(__import__(module, globals(), locals(), [function], 0), function)(self)
        else:
            self.aggregation = self.possible_aggregations[self.central_configs.aggregation](self)

    @emit
    def aggregate(self, rulesets: List[RuleSet]) -> Tuple[str, Union[RuleSet, None]]:
        """Aggregates rulesets in `ifra.central_server` *ruleset* using
        `ifra.central_server.CentralServer` *aggregation*

        Parameters
        ----------
        rulesets: List[RuleSet]
            New rulesets provided by the nodes.

        Returns
        -------
        Tuple[str, Union[RuleSet, None]]
            First item of the tuple can be :
              * "updated" if central model was updates
              * "pass" if no new rules were found but learning should continue
              * "stop" otherwise.
            The second item is the aggregated ruleset if the first item is "updated", None otherwise
        """
        return self.aggregation.aggregate(rulesets)

    @emit
    def ruleset_to_file(self) -> None:
        """Saves `ifra.node.Node` *ruleset* to `ifra.configs.CentralConfig` *central_model_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the ruleset using TableWriter.
        Does not do anything if `ifra.node.Node` *ruleset* is None
        """
        ruleset = self.ruleset

        iteration = 0
        name = self.central_configs.central_model_path.stem
        path = self.central_configs.central_model_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.central_configs.central_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.central_configs.central_model_path)

        try:
            path_table = path.with_suffix(".pdf")
            TableWriter(
                path_table, path.read(index_col=0).apply(lambda x: x.round(3) if x.dtype == float else x), paperwidth=30
            ).compile(clean_tex=True)
        except ValueError:
            logger.warning("Failed to produce tablewriter. Is LaTeX installed ?")

    @emit
    def run(self, timeout: int = 0, sleeptime: int = 5):
        """Monitors new changes in the nodes, every ''sleeptime'' seconds for ''timeout'' seconds, triggering
        aggregation and pushing central model to nodes when enough new node models are available.

        Stops if all nodes gave a model but no new rules could be found from them. Tells the nodes to stop too.

        Parameters
        ----------
        timeout: Optional[int]
            How many seconds should the watch last. If <= 0, will last until all nodes gave a model but no new rules
            could be found from them. Default value = 0.
        sleeptime: int
            How many seconds between each checks for new nodes models. Default value = 5.
        """
        t = time()
        updated_nodes = []
        learning_over = False
        iterations = 0

        if timeout <= 0:
            logger.warning("You did not specify a timeout for your run. It will last until manually stopped.")
        logger.info("")
        logger.info("Starting central server")

        while time() - t < timeout or timeout <= 0:

            if len(self.nodes) == 0:
                logger.warning("No nodes to learn on.")
                learning_over = True
                break

            new_models = False

            for node in self.nodes:
                # Node fetches its latest model from GCP. Returns True if a new model was found.
                if node.interact(self.reference_node_config, self.central_configs.central_model_path) is False:
                    continue

                updated_nodes.append(node)
                if len(updated_nodes) >= self.central_configs.min_number_of_new_models:
                    new_models = True

            if new_models:
                logger.info(f"Found enough ({len(updated_nodes)}) new nodes models. Triggering aggregation.")
                what_now = self.aggregate([node.ruleset for node in updated_nodes])
                if what_now == "stop":
                    learning_over = True
                    break
                elif what_now != "pass":
                    # Aggregation successfully updated central model.
                    updated_nodes = []

                    if self.ruleset is None:
                        raise ValueError("Should never happen !")
                    iterations += 1
                    self.ruleset_to_file()

            sleep(sleeptime)

        if learning_over is False:
            logger.info(f"Timeout of {timeout} seconds reached, stopping learning.")
        if self.ruleset is None:
            logger.warning("Learning failed to produce a central model. No output generated.")
        logger.info(f"Made {iterations} complete iterations between central server and nodes.")
        logger.info(f"Results saved in {self.central_configs.central_model_path}")

import traceback
from copy import deepcopy, copy
from datetime import datetime
from pathlib import Path
from time import time, sleep
from typing import List, Union, Tuple, Optional

from ruleskit import RuleSet
from tablewriter import TableWriter
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, CentralConfig
from .messenger import NodeMessenger
from .aggregations import AdaBoostAggregation

logger = logging.getLogger(__name__)


class NodeGate:

    """This class is the gate used by the central server to interact with the remote nodes. Mainly, it has in memory
    the public configuration of one given node, to which must correspond an instance of the
    `ifra.node.Node` class. This configuration holds among other things, the places where the nodes
    are supposed to save their models and where they will be looking for the central model. It implements a method,
    `ifra.central_server.NodeGate.interact`, that check whether the node produced a new model, and a method,
    `ifra.central_server.NodeGate.push_central_model`, to send the latest central model to the node

    Attributes
    ----------
    id: int
        Unique NodeGate id number. If the corresponding `ifra.node.Node` instance has not custom
        id, then it will use this one.
    public_config_path: TransparentPath
        Path to the node's public configuration file. The corresponding instance of the
         `ifra.node.Node` class must use the same file.
    public_configs: NodePublicConfig
        Node's public configuration. See `ifra.configs.NodePublicConfig`. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    messenger: NodeMessages
        Interface for exchanging messages with node. See `ifra.messenger.NodeMessages`
    ruleset: RuleSet
        Latest node model's ruleset, same as `ifra.node.Node` *ruleset*. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    last_fetch: datetime
        Last time the node produced a new model. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
    new_data: bool
        True if `ifra.central_server.NodeGate` *ruleset* is a ruleset not yet aggregated into the central server's
        model. False at first, modified by `ifra.central_server.NodeGate.interact` and
        `ifra.central_server.NodeGate.push_central_model`
    checked: bool
        True if the node was checked, whether it produced new data or not.
    """

    instances = 0

    def __init__(self, public_config_path: Union[str, TransparentPath]):
        """
        Parameters
        ----------
        public_config_path: Union[str, TransparentPath]
            Directory were node's public configuration file can be found
        """

        self.ok = False
        if type(public_config_path) == str:
            public_config_path = TransparentPath(public_config_path)

        self.id = NodeGate.instances
        try:
            self.public_configs = NodePublicConfig(public_config_path)
        except FileNotFoundError:
            logger.warning(f"No file found for node configuration {public_config_path}. Ignoring it.")
            return
        self.path_public_configs = self.public_configs.path
        self.path_messages = self.path_public_configs.parent / "messages.json"
        self.messenger = NodeMessenger(self.path_messages)

        if self.messenger.running:
            if self.public_configs.id is None:
                raise ValueError(f"Node {self.public_configs.id}'s messenger says it is running, but node's ID is"
                                 f"None. Delete the node's messages file {self.path_messages} and start again.")
            raise ValueError(f"Node {self.public_configs.id} was running before the central server started. Do not"
                             f"do that")
        self.messenger.reset_messages()

        self.ruleset = None
        self.last_fetch = None
        self.new_data = False
        self.checked = False
        self.checked_once = False
        self.ok = True

        NodeGate.instances += 1

    def interact(self):
        """Fetch the node's configuration if not done yet and the node's latest model if it is new compared to central
        server model. Sets `ifra.central_server.NodeGate` *public_configs*,
        `ifra.central_server.NodeGate` *ruleset*, `ifra.central_server.NodeGate` *last_fetch* to now
        and `ifra.central_server.NodeGate` *new_data* to True"""

        def get_ruleset():
            """Fetch the node's latest model's RuleSet.
            `ifra.central_server.NodeGate` *last_fetch* will be set to now and
            `ifra.central_server.NodeGate` *new_data* to True."""
            self.ruleset = RuleSet()
            self.ruleset.load(self.public_configs.local_model_path)
            self.last_fetch = datetime.now()
            self.new_data = True
            logger.info(f"Fetched new ruleset from node {self.id} at {self.last_fetch}")

        if self.new_data:
            # If we are here, then we already have self.ruleset matching the latest node model. It might not have
            # produced any new rules however, so the central server may not have pushed anything new to the node.
            # In that case, it is pointless to interact with the node.
            logger.info(f"Node {self.public_configs.id} already provided new data")
            return

        self.checked = False

        if self.checked_once is False:
            self.checked_once = True
            # First time the ndoe is interacted with
            if self.public_configs.id is None:
                # If the node was not given an id by its local user, then the central server provides one, being
                # this NodeGate's id number
                self.public_configs.id = self.id
                self.public_configs.save()

        self.messenger.get_latest_messages()

        if not self.messenger.running:
            logger.info(f"Node {self.id} is not running yet")
            return

        if self.messenger.fitting:
            logger.info(f"Node {self.id} is fitting.")
            return

        self.checked = True

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
                logger.info(f"Node {self.id} has no new model")

    def push_central_model(self, ruleset: RuleSet):
        """Pushes central model's ruleset to the node remote directory. This means the node's latest model was used in
        central model aggregation, and that the aggregation produced new learning information. In that case, the node's
        model is not new anymore, and `ifra.central_server.NodeGate` *new_data* is thus set to False.

        Parameters
        ----------
        ruleset: RuleSet
            The central model's ruleset
        """
        logger.info(f"Pushing central model to node {self.id} at {self.public_configs.central_model_path}")
        ruleset.save(self.public_configs.central_model_path)
        self.new_data = False


class CentralServer:
    """Implementation of the notion of central server in federated learning.

    It monitors changes in a given list of remote GCP directories, were nodes are expected to write their models.
    Upon changes of the model files, the central server downloads them. When enough model files are downloaded, the
    central server aggregates them to produce/update the central model, which is sent to each nodes' directories.
    "enough" model is defined in the central server configuration (see `ifra.configs.CentralConfig`)

    The aggregation method is that of AdaBoost. See `ifra.central_server.CentralServer.aggregate`

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

    possible_aggregations = {
        "adaboost_aggregation": AdaBoostAggregation
    }
    """Possible string values and corresponding aggregation methods for *aggregation* attribute of
    `ifra.central_server.CentralServer`"""

    def __init__(
        self,
        nodes_configs_paths: List[Union[str, TransparentPath]],
        central_configs_path: Union[str, Path, TransparentPath],
    ):
        """
        Parameters
        ----------
        nodes_configs_paths: List[TransparentPath]
            List of remote paths pointing to each node's public configuration file
        central_configs_path: Union[str, Path, TransparentPath]
            Central server configuration. See `ifra.configs.CentralConfig`
        """
        self.reference_node_config = None

        self.central_configs = CentralConfig(central_configs_path)
        self.nodes = []
        for path in set(nodes_configs_paths):  # cast into set in order not to create twice the same node
            node = NodeGate(path)
            if node.ok:
                self.nodes.append(node)

        if self.central_configs.min_number_of_new_models < 2:
            raise ValueError("Minimum number of new nodes to trigger aggregation must be 2 or more")

        if len(self.nodes) < self.central_configs.min_number_of_new_models:
            raise ValueError(f"The minimum number of nodes to trigger aggregation "
                             f"({self.central_configs.min_number_of_new_models}) is greater than the number of"
                             f" available nodes ({len(self.nodes)}): that can not work !")

        self.ruleset = None

        if self.central_configs.aggregation not in self.possible_aggregations:
            function = self.central_configs.aggregation.split(".")[-1]
            module = self.central_configs.aggregation.replace(f".{function}", "")
            self.aggregation = getattr(__import__(module, globals(), locals(), [function], 0), function)(self)
        else:
            self.aggregation = self.possible_aggregations[self.central_configs.aggregation](self)

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

    def ruleset_to_file(self) -> None:
        """Saves `ifra.node.Node` *ruleset* to `ifra.configs.CentralConfig` *output_path*,
        overwritting any existing file here, and in another file in the same directory but with a unique name.
        Will also produce a .pdf of the ruleset using TableWriter.
        Does not do anything if `ifra.node.Node` *ruleset* is None
        """
        ruleset = self.ruleset

        iteration = 0
        name = self.central_configs.output_path.stem
        path = self.central_configs.output_path.parent / f"{name}_{iteration}.csv"
        while path.is_file():
            iteration += 1
            path = self.central_configs.output_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.central_configs.output_path)

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

    def watch(self, timeout: int = 0, sleeptime: int = 5):
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
        try:
            logger.info("")
            logger.info("Starting central server")
            t = time()
            updated_nodes = []
            checked_nodes = []
            rulesets = []
            learning_over = False
            iterations = 0

            while time() - t < timeout or timeout <= 0:

                if len(checked_nodes) == len(self.nodes):
                    logger.info("All nodes were checked and none is fitting anything : no new data can be extracted."
                                "Stopping learning.")
                    learning_over = True
                    break
                new_models = False

                if len(self.nodes) == 0:
                    raise ValueError("No nodes to learn on !")

                for node in copy(self.nodes):
                    if node in updated_nodes:
                        continue
                    node.interact()  # Node fetches its latest data from GCP
                    if node.checked:
                        checked_nodes.append(node)

                    if node.messenger.error is not None:
                        logger.info(f"Removing crashed node {node.public_configs.id} from learning")
                        self.nodes.remove(node)
                        continue
                    if self.reference_node_config is None:
                        self.reference_node_config = deepcopy(node.public_configs)
                    # Nodes with configurations different from central's are ignored
                    elif node.public_configs != self.reference_node_config:
                        s = "\n"
                        for key in node.public_configs.configs:
                            if node.public_configs.configs[key] != self.reference_node_config.configs[key]:
                                s = f"{s}{key}: {node.public_configs.configs[key]}" \
                                    f" != {self.reference_node_config.configs[key]}\n"
                        logger.warning(
                            f"Node {node.id}'s configuration is not compatible with previous"
                            f" configuration. Differences are: {s}"
                        )
                        continue

                    if node.new_data:
                        updated_nodes.append(node)
                        rulesets.append(node.ruleset)
                        if len(updated_nodes) >= self.central_configs.min_number_of_new_models:
                            new_models = True

                if new_models:
                    logger.info(f"Found enough ({len(updated_nodes)}) new nodes nodels.")
                    what_now = self.aggregate(rulesets)
                    if what_now == "stop":
                        learning_over = True
                        break
                    elif what_now != "pass":
                        # Aggregation successfully updated central model. Forget the node's current model and push the
                        # new model to them.
                        rulesets = []
                        updated_nodes = []
                        checked_nodes = []

                        for node in self.nodes:
                            node.push_central_model(deepcopy(self.ruleset))

                        if self.ruleset is None:
                            raise ValueError("Should never happen !")
                        iterations += 1
                        self.ruleset_to_file()

                sleep(sleeptime)

            if learning_over is False:
                logger.info(f"Timeout of {timeout} seconds reached, stopping learning.")
            for node in self.nodes:
                node.messenger.stop = True
            if self.ruleset is None:
                logger.warning("Learning failed to produce a central model. No output generated.")
            logger.info(f"Made {iterations} complete iterations between central server and nodes.")
            logger.info(f"Results saved in {self.central_configs.output_path}")
        except Exception as e:
            for node in self.nodes:
                node.messenger.central_error = traceback.format_exc()
            raise e

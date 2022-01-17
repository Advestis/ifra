import traceback
from copy import deepcopy, copy
from datetime import datetime
from pathlib import Path
from time import time, sleep
from typing import List, Union, Tuple, Optional

from ruleskit import RuleSet
from transparentpath import TransparentPath
import logging

from .configs import NodePublicConfig, CentralConfig
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
    node_config_path: TransparentPath
        Path to the node's public configuration file. The corresponding instance of the
         `ifra.node.Node` class must use the same file.
    public_configs: NodePublicConfig
        Node's public configuration. See `ifra.configs.NodePublicConfig`. None at initialisation, set by
        `ifra.central_server.NodeGate.interact`
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
    """

    instances = 0

    def __init__(self, node_config_path: Union[str, TransparentPath]):
        """
        Parameters
        ----------
        node_config_path: Union[str, TransparentPath]
            Directory were node's public configuration file can be found
        """

        if type(node_config_path) == str:
            node_config_path = TransparentPath(node_config_path)

        self.id = NodeGate.instances
        self.public_config_path = node_config_path
        self.public_configs = None

        self.ruleset = None
        self.last_fetch = None
        self.new_data = False

        if (self.public_config_path.parent / "running").isfile():
            raise FileExistsError(f"File 'running' found at {self.public_config_path.parent} : node was started"
                                  f"before the central server. Do not do that.")
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
            return

        if self.public_configs is None:
            # If we are here, it means the node has not provided any configurations yet. Either because we interact for
            # the first time, or because last time we checked, there was no file at self.node_config_path. So need to
            # load the configuration, if available
            if not self.public_config_path.isfile():
                return

            # Load the configuration and reset any previous error that was in it
            self.public_configs = NodePublicConfig(self.public_config_path)
            self.public_configs.error = None
            self.public_configs.central_error = None
            if self.public_configs.id is None:
                # If the node was not given an id by its local user, then the central server provides one, being
                # this NodeGate's id number
                self.public_configs.id = self.id
            self.public_configs.save()

        if self.ruleset is None:
            # The node has not produced any model yet if self.ruleset is None. No need to bother with self.last fetch
            # then, just get the ruleset.
            if self.public_configs.local_model_path.isfile():
                get_ruleset()
        else:
            # The node has already produced a model. So we only get its ruleset if the model is new. We know that by
            # checking the modification time of the node's ruleset file.
            if (
                self.public_configs.local_model_path.isfile()
                and self.public_configs.local_model_path.info()["mtime"] > self.last_fetch.timestamp()
            ):
                get_ruleset()

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
        self.nodes = [NodeGate(path) for path in nodes_configs_paths]

        if len(self.nodes) < self.central_configs.min_number_of_new_models:
            raise ValueError(f"The minimum number of nodes to trigger aggregation "
                             f"({self.central_configs.min_number_of_new_models}) is greater than the number of"
                             f" available nodes ({len(self.nodes)}): that can not work !")

        self.ruleset = None

        if self.central_configs.aggregation not in self.possible_aggregations:
            s = f"Aggregation '{self.central_configs.aggregation}' is not known. Can be one of:"
            s = "\n".join([s] + list(self.possible_aggregations.keys()))
            raise ValueError(s)
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
        # noinspection PyBroadException
        try:
            logger.info("Starting central server")
            t = time()
            updated_nodes = []
            rulesets = []
            learning_over = False
            while time() - t < timeout or timeout <= 0:
                new_models = False

                if len(self.nodes) == 0:
                    raise ValueError("No nodes to learning on !")

                for node in copy(self.nodes):
                    if node in updated_nodes:
                        sleep(sleeptime)
                        continue
                    node.interact()  # Node fetches its latest data from GCP

                    if node.public_configs is not None:
                        if node.public_configs.error is not None:
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
                            sleep(sleeptime)
                            continue
                    # Nodes with no available configurations are ignored
                    else:
                        logger.warning(f"Node {node.id}'s has no configurations. Ignoring.")
                        sleep(sleeptime)
                        continue

                    if node.new_data:
                        updated_nodes.append(node)
                        rulesets.append(node.ruleset)
                        if len(updated_nodes) >= self.central_configs.min_number_of_new_models:
                            new_models = True

                if new_models:
                    logger.info("Found enough new nodes nodels.")
                    what_now = self.aggregate(rulesets)
                    if what_now == "stop":
                        learning_over = True
                        for node in self.nodes:
                            node.public_configs.stop = True
                            node.public_configs.save()
                        break
                    elif what_now == "pass":
                        # New model did not provide additionnal information. Keep the current rulesets, but do not
                        # trigger aggregation anymore until another node provides a model
                        sleep(sleeptime)
                        continue
                    # Aggregation successfully updated central model. Forget the node's current model and push the
                    # new model to them.
                    rulesets = []
                    updated_nodes = []
                    for node in self.nodes:
                        node.push_central_model(deepcopy(self.ruleset))
                    if self.ruleset is None:
                        raise ValueError("Should never happen !")
                    iteration = 0
                    name = self.central_configs.output_path.stem
                    path = self.central_configs.output_path.parent / f"{name}_{iteration}.csv"
                    while path.isfile():
                        iteration += 1
                        path = path.parent / f"{name}_{iteration}.csv"
                    self.ruleset.save(self.central_configs.output_path)
                    self.ruleset.save(path)
                sleep(sleeptime)

            if learning_over is False:
                logger.info(f"Timeout of {timeout} seconds reached, stopping learning.")
                for node in self.nodes:
                    node.public_configs.stop = True
                    node.public_configs.save()
            if self.ruleset is None:
                logger.warning("Learning failed to produce a central model. No output generated.")
            logger.info(f"Results saved in {self.central_configs.output_path}")
        except Exception as e:
            for node in self.nodes:
                node.public_configs.central_error = traceback.format_exc()
                node.public_configs.save()
            raise e

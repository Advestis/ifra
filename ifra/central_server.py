from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time, sleep
from typing import List, Union

from ruleskit import RuleSet
from transparentpath import TransparentPath
import logging

from ifra.configs import NodePublicConfig, CentralConfig

logger = logging.getLogger(__name__)


class NodeGate:

    """This class is the gate used by the central server to interact with the remote nodes. Mainly, it has in memory
    the public configuration of one given node, to which must correspond an instance of the
    :class:~ifra.central_server.node.Node class. This configuration holds among other things, the places where the nodes
    are supposed to save their models and where they will be looking for the central model. It implements a method,
    :func:~ifra.central_server.NodeGate.interact, that check whether the node produced a new model, and a method,
    :func:~ifra.central_server.NodeGate.push_central_model, to send the latest central model to the node

    Attributes
    ----------
    id: int
        Unique NodeGate id number. If the corresponding :class:~ifra.central_server.node.Node instance has not custom
        id, then it will use this one.
    node_config_path: TransparentPath
        Path to the node's public configuration file. The corresponding instance of the
         :class:~ifra.central_server.node.Node class must use the same file.
    learning_configs: NodePublicConfig
        Node's public configuration. See :class:~ifra.configs.NodePublicConfig. None at initialisation, set by
        :func:~ifra.central_server.NodeGate.interact
    ruleset: RuleSet
        Latest node model's ruleset, same as :attribute:~ifra.node.Node.ruleset. None at initialisation, set by
        :func:~ifra.central_server.NodeGate.interact
    last_fetch: datetime
        Last time the node produced a new model. None at initialisation, set by
        :func:~ifra.central_server.NodeGate.interact
    new_data: bool
        True if self.ruleset is a ruleset not yet aggregated into the central server's model. False at first, modified
        by :func:~ifra.central_server.NodeGate.interact and :func:~ifra.central_server.NodeGate.push_central_model

    Methods
    -------
    interact()
        Fetch the node's configuration if not done yet and the node's latest model if it is new compared to central
        server model. Sets :attribute:~ifra.central_server.NodeGate.learning_configs,
        :attribute:~ifra.central_server.NodeGate.ruleset, :attribute:~ifra.central_server.NodeGate.last_fetch to now
        and :attribute:~ifra.central_server.NodeGate.new_data to True
    push_central_model(ruleset: RuleSet)
        Pushes latest central model's ruleset to the node's directory. Sets
        :attribute:~ifra.central_server.NodeGate.new_data to False
    """

    instances = 0

    def __init__(self, node_config_path: TransparentPath):
        """
        Parameters
        ----------
        node_config_path: TransparentPath
            Directory were node's public configuration file can be found
        """
        if not isinstance(node_config_path, TransparentPath):
            raise TypeError(f"node_config_path should be a transparentpath, got {type(node_config_path)} instead")

        self.id = NodeGate.instances
        self.node_config_path = node_config_path
        self.learning_configs = None

        self.ruleset = None
        self.last_fetch = None
        self.new_data = False
        NodeGate.instances += 1

    def interact(self):
        """Fetch the node's configuration if not done yet and the node's latest model if it is new compared to central
        server model. Sets :attribute:~ifra.central_server.NodeGate.learning_configs,
        :attribute:~ifra.central_server.NodeGate.ruleset, :attribute:~ifra.central_server.NodeGate.last_fetch to now
        and :attribute:~ifra.central_server.NodeGate.new_data to True"""
        def get_ruleset():
            """Fetch the node's latest model's RuleSet.
            :attribute:~ifra.central_server.NodeGate.last_fetch will be set to now and
            :attribute:~ifra.central_server.NodeGate.new_data to True."""
            self.ruleset = RuleSet()
            self.ruleset.load(self.learning_configs.local_model_path)
            self.last_fetch = datetime.now()
            self.new_data = True
            logger.info(f"Fetched new ruleset from node {self.id} at {self.last_fetch}")

        if self.new_data:
            # If we are here, then we already have self.ruleset matching the latest node model. It might not have
            # produced any new rules however, so the central server may not have pushed anything new to the node.
            # In that case, it is pointless to interact with the node.
            return

        if self.learning_configs is None:
            # If we are here, it means the node has not provided any configurations yet. Either because we interact for
            # the first time, or because last time we checked, there was no file at self.node_config_path. So need to
            # load the configuration, if available
            if not self.node_config_path.isfile():
                return
            self.learning_configs = NodePublicConfig(self.node_config_path)
            if self.learning_configs.id is None:
                # If the node was not given an id by its local user, then the central server provides one, being
                # this NodeGate's id number
                self.learning_configs.id = self.id
                self.learning_configs.save()

        if self.ruleset is None:
            # The node has not produced any model yet if self.ruleset is None. No need to bother with self.last fetch
            # then, just get the ruleset.
            if self.learning_configs.local_model_path.isfile():
                get_ruleset()
        else:
            # The node has already produced a model. So we only get its ruleset if the model is new. We know that by
            # checking the modification time of the node's ruleset file.
            if (
                self.learning_configs.local_model_path.isfile()
                and self.learning_configs.local_model_path.info()["mtime"] > self.last_fetch.timestamp()
            ):
                get_ruleset()

    def push_central_model(self, ruleset: RuleSet):
        """Pushes central model's ruleset to the node remote directory. This means the node's latest model was used in
        central model aggregation, and that the aggregation produced new learning information. In that case, the node's
        model is not new anymore, and :attribute:~ifra.central_server.NodeGate.new_data is thus set to False.

        Parameters
        ----------
        ruleset: RuleSet
            The central model's ruleset
        """
        logger.info(f"Pushing central model to node {self.id} at {self.learning_configs.central_model_path}")
        ruleset.save(self.learning_configs.central_model_path)
        self.new_data = False


class CentralServer:
    """Implementation of the notion of central server in federated learning.

    It monitors changes in a given list of remote GCP directories, were nodes are expected to write their models.
    Upon changes of the model files, the central server downloads them. When enough model files are downloaded, the
    central server aggregates them to produce/update the central model, which is sent to each nodes' directories.
    "enough" model is defined in the central server configuration (see :class:~ifra.configs.CentralConfig)

    The aggregation method is that of AdaBoost. See :func:~ifra.central_server.CentralServer.aggregate

    Attributes
    ----------
    reference_node_config: NodePublicConfig
        Nodes linked to a given central server should have the same public configuration (except for their specific
        paths). This attribute remembers the said configuration, to ignore nodes with a wrong configuration.
    central_configs: CentralConfig
        see :class:~ifra.configs.CentralConfig
    nodes: List[NodeGate]
        List of all gates to the nodes the central server should monitor.
    rulesets: List[RuleSet]
        New rulesets provided by the nodes. When there are enough new rulesets, they are used for aggregation and this
        list is emptied.
    ruleset: RuleSet
        Central server model's ruleset

    Methods
    -------
    _aggregate()
        Aggregate rulesets present in :attribute:~ifra.central_server.CentralServer.rulesets using AdaBoost logic
        and updates :attribute:~ifra.central_server.CentralServer.ruleset
    watch(timeout: int = 60, sleeptime: int = 5)
        Monitors new changes in the nodes, every ''sleeptime'' seconds for ''timeout'' seconds, triggering aggregation
        and pushing central model to nodes when enough new node models are available.
        This is the only method the user should call.
    """

    def __init__(
        self,
        nodes_configs_paths: List[TransparentPath],
        central_configs_path: Union[str, Path, TransparentPath]
    ):
        """
        Parameters
        ----------
        nodes_configs_paths: List[TransparentPath]
            List of remote paths pointing to each node's public configuration file
        central_configs_path: Union[str, Path, TransparentPath]
            Central server configuration. See :class:~ifra.configs.CentralConfig
        """
        self.reference_node_config = None

        if type(central_configs_path) == str:
            central_configs_path = Path(central_configs_path)
        self.central_configs = CentralConfig(central_configs_path)
        self.nodes = [NodeGate(path) for path in nodes_configs_paths]
        self.rulesets = []
        self.ruleset = None

    def _aggregate(self) -> str:
        """Among all rules generated by the nodes in the current iteration, will keep only the most recurent rule(s).

        Those rules will be added to :attribute:~ifra.central_server.CentralServer.ruleset, itself passed to the nodes,
        and the points in each node's data activated by it will be ignored in order to find other relevant rules.
        In order not to remove too many points, a maximum of coverage is imposed, see :class:~ifra.configs.CentralConfig

        If among the rules extracted from the nodes, none are new compared to the current state of the model, nodes are
        not updated. If no new rules are found but each nodes had new data, learning is finished.

        Returns
        -------
        str
            "updated" if central model was updates, "pass" if no new rules were found but learning should continue,
            "stop" otherwise.
        """
        logger.info("Aggregating fit results...")
        all_rules = []
        for ruleset in self.rulesets:
            all_rules += ruleset.rules
        if len(all_rules) == 0:
            logger.info("... no rules found")
            if len(self.rulesets) == len(self.nodes):
                logger.info("No new rules were found, despite all nodes having provided a new model : learning is over")
                return "stop"
            return "pass"
        occurences = {r: all_rules.count(r) for r in set(all_rules) if r.coverage < self.central_configs.max_coverage}
        if len(occurences) == 0:
            logger.warning("No rules matched coverage criterion")
            if len(self.rulesets) == len(self.nodes):
                logger.info("No new rules were found, despite all nodes having provided a new model : learning is over")
                return "stop"
            return "pass"
        max_occurences = max(list(occurences.values()))
        if self.ruleset is None:
            self.ruleset = RuleSet(
                [r for r in occurences if occurences[r] == max_occurences],
                remember_activation=False,
                stack_activation=False,
            )
        else:
            new_rules = [r for r in occurences if occurences[r] == max_occurences and r not in self.ruleset]
            if len(new_rules) == 0:
                logger.warning("No new rules found")
                if len(self.rulesets) == len(self.nodes):
                    logger.info("No new rules were found, despite all nodes having provided a new model"
                                " : learning is over")
                    return "stop"
                return "pass"
            self.ruleset += RuleSet(
                new_rules,
                remember_activation=False,
                stack_activation=False,
            )
        logger.info("... fit results aggregated")
        return "updated"

    def watch(self, timeout: int = 60, sleeptime: int = 5):
        t = time()
        updated_nodes = []
        learning_over = False
        while time() - t < timeout:
            new_models = False

            for node in self.nodes:
                if node in updated_nodes:
                    continue
                node.interact()  # Node fetches its latest data from GCP

                if node.learning_configs is not None:
                    if self.reference_node_config is None:
                        self.reference_node_config = deepcopy(node.learning_configs)
                    # Nodes with configurations different from central's are ignored
                    elif node.learning_configs != self.reference_node_config:
                        logger.warning(
                            f"Node {node.id}'s configuration is not compatible with previous"
                            " configuration. Ignoring."
                        )
                        continue
                # Nodes with no available configurations are ignored
                else:
                    logger.warning(f"Node {node.id}'s has no configurations. Ignoring.")
                    continue

                if node.new_data:
                    updated_nodes.append(node)
                    self.rulesets.append(node.ruleset)
                    if len(updated_nodes) >= self.central_configs.min_number_of_new_models:
                        new_models = True

            if new_models:
                logger.info("Found enough new nodes nodels.")
                what_now = self._aggregate()
                if what_now == "stop":
                    learning_over = True
                    break
                elif what_now == "pass":
                    # New model did not provide additionnal information. Keep the current rulesets, but do not trigger
                    # aggregation anymore until another node provides a model
                    continue
                # Aggregation successfully updated central model. Forget the node's current model and push the new model
                # to them.
                self.rulesets = []
                updated_nodes = []
                for node in self.nodes:
                    node.push_central_model(deepcopy(self.ruleset))
                if self.ruleset is None:
                    raise ValueError("Should never happeb !")
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
        if self.ruleset is None:
            logger.warning("Learning failed to produce a central model. No output generated.")
        logger.info(f"Results saved in {self.central_configs.output_path}")

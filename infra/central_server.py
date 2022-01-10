from typing import Union, List

from ruleskit import RuleSet

from .node import Node
from pathlib import Path
from transparentpath import TransparentPath
import logging
from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class CentralServer:

    """Central server. Can create nodes, pass learning_configs to them, launch their learning and gather their results.
    """

    # If True, calling 'fit' of each node is done in parallel.
    PARALLEL_NODES = False

    def __init__(
        self,
        learning_configs_path: Union[str, Path, TransparentPath] = None,
        nodes: Union[List[Node], None] = None,
        nnodes: Union[None, int] = None,
    ):
        """Must specify one of learning_configs_path and nodes.

        If learning_configs_path, will create nnodes using the given learning configurations
        Else, if nodes is passed, assumes that they are compatible and will get the learning configuration of the first
        node.
        """
        if nodes is not None:
            if not isinstance(nodes, list):
                raise TypeError(f"Node argument must be a list, got {type(nodes)}")
            for i, node in enumerate(nodes):
                if not isinstance(node, Node):
                    raise TypeError(f"Node {i} in nodes is not of class Node, but {type(node)}")
            if nnodes is not None:
                logger.warning("Nodes were given to central server, so number of nodes to create will be ignored")

        if learning_configs_path is None and nodes is None:
            ValueError("At least one of learning_configs_path and nodes must be specified")

        if nnodes is not None:
            if not isinstance(nnodes, int):
                raise TypeError(f"nnodes should be a integer, got a {type(nnodes)}")

        self.nodes = nodes
        if learning_configs_path is not None:
            if type(learning_configs_path) == str:
                learning_configs_path = Path(learning_configs_path)

            self.learning_configs_path = learning_configs_path
            if self.nodes is not None:
                for node in self.nodes:
                    node.learning_configs_path = learning_configs_path
            else:
                for i in range(nnodes):
                    self.add_node()
        else:
            if len(self.nodes) == 0:
                raise ValueError("If learning_configs_path is not specified, nodes must not be empty")
            self.learning_configs_path = nodes[0].learning_configs_path
            for node in self.nodes[1:]:
                node.learning_configs_path = self.learning_configs_path

        self.trees, self.rulesets = [], []
        self.ruleset = None

    def add_node(self, node: Union[Node] = None):
        """Add the node to self, or creates a new one if node is None"""
        if node is not None:
            if not isinstance(node, Node):
                raise TypeError(f"Node is not of class Node, but {type(node)}")
            node.learning_configs_path = self.learning_configs_path
        else:
            node = Node(learning_configs_path=self.learning_configs_path)
        self.nodes.append(node)

    def aggregate(self):
        """TO CODE"""
        logger.info("Aggregating fit results...")
        self.ruleset = RuleSet()
        logger.info("... fit results aggregated")

    def fit(self, niterations: int, make_images: bool = True, save_trees: bool = True, save_rulesets: bool = True):
        logger.info("Started fit...")
        if len(self.nodes) == 0:
            logger.warning("Zero nodes. Fitting canceled.")
            return

        features_names = self.nodes[0].learning_configs.features_names
        for iteration in range(niterations):
            if CentralServer.PARALLEL_NODES:
                with ProcessPoolExecutor(max_workers=cpu_count(), mp_context=get_context("spawn")) as executor:
                    results = list(
                        executor.map(Node.fit, self.nodes)
                    )
            else:
                results = []
                for node in self.nodes:
                    results.append(node.fit())
            for i, tree_ruleset in enumerate(results):
                tree, ruleset = tree_ruleset
                self.trees.append(tree)
                self.rulesets.append(ruleset)
                if make_images:
                    Node.tree_to_graph(f"tests/outputs/node_{i}/tree.dot", None, tree, features_names)
                if save_trees:
                    Node.tree_to_joblib(f"tests/outputs/node_{i}/tree.joblib", None, tree)
                if save_rulesets:
                    Node.rule_to_file(f"tests/outputs/node_{i}/ruleset.csv", None, ruleset)
            self.aggregate()
            for node in self.nodes:
                node.update_from_central(self.ruleset)
        logger.info("...fit finished")

from typing import Union, List

from ruleskit import RuleSet

from .node import Node
from pathlib import Path
import logging
from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class CentralServer:

    """Central server. Can create nodes, pass learning_configs to them, launch their learning and gather their results.
    """

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs_path: Union[str, Path, "TransparentPath"] = None,
        nodes: Union[List[Node], None] = None,
        nnodes: Union[None, int] = None,
    ):
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

    def fit(self, niterations: int):
        for iteration in range(niterations):
            with ProcessPoolExecutor(max_workers=cpu_count(), mp_context=get_context("spawn")) as executor:
                results = list(
                    executor.map(Node.fit, self.nodes)
                )
            for tree, ruleset in results:
                self.trees.append(tree)
                self.rulesets.append(ruleset)
            self.aggregate()
            for node in self.nodes:
                node.update_from_central(self.ruleset)

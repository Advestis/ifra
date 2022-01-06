from typing import Union, List
from .node import Node


class CentralServer:

    def __init__(self, nodes: Union[List[Node], None] = None):
        if nodes is not None:
            if not isinstance(nodes, list):
                raise TypeError(f"Node argument must be a list, got {type(nodes)}")
            for i, node in enumerate(nodes):
                if not isinstance(node, Node):
                    raise TypeError(f"Node {i} in nodes is not of class Node, but {type(node)}")
        self.nodes = nodes

    def add_node(self, node: Node):
        if not isinstance(node, Node):
            raise TypeError(f"Node is not of class Node, but {type(node)}")
        self.nodes.append(node)

"""Interpretable Federated Rule Algorithm"""

from .node import Node
from .central_server import CentralServer

try:
    from ._version import __version__
except ImportError:
    pass

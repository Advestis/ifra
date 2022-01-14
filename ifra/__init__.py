"""Interpretable Federated Rule Algorithm

**Federated Learning** allows several sources (*nodes*) of data to contribute to a single model without sharing their
data between them. It thus answers the problematic of data confidentiality. All federated learning algorithm follow
those steps:\n
    1. Choosing a model to train\n
    2. Having each node training its own model\n
    3. Each node sends its model to a central server\n
    4. The central server aggregates all models into a single one, and send it back to each node which take it into
    account for their next learning phase\n
Steps 2 to 4 are then repeated until either the user stops the algorithm or some threshold is reached.

In IFRA, the model to train produces rules as learning output. Rules are managed by the `ruleskit` pacakge. The
available learning models are:\n
  * decisiontree (see `ifra.fitter.DecisionTreeFitter`)`\n
The user has the liberty to choose the aggregation method among:\n
  * adaboost (see `ifra.aggregations.adaboost_aggregation`)\n
The user has the liberty to choose the update method to be used by the node to take the central model into account.
The available updaters are:\n
  * adaboost_updater (see `ifra.updaters.AdaboostUpdater`)\n

To preserve data separation between each node and between the central server, IFRA assumes that each node has a
dedicated GCS directory where it will send its model, and where it will look for the central model. GCS paths are
handeled by the `transparentpath` package. For testing purposes, those paths can be local if no global filesystem
if set by transparentpath.
"""

from .node import Node
from .central_server import CentralServer

try:
    from ._version import __version__
except ImportError:
    pass

# noinspection PyUnresolvedReferences
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
  * decisiontree_fitter (see `ifra.fitters.DecisionTreeFitter`)\n
The user has the liberty to choose the aggregation method among:\n
  * adaboost_aggregation (see `ifra.aggregations.AdaBoostAggregation`)\n
The user has the liberty to choose the update method to be used by the node to take the central model into account.
The available updaters are:\n
  * adaboost_updater (see `ifra.updaters.AdaboostUpdater`)\n
Each node has the possbility to execute a dataprep before the first learning. The user has the liberty to choose the
dataprep method. The available datapreps are:\n
  * binfeatures_dataprep (see `ifra.datapreps.BinFeaturesDataPrep`)\n

In addition to the available objects listed above, the user can define its own by overloading the `ifra.fitters.Fitter`,
`ifra.aggregations.Aggregation`, `ifra.updaters.Updater` and `ifra.datapreps.DataPrep` classes. Read their
documentation to know how.

To preserve data separation between each node and between the central server, IFRA assumes that each node has a
dedicated GCS directory where it will send its model, and where it will look for the central model. GCS paths are
handeled by the `transparentpath` package. For testing purposes, those paths can be local if no global filesystem
is set by transparentpath.

To use IFRA, you need to do 5 things:\n
  1. Define the nodes public configurations in json files. A given configuration file must be reachable by both the
  node it talks about and the central server. See `ifra.configs.NodePublicConfig` for more information.\n
  2. Define the nodes data configurations in json files. It should NOT be reachable by the central server.
  See `ifra.configs.NodeDataConfig` for more information.\n
  3. Define the central server configuration in a json file. This file needs to be reachable by the central server only.
  see `ifra.configs.CentralConfig` for more information.\n
  4. On each entity that should be a node, instantiate the `ifra.node.Node` class by providing it with its public and
  data configuration, and call its ifra.node.Node.watch` method.
  5. On the entity that should act as the central server, instantiate the `ifra.central_server.CentralServer` class by
  providing it with the list of all nodes configuration paths and its central configuration, then call its
  `ifra.central_server.CentralServer.watch` method.

Example of step 1: you could create nodes public configuration json files contaning:
>>> {
>>>   "features_names": ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"],
>>>   "classes_names": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
>>>   "x_mins": "",
>>>   "x_maxs": "",
>>>   "max_depth": 3,
>>>   "remember_activation": false,
>>>   "stack_activation": false,
>>>   "plot_data": true,
>>>   "get_leaf": false,
>>>   "local_model_path": "tests/outputs/node_0/ruleset.csv",
>>>   "central_model_path": "tests/outputs/ruleset.csv",
>>>   "dataprep": "binfeatures_dataprep",
>>>   "id": "",
>>>   "fitter": "decisiontree_fitter",
>>>   "updater": "adaboost_updater"
>>> }

Example of step 2: you could create nodes data configuration json files contaning:
>>> {
>>>   "x": "tests/data/node_0/x.csv",
>>>   "y": "tests/data/node_0/y.csv",
>>>   "x_read_kwargs": {"index_col": 0},
>>>   "y_read_kwargs": {"index_col": 0},
>>>   "x_fs": "local",
>>>   "y_fs": "local"
>>> }

Example of step 3: you could create a central configuration json file contaning:
>>> {
>>>   "max_coverage": 0.25,
>>>   "output_path": "tests/outputs/ruleset.csv",
>>>   "output_path_fs": "local",
>>>   "min_number_of_new_models": 2,
>>>   "aggregation": "adaboost_aggregation"
>>> }

Example of step 4:
>>> from ifra import Node
>>> public_config_path = "tests/node_0/public_configs.json"
>>> data_config_path = "tests/node_0/data_configs.json"
>>> thenode = Node(path_public_configs=public_config_path, path_data=data_config_path)
>>> thenode.watch()

Example of step 5:
>>> from ifra import CentralServer
>>> nodes_public_config = [
>>>   "tests/node_0/public_configs.json",
>>>   "tests/node_1/public_configs.json",
>>>   "tests/node_2/public_configs.json",
>>>   "tests/node_3/public_configs.json"
>>> ]
>>> central_config_path = "tests/central_configs.json"
>>> server = CentralServer(nodes_configs_paths=nodes_public_config, central_configs_path=central_config_path)
>>> server.watch()
"""

from .node import Node
from .central_server import CentralServer
from .datapreps import DataPrep
from .updaters import Updater
from .fitters import Fitter
from .aggregations import Aggregation
from .setup_logger import setup_logger

try:
    from ._version import __version__
except ImportError:
    pass

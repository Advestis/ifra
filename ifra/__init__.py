# noinspection PyUnresolvedReferences
"""Interpretable Federated Rule Algorithm

**Federated Learning** allows several sources (*nodes*) of data to contribute to a single model without sharing their
data between them. It thus answers the problematic of data confidentiality. All federated learning algorithm follow
those steps:

  1. Choosing a model to train\n
  2. Having each node training its own model\n
  3. Aggregate the models into a single one and share it with the nodes\n
  4. Nodes produce a new model taking the central model into account for their next learning phase. In addition, in IFRA
  the node will also produce a new model if its data is uptdated.\n

Steps 2 to 4 are an 'iteration' of the learning and are repeated until either the user stops the algorithm or some
threshold is reached. In IFRA, no thresholds exist, and the learning stops when the user(s) decide so.

Differential privacy is automatically applied in each node by slightly changing the prediction of the rules
(Only working for regression rule for now).

IFRA consists of 3 independent and asynchronous actors, represented by the abstract class `ifra.actor.Actor`.
Each can be run individually and monitors changes in its inputs. The three actors are :

  * Nodes (**inputs**: data, central model, **output**: node model) (`ifra.node.Node`)
  * Aggregator (**input**: node models, **output**: aggregated model) (`ifra.aggregator.Aggregator`)
  * Central Server (**input**: aggregated model, **output**: central model) (`ifra.central_server.CentralServer`)

This architecture is bug resilient, since if a node is down it can just be restarted without impacting the other actors.
The difference between 'aggregated model' and 'central model' is that the central model will remember all the rules
learned from all previous iterations, while the aggregated model only know the rules of the current iteration.

In IFRA, nodes are anonymous : they all write their model to the same directory that is monitored by the aggregator, and
the aggregator does not have access to any information about the node. It does not know which node produced which model.

In IFRA, one node produces one ruleskit.RuleSet object. Each user is free to define its own model by
overloading the `ifra.fitters.Fitter` class, as long as it produces a ruleskit.RuleSet object.
The available models currently are:

  * decisiontreeregression (see `ifra.fitters.DecisionTreeRegressionFitter` for details)\n
  * decisiontreeclassification (see `ifra.fitters.DecisionTreeClassificationFitter` for details)

The user also has the liberty to define its own aggregation method, by overloading `ifra.aggregation.Aggregation`.
The available aggregation methods currently are:

  * adaboost (see `ifra.aggregations.AdaBoostAggregation` for details)\n
  * reverseadaboost (see `ifra.aggregations.ReverseAdaBoostAggregation` for details)\n
  * keepall (see `ifra.aggregations.AggregateAll`)

The user can implement the update method to be used by the node to take the central model into account by overloading
the `ifra.node_model_updaters.NodeModelUpdater`. The available node updaters currently are:

  * adaboost (adapted for adaboost, reverseadaboost and keepall aggregations)
  (see `ifra.node_model_updaters.AdaBoostNodeModelUpdater`)

Each node has the possbility to execute a dataprep before the first learning. The user has the liberty to define its
dataprep method by overloading the `ifra.datapreps.DataPrep` class. The available datapreps currently are:

  * binfeatures (see `ifra.datapreps.BinFeaturesDataPrep`)

To overload a class, read its documentation. Then, to make the actors use your class specify them in the
actor's configuration file (`ifra.config.NodeLearningConfig` for *fitter*, *node updater* and *dataprep*) or in
`ifra.config.AggregatorConfig` json file for *aggregation*).

To be correctly imported, the line passed in the json for, let's say, the aggregation, must be like

>>> {
>>>    ...
>>>    "aggregation": "some.importable.AggregationClass"
>>>    ...
>>> }
where *some.importable.AggregationClass* can be imported from the current working directory.

To preserve data separation, each actor should be ran on a different machine. Models are shared across actors through
Google Cloud Storage, and it is the reponsability of the user to define buckets to store the models the the appropriate
I/O rights for each actors.

  * Each node should have read access the one unique bucket where the central model is to be written. In addition, each
    node should have a read access its own data. No other actor should have read access to them. It should also have
    write access to a place where the nodes models will be written. The place must be identical for each node of a
    given learning.
  * The aggregator should have read access to the place where nodes are expected to write their models, and write
    access to a place dedicated for the aggregated model
  * The central server should have read access to the aggregated model, and write access to the place where the central
    model is read by the nodes.

GCS paths are handeled by transparentpath.TransparentPath objects. For testing purposes, those paths can be local if no
global filesystem is set by transparentpath.TransparentPath.

To use IFRA, you need to do 5 things:

  1. Define the nodes learning configurations in json files. It should be stored on GCS with read access by the node
     only (set the appropriate access to the service accounts). See `ifra.configs.NodeLearningConfig` for more
     information.\n
  2. Define the nodes data configurations in json files. It should be reachable by the node only.
     See `ifra.configs.NodeDataConfig` for more information.\n
  3. Define the aggregator configuration in a json file. This file needs to be reachable by the aggregator only.
     see `ifra.configs.AggregatorConfig` for more information.\n
  4. Define the central server configuration in a json file. This file needs to be reachable by the central server only.
     see `ifra.configs.CentralConfig` for more information.\n
  5. On each node machine, instantiate the `ifra.node.Node` class by providing it with its learning and
     data configuration, and call the `ifra.node.Node.run` method.
  6. On the machine that should act as the aggregator, instantiate the `ifra.aggregator.Aggregator` class by
     providing it with the list of all nodes configuration and its aggregator configuration, then call its
     `ifra.aggregator.Aggregator.run` method.
  7. On the machine that should act as the central server, instantiate the `ifra.central_server.CentralServer` class by
     providing it with the its central configuration, then call its `ifra.central_server.CentralServer.run` method.

Step 5, 6 and 7 can be done in any order.

Example of step 1: you could create nodes learning configuration json files contaning.

>>> {
>>>     "features_names": ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"],
>>>     "classes_names": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],  # Use an empty list if using regression
>>>     "x_mins": "",
>>>     "x_maxs": "",
>>>     "max_depth": 3,
>>>     "plot_data": true,
>>>     "get_leaf": false,
>>>     "node_models_path": "gs://bucket_node_models",
>>>     "node_models_path_fs": "gcs",
>>>     "central_model_path": "gs://bucket_central_server/ruleset.csv",
>>>     "central_model_path_fs": "gcs",
>>>     "id": "node_name",
>>>     "dataprep": "binfeatures",
>>>     "dataprep_kwargs": {"nbins": 5, "bins": {}, "save_bins": "gs://bucket_node_0/bins.json"},
>>>     "fitter": "decisiontreeclassification",
>>>     "fitter_kwargs": {},
>>>     "updater": "adaboost",
>>>     "updater_kwargs": {},
>>>     "train_test_split": "",
>>>     "thresholds_path": "gs://bucket_node_0/thresholds.json",
>>>     "thresholds_path_fs": "gcs",
>>>     "emitter_path": "gs://bucket_node_0_messages/node_chien_messages.json",
>>>     "emitter_path_fs": "gcs",
>>>     "eval_kwargs": {"criterion_method": "success_rate"},
>>>     "privacy_proba": 0.3
>>> }

See `ifra.configs.NodeLearningConfig` for information about those configurations.


Example of step 2: you could create nodes data configuration json files contaning:

>>> {
>>>   "x_path": "gs://bucket_node_0/x.csv",
>>>   "y_path": "gs://bucket_node_0/y.csv",
>>>   "x_read_kwargs": {"index_col": 0},
>>>   "y_read_kwargs": {"index_col": 0},
>>>   "x_path_fs": "gcs",
>>>   "y_path_fs": "gcs"
>>> }

See `ifra.configs.NodeDataConfig` for information about those configurations.

Example of step 3: you could create a aggragator configuration json file contaning:

>>> {
>>>     "node_models_path": "gs://bucket_node_models",
>>>     "node_models_path_fs": "gcs",
>>>     "aggregated_model_path": "gs://bucket_aggregated_model/ruleset.csv",
>>>     "aggregated_model_path_fs": "gcs",
>>>     "emitter_path": "gs://bucket_aggregator_messages/aggregator_messages.json",
>>>     "emitter_path_fs": "gcs",
>>>     "min_number_of_new_models": 2,
>>>     "aggregation": "adaboost",
>>>     "aggregation_kwargs": {},
>>>     "weight": "",
>>>     "best": ""
>>> }

See `ifra.configs.AggretatorConfig` for information about those configurations.

Example of step 4: you could create a central server configuration json file contaning:

>>> {
>>>     "central_model_path": "gs://bucket_central_model/ruleset.csv",
>>>     "central_model_path_fs": "gcs",
>>>     "aggregated_model_path": "gs://bucket_aggregated_model/ruleset.csv",
>>>     "aggregated_model_path_fs": "gcs",
>>>     "emitter_path": "gs://bucket_central_messages/central_messages.json",
>>>     "emitter_path_fs": "local"
>>> }

See `ifra.configs.CentralConfig` for information about those configurations.

Example of step 5:

>>> from ifra import Node, NodeLearningConfig, NodeDataConfig
>>> from transparentpath import Path
>>> learning_config_path = NodeLearningConfig(Path("gs://bucket_node_learning_configs/learning_configs_0.json"))
>>> data_config_path = NodeDataConfig(Path("data_configs.json", fs="local"))
>>> thenode = Node(learning_configs=path_learning, data=path_data)
>>> thenode.run()

Example of step 6:

>>> from ifra import Aggregator, NodeLearningConfig, AggregatorConfig
>>> nodes_learning_config = [
>>>   NodeLearningConfig(Path("gs://bucket_node_learning_configs/learning_configs_0.json"))
>>>   NodeLearningConfig(Path("gs://bucket_node_learning_configs/learning_configs_1.json"))
>>>   NodeLearningConfig(Path("gs://bucket_node_learning_configs/learning_configs_2.json"))
>>>   NodeLearningConfig(Path("gs://bucket_node_learning_configs/learning_configs_3.json"))
>>> ]
>>> aggregator_config = AggregatorConfig(Path("aggregator_configs.json", fs="local"))
>>> aggr = Aggregator(nodes_configs=nodes_learning_config, aggregator_configs=aggregator_config)
>>> aggr.run()

Example of step 7:

>>> from ifra import CentralServer, CentralConfig
>>> central_config = CentralConfig(Path("central_configs.json", fs="local"))
>>> server = CentralServer(central_configs=central_config)
>>> server.run()

The 'emitter_path' configuration present for each actor will be used in a futur update to monitor what each actor is
doing.
"""

from .node import Node
from .central_server import CentralServer
from .aggregator import Aggregator
from .datapreps import DataPrep
from .node_model_updaters import NodeModelUpdater
from .fitters import Fitter
from .aggregations import Aggregation
from .setup_logger import setup_logger
from .configs import AggregatorConfig, CentralConfig, NodeLearningConfig, NodeDataConfig

try:
    from ._version import __version__
except ImportError:
    pass

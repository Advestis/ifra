from copy import copy, deepcopy
from pathlib import Path
from time import sleep, time
from typing import Union, Optional
from json import load

from transparentpath import TransparentPath


def handle_path(path, fs):
    if fs is not None:
        return TransparentPath(path, fs=fs)
    else:
        return TransparentPath(path)


class Config:

    # noinspection PyUnresolvedReferences
    """Abstract class to load a configuration. Basically just a wrapper around a json file.

    This class should be overloaded and not used as-is, for only keys present in
    `Config.EXPECTED_CONFIGS` and `Config.ADDITIONNAL_CONFIGS` will
    be accepted from the json file, attribute that is empty in this abstract class. Any key present in
    `Config.EXPECTED_CONFIGS` must be present in the json file. Any key present in
    `Config.ADDITIONNAL_CONFIGS` can be present in the file.

    The json file can be on GCP, the use of transparentpath is supported.

    Attributes
    ----------
    path: TransparentPath
        The path to the json file containing the configuration. Will be casted into a transparentpath if a str is
        given, so make sure that if you pass a string that should be local, you did not set a global file system.
    configs: dict
        Content of the json file. Any pair of key value present in it will be seen as an attribute of the class
        instance and can be accessed by doing

        >>> conf = MyConfig(path)
        >>> conf.some_key

        assuming *MyConfig* overloads this class and *path* points to a valid json file containing the key *some_key*
    """

    EXPECTED_CONFIGS = []
    """Configuration keys that MUST figure in the json file"""

    ADDITIONNAL_CONFIGS = []
    """Configuration keys that CAN figure in the json file"""

    # noinspection PyUnresolvedReferences
    def __init__(self, path: Optional[Union[str, Path, TransparentPath]] = None):
        """
        Parameters
        ----------
        path: Optional[Union[str, Path, TransparentPath]]
            The path to the json file containing the configuration. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
            If None, creates an empty configuration.
        """

        if path is None:
            self.configs = {}
            return

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Config path must be a json")

        if not path.is_file():
            raise FileNotFoundError(f"No file found for node configuration {path}")

        lockfile = path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {path}")
        lockfile.touch()

        try:
            if hasattr(path, "read"):
                self.configs = path.read()
            else:
                with open(path) as opath:
                    self.configs = load(opath)

            self.path = path

            for key in self.EXPECTED_CONFIGS:
                if key not in self.configs:
                    raise IndexError(f"Missing required config {key}")
            for key in self.configs:
                if key not in self.EXPECTED_CONFIGS and key not in self.ADDITIONNAL_CONFIGS:
                    raise IndexError(f"Unexpected config {key}")
                if self.configs[key] == "":
                    if key.endswith("_kwargs"):
                        self.configs[key] = {}
                    else:
                        self.configs[key] = None
                elif key.endswith("_path"):
                    self.configs[key] = handle_path(self.configs[key], self.configs.get(f"{key}_fs", None))
                elif key.endswith("_paths"):
                    fss = self.configs.get(f"{key}_fss", None)
                    if fss is not None:
                        new_configs = []
                        for element, fs in zip(self.configs[key], fss):
                            new_configs.append(handle_path(element, fs))
                        self.configs[key] = new_configs

            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d: dict):
        for item in d:
            setattr(self, item, d[item])

    def __getattr__(self, item):
        if item == "configs":
            raise ValueError("Config object's 'configs' not set")
        if item == "path":
            raise ValueError("Config object's 'path' not set")
        if item not in self.configs:
            raise ValueError(f"No configuration named '{item}' was found")
        return self.configs[item]

    def __setattr__(self, item, value):
        if item == "configs" or item == "path":
            super().__setattr__(item, value)
            return
        if item not in self.configs and item not in self.EXPECTED_CONFIGS and item not in self.ADDITIONNAL_CONFIGS:
            raise ValueError(f"Configuration named '{item}' is not allowed")
        self.configs[item] = value

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, "configs", deepcopy(self.configs, memo))
        return result

    def save(self) -> None:
        """Saves the current configuration into the file it used to load. This allows the user to change the
        configuration in code and save it. Note that one can not have added a key not present in
        `Config.EXPECTED_CONFIGS`"""
        to_save = copy(self.configs)

        lockfile = self.path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {self.path}")
        lockfile.touch()

        try:
            for key in to_save:
                if key not in self.EXPECTED_CONFIGS and key not in self.ADDITIONNAL_CONFIGS:
                    raise ValueError(f"Forbidden configuration '{key}'")
                if to_save[key] is None:
                    to_save[key] = ""

            self.path.write(to_save)
            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e


class NodeLearningConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads :`Config`, corresponding to configuration of a node that can be accessed by the
    central server.

    Used by `ifra.node.Node` and `ifra.central_server.NodeGate`

    Overloads \_\_eq\_\_ to allow for node configuration comparison. Two configurations are equal if all their
    configuration values are equal, except *node_models_path* and *central_model_path* that can be different.

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    features_names: List[str]
        Names of the features used in the learning. It should contain all the features, not only those available to
        this node. If not specified, the json file should still contain the key *features_names*, but with value "".
    classes_names: List[str]
        Names of the classes used in the learning. It should contain all the classes, not only those available to
        this node. If not specified, the json file should still contain the key *classes_names*, but with value "".
    x_mins: Union[None, List[Union[int, float]]]
        Mmin values of each feature. Here too, if not None, should contain one entry for each
        feature used in the learning, and not only this node. If not specified, the json file should still contain the
        key *x_mins*, but with value "".
    x_maxs: Union[None, List[Union[int, float]]]
        Max values of each feature. Here too, if not None, should contain one entry for each
        feature used in the learning, and not only this node. If not specified, the json file should still contain the
        key *x_maxs*, but with value "".
    max_depth: int
        Maximum depth of the decision tree to use in the model
    plot_data: bool
        If True, nodes will plot distributions of their data in the same directory they found the said data.
        Plots will beproduced at each learning triggered by a central model update, and will be done both for datapreped
        and not datapreped data if a dataprep method is supplied.
    get_leaf: bool
        If True, will only consider the leaf ofthe tree to create rules. If False, every node of the tree will also be
        a rule.
    node_models_path: TransparentPath
        Path where all nodes are supposed to write their model. Should point to a directory to which the aggregator has
        access.
    node_models_path_fs: TransparentPath
        File system to use for node models directory. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *node_models_path_fs*, but with value "".
    id: Union[None, int, str]
        Name or number of the node. Can not be "".
    dataprep: Union[str]
        Name of the dataprep to use. Can be "" or one of:\n
          * BinFeaturesDataPrep (see `ifra.datapreps.BinFeaturesDataPrep`)\n
        If not specified, the json  file should still contain the key *dataprep*, but with value "".
    dataprep_kwargs: dict
        Keyword arguments for the `ifra.datapreps.Dataprep` init. If not specified, the json file should still contain
        the key *dataprep_kwargs*, but with value "".
    train_test_split: Union[None, str]
        Name of the dataprep to use. Can be "" or one of:\n
          * nothing yet :) \n
        If not specified, the json  file should still contain the key *dataprep*, but with value "".
    fitter: str
        Fitter to use. Can be one of :\n
          * decisiontree_fitter (see `ifra.fitters.DecisionTreeFitter`)\n
    fitter_kwargs: dict
        Keyword arguments for the `ifra.fitters.Fitter` init. If not specified, the json file should still contain the
        key *fitter_kwargs*, but with value "".
    updater: str
        Update method to be used by the node to take the central model into account. Can be one of:\n
          * adaboost_updater (see `ifra.updaters.AdaboostUpdater`)\n
    updater_kwargs: dict
        Keyword arguments for the `ifra.updaters.Updater` init. If not specified, the json file should still contain the
        key *updater_kwargs*, but with value "".
    thresholds_path: TransparentPath
        Path to the json file to use to set `ruleskit.thresholds.Thresholds`. If not specified, the json
        file should still contain the key *thresholds_path*, but with value "".
    thresholds_path_fs: TransparentPath
        File system to use for thresholds json file. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *thresholds_path_fs*, but with value "".
    emitter_path: TransparentPath
        Path to emitter json file. See `ifra.messenger.Emitter`
    emitter_path_fs: TransparentPath
        File system of path to emitter. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *emitter_path_fs*, but with value "".
    central_model_path: TransparentPath
        Path where the node is supposed to look for the central model.
    central_model_path_fs: str
        File system of central model path. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *central_model_path_fs*, but with value "".
    eval_kwargs: dict
        Keyword arguments for ruleskit's eval method. If not specified, the json
        file should still contain the key *eval_kwargs*, but with value "".
    privacy_proba: float
        Probability to use in differential privacy. Higher privacy_proba means worse privay.  If not specified, the json
        file should still contain the key *privacy_proba*, but with value "", in which case differential privacy is not
        used.
    """

    EXPECTED_CONFIGS = [
        "features_names",
        "classes_names",
        "x_mins",
        "x_maxs",
        "max_depth",
        "plot_data",
        "get_leaf",
        "node_models_path",
        "node_models_path_fs",
        "dataprep",
        "dataprep_kwargs",
        "train_test_split",
        "id",
        "fitter",
        "fitter_kwargs",
        "updater",
        "updater_kwargs",
        "thresholds_path",
        "thresholds_path_fs",
        "emitter_path",
        "emitter_path_fs",
        "central_model_path",
        "central_model_path_fs",
        "eval_kwargs",
        "privacy_proba"
    ]

    def __init__(self, path: Optional[Union[str, Path, TransparentPath]] = None):
        super().__init__(path)
        if self.privacy_proba is not None and not 0 < self.privacy_proba < 1:
            raise ValueError("privacy_proba should be between 0 and 1")

    def __eq__(self, other):
        if not isinstance(other, NodeLearningConfig):
            return False
        for key in NodeLearningConfig.EXPECTED_CONFIGS:
            # Those parameters can be different
            if (
                key == "emitter_path"
                or key == "id"
                or (key.endswith("_fs") and key != "node_models_path_fs")
                or key == "plot_data"
            ):
                continue
            if key not in self.configs and key not in other.configs:
                continue
            if key == "thresholds_path":  # Can be either both "", or both not "" and different or both "" and equal
                if (self.configs[key] == "" and other.configs[key] == "") or (
                    self.configs[key] != "" and other.configs[key] != ""
                ):
                    continue
            if self.configs[key] != other.configs[key]:
                return False
        return True


class AggregatorConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, corresponding to the aggregator configuration.

    Used by `ifra.aggregator.Aggregator`

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    node_models_path: TransparentPath
        Path where all nodes will write their models.
    node_models_path_fs: TransparentPath
        File system to use for node models directory. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *node_models_path_fs*, but with value "".
    aggregated_model_path: TransparentPath
        Path where the aggregator will save its model after each aggregations. Will also produce one output each
        time the model is updated. Should point to a csv file.
    aggregated_model_path_fs: str
        File system to use for learning outpout. Can be 'gcs', 'local' or "".
    min_number_of_new_models: int
        Minimum number of nodes that must have prodived a new model to trigger aggregation.
    aggregation: str
        Name of the aggregation method. Can be one of: \n
          * adaboost (see `ifra.aggregations.AdaBoostAggregation`)\n
    aggregation_kwargs: dict
        Keyword arguments for the `ifra.aggregations.Aggregation` init. If not specified, the json file should still
        contain the key *aggregation_kwargs*, but with value "".
    emitter_path: TransparentPath
        Path to emitter json file. See `ifra.messenger.Emitter`.
    emitter_path_fs: TransparentPath
        File system of path to emitter. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *emitter_path_fs*, but with value "".
    weight: str
        If using classification rules, 'weight' is unused. If using regression rules, will be used to average duplicated
        rules' prediction when aggregating. Can be 'equi' or any rlue attribute name that can be used as a weight
        (coverage, criterion...). If not specified, the json file should still contain the key *weight*,
        but with value "". In which case rules will be equally weighted.
    best: str
        If using classification rules, 'best' is unused. If using regression rules, specify whether the rule is best
        when its 'weight' is big ('best'="max") or small ('best'="min").If not specified, the json file should still
        contain the key *best*, but with value "". In which case, it will be set to "max".
    """

    EXPECTED_CONFIGS = [
        "node_models_path",
        "node_models_path_fs",
        "aggregated_model_path",
        "aggregated_model_path_fs",
        "min_number_of_new_models",
        "aggregation",
        "aggregation_kwargs",
        "emitter_path",
        "emitter_path_fs",
        "weight",
        "best"
    ]

    def __init__(self, path: Optional[Union[str, Path, TransparentPath]] = None):
        super().__init__(path)
        if self.weight is None:
            self.weight = "equi"
        if self.best is None:
            self.best = "max"
        if self.best != "min" and self.best != "max":
            raise ValueError(f"Attribute 'best' can be 'min' or 'max', not '{self.best}'")


class CentralConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, corresponding to the aggregator configuration.

    Used by `ifra.aggregator.Aggregator`

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    aggregated_model_path: TransparentPath
        Path where the aggregator will save its model after each aggregations. Will also produce one output each
        time the model is updated. Should point to a csv file.
    aggregated_model_path_fs: str
        File system to use for learning outpout. Can be 'gcs', 'local' or ""
    central_model_path: TransparentPath
        Path where the central model will save its model after each update. Will also produce one output each
        time the model is updated. Should point to a csv file.
    central_model_path_fs: str
        File system to use for learning outpout. Can be 'gcs', 'local' or ""
    emitter_path: TransparentPath
        Path to emitter json file. See `ifra.messenger.Emitter`
    emitter_path_fs: TransparentPath
        File system of path to emitter. Can be 'gcs', 'local' or "". If not specified, the json
        file should still contain the key *emitter_path_fs*, but with value "".
    """

    EXPECTED_CONFIGS = [
        "aggregated_model_path",
        "aggregated_model_path_fs",
        "central_model_path",
        "central_model_path_fs",
        "emitter_path",
        "emitter_path_fs",
    ]


class NodeDataConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, corresponding to one node data paths configuration. Not accessible by
    the central server.

    Used by `ifra.node.Node`

    The node expects two files separate files : one for the features and one for the targets. They both should return
    a dataframe (a single column in the case of the target file).
    The features file should contain one column for EACH FEATURE USED IN THE LEARNING, even if not all of them have data
    in this node. The order of the columns should match the order in
    `ifra.central_server.NodeGate.NodeLearningConfig`'s features_names`.

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    x_path: TransparentPath
        Path to the features file. If it is a csv, it MUST contain the index and columns (can be integers), and
        x_read_kwargs should contain index_col=0
    y_path: TransparentPath
        Path to the target (classes) file. If it is a csv, it MUST contain the index and columns (can be integers), and
        y_read_kwargs should contain index_col=0
    x_read_kwargs: Union[None, dict]
        Keyword arguments to read the features file. If not specified, the json file should still contain the key
        *x_read_kwargs*, but with value "".
    y_read_kwargs: Union[None, dict]
        Keyword arguments to read the target file. If not specified, the json file should still contain the key
        *x_read_kwargs*, but with value "".
    x_path_fs: str
        File system to use for x file. Can be 'gcs', 'local' or ""
    y_path_fs: str
        File system to use for y file. Can be 'gcs', 'local' or ""
    dataprep_kwargs: dict
        Set by `ifra.node.Node`. Keyword arguments for the dataprep.
    x_datapreped_path: TransparentPath
        Set by `ifra.node.Node.fit`. Path to the datapreped features file, before splitting into train/test.
    y_datapreped_path: TransparentPath
        Set by `ifra.node.Node.fit`. Path to the datapreped target file, before splitting into train/test.
    x_train_path: TransparentPath
        Set by `ifra.test_train_split.TrainTestSplit.split`. Path to the features file to use for testing. Can be None
        if no split is done.
    y_train_path: TransparentPath
        Set by `ifra.test_train_split.TrainTestSplit.split`. Path to the features file to use for testing. Can be None
        if no split is done.
    x_test_path: TransparentPath
        Set by `ifra.test_train_split.TrainTestSplit.split`. Path to the features file to use for testing.
    y_test_path: TransparentPath
        Set by `ifra.test_train_split.TrainTestSplit.split`. Path to the target file to use for testing.
    """

    EXPECTED_CONFIGS = ["x_path", "y_path", "x_read_kwargs", "y_read_kwargs", "x_path_fs", "y_path_fs"]
    ADDITIONNAL_CONFIGS = {
        "dataprep_kwargs": {},
        "x_datapreped_path": None,
        "y_datapreped_path": None,
        "x_train_path": None,
        "y_train_path": None,
        "x_test_path": None,
        "y_test_path": None,
    }

    def __init__(self, path: Optional[Union[str, Path, TransparentPath]] = None):
        super().__init__(path)

        if "index_col" in self.x_read_kwargs:
            if self.x_read_kwargs["index_col"] != 0:
                raise ValueError(
                    "If specifying index_col, it should be 0, otherwise you will have problems when saving"
                    " intermediate x learning copies"
                )
        else:
            if self.x.suffix == ".csv":
                raise ValueError("If x file is a csv, then its read kwargs should contain 'index_col=0'")

        if "index_col" in self.y_read_kwargs:
            if self.y_read_kwargs["index_col"] != 0:
                raise ValueError(
                    "If specifying index_col, it should be 0, otherwise you will have problems when saving"
                    " intermediate y learning copies"
                )
        else:
            if self.y.suffix == ".csv":
                raise ValueError("If y file is a csv, then its read kwargs should contain 'index_col=0'")


class MonitoringConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, configuration for monitoring the learning.

    Used by `ifra.monitoring.Monitor`

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    central_server_receiver_path: TransparentPath
    central_server_receiver_path_fs: str
    aggregator_receiver_path: TransparentPath
    aggregator_receiver_path_fs: str
    nodes_receivers_paths: List[TransparentPath]
    nodes_receivers_paths_fss: List[str]
    """

    EXPECTED_CONFIGS = [
        "central_server_receiver_path",
        "central_server_receiver_path_fs",
        "aggregator_receiver_path",
        "aggregator_receiver_path_fs",
        "nodes_receivers_paths",
        "nodes_receivers_paths_fss",
    ]

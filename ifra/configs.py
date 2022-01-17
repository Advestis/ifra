from copy import copy, deepcopy
from pathlib import Path
from time import sleep, time
from typing import Union, Optional
from json import load

from transparentpath import TransparentPath


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
        """

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Config path must be a json")

        if not path.is_file():
            raise FileNotFoundError(f"No file found for node configuration {self.path}")

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

            for key in self.configs:
                if self.configs[key] == "":
                    self.configs[key] = None

            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

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


class NodePublicConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads :`Config`, corresponding to configuration of a node that can be accessed by the
    central server.

    Used by `ifra.node.Node` and `ifra.central_server.NodeGate`

    Overloads \_\_eq\_\_ to allow for node configuration comparison. Two configurations are equal if all their
    configuration values are equal, except *local_model_path* and *central_model_path* that can be different.

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
    remember_activation: bool
        If True, the ruleset created by the node will remember its activation (logical OR of all its rules)
    stack_activation: bool
        If True, the ruleset created by the node will stack its rules' activation vectors into a np.ndarray (watch out
        for memory overload)
    plot_data: bool
        If True, nodes will plot distributions of their data in the same directory they found the said data.
        Plots will beproduced at each learning triggered by a central model update, and will be done both for datapreped
        and not datapreped data if a dataprep method is supplied.
    get_leaf: bool
        If True, will only consider the leaf ofthe tree to create rules. If False, every node of the tree will also be
        a rule.
    local_model_path: TransparentPath
        Path where the node is supposed to write its model. Should point to a csv file to which the central server have
        access.
    central_model_path: TransparentPath
        Path where the node is supposed to look for the central model. Should point to a csv file.
    dataprep: Union[str]
        Name of the dataprep to use. Can be "" or one of:\n
          * BinFeaturesDataPrep (see `ifra.datapreps.BinFeaturesDataPrep`)\n
        If not specified, will be set by central server. If not specified, the json  file should still contain the key
        *dataprep*, but with value "".
    dataprep_kwargs: dict
        Keyword arguments for the dataprep. If not specified, will be set by central server. If not specified, the json
        file should still contain the key *dataprep_kwargs*, but with value "".
    id: Union[None, int, str]
        Name or number of the node. If not specified, will be set by central server. If not specified, the json file
        should still contain the key *id*, but with value "".
    fitter: str
        Fitter to use. Can be one of :\n
          * decisiontree_fitter (see `ifra.fitter.DecisionTreeFitter`)\n
    updater: str
        Update method to be used by the node to take the central model into account. Can be one of:\n
          * adaboost_updater (see `ifra.updaters.AdaboostUpdater`)\n
    """

    EXPECTED_CONFIGS = [
        "features_names",
        "classes_names",
        "x_mins",
        "x_maxs",
        "max_depth",
        "remember_activation",
        "stack_activation",
        "plot_data",
        "get_leaf",
        "local_model_path",
        "central_model_path",
        "dataprep",
        "dataprep_kwargs",
        "id",
        "fitter",
        "updater"
    ]

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)
        if isinstance(self.configs["local_model_path"], str):
            self.configs["local_model_path"] = TransparentPath(self.configs["local_model_path"])
        if isinstance(self.configs["central_model_path"], str):
            self.configs["central_model_path"] = TransparentPath(self.configs["central_model_path"])

    def __eq__(self, other):
        if not isinstance(other, NodePublicConfig):
            return False
        for key in NodePublicConfig.EXPECTED_CONFIGS:
            # Those parameters can change without problem
            if key == "local_model_path" or key == "central_model_path" or key == "id":
                continue
            if self.configs[key] != other.configs[key]:
                return False
        return True


class CentralConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, corresponding to the central server configuration.

    Used by `ifra.central_server.CentralServer`

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    max_coverage: float
        Maximum coverage allowed for a rule of the node to be aggregated into the central mode
    output_path: TransparentPath
        Path where the central server will save its model at the end of the learning. Will also produce one output each
        time the model is updated. Should point to a csv file.
    output_path_fs: str
        File system to use for learning outpout. Can be 'gcs', 'local' or ""
    min_number_of_new_models: int
        Minimum number of nodes that must have prodived a new model to trigger aggregation.
    aggregation: str
        Name of the aggregation method. Can be one of: \n
          * adaboost_aggregation (see `ifra.aggregations.AdaBoostAggregation`)\n
    """

    EXPECTED_CONFIGS = [
        "max_coverage",
        "output_path",
        "output_path_fs",
        "min_number_of_new_models",
        "aggregation"
    ]

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)
        if isinstance(self.configs["output_path"], str):
            if self.configs["output_path_fs"] is not None:
                self.configs["output_path"] = TransparentPath(
                    self.configs["output_path"], fs=self.configs["output_path_fs"]
                )
            else:
                self.configs["output_path"] = TransparentPath(self.configs["output_path"])


class NodeDataConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads `Config`, corresponding to one node data paths configuration. Not accessible by
    the central server.

    Used by `ifra.node.Node`

    The node expects two files separate files : one for the features and one for the targets. They both should return
    a dataframe (a single column in the case of the target file).
    The features file should contain one column for EACH FEATURE USED IN THE LEARNING, even if not all of them have data
    in this node. The order of the columns should match the order in
    `ifra.central_server.NodeGate.NodePublicConfig`'s features_names`.

    Attributes
    ----------
    path: TransparentPath
        see `Config`
    configs: dict
        see `Config`

    x: TransparentPath
        Path to the features file. If it is a csv, it MUST contain the index and columns (can be integers), and
        x_read_kwargs should contain index_col=0
    y: TransparentPath
        Path to the target (classes) file. If it is a csv, it MUST contain the index and columns (can be integers), and
        y_read_kwargs should contain index_col=0
    x_read_kwargs: Union[None, dict]
        Keyword arguments to read the features file. If not specified, the json file should still contain the key
        *x_read_kwargs*, but with value "".
    y_read_kwargs: Union[None, dict]
        Keyword arguments to read the target file. If not specified, the json file should still contain the key
        *x_read_kwargs*, but with value "".
    x_fs: str
        File system to use for x file. Can be 'gcs', 'local' or ""
    y_fs: str
        File system to use for y file. Can be 'gcs', 'local' or ""
    dataprep_kwargs: dict
        Set by `ifra.node.Node`. Keyword arguments for the dataprep.
    """

    EXPECTED_CONFIGS = ["x", "y", "x_read_kwargs", "y_read_kwargs", "x_fs", "y_fs"]
    ADDITIONNAL_CONFIGS = {"dataprep_kwargs": {}}

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)

        if "index_col" in self.x_read_kwargs:
            if self.x_read_kwargs["index_col"] != 0:
                raise ValueError("If specifying index_col, it should be 0, otherwise you will have problem when saving "
                                 "intermediate data")
        else:
            if self.x.suffix == ".csv":
                raise ValueError("If file is a csv, then its read kwargs should contain index_col=0")

        if "index_col" in self.y_read_kwargs:
            if self.y_read_kwargs["index_col"] != 0:
                raise ValueError("If specifying index_col, it should be 0, otherwise you will have problem when saving "
                                 "intermediate data")
        else:
            if self.y.suffix == ".csv":
                raise ValueError("If file is a csv, then its read kwargs should contain index_col=0")

        for item in self.configs:
            if isinstance(self.configs[item], str):
                fs = f"{item}_fs"
                if fs in self.configs and self.configs[fs] is not None:
                    self.configs[item] = TransparentPath(self.configs[item], fs=self.configs[fs])
                else:
                    self.configs[item] = TransparentPath(self.configs[item])

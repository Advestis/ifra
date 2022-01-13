from copy import copy
from pathlib import Path
from typing import Union

from transparentpath import TransparentPath


class Config:

    # noinspection PyUnresolvedReferences
    """Abstract class to load a configuration. Basically just a wrapper around a json file.

        This class should be overloaded and not used as-is, for only keys present in
        :attribute:~ifra.configs.Config.EXPECTED_CONFIGS and :attribute:~ifra.configs.Config.ADDITIONNAL_CONFIGS will
        be accepted from the json file, attribute that is empty in this abstract class. Any key present in
        :attribute:~ifra.configs.Config.EXPECTED_CONFIGS must be present in the json file. Any key present in
        :attribute:~ifra.configs.Config.ADDITIONNAL_CONFIGS can be present in the file.

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
            assuming MyConfig overloads this class and path points to a valid json file containing the key 'some_key'

        Methods
        -------
        save() -> None
            Saves the current configuration into the file it used to load. This allows the user to change the
            configuration in code and save it. Note that one can not have added a key not present in
            :attribute:~ifra.configs.Config.EXPECTED_CONFIGS
        """

    EXPECTED_CONFIGS = []
    ADDITIONNAL_CONFIGS = []

    # noinspection PyUnresolvedReferences
    def __init__(self, path: Union[str, Path, TransparentPath]):
        """
        Parameters
        ----------
         path: Union[str, Path, TransparentPath]
            The path to the json file containing the configuration. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
        """

        if type(path) == str:
            path = TransparentPath(path)
        self.path = path

        if not path.suffix == ".json":
            raise ValueError("Config path must be a json")

        if hasattr(path, "read"):
            self.configs = path.read()
        else:
            with open(path) as opath:
                self.configs = load(opath)

        for key in self.EXPECTED_CONFIGS:
            if key not in self.configs:
                raise IndexError(f"Missing required config {key}")
        for key in self.configs:
            if key not in self.EXPECTED_CONFIGS and key not in self.ADDITIONNAL_CONFIGS:
                raise IndexError(f"Unexpected config {key}")

        for key in self.configs:
            if self.configs[key] == "":
                self.configs[key] = None

    def __getattr__(self, item):
        if item not in self.configs:
            raise ValueError(f"No configuration named '{item}' was found")
        return self.configs[item]

    def __setattr__(self, item, value):
        if item not in self.configs and item not in self.EXPECTED_CONFIGS and item not in self.ADDITIONNAL_CONFIGS:
            raise ValueError(f"No configuration named '{item}' is not allowed")
        self.configs[item] = value

    def save(self):
        """Saves the configuration back to the json file it was read from."""
        to_save = copy(self.configs)

        for key in to_save:
            if key not in self.EXPECTED_CONFIGS:
                raise ValueError(f"Forbidden configuration '{key}'")
            if to_save[key] is None:
                to_save[key] = ""

        self.path.write(to_save)


class NodePublicConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads :class:~ifra.configs.Config, corresponding to configuration of a node that can be accessed by the
    central server.

    Used by :class:~ifra.node.Node and :class:~ifra.central_server.NodeGate

    Overloads __eq__ to allow for node configuration comparison. Two configurations are equal if all their configuration
    values are equal, except "local_model_path" and "central_model_path" that can be different.

    Attributes
    ----------
    path: TransparentPath
        see :class:~ifra.configs.Config
    configs: dict
        see :class:~ifra.configs.Config

    features_names: List[str]
        Names of the features used in the learning. It should contain all the features, not only those available to
        this node. If not specified, the json file should still contain the key 'features_names', but with value "".
    classes_names: List[str]
        Names of the classes used in the learning. It should contain all the classes, not only those available to
        this node. If not specified, the json file should still contain the key 'classes_names', but with value "".
    x_mins: Union[None, List[Union[int, float]]]
        Mmin values of each feature. Here too, if not None, should contain one entry for each
        feature used in the learning, and not only this node. If not specified, the json file should still contain the
        key 'x_mins', but with value "".
    x_maxs: Union[None, List[Union[int, float]]]
        Max values of each feature. Here too, if not None, should contain one entry for each
        feature used in the learning, and not only this node. If not specified, the json file should still contain the
        key 'x_maxs', but with value "".
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
        Path where the node is supposed to write its model. Should point to a csv file.
    central_model_path: TransparentPath
        Path where the node is supposed to look for the central model. Should point to a csv file.
    dataprep_method: Union[str]
        Name of the method used to do dataprep. The given string must be importable from the current working directry.
        Will be None if "" is given. If not specified, the json file should still contain the
        key 'dataprep_method', but with value "".
    id: Union[None, int, str]
        Name or number of the node. If not specified, will be set by central server. If not specified, the json file
        should still contain the key 'id', but with value "".
    fitter: str
        Fitter to use. Can be one of :\n
          * decisiontree\n
    stop: bool
        Set to True by :class:~ifra.central_server.CentralServer when the learning is over.
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
        "dataprep_method",
        "id",
        "fitter"
    ]
    ADDITIONNAL_CONFIGS = ["stop"]

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
            if key == "local_model_path" or key == "central_model_path":  # Those parameters can change without problem
                continue
            if self[key] != other[key]:
                return False
        return True


class CentralConfig(Config):
    # noinspection PyUnresolvedReferences
    """Overloads :class:~ifra.configs.Config, corresponding to the central server configuration.

    Used by :class:~ifra.central_server.CentralServer

    Attributes
    ----------
    path: TransparentPath
        see :class:~ifra.configs.Config
    configs: dict
        see :class:~ifra.configs.Config

    max_coverage: float
        Maximum coverage allowed for a rule of the node to be aggregated into the central mode
    output_path: TransparentPath
        Path where the central server will save its model at the end of the learning. Will also produce one output each
        time the model is updated. Should point to a csv file.
    min_number_of_new_models: int
        Minimum number of nodes that must have prodived a new model to trigger aggregation.
    """

    EXPECTED_CONFIGS = [
        "max_coverage",
        "output_path",
        "min_number_of_new_models"
    ]

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)
        if isinstance(self.configs["output_path"], str):
            self.configs["output_path"] = TransparentPath(self.configs["output_path"])


class Paths(Config):
    # noinspection PyUnresolvedReferences
    """Overloads :class:~ifra.configs.Config, corresponding to one node data paths configuration. Not accessible by
    the central server.

    Used by :class:~ifra.node.Node

    The node expects two files separate files : one for the features and one for the targets. They both should return
    a dataframe (a single column in the case of the target file).
    The features file should contain one column for EACH FEATURE USED IN THE LEARNING, even if not all of them have data
    in this node. The order of the columns should match the order in
    :class:~ifra.configs.NodePublicConfig.features_names.

    Attributes
    ----------
    path: TransparentPath
        see :class:~ifra.configs.Config
    configs: dict
        see :class:~ifra.configs.Config

    x: TransparentPath
        Path to the features file. If it is a csv, it MUST contain the index and columns (can be integers), and
        x_read_kwargs should contain index_col=0
    y: TransparentPath
        Path to the target (classes) file. If it is a csv, it MUST contain the index and columns (can be integers), and
        y_read_kwargs should contain index_col=0
    x_read_kwargs: Union[None, dict]
        Keyword arguments to read the features file. If not specified, the json file should still contain the key
        'x_read_kwargs', but with value "".
    y_read_kwargs: Union[None, dict]
        Keyword arguments to read the target file. If not specified, the json file should still contain the key
        'x_read_kwargs', but with value "".
    """

    EXPECTED_CONFIGS = ["x", "y", "x_read_kwargs", "y_read_kwargs"]

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
                self.configs[item] = TransparentPath(self.configs[item])

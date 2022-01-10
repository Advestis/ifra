import os
from pathlib import Path
from transparentpath import TransparentPath
from typing import Union, List, Optional, Tuple
import joblib
from ruleskit import RuleSet
from sklearn import tree
import numpy as np
from ruleskit.utils.rule_utils import extract_rules_from_tree
from json import load
import logging

logger = logging.getLogger(__name__)


class Config:

    """Class to load a node configuration on its local computer. Basically just a wrapper around a json file."""

    EXPECTED_CONFIGS = []

    # noinspection PyUnresolvedReferences
    def __init__(self, path: Union[str, Path, TransparentPath]):

        if type(path) == str:
            path = TransparentPath(path)

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
            if key not in self.EXPECTED_CONFIGS:
                raise IndexError(f"Unexpected config {key}")

    def __getattr__(self, item):
        if item not in self.configs:
            raise ValueError(f"No configuration named '{item}' was found")
        return self.configs[item]


class Paths(Config):
    EXPECTED_CONFIGS = ["x", "y", "x_read_kwargs", "y_read_kwargs"]

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)
        for item in self.configs:
            if isinstance(self.configs[item], str):
                self.configs[item] = TransparentPath(self.configs[item])


class LearningConfig(Config):
    EXPECTED_CONFIGS = [
        "features_names",
        "classes_names",
        "x_mins",
        "x_maxs",
        "max_depth",
        "remember_activation",
        "stack_activation",
    ]


class Node:

    """Interface through which the central server can create nodes, update them, and get their fit results"""

    instances = []

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs_path: Union[str, Path, TransparentPath],
        path_configs_path: Optional[Union[str, Path, TransparentPath]] = None,
    ):
        self.id = len(Node.instances)
        self.tree = None
        self.ruleset = None
        self.learning_configs_path = learning_configs_path
        if path_configs_path is None:
            path_configs_path = TransparentPath("path_configs.json")
        self.__path_configs_path = path_configs_path

        self.learning_configs = LearningConfig(learning_configs_path)
        self.__paths = Paths(self.__path_configs_path)
        self.__fitter = Fitter(self.learning_configs, self.__paths)
        Node.instances.append(self)

    def fit(self):
        logger.info(f"Fitting node {self.id}...")
        self.tree, self.ruleset = self.__fitter.fit()
        logger.info(f"... node {self.id} fitted.")
        return self.tree, self.ruleset

    def update_from_central(self, ruleset: RuleSet):
        """TO CODE"""
        logger.info(f"Updating node {self.id}...")
        pass
        logger.info(f"... node {self.id} updated.")

    @classmethod
    def tree_to_graph(
        cls,
        path: Union[str, Path, TransparentPath],
        node: Optional["Node"] = None,
        thetree=None,
        features_names: Optional[List[str]] = None,
    ):
        """Saves self.tree to a .dot file and a .svg file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        node: Node
        thetree:
            Tree to save. If None (default) will use self.tree
        features_names: Optional[List[str]]
        """
        if node is None and thetree is None:
            raise ValueError("One of node and thetree must be not None")

        if node is not None and thetree is not None:
            raise ValueError("Must specify only one of node and thetree")

        if node is not None:
            thetree = node.tree
            features_names = node.learning_configs.features_names
        if thetree is None:
            return

        if type(path) == str:
            path = TransparentPath(path)

        path = path.with_suffix(".dot")
        with open(path, "w") as dotfile:
            tree.export_graphviz(
                thetree,
                out_file=dotfile,
                feature_names=features_names,
                filled=True,
                rounded=True,
                special_characters=True,
            )

        # joblib.dump(self.tree, self.__trees_path / (Y_name + ".joblib"))
        os.system(f'dot -Tsvg "{path}" -o "{path.with_suffix(".svg")}"')

    @classmethod
    def tree_to_joblib(
        cls,
        path: Union[str, Path, TransparentPath],
        node: Optional["Node"] = None,
        thetree=None,
    ):
        """Saves self.tree to a .joblib file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        node: Optional[Node]
        thetree:
            Tree to save. If None (default) will use self.tree
        """

        if node is None and thetree is None:
            raise ValueError("One of node and thetree must be not None")

        if node is not None and thetree is not None:
            raise ValueError("Must specify only one of node and thetree")

        if node is not None:
            thetree = node.tree
        if thetree is None:
            return

        if type(path) == str:
            path = TransparentPath(path)

        path = path.with_suffix(".joblib")
        joblib.dump(thetree, path)

    @classmethod
    def rule_to_file(
        cls, path: Union[str, Path, TransparentPath], node: Optional["Node"] = None, ruleset: Optional[RuleSet] = None
    ):
        """Saves self.ruleset to a .csv file. Does not do anything if self.ruleset is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        node: Optional[Node]
        ruleset:
            Tree to save. If None (default) will use self.tree
        """

        if node is None and ruleset is None:
            raise ValueError("One of node and thetree must be not None")

        if node is not None and ruleset is not None:
            raise ValueError("Must specify only one of node and thetree")

        if node is not None:
            ruleset = node.ruleset
        if ruleset is None:
            return

        if type(path) == str:
            path = TransparentPath(path)

        path = path.with_suffix(".csv")
        ruleset.save(path)


class Fitter:

    """Fitter class. Fits a DecisionTreeClassifier on some data."""

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs: LearningConfig,
        paths: Paths,
    ):
        self.learning_configs = learning_configs
        self.paths = paths

    def fit(self):
        return self._fit(
            self.paths.x.read(**self.paths.x_read_kwargs).values,
            self.paths.y.read(**self.paths.y_read_kwargs).values,
            self.learning_configs.max_depth,
            self.learning_configs.x_mins,
            self.learning_configs.x_maxs,
            self.learning_configs.features_names,
            self.learning_configs.classes_names
        )

    # noinspection PyArgumentList
    def _fit(
        self,
        x: np.array,
        y: np.array,
        max_depth: int,
        x_mins: Optional[List[float]],
        x_maxs: Optional[List[float]],
        features_names: Optional[List[str]],
        classes_names: Optional[List[str]],
        remember_activation: bool = True,
        stack_activation: bool = False,
    ) -> Tuple[tree.DecisionTreeClassifier, RuleSet]:
        """Fits x and y using a decision tree cassifier, setting self.tree and self.ruleset

        x array must contain one column for each feature that can exist across all nodes. Some features can contain
        only NaNs.

        Parameters
        ----------
        x: np.ndarray
            Must be of shape (# observations, # features)
        y: np.ndarray
            Must be of shape (# observations,)
        max_depth: int
            Maximum tree depth
        x_mins: Optional[List[float]]
            Lower limits of features
        x_maxs: Optional[List[float]]
            Upper limits of features
        features_names: Optional[List[str]]
            Names of features
        classes_names: Optional[List[str]]
            Names of the classes
        remember_activation: bool
            See extract_rules_from_tree, default = True
        stack_activation: bool
            See extract_rules_from_tree, default = False
        """

        if x_mins is None:
            x_mins = x.min(axis=0)
        elif not isinstance(x_mins, np.ndarray):
            x_mins = np.array(x_mins)
        if x_maxs is None:
            x_maxs = x.max(axis=0)
        elif not isinstance(x_maxs, np.ndarray):
            x_maxs = np.array(x_maxs)

        self.thetree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(x, y)
        self.ruleset = extract_rules_from_tree(
            self.thetree,
            xmins=x_mins,
            xmaxs=x_maxs,
            features_names=features_names,
            classes_names=classes_names,
            get_leaf=True,
            remember_activation=remember_activation,
            stack_activation=stack_activation,
        )
        return self.thetree, self.ruleset

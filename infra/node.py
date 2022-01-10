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
            path = Path(path)

        if not path.suffix == "json":
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
        "x_mins",
        "x_maxs",
        "get_leaf",
        "max_depth",
        "remember_activation",
        "stack_activation",
    ]


class Node:

    """Interface through which the central server can create nodes, update them, and get their fit results"""

    instances = []

    # noinspection PyUnresolvedReferences
    def __init__(self, learning_configs_path: Union[Path, TransparentPath]):
        self.id = len(Node.instances)
        self.tree = None
        self.ruleset = None
        self.learning_configs_path = learning_configs_path
        self.__path_configs_path = Path("path_configs.json")

        self.paths = None
        self.learning_configs = None

        self.load_paths()
        self.load_learning_configs()
        self.__fitter = Fitter(self.learning_configs_path, self.__path_configs_path)
        Node.instances.append(self)

    def fit(self):
        logger.info(f"Fitting node {self.id}...")
        self.tree, self.ruleset = self.__fitter.fit()
        logger.info(f"... node {self.id} fitted.")

    def update_from_central(self, ruleset: RuleSet):
        """TO CODE"""
        logger.info(f"Updating node {self.id}...")
        pass
        logger.info(f"... node {self.id} updated.")


class Fitter:

    """Fitter class. Fits a DecisionTreeClassifier on some data."""

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs_path: Union[str, Path, TransparentPath],
        path_configs_path: Union[str, Path, TransparentPath],
    ):
        self.learning_config = LearningConfig(learning_configs_path)
        self.paths = Paths(path_configs_path)

    def fit(self):
        return self._fit(
            self.paths.x.read(**self.paths.x_read_kwargs).values,
            self.paths.y.read(**self.paths.y_read_kwargs).values,
            self.learning_config.max_depth,
            self.learning_config.x_mins,
            self.learning_config.x_maxs,
            self.learning_config.features_names,
            self.learning_config.get_leaf,
        )

    # noinspection PyArgumentList
    @staticmethod
    def _fit(
        x: np.array,
        y: np.array,
        max_depth: int,
        x_mins: Optional[List[float]],
        x_maxs: Optional[List[float]],
        features_names: Optional[List[str]],
        get_leaf: bool = False,
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
        get_leaf: bool
            default value = False
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

        thetree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(x, y)
        ruleset = extract_rules_from_tree(
            thetree,
            xmins=x_mins,
            xmaxs=x_maxs,
            features_names=features_names,
            get_leaf=get_leaf,
            remember_activation=remember_activation,
            stack_activation=stack_activation,
        )
        return thetree, ruleset

    # noinspection PyUnresolvedReferences
    def tree_to_graph(self, path: Union[str, Path, TransparentPath]):
        """Saves self.tree to a .dot file and a .svg file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        """
        if self.tree is None:
            return

        if type(path) == str:
            path = Path(path)

        path = path.with_suffix(".dot")
        with open(path, "w") as dotfile:
            tree.export_graphviz(
                self.tree,
                out_file=dotfile,
                feature_names=self.features_names,
                filled=True,
                rounded=True,
                special_characters=True,
            )

        # joblib.dump(self.tree, self.__trees_path / (Y_name + ".joblib"))
        os.system(f'dot -Tsvg "{path}" -o "{path.with_suffix(".svg")}"')

    # noinspection PyUnresolvedReferences
    def tree_to_joblib(self, path: Union[str, Path, TransparentPath]):
        """Saves self.tree to a .joblib file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        """
        if self.tree is None:
            return

        if type(path) == str:
            path = Path(path)

        path = path.with_suffix(".joblib")
        joblib.dump(self.tree, path)

    # noinspection PyUnresolvedReferences
    def rule_to_file(self, path):
        """Saves self.ruleset to a .csv file. Does not do anything if self.ruleset is None

        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
        """
        if self.ruleset is None:
            return
        path = path.with_suffix(".csv")
        self.ruleset.save(path)

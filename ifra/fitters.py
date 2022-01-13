import os
from typing import Optional, List

import joblib
import numpy as np
from ruleskit import RuleSet
from sklearn import tree
from ruleskit.utils.rule_utils import extract_rules_from_tree
import logging

from .configs import NodePublicConfig, Paths

logger = logging.getLogger(__name__)


class DecisionTreeFitter:

    """Fits a DecisionTreeClassifier on some data."""

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs: NodePublicConfig,
        data: Paths,
    ):
        self.learning_configs = learning_configs
        self.data = data
        self.tree, self.ruleset = None, None

    def fit(self) -> RuleSet:
        self._fit(
            self.data.x.read(**self.data.x_read_kwargs).values,
            self.data.y.read(**self.data.y_read_kwargs).values,
            self.learning_configs.max_depth,
            self.learning_configs.get_leaf,
            self.learning_configs.x_mins,
            self.learning_configs.x_maxs,
            self.learning_configs.features_names,
            self.learning_configs.classes_names
        )
        self.tree_to_graph()
        self.tree_to_joblib()
        return self.ruleset

    # noinspection PyArgumentList
    def _fit(
        self,
        x: np.array,
        y: np.array,
        max_depth: int,
        get_leaf: bool,
        x_mins: Optional[List[float]],
        x_maxs: Optional[List[float]],
        features_names: Optional[List[str]],
        classes_names: Optional[List[str]],
        remember_activation: bool = True,
        stack_activation: bool = False,
    ):
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

        self.tree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(x, y)
        self.ruleset = extract_rules_from_tree(
            self.tree,
            xmins=x_mins,
            xmaxs=x_maxs,
            features_names=features_names,
            classes_names=classes_names,
            get_leaf=get_leaf,
            remember_activation=remember_activation,
            stack_activation=stack_activation,
        )

        if len(self.ruleset) > 0:
            self.ruleset.calc_activation(x)

    def tree_to_graph(
        self,
    ):
        """Saves self.tree to a .dot file and a .svg file. Does not do anything if self.tree is None
        """
        thetree = self.tree
        features_names = self.learning_configs.features_names
        iteration = 0
        name = self.learning_configs.local_model_path.stem
        path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.dot"

        while path.isfile():
            iteration += 1
            path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.dot"

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

    def tree_to_joblib(
        self,
    ):
        """Saves self.tree to a .joblib file. Does not do anything if self.tree is None
        """

        thetree = self.tree
        iteration = 0
        name = self.learning_configs.local_model_path.stem
        path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.joblib"

        while path.isfile():
            iteration += 1
            path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.joblib"

        path = path.with_suffix(".joblib")
        joblib.dump(thetree, path)

from typing import Optional, List, Tuple

import numpy as np
from ruleskit import RuleSet
from sklearn import tree
from ruleskit.utils.rule_utils import extract_rules_from_tree
import logging

logger = logging.getLogger(__name__)


class DecisionTreeFitter:

    """Fitter class. Fits a DecisionTreeClassifier on some data."""

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        learning_configs: LearningConfig,
        data: Paths,
    ):
        self.learning_configs = learning_configs
        self.data = data

    def fit(self):
        return self._fit(
            self.data.x.read(**self.data.x_read_kwargs).values,
            self.data.y.read(**self.data.y_read_kwargs).values,
            self.learning_configs.max_depth,
            self.learning_configs.get_leaf,
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
        get_leaf: bool,
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
            get_leaf=get_leaf,
            remember_activation=remember_activation,
            stack_activation=stack_activation,
        )

        if len(self.ruleset) > 0:
            self.ruleset.calc_activation(x)
        return self.thetree, self.ruleset

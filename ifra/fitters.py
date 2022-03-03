import os
from typing import Optional, List

import joblib
import numpy as np
from ruleskit import RuleSet
from sklearn import tree
from ruleskit.utils.rule_utils import extract_rules_from_tree
from ruleskit import Rule
import logging

from transparentpath import TransparentPath

from .configs import NodeLearningConfig, NodeDataConfig

logger = logging.getLogger(__name__)


class Fitter:
    """Abstract class of fitter object."""

    def __init__(
        self,
        learning_configs: NodeLearningConfig,
        **kwargs
    ):
        """

        Parameters
        ----------
        learning_configs: NodeLearningConfig
            `ifra.node.Node`'s *learning_config*
        data: NodeDataConfig
            `ifra.node.Node`'s *data*
        kwargs:
            Any additionnal keyword argument that the overleading class accepts. Those arguments will become attributes.
        """
        self.learning_configs = learning_configs
        self.model = None
        Rule.SET_THRESHOLDS(self.learning_configs.thresholds_path)
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def fit(self, x: np.ndarray, y: np.ndarray) -> RuleSet:
        """To be implemented in daughter class"""
        pass


class DecisionTreeFitter(Fitter):

    """Overloads the Fitter class. Fits a DecisionTreeFitter on some data.

    Can be used by giving *decisiontree* as *fitter* configuration when creating a `ifra.node.Node`

    Attributes
    ----------
    learning_configs: `ifra.configs.NodeLearningConfig`
        The learning configuration of the node using this fitter
    model: Union[None, RuleSet]
        Fitted model, or None if fit not done yet
    tree: Union[None, DecisionTreeFitter]
        Fitted tree, or None if fit not done yet
    """

    def __init__(
        self,
        learning_configs: NodeLearningConfig,
    ):
        super().__init__(learning_configs)
        self.tree = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RuleSet:
        """Fits the decision tree on the data pointed by `ifra.fitters.DecisionTreeFitter` *data.x_path* and
        `ifra.fitters.DecisionTreeFitter` *data.y_path*, sets
        `ifra.fitters.DecisionTreeFitter`*tree* saves it as a .dot, .svg and .joblib file in the same place
        the node will save its model. Those files will be unique for each time the fit function is called.
        Also sets `ifra.fitters.DecisionTreeFitter` *model* and returns it.

        Parameters
        ----------
        x: np.ndarray
            The features on which to learn
        y: np.ndarray
            The target to predict

        Returns
        -------
        RuleSet
            `ifra.fitters.DecisionTreeFitter` *model*
        """
        self.make_fit(
            x=x,
            y=y,
            max_depth=self.learning_configs.max_depth,
            get_leaf=self.learning_configs.get_leaf,
            x_mins=self.learning_configs.x_mins,
            x_maxs=self.learning_configs.x_maxs,
            features_names=self.learning_configs.features_names,
            classes_names=self.learning_configs.classes_names,
        )
        return self.model

    def save(self, path: TransparentPath):
        """Calls `ifra.fitters.DecisionTreeFitter.tree_to_graph` and `ifra.fitters.DecisionTreeFitter.tree_to_joblib`"""
        self.tree_to_graph(path)
        self.tree_to_joblib(path)

    # noinspection PyArgumentList
    def make_fit(
        self,
        x: np.array,
        y: np.array,
        max_depth: int,
        get_leaf: bool,
        x_mins: Optional[List[float]],
        x_maxs: Optional[List[float]],
        features_names: Optional[List[str]],
        classes_names: Optional[List[str]],
    ):
        """Fits x and y using a decision tree cassifier, setting
         `ifra.fitters.DecisionTreeFitter` *tree* and
         `ifra.fitters.DecisionTreeFitter` *model*

        x array must contain one column for each feature that can exist across all nodes. Some columns can contain
        only NaNs.

        Parameters
        ----------
        x: np.ndarray
            Must be of shape (# observations, # features)
        y: np.ndarray
            Must be of shape (# observations,)
        max_depth: int
            Maximum tree depth
        get_leaf: bool
            If True, only considers tree's leaves to make rules, else also considers its nodes
        x_mins: Optional[List[float]]
            Lower limits of features. If not specified, will use x.min(axis=0)
        x_maxs: Optional[List[float]]
            Upper limits of features. If not specified, will use x.max(axis=0)
        features_names: Optional[List[str]]
            Names of features
        classes_names: Optional[List[str]]
            Names of the classes
        """

        if x_mins is None:
            x_mins = x.min(axis=0)
        elif not isinstance(x_mins, np.ndarray):
            x_mins = np.array(x_mins)
        if x_maxs is None:
            x_maxs = x.max(axis=0)
        elif not isinstance(x_maxs, np.ndarray):
            x_maxs = np.array(x_maxs)

        self.tree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X=x, y=y)
        self.model = extract_rules_from_tree(
            self.tree,
            xmins=x_mins,
            xmaxs=x_maxs,
            features_names=features_names,
            classes_names=classes_names,
            get_leaf=get_leaf,
            remember_activation=True,
            stack_activation=True,
        )

        if len(self.model) > 0:
            # Compute each rule's activation vector, and the model's if its remember_activation flag is True, and will
            # stack the rules' activation vectors if stack_activation is True
            self.model.fit(y=y, xs=x)
            # self.model.check_duplicated_rules(self.model.rules, name_or_index="name")

    def tree_to_graph(
        self,
        path: TransparentPath
    ):
        """Saves `ifra.fitters.DecisionTreeFitter` *tree* to a .dot file and a .svg file. Does not do anything if
        `ifra.fitters.DecisionTreeFitter` *tree* is None.

        Parameters
        ----------
        path: TransparentPath
            File path where the files should be written. No matter the extension, one file with .dot and another with
            .svg will be created.
        """
        thetree = self.tree
        features_names = self.learning_configs.features_names
        path = path.with_suffix(".dot")

        with open(path, "w") as dotfile:
            tree.export_graphviz(
                decision_tree=thetree,
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
        path: TransparentPath
    ):
        """Saves `ifra.fitters.DecisionTreeFitter` *tree* to a .joblib file. Does not do anything if
        `ifra.fitters.DecisionTreeFitter` *tree* is None.

        Parameters
        ----------
        path: TransparentPath
            File path where the files should be written. No matter the extension, a file with .joblib will be created.
        """

        thetree = self.tree
        path = path.with_suffix(".joblib")
        joblib.dump(value=thetree, filename=path)

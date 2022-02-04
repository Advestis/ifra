import os
from typing import Optional, List

import joblib
import numpy as np
from ruleskit import RuleSet
from sklearn import tree
from ruleskit.utils.rule_utils import extract_rules_from_tree
from ruleskit import Rule
import logging

from .configs import NodePublicConfig, NodeDataConfig

logger = logging.getLogger(__name__)


class Fitter:
    """Abstract class of fitter object."""

    def __init__(
        self,
        public_configs: NodePublicConfig,
        **kwargs
    ):
        """

        Parameters
        ----------
        public_configs: NodePublicConfig
            `ifra.node.Node`'s *public_config*
        data: NodeDataConfig
            `ifra.node.Node`'s *data*
        kwargs:
            Any additionnal keyword argument that the overleading class accepts. Those arguments will become attributes.
        """
        self.public_configs = public_configs
        self.ruleset = None
        Rule.SET_THRESHOLDS(self.public_configs.thresholds_path)
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
    public_configs: `ifra.configs.NodePublicConfig`
        The public configuration of the node using this fitter
    ruleset: Union[None, RuleSet]
        Fitted ruleset, or None if fit not done yet
    tree: Union[None, DecisionTreeFitter]
        Fitted tree, or None if fit not done yet
    """

    def __init__(
        self,
        public_configs: NodePublicConfig,
    ):
        super().__init__(public_configs)
        self.tree = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RuleSet:
        """Fits the decision tree on the data pointed by `ifra.fitters.DecisionTreeFitter` *data.x_path* and
        `ifra.fitters.DecisionTreeFitter` *data.y_path*, sets
        `ifra.fitters.DecisionTreeFitter`*tree* saves it as a .dot, .svg and .joblib file in the same place
        the node will save its ruleset. Those files will be unique for each time the fit function is called.
        Also sets `ifra.fitters.DecisionTreeFitter` *ruleset* and returns it.

        Parameters
        ----------
        x: np.ndarray
            The features on which to learn
        y: np.ndarray
            The target to predict

        Returns
        -------
        RuleSet
            `ifra.fitters.DecisionTreeFitter` *ruleset*
        """
        self.make_fit(
            x,
            y,
            self.public_configs.max_depth,
            self.public_configs.get_leaf,
            self.public_configs.x_mins,
            self.public_configs.x_maxs,
            self.public_configs.features_names,
            self.public_configs.classes_names,
        )
        self.tree_to_graph()
        self.tree_to_joblib()
        return self.ruleset

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
        remember_activation: bool = True,
        stack_activation: bool = False,
    ):
        """Fits x and y using a decision tree cassifier, setting
         `ifra.fitters.DecisionTreeFitter` *tree* and
         `ifra.fitters.DecisionTreeFitter` *ruleset*

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
        remember_activation: bool
            See `ruleskit.utils.rule_utils.extract_rules_from_tree`, default = True
        stack_activation: bool
            See `ruleskit.utils.rule_utils.extract_rules_from_tree`, default = False
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
            # Compute each rule's activation vector, and the ruleset's if remember_activation, and will stack the
            # rules' if stack_activation is True
            self.ruleset.fit(y=y, xs=x)
            # self.ruleset.check_duplicated_rules(self.ruleset.rules, name_or_index="name")

    def tree_to_graph(
        self,
    ):
        """Saves `ifra.fitters.DecisionTreeFitter` *tree* to a .dot file and a .svg file at the same place
         the node will save its ruleset. Does not do anything if `ifra.fitters.DecisionTreeFitter` *tree*
        is None.

        Will create a unique file by looking at existing files and appending a unique integer to the name.
        """
        thetree = self.tree
        features_names = self.public_configs.features_names
        iteration = 0
        name = self.public_configs.node_model_path.stem
        path = self.public_configs.node_model_path.parent / f"{name}_{iteration}.dot"

        while path.is_file():
            iteration += 1
            path = self.public_configs.node_model_path.parent / f"{name}_{iteration}.dot"

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
        """Saves `ifra.fitters.DecisionTreeFitter` *tree* to a .joblib file. Does not do anything if
        `ifra.fitters.DecisionTreeFitter` *tree* is None

        Will create a unique file by looking at existing files and appending a unique integer to the name.
        """

        thetree = self.tree
        iteration = 0
        name = self.public_configs.node_model_path.stem
        path = self.public_configs.node_model_path.parent / f"{name}_{iteration}.joblib"

        while path.is_file():
            iteration += 1
            path = self.public_configs.node_model_path.parent / f"{name}_{iteration}.joblib"

        path = path.with_suffix(".joblib")
        joblib.dump(thetree, path)

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

    """Fits a DecisionTreeClassifier on some data.

    Can be used by giving ''decisiontree'' as ''fitter'' argument when creating a :class:~ifra.node.Node

    Attributes
    ----------
    public_configs: NodePublicConfig
        The public configuration of the node using this fitter
    data: Paths
        The data paths configuration of the node using this fitter
    tree: Union[None, DecisionTreeClassifier]
        Fitted tree, or None if fit not done yet
    ruleset: Union[None, RuleSet]
        Fitted ruleset, or None if fit not done yet

    Methods
    -------
    fit() -> RuleSet
        Calls :func:~ifra.fitters.DecisionTreeClassifier._fit
        Saves :attribute:~ifra.fitters.DecisionTreeClassifier.tree as a .dot, .svg and .joblib file in the same place
        the node will save its ruleset. Those files will be unique for each time the fit function is called.
        Also sets :attribute:~ifra.fitters.DecisionTreeClassifier.ruleset and returns it.
    _fit() -> None
        Fits the decision tree on the data pointed by :attribute:~ifra.fitters.DecisionTreeClassifier.data.x and
        :attribute:~ifra.fitters.DecisionTreeClassifier.y, sets :attribute:~ifra.fitters.DecisionTreeClassifier.tree
    _tree_to_graph() -> None
        Saves the fitted tree in a .dot and .svg
    _tree_to_joblib() -> None
        Saves the fitted tree in a .joblib
    """

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        public_configs: NodePublicConfig,
        data: Paths,
    ):
        self.public_configs = public_configs
        self.paths = data
        self.tree, self.ruleset = None, None

    def fit(self) -> RuleSet:
        """Fits the decision tree on the data pointed by :attribute:~ifra.fitters.DecisionTreeClassifier.paths.x and
        :attribute:~ifra.fitters.DecisionTreeClassifier.paths.y, sets
        :attribute:~ifra.fitters.DecisionTreeClassifier.tree saves it as a .dot, .svg and .joblib file in the same place
        the node will save its ruleset. Those files will be unique for each time the fit function is called.
        Also sets :attribute:~ifra.fitters.DecisionTreeClassifier.ruleset and returns it.

        Returns
        -------
        RuleSet
            :attribute:~ifra.fitters.DecisionTreeClassifier.ruleset
        """
        self._fit(
            self.paths.x.read(**self.paths.x_read_kwargs).values,
            self.paths.y.read(**self.paths.y_read_kwargs).values,
            self.public_configs.max_depth,
            self.public_configs.get_leaf,
            self.public_configs.x_mins,
            self.public_configs.x_maxs,
            self.public_configs.features_names,
            self.public_configs.classes_names
        )
        self._tree_to_graph()
        self._tree_to_joblib()
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
        """Fits x and y using a decision tree cassifier, setting
         :attribute:~ifra.fitters.DecisionTreeClassifier.tree and
         :attribute:~ifra.fitters.DecisionTreeClassifier.ruleset

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
        x_mins: Optional[List[float]]
            Lower limits of features. If not specified, will use x.min(axis=0)
        x_maxs: Optional[List[float]]
            Upper limits of features. If not specified, will use x.max(axis=0)
        features_names: Optional[List[str]]
            Names of features
        classes_names: Optional[List[str]]
            Names of the classes
        remember_activation: bool
            See :func:ruleskit.utils.rule_utils.extract_rules_from_tree, default = True
        stack_activation: bool
            See :func:ruleskit.utils.rule_utils.extract_rules_from_tree, default = False
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
            self.ruleset.calc_activation(x)

    def _tree_to_graph(
        self,
    ):
        """Saves :attribute:~ifra.fitters.DecisionTreeClassifier.tree to a .dot file and a .svg file at the same place
         the node will save its ruleset. Does not do anything if :attribute:~ifra.fitters.DecisionTreeClassifier.tree
        is None.

        Will create a unique file by looking at existing files and appending a unique integer to the name.
        """
        thetree = self.tree
        features_names = self.public_configs.features_names
        iteration = 0
        name = self.public_configs.local_model_path.stem
        path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.dot"

        while path.isfile():
            iteration += 1
            path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.dot"

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

    def _tree_to_joblib(
        self,
    ):
        """Saves :attribute:~ifra.fitters.DecisionTreeClassifier.tree to a .joblib file. Does not do anything if
        :attribute:~ifra.fitters.DecisionTreeClassifier.tree is None

        Will create a unique file by looking at existing files and appending a unique integer to the name.
        """

        thetree = self.tree
        iteration = 0
        name = self.public_configs.local_model_path.stem
        path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.joblib"

        while path.isfile():
            iteration += 1
            path = self.public_configs.local_model_path.parent / f"{name}_{iteration}.joblib"

        path = path.with_suffix(".joblib")
        joblib.dump(thetree, path)

import os
from pathlib import Path
from typing import Union, List
import joblib
from sklearn import tree
import numpy as np
from ruleskit.utils.rule_utils import extract_rules_from_tree


class Node:
    def __init__(
        self,
        id_number: int,
        x_mins: Union[np.ndarray, None] = None,
        x_maxs: Union[np.ndarray, None] = None,
        features_names: List[str] = None,
        get_leaf: bool = False,
    ):
        self.id = id_number
        self.tree = None
        self.ruleset = None

        if features_names is not None:
            if not isinstance(features_names, list):
                raise TypeError(f"features_names must be a list, you gave a {type(features_names)}")
            for name in features_names:
                if not isinstance(name, str):
                    raise TypeError(f"Names in features_names should be strings, but {name} is a {type(name)}")
        self.features_names = features_names

        if x_mins is not None:
            if not isinstance(x_mins, np.ndarray):
                raise TypeError(f"x_mins must be a np.ndarray, you gave a {type(x_mins)}")
            if not isinstance(x_mins.dtype, (float, int)):
                raise TypeError(f"x_mins must contain floating point numbers or integers, you gave {x_mins.dtype}")
        self.x_mins = x_mins

        if x_maxs is not None:
            if not isinstance(x_maxs, np.ndarray):
                raise TypeError(f"x_maxs must be a np.ndarray, you gave a {type(x_maxs)}")
            if not isinstance(x_maxs.dtype, (float, int)):
                raise TypeError(f"x_maxs must contain floating point numbers or integers, you gave {x_maxs.dtype}")
        self.x_maxs = x_maxs

        if not isinstance(get_leaf, bool):
            raise TypeError(f"get_leaf should be a boolean, but you gave a {type(get_leaf)}")
        self.get_leaf = get_leaf

    # noinspection PyArgumentList
    def fit(self, x: np.array, y: np.array, max_depth: int):
        """ Fits x and y using a decision tree cassifier, setting self.tree and self.ruleset

        Parameters
        ----------
        x: np.ndarray
            Must be of shape (# observations, # features)
        y: np.ndarray
            Must be of shape (# observations,)
        max_depth

        """
        # Do not set self.x_mins and self.x_maxs here : it is against the spirit of federated learning, where data
        # should not leave the node
        x_mins = self.x_mins
        if x_mins is None:
            x_mins = x.min(axis=0)
        x_maxs = self.x_maxs
        if x_maxs is None:
            x_maxs = x.max(axis=0)
        self.tree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(x, y)
        self.ruleset = extract_rules_from_tree(
            self.tree, xmins=x_mins, xmaxs=x_maxs, features_names=self.features_names, get_leaf=self.get_leaf
        )

    # noinspection PyUnresolvedReferences
    def tree_to_graph(self, path: Union[str, Path, "TransparentPath"]):
        """Saves self.tree to a .dot file and a .svg file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, "TransparentPath"]
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
    def tree_to_joblib(self, path: Union[str, Path, "TransparentPath"]):
        """Saves self.tree to a .joblib file. Does not do anything if self.tree is None

        Parameters
        ----------
        path: Union[str, Path, "TransparentPath"]
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
        path: Union[str, Path, "TransparentPath"]
        """
        if self.ruleset is None:
            return
        path = path.with_suffix(".csv")
        self.ruleset.save(path)

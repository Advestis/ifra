import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
from transparentpath import TransparentPath
from typing import Union, List, Optional, Tuple, Callable
import joblib
from ruleskit import RuleSet
from sklearn import tree
import numpy as np
from ruleskit.utils.rule_utils import extract_rules_from_tree
from json import load
import logging

logger = logging.getLogger(__name__)

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\boldmath"


def format_x(s):
    """For a given number, will put it in scientific notation if lower than 0.01, using LaTex synthax.

    In addition, if the value is lower than alpha, will change set the color of the value to green.
    """
    if s > 0.01:
        xstr = str(round(s, 2))
    else:
        xstr = "{:.4E}".format(s)
        if "E-" in xstr:
            lead, tail = xstr.split("E-")
            middle = "-"
        else:
            if "E" not in xstr:
                lead, tail = xstr, ""
            else:
                lead, tail = xstr.split("E")
            middle = ""
        while tail.startswith("0"):
            tail = tail[1:]
        while lead.endswith("0") and "." in lead:
            lead = lead[:-1]
            if lead.endswith("."):
                lead = lead[:-1]
        xstr = ("\\times 10^{" + middle).join([lead, tail]) + "}"
    return xstr


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

        for key in self.configs:
            if self.configs[key] == "":
                self.configs[key] = None

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
        "plot_data",
        "get_leaf",
        "max_coverage",
        "output_path",
        "fitter"  # Unused in simulator, decisiontree is hard-coded
    ]


class Node:

    """Interface through which the central server can create nodes, update them, and get their fit results"""

    instances = []

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        public_configs_path: Optional[Union[str, Path, TransparentPath]] = None,
        public_configs: Optional[LearningConfig] = None,
        path_configs_path: Optional[Union[str, Path, TransparentPath]] = None,
        dataprep_method: Callable = None
    ):

        if public_configs_path is None and public_configs is None:
            raise ValueError("One of public_configs_path and public_configs must be not None")
        if public_configs_path is not None and public_configs is not None:
            raise ValueError("Only one of public_configs_path and public_configs must be not None")

        self.id = len(Node.instances)
        self.tree = None
        self.ruleset = None
        if path_configs_path is None:
            path_configs_path = TransparentPath("path_configs.json")

        if public_configs_path is not None:
            self.public_configs = LearningConfig(public_configs_path)
        else:
            self.public_configs = public_configs
        self.dataprep_method = dataprep_method
        self.datapreped = False
        self.copied = False
        self.__paths = Paths(path_configs_path)
        self.__fitter = Fitter(self.public_configs, self.__paths)
        Node.instances.append(self)

    def fit(self):
        if self.public_configs.plot_data:
            self.plot_data_histogram(self.__paths.x.parent / "plots")
        if self.dataprep_method is not None and not self.datapreped:
            logger.info(f"Datapreping data of node {self.id}...")
            x, y = self.dataprep_method(
                self.__paths.x.read(**self.__paths.x_read_kwargs),
                self.__paths.y.read(**self.__paths.y_read_kwargs)
            )
            x_suffix = self.__paths.x.suffix
            y_suffix = self.__paths.y.suffix
            x_datapreped_path = self.__paths.x.with_suffix("").append("_datapreped").with_suffix(x_suffix)
            y_datapreped_path = self.__paths.y.with_suffix("").append("_datapreped").with_suffix(y_suffix)
            x_datapreped_path.write(x)
            y_datapreped_path.write(y)
            self.__fitter.paths.x = x_datapreped_path
            self.__fitter.paths.y = y_datapreped_path
            if self.public_configs.plot_data:
                self.plot_data_histogram(self.__paths.x.parent / "plots_datapreped")
            self.datapreped = True
            logger.info(f"...datapreping done for node {self.id}")

        if not self.copied:
            # Copy x and y in a different file, for it will be changed after each iterations by the update from central
            x_suffix = self.__paths.x.suffix
            self.__paths.x.cp(self.__paths.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix))
            self.__paths.x = self.__paths.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix)
            y_suffix = self.__paths.y.suffix
            self.__paths.y.cp(self.__paths.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix))
            self.__paths.y = self.__paths.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix)
            self.copied = True

        logger.info(f"Fitting node {self.id}...")
        self.tree, self.ruleset = self.__fitter.fit()
        logger.info(f"... node {self.id} fitted.")
        return self.tree, self.ruleset

    def update_from_central(self, ruleset: RuleSet):
        """Ignore points activated by the central server ruleset in order to find other relevant rules in the next
        iterations"""
        logger.info(f"Updating node {self.id}...")
        # Compute activation of the selected rules
        x = self.__paths.x.read(**self.__paths.x_read_kwargs)
        y = self.__paths.y.read(**self.__paths.y_read_kwargs)
        ruleset.remember_activation = True
        ruleset.calc_activation(x.values)
        x = x[ruleset.activation == 0]
        y = y[ruleset.activation == 0]
        self.__paths.x.write(x)
        self.__paths.y.write(y)
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
            features_names = node.public_configs.features_names
        if thetree is None:
            return

        if type(path) == str:
            path = TransparentPath(path)

        iteration = 0
        name = path.stem
        path = path.parent / f"{name}_{iteration}.dot"
        while path.isfile():
            iteration += 1
            path = path.parent / f"{name}_{iteration}.dot"

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

        iteration = 0
        name = path.stem
        path = path.parent / f"{name}_{iteration}.joblib"
        while path.isfile():
            iteration += 1
            path = path.parent / f"{name}_{iteration}.joblib"

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

        ruleset.save(path.with_suffix(".csv"))

        iteration = 0
        name = path.stem
        path = path.parent / f"{name}_{iteration}.csv"
        while path.isfile():
            iteration += 1
            path = path.parent / f"{name}_{iteration}.csv"

        ruleset.save(path)

    def plot_data_histogram(self, path):
        x = self.__paths.x.read(**self.__paths.x_read_kwargs)
        for col, name in zip(x.columns, self.public_configs.features_names):
            fig = plot_histogram(
                data=x[col],
                xlabel=name,
                figsize=(10, 7)
            )
            iteration = 0
            name = name.replace(' ', '_')
            path_x = path.parent / f"{name}_{iteration}.dot"
            while path_x.isfile():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.dot"
            fig.savefig(path_x)

        y = self.__paths.y.read(**self.__paths.y_read_kwargs)
        fig = plot_histogram(
            data=y.squeeze(),
            xlabel="Class",
            figsize=(10, 7)
        )

        iteration = 0
        path_y = path.parent / f"classes_{iteration}.dot"
        while path_y.isfile():
            iteration += 1
            path_y = path_y.parent / f"classes_{iteration}.dot"
        fig.savefig(path_y)


def plot_histogram(**kwargs):
    data = kwargs.get("data")
    xlabel = kwargs.get("xlabel", "x")
    xlabel = "".join(["\\textbf{", xlabel, "}"])
    ylabel = kwargs.get("ylabel", "count")
    ylabel = "".join(["\\textbf{", ylabel, "}"])
    xlabelfontsize = kwargs.get("xlabelfontsize", 20)
    ylabelfontsize = kwargs.get("ylabelfontsize", 20)
    title = kwargs.get("title", None)
    titlefontsize = kwargs.get("titlefontsize", 20)
    xticks = kwargs.get("xticks", None)
    yticks = kwargs.get("yticks", None)
    xtickslabels = kwargs.get("xtickslabels", None)
    xticksfontsize = kwargs.get("xticksfontsize", 15)
    yticksfontsize = kwargs.get("yticksfontsize", 15)
    force_xlim = kwargs.get("force_xlim", False)
    xlim = kwargs.get("xlim", [])
    ylim = kwargs.get("ylim", [])
    figsize = kwargs.get("figsize", None)
    legendfontsize = kwargs.get("legendfontsize", 12)
    plt.close("all")
    fig = plt.figure(0, figsize=figsize)
    ax1 = fig.gca()

    # If xlim is specified, set it. Otherwise, if force_xlim is specified,
    # redefine the x limites to match the histo ranges.
    if len(xlim) == 2:
        ax1.set_xlim(left=xlim[0], right=xlim[1])
    elif force_xlim:
        ax1.set_xlim([0.99 * data.min(), 1.01 * data.max()])
    if len(ylim) == 2:
        ax1.set_ylim(ylim)

    label = None

    try:
        mean = format_x(data.mean())
        std = format_x(data.std())
        count = format_x(len(data))
        integral = format_x(sum(data))
        label = (
                f"\\flushleft$\\mu={mean}$ \\\\ $\\sigma={std}$ \\\\ "
                + "\\textbf{Count}$="
                + f"{count}$ \\\\ "
                + "\\textbf{Integral}$="
                + f"{integral}$"
        )
    except TypeError:
        pass

    data.hist(ax=ax1, bins=100, label=label)

    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=ylabelfontsize)
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=xlabelfontsize)
    if title is not None:
        ax1.set_title(title, fontsize=titlefontsize)

    if xticksfontsize is not None:
        ax1.tick_params(axis="x", labelsize=xticksfontsize)
    if yticksfontsize is not None:
        ax1.tick_params(axis="y", labelsize=yticksfontsize)

    if xticks is not None:
        ax1.set_xticks(xticks)
    if yticks is not None:
        ax1.set_yticks(yticks)
    if xtickslabels is not None:
        ax1.set_xticklabels(xtickslabels)

    if label:
        ax1.legend(
            # bbox_to_anchor=(0.9, 0, 0, 1),
            loc="upper right",
            facecolor="wheat",
            fontsize=legendfontsize,
            shadow=True,
            title=None,
        )
    plt.gcf().tight_layout()

    return fig


class Fitter:

    """Fitter class. Fits a DecisionTreeClassifier on some data."""

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        public_configs: LearningConfig,
        paths: Paths,
    ):
        self.public_configs = public_configs
        self.paths = paths

    def fit(self):
        return self._fit(
            self.paths.x.read(**self.paths.x_read_kwargs).values,
            self.paths.y.read(**self.paths.y_read_kwargs).values,
            self.public_configs.max_depth,
            self.public_configs.get_leaf,
            self.public_configs.x_mins,
            self.public_configs.x_maxs,
            self.public_configs.features_names,
            self.public_configs.classes_names
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

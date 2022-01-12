from datetime import datetime
from pathlib import Path
from time import time, sleep

from sklearn import tree
import numpy as np
from ruleskit.utils.rule_utils import extract_rules_from_tree
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Union, Optional, Tuple

from ruleskit import RuleSet
from transparentpath import TransparentPath
import logging

from ifra.configs import NodePublicConfig, Paths

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


class Node:
    def __init__(
        self,
        path_learning_configs: TransparentPath,
        path_data: Union[str, Path, TransparentPath],
    ):
        if type(path_learning_configs) == str:
            path_learning_configs = TransparentPath(path_learning_configs)
        self.path_learning_configs = path_learning_configs
        if type(path_data) == str:
            path_data = TransparentPath(path_data)
        self.learning_configs = NodePublicConfig(path_learning_configs)

        if not len(list(self.learning_configs.local_model_path.parent.ls(""))) == 0:
            raise ValueError(
                "Path were Node model should be saved is not empty. Make sure you cleaned all previous learning output"
            )

        self.__data = Paths(path_data)
        self.__fitter = Fitter(self.learning_configs, self.__data)

        self.dataprep_method = None
        self.datapreped = False
        self.copied = False
        self.tree = None
        self.ruleset = None
        self.last_fetch = None
        if self.learning_configs.dataprep_method is not None:
            function = self.learning_configs.dataprep_method.split(".")[-1]
            module = self.learning_configs.dataprep_method.replace(f".{function}", "")
            self.dataprep_method = __import__(module, globals(), locals(), [function], 0)

    def fit(self):
        if self.learning_configs.plot_data:
            self.plot_data_histogram(self.__data.x.parent / "plots")
        if self.dataprep_method is not None and not self.datapreped:
            logger.info(f"Datapreping data of node {self.learning_configs.id}...")
            x, y = self.dataprep_method(
                self.__data.x.read(**self.__data.x_read_kwargs),
                self.__data.y.read(**self.__data.y_read_kwargs)
            )
            x_suffix = self.__data.x.suffix
            y_suffix = self.__data.y.suffix
            x_datapreped_path = self.__data.x.with_suffix("").append("_datapreped").with_suffix(x_suffix)
            y_datapreped_path = self.__data.y.with_suffix("").append("_datapreped").with_suffix(y_suffix)
            x_datapreped_path.write(x, index=False)
            y_datapreped_path.write(y, index=False)
            self.__fitter.data.x = x_datapreped_path
            self.__fitter.data.y = y_datapreped_path
            if self.learning_configs.plot_data:
                self.plot_data_histogram(self.__data.x.parent / "plots_datapreped")
            self.datapreped = True
            logger.info(f"...datapreping done for node {self.learning_configs.id}")

        if not self.copied:
            # Copy x and y in a different file, for it will be changed after each iterations by the update from central
            x_suffix = self.__data.x.suffix
            self.__data.x.cp(self.__data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix))
            self.__data.x = self.__data.x.with_suffix("").append("_copy_for_learning").with_suffix(x_suffix)
            y_suffix = self.__data.y.suffix
            self.__data.y.cp(self.__data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix))
            self.__data.y = self.__data.y.with_suffix("").append("_copy_for_learning").with_suffix(y_suffix)
            self.copied = True

        logger.info(f"Fitting node {self.learning_configs.id}...")
        self.tree, self.ruleset = self.__fitter.fit()
        self.tree_to_graph()
        self.tree_to_joblib()
        self.ruleset_to_file()
        logger.info(f"... node {self.learning_configs.id} fitted, results saved in"
                    f" {self.learning_configs.local_model_path.parent}.")
        return self.tree, self.ruleset

    def update_from_central(self, ruleset):
        """Ignore points activated by the central server ruleset in order to find other relevant rules in the next
        iterations"""
        logger.info(f"Updating node {self.learning_configs.id}...")
        # Compute activation of the selected rules
        x = self.__data.x.read(**self.__data.x_read_kwargs)
        y = self.__data.y.read(**self.__data.y_read_kwargs)
        ruleset.remember_activation = True
        ruleset.calc_activation(x.values)
        x = x[ruleset.activation == 0]
        y = y[ruleset.activation == 0]
        self.__data.x.write(x, index=False)
        self.__data.y.write(y, index=False)
        logger.info(f"... node {self.learning_configs.id} updated.")

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

    def ruleset_to_file(
        self
    ):
        """Saves self.ruleset to 2 .csv files. Does not do anything if self.ruleset is None
        """

        ruleset = self.ruleset

        iteration = 0
        name = self.learning_configs.local_model_path.stem
        path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.csv"
        while path.isfile():
            iteration += 1
            path = self.learning_configs.local_model_path.parent / f"{name}_{iteration}.csv"

        path = path.with_suffix(".csv")
        ruleset.save(path)
        ruleset.save(self.learning_configs.local_model_path)

    def plot_data_histogram(self, path):
        x = self.__data.x.read(**self.__data.x_read_kwargs)
        for col, name in zip(x.columns, self.learning_configs.features_names):
            fig = plot_histogram(
                data=x[col],
                xlabel=name,
                figsize=(10, 7)
            )

            iteration = 0
            name = name.replace(' ', '_')
            path_x = path.parent / f"{name}_{iteration}.pdf"
            while path_x.isfile():
                iteration += 1
                path_x = path_x.parent / f"{name}_{iteration}.pdf"
            fig.savefig(path_x)

        y = self.__data.y.read(**self.__data.y_read_kwargs)
        fig = plot_histogram(
            data=y.squeeze(),
            xlabel="Class",
            figsize=(10, 7)
        )

        iteration = 0
        path_y = path.parent / f"classes_{iteration}.pdf"
        while path_y.isfile():
            iteration += 1
            path_y = path_y.parent / f"classes_{iteration}.pdf"

        fig.savefig(path_y)

    def watch(self, timeout: int = 60, sleeptime: int = 5):

        def get_ruleset():
            central_ruleset = RuleSet()
            central_ruleset.load(self.learning_configs.central_model_path)
            self.last_fetch = datetime.now()
            self.update_from_central(central_ruleset)
            logger.info(f"Fetched new central ruleset in node {self.learning_configs.id} at {self.last_fetch}")
            self.new_data = True

        t = time()
        new_central_model = True  # start at true to trigger fit even if no central model is here at first iteration
        while time() - t < timeout:
            if self.learning_configs.id is None:
                self.learning_configs = NodePublicConfig(self.path_learning_configs)
            if self.learning_configs.id is None:
                logger.warning(
                    "Node id is not set, meaning central server is not running. Waiting for central server to start."
                )
                continue

            if self.last_fetch is None:
                if self.learning_configs.central_model_path.isfile():
                    get_ruleset()
                    new_central_model = True
            else:
                if (
                    self.learning_configs.central_model_path.isfile()
                    and self.learning_configs.central_model_path.info()["mtime"] > self.last_fetch.timestamp()
                ):
                    get_ruleset()
                    new_central_model = True

            if new_central_model:
                self.fit()
                new_central_model = False
            sleep(sleeptime)

        logger.info(f"Timeout of {timeout} seconds reached, stopping learning in node {self.learning_configs.id}.")


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

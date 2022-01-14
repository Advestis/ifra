from typing import Union
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\boldmath"


def format_x(s: Union[float, int], with_dollar: bool = False) -> str:
    """For a given number, will put it in scientific notation if lower than 0.01 or greater than 1000,
    using LaTex synthax.
    """
    if 1000 > s > 0.01:
        xstr = str(round(s, 2))
    else:
        xstr = "{:.4E}".format(s)
        if "E-" in xstr:
            lead, tail = xstr.split("E-")
            middle = "-"
        else:
            lead, tail = xstr.split("E")
            middle = ""
        lead = round(float(lead), 2)
        tail = round(float(tail), 2)
        if with_dollar:
            xstr = ("$\\cdot 10^{" + middle).join([str(lead), str(tail)]) + "}$"
        else:
            xstr = ("\\cdot 10^{" + middle).join([str(lead), str(tail)]) + "}"
    return xstr


def plot_histogram(**kwargs) -> plt.Figure:
    """Plot the distribution of some data, including a statbox as legend"""
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

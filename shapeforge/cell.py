from typing import List

import gbox as gb
import matplotlib.pyplot as plt

from .cell_domain import CellDomain
from .inclusion import Inclusion


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        inclusions: dict[str, List[Inclusion | gb.CirclesArray]],
    ):
        self.domain = domain
        self.inclusions = inclusions

    def plot(self, fig=None, ax=None, show=False, f_path=None):
        """
        Plot the cell with its domain and inclusions.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. If None, a new figure will be created.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new axes will be created.
        show : bool, optional
            If True, display the plot. Default is False.
        f_path : str, optional
            If provided, save the plot to this file path.
        kwargs : dict, optional
            Additional keyword arguments for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        domain_bbox = self.domain.domain
        ax = domain_bbox.plot(
            ax, facecolor="None", edgecolor="b", lw=1, alpha=0.5
        )

        for a_group_of_inclusions in self.inclusions.values():
            for a_inclusion in a_group_of_inclusions:
                a_inclusion.plot(axs=ax, facecolor="None", edgecolor="k")

        # TODO replace this custom axis formatting with a dedicated function
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(domain_bbox.p_min[0], domain_bbox.p_max[0])
        ax.set_ylim(domain_bbox.p_min[1], domain_bbox.p_max[1])

        if f_path is not None:
            plt.savefig(f_path)

        if show:
            plt.show()

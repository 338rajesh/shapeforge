from typing import List

import gbox as gb
import matplotlib.pyplot as plt
import numpy as np

from .cell_domain import CellDomain
from .optim import nmspg, OptimisationProblem


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        inclusions: dict[str, List[gb.GShape | gb.CirclesArray]],
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
            ax, facecolor="grey", edgecolor="b", lw=1, alpha=0.5
        )

        for a_group_of_inclusions in self.inclusions.values():
            for a_inclusion in a_group_of_inclusions:
                a_inclusion.plot(
                    axs=ax, facecolor="y", edgecolor="blue", lw=1.0
                )

        # TODO replace this custom axis formatting with a dedicated function
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(domain_bbox.p_min[0], domain_bbox.p_max[0])
        ax.set_ylim(domain_bbox.p_min[1], domain_bbox.p_max[1])

        if f_path is not None:
            plt.savefig(f_path)

        if show:
            plt.show()

    def remove_inclusion_overlaps(self) -> None:
        # Run Optim Loop to ensure there are no overlaps among inclusions
        #   Evaluate the cost function and gradients
        #     Add Periodic copies, if required
        #     cost evaluation
        #     gradients evaluation
        #   Update the inclusions positions
        #   Check for convergence
        #   If converged, return the optimised inclusions positions
        opt_problem = InclusionOverlap()
        result = nmspg(
            objective=opt_problem,
            x0=None,
            iter_max=100,
            iter_memory=10,
            epsilon=1e-6,
            spectral_step_min=1e-30,
            spectral_step_max=1e30,
            gamma=0.0001,
            sigma1=0.1,
            sigma2=0.9,
            ls_iter_max=20,
            p_bar=None,
        )


class InclusionOverlap(OptimisationProblem):
    def __init__(self):
        pass

    def f(self, x: np.ndarray) -> float:
        pass

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        pass

    def projection(self, x: np.ndarray) -> np.ndarray:
        pass

from typing import List, Optional, Sequence

import gbox as gb
import matplotlib.pyplot as plt
import numpy as np

from .optim import nmspg, OptimisationProblem
from .utils import _validate_dict, DistributionSampler


class CellDomain:
    """
    It represents a domain for a cell, without any inclusions. It can be
    considered as a host for inclusions or other geometrical features, but it
    does not contain any by itself.
    """

    def __init__(self, bounds: Sequence):
        self._bounds = bounds
        self.domain = gb.BoundingBox(
            lower_bound=(bounds[0], bounds[1]),
            upper_bound=(bounds[2], bounds[3]),
        )

    @property
    def domain_bounds(self):
        return self._bounds

    @property
    def x_bounds(self) -> tuple[float, float]:
        """
        Get the x bounds of the cell domain.
        """
        return self._bounds[0], self._bounds[2]

    @property
    def y_bounds(self) -> tuple[float, float]:
        """
        Get the y bounds of the cell domain.
        """
        return self._bounds[1], self._bounds[3]

    @property
    def cell_volume(self) -> float:
        """
        Calculate the volume of the cell domain.
        """
        return self.domain.volume

    @classmethod
    def from_config(cls, config: dict) -> "CellDomain":
        """
        Create a CellDomain instance from a configuration dictionary. It is
        expected that the configuration contains `shape` and `bounds` keys
        that define the shape of the cell domain and its bounds, respectively.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing bounds and other parameters.

        Returns
        -------
        CellDomain
            An instance of CellDomain initialized with the provided bounds.
        """
        _validate_dict(config, ["shape", "bounds"])
        cell_shape = config["shape"]
        cell_bounds = config["bounds"]

        if cell_shape != "rectangle":
            raise NotImplementedError(
                f"Unsupported cell shape '{cell_shape}'. Only 'rectangle' is supported."
            )

        _validate_dict(cell_bounds, ["x_min", "y_min", "x_max", "y_max"])
        x_min = cell_bounds["x_min"]
        y_min = cell_bounds["y_min"]
        x_max = cell_bounds["x_max"]
        y_max = cell_bounds["y_max"]

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                "Invalid bounds: x_min must be less than x_max and "
                "y_min must be less than y_max."
            )

        return cls(bounds=[x_min, y_min, x_max, y_max])


class InclusionSampler:
    def __init__(
        self,
        shape: str | gb.GShape,
        spatial_param_sampler: DistributionSampler,
        size_param_sampler: DistributionSampler,
    ):
        g_shape = getattr(gb, shape) if isinstance(shape, str) else shape

        if not issubclass(g_shape, gb.GShape):
            raise ValueError(
                f"Unsupported shape: {shape}, must be a subclass of gb.GShape"
            )
        # check if g_shape has a method called from_params
        if not callable(getattr(g_shape, "from_params")):
            raise ValueError(
                f"Unsupported shape: {shape}, must have a from_params method"
            )

        self.shape: gb.GShape = g_shape
        self.spatial_param_sampler = spatial_param_sampler
        self.size_param_sampler = size_param_sampler

    def __call__(self) -> gb.GShape:
        pos_params = self.spatial_param_sampler.sample()
        size_params = self.size_param_sampler.sample()
        return self.shape.from_params(
            positional_params=pos_params, size_params=size_params
        )


def initialise_inclusions(
    incl_config: dict | list[dict],
    cell_domain: CellDomain,
    *,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, List[gb.GShape]]:
    """
    It is the main function to initialize inclusions based on the provided
    configuration. It iterates through the configuration list, extracting
    the shape, volume fraction, and parameters for each inclusion type.
    It then calls the appropriate function to generate and initialize the
    inclusions based on the shape specified in the configuration.

    Parameters
    ----------
    cell_domain : CellDomain
        The cell domain in which the inclusions will be placed.
    config : list[dict] | dict
        A single dictionary or a list of dictionaries, each containing
        configuration for a specific inclusion type. Each dictionary should
        have the following keys:
        - `shape`: The shape of the inclusion (e.g., "circle", "ellipse").
        - `volume_fraction`: The volume fraction of the inclusion.
        - `parameters`: Additional parameters specific to the inclusion shape.

    Returns
    -------
    dict[str, list[gb.GShape]]
        A dictionary where the keys are the shape names and the values are
        lists of initialized inclusion objects of that shape.
    """
    if isinstance(incl_config, dict):
        incl_config = [incl_config]

    non_dict_items = [i for i in incl_config if not isinstance(i, dict)]
    if non_dict_items:
        raise TypeError(
            "Configuration must be a single dictionary or a list of "
            f"dictionaries. Found non-dictionary items: {non_dict_items}."
        )

    # xc and yc distributions are not specified in the config,
    x_min, x_max = cell_domain.x_bounds
    y_min, y_max = cell_domain.y_bounds
    xy_init_config = {
        "xc": {
            "distribution": {
                "name": "uniform",
                "loc": x_min,
                "scale": x_max - x_min,
            }
        },
        "yc": {
            "distribution": {
                "name": "uniform",
                "loc": y_min,
                "scale": y_max - y_min,
            }
        },
        "major_axis_angle": {
            "distribution": {
                "name": "uniform",
                "loc": 0.0,
                "scale": 2.0 * np.pi,
            }
        },
    }
    xy_sampler = DistributionSampler(xy_init_config, rng=rng)

    initialised_inclusions: dict[str, List[gb.GShape]] = {}
    for ith_incl_config in incl_config:
        shape, vf, params = _validate_dict(
            ith_incl_config,
            keys=["shape", "volume_fraction", "parameters"],
            val_types=[str, float, dict],
            ret_val=True,
        )
        incl_sampler = InclusionSampler(
            shape=shape,
            spatial_param_sampler=xy_sampler,
            size_param_sampler=DistributionSampler(params, rng=rng),
        )
        required_volume = cell_domain.cell_volume * vf
        cumulative_volume = 0.0
        generated_inclusions = []
        while cumulative_volume < required_volume:
            a_inclusion = incl_sampler()
            generated_inclusions.append(a_inclusion)
            cumulative_volume += a_inclusion.volume()

        initialised_inclusions[shape] = generated_inclusions
    return initialised_inclusions


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        inclusions: dict[str, List[gb.GShape]],
    ):
        self.domain = domain
        self.inclusions = inclusions
        #
        self.inclusions_uns = None  # type: List[gb.CirclesArray] | None

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
        ax = domain_bbox.plot(ax, facecolor="grey", edgecolor="b", lw=1, alpha=0.5)

        for a_group_of_inclusions in self.inclusions.values():
            for a_inclusion in a_group_of_inclusions:
                a_inclusion.plot(axs=ax, facecolor="y", edgecolor="blue", lw=1.0)

        # TODO replace this custom axis formatting with a dedicated function
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(domain_bbox.p_min[0], domain_bbox.p_max[0])
        ax.set_ylim(domain_bbox.p_min[1], domain_bbox.p_max[1])

        if f_path is not None:
            plt.savefig(f_path)

        if show:
            plt.show()

    def _get_inclusions_overlap_opt_problem(self) -> OptimisationProblem:
        self.inclusions_uns = (
            {k: [gs.uns() for gs in v] for k, v in self.inclusions.items()}
            if self.inclusions_uns is None
            else self.inclusions_uns
        )
        opt_problem = CellInclusionsOverlapProblem()
        return opt_problem

    def remove_inclusion_overlaps(self) -> None:
        # Run Optim Loop to ensure there are no overlaps among inclusions
        #   Evaluate the cost function and gradients
        #     Add Periodic copies, if required
        #     cost evaluation
        #     gradients evaluation
        #   Update the inclusions positions
        #   Check for convergence
        #   If converged, return the optimised inclusions positions
        opt_problem = self._get_inclusions_overlap_opt_problem()
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


class CellInclusionsOverlapProblem(OptimisationProblem):
    def __init__(self):
        pass

    def f(self, x: np.ndarray) -> float:
        pass

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        pass

    def projection(self, x: np.ndarray) -> np.ndarray:
        pass

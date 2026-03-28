import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Any

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
    def bounds(self):
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

    def plot(self, axs: plt.Axes, **kwargs):
        return self.domain.plot(axs=axs, **kwargs)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "CellDomain":
        """
        Create a CellDomain instance from a dictionary, with the following
        structure:

        ```py
        # dict(shape=<name>, bounds=[x_min, y_min, x_max, y_max])
        domain_config = dict(shape=rectangle, bounds=[0, 0, 20, 25])
        ```

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
        x_min, y_min, x_max, y_max = cell_bounds

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                "Invalid bounds: x_min must be less than x_max and "
                "y_min must be less than y_max."
            )

        return cls(bounds=[x_min, y_min, x_max, y_max])


@dataclass
class PositionSampler:
    xc_sampler: DistributionSampler
    yc_sampler: DistributionSampler
    zc_sampler: DistributionSampler = None
    azimuthal_angle_sampler: DistributionSampler = None
    polar_angle_sampler: DistributionSampler = None

    def sample(self) -> dict[str, float]:
        out = dict(
            xc=self.xc_sampler.sample(),
            yc=self.yc_sampler.sample(),
        )
        if self.zc_sampler is not None:
            out["zc"] = self.zc_sampler.sample()
        if self.azimuthal_angle_sampler is not None:
            out["azimuthal_angle"] = self.azimuthal_angle_sampler.sample()
        if self.polar_angle_sampler is not None:
            out["polar_angle"] = self.polar_angle_sampler.sample()

        return out


class InclusionSampler:
    def __init__(
        self,
        shape: str | gb.GShape,
        pos_sampler: DistributionSampler,
        size_samplers: dict[str, DistributionSampler],
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
        self.pos_sampler = pos_sampler
        self.size_samplers = size_samplers

    def sample(self) -> gb.GShape:
        pos_params = self.pos_sampler.sample()
        size_params = {k: v.sample() for k, v in self.size_samplers.items()}
        return self.shape.from_params(
            positional_params=pos_params, size_params=size_params
        )


def initialise_shapes(
    shapes_params: dict | list[dict],
    cell_domain: CellDomain,
    *,
    init_method: str = "uniform",
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
        - `name`: The shape of the inclusion (e.g., "circle", "ellipse").
        - `vf`: The volume fraction of the inclusion.
        - `params`: Additional parameters specific to the inclusion shape.

    Returns
    -------
    dict[str, list[gb.GShape]]
        A dictionary where the keys are the shape names and the values are
        lists of initialized inclusion objects of that shape.
    """
    if isinstance(shapes_params, dict):
        shapes_params = [shapes_params]

    non_dict_items = [nd for nd in shapes_params if not isinstance(nd, dict)]
    if non_dict_items:
        raise TypeError(
            "Configuration must be a single dictionary or a list of "
            f"dictionaries. Found non-dictionary items: {non_dict_items}."
        )

    # xc and yc distributions are not specified in the config,
    x_min, x_max = cell_domain.x_bounds
    y_min, y_max = cell_domain.y_bounds
    if init_method != "uniform":
        raise NotImplementedError(
            f"Init method {init_method} for inclusions is not supported."
        )
    pos_sampler = PositionSampler(
        xc_sampler=DistributionSampler.from_signature(
            f"uniform({x_min}, {x_max - x_min})", rng=rng
        ),
        yc_sampler=DistributionSampler.from_signature(
            f"uniform({y_min}, {y_max - y_min})", rng=rng
        ),
    )

    initialised_inclusions: dict[str, List[gb.GShape]] = {}
    cum_vf = 0.0
    for ith_incl_config in shapes_params:
        name, vf, size_sampler_sigs = _validate_dict(
            ith_incl_config,
            keys=["name", "vf", "params"],
            val_types=[str, float, dict],
            val_ranges=[None, (0.0, 1.0), None],
            ret_val=True,
        )
        cum_vf += vf
        if cum_vf > 1.0:
            raise ValueError(
                "Cumulative volume fraction of given shapes exceeds 1.0."
            )
        size_samplers = {
            p_name: DistributionSampler.from_signature(sig, rng=rng)
            for p_name, sig in size_sampler_sigs.items()
        }

        # xy_sampler independent of shape
        # params: dependent of shape
        incl_sampler = InclusionSampler(
            shape=name,
            pos_sampler=pos_sampler,
            size_samplers=size_samplers,
        )
        required_volume = cell_domain.cell_volume * vf
        cumulative_volume = 0.0
        generated_inclusions = []
        while cumulative_volume < required_volume:
            a_inclusion = incl_sampler.sample()
            generated_inclusions.append(a_inclusion)
            cumulative_volume += a_inclusion.volume()

        initialised_inclusions[name] = generated_inclusions
    return initialised_inclusions


class CellCirclesOverlap(OptimisationProblem):
    """
    Overlap cost for a 2-D cell with circular inclusions, no periodicity.

    Parameters
    ----------
    domain : CellDomain
    shapes : dict[str, list[gb.GShape]]
        Output of ``initialise_shapes``; all shapes must be circles.
    ssd_ratio : float
        Minimum surface-to-surface gap as a fraction of each circle's
        radius.  Default 0.04 (4 %).
    proj_buffer_ratio : float
        Projection buffer thickness = proj_buffer_ratio × radius.
        Default 2.0 (mirrors Julia default).
    """

    def __init__(
        self,
        domain: CellDomain,
        shapes: dict[str, list[gb.Circle]],
        *,
        ssd_ratio: float = 0.05,
        proj_buffer_ratio: float = 2.0,
    ):
        super().__init__()
        self.domain = domain
        self.shapes = shapes

        self._inclusions: list[gb.Circle] = []
        for g in shapes.values():
            self._inclusions.extend(g)
        self._num_inclusions = len(self._inclusions)

        self.x0 = [i.centre.x for i in self._inclusions] + [
            i.centre.y for i in self._inclusions
        ]
        self._radii = np.array([i.radius for i in self._inclusions])
        self._ssd = ssd_ratio * self._radii
        self._proj_buffer = proj_buffer_ratio * self._radii

    def _overlap_cost_and_gradient(self, positions: np.ndarray):
        xs, ys = positions.T

        cost = 0.0
        grad_x = np.zeros(self._num_inclusions)
        grad_y = np.zeros(self._num_inclusions)

        for i in range(self._num_inclusions):
            for j in range(1 + i, self._num_inclusions):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.hypot(dx, dy)

                dca = self._radii[i] + self._radii[j] + self._ssd[i]
                c = dca - dist

                if c > 0.0:
                    dol = c / (dist + 1e-6)  # degree of overlap

                    cost += c * c  # making convex

                    tmp_gx = dol * dx
                    tmp_gy = dol * dy
                    grad_x[i] += tmp_gx
                    grad_x[j] -= tmp_gx
                    grad_y[i] += tmp_gy
                    grad_y[j] -= tmp_gy

        grad = -2.0 * np.column_stack([grad_x, grad_y])
        return cost, grad

    def f_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        positions = x.reshape(-1, 2, order="F")  # x, y
        f, g = self._overlap_cost_and_gradient(positions)
        self.eval_count["f_and_g"] += 1
        return f, g.flatten(order="F")

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = x.reshape(-1, 2, order="F")
        xlb, xub = self.domain.x_bounds
        ylb, yub = self.domain.y_bounds

        for i in range(self._num_inclusions):
            buf_len = self._proj_buffer[i] * np.random.random()
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            if positions[i, 1] > yub:
                positions[i, 1] = yub - buf_len
            elif positions[i, 1] < ylb:
                positions[i, 1] = ylb + buf_len

        self.eval_count["proj"] += 1
        return positions.flatten(order="F")


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        shapes: dict[str, List[gb.GShape]] = None,
    ):
        self.domain = domain
        self.shapes = shapes
        #
        self._opt_problem = None

    def plot(
        self,
        shape_vis_options: dict | None = None,
        domain_vis_options: dict | None = None,
        *,
        image_size: tuple[int, int] = (256, 256),
        f_path: Path | str | None = None,
    ):
        """
        Plot the cell with its domain and inclusions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new axes will be created.
        shape_options : dict, optional
            Keyword arguments passed to the respective shape's
            `plot` method.
        domain_vis_options : dict, optional
            Keyword arguments passed to the domain's `plot` method.
        """
        shape_vis_options = shape_vis_options or {}
        domain_vis_options = domain_vis_options or {}

        shapes_plotter = gb.utils.ShapesPlotter(
            shape_options=None,
            bg_options={
                **domain_vis_options,
                "bounds": self.domain.bounds,
            },
            fig_options=None,
            image_options={
                "dpi": 100,
                "size": image_size,
                "mode": "L",
                "as_array": False,
                "dtype": "uint8",
            }
        )

        for a_group_of_inclusions in self.shapes.values():
            for a_inclusion in a_group_of_inclusions:
                incl_patch = a_inclusion.get_patch(**shape_vis_options)
                shapes_plotter.add_patch(incl_patch)

        shapes_plotter.saveas(f_path)
        shapes_plotter.close()
        # # TODO replace this custom axis formatting with a dedicated function
        # axs.set_aspect("equal")
        # axs.axis("off")
        # axs.set_xlim(*self.domain.x_bounds)
        # axs.set_ylim(*self.domain.y_bounds)

    def remove_inclusion_overlaps(self, ssd_ratio, proj_buffer_ratio) -> None:
        # Run Optim Loop to ensure there are no overlaps among inclusions
        #   Evaluate the cost function and gradients
        #     Add Periodic copies, if required
        #     cost evaluation
        #     gradients evaluation
        #   Update the inclusions positions
        #   Check for convergence
        #   If converged, return the optimised inclusions positions
        self._opt_problem = CellCirclesOverlap(
            domain=self.domain,
            shapes=self.shapes,
            ssd_ratio=ssd_ratio,
            proj_buffer_ratio=proj_buffer_ratio,
        )
        result = nmspg(
            objective=self._opt_problem,
            x0=self._opt_problem.x0,
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

        positions = result.x_optimal.reshape(-1, 2, order="F")
        for k, shapes_list in self.shapes.items():
            for idx, a_shape in enumerate(shapes_list):
                a_shape.centre = gb.Point2D(
                    positions[idx, 0], positions[idx, 1]
                )

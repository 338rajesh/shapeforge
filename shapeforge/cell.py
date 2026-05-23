import math
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Any

import gbox as gb
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .optim import nmspg, OptimisationProblem
from .utils import _validate_dict, DistributionSampler


class CellDomain:
    """
    It represents a domain for a cell, without any inclusions. It can be
    considered as a host for inclusions or other geometrical features, but it
    does not contain any by itself.
    """

    def __init__(self, **options):
        self.domain: gb.GShape2D = self._get_domain(options)

    def _get_domain(self, options: dict):
        shape = options.get("shape")
        if shape is None:
            raise ValueError("Shape not specified.")
        if shape == "rectangle":
            centre = options.get("centre")
            if centre is None:
                print("Warning: centre not specified, using (0, 0).")
                centre = (0.0, 0.0)
            semi_major_length = options.get("semi_major_length")
            semi_minor_length = options.get("semi_minor_length")
            major_axis_angle = options.get("major_axis_angle")
            return gb.Rectangle(
                semi_major_length=semi_major_length,
                semi_minor_length=semi_minor_length,
                major_axis_angle=major_axis_angle,
                centre=centre,
            )
        else:
            raise NotImplementedError(f"Shape {shape} not implemented.")

    @property
    def bounds(self) -> list[float]:
        _bounds = getattr(self.domain, "bounds", None)
        if _bounds is None:
            raise ValueError("Bounds not set.")
        return list(_bounds)

    @property
    def volume(self) -> float:
        """
        Calculate the volume of the cell domain.
        """
        return self.domain.volume

    def plot(self, axs: plt.Axes, **kwargs):
        return self.domain.plot(axs=axs, **kwargs)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "CellDomain":
        """
        Create a cell domain from a dictionary.
        """
        return cls(**config)


@dataclass
class CellElement:
    name: str
    element: gb.GShape

    def __post_init__(self):
        if not isinstance(self.element, gb.GShape):
            raise ValueError(f"Element {self.element} is not a GShape.")

    def initialise(self, domain: CellDomain):
        pass


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        elements: Sequence[CellElement] = None,
    ):
        self.domain = domain
        self.elements = elements
        #
        self._opt_problem = None

    def clone(self) -> "Cell":
        return Cell(self.domain, self.elements)

    @classmethod
    def initialise(cls, cell_config: dict, rng_seed, init_method="uniform"):
        cell_domain = CellDomain.from_dict(cell_config["domain"])
        elements = cell_config.get("elements", [])
        if len(elements) == 0:
            print(
                "WARNING: no elements are found in the cell config, so returning "
                "empty cell domain."
            )
            return Cell(cell_domain)  # just return the empty cell domain

        elements = initialise_elements(
            elements,
            cell_domain,
            rng=np.random.default_rng(seed=rng_seed),
            init_method=init_method,
        )

        return cls(cell_domain, elements)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError()

    def from_dict(config: dict[str, Any]) -> "Cell":
        raise NotImplementedError()

    def save(self, f_path: Path | str) -> Path:
        """
        Save the cell config to a file, for creating the same cell later
        using the `load` method.

        Parameters
        ----------
        f_path: str or Path
            File path to save the cell. If not provided the cell will be saved
            in the current directory with a default name.

        """
        f_path = Path(f_path)
        if f_path.exists():
            raise FileExistsError(f_path)
        if f_path.suffix == "":
            f_path = f_path.with_suffix(".json")

        if f_path.suffix != ".json":
            raise ValueError(
                f"Unsupported file type: {f_path.suffix}. "
                f"Only .json files are supported."
            )
        with open(f_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(f_path: Path | str) -> "Cell":
        with open(f_path, "r") as f:
            config = json.load(f)
        return Cell.from_dict(config)

    def export(self, f_path: Path | str, options: dict):
        f_path = Path(f_path)
        f_suffix = f_path.suffix.lower()
        if f_suffix in (".pickle", ".pkl"):
            with open(f_path, "wb") as f:
                pickle.dump(self, f)
        elif f_suffix in (".yaml", ".yml"):
            with open(f_path, "w") as f:
                yaml.dump(self.to_dict(), f)
        elif f_suffix == ".json":
            with open(f_path, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        elif f_suffix == ".npz":
            with open(f_path, "wb") as f:
                np.savez(f, **self.to_dict())
        elif f_suffix == ".png":
            self._plot(f_path=f_path, **options)
        else:
            raise ValueError(f"Unsupported file type: {f_suffix}.")

    def _plot(
        self,
        element_facecolor: str = "white",
        element_edgecolor: str = "black",
        bg_facecolor: str = "black",
        bg_edgecolor: str = "white",
        size: tuple[int, int] = (256, 256),
        dpi: int = 100,
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
        shapes_plotter = gb.utils.ShapesPlotter(
            shape_options={
                "facecolor": element_facecolor,
                "edgecolor": element_edgecolor,
            },
            bg_options={
                "edgecolor": bg_edgecolor,
                "facecolor": bg_facecolor,
                "bounds": self.domain.bounds,
            },
            image_options={
                "dpi": dpi,
                "size": size,
                "mode": "L",
                "dtype": "uint8",
            },
        )

        for a_group_of_inclusions in self.elements.values():
            for a_inclusion in a_group_of_inclusions:
                shapes_plotter.add_shape(a_inclusion)

        shapes_plotter.saveas(f_path)
        shapes_plotter.close()

    def remove_inclusion_overlaps(self, ssd_ratio, proj_buffer_ratio) -> None:
        # Run Optim Loop to ensure there are no overlaps among inclusions
        #   Evaluate the cost function and gradients
        #     Add Periodic copies, if required
        #     cost evaluation
        #     gradients evaluation
        #   Update the inclusions positions
        #   Check for convergence
        #   If converged, return the optimised inclusions positions
        self._opt_problem = GShapes2DOverlap(
            domain=self.domain,
            shapes=self.elements,
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
        for k, shapes_list in self.elements.items():
            for idx, a_shape in enumerate(shapes_list):
                a_shape.centre = positions[idx, 0:2]


@dataclass
class PositionSampler:
    xc_sampler: DistributionSampler
    yc_sampler: DistributionSampler
    zc_sampler: DistributionSampler = None
    azimuthal_angle_sampler: DistributionSampler = None
    polar_angle_sampler: DistributionSampler = None

    def sample(self) -> dict[str, float]:
        """
        Sample the position of the inclusion in the cell

        Returns
        -------
        dict[str, float]
            A dictionary containing the sampled position of the inclusion
            in the cell. The keys are "xc", "yc", "zc", "major_axis_angle",
            and "polar_angle".
        """
        out = dict(
            xc=self.xc_sampler.sample(),
            yc=self.yc_sampler.sample(),
        )
        if self.zc_sampler is not None:
            out["zc"] = self.zc_sampler.sample()
        if self.azimuthal_angle_sampler is not None:
            out["major_axis_angle"] = self.azimuthal_angle_sampler.sample()
        if self.polar_angle_sampler is not None:
            out["polar_angle"] = self.polar_angle_sampler.sample()

        return out


class InclusionSampler:
    def __init__(
        self,
        shape: str | gb.GShape,
        pos_sampler: PositionSampler,
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
        self.pos_sampler: PositionSampler = pos_sampler
        self.size_samplers = size_samplers

    def sample(self) -> gb.GShape:
        pos_params = self.pos_sampler.sample()
        size_params = {k: v.sample() for k, v in self.size_samplers.items()}
        return self.shape.from_params(
            positional_params=pos_params, size_params=size_params
        )


def initialise_elements(
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
    x_min, x_max, y_min, y_max = cell_domain.bounds
    if init_method != "uniform":
        raise NotImplementedError(
            f"Init method {init_method} for inclusions is not supported."
        )
    pos_sampler = PositionSampler(
        xc_sampler=DistributionSampler.from_signature(
            f"{init_method}({x_min}, {x_max - x_min})", rng=rng
        ),
        yc_sampler=DistributionSampler.from_signature(
            f"{init_method}({y_min}, {y_max - y_min})", rng=rng
        ),
        azimuthal_angle_sampler=DistributionSampler.from_signature(
            f"{init_method}(0.0, 360.0)", rng=rng
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
        required_volume = cell_domain.volume * vf
        cumulative_volume = 0.0
        generated_inclusions = []
        while cumulative_volume < required_volume:
            a_inclusion = incl_sampler.sample()
            generated_inclusions.append(a_inclusion)
            cumulative_volume += a_inclusion.volume

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
        Projection buffer thickness = proj_buffer_ratio x radius.
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


class GShapes2DOverlap(OptimisationProblem):
    """
    Cumulative overlap cost of various `gbox.GShape2D` shapes

    Parameters
    ----------
    domain : CellDomain
    shapes : dict[str, list[gb.GShape2D]]
        Output of ``initialise_shapes``; all shapes must be circles.
    ssd_ratio : float
        Minimum surface-to-surface gap as a fraction of each circle's
        radius.  Default 0.04 (4 %).
    proj_buffer_ratio : float
        Projection buffer thickness = proj_buffer_ratio x radius.
        Default 2.0 (mirrors Julia default).
    """

    def __init__(
        self,
        domain: CellDomain,
        shapes: dict[str, list[gb.GShape2D]],
        *,
        ssd_ratio: float = 0.05,
        proj_buffer_ratio: float = 2.0,
    ):
        super().__init__()
        self.domain = domain
        self.shapes = shapes

        self._inclusions: list[gb.GShape2D] = []
        for g in shapes.values():
            self._inclusions.extend(g)
        self._num_inclusions = len(self._inclusions)

        self.x0 = (
            [i.centre.x for i in self._inclusions]
            + [i.centre.y for i in self._inclusions]
            + [i.major_axis_angle for i in self._inclusions]
        )
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

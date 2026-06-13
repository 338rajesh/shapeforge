import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gbox as gb
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .optim import OptimisationProblem, nmspg
from .utils import DistributionSampler, _validate_dict


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

    @classmethod
    def from_signatures(
        cls, x_min, x_max, y_min, y_max, init_method, rng
    ) -> "PositionSampler":
        if init_method != "uniform":
            raise NotImplementedError(
                f"Init method {init_method} is not supported for "
                "position sampling of elements."
            )
        return cls(
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
        """
        Get bounds of the domain in the order of `[xmin, ymin, xmax, ymax]`.
        """
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


# ----------------------------------------------------
# region CellElement
# ----------------------------------------------------


@dataclass
class CellElement:
    name: str
    element: gb.GShape

    def __post_init__(self):
        if not isinstance(self.element, gb.GShape):
            raise ValueError(f"Element {self.element} is not a GShape.")

    @property
    def volume(self):
        return self.element.volume

    @property
    def centre(self):
        return self.element.centre

    @property
    def major_axis_angle(self):
        return self.element.major_axis_angle

    @property
    def pose(self):
        return self.element.pose

    @property
    def bounding_box(self):
        return self.element.bounding_box

    def translate_and_rotate(self, *args, **kwargs):
        return self.__class__(
            name=self.name,
            element=self.element.translate_and_rotate(*args, **kwargs),
        )

    def sample(self, domain: CellDomain):
        pass

    def union_of_nspheres(self, *args, **kwargs) -> gb.CirclesArray:
        return self.element.union_of_nspheres(*args, **kwargs)

    @classmethod
    def initialise(
        cls,
        domain: CellDomain,
        elements_config: dict,
        *,
        init_method="uniform",
        rng=None,
    ):
        x_min, y_min, x_max, y_max = domain.bounds
        pos_sampler = PositionSampler.from_signatures(
            x_min, x_max, y_min, y_max, init_method="uniform", rng=rng
        )

        elements: list[CellElement] = []
        cum_vf = 0.0
        for ith_ele_config in elements_config:
            name, vf, size_sampler_sigs = _validate_dict(
                ith_ele_config,
                keys=["name", "vf", "params"],
                val_types=[str, float, dict],
                val_ranges=[None, (0.0, 1.0), None],
                ret_val=True,
            )
            cum_vf += vf
            if cum_vf > 1.0:
                raise ValueError(
                    f"Total volume fraction of inclusions exceeds 1.0. Found {cum_vf}."
                )
            size_samplers = {
                p_name: DistributionSampler.from_signature(sig, rng=rng)
                for p_name, sig in size_sampler_sigs.items()
            }

            g_element = getattr(gb, name, None)
            if not callable(getattr(g_element, "from_params")):
                raise ValueError(
                    f"Element name '{name}' is not valid."
                    f"It must contain a method 'from_params'."
                )

            required_volume = domain.volume * vf
            cumulative_volume = 0.0
            generated_elements = []
            while cumulative_volume < required_volume:
                pos_params = pos_sampler.sample()
                size_params = {k: v.sample() for k, v in size_samplers.items()}
                g_element = g_element.from_params(
                    positional_params=pos_params, size_params=size_params
                )
                a_element = cls(name=name, element=g_element)
                generated_elements.append(a_element)
                cumulative_volume += a_element.volume

            elements.extend(generated_elements)

        return elements


# ----------------------------------------------------------------------------
# region Cell
# ----------------------------------------------------------------------------


class Cell:
    def __init__(
        self,
        domain: CellDomain,
        elements: Sequence[CellElement] = None,
    ):
        self.domain = domain
        self.elements: Sequence[CellElement] = elements
        #
        self._opt_problem = None

    def clone(self) -> "Cell":
        return Cell(self.domain, self.elements)

    @classmethod
    def initialise(
        cls,
        cell_config: dict,
        rng_seed,
        init_method="uniform",
    ):
        cell_domain = CellDomain.from_dict(cell_config["domain"])

        elements = cell_config.get("elements", [])
        if len(elements) == 0:
            print(
                "WARNING: no elements are found in the cell config, so returning "
                "empty cell domain."
            )
            return Cell(cell_domain)  # just return the empty cell domain

        elements = CellElement.initialise(
            cell_domain,
            elements,
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
        domain_facecolor: str = "black",
        domain_edgecolor: str = "white",
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
                "edgecolor": domain_edgecolor,
                "facecolor": domain_facecolor,
                "bounds": self.domain.bounds,
            },
            image_options={
                "dpi": dpi,
                "size": size,
                "mode": "L",
                "dtype": "uint8",
            },
        )

        for a_element in self.elements:
            shapes_plotter.add_shape(a_element.element)

        shapes_plotter.saveas(f_path)
        shapes_plotter.close()

    def remove_inclusion_overlaps(self, ssd_ratio, *, logger=None) -> None:

        self._opt_problem = GShapes2DOverlap(
            domain=self.domain,
            elements=self.elements,
            ssd_ratio=ssd_ratio,
        )

        logger.log(">> Solving the optimisation problem...")
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

        self.elements = self._opt_problem._get_updated_ele(result.x_optimal)
        # positions = result.x_optimal.reshape(len(self.elements), 3, order="F")
        # for idx, a_element in enumerate(self.elements):
        #     a_element.element.centre = positions[idx, 0:2]


# ---------------------------------------------------------
# region OverlapProblem
# ---------------------------------------------------------


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
        elements: Sequence[gb.GShape2D],
        *,
        ssd_ratio: float = 0.05,
        # proj_buffer_ratio: float = 2.0,
    ):
        super().__init__()

        self._num_inclusions = len(elements)
        self.domain = domain
        self.elements: list[gb.GShape2D] = elements
        self._ssd_factor = float(ssd_ratio)
        self.x0 = self._poses_to_x0(elements)

    def _poses_to_x0(self, elements: list[gb.GShape2D]) -> np.ndarray:
        """
        Collect (x_i, y_i, θ_i) from each element and flatten to 1-D.

        Parameters
        ----------
        elements : list[gb.GShape2D]
            List of geometric shapes.

        Returns
        -------
        x0 : np.ndarray
            1-D array of (x_i, y_i, θ_i) values.
        """
        xyt_i = np.array([ele.pose for ele in elements], dtype=np.float32)

        return xyt_i.flatten()

    def _x0_to_poses(self, x0: np.ndarray):
        """Return (N, 3) array of [x_i, y_i, θ_i] from flat vector."""
        return np.asarray(x0, dtype=np.float32).reshape(self._num_inclusions, 3)

    def _eval_cost_and_gradient(self, ele: list[gb.GShape2D], ssd_factor: float):
        num_inclusions = len(ele)
        cost = 0.0
        grad_x = np.zeros(num_inclusions)
        grad_y = np.zeros(num_inclusions)
        grad_t = np.zeros(num_inclusions)

        for i in range(num_inclusions):
            ei = ele[i]
            for j in range(i + 1, num_inclusions):
                ej = ele[j]

                if not ei.bounding_box.overlaps(ej.bounding_box):
                    continue

                ca_i = ei.union_of_nspheres(dh=0.02).data
                ca_j = ej.union_of_nspheres(dh=0.02).data

                for ca_ik_x, ca_ik_y, ca_ik_r in ca_i:
                    for ca_jl_x, ca_jl_y, ca_jl_r in ca_j:
                        dx = ca_ik_x - ca_jl_x
                        dy = ca_ik_y - ca_jl_y
                        doc = (ca_ik_r + ca_jl_r) * (1.0 + ssd_factor)
                        d_ikjl = math.hypot(dx, dy)
                        cost_ikjl = doc - d_ikjl
                        if cost_ikjl > 0:
                            cost += cost_ikjl * cost_ikjl

                            dol = cost_ikjl / (d_ikjl + 1e-6)
                            tmp_gx = dol * dx
                            tmp_gy = dol * dy
                            grad_x[i] += tmp_gx
                            grad_x[j] -= tmp_gx
                            grad_y[i] += tmp_gy
                            grad_y[j] -= tmp_gy
                            grad_t[i] += dol * (
                                dx * (ei.centre.y - ca_ik_y)
                                - (dy * (ei.centre.x - ca_ik_x))
                            )
                            grad_t[j] -= dol * (
                                dx * (ej.centre.y - ca_jl_y)
                                - (dy * (ej.centre.x - ca_jl_x))
                            )

        grad = -2.0 * np.column_stack([grad_x, grad_y, grad_t])
        return cost, grad

    def _get_updated_ele(self, x: np.ndarray):
        pose = self._x0_to_poses(x)
        updated_ele = []
        for idx, (dx, dy, dtheta) in enumerate(pose):
            ei = self.elements[idx]
            updated_ele.append(ei.translate_and_rotate(dx, dy, dtheta))
        return updated_ele

    def f_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        new_ele_pose = self._get_updated_ele(x)
        f, g = self._eval_cost_and_gradient(new_ele_pose, self._ssd_factor)
        self.eval_count["f_and_g"] += 1
        return f, g.flatten(order="F")

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._x0_to_poses(x)  # xyr
        xlb, ylb, xub, yub = self.domain.bounds

        for i in range(self._num_inclusions):
            if positions[i, 0] > xub:
                positions[i, 0] = 2.0 * xub - positions[i, 0]
            elif positions[i, 0] < xlb:
                positions[i, 0] = 2.0 * xlb - positions[i, 0]

            if positions[i, 1] > yub:
                positions[i, 1] = 2.0 * yub - positions[i, 1]
            elif positions[i, 1] < ylb:
                positions[i, 1] = 2.0 * ylb - positions[i, 1]

        self.eval_count["proj"] += 1
        return positions.flatten(order="F")

    def update_pose(self):
        return

import json
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import gbox as gb
import numpy as np

from .config import ShapeConfig
from .optim import OptimisationProblem, nmspg
from .utils import DistributionSampler, DistributionSpec, Validator, get_logger

logger = get_logger(__name__)


class CellDomain2D:
    """
    It represents a 2D domain for a cell, without any inclusions. It can be
    considered as a host for inclusions or other geometrical features, but it
    does not contain any by itself.
    """

    __slots__ = ("_domain",)

    def __init__(self, bounds: gb.Bounds2DRectangular | dict[str, float]):
        if isinstance(bounds, gb.Bounds2DRectangular):
            pass
        if isinstance(bounds, Mapping):
            bounds = gb.Bounds2DRectangular.from_mapping(bounds)
        elif isinstance(bounds, Sequence):
            bounds = gb.Bounds2DRectangular.from_sequence(bounds)
        else:
            raise TypeError(
                "Invalid type for bounds. Expected gb.Bounds2DRectangular or "
                f"Sequence or Mapping, but got {type(bounds).__name__}"
            )
        self._domain = bounds
        logger.debug("> Initialised the `CellDomain2D`")

    @property
    def bounds(self) -> Mapping[str, float]:
        return self._domain.bounds

    @property
    def x_bounds(self) -> tuple[float, float]:
        """
        Get the x bounds of the cell domain.
        """
        return self._domain.x_min, self._domain.x_max

    @property
    def y_bounds(self) -> tuple[float, float]:
        """
        Get the y bounds of the cell domain.
        """
        return self._domain.y_min, self._domain.y_max

    @property
    def area(self) -> float:
        """
        Calculate the volume of the cell domain.
        """
        return self._domain.area

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(d, keys=["bounds"], types=[dict])
        return cls(gb.Bounds2DRectangular.from_mapping(d["bounds"]))

    def to_dict(self) -> dict:
        return {"bounds": dict(self._domain.bounds)}


class Inclusions2D(Collection[gb.Shape2D]):
    """
    An unordered collection of geometric inclusions.It stores
    ``gb.Shape2D`` instances.  It does not concern itself
    with inclusions generation or sampling.

    Parameters
    ----------
    inclusions
        Initial inclusions to add to the collection.

    Examples
    --------
    >>> from gbox.shape import Circle, Ellipse
    >>> inclusions = Inclusions2D([Circle(3.0), Circle(1.2)])
    >>> inclusions.add(Circle(2.0))
    >>> inclusions.add(Ellipse(2.0, 1.0))

    The collection behaves like a collection:

    >>> len(inclusions)
    2
    """

    __slots__ = ("_inclusions",)

    def __init__(self, inclusions: Iterable[gb.Shape2D] | None = None) -> None:
        self._inclusions: list[gb.Shape2D] = []

        if inclusions is not None:
            self.extend(inclusions)

    def __len__(self) -> int:
        return len(self._inclusions)

    def __iter__(self) -> Iterator[gb.Shape2D]:
        return iter(self._inclusions)

    def __contains__(self, inclusion: object) -> bool:
        return inclusion in self._inclusions

    def add(self, inclusion: gb.Shape2D) -> None:
        """Add a single inclusion to the collection."""
        Validator.is_type(inclusion, gb.Shape2D, name="inclusion")
        self._inclusions.append(inclusion)

    def extend(self, inclusions: Iterable[gb.Shape2D]) -> None:
        """Add multiple inclusions to the collection."""
        for inclusion in inclusions:
            self.add(inclusion)

    @property
    def area(self) -> float:
        """Total area occupied by the inclusions."""
        return sum(inclusion.area for inclusion in self._inclusions)

    def area_fraction(self, domain_area: float) -> float:
        """Return the fraction of the domain occupied by the inclusions."""
        if domain_area <= 0:
            raise ValueError("domain_area must be greater than zero.")

        return self.area / domain_area

    def get_by_shape(self, shape: type[gb.Shape2D]) -> list[gb.Shape2D]:
        """Return all inclusions of the specified shape type."""
        return [
            inclusion
            for inclusion in self._inclusions
            if isinstance(inclusion, shape)
        ]

    def get_count(self) -> dict[type[gb.Shape2D], int]:
        """Return the number of inclusions of each shape type."""
        counts: dict[type[gb.Shape2D], int] = defaultdict(int)

        for inclusion in self._inclusions:
            counts[type(inclusion)] += 1

        return dict(counts)

    def clear(self) -> None:
        """Remove all inclusions from the collection."""
        self._inclusions.clear()

    def clone(self) -> Self:
        """Create a deep copy of this instance."""
        return self.__class__(self._inclusions)

    def to_dict(self) -> dict:
        return {
            "inclusions": [
                inclusion.to_dict() for inclusion in self._inclusions
            ]
        }

    @classmethod
    def from_dict(self, d: dict) -> Self:
        d = Validator.as_dict(
            d, key_type_map={"inclusions": Sequence}, reject_extra_keys=True
        )
        inclusions = [
            gb.Shape2D.from_dict(inclusion) for inclusion in d["inclusions"]
        ]
        return self.__class__(inclusions)


@dataclass
class PositionSampler2D:
    xc_sampler: DistributionSampler | None = None
    yc_sampler: DistributionSampler | None = None
    orientation_sampler: DistributionSampler | None = None

    def sample(self) -> dict[str, float | gb.Angle]:
        out = {}
        if isinstance(self.xc_sampler, DistributionSampler):
            out["xc"] = self.xc_sampler.sample()

        if isinstance(self.yc_sampler, DistributionSampler):
            out["yc"] = self.yc_sampler.sample()

        if isinstance(self.orientation_sampler, DistributionSampler):
            out["major_axis_angle"] = gb.Angle.rad(
                self.orientation_sampler.sample()
            )

        return out


class Inclusion2DSampler:
    __slots__ = ("_pos_sampler", "_shape", "_size_samplers")

    def __init__(
        self,
        shape: str,
        pos_sampler: DistributionSampler,
        size_samplers: dict[str, float | DistributionSampler],
    ):
        shape = Validator.as_string(shape).lower()
        gb_shape = gb.shapes.SHAPES_2D_MAPPING[shape]

        # check if g_shape has a method called from_params
        if not hasattr(gb_shape, "from_params") or not callable(
            gb_shape.from_params
        ):
            raise ValueError(
                f"Unsupported shape: {shape}, must have a from_params method"
            )

        self._shape: gb.Shape2D = gb_shape
        self._pos_sampler = pos_sampler
        self._size_samplers = Validator.as_dict(size_samplers)

    def sample(self) -> gb.Shape2D:
        pos_params = self._pos_sampler.sample()
        size_params = {}
        for k, v in self._size_samplers.items():
            if isinstance(v, (int, float)):
                size_params[k] = v
            else:
                size_params[k] = v.sample()
        return self._shape.from_params(
            position_params=pos_params, size_params=size_params
        )


def initialise_shapes_2d(
    cell_domain: CellDomain2D,
    shapes_config: Sequence[ShapeConfig],
    rng: np.random.Generator,
) -> Inclusions2D:
    """
    It is the main function to initialize inclusions based on the provided
    configuration. It iterates through the configuration list, extracting
    the shape, volume fraction, and parameters for each inclusion type.
    It then calls the appropriate function to generate and initialize the
    inclusions based on the shape specified in the configuration.

    Parameters
    ----------
    cell_domain : CellDomain2D
        The 2D cell domain in which the inclusions will be placed.
    shapes_config : Sequence[ShapeConfig]
        A sequence of ShapeConfig objects, each containing the shape type,
        volume fraction, and parameters for each inclusion type.
    rng: np.random.Generator
        A random number generator to ensure reproducibility.

    Returns
    -------
    Inclusions2D
       A Inclusions2D object containing the initialized inclusions.
    """
    x_min, x_max = cell_domain.x_bounds
    y_min, y_max = cell_domain.y_bounds

    pos_sampler = PositionSampler2D(
        xc_sampler=DistributionSampler.from_signature(
            f"uniform({x_min}, {x_max - x_min})", rng=rng
        ),
        yc_sampler=DistributionSampler.from_signature(
            f"uniform({y_min}, {y_max - y_min})", rng=rng
        ),
    )

    inclusions = Inclusions2D()
    cum_vf = 0.0
    for a_shape_cfg in shapes_config:
        cum_vf += a_shape_cfg.volume_fraction
        if cum_vf > 1.0:
            raise ValueError(
                "Cumulative volume fraction of given shapes exceeds 1.0."
            )

        size_samplers = {}
        for p_name, p_spec in a_shape_cfg.params.items():
            if isinstance(p_name, (int, float)):
                size_samplers[p_name] = p_spec
            elif isinstance(p_name, DistributionSpec):
                size_samplers[p_name] = DistributionSampler(p_spec, rng)
            elif isinstance(p_name, str):
                size_samplers[p_name] = DistributionSampler.from_signature(
                    p_spec, rng
                )
            else:
                raise TypeError(
                    f"Invalid distribution spec type: {type(p_spec).__name__}"
                )

        incl_sampler = Inclusion2DSampler(
            shape=a_shape_cfg.name,
            pos_sampler=pos_sampler,
            size_samplers=size_samplers,
        )

        required_area = cell_domain.area * a_shape_cfg.volume_fraction
        cumulative_area = 0.0

        while cumulative_area < required_area:
            a_inclusion = incl_sampler.sample()
            inclusions.add(a_inclusion)
            cumulative_area += a_inclusion.area()

    return inclusions


class CellShapes2DOverlap(OptimisationProblem):
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
        domain: CellDomain2D,
        shapes: Inclusions2D,
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


class Cell2D:
    __slots__ = ("_domain", "_opt_problem", "_shapes")

    def __init__(self, domain: CellDomain2D, shapes: Inclusions2D):
        self._domain = domain
        self._shapes = shapes
        self._opt_problem = None

    @property
    def domain(self) -> CellDomain2D:
        return self._domain

    @property
    def shapes(self) -> Inclusions2D:
        return self._shapes

    def clone(self) -> Self:
        return self.__class__(self._domain, self._shapes)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(
            d,
            key_type_map={"domain": dict, "shapes": dict},
            reject_extra_keys=True,
            name="Cell2D.from_dict.d",
        )

    def to_dict(self) -> dict:
        return {
            "domain": self._domain.to_dict(),
            "shapes": self._shapes.to_dict(),
        }

    def save(self, f_path: Path | str, *, overwrite: bool = False) -> Path:
        """Save the configuration of the cell such that it can be loaded later.

        Parameters
        ----------
        f_path: str or Path
            Path to save the cell as a json file.

        Returns
        -------
        Path
            Path to the saved file.
        """
        f_path = Validator.file_path(
            f_path,
            must_exist=None if overwrite else False,
            extensions=[".json"],
        )
        with open(f_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return f_path

    @classmethod
    def load(cls, f_path: str | Path) -> Self:
        """Returns a new cell object with the configuration loaded from
        a json file.

        Parameters
        ----------
        f_path: str | Path
            Path to the json file containing the cell configuration.

        Returns
        -------
        Cell2D
            A new cell object with the configuration loaded from the json file.
        """
        f_path = Validator.file_path(
            f_path, must_exist=True, extensions=[".json"]
        )
        with open(f_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def remove_inclusion_overlaps(self, ssd_ratio, proj_buffer_ratio) -> None:
        # Run Optim Loop to ensure there are no overlaps among inclusions
        #   Evaluate the cost function and gradients
        #     Add Periodic copies, if required
        #     cost evaluation
        #     gradients evaluation
        #   Update the inclusions positions
        #   Check for convergence
        #   If converged, return the optimised inclusions positions
        self._opt_problem = CellShapes2DOverlap(
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
        for k, shapes_list in self._shapes.values():
            for idx, a_shape in enumerate(shapes_list):
                a_shape.centre = gb.Point2D(
                    positions[idx, 0], positions[idx, 1]
                )

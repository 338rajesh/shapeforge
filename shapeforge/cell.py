import json
import pickle
from abc import ABC
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import gbox as gb
import numpy as np
import yaml
from scipy import stats

from .config import ShapeConfig
from .utils import (
    PI,
    DistributionSampler,
    DistributionSpec,
    Validator,
    get_logger,
)

logger = get_logger(__name__)


def _shape_to_dict(shape: gb.Shape2D) -> dict:
    if type(shape) is gb.Circle:
        d = {
            "shape": "circle",
            "radius": float(shape.radius),
            "centre": [float(c) for c in shape.centre],
        }
    elif isinstance(shape, gb.Ellipse):
        d = {
            "shape": "ellipse",
            "semi_major_length": float(shape.semi_major_length),
            "semi_minor_length": float(shape.semi_minor_length),
            "centre": [float(c) for c in shape.centre],
            "major_axis_angle": [float(shape.position.orientation.radians), "rad"],
        }
    else:
        d_inner = shape.to_dict()
        d = {"shape": shape.__class__.__name__.lower(), **d_inner}
    if getattr(shape, "is_periodic_copy", False):
        d["is_periodic_copy"] = True
    return d


def _shape_from_dict(d: dict) -> gb.Shape2D:
    shape_name = d.get("shape", "").lower()
    is_periodic = d.get("is_periodic_copy", False)
    if shape_name == "circle" or ("radius" in d and "semi_major_length" not in d):
        s = gb.Circle(
            radius=float(d["radius"]),
            centre=[float(c) for c in d.get("centre", (0.0, 0.0))],
        )
    elif shape_name == "ellipse":
        major_axis_angle = d.get("major_axis_angle", 0.0)
        if isinstance(major_axis_angle, (list, tuple)):
            major_axis_angle = gb.Angle(
                float(major_axis_angle[0]), str(major_axis_angle[1])
            )
        elif isinstance(major_axis_angle, (int, float)):
            major_axis_angle = gb.Angle.rad(float(major_axis_angle))
        s = gb.Ellipse(
            semi_major_length=float(d["semi_major_length"]),
            semi_minor_length=float(d["semi_minor_length"]),
            centre=[float(c) for c in d.get("centre", (0.0, 0.0))],
            major_axis_angle=major_axis_angle,
        )
    else:
        cls = gb.shapes.SHAPES_2D_MAPPING.get(shape_name)
        if cls is None:
            raise ValueError(f"Unknown shape type '{shape_name}'")
        d_copy = {k: v for k, v in d.items() if k not in ("shape", "is_periodic_copy")}
        s = cls.from_dict(d_copy)
    if is_periodic:
        s.is_periodic_copy = True
    return s


class CellDomain(ABC):
    __slots__ = ("_bounds",)

    def __init__(
        self, bounds: Mapping[str, float] | Sequence[float] | gb.Bounds
    ):
        if isinstance(bounds, gb.Bounds):
            pass
        elif isinstance(bounds, Mapping):
            bounds = gb.Bounds.from_mapping(bounds)
        elif isinstance(bounds, Sequence):
            bounds = gb.Bounds.from_sequence(bounds)
        else:
            raise TypeError(
                f"bounds must be a gb.Bounds, or a Mapping or a Sequence, "
                f"not {type(bounds).__name__}"
            )
        self._bounds = bounds

    @property
    def bounds(self) -> gb.Bounds:
        return gb.Bounds.from_mapping(self._bounds.bounds)


class CellDomain2D(CellDomain):
    """
    It represents a 2D domain for a cell, without any inclusions. It can be
    considered as a host for inclusions or other geometrical features, but it
    does not contain any by itself.
    """

    def __init__(
        self,
        bounds: gb.Bounds2DRectangular | Mapping[str, float] | Sequence[float],
    ):
        if isinstance(bounds, Mapping):
            bounds = gb.Bounds2DRectangular.from_mapping(bounds)
        elif isinstance(bounds, Sequence) and not isinstance(bounds, gb.Bounds):
            bounds = gb.Bounds2DRectangular.from_sequence(bounds)
        super().__init__(bounds)

    @property
    def bounds(self) -> gb.Bounds2DRectangular:
        return gb.Bounds2DRectangular.from_mapping(self._bounds.bounds)

    @property
    def x_bounds(self) -> tuple[float, float]:
        """
        Get the x bounds of the cell domain.
        """
        return self._bounds.x_min, self._bounds.x_max

    @property
    def y_bounds(self) -> tuple[float, float]:
        """
        Get the y bounds of the cell domain.
        """
        return self._bounds.y_min, self._bounds.y_max

    @property
    def area(self) -> float:
        """
        Calculate the volume of the cell domain.
        """
        return self._bounds.area

    @property
    def domain(self) -> gb.Bounds2DRectangular:
        """Returns the ``gb.Bounds2DRectangular`` object"""
        return self._bounds

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(d, keys=["bounds"], types=[dict])
        return cls(gb.Bounds2DRectangular.from_mapping(d["bounds"]))

    def to_dict(self) -> dict:
        return {"bounds": dict(self._bounds.bounds)}


def circle_intersects_rect(
    cx: float,
    cy: float,
    r: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    """Return True if a 2D circle intersects or touches a rectangle [xmin, xmax] x [ymin, ymax]."""
    dx = max(xmin - cx, 0.0, cx - xmax)
    dy = max(ymin - cy, 0.0, cy - ymax)
    return (dx * dx + dy * dy) < (r * r)


def get_shape_circles(shape: gb.Shape2D) -> list[tuple[float, float, float]]:
    """Return list of (cx, cy, r) circles representing the shape at its actual pose (x, y, theta)."""
    if type(shape) is gb.Circle:
        return [
            (float(shape.position.x), float(shape.position.y), float(shape.radius))
        ]

    pos = shape.position
    x = float(pos.x)
    y = float(pos.y)
    th = float(pos.orientation.radians)
    c, s = np.cos(th), np.sin(th)

    original_pose = shape.position
    shape.position = gb.shapes_2d.Shape2DPose(0.0, 0.0, 0.0)
    body_circles = [
        (float(circ.centre[0]), float(circ.centre[1]), float(circ.radius))
        for circ in shape.union_of_circles()
    ]
    shape.position = original_pose

    return [
        (x + c * px - s * py, y + s * px + c * py, r)
        for (px, py, r) in body_circles
    ]


def shape_intersects_bounds(
    shape: gb.Shape2D, bounds: gb.Bounds2DRectangular
) -> bool:
    """Check if a 2D shape intersects the rectangular domain bounds."""
    if not shape.bounding_box.overlaps(bounds):
        return False
    b = bounds.bounds
    xmin, ymin, xmax, ymax = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
    for cx, cy, r in get_shape_circles(shape):
        if circle_intersects_rect(cx, cy, r, xmin, ymin, xmax, ymax):
            return True
    return False


def generate_periodic_copies(
    inclusions: "Inclusions2D",
    domain: CellDomain2D,
) -> "Inclusions2D":
    """Generate periodic copies of inclusions that cross domain boundaries.

    For each inclusion crossing any of the domain's boundaries (edges or corners),
    translated copies shifted by +/- Lx and/or +/- Ly are created and placed on
    the opposite boundary to ensure spatial continuity across the unit cell.
    """
    x_min, x_max = domain.x_bounds
    y_min, y_max = domain.y_bounds
    Lx = x_max - x_min
    Ly = y_max - y_min

    periodic_copies: list[gb.Shape2D] = []

    for shape in inclusions:
        if getattr(shape, "is_periodic_copy", False):
            continue

        bb = shape.bounding_box.bounds
        dx_shifts: list[float] = []
        if bb["x_min"] < x_min:
            dx_shifts.append(Lx)
        if bb["x_max"] > x_max:
            dx_shifts.append(-Lx)

        dy_shifts: list[float] = []
        if bb["y_min"] < y_min:
            dy_shifts.append(Ly)
        if bb["y_max"] > y_max:
            dy_shifts.append(-Ly)

        candidate_shifts: list[tuple[float, float]] = []
        for dx in dx_shifts:
            candidate_shifts.append((dx, 0.0))
        for dy in dy_shifts:
            candidate_shifts.append((0.0, dy))
        for dx in dx_shifts:
            for dy in dy_shifts:
                candidate_shifts.append((dx, dy))

        ori_rad = gb.Angle.rad(float(shape.position.orientation.radians))
        for dx, dy in candidate_shifts:
            shifted = shape.clone()
            shifted.position = gb.shapes_2d.Shape2DPose(
                float(shape.position.x + dx),
                float(shape.position.y + dy),
                ori_rad,
            )
            shifted.is_periodic_copy = True
            if shape_intersects_bounds(shifted, domain.domain):
                periodic_copies.append(shifted)

    return Inclusions2D(periodic_copies)


class Inclusions(Collection[gb.Shape2D]):
    def get_positions(self, flat: bool = False) -> np.ndarray:
        pass

    def set_positions(self, x_new: np.ndarray):
        """Sets the positions of the inclusions.

        ***All the angles must be in radians.***
        """
        raise NotImplementedError("Subclasses must implement this method")


class Inclusions2D(Inclusions):
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
        """Total area occupied by the inclusions (excluding periodic copies if primary exist)."""
        primary = self.primary_inclusions
        if primary:
            return sum(inclusion.area for inclusion in primary)
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
        """Return the number of inclusions of each shape type (excluding periodic copies if primary exist)."""
        counts: dict[type[gb.Shape2D], int] = defaultdict(int)

        items = self.primary_inclusions or self._inclusions
        for inclusion in items:
            counts[type(inclusion)] += 1

        return dict(counts)

    @property
    def primary_inclusions(self) -> list[gb.Shape2D]:
        """Return only primary inclusions (excluding periodic copies)."""
        return [
            inclusion
            for inclusion in self._inclusions
            if not getattr(inclusion, "is_periodic_copy", False)
        ]

    @property
    def periodic_copies(self) -> list[gb.Shape2D]:
        """Return only periodic copy inclusions."""
        return [
            inclusion
            for inclusion in self._inclusions
            if getattr(inclusion, "is_periodic_copy", False)
        ]

    def make_periodic(self, domain: CellDomain2D) -> Self:
        """Return a new Inclusions2D with wrapped primary inclusions and their periodic copies."""
        x_min, x_max = domain.x_bounds
        y_min, y_max = domain.y_bounds
        Lx = x_max - x_min
        Ly = y_max - y_min

        primary = []
        for shape in self.primary_inclusions:
            pos = shape.position
            wrapped_x = x_min + ((pos.x - x_min) % Lx)
            wrapped_y = y_min + ((pos.y - y_min) % Ly)
            if not np.isclose(wrapped_x, pos.x) or not np.isclose(wrapped_y, pos.y):
                s_wrapped = shape.clone()
                s_wrapped.position = gb.shapes_2d.Shape2DPose(
                    float(wrapped_x),
                    float(wrapped_y),
                    gb.Angle.rad(float(pos.orientation.radians)),
                )
                primary.append(s_wrapped)
            else:
                primary.append(shape.clone())

        copies = generate_periodic_copies(Inclusions2D(primary), domain)
        return self.__class__(primary + list(copies))

    def clear(self) -> None:
        """Remove all inclusions from the collection."""
        self._inclusions.clear()

    def clone(self) -> Self:
        """Create a deep copy of this instance."""
        cloned = []
        for inclusion in self._inclusions:
            c = inclusion.clone()
            if getattr(inclusion, "is_periodic_copy", False):
                c.is_periodic_copy = True
            cloned.append(c)
        return self.__class__(cloned)

    def to_dict(self) -> dict:
        return {
            "inclusions": [
                _shape_to_dict(inclusion) for inclusion in self._inclusions
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(
            d, key_type_map={"inclusions": Sequence}, reject_extra_keys=True
        )
        inclusions = [
            _shape_from_dict(inclusion) for inclusion in d["inclusions"]
        ]
        return cls(inclusions)

    def get_positions(self, flat: bool = False) -> np.ndarray:
        """Returns a numpy array of shape (N, 3) where N is the number of
        inclusions and each row contains x, y and orientation (in radians)
        of the shape in the 2D plane. If ``flat`` is True, an array of shape
        ``(3N, )`` will be returned where the first N elements are the x,
        next N elements are the y and the last N elements are the
        orientations.
        """
        positions = np.array(
            [inclusion.position.to_tuple() for inclusion in self._inclusions]
        )
        return positions.flatten(order="F") if flat else positions

    def set_positions(self, xyo: np.ndarray) -> None:
        """Sets the positions of the inclusions. The input array should
        have shape (N, 3) where N is the number of inclusions and each row
        contains x, y and orientation (in radians)
        """
        num_inclusions = len(self)
        if xyo.shape != (num_inclusions, 3):
            raise ValueError(
                f"Expected an array of shape ({num_inclusions, 3}), "
                f"Got {xyo.shape}"
            )
        for i in range(num_inclusions):
            self._inclusions[i].position = gb.shapes_2d.Shape2DPose.from_dict(
                {
                    "x": float(xyo[i, 0]),
                    "y": float(xyo[i, 1]),
                    "orientation": gb.Angle.rad(float(xyo[i, 2])),
                }
            )

    def get_equivalent_circle_radii(self) -> np.ndarray:
        return np.array([i.equivalent_circle_radius for i in self])

    def get_bbox_overlap_pairs(self) -> set[tuple[int, int]]:
        """Returns a set of 2-tuples (i, j) where i > j and the bounding
        boxes of inclusions i and j overlap.
        """
        n = len(self)
        # overlap_matrix = np.zeros((n, n), dtype=int)
        overlappping_pairs = set()
        for i in range(n):
            for j in range(n):
                if i <= j:
                    continue
                if self._inclusions[i].bounding_box.overlaps(
                    self._inclusions[j].bounding_box
                ):
                    overlappping_pairs.add((i, j))
        return overlappping_pairs


@dataclass
class PositionSampler2D:
    xc_sampler: DistributionSampler | None = None
    yc_sampler: DistributionSampler | None = None
    orientation_sampler: DistributionSampler | float | None = None
    qmc_sampler: Any | None = None
    bounds: tuple[float, float, float, float] | None = None

    def sample(self) -> dict[str, float | gb.Angle]:
        out: dict[str, float | gb.Angle] = {}
        if self.qmc_sampler is not None and self.bounds is not None:
            pt = self.qmc_sampler.random(1)[0]
            x_min, x_max, y_min, y_max = self.bounds
            out["xc"] = float(x_min + pt[0] * (x_max - x_min))
            out["yc"] = float(y_min + pt[1] * (y_max - y_min))
        else:
            if isinstance(self.xc_sampler, DistributionSampler):
                out["xc"] = float(self.xc_sampler.sample())

            if isinstance(self.yc_sampler, DistributionSampler):
                out["yc"] = float(self.yc_sampler.sample())

        if isinstance(self.orientation_sampler, DistributionSampler):
            out["major_axis_angle"] = gb.Angle.rad(
                float(self.orientation_sampler.sample())
            )
        elif isinstance(self.orientation_sampler, (int, float)):
            out["major_axis_angle"] = gb.Angle.rad(
                float(self.orientation_sampler)
            )

        return out


class Inclusion2DSampler:
    __slots__ = ("_is_circular", "_pos_sampler", "_shape", "_size_samplers")

    def __init__(
        self,
        shape: str | type[gb.Shape2D],
        pos_sampler: PositionSampler2D,
        size_samplers: dict[str, float | DistributionSampler],
    ):
        if isinstance(shape, str):
            shape_str = Validator.as_string(shape).lower()
            gb_shape = gb.shapes.SHAPES_2D_MAPPING[shape_str]
        elif isinstance(shape, type) and issubclass(shape, gb.Shape2D):
            gb_shape = shape
        else:
            raise TypeError(
                f"Unsupported shape type: {type(shape).__name__}. "
                "Expected str or gb.Shape2D subclass."
            )

        # check if g_shape has a method called from_params
        if not hasattr(gb_shape, "from_params") or not callable(
            gb_shape.from_params
        ):
            raise ValueError(
                f"Unsupported shape: {shape}, must have a from_params method"
            )

        self._shape: type[gb.Shape2D] = gb_shape
        self._pos_sampler = pos_sampler
        self._size_samplers = Validator.as_dict(size_samplers)
        self._is_circular = issubclass(gb_shape, gb.Circle)

    def sample(self) -> gb.Shape2D:
        pos_params = self._pos_sampler.sample()
        if self._is_circular:
            # Circle only expects xc and yc, rejects major_axis_angle
            pos_params = {
                "xc": float(pos_params["xc"]),
                "yc": float(pos_params["yc"]),
            }
        else:
            # Non-circular shapes require orientation (major_axis_angle)
            if "major_axis_angle" not in pos_params:
                pos_params["major_axis_angle"] = gb.Angle.rad(
                    float(np.random.uniform(0.0, 2.0 * PI))
                )

        size_params = {}
        for k, v in self._size_samplers.items():
            if isinstance(v, (int, float)):
                size_params[k] = float(v)
            else:
                size_params[k] = float(v.sample())

        return self._shape.from_params(
            position_params=pos_params, size_params=size_params
        )


def _normalize_shape_size_params(shape_name: str, params: dict) -> dict:
    """Normalize shape size parameter keys and aliases."""
    normalized: dict = {}
    if shape_name == "circle":
        for k, v in params.items():
            if k in ("radius", "r"):
                normalized["radius"] = v
            else:
                normalized[k] = v
    elif shape_name == "ellipse":
        for k, v in params.items():
            if k in (
                "semi_major_length",
                "semi_major_axis",
                "major_axis",
                "major_radius",
                "a",
            ):
                normalized["semi_major_length"] = v
            elif k in (
                "semi_minor_length",
                "semi_minor_axis",
                "minor_axis",
                "minor_radius",
                "b",
            ):
                normalized["semi_minor_length"] = v
            else:
                normalized[k] = v

        # Compute missing axis from aspect_ratio if provided
        ar = normalized.pop("aspect_ratio", None) or normalized.pop("ar", None)
        if ar is not None and isinstance(ar, (int, float)):
            if (
                "semi_major_length" in normalized
                and "semi_minor_length" not in normalized
            ):
                val = normalized["semi_major_length"]
                if isinstance(val, (int, float)):
                    normalized["semi_minor_length"] = float(val) / float(ar)
            elif (
                "semi_minor_length" in normalized
                and "semi_major_length" not in normalized
            ):
                val = normalized["semi_minor_length"]
                if isinstance(val, (int, float)):
                    normalized["semi_major_length"] = float(val) * float(ar)
    else:
        normalized = dict(params)
    return normalized


def initialise_shapes_2d(
    cell_domain: CellDomain2D,
    shapes_config: Sequence[ShapeConfig],
    rng: np.random.Generator,
    init_method: str = "uniform",
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
    init_method: str, optional
        Sampling method for spatial positions: "uniform", "lhs", or "sobol"
        (default: "uniform").

    Returns
    -------
    Inclusions2D
       A Inclusions2D object containing the initialized inclusions.
    """
    x_min, x_max = cell_domain.x_bounds
    y_min, y_max = cell_domain.y_bounds

    qmc_sampler = None
    method_lower = init_method.lower() if isinstance(init_method, str) else "uniform"
    if method_lower in ("lhs", "latin_hypercube"):
        qmc_sampler = stats.qmc.LatinHypercube(d=2, seed=rng)
    elif method_lower in ("sobol", "sobol_sequence"):
        qmc_sampler = stats.qmc.Sobol(d=2, seed=rng)

    inclusions = Inclusions2D()
    cum_vf = 0.0
    for a_shape_cfg in shapes_config:
        cum_vf += a_shape_cfg.volume_fraction
        if cum_vf > 1.0:
            raise ValueError(
                "Cumulative volume fraction of given shapes exceeds 1.0."
            )

        shape_name = Validator.as_string(a_shape_cfg.name).lower()
        gb_shape_cls = gb.shapes.SHAPES_2D_MAPPING[shape_name]
        is_circular = issubclass(gb_shape_cls, gb.Circle)

        # Separate orientation parameters from size parameters
        raw_params = dict(a_shape_cfg.params)
        orientation_spec = None
        for ori_key in ("major_axis_angle", "orientation", "angle", "theta"):
            if ori_key in raw_params:
                orientation_spec = raw_params.pop(ori_key)
                break

        # Setup orientation sampler
        orientation_sampler = None
        if not is_circular:
            if orientation_spec is not None:
                if isinstance(orientation_spec, (int, float)):
                    orientation_sampler = float(orientation_spec)
                elif isinstance(orientation_spec, DistributionSpec):
                    orientation_sampler = DistributionSampler(
                        orientation_spec, rng
                    )
                elif isinstance(orientation_spec, str):
                    orientation_sampler = DistributionSampler.from_signature(
                        orientation_spec, rng
                    )
            else:
                # Default to uniform orientation in [0, 2*pi)
                orientation_sampler = DistributionSampler.from_signature(
                    f"uniform(0.0, {2.0 * PI})", rng=rng
                )

        pos_sampler = PositionSampler2D(
            xc_sampler=DistributionSampler.from_signature(
                f"uniform({x_min}, {x_max - x_min})", rng=rng
            ),
            yc_sampler=DistributionSampler.from_signature(
                f"uniform({y_min}, {y_max - y_min})", rng=rng
            ),
            orientation_sampler=orientation_sampler,
            qmc_sampler=qmc_sampler,
            bounds=(x_min, x_max, y_min, y_max),
        )

        size_params_specs = _normalize_shape_size_params(
            shape_name, raw_params
        )

        size_samplers = {}
        for p_name, p_spec in size_params_specs.items():
            if isinstance(p_spec, (int, float)):
                size_samplers[p_name] = float(p_spec)
            elif isinstance(p_spec, DistributionSpec):
                size_samplers[p_name] = DistributionSampler(p_spec, rng)
            elif isinstance(p_spec, str):
                size_samplers[p_name] = DistributionSampler.from_signature(
                    p_spec, rng
                )
            else:
                raise TypeError(
                    f"Invalid distribution spec type: {type(p_spec).__name__}"
                )

        incl_sampler = Inclusion2DSampler(
            shape=shape_name,
            pos_sampler=pos_sampler,
            size_samplers=size_samplers,
        )

        required_area = cell_domain.area * a_shape_cfg.volume_fraction
        cumulative_area = 0.0

        while cumulative_area < required_area:
            a_inclusion = incl_sampler.sample()
            inclusions.add(a_inclusion)
            cumulative_area += a_inclusion.area

    return inclusions


class Cell:
    __slots__ = ("_domain", "_opt_problem", "_shapes")

    def __init__(self, domain: CellDomain, shapes: Inclusions):
        self._domain: CellDomain = domain
        self._shapes: Inclusions = shapes
        self._opt_problem = None

    @property
    def domain(self) -> CellDomain2D:
        return self._domain

    @property
    def shapes(self) -> Inclusions2D:
        return self._shapes

    @property
    def primary_shapes(self) -> Inclusions2D:
        """Return an Inclusions2D collection of only the primary inclusions."""
        return Inclusions2D(self._shapes.primary_inclusions)

    @property
    def periodic_shapes(self) -> Inclusions2D:
        """Return an Inclusions2D collection of only the periodic copy inclusions."""
        return Inclusions2D(self._shapes.periodic_copies)

    def make_periodic(self) -> Self:
        """Return a new Cell with periodic copies added for boundary-crossing shapes."""
        new_shapes = self._shapes.make_periodic(self._domain)
        return self.__class__(self._domain, new_shapes)

    def clone(self) -> Self:
        return self.__class__(self._domain, self._shapes.clone())

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(
            d,
            key_type_map={"domain": dict, "shapes": dict},
            reject_extra_keys=True,
            name=f"{cls.__name__}.from_dict.d",
        )
        return cls(
            CellDomain2D.from_dict(d["domain"]),
            Inclusions2D.from_dict(d["shapes"]),
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
            Path to save the cell file (.json, .yaml, .yml, .pkl, .pickle, .npz).
        overwrite: bool, optional
            Whether to overwrite an existing file (default: False).

        Returns
        -------
        Path
            Path to the saved file.
        """
        f_path = Validator.file_path(
            f_path,
            must_exist=None if overwrite else False,
            extensions=[".json", ".yaml", ".yml", ".pkl", ".pickle", ".npz"],
        )
        ext = f_path.suffix.lower()
        if ext == ".json":
            with open(f_path, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        elif ext in (".yaml", ".yml"):
            with open(f_path, "w") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        elif ext in (".pkl", ".pickle"):
            with open(f_path, "wb") as f:
                pickle.dump(self.to_dict(), f)
        elif ext == ".npz":
            positions = self.shapes.get_positions()
            bounds_dict = self.domain.to_dict()["bounds"]
            bounds_arr = np.array([
                bounds_dict["x_min"],
                bounds_dict["y_min"],
                bounds_dict["x_max"],
                bounds_dict["y_max"],
            ])
            np.savez(
                f_path,
                positions=positions,
                bounds=bounds_arr,
                cell_json=json.dumps(self.to_dict()),
            )
        return f_path

    @classmethod
    def load(cls, f_path: str | Path) -> Self:
        """Returns a new cell object with the configuration loaded from file.

        Parameters
        ----------
        f_path: str | Path
            Path to the file containing the cell configuration.

        Returns
        -------
        Cell
            A new cell object with the configuration loaded from the file.
        """
        f_path = Validator.file_path(
            f_path,
            must_exist=True,
            extensions=[".json", ".yaml", ".yml", ".pkl", ".pickle", ".npz"],
        )
        ext = f_path.suffix.lower()
        if ext == ".json":
            with open(f_path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        elif ext in (".yaml", ".yml"):
            with open(f_path, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        elif ext in (".pkl", ".pickle"):
            with open(f_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, cls):
                return data
            return cls.from_dict(data)
        elif ext == ".npz":
            npz_data = np.load(f_path)
            if "cell_json" in npz_data:
                data = json.loads(str(npz_data["cell_json"]))
                return cls.from_dict(data)
            raise ValueError(
                f"NPZ file {f_path} does not contain valid Cell data ('cell_json')"
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def plot(self, f_path: str | Path | None = None, **kwargs) -> np.ndarray:
        """Plots the cell, optionally saves the plot to specified file, and returns the rendered image array."""
        sp = gb.ShapesPlotter(**kwargs)
        return sp.plot(
            shapes=[i for i in self.shapes],
            bounds=self.domain.domain,
            path=f_path,
        )

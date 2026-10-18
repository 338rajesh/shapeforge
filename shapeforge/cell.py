"""
Domain, inclusion collections, and unit cell representations.
=============================================================

Provides data structures representing spatial bounding domains (:class:`CellDomain2D`),
ordered/unordered inclusion collections (:class:`Inclusions2D`), spatial and parameter
samplers (:class:`Inclusion2DSampler`, :class:`PositionSampler2D`), periodic boundary
wrapping routines, and the primary unit cell container (:class:`Cell`).
"""

from __future__ import annotations

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
from .plotter import CellPlotter
from .registry import ShapeRegistry
from .utils import (
    PI,
    DistributionSampler,
    DistributionSpec,
    Validator,
    get_logger,
)

logger = get_logger(__name__)


def _shape_to_dict(shape: gb.Shape2D) -> dict[str, Any]:
    """
    Serialize a 2D geometric shape instance into a dictionary.

    Handles native Circle and Ellipse representations as well as arbitrary
    registered shapes defining ``to_dict()``.

    Parameters
    ----------
    shape : gb.Shape2D
        Shape instance to serialize.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the shape.
    """
    if type(shape) is gb.Circle:
        d: dict[str, Any] = {
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


def _shape_from_dict(d: dict[str, Any]) -> gb.Shape2D:
    """
    Reconstruct a 2D geometric shape instance from a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary containing shape attributes.

    Returns
    -------
    gb.Shape2D
        Instantiated Shape2D subclass instance.

    Raises
    ------
    ValueError
        If the shape type is unrecognized or not registered.
    """
    shape_name = d.get("shape", "").lower()
    is_periodic = d.get("is_periodic_copy", False)
    if shape_name == "circle" or (
        not shape_name and "radius" in d and "semi_major_length" not in d
    ):
        s: gb.Shape2D = gb.Circle(
            radius=float(d["radius"]),
            centre=[float(c) for c in d.get("centre", (0.0, 0.0))],
        )
    elif shape_name == "ellipse" or (
        not shape_name and "semi_major_length" in d
    ):
        major_axis_angle = d.get("major_axis_angle", 0.0)
        if isinstance(major_axis_angle, (list, tuple)):
            ang = gb.Angle(
                float(major_axis_angle[0]), str(major_axis_angle[1])
            )
        elif isinstance(major_axis_angle, (int, float)):
            ang = gb.Angle.rad(float(major_axis_angle))
        else:
            ang = gb.Angle.rad(0.0)
        s = gb.Ellipse(
            semi_major_length=float(d["semi_major_length"]),
            semi_minor_length=float(d["semi_minor_length"]),
            centre=[float(c) for c in d.get("centre", (0.0, 0.0))],
            major_axis_angle=ang,
        )
    else:
        if ShapeRegistry.has(shape_name):
            cls = ShapeRegistry.get(shape_name)
        else:
            raise ValueError(f"Unknown shape type '{shape_name}'")
        d_copy = {k: v for k, v in d.items() if k not in ("shape", "is_periodic_copy")}
        s = cls.from_dict(d_copy)
    if is_periodic:
        s.is_periodic_copy = True
    return s


class CellDomain(ABC):
    """
    Abstract base class for spatial cell domains.

    Parameters
    ----------
    bounds : Mapping[str, float] or Sequence[float] or gb.Bounds
        Bounding coordinates of the domain.

    Raises
    ------
    TypeError
        If bounds is not a valid Bounds, Mapping, or Sequence.
    """

    __slots__ = ("_bounds",)

    def __init__(
        self, bounds: Mapping[str, float] | Sequence[float] | gb.Bounds
    ) -> None:
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
        self._bounds: gb.Bounds = bounds

    @property
    def bounds(self) -> gb.Bounds:
        """
        Copy of the bounding box object.

        Returns
        -------
        gb.Bounds
            Bounding box instance.
        """
        return gb.Bounds.from_mapping(self._bounds.bounds)


class CellDomain2D(CellDomain):
    """
    2D rectangular spatial domain bounding a unit cell or RVE.

    Acts as the spatial container for inclusions and defines the geometry
    over which boundary conditions and volume fractions are evaluated.

    Parameters
    ----------
    bounds : gb.Bounds2DRectangular or Mapping[str, float] or Sequence[float]
        Bounding coordinates specified as ``[x_min, y_min, x_max, y_max]`` or
        a mapping with keys ``'x_min'``, ``'y_min'``, ``'x_max'``, ``'y_max'``.

    Examples
    --------
    >>> from shapeforge.cell import CellDomain2D
    >>> domain = CellDomain2D([0.0, 0.0, 10.0, 10.0])
    >>> domain.area
    100.0
    >>> domain.x_bounds
    (0.0, 10.0)
    """

    def __init__(
        self,
        bounds: gb.Bounds2DRectangular | Mapping[str, float] | Sequence[float],
    ) -> None:
        if isinstance(bounds, Mapping):
            bounds = gb.Bounds2DRectangular.from_mapping(bounds)
        elif isinstance(bounds, Sequence) and not isinstance(bounds, gb.Bounds):
            bounds = gb.Bounds2DRectangular.from_sequence(bounds)
        super().__init__(bounds)

    @property
    def bounds(self) -> gb.Bounds2DRectangular:
        """
        Copy of the 2D rectangular bounds object.

        Returns
        -------
        gb.Bounds2DRectangular
            Rectangular bounds instance.
        """
        return gb.Bounds2DRectangular.from_mapping(self._bounds.bounds)

    @property
    def x_bounds(self) -> tuple[float, float]:
        """
        Tuple of minimum and maximum x-coordinates $(x_{\\min}, x_{\\max})$.

        Returns
        -------
        tuple of float
            $(x_{\\min}, x_{\\max})$.
        """
        return self._bounds.x_min, self._bounds.x_max

    @property
    def y_bounds(self) -> tuple[float, float]:
        """
        Tuple of minimum and maximum y-coordinates $(y_{\\min}, y_{\\max})$.

        Returns
        -------
        tuple of float
            $(y_{\\min}, y_{\\max})$.
        """
        return self._bounds.y_min, self._bounds.y_max

    @property
    def area(self) -> float:
        """
        Total surface area of the domain: $(x_{\\max} - x_{\\min}) \\times (y_{\\max} - y_{\\min})$.

        Returns
        -------
        float
            Surface area.
        """
        return self._bounds.area

    @property
    def domain(self) -> gb.Bounds2DRectangular:
        """
        Internal ``gb.Bounds2DRectangular`` reference.

        Returns
        -------
        gb.Bounds2DRectangular
            Direct bounds reference.
        """
        return self._bounds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Construct a CellDomain2D from a dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary containing a ``'bounds'`` sub-dictionary.

        Returns
        -------
        CellDomain2D
            Constructed domain instance.
        """
        d = Validator.as_dict(d, keys=["bounds"], types=[dict])
        return cls(gb.Bounds2DRectangular.from_mapping(d["bounds"]))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize domain bounds to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing ``{'bounds': {'x_min': ..., ...}}``.
        """
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
    """
    Check whether a 2D circle intersects or touches a rectangle.

    Parameters
    ----------
    cx : float
        Circle center x-coordinate.
    cy : float
        Circle center y-coordinate.
    r : float
        Circle radius.
    xmin : float
        Rectangle minimum x-coordinate.
    ymin : float
        Rectangle minimum y-coordinate.
    xmax : float
        Rectangle maximum x-coordinate.
    ymax : float
        Rectangle maximum y-coordinate.

    Returns
    -------
    bool
        True if the circle intersects or overlaps the rectangle, False otherwise.
    """
    dx = max(xmin - cx, 0.0, cx - xmax)
    dy = max(ymin - cy, 0.0, cy - ymax)
    return (dx * dx + dy * dy) < (r * r)


def get_shape_circles(shape: gb.Shape2D) -> list[tuple[float, float, float]]:
    """
    Extract world-frame disk representations $(c_x, c_y, r)$ for a given shape.

    Transforms the body-frame circles returned by ``shape.union_of_circles()``
    using the shape's current position pose $(x, y, \\theta)$.

    Parameters
    ----------
    shape : gb.Shape2D
        Shape instance to convert.

    Returns
    -------
    list of tuple
        List of 3-tuples ``(world_cx, world_cy, radius)``.
    """
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
    """
    Check if a 2D shape physically intersects a rectangular domain boundary.

    Performs a broad-phase bounding box rejection test followed by narrow-phase
    circle-rectangle intersection tests for all constituent disks of the shape.

    Parameters
    ----------
    shape : gb.Shape2D
        Geometric shape to check.
    bounds : gb.Bounds2DRectangular
        Domain rectangular boundaries.

    Returns
    -------
    bool
        True if the shape intersects the domain rectangle, False otherwise.
    """
    if not shape.bounding_box.overlaps(bounds):
        return False
    b = bounds.bounds
    xmin, ymin, xmax, ymax = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
    for cx, cy, r in get_shape_circles(shape):
        if circle_intersects_rect(cx, cy, r, xmin, ymin, xmax, ymax):
            return True
    return False


def generate_periodic_copies(
    inclusions: Inclusions2D,
    domain: CellDomain2D,
) -> Inclusions2D:
    """
    Generate periodic ghost copies of inclusions that cross domain boundaries.

    For each inclusion crossing any domain edge or corner, translated copies
    shifted by $\\pm L_x$ and/or $\\pm L_y$ are created and placed on the opposite
    boundary to enforce spatial continuity in periodic representative volume elements.

    Parameters
    ----------
    inclusions : Inclusions2D
        Collection of primary inclusions.
    domain : CellDomain2D
        The periodic domain defining dimensions $L_x$ and $L_y$.

    Returns
    -------
    Inclusions2D
        Collection containing only the newly generated periodic ghost copies.
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
    """
    Abstract base collection representing a set of geometric inclusions.
    """

    def get_positions(self, flat: bool = False) -> np.ndarray:
        """
        Get positions and orientations of all inclusions.

        Parameters
        ----------
        flat : bool, optional
            Whether to return a flattened 1D array (default: False).

        Returns
        -------
        np.ndarray
            Matrix of shape ``(N, 3)`` or flattened 1D array.
        """
        pass

    def set_positions(self, x_new: np.ndarray) -> None:
        """
        Set the positions and orientations of all inclusions in the collection.

        Parameters
        ----------
        x_new : np.ndarray
            Pose array of shape ``(N, 3)`` with angles in radians.

        Raises
        ------
        NotImplementedError
            If subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")


class Inclusions2D(Inclusions):
    """
    Collection container for 2D geometric inclusions (:class:`gbox.Shape2D`).

    Manages inclusion storage, area calculations, serialization, periodic ghost
    creation, and vectorized pose retrieval for the optimization engine.

    Parameters
    ----------
    inclusions : Iterable[gb.Shape2D], optional
        Initial inclusions to populate the collection with.

    Examples
    --------
    >>> import gbox as gb
    >>> from shapeforge.cell import Inclusions2D
    >>> inclusions = Inclusions2D([gb.Circle(1.0), gb.Circle(2.0)])
    >>> len(inclusions)
    2
    >>> round(inclusions.area, 2)
    15.71
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
        """
        Append a single inclusion to the collection.

        Parameters
        ----------
        inclusion : gb.Shape2D
            Shape instance to add.

        Raises
        ------
        TypeError
            If ``inclusion`` is not an instance of :class:`gbox.Shape2D`.
        """
        Validator.is_type(inclusion, gb.Shape2D, name="inclusion")
        self._inclusions.append(inclusion)

    def extend(self, inclusions: Iterable[gb.Shape2D]) -> None:
        """
        Append multiple inclusions to the collection.

        Parameters
        ----------
        inclusions : Iterable[gb.Shape2D]
            Sequence or iterable of shapes to add.
        """
        for inclusion in inclusions:
            self.add(inclusion)

    @property
    def area(self) -> float:
        """
        Total surface area occupied by primary inclusions.

        Excludes periodic ghost copies if primary inclusions are present.

        Returns
        -------
        float
            Total primary inclusion area.
        """
        primary = self.primary_inclusions
        if primary:
            return sum(inclusion.area for inclusion in primary)
        return sum(inclusion.area for inclusion in self._inclusions)

    def area_fraction(self, domain_area: float) -> float:
        """
        Compute the fraction of the domain area occupied by the inclusions.

        Parameters
        ----------
        domain_area : float
            Total area of the host domain. Must be $> 0$.

        Returns
        -------
        float
            Volume (area) fraction in $[0, 1]$.

        Raises
        ------
        ValueError
            If ``domain_area <= 0``.
        """
        if domain_area <= 0:
            raise ValueError("domain_area must be greater than zero.")

        return self.area / domain_area

    def get_by_shape(self, shape: type[gb.Shape2D]) -> list[gb.Shape2D]:
        """
        Retrieve all inclusions matching a specific shape class.

        Parameters
        ----------
        shape : type[gb.Shape2D]
            Class type to filter by (e.g. ``gb.Circle``).

        Returns
        -------
        list of gb.Shape2D
            Matching inclusion instances.
        """
        return [
            inclusion
            for inclusion in self._inclusions
            if isinstance(inclusion, shape)
        ]

    def get_count(self) -> dict[type[gb.Shape2D], int]:
        """
        Count occurrences of each inclusion shape type.

        Excludes periodic copies if primary inclusions exist.

        Returns
        -------
        dict[type[gb.Shape2D], int]
            Mapping of shape class types to integer counts.
        """
        counts: dict[type[gb.Shape2D], int] = defaultdict(int)

        items = self.primary_inclusions or self._inclusions
        for inclusion in items:
            counts[type(inclusion)] += 1

        return dict(counts)

    @property
    def primary_inclusions(self) -> list[gb.Shape2D]:
        """
        List of original, primary inclusions (excluding periodic ghost copies).

        Returns
        -------
        list of gb.Shape2D
            Primary inclusion instances.
        """
        return [
            inclusion
            for inclusion in self._inclusions
            if not getattr(inclusion, "is_periodic_copy", False)
        ]

    @property
    def periodic_copies(self) -> list[gb.Shape2D]:
        """
        List of periodic ghost copies generated across boundaries.

        Returns
        -------
        list of gb.Shape2D
            Periodic copy shape instances.
        """
        return [
            inclusion
            for inclusion in self._inclusions
            if getattr(inclusion, "is_periodic_copy", False)
        ]

    def make_periodic(self, domain: CellDomain2D) -> Self:
        """
        Return a new collection with boundary-crossing shapes wrapped and duplicated.

        Wraps out-of-bounds coordinates into $[x_{\\min}, x_{\\max}) \\times [y_{\\min}, y_{\\max})$
        and generates translated ghost copies on opposite edges.

        Parameters
        ----------
        domain : CellDomain2D
            The periodic spatial domain.

        Returns
        -------
        Inclusions2D
            New collection containing wrapped primaries and their periodic copies.
        """
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
        """
        Create a deep copy of the inclusions collection.

        Returns
        -------
        Inclusions2D
            Cloned collection instance.
        """
        cloned = []
        for inclusion in self._inclusions:
            c = inclusion.clone()
            if getattr(inclusion, "is_periodic_copy", False):
                c.is_periodic_copy = True
            cloned.append(c)
        return self.__class__(cloned)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize all inclusions into a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with key ``'inclusions'`` mapping to list of serialized shapes.
        """
        return {
            "inclusions": [
                _shape_to_dict(inclusion) for inclusion in self._inclusions
            ]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Deserialize an Inclusions2D collection from a dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary containing an ``'inclusions'`` list.

        Returns
        -------
        Inclusions2D
            Deserialized collection.
        """
        d = Validator.as_dict(
            d, key_type_map={"inclusions": Sequence}, reject_extra_keys=True
        )
        inclusions = [
            _shape_from_dict(inclusion) for inclusion in d["inclusions"]
        ]
        return cls(inclusions)

    def get_positions(self, flat: bool = False) -> np.ndarray:
        """
        Extract positions and orientations of all inclusions.

        Parameters
        ----------
        flat : bool, optional
            If False (default), returns an $(N, 3)$ matrix where each row is
            $[x, y, \\theta]$ with $\\theta$ in radians. If True, returns a 1D
            Fortran-ordered array of shape $(3N,)$ where the first $N$ elements
            are $x$, the next $N$ are $y$, and the last $N$ are $\\theta$.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 3)`` or ``(3 * N,)``.
        """
        positions = np.array(
            [inclusion.position.to_tuple() for inclusion in self._inclusions]
        )
        return positions.flatten(order="F") if flat else positions

    def set_positions(self, xyo: np.ndarray) -> None:
        """
        Update the poses of all inclusions in the collection.

        Parameters
        ----------
        xyo : np.ndarray
            Matrix of shape ``(N, 3)`` where rows contain $[x, y, \\theta]$
            with angles in radians.

        Raises
        ------
        ValueError
            If ``xyo`` does not match shape ``(len(self), 3)``.
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
        """
        Compute equivalent circle radii $r = \\sqrt{A / \\pi}$ for all inclusions.

        Returns
        -------
        np.ndarray
            1D array of equivalent circle radii of shape ``(len(self),)``.
        """
        return np.array([i.equivalent_circle_radius for i in self])

    def get_bbox_overlap_pairs(self) -> set[tuple[int, int]]:
        """
        Find all pairs of inclusions whose bounding boxes overlap.

        Returns
        -------
        set of tuple
            Set of index pairs ``(i, j)`` with $i > j$ indicating overlapping
            bounding boxes.
        """
        n = len(self)
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
    """
    Spatial location and orientation sampler for 2D inclusions.

    Supports independent continuous distribution sampling for $x, y$ coordinates
    as well as quasi-Monte Carlo (LHS, Sobol) point generation across domain bounds.

    Parameters
    ----------
    xc_sampler : DistributionSampler or None, optional
        Sampler for x-center coordinates.
    yc_sampler : DistributionSampler or None, optional
        Sampler for y-center coordinates.
    orientation_sampler : DistributionSampler or float or None, optional
        Sampler or constant angle for shape orientation.
    qmc_sampler : Any or None, optional
        Quasi-Monte Carlo point sequence sampler (e.g. ``scipy.stats.qmc.Sobol``).
    bounds : tuple of float or None, optional
        Spatial domain bounds ``(x_min, x_max, y_min, y_max)``.
    """

    xc_sampler: DistributionSampler | None = None
    yc_sampler: DistributionSampler | None = None
    orientation_sampler: DistributionSampler | float | None = None
    qmc_sampler: Any | None = None
    bounds: tuple[float, float, float, float] | None = None

    def sample(self) -> dict[str, float | gb.Angle]:
        """
        Draw a single sampled position and orientation dictionary.

        Returns
        -------
        dict[str, float | gb.Angle]
            Dictionary containing ``'xc'``, ``'yc'``, and optionally ``'major_axis_angle'``.
        """
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
    """
    Combined position, orientation, and size sampler for geometric shapes.

    Integrates spatial positioning (:class:`PositionSampler2D`) with parametric
    size distributions to instantiate fully realized :class:`gbox.Shape2D` objects.

    Parameters
    ----------
    shape : str or type[gb.Shape2D]
        Shape identifier string or class type registered in :class:`~shapeforge.registry.ShapeRegistry`.
    pos_sampler : PositionSampler2D
        Spatial position and orientation sampler.
    size_samplers : dict[str, float | DistributionSampler]
        Dictionary mapping geometric parameter names (e.g. ``'radius'``) to
        constant floats or :class:`~shapeforge.utils.DistributionSampler` instances.

    Raises
    ------
    TypeError
        If ``shape`` is neither a string nor a :class:`gbox.Shape2D` subclass.
    ValueError
        If the shape class lacks a callable ``from_params`` method.
    """

    __slots__ = ("_is_circular", "_pos_sampler", "_shape", "_size_samplers")

    def __init__(
        self,
        shape: str | type[gb.Shape2D],
        pos_sampler: PositionSampler2D,
        size_samplers: dict[str, float | DistributionSampler],
    ) -> None:
        if isinstance(shape, str):
            shape_str = Validator.as_string(shape).lower()
            gb_shape = ShapeRegistry.get(shape_str)
        elif isinstance(shape, type) and issubclass(shape, gb.Shape2D):
            gb_shape = shape
        else:
            raise TypeError(
                f"Unsupported shape type: {type(shape).__name__}. "
                "Expected str or gb.Shape2D subclass."
            )

        if not hasattr(gb_shape, "from_params") or not callable(
            gb_shape.from_params
        ):
            raise ValueError(
                f"Unsupported shape: {shape}, must have a from_params method"
            )

        self._shape: type[gb.Shape2D] = gb_shape
        self._pos_sampler: PositionSampler2D = pos_sampler
        self._size_samplers: dict[str, Any] = Validator.as_dict(size_samplers)
        self._is_circular: bool = issubclass(gb_shape, gb.Circle)

    def sample(self) -> gb.Shape2D:
        """
        Draw a single randomized geometric inclusion instance.

        Returns
        -------
        gb.Shape2D
            Constructed shape instance positioned and sized according to the samplers.
        """
        pos_params = self._pos_sampler.sample()
        if self._is_circular:
            pos_params = {
                "xc": float(pos_params["xc"]),
                "yc": float(pos_params["yc"]),
            }
        else:
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


def _normalize_shape_size_params(shape_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize shape size parameter keys and alias variants.

    Parameters
    ----------
    shape_name : str
        Lowercase shape type name.
    params : dict[str, Any]
        Raw parameter dictionary from user configuration.

    Returns
    -------
    dict[str, Any]
        Standardized parameter mapping.
    """
    normalized: dict[str, Any] = {}
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
    Initialize and place inclusions across the domain to satisfy prescribed volume fractions.

    Iterates through each phase in ``shapes_config``, instantiates the corresponding
    spatial position and size samplers, and sequentially adds inclusions until the
    accumulated area satisfies the target volume fraction:

    $$A_{\\text{target}} = A_{\\text{domain}} \\times \\phi_i$$

    Parameters
    ----------
    cell_domain : CellDomain2D
        The 2D cell domain host bounding the inclusions.
    shapes_config : Sequence[ShapeConfig]
        Sequence of shape configurations specifying name, target volume fraction,
        and parameter distributions.
    rng : np.random.Generator
        Random number generator for reproducible sampling.
    init_method : {'uniform', 'lhs', 'sobol'}, optional
        Spatial position placement algorithm (default: ``'uniform'``).

    Returns
    -------
    Inclusions2D
        Collection of initialized inclusions scattered across the domain.

    Raises
    ------
    ValueError
        If cumulative volume fraction of the configured shapes exceeds 1.0.
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
        gb_shape_cls = ShapeRegistry.get(shape_name)
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
    """
    Representation of a microstructure unit cell or RVE.

    Encapsulates the bounding domain (:class:`CellDomain2D`) and the collection
    of geometric inclusions (:class:`Inclusions2D`). Supports serialization to
    JSON, YAML, Pickle, and NPZ formats, plotting to raster/vector files, and
    periodic boundary extension.

    Parameters
    ----------
    domain : CellDomain
        The spatial bounding domain.
    shapes : Inclusions
        The collection of inclusions contained within the cell.

    Examples
    --------
    >>> import gbox as gb
    >>> from shapeforge.cell import Cell, CellDomain2D, Inclusions2D
    >>> domain = CellDomain2D([0, 0, 10, 10])
    >>> inclusions = Inclusions2D([gb.Circle(1.0, centre=[5, 5])])
    >>> cell = Cell(domain, inclusions)
    >>> len(cell.shapes)
    1
    >>> cell.domain.area
    100.0
    """

    __slots__ = ("_domain", "_opt_problem", "_shapes")

    def __init__(self, domain: CellDomain, shapes: Inclusions) -> None:
        self._domain: CellDomain = domain
        self._shapes: Inclusions = shapes
        self._opt_problem = None

    @property
    def domain(self) -> CellDomain2D:
        """
        The 2D spatial domain of the cell.

        Returns
        -------
        CellDomain2D
            Bounding domain.
        """
        return self._domain

    @property
    def shapes(self) -> Inclusions2D:
        """
        Collection of all inclusions within the cell (primary and periodic copies).

        Returns
        -------
        Inclusions2D
            Inclusion collection.
        """
        return self._shapes

    @property
    def primary_shapes(self) -> Inclusions2D:
        """
        Collection containing only primary inclusions (excluding periodic ghost copies).

        Returns
        -------
        Inclusions2D
            Primary inclusion collection.
        """
        return Inclusions2D(self._shapes.primary_inclusions)

    @property
    def periodic_shapes(self) -> Inclusions2D:
        """
        Collection containing only periodic copy inclusions.

        Returns
        -------
        Inclusions2D
            Periodic copy inclusion collection.
        """
        return Inclusions2D(self._shapes.periodic_copies)

    def make_periodic(self) -> Self:
        """
        Create a new Cell with boundary-crossing shapes wrapped and periodic copies added.

        Returns
        -------
        Cell
            New Cell instance enforcing periodic boundary conditions.
        """
        new_shapes = self._shapes.make_periodic(self._domain)
        return self.__class__(self._domain, new_shapes)

    def clone(self) -> Self:
        """
        Create a deep copy of this Cell.

        Returns
        -------
        Cell
            Cloned Cell instance.
        """
        return self.__class__(self._domain, self._shapes.clone())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Reconstruct a Cell instance from a serialized dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary containing ``'domain'`` and ``'shapes'`` mappings.

        Returns
        -------
        Cell
            Reconstructed Cell instance.
        """
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the Cell into a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``'domain'`` and ``'shapes'`` entries.
        """
        return {
            "domain": self._domain.to_dict(),
            "shapes": self._shapes.to_dict(),
        }

    def save(self, f_path: Path | str, *, overwrite: bool = False) -> Path:
        """
        Save the cell configuration to disk.

        Supported file extensions: ``.json``, ``.yaml``, ``.yml``, ``.pkl``,
        ``.pickle``, and ``.npz``.

        Parameters
        ----------
        f_path : str or Path
            Destination file path.
        overwrite : bool, optional
            Whether to overwrite an existing file (default: False).

        Returns
        -------
        Path
            Path to the saved file.

        Raises
        ------
        FileExistsError
            If ``overwrite=False`` and destination file already exists.
        ValueError
            If extension is unsupported.
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
        """
        Load a Cell instance from a serialized file on disk.

        Supports ``.json``, ``.yaml``, ``.yml``, ``.pkl``, ``.pickle``, and ``.npz``.

        Parameters
        ----------
        f_path : str or Path
            Path to the file containing serialized cell data.

        Returns
        -------
        Cell
            Reconstructed Cell instance.

        Raises
        ------
        FileNotFoundError
            If ``f_path`` does not exist.
        ValueError
            If file format is unsupported or file content is invalid.
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

    def plot(self, f_path: str | Path | None = None, **kwargs: Any) -> np.ndarray:
        """
        Render the cell into an image plot and return the rasterized pixel array.

        Parameters
        ----------
        f_path : str or Path or None, optional
            File path where the rendered figure should be saved (e.g. ``'cell.png'``).
            If None, image is not saved to disk.
        **kwargs : Any
            Plotting keyword arguments forwarded to :class:`~shapeforge.plotter.CellPlotter`
            (e.g., ``size``, ``background``, ``foreground``, ``dpi``).

        Returns
        -------
        np.ndarray
            Rendered RGB image array of shape ``(height, width, 3)``.
        """
        sp = CellPlotter(**kwargs)
        return sp.plot(
            shapes=[i for i in self.shapes],
            bounds=self.domain.domain,
            path=f_path,
        )

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Self

import gbox as gb
import matplotlib.patches as mpatches
import numpy as np
from gbox.core.points import Point2D
from gbox.core.utils import Angle, Bounds2DRectangular, Validator

from shapeforge.registry import register_shape


@register_shape("stadium")
class Stadium(gb.Shape2D):
    """A stadium (capsule) 2D shape composed of a rectangle and two semicircles."""

    def __init__(
        self,
        length: float,
        radius: float,
        centre: Sequence[float] = (0.0, 0.0),
        orientation: Angle | None = None,
    ):
        if length <= 0 or radius <= 0:
            raise ValueError("length and radius must be positive")
        self._length = float(length)
        self._radius = float(radius)
        orientation = orientation or Angle.rad(0.0)
        self._position = gb.shapes_2d.Shape2DPose(
            float(centre[0]), float(centre[1]), orientation
        )

    @property
    def length(self) -> float:
        return self._length

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def position(self) -> gb.shapes_2d.Shape2DPose:
        return self._position

    @position.setter
    def position(
        self,
        position: gb.shapes_2d.Shape2DPose
        | tuple[float, float, Angle]
        | dict[str, float | Angle],
    ) -> None:
        if isinstance(position, gb.shapes_2d.Shape2DPose):
            self._position = position
        elif isinstance(position, tuple):
            self._position = gb.shapes_2d.Shape2DPose(*position)
        elif isinstance(position, dict):
            self._position = gb.shapes_2d.Shape2DPose.from_dict(position)
        else:
            raise TypeError("Invalid position type")

    @property
    def area(self) -> float:
        return float(2.0 * self._length * self._radius + math.pi * self._radius**2)

    @property
    def perimeter(self) -> float:
        return float(2.0 * self._length + 2.0 * math.pi * self._radius)

    @property
    def equivalent_circle_radius(self) -> float:
        return float(math.sqrt(self.area / math.pi))

    def contains_point(
        self, point: Point2D | Sequence[float]
    ) -> Literal[-1, 0, 1]:
        px = float(point[0]) if not isinstance(point, Point2D) else float(point.x)
        py = float(point[1]) if not isinstance(point, Point2D) else float(point.y)

        # Transform point into body frame
        dx = px - self._position.x
        dy = py - self._position.y
        th = float(self._position.orientation.radians)
        c, s = math.cos(th), math.sin(th)
        bx = c * dx + s * dy
        by = -s * dx + c * dy

        # Clamp bx to segment [-length/2, length/2]
        half_l = self._length / 2.0
        clamped_x = max(-half_l, min(half_l, bx))
        dist_sq = (bx - clamped_x) ** 2 + by**2
        r_sq = self._radius**2

        if math.isclose(dist_sq, r_sq, rel_tol=1e-5):
            return 0
        return 1 if dist_sq < r_sq else -1

    def union_of_circles(self) -> gb.CirclesArray:
        # Generate overlapping circles along the centerline [-length/2, length/2]
        half_l = self._length / 2.0
        step = max(self._radius * 0.5, 0.05)
        n_pts = max(3, math.ceil(self._length / step) + 1)
        xs = np.linspace(-half_l, half_l, n_pts)
        ys = np.zeros_like(xs)
        radii = np.full_like(xs, self._radius)
        centres = np.column_stack([xs, ys])
        return gb.CirclesArray(centres, radii)

    @property
    def bounding_box(self) -> Bounds2DRectangular:
        # Conservative bounding box using centerline endpoints expanded by radius
        half_l = self._length / 2.0
        th = float(self._position.orientation.radians)
        c, s = math.cos(th), math.sin(th)
        x0, y0 = self._position.x, self._position.y
        p1_x, p1_y = x0 - half_l * c, y0 - half_l * s
        p2_x, p2_y = x0 + half_l * c, y0 + half_l * s
        r = self._radius
        return Bounds2DRectangular(
            x_min=min(p1_x, p2_x) - r,
            y_min=min(p1_y, p2_y) - r,
            x_max=max(p1_x, p2_x) + r,
            y_max=max(p1_y, p2_y) + r,
        )

    def boundary_points(self) -> np.ndarray:
        """Return (N, 2) array of boundary coordinates in world space."""
        half_l = self._length / 2.0
        r = self._radius
        # Body frame boundary: right semicircle, top line, left semicircle, bottom line
        thetas_right = np.linspace(-math.pi / 2, math.pi / 2, 20)
        xs_right = half_l + r * np.cos(thetas_right)
        ys_right = r * np.sin(thetas_right)

        thetas_left = np.linspace(math.pi / 2, 3 * math.pi / 2, 20)
        xs_left = -half_l + r * np.cos(thetas_left)
        ys_left = r * np.sin(thetas_left)

        bx = np.concatenate([xs_right, xs_left])
        by = np.concatenate([ys_right, ys_left])

        # Transform to world space
        x0, y0 = self._position.x, self._position.y
        th = float(self._position.orientation.radians)
        c, s = math.cos(th), math.sin(th)
        wx = x0 + c * bx - s * by
        wy = y0 + s * bx + c * by
        return np.column_stack([wx, wy])

    def clone(self) -> Self:
        c = self.__class__(
            self._length,
            self._radius,
            centre=(self._position.x, self._position.y),
            orientation=gb.Angle.rad(float(self._position.orientation.radians)),
        )
        if getattr(self, "is_periodic_copy", False):
            c.is_periodic_copy = True
        return c

    @classmethod
    def from_params(
        cls,
        position_params: dict[str, float | gb.Angle],
        size_params: dict[str, float],
    ) -> Self:
        Validator.as_dict(
            position_params,
            ["xc", "yc", "major_axis_angle"],
            [float, float, gb.Angle],
            name="position_params",
        )
        Validator.as_dict(
            size_params,
            ["length", "radius"],
            [float, float],
            name="size_params",
        )
        return cls(
            length=float(size_params["length"]),
            radius=float(size_params["radius"]),
            centre=(float(position_params["xc"]), float(position_params["yc"])),
            orientation=position_params["major_axis_angle"],
        )

    def to_dict(self) -> dict:
        return {
            "length": self._length,
            "radius": self._radius,
            "centre": [self._position.x, self._position.y],
            "major_axis_angle": [
                float(self._position.orientation.radians),
                "rad",
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d = Validator.as_dict(
            d,
            ["length", "radius"],
            [float, float],
            name="stadium_dict",
        )
        centre = d.get("centre", (0.0, 0.0))
        angle_raw = d.get("major_axis_angle", 0.0)
        if isinstance(angle_raw, (list, tuple)):
            ori = gb.Angle(float(angle_raw[0]), str(angle_raw[1]))
        elif isinstance(angle_raw, (int, float)):
            ori = gb.Angle.rad(float(angle_raw))
        else:
            ori = gb.Angle.rad(0.0)
        return cls(
            length=float(d["length"]),
            radius=float(d["radius"]),
            centre=centre,
            orientation=ori,
        )


class CustomSquare(gb.Shape2D):
    """A square shape defined by side length, using to_patch() for rendering."""

    def __init__(
        self,
        side: float,
        centre: Sequence[float] = (0.0, 0.0),
        orientation: gb.Angle | None = None,
    ):
        self._side = float(side)
        self._position = gb.shapes_2d.Shape2DPose(
            float(centre[0]),
            float(centre[1]),
            orientation or gb.Angle.rad(0.0),
        )

    @property
    def side(self) -> float:
        return self._side

    @property
    def position(self) -> gb.shapes_2d.Shape2DPose:
        return self._position

    @position.setter
    def position(self, pos: gb.shapes_2d.Shape2DPose) -> None:
        self._position = pos

    @property
    def area(self) -> float:
        return self._side**2

    @property
    def perimeter(self) -> float:
        return 4.0 * self._side

    def contains_point(self, point: Point2D | Sequence[float]) -> Literal[-1, 0, 1]:
        return 1

    def union_of_circles(self) -> gb.CirclesArray:
        # Approximate square with a 3x3 grid of circles
        r = self._side / 6.0
        coords = np.linspace(-self._side / 3.0, self._side / 3.0, 3)
        gx, gy = np.meshgrid(coords, coords)
        centres = np.column_stack([gx.ravel(), gy.ravel()])
        radii = np.full(len(centres), r)
        return gb.CirclesArray(centres, radii)

    @property
    def bounding_box(self) -> Bounds2DRectangular:
        r = self._side * math.sqrt(2.0) / 2.0
        return Bounds2DRectangular(
            self._position.x - r,
            self._position.y - r,
            self._position.x + r,
            self._position.y + r,
        )

    def to_patch(self) -> mpatches.Rectangle:
        """Render as a matplotlib Rectangle."""
        half = self._side / 2.0
        th_deg = self._position.orientation.degrees
        # Body frame lower-left (-half, -half) transformed to world space
        th = math.radians(th_deg)
        c, s = math.cos(th), math.sin(th)
        ll_x = self._position.x - half * c + half * s
        ll_y = self._position.y - half * s - half * c
        return mpatches.Rectangle(
            (ll_x, ll_y),
            self._side,
            self._side,
            angle=th_deg,
        )

    def clone(self) -> Self:
        c = self.__class__(
            self._side,
            centre=(self._position.x, self._position.y),
            orientation=gb.Angle.rad(float(self._position.orientation.radians)),
        )
        if getattr(self, "is_periodic_copy", False):
            c.is_periodic_copy = True
        return c

    @classmethod
    def from_params(
        cls,
        position_params: dict[str, float | gb.Angle],
        size_params: dict[str, float],
    ) -> Self:
        return cls(
            side=float(size_params["side"]),
            centre=(float(position_params["xc"]), float(position_params["yc"])),
            orientation=position_params.get("major_axis_angle", gb.Angle.rad(0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "side": self._side,
            "centre": [self._position.x, self._position.y],
            "major_axis_angle": float(self._position.orientation.radians),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            side=float(d["side"]),
            centre=d.get("centre", (0.0, 0.0)),
            orientation=gb.Angle.rad(float(d.get("major_axis_angle", 0.0))),
        )

from __future__ import annotations

from typing import Self

import gbox as gb
import numpy as np

from shapeforge import register_shape


@register_shape("cshape")
class CShape(gb.Shape2D):
    """C-shaped inclusion formed by sweeping disk r_small along arc r_curve."""

    def __init__(
        self,
        r_small: float,
        r_curve: float,
        c_theta: float | gb.Angle = 0.0,
        centre=(0.0, 0.0),
        orientation: float | gb.Angle | None = None,
    ):
        self._r_small = float(r_small)
        self._r_curve = float(r_curve)
        self._c_theta = (
            c_theta
            if isinstance(c_theta, gb.Angle)
            else gb.Angle.rad(float(c_theta))
        )
        ori = (
            orientation
            if isinstance(orientation, gb.Angle)
            else gb.Angle.rad(float(orientation or 0.0))
        )
        self._position = gb.shapes_2d.Shape2DPose(centre[0], centre[1], ori)

    # 1. Pose for optimizer and periodicity
    @property
    def position(self) -> gb.shapes_2d.Shape2DPose:
        return self._position

    @position.setter
    def position(self, pos) -> None:
        self._position = pos

    # 2. Area for volume fraction tracking
    @property
    def area(self) -> float:
        ends_area = np.pi * self._r_small**2
        strip_area = 2.0 * self._r_curve * self._r_small * self._c_theta.radians
        return float(ends_area + strip_area)

    # 3. Contact physics for overlap optimizer
    def union_of_circles(self) -> gb.CirclesArray:
        th = np.linspace(0, self._c_theta.radians, 20)
        xs = self._r_curve * np.cos(th)
        ys = self._r_curve * np.sin(th)
        points = gb.PointArray2D.from_named_dims(xs=xs, ys=ys)
        points = points.transform(
            dx=self._position.x,
            dy=self._position.y,
            angle=self._position.orientation,
        )
        return gb.CirclesArray(points, np.full_like(th, self._r_small))

    # 4. Boundary polygon for plotting
    def boundary_points(self) -> np.ndarray:
        th_c = self._c_theta.radians
        r_out = self._r_curve + self._r_small
        r_in = self._r_curve - self._r_small

        th_out = np.linspace(0, th_c, 30)
        phi = np.linspace(th_c, th_c + np.pi, 15)
        th_in = np.linspace(th_c, 0, 30)
        psi = np.linspace(np.pi, 2 * np.pi, 15)

        x_out, y_out = r_out * np.cos(th_out), r_out * np.sin(th_out)
        x_cap_end = self._r_curve * np.cos(th_c) + self._r_small * np.cos(phi)
        y_cap_end = self._r_curve * np.sin(th_c) + self._r_small * np.sin(phi)
        x_in, y_in = r_in * np.cos(th_in), r_in * np.sin(th_in)
        x_cap_start = self._r_curve + self._r_small * np.cos(psi)
        y_cap_start = self._r_small * np.sin(psi)

        xs = np.concatenate(
            [x_out, x_cap_end[1:], x_in[1:], x_cap_start[1:-1]]
        )
        ys = np.concatenate(
            [y_out, y_cap_end[1:], y_in[1:], y_cap_start[1:-1]]
        )

        points = gb.PointArray2D.from_named_dims(xs=xs, ys=ys)
        transformed = points.transform(
            dx=self._position.x,
            dy=self._position.y,
            angle=self._position.orientation,
        )
        return transformed.coordinates

    # 5. Bounds for broadphase culling & periodicity
    @property
    def bounding_box(self) -> gb.Bounds2DRectangular:
        pts = self.boundary_points()
        return gb.Bounds2DRectangular(
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        )

    # 6. Cloning for periodic copies and cell snapshots
    def clone(self) -> Self:
        c = self.__class__(
            self._r_small,
            self._r_curve,
            self._c_theta.radians,
            (self._position.x, self._position.y),
            self._position.orientation,
        )
        if getattr(self, "is_periodic_copy", False):
            c.is_periodic_copy = True
        return c

    # 7. Factory constructor for random sampling
    @classmethod
    def from_params(cls, position_params: dict, size_params: dict) -> Self:
        ori = position_params.get("major_axis_angle") or position_params.get(
            "orientation"
        )
        return cls(
            size_params["r_small"],
            size_params["r_curve"],
            size_params["c_theta"],
            centre=(position_params["xc"], position_params["yc"]),
            orientation=ori,
        )

    # 8. Serialization to / from JSON, YAML, NPZ
    def to_dict(self) -> dict:
        return {
            "r_small": self._r_small,
            "r_curve": self._r_curve,
            "c_theta": self._c_theta.radians,
            "centre": [self._position.x, self._position.y],
            "orientation": self._position.orientation.radians,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            d["r_small"],
            d["r_curve"],
            d["c_theta"],
            d.get("centre", (0.0, 0.0)),
            d.get("orientation", 0.0),
        )

    # 9. Stubs to satisfy gbox.Shape2D ABC (not used by ShapeForge)
    @property
    def perimeter(self) -> float:
        return float(
            2.0 * np.pi * self._r_small
            + 2.0 * self._r_curve * self._c_theta.radians
        )

    def contains_point(self, point) -> int:
        return -1

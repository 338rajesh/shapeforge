from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gbox as gb
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon


def _to_rgb(color: Any) -> tuple[int, int, int]:
    """Convert color specification (name, hex, or tuple) to (R, G, B) in 0-255."""
    try:
        from gbox.plots.render import _to_rgb as gbox_to_rgb

        return gbox_to_rgb(color)
    except (ImportError, ValueError, AttributeError):
        import matplotlib.colors as mcolors

        r, g, b = mcolors.to_rgb(color)
        return (round(r * 255), round(g * 255), round(b * 255))


class CellPlotter(gb.ShapesPlotter):
    """Plotter supporting both native gbox shapes and user-defined custom shapes.

    For custom shapes, visual rendering adheres to:
    1. If the shape defines ``to_patch()`` or ``as_patch()``, it uses the returned
       Matplotlib ``Patch`` (or collection of patches) directly.
    2. Otherwise, if the shape defines ``boundary_points()`` or ``get_boundary_points()``,
       it renders the boundary as a closed ``matplotlib.patches.Polygon``.
    3. If the shape defines ``body_boundary_points()`` or ``local_boundary_points()``,
       it transforms the body-frame coordinates using the shape's position pose into world
       coordinates and renders as a closed ``Polygon``.
    4. If neither is defined, a descriptive ``TypeError`` is raised.
    """

    def _normalize(
        self, shapes: Any
    ) -> list[tuple[Any, tuple[int, int, int]]]:
        """Convert the user's input into a list of (shape, rgb_color) pairs."""
        if not isinstance(shapes, Sequence) or isinstance(
            shapes, (str, bytes, np.ndarray)
        ):
            shapes = [shapes]

        items: list[tuple[Any, tuple[int, int, int]]] = []
        for item in shapes:
            if isinstance(item, tuple) and len(item) == 2:
                items.append((item[0], _to_rgb(item[1])))
            else:
                items.append((item, self._foreground))
        return items

    def _to_patches(self, shape: Any) -> list[Patch]:
        # Built-in native gbox shapes
        if isinstance(shape, (gb.Circle, gb.Ellipse, gb.CirclesArray)):
            return super()._to_patches(shape)

        # 1. Custom shape defines to_patch() or as_patch()
        for method_name in ("to_patch", "as_patch"):
            if hasattr(shape, method_name) and callable(
                getattr(shape, method_name)
            ):
                res = getattr(shape, method_name)()
                if isinstance(res, Patch):
                    return [res]
                elif isinstance(res, (list, tuple)):
                    return list(res)

        # 2. Custom shape defines boundary_points() or get_boundary_points()
        for method_name in ("boundary_points", "get_boundary_points"):
            if hasattr(shape, method_name):
                attr = getattr(shape, method_name)
                pts = attr() if callable(attr) else attr
                if pts is not None:
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                        return [MplPolygon(arr, closed=True)]

        # 3. Custom shape defines body_boundary_points() or local_boundary_points()
        for method_name in ("body_boundary_points", "local_boundary_points"):
            if hasattr(shape, method_name):
                attr = getattr(shape, method_name)
                pts = attr() if callable(attr) else attr
                if pts is not None:
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                        x = float(shape.position.x)
                        y = float(shape.position.y)
                        th = float(shape.position.orientation.radians)
                        c, s = np.cos(th), np.sin(th)
                        xs = x + c * arr[:, 0] - s * arr[:, 1]
                        ys = y + s * arr[:, 0] + c * arr[:, 1]
                        world_pts = np.column_stack([xs, ys])
                        return [MplPolygon(world_pts, closed=True)]

        raise TypeError(
            f"Custom shape '{type(shape).__name__}' cannot be plotted: "
            "it must implement either 'to_patch()' returning a matplotlib Patch, "
            "or 'boundary_points()' returning an (N, 2) array of boundary coordinates."
        )

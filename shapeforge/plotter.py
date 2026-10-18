"""
Microstructure cell plotting and Matplotlib rendering engine.
=============================================================

Extends ``gbox.ShapesPlotter`` to support both native geometric shapes
(:class:`gbox.Circle`, :class:`gbox.Ellipse`, :class:`gbox.CirclesArray`) and
user-defined custom subclasses of :class:`gbox.Shape2D`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gbox as gb
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon


def _to_rgb(color: Any) -> tuple[int, int, int]:
    """
    Convert a flexible color specification into an integer (R, G, B) tuple.

    Supports named colors (e.g. ``'white'``, ``'red'``), hex codes (e.g. ``'#ffffff'``),
    integer grayscale values in [0, 255], or 3-element float/int tuples.

    Parameters
    ----------
    color : Any
        Color specification in string, integer, or tuple format.

    Returns
    -------
    tuple of int
        3-tuple ``(R, G, B)`` with values in $[0, 255]$.
    """
    try:
        from gbox.plots.render import _to_rgb as gbox_to_rgb

        return gbox_to_rgb(color)
    except (ImportError, ValueError, AttributeError):
        import matplotlib.colors as mcolors

        r, g, b = mcolors.to_rgb(color)
        return (round(r * 255), round(g * 255), round(b * 255))


class CellPlotter(gb.ShapesPlotter):
    """
    Plotter supporting both native gbox shapes and user-defined custom shapes.

    Renders collections of 2D geometric inclusions into Matplotlib figures
    or directly into rasterized NumPy image arrays.

    For custom shapes (subclasses of :class:`gbox.Shape2D`), rendering priority is:
    1. If the shape defines :meth:`to_patch` or :meth:`as_patch`, it uses the returned
       Matplotlib :class:`matplotlib.patches.Patch` (or sequence of patches) directly.
    2. If the shape defines :meth:`boundary_points` or :meth:`get_boundary_points`,
       it renders the boundary coordinates as a closed :class:`matplotlib.patches.Polygon`.
    3. If the shape defines :meth:`body_boundary_points` or :meth:`local_boundary_points`,
       it transforms the body-frame coordinates using the shape's pose into world
       coordinates and renders as a closed polygon.
    4. If none of these are defined, a descriptive :exc:`TypeError` is raised.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to ``gbox.ShapesPlotter`` (e.g. ``size``,
        ``background``, ``foreground``, ``dpi``).

    Examples
    --------
    >>> import gbox as gb
    >>> from shapeforge.plotter import CellPlotter
    >>> plotter = CellPlotter(background='black', foreground='white', size=(256, 256))
    >>> circles = [gb.Circle(1.0, centre=[2, 2]), gb.Circle(1.0, centre=[5, 5])]
    >>> arr = plotter.plot(circles, bounds=gb.Bounds2DRectangular(0, 0, 10, 10))
    >>> arr.shape
    (256, 256, 3)
    """

    def _normalize(
        self, shapes: Any
    ) -> list[tuple[Any, tuple[int, int, int]]]:
        """
        Normalize input shapes into a sequence of ``(shape, rgb_color)`` tuples.

        Parameters
        ----------
        shapes : Any
            Single shape, list of shapes, or list of ``(shape, color)`` tuples.

        Returns
        -------
        list of tuple
            List of ``(shape_instance, (r, g, b))`` pairs.
        """
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
        """
        Convert a shape instance into a list of Matplotlib Patch objects.

        Parameters
        ----------
        shape : gb.Shape2D
            Shape instance to convert.

        Returns
        -------
        list of Patch
            One or more Matplotlib patch objects representing the shape geometry.

        Raises
        ------
        TypeError
            If the shape cannot be converted to patches via built-in handlers,
            ``to_patch()``, or ``boundary_points()``.
        """
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

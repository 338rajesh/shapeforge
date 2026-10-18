"""
Overlap potential formulation and spatial contact evaluation.
=============================================================

Defines the mathematical contact potential and analytic gradient evaluation
for multi-body shape arrangements in 2D bounded or periodic domains.

Each geometric shape is represented as a union of bounding disks
(obtained via ``shape.union_of_circles()``). Inter-particle penetration is
evaluated using accelerated $k$-d tree spatial queries (``scipy.spatial.cKDTree``)
with periodic toroidal distance metrics when periodicity is enabled.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gbox.shapes.shapes_2d import Shape2DPose
from scipy.spatial import cKDTree

from .cell import CellDomain, CellDomain2D, Inclusions, Inclusions2D
from .optim import OptimisationProblem
from .utils import PI


class ShapesOverlap(OptimisationProblem):
    """
    Base optimization problem formulation for multi-body overlap minimization.

    Parameters
    ----------
    domain : CellDomain
        Spatial bounding domain.
    shapes : Inclusions
        Collection of inclusions to be packed.
    periodicity : bool, optional
        Whether to enforce periodic boundary conditions across domain edges
        (default: False).
    buffer_thickness_ratio : float, optional
        Minimum surface-to-surface gap buffer specified as a fraction of each
        shape's equivalent circle radius (default: 0.05).
    proj_buffer_ratio : float, optional
        Buffer ratio used for boundary clipping in non-periodic domains
        (default: 2.0).
    rng : np.random.Generator, optional
        NumPy random number generator for randomized projection damping.
        If None, ``np.random.default_rng()`` is used.

    Attributes
    ----------
    num_inclusions : int
        Number of inclusions in the system.
    periodicity : bool
        Flag indicating whether boundary conditions are periodic.
    x0 : np.ndarray
        Flattened initial pose vector of shape ``(3 * N,)``.
    """

    def __init__(
        self,
        domain: CellDomain,
        shapes: Inclusions,
        *,
        periodicity: bool = False,
        buffer_thickness_ratio: float = 0.05,
        proj_buffer_ratio: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.num_inclusions: int = len(shapes)
        self.periodicity: bool = periodicity
        self.x0: np.ndarray = shapes.get_positions(flat=True)

        self.shapes: Inclusions = shapes
        self.domain: CellDomain = domain
        self.buffer_thickness_ratio: float = buffer_thickness_ratio
        self.proj_buffer_ratio: float = proj_buffer_ratio
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    def _as_flat(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten an $(N, 3)$ pose matrix into a $1$D Fortran-ordered vector.

        Parameters
        ----------
        x : np.ndarray
            Matrix of shape ``(N, 3)`` where columns represent $x$, $y$, $\\theta$.

        Returns
        -------
        np.ndarray
            Flattened 1D array of shape ``(3 * N,)``.
        """
        return x.flatten(order="F")

    def _as_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape a $1$D flattened vector of length $3N$ into an $(N, 3)$ pose matrix.

        Parameters
        ----------
        x : np.ndarray
            Flattened 1D parameter array of shape ``(3 * N,)``.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(N, 3)``.
        """
        return x.reshape(self.num_inclusions, -1, order="F")


class CellShapes2DOverlap(ShapesOverlap):
    r"""
    Overlap penalty cost and analytic gradient for 2D cell packing.

    Approximates each inclusion as a union of body-frame disks. Computes the
    aggregate overlap potential:

    $$f(x) = \sum_{k < l} \max\left(0, (r_k^{\text{eff}} + r_l^{\text{eff}}) - d_{kl}\right)^2$$

    and its analytic gradient $\nabla f(x) = [\nabla_x f, \nabla_y f, \nabla_\theta f]$
    via the kinematic chain rule.

    Parameters
    ----------
    domain : CellDomain2D
        The 2D rectangular domain.
    shapes : Inclusions2D
        The collection of 2D inclusions to arrange.
    periodicity : bool, optional
        Whether periodic toroidal boundary conditions are enabled (default: False).
    buffer_thickness_ratio : float, optional
        Buffer gap fraction applied to effective radii (default: 0.05).
    proj_buffer_ratio : float, optional
        Projection boundary buffer ratio for non-periodic domains (default: 2.0).
    rng : np.random.Generator, optional
        Random generator for projection damping. If None, default generator is used.

    Examples
    --------
    >>> import gbox as gb
    >>> from shapeforge.cell import CellDomain2D, Inclusions2D
    >>> from shapeforge.overlap_optim import CellShapes2DOverlap
    >>> domain = CellDomain2D([0, 0, 10, 10])
    >>> shapes = Inclusions2D([gb.Circle(1.0, centre=[2, 2]), gb.Circle(1.0, centre=[2.5, 2])])
    >>> prob = CellShapes2DOverlap(domain, shapes)
    >>> cost = prob.func(prob.x0)
    >>> cost > 0.0
    True
    """

    def __init__(
        self,
        domain: CellDomain2D,
        shapes: Inclusions2D,
        *,
        periodicity: bool = False,
        buffer_thickness_ratio: float = 0.05,
        proj_buffer_ratio: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            domain,
            shapes,
            periodicity=periodicity,
            buffer_thickness_ratio=buffer_thickness_ratio,
            proj_buffer_ratio=proj_buffer_ratio,
            rng=rng,
        )
        bounds_dict = domain.bounds.to_dict()
        self._xmin: float = float(bounds_dict["x_min"])
        self._xmax: float = float(bounds_dict["x_max"])
        self._ymin: float = float(bounds_dict["y_min"])
        self._ymax: float = float(bounds_dict["y_max"])
        self._Lx: float = self._xmax - self._xmin
        self._Ly: float = self._ymax - self._ymin
        self._boxsize: np.ndarray | None = (
            np.array([self._Lx, self._Ly], dtype=float)
            if periodicity
            else None
        )

        if self.periodicity:
            self.x0 = self.projection(self.x0)

        equivalent_circle_radii = shapes.get_equivalent_circle_radii()
        self._ssd: np.ndarray = equivalent_circle_radii * buffer_thickness_ratio
        self._radii: np.ndarray = equivalent_circle_radii

        self._templates: list[np.ndarray] = self._build_templates()
        self._build_circle_arrays()
        self._cache_x: np.ndarray | None = None
        self._cache: tuple[float, np.ndarray] | None = None

    def _build_circle_arrays(self) -> None:
        """Precompute flattened circle attributes for vectorized cKDTree queries."""
        shape_idx_list = []
        px_list = []
        py_list = []
        r_list = []
        ssd_list = []
        for i, tpl in enumerate(self._templates):
            for row in tpl:
                shape_idx_list.append(i)
                px_list.append(row[0])
                py_list.append(row[1])
                r_list.append(row[2])
                ssd_list.append(self._ssd[i])

        self._shape_idx = np.array(shape_idx_list, dtype=int)
        self._local_px = np.array(px_list, dtype=float)
        self._local_py = np.array(py_list, dtype=float)
        self._circle_radii = np.array(r_list, dtype=float)
        self._circle_ssds = np.array(ssd_list, dtype=float)
        self._circle_eff_radii = self._circle_radii + self._circle_ssds

        if len(self._circle_eff_radii) >= 2:
            sorted_eff = np.sort(self._circle_eff_radii)
            self._max_reach = float(sorted_eff[-1] + sorted_eff[-2])
        elif len(self._circle_eff_radii) == 1:
            self._max_reach = float(2.0 * self._circle_eff_radii[0])
        else:
            self._max_reach = 0.0

    def _build_templates(self) -> list[np.ndarray]:
        """
        Extract body-frame circle representations $(c_x, c_y, r)$ at pose $(0, 0, 0)$.

        Returns
        -------
        list of np.ndarray
            List of template arrays for each shape, where each array has shape
            ``(M_i, 3)``.
        """
        templates = []
        for shape in self.shapes:
            original_pose = shape.position
            shape.position = Shape2DPose(0.0, 0.0, 0.0)
            circles = [
                (c.centre[0], c.centre[1], c.radius)
                for c in shape.union_of_circles()
            ]
            shape.position = original_pose
            templates.append(np.asarray(circles, dtype=float))
        return templates

    def _circles_at(
        self, positions: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform body-frame circles to world coordinates for given poses.

        Parameters
        ----------
        positions : np.ndarray
            Matrix of poses with shape ``(N, 3)``.

        Returns
        -------
        list of tuple
            List of tuples ``(world_cx, world_cy, radii)`` per shape.
        """
        out = []
        for (x, y, th), tpl in zip(positions, self._templates):
            c, s = np.cos(th), np.sin(th)
            px, py, r = tpl[:, 0], tpl[:, 1], tpl[:, 2]
            out.append((x + c * px - s * py, y + s * px + c * py, r))
        return out

    def _evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluate cost and analytic gradient jointly, caching on parameter array `x`.

        Parameters
        ----------
        x : np.ndarray
            Flattened pose vector of shape ``(3 * N,)``.

        Returns
        -------
        tuple of (float, np.ndarray)
            Tuple containing the scalar overlap cost and 1D gradient array.
        """
        x = np.asarray(x, dtype=float)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache

        pos = self._as_matrix(x)
        N = self.num_inclusions
        K = len(self._shape_idx)

        xs = pos[:, 0]
        ys = pos[:, 1]
        ths = pos[:, 2]

        c_th = np.cos(ths)
        s_th = np.sin(ths)

        # Vectorized world-frame circle centres across all shapes
        cx = (
            xs[self._shape_idx]
            + c_th[self._shape_idx] * self._local_px
            - s_th[self._shape_idx] * self._local_py
        )
        cy = (
            ys[self._shape_idx]
            + s_th[self._shape_idx] * self._local_px
            + c_th[self._shape_idx] * self._local_py
        )

        if self.periodicity:
            cx_tree = np.mod(cx - self._xmin, self._Lx)
            cx_tree = np.clip(cx_tree, 0.0, np.nextafter(self._Lx, 0.0))
            cy_tree = np.mod(cy - self._ymin, self._Ly)
            cy_tree = np.clip(cy_tree, 0.0, np.nextafter(self._Ly, 0.0))
            pts = np.column_stack((cx_tree, cy_tree))
            tree = cKDTree(pts, boxsize=self._boxsize)
        else:
            pts = np.column_stack((cx, cy))
            tree = cKDTree(pts)

        pairs = tree.query_pairs(r=self._max_reach, output_type="ndarray")

        cost = 0.0
        gcx = np.zeros(K, dtype=float)
        gcy = np.zeros(K, dtype=float)

        if len(pairs) > 0:
            k = pairs[:, 0]
            l = pairs[:, 1]
            shape_k = self._shape_idx[k]
            shape_l = self._shape_idx[l]
            diff_shape = shape_k != shape_l

            if diff_shape.any():
                k = k[diff_shape]
                l = l[diff_shape]

                if self.periodicity:
                    dx = cx_tree[k] - cx_tree[l]
                    dy = cy_tree[k] - cy_tree[l]
                    dx = dx - self._Lx * np.round(dx / self._Lx)
                    dy = dy - self._Ly * np.round(dy / self._Ly)
                else:
                    dx = cx[k] - cx[l]
                    dy = cy[k] - cy[l]

                d = np.hypot(dx, dy)
                c = (
                    self._circle_eff_radii[k] + self._circle_eff_radii[l]
                ) - d

                mask = c > 0.0
                if mask.any():
                    k_act = k[mask]
                    l_act = l[mask]
                    c_act = c[mask]
                    d_act = d[mask]
                    dx_act = dx[mask]
                    dy_act = dy[mask]

                    cost = float(np.sum(c_act**2))
                    w = -2.0 * c_act / np.maximum(d_act, 1e-12)
                    w_dx = w * dx_act
                    w_dy = w * dy_act

                    np.add.at(gcx, k_act, w_dx)
                    np.add.at(gcx, l_act, -w_dx)
                    np.add.at(gcy, k_act, w_dy)
                    np.add.at(gcy, l_act, -w_dy)

        gx = np.zeros(N, dtype=float)
        gy = np.zeros(N, dtype=float)
        gth = np.zeros(N, dtype=float)

        np.add.at(gx, self._shape_idx, gcx)
        np.add.at(gy, self._shape_idx, gcy)
        dth_k = gcx * (
            -s_th[self._shape_idx] * self._local_px
            - c_th[self._shape_idx] * self._local_py
        ) + gcy * (
            c_th[self._shape_idx] * self._local_px
            - s_th[self._shape_idx] * self._local_py
        )
        np.add.at(gth, self._shape_idx, dth_k)

        grad = np.concatenate([gx, gy, gth])
        self._cache_x = x.copy()
        self._cache = (cost, grad)
        return self._cache

    def func(self, x: np.ndarray) -> float:
        """
        Evaluate scalar overlap cost at pose vector `x`.

        Parameters
        ----------
        x : np.ndarray
            Flattened pose vector of shape ``(3 * N,)``.

        Returns
        -------
        float
            Sum of squared overlapping penetrations across all shape pairs.
        """
        self._eval_count["func"] += 1
        return self._evaluate(x)[0]

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the analytic gradient vector at pose vector `x`.

        Parameters
        ----------
        x : np.ndarray
            Flattened pose vector of shape ``(3 * N,)``.

        Returns
        -------
        np.ndarray
            1D gradient array of shape ``(3 * N,)`` ordered as
            ``[gx_1, ..., gx_N, gy_1, ..., gy_N, gth_1, ..., gth_N]``.
        """
        self._eval_count["grad"] += 1
        return self._evaluate(x)[1]

    def projection(self, x: np.ndarray) -> np.ndarray:
        """
        Project candidate poses onto the domain bounding constraints.

        In periodic mode, the projection is identity (all poses on the torus
        manifold are feasible). In non-periodic mode, coordinates are clipped
        within domain boundaries with randomized buffer damping.

        Parameters
        ----------
        x : np.ndarray
            Candidate pose vector of shape ``(3 * N,)``.

        Returns
        -------
        np.ndarray
            Feasible projected pose vector of shape ``(3 * N,)``.
        """
        if self.periodicity:
            self._eval_count["proj"] += 1
            return x.copy()

        positions = self._as_matrix(x)
        bounds = self.domain.bounds.to_dict()
        xlb, xub = bounds["x_min"], bounds["x_max"]
        ylb, yub = bounds["y_min"], bounds["y_max"]

        olb, oub = 0.0, 2.0 * PI

        for i in range(self.num_inclusions):
            buf_len = min(
                self.proj_buffer_ratio * self._radii[i] * float(self.rng.random()),
                0.5 * (xub - xlb),
            )
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            buf_len = min(
                self.proj_buffer_ratio * self._radii[i] * float(self.rng.random()),
                0.5 * (yub - ylb),
            )
            if positions[i, 1] > yub:
                positions[i, 1] = yub - buf_len
            elif positions[i, 1] < ylb:
                positions[i, 1] = ylb + buf_len

            if positions[i, 2] > oub:
                positions[i, 2] = oub
            elif positions[i, 2] < olb:
                positions[i, 2] = olb

        self._eval_count["proj"] += 1
        return self._as_flat(positions)

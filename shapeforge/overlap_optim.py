import numpy as np
from gbox.shapes.shapes_2d import Shape2DPose
from scipy.spatial import cKDTree

from .cell import CellDomain, CellDomain2D, Inclusions, Inclusions2D
from .optim import OptimisationProblem
from .utils import PI


class ShapesOverlap(OptimisationProblem):
    def __init__(
        self,
        domain: CellDomain,
        shapes: Inclusions,
        *,
        periodicity: bool = False,
        buffer_thickness_ratio: float = 0.05,
        proj_buffer_ratio: float = 2.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        self.num_inclusions = len(shapes)
        self.periodicity = periodicity
        self.x0 = shapes.get_positions(flat=True)

        self.shapes = shapes
        self.domain = domain
        self.buffer_thickness_ratio = buffer_thickness_ratio
        self.proj_buffer_ratio = proj_buffer_ratio
        self.rng = rng if rng is not None else np.random.default_rng()

    def _as_flat(self, x: np.ndarray) -> np.ndarray:
        return x.flatten(order="F")

    def _as_matrix(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.num_inclusions, -1, order="F")


class CellShapes2DOverlap(ShapesOverlap):
    """
    Overlap cost for a 2-D cell with inclusions, supporting periodic boundaries.

    Parameters
    ----------
    domain : CellDomain2D
    shapes : Inclusions2D
    periodicity : bool, optional
        Whether to enforce periodic boundary conditions across domain edges (default: False).
    buffer_thickness_ratio : float, optional
        Minimum surface-to-surface gap as a fraction of each shape's equivalent radius.
    proj_buffer_ratio : float, optional
        Projection buffer thickness ratio (used in non-periodic projection).
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
    ):
        super().__init__(
            domain,
            shapes,
            periodicity=periodicity,
            buffer_thickness_ratio=buffer_thickness_ratio,
            proj_buffer_ratio=proj_buffer_ratio,
            rng=rng,
        )
        bounds_dict = domain.bounds.to_dict()
        self._xmin = float(bounds_dict["x_min"])
        self._xmax = float(bounds_dict["x_max"])
        self._ymin = float(bounds_dict["y_min"])
        self._ymax = float(bounds_dict["y_max"])
        self._Lx = self._xmax - self._xmin
        self._Ly = self._ymax - self._ymin
        self._boxsize = (
            np.array([self._Lx, self._Ly], dtype=float)
            if periodicity
            else None
        )

        if self.periodicity:
            self.x0 = self.projection(self.x0)

        equivalent_circle_radii = shapes.get_equivalent_circle_radii()
        self._ssd = equivalent_circle_radii * buffer_thickness_ratio
        self._radii = equivalent_circle_radii

        self._templates = self._build_templates()  # body-frame circles
        self._build_circle_arrays()
        self._cache_x = None
        self._cache = None

    def _build_circle_arrays(self) -> None:
        """Precompute flattened circle attributes for vectorized cKDTree evaluation."""
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

    # ------------------------------------------------------------------
    # Geometry: a pure function of x, never reads self.shapes
    # ------------------------------------------------------------------
    def _build_templates(self) -> list[np.ndarray]:
        """Body-frame circles (cx, cy, r) for each shape, at pose (0, 0, 0)."""
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

    def _circles_at(self, positions: np.ndarray):
        """World-frame circle centres and radii for every shape at (N, 3) poses."""
        out = []
        for (x, y, th), tpl in zip(positions, self._templates):
            c, s = np.cos(th), np.sin(th)
            px, py, r = tpl[:, 0], tpl[:, 1], tpl[:, 2]
            out.append((x + c * px - s * py, y + s * px + c * py, r))
        return out

    # ------------------------------------------------------------------
    # Cost and gradient, evaluated together and cached on x
    # ------------------------------------------------------------------
    def _evaluate(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache

        pos = self._as_matrix(x)  # (N, 3): x, y, theta
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

                dx = cx[k] - cx[l]
                dy = cy[k] - cy[l]
                if self.periodicity:
                    dx = dx - self._Lx * np.round(dx / self._Lx)
                    dy = dy - self._Ly * np.round(dy / self._Ly)

                d = np.hypot(dx, dy)
                c = (
                    self._circle_eff_radii[k] + self._circle_eff_radii[l]
                ) - d  # penetration depth

                mask = c > 0.0
                if mask.any():
                    k_act = k[mask]
                    l_act = l[mask]
                    c_act = c[mask]
                    d_act = d[mask]
                    dx_act = dx[mask]
                    dy_act = dy[mask]

                    cost = float(np.sum(c_act**2))
                    # d(c^2)/d(centre_k) = -2c * (ck - cl) / d
                    w = -2.0 * c_act / np.maximum(d_act, 1e-12)
                    w_dx = w * dx_act
                    w_dy = w * dy_act

                    np.add.at(gcx, k_act, w_dx)
                    np.add.at(gcx, l_act, -w_dx)
                    np.add.at(gcy, k_act, w_dy)
                    np.add.at(gcy, l_act, -w_dy)

        # Chain rule: centre = (x, y) + R(theta) p_k
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

        grad = np.concatenate([gx, gy, gth])  # matches _as_flat order
        self._cache_x = x.copy()
        self._cache = (cost, grad)
        return self._cache

    def func(self, x: np.ndarray) -> float:
        self._eval_count["func"] += 1
        return self._evaluate(x)[0]

    def grad(self, x: np.ndarray) -> None:
        self._eval_count["grad"] += 1
        return self._evaluate(x)[1]

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._as_matrix(x)  # -> (N, 3) shaped matrix
        if self.periodicity:
            two_pi = 2.0 * PI
            positions[:, 0] = self._xmin + np.mod(positions[:, 0] - self._xmin, self._Lx)
            positions[:, 1] = self._ymin + np.mod(positions[:, 1] - self._ymin, self._Ly)
            positions[:, 2] = np.mod(positions[:, 2], two_pi)
        else:
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

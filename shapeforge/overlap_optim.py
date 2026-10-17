import numpy as np

# from gbox import Circle, CirclesArray
from gbox.shapes.shapes_2d import Shape2DPose

from .cell import CellDomain, CellDomain2D, Inclusions, Inclusions2D
from .optim import OptimisationProblem
from .utils import PI


class ShapesOverlap(OptimisationProblem):
    def __init__(
        self,
        domain: CellDomain,
        shapes: Inclusions,
        *,
        buffer_thickness_ratio: float = 0.05,
    ):
        super().__init__()
        self.num_inclusions = len(shapes)
        self.x0 = shapes.get_positions(flat=True)

        self.shapes = shapes
        self.domain = domain

    def _as_flat(self, x: np.ndarray) -> np.ndarray:
        return x.flatten(order="F")

    def _as_matrix(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.num_inclusions, -1, order="F")


class CellShapes2DOverlap(ShapesOverlap):
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
        Projection buffer thickness = proj_buffer_ratio x radius.
        Default 2.0 (mirrors Julia default).
    """

    def __init__(
        self,
        domain: CellDomain2D,
        shapes: Inclusions2D,
        *,
        buffer_thickness_ratio: float = 0.05,
    ):
        super().__init__(
            domain, shapes, buffer_thickness_ratio=buffer_thickness_ratio
        )
        equivalent_circle_radii = shapes.get_equivalent_circle_radii()
        self._ssd = equivalent_circle_radii * buffer_thickness_ratio
        self._radii = equivalent_circle_radii

        self._templates = self._build_templates()  # body-frame circles
        self._cache_x = None
        self._cache = None

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
        circles = self._circles_at(pos)
        N = self.num_inclusions

        cost = 0.0
        gcx = [np.zeros_like(c[0]) for c in circles]  # dCost/d(circle centre)
        gcy = [np.zeros_like(c[0]) for c in circles]

        for i in range(N):
            cxi, cyi, ri = circles[i]
            for j in range(i + 1, N):
                cxj, cyj, rj = circles[j]

                dx = cxi[:, None] - cxj[None, :]
                dy = cyi[:, None] - cyj[None, :]
                d = np.hypot(dx, dy)
                c = (
                    ri[:, None] + rj[None, :] + self._ssd[i] + self._ssd[j]
                ) - d  # penetration depth

                mask = c > 0.0
                if not mask.any():
                    continue

                cost += float(np.sum(c[mask] ** 2))
                # d(c^2)/d(centre_k) = -2c * (ck - cl) / d
                w = np.where(mask, -2.0 * c / np.maximum(d, 1e-12), 0.0)
                gcx[i] += (w * dx).sum(axis=1)
                gcx[j] -= (w * dx).sum(axis=0)
                gcy[i] += (w * dy).sum(axis=1)
                gcy[j] -= (w * dy).sum(axis=0)

        # Chain rule: centre = (x, y) + R(theta) p_k
        gx, gy, gth = np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            tpl = self._templates[i]
            th = pos[i, 2]
            c, s = np.cos(th), np.sin(th)
            px, py = tpl[:, 0], tpl[:, 1]
            gx[i] = gcx[i].sum()
            gy[i] = gcy[i].sum()
            gth[i] = np.sum(
                gcx[i] * (-s * px - c * py) + gcy[i] * (c * px - s * py)
            )

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

    def update_x(self, x_new: np.ndarray):
        pass

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._as_matrix(x)  # -> (N, 3) shaped matrix
        bounds = self.domain.bounds.to_dict()
        xlb, xub = bounds["x_min"], bounds["x_max"]
        ylb, yub = bounds["y_min"], bounds["y_max"]

        olb, oub = 0.0, 2.0 * PI

        for i in range(self.num_inclusions):
            buf_len = 0.25 * (xub - xlb) * np.random.random()
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            buf_len = 0.25 * (yub - ylb) * np.random.random()
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

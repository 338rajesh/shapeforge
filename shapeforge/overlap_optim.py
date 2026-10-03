import numpy as np
from gbox import Angle, CirclesArray

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
        self._x_prev = None

    def update_x(self, x_new: np.ndarray):
        x_new = self._as_matrix(x_new)
        self.shapes.set_positions(x_new)

    # def _get_bbox_overlap_matrix(self, uns: list[CirclesArray]) -> np.ndarray:
    #     bbox_overla_matrix = np.zeros(
    #         (self.num_inclusions, self.num_inclusions), dtype=np.bool_
    #     )
    #     for i in range(self.num_inclusions):
    #         for j in range(1 + i, self.num_inclusions):
    #             bbox_overla_matrix[i, j] = uns[i].bounding_box.overlaps(
    #                 uns[j].bounding_box
    #             )
    #     return bbox_overla_matrix

    def _overlap_cost_and_gradient(
        self,
        positions: np.ndarray,
        # uns: list[CirclesArray],
    ):
        xs, ys = positions.T

        cost = 0.0
        grad_x = np.zeros(self.num_inclusions)
        grad_y = np.zeros(self.num_inclusions)

        for i in range(self.num_inclusions):
            for j in range(1 + i, self.num_inclusions):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = float(np.hypot(dx, dy))

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
        # grad_o = np.zeros(self.num_inclusions)

        # bb_overlap_matrix = self._get_bbox_overlap_matrix(uns)

        # for i in range(self.num_inclusions):
        #     ixc, iyc = xs[i], ys[i]

        #     for j in range(1 + i, self.num_inclusions):
        #         jxc, jyc = xs[j], ys[j]

        #         if bb_overlap_matrix[i, j]:  # Proceed only if bbox overlaps
        #             for circle_k in uns[i]:
        #                 kxc, kyc = circle_k.centre
        #                 kr = circle_k.radius
        #                 k_buff_t = self._shapes_buffer_thickness[i]
        #                 for circle_l in uns[j]:
        #                     lxc, lyc = circle_l.centre
        #                     lr = circle_l.radius
        #                     l_buff_t = self._shapes_buffer_thickness[j]

        #                     dx_ikjl = kxc - lxc
        #                     dy_ikjl = kyc - lyc

        #                     distance_ikjl = float(np.hypot(dx_ikjl, dy_ikjl))
        #                     min_distance_ik_jl = kr + lr + k_buff_t + l_buff_t

        #                     c_ikjl = min_distance_ik_jl - distance_ikjl

        #                     if c_ikjl > 0.0:  # actual_distance < min_distance
        #                         dx_iik = ixc - kxc
        #                         dy_iik = iyc - kyc
        #                         dx_jjl = jxc - lxc
        #                         dy_jjl = jyc - lyc

        #                         cost += c_ikjl * c_ikjl

        #                         eps = 1e-06 if distance_ikjl < 1e-06 else 0.0
        #                         overlap_degree = c_ikjl / (distance_ikjl + eps)

        #                         temp_grad_x = overlap_degree * dx_ikjl
        #                         temp_grad_y = overlap_degree * dy_ikjl
        #                         grad_x[i] += temp_grad_x
        #                         grad_x[j] -= temp_grad_x
        #                         grad_y[i] += temp_grad_y
        #                         grad_y[j] -= temp_grad_y
        #                         grad_o[i] += overlap_degree * (
        #                             (dx_ikjl * dy_iik) - (dy_ikjl * dx_iik)
        #                         )
        #                         grad_o[j] -= overlap_degree * (
        #                             (dx_ikjl * dy_jjl) - (dy_ikjl * dx_jjl)
        #                         )
        # grad = -2.0 * np.column_stack([grad_x, grad_y, grad_o])
        # return cost, grad

    def f_and_grad(
        self, x: np.ndarray, x_prev: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        xy = self._as_matrix(x)
        f, g = self._overlap_cost_and_gradient(xy)
        self._eval_count["f_and_g"] += 1
        return f, self._as_flat(g)

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._as_matrix(x)  # -> (N, 3) shaped matrix
        bounds = self.domain.bounds.to_dict()
        xlb, xub = bounds["x_min"], bounds["x_max"]
        ylb, yub = bounds["y_min"], bounds["y_max"]

        # olb, oub = 0.0, 2.0 * PI

        for i in range(self.num_inclusions):
            buf_len = 0.05 * (xub - xlb) * np.random.random()
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            buf_len = 0.05 * (yub - ylb) * np.random.random()
            if positions[i, 1] > yub:
                positions[i, 1] = yub - buf_len
            elif positions[i, 1] < ylb:
                positions[i, 1] = ylb + buf_len

            # if positions[i, 2] > oub:
            #     positions[i, 2] = oub
            # elif positions[i, 2] < olb:
            #     positions[i, 2] = olb

        self._eval_count["proj"] += 1
        return self._as_flat(positions)

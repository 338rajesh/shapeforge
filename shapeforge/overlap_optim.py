import numpy as np
from gbox import CirclesArray

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
        self.num_inclusions = len(shapes)
        self.x0 = shapes.get_positions(flat=True)

        self.shapes = shapes
        self.domain = domain

    def _as_flat(self, x: np.ndarray) -> np.ndarray:
        return x.flatten(order="F")

    def _as_matrix(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.num_inclusions, 3, order="F")


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

        # Evaluate UnS
        self._uns = [a_shape.union_of_circles() for a_shape in shapes]

        self._shapes_buffer_thickness = (
            equivalent_circle_radii * buffer_thickness_ratio
        )

        self._x_prev = None

    def _get_bbox_overlap_matrix(self) -> np.ndarray:
        # use `uns` to get the overlap matrix of all shape combinations
        bbox_overla_matrix = np.zeros(
            (self.num_inclusions, self.num_inclusions), dtype=np.bool
        )
        for i in range(self.num_inclusions):
            for j in range(self.num_inclusions):
                bbox_overla_matrix[i, j] = self._uns[i].bounding_box.overlaps(
                    self._uns[j].bounding_box
                )
        return bbox_overla_matrix

    def _overlap_cost_and_gradient(
        self,
        positions: np.ndarray,
        uns: list[CirclesArray],
    ):
        xs, ys, _ = positions.T

        cost = 0.0
        grad_x = np.zeros(self.num_inclusions)
        grad_y = np.zeros(self.num_inclusions)
        grad_o = np.zeros(self.num_inclusions)

        bb_overlap_matrix = self._get_bbox_overlap_matrix()

        for i in range(self.num_inclusions):
            for j in range(1 + i, self.num_inclusions):
                if bb_overlap_matrix[i, j]:  # Proceed only if bbox overlaps
                    ith_circles_array = self._uns[i]
                    jth_circles_array = self._uns[j]

                    for k_xc, k_yc, k_r in enumerate(ith_circles_array):
                        for l_xc, l_yc, l_r in enumerate(jth_circles_array):
                            dx_ik_jl = k_xc - l_xc
                            dy_ik_jl = k_yc - l_yc
                            distance_ik_jl = np.hypot(dx_ik_jl, dy_ik_jl)
                            min_distance_ik_jl = (
                                k_r
                                + l_r
                                + self._shapes_buffer_thickness[i]
                                + self._shapes_buffer_thickness[j]
                            )
                            c_ik_jl = min_distance_ik_jl - distance_ik_jl
                            if c_ik_jl > 1e-06:
                                dx_i_ik = xs[i] - k_xc
                                dy_i_ik = ys[i] - k_yc
                                dx_j_jl = xs[j] - l_xc
                                dy_j_jl = ys[j] - l_yc

                                cost += c_ik_jl * c_ik_jl
                                if distance_ik_jl > 1e-06:
                                    overlap_degree = c_ik_jl / distance_ik_jl
                                else:
                                    overlap_degree = c_ik_jl / (
                                        distance_ik_jl + 1e-06
                                    )

                                temp_grad_x = overlap_degree * dx_ik_jl
                                temp_grad_y = overlap_degree * dy_ik_jl
                                grad_x[i] += temp_grad_x
                                grad_x[j] -= temp_grad_x
                                grad_y[i] += temp_grad_y
                                grad_y[j] -= temp_grad_y
                                grad_o[i] += overlap_degree * (
                                    (dx_ik_jl * dy_i_ik) - (dy_ik_jl * dx_i_ik)
                                )
                                grad_o[j] -= overlap_degree * (
                                    (dx_ik_jl * dy_j_jl) - (dy_ik_jl * dx_j_jl)
                                )

        grad = -2.0 * np.column_stack([grad_x, grad_y])
        return cost, grad

    def get_trial_state(self, x, shift):
        # Update UnS
        shift_matrix = self._as_matrix(shift)
        for i, a_uns in enumerate(self._uns):
            a_uns.centres.transform(
                dx=shift_matrix[i, 0], dy=shift_matrix[i, 1], in_place=True
            )
        return np.add(x, shift)

    def f_and_grad(
        self, x: np.ndarray, x_prev: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        xyo = self._as_matrix(x)
        uns = self._uns
        if isinstance(x_prev, np.ndarray):
            xyo_prev = self._as_matrix(x_prev)
            for i in range(self.num_inclusions):
                dx, dy, dt = xyo[i] - xyo_prev[i]
                uns[i].centres.transform(
                    dx=dx,
                    dy=dy,
                    angle=dt,
                    in_place=True,
                    pivot=(xyo_prev[0], xyo_prev[1]),
                )

        f, g = self._overlap_cost_and_gradient(xyo, uns=uns)
        self.eval_count["f_and_g"] += 1
        return f, self._as_flat(g)

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._as_matrix(x)  # -> (N, 3) shaped matrix
        xlb, xub = self.domain
        ylb, yub = self.domain.y_bounds
        olb, oub = 0.0, 2.0 * PI

        for i in range(self.num_inclusions):
            buf_len = self._proj_buffer[i] * np.random.random()
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            if positions[i, 1] > yub:
                positions[i, 1] = yub - buf_len
            elif positions[i, 1] < ylb:
                positions[i, 1] = ylb + buf_len

            if positions[i, 2] > oub:
                positions[i, 2] = oub
            elif positions[i, 2] < olb:
                positions[i, 2] = olb

        self.eval_count["proj"] += 1
        return self._as_flat(positions)

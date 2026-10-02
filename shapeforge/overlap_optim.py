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

        # Evaluate UnS
        self._uns = [a_shape.union_of_circles() for a_shape in shapes]

        self._shapes_buffer_thickness = (
            equivalent_circle_radii * buffer_thickness_ratio
        )

        self._x_prev = None

    def _get_bbox_overlap_matrix(self, uns=None) -> np.ndarray:
        uns = uns or self._uns
        # use `uns` to get the overlap matrix of all shape combinations
        bbox_overla_matrix = np.zeros(
            (self.num_inclusions, self.num_inclusions), dtype=np.bool_
        )
        for i in range(self.num_inclusions):
            for j in range(1 + i, self.num_inclusions):
                bbox_overla_matrix[i, j] = uns[i].bounding_box.overlaps(
                    uns[j].bounding_box
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

        bb_overlap_matrix = self._get_bbox_overlap_matrix(uns)

        num_bbox_overlaps = 0
        num_circle_overlaps = 0
        for i in range(self.num_inclusions):
            for j in range(1 + i, self.num_inclusions):
                if bb_overlap_matrix[i, j]:  # Proceed only if bbox overlaps
                    num_bbox_overlaps += 1
                    for circle_k in uns[i]:
                        for circle_l in uns[j]:
                            dx_ik_jl = circle_k.centre[0] - circle_l.centre[0]
                            dy_ik_jl = circle_k.centre[1] - circle_l.centre[1]

                            distance_ik_jl = float(
                                np.sqrt(
                                    dx_ik_jl * dx_ik_jl + dy_ik_jl * dy_ik_jl
                                )
                            )
                            min_distance_ik_jl = (
                                circle_k.radius
                                + circle_l.radius
                                + self._shapes_buffer_thickness[i]
                                + self._shapes_buffer_thickness[j]
                            )
                            c_ik_jl = min_distance_ik_jl - distance_ik_jl
                            if c_ik_jl > 0.0:
                                num_circle_overlaps += 1
                                dx_i_ik = xs[i] - circle_k.centre[0]
                                dy_i_ik = ys[i] - circle_k.centre[1]
                                dx_j_jl = xs[j] - circle_l.centre[0]
                                dy_j_jl = ys[j] - circle_l.centre[1]

                                cost += c_ik_jl * c_ik_jl

                                if abs(distance_ik_jl) >= 1e-06:
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

        num_overlaps = int(np.triu(bb_overlap_matrix, k=1).sum())
        print(
            f"Total Number of bbox overlaps: {num_overlaps}; "
            f"number of circle overlaps: {num_circle_overlaps}, "
            f"number of actual bbox overlaps: {num_bbox_overlaps}"
            f"Cost: {cost}"
        )
        grad = -2.0 * np.column_stack([grad_x, grad_y, grad_o])
        return cost, grad

    def f_and_grad(
        self, x: np.ndarray, x_prev: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        xyo = self._as_matrix(x)
        uns_trial = self._uns
        if isinstance(x_prev, np.ndarray):
            xyo_prev = self._as_matrix(x_prev)
            for i in range(self.num_inclusions):
                xi_prev, yi_prev, oi_prev = xyo_prev[i]
                uns_trial[i] = uns_trial[i].transform(
                    dx=xyo[i, 0] - xi_prev,
                    dy=xyo[i, 1] - yi_prev,
                    rot_angle=Angle.rad(xyo[i, 2] - oi_prev),
                    pivot=(float(xi_prev), float(yi_prev)),
                )

        f, g = self._overlap_cost_and_gradient(xyo, uns=uns_trial)
        self._eval_count["f_and_g"] += 1
        return f, self._as_flat(g)

    def projection(self, x: np.ndarray) -> np.ndarray:
        positions = self._as_matrix(x)  # -> (N, 3) shaped matrix
        bounds = self.domain.bounds.to_dict()
        xlb, xub = bounds["x_min"], bounds["x_max"]
        ylb, yub = bounds["y_min"], bounds["y_max"]

        olb, oub = 0.0, 2.0 * PI

        for i in range(self.num_inclusions):
            buf_len = 0.2 * (xub - xlb) * np.random.random()
            if positions[i, 0] > xub:
                positions[i, 0] = xub - buf_len
            elif positions[i, 0] < xlb:
                positions[i, 0] = xlb + buf_len

            buf_len = 0.2 * (yub - ylb) * np.random.random()
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

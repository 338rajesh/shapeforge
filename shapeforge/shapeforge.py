import argparse

from .cell_domain import CellDomain
from .inclusion import initialise_inclusions
from .cell import Cell
from .utils import load_yaml_config


# def get_initialized_inclusions(
#     vf: float = 0.5,
#     bounds: tuple = (0.0, 0.0, 1.0, 1.0),
#     radii_mean: float = 1.5,
#     radii_std: float = 0.5,
# ):
#     xlb, ylb, xub, yub = bounds
#     domain_area = (xub - xlb) * (yub - ylb)
#     inclusion_area = vf * domain_area

#     xc_distribution = stats.uniform(xlb, xub - xlb)
#     yc_distribution = stats.uniform(ylb, yub - ylb)

#     # Ensure the radius distribution is valid (non-negative)
#     r_min = max(0, radii_mean - 3 * radii_std)
#     r_max = radii_mean + 3 * radii_std
#     if r_min < 0:
#         raise ValueError("Radius distribution must have non-negative support.")
#     if r_max <= 0:
#         raise ValueError("Radius distribution must have positive support.")

#     radius_distribution = stats.truncnorm(
#         a=(r_min - radii_mean) / radii_std,
#         b=(r_max - radii_mean) / radii_std,
#         loc=radii_mean,
#         scale=radii_std,
#     )
#     inclusion = CircularInclusion(
#         xc_distribution=xc_distribution,
#         yc_distribution=yc_distribution,
#         radius_distribution=radius_distribution,
#         name="Circular Inclusion",
#     )
#     return inclusion.generate(cum_area=inclusion_area)


# class InclusionOverlapOptimisationProblem(OptimisationProblem):
#     def __init__(self, radii=None, bounds=None):
#         super().__init__()
#         self.radii = radii
#         self.radii_sum = radii.reshape(-1, 1) + radii.reshape(-1)
#         self.cost = None
#         self.bounds = bounds
#         self.pw_centres_distances: np.ndarray = None
#         self.pw_cost_matrix = None

#     def _eval_cost(self, x: np.ndarray) -> float:
#         x, y = np.split(x, 2)
#         if len(x) != len(y):
#             raise ValueError("Input arrays x and y must have the same length.")

#         # Find the pairwise distances between points
#         self.pw_centres_distances = np.sqrt(
#             np.add.outer(x**2, y**2) - 2 * np.outer(x, y)
#         )
#         self.pw_cost_matrix = self.radii_sum - self.pw_centres_distances
#         self.pw_cost_matrix[self.pw_cost_matrix < 0] = 0
#         return self

#     def f(self, x: np.ndarray) -> float:
#         self._eval_cost(x)
#         return self.pw_cost_matrix.sum()

#     def grad_f(self, x: np.ndarray) -> np.ndarray:
#         x, y = np.split(x, 2)
#         dx = x.reshape(-1, 1) - x
#         dy = y.reshape(-1, 1) - y
#         pw_cd = self.pw_centres_distances
#         pw_cd[pw_cd == 0] = 1e-10  # Avoid division by zero
#         grad_matrix = self.pw_cost_matrix / pw_cd
#         # remove diagonal elements from grad_matrix
#         np.fill_diagonal(grad_matrix, 0)
#         grad_x = (grad_matrix * dx).sum(axis=1)
#         grad_y = (grad_matrix * dy).sum(axis=1)
#         return -2.0 * np.concatenate([grad_x, grad_y])

#     def projection(self, x):
#         x, y = np.split(x, 2)
#         x = np.clip(x, self.bounds[0], self.bounds[2])
#         y = np.clip(y, self.bounds[1], self.bounds[3])
#         return np.concatenate([x, y])


# def undo_overlaps(ucd: UnitCellDomain, optimiser="nmspg", iter_max=1000):
#     """Undo overlaps in the UnitCellDomain using an optimisation algorithm."""
#     if optimiser != "nmspg":
#         raise ValueError(f"Unsupported optimiser: {optimiser}")

#     x_init = []
#     y_init = []
#     r_init = []
#     a_incl_circles_array: gb.CirclesArray = None
#     for a_incl_circles_array in ucd.inclusions:
#         x_init.extend(a_incl_circles_array.xc.tolist())
#         y_init.extend(a_incl_circles_array.yc.tolist())
#         r_init.extend(a_incl_circles_array.radii.tolist())

#     x_0 = np.concatenate([x_init, y_init])
#     opt_problem = InclusionOverlapOptimisationProblem(
#         radii=np.array(r_init),
#         bounds=ucd.domain_bounds,
#     )

#     print("Initial circles:", len(x_init))
#     print("Initial radii:", len(r_init))
#     print("Initial domain bounds:", ucd.domain_bounds)
#     print("Intial cost:", opt_problem.f(x_0))

#     opt_result = nmspg(
#         objective=opt_problem,
#         x0=x_0,
#         iter_max=iter_max,
#         p_bar=True,
#     )
#     opt_xy = opt_result.x_optimal
#     xyr = np.concatenate((opt_xy, r_init)).reshape(-1, 3)

#     opt_incl_circles_array = [
#         gb.CirclesArray(circles=a_xyr.reshape(1, 3), initial_capacity=1)
#         for a_xyr in xyr
#     ]

#     opt_ucd = UnitCellDomain(
#         bounds=ucd.domain_bounds,
#         inclusions=opt_incl_circles_array,
#     )

#     return opt_ucd


# def make_rve_with_circular_inclusions(
#     vf: float = 0.5,
#     bounds: tuple = (0.0, 0.0, 10.0, 10.0),
#     radii_mean: float = 1.5,
#     radii_std: float = 0.5,
# ) -> tuple[UnitCellDomain, UnitCellDomain]:
#     init_inclusions = get_initialized_inclusions(
#         vf, bounds, radii_mean=radii_mean, radii_std=radii_std
#     )
#     init_ucd = UnitCellDomain(inclusions=init_inclusions, bounds=bounds)
#     opt_ucd = undo_overlaps(init_ucd, iter_max=1000)
#     return init_ucd, opt_ucd


def generate_unit_cell(config: dict) -> None:
    """
    Generate a unit cell with the specified configuration.
    """
    # Cell Initialization
    cell_domain = CellDomain.from_config(config.get("cell_domain", {}))
    inclusions = initialise_inclusions(
        cell_domain, config.get("inclusions", [])
    )
    cell = Cell(cell_domain, inclusions)

    # Run Optim Loop to ensure there are no overlaps among inclusions
    #   Evaluate the cost function and gradients
    #     Add Periodic copies, if required
    #     cost evaluation
    #     gradients evaluation
    #   Update the inclusions positions
    #   Check for convergence
    #   If converged, return the optimised inclusions positions
    return cell


def main():
    parser = argparse.ArgumentParser(description=("ShapeForge CLI"))
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the shape forge.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config_file, show=True)
    generate_unit_cell(config)


if __name__ == "__main__":
    main()

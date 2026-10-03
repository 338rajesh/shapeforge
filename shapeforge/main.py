import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gbox.shapes.shapes_2d import Shape2DPose

from .cell import Cell, CellDomain2D, initialise_shapes_2d
from .config import ShapeForgeConfig
from .overlap_optim import CellShapes2DOverlap
from .utils import _load_dict, get_logger, plot_f_and_g_norms

logger = get_logger(__name__)

SCRATCH_DIR = Path("/home/rajesh/workshop/repos/shapeforge/scratch")


def generate_cell_2d(
    config: dict[str, Any] | str | Path,
) -> Sequence[tuple[Cell, Cell]]:
    """
    Generate a unit cell with the specified configuration.
    """
    logger.info("Starting the cell 2D genration...")
    config_dict = _load_dict(config)
    cfg = ShapeForgeConfig.from_dict(config_dict)
    cells: list[tuple[Cell, Cell]] = [
        (None, None) for _ in range(cfg.num_cells)
    ]
    solver_options = {
        "method": "nmspg",
        "iter_max": 1000,
        "iter_memory": 10,
        "epsilon": 1e-6,
        "spectral_step_min": 1e-30,
        "spectral_step_max": 1e30,
        "gamma": 0.0001,
        "sigma1": 0.1,
        "sigma2": 0.9,
        "ls_iter_max": 20,
        "p_bar": True,
    }
    for index in range(cfg.num_cells):
        rng_seed = cfg.metadata.rng_seed + index
        logger.info(f"Generating cell {index} with seed {rng_seed}")

        cell_domain = CellDomain2D(bounds=cfg.domain.bounds)
        logger.debug("  Domain of cell is created")

        shapes = initialise_shapes_2d(
            cell_domain, cfg.shapes, rng=np.random.default_rng(seed=rng_seed)
        )
        logger.debug(f"  Cell is initialised with {len(shapes)} shapes.")

        cell = Cell(cell_domain, shapes)
        logger.debug("  Cell object containing shapes and domain is created")

        initial_copy = cell.clone()
        initial_copy.plot(f_path=SCRATCH_DIR.joinpath(f"initial_{index}.png"))
        logger.debug("  A copy of the cell is created for comparison")

        # Solving the overlap
        overlap_problem = CellShapes2DOverlap(
            cell.domain, cell.shapes.clone(), buffer_thickness_ratio=0.05
        )

        _x0 = overlap_problem.x0
        import json

        with open(SCRATCH_DIR.joinpath(f"x0_{index}.json"), "w") as f:
            json.dump(
                {
                    "x0_flat": _x0.tolist(),
                    "x0_matrix": overlap_problem._as_matrix(_x0).tolist(),
                },
                f,
                indent=4,
            )

        solution = overlap_problem.solve(**solver_options)

        print(
            f"status: {solution.status}, \t iter count: {solution.iter_count}"
        )

        # Updating the shapes with the optimal positions
        positions = overlap_problem._as_matrix(solution.x_optimal)
        for idx, a_shape in enumerate(shapes):
            a_shape.position = Shape2DPose(*positions[idx], 0.0)

        cell.plot(f_path=SCRATCH_DIR.joinpath(f"final_{index}.png"))
        # cell.save(
        #     f_path=SCRATCH_DIR.joinpath(f"final_{index}.json"), overwrite=True
        # )
        plot_f_and_g_norms(
            sol=solution,
            f_path=SCRATCH_DIR.joinpath(f"fg_variation_{index}.png"),
        )
        cells[index] = (initial_copy, cell)
    return cells


def build_parser() -> argparse.Namespace:
    _log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    parser = argparse.ArgumentParser(description=("ShapeForge CLI"))
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the shape forge.",
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        choices=_log_levels + list(map(str.lower, _log_levels)),
        help="Set the logging level (default: WARNING)",
    )
    return parser.parse_args()


def main():
    args = build_parser()
    logger.setLevel(args.log.upper())
    generate_cell_2d(args.config_file)


if __name__ == "__main__":
    main()

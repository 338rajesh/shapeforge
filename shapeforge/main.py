import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gbox.shapes.shapes_2d import Shape2DPose

from .cell import Cell, CellDomain2D, initialise_shapes_2d
from .config import ShapeForgeConfig
from .overlap_optim import CellShapes2DOverlap
from .utils import get_logger

logger = get_logger(__name__)


def generate_cell_2d(
    config: dict[str, Any] | str | Path,
) -> Sequence[tuple[Cell, Cell]]:
    """
    Generate a unit cell with the specified configuration.
    """
    logger.info("Starting the cell 2D genration...")

    cfg = ShapeForgeConfig.from_dict(config)
    cells: list[tuple[Cell, Cell]] = [
        (None, None) for _ in range(cfg.num_cells)
    ]
    solver_options = {
        "method": "nmspg",
        "iter_max": 100,
        "iter_memory": 10,
        "epsilon": 1e-6,
        "spectral_step_min": 1e-30,
        "spectral_step_max": 1e30,
        "gamma": 0.0001,
        "sigma1": 0.1,
        "sigma2": 0.9,
        "ls_iter_max": 20,
        "p_bar": None,
    }
    for index in range(cfg.num_cells):
        rng_seed = cfg.metadata.rng_seed + index
        logger.info(f"Generating cell {index} with seed {rng_seed}")

        cell_domain = CellDomain2D(bounds=cfg.domain.bounds)
        shapes = initialise_shapes_2d(
            cell_domain, cfg.shapes, rng=np.random.default_rng(seed=rng_seed)
        )
        cell = Cell(cell_domain, shapes)
        initial_copy = cell.clone()

        # Solving the overlap
        overlap_problem = CellShapes2DOverlap(
            cell.domain, cell.shapes, ssd_ratio=0.05, proj_buffer_ratio=2.0
        )
        solution = overlap_problem.solve(**solver_options)

        # Updating the shapes with the optimal positions
        positions = solution.x_optimal.reshape(len(shapes), 3, order="F")
        for idx, a_shape in enumerate(shapes):
            a_shape.position = Shape2DPose(*positions[idx])

        cells[index] = (initial_copy, cell)
    return cells


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("ShapeForge CLI"))
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the shape forge.",
    )
    return parser.parse_args()


def main():
    args = build_parser()
    generate_cell_2d(args.config_file)


if __name__ == "__main__":
    main()

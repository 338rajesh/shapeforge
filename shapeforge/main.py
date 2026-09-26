import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cell import Cell2D, CellDomain2D, initialise_shapes_2d
from .config import ShapeForgeConfig
from .utils import get_logger

logger = get_logger(__name__)


def generate_cell_2d(
    config: dict[str, Any] | str | Path,
) -> Sequence[tuple[Cell2D, Cell2D]]:
    """
    Generate a unit cell with the specified configuration.
    """
    logger.info("Starting the cell 2D genration...")

    cfg = ShapeForgeConfig.from_dict(config)
    cells = [(None, None) for _ in range(cfg.num_cells)]
    for index in range(cfg.num_cells):
        rng_seed = cfg.metadata.rng_seed + index
        logger.info(f"Generating cell {index} with seed {rng_seed}")

        cell_domain = CellDomain2D(bounds=cfg.domain.bounds)
        shapes = initialise_shapes_2d(
            cell_domain, cfg.shapes, rng=np.random.default_rng(seed=rng_seed)
        )
        cell = Cell2D(cell_domain, shapes)
        cells[index][0] = cell.clone()

        cell.remove_inclusion_overlaps(
            ssd_ratio=cfg.get("min_gap", 0.05),
            proj_buffer_ratio=cfg.get("proj_buffer_ratio", 0.5),
        )
        cells[index][1] = cell


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

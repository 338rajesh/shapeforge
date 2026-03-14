import argparse
from pathlib import Path

import numpy as np

from .cell import CellDomain, initialise_inclusions
from .cell import Cell
from .utils import load_yaml


def generate_unit_cell(config: dict | str | Path) -> Cell:
    """
    Generate a unit cell with the specified configuration.
    """
    if isinstance(config, (str, Path)):
        config = load_yaml(config)

    if not isinstance(config, dict):
        raise ValueError("Expecting config to be a dictionary.")

    # Cell Initialization
    cell_domain = CellDomain.from_config(config.get("cell_domain", {}))
    inclusions = initialise_inclusions(
        config.get("inclusions", []),
        cell_domain,
        rng=np.random.default_rng(seed=config.get("randomness_seed")),
    )
    cell = Cell(cell_domain, inclusions)
    # cell.remove_inclusion_overlaps()
    print("Cell generation completed!.")
    return cell


def main():
    parser = argparse.ArgumentParser(description=("ShapeForge CLI"))
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the shape forge.",
    )
    args = parser.parse_args()

    cfg_fp = args.config_file

    print(f"Generating the Cell with configuration from '{cfg_fp}'")
    generate_unit_cell(args.config_file)


if __name__ == "__main__":
    main()

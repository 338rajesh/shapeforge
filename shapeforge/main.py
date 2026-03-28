import argparse
import json
from pathlib import Path

import numpy as np

from .cell import CellDomain, initialise_shapes
from .cell import Cell
from .utils import load_yaml


def generate_cell(config: dict | str | Path) -> Cell:
    """
    Generate a unit cell with the specified configuration.
    """
    print("Starting the cell genration...")
    if isinstance(config, (str, Path)):
        config = load_yaml(config)

    if not isinstance(config, dict):
        raise ValueError("Expecting config to be a dictionary.")

    print("> Loaded the configuration")
    # Output dir
    output_dir = Path(config.get("output_dir", Path.cwd()))
    output_dir.mkdir(exist_ok=True)
    verbose = config.get("verbose", False)
    if verbose:
        print("Generating the Cell with configuration:")
        print(json.dumps(config, indent=4))

    # Cell Initialization
    cell_domain = CellDomain.from_dict(config["domain"])
    print("> Initialized the domain")
    shapes = config.get("shapes", [])
    if len(shapes) == 0:
        print(
            "WARNING: no shapes are found in the config, so returning "
            "empty cell domain."
        )
        return Cell(cell_domain)  # just return the empty cell domain

    shapes = initialise_shapes(
        shapes,
        cell_domain,
        rng=np.random.default_rng(seed=config.get("rng_seed")),
        init_method=config.get("engine", {}).get("init_method", "uniform")
    )
    print("> Shapes are initialised in the domain.")
    
    cell = Cell(cell_domain, shapes)
    cell.plot(f_path=output_dir.joinpath("initial.png"))
    cell.remove_inclusion_overlaps(
        ssd_ratio=config.get("min_gap", 0.05),
        proj_buffer_ratio=config.get("proj_buffer_ratio", 0.5)
    )
    print("> Cell generation completed!")


    image_options = config.get("image", {})
    cell.plot(
        f_path=output_dir.joinpath("final.png"),
        shape_vis_options=image_options.get("shapes", {}),
        domain_vis_options=image_options.get("domain", {}),
        image_size=image_options.get("size", (256, 256))
    )

    return cell


def main():
    parser = argparse.ArgumentParser(description=("ShapeForge CLI"))
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the shape forge.",
    )
    args = parser.parse_args()
    generate_cell(args.config_file)


if __name__ == "__main__":
    main()

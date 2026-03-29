import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .cell import CellDomain, initialise_shapes
from .cell import Cell
from .utils import load_yaml


def _generate_cell(
    cfg: dict, get_init_cell: bool = False
) -> Cell | tuple[Cell, Cell]:
    # --------------------------------------------------------- #
    #               Cell Initialization                         #
    # --------------------------------------------------------- #
    cell_domain = CellDomain.from_dict(cfg["domain"])
    print("> Initialized the domain")
    shapes = cfg.get("shapes", [])
    if len(shapes) == 0:
        print(
            "WARNING: no shapes are found in the config, so returning "
            "empty cell domain."
        )
        return Cell(cell_domain)  # just return the empty cell domain

    shapes = initialise_shapes(
        shapes,
        cell_domain,
        rng=np.random.default_rng(seed=cfg.get("rng_seed")),
        init_method=cfg.get("engine", {}).get("init_method", "uniform"),
    )
    print("> Shapes are initialised in the domain.")
    cell = Cell(cell_domain, shapes)
    init_cell_copy = cell.clone()

    # --------------------------------------------------------- #
    #               Cell Optimisation                           #
    # --------------------------------------------------------- #
    cell.remove_inclusion_overlaps(
        ssd_ratio=cfg.get("min_gap", 0.05),
        proj_buffer_ratio=cfg.get("proj_buffer_ratio", 0.5),
    )
    print("> Cell generation completed!")

    if get_init_cell:
        return cell, init_cell_copy
    return cell


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

    verbose = int(config.get("verbose", 1))
    export_options = config.get("export", {})
    if not export_options:
        raise ValueError("export options are required in the config.")
    output_dir = export_options.get("output_dir")
    if not output_dir:
        raise ValueError(
            "output_dir is required in export options for exporting."
        )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        rm_output_dir = input(
            "Output directory already exists. Remove it? (y/n): "
        )
        if rm_output_dir.lower() == "y":
            shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    export_fmt = export_options.get("format", "png")

    
    if verbose > 10:
        print("Generating the Cell with configuration:")
        print(json.dumps(config, indent=4))

    num_cells = config.get("num_cells", 1)
    if not isinstance(num_cells, int) or num_cells < 1:
        raise ValueError(
            "num_cells should be an integer greater than or equal to 1."
        )

    for i in range(num_cells):
        config["rng_seed"] = config["rng_seed"] + i
        if 0 < verbose < 10:
            print(f"Generating cell {i} with seed {config['rng_seed']}")
        cell = _generate_cell(config)

        cell.save(
            f_path=output_dir.joinpath(f"cell_{i}.{export_fmt}"),
            plot_options=export_options.get("plot_with"),
        )
        # image_options = config.get("image", {})
        # if image_options:
        #     cell.plot(
        #         f_path=output_dir.joinpath("final.png"),
        #         shape_vis_options=image_options.get("shapes", {}),
        #         domain_vis_options=image_options.get("domain", {}),
        #         image_size=image_options.get("size", (256, 256)),
        #     )


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

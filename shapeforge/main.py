import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .cell import CellDomain, initialise_shapes
from .cell import Cell
from .utils import load_yaml, Event


def _generate_cell(
    cfg: dict, get_init_cell: bool = False, log_handle: Event = None
) -> Cell | tuple[Cell, Cell]:
    # --------------------------------------------------------- #
    #               Cell Initialization                         #
    # --------------------------------------------------------- #
    cell_domain = CellDomain.from_dict(cfg["domain"])
    if log_handle is not None:
        log_handle.log("Initializing the domain...")
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
    if log_handle is not None:
        log_handle.log("Shapes are initialised in the domain.")
    cell = Cell(cell_domain, shapes)
    init_cell_copy = cell.clone()

    # --------------------------------------------------------- #
    #               Cell Optimisation                           #
    # --------------------------------------------------------- #
    cell.remove_inclusion_overlaps(
        ssd_ratio=cfg.get("min_gap", 0.05),
        proj_buffer_ratio=cfg.get("proj_buffer_ratio", 0.5),
    )
    if log_handle is not None:
        log_handle.log("Inclusions overlaps are removed.")

    if get_init_cell:
        return cell, init_cell_copy
    return cell


def _load_input_file(config: Path):
    if isinstance(config, (str, Path)):
        config = load_yaml(config)
    if not isinstance(config, dict):
        raise ValueError("Expecting config to be a dictionary.")

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
        # rm_output_dir = input(
        #     "Output directory already exists. Remove it? (y/n): "
        # )  # TODO: add confirmation
        rm_output_dir = "y"
        if rm_output_dir.lower() == "y":
            shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    return config, verbose, output_dir, export_options


def _validate_num_cells(num_cells):
    if not isinstance(num_cells, int) or num_cells < 1:
        raise ValueError(
            "num_cells should be an integer greater than or equal to 1."
        )
    return num_cells


def generate_cell(config: dict | str | Path) -> Cell:
    """
    Generate a unit cell with the specified configuration.
    """
    Event.log("Starting the cell(s) genration...")

    with Event("Loading the input file..."):
        config, verbose, output_dir, export_options = _load_input_file(config)
    export_fmt = export_options.get("format", "png")

    if verbose > 10:
        Event.log("Generating the cell with configuration:")
        Event.log(json.dumps(config, indent=4))

    num_cells = _validate_num_cells(config.get("num_cells", 1))
    for i in range(num_cells):
        config["rng_seed"] = config["rng_seed"] + i

        with Event(f"Generating cell {i} with seed {config['rng_seed']}...") as lh:
            cell = _generate_cell(config, get_init_cell=False, log_handle=lh)

        with Event(f"Exporting cell {i}..."):
            cell.save(
                f_path=output_dir.joinpath(f"cell_{i}.{export_fmt}"),
                plot_options=export_options.get("plot_with"),
            )
    Event.log("Completed cell generation.")


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

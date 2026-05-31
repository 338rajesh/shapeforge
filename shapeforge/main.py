import argparse
import json
import shutil
from pathlib import Path

from .cell import Cell
from .utils import load_yaml, Event


# def _generate_cell(
#     cfg: dict, get_init_cell: bool = False, log_handle: Event = None
# ) -> Cell | tuple[Cell, Cell]:
#     cell_cfg = cfg["cell"]
#     # --------------------------------------------------------- #
#     #               Cell Initialization                         #
#     # --------------------------------------------------------- #
#     cell = Cell.initialise(
#         cell_cfg, cfg["rng_seed"], init_method=cfg["engine"]["init_method"]
#     )
#     init_cell_copy = cell.clone()

#     # --------------------------------------------------------- #
#     #               Cell Optimisation                           #
#     # --------------------------------------------------------- #
#     cell.remove_inclusion_overlaps(
#         ssd_ratio=cell_cfg.get("element_min_gap", 0.05),
#         proj_buffer_ratio=cfg.get("engine", {}).get("proj_buffer_ratio", 0.5),
#     )
#     if log_handle is not None:
#         log_handle.log("Inclusions overlaps are removed.")

#     if get_init_cell:
#         return cell, init_cell_copy
#     return cell


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
    export_options["output_dir"] = output_dir
    config["export"] = export_options
    config["verbose"] = verbose
    return config


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
        config = _load_input_file(config)

    if config["verbose"] > 10:
        Event.log("Generating the cell with configuration:")
        Event.log(json.dumps(config, indent=4))

    def get_file_path(idx: int):
        return config["export"]["output_dir"].joinpath(
            f"cell_{idx}.{config['export']['format']}"
        )

    num_cells = _validate_num_cells(config.get("num_cells", 1))
    for i in range(num_cells):
        config["rng_seed"] = config["rng_seed"] + i
        Event.log(f"Generating cell {i} with seed {config['rng_seed']}...")

        # f"Generating cell {i} with seed {config['rng_seed']}..."
        with Event("Initializing the cell..."):
            cell = Cell.initialise(
                config["cell"], config["rng_seed"], init_method="uniform"
            )

        with Event("Removing inclusions overlaps..."):
            cell.remove_inclusion_overlaps(
                ssd_ratio=config["cell"].get("element_min_gap", 0.05),
                # proj_buffer_ratio=config.get("engine", {}).get(
                #     "proj_buffer_ratio", 0.5
                # ),
            )

        with Event(f"Exporting cell {i}..."):
            cell.export(
                f_path=get_file_path(i), options=config["export"]["options"]
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

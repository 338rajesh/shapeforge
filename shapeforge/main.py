import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# import gbox as gb
import numpy as np
from gbox.shapes.shapes_2d import Shape2DPose

from .cell import Cell, CellDomain2D, initialise_shapes_2d
from .config import ShapeForgeConfig
from .overlap_optim import CellShapes2DOverlap
from .utils import _load_dict, get_logger, plot_f_and_g_norms

logger = get_logger(__name__)

SCRATCH_DIR = Path("/home/rajesh/workshop/repos/shapeforge/scratch")


def _generate_cell_2d_core(
    cfg: ShapeForgeConfig,
) -> Sequence[tuple[Cell, Cell]]:
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
        t0 = time.perf_counter()
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
            cell.domain, cell.shapes, buffer_thickness_ratio=0.05
        )

        solution = overlap_problem.solve(**solver_options)

        print()
        logger.info(
            f"[status: {solution.status}]"
            f"[iter count: {solution.iter_count}]"
            f"[Time: {(time.perf_counter() - t0):4.3f}sec]"
        )

        # Updating the shapes with the optimal positions
        positions = overlap_problem._as_matrix(solution.x_optimal)
        cell.shapes.set_positions(positions)

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


def generate_cell_2d(
    config: dict[str, Any] | str | Path,
    *,
    profile: bool = False,
    profile_output: str | Path | None = None,
    profile_lines: int = 25,
    profile_sort: str = "cumulative",
) -> Sequence[tuple[Cell, Cell]]:
    """
    Generate a unit cell with the specified configuration.

    Parameters
    ----------
    config : dict | str | Path
        Configuration mapping or path to YAML/JSON configuration file.
    profile : bool, optional
        Whether to profile the cell generation run (default: False).
    profile_output : str | Path | None, optional
        Destination file path for the .prof binary profile data. If None,
        defaults to '<export.output_dir>/shapeforge.prof'.
    profile_lines : int, optional
        Number of hotspot entries to display in the terminal summary (default: 25).
    profile_sort : str, optional
        Sort criterion for profiling stats: 'cumulative', 'time', or 'calls'
        (default: 'cumulative').
    """
    logger.info("Starting the cell 2D generation...")
    config_dict = _load_dict(config)
    cfg = ShapeForgeConfig.from_dict(config_dict)

    if profile:
        from .profiler import Profiler

        if profile_output is None:
            profile_output = Path(cfg.export.output_dir) / "shapeforge.prof"

        with Profiler(
            output_path=profile_output,
            top_n=profile_lines,
            sort_by=profile_sort,
            print_summary=True,
        ):
            return _generate_cell_2d_core(cfg)
    else:
        return _generate_cell_2d_core(cfg)


def build_parser() -> argparse.ArgumentParser:
    _log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    parser = argparse.ArgumentParser(
        prog="shapeforge",
        description="ShapeForge: RVE and Unit Cell Generation Engine",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for ShapeForge.",
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        choices=_log_levels + list(map(str.lower, _log_levels)),
        help="Set the logging level (default: WARNING)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile execution with cProfile to identify performance bottlenecks.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Destination path for .prof profiling output file (default: <output_dir>/shapeforge.prof).",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=25,
        help="Number of hotspot lines to show in the console summary (default: 25).",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        choices=["cumulative", "time", "calls"],
        default="cumulative",
        help="Sort metric for the profiling summary (default: cumulative).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logger.setLevel(args.log.upper())
    generate_cell_2d(
        args.config_file,
        profile=args.profile,
        profile_output=args.profile_output,
        profile_lines=args.profile_lines,
        profile_sort=args.profile_sort,
    )


if __name__ == "__main__":
    main()

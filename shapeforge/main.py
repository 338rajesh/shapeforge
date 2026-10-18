"""
Command-Line Interface (CLI) and high-level cell generation entry points.
========================================================================

Provides the CLI executable ``shapeforge`` and the primary Python API function
:func:`generate_cell_2d`. Coordinates configuration parsing, custom shape loading,
spatial domain construction, inclusion sampling, NM-SPG overlap resolution,
periodic boundary wrapping, image plotting, and file exports.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gbox as gb
import numpy as np

from .cell import Cell, CellDomain2D, initialise_shapes_2d
from .config import ShapeForgeConfig
from .overlap_optim import CellShapes2DOverlap
from .utils import _load_dict, get_logger, plot_f_and_g_norms

logger = get_logger(__name__)


def _generate_cell_2d_core(
    cfg: ShapeForgeConfig,
) -> Sequence[tuple[Cell, Cell]]:
    """
    Execute the core cell generation loop across all requested unit cells.

    Parameters
    ----------
    cfg : ShapeForgeConfig
        Fully validated configuration object.

    Returns
    -------
    Sequence of tuple of (Cell, Cell)
        List of 2-tuples ``(initial_cell, final_cell)`` for each generated unit cell.
    """
    out_dir = Path(cfg.export.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[Cell, Cell]] = [
        (None, None) for _ in range(cfg.num_cells)  # type: ignore[misc]
    ]

    solver_options: dict[str, Any] = {
        "method": "nmspg",
        "iter_max": 1000,
        "iter_memory": 10,
        "epsilon": 1e-6,
        "f_tol": 1e-6,
        "spectral_step_min": 1e-10,
        "spectral_step_max": 1e3,
        "gamma": 0.0001,
        "sigma1": 0.1,
        "sigma2": 0.9,
        "ls_iter_max": 20,
        "p_bar": True,
    }
    opt_user_cfg = dict(cfg.solver.optimiser)
    if "name" in opt_user_cfg:
        solver_options["method"] = opt_user_cfg.pop("name")
    if "max_iter" in opt_user_cfg:
        solver_options["iter_max"] = opt_user_cfg.pop("max_iter")
    solver_options.update(opt_user_cfg)

    plot_kwargs = {
        "size": cfg.export.image_dimensions,
        "background": cfg.export.background,
        "foreground": cfg.export.facecolor,
        "dpi": cfg.export.dpi,
    }

    img_formats = [
        fmt
        for fmt in cfg.export.formats
        if fmt.lower() in ("png", "jpg", "jpeg", "pdf")
    ]
    data_formats = [
        fmt
        for fmt in cfg.export.formats
        if fmt.lower() in ("json", "pkl", "pickle", "yaml", "yml", "npz")
    ]

    for index in range(cfg.num_cells):
        t0 = time.perf_counter()
        rng_seed = cfg.metadata.rng_seed + index
        rng = np.random.default_rng(seed=rng_seed)
        logger.info(f"Generating cell {index} with seed {rng_seed}")

        cell_domain = CellDomain2D(bounds=cfg.domain.bounds)
        logger.debug("  Domain of cell is created")

        shapes = initialise_shapes_2d(
            cell_domain,
            cfg.shapes,
            rng=rng,
            init_method=cfg.solver.init_method,
        )
        logger.debug(f"  Cell is initialised with {len(shapes)} shapes.")

        if cfg.packing.adjust_bounds_to_exact_vf:
            target_vf = sum(s.volume_fraction for s in cfg.shapes)
            if target_vf > 0.0:
                actual_area = shapes.area
                required_domain_area = actual_area / target_vf
                scale = np.sqrt(required_domain_area / cell_domain.area)
                x_min, x_max = cell_domain.x_bounds
                y_min, y_max = cell_domain.y_bounds
                new_x_max = x_min + (x_max - x_min) * scale
                new_y_max = y_min + (y_max - y_min) * scale
                cell_domain = CellDomain2D(
                    gb.Bounds2DRectangular(x_min, y_min, new_x_max, new_y_max)
                )
                logger.debug(
                    f"  Adjusted domain bounds to {cell_domain.domain} "
                    f"for exact volume fraction {target_vf:.4f}"
                )

        cell = Cell(cell_domain, shapes)
        logger.debug("  Cell object containing shapes and domain is created")

        initial_copy = cell.clone()
        if cfg.packing.periodicity:
            initial_export = initial_copy.make_periodic()
        else:
            initial_export = initial_copy

        # Render/export initial cell
        initial_arr = None
        for fmt in img_formats:
            initial_arr = initial_export.plot(
                f_path=out_dir / f"initial_{index}.{fmt}", **plot_kwargs
            )
        if initial_arr is None and cfg.export.as_array:
            initial_arr = initial_export.plot(f_path=None, **plot_kwargs)

        for fmt in data_formats:
            initial_export.save(
                f_path=out_dir / f"initial_{index}.{fmt}", overwrite=True
            )

        logger.debug("  A copy of the cell is created for comparison")

        # Solving the overlap
        overlap_problem = CellShapes2DOverlap(
            cell.domain,
            cell.shapes,
            periodicity=cfg.packing.periodicity,
            buffer_thickness_ratio=cfg.packing.min_gap_ratio,
            proj_buffer_ratio=(
                cfg.packing.proj_buffer_ratio
                if cfg.packing.proj_buffer_ratio is not None
                else 2.0
            ),
            rng=rng,
        )

        solution = overlap_problem.solve(**solver_options)

        print()
        final_f = solution.f_history[-1] if solution.f_history else float("nan")
        logger.info(
            f"[status: {solution.status}]"
            f"[f: {final_f:.4e}]"
            f"[iter count: {solution.iter_count}]"
            f"[Time: {(time.perf_counter() - t0):4.3f}sec]"
        )
        if solution.status == "failure" and solution.failure_message:
            logger.warning(f"  {solution.failure_message}")

        # Updating the shapes with the optimal positions
        positions = overlap_problem._as_matrix(solution.x_optimal)
        if cfg.packing.periodicity:
            x_min, x_max = cell.domain.x_bounds
            y_min, y_max = cell.domain.y_bounds
            Lx = x_max - x_min
            Ly = y_max - y_min
            positions[:, 0] = x_min + np.mod(positions[:, 0] - x_min, Lx)
            positions[:, 1] = y_min + np.mod(positions[:, 1] - y_min, Ly)
            positions[:, 2] = np.mod(positions[:, 2], 2.0 * np.pi)

        cell.shapes.set_positions(positions)

        if cfg.packing.periodicity:
            final_cell = cell.make_periodic()
        else:
            final_cell = cell

        final_arr = None
        for fmt in img_formats:
            final_arr = final_cell.plot(
                f_path=out_dir / f"final_{index}.{fmt}", **plot_kwargs
            )
        if final_arr is None and cfg.export.as_array:
            final_arr = final_cell.plot(f_path=None, **plot_kwargs)

        for fmt in data_formats:
            final_cell.save(f_path=out_dir / f"final_{index}.{fmt}", overwrite=True)

        if cfg.export.as_array:
            if initial_arr is not None:
                np.save(out_dir / f"initial_{index}.npy", initial_arr)
            if final_arr is not None:
                np.save(out_dir / f"final_{index}.npy", final_arr)

        plot_f_and_g_norms(
            sol=solution,
            f_path=out_dir / f"fg_variation_{index}.png",
        )
        cells[index] = (initial_export, final_cell)
    return cells


def generate_cell_2d(
    config: dict[str, Any] | str | Path,
    *,
    custom_shapes: Sequence[str | Path] | None = None,
    log_level: str | None = None,
    profile: bool = False,
    profile_output: str | Path | None = None,
    profile_lines: int = 25,
    profile_sort: str = "cumulative",
) -> Sequence[tuple[Cell, Cell]]:
    """
    Generate 2D unit cells from a dictionary or configuration file.

    This is the primary Python API entry point for ShapeForge. It loads the
    configuration, imports any declared custom shapes, initializes inclusion
    geometries, resolves spatial overlaps using the NM-SPG solver, enforces
    periodicity, exports plots/data, and optionally profiles the run.

    Parameters
    ----------
    config : dict or str or Path
        Configuration dictionary or filesystem path to a YAML/JSON configuration file.
    custom_shapes : Sequence of (str or Path), optional
        Additional paths to Python files defining custom shapes to register.
    log_level : str, optional
        Logging level override (e.g. ``'DEBUG'``, ``'INFO'``). If None, uses
        ``metadata.log_level`` from the configuration.
    profile : bool, optional
        Whether to profile the cell generation execution with cProfile (default: False).
    profile_output : str or Path, optional
        Destination path for the binary ``.prof`` file. If None, defaults to
        ``<export.output_dir>/shapeforge.prof``.
    profile_lines : int, optional
        Number of bottleneck functions to display in the terminal summary (default: 25).
    profile_sort : {'cumulative', 'time', 'calls'}, optional
        Sorting metric for profiling diagnostics (default: ``'cumulative'``).

    Returns
    -------
    Sequence of tuple of (Cell, Cell)
        List of 2-tuples ``(initial_cell, final_cell)`` corresponding to each
        generated unit cell.

    Examples
    --------
    >>> from shapeforge import generate_cell_2d
    >>> cells = generate_cell_2d("demo/custom_shape/config.yaml")
    >>> len(cells)
    2
    >>> init_0, final_0 = cells[0]
    >>> final_0.domain.area
    900.0
    """
    base_dir = (
        Path(config).parent.resolve()
        if isinstance(config, (str, Path))
        else None
    )

    if custom_shapes:
        from .registry import ShapeRegistry

        for cs in custom_shapes:
            cs_path = Path(cs)
            if not cs_path.is_absolute() and base_dir is not None:
                cs_path = (base_dir / cs_path).resolve()
            ShapeRegistry.load_file(cs_path)

    config_dict = _load_dict(config)
    cfg = ShapeForgeConfig.from_dict(config_dict, base_dir=base_dir)

    effective_log_level = (log_level or cfg.metadata.log_level).upper()
    logging.getLogger("shapeforge").setLevel(effective_log_level)
    logger.setLevel(effective_log_level)

    logger.info("Starting the cell 2D generation...")

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
    """
    Construct the command-line argument parser for the ShapeForge CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for CLI execution.
    """
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
        default=None,
        choices=_log_levels + list(map(str.lower, _log_levels)),
        help="Set the logging level (default: from config metadata.log_level, or WARNING)",
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
    parser.add_argument(
        "--custom-shapes",
        nargs="*",
        default=None,
        help="Optional paths to Python files defining custom shapes to register.",
    )
    return parser


def main() -> None:
    """
    Primary CLI command entry point for the ``shapeforge`` command.

    Parses command-line arguments and delegates to :func:`generate_cell_2d`.
    """
    parser = build_parser()
    args = parser.parse_args()
    generate_cell_2d(
        args.config_file,
        custom_shapes=args.custom_shapes,
        log_level=args.log,
        profile=args.profile,
        profile_output=args.profile_output,
        profile_lines=args.profile_lines,
        profile_sort=args.profile_sort,
    )

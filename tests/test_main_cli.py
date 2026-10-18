import sys

import pytest

from shapeforge.main import generate_cell_2d, main


def test_periodicity_supported(tmp_path):
    cfg_dict = {
        "num_cells": 1,
        "metadata": {
            "title": "periodic-test",
            "rng_seed": 42,
            "log_level": "info",
        },
        "domain": {
            "shape": "rectangle",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20},
        },
        "shapes": [
            {
                "name": "Circle",
                "volume_fraction": 0.1,
                "params": {"radius": 1.0},
            }
        ],
        "packing": {
            "periodicity": True,
            "min_gap_ratio": 0.05,
            "proj_buffer_ratio": 2.0,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {"init_method": "uniform", "optimiser": {"name": "nmspg"}},
        "export": {
            "output_dir": str(tmp_path),
            "as_array": False,
            "formats": ["json"],
            "background": 0,
            "facecolor": 255,
            "image_dimensions": [64, 64],
            "dpi": 72,
        },
    }
    cells = generate_cell_2d(cfg_dict)
    assert len(cells) == 1
    init_cell, final_cell = cells[0]
    assert len(final_cell.shapes) >= len(final_cell.primary_shapes)
    assert (tmp_path / "final_0.json").exists()


def test_cli_main_invocation(tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "test_config.yaml"
    out_dir = tmp_path / "out"
    cfg_dict = {
        "num_cells": 1,
        "metadata": {"title": "cli-test", "rng_seed": 10, "log_level": "info"},
        "domain": {
            "shape": "rectangle",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 15, "y_max": 15},
        },
        "shapes": [
            {
                "name": "Circle",
                "volume_fraction": 0.08,
                "params": {"radius": 1.0},
            }
        ],
        "packing": {
            "periodicity": False,
            "min_gap_ratio": 0.05,
            "proj_buffer_ratio": 2.0,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {
            "init_method": "uniform",
            "optimiser": {"name": "nmspg", "max_iter": 30},
        },
        "export": {
            "output_dir": str(out_dir),
            "as_array": False,
            "formats": ["json"],
            "background": 0,
            "facecolor": 255,
            "image_dimensions": [64, 64],
            "dpi": 72,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    prof_file = tmp_path / "run.prof"
    test_args = [
        "shapeforge",
        str(cfg_path),
        "--log",
        "debug",
        "--profile",
        "--profile-output",
        str(prof_file),
        "--profile-lines",
        "10",
        "--profile-sort",
        "time",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Invoke main()
    main()

    assert out_dir.exists()
    assert (out_dir / "final_0.json").exists()
    assert prof_file.exists()

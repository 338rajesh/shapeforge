import numpy as np
import pytest
from PIL import Image

from shapeforge.cell import Cell, CellDomain2D
from shapeforge.main import build_parser, generate_cell_2d


@pytest.fixture
def base_config_dict(tmp_path):
    out_dir = tmp_path / "custom_output"
    return {
        "num_cells": 1,
        "metadata": {
            "title": "test-respect-config",
            "rng_seed": 42,
            "log_level": "info",
        },
        "domain": {
            "shape": "rectangle",
            "bounds": {
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 20.0,
                "y_max": 20.0,
            },
        },
        "shapes": [
            {
                "name": "Circle",
                "volume_fraction": 0.10,
                "params": {"radius": 1.0},
            },
        ],
        "packing": {
            "periodicity": False,
            "min_gap_ratio": 0.08,
            "proj_buffer_ratio": 1.5,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {
            "init_method": "uniform",
            "optimiser": {
                "name": "nmspg",
                "max_iter": 50,
                "epsilon": 1e-4,
            },
        },
        "export": {
            "output_dir": str(out_dir),
            "as_array": False,
            "formats": ["png", "json"],
            "background": 0,
            "facecolor": 255,
            "image_dimensions": [256, 256],
            "dpi": 72,
        },
    }


def test_output_dir_and_image_dimensions_respected(base_config_dict, tmp_path):
    out_dir = tmp_path / "custom_output"
    base_config_dict["export"]["image_dimensions"] = [300, 300]
    base_config_dict["export"]["output_dir"] = str(out_dir)

    generate_cell_2d(base_config_dict)

    assert out_dir.exists()
    initial_png = out_dir / "initial_0.png"
    final_png = out_dir / "final_0.png"
    final_json = out_dir / "final_0.json"
    fg_plot = out_dir / "fg_variation_0.png"

    assert initial_png.exists()
    assert final_png.exists()
    assert final_json.exists()
    assert fg_plot.exists()

    # Verify image dimensions
    with Image.open(initial_png) as img:
        assert img.size == (300, 300)

    with Image.open(final_png) as img:
        assert img.size == (300, 300)

    # Verify cell loading from exported json
    cell = Cell.load(final_json)
    assert isinstance(cell.domain, CellDomain2D)
    assert len(cell.shapes) > 0


def test_as_array_export_respected(base_config_dict, tmp_path):
    out_dir = tmp_path / "array_output"
    base_config_dict["export"]["output_dir"] = str(out_dir)
    base_config_dict["export"]["as_array"] = True
    base_config_dict["export"]["image_dimensions"] = [128, 128]

    generate_cell_2d(base_config_dict)

    initial_npy = out_dir / "initial_0.npy"
    final_npy = out_dir / "final_0.npy"

    assert initial_npy.exists()
    assert final_npy.exists()

    arr = np.load(final_npy)
    assert arr.shape == (128, 128)
    assert arr.dtype == np.uint8


def test_multiple_formats_export_and_load(base_config_dict, tmp_path):
    out_dir = tmp_path / "formats_output"
    base_config_dict["export"]["output_dir"] = str(out_dir)
    base_config_dict["export"]["formats"] = [
        "png",
        "pdf",
        "json",
        "yaml",
        "pkl",
        "npz",
    ]

    generate_cell_2d(base_config_dict)

    for ext in ["png", "pdf", "json", "yaml", "pkl", "npz"]:
        final_file = out_dir / f"final_0.{ext}"
        assert final_file.exists(), f"File {final_file} was not exported!"

    # Verify round-trip loading for all data formats
    for ext in ["json", "yaml", "pkl", "npz"]:
        loaded_cell = Cell.load(out_dir / f"final_0.{ext}")
        assert isinstance(loaded_cell, Cell)
        assert len(loaded_cell.shapes) == len(
            Cell.load(out_dir / "final_0.json").shapes
        )


def test_adjust_bounds_to_exact_vf(base_config_dict, tmp_path):
    base_config_dict["packing"]["adjust_bounds_to_exact_vf"] = True
    base_config_dict["shapes"] = [
        {
            "name": "Circle",
            "volume_fraction": 0.20,
            "params": {"radius": 1.0},
        }
    ]
    cells = generate_cell_2d(base_config_dict)
    _, final_cell = cells[0]

    actual_vf = final_cell.shapes.area / final_cell.domain.area
    np.testing.assert_allclose(actual_vf, 0.20, rtol=1e-5)


def test_init_methods(base_config_dict):
    for method in ["uniform", "lhs", "sobol"]:
        base_config_dict["solver"]["init_method"] = method
        cells = generate_cell_2d(base_config_dict)
        assert len(cells) == 1
        assert len(cells[0][1].shapes) > 0


def test_optimiser_options_respected(base_config_dict, monkeypatch):
    captured_options = {}

    from shapeforge.overlap_optim import CellShapes2DOverlap

    original_solve = CellShapes2DOverlap.solve

    def mock_solve(self, **kwargs):
        captured_options.update(kwargs)
        return original_solve(self, **kwargs)

    monkeypatch.setattr(CellShapes2DOverlap, "solve", mock_solve)

    base_config_dict["solver"]["optimiser"] = {
        "name": "nmspg",
        "max_iter": 42,
        "epsilon": 2.5e-5,
        "iter_memory": 7,
    }
    base_config_dict["packing"]["min_gap_ratio"] = 0.12
    base_config_dict["packing"]["proj_buffer_ratio"] = 3.5

    generate_cell_2d(base_config_dict)

    assert captured_options["iter_max"] == 42
    assert captured_options["epsilon"] == 2.5e-5
    assert captured_options["iter_memory"] == 7


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["demo/config.yaml"])
    assert args.log is None
    assert args.profile is False
    assert args.profile_output is None

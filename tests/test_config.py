import pytest

from shapeforge.config import (
    DomainConfig,
    ExportConfig,
    MetadataConfig,
    PackingConfig,
    ShapeConfig,
    ShapeForgeConfig,
    SolverConfig,
)
from shapeforge.utils import DistributionSpec


def test_metadata_config():
    valid = {"title": "run1", "rng_seed": 123, "log_level": "info"}
    meta = MetadataConfig.from_dict(valid)
    assert meta.title == "run1"
    assert meta.rng_seed == 123
    assert meta.log_level == "info"

    with pytest.raises(Exception):
        MetadataConfig.from_dict(
            {"title": "run1", "rng_seed": 123, "log_level": "info", "extra": 1}
        )


def test_domain_config():
    valid = {
        "shape": "rectangle",
        "bounds": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 50},
    }
    dom = DomainConfig.from_dict(valid)
    assert dom.shape == "rectangle"
    assert dom.bounds.x_min == 0.0
    assert dom.bounds.x_max == 100.0


def test_shape_config_valid():
    valid = {
        "name": "Circle",
        "volume_fraction": 0.25,
        "params": {"radius": 2.5},
    }
    shape = ShapeConfig.from_dict(valid)
    assert shape.name == "circle"
    assert shape.volume_fraction == 0.25
    assert shape.params["radius"] == 2.5

    # With distribution string
    valid_dist = {
        "name": "Ellipse",
        "volume_fraction": 0.15,
        "params": {
            "semi_major_length": "uniform(1.0, 0.5)",
            "semi_minor_length": 1.0,
        },
    }
    shape_dist = ShapeConfig.from_dict(valid_dist)
    assert isinstance(shape_dist.params["semi_major_length"], DistributionSpec)


def test_shape_config_invalid():
    # Volume fraction > 1.0
    with pytest.raises(ValueError):
        ShapeConfig.from_dict(
            {
                "name": "Circle",
                "volume_fraction": 1.5,
                "params": {"radius": 2.5},
            }
        )

    # Unknown shape name
    with pytest.raises(ValueError):
        ShapeConfig.from_dict(
            {
                "name": "SuperHexagon",
                "volume_fraction": 0.2,
                "params": {"radius": 2.5},
            }
        )

    # Bool param rejected
    with pytest.raises(TypeError):
        ShapeConfig.from_dict(
            {
                "name": "Circle",
                "volume_fraction": 0.2,
                "params": {"radius": True},
            }
        )

    # Invalid param type
    with pytest.raises(TypeError):
        ShapeConfig.from_dict(
            {
                "name": "Circle",
                "volume_fraction": 0.2,
                "params": {"radius": [1.0, 2.0]},
            }
        )


def test_packing_config():
    valid = {
        "periodicity": False,
        "min_gap_ratio": 0.05,
        "proj_buffer_ratio": 2.0,
        "adjust_bounds_to_exact_vf": True,
    }
    p = PackingConfig.from_dict(valid)
    assert p.min_gap_ratio == 0.05
    assert p.proj_buffer_ratio == 2.0
    assert p.adjust_bounds_to_exact_vf is True

    # Negative min_gap_ratio
    with pytest.raises(Exception):
        PackingConfig.from_dict(
            {
                "periodicity": False,
                "min_gap_ratio": -0.05,
                "proj_buffer_ratio": 2.0,
                "adjust_bounds_to_exact_vf": False,
            }
        )


def test_solver_config():
    valid = {
        "init_method": "lhs",
        "optimiser": {"name": "nmspg", "max_iter": 500},
    }
    s = SolverConfig.from_dict(valid)
    assert s.init_method == "lhs"
    assert s.optimiser["max_iter"] == 500

    # Invalid init_method
    with pytest.raises(Exception):
        SolverConfig.from_dict(
            {
                "init_method": "random_walk",
                "optimiser": {"name": "nmspg"},
            }
        )


def test_export_config(tmp_path):
    out = tmp_path / "exp"
    valid = {
        "output_dir": str(out),
        "as_array": True,
        "formats": ["png", "yaml", "npz"],
        "background": 0,
        "facecolor": 255,
        "image_dimensions": [256, 256],
        "dpi": 100,
    }
    exp = ExportConfig.from_dict(valid)
    assert exp.output_dir == out
    assert exp.as_array is True
    assert "png" in exp.formats

    # Invalid export format
    with pytest.raises(ValueError):
        ExportConfig.from_dict(
            {
                **valid,
                "formats": ["unsupported_format_xyz"],
            }
        )

    # Invalid background color
    with pytest.raises(Exception):
        ExportConfig.from_dict(
            {
                **valid,
                "background": 300,
            }
        )

    # Invalid image dimensions
    with pytest.raises(Exception):
        ExportConfig.from_dict(
            {
                **valid,
                "image_dimensions": [0, -10],
            }
        )


def test_shapeforge_config_complete(tmp_path):
    cfg_dict = {
        "num_cells": 2,
        "metadata": {"title": "full", "rng_seed": 1, "log_level": "debug"},
        "domain": {
            "shape": "rectangle",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
        },
        "shapes": [
            {
                "name": "Circle",
                "volume_fraction": 0.2,
                "params": {"radius": 1.0},
            },
            {
                "name": "Ellipse",
                "volume_fraction": 0.3,
                "params": {"semi_major_length": 2.0, "semi_minor_length": 1.0},
            },
        ],
        "packing": {
            "periodicity": False,
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
            "image_dimensions": [100, 100],
            "dpi": 72,
        },
    }
    cfg = ShapeForgeConfig.from_(cfg_dict)
    assert cfg.num_cells == 2
    assert len(cfg.shapes) == 2

    # Cumulative VF > 1.0
    cfg_dict["shapes"][0]["volume_fraction"] = 0.8
    with pytest.raises(
        ValueError, match="Total shape volume fraction must not exceed 1.0"
    ):
        ShapeForgeConfig.from_dict(cfg_dict)

    # Invalid source type
    with pytest.raises(TypeError):
        ShapeForgeConfig.from_(12345)

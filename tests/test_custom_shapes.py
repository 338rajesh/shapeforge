from __future__ import annotations

from pathlib import Path

import gbox as gb
import numpy as np
import pytest
import yaml

from shapeforge.cell import (
    Cell,
    CellDomain2D,
    Inclusions2D,
    initialise_shapes_2d,
)
from shapeforge.config import ShapeConfig, ShapeForgeConfig
from shapeforge.main import generate_cell_2d
from shapeforge.overlap_optim import CellShapes2DOverlap
from shapeforge.plotter import CellPlotter
from shapeforge.registry import ShapeRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "custom_shapes.py"


@pytest.fixture(autouse=True)
def reset_registry():
    ShapeRegistry.reset()
    yield
    ShapeRegistry.reset()


def test_registry_builtins():
    assert ShapeRegistry.has("circle")
    assert ShapeRegistry.has("ellipse")
    assert ShapeRegistry.get("circle") is gb.Circle
    assert ShapeRegistry.get("ellipse") is gb.Ellipse
    assert "circle" in ShapeRegistry.available_shapes()
    assert "ellipse" in ShapeRegistry.available_shapes()


def test_registry_load_file():
    loaded = ShapeRegistry.load_file(FIXTURE_PATH)
    assert "stadium" in loaded
    assert "customsquare" in loaded
    assert ShapeRegistry.has("stadium")
    assert ShapeRegistry.has("customsquare")

    stadium_cls = ShapeRegistry.get("stadium")
    assert stadium_cls.__name__ == "Stadium"


def test_registry_manual_register():
    class DummyShape(gb.Shape2D):
        @classmethod
        def from_params(cls, p, s):
            pass

    ShapeRegistry.register("dummy", DummyShape)
    assert ShapeRegistry.has("dummy")
    assert ShapeRegistry.get("dummy") is DummyShape

    with pytest.raises(TypeError, match="subclass of gbox.Shape2D"):
        ShapeRegistry.register("not_shape", dict)

    class InvalidShape(gb.Shape2D):
        pass

    with pytest.raises(ValueError, match="from_params"):
        ShapeRegistry.register("invalid", InvalidShape)


def test_config_loading_with_custom_shapes(tmp_path):
    config_dict = {
        "custom_shapes": [str(FIXTURE_PATH)],
        "num_cells": 1,
        "metadata": {
            "title": "Custom Shape Test",
            "rng_seed": 42,
            "log_level": "WARNING",
        },
        "domain": {
            "shape": "box",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20},
        },
        "shapes": [
            {
                "name": "stadium",
                "volume_fraction": 0.10,
                "params": {"length": 2.0, "radius": 0.5},
            }
        ],
        "packing": {
            "min_gap_ratio": 0.05,
            "periodicity": True,
            "proj_buffer_ratio": 2.0,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {
            "init_method": "uniform",
            "optimiser": {"name": "nmspg", "max_iter": 50},
        },
        "export": {
            "as_array": False,
            "formats": ["json"],
            "output_dir": str(tmp_path / "out"),
            "image_dimensions": [100, 100],
            "background": 0,
            "facecolor": 255,
            "dpi": 72,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

    cfg = ShapeForgeConfig.from_(cfg_file)
    assert ShapeRegistry.has("stadium")
    assert cfg.shapes[0].name == "stadium"
    assert cfg.custom_shapes == (str(FIXTURE_PATH),)


def test_initialise_shapes_2d_with_custom_shape():
    ShapeRegistry.load_file(FIXTURE_PATH)
    domain = CellDomain2D([0, 0, 50, 50])
    rng = np.random.default_rng(42)

    shape_cfg = ShapeConfig(
        name="stadium",
        volume_fraction=0.15,
        params={"length": 3.0, "radius": 0.8},
    )
    inclusions = initialise_shapes_2d(domain, [shape_cfg], rng=rng)
    assert len(inclusions) > 0
    stadium_cls = ShapeRegistry.get("stadium")
    for inc in inclusions:
        assert isinstance(inc, stadium_cls)

    total_area = sum(inc.area for inc in inclusions)
    expected_area = domain.area * 0.15
    assert total_area >= expected_area


def test_overlap_optimization_with_custom_shapes():
    ShapeRegistry.load_file(FIXTURE_PATH)
    domain = CellDomain2D([0, 0, 30, 30])
    rng = np.random.default_rng(123)

    shape_cfg = ShapeConfig(
        name="stadium",
        volume_fraction=0.10,
        params={"length": 2.0, "radius": 0.5},
    )
    inclusions = initialise_shapes_2d(domain, [shape_cfg], rng=rng)

    prob = CellShapes2DOverlap(domain, inclusions, periodicity=True, rng=rng)
    f0, g0 = prob._evaluate(prob.x0)

    # Simple step along negative gradient
    step = 0.01 * g0 / (np.linalg.norm(g0) + 1e-12)
    x1 = prob.projection(prob.x0 - step)
    f1, _ = prob._evaluate(x1)
    # Evaluator is functioning
    assert isinstance(f0, float)
    assert isinstance(f1, float)


def test_periodicity_with_custom_shapes():
    ShapeRegistry.load_file(FIXTURE_PATH)
    domain = CellDomain2D([0, 0, 20, 20])
    stadium_cls = ShapeRegistry.get("stadium")

    # Place a stadium crossing the left boundary (x = 0)
    s = stadium_cls(
        length=4.0,
        radius=1.0,
        centre=(0.5, 10.0),
        orientation=gb.Angle.rad(0.0),
    )
    inclusions = Inclusions2D([s])
    cell = Cell(domain, inclusions)
    periodic_cell = cell.make_periodic()

    # Should have primary + 1 periodic copy shifted by +Lx (20)
    assert len(periodic_cell.shapes) == 2
    assert len(periodic_cell.periodic_shapes) == 1
    copy = next(iter(periodic_cell.periodic_shapes))
    assert np.isclose(copy.position.x, 20.5)


def test_plot_custom_shape_boundary_points(tmp_path):
    ShapeRegistry.load_file(FIXTURE_PATH)
    stadium_cls = ShapeRegistry.get("stadium")
    s = stadium_cls(
        length=10.0,
        radius=2.0,
        centre=(25.0, 25.0),
        orientation=gb.Angle.deg(45),
    )
    domain = CellDomain2D([0, 0, 50, 50])
    cell = Cell(domain, Inclusions2D([s]))

    img_path = tmp_path / "stadium.png"
    arr = cell.plot(f_path=img_path, size=(100, 100))
    assert img_path.exists()
    assert arr.shape == (100, 100)
    # Ensure something was drawn (image is not blank)
    assert np.any(arr != 255)


def test_plot_custom_shape_to_patch(tmp_path):
    ShapeRegistry.load_file(FIXTURE_PATH)
    square_cls = ShapeRegistry.get("customsquare")
    sq = square_cls(
        side=10.0,
        centre=(25.0, 25.0),
        orientation=gb.Angle.deg(30),
    )
    domain = CellDomain2D([0, 0, 50, 50])
    cell = Cell(domain, Inclusions2D([sq]))

    img_path = tmp_path / "square.png"
    arr = cell.plot(f_path=img_path, size=(100, 100))
    assert img_path.exists()
    assert arr.shape == (100, 100)
    assert np.any(arr != 255)


def test_plot_missing_methods_raises():
    class IncompleteShape:
        pass

    plotter = CellPlotter(size=(100, 100))
    with pytest.raises(TypeError, match="to_patch"):
        plotter.plot([IncompleteShape()], bounds=(0, 0, 50, 50))


def test_serialization_roundtrip(tmp_path):
    ShapeRegistry.load_file(FIXTURE_PATH)
    stadium_cls = ShapeRegistry.get("stadium")
    s = stadium_cls(
        length=5.0,
        radius=1.5,
        centre=(10.0, 15.0),
        orientation=gb.Angle.rad(0.5),
    )
    domain = CellDomain2D([0, 0, 30, 30])
    cell = Cell(domain, Inclusions2D([s]))

    # JSON roundtrip
    json_path = tmp_path / "cell.json"
    cell.save(json_path)
    loaded_cell = Cell.load(json_path)
    assert len(loaded_cell.shapes) == 1
    loaded_s = next(iter(loaded_cell.shapes))
    assert isinstance(loaded_s, stadium_cls)
    assert np.isclose(loaded_s.length, 5.0)
    assert np.isclose(loaded_s.radius, 1.5)
    assert np.isclose(loaded_s.position.x, 10.0)

    # NPZ roundtrip
    npz_path = tmp_path / "cell.npz"
    cell.save(npz_path)
    loaded_npz = Cell.load(npz_path)
    assert len(loaded_npz.shapes) == 1
    assert isinstance(next(iter(loaded_npz.shapes)), stadium_cls)


def test_generate_cell_2d_e2e(tmp_path):
    config_dict = {
        "custom_shapes": [str(FIXTURE_PATH)],
        "num_cells": 1,
        "metadata": {
            "title": "E2E Custom Shape Test",
            "rng_seed": 99,
            "log_level": "WARNING",
        },
        "domain": {
            "shape": "box",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 25, "y_max": 25},
        },
        "shapes": [
            {
                "name": "stadium",
                "volume_fraction": 0.12,
                "params": {"length": 2.5, "radius": 0.6},
            }
        ],
        "packing": {
            "min_gap_ratio": 0.05,
            "periodicity": True,
            "proj_buffer_ratio": 2.0,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {
            "init_method": "uniform",
            "optimiser": {"name": "nmspg", "max_iter": 20},
        },
        "export": {
            "as_array": False,
            "formats": ["png", "json"],
            "output_dir": str(tmp_path / "e2e_out"),
            "image_dimensions": [100, 100],
            "background": 0,
            "facecolor": 255,
            "dpi": 72,
        },
    }
    cfg_file = tmp_path / "e2e_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

    results = generate_cell_2d(cfg_file)
    assert len(results) == 1
    _, final_cell = results[0]
    assert len(final_cell.shapes) > 0
    assert (tmp_path / "e2e_out" / "initial_0.png").exists()
    assert (tmp_path / "e2e_out" / "initial_0.json").exists()
    assert (tmp_path / "e2e_out" / "final_0.json").exists()

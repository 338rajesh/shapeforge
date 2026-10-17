import gbox as gb
import numpy as np
import pytest
from shapeforge.cell import CellDomain2D, initialise_shapes_2d
from shapeforge.config import ShapeConfig
from shapeforge.overlap_optim import CellShapes2DOverlap


@pytest.fixture
def sample_domain():
    return CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 50.0, 50.0))


def test_initialise_circle_shapes(sample_domain):
    rng = np.random.default_rng(42)
    shapes_cfg = [
        ShapeConfig(name="circle", volume_fraction=0.10, params={"radius": 1.0})
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    assert all(isinstance(s, gb.Circle) for s in inclusions)
    # Area should meet or slightly exceed the requested volume fraction
    expected_area = sample_domain.area * 0.10
    assert inclusions.area >= expected_area
    assert inclusions.area - expected_area < (np.pi * 1.0**2)


def test_initialise_ellipse_default_orientation(sample_domain):
    rng = np.random.default_rng(42)
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.10,
            params={"semi_major_length": 2.0, "semi_minor_length": 1.0},
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    assert all(isinstance(s, gb.Ellipse) for s in inclusions)

    # Check orientations are sampled and distributed within [0, 2*pi)
    orientations = [s.position.orientation.radians for s in inclusions]
    assert all(0.0 <= ori <= 2.0 * np.pi for ori in orientations)
    # They should not all be identical
    assert len(set(np.round(orientations, 4))) > 1


def test_initialise_ellipse_custom_orientation(sample_domain):
    rng = np.random.default_rng(42)
    # Fixed angle orientation
    fixed_angle = 0.785398  # pi/4
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={
                "semi_major_length": 2.5,
                "semi_minor_length": 1.0,
                "orientation": fixed_angle,
            },
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    for s in inclusions:
        assert np.isclose(s.position.orientation.radians, fixed_angle, atol=1e-5)


def test_initialise_ellipse_distribution_orientation(sample_domain):
    rng = np.random.default_rng(42)
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={
                "semi_major_length": 2.0,
                "semi_minor_length": 1.0,
                "major_axis_angle": "uniform(0.1, 0.2)",
            },
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    for s in inclusions:
        rad = s.position.orientation.radians
        assert 0.1 <= rad <= 0.35


def test_initialise_ellipse_aliases(sample_domain):
    rng = np.random.default_rng(42)
    # Using 'semi_major_axis' and 'semi_minor_axis'
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={"semi_major_axis": 3.0, "semi_minor_axis": 1.5},
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    for s in inclusions:
        assert isinstance(s, gb.Ellipse)
        assert np.isclose(s.semi_major_length, 3.0)
        assert np.isclose(s.semi_minor_length, 1.5)


def test_initialise_ellipse_aspect_ratio(sample_domain):
    rng = np.random.default_rng(42)
    # Using 'semi_major_length' and 'aspect_ratio'
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={"semi_major_length": 4.0, "aspect_ratio": 2.0},
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    assert len(inclusions) > 0
    for s in inclusions:
        assert np.isclose(s.semi_major_length, 4.0)
        assert np.isclose(s.semi_minor_length, 2.0)


def test_initialise_mixed_shapes(sample_domain):
    rng = np.random.default_rng(42)
    shapes_cfg = [
        ShapeConfig(name="circle", volume_fraction=0.05, params={"radius": 1.0}),
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={"semi_major_length": 2.0, "semi_minor_length": 1.0},
        ),
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    circles = [s for s in inclusions if type(s) is gb.Circle]
    ellipses = [s for s in inclusions if type(s) is gb.Ellipse]

    assert len(circles) > 0
    assert len(ellipses) > 0
    assert len(circles) + len(ellipses) == len(inclusions)


def test_ellipse_overlap_optimization(sample_domain):
    rng = np.random.default_rng(42)
    shapes_cfg = [
        ShapeConfig(
            name="ellipse",
            volume_fraction=0.05,
            params={"semi_major_length": 2.0, "semi_minor_length": 1.0},
        )
    ]
    inclusions = initialise_shapes_2d(sample_domain, shapes_cfg, rng)
    problem = CellShapes2DOverlap(sample_domain, inclusions)
    sol = problem.solve(method="nmspg", iter_max=100)
    assert sol.status == "success"
    # Ensure final overlap cost is resolved or decreased
    assert sol.f_history[-1] <= sol.f_history[0]

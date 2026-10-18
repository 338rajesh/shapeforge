import json
from pathlib import Path

import gbox as gb
import numpy as np
import pytest

from shapeforge.cell import (
    Cell,
    CellDomain2D,
    Inclusions2D,
    circle_intersects_rect,
    generate_periodic_copies,
    shape_intersects_bounds,
)
from shapeforge.main import generate_cell_2d
from shapeforge.overlap_optim import CellShapes2DOverlap


# ============================================================================
# 1. Geometric Intersection Tests
# ============================================================================


def test_circle_intersects_rect():
    # Domain: [0, 10] x [0, 10]
    # Center inside
    assert circle_intersects_rect(5.0, 5.0, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Center crossing left edge (cx = -0.5, r = 1.0 -> reaches x = 0.5)
    assert circle_intersects_rect(-0.5, 5.0, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Center crossing right edge (cx = 10.5, r = 1.0 -> reaches x = 9.5)
    assert circle_intersects_rect(10.5, 5.0, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Center crossing bottom edge (cy = -0.3, r = 1.0)
    assert circle_intersects_rect(5.0, -0.3, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Center crossing top edge (cy = 10.3, r = 1.0)
    assert circle_intersects_rect(5.0, 10.3, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Corner crossing: (cx, cy) = (-0.5, -0.5), dist to (0, 0) is sqrt(0.5) ~ 0.707 < 1.0
    assert circle_intersects_rect(-0.5, -0.5, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Corner miss: (cx, cy) = (-0.8, -0.8), dist to (0, 0) is sqrt(1.28) ~ 1.131 > 1.0
    assert not circle_intersects_rect(-0.8, -0.8, 1.0, 0.0, 0.0, 10.0, 10.0)

    # Completely far away
    assert not circle_intersects_rect(20.0, 20.0, 1.0, 0.0, 0.0, 10.0, 10.0)


def test_shape_intersects_bounds():
    bounds = gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0)

    # Circle inside
    c_inside = gb.Circle(radius=1.0, centre=[5.0, 5.0])
    assert shape_intersects_bounds(c_inside, bounds)

    # Circle crossing edge
    c_edge = gb.Circle(radius=1.0, centre=[10.5, 5.0])
    assert shape_intersects_bounds(c_edge, bounds)

    # Circle completely outside
    c_outside = gb.Circle(radius=1.0, centre=[15.0, 5.0])
    assert not shape_intersects_bounds(c_outside, bounds)

    # Ellipse crossing boundary
    e_edge = gb.Ellipse(
        semi_major_length=2.0,
        semi_minor_length=1.0,
        centre=[0.2, 5.0],
        major_axis_angle=gb.Angle.rad(0.5),
    )
    assert shape_intersects_bounds(e_edge, bounds)


# ============================================================================
# 2. Periodic Copies Generation Tests
# ============================================================================


def test_periodic_copies_edge_and_corner():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))

    # Shape 1: Interior circle -> 0 copies
    c_interior = gb.Circle(radius=1.0, centre=[5.0, 5.0])

    # Shape 2: Left edge circle -> 1 copy on right (x + 10)
    c_left = gb.Circle(radius=1.0, centre=[0.2, 5.0])

    # Shape 3: Right edge circle -> 1 copy on left (x - 10)
    c_right = gb.Circle(radius=1.0, centre=[9.8, 5.0])

    # Shape 4: Corner circle at (0.2, 0.2) -> 3 copies: (10.2, 0.2), (0.2, 10.2), (10.2, 10.2)
    c_corner = gb.Circle(radius=1.0, centre=[0.2, 0.2])

    incls = Inclusions2D([c_interior, c_left, c_right, c_corner])
    copies = generate_periodic_copies(incls, domain)

    # 0 + 1 + 1 + 3 = 5 copies total
    assert len(copies) == 5
    for cp in copies:
        assert getattr(cp, "is_periodic_copy", False) is True

    # Test full make_periodic
    periodic_incls = incls.make_periodic(domain)
    assert len(periodic_incls) == 4 + 5  # 4 primary + 5 periodic copies
    assert len(periodic_incls.primary_inclusions) == 4
    assert len(periodic_incls.periodic_copies) == 5

    # Idempotence: calling make_periodic again shouldn't duplicate copies
    periodic_twice = periodic_incls.make_periodic(domain)
    assert len(periodic_twice) == 9
    assert len(periodic_twice.primary_inclusions) == 4
    assert len(periodic_twice.periodic_copies) == 5


def test_periodic_copies_non_zero_origin():
    # Domain [-20, 20] x [-20, 20], Lx = 40, Ly = 40
    domain = CellDomain2D(gb.Bounds2DRectangular(-20.0, -20.0, 20.0, 20.0))

    # Circle on bottom-left corner at (-19.5, -19.5) with r = 1.0
    c_bl = gb.Circle(radius=1.0, centre=[-19.5, -19.5])
    incls = Inclusions2D([c_bl])

    periodic = incls.make_periodic(domain)
    # Primary + 3 copies
    assert len(periodic) == 4
    positions = periodic.get_positions()
    xs = positions[:, 0]
    ys = positions[:, 1]

    # Primary is at (-19.5, -19.5)
    # Copies at (20.5, -19.5), (-19.5, 20.5), (20.5, 20.5)
    assert np.any(np.isclose(xs, 20.5) & np.isclose(ys, -19.5))
    assert np.any(np.isclose(xs, -19.5) & np.isclose(ys, 20.5))
    assert np.any(np.isclose(xs, 20.5) & np.isclose(ys, 20.5))


def test_periodic_copies_with_ellipse():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 20.0, 20.0))
    angle = gb.Angle.deg(45)
    e = gb.Ellipse(
        semi_major_length=2.0,
        semi_minor_length=1.0,
        centre=[0.3, 0.3],
        major_axis_angle=angle,
    )
    incls = Inclusions2D([e])
    periodic = incls.make_periodic(domain)

    # 1 primary + 3 corner copies
    assert len(periodic) == 4
    for shape in periodic:
        assert isinstance(shape, gb.Ellipse)
        assert np.isclose(shape.semi_major_length, 2.0)
        assert np.isclose(shape.semi_minor_length, 1.0)
        # Orientation preserved
        assert np.isclose(
            shape.position.orientation.radians, angle.radians
        )


def test_cell_make_periodic_properties():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))
    c1 = gb.Circle(radius=1.0, centre=[0.2, 5.0])
    c2 = gb.Circle(radius=1.0, centre=[5.0, 5.0])
    incls = Inclusions2D([c1, c2])
    cell = Cell(domain, incls)

    assert len(cell.shapes) == 2
    assert len(cell.primary_shapes) == 2
    assert len(cell.periodic_shapes) == 0

    p_cell = cell.make_periodic()
    assert len(p_cell.primary_shapes) == 2
    assert len(p_cell.periodic_shapes) == 1
    assert len(p_cell.shapes) == 3

    # Primary area is preserved (not doubled by periodic copy)
    expected_primary_area = 2.0 * np.pi * 1.0**2
    assert np.isclose(p_cell.shapes.area, expected_primary_area)
    assert np.isclose(
        p_cell.shapes.area_fraction(domain.area),
        expected_primary_area / 100.0,
    )
    # Count reflects primary shapes
    counts = p_cell.shapes.get_count()
    assert counts[gb.Circle] == 2


def test_periodic_cell_serialization(tmp_path):
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))
    c = gb.Circle(radius=1.0, centre=[0.2, 5.0])
    cell = Cell(domain, Inclusions2D([c])).make_periodic()

    # Save to JSON
    json_path = tmp_path / "cell_periodic.json"
    cell.save(json_path)

    # Load from JSON
    loaded = Cell.load(json_path)
    assert len(loaded.shapes) == 2
    assert len(loaded.primary_shapes) == 1
    assert len(loaded.periodic_shapes) == 1

    # Save to NPZ
    npz_path = tmp_path / "cell_periodic.npz"
    cell.save(npz_path)
    loaded_npz = Cell.load(npz_path)
    assert len(loaded_npz.shapes) == 2
    assert len(loaded_npz.primary_shapes) == 1
    assert len(loaded_npz.periodic_shapes) == 1


# ============================================================================
# 3. Periodic Overlap Optimization Tests
# ============================================================================


def test_periodic_vs_nonperiodic_overlap():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))
    # Two circles near opposite boundaries: x = 0.2 and x = 9.8.
    # Non-periodic Euclidean distance = 9.6 > 2.0 (no overlap).
    # Periodic minimum-image distance = 0.4 < 2.0 (severe overlap).
    c0 = gb.Circle(radius=1.0, centre=[0.2, 5.0])
    c1 = gb.Circle(radius=1.0, centre=[9.8, 5.0])

    incls = Inclusions2D([c0, c1])

    # 1. Non-periodic
    prob_nonperiodic = CellShapes2DOverlap(
        domain, incls, periodicity=False, buffer_thickness_ratio=0.0
    )
    cost_np = prob_nonperiodic.func(prob_nonperiodic.x0)
    assert cost_np == 0.0

    # 2. Periodic
    prob_periodic = CellShapes2DOverlap(
        domain, incls, periodicity=True, buffer_thickness_ratio=0.0
    )
    cost_p = prob_periodic.func(prob_periodic.x0)
    # Penetration depth: 2.0 - 0.4 = 1.6 -> cost = 1.6^2 = 2.56
    assert cost_p > 2.0
    assert np.isclose(cost_p, 1.6**2)


def test_periodic_gradient_finite_difference():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))
    # Shapes overlapping across boundaries (horizontal and vertical)
    c0 = gb.Circle(radius=1.0, centre=[0.2, 5.0])
    e1 = gb.Ellipse(
        semi_major_length=1.5,
        semi_minor_length=0.8,
        centre=[9.8, 5.0],
        major_axis_angle=gb.Angle.deg(35),
    )
    c2 = gb.Circle(radius=1.0, centre=[5.0, 0.3])
    c3 = gb.Circle(radius=1.0, centre=[5.0, 9.7])

    incls = Inclusions2D([c0, e1, c2, c3])
    prob = CellShapes2DOverlap(
        domain, incls, periodicity=True, buffer_thickness_ratio=0.05
    )

    x0 = incls.get_positions(flat=True)
    cost0, grad0 = prob._evaluate(x0)
    assert cost0 > 0.0

    # Numerical finite differences
    grad_fd = np.zeros_like(grad0)
    eps = 1e-7
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_plus[i] += eps
        prob._cache_x = None
        c_plus, _ = prob._evaluate(x_plus)

        x_minus = x0.copy()
        x_minus[i] -= eps
        prob._cache_x = None
        c_minus, _ = prob._evaluate(x_minus)

        grad_fd[i] = (c_plus - c_minus) / (2 * eps)

    assert np.allclose(grad0, grad_fd, atol=1e-5)


def test_periodic_optimization_solve():
    domain = CellDomain2D(gb.Bounds2DRectangular(0.0, 0.0, 10.0, 10.0))
    # Two overlapping shapes across periodic boundary
    c0 = gb.Circle(radius=1.0, centre=[0.2, 5.0])
    c1 = gb.Circle(radius=1.0, centre=[9.8, 5.0])

    incls = Inclusions2D([c0, c1])
    prob = CellShapes2DOverlap(
        domain, incls, periodicity=True, buffer_thickness_ratio=0.05
    )

    sol = prob.solve(method="nmspg", iter_max=100, p_bar=False)
    assert sol.status == "success"
    assert np.isclose(sol.f_history[-1], 0.0, atol=1e-6)

    # Check that after solve, minimum periodic distance >= 2 * radius + buffer
    final_pos = prob._as_matrix(sol.x_optimal)
    dx = final_pos[0, 0] - final_pos[1, 0]
    dx = dx - 10.0 * np.round(dx / 10.0)
    dy = final_pos[0, 1] - final_pos[1, 1]
    dy = dy - 10.0 * np.round(dy / 10.0)
    dist = np.hypot(dx, dy)
    assert dist >= 2.0 * (1.0 + 0.05) - 1e-5


# ============================================================================
# 4. End-to-End Pipeline & Tiling Continuity
# ============================================================================


def test_end_to_end_periodic_generation(tmp_path):
    cfg_dict = {
        "num_cells": 1,
        "metadata": {
            "title": "e2e-periodic-test",
            "rng_seed": 123,
            "log_level": "info",
        },
        "domain": {
            "shape": "rectangle",
            "bounds": {"x_min": 0, "y_min": 0, "x_max": 25, "y_max": 25},
        },
        "shapes": [
            {
                "name": "Circle",
                "volume_fraction": 0.15,
                "params": {"radius": 1.5},
            },
            {
                "name": "Ellipse",
                "volume_fraction": 0.10,
                "params": {
                    "semi_major_length": 2.0,
                    "semi_minor_length": 1.0,
                    "major_axis_angle": "uniform(0.0, 3.1415)",
                },
            },
        ],
        "packing": {
            "periodicity": True,
            "min_gap_ratio": 0.05,
            "proj_buffer_ratio": 2.0,
            "adjust_bounds_to_exact_vf": False,
        },
        "solver": {
            "init_method": "uniform",
            "optimiser": {"name": "nmspg", "max_iter": 50},
        },
        "export": {
            "output_dir": str(tmp_path),
            "as_array": True,
            "formats": ["png", "json"],
            "background": 0,
            "facecolor": 255,
            "image_dimensions": [128, 128],
            "dpi": 72,
        },
    }

    cells = generate_cell_2d(cfg_dict)
    assert len(cells) == 1
    init_cell, final_cell = cells[0]

    # Verify output files
    assert (tmp_path / "final_0.png").exists()
    assert (tmp_path / "final_0.json").exists()
    assert (tmp_path / "final_0.npy").exists()

    # Load exported numpy array
    img = np.load(tmp_path / "final_0.npy")
    assert img.shape == (128, 128)

    # Verify periodicity: periodic copies should exist
    assert len(final_cell.shapes) >= len(final_cell.primary_shapes)
    assert len(final_cell.periodic_shapes) > 0

    # Mathematical periodicity: every periodic copy matches a primary shape shifted by (+/- Lx, +/- Ly)
    Lx = 25.0
    Ly = 25.0
    for p_copy in final_cell.periodic_shapes:
        matched = False
        for primary in final_cell.primary_shapes:
            dx = abs(p_copy.position.x - primary.position.x)
            dy = abs(p_copy.position.y - primary.position.y)
            valid_x = np.isclose(dx, 0.0, atol=1e-5) or np.isclose(dx, Lx, atol=1e-5)
            valid_y = np.isclose(dy, 0.0, atol=1e-5) or np.isclose(dy, Ly, atol=1e-5)
            if valid_x and valid_y and not (np.isclose(dx, 0.0, atol=1e-5) and np.isclose(dy, 0.0, atol=1e-5)):
                matched = True
                break
        assert matched, f"Periodic copy at {p_copy.position} did not match any primary shape"

    # Tiling continuity check:
    # Opposing borders of the exported cell have matching active pixel presence
    left_edge_active = np.sum(img[:, 0] > 0)
    right_edge_active = np.sum(img[:, -1] > 0)
    top_edge_active = np.sum(img[0, :] > 0)
    bottom_edge_active = np.sum(img[-1, :] > 0)

    # Edge continuity: opposing borders both intersect crossing shapes
    assert (left_edge_active > 0) == (right_edge_active > 0)
    assert (top_edge_active > 0) == (bottom_edge_active > 0)
    assert abs(left_edge_active - right_edge_active) <= 10
    assert abs(top_edge_active - bottom_edge_active) <= 10

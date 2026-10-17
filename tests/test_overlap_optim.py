import gbox as gb
import numpy as np
import pytest

from shapeforge.cell import CellDomain2D, Inclusions2D
from shapeforge.optim import OptimisationProblem
from shapeforge.overlap_optim import CellShapes2DOverlap, ShapesOverlap


def test_optimisation_problem_base():
    prob = OptimisationProblem()
    with pytest.raises(NotImplementedError):
        prob.func(np.zeros(3))

    with pytest.raises(NotImplementedError):
        prob.grad(np.zeros(3))

    with pytest.raises(NotImplementedError):
        prob.projection(np.zeros(3))

    with pytest.raises(ValueError, match="Unknown optimisation method"):
        prob.solve(method="bfgs_unsupported")


def test_shapes_overlap_representation():
    dom = CellDomain2D([0, 0, 10, 10])
    c1 = gb.Circle(1.0, centre=[2.0, 3.0])
    c2 = gb.Circle(1.5, centre=[6.0, 7.0])
    incls = Inclusions2D([c1, c2])

    prob = ShapesOverlap(dom, incls)
    assert prob.num_inclusions == 2

    mat = np.array([[2.0, 3.0, 0.0], [6.0, 7.0, 0.0]])
    flat = prob._as_flat(mat)
    assert len(flat) == 6
    restored = prob._as_matrix(flat)
    np.testing.assert_allclose(mat, restored)


def test_cell_shapes_2d_overlap_zero_and_one_shape():
    dom = CellDomain2D([0, 0, 10, 10])

    # Zero shapes
    prob_empty = CellShapes2DOverlap(dom, Inclusions2D([]))
    assert prob_empty._max_reach == 0.0

    # One shape
    prob_one = CellShapes2DOverlap(
        dom, Inclusions2D([gb.Circle(2.0, centre=[5.0, 5.0])])
    )
    # Eff radius = 2.0 + 2.0 * 0.05 = 2.1; max reach = 2 * 2.1 = 4.2
    np.testing.assert_allclose(prob_one._max_reach, 4.2)


def test_cell_shapes_2d_overlap_caching_and_eval_counts():
    dom = CellDomain2D([0, 0, 20, 20])
    c1 = gb.Circle(2.0, centre=[5.0, 5.0])
    c2 = gb.Circle(2.0, centre=[6.0, 5.0])  # Overlapping
    prob = CellShapes2DOverlap(dom, Inclusions2D([c1, c2]))

    x0 = prob.x0.copy()

    f1 = prob.func(x0)
    assert f1 > 0.0  # Overlap cost is positive

    # Second call with same x should reuse cache
    f2 = prob.func(x0)
    assert f1 == f2
    assert prob._eval_count["func"] == 2

    # Grad should also reuse cache
    g1 = prob.grad(x0)
    assert g1 is not None
    assert len(g1) == len(x0)


def test_gradient_numerical_check():
    dom = CellDomain2D([0, 0, 20, 20])
    c1 = gb.Circle(2.0, centre=[5.0, 5.0])
    c2 = gb.Circle(2.0, centre=[6.5, 5.0])
    prob = CellShapes2DOverlap(dom, Inclusions2D([c1, c2]))

    x = prob.x0.copy()
    analytical_grad = prob.grad(x)

    # Finite difference check for positions
    eps = 1e-6
    num_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = prob._evaluate(x_plus)[0]
        f_minus = prob._evaluate(x_minus)[0]
        num_grad[i] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-4)


def test_projection_bounds_and_determinism():
    dom = CellDomain2D([0, 0, 10, 10])
    # Place circles outside bounds
    c1 = gb.Circle(1.0, centre=[15.0, -5.0])
    e1 = gb.Ellipse(
        1.5, 0.8, centre=[-2.0, 12.0], major_axis_angle=gb.Angle.rad(7.0)
    )
    prob = CellShapes2DOverlap(
        dom,
        Inclusions2D([c1, e1]),
        proj_buffer_ratio=2.0,
        rng=np.random.default_rng(42),
    )

    x_out = prob.x0.copy()
    x_proj = prob.projection(x_out)
    mat_proj = prob._as_matrix(x_proj)

    # All x and y coordinates must be inside [0, 10]
    assert np.all(mat_proj[:, 0] >= 0.0)
    assert np.all(mat_proj[:, 0] <= 10.0)
    assert np.all(mat_proj[:, 1] >= 0.0)
    assert np.all(mat_proj[:, 1] <= 10.0)

    # Orientation must be within [0, 2*pi]
    assert np.all(mat_proj[:, 2] >= 0.0)
    assert np.all(mat_proj[:, 2] <= 2.0 * np.pi)

    # Check determinism with seeded rng
    prob2 = CellShapes2DOverlap(
        dom,
        Inclusions2D([c1, e1]),
        proj_buffer_ratio=2.0,
        rng=np.random.default_rng(42),
    )
    x_proj2 = prob2.projection(x_out)
    np.testing.assert_allclose(x_proj, x_proj2)

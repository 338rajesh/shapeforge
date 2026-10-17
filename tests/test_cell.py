import gbox as gb
import numpy as np
import pytest

from shapeforge.cell import (
    Cell,
    CellDomain2D,
    Inclusions2D,
    _normalize_shape_size_params,
    _shape_from_dict,
    _shape_to_dict,
)


def test_cell_domain_2d():
    # From mapping
    dom = CellDomain2D({"x_min": 0, "y_min": -5, "x_max": 20, "y_max": 15})
    assert dom.x_bounds == (0.0, 20.0)
    assert dom.y_bounds == (-5.0, 15.0)
    assert dom.area == 20.0 * 20.0
    assert isinstance(dom.domain, gb.Bounds2DRectangular)

    # From sequence
    dom_seq = CellDomain2D([0, 0, 10, 10])
    assert dom_seq.area == 100.0

    # Serialization
    d = dom.to_dict()
    dom_loaded = CellDomain2D.from_dict(d)
    assert dom_loaded.area == dom.area


def test_inclusions_2d_collection():
    c1 = gb.Circle(1.0, centre=[2.0, 2.0])
    c2 = gb.Circle(2.0, centre=[5.0, 5.0])
    e1 = gb.Ellipse(
        3.0, 1.5, centre=[10.0, 10.0], major_axis_angle=gb.Angle.rad(0.5)
    )

    incls = Inclusions2D([c1, c2])
    assert len(incls) == 2
    assert c1 in incls

    incls.add(e1)
    assert len(incls) == 3
    assert e1 in incls

    # Add invalid object
    with pytest.raises(TypeError):
        incls.add("not_a_shape")

    # Counts and queries
    counts = incls.get_count()
    assert counts[gb.Circle] == 2
    assert counts[gb.Ellipse] == 1

    circles = incls.get_by_shape(gb.Circle)
    assert len(circles) == 2

    # Area calculations
    total_area = c1.area + c2.area + e1.area
    np.testing.assert_allclose(incls.area, total_area)
    np.testing.assert_allclose(
        incls.area_fraction(1000.0), total_area / 1000.0
    )

    with pytest.raises(ValueError, match="greater than zero"):
        incls.area_fraction(0.0)

    # Positions get and set
    pos = incls.get_positions(flat=False)
    assert pos.shape == (3, 3)
    flat_pos = incls.get_positions(flat=True)
    assert flat_pos.shape == (9,)

    new_pos = pos.copy()
    new_pos[:, 0] += 1.0
    incls.set_positions(new_pos)
    np.testing.assert_allclose(incls.get_positions()[:, 0], pos[:, 0] + 1.0)

    # Invalid positions shape
    with pytest.raises(ValueError):
        incls.set_positions(np.zeros((2, 3)))

    # Equivalent circle radii
    radii = incls.get_equivalent_circle_radii()
    assert len(radii) == 3

    # Bounding box overlap pairs
    incls_overlap = Inclusions2D(
        [
            gb.Circle(5.0, centre=[0.0, 0.0]),
            gb.Circle(5.0, centre=[1.0, 1.0]),
            gb.Circle(1.0, centre=[50.0, 50.0]),
        ]
    )
    pairs = incls_overlap.get_bbox_overlap_pairs()
    assert (1, 0) in pairs

    # Clone and clear
    cloned = incls.clone()
    assert len(cloned) == len(incls)
    incls.clear()
    assert len(incls) == 0
    assert len(cloned) == 3


def test_shape_serialization_roundtrip():
    c = gb.Circle(3.2, centre=[12.0, 15.0])
    e = gb.Ellipse(
        4.0, 1.2, centre=[20.0, 30.0], major_axis_angle=gb.Angle.rad(1.23)
    )

    c_dict = _shape_to_dict(c)
    e_dict = _shape_to_dict(e)

    c_restored = _shape_from_dict(c_dict)
    e_restored = _shape_from_dict(e_dict)

    assert isinstance(c_restored, gb.Circle)
    assert c_restored.radius == 3.2
    assert c_restored.centre == (12.0, 15.0)

    assert isinstance(e_restored, gb.Ellipse)
    assert e_restored.semi_major_length == 4.0
    assert e_restored.semi_minor_length == 1.2
    assert e_restored.centre == (20.0, 30.0)
    np.testing.assert_allclose(
        e_restored.position.orientation.radians, 1.23, rtol=1e-5
    )


def test_normalize_shape_size_params():
    # Circle radius aliases
    norm_c = _normalize_shape_size_params("circle", {"r": 3.0})
    assert norm_c["radius"] == 3.0

    # Ellipse aliases
    norm_e = _normalize_shape_size_params(
        "ellipse", {"major_axis": 4.0, "b": 2.0}
    )
    assert norm_e["semi_major_length"] == 4.0
    assert norm_e["semi_minor_length"] == 2.0

    # Ellipse aspect ratio calculation
    norm_ar = _normalize_shape_size_params(
        "ellipse", {"semi_major_length": 6.0, "aspect_ratio": 3.0}
    )
    assert norm_ar["semi_minor_length"] == 2.0

    norm_ar2 = _normalize_shape_size_params(
        "ellipse", {"semi_minor_length": 2.0, "ar": 4.0}
    )
    assert norm_ar2["semi_major_length"] == 8.0


def test_cell_save_and_load(tmp_path):
    dom = CellDomain2D([0, 0, 50, 50])
    c = gb.Circle(2.0, centre=[10.0, 10.0])
    e = gb.Ellipse(
        3.0, 1.5, centre=[25.0, 25.0], major_axis_angle=gb.Angle.rad(0.8)
    )
    cell = Cell(dom, Inclusions2D([c, e]))

    # Test all supported formats
    for ext in [".json", ".yaml", ".yml", ".pkl", ".pickle", ".npz"]:
        file_path = tmp_path / f"cell_test{ext}"
        saved_path = cell.save(file_path)
        assert saved_path == file_path
        assert file_path.exists()

        # Overwrite without flag raises
        with pytest.raises(FileExistsError):
            cell.save(file_path, overwrite=False)

        # Overwrite with flag succeeds
        cell.save(file_path, overwrite=True)

        # Load back and verify
        loaded = Cell.load(file_path)
        assert isinstance(loaded, Cell)
        assert len(loaded.shapes) == 2
        assert loaded.domain.area == 2500.0

    # Unsupported format on save raises ValueError
    with pytest.raises(ValueError):
        cell.save(tmp_path / "cell.txt")

    # Non-existent file on load raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        Cell.load(tmp_path / "non_existent_cell.json")

    # Existing file with invalid extension raises ValueError
    invalid_file = tmp_path / "cell.txt"
    invalid_file.write_text("hello")
    with pytest.raises(ValueError):
        Cell.load(invalid_file)

    # Corrupt npz missing cell_json
    corrupt_npz = tmp_path / "corrupt.npz"
    np.savez(corrupt_npz, dummy=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="does not contain valid Cell data"):
        Cell.load(corrupt_npz)


def test_cell_plot_array(tmp_path):
    dom = CellDomain2D([0, 0, 20, 20])
    cell = Cell(dom, Inclusions2D([gb.Circle(2.0, centre=[10.0, 10.0])]))

    # Plot without path returns array
    arr = cell.plot(size=(128, 128), background=0, foreground=255, dpi=72)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (128, 128)

    # Plot with path writes file and returns array
    p = tmp_path / "plot.png"
    arr2 = cell.plot(
        f_path=p, size=(128, 128), background=0, foreground=255, dpi=72
    )
    assert p.exists()
    assert arr2.shape == (128, 128)

# ShapeForge

[![CI](https://github.com/338rajesh/shapeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/338rajesh/shapeforge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/338rajesh/shapeforge/graph/badge.svg)](https://codecov.io/gh/338rajesh/shapeforge)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python package for generating 2D and 3D Representative Volume Elements (RVE) and unit cells containing geometric inclusions following specified size, orientation, and spatial distributions.

---

Sample generated cells with $65\%$ of area fraction occupied by shapes. 
| Initial Random Cell | Optimized Non-Overlapping Cell |
| :-----------------: | :----------------------------: |
| ![Example initial output](docs/_static/sample_cells/initial_0.png) | ![Example final output](docs/_static/sample_cells/final_0.png) |
| ![Example initial output](docs/_static/sample_cells/initial_1.png) | ![Example final output](docs/_static/sample_cells/final_1.png) |

---

## Installation

```bash
pip install shapeforge
```

Or for development with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/338rajesh/shapeforge.git
cd shapeforge
uv sync
```

---

## Quick Start

### 1. Command-Line Interface (CLI)

Run cell generation directly from a YAML configuration file:

```bash
# Generate RVE with default configuration
shapeforge demo/config.yaml

# Run with verbose info/debug logging
shapeforge demo/config.yaml --log debug

# Profile execution to find bottlenecks
shapeforge demo/config.yaml --profile --profile-sort cumulative --profile-lines 25
```

### 2. Python API

```python
from pathlib import Path
from shapeforge.main import generate_cell_2d
from shapeforge.cell import Cell

# Generate cells using a configuration file or mapping
cells = generate_cell_2d("demo/config.yaml")

# Each cell pair contains (initial_cell, final_optimized_cell)
initial_cell, final_cell = cells[0]

# Inspect properties
print(f"Number of inclusions: {len(final_cell.shapes)}")
print(f"Domain area: {final_cell.domain.area:.2f}")
print(f"Inclusions area: {final_cell.shapes.area:.2f}")
print(
    f"Actual volume fraction: {final_cell.shapes.area / final_cell.domain.area:.4f}"
)

# Save or plot programmatically
final_cell.save("output/cell_0.json", overwrite=True)
final_cell.plot(
    "output/cell_0.png", size=(512, 512), background=0, foreground=255, dpi=100
)
```

---

## Key Features

- **Geometric Inclusions**: Circular and non-circular shapes (such as `Circle`, `Ellipse`), with constant or statistical parameter distributions (via `scipy.stats`).
- **High-Performance Overlap Optimization**:
  - Vectorized distance queries with `scipy.spatial.cKDTree` and analytical gradients.
  - Non-Monotone Spectral Projected Gradient (`nmspg`) optimization engine for fast, robust overlap resolution.
- **Advanced Spatial Initialization**:
  - Spatial point initialization with `uniform` random sampling, **Latin Hypercube Sampling (`lhs`)**, or **Sobol low-discrepancy sequences (`sobol`)** via `scipy.stats.qmc`.
- **Exact Volume Fraction Bounds**:
  - Option to dynamically scale domain bounds (`packing.adjust_bounds_to_exact_vf`) to achieve exact target volume fractions.
- **Config-Driven Multi-Format Export**:
  - Raster images: `.png`, `.jpg`, `.pdf` with customizable dimensions, DPI, background, and facecolor.
  - Cell serialization: `.json`, `.yaml`, `.pkl`, and `.npz` with lossless round-trip saving and loading (`Cell.save()` / `Cell.load()`).
  - Raw microstructure arrays: `.npy` binary arrays ready for machine learning and finite element pipelines.
  - Diagnostic convergence curves: Automatically plots objective function and gradient norm histories.
- **Built-in Profiling Suite**:
  - Profile runs anytime with `--profile` to pinpoint hotspots in execution time and memory. Compatible with [SnakeViz](https://jiffyclub.github.io/snakeviz/).


---

## Configuration File Structure

ShapeForge is completely driven by declarative YAML configuration files:

```yaml
# ============================================================================
# ShapeForge Configuration
# ============================================================================

num_cells: 1

metadata:
  title: shapeforge-demo
  rng_seed: 42
  log_level: info        # debug | info | warning | error

domain:
  shape: rectangle
  bounds:
    x_min: 0.0
    y_min: 0.0
    x_max: 50.0
    y_max: 50.0

shapes:
  - name: Circle
    volume_fraction: 0.25
    params:
      radius: 1.0         # constant or distribution, e.g. uniform(1.0, 0.2)
  - name: Ellipse
    volume_fraction: 0.20
    params:
      semi_major_length: 2.0
      semi_minor_length: 1.0
      major_axis_angle: uniform(0.0, 1.0)

packing:
  periodicity: false
  min_gap_ratio: 0.05
  proj_buffer_ratio: 2.0
  adjust_bounds_to_exact_vf: false

solver:
  init_method: uniform   # uniform | lhs | sobol
  optimiser:
    name: nmspg
    max_iter: 1000
    epsilon: 1.0e-6

export:
  output_dir: ./scratch/outputs/test_1
  as_array: false        # save raw uint8 array as .npy
  formats:               # png, jpg, pdf, json, pkl, yml, npz
    - png
    - json
  background: 0
  facecolor: 255
  image_dimensions: [512, 512]
  dpi: 100
```

---

## User-Defined Custom Shapes

ShapeForge supports user-defined shapes beyond the built-in `Circle` and `Ellipse` from `gbox`. Users can implement custom 2D particles by subclassing `gbox.Shape2D` in a standalone Python file.

### 1. Defining a Custom Shape

A custom shape needs:
- `from_params(position_params, size_params)`: Factory method for spatial sampling.
- `union_of_circles() -> gb.CirclesArray`: Multi-circle decomposition used by the SPG overlap optimizer.
- `bounding_box`: Rectangular bounds for broadphase culling & periodic boundary copying.
- `area` and `clone()`: Area calculation and deep copy.
- **Plotting**: Either `to_patch()` returning a Matplotlib `Patch` (e.g. `Rectangle`, `PathPatch`) or `boundary_points()` returning an $(N, 2)$ array of boundary coordinates (rendered cleanly as a closed `Polygon`).

```python
import numpy as np
import gbox as gb
from shapeforge import register_shape

@register_shape("stadium")
class Stadium(gb.Shape2D):
    def __init__(self, length: float, radius: float, centre=(0.0, 0.0), orientation=None):
        self._length = float(length)
        self._radius = float(radius)
        self._position = gb.shapes_2d.Shape2DPose(centre[0], centre[1], orientation or gb.Angle.rad(0.0))

    @property
    def area(self) -> float:
        return 2.0 * self._length * self._radius + np.pi * self._radius**2

    def union_of_circles(self) -> gb.CirclesArray:
        xs = np.linspace(-self._length / 2.0, self._length / 2.0, max(3, int(np.ceil(self._length / self._radius))))
        return gb.CirclesArray(np.column_stack([xs, np.zeros_like(xs)]), np.full_like(xs, self._radius))

    def boundary_points(self) -> np.ndarray:
        # Return (N, 2) world-space boundary coordinates for crisp vector rendering
        ...

    @classmethod
    def from_params(cls, position_params, size_params):
        return cls(size_params["length"], size_params["radius"],
                   centre=(position_params["xc"], position_params["yc"]),
                   orientation=position_params.get("major_axis_angle"))
```

### 2. Specifying in `config.yaml` or CLI

Include the script in your configuration file:

```yaml
custom_shapes:
  - "shapes/my_custom_shapes.py"

shapes:
  - name: stadium
    volume_fraction: 0.15
    params:
      length: 3.0
      radius: 0.8
```

Or pass via the CLI flag:

```bash
shapeforge config.yaml --custom-shapes shapes/my_custom_shapes.py
```

---

## Architecture

A **Cell** consists of:
- **Domain**: Spatial boundary (e.g. `CellDomain2D` with rectangular bounds).
- **Inclusions**: Collection of shapes (`Inclusions2D`) containing `gb.Shape2D` objects.
  - **Positional parameters**: Centroid coordinates $(x, y)$ and orientation angle $\theta$. These are optimized by the solver while preserving shape sizes.
  - **Size parameters**: Semi-major length, semi-minor length, radius, etc. Defined either as fixed numbers or continuous distributions (`norm`, `uniform`, `lognorm`, etc.).

---

## Profiling & Performance Diagnostics

ShapeForge includes an integrated profiler. You can profile any generation run to analyze runtime hotspots:

```bash
# Profile and display the top 20 cumulative time hotspots
shapeforge demo/config.yaml --profile --profile-sort cumulative --profile-lines 20

# Visualize binary profile with SnakeViz
snakeviz scratch/outputs/test_1/shapeforge.prof
```

---

## Testing

Run the test suite with `pytest`:

```bash
pytest -v
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

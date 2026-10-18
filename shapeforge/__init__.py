"""
ShapeForge: Microstructure and Representative Volume Element (RVE) Generation.
==============================================================================

ShapeForge is a high-performance Python package for generating 2D (and extensible
to 3D) geometric domains packed with inclusions of arbitrary shapes, sizes, and
orientations according to prescribed volume fractions and statistical distributions.

Features
--------
- Flexible shape definitions: built-in circles and ellipses via ``gbox``, plus
  arbitrary user-defined custom shapes via ``@register_shape``.
- Advanced spatial initialisation: uniform, Latin Hypercube Sampling (LHS), and
  Sobol low-discrepancy sequences.
- Robust overlap resolution: Non-monotone Spectral Projected Gradient (NM-SPG)
  optimization with Barzilai-Borwein step lengths and $k$-d tree accelerated
  contact potential calculations.
- Strict periodic boundary conditions (PBC) with automatic boundary wrapping and
  ghost shape generation.
- Integrated visualization, profiling, and serialization (JSON, YAML, Pickle, NPZ).

Exported Symbols
----------------
Cell : class
    Host microstructure cell containing domain bounds and inclusions.
CellDomain2D : class
    2D rectangular spatial bounding domain for unit cells.
generate_cell_2d : function
    Primary programmatic API entry point to generate cells from a configuration.
ShapeForgeConfig : class
    Validated data container for complete task configurations.
CellPlotter : class
    Visualization engine supporting built-in and custom polygon shapes.
Profiler : class
    Context manager for performance profiling and hotspot analysis.
profile_function : function
    Decorator for targeted function execution profiling.
ShapeRegistry : class
    Global registry for built-in and user-defined shape classes.
register_shape : function
    Decorator to register a custom shape class with ShapeForge.

Examples
--------
Programmatically generate a periodic unit cell:

>>> from shapeforge import generate_cell_2d
>>> cells = generate_cell_2d("config.yaml")
>>> initial_cell, final_cell = cells[0]
>>> final_cell.plot("microstructure.png")
"""

from .cell import Cell, CellDomain2D
from .config import ShapeForgeConfig
from .main import generate_cell_2d
from .plotter import CellPlotter
from .profiler import Profiler, profile_function
from .registry import ShapeRegistry, register_shape

__all__ = [
    "Cell",
    "CellDomain2D",
    "CellPlotter",
    "Profiler",
    "ShapeForgeConfig",
    "ShapeRegistry",
    "generate_cell_2d",
    "profile_function",
    "register_shape",
]

.. _api_reference:

=============
API Reference
=============

This document details the complete public Python API of ShapeForge. All classes and functions are fully documented with NumPy style docstrings and type annotations.

Modules Overview
================

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`api/top_level`
     - Core high-level package exports: :class:`~shapeforge.cell.Cell`, :class:`~shapeforge.cell.CellDomain2D`, :func:`~shapeforge.main.generate_cell_2d`, and :class:`~shapeforge.config.ShapeForgeConfig`.
   * - :doc:`api/cell`
     - Domain bounds, inclusion containers, position samplers, and unit cell representations.
   * - :doc:`api/config`
     - Configuration schemas, dataclasses, and YAML/JSON deserializers.
   * - :doc:`api/optim`
     - Optimization problem formulations and the Non-monotone Spectral Projected Gradient (:func:`~shapeforge.optim.nmspg`) solver.
   * - :doc:`api/overlap_optim`
     - Overlap potential formulation, broad-phase $k$-d tree accelerated search, and analytic gradients.
   * - :doc:`api/plotter`
     - Matplotlib patch rendering engine for native gbox shapes and user-defined custom shapes.
   * - :doc:`api/registry`
     - Dynamic custom shape registry and loader.
   * - :doc:`api/profiler`
     - Deterministic profiling context managers, runtime reports, and stats dumpers.
   * - :doc:`api/utils`
     - Statistical distribution samplers, signature parsers, and convergence plotting helpers.
   * - :doc:`api/main`
     - CLI parsing, workflow pipeline orchestration, and multi-format exporters.

.. toctree::
   :maxdepth: 1
   :caption: API Modules
   :hidden:

   api/top_level
   api/cell
   api/config
   api/optim
   api/overlap_optim
   api/plotter
   api/registry
   api/profiler
   api/utils
   api/main

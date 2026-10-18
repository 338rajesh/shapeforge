.. _examples:

========
Examples
========

This guide provides practical, ready-to-run examples demonstrating common workflows in ShapeForge, from basic monodisperse packings to advanced multi-phase periodic composites and custom shape registration.

Example 1: Polydisperse Circular Inclusions
===========================================

In this example, we generate a 2D microstructure with circular inclusions following a uniform size distribution in a $25 \times 25$ square domain.

Configuration (``circles.yaml``)
--------------------------------

.. code-block:: yaml

   num_cells: 1
   metadata:
     title: polydisperse-circles
     rng_seed: 100
     log_level: info

   domain:
     shape: rectangle
     bounds:
       x_min: 0.0
       y_min: 0.0
       x_max: 25.0
       y_max: 25.0

   shapes:
     - name: Circle
       volume_fraction: 0.35
       params:
         radius: uniform(0.8, 0.6)  # Radii drawn uniformly from [0.8, 1.4]

   packing:
     periodicity: false
     min_gap_ratio: 0.05
     proj_buffer_ratio: 1.0
     adjust_bounds_to_exact_vf: false

   solver:
     init_method: sobol
     optimiser:
       name: nmspg
       iter_max: 1000
       iter_memory: 10
       epsilon: 1.0e-6

   export:
     output_dir: ./output_circles
     as_array: false
     formats:
       - png
       - json
     background: 0
     facecolor: 255
     image_dimensions: [512, 512]
     dpi: 100

Running from the CLI
--------------------

.. code-block:: bash

   shapeforge circles.yaml

Example 2: Multi-Phase Packing with Circles and Ellipses
========================================================

Here, we simulate a dual-phase inclusion microstructure comprising both circular aggregates and oriented elliptical fibers.

Configuration (``multiphase.yaml``)
-----------------------------------

.. code-block:: yaml

   num_cells: 1
   metadata:
     title: dual-phase-composite
     rng_seed: 42
     log_level: info

   domain:
     shape: rectangle
     bounds:
       x_min: 0.0
       y_min: 0.0
       x_max: 30.0
       y_max: 30.0

   shapes:
     - name: Circle
       volume_fraction: 0.20
       params:
         radius: 1.2

     - name: Ellipse
       volume_fraction: 0.20
       params:
         semi_major_length: 2.2
         semi_minor_length: 0.9
         major_axis_angle: uniform(0.0, 3.14159)  # Uniform random angles in [0, pi)

   packing:
     periodicity: true
     min_gap_ratio: 0.08
     proj_buffer_ratio: 1.0
     adjust_bounds_to_exact_vf: false

   solver:
     init_method: lhs
     optimiser:
       name: nmspg
       iter_max: 1000
       iter_memory: 15
       epsilon: 1.0e-6

   export:
     output_dir: ./output_multiphase
     as_array: false
     formats:
       - png
       - json
     background: 0
     facecolor: 255
     image_dimensions: [512, 512]
     dpi: 100

Python API Execution
--------------------

.. code-block:: python

   from shapeforge import generate_cell_2d

   cells = generate_cell_2d("multiphase.yaml")
   initial_cell, final_cell = cells[0]

   print(f"Generated {len(final_cell.shapes)} total inclusions.")
   counts = final_cell.shapes.get_count()
   for shape_type, count in counts.items():
       print(f"  {shape_type.__name__}: {count}")

Example 3: Periodic Boundary Conditions & Tiling
================================================

When ``packing.periodicity: true`` is enabled, ShapeForge enforces toroidal boundary conditions. Inclusions that cross any of the four domain walls are automatically split, and translated periodic ghost copies are placed on the opposing wall.

Checking Periodic Continuity in Python
--------------------------------------

.. code-block:: python

   from shapeforge import generate_cell_2d
   import numpy as np

   cells = generate_cell_2d("multiphase.yaml")
   _, periodic_cell = cells[0]

   # The cell contains both primary inclusions and periodic boundary copies
   print("Primary shapes:", len(periodic_cell.primary_shapes))
   print("Periodic boundary copies:", len(periodic_cell.periodic_shapes))

   # Render an image
   img = periodic_cell.plot("periodic_rve.png", size=(512, 512))

   # Replicate the RVE into a 2x2 periodic grid to visually confirm seam continuity
   tiled_2x2 = np.block([[img, img], [img, img]])
   import matplotlib.pyplot as plt
   plt.imsave("tiled_2x2.png", tiled_2x2)

Example 4: Implementing and Registering a Custom Shape
======================================================

ShapeForge allows users to define arbitrary 2D geometric shapes by subclassing ``gbox.Shape2D``.

To integrate a custom shape with the optimizer and visualization engines, implement:

1. ``area`` property (for target volume fraction tracking).
2. ``position`` property getter and setter (for coordinate updates).
3. ``union_of_circles()`` method (approximates the shape as disks for contact potential evaluation).
4. ``boundary_points()`` method (coordinates of the perimeter for Polygon plotting).
5. ``from_params()`` classmethod (instantiates the shape from configuration parameters).

Custom Shape Definition (``cshape.py``)
---------------------------------------

.. code-block:: python

   from __future__ import annotations
   import gbox as gb
   import numpy as np
   from shapeforge import register_shape

   @register_shape("cshape")
   class CShape(gb.Shape2D):
       """C-shaped inclusion formed by sweeping disk r_small along arc r_curve."""

       def __init__(
           self,
           r_small: float,
           r_curve: float,
           c_theta: float | gb.Angle = 0.0,
           centre=(0.0, 0.0),
           orientation: float | gb.Angle | None = None,
       ):
           self._r_small = float(r_small)
           self._r_curve = float(r_curve)
           self._c_theta = (
               c_theta if isinstance(c_theta, gb.Angle)
               else gb.Angle.rad(float(c_theta))
           )
           ori = (
               orientation if isinstance(orientation, gb.Angle)
               else gb.Angle.rad(float(orientation or 0.0))
           )
           self._position = gb.shapes_2d.Shape2DPose(centre[0], centre[1], ori)

       @property
       def position(self) -> gb.shapes_2d.Shape2DPose:
           return self._position

       @position.setter
       def position(self, pos) -> None:
           self._position = pos

       @property
       def area(self) -> float:
           ends_area = np.pi * self._r_small**2
           strip_area = 2.0 * self._r_curve * self._r_small * self._c_theta.radians
           return float(ends_area + strip_area)

       def union_of_circles(self) -> gb.CirclesArray:
           th = np.linspace(0, self._c_theta.radians, 20)
           xs = self._r_curve * np.cos(th)
           ys = self._r_curve * np.sin(th)
           points = gb.PointArray2D.from_named_dims(xs=xs, ys=ys)
           points = points.transform(
               dx=self._position.x,
               dy=self._position.y,
               orientation=self._position.orientation,
           )
           return gb.CirclesArray.from_radii_and_centres(
               radii=self._r_small, centres=points
           )

       def boundary_points(self) -> np.ndarray:
           th_arc = np.linspace(0, self._c_theta.radians, 30)
           r_in = max(self._r_curve - self._r_small, 0.01)
           r_out = self._r_curve + self._r_small

           inner_x = r_in * np.cos(th_arc)
           inner_y = r_in * np.sin(th_arc)
           outer_x = r_out * np.cos(th_arc[::-1])
           outer_y = r_out * np.sin(th_arc[::-1])

           body_pts = np.vstack([
               np.column_stack([inner_x, inner_y]),
               np.column_stack([outer_x, outer_y]),
           ])

           x, y = float(self._position.x), float(self._position.y)
           th = float(self._position.orientation.radians)
           c, s = np.cos(th), np.sin(th)
           wx = x + c * body_pts[:, 0] - s * body_pts[:, 1]
           wy = y + s * body_pts[:, 0] + c * body_pts[:, 1]
           return np.column_stack([wx, wy])

       @classmethod
       def from_params(cls, position_params=None, size_params=None):
           pos = position_params or {}
           sz = size_params or {}
           return cls(
               r_small=sz.get("r_small", 0.4),
               r_curve=sz.get("r_curve", 1.0),
               c_theta=sz.get("c_theta", np.pi),
               centre=(pos.get("xc", 0.0), pos.get("yc", 0.0)),
               orientation=pos.get("major_axis_angle", 0.0),
           )

Using the Custom Shape in Configuration
---------------------------------------

In your configuration YAML file, list ``cshape.py`` under ``custom_shapes`` and define inclusions with ``name: CShape``:

.. code-block:: yaml

   custom_shapes:
     - "cshape.py"

   shapes:
     - name: CShape
       volume_fraction: 0.15
       params:
         r_small: 0.4
         r_curve: 1.0
         c_theta: uniform(1.0, 2.0)

Example 5: Exact Volume Fraction Matching
=========================================

Because shapes are sampled as discrete objects, packing a discrete number of inclusions into a fixed domain will slightly undershoot or overshoot the continuous target volume fraction $\phi$.

To match the target volume fraction exactly, enable ``adjust_bounds_to_exact_vf: true`` under ``packing``:

.. code-block:: yaml

   packing:
     periodicity: true
     min_gap_ratio: 0.05
     adjust_bounds_to_exact_vf: true

When enabled, ShapeForge computes the exact required domain area:

$$A_{\text{required}} = \frac{A_{\text{actual inclusions}}}{\sum \phi_i}$$

and smoothly expands or contracts the domain boundaries ($x_{\max}, y_{\max}$) before solving the packing, ensuring the final cell achieves the prescribed volume fraction to high precision.

Example 6: Cell Loading and Programmatic Inspection
===================================================

Saved unit cell configurations can be loaded back into memory for further geometric interrogation or FE mesh generation:

.. code-block:: python

   from shapeforge import Cell
   import numpy as np

   # Load saved cell from JSON, YAML, Pickle, or NPZ
   cell = Cell.load("output_circles/final_0.json")

   # Access domain bounds
   print("Domain bounds:", cell.domain.bounds.to_dict())

   # Retrieve poses of all inclusions as an (N, 3) numpy matrix
   poses = cell.shapes.get_positions()
   print("Shape poses matrix shape:", poses.shape)

   # Calculate equivalent radii
   eq_radii = cell.shapes.get_equivalent_circle_radii()
   print(f"Mean equivalent radius: {np.mean(eq_radii):.3f}")

   # Find overlapping bounding boxes (should be 0 for relaxed cell)
   overlaps = cell.shapes.get_bbox_overlap_pairs()
   print(f"Number of overlapping bounding box pairs: {len(overlaps)}")


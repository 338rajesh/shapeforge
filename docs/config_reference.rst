.. _config_reference:

======================
Config File Reference
======================

ShapeForge uses YAML (or JSON) configuration files to describe all parameters of a unit cell generation run. This document provides a complete specification for all configuration sections, keys, allowed values, and statistical distributions.

Top-Level Structure
===================

A ShapeForge configuration file is divided into seven main sections:

.. code-block:: yaml

   num_cells: <integer>
   metadata: { ... }
   domain: { ... }
   custom_shapes: [ ... ]  # Optional
   shapes: [ ... ]
   packing: { ... }
   solver: { ... }
   export: { ... }

General Execution Keys
======================

``num_cells``
-------------
:Type: ``integer``
:Required: Yes
:Validation: Must be $\ge 1$.

Specifies the number of independent, statistically equivalent unit cells to generate during the run. Each cell $i \in \{0, \dots, \text{num\_cells}-1\}$ is seeded deterministically with ``metadata.rng_seed + i``.

``custom_shapes`` (Optional)
-----------------------------
:Type: ``list[string]``
:Default: ``[]``

List of paths to Python source files defining custom subclasses of ``gbox.Shape2D``. Paths can be absolute or relative to the directory containing the configuration file. Classes decorated with ``@register_shape`` or defining ``from_params`` will be loaded dynamically and made available under the ``shapes`` section.

Example:

.. code-block:: yaml

   custom_shapes:
     - "custom_shapes.py"
     - "inclusions/cshape.py"

Section: ``metadata``
=====================

Controls task-level identification, reproducibility, and logging.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``title``
     - ``string``
     - Required
     - Human-readable name for the simulation or task.
   * - ``rng_seed``
     - ``integer``
     - Required
     - Master base seed for the pseudo-random number generator (PRNG).
   * - ``log_level``
     - ``string``
     - Required
     - Logging level: ``'debug'``, ``'info'``, ``'warning'``, or ``'error'``.

Example:

.. code-block:: yaml

   metadata:
     title: polymer-composite-rve
     rng_seed: 12345
     log_level: info

Section: ``domain``
===================

Defines the spatial boundary envelope in which the inclusions will reside.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``shape``
     - ``string``
     - Required
     - Domain boundary geometry. Currently ``'rectangle'`` is supported for 2D.
   * - ``bounds``
     - ``mapping``
     - Required
     - Bounding box limits: ``x_min``, ``y_min``, ``x_max``, ``y_max``.

Example:

.. code-block:: yaml

   domain:
     shape: rectangle
     bounds:
       x_min: 0.0
       y_min: 0.0
       x_max: 30.0
       y_max: 30.0

Section: ``shapes``
===================

Defines the geometric inclusion phases to pack into the domain. Contains a list of shape specifications.

Each entry in ``shapes`` contains:

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``name``
     - ``string``
     - Required
     - Name of the shape class (case-insensitive). Built-in: ``'Circle'``, ``'Ellipse'``. Custom shapes registered via ``custom_shapes`` are also referenced here.
   * - ``volume_fraction``
     - ``float``
     - Required
     - Target area fraction $\phi_i \in (0, 1.0)$. The sum of all volume fractions must not exceed $1.0$.
   * - ``params``
     - ``mapping``
     - Required
     - Geometric parameters required to instantiate the shape.

Parameter Signatures & Statistical Distributions
------------------------------------------------

Values inside ``params`` can be specified as either:

1. **Constant Floats**: A single numerical value.
   Example: ``radius: 1.5``
2. **Distribution Signatures**: A continuous distribution name and parameters adhering to the format ``"<dist_name>(<loc>, <scale>)"``.
   Any continuous distribution from ``scipy.stats`` is supported:

   * ``uniform(loc, scale)``: Uniform random variate in $[loc, loc + scale]$.
   * ``norm(loc, scale)``: Gaussian normal distribution with mean $loc$ and standard deviation $scale$.
   * ``lognorm(loc, scale)``: Lognormal distribution.

Built-in Shape Parameters
-------------------------

Circle
~~~~~~
* ``radius`` (or ``r``): Radius of the circular inclusion.

Ellipse
~~~~~~~
* ``semi_major_length`` (or ``a``): Semi-major axis length.
* ``semi_minor_length`` (or ``b``): Semi-minor axis length.
* Alternatively, supply ``semi_major_length`` and ``aspect_ratio`` (ratio $a/b$).
* ``major_axis_angle`` (or ``orientation``, ``angle``, ``theta``): Orientation angle in radians. If omitted, defaults to ``uniform(0.0, 6.2831853)`` ($[0, 2\pi)$).

Example:

.. code-block:: yaml

   shapes:
     - name: Circle
       volume_fraction: 0.20
       params:
         radius: uniform(0.8, 0.4)

     - name: Ellipse
       volume_fraction: 0.15
       params:
         semi_major_length: 2.0
         semi_minor_length: 1.0
         major_axis_angle: uniform(0.0, 3.14159)

Section: ``packing``
====================

Controls physical distance buffers, boundary conditions, and volume adjustments.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``min_gap_ratio``
     - ``float``
     - Required
     - Minimum surface-to-surface gap buffer specified as a fraction of each shape's equivalent radius ($> 0$).
   * - ``periodicity``
     - ``boolean``
     - ``false``
     - If ``true``, toroidal periodic boundary conditions (PBC) are enforced. Boundary-crossing shapes generate translated ghost copies on opposite boundaries.
   * - ``proj_buffer_ratio``
     - ``float``
     - ``2.0``
     - Projection buffer thickness ratio used when clipping inclusions inside non-periodic boundaries.
   * - ``adjust_bounds_to_exact_vf``
     - ``boolean``
     - ``false``
     - If ``true``, automatically scales domain boundaries after discrete sampling so that the actual particle count matches the target volume fraction exactly.

Example:

.. code-block:: yaml

   packing:
     periodicity: true
     min_gap_ratio: 0.05
     proj_buffer_ratio: 1.0
     adjust_bounds_to_exact_vf: true

Section: ``solver``
===================

Specifies the spatial initialisation method and numerical optimization hyperparameters.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``init_method``
     - ``string``
     - Required
     - Spatial initialisation algorithm: ``'uniform'`` (pseudorandom), ``'lhs'`` (Latin Hypercube Sampling), or ``'sobol'`` (Sobol low-discrepancy sequence).
   * - ``optimiser``
     - ``mapping``
     - Required
     - Optimization engine settings.

Optimiser Hyperparameters
-------------------------

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``name``
     - ``string``
     - ``'nmspg'``
     - Optimization algorithm.
   * - ``iter_max``
     - ``integer``
     - ``1000``
     - Maximum number of optimization iterations.
   * - ``iter_memory``
     - ``integer``
     - ``10``
     - History memory length $M$ for non-monotone line search.
   * - ``epsilon``
     - ``float``
     - ``1e-6``
     - Termination tolerance for projected gradient norm $\|d_k\|_\infty \le \epsilon$.
   * - ``f_tol``
     - ``float``
     - ``1e-6``
     - Target objective function value tolerance. Terminates successfully as soon as overlap potential $f_k \le f_{\text{tol}}$. If gradient norm reaches $\epsilon$ with $f > f_{\text{tol}}$, reports stagnation failure.

Example:

.. code-block:: yaml

   solver:
     init_method: sobol
     optimiser:
       name: nmspg
       iter_max: 1000
       iter_memory: 10
       epsilon: 1.0e-6
       f_tol: 1.0e-6

Section: ``export``
===================

Controls export file formats, output directory, and image rendering properties.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Description
   * - ``output_dir``
     - ``string`` / ``path``
     - Required
     - Destination folder for exported files. Created automatically if it does not exist.
   * - ``as_array``
     - ``boolean``
     - ``false``
     - If ``true``, saves raw image arrays as NumPy binaries (``.npy``).
   * - ``formats``
     - ``list[string]``
     - Required
     - List of file formats to save. Allowed: ``png``, ``jpg``, ``jpeg``, ``pdf``, ``json``, ``yaml``, ``pkl``, ``npz``.
   * - ``background``
     - ``integer``
     - Required
     - Grayscale background intensity $[0, 255]$ ($0$ = black).
   * - ``facecolor``
     - ``integer``
     - Required
     - Grayscale inclusion fill color $[0, 255]$ ($255$ = white).
   * - ``image_dimensions``
     - ``list[int, int]``
     - Required
     - Image pixel dimensions ``[width, height]``.
   * - ``dpi``
     - ``integer``
     - Required
     - Resolution in dots per inch for saved plots.

Example:

.. code-block:: yaml

   export:
     output_dir: ./scratch/outputs/demo
     as_array: false
     formats:
       - png
       - json
     background: 0
     facecolor: 255
     image_dimensions: [512, 512]
     dpi: 100


.. _cli_reference:

=============
CLI Reference
=============

The ShapeForge Command-Line Interface (CLI) provides a direct terminal tool for generating microstructures, managing custom shape imports, and diagnosing optimization performance.

Synopsis
========

.. code-block:: bash

   shapeforge [-h] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
              [--profile] [--profile-output PROFILE_OUTPUT]
              [--profile-lines PROFILE_LINES]
              [--profile-sort {cumulative,time,calls}]
              [--custom-shapes [CUSTOM_SHAPES ...]]
              config_file

Arguments
=========

Positional Arguments
--------------------

``config_file``
    Path to the YAML or JSON configuration file specifying domain bounds, inclusion phases, solver parameters, and export settings.

    Example:

    .. code-block:: bash

       shapeforge path/to/config.yaml

Options and Flags
-----------------

``--log {DEBUG, INFO, WARNING, ERROR, CRITICAL}``
    Override the logging verbosity level for the run. If omitted, ShapeForge uses the ``metadata.log_level`` defined in the configuration file. Case-insensitive.

    Example:

    .. code-block:: bash

       shapeforge config.yaml --log debug

``--custom-shapes [CUSTOM_SHAPES ...]``
    Optional list of paths to Python files defining custom subclasses of ``gbox.Shape2D``. Dynamically imports and registers the shape classes prior to configuration parsing.

    Example:

    .. code-block:: bash

       shapeforge config.yaml --custom-shapes my_shapes.py extra_shapes.py

``--profile``
    Enable execution profiling using Python's deterministic ``cProfile`` engine. Captures function call frequencies, per-call execution times, and cumulative durations. On completion, prints a formatted bottleneck report table to stdout.

    Example:

    .. code-block:: bash

       shapeforge config.yaml --profile

``--profile-output PROFILE_OUTPUT``
    Filesystem destination path for the binary profile data file. Defaults to ``<export.output_dir>/shapeforge.prof`` if not explicitly provided.

    Example:

    .. code-block:: bash

       shapeforge config.yaml --profile --profile-output ./perf.prof

``--profile-lines PROFILE_LINES``
    Number of top bottleneck functions to display in the console summary table (default: ``25``).

    Example:

    .. code-block:: bash

       shapeforge config.yaml --profile --profile-lines 15

``--profile-sort {cumulative, time, calls}``
    Sorting metric for the terminal profiling report (default: ``cumulative``):

    * ``cumulative``: Total time spent in the function and all subfunctions.
    * ``time``: Internal time spent solely within the function body (self-time).
    * ``calls``: Total number of times the function was invoked.

    Example:

    .. code-block:: bash

       shapeforge config.yaml --profile --profile-sort time

``-h, --help``
    Display standard command help message and exit.

Output Directory Structure
==========================

Upon execution, ShapeForge generates outputs inside the directory specified by ``export.output_dir``:

.. code-block:: text

   output_dir/
   ├── initial_0.png            # Unrelaxed initial state rendering (cell 0)
   ├── initial_0.json           # Serialized initial cell geometry (cell 0)
   ├── final_0.png              # Optimized, overlap-free microstructure (cell 0)
   ├── final_0.json             # Serialized final cell geometry (cell 0)
   ├── fg_variation_0.png       # Objective function f and gradient norm plot
   ├── initial_1.png            # (Cell 1 if num_cells >= 2)
   ├── final_1.png
   ├── ...
   └── shapeforge.prof          # Binary profiling trace (when --profile is passed)

Visualizing Performance Profiles
================================

When ``--profile`` is used, the generated ``.prof`` file can be analyzed using external profiling tools:

SnakeViz (Browser GUI)
----------------------

.. code-block:: bash

   pip install snakeviz
   snakeviz output_dir/shapeforge.prof

Tuna (Icicle Diagram Viewer)
----------------------------

.. code-block:: bash

   pip install tuna
   tuna output_dir/shapeforge.prof

VS Code Extension
-----------------

Install the **Python Profile Viewer** extension in VS Code and click on the ``shapeforge.prof`` file in the explorer panel.

Exit Codes
==========

* ``0``: Successful completion.
* ``1``: Runtime error, configuration validation failure, or unhandled exception.
* ``2``: Invalid command-line arguments or missing required parameters.


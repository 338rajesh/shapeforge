"""
Configuration models and validation schemas for ShapeForge.
===========================================================

Defines immutable, validated data containers for configuring ShapeForge unit cell
generation tasks. Maps directly to user YAML/JSON configuration files.

Classes
-------
MetadataConfig
    Task metadata including title, random seed, and logging verbosity.
DomainConfig
    Spatial domain shape and bounding coordinates.
ShapeConfig
    Individual inclusion phase definition with target volume fraction and size distributions.
PackingConfig
    Physical packing constraints including minimum gap buffers, periodicity, and boundary adjustments.
SolverConfig
    Spatial initialisation strategy and optimization engine hyperparameters.
ExportConfig
    Output directory, image rendering settings, and file formats.
ShapeForgeConfig
    Master top-level configuration encapsulating all subsections.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

from gbox.core.utils import Bounds2DRectangular, Validator

from .registry import ShapeRegistry
from .utils import DistributionSpec, _load_dict

ALLOWABLE_EXPORT_FORMATS: set[str] = {
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "json",
    "pkl",
    "pickle",
    "yml",
    "yaml",
    "npz",
}


# ============================================================================
# Metadata
# ============================================================================


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    """
    Task-level metadata and execution control settings.

    Parameters
    ----------
    title : str
        Human-readable title or project identifier.
    rng_seed : int
        Base seed for pseudo-random number generation ensuring reproducibility.
    log_level : str
        Logging verbosity (e.g., ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``).
    """

    title: str
    rng_seed: int
    log_level: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetadataConfig:
        """
        Construct and validate a MetadataConfig from a raw dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing ``'title'`` (str), ``'rng_seed'`` (int),
            and ``'log_level'`` (str).

        Returns
        -------
        MetadataConfig
            Validated metadata configuration instance.

        Raises
        ------
        ValueError
            If required keys are missing or invalid types are provided.
        """
        data = Validator.as_dict(
            data,
            name="metadata",
            keys=["title", "rng_seed", "log_level"],
            types=[str, int, str],
            reject_extra_keys=True,
        )
        return cls(**data)


# ============================================================================
# Domain
# ============================================================================


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """
    Spatial domain geometry specification.

    Parameters
    ----------
    shape : str
        Domain boundary type (e.g., ``'rectangle'``).
    bounds : Bounds2DRectangular
        Bounding box specifying $[x_{\\min}, y_{\\min}, x_{\\max}, y_{\\max}]$.
    """

    shape: str
    bounds: Bounds2DRectangular

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        """
        Construct and validate a DomainConfig from a raw dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with keys ``'shape'`` and ``'bounds'`` (mapping of
            ``'x_min'``, ``'y_min'``, ``'x_max'``, ``'y_max'``).

        Returns
        -------
        DomainConfig
            Validated domain configuration instance.

        Raises
        ------
        ValueError
            If bounds values are not numbers or bounding keys are missing.
        """
        data = Validator.as_dict(
            data,
            name="domain",
            keys=["shape", "bounds"],
            types=[str, dict],
            reject_extra_keys=True,
        )

        bounds = Validator.as_dict(
            data["bounds"],
            name="domain.bounds",
            keys=["x_min", "y_min", "x_max", "y_max"],
            types=[Number, Number, Number, Number],
            reject_extra_keys=True,
        )
        bounds = {k: float(v) for k, v in bounds.items()}
        bounds = Bounds2DRectangular(**bounds)

        return cls(
            shape=data["shape"],
            bounds=bounds,
        )


# ============================================================================
# Shape parameters
# ============================================================================


@dataclass(frozen=True, slots=True)
class ShapeConfig:
    """
    Configuration specification for a single inclusion shape phase.

    Parameters
    ----------
    name : str
        Registered shape identifier (case-insensitive, e.g. ``'circle'``,
        ``'ellipse'``, or a registered custom shape).
    volume_fraction : float
        Target phase volume (area) fraction in $(0, 1)$.
    params : dict[str, float | DistributionSpec]
        Shape geometric parameters, where values are either deterministic floats
        or statistical distribution specifications (e.g. ``'uniform(1.0, 2.0)'``).
    """

    name: str
    volume_fraction: float
    params: dict[str, float | DistributionSpec]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapeConfig:
        """
        Construct and validate a ShapeConfig instance from a raw dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing ``'name'``, ``'volume_fraction'``, and ``'params'``.

        Returns
        -------
        ShapeConfig
            Validated shape configuration.

        Raises
        ------
        ValueError
            If the shape name is not registered, or if ``volume_fraction`` is
            outside $(0, 1)$.
        TypeError
            If parameter values are invalid types.
        """
        data = Validator.as_dict(
            data,
            name="Shapes",
            keys=["name", "volume_fraction", "params"],
        )

        shape_name = Validator.as_string(
            data["name"], min_length=1, name="shape.name"
        ).lower()
        if not ShapeRegistry.has(shape_name):
            available = ShapeRegistry.available_shapes()
            raise ValueError(
                f"Unknown shape '{shape_name}'. Available shapes: {available}"
            )

        volume_fraction = Validator.as_float(
            data["volume_fraction"],
            low=0.0,
            high=1.0,
            closed_bounds=False,
            name="shape.volume_fraction",
        )

        Validator.as_dict(data["params"], name="shape.params")
        params: dict[str, float | DistributionSpec] = {}
        for p_name, p_value in data["params"].items():
            Validator.as_string(
                p_name, min_length=1, name=f"shape.params.{p_name}"
            )

            if isinstance(p_value, bool):
                raise TypeError(
                    f"Shape parameter '{p_name}' must be a float "
                    f"or distribution expression, but got bool"
                )

            if isinstance(p_value, (int, float)):
                params[p_name] = Validator.as_float(
                    p_value, name=f"shape.params.{p_name}"
                )
            elif isinstance(p_value, str):
                params[p_name] = DistributionSpec.from_signature(p_value)
            else:
                raise TypeError(
                    f"Shape parameter '{p_name}' must be a float "
                    f"or distribution expression, but got {type(p_value)}"
                )

        return cls(
            name=shape_name,
            volume_fraction=volume_fraction,
            params=params,
        )


# ============================================================================
# Packing
# ============================================================================


@dataclass(frozen=True, slots=True)
class PackingConfig:
    """
    Physical packing constraints and boundary conditions.

    Parameters
    ----------
    min_gap_ratio : float
        Minimum surface-to-surface buffer distance expressed as a ratio of
        each shape's equivalent radius. Must be $> 0$.
    periodicity : bool, optional
        Whether to enforce periodic boundary conditions across opposite domain
        edges (default: False).
    proj_buffer_ratio : float or None, optional
        Damping buffer ratio used when projecting shapes within non-periodic
        box boundaries (default: None).
    adjust_bounds_to_exact_vf : bool, optional
        Whether to automatically rescale the domain boundary dimensions after
        initial sampling so the actual discrete shape count matches the exact
        prescribed volume fraction (default: False).
    """

    min_gap_ratio: float
    periodicity: bool = False
    proj_buffer_ratio: float | None = None
    adjust_bounds_to_exact_vf: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackingConfig:
        """
        Construct and validate PackingConfig from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing packing configuration parameters.

        Returns
        -------
        PackingConfig
            Validated packing configuration.

        Raises
        ------
        ValueError
            If numeric ratios are $\\le 0$.
        """
        data = Validator.as_dict(
            data,
            name="Packing",
            key_type_map={
                "min_gap_ratio": float,
                "periodicity": bool,
                "proj_buffer_ratio": float,
                "adjust_bounds_to_exact_vf": bool,
            },
            reject_extra_keys=True,
        )
        data["min_gap_ratio"] = Validator.as_float(
            data["min_gap_ratio"],
            low=0.0,
            closed_bounds=False,
            name="Minimum Gap Ratio",
        )
        data["proj_buffer_ratio"] = Validator.as_float(
            data["proj_buffer_ratio"],
            low=0.0,
            closed_bounds=False,
            name="Projection Buffer Ratio",
        )
        return cls(**data)


# ============================================================================
# Solver
# ============================================================================


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """
    Spatial initialisation strategy and numerical optimization settings.

    Parameters
    ----------
    init_method : {'uniform', 'lhs', 'sobol'}
        Sampling algorithm used to initially scatter shapes across the domain.
    optimiser : dict
        Hyperparameters for the numerical solver (e.g. ``'name'``, ``'iter_max'``,
        ``'iter_memory'``, ``'epsilon'``, ``'f_tol'``).
    """

    init_method: str
    optimiser: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverConfig:
        """
        Construct and validate SolverConfig from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing ``'init_method'`` and ``'optimiser'``.

        Returns
        -------
        SolverConfig
            Validated solver configuration.

        Raises
        ------
        ValueError
            If ``init_method`` is not one of ``{'uniform', 'lhs', 'sobol'}``.
        """
        data = Validator.as_dict(
            data,
            name="solver",
            key_type_map={
                "init_method": str,
                "optimiser": dict,
            },
            reject_extra_keys=True,
        )
        Validator.has(
            data["init_method"],
            {"uniform", "lhs", "sobol"},
            name="solver.init_method",
        )

        return cls(**data)


# ============================================================================
# Export
# ============================================================================


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """
    File output and visual rendering export configuration.

    Parameters
    ----------
    output_dir : Path
        Directory on disk where generated cells, plots, and metadata are saved.
    as_array : bool
        Whether to export raw rasterized image arrays as NumPy ``.npy`` binaries.
    formats : tuple of str
        List of target export extensions (e.g. ``('png', 'json')``).
    background : int
        Grayscale background pixel value in $[0, 255]$ (0 = black).
    facecolor : int
        Grayscale foreground shape pixel value in $[0, 255]$ (255 = white).
    image_dimensions : tuple of int
        Image resolution as ``(width, height)`` in pixels.
    dpi : int
        Dots per inch for rendered vector/raster outputs.
    """

    output_dir: Path
    as_array: bool
    formats: tuple[str, ...]
    background: int
    facecolor: int
    image_dimensions: tuple[int, int]
    dpi: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportConfig:
        """
        Construct and validate ExportConfig from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary of export parameters.

        Returns
        -------
        ExportConfig
            Validated export configuration.

        Raises
        ------
        ValueError
            If export formats are unrecognized or dimensions are invalid.
        """
        data = Validator.as_dict(
            data,
            keys=[
                "output_dir",
                "as_array",
                "formats",
                "background",
                "facecolor",
                "image_dimensions",
                "dpi",
            ],
            types=[(str, Path), bool, Collection, int, int, Sequence, int],
            name="export",
        )
        data = dict(data)
        data["output_dir"] = Validator.dir_path(data["output_dir"], mkdir=True)

        invalid_formats = [
            fmt
            for fmt in data["formats"]
            if fmt not in ALLOWABLE_EXPORT_FORMATS
        ]
        if invalid_formats:
            raise ValueError(
                f"Invalid export formats: {invalid_formats}. "
                f"Allowed formats: {ALLOWABLE_EXPORT_FORMATS}"
            )
        data["formats"] = tuple(data["formats"])

        data["background"] = Validator.as_int(
            data["background"],
            low=0,
            high=255,
            name="export.background",
        )
        data["facecolor"] = Validator.as_int(
            data["facecolor"],
            low=0,
            high=255,
            name="export.facecolor",
        )

        image_dimensions = Validator.as_sequence(
            data["image_dimensions"],
            name="export.image_dimensions",
            ele_type=int,
            length=2,
        )

        data["image_dimensions"] = (
            Validator.as_int(
                image_dimensions[0],
                low=1,
                name="export.image_dimensions[0]",
            ),
            Validator.as_int(
                image_dimensions[1],
                low=1,
                name="export.image_dimensions[1]",
            ),
        )

        data["dpi"] = Validator.as_int(
            data["dpi"],
            low=1,
            name="export.dpi",
        )

        return cls(**data)


# ============================================================================
# Complete configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ShapeForgeConfig:
    """
    Master configuration schema for ShapeForge cell generation jobs.

    Encapsulates all task parameters, domain boundaries, multi-phase inclusion
    definitions, packing rules, solver options, and export preferences.

    Parameters
    ----------
    num_cells : int
        Number of statistically independent microstructures to generate.
    metadata : MetadataConfig
        Task title, random seed, and logging level.
    domain : DomainConfig
        Domain shape and spatial coordinates.
    shapes : tuple of ShapeConfig
        Inclusion phase specifications.
    packing : PackingConfig
        Overlap buffers, periodicity, and domain volume adjustments.
    solver : SolverConfig
        Initialisation sampler and NM-SPG optimization parameters.
    export : ExportConfig
        Output directory and export formats.
    custom_shapes : tuple of str, optional
        Paths to external Python files defining custom shape classes (default: ()).
    """

    num_cells: int
    metadata: MetadataConfig
    domain: DomainConfig
    shapes: tuple[ShapeConfig, ...]
    packing: PackingConfig
    solver: SolverConfig
    export: ExportConfig
    custom_shapes: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], base_dir: Path | None = None
    ) -> ShapeForgeConfig:
        """
        Parse, validate, and construct a ShapeForgeConfig from a dictionary.

        If custom shape files are declared under ``custom_shapes``, they are
        resolved relative to ``base_dir`` and automatically imported into
        :class:`~shapeforge.registry.ShapeRegistry`.

        Parameters
        ----------
        data : dict[str, Any]
            Top-level configuration dictionary.
        base_dir : Path, optional
            Base directory for resolving relative custom shape file paths.

        Returns
        -------
        ShapeForgeConfig
            Complete validated configuration instance.

        Raises
        ------
        ValueError
            If total volume fraction exceeds 1.0 or if required sections are missing.
        """
        data_copy = dict(data)
        custom_shapes_raw = data_copy.pop("custom_shapes", None)
        custom_shapes_list: list[str] = []
        if custom_shapes_raw is not None:
            raw_seq = Validator.as_sequence(
                custom_shapes_raw, name="custom_shapes"
            )
            for item in raw_seq:
                p_str = Validator.as_string(
                    item, min_length=1, name="custom_shapes item"
                )
                custom_shapes_list.append(p_str)
                p_obj = Path(p_str)
                if not p_obj.is_absolute() and base_dir is not None:
                    p_obj = (base_dir / p_obj).resolve()
                ShapeRegistry.load_file(p_obj)

        data_validated = Validator.as_dict(
            data_copy,
            name="ShapeForge Config",
            key_type_map={
                "num_cells": int,
                "metadata": dict,
                "domain": dict,
                "shapes": list,
                "packing": dict,
                "solver": dict,
                "export": dict,
            },
            reject_extra_keys=True,
        )
        num_cells = Validator.as_int(
            data_validated["num_cells"], low=1, name="Number of cells"
        )
        metadata = MetadataConfig.from_dict(data_validated["metadata"])
        solver = SolverConfig.from_dict(data_validated["solver"])
        export = ExportConfig.from_dict(data_validated["export"])
        domain = DomainConfig.from_dict(data_validated["domain"])
        packing = PackingConfig.from_dict(data_validated["packing"])
        raw_shapes = Validator.as_sequence(
            data_validated["shapes"], min_length=1
        )
        shapes = tuple(ShapeConfig.from_dict(shape) for shape in raw_shapes)
        total_volume_fraction = sum(shape.volume_fraction for shape in shapes)
        if total_volume_fraction > 1.0:
            raise ValueError(
                "Total shape volume fraction must not exceed 1.0, "
                f"but got {total_volume_fraction}"
            )

        return cls(
            num_cells=num_cells,
            metadata=metadata,
            domain=domain,
            shapes=shapes,
            packing=packing,
            solver=solver,
            export=export,
            custom_shapes=tuple(custom_shapes_list),
        )

    @classmethod
    def from_(
        cls, source: dict[str, Any] | str | Path, base_dir: Path | None = None
    ) -> ShapeForgeConfig:
        """
        Load a ShapeForgeConfig from a dictionary, YAML file, or JSON file.

        Parameters
        ----------
        source : dict, str, or Path
            Configuration mapping or filesystem path to YAML/JSON file.
        base_dir : Path, optional
            Base directory for resolving relative paths. If omitted and `source`
            is a file path, the parent directory of `source` is used.

        Returns
        -------
        ShapeForgeConfig
            Constructed and validated configuration.

        Raises
        ------
        TypeError
            If `source` is neither a dict nor a string/Path.
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source).resolve()
            data = _load_dict(source_path)
            if base_dir is None:
                base_dir = source_path.parent
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError(
                f"Invalid type '{type(source).__name__}' of source. "
                "Expecting a str or pathlib.Path or dict"
            )
        return cls.from_dict(data, base_dir=base_dir)

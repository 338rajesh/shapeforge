from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path

from gbox.core.utils import Bounds2DRectangular, Validator
from gbox.shapes import SHAPES_2D_MAPPING as GBOX_2D_SHAPES_MAPPING

from .utils import DistributionSpec, _load_dict

ALLOWABLE_EXPORT_FORMATS = {"png", "jpg", "pdf", "json", "pkl", "yml", "npz"}


# ============================================================================
# Metadata
# ============================================================================


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    title: str
    rng_seed: int
    log_level: str

    @classmethod
    def from_dict(cls, data: dict) -> MetadataConfig:
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
    shape: str
    bounds: Bounds2DRectangular

    @classmethod
    def from_dict(cls, data: dict) -> DomainConfig:
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
    name: str
    volume_fraction: float
    params: dict[str, float | DistributionSpec]

    @classmethod
    def from_dict(cls, data: dict) -> ShapeConfig:
        data = Validator.as_dict(
            data,
            name="Shapes",
            keys=["name", "volume_fraction", "params"],
        )

        shape_name = Validator.as_string(
            data["name"], min_length=1, name="shape.name"
        ).lower()
        Validator.has(shape_name, list(GBOX_2D_SHAPES_MAPPING.keys()))

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
    min_gap_ratio: float
    periodicity: bool = False
    proj_buffer_ratio: float | None = None
    adjust_bounds_to_exact_vf: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> PackingConfig:
        data = Validator.as_dict(
            data,
            name="Packing",
            keys=[
                "min_gap_ratio",
                "periodicity",
                "proj_buffer_ratio",
                "adjust_bounds_to_exact_vf",
            ],
            types=[float, bool, float, bool],
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
    init_method: str
    optimiser: dict

    @classmethod
    def from_dict(cls, data: dict) -> SolverConfig:
        data = Validator.as_dict(
            data,
            name="solver",
            required_keys=["init_method", "optimiser"],
            types=[str, dict],
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
    output_dir: Path
    as_array: bool
    formats: tuple[str, ...]
    background: int
    facecolor: int
    image_dimensions: tuple[int, int]
    dpi: int

    @classmethod
    def from_dict(cls, data: dict) -> ExportConfig:
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
            types=[str, bool, Collection, int, int, Sequence, int],
            name="export",
        )

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
    num_cells: int
    metadata: MetadataConfig
    domain: DomainConfig
    shapes: tuple[ShapeConfig, ...]
    packing: PackingConfig
    solver: SolverConfig
    export: ExportConfig

    @classmethod
    def from_dict(cls, data: dict) -> ShapeForgeConfig:
        data = Validator.as_dict(
            data,
            name="ShapeForge Config",
            required_keys=[
                "num_cells",
                "metadata",
                "domain",
                "shapes",
                "packing",
                "solver",
                "export",
            ],
            reject_extra_keys=True,
        )
        num_cells = Validator.as_int(
            data["num_cells"], low=1, name="Number of cells"
        )
        metadata = MetadataConfig.from_dict(data["metadata"])
        solver = SolverConfig.from_dict(data["solver"])
        export = ExportConfig.from_dict(data["export"])
        domain = DomainConfig.from_dict(data["domain"])
        packing = PackingConfig.from_dict(data["packing"])
        raw_shapes = Validator.as_sequence(data["shapes"], min_length=1)
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
        )

    @classmethod
    def from_(cls, source: dict | str | Path) -> ShapeForgeConfig:
        if isinstance(source, (str, Path)):
            data = _load_dict(source)
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError(
                f"Invalid type '{type(source).__name__}' of source."
                "Expecting a str or pathlib.Path or dict"
            )
        return cls.from_dict(data)

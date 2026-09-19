from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gbox.core.utils import Bounds2DRectangular, Validator

# ============================================================================
# Helpers
# ============================================================================


def _require_dict(value: Any, name: str) -> dict:
    return Validator.dict(value, name=name)


def _required_keys(
    value: dict,
    keys: list[str],
    name: str,
    *,
    reject_extra_keys: bool = False,
) -> dict:
    return Validator.dict(
        value,
        keys=keys,
        name=name,
        reject_extra_keys=reject_extra_keys,
    )


def _string(
    value: Any,
    *,
    name: str,
    allowed: set[str] | None = None,
) -> str:
    Validator.is_type(value, str, name=name)

    if allowed is not None and value not in allowed:
        raise ValueError(
            f"Given value '{name}' must be one of {sorted(allowed)}, "
            f"but got {value!r}"
        )

    return value


def _bool(value: Any, *, name: str) -> bool:
    return Validator.is_type(value, bool, name=name) and value


# ============================================================================
# Metadata
# ============================================================================


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    project_name: str
    rng_seed: int
    log_level: str
    num_cells: int

    @classmethod
    def from_dict(cls, data: dict) -> MetadataConfig:
        _required_keys(
            data,
            ["project_name", "rng_seed", "log_level", "num_cells"],
            "metadata",
        )

        project_name = _string(
            data["project_name"],
            name="metadata.project_name",
        )

        rng_seed = Validator.int(
            data["rng_seed"],
            low=0,
            name="metadata.rng_seed",
        )

        log_level = _string(
            data["log_level"],
            name="metadata.log_level",
            allowed={"debug", "info", "warning", "error", "critical"},
        )

        num_cells = Validator.int(
            data["num_cells"],
            low=1,
            name="metadata.num_cells",
        )

        return cls(
            project_name=project_name,
            rng_seed=rng_seed,
            log_level=log_level,
            num_cells=num_cells,
        )


# ============================================================================
# Domain
# ============================================================================


@dataclass(frozen=True, slots=True)
class DomainConfig:
    shape: str
    bounds: Bounds2DRectangular

    @classmethod
    def from_dict(cls, data: dict) -> DomainConfig:
        _required_keys(
            data,
            ["shape", "bounds"],
            "domain",
        )

        shape = _string(
            data["shape"],
            name="domain.shape",
            allowed={"rectangle"},
        )

        bounds_data = _required_keys(
            data["bounds"],
            ["x_min", "y_min", "x_max", "y_max"],
            "domain.bounds",
        )

        bounds = Bounds2DRectangular(
            x_min=Validator.float(
                bounds_data["x_min"],
                name="domain.bounds.x_min",
            ),
            y_min=Validator.float(
                bounds_data["y_min"],
                name="domain.bounds.y_min",
            ),
            x_max=Validator.float(
                bounds_data["x_max"],
                name="domain.bounds.x_max",
            ),
            y_max=Validator.float(
                bounds_data["y_max"],
                name="domain.bounds.y_max",
            ),
        )

        return cls(
            shape=shape,
            bounds=bounds,
        )


# ============================================================================
# Shape parameters
# ============================================================================


@dataclass(frozen=True, slots=True)
class ShapeConfig:
    name: str
    volume_fraction: float
    params: dict[str, float | str]

    @classmethod
    def from_dict(cls, data: dict) -> ShapeConfig:
        _required_keys(
            data,
            ["name", "volume_fraction", "params"],
            "shape",
        )

        name = _string(
            data["name"],
            name="shape.name",
        )

        volume_fraction = Validator.float(
            data["volume_fraction"],
            low=0.0,
            high=1.0,
            closed_bounds=False,
            name="shape.volume_fraction",
        )

        params_data = _require_dict(
            data["params"],
            "shape.params",
        )

        params: dict[str, float | str] = {}

        for parameter_name, value in params_data.items():
            _string(
                parameter_name,
                name="shape parameter name",
            )

            if isinstance(value, bool):
                raise TypeError(
                    f"Shape parameter '{parameter_name}' must be a float "
                    f"or distribution expression, but got bool"
                )

            if isinstance(value, (int, float)):
                params[parameter_name] = Validator.float(
                    value,
                    name=f"shape.params.{parameter_name}",
                )

            elif isinstance(value, str):
                params[parameter_name] = value

            else:
                raise TypeError(
                    f"Shape parameter '{parameter_name}' must be a float "
                    f"or distribution expression, but got {type(value)}"
                )

        return cls(
            name=name,
            volume_fraction=volume_fraction,
            params=params,
        )


# ============================================================================
# Optimiser
# ============================================================================


@dataclass(frozen=True, slots=True)
class OptimiserConfig:
    name: str
    max_iter: int
    epsilon: float

    @classmethod
    def from_dict(cls, data: dict) -> OptimiserConfig:
        _required_keys(
            data,
            ["name", "max_iter", "epsilon"],
            "solver.optimiser",
        )

        name = _string(
            data["name"],
            name="solver.optimiser.name",
        )

        max_iter = Validator.int(
            data["max_iter"],
            low=1,
            name="solver.optimiser.max_iter",
        )

        epsilon = Validator.float(
            data["epsilon"],
            low=0.0,
            closed_bounds=False,
            name="solver.optimiser.epsilon",
        )

        return cls(
            name=name,
            max_iter=max_iter,
            epsilon=epsilon,
        )


# ============================================================================
# Packing
# ============================================================================


@dataclass(frozen=True, slots=True)
class PackingConfig:
    periodic: bool
    min_gap_ratio: float
    proj_buffer_ratio: float
    adjust_bounds_to_exact_vf: bool

    @classmethod
    def from_dict(cls, data: dict) -> PackingConfig:
        _required_keys(
            data,
            [
                "periodic",
                "min_gap_ratio",
                "proj_buffer_ratio",
                "adjust_bounds_to_exact_vf",
            ],
            "solver.packing",
        )

        periodic = _bool(
            data["periodic"],
            name="solver.packing.periodic",
        )

        min_gap_ratio = Validator.float(
            data["min_gap_ratio"],
            low=0.0,
            name="solver.packing.min_gap_ratio",
        )

        proj_buffer_ratio = Validator.float(
            data["proj_buffer_ratio"],
            low=0.0,
            name="solver.packing.proj_buffer_ratio",
        )

        adjust_bounds_to_exact_vf = _bool(
            data["adjust_bounds_to_exact_vf"],
            name="solver.packing.adjust_bounds_to_exact_vf",
        )

        return cls(
            periodic=periodic,
            min_gap_ratio=min_gap_ratio,
            proj_buffer_ratio=proj_buffer_ratio,
            adjust_bounds_to_exact_vf=adjust_bounds_to_exact_vf,
        )


# ============================================================================
# Solver
# ============================================================================


@dataclass(frozen=True, slots=True)
class SolverConfig:
    init_method: str
    optimiser: OptimiserConfig
    packing: PackingConfig

    @classmethod
    def from_dict(cls, data: dict) -> SolverConfig:
        _required_keys(
            data,
            ["init_method", "optimiser", "packing"],
            "solver",
        )

        init_method = _string(
            data["init_method"],
            name="solver.init_method",
            allowed={"uniform", "lhs", "sobol"},
        )

        optimiser = OptimiserConfig.from_dict(
            _require_dict(
                data["optimiser"],
                "solver.optimiser",
            )
        )

        packing = PackingConfig.from_dict(
            _require_dict(
                data["packing"],
                "solver.packing",
            )
        )

        return cls(
            init_method=init_method,
            optimiser=optimiser,
            packing=packing,
        )


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
        _required_keys(
            data,
            [
                "output_dir",
                "as_array",
                "formats",
                "background",
                "facecolor",
                "image_dimensions",
                "dpi",
            ],
            "export",
        )

        output_dir = Validator.file_path(data["output_dir"])

        as_array = _bool(
            data["as_array"],
            name="export.as_array",
        )

        formats = Validator.sequence(
            data["formats"],
            name="export.formats",
            ele_type=str,
            length=None,
        )

        allowed_formats = {
            "png",
            "jpg",
            "pdf",
            "json",
            "pkl",
            "yml",
            "npz",
        }

        invalid_formats = [
            fmt for fmt in formats if fmt not in allowed_formats
        ]

        if invalid_formats:
            raise ValueError(
                f"Invalid export formats: {invalid_formats}. "
                f"Allowed formats: {sorted(allowed_formats)}"
            )

        background = Validator.int(
            data["background"],
            low=0,
            high=255,
            name="export.background",
        )

        facecolor = Validator.int(
            data["facecolor"],
            low=0,
            high=255,
            name="export.facecolor",
        )

        image_dimensions = Validator.sequence(
            data["image_dimensions"],
            name="export.image_dimensions",
            ele_type=int,
            length=2,
        )

        image_dimensions = (
            Validator.int(
                image_dimensions[0],
                low=1,
                name="export.image_dimensions[0]",
            ),
            Validator.int(
                image_dimensions[1],
                low=1,
                name="export.image_dimensions[1]",
            ),
        )

        dpi = Validator.int(
            data["dpi"],
            low=1,
            name="export.dpi",
        )

        return cls(
            output_dir=output_dir,
            as_array=as_array,
            formats=tuple(formats),
            background=background,
            facecolor=facecolor,
            image_dimensions=image_dimensions,
            dpi=dpi,
        )


# ============================================================================
# Complete configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ShapeForgeConfig:
    metadata: MetadataConfig
    domain: DomainConfig
    shapes: tuple[ShapeConfig, ...]
    solver: SolverConfig
    export: ExportConfig

    @classmethod
    def from_dict(cls, data: dict) -> ShapeForgeConfig:
        _required_keys(
            data,
            [
                "metadata",
                "domain",
                "shapes",
                "solver",
                "export",
            ],
            "configuration",
        )

        metadata = MetadataConfig.from_dict(
            _require_dict(data["metadata"], "metadata")
        )

        domain = DomainConfig.from_dict(
            _require_dict(data["domain"], "domain")
        )

        raw_shapes = Validator.sequence(
            data["shapes"],
            name="shapes",
        )

        if len(raw_shapes) == 0:
            raise ValueError("At least one shape must be specified")

        shapes = tuple(
            ShapeConfig.from_dict(_require_dict(shape, f"shapes[{index}]"))
            for index, shape in enumerate(raw_shapes)
        )

        total_volume_fraction = sum(shape.volume_fraction for shape in shapes)

        if total_volume_fraction > 1.0:
            raise ValueError(
                "Total shape volume fraction must not exceed 1.0, "
                f"but got {total_volume_fraction}"
            )

        solver = SolverConfig.from_dict(
            _require_dict(data["solver"], "solver")
        )

        export = ExportConfig.from_dict(
            _require_dict(data["export"], "export")
        )

        return cls(
            metadata=metadata,
            domain=domain,
            shapes=shapes,
            solver=solver,
            export=export,
        )

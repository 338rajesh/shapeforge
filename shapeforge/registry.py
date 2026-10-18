from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import gbox as gb


class ShapeRegistry:
    """Central registry for 2D shapes supported by ShapeForge.

    Starts pre-populated with built-in gbox shapes ('circle', 'ellipse').
    Users can register custom subclasses of ``gb.Shape2D`` either via code:
        ShapeRegistry.register("capsule", Capsule)
    via decorator:
        @register_shape("capsule")
        class Capsule(gb.Shape2D): ...
    or by dynamically loading from Python files:
        ShapeRegistry.load_file("path/to/my_shapes.py")
    """

    _shapes_2d: dict[str, type[gb.Shape2D]] = {}  # noqa: RUF012
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._initialized:
            cls.reset()

    @classmethod
    def reset(cls) -> None:
        """Reset registry back to default built-in gbox shapes."""
        cls._shapes_2d = dict(gb.shapes.SHAPES_2D_MAPPING)
        cls._initialized = True

    @classmethod
    def register(
        cls,
        name: str | None = None,
        shape_cls: type[gb.Shape2D] | None = None,
    ) -> Callable[[type[gb.Shape2D]], type[gb.Shape2D]] | type[gb.Shape2D]:
        """Register a Shape2D subclass in the registry.

        Parameters
        ----------
        name : str, optional
            Name/identifier for the shape (case-insensitive).
        shape_cls : type[gb.Shape2D], optional
            The Shape2D subclass to register.
        """
        cls._ensure_initialized()

        def _do_register(
            name_str: str, s_cls: type[gb.Shape2D]
        ) -> type[gb.Shape2D]:
            if not isinstance(s_cls, type) or not issubclass(s_cls, gb.Shape2D):
                raise TypeError(
                    f"Class '{s_cls}' must be a subclass of gbox.Shape2D"
                )
            if (
                not hasattr(s_cls, "from_params")
                or s_cls.from_params is gb.Shape2D.from_params
                or not callable(s_cls.from_params)
            ):
                raise ValueError(
                    f"Shape class '{s_cls.__name__}' must implement a callable 'from_params' method"
                )
            key = name_str.strip().lower()
            if not key:
                raise ValueError("Shape name cannot be empty")
            cls._shapes_2d[key] = s_cls
            return s_cls

        if shape_cls is not None:
            actual_name = name or shape_cls.__name__.lower()
            return _do_register(actual_name, shape_cls)

        def decorator(s_cls: type[gb.Shape2D]) -> type[gb.Shape2D]:
            actual_name = name or s_cls.__name__.lower()
            return _do_register(actual_name, s_cls)

        if isinstance(name, type) and issubclass(name, gb.Shape2D):
            # Used as @ShapeRegistry.register without arguments
            s_cls = name
            return _do_register(s_cls.__name__.lower(), s_cls)

        return decorator

    @classmethod
    def get(cls, name: str) -> type[gb.Shape2D]:
        """Get registered shape class by name (case-insensitive)."""
        cls._ensure_initialized()
        key = name.strip().lower()
        if key not in cls._shapes_2d:
            avail = cls.available_shapes()
            raise KeyError(
                f"Unknown shape '{name}'. Available shapes: {avail}"
            )
        return cls._shapes_2d[key]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if shape name is registered (case-insensitive)."""
        cls._ensure_initialized()
        return name.strip().lower() in cls._shapes_2d

    @classmethod
    def available_shapes(cls) -> list[str]:
        """Return sorted list of available shape names."""
        cls._ensure_initialized()
        return sorted(cls._shapes_2d.keys())

    @classmethod
    def load_file(cls, path: str | Path) -> list[str]:
        """Dynamically load and register custom shapes from a Python file.

        Parameters
        ----------
        path : str or Path
            Path to the Python file containing Shape2D subclass(es).

        Returns
        -------
        list[str]
            List of registered shape names loaded from the file.
        """
        cls._ensure_initialized()
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Custom shape file not found: {file_path}"
            )

        module_name = (
            f"shapeforge_custom_{file_path.stem}_{abs(hash(str(file_path)))}"
        )
        spec = importlib.util.spec_from_file_location(
            module_name, str(file_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from '{file_path}'")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Inspect classes defined in this module
        registered_names: list[str] = []
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, gb.Shape2D)
                and obj is not gb.Shape2D
                and getattr(obj, "__module__", None) == module_name
            ):
                shape_name = obj.__name__.lower()
                # If not registered via decorator during import, auto-register
                if not any(v is obj for v in cls._shapes_2d.values()):
                    cls.register(shape_name, obj)
                # Collect all registry keys pointing to this class
                for k, v in cls._shapes_2d.items():
                    if v is obj and k not in registered_names:
                        registered_names.append(k)

        return registered_names


def register_shape(
    name: str | None = None,
    shape_cls: type[gb.Shape2D] | None = None,
):
    """Decorator or helper to register a custom shape in ShapeRegistry."""
    return ShapeRegistry.register(name, shape_cls)

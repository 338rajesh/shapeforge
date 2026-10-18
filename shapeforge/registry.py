"""
Dynamic shape registry and custom shape loader.
===============================================

Provides the central registry :class:`ShapeRegistry` and convenience decorator
:func:`register_shape` allowing users to define, register, and import custom 2D
geometric inclusion types extending :class:`gbox.Shape2D`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gbox as gb


class ShapeRegistry:
    """
    Central registry for 2D inclusion shape types supported by ShapeForge.

    Starts pre-populated with built-in ``gbox`` shapes (``'circle'``, ``'ellipse'``).
    Users can register custom subclasses of :class:`gbox.Shape2D` via code:

    >>> ShapeRegistry.register("capsule", Capsule)

    via decorator:

    >>> @register_shape("capsule")
    ... class Capsule(gb.Shape2D):
    ...     pass

    or dynamically from external Python files:

    >>> ShapeRegistry.load_file("my_shapes.py")

    Attributes
    ----------
    _shapes_2d : dict[str, type[gb.Shape2D]]
        Internal mapping of normalized shape names to class objects.
    _initialized : bool
        Flag indicating whether defaults have been loaded.
    """

    _shapes_2d: dict[str, type[gb.Shape2D]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in shape classes are populated in the registry."""
        if not cls._initialized:
            cls.reset()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry back to default built-in ``gbox`` shapes.

        Clears any dynamically added custom shapes and restores standard
        built-in shape mappings (``'circle'``, ``'ellipse'``).
        """
        cls._shapes_2d = dict(gb.shapes.SHAPES_2D_MAPPING)
        cls._initialized = True

    @classmethod
    def register(
        cls,
        name: str | type[gb.Shape2D] | None = None,
        shape_cls: type[gb.Shape2D] | None = None,
    ) -> Callable[[type[gb.Shape2D]], type[gb.Shape2D]] | type[gb.Shape2D]:
        """
        Register a :class:`gbox.Shape2D` subclass in the registry.

        Can be called as a direct function:
        ``ShapeRegistry.register('my_shape', MyShape)``
        or as a decorator:
        ``@ShapeRegistry.register('my_shape')`` or ``@ShapeRegistry.register``.

        Parameters
        ----------
        name : str or type[gb.Shape2D] or None, optional
            Identifier string for the shape (case-insensitive). If omitted when
            used as a decorator, defaults to the lowercase class name.
        shape_cls : type[gb.Shape2D], optional
            The :class:`gbox.Shape2D` subclass to register.

        Returns
        -------
        Callable or type[gb.Shape2D]
            The registered class or a decorator function returning the class.

        Raises
        ------
        TypeError
            If ``shape_cls`` is not a subclass of :class:`gbox.Shape2D`.
        ValueError
            If the shape class does not implement a callable ``from_params`` method
            or if the name is an empty string.

        Examples
        --------
        >>> from shapeforge.registry import ShapeRegistry
        >>> @ShapeRegistry.register("custom_disk")
        ... class CustomDisk(gb.Circle):
        ...     @classmethod
        ...     def from_params(cls, **params):
        ...         return cls(radius=params.get("r", 1.0))
        >>> ShapeRegistry.has("custom_disk")
        True
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
            actual_name = str(name) if name is not None else shape_cls.__name__.lower()
            return _do_register(actual_name, shape_cls)

        def decorator(s_cls: type[gb.Shape2D]) -> type[gb.Shape2D]:
            actual_name = str(name) if name is not None else s_cls.__name__.lower()
            return _do_register(actual_name, s_cls)

        if isinstance(name, type) and issubclass(name, gb.Shape2D):
            s_cls = name
            return _do_register(s_cls.__name__.lower(), s_cls)

        return decorator

    @classmethod
    def get(cls, name: str) -> type[gb.Shape2D]:
        """
        Retrieve a registered shape class by name (case-insensitive).

        Parameters
        ----------
        name : str
            Shape identifier.

        Returns
        -------
        type[gb.Shape2D]
            Class object associated with the identifier.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
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
        """
        Check whether a shape name is registered (case-insensitive).

        Parameters
        ----------
        name : str
            Shape identifier to check.

        Returns
        -------
        bool
            True if registered, False otherwise.
        """
        cls._ensure_initialized()
        return name.strip().lower() in cls._shapes_2d

    @classmethod
    def available_shapes(cls) -> list[str]:
        """
        Return a sorted list of all currently available shape names.

        Returns
        -------
        list of str
            Sorted list of registered shape identifiers.
        """
        cls._ensure_initialized()
        return sorted(cls._shapes_2d.keys())

    @classmethod
    def load_file(cls, path: str | Path) -> list[str]:
        """
        Dynamically import and register custom shape classes from a Python file.

        Executes the file in an isolated namespace and auto-registers any
        :class:`gbox.Shape2D` subclasses defined within it.

        Parameters
        ----------
        path : str or Path
            Path to the Python file containing custom shape definitions.

        Returns
        -------
        list of str
            List of registered shape identifiers loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist on disk.
        ImportError
            If module specification cannot be created or loaded.
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
    name: str | type[gb.Shape2D] | None = None,
    shape_cls: type[gb.Shape2D] | None = None,
) -> Callable[[type[gb.Shape2D]], type[gb.Shape2D]] | type[gb.Shape2D]:
    """
    Decorator or helper function to register a custom shape in :class:`ShapeRegistry`.

    Parameters
    ----------
    name : str or type[gb.Shape2D] or None, optional
        Shape name/identifier. If omitted, uses lowercase class name.
    shape_cls : type[gb.Shape2D], optional
        The shape class being registered.

    Returns
    -------
    Callable or type[gb.Shape2D]
        The registered class or decorator.

    Examples
    --------
    >>> from shapeforge import register_shape
    >>> import gbox as gb
    >>> @register_shape("star")
    ... class Star(gb.Shape2D):
    ...     @classmethod
    ...     def from_params(cls, **params):
    ...         return cls()
    """
    return ShapeRegistry.register(name, shape_cls)

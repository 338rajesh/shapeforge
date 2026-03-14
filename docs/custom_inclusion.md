# Guidelines for custom inclusion shapes in Shapeforge

- The shape must subclass `gbox.GShape2D` or `gbox.GShape3D` from the `gbox` library.
- It must implement the following methods:
  - `volume(thickness: float = 1.0) -> float`: Returns the volume (or area for 2D shapes) of the shape.
  - `from_params(positional_params: Dict[str, float], size_params: Dict[str, float]) -> Self`: A class method that creates an instance of the shape from given positional and size parameters.
  - `union_of_spheres()` or `union_of_circles()`: Returns a list of spheres (or circles for 2D shapes) that approximate the shape.

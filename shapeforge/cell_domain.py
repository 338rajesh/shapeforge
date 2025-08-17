from typing import Sequence

import gbox as gb

from .utils import _validate_dict


class CellDomain:
    """
    It represents a domain for a cell, without any inclusions. It can be
    considered as a host for inclusions or other geometrical features, but it
    does not contain any by itself.
    """

    def __init__(self, bounds: Sequence):
        self._bounds = bounds
        self.domain = gb.BoundingBox(
            lower_bound=(bounds[0], bounds[1]),
            upper_bound=(bounds[2], bounds[3]),
        )

    @property
    def domain_bounds(self):
        return self._bounds

    @property
    def x_bounds(self) -> tuple[float, float]:
        """
        Get the x bounds of the cell domain.
        """
        return self._bounds[0], self._bounds[2]

    @property
    def y_bounds(self) -> tuple[float, float]:
        """
        Get the y bounds of the cell domain.
        """
        return self._bounds[1], self._bounds[3]

    @property
    def cell_volume(self) -> float:
        """
        Calculate the volume of the cell domain.
        """
        return self.domain.volume

    @classmethod
    def from_config(cls, config: dict) -> "CellDomain":
        """
        Create a CellDomain instance from a configuration dictionary. It is
        expected that the configuration contains `shape` and `bounds` keys
        that define the shape of the cell domain and its bounds, respectively.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing bounds and other parameters.

        Returns
        -------
        CellDomain
            An instance of CellDomain initialized with the provided bounds.
        """
        _validate_dict(config, ["shape", "bounds"])
        cell_shape = config["shape"]
        cell_bounds = config["bounds"]

        if cell_shape != "rectangle":
            raise ValueError(
                f"Unsupported cell shape '{cell_shape}'. "
                f"Only 'rectangle' is supported."
            )

        _validate_dict(cell_bounds, ["x_min", "y_min", "x_max", "y_max"])
        x_min = cell_bounds["x_min"]
        y_min = cell_bounds["y_min"]
        x_max = cell_bounds["x_max"]
        y_max = cell_bounds["y_max"]

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                "Invalid bounds: x_min must be less than x_max and "
                "y_min must be less than y_max."
            )

        return cls(bounds=[x_min, y_min, x_max, y_max])

from typing import List, Optional

import gbox as gb
import numpy as np

from .cell_domain import CellDomain
from .utils import _validate_dict
from .base import DistributionSampler


class InclusionSampler:
    def __init__(
        self,
        shape: str | gb.GShape,
        spatial_param_sampler: DistributionSampler,
        size_param_sampler: DistributionSampler,
    ):
        g_shape = getattr(gb, shape) if isinstance(shape, str) else shape

        if not issubclass(g_shape, gb.GShape):
            raise ValueError(
                f"Unsupported shape: {shape}, must be a subclass of gb.GShape"
            )
        # check if g_shape has a method called from_params
        if not callable(getattr(g_shape, "from_params")):
            raise ValueError(
                f"Unsupported shape: {shape}, must have a from_params method"
            )

        self.shape: gb.GShape = g_shape
        self.spatial_param_sampler = spatial_param_sampler
        self.size_param_sampler = size_param_sampler

    def __call__(self) -> gb.GShape:
        pos_params = self.spatial_param_sampler.sample()
        size_params = self.size_param_sampler.sample()
        return self.shape.from_params(
            positional_params=pos_params, size_params=size_params
        )


def initialise_inclusions(
    cell_domain: CellDomain,
    config: dict | list[dict],
    uns: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, List[gb.CirclesArray | gb.GShape]]:
    """
    It is the main function to initialize inclusions based on the provided
    configuration. It iterates through the configuration list, extracting
    the shape, volume fraction, and parameters for each inclusion type.
    It then calls the appropriate function to generate and initialize the
    inclusions based on the shape specified in the configuration.

    Parameters
    ----------
    cell_domain : CellDomain
        The cell domain in which the inclusions will be placed.
    config : list[dict] | dict
        A single dictionary or a list of dictionaries, each containing
        configuration for a specific inclusion type. Each dictionary should
        have the following keys:
        - `shape`: The shape of the inclusion (e.g., "circle", "ellipse").
        - `volume_fraction`: The volume fraction of the inclusion.
        - `parameters`: Additional parameters specific to the inclusion shape.
    uns : bool, optional
        If True, the function will return inclusions as an
        union of n-sphere representation or as a CirclesArray.

    Returns
    -------
    List[Inclusion | gb.CirclesArray]
        A list of inclusions initialized based on the configuration.

    """
    if not isinstance(config, (list, dict)):
        raise TypeError(
            "Configuration must be a dictionary or a list of dictionaries."
        )

    if isinstance(config, dict):
        config = [config]

    # xc and yc distributions are not specified in the config,
    x_min, x_max = cell_domain.x_bounds
    y_min, y_max = cell_domain.y_bounds
    xy_init_config = {
        "xc": {
            "distribution": {
                "name": "uniform",
                "loc": x_min,
                "scale": x_max - x_min,
            }
        },
        "yc": {
            "distribution": {
                "name": "uniform",
                "loc": y_min,
                "scale": y_max - y_min,
            }
        },
        "major_axis_angle": {
            "distribution": {
                "name": "uniform",
                "loc": 0.0,
                "scale": 2.0 * np.pi,
            }
        },
    }
    xy_sampler = DistributionSampler(xy_init_config, rng=rng)

    initialised_inclusions: dict[str, List[gb.CirclesArray | gb.GShape]] = {}
    for ith_incl_config in config:
        shape, vf, params = _validate_dict(
            ith_incl_config,
            keys=["shape", "volume_fraction", "parameters"],
            val_types=[str, float, dict],
            ret_val=True,
        )
        incl_sampler = InclusionSampler(
            shape=shape,
            spatial_param_sampler=xy_sampler,
            size_param_sampler=DistributionSampler(params, rng=rng),
        )
        required_volume = cell_domain.cell_volume * vf
        cumulative_volume = 0.0
        generated_inclusions = []
        while cumulative_volume < required_volume:
            a_inclusion = incl_sampler()
            generated_inclusions.append(a_inclusion)
            cumulative_volume += a_inclusion.volume()

        # make uns representation
        if uns:
            raise NotImplementedError(
                "Union of n-sphere representation is not implemented yet."
            )

        initialised_inclusions[shape] = generated_inclusions
    return initialised_inclusions

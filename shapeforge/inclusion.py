from typing import List, Optional, Union

import gbox as gb
import numpy as np

from .cell_domain import CellDomain
from .utils import _validate_dict
from .base import DistributionSampler

Inclusion = Union[
    gb.Circle,
    gb.Ellipse,
]


def generate_and_initialise_inclusions(
    required_volume: float,
    shape: str,
    param_sampler: DistributionSampler,
    spatial_param_sampler: Optional[DistributionSampler] = None,
    uns: bool = False,
) -> List[gb.CirclesArray | Inclusion]:
    """
    Generate and initialize inclusions of a specific shape,
    in the given domain, based on the provided configuration dictionary.

    Parameters
    ----------
    domain : CellDomain
        The cell domain in which the inclusions will be placed.
    incl_config : dict
        A dictionary containing the configuration for the inclusion.
        It should have the following keys:
        - `shape`: The shape of the inclusion (e.g., "circle", "ellipse").
        - `volume_fraction`: The volume fraction of the inclusion.
        - `parameters`: Additional parameters specific to the inclusion shape.
    uns : bool, optional
        If True, the function will return inclusions as an
        union of n-sphere representation.

    Returns
    -------
    gb.CirclesArray | Inclusion
        A GBox CirclesArray or an Inclusion object representing the
        generated inclusions based on the configuration.
    """

    cumulative_volume = 0.0
    generated_inclusions = []
    while cumulative_volume < required_volume:
        incl_params = param_sampler.sample()  # sample inclusion parameters

        # spatial parameter sampling, initializing the inclusions
        if spatial_param_sampler is not None:
            xy_c = spatial_param_sampler.sample()
        else:
            xy_c = {"xc": 0, "yc": 0}  # Default to origin if not provided

        # create inclusion based on the shape
        if shape == "circle":
            a_inclusion = gb.Circle.from_xyr(**xy_c, **incl_params)
        elif shape == "ellipse":
            a_inclusion = gb.Ellipse.from_xy_semi_axes(
                **xy_c,
                semi_major_axis=incl_params["semi_major_axis"],
                semi_minor_axis=incl_params["semi_minor_axis"],
            )
        else:
            raise ValueError(f"Unsupported shape: {shape}")
        generated_inclusions.append(a_inclusion)

        cumulative_volume += a_inclusion.volume()  # update cumulative volume

    # make uns representation
    if uns:
        raise NotImplementedError(
            "Union of n-sphere representation is not implemented yet."
        )

    return generated_inclusions


def initialise_inclusions(
    cell_domain: CellDomain,
    config: dict | list[dict],
    uns: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, List[gb.CirclesArray | Inclusion]]:
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
    }
    xy_sampler = DistributionSampler(xy_init_config, rng=rng)

    initialised_inclusions = {}
    for ith_incl_config in config:
        shape, vf, params = _validate_dict(
            ith_incl_config,
            keys=["shape", "volume_fraction", "parameters"],
            val_types=[str, float, dict],
            ret_val=True,
        )
        initialised_inclusions[shape] = generate_and_initialise_inclusions(
            required_volume=cell_domain.cell_volume * vf,
            shape=shape,
            param_sampler=DistributionSampler(params, rng=rng),
            spatial_param_sampler=xy_sampler,
            uns=uns,
        )
    return initialised_inclusions

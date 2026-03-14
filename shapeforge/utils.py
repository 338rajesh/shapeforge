import json
from pathlib import Path

import yaml

from functools import partial
from typing import Optional

import numpy as np
from scipy import stats


class DistributionSampler:
    """
    A class to create a sampler for various distributions.
    """

    def __init__(self, spec: dict, rng: Optional[np.random.Generator] = None):
        """
        Initialize the sampler with a specification of distributions.

        Parameters
        ----------
        spec : dict
            A specification dictionary where each key is the name of a
            quantity to sample, and the value is a dictionary defining its
            characteristics, including the distribution details. For this
            implementation, the `distribution` key must be specified with
            a dictionary containing the distribution name and its parameters,
            as defined in `scipy.stats`.

        rng : np.random.Generator, optional
            A random number generator instance. If None, a default RNG is used.

        Examples
        --------

        ```
        >>> a_spec = {
        "radius": {"distribution": {"name": "norm", "loc": 0, "scale": 1}}
        }
        >>> a = DistributionSampler(a_spec)
        >>> a.sample(size=10)  # returns samples in a dict with key 'radius'
        >>> b_spec = {
            "semi_major_axis": {
                "distribution": {"name": "uniform", "loc": 2.0, "scale": 0.5}
            },
            "semi_minor_axis": {
                "distribution": {"name": "uniform", "loc": 1.0, "scale": 0.5}
            }
        }
        >>> b = DistributionSampler(b_spec)
        >>> b.sample(size=10)  # returns samples in a dict with
        # keys 'semi_major_axis' and 'semi_minor_axis'
        ```
        """
        self.samplers = self._make_samplers(spec, rng)

    @staticmethod
    def _make_samplers(spec: dict, rng: np.random.Generator) -> dict:
        samplers = {}
        if rng is None:
            rng = np.random.default_rng()
        for a_quant_name in spec:
            a_quant_dist_spec = spec[a_quant_name]

            _validate_dict(a_quant_dist_spec, keys=["distribution"], val_types=[dict])
            dist_spec = a_quant_dist_spec.get("distribution")

            _validate_dict(dist_spec, keys=["name"], val_types=[str])
            dist_name = dist_spec["name"].lower()

            rv: stats.rv_continuous = getattr(stats, dist_name, None)
            if rv is None or not hasattr(rv, "rvs"):
                raise ValueError(
                    f"Unsupported distribution: {dist_name}. "
                    "Ensure it is a valid scipy.stats distribution."
                )

            params = {k: v for k, v in dist_spec.items() if k != "name"}
            samplers[a_quant_name] = partial(rv.rvs, random_state=rng, **params)
        return samplers

    def sample(self, size: int = 1) -> dict[str, np.ndarray | float]:
        """
        Sample from the distributions defined in the spec.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate per each specified distribution.
            Default is 1.

        Returns
        -------
        list or np.ndarray
            A list of samples from each distribution. If only one distribution
            is specified, returns a single array of samples.
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer > 0.")

        gen_samples = {}
        for a_quant_name, a_sampler in self.samplers.items():
            a = np.asarray(a_sampler(size=size))
            if size == 1:
                a = a[0].item()
            gen_samples[a_quant_name] = a
        return gen_samples
        # outcomes = []
        # for a_sampler in self.samplers:
        #     outcomes.append(np.asarray(a_sampler(size=size)))
        # if self._single:
        #     return outcomes[0]

        # handle size=1 case
        # if size == 1 and len(outcomes) == 1:
        #     return outcomes[0][0]
        # elif size == 1:
        #     return [outcome[0] for outcome in outcomes]
        # else:
        #     return outcomes


def load_yaml(file_path: str, show: bool = False) -> dict:
    """
    Load a YAML configuration file and optionally print its contents.

    Parameters
    ----------
    file_path : str
        The path to the YAML configuration file.
    show : bool, optional
        If True, print the loaded configuration. Default is False.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    fp = Path(file_path).resolve()
    if not fp.exists():
        raise FileNotFoundError(f"Configuration file '{file_path}' does not exist.")
    if not fp.suffix == ".yaml":
        raise ValueError(
            f"Configuration file '{file_path}' must have a .yaml extension."
        )

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    if show:
        json_str = json.dumps(config, indent=4)
        print(f"Loaded configuration from {file_path}:\n{json_str}")

    return config


def _validate_dict(
    d: dict,
    keys: list,
    val_types: list = None,
    ret_val: bool = False,
) -> None | tuple:
    """
    Validate that a dictionary contains specific keys.

    Parameters
    ----------
    d : dict
        The dictionary to validate.
    keys : list
        The list of keys that must be present in the dictionary.
    val_types : list, optional
        A list of types corresponding to each key in `keys`. If provided,
        the function will also check that the values associated with each key
        are of the specified type.
    ret_val : bool, optional
        If True, the function will return the values associated with the keys
        in the same order as the keys. Default is False.
    Raises
    ------
    ValueError
        If any of the specified keys are missing from the dictionary.
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(keys, list):
        raise TypeError("Expected keys must be provided as a list.")

    missing_keys = [key for key in keys if key not in d]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    if val_types is None:
        val_types = [None] * len(keys)
    if len(keys) != len(val_types):
        raise ValueError("Length of keys and val_types must match.")
    for key, val_type in zip(keys, val_types):
        if val_type is None:
            continue
        if not isinstance(d[key], val_type):
            raise TypeError(
                f"Value for key '{key}' must be of type {val_type.__name__}, "
                f"but got {type(d[key]).__name__}."
            )
    if ret_val:
        return tuple(d[key] for key in keys)

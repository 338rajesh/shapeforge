import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import yaml
from gbox.core.utils import Validator, get_logger
from scipy import stats

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """
    A dataclass to specify the distribution of a quantity.
    """

    distribution: str
    loc: float
    scale: float

    def __post_init__(self):
        Validator.as_float(self.scale, low=0.0, closed_bounds=False)

        rv: stats.rv_continuous = getattr(stats, self.distribution, None)
        if rv is None or not hasattr(rv, "rvs"):
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                "Ensure it is a valid scipy.stats distribution."
            )

    def rvs_partial(self) -> Callable[[], float]:
        rv: stats.rv_continuous = getattr(stats, self.distribution, None)
        return partial(rv.rvs, loc=self.loc, scale=self.scale)

    @classmethod
    def from_signature(cls, sig: str) -> Self:
        Validator.is_type(sig, str, name="Distribution Signature")
        match = re.match(r"(\w+)\((.*)\)", sig)
        if not match:
            raise ValueError(
                "Invalid signature: it must be in the form of"
                "<method_name>(<loc>, <scale>). Example: uniform(2.0, 0.1)"
            )
        method = match.group(1)
        loc, scale = [float(a.strip()) for a in match.group(2).split(",")]
        return cls(method, loc, scale)


class DistributionSampler:
    """
    A class to create a sampler for various distributions.
    """

    __slots__ = ("_sampler",)

    def __init__(
        self,
        distribution_spec: DistributionSpec,
        rng: np.random.Generator | None = None,
        rvs_kwargs=None,
    ):
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

        rng = rng or np.random.default_rng()
        rvs_kwargs = rvs_kwargs or {}
        self._sampler = partial(
            distribution_spec.rvs_partial(), random_state=rng, **rvs_kwargs
        )

    @classmethod
    def from_signature(cls, sig: str, rng=None, rvs_kwargs=None) -> Self:
        distr_spec = DistributionSpec.from_signature(sig)
        return cls(distr_spec, rng, rvs_kwargs)

    def sample(self, size: int = 1) -> npt.NDArray | float:
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
        size = Validator.as_int(size, low=1, name="Sample Size")
        a = np.asarray(self.sampler(size=size))
        if size == 1:
            return a[0].item()
        return a


def _load_dict(file_path: str | Path) -> dict:
    """
    Load a dictionary from a YAML/JSON file and optionally print its contents.

    Parameters
    ----------
    file_path : str
        The path to the YAML/JSON file.
    show : bool, optional
        If True, print the loaded configuration. Default is False.

    Returns
    -------
    dict
        The loaded dictionary.
    """
    fp = Validator.file_path(
        file_path, must_exist=True, extensions=[".json", ".yml", ".yaml"]
    )
    with open(fp, "r") as file:
        if fp.suffix in (".yaml", "yml"):
            config = yaml.safe_load(file)
        else:  # ".json"
            config = json.load(file)

    Validator.is_type(config, dict, name="loaded values")

    logger.debug(
        f"Loaded configuration from {file_path}:\n"
        f"{json.dumps(config, indent=4)}"
    )

    return config

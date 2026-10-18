"""
Utility functions and distribution helpers for ShapeForge.
==========================================================

Provides statistical distribution specifications, random sampling wrappers
interfacing with ``scipy.stats``, configuration loading utilities, and convergence
visualization routines.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import yaml
from gbox.core.utils import Validator, get_logger
from scipy import stats

logger = get_logger(__name__)
PI: float = float(np.pi)


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """
    Specification for a continuous random variable distribution.

    Wraps a named ``scipy.stats`` continuous distribution along with its
    location (``loc``) and scale (``scale``) parameters.

    Parameters
    ----------
    distribution : str
        Name of the distribution as defined in ``scipy.stats`` (e.g., ``'uniform'``,
        ``'norm'``, ``'lognorm'``).
    loc : float
        Location parameter of the distribution (e.g. lower bound for uniform,
        mean for normal).
    scale : float
        Scale parameter of the distribution. Must be strictly positive ($> 0$).

    Raises
    ------
    ValueError
        If ``scale <= 0`` or if ``distribution`` is not a recognized continuous
        distribution in ``scipy.stats``.

    Examples
    --------
    >>> spec = DistributionSpec("uniform", loc=1.0, scale=2.0)
    >>> spec.distribution
    'uniform'
    >>> spec = DistributionSpec.from_signature("norm(10.0, 1.5)")
    >>> spec.loc, spec.scale
    (10.0, 1.5)
    """

    distribution: str
    loc: float
    scale: float

    def __post_init__(self) -> None:
        """Validate distribution name and parameter bounds."""
        Validator.as_float(self.scale, low=0.0, closed_bounds=False)

        rv: stats.rv_continuous | None = getattr(stats, self.distribution, None)
        if rv is None or not hasattr(rv, "rvs"):
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                "Ensure it is a valid scipy.stats distribution."
            )

    def rvs_partial(self) -> Callable[..., float]:
        """
        Return a partially evaluated random variate sampler function.

        Returns
        -------
        Callable[..., float]
            Function binding the distribution's ``rvs`` method with ``loc``
            and ``scale``.
        """
        rv: stats.rv_continuous = getattr(stats, self.distribution)
        return partial(rv.rvs, loc=self.loc, scale=self.scale)

    @classmethod
    def from_signature(cls, sig: str | Self) -> Self:
        """
        Parse a distribution specification from a string signature.

        Parameters
        ----------
        sig : str or DistributionSpec
            Signature string in the format ``'<dist_name>(<loc>, <scale>)'``,
            e.g., ``'uniform(1.0, 2.0)'`` or ``'norm(0.0, 1.0)'``. If already an
            instance of ``DistributionSpec``, it is returned unchanged.

        Returns
        -------
        DistributionSpec
            Parsed distribution specification instance.

        Raises
        ------
        ValueError
            If the string signature cannot be parsed or parameters cannot be
            converted to floats.
        TypeError
            If ``sig`` is neither a string nor a ``DistributionSpec``.
        """
        if isinstance(sig, cls):
            return sig

        Validator.is_type(sig, str, name="Distribution Signature")
        match = re.match(r"(\w+)\((.*)\)", sig)
        if not match:
            raise ValueError(
                "Invalid signature: it must be in the form of "
                "<method_name>(<loc>, <scale>). Example: uniform(2.0, 0.1)"
            )
        method = match.group(1)
        loc, scale = [float(a.strip()) for a in match.group(2).split(",")]
        return cls(method, loc, scale)


class DistributionSampler:
    """
    Random variate sampler wrapping a :class:`DistributionSpec`.

    Provides reproducible pseudo-random sampling using NumPy's modern
    random number generator API (``np.random.Generator``).

    Parameters
    ----------
    distribution_spec : DistributionSpec
        The distribution specification defining the statistical family,
        location, and scale.
    rng : np.random.Generator, optional
        NumPy random number generator instance for reproducible sampling.
        If None, ``np.random.default_rng()`` is used.
    rvs_kwargs : dict, optional
        Additional keyword arguments passed to the underlying ``scipy.stats``
        sampling function.

    Examples
    --------
    >>> import numpy as np
    >>> spec = DistributionSpec("uniform", loc=1.0, scale=0.5)
    >>> sampler = DistributionSampler(spec, rng=np.random.default_rng(42))
    >>> val = sampler.sample()
    >>> isinstance(val, float)
    True
    >>> samples = sampler.sample(size=5)
    >>> len(samples)
    5
    """

    __slots__ = ("_sampler",)

    def __init__(
        self,
        distribution_spec: DistributionSpec,
        rng: np.random.Generator | None = None,
        rvs_kwargs: dict[str, Any] | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        rvs_kwargs = rvs_kwargs or {}
        self._sampler = partial(
            distribution_spec.rvs_partial(), random_state=rng, **rvs_kwargs
        )

    @classmethod
    def from_signature(
        cls,
        sig: str,
        rng: np.random.Generator | None = None,
        rvs_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Create a sampler directly from a string distribution signature.

        Parameters
        ----------
        sig : str
            Distribution signature formatted as ``'<dist_name>(<loc>, <scale>)'``,
            e.g., ``'uniform(0.5, 1.0)'``.
        rng : np.random.Generator, optional
            Random generator for reproducible sampling.
        rvs_kwargs : dict, optional
            Extra arguments passed to the ``scipy.stats`` variate generator.

        Returns
        -------
        DistributionSampler
            Configured sampler instance ready to draw random variates.
        """
        distr_spec = DistributionSpec.from_signature(sig)
        return cls(distr_spec, rng, rvs_kwargs)

    def sample(self, size: int = 1) -> npt.NDArray | float:
        """
        Draw one or more samples from the distribution.

        Parameters
        ----------
        size : int, optional
            Number of random variates to generate. Must be $\\ge 1$.
            Default is 1.

        Returns
        -------
        float or np.ndarray
            If ``size == 1``, returns a single scalar float. If ``size > 1``,
            returns a 1D NumPy array of shape ``(size,)``.

        Raises
        ------
        ValueError
            If ``size < 1``.
        """
        size = Validator.as_int(size, low=1, name="Sample Size")
        a = np.asarray(self._sampler(size=size))
        if size == 1:
            return a[0].item()
        return a


def _load_dict(file_path: str | Path | dict) -> dict:
    """
    Load a dictionary from a file path or return it directly if already a dict.

    Supports JSON (``.json``) and YAML (``.yaml``, ``.yml``) formats.

    Parameters
    ----------
    file_path : str, Path, or dict
        Path to the configuration file, or an already-parsed dictionary.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist on disk.
    ValueError
        If the file has an unsupported extension or does not contain a valid mapping.
    """
    if isinstance(file_path, dict):
        return file_path
    fp = Validator.file_path(
        file_path, must_exist=True, extensions=[".json", ".yml", ".yaml"]
    )
    with open(fp, "r") as file:
        if fp.suffix in (".yaml", ".yml"):
            config = yaml.safe_load(file)
        else:  # ".json"
            config = json.load(file)

    Validator.is_type(config, dict, name="loaded values")

    logger.debug(
        f"Loaded configuration from {file_path}:\n"
        f"{json.dumps(config, indent=4)}"
    )

    return config


def plot_f_and_g_norms(sol: Any, f_path: Path | None = None) -> None:
    r"""
    Plot convergence histories of the objective function and gradient norm.

    Renders a two-panel vertical plot displaying:
    1. Objective function values $f_k$ across iterations (log scale).
    2. Projected gradient infinity-norm $\|g_k\|_\\infty$ across iterations (log scale).

    Parameters
    ----------
    sol : OptimisationResult
        Optimization result object containing ``f_history`` and ``g_norm_history``.
    f_path : Path, optional
        Destination image path (e.g. PNG). If None, the plot is displayed
        interactively via ``matplotlib.pyplot.show()``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(18, 6))
    f_hist = getattr(sol, "f_history", None)
    g_hist = getattr(sol, "g_norm_history", None)
    if f_hist is not None and len(f_hist) > 0:
        n = list(range(len(f_hist)))
        ax[0].plot(n, f_hist, color="r", label="Function Values")
        ax[0].legend()
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Function Values")
        if any(val > 0 for val in f_hist):
            ax[0].set_yscale("log")
    if g_hist is not None and len(g_hist) > 0:
        n = list(range(len(g_hist)))
        ax[1].plot(n, g_hist, color="k", label="Gradient Norms")
        ax[1].legend()
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Gradient Norms")
        if any(val > 0 for val in g_hist):
            ax[1].set_yscale("log")
    if f_path is not None:
        fig.savefig(f_path)
    else:
        plt.show()
    plt.close(fig)

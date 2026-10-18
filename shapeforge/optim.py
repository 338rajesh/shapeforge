"""
Optimisation algorithms and problem abstractions for ShapeForge.
=================================================================

Provides the abstract base class :class:`OptimisationProblem`, the structured
result container :class:`OptimisationResult`, and the Non-Monotone Spectral
Projected Gradient (:func:`nmspg`) solver.

Available Algorithms
--------------------
- Non-Monotone Spectral Projected Gradient (NM-SPG):
  A gradient projection method using Barzilai-Borwein spectral step lengths,
  non-monotone Armijo line search over historical function values, and safeguard
  quadratic interpolation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gbox.core.utils import Validator


class OptimisationProblem:
    """
    Abstract base class for bound-constrained or projected optimization problems.

    Subclasses must implement the objective function :meth:`f` (or override
    :meth:`func`), the gradient :meth:`grad_f` (or override :meth:`grad`),
    and the feasible projection :meth:`projection`.

    Attributes
    ----------
    _eval_count : dict[str, int]
        Counter dictionary tracking evaluations of ``'func'``, ``'grad'``, and ``'proj'``.
    _x0 : np.ndarray or None
        Initial point vector.
    """

    __slots__ = ("_eval_count", "_x0")

    def __init__(self) -> None:
        self._x0: np.ndarray | None = None
        self._eval_count: dict[str, int] = {"func": 0, "grad": 0, "proj": 0}

    def func(self, x: np.ndarray, *args: Any, **kwargs: Any) -> float:
        """
        Evaluate the objective function at candidate point `x`.

        Parameters
        ----------
        x : np.ndarray
            Current optimization parameter vector.
        *args : Any
            Additional positional arguments forwarded to :meth:`f`.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`f`.

        Returns
        -------
        float
            Scalar objective function value $f(x)$.

        Raises
        ------
        NotImplementedError
            If neither :meth:`f` nor :meth:`func` is implemented.
        """
        if hasattr(self, "f") and callable(self.f):
            return self.f(x, *args, **kwargs)
        raise NotImplementedError("Objective function evaluation not implemented.")

    def grad(
        self, x: np.ndarray, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function at candidate point `x`.

        Parameters
        ----------
        x : np.ndarray
            Current optimization parameter vector.
        *args : Any
            Additional positional arguments forwarded to :meth:`grad_f`.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`grad_f`.

        Returns
        -------
        np.ndarray
            Gradient vector $\\nabla f(x)$ matching the shape of `x`.

        Raises
        ------
        NotImplementedError
            If neither :meth:`grad_f` nor :meth:`grad` is implemented.
        """
        if hasattr(self, "grad_f") and callable(self.grad_f):
            return self.grad_f(x, *args, **kwargs)
        raise NotImplementedError("Gradient evaluation not implemented.")

    def projection(self, x: np.ndarray) -> np.ndarray:
        """
        Project candidate point `x` onto the feasible constraint set $\\Omega$.

        Parameters
        ----------
        x : np.ndarray
            Point to project.

        Returns
        -------
        np.ndarray
            Feasible point $P_\\Omega(x)$.

        Raises
        ------
        NotImplementedError
            If projection is not implemented by the subclass.
        """
        raise NotImplementedError("Projection function not implemented.")

    def solve(self, method: str = "nmspg", **options: Any) -> OptimisationResult:
        """
        Solve the optimization problem using the specified numerical solver.

        Parameters
        ----------
        method : str, optional
            Algorithm identifier. Currently only ``'nmspg'`` is supported
            (default: ``'nmspg'``).
        **options : Any
            Solver-specific hyperparameters forwarded to :func:`nmspg`.

        Returns
        -------
        OptimisationResult
            Structured result containing the optimal point, status, and convergence history.

        Raises
        ------
        ValueError
            If `method` is unknown or unsupported.
        """
        method = Validator.as_string(method).lower()
        if method == "nmspg":
            return nmspg(self, self.x0, **options)
        else:
            raise ValueError(f"Unknown optimisation method '{method}'")


@dataclass(slots=True, frozen=True)
class OptimisationResult:
    """
    Immutable container storing the outcome and diagnostics of an optimization run.

    Parameters
    ----------
    x_optimal : np.ndarray
        Best parameter vector achieved by the optimizer.
    status : {'success', 'failure'}, optional
        Termination status indicator (default: ``'success'``).
    failure_message : str | None, optional
        Human-readable diagnostic message explaining reason for failure.
    f_history : list of float | None, optional
        Recorded objective function values across iterations.
    g_norm_history : list of float | None, optional
        Recorded projected gradient infinity-norms across iterations.
    iter_count : int, optional
        Total iterations performed (default: 0).
    f_eval_count : int, optional
        Total objective function evaluations (default: 0).
    g_eval_count : int, optional
        Total gradient evaluations (default: 0).
    proj_eval_count : int, optional
        Total projection evaluations (default: 0).

    Raises
    ------
    ValueError
        If the length of ``f_history`` or ``g_norm_history`` does not equal
        ``iter_count + 1``.
    """

    x_optimal: np.ndarray
    status: Literal["success", "failure"] = "success"
    failure_message: str | None = None
    f_history: list[float] | None = None
    g_norm_history: list[float] | None = None
    iter_count: int = 0
    f_eval_count: int = 0
    g_eval_count: int = 0
    proj_eval_count: int = 0

    def __post_init__(self) -> None:
        n = self.iter_count
        if self.f_history and (len(self.f_history) != n + 1):
            raise ValueError("Length of f_history must match iter_count + 1.")
        if self.g_norm_history and (len(self.g_norm_history) != n + 1):
            raise ValueError("Length of g_norm_history must match iter_count + 1.")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result into a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of optimal variables and iteration history.
        """
        return {
            "x_optimal": self.x_optimal.tolist(),
            "status": self.status,
            "failure_message": self.failure_message,
            "iter_count": self.iter_count,
            "f_eval_count": self.f_eval_count,
            "g_eval_count": self.g_eval_count,
            "proj_eval_count": self.proj_eval_count,
            "f_history": self.f_history,
            "g_norm_history": self.g_norm_history,
        }


def nmspg(
    objective: OptimisationProblem,
    x0: np.ndarray | list[float],
    iter_max: int = 1000,
    iter_memory: int = 10,
    epsilon: float = 1e-6,
    f_tol: float | None = None,
    spectral_step_min: float = 1e-10,
    spectral_step_max: float = 1e3,
    gamma: float = 0.0001,
    sigma1: float = 0.1,
    sigma2: float = 0.9,
    ls_iter_max: int = 20,
    p_bar: bool | None = None,
) -> OptimisationResult:
    r"""
    Non-monotone Spectral Projected Gradient (NM-SPG) algorithm.

    Solves convex or non-convex constrained optimization problems of the form:

    $$\\min_{x \\in \\Omega} f(x)$$

    where $\\Omega$ is a closed set onto which projections $P_\\Omega(x)$ can be computed.

    Parameters
    ----------
    objective : OptimisationProblem
        Problem instance defining :meth:`~OptimisationProblem.func`,
        :meth:`~OptimisationProblem.grad`, and :meth:`~OptimisationProblem.projection`.
    x0 : np.ndarray or list of float
        Initial parameter vector $x_0$.
    iter_max : int, optional
        Maximum number of iterations allowed (default: 1000).
    iter_memory : int, optional
        Memory size $M$ for non-monotone line search tracking the maximum function
        value over the previous $M$ iterations (default: 10).
    epsilon : float, optional
        Optimality tolerance for the projected gradient infinity norm:
        $\\|P_\\Omega(x_k - g_k) - x_k\\|_\\infty \\le \\epsilon$ (default: 1e-6).
    f_tol : float or None, optional
        Objective function target tolerance. If specified, the optimization
        terminates successfully as soon as $f_k \\le f_{\\text{tol}}$. If a stationary
        point is reached ($\\|d_k\\|_\\infty \\le \\epsilon$) with $f_k > f_{\\text{tol}}$,
        the solver reports failure due to stagnation in a local minimum (default: None).
    spectral_step_min : float, optional
        Lower safeguard bound for the Barzilai-Borwein spectral step length
        (default: 1e-10).
    spectral_step_max : float, optional
        Upper safeguard bound for the Barzilai-Borwein spectral step length
        (default: 1e3).
    gamma : float, optional
        Sufficient decrease parameter for the non-monotone Armijo condition:
        $f(x_k + \\alpha d_k) \\le \\max_{0 \\le j \\le M} f_{k-j} + \\gamma \\alpha g_k^T d_k$
        (default: 1e-4).
    sigma1 : float, optional
        Lower safeguard interpolation factor for line search quadratic step
        reduction (default: 0.1).
    sigma2 : float, optional
        Upper safeguard interpolation factor for line search quadratic step
        reduction (default: 0.9).
    ls_iter_max : int, optional
        Maximum line search trials per iteration before engaging fallback or
        terminating (default: 20).
    p_bar : bool or None, optional
        Whether to print per-iteration progress logs to standard output.
        If None, logging is suppressed.

    Returns
    -------
    OptimisationResult
        Optimal point, termination status, iteration counts, and convergence history.

    Notes
    -----
    The algorithm implements the spectral projected gradient framework of Birgin
    et al. [1]_. The spectral step length $\lambda_k$ is computed using the
    Barzilai-Borwein formula:

    $$\\lambda_k = \\frac{s_{k-1}^T s_{k-1}}{s_{k-1}^T y_{k-1}}$$

    where $s_{k-1} = x_k - x_{k-1}$ and $y_{k-1} = g_k - g_{k-1}$. When
    $s_{k-1}^T y_{k-1} \\le 0$, $\\lambda_k$ is safely clamped to ``spectral_step_max``.

    References
    ----------
    .. [1] Birgin, E. G., Martínez, J. M., & Raydan, M. (2000). Nonmonotone
       spectral projected gradient methods on convex sets. SIAM Journal on
       Optimization, 10(4), 1196-1211.
    """

    def get_d_k(x_: np.ndarray, g_: np.ndarray, ssl_: float) -> np.ndarray:
        x_trial = x_ - ssl_ * g_
        d_ = objective.projection(x_trial) - x_
        objective._eval_count["proj"] += 1
        return d_

    def d_inf_norm(x_: np.ndarray, g_: np.ndarray, ssl_: float = 1.0) -> float:
        return float(np.linalg.norm(get_d_k(x_, g_, ssl_), ord=np.inf))

    def get_init_ssl(x_: np.ndarray, g_: np.ndarray) -> float:
        d_inf_norm_ = d_inf_norm(x_, g_, ssl_=1.0)
        ssl_ = 1 / d_inf_norm_ if d_inf_norm_ > 0 else 1.0
        return max(spectral_step_min, min(spectral_step_max, ssl_))

    def get_ssl(
        x_k_: np.ndarray,
        x_kp1_: np.ndarray,
        g_k_: np.ndarray,
        g_kp1_: np.ndarray,
    ) -> float:
        s_k = x_kp1_ - x_k_
        y_k = g_kp1_ - g_k_
        sy = np.dot(s_k, y_k)
        if sy > 0:
            ssl_ = np.dot(s_k, s_k) / sy
            return max(spectral_step_min, min(spectral_step_max, ssl_))
        else:
            return spectral_step_max

    x_k = np.array(x0, dtype=np.float32).copy()
    f_k = objective.func(x_k)
    g_k = objective.grad(x_k)
    f_history = [f_k]
    g_norm_history = [float(np.linalg.norm(g_k, ord=np.inf))]

    f_memory = deque([f_k] * iter_memory)

    k = 0
    status: Literal["success", "failure"] = "failure"
    failure_message: str | None = None
    ssl_k = get_init_ssl(x_k, g_k)
    while k < iter_max:
        # Check convergence
        d_inf_norm_k = d_inf_norm(x_k, g_k, ssl_=1.0)
        if f_tol is not None and f_k <= f_tol:
            status = "success"
            break

        if d_inf_norm_k <= epsilon:
            if f_tol is not None and f_k > f_tol:
                status = "failure"
                failure_message = (
                    f"Optimisation reached a stationary point (d_inf_norm = {d_inf_norm_k:.4g} <= {epsilon}), "
                    f"but residual objective value f = {f_k:.4g} exceeds f_tol ({f_tol})."
                )
            else:
                status = "success"
            break

        if p_bar:
            print(
                f"[Iteration {k + 1}/{iter_max}]"
                f"[d_inf_norm: {d_inf_norm_k:.4g}]"
                f"[f: {f_k:.4g}]"
                f"[g_norm: {g_norm_history[-1]:.4g}]",
                end="",
            )

        d_k = get_d_k(x_k, g_k, ssl_k)
        f_max = max(f_memory)

        alpha = 1.0
        ls_iter = 0
        slope_local = np.dot(g_k, d_k)
        armijo_satisfied = False

        while ls_iter < ls_iter_max:
            x_trial = x_k + alpha * d_k
            f_trial = objective.func(x_trial)

            if f_trial <= f_max + (gamma * alpha * slope_local):
                armijo_satisfied = True
                break

            denom = f_trial - f_k - (alpha * slope_local)
            numer = -0.5 * (alpha**2) * slope_local
            if denom != 0:
                alpha_temp = numer / denom
                if sigma1 < alpha_temp < sigma2 * alpha:
                    alpha = alpha_temp
                else:
                    alpha = alpha / 2
            else:
                alpha = alpha / 2

            ls_iter += 1

        if not armijo_satisfied:
            # Fallback: projected gradient step with conservative step length
            d_fallback = get_d_k(
                x_k, g_k, ssl_=min(1.0, 1.0 / max(d_inf_norm_k, 1e-12))
            )
            slope_fallback = np.dot(g_k, d_fallback)
            alpha_fb = 1.0
            ls_iter_fb = 0
            while ls_iter_fb < ls_iter_max:
                x_trial = x_k + alpha_fb * d_fallback
                f_trial = objective.func(x_trial)
                if f_trial <= f_max + (gamma * alpha_fb * slope_fallback):
                    armijo_satisfied = True
                    break
                alpha_fb = alpha_fb / 2
                ls_iter_fb += 1

        if not armijo_satisfied:
            status = "failure"
            failure_message = (
                f"Line search failed to find an acceptable step at iteration {k + 1}."
            )
            break

        x_kp1 = x_trial
        g_kp1 = objective.grad(x_kp1)
        ssl_kp1 = get_ssl(x_k, x_kp1, g_k, g_kp1)

        x_k = x_kp1
        f_k = f_trial
        g_k = g_kp1
        ssl_k = ssl_kp1

        f_history.append(f_k)
        g_norm_history.append(float(np.linalg.norm(g_k, ord=np.inf)))
        f_memory.popleft()
        f_memory.append(f_k)

        k += 1
    else:
        failure_message = (
            f"Maximum number of iterations ({iter_max}) reached without convergence."
        )

    return OptimisationResult(
        x_optimal=np.asarray(x_k),
        status=status,
        failure_message=failure_message,
        f_history=[float(i) for i in f_history],
        g_norm_history=g_norm_history,
        iter_count=k,
        f_eval_count=objective._eval_count["func"],
        g_eval_count=objective._eval_count["grad"],
        proj_eval_count=objective._eval_count["proj"],
    )

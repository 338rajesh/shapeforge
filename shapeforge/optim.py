"""
This module provides the optimisation methods used for cell generation.

Available Algorithms
  - Non-monotone spectral projection gradient (NM-SPG)

"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


class OptimisationProblem:
    def __init__(self, params: Optional[dict] = None):
        pass
        self.params = params or {}
        self.eval_count = {"f_and_g": 0, "proj": 0}

    def f_and_grad(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function and its gradient
        """
        raise NotImplementedError(
            "Objective function and gradient evaluation not implemented. "
        )

    def projection(self, x: np.ndarray) -> np.ndarray:
        """
        Projection onto the feasible set.
        """
        raise NotImplementedError("Projection function not implemented.")


@dataclass
class OptimisationResult:
    x_optimal: np.ndarray
    status: Literal["success", "failure"] = "success"
    failure_message: Optional[str] = None
    f_history: Optional[list] = None
    g_norm_history: Optional[list] = None
    iter_count: int = 0
    f_and_g_eval_count: int = 0
    proj_eval_count: int = 0

    def __post_init__(self):
        n = self.iter_count
        self.x_optimal = np.array(self.x_optimal)
        if self.f_history and (not len(self.f_history) == n + 1):
            raise ValueError("Length of f_history must match iter_count.")
        if self.g_norm_history and (not len(self.g_norm_history) == n + 1):
            raise ValueError("Length of g_norm_history must match iter_count.")

    def to_dict(self) -> dict:
        return {
            "x_optimal": self.x_optimal,
            "iter_count": self.iter_count,
            "f_and_g_eval_count": self.f_and_g_eval_count,
            "f_history": self.f_history,
            "g_norm_history": self.g_norm_history,
            "proj_eval_count": self.proj_eval_count,
        }


def nmspg(
    objective: OptimisationProblem,
    x0: np.ndarray | list,
    iter_max=1000,
    iter_memory=10,
    epsilon=1e-6,
    spectral_step_min=1e-30,
    spectral_step_max=1e30,
    gamma=0.0001,
    sigma1=0.1,
    sigma2=0.9,
    ls_iter_max=20,
    p_bar=None,
) -> OptimisationResult:
    """
    Non-monotone spectral projection gradient (NM-SPG) algorithm for solving
    bound-constrained optimization problems.

    Parameters
    ----------
    objective : OptimisationProblem
        An instance of the OptimisationProblem class that defines the
        objective function, its gradient, and projection methods.
    x0 : np.ndarray or list
        Initial point for the optimization.
    iter_max : int, optional, defaults to 1000
        Maximum number of iterations.
    iter_memory : int, optional, defaults to 10
        Number of previous function values to store for
        non-monotone line search.
    epsilon : float, optional, defaults to 1e-6
        Convergence tolerance for the optimization.
    spectral_step_min : float, optional, defaults to 1e-10
        Lower bound of Barzilai-Borwein spectral step length.
    spectral_step_max : float, optional, defaults to 1e10
        Upper bound of Barzilai-Borwein spectral step length.
    gamma : float, optional, defaults to 0.0001
        A sufficent decrease parameter for checking the
        Armijo condition in the line search.
    sigma1 : float, optional, defaults to 0.1
        Lower bound safeguard parameter for the quadratic line search
        used in the non-monotone line search. It should be in (0, 1)
        but less than sigma2.
    sigma2 : float, optional, defaults to 0.9
        Upper bound safeguard parameter for the quadratic line search
        used in the non-monotone line search. It should be in (0, 1)
        but greater than sigma1.


    """

    def get_d_k(x_: np.ndarray, g_: np.ndarray, ssl_: float) -> np.ndarray:
        """
        Compute the direction for the next step.
        """
        x_trial = x_ - ssl_ * g_
        d_ = objective.projection(x_trial) - x_
        objective.eval_count["proj"] += 1
        return d_

    def d_inf_norm(x_: np.ndarray, g_: np.ndarray, ssl_: float = 1.0) -> bool:
        """
        Check if the termination condition is met.
        """
        return np.linalg.norm(get_d_k(x_, g_, ssl_), ord=np.inf)

    def get_init_ssl(x_: np.ndarray, g_: np.ndarray) -> float:
        """
        Compute the spectral step length using the Barzilai-Borwein method.
        """
        d_inf_norm_ = d_inf_norm(x_, g_, ssl_=1.0)
        ssl_ = 1 / d_inf_norm_ if d_inf_norm_ > 0 else spectral_step_max
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
    f_k, g_k = objective.f_and_grad(x_k)
    f_history = [f_k]
    g_norm_history = [np.linalg.norm(g_k, ord=np.inf)]

    f_memory = deque([f_k] * iter_memory)  # Store function values

    k = 0
    status = "failure"
    failure_message = None
    ssl_k = get_init_ssl(x_k, g_k)
    while k < iter_max:
        # --------------------------------------------
        #       Check convergence
        # --------------------------------------------
        if d_inf_norm(x_k, g_k, ssl_=1.0) <= epsilon:
            status = "success"
            break

        if p_bar is not None:
            print(f"Iteration {k + 1}/{iter_max}, f: {f_k:.6f}", end="\r")

        # --------------------------------------------
        #       Get the search direction, d_k
        # --------------------------------------------
        d_k = get_d_k(x_k, g_k, ssl_=ssl_k)

        # --------------------------------------------
        #       Compute the step length, alpha_k
        # --------------------------------------------
        f_max = max(f_memory)  # Maximum of the previous function values

        alpha = 1.0  # Initial step length
        ls_iter = 0
        slope_local = np.dot(g_k, d_k)
        while True:
            x_trial = x_k + alpha * d_k
            f_trial, _ = objective.f_and_grad(x_trial)

            # terminate if reached the max number of line search iterations
            if ls_iter >= ls_iter_max:
                break

            # terminate if the Armijo condition is satisfied
            if f_trial <= f_max + (gamma * alpha * slope_local):
                break

            # update the step length with safeguard parameters
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

        # --------------------------------------------
        #       Update the point, x_k
        # --------------------------------------------
        x_kp1 = x_trial
        _, g_kp1 = objective.f_and_grad(x_kp1)

        # ---------------------------------------------
        #       Compute the spectral step length
        # ---------------------------------------------
        ssl_kp1 = get_ssl(x_k, x_kp1, g_k, g_kp1)

        # ---------------------------------------------
        #       Update variables for the next iteration
        # ---------------------------------------------
        x_k = x_kp1
        f_k = f_trial
        g_k = g_kp1
        ssl_k = ssl_kp1

        f_history.append(f_k)
        g_norm_history.append(np.linalg.norm(g_k, ord=np.inf))
        f_memory.popleft()
        f_memory.append(f_k)

        k += 1
    else:
        # If the loop completes without breaking, we reached iter_max
        failure_message = (
            "Maximum number of iterations reached without convergence."
        )

    return OptimisationResult(
        x_optimal=x_k.tolist(),
        status=status,
        failure_message=failure_message,
        f_history=f_history,
        g_norm_history=g_norm_history,
        iter_count=k,
        f_and_g_eval_count=objective.eval_count["f_and_g"],
        proj_eval_count=objective.eval_count["proj"],
    )

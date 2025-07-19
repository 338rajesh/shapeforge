from typing import Optional, Literal
from dataclasses import dataclass

import numpy as np


class OptimisationProblem:
    def __init__(self, params: Optional[dict] = None):
        pass
        self.params = params or {}
        self.eval_count = {"f": 0, "grad_f": 0, "proj": 0}

    def f(self, x: np.ndarray) -> float:
        """
        Objective function to be minimized.
        """
        raise NotImplementedError("Objective function not implemented.")

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of the objective function.
        """
        raise NotImplementedError("Gradient function not implemented.")

    def projection(self, x: np.ndarray) -> np.ndarray:
        """
        Projection onto the feasible set.
        """
        raise NotImplementedError("Projection function not implemented.")


@dataclass
class OptimisationResult:
    x_optimal: list
    status: Literal["success", "failure"] = "success"
    failure_message: Optional[str] = None
    f_history: Optional[list] = None
    g_norm_history: Optional[list] = None
    iter_count: int = 0
    f_eval_count: int = 0
    g_eval_count: int = 0
    proj_eval_count: int = 0

    def __post_init__(self):
        n = self.iter_count
        if self.f_history and (not len(self.f_history) == n + 1):
            raise ValueError("Length of f_history must match iter_count.")
        if self.g_norm_history and (not len(self.g_norm_history) == n + 1):
            raise ValueError("Length of g_norm_history must match iter_count.")

    def to_dict(self) -> dict:
        return {
            "x_optimal": self.x_optimal,
            "iter_count": self.iter_count,
            "f_eval_count": self.f_eval_count,
            "g_eval_count": self.g_eval_count,
            "f_history": self.f_history,
            "g_norm_history": self.g_norm_history,
            "proj_eval_count": self.proj_eval_count,
        }
from shapeforge.optim import nmspg, OptimisationProblem

import numpy as np


class QuadraticProblem(OptimisationProblem):
    def __init__(self, A=None, b=None, lower=None, upper=None):
        super().__init__()
        self.A = A
        self.b = b
        self.lower = lower
        self.upper = upper

    def f(self, x):
        return 0.5 * x @ self.A @ x + self.b @ x

    def grad_f(self, x):
        return self.A @ x + self.b

    def projection(self, x):
        return np.clip(x, self.lower, self.upper)


class RosenbrockProblem(OptimisationProblem):
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim
        self.lower = -1.5 * np.ones(dim)
        self.upper = 1.5 * np.ones(dim)

    def f(self, x):
        return np.sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )

    def grad_f(self, x):
        grad = np.zeros_like(x)
        grad[0:-1] = -400 * (x[1:] - x[:-1] ** 2) * x[:-1] - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1] ** 2)
        return grad

    def projection(self, x):
        return np.clip(x, self.lower, self.upper)


def test_nmspg_quadratic_basic():
    # np.random.seed(42)
    dim = 100
    A = np.eye(dim)
    b = -np.ones(dim)
    lower = np.zeros(dim)
    upper = np.ones(dim)

    problem = QuadraticProblem(A=A, b=b, lower=lower, upper=upper)
    x0 = np.random.uniform(low=0.0, high=1.0, size=dim)

    result = nmspg(
        objective=problem,
        x0=x0,
        iter_max=500,
        epsilon=1e-6,
    )

    x_star = result.x_optimal
    assert result.status == "success", "Optimization did not succeed."
    assert np.allclose(x_star, np.ones(dim), atol=1e-4), (
        f"Did not converge to expected solution: {x_star}"
    )


def test_nmspg_quadratic():
    # np.random.seed(42)
    dim = 10
    A = np.diag(np.arange(1, dim + 1))
    b = -np.ones(dim)
    lower = np.zeros(dim)
    upper = np.ones(dim)

    problem = QuadraticProblem(A=A, b=b, lower=lower, upper=upper)
    x0 = np.random.uniform(low=0.0, high=1.0, size=dim)

    result = nmspg(
        objective=problem,
        x0=x0,
        iter_max=500,
        epsilon=1e-6,
    )

    x_star = result.x_optimal
    assert result.status == "success", "Optimization did not succeed."

    print(f"Optimal solution: {result.to_dict()}")
    print(f"Number of iterations: {result.iter_count}")
    print(f"Function evaluations: {result.f_eval_count}")
    print(f"Gradient evaluations: {result.g_eval_count}")
    print(f"Projection evaluations: {result.proj_eval_count}")


def test_nmspg_rosenbrock():
    # np.random.seed(42)
    dim = 10
    problem = RosenbrockProblem(dim=dim)
    x0 = np.random.uniform(low=-1.24, high=1.24, size=dim)

    result = nmspg(
        objective=problem,
        x0=x0,
        iter_max=5000,
        epsilon=1e-6,
    )

    x_star = result.x_optimal
    assert result.status == "success", "Optimization did not succeed."

    print("\n\nRosenbrock Problem Results:")
    print(f"Optimal solution: {x_star}")
    print(f"Number of iterations: {result.iter_count}")
    print(f"Function evaluations: {result.f_eval_count}")
    print(f"Gradient evaluations: {result.g_eval_count}")
    print(f"Projection evaluations: {result.proj_eval_count}")
    print(f"Final function value: {result.f_history[-1]}")



# if __name__ == "__main__":
#     test_nmspg_quadratic_basic()

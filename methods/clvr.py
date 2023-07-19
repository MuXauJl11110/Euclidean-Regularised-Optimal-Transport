import time
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np

from problems import BaseProblem


class CLVR:
    def __init__(
        self,
        alpha: float,
        epsilon: Optional[float] = 0.001,
        log: Optional[bool] = False,
    ):
        self.alpha = alpha
        self.epsilon = epsilon

        self.log = log
        if log:
            print("–––––––––––––––––––––––––––––")
            print("CLVR configuration:")
            print(f"alpha = {alpha}")
            print(f"epsilon = {epsilon}")
            print("–––––––––––––––––––––––––––––\n")

    def fit(
        self,
        problem: BaseProblem,
        max_iter: Optional[int] = 2000,
    ) -> Tuple[np.ndarray, int]:
        history = defaultdict(list)

        A = a = 1 / (4 * np.sqrt(problem.n + problem.n))
        X_0 = np.ones((problem.n, problem.n)) / (problem.n * problem.n)
        X_hat = X_0.copy()  # np.zeros_like(X_0)
        lambda_ = np.zeros(problem.n)
        lambda_next = np.zeros_like(lambda_)
        mu_ = np.zeros(problem.n)
        mu_next = np.zeros_like(mu_)
        z = np.zeros_like(problem.C)
        z_next = np.zeros_like(z)
        q = a * problem.C
        q_next = np.zeros_like(q)

        k = 0
        start_time = time.perf_counter()
        while True:
            X = np.maximum(self.alpha * X_0 - q, 0) / (self.alpha + problem.gamma * A)
            p = np.random.rand()
            if p > 0.5:
                lambda_next = lambda_ + 2 * self.alpha * a * (np.sum(X, axis=1) - problem.p)
            else:
                mu_next = mu_ + 2 * self.alpha * a * (np.sum(X, axis=0) - problem.q)

            a_next = 1 / 4 * np.sqrt((1 + problem.gamma * A / self.alpha) / (problem.n + problem.n))
            A_next = A + a_next

            if p > 0.5:
                z_next = z + np.outer(lambda_next - lambda_, problem.one)
            else:
                z_next = z + np.outer(problem.one, mu_next - mu_)
            q_next = q + a_next * (z_next + problem.C) + 2 * a * (z_next - z)

            X_hat = 1 / A_next * (A * X_hat + a_next * X)

            A, a = A_next, a_next
            lambda_, mu_, q, z = lambda_next.copy(), mu_next.copy(), q_next.copy(), z_next.copy()
            lamu = np.concatenate((lambda_, mu_), axis=0)
            phi = problem.phi(lamu)

            criteria_one = problem.first_crit(X_hat)
            criteria_two = problem.second_crit(X_hat, phi)
            criteria_one_X = problem.first_crit(X)
            criteria_two_X = problem.second_crit(X, phi)
            # print(criteria_one, criteria_two)
            if self.log and k % 200 == 0:
                print(
                    f"– Outer iteration {k}: {criteria_one:.6f} > {self.epsilon:.6f} "
                    + f"or {criteria_two:.6f} > {self.epsilon:.6f}"
                )
            if (criteria_one <= self.epsilon and criteria_two <= self.epsilon) or k >= max_iter:
                if self.log:
                    print(
                        f"– Outer iteration {k}: {criteria_one:.6f} <= {self.epsilon:.6f} "
                        + f"or {criteria_two:.6f} <= {self.epsilon:.6f}"
                    )
                return X_hat, history

            k += 1
            history["criteria_one"].append(criteria_one)
            history["criteria_two"].append(criteria_two)
            history["criteria_one_X"].append(criteria_one_X)
            history["criteria_two_X"].append(criteria_two_X)
            history["time"].append(time.perf_counter() - start_time)

import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from problems.base import BaseProblem


class PrimalDualAAMLS:
    def __init__(
        self,
        epsilon: float,
        log: Optional[bool] = False,
    ):
        """
        Primal-Dual Accelerated Alternating Minimization for Transport Problem
        Algorithm 2 https://arxiv.org/pdf/1906.03622.pdf
        :param epsilon: desired accuracy
        :param log: if true log output will be printed
        """
        self.epsilon = epsilon

        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("PDAAM-LS configuration:")
            print(f"eps = {epsilon}")
            print("–––––––––––––––––––––––––––––\n")

    def ternary_search_beta(self, l: float, r: float, f: Callable[[float], float]):
        while r - l > self.epsilon:
            m1 = l + (r - l) / 3
            m2 = r - (r - l) / 3
            if f(m1) < f(m2):
                r = m2
            else:
                l = m1
        return r

    def fit(
        self,
        problem: BaseProblem,
        max_iter: Optional[int] = 2000,
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        history = defaultdict(list)
        A = A_new = 0
        eta = np.zeros(2 * problem.n)
        dzeta = np.zeros(2 * problem.n)
        kappa = np.zeros(2 * problem.n)

        dual_func_beta = lambda beta: problem.phi(eta + beta * (dzeta - eta))
        X_hat = np.zeros((problem.n, problem.n))

        k = 0
        start_time = time.perf_counter()
        while True:
            beta = self.ternary_search_beta(0, 1, dual_func_beta)
            kappa = beta * dzeta + (1 - beta) * eta

            grad_phi_lambda = problem.grad_phi_lambda(kappa)
            grad_phi_mu = problem.grad_phi_mu(kappa)
            grad_phi_kappa = np.concatenate((grad_phi_lambda, grad_phi_mu), axis=0)

            grad_phi_lambda_norm = np.sum(grad_phi_lambda**2)
            grad_phi_mu_norm = np.sum(grad_phi_mu**2)
            grad_phi_kappa_norm = grad_phi_lambda_norm + grad_phi_mu_norm

            eta_new = kappa.copy()
            if grad_phi_lambda_norm > grad_phi_mu_norm:
                eta_new = problem.update_lambda(eta_new)
            else:
                eta_new = problem.update_mu(eta_new)

            phi_eta_new = problem.phi(eta_new)
            phi_kappa = problem.phi(kappa)
            delta = problem.pdaam_delta(phi_eta_new, phi_kappa)
            d = delta**2 - 2 * grad_phi_kappa_norm * A * delta
            a = (-delta + np.sqrt(d)) / grad_phi_kappa_norm
            A_new = A + a

            dzeta = problem.update_momentum(dzeta, a, grad_phi_kappa)
            X_hat = (a * problem.X_hat(kappa) + A * X_hat) / (A_new)

            A = A_new

            criteria_one = problem.first_crit(X_hat)
            criteria_two = problem.second_crit(X_hat, phi_eta_new)
            if self.log and k % 1000 == 0:
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
            history["time"].append(time.perf_counter() - start_time)

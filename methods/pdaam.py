import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from problems.base import BaseProblem


class PrimalDualAAM:
    def __init__(
        self,
        epsilon: float,
        log: Optional[bool] = False,
    ):
        """
        Primal-Dual Accelerated Alternating Minimization for Transport Problem
        Algorithm 4 https://arxiv.org/pdf/1906.03622.pdf
        :param epsilon: desired accuracy
        :param log: if true log output will be printed
        """
        self.epsilon = epsilon

        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("PDAAM configuration:")
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
        L: Optional[float] = 1.0,
        max_iter: Optional[int] = 2000,
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        history = defaultdict(list)
        alpha = 0
        eta = np.zeros(2 * problem.n)
        dzeta = np.zeros(2 * problem.n)
        kappa = np.zeros(2 * problem.n)

        X_hat = np.zeros((problem.n, problem.n))

        k = 0
        start_time = time.perf_counter()
        while True:
            L_new = L / 2

            t = 0
            while True:
                t += 1
                alpha_new = 1 / 2 / L_new + np.sqrt(1 / (4 * L_new**2) + alpha**2 * L / L_new)
                tau = 1 / alpha_new / L_new
                kappa = tau * dzeta + (1 - tau) * eta

                grad_phi_lambda = problem.grad_phi_lambda(kappa)
                grad_phi_mu = problem.grad_phi_mu(kappa)
                grad_phi_kappa = np.concatenate((grad_phi_lambda, grad_phi_mu), axis=0)

                grad_phi_lambda_norm = np.sum(grad_phi_lambda**2)
                grad_phi_mu_norm = np.sum(grad_phi_mu**2)
                grad_phi_kappa_norm = grad_phi_lambda_norm + grad_phi_mu_norm

                eta_new = kappa.copy()  # because kappa is used in X_hat update
                if grad_phi_lambda_norm > grad_phi_mu_norm:
                    eta_new = problem.update_lambda(eta_new)
                else:
                    eta_new = problem.update_mu(eta_new)

                phi_eta_new = problem.phi(eta_new)
                phi_kappa = problem.phi(kappa)
                if problem.pdaam_ls_condition(phi_eta_new, phi_kappa, grad_phi_kappa_norm / 2 / L_new):
                    X_hat = (alpha_new * problem.X_hat(kappa) + L * alpha**2 * X_hat) / (L_new * alpha_new**2)
                    dzeta = problem.update_momentum(dzeta, alpha_new, grad_phi_kappa)
                    eta = eta_new.copy()
                    alpha = alpha_new
                    L = L_new
                    break
                L_new *= 2

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

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from problems import BaseProblem


class APDAGD:
    def __init__(
        self,
        epsilon: Optional[float] = 0.001,
        log: Optional[bool] = False,
    ):
        """
        Algorithm 3 from https://arxiv.org/pdf/1802.04367.pdf
        :param epsilon: desired accuracy
        :param log: if true log output will be printed
        """
        self.epsilon = epsilon

        self.log = log
        if log:
            print("–––––––––––––––––––––––––––––")
            print("APDAGD configuration:")
            print(f"epsilon = {epsilon}")
            print("–––––––––––––––––––––––––––––\n")

    def fit(
        self,
        problem: BaseProblem,
        L: Optional[float] = 1.0,
        max_iter: Optional[int] = 2000,
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        history = defaultdict(list)

        beta = 0.0
        dzeta = np.zeros(2 * problem.n)
        eta = np.zeros(2 * problem.n)

        X_hat = np.zeros((problem.n, problem.n))
        k = 0
        T = 0
        start_time = time.perf_counter()
        while True:
            L = L / 2

            t = 0
            while True:
                t += 1
                T += 1

                alpha_new = (1 + np.sqrt(4 * L * beta + 1)) / 2 / L
                beta_new = beta + alpha_new
                tau = alpha_new / beta_new
                kappa = tau * dzeta + (1 - tau) * eta
                grad_phi_kappa = np.concatenate((problem.grad_phi_lambda(kappa), problem.grad_phi_mu(kappa)), axis=0)

                dzeta_new = problem.update_momentum(dzeta, alpha_new, grad_phi_kappa)
                eta_new = tau * dzeta_new + (1 - tau) * eta

                phi_eta_new = problem.phi(eta_new)
                phi_approx = problem.phi_approx(kappa, grad_phi_kappa, eta_new, L)

                if problem.apdagd_ls_condition(phi_eta_new, phi_approx):
                    dzeta = dzeta_new.copy()
                    eta = eta_new.copy()
                    beta = beta_new
                    break
                L = L * 2

            X_hat = tau * problem.X_hat(kappa) + (1 - tau) * X_hat

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

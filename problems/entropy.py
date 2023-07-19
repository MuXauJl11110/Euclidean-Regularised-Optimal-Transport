import warnings
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from problems.base import BaseProblem


class EntropyRegularizedOTProblem(BaseProblem):
    def f(self, x: np.ndarray) -> float:
        y = x.copy()
        y[x == 0.0] = 1.0
        return (self.C * x).sum() + self.gamma * (x * np.log(y)).sum()

    def phi(self, lamu: np.ndarray) -> float:
        log_B = self._log_B(lamu)

        return self.gamma * (logsumexp(log_B) - lamu[: self.n].dot(self.p) - lamu[self.n :].dot(self.q))

    def phi_approx(self, x: np.ndarray, grad_x: np.ndarray, y: np.ndarray, L: float) -> bool:
        delta = y - x
        return self.phi(x) + grad_x.dot(delta) + L / 2 * np.sum(delta**2)

    def grad_phi_lambda(self, lamu: np.ndarray) -> np.ndarray:
        B_stable, Bs_stable = self.B_stable(lamu)

        lambda_stable = B_stable.sum(axis=1)
        return self.gamma * (lambda_stable / Bs_stable - self.p)

    def grad_phi_mu(self, lamu: np.ndarray) -> np.ndarray:
        B_stable, Bs_stable = self.B_stable(lamu)

        mu_stable = B_stable.sum(axis=0)
        return self.gamma * (mu_stable / Bs_stable - self.q)

    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        B_stable, _ = self.B_stable(lamu)
        lambda_stable = B_stable.sum(axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                lambda_step = self.p / lambda_stable
            except Warning as e:
                lambda_stable /= lambda_stable.max()
                lambda_stable[lambda_stable < 1e-150] = 1e-150
                lambda_step = self.p / lambda_stable
        lambda_step /= lambda_step.max()
        lamu[: self.n] += np.log(lambda_step)
        return lamu

    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        B_stable, _ = self.B_stable(lamu)
        mu_stable = B_stable.sum(axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                mu_step = self.q / mu_stable
            except Warning as e:
                mu_stable /= mu_stable.max()
                mu_stable[mu_stable < 1e-150] = 1e-150
                mu_step = self.q / mu_stable
        mu_step /= mu_step.max()
        lamu[self.n :] += np.log(mu_step)
        return lamu

    def update_momentum(self, x: np.ndarray, alpha: float, grad: float) -> np.ndarray:
        return x - alpha * grad

    def _log_B(self, lamu: np.ndarray) -> np.ndarray:
        return np.outer(lamu[: self.n], self.one) + np.outer(self.one, lamu[self.n :]) - self.C / self.gamma

    def B_stable(self, lamu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        log_B = self._log_B(lamu)
        max_log_B = log_B.max()
        log_B_stable = log_B - max_log_B
        B_stable = np.exp(log_B_stable)

        return B_stable, np.sum(B_stable)

    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        B_stable, Bs_stable = self.B_stable(lamu)

        return B_stable / Bs_stable

    def conv_crit(self, lamu: np.ndarray) -> float:
        B_stable, Bs_stable = self.B_stable(lamu)
        lambda_stable = np.sum(B_stable, axis=1)
        mu_stable = np.sum(B_stable, axis=0)

        return np.sum(np.abs(lambda_stable / Bs_stable - self.p)) + np.sum(np.abs(mu_stable / Bs_stable - self.q))

    def apdagd_ls_condition(self, phi_x: float, phi_approx_y: float) -> bool:
        return phi_x <= phi_approx_y

    def pdaam_ls_condition(self, phi_x: float, phi_y: float, term: float) -> bool:
        return phi_x <= phi_y - term

    def pdaam_delta(self, phi_x: float, phi_y) -> float:
        return phi_x - phi_y

    def first_crit(self, X_hat: np.ndarray) -> float:
        return np.sum(self.C * (self.B_round(X_hat) - X_hat))

    def second_crit(self, X_hat: np.ndarray, phi_eta: float) -> float:
        return abs(self.f(X_hat) + phi_eta)

from abc import ABC, abstractmethod

import numpy as np


class BaseProblem(ABC):
    r"""
    Base class for implementation $\gamma$ regularized OT problems with two dual variables $\lambda$ and $\mu$.
    """

    def __init__(self, gamma: float, n: int, C: np.ndarray, p: np.ndarray, q: np.ndarray):
        """
        :param float gamma: Regularization parameter.
        :param int n: Problem size.
        :param np.ndarray C: Cost matrix.
        :param np.ndarray p: First discrete probability measure.
        :param np.ndarray q: Second discrete probability measure.
        """
        self.gamma, self.n = gamma, n

        self.C, self.p, self.q = C, p, q

        self.one = np.ones(n)

    def B_round(self, x: np.ndarray) -> np.ndarray:
        x[x == 0.0] = 1e-16
        r = self.p / x.dot(self.one)
        r[r > 1] = 1.0
        F = np.diag(r).dot(x)

        c = self.q / (x.T).dot(self.one)
        c[c > 1] = 1.0
        F = F.dot(np.diag(c))

        err_r = self.p - F.dot(self.one)
        err_c = self.q - (F.T).dot(self.one)

        return F + np.outer(err_r, err_c) / abs(err_r).sum()

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        """
        Computes the value of function at point x.
        """
        pass

    @abstractmethod
    def phi(self, lamu: np.ndarray) -> float:
        r"""
        Computes the value of the dual function at point ($\lambda$, $\mu$).
        """
        pass

    @abstractmethod
    def phi_approx(self, x: np.ndarray, grad_x: np.ndarray, y: np.ndarray, L: float) -> bool:
        r"""
        Return phi approximation in x.
        """
        pass

    @abstractmethod
    def grad_phi_lambda(self, lamu: np.ndarray) -> np.ndarray:
        r"""
        Computes gradient by $\lambda$ of the dual function at point ($\lambda$, $\mu$).
        """
        pass

    @abstractmethod
    def grad_phi_mu(self, lamu: np.ndarray) -> np.ndarray:
        r"""
        Computes gradient by $\mu$ of the dual function at point ($\lambda$, $\mu$).
        """
        pass

    @abstractmethod
    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        r"""
        Updates dual $\lambda$ variable inplace.
        """
        pass

    @abstractmethod
    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        r"""
        Updates dual $\mu$ variable inplace.
        """
        pass

    @abstractmethod
    def update_momentum(self, x: np.ndarray, alpha: float, grad: float) -> np.ndarray:
        r"""
        Updates momentum's term.
        """
        pass

    @abstractmethod
    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        r"""
        Some solution of the convex problem
        $$
        \max_{x \in Q} \big(-f(x)-\langel A^T\lambda, x \rangle\big)
        $$
        """
        pass

    @abstractmethod
    def conv_crit(self, lamu: np.ndarray) -> float:
        r"""
        Computes convergency criteria.
        """
        pass

    @abstractmethod
    def first_crit(self, X_hat: np.ndarray) -> float:
        r"""
        Computes first convergency criteria.
        """
        pass

    @abstractmethod
    def second_crit(self, X_hat: np.ndarray, phi_eta: float) -> float:
        r"""
        Computes second convergency criteria.
        """
        pass

    @abstractmethod
    def apdagd_ls_condition(self, phi_x: float, phi_approx_y: float) -> bool:
        r"""
        Return true if line search conditions of APDAGD method are met.
        """
        pass

    @abstractmethod
    def pdaam_ls_condition(self, phi_x: float, phi_y: float, term: float) -> bool:
        r"""
        Return true if line search conditions of PDAAM method are met.
        """
        pass

    @abstractmethod
    def pdaam_delta(self, phi_x: float, phi_y) -> float:
        r"""
        Return delta for solving quadratic equation.
        """
        pass

import numpy as np

from problems.base import BaseProblem


class EuclideanRegularizedOTProblem(BaseProblem):
    def f(self, x: np.ndarray) -> float:
        return np.sum(self.C * x) + self.gamma / 2 * np.sum(x**2)

    def phi(self, lamu: np.ndarray) -> float:
        gamma_X = self.gamma_X(lamu)
        norm_X = -1 / (2 * self.gamma) * np.sum(gamma_X**2)

        return norm_X - lamu[: self.n].dot(self.p) - lamu[self.n :].dot(self.q)

    def phi_approx(self, x: np.ndarray, grad_x: np.ndarray, y: np.ndarray, L: float) -> bool:
        delta = y - x
        return self.phi(x) + grad_x.dot(delta) - L / 2 * np.sum(delta**2)

    def grad_phi_lambda(self, lamu: np.ndarray) -> np.ndarray:
        gamma_X = self.gamma_X(lamu)

        return 1 / self.gamma * np.sum(gamma_X, axis=1) - self.p

    def grad_phi_mu(self, lamu: np.ndarray) -> np.ndarray:
        gamma_X = self.gamma_X(lamu)

        return 1 / self.gamma * np.sum(gamma_X, axis=0) - self.q

    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        lambda_, mu_ = lamu[: self.n], lamu[self.n :]

        for i, row_i in enumerate(self.C):
            indexes = np.where(
                np.sum(
                    np.maximum(-np.outer(self.one, row_i + mu_) - np.outer(-(row_i + mu_), self.one), 0),
                    axis=1,
                )
                <= self.gamma * self.p[i],
            )[0]

            if len(indexes) > 0:
                lambda_[i] = -(self.gamma * self.p[i] + np.sum(row_i[indexes] + mu_[indexes])) / len(indexes)
            else:
                lambda_[i] = -self.gamma * self.p[i]
        lamu[: self.n] = lambda_
        return lamu

    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        lambda_, mu_ = lamu[: self.n], lamu[self.n :]

        for j, column_j in enumerate(self.C.T):
            indexes = np.where(
                np.sum(
                    np.maximum(
                        -np.outer(column_j + lambda_, self.one) - np.outer(self.one, -(column_j + lambda_)),
                        0,
                    ),
                    axis=0,
                )
                <= self.gamma * self.q[j],
            )[0]
            if len(indexes) > 0:
                mu_[j] = -(self.gamma * self.q[j] + np.sum(column_j[indexes] + lambda_[indexes])) / len(indexes)
            else:
                mu_[j] = -self.gamma * self.q[j]
        lamu[self.n :] = mu_
        return lamu

    def update_momentum(self, x: np.ndarray, alpha: float, grad: float) -> np.ndarray:
        return x + alpha * grad

    def gamma_X(self, lamu: np.ndarray) -> np.ndarray:
        gamma_X = self.C + np.outer(lamu[: self.n], self.one) + np.outer(self.one, lamu[self.n :])
        return np.maximum(-gamma_X, 0)

    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        return 1 / self.gamma * self.gamma_X(lamu)

    def conv_crit(self, lamu: np.ndarray) -> float:
        X = self.X_hat(lamu)

        return np.sum(np.abs(np.sum(X, axis=1) - self.p)) + np.sum(np.abs(np.sum(X, axis=0) - self.q))

    def apdagd_ls_condition(self, phi_x: float, phi_approx_y: float) -> bool:
        return phi_x >= phi_approx_y

    def pdaam_ls_condition(self, phi_x: float, phi_y: float, term: float) -> bool:
        return phi_x >= phi_y + term

    def pdaam_delta(self, phi_x: float, phi_y) -> float:
        return phi_y - phi_x

    def first_crit(self, X_hat: np.ndarray) -> float:
        # return np.sqrt(np.sum((np.sum(X_hat, axis=1) - self.p) ** 2 + (np.sum(X_hat, axis=0) - self.q) ** 2))
        return np.sum(self.C * (self.B_round(X_hat) - X_hat))

    def second_crit(self, X_hat: np.ndarray, phi_eta: float) -> float:
        return abs(self.f(X_hat) - phi_eta)

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from problems.base import BaseProblem


class Sinkhorn:
    def __init__(
        self,
        epsilon: Optional[float] = 0.001,
        log: Optional[bool] = False,
    ):
        """
        Algorithm 1 from https://arxiv.org/pdf/1802.04367.pdf
        :param epsilon: desired accuracy
        :param log: if true log output will be printed
        """
        self.epsilon = epsilon
        self.log = log
        if log:
            print("–––––––––––––––––––––––––––––")
            print("Sinkhorn configuration:")
            print(f"eps = {self.epsilon}")
            print("–––––––––––––––––––––––––––––\n")

    def fit(
        self, problem: BaseProblem, lamu: np.ndarray, max_iter: Optional[int] = 2000
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        history = defaultdict(list)

        k = 0
        start_time = time.perf_counter()
        while True:
            if k % 2 == 0:
                lamu = problem.update_lambda(lamu)
            else:
                lamu = problem.update_mu(lamu)

            conv_crit = problem.conv_crit(lamu)
            if self.log and k % 1000 == 0:
                print(f"Iteration {k}: metric {conv_crit} > {self.epsilon}")

            if conv_crit < self.epsilon or k >= max_iter:
                if self.log:
                    print(f"Iteration {k}: metric {conv_crit} < {self.epsilon}")
                return problem.B_round(problem.X_hat(lamu)), history
            k += 1
            history["conv_crit"].append(conv_crit)
            history["time"].append(time.perf_counter() - start_time)

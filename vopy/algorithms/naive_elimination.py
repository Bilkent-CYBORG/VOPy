import logging
from typing import Optional

import numpy as np

from vopy.algorithms.algorithm import PALAlgorithm
from vopy.datasets import get_dataset_instance
from vopy.maximization_problem import ProblemFromDataset

from vopy.order import PolyhedralConeOrder


class NaiveElimination(PALAlgorithm):
    """
    Implement the Naive Elimination algorithm.

    :param epsilon: Determines the accuracy of the PAC-learning framework.
    :type epsilon: float
    :param delta: Determines the success probability of the PAC-learning framework.
    :type delta: float
    :param dataset_name: Name of the dataset to be used.
    :type dataset_name: str
    :param order: Order to be used.
    :type order: Order
    :param noise_var: Variance of the Gaussian sampling noise.
    :type noise_var: float
    :param L: Number of samples to be taken for each arm. If `None`, theoretical sampling count
        is used.
    :type L: Optional[int]

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.

    Example:
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.algorithms import NaiveElimination
        >>>
        >>> epsilon, delta, noise_var = 0.1, 0.05, 0.01
        >>> dataset_name = "DiscBrake"
        >>> order_right = ComponentwiseOrder(2)
        >>>
        >>> algorithm = NaiveElimination(epsilon, delta, dataset_name, order_right, noise_var)
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = algorithm.P

    Reference:
        "Vector Optimization with Stochastic Bandit Feedback",
        Ararat, Tekin, AISTATS, '23
        https://proceedings.mlr.press/v206/ararat23a.html
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        dataset_name: str,
        order: PolyhedralConeOrder,
        noise_var: float,
        L: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order

        self.dataset = get_dataset_instance(dataset_name)
        self.m = self.dataset.out_dim

        self.K = len(self.dataset.in_data)
        if L is None:  # Use theoretical sampling count if not given.
            if not hasattr(order.ordering_cone, "beta"):
                raise AttributeError("Ordering complexity needs to be defined.")
            ordering_complexity = order.ordering_cone.beta

            c = 1 + np.sqrt(2)  # Any c>0 should suffice according to Lemma B.12.
            self.L = np.ceil(
                4
                * ((c * noise_var * ordering_complexity / self.epsilon) ** 2)
                * np.log(4 * self.m / (2 * self.delta / (self.K * (self.K - 1))))
            ).astype(int)
        else:
            self.L = L

        self.problem = ProblemFromDataset(self.dataset, noise_var)

        self.samples = np.empty((self.K, 0, self.m))

        self.round = 0
        self.sample_count = 0

    def run_one_step(self) -> bool:
        """
        Run one step of the algorithm and return algorithm status.

        :return: True if the algorithm is over, False otherwise.
        :rtype: bool
        """
        if self.round == self.L:
            return True

        self.round += 1

        new_samples = self.problem.evaluate(self.dataset.in_data)
        self.samples = np.concatenate([self.samples, np.expand_dims(new_samples, axis=-2)], axis=-2)

        self.sample_count += self.K

        if self.L <= 50 or self.round % 50 == 0:
            round_str = f"Round {self.round}"

            logging.info(f"{round_str}:Sample count {self.sample_count}")

        return self.round == self.L

    @property
    def P(self) -> np.ndarray:
        """
        Calculate the Pareto set ordering by sample means.

        :return: Indices for the Pareto set.
        :rtype: np.ndarray
        """
        return self.order.get_pareto_set(self.samples.mean(axis=-2))

import logging

import numpy as np

from vopy.acquisition import optimize_decoupled_acqf_discrete, ThompsonEntropyDecoupledAcquisition
from vopy.algorithms.algorithm import Algorithm
from vopy.datasets import get_dataset_instance
from vopy.maximization_problem import DecoupledEvaluationProblem, ProblemFromDataset
from vopy.models import get_gpytorch_modellist_w_known_hyperparams, GPyTorchModelListExactModel

from vopy.order import PolyhedralConeOrder


class DecoupledGP(Algorithm):
    """
    Implement a partially observable GP-based minimal algorithm that runs an acquisition method
    for until a budget is reached.

    :param dataset_name: Name of the dataset to be used.
    :type dataset_name: str
    :param order: Order to be used.
    :type order: Order
    :param noise_var: Variance of the Gaussian sampling noise.
    :type noise_var: float
    :param cost_budget: Cost budget for the algorithm.
    :type cost_budget: float
    :param costs: Cost associated with sampling each objective.
    :type costs: list
    :param batch_size: Number of samples to be taken in each round. Defaults to 1.
    :type batch_size: int

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.
    It uses Gaussian Process regression to model the rewards and confidence
    regions.

    Example:
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.algorithms import DecoupledGP
        >>>
        >>> noise_var = 0.01
        >>> cost_budget = 64
        >>> dataset_name = "DiscBrake"
        >>> order_right = ComponentwiseOrder(2)
        >>>
        >>> algorithm = DecoupledGP(
        >>>     dataset_name, order_right, noise_var, cost_budget
        >>> )
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = algorithm.P
    """

    def __init__(
        self,
        dataset_name: str,
        order: PolyhedralConeOrder,
        noise_var: float,
        cost_budget: float,
        costs: list,
        batch_size: int = 1,
    ) -> None:
        super().__init__()

        self.order = order
        self.batch_size = batch_size
        self.costs = np.array(costs)
        self.cost_budget = cost_budget

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        self.points = dataset.in_data
        self.problem = DecoupledEvaluationProblem(ProblemFromDataset(dataset, noise_var))

        self.model: GPyTorchModelListExactModel = get_gpytorch_modellist_w_known_hyperparams(
            self.problem,
            initial_sample_cnt=1,
            noise_var=noise_var,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.P = set()
        self.round = 0
        # TODO: Initial samples are managed in model preparation, they're not taken into account.
        self.sample_count = 0
        self.total_cost = 0.0

    def pareto_updating(self):
        """
        Identify the designs that are optimal using the mean estimates.
        """
        mu, covars = self.model.predict(self.points)
        self.P = self.order.get_pareto_set(mu)
        logging.debug(f"Pareto: {str(self.P)}")

    def evaluating(self):
        """
        Observe the self.batch_size number of designs from active designs, selecting by
        the `ThompsonEntropyDecoupledAcquisition` acquisition method and update the model.
        """
        acq = ThompsonEntropyDecoupledAcquisition(self.model, order=self.order, costs=self.costs)
        candidate_list, acq_values, eval_indices = optimize_decoupled_acqf_discrete(
            acq, self.batch_size, choices=self.points
        )

        observations = self.problem.evaluate(candidate_list, eval_indices)

        self.sample_count += len(candidate_list)
        if self.costs is not None:
            self.total_cost += np.sum(self.costs[eval_indices])
        self.model.add_sample(candidate_list, observations, eval_indices)
        self.model.update()

    def run_one_step(self) -> bool:
        """
        Run one step of the algorithm and return algorithm status.

        :return: True if the algorithm is over, False otherwise.
        :rtype: bool
        """
        if self.total_cost >= self.cost_budget:
            return True

        self.round += 1

        round_str = f"Round {self.round}"

        logging.info(f"{round_str}:Evaluating")
        self.evaluating()

        logging.info(f"{round_str}:Pareto update")
        self.pareto_updating()

        logging.info(f"{round_str}:There are {len(self.P)} designs in set P.")

        logging.info(
            f"{round_str}:" f"Sample count {self.sample_count}, Cost {self.total_cost:.2f}"
        )

        return self.total_cost >= self.cost_budget

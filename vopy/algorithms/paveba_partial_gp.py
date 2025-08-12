import logging
from typing import Literal, Optional

import numpy as np

from vopy.acquisition import MaxVarianceDecoupledAcquisition, optimize_decoupled_acqf_discrete
from vopy.algorithms.algorithm import PALAlgorithm
from vopy.confidence_region import confidence_region_is_covered, confidence_region_is_dominated
from vopy.datasets import get_dataset_instance
from vopy.design_space import FixedPointsDesignSpace
from vopy.maximization_problem import DecoupledEvaluationProblem, ProblemFromDataset
from vopy.models import get_gpytorch_modellist_w_known_hyperparams, GPyTorchModelListExactModel

from vopy.order import PolyhedralConeOrder


class PaVeBaPartialGP(PALAlgorithm):
    """
    Implement the partially observable GP-based Pareto Vector Bandits (PaVeBa) algorithm.

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
    :param conf_contraction: Contraction coefficient to shrink the confidence
        regions empirically. Defaults to 32.
    :type conf_contraction: float
    :param costs: Cost associated with sampling each objective. Defaults to None.
    :type costs: Optional[list]
    :param cost_budget: Cost budget for the algorithm. Defaults to None.
    :type cost_budget: Optional[float]
    :param confidence_type: Specifies if the algorithm uses ellipsoidal or
        hyperrectangular confidence regions. Defaults to "hyperrectangle".
    :type confidence_type: Literal["hyperrectangle", "hyperellipsoid"]
    :param batch_size: Number of samples to be taken in each round. Defaults to 1.
    :type batch_size: int

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.
    It uses Gaussian Process regression to model the rewards and confidence
    regions.

    Example:
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.algorithms import PaVeBaPartialGP
        >>>
        >>> epsilon, delta, noise_var = 0.1, 0.05, 0.01
        >>> cost_budget = 64
        >>> dataset_name = "DiscBrake"
        >>> order_right = ComponentwiseOrder(2)
        >>>
        >>> algorithm = PaVeBaPartialGP(
        >>>     epsilon, delta, dataset_name, order_right, noise_var, cost_budget=cost_budget
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
        epsilon: float,
        delta: float,
        dataset_name: str,
        order: PolyhedralConeOrder,
        noise_var: float,
        conf_contraction: float = 32,
        costs: Optional[list] = None,
        cost_budget: Optional[float] = None,
        confidence_type: Literal["hyperrectangle", "hyperellipsoid"] = "hyperrectangle",
        batch_size: int = 1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction
        self.costs = np.array(costs) if costs is not None else costs
        self.cost_budget = cost_budget if cost_budget is not None else np.inf

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type=confidence_type
        )
        self.problem = DecoupledEvaluationProblem(ProblemFromDataset(dataset, noise_var))

        self.model: GPyTorchModelListExactModel = get_gpytorch_modellist_w_known_hyperparams(
            self.problem,
            initial_sample_cnt=1,
            noise_var=noise_var,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.cone_alpha = self.order.ordering_cone.alpha.flatten()
        self.cone_alpha_eps = self.cone_alpha * self.epsilon

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.U = set()
        self.round = 0
        # TODO: Initial samples are managed in model preparation, they're not taken into account.
        self.sample_count = 0
        self.total_cost = 0.0

    def modeling(self):
        """
        Construct the confidence regions of all active designs given all past observations.
        """
        self.alpha_t = self.compute_alpha()
        A = self.S.union(self.U)
        self.design_space.update(self.model, self.alpha_t, list(A))

    def discarding(self):
        """
        Discard the designs that are highly likely to be suboptimal using the confidence regions.
        """
        A = self.S.union(self.U)

        to_be_discarded = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in A:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, 0):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def pareto_updating(self):
        """
        Identify the designs that are highly likely to be `epsilon`-optimal
        using the confidence regions.
        """
        A = self.S.union(self.U)

        new_pareto_pts = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in A:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_conf, pt_p_conf, self.cone_alpha_eps
                ):
                    break
            else:
                new_pareto_pts.append(pt)

        for pt in new_pareto_pts:
            self.S.remove(pt)
            self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def useful_updating(self):
        """
        Identify the designs that are decided to be Pareto, that would help with decisions of
        other designs.
        """
        self.U = set()
        for pt in self.P:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in self.S:
                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_p_conf, pt_conf, self.cone_alpha_eps
                ):
                    self.U.add(pt)
                    break
        logging.debug(f"Useful: {str(self.U)}")

    def evaluating(self):
        """
        Observe the self.batch_size number of designs from active designs, selecting by
        largest variance across designs and objectives and update the model.
        """
        A = self.S.union(self.U)
        acq = MaxVarianceDecoupledAcquisition(self.model, costs=self.costs)
        active_pts = self.design_space.points[list(A)]
        candidate_list, acq_values, eval_indices = optimize_decoupled_acqf_discrete(
            acq, self.batch_size, choices=active_pts
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
        if len(self.S) == 0 or self.total_cost >= self.cost_budget:
            return True

        self.round += 1

        round_str = f"Round {self.round}"

        logging.debug(f"{round_str}:Evaluating")
        self.evaluating()

        logging.debug(f"{round_str}:Modeling")
        self.modeling()

        logging.debug(f"{round_str}:Discarding")
        self.discarding()

        logging.debug(f"{round_str}:Pareto update")
        self.pareto_updating()

        logging.debug(f"{round_str}:Useful update")
        self.useful_updating()

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        logging.debug(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0 or self.total_cost >= self.cost_budget

    def compute_alpha(self):
        """
        Compute the radius of the confidence regions of the current round to be used in modeling.

        :return: The radius of the confidence regions.
        :rtype: float
        """
        alpha = 2 * np.log(
            (np.pi**2 * self.round**2 * self.design_space.cardinality) / (3 * self.delta)
        )

        return alpha / self.conf_contraction

import logging

import numpy as np

from vopy.acquisition import MaxDiagonalAcquisition, optimize_acqf_discrete
from vopy.algorithms.algorithm import PALAlgorithm
from vopy.confidence_region import (
    confidence_region_check_dominates,
    confidence_region_is_covered,
    confidence_region_is_dominated,
)
from vopy.datasets import get_dataset_instance
from vopy.design_space import FixedPointsDesignSpace
from vopy.maximization_problem import ProblemFromDataset
from vopy.models import get_gpytorch_model_w_known_hyperparams, IndependentExactGPyTorchModel

from vopy.order import ComponentwiseOrder


class EpsilonPAL(PALAlgorithm):
    r"""
    Implement the GP-based :math:`\epsilon`-Pareto Active Learning (:math:`\epsilon`-PAL) algorithm.

    :param epsilon: Determines the accuracy of the PAC-learning framework.
    :type epsilon: float
    :param delta: Determines the success probability of the PAC-learning framework.
    :type delta: float
    :param dataset_name: Name of the dataset to be used.
    :type dataset_name: str
    :param noise_var: Variance of the Gaussian sampling noise.
    :type noise_var: float
    :param conf_contraction: Contraction coefficient to shrink the
        confidence regions empirically. Defaults to 9.
    :type conf_contraction: float
    :param batch_size: Number of samples to be taken in each round. Defaults to 1.
    :type batch_size: int

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.
    It uses Gaussian Process regression to model the rewards and confidence
    regions.

    Example:
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.algorithms import EpsilonPAL
        >>>
        >>> epsilon, delta, noise_var = 0.1, 0.05, 0.01
        >>> dataset_name = "DiscBrake"
        >>>
        >>> algorithm = EpsilonPAL(epsilon, delta, dataset_name, noise_var)
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = algorithm.P

    Reference:
        ":math:`\epsilon`-PAL: An Active Learning Approach to the Multi-Objective Optimization
        Problem",
        Zuluaga, Krause, Püschel, JMLR, '16
        https://jmlr.org/papers/v17/15-047.html
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        dataset_name: str,
        noise_var: float,
        conf_contraction: float = 9,
        batch_size: int = 1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim
        self.order = ComponentwiseOrder(dim=self.m)

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            IndependentExactGPyTorchModel,
            self.problem,
            initial_sample_cnt=1,
            noise_var=noise_var,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.round = 0
        self.sample_count = 0

    def modeling(self):
        """
        Updates confidence regions for all active designs.
        """
        self.beta = self.compute_beta()
        # Active nodes, union of sets s_t and p_t at the beginning of round t
        W = self.S.union(self.P)
        self.design_space.update(self.model, self.beta, list(W))

    def discarding(self):
        """
        Discards designs that are highly likely to be dominated based on
        current confidence regions.
        """
        pessimistic_set = self.compute_pessimistic_set()
        difference = self.S.difference(pessimistic_set)

        to_be_discarded = []
        for pt in difference:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in pessimistic_set:
                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, self.epsilon):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def epsiloncovering(self):
        """
        Identify and remove designs from `S` that are not covered by the confidence region of
        other designs, adding them to `P` as Pareto-optimal.
        """
        W = self.S.union(self.P)

        new_pareto_pts = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in W:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(self.order, pt_conf, pt_p_conf, self.epsilon):
                    break
            else:
                new_pareto_pts.append(pt)

        for pt in new_pareto_pts:
            self.S.remove(pt)
            self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def evaluating(self):
        """
        Selects self.batch_size number of designs for evaluation based on maximum diagonals and
        updates the model with new observations.
        """
        W = self.S.union(self.P)
        acq = MaxDiagonalAcquisition(self.design_space)
        active_pts = self.design_space.points[list(W)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        observations = self.problem.evaluate(candidate_list)

        self.sample_count += len(candidate_list)
        self.model.add_sample(candidate_list, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        r"""
        Executes one iteration of the :math:`\epsilon`-PAL algorithm, performing modeling,
        discarding, epsilon-covering, and evaluating phases. Returns the algorithm termination
        status.

        :return: True if the set `S` is empty, indicating termination, False otherwise.
        :rtype: bool
        """
        if len(self.S) == 0:
            return True

        round_str = f"Round {self.round}"

        logging.debug(f"{round_str}:Modeling")
        self.modeling()

        logging.debug(f"{round_str}:Discarding")
        self.discarding()

        logging.debug(f"{round_str}:Epsilon-Covering")
        self.epsiloncovering()

        logging.debug(f"{round_str}:Evaluating")
        if self.S:  # If S_t is not empty
            self.evaluating()

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        self.round += 1

        logging.debug(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self) -> np.ndarray:
        """
        Compute the confidence scaling parameter `beta` for Gaussian Process modeling.

        :return: A vector representing `beta` for each dimension of the output space.
        :rtype: np.ndarray
        """
        # This is according to the proofs.
        beta_sqr = 2 * np.log(
            self.m
            * self.design_space.cardinality
            * (np.pi**2)
            * ((self.round + 1) ** 2)
            / (6 * self.delta)
        )
        return np.sqrt(beta_sqr / self.conf_contraction)

    def compute_pessimistic_set(self) -> set:
        """
        The pessimistic Pareto set of the set S+P of designs.

        :return: Set of pessimistic Pareto indices.
        :rtype: set
        """
        W = self.S.union(self.P)

        pessimistic_set = set()
        for pt in W:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in W:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                # Check if there is another point j that dominates i, if so,
                # do not include i in the pessimistic set
                if confidence_region_check_dominates(self.order, pt_p_conf, pt_conf):
                    break
            else:
                pessimistic_set.add(pt)

        return pessimistic_set

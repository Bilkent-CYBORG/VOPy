import logging
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

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
from vopy.models import CorrelatedExactGPyTorchModel, get_gpytorch_model_w_known_hyperparams

from vopy.order import PolyhedralConeOrder


class VOGP(PALAlgorithm):
    """
    Implement the Vector Optimization with Gaussian Process (VOGP) algorithm.

    :param epsilon: Accuracy parameter for the PAC algorithm.
    :type epsilon: float
    :param delta: Confidence parameter for the PAC algorithm.
    :type delta: float
    :param dataset_name: Name of the dataset to be used.
    :type dataset_name: str
    :param order: An instance of the Order class for managing comparisons.
    :type order: Order
    :param noise_var: Variance of the Gaussian sampling noise.
    :type noise_var: float
    :param conf_contraction: Contraction coefficient to shrink the
        confidence regions empirically. Defaults to 32.
    :type conf_contraction: float
    :param batch_size: Number of samples to be taken in each round. Defaults to 1.
    :type batch_size: int

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.
    It uses Gaussian Process regression to model the rewards and confidence
    regions.

    Example Usage:
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.algorithms import VOGP
        >>>
        >>> epsilon, delta, noise_var = 0.1, 0.05, 0.01
        >>> dataset_name = "DiscBrake"
        >>> order_right = ComponentwiseOrder(2)
        >>>
        >>> algorithm = VOGP(epsilon, delta, dataset_name, order_right, noise_var)
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_set = algorithm.P

    Reference:
        "Vector Optimization with Gaussian Process Bandits",
        Korkmaz, Yıldırım, Ararat, Tekin, arXiv preprint
        https://arxiv.org/abs/2412.02484
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        dataset_name: str,
        order: PolyhedralConeOrder,
        noise_var: float,
        conf_contraction: float = 32,
        batch_size: int = 1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            CorrelatedExactGPyTorchModel,
            self.problem,
            initial_sample_cnt=1,
            noise_var=noise_var,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.u_star, self.d1 = self.compute_u_star()
        self.u_star_eps = self.u_star * epsilon

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
                # Function to check if ∃z' in R(x') such that R(x) <_C z + u, where u < epsilon
                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, self.u_star_eps):
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

                if confidence_region_is_covered(self.order, pt_conf, pt_p_conf, self.u_star_eps):
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
        """
        Executes one iteration of the VOGP algorithm, performing modeling, discarding,
        epsilon-covering, and evaluating phases. Returns the algorithm termination status.

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
            / (3 * self.delta)
        )
        return np.sqrt(beta_sqr / self.conf_contraction)

    def compute_u_star(self) -> Tuple[np.ndarray, float]:
        """
        Computes the normalized direction vector `u_star` and the ordering difficulty `d1` of
        a polyhedral ordering cone defined in the order self.order.ordering_cone.

        :return: A tuple containing `u_star`, the normalized direction vector of the cone, and
            `d1`, the Euclidean norm of the vector that gives `u_star` when normalized, _i.e._, `z`.
        :rtype: Tuple[np.ndarray, float]
        """
        cone_matrix = self.order.ordering_cone.W

        n = cone_matrix.shape[0]

        def objective(z):
            return np.linalg.norm(z)

        def constraint_func(z):
            constraints = []
            constraint = cone_matrix @ (z) - np.ones((n,))
            constraints.extend(constraint)
            return np.array(constraints)

        z_init = np.ones(self.m)  # Initial guess
        cons = [{"type": "ineq", "fun": lambda z: constraint_func(z)}]  # Constraints
        res = minimize(
            objective,
            z_init,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": 1000000, "ftol": 1e-30},
        )
        norm = np.linalg.norm(res.x)
        construe = np.all(constraint_func(res.x) + 1e-14)

        if not construe:
            raise ValueError("Could not compute the u_star vector for the given order.")

        return res.x / norm, norm

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

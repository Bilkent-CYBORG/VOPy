import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from vopy.acquisition import MaxDiagonalAcquisition, optimize_acqf_discrete
from vopy.algorithms.algorithm import PALAlgorithm
from vopy.confidence_region import (
    confidence_region_check_dominates,
    confidence_region_is_covered,
    confidence_region_is_dominated,
)
from vopy.utils.transforms import StandardizeOutput
from vopy.design_space import AdaptivelyDiscretizedDesignSpace
from vopy.maximization_problem import ContinuousProblem
from vopy.models import Model, CorrelatedExactGPyTorchModel
from vopy.order import PolyhedralConeOrder


class VOGP_ADOnline(PALAlgorithm):
    """
    Implement the Vector Optimization with Gaussian Process (VOGP) algorithm for continuous
    domains using adaptive discretization with online model training.

    :param epsilon: Accuracy parameter for the PAC algorithm.
    :type epsilon: float
    :param delta: Confidence parameter for the PAC algorithm.
    :type delta: float
    :param problem: Problem instance to optimize on. It should be a :obj:`ContinuousProblem`.
    :type problem: ContinuousProblem
    :param order: An instance of the Order class for managing comparisons.
    :type order: PolyhedralConeOrder
    :param conf_contraction: Contraction coefficient to shrink the
        confidence regions empirically. Defaults to 32.
    :type conf_contraction: float
    :param batch_size: Number of samples to be taken in each round. Defaults to 1.
    :type batch_size: int
    :param initial_sample_cnt: Number of initial samples to be taken before the algorithm starts.
    :type initial_sample_cnt: int
    :param reset_on_retrain: If True, resets the sets after each model retraining.
        Defaults to False.
    :type reset_on_retrain: bool
    :param model: Predefined model to be used. If None, a default model is created.
    :type model: Optional[Model]

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.
    It uses Gaussian Process regression to model the rewards and confidence
    regions of the continuous function defined with problem. It continuously
    retratins the model after every observation and uses adaptive discretization.

    Example Usage:
        >>> from vopy.algorithms import VOGP_ADOnline
        >>> from vopy.order import ComponentwiseOrder
        >>> from vopy.maximization_problem import get_continuous_problem
        >>>
        >>> epsilon, delta = 0.1, 0.05
        >>> problem = get_continuous_problem("BraninCurrin")
        >>> order_right = ComponentwiseOrder(2)
        >>>
        >>> algorithm = VOGP_ADOnline(epsilon, delta, problem, order_right)
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> predictive_model = algorithm.model

    Reference:
        Based on "Vector Optimization with Gaussian Process Bandits",
        Korkmaz, Yıldırım, Ararat, Tekin, arXiv preprint
        https://arxiv.org/abs/2412.02484
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        problem: ContinuousProblem,
        order: PolyhedralConeOrder,
        conf_contraction: float = 32,
        batch_size: int = 1,
        initial_sample_cnt: int = 1,
        reset_on_retrain: bool = False,
        model: Optional[Model] = None,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.conf_contraction = conf_contraction
        self.reset_on_retrain = reset_on_retrain

        self.batch_size = batch_size
        if self.batch_size != 1:
            raise AssertionError(
                "Currently, only batch size of 1 is supported"
                " with VOGP_AD that uses adaptive discretization."
            )

        if not hasattr(problem, "depth_max"):
            raise AssertionError("Problem does not have a max depth defined.")

        self.max_discretization_depth = problem.depth_max
        self.problem = problem

        self.d = self.problem.in_dim
        self.m = self.problem.out_dim

        self.design_space = AdaptivelyDiscretizedDesignSpace(
            problem.in_dim,
            problem.out_dim,
            delta=self.delta,
            max_depth=self.max_discretization_depth,
            confidence_type="hyperrectangle",
        )

        if model is None:
            output_transform = StandardizeOutput(self.m)
            self.model = CorrelatedExactGPyTorchModel(
                self.d,
                self.m,
                noise_rank=self.m,
                output_transform=output_transform,
            )
        else:
            self.model = model

        self.u_star, self.d1 = self.compute_u_star()
        self.u_star_eps = self.u_star * epsilon

        self.sample_count = 0
        self.reset_sets()
        self.round = 0
        self.enable_epsilon_covering = False

        self.initial_sampling(initial_sample_cnt=initial_sample_cnt)

    def reset_sets(self):
        self.S = set(range(1))
        self.P = set()

    def initial_sampling(self, initial_sample_cnt: int):
        """
        Initial sampling from the design space to start the algorithm.

        :param initial_sample_cnt: Number of initial samples to be taken.
        :type initial_sample_cnt: int
        """
        # Sample randomly from the continuous domain
        bounds = self.problem.bounds
        initial_points = np.random.uniform(
            low=bounds[0], high=bounds[1], size=(initial_sample_cnt, self.d)
        )
        initial_values = self.problem.evaluate(initial_points)

        self.model.add_sample(initial_points, initial_values)
        self.model.update()
        self.model.train()
        self.sample_count += initial_sample_cnt

    def modeling(self):
        """
        Updates confidence regions for all currently active design nodes.
        """
        # Active nodes, union of sets s_t and p_t at the beginning of round t
        W = self.S.union(self.P)
        self.beta = self.compute_beta()
        self.design_space.update(self.model, self.beta, list(W))

    def discarding(self):
        """
        Discards design nodes that are highly likely to be dominated based on
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
        Identify and remove design nodes from `S` that are not covered by the confidence region of
        other design nodes, adding them to `P` as Pareto-optimal. This stage is only enabled after
        all design nodes in `S` have reached the maximum discretization depth.
        """
        if not self.enable_epsilon_covering:
            for design_i in self.S:
                if self.design_space.point_depths[design_i] != self.max_discretization_depth:
                    return
            else:
                self.enable_epsilon_covering = True

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

    def evaluate_refine(self):
        """
        Selects a design node for based on maximum diagonals and either updates the model with new
        observations or refines the design node. If `self.reset_on_retrain` is True,
        reset the running sets after retraining the model.
        """
        W = self.S.union(self.P)
        acq = MaxDiagonalAcquisition(self.design_space)
        active_pts = self.design_space.points[list(W)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        candidate_pt = candidate_list[0]
        candidate_i = np.where(np.all(self.design_space.points == candidate_pt, axis=1))[0].item()
        should_refine = self.design_space.should_refine_design(self.model, candidate_i, self.beta)
        if should_refine:
            child_designs = self.design_space.refine_design(candidate_i)
            if candidate_i in self.S:
                self.S.remove(candidate_i)
                self.S = self.S.union(child_designs)
            else:  # candidate_pt in self.P
                self.P.remove(candidate_i)
                self.P = self.P.union(child_designs)
        else:
            observations = self.problem.evaluate(candidate_list)

            self.sample_count += len(candidate_list)
            self.model.add_sample(candidate_list, observations)
            self.model.update()
            self.model.train()
            if self.reset_on_retrain:
                self.reset_sets()

    def run_one_step(self) -> bool:
        """
        Run one step of the algorithm and return algorithm status.

        :return: True if the algorithm is over, *i.e.*, `S` is empty, False otherwise.
        :rtype: bool
        """
        if len(self.S) == 0:
            return True

        round_str = f"Round {self.round}"

        logging.info(f"{round_str}:Modeling")
        self.modeling()

        logging.info(f"{round_str}:Discarding")
        self.discarding()

        logging.info(f"{round_str}:Epsilon-Covering")
        self.epsiloncovering()

        logging.info(f"{round_str}:Evaluating")
        if self.S:  # If S_t is not empty
            self.evaluate_refine()

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )
        logging.info(self.design_space.point_depths)

        self.round += 1

        logging.info(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self):
        """
        Compute the confidence scaling parameter `beta` for Gaussian Process modeling.

        :return: A vector representing `beta` for each dimension of the output space.
        :rtype: np.ndarray
        """
        # Get the learned likelihood noise variance from the model.
        if self.model.likelihood.rank == 0:
            noise_var = np.max(self.model.likelihood.task_noises.numpy(force=True))
        else:
            noise_var = np.max(self.model.likelihood.task_noise_covar.diag().numpy(force=True))

        Kn = self.model.evaluate_kernel()
        rkhs_bound = 0.1
        beta_sqr = rkhs_bound + np.sqrt(
            noise_var * np.log((1 / noise_var) * np.linalg.det(Kn + np.eye(len(Kn))))
            - 2 * np.log(self.delta)
        )

        beta_sqr = beta_sqr**2  # Instead of dividing with sqrt of conf_contraction.
        return np.sqrt(beta_sqr / self.conf_contraction)

    def compute_u_star(self):
        """
        Computes the normalized direction vector `u_star` and the ordering difficulty `d1` of
        a polyhedral ordering cone defined in the order self.order.ordering_cone.

        :return: A tuple containing `u_star`, the normalized direction vector of the cone, and
            `d1`, the Euclidean norm of the vector that gives `u_star` when normalized, *i.e.*, `z`.
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

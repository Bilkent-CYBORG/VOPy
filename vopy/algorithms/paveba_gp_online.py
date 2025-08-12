import logging
from typing import Optional

import torch
import gpytorch
import numpy as np

from vopy.acquisition import optimize_acqf_discrete, SumVarianceAcquisition
from vopy.algorithms.algorithm import PALAlgorithm
from vopy.confidence_region import confidence_region_is_covered, confidence_region_is_dominated
from vopy.design_space import FixedPointsDesignSpace
from vopy.maximization_problem import FixedPointsProblem
from vopy.models import IndependentExactGPyTorchModel

from vopy.models import Model
from vopy.order import PolyhedralConeOrder
from vopy.utils.transforms import NormalizeInput, StandardizeOutput


class PaVeBaGPOnline(PALAlgorithm):
    """
    Implement the GP-based Pareto Vector Bandits (PaVeBa) algorithm for online optimization.

    :param epsilon: Determines the accuracy of the PAC-learning framework.
    :type epsilon: float
    :param delta: Determines the success probability of the PAC-learning framework.
    :type delta: float
    :param problem: Problem instance to optimize on. It should be a :obj:`FixedPointsProblem`.
    :type problem: FixedPointsProblem
    :param order: Order to be used.
    :type order: Order
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

    The algorithm sequentially samples design rewards. It uses Gaussian Process regression to model
    the rewards and confidence regions. It retratins the model after every observation.

    Reference:
        "Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector Bandits",
        Karagözlü, Yıldırım, Ararat, Tekin, AISTATS, '24
        https://proceedings.mlr.press/v238/karagozlu24a.html
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        problem: FixedPointsProblem,
        order: PolyhedralConeOrder,
        conf_contraction: float = 32,
        batch_size: int = 1,
        initial_sample_cnt: int = 10,
        reset_on_retrain: bool = False,
        model: Optional[Model] = None,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction
        self.reset_on_retrain = reset_on_retrain

        self.problem = problem

        self.d = self.problem.in_dim
        self.m = self.problem.out_dim

        self.design_space = FixedPointsDesignSpace(
            self.problem.in_data, self.m, confidence_type="hyperrectangle"
        )

        if model is None:
            mean_module = self.mean_module = gpytorch.means.ZeroMean(
                batch_shape=torch.Size([self.m])
            )

            rbf_kernel = gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([self.m]),
                ard_num_dims=self.d,
            )
            covar_module = gpytorch.kernels.ScaleKernel(
                rbf_kernel, batch_shape=torch.Size([self.m])
            )

            input_transform = NormalizeInput(self.d, bounds=self.problem.bounds)
            output_transform = StandardizeOutput(self.m)
            self.model = IndependentExactGPyTorchModel(
                self.d,
                self.m,
                noise_rank=self.m,
                input_transform=input_transform,
                output_transform=output_transform,
                mean_module=mean_module,
                covar_module=covar_module,
            )
        else:
            self.model = model

        self.sample_count = 0
        self.reset_sets()
        self.round = 0

        self.initial_sampling(initial_sample_cnt=initial_sample_cnt)

        self.cone_alpha = self.order.ordering_cone.alpha.flatten()
        self.cone_alpha_eps = self.cone_alpha * self.epsilon

    def reset_sets(self):
        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.U = set()

    def initial_sampling(self, initial_sample_cnt: int):
        """
        Initial sampling from the design space to start the algorithm.

        :param initial_sample_cnt: Number of initial samples to be taken.
        :type initial_sample_cnt: int
        """
        initial_indices = np.random.choice(len(self.problem.in_data), initial_sample_cnt)
        initial_points = self.problem.in_data[initial_indices]
        initial_values = self.problem.evaluate(initial_points)

        self.model.add_sample(initial_points, initial_values)
        self.model.update()
        self.model.train()
        self.sample_count += initial_sample_cnt

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
        Identify the designs that are highly likely to be `epsilon`-optimal using
        the confidence regions.
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
        largest sum of variances and update the model. If `self.reset_on_retrain` is True,
        reset the running sets after retraining the model.
        """
        A = self.S.union(self.U)
        acq = SumVarianceAcquisition(self.model)
        active_pts = self.design_space.points[list(A)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

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

        :return: True if the algorithm is over, False otherwise.
        :rtype: bool
        """
        if len(self.S) == 0:
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

        return len(self.S) == 0

    def compute_alpha(self) -> float:
        """
        Compute the radius of the confidence regions of the current round to be used in modeling.

        :return: The radius of the confidence regions.
        :rtype: float
        """
        alpha = 8 * self.m * np.log(6) + 4 * np.log(
            (np.pi**2 * self.round**2 * self.design_space.cardinality) / (6 * self.delta)
        )

        return alpha / self.conf_contraction

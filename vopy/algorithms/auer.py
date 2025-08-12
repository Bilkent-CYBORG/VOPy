import logging

import numpy as np

from vopy.algorithms.algorithm import PALAlgorithm
from vopy.datasets import get_dataset_instance
from vopy.design_space import FixedPointsDesignSpace
from vopy.maximization_problem import ProblemFromDataset
from vopy.models import EmpiricalMeanVarModel

from vopy.order import ComponentwiseOrder


class Auer(PALAlgorithm):
    """
    Implement the Algorithm 1 from Auer et al. (2016).

    :param epsilon: Determines the accuracy of the PAC-learning framework.
    :type epsilon: float
    :param delta: Determines the success probability of the PAC-learning framework.
    :type delta: float
    :param dataset_name: Name of the dataset to be used.
    :type dataset_name: str
    :param noise_var: Variance of the Gaussian sampling noise.
    :type noise_var: float
    :param conf_contraction: Contraction coefficient to shrink
        the confidence regions empirically.
    :type conf_contraction: float

    The algorithm sequentially samples design rewards with a multivariate
    white Gaussian noise whose diagonal entries are specified by the user.

    Example:
        >>> from vopy.algorithms import Auer
        >>>
        >>> epsilon, delta, noise_var = 0.1, 0.05, 0.01
        >>> dataset_name = "DiscBrake"
        >>>
        >>> algorithm = Auer(epsilon, delta, dataset_name, order_right, noise_var)
        >>>
        >>> while True:
        >>>     is_done = algorithm.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = algorithm.P

    Reference:
        "Pareto Front Identification from Stochastic Bandit Feedback",
        Auer, Chiang, Ortner, Drugan, AISTATS, '16
        https://proceedings.mlr.press/v51/auer16.html
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        dataset_name: str,
        noise_var: float,
        conf_contraction: int = 32,
        use_empirical_beta: bool = False,
    ) -> None:
        super().__init__(epsilon, delta)

        self.noise_var = noise_var
        self.conf_contraction = conf_contraction
        self._use_empirical_beta = use_empirical_beta

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim
        self.order = ComponentwiseOrder(self.m)

        # Trick to keep indices alongside points. This is for predictions from the model.
        in_data = np.hstack((dataset.in_data, np.arange(len(dataset.in_data))[:, None]))
        self.design_space = FixedPointsDesignSpace(
            in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = EmpiricalMeanVarModel(
            dataset.in_dim,
            self.m,
            noise_var,
            self.design_space.cardinality,
            track_variances=self._use_empirical_beta,
        )

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.round = 0
        self.sample_count = 0

    @property
    def use_empirical_beta(self) -> bool:
        """
        Property for the use_empirical_beta attribute.

        :return: The value of the use_empirical_beta attribute.
        :rtype: bool
        """
        return self._use_empirical_beta

    @use_empirical_beta.setter
    def use_empirical_beta(self, value: bool):
        """
        Setter for the use_empirical_beta attribute, which updates model's variance tracking.

        :param value: The new value for the use_empirical_beta attribute.
        :type value: bool
        """
        self._use_empirical_beta = value
        self.model.track_variances = self._use_empirical_beta

    def small_m(self, i: np.ndarray, j: np.ndarray) -> float:
        """
        This method calculates the m(i, j) value, which is the amount by which the vector i
        have to be increased such that it would not be strongly dominated by vector j.

        :param vi: A D-vector.
        :type vi: np.ndarray
        :param vj: A D-vector.
        :type vj: np.ndarray
        :return: The computed m(i, j) value.
        :rtype: float
        """
        return max(0, np.min(j - i))

    def big_m(self, i: np.ndarray, j: np.ndarray) -> float:
        """
        This method calculates the M(i, j) value, which is the amount by which the values of
        vector j have to be increased such that vector i + epsilon would be weakly dominated by it.

        :param i: A D-vector.
        :type i: np.ndarray
        :param j: A D-vector.
        :type j: np.ndarray
        :return: The computed M(i, j) value.
        :rtype: float
        """
        return max(0, np.max((i + self.epsilon) - j))

    def modeling(self):
        """
        Construct the confidence regions of all active designs given all past observations.
        """
        # All active designs have the same beta value if empirical beta is not used. In that case;
        # model does not track variances. If the empirical beta is used, variances are tracked and
        # used here to provide individual scales for each design but disabled when updating the
        # design space.

        self.beta_t = self.compute_beta()

        self.model.track_variances = self.use_empirical_beta and False  # always False
        self.design_space.update(self.model, self.beta_t, list(self.S))
        self.model.track_variances = self.use_empirical_beta and True

    def discarding(self):
        """
        Discard the designs that are highly likely to be suboptimal using the confidence regions.
        """
        to_be_discarded = []
        for pt_i, pt in enumerate(self.S):
            pt_conf = self.design_space.confidence_regions[pt]
            pt_beta = self.beta_t[pt_i]
            for pt_prime_i, pt_prime in enumerate(self.S):
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                pt_p_beta = self.beta_t[pt_prime_i]

                beta = pt_beta + pt_p_beta
                if np.all(self.small_m(pt_conf.center, pt_p_conf.center) > beta):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def pareto_updating(self):
        """
        Identify the designs that are highly likely to be `epsilon`-optimal
        using the confidence regions, by first identifying the designs that are useful.
        """
        P1_pt_is = []
        P1_pts = []
        for pt_i, pt in enumerate(self.S):
            pt_conf = self.design_space.confidence_regions[pt]
            pt_beta = self.beta_t[pt_i]
            for pt_prime_i, pt_prime in enumerate(self.S):
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                pt_p_beta = self.beta_t[pt_prime_i]

                beta = pt_beta + pt_p_beta
                if np.all(self.big_m(pt_conf.center, pt_p_conf.center) < beta):
                    break
            else:
                P1_pt_is.append(pt_i)
                P1_pts.append(pt)

        new_pareto_pts = []
        for p1_pt_i, p1_pt in zip(P1_pt_is, P1_pts):
            p1_pt_conf = self.design_space.confidence_regions[p1_pt]
            p1_pt_beta = self.beta_t[p1_pt_i]
            for pt_i, pt in enumerate(self.S):
                if pt in P1_pts:
                    continue

                pt_conf = self.design_space.confidence_regions[pt]
                pt_beta = self.beta_t[pt_i]

                beta = p1_pt_beta + pt_beta
                if np.all(self.big_m(pt_conf.center, p1_pt_conf.center) <= beta):
                    # Leave as useful
                    break
            else:
                # Add to P
                new_pareto_pts.append(p1_pt)

        for pt in new_pareto_pts:
            self.S.remove(pt)
            self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def evaluating(self):
        """
        Observe the active designs via sampling.
        """
        active_pts = self.design_space.points[list(self.S)]

        observations = self.problem.evaluate(active_pts[:, :-1])

        self.sample_count += len(self.S)
        self.model.add_sample(self.S, observations)
        self.model.update()

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

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        logging.debug(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self) -> np.ndarray:
        """
        Compute the beta values for the confidence regions of the current round to be
        used in modeling.

        :return: The beta velues of the confidence regions.
        :rtype: np.ndarray
        """
        # Original beta
        if not self.use_empirical_beta:
            t1 = np.log((4 * self.design_space.cardinality * self.m * self.round**2) / self.delta)
            t2 = np.ones((len(self.S), self.m))
        else:  # Empirical beta
            # Indices are enough for prediction.
            active_pts = self.design_space.points[list(self.S)]

            v_hat = self.model.predict(active_pts)[1].diagonal(axis1=-2, axis2=-1)
            v_bar = 1
            t1 = np.log((self.design_space.cardinality * self.m * self.round) / self.delta)
            t2 = v_hat + v_bar * np.sqrt((4 * t1) / self.round)

        beta = np.sqrt((2 * t1 * t2) / self.round)

        return beta / self.conf_contraction

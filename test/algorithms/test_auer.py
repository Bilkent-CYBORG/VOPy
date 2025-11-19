from unittest import TestCase

import numpy as np
from vopy.algorithms import Auer
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestAuer(TestCase):
    """Test the Auer class."""

    def setUp(self):
        """A basic setup for the model."""
        set_seed(SEED)

        self.epsilon = 0.1
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.noise_var = 0.00001
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.conf_contraction = 32
        self.algorithm = Auer(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_small_m(self):
        """Test the m(i, j) method."""
        i = np.array([0, 1])
        j = np.array([1, 0])
        m = self.algorithm.small_m(i, j)

        self.assertTrue(m == 0)

    def test_big_m(self):
        """Test the M(i, j) method."""
        i = np.array([0, 1])
        j = np.array([1, 0])
        m = self.algorithm.big_m(i, j)

        self.assertTrue(m == 1 + self.epsilon)

    def test_evaluating(self):
        """Test the evaluating method."""
        sample_test = self.algorithm.sample_count
        self.algorithm.evaluating()
        self.assertTrue(self.algorithm.sample_count > sample_test)

    def test_whole_class(self):
        """Test the whole class by running it until the end end checking its score."""
        while True:
            is_done = self.algorithm.run_one_step()
            if is_done:
                break

        self.assertTrue(self.algorithm.run_one_step())

        pareto_indices = self.algorithm.P
        dataset = get_dataset_instance(self.dataset_name)
        eps_f1 = calculate_epsilonF1_score(
            dataset,
            self.order,
            self.order.get_pareto_set(dataset.out_data),
            list(pareto_indices),
            self.epsilon,
        )
        self.assertGreaterEqual(eps_f1, 0.9)

    def test_run_one_step(self):
        """Test the run_one_step method."""
        num_rounds = 10
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 3:  # Save the state at round 3 at the latest.
                S_test = self.algorithm.S
                P_test = self.algorithm.P
            alg_done = self.algorithm.run_one_step()

        S = self.algorithm.S
        P = self.algorithm.P

        self.assertTrue(num_rounds >= self.algorithm.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))

    def test_compute_beta(self):
        """Test the compute_beta method."""
        self.algorithm.run_one_step()
        t1 = np.log((4 * self.dataset_cardinality * 2 * self.algorithm.round**2) / self.delta)
        r1 = np.sqrt(2 * t1 / self.algorithm.round) / self.conf_contraction
        r2 = self.algorithm.compute_beta()
        self.assertTrue((r1 == r2).all())

        self.algorithm.run_one_step()
        r3 = self.algorithm.compute_beta()
        self.assertTrue((r3[0] <= r2[0]).all())

        self.algorithm.use_empirical_beta = True
        self.algorithm.run_one_step()
        r4 = self.algorithm.compute_beta()
        self.algorithm.run_one_step()
        r5 = self.algorithm.compute_beta()

        self.assertTrue((r5 <= r4).all())

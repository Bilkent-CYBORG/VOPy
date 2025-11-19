import unittest

import numpy as np
from vopy.algorithms import EpsilonPAL
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestEpsilonPAL(unittest.TestCase):
    """Test the EpsilonPAL algorithm class."""

    def setUp(self):
        # Set random seed for reproducibility
        set_seed(SEED)

        self.epsilon = 0.2
        self.delta = 0.1
        self.noise_var = 0.00001
        self.dataset_name = "Test"
        self.conf_contraction = 128

        self.iter_count = 5
        self.output_dim = 2
        self.order = ComponentwiseOrder(self.output_dim)
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality

        self.algorithm = EpsilonPAL(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_evaluating(self):
        """Test the evaluating method."""
        initial_sample_count = self.algorithm.sample_count
        self.algorithm.evaluating()
        self.assertGreater(self.algorithm.sample_count, initial_sample_count)

    def test_run_one_step(self):
        """Test the run_one_step method."""
        is_done = False
        for i in range(self.iter_count):  # Run for 10 rounds, it should be enough.
            if not is_done and i <= 3:  # Save the state at round 3 at the latest.
                S_test = self.algorithm.S
                P_test = self.algorithm.P
            is_done = self.algorithm.run_one_step()

        S = self.algorithm.S
        P = self.algorithm.P

        self.assertTrue(self.iter_count >= self.algorithm.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))

    def test_whole_class(self):
        """
        This test performs a full run of the EpsilonPAL algorithm and calculates the epsilon-F1
        score.
        """
        while True:
            is_done = self.algorithm.run_one_step()
            if is_done:
                break

        dataset = get_dataset_instance(self.dataset_name)
        pareto_indices = self.algorithm.P
        eps_f1 = calculate_epsilonF1_score(
            dataset,
            self.order,
            self.order.get_pareto_set(dataset.out_data),
            list(pareto_indices),
            self.epsilon,
        )

        self.assertGreaterEqual(eps_f1, 0.9)

    def test_compute_beta(self):
        """Test the compute_beta method."""
        self.algorithm.run_one_step()
        beta = np.sqrt(
            2
            * np.log(
                self.output_dim
                * self.dataset_cardinality
                * (np.pi**2)
                * ((self.algorithm.round + 1) ** 2)
                / (6 * self.delta)
            )
        )
        r1 = beta
        r2 = self.algorithm.compute_beta()
        self.assertAlmostEqual((r1 / np.sqrt(self.conf_contraction)), r2)

        self.algorithm.run_one_step()
        r3 = self.algorithm.compute_beta()
        self.assertTrue((r3 > r2).all())

    def test_compute_pessimistic_set(self):
        """Test the compute_pessimistic_set method."""
        for _ in range(self.iter_count):
            self.algorithm.run_one_step()
        pess = self.algorithm.compute_pessimistic_set()
        self.assertGreater(len(pess), 0)
        self.assertLessEqual(len(pess), len(self.algorithm.S) + len(self.algorithm.P))

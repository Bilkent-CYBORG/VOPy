from unittest import TestCase

import numpy as np
from vopy.algorithms import PaVeBa
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestPaVeBa(TestCase):
    """Test the PaVeBa class."""

    def setUp(self):
        """A basic setup for the model."""
        set_seed(SEED)

        self.epsilon = 0.2
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.noise_var = 0.00001
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.conf_contraction = 1024
        self.algo = PaVeBa(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_evaluating(self):
        """Test the evaluating method."""
        sample_test = self.algo.sample_count
        self.algo.evaluating()
        self.assertTrue(self.algo.sample_count > sample_test)

    def test_whole_class(self):
        """Test the whole class by running it until the end end checking its score."""
        while True:
            is_done = self.algo.run_one_step()
            if is_done:
                break

        self.assertTrue(self.algo.run_one_step())

        pareto_indices = self.algo.P
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
        num_rounds = 5
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 3:  # Save the state at round 3 at the latest.
                S_test = self.algo.S
                P_test = self.algo.P
            alg_done = self.algo.run_one_step()

        S = self.algo.S
        P = self.algo.P

        self.assertTrue(num_rounds >= self.algo.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))

    def test_compute_radius(self):
        """Test the compute_radius method."""
        self.algo.run_one_step()
        t1 = 8 * self.noise_var
        t2 = np.log((np.pi**2 * (3) * self.dataset_cardinality) / (6 * 0.1))
        r1 = np.sqrt(t1 * t2) / self.conf_contraction
        r2 = self.algo.compute_radius()
        self.assertTrue(r1 == r2)

        self.algo.run_one_step()
        r3 = self.algo.compute_radius()
        self.assertTrue((r3 <= r2).all())

from unittest import TestCase

import numpy as np
from vopy.algorithms import PaVeBaGP
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestPaVeBaGP(TestCase):
    """Test the PaVeBaGP class."""

    def setUp(self):
        # A basic setup for the model.
        set_seed(SEED)

        self.epsilon = 0.2
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.noise_var = 0.00001
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.conf_contraction = 256
        self.algo = PaVeBaGP(
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

    def test_run_one_step_with_hyperrectangle(self):
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

    def test_run_one_step_with_ellipsoid(self):
        """Test the run_one_step method."""

        self.algo = PaVeBaGP(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
            type="DE",
        )

        num_rounds = 3
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 2:  # Save the state at round 3 at the latest.
                S_test = self.algo.S
                P_test = self.algo.P
            alg_done = self.algo.run_one_step()

        S = self.algo.S
        P = self.algo.P

        self.assertTrue(num_rounds >= self.algo.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))

    def test_compute_alpha(self):
        """Test the compute_alpha method."""
        self.algo.run_one_step()
        alpha = 8 * 2 * np.log(6) + 4 * np.log(
            (np.pi**2 * self.dataset_cardinality) / (6 * self.delta)
        )
        r1 = alpha
        r2 = self.algo.compute_alpha()
        self.assertTrue((r1 / self.conf_contraction) == r2)

        self.algo.run_one_step()
        r3 = self.algo.compute_alpha()
        self.assertTrue((r3 > r2).all())

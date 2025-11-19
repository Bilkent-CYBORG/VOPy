from unittest import TestCase

import numpy as np
from vopy.algorithms import PaVeBaPartialGP
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestPaVeBaPartialGP(TestCase):
    """Test the PaVeBaPartialGP class."""

    def setUp(self):
        """A basic setup for the model."""
        set_seed(SEED)

        self.epsilon = 0.2
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.noise_var = 0.00001
        self.conf_contraction = 32
        self.costs = [1.0, 1.5]
        self.cost_budget = 24
        self.algo = PaVeBaPartialGP(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
            costs=self.costs,
            cost_budget=self.cost_budget,
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
        self.assertLess(self.algo.total_cost, self.cost_budget + max(self.costs))
        self.assertLessEqual(self.algo.total_cost, self.algo.round * max(self.costs))
        self.assertGreaterEqual(self.algo.total_cost, self.algo.round * min(self.costs))

    def test_run_one_step(self):
        """Test the run_one_step method."""
        num_rounds = 5
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 3:  # Save the state at round 3 at the latest.
                S_test = self.algo.S
                P_test = self.algo.P
                cost_test = self.algo.total_cost
            alg_done = self.algo.run_one_step()

        S = self.algo.S
        P = self.algo.P
        cost = self.algo.total_cost

        self.assertTrue(num_rounds >= self.algo.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))
        self.assertTrue(cost_test < cost)

    def test_compute_alpha(self):
        """Test the compute_alpha method."""
        self.algo.run_one_step()
        alpha = 2 * np.log((np.pi**2 * self.dataset_cardinality) / (3 * self.delta))
        r1 = alpha
        r2 = self.algo.compute_alpha()
        self.assertTrue((r1 / self.conf_contraction) == r2)

        self.algo.run_one_step()
        r3 = self.algo.compute_alpha()
        self.assertTrue((r3 > r2).all())

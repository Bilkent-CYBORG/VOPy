from unittest import TestCase

import numpy as np
from vopy.algorithms import PaVeBaGPOnline
from vopy.datasets import get_dataset_instance
from vopy.order import ComponentwiseOrder

from vopy.utils import set_seed
from vopy.utils.seed import SEED
from vopy.datasets import Dataset
from vopy.maximization_problem import FixedPointsProblem, ProblemFromDataset
from vopy.utils.evaluate import calculate_epsilonF1_score


class TestPaVeBaGPOnline(TestCase):
    """Test the PaVeBaGPOnline class."""

    def setUp(self):
        # A basic setup for the model.
        set_seed(SEED)

        # Test dataset
        self.dataset_name = "Test"
        self.dataset: Dataset = get_dataset_instance(self.dataset_name)
        self.noise_var = 0.00001
        self.dset_problem = ProblemFromDataset(self.dataset, self.noise_var)
        self.problem = FixedPointsProblem(
            in_points=self.dset_problem.dataset.in_data,
            out_dim=self.dset_problem.out_dim,
            objective=self.dset_problem.evaluate,
        )

        self.epsilon = 0.2
        self.delta = 0.1
        self.order = ComponentwiseOrder(2)
        self.conf_contraction = 256
        self.algo = PaVeBaGPOnline(
            epsilon=self.epsilon,
            delta=self.delta,
            problem=self.problem,
            order=self.order,
            reset_on_retrain=True,
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

    def test_initial_samples(self):
        """Test the initial sample count."""
        self.assertTrue(len(self.algo.model.train_inputs) == self.algo.sample_count)

    def test_reset_sets(self):
        """Test the reset_sets method."""
        while len(self.algo.P) == 0:
            self.algo.run_one_step()
        initial_S = self.algo.S.copy()
        initial_P = self.algo.P.copy()

        self.algo.evaluating()

        self.assertTrue(len(self.algo.S) == len(self.problem.in_data))
        self.assertTrue(len(self.algo.P) == 0)
        self.assertTrue(initial_S != self.algo.S)
        self.assertTrue(initial_P != self.algo.P)

    def test_compute_alpha(self):
        """Test the compute_alpha method."""
        self.algo.run_one_step()
        r2 = self.algo.compute_alpha()

        self.algo.run_one_step()
        r3 = self.algo.compute_alpha()

        self.assertTrue((r3 > r2).all())

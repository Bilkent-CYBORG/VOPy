import unittest
import numpy as np

from vopy.algorithms.vogp_ad_online import VOGP_ADOnline
from vopy.maximization_problem import get_continuous_problem
from vopy.order import ComponentwiseOrder


class TestVOGPADOnline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
        self.epsilon = 0.1
        self.delta = 0.05
        self.noise_var = 0.01
        self.problem = get_continuous_problem("BraninCurrin", self.noise_var)
        self.order = ComponentwiseOrder(2)

    def test_initialization(self):
        """Test that VOGP_ADOnline initializes correctly."""
        algorithm = VOGP_ADOnline(
            self.epsilon, self.delta, self.problem, self.order, self.noise_var
        )

        self.assertEqual(algorithm.epsilon, self.epsilon)
        self.assertEqual(algorithm.delta, self.delta)
        self.assertEqual(algorithm.noise_var, self.noise_var)
        self.assertEqual(algorithm.m, self.problem.out_dim)
        self.assertEqual(algorithm.d, self.problem.in_dim)
        self.assertIsInstance(algorithm.S, set)
        self.assertIsInstance(algorithm.P, set)
        self.assertEqual(len(algorithm.S), 1)  # Should start with one design
        self.assertEqual(len(algorithm.P), 0)  # Should start with no Pareto points

    def test_run_one_step(self):
        """Test that run_one_step executes without error."""
        algorithm = VOGP_ADOnline(
            self.epsilon, self.delta, self.problem, self.order, self.noise_var
        )

        initial_sample_count = algorithm.sample_count
        result = algorithm.run_one_step()

        # Should not be done after one step
        self.assertFalse(result)
        # Round should increment
        self.assertEqual(algorithm.round, 1)
        # Sample count should be at least the initial count
        self.assertGreaterEqual(algorithm.sample_count, initial_sample_count)

    def test_batch_size_constraint(self):
        """Test that batch size other than 1 raises an assertion error."""
        with self.assertRaises(AssertionError):
            VOGP_ADOnline(
                self.epsilon, self.delta, self.problem, self.order, self.noise_var, batch_size=2
            )

    def test_reset_on_retrain(self):
        """Test that reset_on_retrain parameter is properly handled."""
        algorithm = VOGP_ADOnline(
            self.epsilon,
            self.delta,
            self.problem,
            self.order,
            self.noise_var,
            reset_on_retrain=True,
        )

        self.assertTrue(algorithm.reset_on_retrain)

    def test_custom_parameters(self):
        """Test that custom parameters are properly set."""
        conf_contraction = 16.0
        initial_sample_cnt = 5

        algorithm = VOGP_ADOnline(
            self.epsilon,
            self.delta,
            self.problem,
            self.order,
            self.noise_var,
            conf_contraction=conf_contraction,
            initial_sample_cnt=initial_sample_cnt,
        )

        self.assertEqual(algorithm.conf_contraction, conf_contraction)
        # Should have taken initial samples
        self.assertGreater(algorithm.sample_count, 0)


if __name__ == "__main__":
    unittest.main()

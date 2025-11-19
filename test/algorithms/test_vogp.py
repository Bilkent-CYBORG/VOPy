import unittest

import numpy as np
from vopy.algorithms import VOGP
from vopy.datasets import get_dataset_instance
from vopy.design_space import FixedPointsDesignSpace
from vopy.order import ComponentwiseOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score
from vopy.utils.seed import SEED


class TestVOGP(unittest.TestCase):
    """Test the VOGP algorithm class."""

    def setUp(self):
        """
        This method is run before each test to initialize parameters
        for the VOGP instance and prepare for testing both individual
        phases and full runs.
        """
        # Set random seed for reproducibility
        set_seed(SEED)

        # Parameters for VOGP instance
        self.epsilon = 0.2
        self.delta = 0.1
        self.noise_var = 0.00001
        self.conf_contraction = 64
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)

        self.iter_count = 5

        # Create the VOGP instance
        self.algorithm = VOGP(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_ustar(self):
        """
        Test the calculation of u_star. Ensures that the computed u* is inside the cone and has
        a norm of 1.
        """
        self.assertAlmostEqual(np.linalg.norm(self.algorithm.u_star), 1.0)
        self.assertTrue(self.order.ordering_cone.is_inside(self.algorithm.u_star))

    def test_discarding(self):
        """
        Tests the discarding phase of VOGP. Compares confidence regions where one is supposed
        to be discarded by the other.
        """
        self.manual_in_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.manual_out_data = np.array([[1.0, 2.0], [1.5, 2.5]])

        # Manually update the confidence regions with means and covariances
        mean_array = np.array([[1.0, 1.0], [1.0, 1.0]])
        mean_array[1] += self.algorithm.u_star * (2 * self.algorithm.d1 - self.epsilon)
        cov_array = np.array([np.diag([0.75, 0.25]), np.diag([0.25, 0.75])])

        self.algorithm.design_space = FixedPointsDesignSpace(
            self.manual_in_data, len(self.manual_out_data[0]), confidence_type="hyperrectangle"
        )

        # Manually setting the confidence regions:
        for pt_i in range(self.algorithm.design_space.cardinality):
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = False
            self.algorithm.design_space.confidence_regions[pt_i].update(
                mean=mean_array[pt_i], covariance=cov_array[pt_i], scale=np.ones(2)
            )

        self.algorithm.S = set(range(self.algorithm.design_space.cardinality))
        self.algorithm.P = set()

        self.algorithm.discarding()

        self.assertEqual(self.algorithm.S, {1}, "Wrong design was discarded.")

    def test_epsilon_covering(self):
        """
        Tests the Pareto identification phase of VOGP. Compares confidence regions where one
        design prevents the other design from being added to the Pareto set but not _vice versa_.
        """
        self.manual_in_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.manual_out_data = np.array([[1.0, 2.0], [1.5, 2.5]])

        # Manually update the confidence regions with means and covariances
        mean_array = np.array([[1.0, 1.0], [1.0, 1.0]])
        mean_array[1] += self.algorithm.u_star * (2 * self.algorithm.d1 - self.epsilon)
        cov_array = np.array([np.diag([0.75, 0.25]), np.diag([0.25, 0.75])])

        self.algorithm.design_space = FixedPointsDesignSpace(
            self.manual_in_data, len(self.manual_out_data[0]), confidence_type="hyperrectangle"
        )

        # Manually setting the confidence regions:
        for pt_i in range(self.algorithm.design_space.cardinality):
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = False
            self.algorithm.design_space.confidence_regions[pt_i].update(
                mean=mean_array[pt_i], covariance=cov_array[pt_i], scale=np.ones(2)
            )

        self.algorithm.S = set(range(self.algorithm.design_space.cardinality))
        self.algorithm.P = set()

        self.algorithm.epsiloncovering()

        self.assertEqual(self.algorithm.P, {1}, "Wrong design was added to Pareto set.")

    def test_evaluating(self):
        """
        Test the evaluating function of VOGP. Checks if the sample count is increased after
        the evaluation phase.
        """
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
        This test performs a full run of the VOGP algorithm and calculates the epsilon-F1 score.
        """
        # Run VOGP until completion
        while True:
            is_done = self.algorithm.run_one_step()
            if is_done:
                break

        # Get Pareto indices and calculate epsilon-F1 score
        dataset = get_dataset_instance(self.dataset_name)
        pareto_indices = self.algorithm.P
        eps_f1 = calculate_epsilonF1_score(
            dataset,
            self.order,
            self.order.get_pareto_set(dataset.out_data),
            list(pareto_indices),
            self.epsilon,
        )

        # Check the epsilon-F1 score
        self.assertGreaterEqual(eps_f1, 0.9, "eps-F1 score should be reasonably high.")

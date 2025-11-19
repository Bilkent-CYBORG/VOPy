from unittest import mock, TestCase

import numpy as np
from vopy.order import ComponentwiseOrder
from vopy.utils.evaluate import (
    calculate_epsilonF1_score,
    calculate_hypervolume_discrepancy_for_model,
)


class TestEpsilonF1Score(TestCase):
    """Test the epsilon-F1 evaluation metric."""

    def test_calculate_epsilonF1_score(self):
        """Test the calculate_epsilonF1_score function."""

        dataset = mock.Mock()
        dataset.out_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        order = ComponentwiseOrder(dim=2)

        true_indices = [1, 2, 3]
        pred_indices = [1, 3]
        epsilon = 0.1

        result = calculate_epsilonF1_score(dataset, order, true_indices, pred_indices, epsilon)

        self.assertEqual(result, 2 / 3)


class TestHypervolumeDiscrepancy(TestCase):
    """Test the hypervolume discrepancy evaluation metric."""

    @mock.patch("vopy.utils.evaluate.generate_sobol_samples")
    def test_calculate_hypervolume_discrepancy_for_model(self, mock_generate_sobol_samples):
        """Test the calculate_hypervolume_discrepancy_for_model function."""

        mock_generate_sobol_samples.return_value = np.array([[0, 0], [1, 1], [2, 2]])

        order = ComponentwiseOrder(dim=2)
        mock_problem = mock.Mock()
        mock_problem.in_dim = 2

        mock_model = mock.Mock()

        with self.assertRaises(AssertionError):
            mock_problem.evaluate.return_value = np.array([[1, 3], [2.5, 2.5], [3, 1]])
            mock_model.predict.return_value = (np.array([[1, 3], [2.5, 2.5], [3, 1]]), None)
            calculate_hypervolume_discrepancy_for_model(order, mock_problem, mock_model)

        mock_problem.evaluate.return_value = np.array([[1, 3], [2.5, 2.5], [3, 1]])
        mock_model.predict.return_value = (np.array([[3, 3], [2.5, 2.5], [3, 1]]) - 0.25, None)
        result = calculate_hypervolume_discrepancy_for_model(order, mock_problem, mock_model)

        self.assertEqual(mock_problem.evaluate.call_count, 2)
        self.assertEqual(mock_model.predict.call_count, 2)
        self.assertAlmostEqual(result, 0.8109302)

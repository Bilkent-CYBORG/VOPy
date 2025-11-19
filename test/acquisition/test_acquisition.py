from unittest import mock, TestCase

import numpy as np
from vopy.acquisition import (
    MaxDiagonalAcquisition,
    MaxVarianceDecoupledAcquisition,
    optimize_acqf_discrete,
    optimize_decoupled_acqf_discrete,
    SumVarianceAcquisition,
    ThompsonEntropyDecoupledAcquisition,
)
from vopy.design_space import FixedPointsDesignSpace
from vopy.models.gpytorch import GPyTorchModelListExactModel
from vopy.order import PolyhedralConeOrder
from vopy.ordering_cone import OrderingCone
from vopy.utils import set_seed
from vopy.utils.seed import SEED


class TestSumVarianceAcquisition(TestCase):
    """Test the SumVarianceAcquisition class."""

    def test_forward(self):
        """Test the forward method."""
        choices = np.array([[1, 2], [3, 4], [5, 6]])
        values = np.array([np.eye(2), 3 * np.eye(2), 4 * np.eye(2)])

        def side_effect_func(inp):
            vals = []
            for idx in range(inp.shape[0]):
                vals.append(values[np.where((choices == inp[idx]).all(axis=1))][0])
            return 12, np.array(vals)

        mock_model = mock.MagicMock(side_effect=side_effect_func)
        mock_model.predict.side_effect = side_effect_func
        acq = SumVarianceAcquisition(mock_model)
        ret_values = acq(choices)
        self.assertTrue(np.allclose(ret_values, np.array([2, 6, 8])))


class TestMaxVarianceDecoupledAcquisition(TestCase):
    """Test the MaxVarianceDecoupledAcquisition class."""

    def test_forward(self):
        """Test the forward method."""
        choices = np.array([[1, 2], [3, 4], [5, 6]])
        values = np.array([np.eye(2), np.array([[4, 0], [0, 0.9]]), np.array([[0.5, 0], [0, 7]])])

        def side_effect_func(inp):
            vals = []
            for idx in range(inp.shape[0]):
                vals.append(values[np.where((choices == inp[idx]).all(axis=1))][0])
            return 12, np.array(vals)

        mock_model = mock.MagicMock(side_effect=side_effect_func)
        mock_model.predict.side_effect = side_effect_func
        acq = MaxVarianceDecoupledAcquisition(mock_model)
        acq.evaluation_index = 1
        ret_values = acq(choices)
        self.assertTrue(np.allclose(ret_values, np.array([1, 0.9, 7])))


class TestThompsonEntropyDecoupledAcquisition(TestCase):
    """Test the ThompsonEntropyDecoupledAcquisition class."""

    def setUp(self) -> None:
        """Set up the seed for the test."""
        set_seed(SEED)

    def test_forward(self):
        """Test the forward method."""
        GP = GPyTorchModelListExactModel(2, 2, 0.3)
        choices = np.array([[1, 0], [-1, 0]])
        GP.add_sample(choices, np.array([1, -1]), [0, 0])
        GP.add_sample(choices, np.zeros(2), [1, 1])
        GP.update()
        ordering1 = OrderingCone(np.array([[1, 0], [1, 0]]))
        ordering2 = OrderingCone(np.array([[0, 1], [0, 1]]))
        order1 = PolyhedralConeOrder(ordering1)
        order2 = PolyhedralConeOrder(ordering2)
        acq1 = ThompsonEntropyDecoupledAcquisition(GP, order1, 1)
        acq2 = ThompsonEntropyDecoupledAcquisition(GP, order2, 1)
        acq3 = ThompsonEntropyDecoupledAcquisition(GP, order2, 0)
        ret_values1 = acq1(choices)
        ret_values2 = acq2(choices)
        ret_values3 = acq3(choices)
        self.assertTrue((ret_values1 < ret_values2).all())
        self.assertTrue((ret_values3 < ret_values2).all())


class TestMaxDiagonalAcquisition(TestCase):
    """Test the MaxDiagonalAcquisition class."""

    def test_forward(self):
        """Test the forward method."""
        choices = np.array([[1, 2], [3, 4], [5, 6]])
        design_space = FixedPointsDesignSpace(choices, 2)
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = (
            np.array([12, 12, 12]),
            np.array([np.eye(2), 3 * np.eye(2), 4 * np.eye(2)]),
        )
        design_space.update(mock_model, np.array([1]), [0, 1, 2])
        axq = MaxDiagonalAcquisition(design_space)
        ret_values = axq(choices)
        self.assertTrue(np.allclose(ret_values, np.sqrt(2) * np.array([2, 2 * np.sqrt(3), 4])))


class TestOptimizeAcqfDiscrete(TestCase):
    """Test the optimize_acqf_discrete function."""

    def test_optimize_acqf_discrete(self):
        q = 2
        choices = np.array([[1, 2], [3, 4], [5, 6]])
        values = np.array([1, 3, 2])

        def side_effect_func(inp):
            vals = []
            for idx in range(inp.shape[0]):
                vals.append(values[np.where((choices == inp[idx]).all(axis=1))][0])
            return np.array(vals)

        mock_acqf = mock.MagicMock(side_effect=side_effect_func)
        points, ret_values = optimize_acqf_discrete(mock_acqf, q, choices)
        self.assertTrue(
            np.allclose(points, np.array([[3, 4], [5, 6]]))
            and np.allclose(ret_values, np.array([3, 2]))
        )


class TestOptimizeDecoupledAcqfDiscrete(TestCase):
    """Test the optimize_decoupled_acqf_discrete function."""

    def test_optimize_decoupled_acqf_discrete(self):
        q = 2
        choices = np.array([[1, 2], [3, 4], [5, 6]])
        values = [np.array([1, 3.1, 2]), np.array([7, -1, 3])]

        def side_effect_func_0(inp):
            return side_effect_func(inp, mock_acqf.evaluation_index)

        def side_effect_func(inp, eval):
            vals = []
            for idx in range(inp.shape[0]):
                x = values[eval][np.where((choices == inp[idx]).all(axis=1))][0]
                vals.append(x)
            return np.array(vals)

        mock_acqf = mock.MagicMock(side_effect=side_effect_func_0)
        mock_acqf.out_dim = 2
        mock_acqf.evaluation_index = 0

        points, ret_values, idxs = optimize_decoupled_acqf_discrete(mock_acqf, q, choices)
        self.assertTrue(np.allclose(points, np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.allclose(ret_values, np.array([7, 3.1])))
        self.assertTrue(np.allclose(idxs, np.array([1, 0])))

import unittest

import torch
from scipy.linalg import issymmetric

from vopy.models.kernels import MixedKernel

from vopy.utils import set_seed
from vopy.utils.seed import SEED


class TestMixedKernel(unittest.TestCase):
    """Tests for the MixedKernel class."""

    def setUp(self):
        """Set up sample data and model for tests."""
        set_seed(SEED)
        self.kernel = MixedKernel(continuous_dims=(0, 1), categorical_dims=(2, 3))

    def test_forward(self):
        """Test forward pass of MixedKernel."""
        X = torch.tensor([[1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 1.0, 0.0], [5.0, 6.0, 0.0, 1.0]])

        output = self.kernel(X, X).to_dense()
        output.to(dtype=torch.float64)
        self.assertEqual(output.shape, (3, 3))
        self.assertTrue(issymmetric(output.detach().numpy()))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(output) > -1e-6))

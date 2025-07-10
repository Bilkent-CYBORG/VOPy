from unittest import TestCase

import numpy as np

from vopy.utils.transforms import NormalizeInput, StandardizeOutput


class TestNormalizeInput(TestCase):
    """Tests for the `NormalizeInput` `InputTransform` class."""

    def setUp(self):
        self.X = np.array([[0.5, 0.5], [0.2, 0.8]])

    def test_init(self):
        """Test the initialization of `NormalizeInput`."""
        with self.assertRaises(ValueError):
            NormalizeInput(d=2, bounds=np.array([[0, 1], [0, 1], [0, 1]]))

    def test_transform(self):
        transform = NormalizeInput(d=2, bounds=np.array([[-1, 1], [-1, 1]]))
        transformed_X = transform.transform(self.X)
        expected_X = (self.X + 1) / 2
        np.testing.assert_array_almost_equal(transformed_X, expected_X)

        # Checks
        with self.assertRaises(RuntimeError):
            transform.transform(np.array([[0.5, 0.2, 0.1], [0.5, 0.5, 0.5]]))
        with self.assertRaises(ValueError):
            transform.bounds = None
            transform.transform(self.X)

    def test_untransform(self):
        """Test the untransform method of `NormalizeInput`."""
        transform = NormalizeInput(d=2, bounds=np.array([[-1, 1], [-1, 1]]))
        transformed_X = transform.untransform((self.X + 1) / 2)
        expected_X = self.X
        np.testing.assert_array_almost_equal(transformed_X, expected_X)

        # Checks
        with self.assertRaises(ValueError):
            transform.bounds = None
            transform.untransform((self.X + 1) / 2)

    def test_fit_transform(self):
        """Test the fit_transform method of `NormalizeInput`."""
        transform = NormalizeInput(d=2)

        # Checks
        with self.assertRaises(ValueError):
            transform.fit_transform(np.array([0.5, 0.2]))
        with self.assertRaises(RuntimeError):
            transform.fit_transform(np.array([[0.5, 0.2, 0.1], [0.5, 0.5, 0.5]]))
        with self.assertRaises(ValueError):
            transform.fit_transform(np.zeros((0, 2)))

        # Calculated bounds
        X = np.vstack((self.X, np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])))
        transformed_X = transform.fit_transform(X)
        expected_X = (X + 1) / 2
        np.testing.assert_array_almost_equal(transformed_X, expected_X)


class TestStandardizeOutput(TestCase):
    """Tests for the `StandardizeOutput` `InputTransform` class."""

    def setUp(self):
        self.X = np.array([[0.5, 0.5], [0.2, -0.8], [0.3, 1.2], [-0.2, 0.1]])

    def test_transform(self):
        """Test the transform method of `StandardizeOutput`."""
        transform = StandardizeOutput(m=2)
        # Checks
        with self.assertRaises(ValueError):
            transform.transform(self.X)

    def test_untransform(self):
        """Test the untransform method of `StandardizeOutput`."""
        transform = StandardizeOutput(m=2)
        # Checks
        with self.assertRaises(ValueError):
            transform.untransform(self.X)

        transformed_X = transform.fit_transform(self.X)
        untransformed_X = transform.untransform(transformed_X)

        np.testing.assert_array_almost_equal(untransformed_X, self.X)

    def test_fit_transform(self):
        """Test the fit_transform method of `StandardizeOutput`."""
        transform = StandardizeOutput(m=2)

        # Checks
        with self.assertRaises(ValueError):
            transform.fit_transform(np.array([0.5, 0.2]))
        with self.assertRaises(RuntimeError):
            transform.fit_transform(np.array([[0.5, 0.2, 0.1], [0.5, 0.5, 0.5]]))
        with self.assertRaises(ValueError):
            transform.fit_transform(np.zeros((0, 2)))

        # Calculated bounds
        transformed_X = transform.fit_transform(self.X)
        np.testing.assert_array_almost_equal(np.mean(transformed_X, axis=0), np.zeros(2))
        np.testing.assert_array_almost_equal(np.std(transformed_X, axis=0), np.ones(2))

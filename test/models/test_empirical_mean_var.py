from unittest import TestCase

import numpy as np
from vopy.models import EmpiricalMeanVarModel


class TestEmpiricalMeanVarModel(TestCase):
    """Test the EmpiricalMeanVarModel class."""

    def setUp(self):
        # A basic setup for the model.
        self.input_dim = 2
        self.output_dim = 2
        self.noise_var = 0.1
        self.design_count = 3
        self.track_means = True
        self.track_variances = True
        self.model = EmpiricalMeanVarModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            noise_var=self.noise_var,
            design_count=self.design_count,
            track_means=self.track_means,
            track_variances=self.track_variances,
        )

    def test_add_sample(self):
        """Test the add_sample method."""
        with self.assertRaises(ValueError):
            indices = [0]
            Y_t = np.array([[1, 2], [3, 4]])
            self.model.add_sample(indices, Y_t)

        with self.assertRaises(ValueError):
            indices = [6]
            Y_t = np.array([[1, 2]])
            self.model.add_sample(indices, Y_t)

        indices = [0, 0, 1]
        Y_t = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.add_sample(indices, Y_t)
        sample_counts = [2, 1, 0]
        for idx, count in enumerate(sample_counts):
            with self.subTest(idx=idx):
                self.assertEqual(self.model.design_samples[idx].shape[0], count)

    def test_clear_data(self):
        """Test the clear_data method."""
        self.model.clear_data()
        for i in range(self.design_count):
            with self.subTest(i=i):
                self.assertEqual(self.model.design_samples[i].shape, (0, self.output_dim))

    def test_update(self):
        """Test the update method."""
        self.model.add_sample([0, 0, 1], np.array([[1, 2], [3, 4], [5, 6]]))

        self.model.track_means = False
        self.model.track_variances = False
        self.model.update()
        self.assertIsNone(self.model.means)
        self.assertIsNone(self.model.variances)

        self.model.track_means = True
        self.model.track_variances = True
        self.model.update()
        means_tracked = np.array([[2, 3], [5, 6], [0, 0]])
        variances_tracked = np.array(
            [
                [[1, 0], [0, 1]],
                [[self.noise_var, 0], [0, self.noise_var]],
                [[self.noise_var, 0], [0, self.noise_var]],
            ]
        )
        np.testing.assert_allclose(self.model.means, means_tracked)
        np.testing.assert_allclose(self.model.variances, variances_tracked)

    def test_train(self):
        """Test the train method."""
        self.model.train()
        self.assertTrue(True)

    def test_predict(self):
        """Test the predict method."""
        self.model.add_sample([0, 0, 1], np.array([[1, 2], [3, 4], [5, 6]]))
        self.model.update()

        # NOTE: Actual coordinates of test points are not used, only the indices are used.

        with self.assertRaises(ValueError):
            self.model.predict(np.array([[-1, -1]]))

        X = np.array([[-1, -1, 0], [-1, -1, 1]])

        # Test if the predict method returns the correct output when track_means
        # and track_variances are disabled.
        self.model.track_variances = False
        self.model.track_means = False
        means = np.array([[0, 0], [0, 0]])
        variances = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        out_mean, out_variances = self.model.predict(X)
        np.testing.assert_allclose(out_mean, means)
        np.testing.assert_allclose(out_variances, variances)

        # Test if the predict method returns the correct output when track_means
        # and track_variances are enabled.
        self.model.track_variances = True
        self.model.track_means = True
        means = np.array([[2, 3], [5, 6]])
        variances = np.array([[[1, 0], [0, 1]], [[self.noise_var, 0], [0, self.noise_var]]])
        out_mean, out_variances = self.model.predict(X)
        np.testing.assert_allclose(out_mean, means)
        np.testing.assert_allclose(out_variances, variances)

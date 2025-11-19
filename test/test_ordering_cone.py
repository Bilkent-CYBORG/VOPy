from unittest import mock, TestCase

import numpy as np
from vopy.ordering_cone import ConeTheta2D, OrderingCone


class TestOrderingCone(TestCase):
    """
    Test the OrderingCone class.
    """

    def test_is_inside(self):
        W = [[1, 0], [0, 1]]
        ordering_cone = OrderingCone(W)

        self.assertTrue(ordering_cone.is_inside([1, 1]))
        self.assertTrue(ordering_cone.is_inside([0, 0]))
        self.assertFalse(ordering_cone.is_inside([-1, -1]))

        W = [[-1, 0], [0, -1]]
        ordering_cone = OrderingCone(W)

        self.assertFalse(ordering_cone.is_inside([1, 1]))
        self.assertTrue(ordering_cone.is_inside([-1, -1]))

    def test_plot(self):
        with mock.patch("vopy.ordering_cone.plot_2d_cone") as mock_plot:
            W = np.eye(2)
            ordering_cone = OrderingCone(W)
            ordering_cone.plot()

            mock_plot.assert_called_once()

        with mock.patch("vopy.ordering_cone.plot_3d_cone") as mock_plot:
            W = np.eye(3)
            ordering_cone = OrderingCone(W)
            ordering_cone.plot()

            mock_plot.assert_called_once()

        with self.assertRaises(ValueError):
            W = np.eye(4)
            ordering_cone = OrderingCone(W)
            ordering_cone.plot()

    def test_equal(self):
        W = [[1, 0], [0, 1]]
        ordering_cone1 = OrderingCone(W)
        ordering_cone2 = OrderingCone(W)

        self.assertEqual(ordering_cone1, ordering_cone2)

        W = [[1, 0], [0, 2]]
        ordering_cone2 = OrderingCone(W)

        self.assertNotEqual(ordering_cone1, ordering_cone2)


class TestConeTheta2D(TestCase):
    """
    Test the OrderingCone class.
    """

    def test_init(self):
        degree = 45
        angle_rad = np.deg2rad(degree)

        ordering_cone = ConeTheta2D(degree)
        self.assertEqual(ordering_cone.cone_degree, degree)
        self.assertAlmostEqual(ordering_cone.beta, 1 / np.sin(angle_rad))

    def test_plot(self):
        with mock.patch("vopy.ordering_cone.plot_2d_theta_cone") as mock_plot:
            ordering_cone = ConeTheta2D(45)
            ordering_cone.plot()

            mock_plot.assert_called_once()

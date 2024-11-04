from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from vectoptal.order import ConeTheta2DOrder, ConeOrder3D
from vectoptal.utils.plotting import (
    plot_2d_cone,
    plot_3d_cone,
    plot_2d_theta_cone,
    plot_pareto_front,
)


class TestPlotting(TestCase):
    """
    Test the plotting functions.
    """

    def setUp(self):
        self.cone_degree = 45
        self.order_2d = ConeTheta2DOrder(self.cone_degree)
        self.order_3d = ConeOrder3D("acute")

    def test_plot_2d_theta_cone(self):
        result = plot_2d_theta_cone(self.cone_degree)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_2d_cone(self):
        result = plot_2d_cone(self.order_2d.ordering_cone.is_inside)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_3d_cone(self):
        result = plot_3d_cone(self.order_3d.ordering_cone.is_inside)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_pareto_front(self):
        elements_2d = np.array([[1, 2], [2, 1], [3, 3]])
        elements_3d = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 3]])
        result = plot_pareto_front(elements_2d, self.order_2d.get_pareto_set(elements_2d))
        self.assertIsInstance(result, plt.Figure)
        result = plot_pareto_front(elements_3d, self.order_3d.get_pareto_set(elements_3d))
        self.assertIsInstance(result, plt.Figure)

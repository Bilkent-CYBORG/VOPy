from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from vopy.order import ConeOrder3D, ConeTheta2DOrder
from vopy.utils.plotting import (
    plot_2d_cone,
    plot_2d_theta_cone,
    plot_3d_cone,
    plot_cells_with_centers,
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
        result = plot_2d_theta_cone(self.cone_degree, show=False)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_2d_cone(self):
        result = plot_2d_cone(self.order_2d.ordering_cone.is_inside, show=False)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_3d_cone(self):
        result = plot_3d_cone(self.order_3d.ordering_cone.is_inside, show=False)
        self.assertIsInstance(result, plt.Figure)

    def test_plot_pareto_front(self):
        elements_2d = np.array([[1, 2], [2, 1], [3, 3]])
        elements_3d = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 3]])
        result = plot_pareto_front(
            elements_2d, self.order_2d.get_pareto_set(elements_2d), show=False
        )
        self.assertIsInstance(result, plt.Figure)
        result = plot_pareto_front(
            elements_3d, self.order_3d.get_pareto_set(elements_3d), show=False
        )
        self.assertIsInstance(result, plt.Figure)

    def test_plot_cells_with_centers(self):
        cells = np.array([[[0, 0.5], [0, 1]], [[0.5, 1], [0, 1]]])
        centers = np.array([[0.25, 0.5], [0.75, 0.5]])
        result = plot_cells_with_centers(cells, centers, show=False)
        self.assertIsInstance(result, plt.Figure)

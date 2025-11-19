from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from vopy.order import (
    ComponentwiseOrder,
    ConeOrder3D,
    ConeOrder3DIceCream,
    ConeTheta2DOrder,
    PolyhedralConeOrder,
)
from vopy.ordering_cone import ConeTheta2D, OrderingCone


class TestPolyhedralConeOrder(TestCase):
    """Test the PolyhedralConeOrder class."""

    def test_dominates(self):
        ordering_cone = OrderingCone(np.eye(2))
        order = PolyhedralConeOrder(ordering_cone)

        self.assertTrue(order.dominates(np.array([1, 1]), np.array([0, 0])))
        self.assertFalse(order.dominates(np.array([0, 0]), np.array([1, 1])))

    def test_get_pareto_set(self):
        ordering_cone = OrderingCone(np.eye(2))
        order = PolyhedralConeOrder(ordering_cone)
        elements = np.array([[0.5, 1], [0, 0], [1, 0.5]])
        self.assertListEqual(order.get_pareto_set(elements).tolist(), [0, 2])

        ordering_cone = OrderingCone(np.eye(2))
        order = PolyhedralConeOrder(ordering_cone)
        elements = np.array([[0.5, 1], [0, 0], [1, 0.5]])
        self.assertListEqual(order.get_pareto_set_naive(elements).tolist(), [0, 2])

    def test_plot_pareto_set(self):
        ordering_cone = OrderingCone(np.eye(2))
        order = PolyhedralConeOrder(ordering_cone)
        elements = np.array([[1, 1], [0, 0], [-1, -1]])

        fig = order.plot_pareto_set(elements)
        self.assertIsInstance(fig, plt.Figure)

        with self.assertRaises(ValueError):
            elements = np.zeros((3))
            order.plot_pareto_set(elements)

        with self.assertRaises(ValueError):
            elements = np.zeros((3, 5))
            order.plot_pareto_set(elements)


class TestComponentwiseOrder(TestCase):
    """Test the ComponentwiseOrder class."""

    def test_init(self):
        order = ComponentwiseOrder(2)
        self.assertIsInstance(order, PolyhedralConeOrder)
        self.assertEqual(order.ordering_cone, OrderingCone(np.eye(2)))


class TestConeTheta2DOrder(TestCase):
    """Test the ConeTheta2DOrder class."""

    def test_init(self):
        order = ConeTheta2DOrder(45)
        self.assertIsInstance(order, PolyhedralConeOrder)
        self.assertEqual(order.ordering_cone, ConeTheta2D(45))


class TestConeOrder3D(TestCase):
    """Test the ConeOrder3D class."""

    def test_init(self):
        with self.assertRaises(ValueError):
            order = ConeOrder3D(cone_type="weird_3d_cone_type")

        for cone_type in ["acute", "right", "obtuse"]:
            with self.subTest(cone_type=cone_type):
                order = ConeOrder3D(cone_type)
                self.assertIsInstance(order, PolyhedralConeOrder)
                self.assertTrue(np.allclose(np.linalg.norm(order.ordering_cone.W, axis=1), 1))


class TestConeOrder3DIceCream(TestCase):
    """Test the ConeOrder3DIceCream class."""

    def test_init(self):
        for num_halfspace in range(3, 5):
            with self.subTest(num_halfspace=num_halfspace):
                order = ConeOrder3DIceCream(60, num_halfspace)
                self.assertIsInstance(order, PolyhedralConeOrder)
                self.assertTrue(np.allclose(np.linalg.norm(order.ordering_cone.W, axis=1), 1))

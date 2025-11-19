import copy
from unittest import TestCase

import numpy as np
from vopy.confidence_region import EllipsoidalConfidenceRegion, RectangularConfidenceRegion
from vopy.order import ComponentwiseOrder


class TestRectangularConfidenceRegion(TestCase):
    """Test the RectangularConfidenceRegion class."""

    def setUp(self):
        self.dim = 2
        self.slackness = np.zeros(2)
        self.intersect_iteratively = True

        self.lower = np.array([0, 0])
        self.upper = np.array([1, 1])
        self.confidence_region = RectangularConfidenceRegion(
            self.dim, self.lower, self.upper, self.intersect_iteratively
        )

        self.lower2 = np.array([1.1, 1.1])
        self.upper2 = np.array([1.5, 1.5])
        self.confidence_region2 = RectangularConfidenceRegion(
            self.dim, self.lower2, self.upper2, self.intersect_iteratively
        )

        self.lower3 = np.array([0.9, 0.9])
        self.upper3 = np.array([1.2, 1.2])
        self.confidence_region3 = RectangularConfidenceRegion(
            self.dim, self.lower3, self.upper3, self.intersect_iteratively
        )

        self.lower4 = np.array([0.5, -0.1])
        self.upper4 = np.array([1.5, 1.5])
        self.confidence_region4 = RectangularConfidenceRegion(
            self.dim, self.lower4, self.upper4, self.intersect_iteratively
        )

        self.lower5 = np.array([2, -2])
        self.upper5 = np.array([3, -1])
        self.confidence_region5 = RectangularConfidenceRegion(
            self.dim, self.lower5, self.upper5, self.intersect_iteratively
        )

        self.lower6 = np.array([2, -2])
        self.upper6 = np.array([3, 0.5])
        self.confidence_region6 = RectangularConfidenceRegion(
            self.dim, self.lower6, self.upper6, self.intersect_iteratively
        )

        self.order = ComponentwiseOrder(2)

    def test_diagonal(self):
        """Test the diagonal method."""
        self.assertTrue(np.sqrt(2) == self.confidence_region.diagonal())

    def test_update(self):
        """Test the update method."""
        with self.assertRaises(ValueError):
            self.confidence_region.update(np.array([1, 1]), np.ones([2, 3]))

        self.confidence_region.update(np.array([1.25, 1.25]), 0.25 * np.eye(2), np.array(2))
        self.assertTrue(
            np.allclose(self.confidence_region.lower, np.array([0.25, 0.25]))
            and np.allclose(self.confidence_region.upper, np.array([1, 1]))
        )
        self.confidence_region.intersect_iteratively = False
        self.confidence_region.update(np.array([-0.4, -0.25]), 0.25 * np.eye(2))
        self.assertTrue(
            np.allclose(self.confidence_region.lower, np.array([-0.9, -0.75]))
            and np.allclose(self.confidence_region.upper, np.array([0.1, 0.25]))
        )

    def test_center(self):
        """Test the center method."""
        self.assertTrue((self.confidence_region.center == np.array([0.5, 0.5])).all())

    def test_intersect(self):
        """Test the intersect method."""
        self.confidence_region.intersect(np.array([-0.1, -0.1]), np.array([0.5, 0.5]))
        self.assertTrue(
            (self.confidence_region.lower == np.array([0, 0])).all()
            and (self.confidence_region.upper == np.array([0.5, 0.5])).all()
        )
        self.confidence_region.intersect(np.array([0.6, 0.6]), np.array([1.0, 1.0]))
        self.assertTrue(
            (self.confidence_region.lower == np.array([0.6, 0.6])).all()
            and (self.confidence_region.upper == np.array([1.0, 1.0])).all()
        )

    def test_is_dominated(self):
        """Test the is_dominated method."""
        self.assertTrue(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region2, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region3, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region4, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region5, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region6, self.slackness
            )
        )

    def test_check_dominates(self):
        """Test the check_dominates method."""
        self.assertTrue(
            self.confidence_region.check_dominates(
                self.order, self.confidence_region2, self.confidence_region, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.check_dominates(
                self.order, self.confidence_region3, self.confidence_region, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.check_dominates(
                self.order, self.confidence_region4, self.confidence_region, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.check_dominates(
                self.order, self.confidence_region5, self.confidence_region, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.check_dominates(
                self.order, self.confidence_region6, self.confidence_region, self.slackness
            )
        )

    def test_is_covered(self):
        """Test the is_covered method."""
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region2, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region3, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region4, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region5, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region6, self.slackness
            )
        )


class TestEllipsoidalConfidenceRegion(TestCase):
    """Test the EllipsoidalConfidenceRegion class."""

    def setUp(self):
        self.dim = 2
        self.alpha = 1
        self.slackness = np.zeros(2)

        self.center = np.array([1, 1])
        self.sigma = np.eye(2)
        self.confidence_region = EllipsoidalConfidenceRegion(
            self.dim, self.center, self.sigma, self.alpha
        )

        self.center2 = np.array([3, 3])
        self.sigma2 = np.eye(2)
        self.confidence_region2 = EllipsoidalConfidenceRegion(
            self.dim, self.center2, self.sigma2, self.alpha
        )

        self.center3 = np.array([2, 2])
        self.sigma3 = np.eye(2)
        self.confidence_region3 = EllipsoidalConfidenceRegion(
            self.dim, self.center3, self.sigma3, self.alpha
        )

        self.center4 = np.array([2, 1])
        self.sigma4 = np.eye(2) * 1.1
        self.confidence_region4 = EllipsoidalConfidenceRegion(
            self.dim, self.center4, self.sigma4, self.alpha
        )

        self.center5 = np.array([7, -3])
        self.sigma5 = np.eye(2)
        self.confidence_region5 = EllipsoidalConfidenceRegion(
            self.dim, self.center5, self.sigma5, self.alpha
        )

        self.center6 = np.array([7, 0])
        self.sigma6 = np.eye(2)
        self.confidence_region6 = EllipsoidalConfidenceRegion(
            self.dim, self.center6, self.sigma6, self.alpha
        )

        self.order = ComponentwiseOrder(2)

    def test_update(self):
        """Test the update method."""
        with self.assertRaises(ValueError):
            self.confidence_region.update(np.array([1, 1]), np.ones([2, 3]))

        self.confidence_region_new = copy.deepcopy(self.confidence_region)
        self.confidence_region.update(self.center, self.sigma)
        self.assertTrue((self.confidence_region.center == self.confidence_region_new.center).all())

    def test_is_dominated(self):
        """Test the is_dominated method."""
        self.assertTrue(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region2, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region3, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region4, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region5, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_dominated(
                self.order, self.confidence_region, self.confidence_region6, self.slackness
            )
        )

    def test_check_dominates(self):
        """Test the check_dominates method."""
        with self.assertRaises(NotImplementedError):
            self.confidence_region.check_dominates(
                self.order, self.confidence_region, self.confidence_region2, self.slackness
            )

    def test_is_covered(self):
        """Test the is_covered method."""
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region2, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region3, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region4, self.slackness
            )
        )
        self.assertFalse(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region5, self.slackness
            )
        )
        self.assertTrue(
            self.confidence_region.is_covered(
                self.order, self.confidence_region, self.confidence_region6, self.slackness
            )
        )

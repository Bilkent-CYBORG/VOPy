from unittest import mock, TestCase

import numpy as np
from vopy.datasets import get_dataset_instance
from vopy.design_space import AdaptivelyDiscretizedDesignSpace, FixedPointsDesignSpace
from vopy.models.model import UncertaintyPredictiveModel
from vopy.utils import set_seed
from vopy.utils.seed import SEED


class TestFixedPointsDesignSpace(TestCase):
    """
    Test the FixedPointsDesignSpace class.
    """

    def setUp(self):
        set_seed(SEED)

        self.dataset_name = "Test"
        self.dataset = get_dataset_instance(self.dataset_name)

        self.mock_model = mock.MagicMock(spec=UncertaintyPredictiveModel)

        self.points = self.dataset.in_data
        self.out_dim = self.dataset._out_dim
        self.design_space = FixedPointsDesignSpace(self.points, self.out_dim, "hyperrectangle")

    def test_init(self):
        """
        Test the __init__ method.
        """
        self.assertTrue(np.allclose(self.design_space.points, self.points))
        self.assertEqual(len(self.design_space.confidence_regions), len(self.points))
        self.assertEqual(self.design_space.cardinality, len(self.points))

    def test_update_raises(self):
        """
        Test the update method raises an error.
        """

        with self.assertRaises(NotImplementedError):
            FixedPointsDesignSpace(self.points, self.out_dim, "weird_confidence_shape")

        with self.assertRaises(ValueError):
            self.design_space.update(self.mock_model, scale=np.ones((3, 5, 6)))

    @mock.patch("vopy.design_space.RectangularConfidenceRegion.update")
    def test_update_all_pts(self, mock_conf_region_update):
        """
        Test the update method with all points.
        """
        self.mock_model.predict.return_value = (
            np.random.randn(self.points.shape[0], self.out_dim),
            np.eye(self.out_dim)
            * np.random.rand(self.points.shape[0], self.out_dim)[:, np.newaxis, :],
        )

        self.design_space.update(self.mock_model, scale=np.ones(1))
        self.assertEqual(mock_conf_region_update.call_count, len(self.points))


class TestAdaptivelyDiscretizedDesignSpace(TestCase):
    """
    Test the AdaptivelyDiscretizedDesignSpace class.
    """

    def setUp(self):
        set_seed(SEED)

        self.mock_model = mock.MagicMock(spec=UncertaintyPredictiveModel)

        self.delta = 0.1
        self.max_depth = 2
        self.in_dim = 3
        self.out_dim = 2
        self.design_space = AdaptivelyDiscretizedDesignSpace(
            self.in_dim, self.out_dim, self.delta, self.max_depth, "hyperrectangle"
        )

    def test_init(self):
        """
        Test the __init__ method.
        """
        self.assertEqual(self.design_space.cardinality, 1)
        self.assertEqual(len(self.design_space.confidence_regions), 1)

    def test_raises(self):
        """
        Test the raised errors of the class.
        """

        with self.assertRaises(NotImplementedError):
            AdaptivelyDiscretizedDesignSpace(
                self.in_dim, self.out_dim, self.delta, self.max_depth, "hyperellipsoid"
            )

        with self.assertRaises(ValueError):
            self.design_space.update(self.mock_model, scale=np.ones((3, 5, 6)))

    @mock.patch("vopy.design_space.RectangularConfidenceRegion.update")
    def test_update_all_pts(self, mock_conf_region_update):
        """
        Test the update method with all points.
        """
        self.mock_model.predict.return_value = (
            np.random.randn(self.design_space.points.shape[0], self.out_dim),
            np.eye(self.out_dim)
            * np.random.rand(self.design_space.points.shape[0], self.out_dim)[:, np.newaxis, :],
        )

        self.design_space.update(self.mock_model, scale=np.ones(1))
        self.assertEqual(mock_conf_region_update.call_count, len(self.design_space.points))

    def test_refine_design(self):
        """
        Test the refine_design method.
        """
        previous_cardinality = self.design_space.cardinality
        self.design_space.refine_design(0)

        self.assertEqual(self.design_space.cardinality, previous_cardinality + 2**self.in_dim)
        self.assertEqual(len(self.design_space.confidence_regions), self.design_space.cardinality)
        self.assertEqual(self.design_space.point_depths[-1], self.design_space.point_depths[0] + 1)

    def test_should_refine(self):
        """
        Test the should_refine method.
        """
        self.design_space.refine_design(0)
        self.mock_model.predict.return_value = (
            np.random.randn(self.design_space.points.shape[0], self.out_dim),
            np.eye(self.out_dim)
            * np.random.rand(self.design_space.points.shape[0], self.out_dim)[:, np.newaxis, :],
        )
        self.design_space.update(self.mock_model, scale=np.ones(1))

        self.assertFalse(self.design_space.should_refine_design(self.mock_model, 1, np.array(1)))

    def visualize_design_space(self):
        """
        Test the visualize_design_space method.
        """
        fig = self.design_space.visualize_design_space()

        self.assertIsNotNone(fig)

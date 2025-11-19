from unittest import mock, TestCase

import numpy as np

import torch

from vopy.datasets import get_dataset_instance
from vopy.utils import (
    binary_entropy,
    generate_sobol_samples,
    get_2d_w,
    get_alpha,
    get_alpha_vec,
    get_bigmij,
    get_closest_indices_from_points,
    get_delta,
    get_noisy_evaluations_chol,
    get_smallmij,
    get_uncovered_set,
    get_uncovered_size,
    hyperrectangle_check_intersection,
    hyperrectangle_get_region_matrix,
    hyperrectangle_get_vertices,
    is_covered,
    is_pt_in_extended_polytope,
    line_seg_pt_intersect_at_dim,
    normalize,
    set_seed,
    unnormalize,
)
from vopy.utils.seed import SEED


class TestSetSeed(TestCase):
    """Test seed setting."""

    def test_set_seed(self):
        """Test the set_seed function."""
        seeds = [0, 42, 123]
        for seed in seeds:
            with self.subTest(seed=seed):
                set_seed(seed)

                self.assertEqual(np.random.get_state()[1][0], seed)
                self.assertListEqual(
                    torch.random.get_rng_state().tolist(),
                    torch.manual_seed(seed).get_state().tolist(),
                )


class TestGet2DW(TestCase):
    """Test 2D cone matrix generation."""

    def test_get_2d_w(self):
        """Test the get_2d_w function."""
        cone_angles = [45, 60, 90, 120, 135]
        for cone_angle in cone_angles:
            with self.subTest(cone_angle=cone_angle):
                W = get_2d_w(cone_angle)

                for i in range(2):
                    self.assertAlmostEqual(np.linalg.norm(W[i]), 1.0)

                self.assertAlmostEqual(-np.dot(W[0], W[1]), np.cos(np.deg2rad(cone_angle)))


class TestGetAlpha(TestCase):
    """Test alpha computation."""

    def test_get_alpha(self):
        """Test the get_alpha function."""
        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            cos_ang = np.cos(np.deg2rad(max(90 - angle, 0)))
            W = get_2d_w(angle)
            for rind in range(W.shape[0]):
                with self.subTest(rind=rind):
                    alpha = get_alpha(rind, W)
                    self.assertAlmostEqual(alpha, cos_ang)

    def test_get_alpha_vec(self):
        """Test the get_alpha_vec function."""
        angle = 45
        cos_ang = np.cos(np.deg2rad(max(90 - angle, 0)))
        W = get_2d_w(angle)
        alphas = get_alpha_vec(W)

        np.testing.assert_allclose(alphas, [[cos_ang]] * W.shape[0])


class TestGetClosestIndicesFromPoints(TestCase):
    """Test closest indices computation."""

    def setUp(self):
        self.points = torch.tensor([[0, 0], [1, 1]])
        self.queries = torch.tensor([[0.1, 0.1], [0.5, 0.9], [1.2, 1.2]])

    def test_does_return_first_match(self):
        """Test if the get_closest_indices_from_points returns the first between equals."""
        result = get_closest_indices_from_points(
            torch.tensor([[0.5, 0.5]]), self.points, return_distances=False, squared=False
        )
        self.assertListEqual(result.tolist(), [0])

    def test_get_closest_indices_from_points(self):
        """Test the get_closest_indices_from_points function."""
        self.assertListEqual(
            get_closest_indices_from_points(
                self.queries, [], return_distances=False, squared=False
            ),
            [],
        )

        result_sq = get_closest_indices_from_points(
            self.queries, self.points, return_distances=False, squared=True
        )
        result = get_closest_indices_from_points(
            self.queries, self.points, return_distances=False, squared=False
        )
        self.assertListEqual(result_sq.tolist(), result.tolist())
        self.assertListEqual(result.tolist(), [0, 1, 1])

        result_sq, dists_sq = get_closest_indices_from_points(
            self.queries, self.points, return_distances=True, squared=True
        )
        result, dists = get_closest_indices_from_points(
            self.queries, self.points, return_distances=True, squared=False
        )

        self.assertListEqual(result_sq.tolist(), result.tolist())
        self.assertListEqual(result.tolist(), [0, 1, 1])
        np.testing.assert_allclose(dists_sq, dists**2)
        np.testing.assert_allclose(dists_sq, [0.02, 0.26, 0.08], atol=1e-7)


class TestGetNoisyEvaluationsChol(TestCase):
    """Test noisy evaluations generation."""

    @mock.patch("vopy.utils.utils.np.random.normal")
    def test_get_noisy_evaluations_chol(self, mock_normal):
        """Test the get_noisy_evaluations_chol function."""
        n = 10
        x = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sin(2 * np.pi * x)

        with self.assertRaises(AssertionError):
            get_noisy_evaluations_chol(y, np.zeros((2,)))

        mock_normal.return_value = x
        np.testing.assert_allclose(get_noisy_evaluations_chol(y, np.zeros((1, 1))), y)

        sigma = np.ones((1, 1)) * 0.1
        y_noisy = get_noisy_evaluations_chol(y, sigma)
        self.assertEqual(y_noisy.shape, y.shape)
        np.testing.assert_allclose(y_noisy, y + sigma * x)


class TestGenerateSobolSamples(TestCase):
    """Test Sobol samples generation."""

    def test_generate_sobol_samples(self):
        """Test the generate_sobol_samples function."""
        dim = 2
        n = 16
        samples = generate_sobol_samples(dim, n)

        self.assertEqual(samples.shape, (n, dim))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < 1))
        self.assertEqual(len(np.unique(samples, axis=0)), n)

    def test_generate_sobol_samples_randomness(self):
        """Test the randomness of generate_sobol_samples function."""
        dim = 2
        n = 16
        set_seed(SEED)
        samples_fst = generate_sobol_samples(dim, n)
        set_seed(SEED)
        samples_scd = generate_sobol_samples(dim, n)

        np.testing.assert_allclose(samples_fst, samples_scd)


class Testmij(TestCase):
    """Test m(i, j) and M(i, j) computations."""

    def test_get_smallmij(self):
        """Test the get_smallmij function."""
        vi = np.array([1, 0])
        vj = np.array([1.1, 0.2])

        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            with self.subTest(angle=angle):
                W = get_2d_w(angle)
                alpha_vec = get_alpha_vec(W)

                W_normalized = (W.T / alpha_vec.flatten()).T
                diff = vj - vi

                m = get_smallmij(vi, vj, W, alpha_vec)
                np.testing.assert_allclose(
                    m, min(np.clip(W_normalized @ diff, a_min=0, a_max=None))
                )

    def test_get_bigmij(self):
        """Test the get_bigmij function."""
        vi = np.array([1, 0])
        vj = np.array([1.1, 0.2])

        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            with self.subTest(angle=angle):
                W = get_2d_w(angle)
                alpha_vec = get_alpha_vec(W)

                W_normalized = (W.T / alpha_vec.flatten()).T
                diff = vj - vi
                M_pred = min(np.clip(-(W_normalized @ diff), a_min=0, a_max=None))

                M = get_bigmij(vi, vj, W)
                np.testing.assert_allclose(M, M_pred)


class TestGetDelta(TestCase):
    """Test delta gap computation."""

    def test_get_delta(self):
        """Test the get_delta function."""
        means = np.array([[1, 0], [1.1, 0.2], [0.5, 0.8]])

        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            with self.subTest(angle=angle):
                W = get_2d_w(angle)
                alpha_vec = get_alpha_vec(W)

                delta_true = get_delta(means, W, alpha_vec)
                delta_expected = np.zeros_like(delta_true)
                for i in range(means.shape[0]):
                    vi = means[i]
                    for j in range(means.shape[0]):
                        vj = means[j]
                        mij = get_smallmij(vi, vj, W, alpha_vec)
                        delta_expected[i] = max(delta_expected[i], mij)
                        self.assertLessEqual(mij, delta_true[i])
                np.testing.assert_allclose(delta_true, delta_expected)


class TestEpsilonCover(TestCase):
    """Test epsilon coverage of Pareto points w.r.t. the ordering."""

    def setUp(self):
        self.epsilon = 0.1

    def test_is_covered_specific_data(self):
        """
        Test the is_covered function with a specific case that fails with CLARABEL.
        Details: https://github.com/cvxpy/cvxpy/issues/2610
        """

        dataset = get_dataset_instance("Test")
        vi = dataset.out_data[30]
        vj = dataset.out_data[18]
        W = get_2d_w(90)

        self.assertFalse(is_covered(vi, vj, self.epsilon, W))

    def test_is_covered(self):
        """Test the is_covered function."""
        means = np.array(
            [
                [1, 0.2],
                [1.1, 0.2],
                [0.5, 0.8],
            ]
        )

        W = get_2d_w(90)

        # vi equal to vj
        self.assertTrue(is_covered(means[0], means[0], self.epsilon, W))
        # vj is already better than vi, shouldn't happen if vi is pareto optimal
        self.assertTrue(is_covered(means[0], means[1], self.epsilon, W))
        # vi can be reached from vj
        self.assertTrue(is_covered(means[1], means[0], self.epsilon, W))
        # vi incomparable with vj
        self.assertFalse(is_covered(means[1], means[2], self.epsilon, W))

    def test_get_uncovered_set(self):
        """Test the get_uncovered_set function."""
        means = np.array(
            [
                [1, 0.2],
                [1.1, 0.2],
                [0.5, 0.8],
            ]
        )

        W = get_2d_w(90)

        self.assertListEqual(get_uncovered_set([1], [0, 2], means, self.epsilon, W), [])
        self.assertListEqual(get_uncovered_set([1], [2], means, self.epsilon, W), [1])

    def test_get_uncovered_size(self):
        """Test the get_uncovered_size function."""
        means = np.array(
            [
                [1, 0.2],
                [1.1, 0.2],
                [0.5, 0.8],
            ]
        )

        W = get_2d_w(90)

        self.assertEqual(get_uncovered_size(means[[1]], means[[0, 2]], self.epsilon, W), 0)
        self.assertEqual(get_uncovered_size(means[[1]], means[[2]], self.epsilon, W), 1)


class TestHyperrectangleCheckIntersection(TestCase):
    """Test hyperrectangle intersection check."""

    def test_hyperrectangle_check_intersection_2d(self):
        """Test the hyperrectangle_check_intersection function with 2D rectangles."""
        lower1, upper1 = np.array([0, 0]), np.array([1, 1])
        lower2, upper2 = np.array([0.5, 0.5]), np.array([1.5, 1.5])

        self.assertTrue(hyperrectangle_check_intersection(lower1, upper1, lower2, upper2))
        self.assertTrue(hyperrectangle_check_intersection(lower2, upper2, lower1, upper1))

        lower3, upper3 = np.array([2, 2]), np.array([3, 3])
        self.assertFalse(hyperrectangle_check_intersection(lower1, upper1, lower3, upper3))
        self.assertFalse(hyperrectangle_check_intersection(lower3, upper3, lower1, upper1))

    def test_hyperrectangle_check_intersection_3d(self):
        """Test the hyperrectangle_check_intersection function with 3D rectangles."""
        lower1, upper1 = [np.array([-1, -1, -1]), np.array([0, 0, 0])]
        lower2, upper2 = [np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])]
        self.assertFalse(hyperrectangle_check_intersection(lower1, upper1, lower2, upper2))
        self.assertFalse(hyperrectangle_check_intersection(lower2, upper2, lower1, upper1))

        lower3, upper3 = [np.array([-0.5, -0.5, -0.5]), np.array([-0.25, -0.25, -0.25])]
        self.assertTrue(hyperrectangle_check_intersection(lower1, upper1, lower3, upper3))
        self.assertTrue(hyperrectangle_check_intersection(lower3, upper3, lower1, upper1))


class TestHyperrectangleGetVertices(TestCase):
    """Test hyperrectangle vertices computation."""

    def test_hyperrectangle_get_vertices_2d(self):
        """Test the hyperrectangle_get_vertices function with 2D rectangles."""
        lower, upper = np.array([0, 0]), np.array([1, 1])
        vertices = hyperrectangle_get_vertices(lower, upper)

        self.assertEqual(vertices.shape, (2**2, 2))
        self.assertTrue(np.all(vertices >= lower))
        self.assertTrue(np.all(vertices <= upper))

    def test_hyperrectangle_get_vertices_3d(self):
        """Test the hyperrectangle_get_vertices function with 3D rectangles."""
        lower, upper = np.array([0, 0, 0]), np.array([1, 1, 1])
        vertices = hyperrectangle_get_vertices(lower, upper)

        self.assertEqual(vertices.shape, (2**3, 3))
        self.assertTrue(np.all(vertices >= lower))
        self.assertTrue(np.all(vertices <= upper))


class TestHyperrectangleGetRegionMatrix(TestCase):
    """Test hyperrectangle region matrix computation."""

    def test_hyperrectangle_get_region_matrix_2d(self):
        """Test the hyperrectangle_get_region_matrix function with 2D points."""
        lower, upper = np.array([0, 0]), np.array([1, 1])
        region_matrix, region_boundary = hyperrectangle_get_region_matrix(lower, upper)

        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [2, 2], [-1, -1]])
        results = [True, True, True, True, True, False, False]

        self.assertTrue(
            np.all(np.all(points @ region_matrix.T >= region_boundary, axis=1) == results)
        )

    def test_hyperrectangle_get_region_matrix_3d(self):
        """Test the hyperrectangle_get_region_matrix function with 3D points."""
        lower, upper = np.array([-1, -1, -1]), np.array([-0.5, -0.5, -0.5])
        region_matrix, region_boundary = hyperrectangle_get_region_matrix(lower, upper)

        points = np.array([[-0.75, -0.75, -0.75], [0, 0, 0]])
        results = [True, False]

        self.assertTrue(
            np.all(np.all(points @ region_matrix.T >= region_boundary, axis=1) == results)
        )


class TestIsPtInExtendedPolytope(TestCase):
    """Test extended polytope point inclusion check."""

    def test_is_pt_in_extended_polytope(self):
        """Test the is_pt_in_extended_polytope function."""
        polytope = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        # Inside the polytope
        points_inside = [np.array([0.5, 0.5]), np.array([0.1, 0.1]), np.array([0.9, 0.9])]
        for pt in points_inside:
            with self.subTest():
                self.assertTrue(is_pt_in_extended_polytope(pt, polytope))

        # Outside the polytope
        points_outside = [np.array([-0.1, 0.5]), np.array([0.5, -0.1])]
        for pt in points_outside:
            with self.subTest():
                self.assertFalse(is_pt_in_extended_polytope(pt, polytope))

        # Inside the extended polytope
        points_extended = [np.array([1.5, 0.5]), np.array([0.5, 1.5])]
        for pt in points_extended:
            with self.subTest():
                self.assertTrue(is_pt_in_extended_polytope(pt, polytope))

        # Inside the inverted extended polytope
        points_inverted_extended = [np.array([-0.5, 0.5]), np.array([0.5, -0.5])]
        for pt in points_inverted_extended:
            with self.subTest():
                self.assertTrue(is_pt_in_extended_polytope(pt, polytope, invert_extension=True))

        # Outside the inverted extended polytope
        points_outside_inverted_extended = [np.array([1.5, 0.5]), np.array([0.5, 1.5])]
        for pt in points_outside_inverted_extended:
            with self.subTest():
                self.assertFalse(is_pt_in_extended_polytope(pt, polytope, invert_extension=True))

        # Inside the extended parallelogram polytope
        polytope = np.array([[0, 0], [1, 0], [-0.5, 1], [0.5, 1]])
        points_extended = [np.array([-0.25, 0.5])]
        for pt in points_extended:
            with self.subTest():
                self.assertTrue(is_pt_in_extended_polytope(pt, polytope))


class TestLineSegPtIntersectAtDim(TestCase):
    """Test line segment intersection at a specific dimension."""

    def test_line_seg_pt_intersect_at_dim(self):
        """Test the line_seg_pt_intersect_at_dim function."""
        P1 = np.array([0, 0])
        P2 = np.array([1, 1])

        # Intersection at the endpoints
        target_dim = 0
        intersection = line_seg_pt_intersect_at_dim(P1, P2, P1, target_dim)
        np.testing.assert_allclose(intersection, P1)
        intersection = line_seg_pt_intersect_at_dim(P1, P2, P2, target_dim)
        np.testing.assert_allclose(intersection, P2)

        # Intersection at target dimension
        target_pt = np.array([0.8, 0.5])
        target_dim = 0
        intersection = line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim)
        np.testing.assert_allclose(intersection, np.array([0.8, 0.8]))
        target_dim = 1
        intersection = line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim)
        np.testing.assert_allclose(intersection, np.array([0.5, 0.5]))

        # No intersection (target point outside the segment)
        target_pt = np.array([1.5, 1.5])
        target_dim = 0
        intersection = line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim)
        self.assertIsNone(intersection)
        target_dim = 1
        intersection = line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim)
        self.assertIsNone(intersection)


class TestNormalization(TestCase):
    """Test normalization and unnormalization functions."""

    def setUp(self):
        self.data = np.array([[1, 0], [2, 1], [3, 2]], dtype=float)
        self.bounds = [(-1.0, 3.0), (0.0, 2.0)]

    def test_normalize(self):
        """Test the normalize function."""
        normalized_data = normalize(self.data, self.bounds)
        np.testing.assert_allclose(normalized_data, np.array([[0.5, 0.0], [0.75, 0.5], [1.0, 1.0]]))

    def test_unnormalize(self):
        """Test the unnormalize function."""
        unnormalized_data = unnormalize(normalize(self.data, self.bounds), self.bounds)
        np.testing.assert_allclose(unnormalized_data, self.data)


class TestBinaryEntropy(TestCase):
    """Test binary entropy computation."""

    def test_binary_entropy(self):
        """Test the binary_entropy function."""
        p = np.array([0, 0.5, 1])
        expected = np.array([0, 1, 0])

        np.testing.assert_allclose(binary_entropy(p), expected)

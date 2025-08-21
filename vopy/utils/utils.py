import itertools
import random
from typing import Iterable, Optional, Union

import cvxpy as cp
import numpy as np
import scipy.special

import torch
from scipy.stats.qmc import Sobol
from sklearn.metrics.pairwise import euclidean_distances


def set_seed(seed: int) -> None:
    """
    This function sets the seed for both NumPy and PyTorch random number generators.

    :param seed: The seed value to set for the random number generators.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_2d_w(cone_degree: float) -> np.ndarray:
    """
    This function generates a 2D cone matrix W with boundaries at an angle cone_angle and
    symmetric around :math:`y=x`.

    :param cone_angle: The angle of the cone in degrees.
    :type cone_angle: float
    :return: A 2x2 numpy array where each row is a normalized normal vector.
    :rtype: numpy.ndarray
    """
    angle_radian = (cone_degree / 180) * np.pi
    if cone_degree <= 90:
        W_1 = np.array([-np.tan(np.pi / 4 - angle_radian / 2), 1])
        W_2 = np.array([+np.tan(np.pi / 4 + angle_radian / 2), -1])
    else:
        W_1 = np.array([-np.tan(np.pi / 4 - angle_radian / 2), 1])
        W_2 = np.array([-np.tan(np.pi / 4 + angle_radian / 2), 1])
    W_1 = W_1 / np.linalg.norm(W_1)
    W_2 = W_2 / np.linalg.norm(W_2)
    W = np.vstack((W_1, W_2))

    return W


def get_alpha(rind: int, W: np.ndarray) -> np.ndarray:
    """
    Compute alpha_rind for row rind of W.

    :param rind: The row index of W for which to compute alpha.
    :type rind: int
    :param W: The cone matrix.
    :type W: numpy.ndarray
    :return: The computed alpha value for the specified row of W.
    :rtype: numpy.ndarray
    """
    m = W.shape[0] + 1  # number of constraints
    D = W.shape[1]
    f = -W[rind, :]
    A = []
    b = []
    c = []
    d = []
    # x inside the cone
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i, :])
        d.append(np.zeros(1))

    # norm of x is less than or equal to 1
    A.append(np.eye(D))
    b.append(np.zeros(D))
    c.append(np.zeros(D))
    d.append(np.ones(1))

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)]
    prob = cp.Problem(cp.Minimize(f.T @ x), soc_constraints)
    prob.solve()

    return -prob.value


def get_alpha_vec(W: np.ndarray) -> np.ndarray:
    """
    The alpha vector is computed using the `get_alpha` function for each row index.

    :param W: The cone matrix.
    :type W: numpy.ndarray
    :return: An ndarray of shape (n_constraint, 1) representing the computed alpha vector.
    :rtype: numpy.ndarray
    """
    alpha_vec = np.zeros((W.shape[0], 1))
    for i in range(W.shape[0]):
        alpha_vec[i] = get_alpha(i, W)
    return alpha_vec


def get_closest_indices_from_points(
    pts_to_find: np.ndarray,
    pts_to_check: np.ndarray,
    return_distances: bool = False,
    squared: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    This method calculates the closest indices in `pts_to_check` for each point in `pts_to_find`
    using Euclidean distances. Optionally, it can return the distances as well.

    :param pts_to_find: An array of points for which the closest indices need to be found.
    :type pts_to_find: np.ndarray
    :param pts_to_check: An array of points to check against.
    :type pts_to_check: np.ndarray
    :param return_distances: If True, the method also returns the distances to the closest points.
    :type return_distances: bool, optional
    :param squared: If True, the squared Euclidean distances are used and returned.
    :type squared: bool, optional
    :return: An array of the closest indices, or a tuple containing the closest indices and the
        distances.
    :rtype: Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
    """
    if len(pts_to_find) == 0 or len(pts_to_check) == 0:
        return []

    distances = euclidean_distances(pts_to_find, pts_to_check, squared=squared)
    x_inds = np.argmin(distances, axis=1)
    if return_distances:
        return x_inds.astype(int), np.min(distances, axis=1)
    return x_inds.astype(int)


def get_noisy_evaluations_chol(means: np.ndarray, cholesky_cov: np.ndarray) -> np.ndarray:
    """
    This method generates noisy samples from a multivariate normal distribution using the provided
    means and Cholesky decomposition of the covariance matrix. It is used for vectorized
    multivariate normal sampling.

    :param means: An array of mean values for the multivariate normal distribution.
    :type means: np.ndarray
    :param cholesky_cov: The Cholesky decomposition of the covariance matrix, this is a 2D array.
    :type cholesky_cov: np.ndarray
    :return: An array of noisy samples.
    :rtype: np.ndarray
    """
    if cholesky_cov.ndim != 2 or means.shape[1] != cholesky_cov.shape[1]:
        raise AssertionError("Invalid dimensions.")
    n, d = means.shape[0], len(cholesky_cov)
    X = np.random.normal(size=(n, d))
    complicated_X = np.dot(X, cholesky_cov)

    noisy_samples = means + complicated_X

    return noisy_samples


def generate_sobol_samples(dim: int, n: int) -> np.ndarray:
    """
    This method generates `n` samples from a Sobol sequence of dimension `dim`. `n` should be a
    power of 2 in order to generate a balanced sequence. Returned samples are in the range [0, 1).

    :param dim: The dimension of the Sobol sequence.
    :type dim: int
    :param n: The number of samples to generate.
    :type n: int
    :return: An array of Sobol sequence samples.
    :rtype: np.ndarray
    """
    local_generator = np.random.default_rng(np.random.randint(0, 1000))
    sampler = Sobol(dim, scramble=True, seed=local_generator)
    samples = sampler.random(n)

    return samples


def get_bigmij(vi: np.ndarray, vj: np.ndarray, W: np.ndarray) -> float:
    """
    This method calculates the M(i,j) value, which is used to measure the difference between two
    designs `vi` and `vj` based on the constraint matrix `W`.

    :param vi: A D-vector representing the design vector i.
    :type vi: np.ndarray
    :param vj: A D-vector representing the design vector j.
    :type vj: np.ndarray
    :param W: A (n_constraint, D) ndarray representing the constraint matrix.
    :type W: np.ndarray
    :return: The computed M(i,j) value.
    :rtype: float
    """
    m = W.shape[1]

    x = cp.Variable(m)
    prob = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(x)), constraints=[W @ x >= 0, W @ (vj + x - vi) >= 0]
    )

    try:
        prob.solve()
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)

    bigmij = np.sqrt(prob.value)

    return bigmij


def get_smallmij(vi: np.ndarray, vj: np.ndarray, W: np.ndarray, alpha_vec: np.ndarray) -> float:
    """
    This method calculates the m(i,j) value, which is used to measure the difference between two
    designs `vi` and `vj` based on the constraint matrix `W` and the alpha vector `alpha_vec`.

    :param vi: A D-vector representing the design vector i.
    :type vi: np.ndarray
    :param vj: A D-vector representing the design vector j.
    :type vj: np.ndarray
    :param W: A (n_constraint, D) ndarray representing the constraint matrix.
    :type W: np.ndarray
    :param alpha_vec: A (n_constraint, 1) ndarray representing the alphas of W.
    :type alpha_vec: np.ndarray
    :return: The computed m(i,j) value.
    :rtype: float
    """
    prod = np.matmul(W, vj - vi)
    prod[prod < 0] = 0
    smallmij = (prod / alpha_vec).min()

    return smallmij


def get_delta(mu: np.ndarray, W: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    r"""
    This method computes :math:`\Delta^*_i` gap value for each point in the input array `mu`.
    :math:`\Delta^*_i` is calculated based on the provided constraint matrix `W` and the alpha
    vector `alpha_vec`.

    :param mu: An array of shape (n_points, D) representing the points.
    :type mu: np.ndarray
    :param W: An array of shape (n_constraint, D) representing the constraint matrix.
    :type W: np.ndarray
    :param alpha_vec: An array of shape (n_constraint, 1) representing the alphas of W.
    :type alpha_vec: np.ndarray
    :return: An array of shape (n_points, 1) containing :math:`\Delta^*_i` for each point.
    :rtype: np.ndarray
    """
    num_points = mu.shape[0]
    delta_values = np.zeros(num_points)
    for i in range(num_points):
        for j in range(num_points):
            vi = mu[i, :]
            vj = mu[j, :]
            mij = get_smallmij(vi, vj, W, alpha_vec)
            delta_values[i] = max(delta_values[i], mij)

    return delta_values.reshape(-1, 1)


def get_uncovered_set(
    p_inds: Iterable, p_hat_inds: Iterable, mu: np.ndarray, eps: float, W: np.ndarray
) -> list:
    """
    Identify the set of elements in `p_inds` that are not covered by any element in `p_hat_inds`.

    This function checks each element in `p_inds` to see if it is covered by any element
    in `p_hat_inds` based on the provided `mu`, `eps`, and `W` parameters.

    :param p_inds: Array of indices representing the Pareto elements to check for coverage.
    :type p_inds: Iterable
    :param p_hat_inds: Array of indices representing the estimated Pareto elements that may cover
        elements in `p_inds`.
    :type p_hat_inds: Iterable
    :param mu: (N, D) array where each row corresponds to a designs D-dim feature vector.
    :type mu: np.ndarray
    :param eps: Coverage slackness.
    :type eps: float
    :param W: An (n_constraint, D) array representing the cone matrix.
    :type W: np.ndarray
    :return: List of indices from `p_inds` that are not covered by any element in `p_hat_inds`.
    :rtype: list
    """
    uncovered_set = []

    for i in p_inds:
        for j in p_hat_inds:
            if is_covered(mu[i, :], mu[j, :], eps, W):
                break
        else:
            uncovered_set.append(i)

    return uncovered_set


def get_uncovered_size(
    pareto_pts: np.ndarray, pareto_hat_pts: np.ndarray, eps: float, W: np.ndarray
) -> int:
    """
    Identify the set of elements in `pareto_pts` that are not covered by any element in
    `pareto_hat_pts`.

    This function checks each element in `pareto_pts` to see if it is covered by any element
    in `pareto_hat_pts` based on the provided `mu`, `eps`, and `W` parameters.

    :param pareto_pts: An (N_pareto, D) array of Pareto elements to check for coverage.
    :type pareto_pts: np.ndarray
    :param pareto_hat_pts: An (N_pareto_hat, D) array of estimated Pareto elements to check if
        they cover any element of `pareto_pts`.
    :type pareto_hat_pts: np.ndarray
    :param eps: Coverage slackness.
    :type eps: float
    :param W: An (n_constraint, D) array representing the cone matrix.
    :type W: np.ndarray
    :return: Number of points from `pareto_pts` that are not covered by any element in
        `pareto_hat_pts`.
    :rtype: int
    """
    count = 0

    for i, ip in enumerate(pareto_pts):
        for jp in pareto_hat_pts:
            if is_covered(ip, jp, eps, W):
                break
        else:
            count += 1

    return count


def is_covered(vi: np.ndarray, vj: np.ndarray, eps: float, W: np.ndarray) -> bool:
    """
    Check if vector `vi` is epsilon-covered by vector `vj` for a given cone matrix `W`.

    This function determines if the vector `vi` can be weakly dominated by `vj` with an epsilon
    length vector chosen from the cone.

    :param vi: A (D,) array representing the vector to be checked.
    :type vi: np.ndarray
    :param vj: A (D,) array representing the reference vector.
    :type vj: np.ndarray
    :param eps: A float representing the slackness.
    :type eps: float
    :param W: An (n_constraint, D) array representing the cone matrix.
    :type W: np.ndarray
    :return: True if `vi` is epsilon-covered by `vj` under the constraints of `W`, False otherwise.
    :rtype: bool
    """
    x = cp.Variable(W.shape[1])

    constraints = [
        W @ x >= 0,
        cp.norm(x + (vi - vj)) <= eps,
        W @ (x + (vi - vj)) >= 0,
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    return x.value is not None


def hyperrectangle_check_intersection(
    lower1: np.ndarray, upper1: np.ndarray, lower2: np.ndarray, upper2: np.ndarray
) -> bool:
    """
    This function takes the lower and upper bounds of two hyperrectangles and
    determines if they intersect. A hyperrectangle is defined by its lower and
    upper points in an n-dimensional space.

    :param lower1: Lower bounds of the first hyperrectangle.
    :type lower1: np.ndarray
    :param upper1: Upper bounds of the first hyperrectangle.
    :type upper1: np.ndarray
    :param lower2: Lower bounds of the second hyperrectangle.
    :type lower2: np.ndarray
    :param upper2: Upper bounds of the second hyperrectangle.
    :type upper2: np.ndarray
    :return: True if the hyperrectangles intersect, False otherwise.
    :rtype: bool
    """
    if np.any(lower1 >= upper2) or np.any(upper1 <= lower2):
        return False

    return True


def hyperrectangle_get_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    This method takes an n-dimensional lower bound array and an n-dimensional upper bound array,
    and constructs the vertices of the corresponding hyperrectangle by combining these bounds.

    :param lower: An array of shape (n,) representing the lower bounds of the hyperrectangle.
    :type lower: np.ndarray
    :param upper: An array of shape (n,) representing the upper bounds of the hyperrectangle.
    :type upper: np.ndarray
    :return: An array of shape (2^n, n) containing the vertices of the hyperrectangle.
    :rtype: np.ndarray
    """
    dim_bounds = [[dim_low, dim_high] for dim_low, dim_high in zip(lower, upper)]
    vertex_list = [vertex for vertex in itertools.product(*dim_bounds)]
    return np.array(vertex_list)


def hyperrectangle_get_region_matrix(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    This method takes an n-dimensional lower bound array and an n-dimensional upper bound array,
    and constructs a matrix form for the hyperrectangle. For all points `x` inside the region of
    the hyperrectangle, the condition `region_matrix @ x >= region_boundary` holds true.

    :param lower: An array of shape (n,) representing the lower bounds of the hyperrectangle.
    :type lower: np.ndarray
    :param upper: An array of shape (n,) representing the upper bounds of the hyperrectangle.
    :type upper: np.ndarray
    :return: A tuple containing two elements:
        - region_matrix: An array of shape (2*n, n) representing the matrix form of the
        hyperrectangle.
        - region_boundary: An array of shape (2*n,) representing the boundary conditions of the
        hyperrectangle.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    dim = len(lower)
    region_matrix = np.vstack((np.eye(dim), -np.eye(dim)))
    region_boundary = np.hstack((np.array(lower), -np.array(upper)))

    return region_matrix, region_boundary


def is_pt_in_extended_polytope(
    pt: np.ndarray, polytope: np.ndarray, invert_extension: bool = False
):
    """
    Check if `pt` is an element of the extended polytope with vertices represented with `polytope`.

    This method checks if a point `pt` is an element of the polytope defined by the vertices in
    `polytope` and extended along to infinity along the axes. This corresponds to the Minkowski
    addition of a hyperrectangle with a right-angled cone. The `invert_extension` parameter can be
    used to invert the extension to negative infinity, *i.e.*, Minkowski addition with negative of
    the right angle.

    :param pt: A (D, ) array for the point to check.
    :type pt: np.ndarray
    :param polytope: An (N_vertices, D) array of vertices defining the polytope.
    :type polytope: np.ndarray
    :param invert_extension: If True, the extension is inverted.
    :type invert_extension: bool
    :return: True if the point is an element of the extended polytope, False otherwise.
    :rtype: bool
    """
    dim = polytope.shape[1]

    if invert_extension:

        def comp_func(x, y):
            return x >= y

    else:

        def comp_func(x, y):
            return x <= y

    # Vertex is trivially an element
    for vert in polytope:
        if comp_func(vert, pt).all():
            return True

    # Check intersections with polytope. If any intersection is dominated, then an element.
    for dim_i in range(dim):
        edges_of_interest = np.empty((0, 2, dim), dtype=np.float64)
        for vert_i, vert1 in enumerate(polytope):
            for vert_j, vert2 in enumerate(polytope):
                if vert_i == vert_j:
                    continue

                if vert1[dim_i] <= pt[dim_i] and pt[dim_i] <= vert2[dim_i]:
                    edges_of_interest = np.vstack(
                        (edges_of_interest, np.expand_dims(np.vstack((vert1, vert2)), axis=0))
                    )

        for edge in edges_of_interest:
            intersection = line_seg_pt_intersect_at_dim(edge[0], edge[1], pt, dim_i)
            if intersection is not None and comp_func(intersection, pt).all():
                # Vertex is an element due to the intersection point
                return True

    return False


def line_seg_pt_intersect_at_dim(
    P1: np.ndarray, P2: np.ndarray, target_pt: np.ndarray, target_dim: int
) -> Optional[np.ndarray]:
    """
    Check if a line segment intersects with a point at a specific dimension.

    This function determines if the line segment defined by points P1 and P2
    intersects with the point `target_pt` when only the dimension `target_dim`
    is considered. If there is an intersection, it returns the point on the line
    segment where the intersection occurs. Otherwise, it returns None.

    :param P1: A (D, ) array for the first endpoint of the line segment.
    :type P1: np.ndarray
    :param P2: A (D, ) array for the second endpoint of the line segment.
    :type P2: np.ndarray
    :param target_pt: A (D, ) array for the target point to check intersection.
    :type target_pt: np.ndarray
    :param target_dim: The dimension to consider for the intersection check.
    :type target_dim: int
    :return: The point on the line segment where the intersection occurs, or None if there is no
        intersection.
    :rtype: Optional[np.ndarray]
    """
    t = (target_pt[target_dim] - P1[target_dim]) / (P2[target_dim] - P1[target_dim])

    if t < 0 or t > 1:
        # No intersection
        return None

    point_on_line = P1 + t * (P2 - P1)
    return point_on_line


def normalize(data: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Normalize the data based on the provided bounds. Each column of the data array is normalized
    based on the corresponding bounds.

    :param data: The data array to normalize.
    :type data: np.ndarray
    :param bounds: A list of tuples containing the lower and upper bounds for each column.
    :type bounds: list[tuple[float, float]]
    :return: The normalized data array.
    :rtype: np.ndarray
    """

    if len(bounds) != data.shape[1]:
        raise ValueError("Bounds must have the same number of elements as the data columns.")

    normalized_data = np.empty_like(data)
    for i, (lower, upper) in enumerate(bounds):
        normalized_data[:, i] = (data[:, i] - lower) / (upper - lower)

    return normalized_data


def unnormalize(data: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Unnormalize the data based on the provided bounds. Each column of the data array is unnormalized
    based on the corresponding bounds.

    :param data: The data array to unnormalize.
    :type data: np.ndarray
    :param bounds: A list of tuples containing the lower and upper bounds for each column.
    :type bounds: list[tuple[float, float]]
    :return: The unnormalized data array.
    :rtype: np.ndarray
    """

    if len(bounds) != data.shape[1]:
        raise ValueError("Bounds must have the same number of elements as the data columns.")

    unnormalized_data = np.empty_like(data)
    for i, (lower, upper) in enumerate(bounds):
        unnormalized_data[:, i] = data[:, i] * (upper - lower) + lower

    return unnormalized_data


def binary_entropy(x: np.ndarray) -> np.ndarray:
    """
    Calculate the binary entropy of a given probability.

    This method computes the binary entropy for each element in the input array `x`.
    Binary entropy is a measure of the uncertainty associated with a Bernoulli random variable.

    :param x: An array of probabilities.
    :type x: np.ndarray
    :return: An array of binary entropy values corresponding to the input probabilities.
    :rtype: np.ndarray
    """
    return -(scipy.special.xlogy(x, x) + scipy.special.xlog1py(1 - x, -x)) / np.log(2)

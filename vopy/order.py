from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from vopy.ordering_cone import ConeTheta2D, OrderingCone

from vopy.utils.plotting import plot_pareto_front


class Order(ABC):
    """
    Abstract base class for defining an ordering relation between points in a space. Any deriving
    class must implement the :meth:`dominates` that induces the preorder.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def dominates(self) -> bool:
        pass


class PolyhedralConeOrder(Order):
    """
    Base class for defining an ordering relation using a specified polyhedral ordering cone.

    :param ordering_cone: An instance of :obj:`OrderingCone` that defines the ordering relation.
    :type ordering_cone: OrderingCone
    """

    def __init__(self, ordering_cone: OrderingCone):
        self.ordering_cone = ordering_cone

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Determines if point :obj:`a` dominates point :obj:`b` according to the ordering cone.

        :param a: The vector representing the point to check for dominance.
        :type a: np.ndarray
        :param b: The vector representing the point to check if dominated.
        :type b: np.ndarray
        :return: `True` if :obj:`a` dominates :obj:`b` according to the order; otherwise, `False`.
        :rtype: bool
        """
        return self.ordering_cone.is_inside(a - b)

    def get_pareto_set(self, elements: np.ndarray) -> np.ndarray:
        """
        Computes the Pareto set from a set of elements, retaining only non-dominated points.
        See: https://stackoverflow.com/a/40239615

        :param elements: An array of shape (N, dim), where `N` is the number of elements
            and `dim` is the dimension of the elements and the ordering cone.
        :type elements: np.ndarray
        :return: Indices of the elements that belong to the Pareto set.
        :rtype: np.ndarray
        :raises ValueError: If :obj:`elements` is not a 2D array.
        """
        if elements.ndim != 2:
            raise ValueError("Elements array should be N-by-dim.")

        is_pareto = np.arange(len(elements))

        # Next index in the is_pareto array to search for
        next_point_index = 0
        while next_point_index < len(elements):
            nondominated_point_mask = np.zeros(len(elements), dtype=bool)
            vj = elements[next_point_index]

            for i in range(len(elements)):
                vi = elements[i]
                nondominated_point_mask[i] = not self.dominates(vj, vi)

            nondominated_point_mask[next_point_index] = True

            # Remove dominated points
            is_pareto = is_pareto[nondominated_point_mask]
            elements = elements[nondominated_point_mask]

            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        return is_pareto

    def get_pareto_set_naive(self, elements: np.ndarray) -> np.ndarray:
        """
        Computes the Pareto set using a naive method by iterating over each element
        and checking if it is dominated by any other.

        :param elements: An array of shape (N, dim), where `N` is the number of elements
            and `dim` is the dimension of the elements and the ordering cone.
        :type elements: np.ndarray
        :return: Indices of the elements that belong to the Pareto set.
        :rtype: np.ndarray
        :raises ValueError: If :obj:`elements` is not a 2D array.
        """
        if elements.ndim != 2:
            raise ValueError("Elements array should be N-by-dim.")

        pareto_indices = []
        for el_i, el in enumerate(elements):
            for other_el in elements:
                if np.allclose(el, other_el):
                    continue

                if self.dominates(other_el, el):
                    break
            else:
                pareto_indices.append(el_i)

        return np.array(pareto_indices, dtype=int)

    def plot_pareto_set(
        self, elements: np.ndarray, path: Optional[Union[str, PathLike]] = None, show: bool = True
    ) -> plt.Figure:
        """
        Plots the Pareto front of the provided elements if the dimension is 2D or 3D.

        :param elements: An array of shape (N, dim), where `N` is the number of elements
            and `dim` is the dimension of the elements and the ordering cone.
        :type elements: np.ndarray
        :param path: The file path where the plot will be saved. If not provided,
            the plot will only be displayed. Default is `None`.
        :type path: Optional[Union[str, PathLike]]
        :param show: Whether to show the plot or close it. If False, the figure is closed.
        :type show: bool
        :return: The matplotlib figure containing the plot.
        :rtype: plt.Figure
        :raises ValueError: If :obj:`elements` is not a 2D array
            or has a dimension other than 2 or 3.
        """
        if elements.ndim != 2:
            raise ValueError("Elements array should be N-by-dim.")
        if elements.shape[1] not in [2, 3]:
            raise ValueError("Only 2D and 3D plots are supported.")

        fig = plot_pareto_front(elements, self.get_pareto_set(elements), path, show)

        return fig


class ComponentwiseOrder(PolyhedralConeOrder):
    """
    Component-wise ordering class that defines an ordering relation where each
    dimension is considered independently. Vector optimization with this order corresponds to the
    multi-objective optimization.

    :param dim: The dimension of the space in which the ordering relation is defined.
    :type dim: int
    """

    def __init__(self, dim: int):
        W = np.eye(dim)
        ordering_cone = OrderingCone(W)

        super().__init__(ordering_cone)


class ConeTheta2DOrder(PolyhedralConeOrder):
    """
    Defines an ordering relation in 2D using a cone with a specified opening angle. The ordering
    cone is an instance of :class:`vopy.ordering_cone.ConeTheta2D`.

    :param cone_degree: The opening angle of the cone in degrees.
    :type cone_degree: float
    """

    def __init__(self, cone_degree) -> None:
        self.cone_degree = cone_degree
        ordering_cone = ConeTheta2D(cone_degree)

        super().__init__(ordering_cone)


class ConeOrder3D(PolyhedralConeOrder):
    """
    Defines a 3D ordering relation using a specified cone type. The class supports
    three predefined cone types—'acute', 'right', and 'obtuse'—each with its
    unique constraint matrix, as used in [Karagozlu2024]_.

    :param cone_type: The type of cone to use for ordering. Must be one of
        'acute', 'right', or 'obtuse'.
    :type cone_type: str

    :raises ValueError: If :obj:`cone_type` is not one of the allowed values.

    References:
        .. [Karagozlu2024]
            Karagözlü, Yıldırım, Ararat, Tekin.
            Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector
            Bandits.
            Artificial Intelligence and Statistics (AISTATS), 2024.
    """

    def __init__(self, cone_type: str):
        if cone_type == "acute":
            W = np.array(
                [
                    [1.0, -2, 4],
                    [4, 1.0, -2],
                    [-2, 4, 1.0],
                ]
            )
            norm = np.linalg.norm(W[0])
            W /= norm
        elif cone_type == "right":
            W = np.eye(3)
        elif cone_type == "obtuse":
            W = np.array(
                [
                    [1, 0.4, 1.6],
                    [1.6, 1, 0.4],
                    [0.4, 1.6, 1],
                ]
            )
            norm = np.linalg.norm(W[0])
            W /= norm
        else:
            raise ValueError("cone_type must be one of 'acute', 'right', or 'obtuse'.")
        ordering_cone = OrderingCone(W)

        super().__init__(ordering_cone)


class ConeOrder3DIceCream(PolyhedralConeOrder):
    """
    Defines a 3D ordering relation approximating the shape of an ice cream cone with opening
    defined by :obj:`cone_angle`. The ordering cone is constructed with equally rotated
    half-spaces based on the specified number of half-spaces.

    :param cone_degree: The opening angle of the cone in degrees.
    :type cone_degree: float
    :param num_halfspace: The number of half-spaces used to approximate the cone.
    :type num_halfspace: int
    """

    def __init__(self, cone_degree: float, num_halfspace: int):
        W = self.compute_ice_cream_cone(num_halfspace, cone_degree)
        ordering_cone = OrderingCone(W)

        super().__init__(ordering_cone)

    def compute_ice_cream_cone(self, K: int, theta: float) -> np.ndarray:
        r"""
        Computes the constraint matrix `W` for the ice cream cone approximation.
        The cone is constructed by rotating half-spaces around a central axis.

        :param K: The number of half-spaces used to approximate the cone.
        :type K: int
        :param theta: The opening angle of the cone in degrees.
        :type theta: float
        :return: A 2D array representing the normal vectors for each half-space, *i.e.*, a cone
            matrix :math:`\mathbf{W}` that approximates the ice cream cone.
        :rtype: np.ndarray
        """
        delta_angle = 2 * np.pi / K
        theta_rad = np.pi / 2 - np.radians(theta)

        radius = np.tan(theta_rad)
        W = []
        for i in range(K):
            angle = i * delta_angle
            rotated_ny = radius * np.sin(angle)
            rotated_nx = radius * np.cos(angle)
            W.append([rotated_nx, rotated_ny, 1])
        W = np.array(W)

        rot_axis = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        rot_rad = np.pi / 4
        C = np.array(
            [
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0],
            ]
        )

        r = np.eye(3) + C * np.sin(rot_rad) + (C @ C) * (1 - np.cos(rot_rad))
        W = (r @ W.T).T

        # Normalize half plane normal vectors
        for i in range(K):
            W[i] = W[i] / np.linalg.norm(W[i])

        return W

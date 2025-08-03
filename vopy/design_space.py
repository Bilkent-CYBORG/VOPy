from abc import ABC, abstractmethod
from itertools import product
from os import PathLike
from typing import Literal, Optional, Union

import numpy as np

from vopy.confidence_region import (
    ConfidenceRegion,
    EllipsoidalConfidenceRegion,
    RectangularConfidenceRegion,
)
from vopy.models.model import GPModel, Model
from vopy.utils import get_closest_indices_from_points
from vopy.utils.plotting import plot_cells_with_centers


class DesignSpace(ABC):
    """
    Abstract base class for design spaces.

    This class defines the interface for design spaces, which are used to represent the space of
    possible designs in an optimization problem. Subclasses must implement the `update` method
    to update the design space based on a given model.
    """

    def __init__(self):
        pass

    @abstractmethod
    def update(self, model: Model):
        """
        Update the design space based on the given model.

        :param model: The model used to update the design space.
        :type model: Model
        """
        pass


class DiscreteDesignSpace(DesignSpace):
    """
    Represents a design space that consists of dicrete points.

    This class is an abstract implementation of the `DesignSpace` abstract base class. It
    represents a design space where the points are discrete. The class also maintains a list of
    confidence regions associated with the design points.

    A derived class must define the following attributes:

    - :obj:`points`: :type:`np.ndarray`
    - :obj:`confidence_regions`: :type:`list[ConfidenceRegion]`
    """

    points: np.ndarray
    confidence_regions: list[ConfidenceRegion]

    def __init__(self):
        super().__init__()

    def locate_points(self, x: np.ndarray, atol: float = 1e-6) -> list[int]:
        """
        Find indices of points given as `x` in the design space.

        This method finds the indices of the points given as `x` in the design space. Instead
        of exact equality, the method uses `np.allclose` for tolerated comparisons. If any of the
        distances to the points found is larger than `atol`, an error is raised.

        :param x: An array of points to locate in the design space.
        :type x: np.ndarray
        :param atol: The absolute tolerance parameter, defaults to :math:`1e-6`.
        :type atol: float
        :return: A list of indices representing the positions of the points in the design space.
        :rtype: list[int]
        :raises ValueError: If any of the distances to the points found is larger than the
            specified tolerance.
        """
        indices, distances = get_closest_indices_from_points(x, self.points, return_distances=True)
        if distances.max() > atol:
            raise ValueError("Some points are not in the design space.")

        return indices


class FixedPointsDesignSpace(DiscreteDesignSpace):
    """
    Represents a design space that has fixed number points.

    This class is a concrete implementation of the `DiscreteDesignSpace` abstract class. It
    represents a design space where the points are fixed and does not get updated.

    :param points: An array representing the points in the design space.
    :type points: np.ndarray
    :param objective_dim: The dimension of the objective space.
    :type objective_dim: int
    :param confidence_type: The type of confidence region to use. Can be "hyperrectangle"
        or "hyperellipsoid", defaults to "hyperrectangle".
    :type confidence_type: str
    :raises NotImplementedError: If an unsupported confidence type is provided.
    """

    def __init__(
        self,
        points: np.ndarray,
        objective_dim: int,
        confidence_type: Literal["hyperrectangle", "hyperellipsoid"] = "hyperrectangle",
    ) -> None:
        super().__init__()

        if confidence_type == "hyperrectangle":
            confidence_cls = RectangularConfidenceRegion
        elif confidence_type == "hyperellipsoid":
            confidence_cls = EllipsoidalConfidenceRegion
        else:
            raise NotImplementedError(f"Unsupported confidence type {confidence_type}.")

        self.points = points
        self.confidence_regions = []
        for _ in range(len(points)):
            self.confidence_regions.append(confidence_cls(objective_dim))

        self.cardinality = len(self.points)

    def update(self, model: Model, scale: np.ndarray, indices_to_update: Optional[list] = None):
        """
        Update the confidence regions based on the given model and scale.

        This method updates the confidence regions for the specified points in the design space
        based on the predictions from the given model and the provided scale.

        :param model: The model used to predict the means and covariances for the points.
        :type model: Model
        :param scale: An array representing the scale for each objective. Can be a scalar, a
            vector with size output dim, or a 2D array with shape (N_indices, output dim).
        :type scale: np.ndarray
        :param indices_to_update: A list of indices of the points to update. If None, all points
            are updated. Defaults to None.
        :type indices_to_update: Optional[list]
        :raises AssertionError: If the shape of the scale array is invalid.
        """
        if indices_to_update is None:
            indices_to_update = list(range(self.cardinality))

        # If scale is a scalar, it is broadcasted to the length of indices_to_update.
        # Note that scale can have different values for each objective.
        if scale.ndim < 2:
            scale = np.repeat(np.atleast_1d(scale)[None, :], len(indices_to_update), axis=0)
        elif scale.ndim != 2 or len(scale) != len(indices_to_update):
            raise ValueError("Invalid scale shape.")

        mus, covs = model.predict(self.points[indices_to_update])
        for pt_i, mu, cov, s in zip(indices_to_update, mus, covs, scale):
            self.confidence_regions[pt_i].update(mu, cov, s)


class AdaptivelyDiscretizedDesignSpace(DiscreteDesignSpace):
    """
    Represents an adaptively discretized design space.

    This class is a concrete implementation of the `DiscreteDesignSpace` class. It represents a
    design space where the domain is adaptively discretized based on a given model. The class
    maintains a list of points in a tree like structure as a representation of the design space,
    and allows for refinement of the design space based on the model's predictions. Note that
    the design space is assumed to be the unit hypercube.

    :param domain_dim: The dimension of the input space.
    :type domain_dim: int
    :param objective_dim: The dimension of the objective space.
    :type objective_dim: int
    :param delta: Determines confidence level for the confidence regions.
    :type delta: float
    :param max_depth: The maximum depth for the adaptive discretization. Number of points increases
        exponentially with depth.
    :type max_depth: int
    :param confidence_type: The type of confidence region to use, defaults to "hyperrectangle".
    :type confidence_type: Literal["hyperrectangle"]
    :raises NotImplementedError: If an unsupported confidence type is provided. Currently, only the
        "hyperrectangle" confidence type is supported.
    """

    def __init__(
        self,
        domain_dim: int,
        objective_dim: int,
        delta: float,
        max_depth: int,
        confidence_type: Literal["hyperrectangle"] = "hyperrectangle",
    ) -> None:
        super().__init__()

        if confidence_type == "hyperrectangle":
            confidence_cls = RectangularConfidenceRegion
        else:
            raise NotImplementedError(
                "Only hyperrectangular confidence region is supported for adaptive discretization."
            )

        self.domain_dim = domain_dim
        self.objective_dim = objective_dim
        self.max_depth = max_depth
        self.delta = delta
        self.confidence_cls = confidence_cls

        # TODO: Consider creating a Design class.
        # If parent design becomes necessary, a reference can be held as an attribute.
        self.points = np.array([[0.5] * domain_dim])
        self.point_depths = [1]
        self.cells = [[[0, 1] for _ in range(domain_dim)]]
        self.confidence_regions = [confidence_cls(objective_dim)]

        self.cardinality = len(self.points)

    def update(self, model: GPModel, scale: np.ndarray, indices_to_update: Optional[list] = None):
        """
        Update the confidence regions based on the given model and scale.

        This method updates the confidence regions for the specified points in the design space
        based on the predictions from the given model and the provided scale.

        :param model: The model used to predict the means and covariances for the points.
        :type model: GPModel
        :param scale: An array representing the scale for each objective. Can be a scalar, a
            vector with size output dim, or a 2D array with shape (N_indices, output dim).
        :type scale: np.ndarray
        :param indices_to_update: A list of indices of the points to update. If None, all points
            are updated. Defaults to None.
        :type indices_to_update: Optional[list]
        :raises AssertionError: If the shape of the scale array is invalid.
        """
        if indices_to_update is None:
            indices_to_update = list(range(len(self.points)))

        # If scale is a scalar, it is broadcasted to the length of indices_to_update.
        # Note that scale can have different values for each objective.
        if scale.ndim < 2:
            scale = np.repeat(np.atleast_1d(scale)[None, :], len(indices_to_update), axis=0)
        elif scale.ndim != 2 or len(scale) != len(indices_to_update):
            raise ValueError("Invalid scale shape.")

        mus, covs = model.predict(self.points[indices_to_update])
        for pt_i, mu, cov, s in zip(indices_to_update, mus, covs, scale):
            self.confidence_regions[pt_i].update(mu, cov, s)

    def refine_design(self, index_to_refine: int) -> list:
        """
        Refine the design space by generating child designs for the point specified by its
        index `index_to_refine`.

        This method generates child designs for the specified index in the design space,
        effectively refining the design space.

        :param index_to_refine: The index of the design to refine.
        :type index_to_refine: int
        :return: A list of indices representing the new child designs.
        :rtype: list
        """
        return self.generate_child_designs(index_to_refine)

    def generate_child_designs(self, design_index: int) -> list:
        """
        Generate child designs for the specified design index.

        This method generates child designs for the specified design index by splitting the design
        along each dimension and creating new designs at the midpoints.

        :param design_index: The index of the design to generate child designs for.
        :type design_index: int
        :return: A list of indices representing the new child designs.
        :rtype: list
        """
        options = []
        for dim_i in range(self.domain_dim):
            options.append(
                [
                    [
                        self.cells[design_index][dim_i][0],
                        (self.cells[design_index][dim_i][0] + self.cells[design_index][dim_i][1])
                        / 2,
                    ],
                    [
                        (self.cells[design_index][dim_i][0] + self.cells[design_index][dim_i][1])
                        / 2,
                        self.cells[design_index][dim_i][1],
                    ],
                ]
            )
        new_bounds = list(map(list, product(*options)))

        list_children = []
        for bound in new_bounds:
            list_children.append(len(self.points))

            x = np.array(bound, dtype=float).mean(axis=1)
            self.points = np.append(self.points, [x], axis=0)
            self.point_depths.append(self.point_depths[design_index] + 1)
            self.cells.append(bound)
            self.confidence_regions.append(
                # TODO: Create a copy constructor for ConfidenceRegion
                self.confidence_cls(
                    self.objective_dim,
                    self.confidence_regions[design_index].lower,
                    self.confidence_regions[design_index].upper,
                )
            )

        self.cardinality = len(self.points)

        return list_children

    def should_refine_design(self, model: GPModel, design_index: int, scale: np.ndarray) -> bool:
        """
        Determine whether the design at the specified index should be refined.

        This method determines whether the design at the specified index should be refined
        based on the model's predictions and the provided scale.

        :param model: The model used to predict the means and covariances for the points.
        :type model: GPModel
        :param design_index: The index of the design to check for refinement.
        :type design_index: int
        :param scale: An array representing the scale for each objective. Can be a scalar and a
            vector with size output dim.
        :type scale: np.ndarray
        :return: True if the design should be refined, False otherwise.
        :rtype: bool
        """
        if self.point_depths[design_index] >= self.max_depth:
            return False

        vh = self.calculate_design_vh(model, design_index)
        mu, cov = model.predict(self.points[[design_index]])
        std = np.sqrt(np.diag(cov.squeeze()))

        return np.all(scale * np.linalg.norm(std) <= np.linalg.norm(vh))

    def calculate_design_vh(
        self, model: GPModel, design_index: int, depth_offset: int = 0
    ) -> np.ndarray:
        """
        Calculate the Vh value for the design at the specified index.

        This method calculates the Vh value for the design at the specified index based on the
        model's predictions and the depth offset. Vh value is a measure of the uncertainty for
        the design point depending on its depth.

        :param model: The model used to predict the means and covariances for the points.
        :type model: GPModel
        :param design_index: The index of the design to calculate the Vh value for.
        :type design_index: int
        :param depth_offset: The depth offset for the calculation, defaults to 0. If given as -1,
            the Vh value is calculated for the parent design.
        :type depth_offset: int
        :return: The calculated Vh value.
        :rtype: np.ndarray
        :raises ValueError: If a value for the kernel type of the model is not defined.
        """
        # TODO: magic number
        rho = 0.5
        alpha = 1
        N = 4

        lengthscales, variances = model.get_lengthscale_and_var()
        Vh = np.zeros([self.objective_dim, 1])
        depth = self.point_depths[design_index] + depth_offset
        diam_x = np.sqrt(self.domain_dim)
        v1 = 0.5 * np.sqrt(self.domain_dim)

        for i in range(self.objective_dim):
            if model.get_kernel_type() == "RBF":
                Cki = np.sqrt(variances[i]) / lengthscales[i]
            else:
                raise ValueError("Kernel type undefined.")

            C1 = np.power((diam_x + 1) * diam_x / 2, self.domain_dim) * np.power(Cki, 1 / alpha)
            C2 = 2 * np.log(2 * np.power(C1, 2) * np.power(np.pi, 2) / 6)
            C3 = 1.0 + 2.7 * np.sqrt(2 * self.domain_dim * alpha * np.log(2))

            term1 = Cki * np.power(v1 * np.power(rho, depth), alpha)
            term2 = np.log(
                2
                * np.power(depth + 1, 2)
                * np.power(np.pi, 2)
                * self.objective_dim
                / (6 * self.delta)
            )
            term3 = depth * np.log(N)
            term4 = np.maximum(0, -4 * self.domain_dim / alpha * np.log(term1))

            Vh[i] = 4 * term1 * (np.sqrt(C2 + 2 * term2 + term3 + term4) + C3)
        return Vh

    def visualize_design_space(
        self, path: Optional[Union[str, PathLike]] = None, show: bool = True
    ):
        """
        Visualize the design space's current cell structure.

        This method visualizes the cells that are defined by their centers at `self.points` and
        their bounds at `self.cells`. It uses Matplotlib to create a plot of the design space.

        :param path: The path to save the plot to. If not provided, the plot
            will only be displayed. Defaults to `None`.
        :type path: Optional[Union[str, PathLike]]
        :param show: Whether to show the plot or close it. If False, the figure is closed.
        :type show: bool
        :return: The Matplotlib figure object containing the plot.
        :rtype: plt.Figure
        """

        fig = plot_cells_with_centers(self.cells, self.points, path, show)

        return fig

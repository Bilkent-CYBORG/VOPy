from os import PathLike
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from vopy.utils import get_2d_w, get_alpha_vec
from vopy.utils.plotting import plot_2d_cone, plot_2d_theta_cone, plot_3d_cone


class OrderingCone:
    r"""
    Represents a polyhedral ordering cone in the form :math:`C := \{ x | \mathbf{W}x \geq 0 \}`,
    where :math:`\mathbf{W}` is a matrix defining the cone's boundaries. The ordering cone is
    used to check if points lie within the cone by testing if they satisfy the inequality.

    :param W: A 2D array (matrix) that defines the ordering cone. The shape of
        :obj:`W` should be number of constraints by number of dimensions.
    :type W: np.ndarray

    :attr W: The matrix defining the ordering cone.
    :type W: np.ndarray
    :attr dim: The number of dimensions of the ordering cone.
    :type dim: int
    :attr alpha: The alpha vector of the ordering cone.
    :type alpha: np.ndarray

    Examples:
        >>> import numpy as np
        >>> W = np.array([[1, 0], [0, 1]])
        >>> cone = OrderingCone(W)
        >>> x = np.array([0.5, 0.5])
        >>> cone.is_inside(x)
        array([ True])

        >>> x_outside = np.array([-1, 0.5])
        >>> cone.is_inside(x_outside)
        array([False])
    """

    def __init__(self, W: np.ndarray):
        self.W = np.array(W)
        self.dim = self.W.shape[1]
        self.alpha = get_alpha_vec(self.W)

    def __eq__(self, other: "OrderingCone") -> bool:
        """
        Check if this ordering cone is equal to another.

        :param other: The other ordering cone to compare.
        :type other: OrderingCone
        :return: True if the two cones are equivalent (same :obj:`W` matrix), False otherwise.
        :rtype: bool
        """
        if isinstance(other, OrderingCone):
            return np.allclose(self.W, other.W)

        return False

    def is_inside(self, x: np.ndarray) -> np.ndarray:
        r"""
        Determines if a point is or an array of points are inside the ordering cone.

        :param x: A point or an array of points to test. The points should have the same
            dimension as the cone's :math:`\mathbf{W}` matrix.
        :type x: np.ndarray
        :return: A boolean array where `True` indicates that the point is within the cone.
        :rtype: np.ndarray
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x @ self.W.T >= 0).all(axis=-1)

    def plot(
        self, path: Optional[Union[str, PathLike]] = None, show: bool = True, **kwargs
    ) -> plt.Figure:
        """
        Generate a plot of the ordering cone if the cone is 2D or 3D.

        :param path: The file path where the plot will be saved. If not provided, the plot
            will only be displayed. Defaults to `None`.
        :type path: Optional[Union[str, PathLike]]
        :param show: Whether to show the plot or close it. If False, the figure is closed.
        :type show: bool
        :param kwargs: Additional keyword arguments to pass to the plotting function.
        :type kwargs: dict
        :return: The matplotlib figure containing the plot.
        :rtype: plt.Figure

        :raises ValueError: If the dimension of the ordering cone is not 2 or 3.
        """
        if self.dim not in [2, 3]:
            raise ValueError("Only 2D and 3D plots are supported.")

        if self.dim == 2:
            fig = plot_2d_cone(self.is_inside, path, show=show, **kwargs)
        else:
            fig = plot_3d_cone(self.is_inside, path, show=show, **kwargs)

        return fig


class ConeTheta2D(OrderingCone):
    r"""
    Represents a 2D ordering cone defined by a specified degree.

    This class is a specific type of `OrderingCone` and it constructs the `W` matrix based on
    the given cone degree. The ordering cone corresponds to a symmetric cone with its height
    overlapping :math:`y=x`. This class also defines a :math:`\beta` attribute that is given in
    [Ararat2023]_ as ordering complexity of the ordering cone.

    :param cone_degree: The degree of the cone in degrees. The cone is symmetric about the line
        :math:`y=x` and the degree specifies the angle of the cone.
    :type cone_degree: float

    :attr cone_degree: The degree of the cone.
    :type cone_degree: float
    :attr beta: The :math:`\beta` parameter of the cone.
    :type beta: float

    Example:
        >>> cone = ConeTheta2D(45)
        >>> x = np.array([1, 1])
        >>> cone.is_inside(x)
        array([ True])
        >>> cone.beta
        1.4142135623730951

    References:
        .. [Ararat2023]
            Ararat, Tekin.
            Vector Optimization with Stochastic Bandit Feedback.
            Artificial Intelligence and Statistics (AISTATS), 2023.
    """

    def __init__(self, cone_degree: float) -> None:
        self.cone_degree = cone_degree
        W = get_2d_w(cone_degree)

        super().__init__(W)

    @property
    def beta(self) -> float:
        r"""
        Calculate the ordering complexity :math:`\beta` based on the cone's angle in degrees.

        :return: The ordering complexity of the cone.
        :rtype: float
        """
        cone_rad = (self.cone_degree / 180) * np.pi
        if cone_rad < np.pi / 2:
            return 1 / np.sin(cone_rad)
        else:
            return 1.0

    def plot(self, path: Optional[Union[str, PathLike]] = None, show: bool = True):
        """
        Plots the 2D ordering cone based on the specified cone degree.

        :param path: The file path where the plot will be saved. If not provided, the plot
            will only be displayed. Defaults to `None`.
        :type path: Optional[Union[str, PathLike]]
        :param show: Whether to show the plot or close it. If False, the figure is closed.
        :type show: bool
        :return: The matplotlib figure containing the plot.
        :rtype: plt.Figure
        """
        fig = plot_2d_theta_cone(self.cone_degree, path, show)
        return fig

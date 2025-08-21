from os import PathLike
from typing import Callable, Optional, Union

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from vopy.utils import get_2d_w


def plot_2d_theta_cone(
    cone_degree: float, path: Optional[Union[str, PathLike]] = None, show: bool = True
) -> plt.Figure:
    """
    Plot the 2D cone defined by the given cone degree, symmetric around :math:`y=x`.

    This function plots a 2D cone based on the specified cone degree. The plot is created using
    Matplotlib and can be saved to a specified path if provided.

    :param cone_degree: The degree of the cone to be plotted.
    :type cone_degree: float
    :param path: The file path where the plot will be saved. If None, the plot is not saved.
    :type path: Optional[Union[str, PathLike]]
    :param show: Whether to show the plot or close it. If False, the figure is closed.
    :type show: bool
    :return: The Matplotlib figure object containing the plot.
    :rtype: plt.Figure
    """
    xlim = [-5, 5]
    ylim = [-5, 5]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    W = get_2d_w(cone_degree)

    m1 = W[0][0] / -W[0][1] if cone_degree != 90 else 0
    m2 = W[1][0] / -W[1][1]

    # For the x basis
    x_right = np.array([0, xlim[-1]])
    y_right = 0 + m1 * (x_right - 0)

    # For the y basis
    if cone_degree > 90:
        x_left = np.array([xlim[0], 0])
    else:
        x_left = np.array([0, xlim[-1]])
    y_left = 0 + m2 * (x_left - 0)

    if cone_degree > 90:
        verts = np.array(
            [
                [0, 0],
                [x_left[0], y_left[0]],
                [xlim[1], ylim[1]],
                [x_right[-1], y_right[-1]],
            ]
        )
    elif cone_degree == 90:
        verts = np.array(
            [
                [0, 0],
                [0, ylim[1]],
                [xlim[1], ylim[1]],
                [xlim[1], 0],
            ]
        )
    else:
        verts = np.array(
            [
                [0, 0],
                [x_left[-1], y_left[-1]],
                [xlim[1], ylim[1]],
                [x_right[-1], y_right[-1]],
            ]
        )

    ax.add_patch(Polygon(verts, color="blue", alpha=0.5))

    if path is not None:
        fig.savefig(path)

    result_fig = fig
    if not show:
        plt.close(fig)

    return result_fig


def plot_2d_cone(
    cone_membership: Callable[[np.ndarray], np.ndarray],
    path: Optional[Union[str, PathLike]] = None,
    density: int = 50,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the 2D cone by checking membership of the points in the cone.

    This function plots a 2D cone by checking the membership of points within the cone using the
    provided `cone_membership` function. The plot is created using Matplotlib and can be saved
    to a specified path if provided.

    :param cone_membership: A callable that takes an array of points and returns a boolean array
        indicating whether each point is inside the cone.
    :type cone_membership: Callable[[np.ndarray], np.ndarray]
    :param path: The file path where the plot will be saved. If None, the plot is not saved.
    :type path: Optional[Union[str, PathLike]]
    :param density: The number of points to sample along each axis for plotting.
    :type density: int
    :param show: Whether to show the plot or close it. If False, the figure is closed.
    :type show: bool
    :return: The Matplotlib figure object containing the plot.
    :rtype: plt.Figure
    """
    xlim = [-5, 5]
    ylim = [-5, 5]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot([xlim[0], xlim[1]], [0, 0], color="black")
    ax.plot([0, 0], [ylim[0], ylim[1]], color="black")

    x_pts = np.linspace(xlim[0], xlim[1], density)
    y_pts = np.linspace(ylim[0], ylim[1], density)

    X, Y = np.meshgrid(x_pts, y_pts)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    pts = pts[cone_membership(pts)]
    X, Y = pts[:, 0], pts[:, 1]

    ax.scatter(X, Y, alpha=0.3, c="blue", s=8)

    if path is not None:
        fig.savefig(path)

    result_fig = fig
    if not show:
        plt.close(fig)

    return result_fig


def plot_3d_cone(
    cone_membership: Callable[[np.ndarray], np.ndarray],
    path: Optional[Union[str, PathLike]] = None,
    density: int = 50,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the 3D cone by checking membership of the points in the cone.

    This function plots a 3D cone by checking the membership of points within the cone using the
    provided `cone_membership` function. The plot is created using Matplotlib and can be saved
    to a specified path if provided.

    :param cone_membership: A callable that takes an array of points and returns a boolean array
        indicating whether each point is inside the cone.
    :type cone_membership: Callable[[np.ndarray], np.ndarray]
    :param path: The file path where the plot will be saved. If None, the plot is not saved.
    :type path: Optional[Union[str, PathLike]]
    :param density: The number of points to sample along each axis for plotting.
    :type density: int
    :param show: Whether to show the plot or close it. If False, the figure is closed.
    :type show: bool
    :return: The Matplotlib figure object containing the plot.
    :rtype: plt.Figure
    """
    xlim = [-5, 5]
    ylim = [-5, 5]
    zlim = [-5, 5]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[], zticks=[], zticklabels=[])
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add X, Y, and Z axis lines at the middle of the region
    ax.plot([xlim[0], xlim[1]], [0, 0], [0, 0], color="black")
    ax.plot([0, 0], [ylim[0], ylim[1]], [0, 0], color="black")
    ax.plot([0, 0], [0, 0], [zlim[0], zlim[1]], color="black")

    x_pts = np.linspace(xlim[0], xlim[1], density)
    y_pts = np.linspace(ylim[0], ylim[1], density)
    z_pts = np.linspace(zlim[0], zlim[1], density)

    X, Y, Z = np.meshgrid(x_pts, y_pts, z_pts)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    pts = pts[cone_membership(pts)]
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax.scatter(X, Y, Z, alpha=0.3, c="blue", s=8)

    if path is not None:
        fig.savefig(path)

    result_fig = fig
    if not show:
        plt.close(fig)

    return result_fig


def plot_pareto_front(
    elements: np.ndarray,
    pareto_indices: np.ndarray,
    path: Optional[Union[str, PathLike]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the Pareto front for a given set of elements.

    This function plots the Pareto front for a given set of elements in either 2D or 3D space. The
    elements that belong to the Pareto front are highlighted in a different color.
    The plot is created using Matplotlib and can be saved to a specified path if provided.

    :param elements: An array of shape (N, dim) representing the elements to be plotted.
    :type elements: np.ndarray
    :param pareto_indices: An array of indices indicating which elements belong to the Pareto front.
    :type pareto_indices: np.ndarray
    :param path: The file path where the plot will be saved. If None, the plot is not saved.
    :type path: Optional[Union[str, PathLike]]
    :param show: Whether to show the plot or close it. If False, the figure is closed.
    :type show: bool
    :return: The Matplotlib figure object containing the plot.
    :rtype: plt.Figure

    :raises AssertionError: If the elements array is not 2-dimensional.
    :raises AssertionError: If the dimension of the elements is not 2 or 3.
    """
    dim = elements.shape[1]

    if elements.ndim != 2:
        raise AssertionError("Elements array should be N-by-dim.")
    if dim not in [2, 3]:
        raise AssertionError("Only 2D and 3D plots are supported.")

    fig = plt.figure(figsize=(6, 4))

    mask = np.ones(len(elements), dtype=np.uint8)
    mask[pareto_indices] = 0
    non_pareto_indices = np.nonzero(mask)
    if dim == 2:
        ax = fig.add_subplot(111)

        ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        ax.spines["bottom"].set_position("center")
        ax.spines["left"].set_position("center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.scatter(
            elements[pareto_indices][:, 0],
            elements[pareto_indices][:, 1],
            c="mediumslateblue",
            label="Pareto",
            alpha=0.7,
        )
        ax.scatter(
            elements[non_pareto_indices][:, 0],
            elements[non_pareto_indices][:, 1],
            c="tab:blue",
            label="Non Pareto",
            alpha=0.5,
        )
    else:
        ax = fig.add_subplot(111, projection="3d")

        ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[], zticks=[], zticklabels=[])
        ax.spines["bottom"].set_position("center")
        ax.spines["left"].set_position("center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.scatter(
            elements[pareto_indices][:, 0],
            elements[pareto_indices][:, 1],
            elements[pareto_indices][:, 2],
            c="mediumslateblue",
            label="Pareto",
            alpha=0.6,
        )
        ax.scatter(
            elements[non_pareto_indices][:, 0],
            elements[non_pareto_indices][:, 1],
            elements[non_pareto_indices][:, 2],
            c="tab:blue",
            label="Non Pareto",
            alpha=0.6,
        )

        # Add X, Y, and Z axis lines at the middle of the region
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], [0, 0], color="black")
        ax.plot([0, 0], [ax.get_ylim()[0], ax.get_ylim()[1]], [0, 0], color="black")
        ax.plot([0, 0], [0, 0], [ax.get_xlim()[0], ax.get_zlim()[1]], color="black")

    ax.legend(loc="lower left")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path)

    result_fig = fig
    if not show:
        plt.close(fig)

    return result_fig


def plot_cells_with_centers(
    cells: np.ndarray,
    centers: np.ndarray,
    path: Optional[Union[str, PathLike]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the given cells with their corresponding centers. The plot is created using
    Matplotlib and can be saved to a specified path if provided.

    :param cells: An array of shape (N, 2, 2) representing the cells to be plotted.
    :type cells: np.ndarray
    :param centers: An array of shape (N, 2) representing the centers of the cells.
    :type centers: np.ndarray
    :param path: The file path where the plot will be saved. If None, the plot is not saved.
    :type path: Optional[Union[str, PathLike]]
    :param show: Whether to show the plot or close it. If False, the figure is closed.
    :type show: bool
    :return: The Matplotlib figure object containing the plot.
    :rtype: plt.Figure
    """
    dim = centers.shape[1]
    if dim != 2:
        raise NotImplementedError("Visualization of cells is only implemented for 2D data.")

    fig, ax = plt.subplots()
    for point, cell in zip(centers, cells):
        rect = plt.Rectangle(
            (cell[0][0], cell[1][0]),
            cell[0][1] - cell[0][0],
            cell[1][1] - cell[1][0],
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(point[0], point[1], c="tab:red", marker="o", markersize=2)  # Plot the center point

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Cells with centers")

    if path is not None:
        fig.savefig(path)

    result_fig = fig
    if not show:
        plt.close(fig)

    return result_fig

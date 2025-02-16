from typing import Optional
from abc import ABC, abstractmethod

import numpy as np


class InputTransform(ABC):
    """
    Abstract base class for on-the-fly input transformations.
    """

    @abstractmethod
    def transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Transform the inputs to a model.

        :param Y: An `n x d`-dim array of training inputs.
        :type Y: np.ndarray
        :return: The transformed input observations.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def untransform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Un-transform the inputs (might be previously transformed or sampled within bounds).

        :param Y: An `n x d`-dim array of training inputs.
        :type Y: np.ndarray
        :return: The un-transformed input observations.
        :rtype: np.ndarray
        """
        pass


class NormalizeInput(InputTransform):
    r"""
    Normalize the inputs to a unit hypercube.

    :param d: The input dimension.
    :type d: int
    :param min_std: The minimum standard deviation to avoid division by zero.
    :type min_std: float
    """

    def __init__(self, d: int, bounds: Optional[list] = None):
        self.d = d

        self.calculated_bounds = bounds is None
        if bounds is not None:
            self.bounds = bounds

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Calculate the statistics and standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The standardized output observations.
        :rtype: np.ndarray
        """

        if Y.ndim != 2:
            raise ValueError(f"Expected 2D array; got {Y.ndim}D array.")
        if Y.shape[-1] != self.m:
            raise RuntimeError(f"Wrong output dimension. Given {Y.shape[-1]}; expected {self.m}.")
        if Y.shape[-2] < 1:
            raise ValueError(f"Can't standardize with no observations. {Y.shape=}.")

        if Y.shape[0] == 1:
            self.stds = np.ones((1, self.m))
        else:
            self.stds = np.std(Y, axis=0, keepdims=True)

        self.stds[self.stds < self.min_std] = 1
        self.means = np.mean(Y, axis=0, keepdims=True)

        self.fitted = True
        return self.transform(Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The standardized output observations.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("The transformation has not been fitted yet.")

        Y_standardized = (Y - self.means) / self.stds
        return Y_standardized

    def untransform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Un-standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The un-standardized output observations.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("The transformation has not been fitted yet.")

        Y_untransformed = Y * self.stds + self.means
        return Y_untransformed


class OutputTransform(ABC):
    """
    Abstract base class for on-the-fly output transformations.
    """

    @abstractmethod
    def transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Transform the outputs given as a model's training targets.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The transformed output observations.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def untransform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Un-transform the outputs (might be previously transformed or sampled from the model).

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The un-transformed output observations.
        :rtype: np.ndarray
        """
        pass


class StandardizeOutput(OutputTransform):
    r"""
    Standardize the outputs (to zero mean, unit variance) by subtracting the mean and dividing
    by the standard deviation.

    .. math::

        Y_{\text{standardized}} = \frac{Y - \mu}{\sigma}

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation of the outputs.

    :param m: The number of outputs.
    :type m: int
    :param min_std: The minimum standard deviation to avoid division by zero.
    :type min_std: float
    """

    def __init__(self, m: int, min_std: float = 1e-8):
        self.fitted = False
        self.means = np.zeros((1, m))
        self.stds = np.ones((1, m))

        self.m = m
        self.min_std = min_std

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Calculate the statistics and standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The standardized output observations.
        :rtype: np.ndarray
        """

        if Y.ndim != 2:
            raise ValueError(f"Expected 2D array; got {Y.ndim}D array.")
        if Y.shape[-1] != self.m:
            raise RuntimeError(f"Wrong output dimension. Given {Y.shape[-1]}; expected {self.m}.")
        if Y.shape[-2] < 1:
            raise ValueError(f"Can't standardize with no observations. {Y.shape=}.")

        if Y.shape[0] == 1:
            self.stds = np.ones((1, self.m))
        else:
            self.stds = np.std(Y, axis=0, keepdims=True)

        self.stds[self.stds < self.min_std] = 1
        self.means = np.mean(Y, axis=0, keepdims=True)

        self.fitted = True
        return self.transform(Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The standardized output observations.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("The transformation has not been fitted yet.")

        Y_standardized = (Y - self.means) / self.stds
        return Y_standardized

    def untransform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Un-standardize the outputs.

        :param Y: An `n x m`-dim array of training targets.
        :type Y: np.ndarray
        :return: The un-standardized output observations.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("The transformation has not been fitted yet.")

        Y_untransformed = Y * self.stds + self.means
        return Y_untransformed

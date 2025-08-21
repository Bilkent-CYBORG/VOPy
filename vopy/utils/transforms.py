from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


class InputTransform(ABC):
    """
    Abstract base class for on-the-fly input transformations.
    """

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Does necessary calculations and transform the inputs to a model.

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The transformed input observations.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Transform the inputs to a model.

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The transformed input observations.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def untransform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Un-transform the inputs (might be previously transformed or sampled within bounds).

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The un-transformed input observations.
        :rtype: np.ndarray
        """
        pass


class NormalizeInput(InputTransform):
    r"""
    Normalize the inputs to a unit hypercube.

    :param d: The input dimension.
    :type d: int
    :param bounds: The bounds of the input space. If not provided, they are calculated from the
        data.
    :type bounds: Optional[ArrayLike]
    :param min_range: The minimum range for each dimension.
    :type min_range: float
    """

    def __init__(self, d: int, bounds: Optional[ArrayLike] = None, min_range: float = 1e-6):
        self.d = d
        self.min_range = min_range

        self.calculated_bounds = bounds is None
        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.array(self.bounds)
            if bounds.shape != (self.d, 2):
                raise ValueError(f"Expected bound with shape ({self.d}, 2); got {bounds.shape}.")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Calculate the bounds and normalize the inputs. If the bounds are provided initially, they
        are used directly. Otherwise, the bounds are calculated from the data.

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The normalized inputs.
        :rtype: np.ndarray
        """

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array; got {X.ndim}D array.")
        if X.shape[-1] != self.d:
            raise RuntimeError(f"Wrong input dimension. Given {X.shape[-1]}; expected {self.d}.")

        if self.calculated_bounds:
            if X.shape[-2] < 1:
                raise ValueError(f"Can't normalize with no observations. {X.shape=}.")
            mins = np.min(X, axis=0, keepdims=True)
            maxs = np.max(X, axis=0, keepdims=True)
            ranges = maxs - mins
            range_mask = ranges < self.min_range
            mins[range_mask] = 0
            maxs[range_mask] = 1
            self.bounds = np.concatenate([mins.reshape(-1, 1), maxs.reshape(-1, 1)], axis=1)

        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Normalize the inputs.

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The normalized inputs.
        :rtype: np.ndarray
        """

        if self.bounds is None:
            raise ValueError("The bounds have not been set yet.")

        if X.shape[-1] != self.d:
            raise RuntimeError(f"Wrong input dimension. Given {X.shape[-1]}; expected {self.d}.")

        X_transformed = (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return X_transformed

    def untransform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Un-normalize the inputs.

        :param X: An `n x d`-dim array of training inputs.
        :type X: np.ndarray
        :return: The un-normalized inputs.
        :rtype: np.ndarray
        """

        if self.bounds is None:
            raise ValueError("The bounds have not been set yet.")

        X_untransformed = X * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return X_untransformed


class OutputTransform(ABC):
    """
    Abstract base class for on-the-fly output transformations.
    """

    @abstractmethod
    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Does necessary calculations and transform the targets of a model.

        :param Y: An `n x m`-dim array of training inputs.
        :type Y: np.ndarray
        :return: The transformed output observations.
        :rtype: np.ndarray
        """
        pass

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

        Y_transformed = (Y - self.means) / self.stds
        return Y_transformed

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

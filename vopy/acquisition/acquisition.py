from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Optional, Tuple

import numpy as np

from vopy.design_space import DiscreteDesignSpace
from vopy.models import Model, ModelList
from vopy.order import PolyhedralConeOrder
from vopy.utils import binary_entropy


class AcquisitionStrategy(ABC):
    """
    Abstract class for acquisition functions.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class DecoupledAcquisitionStrategy(AcquisitionStrategy):
    """
    Abstract class for decoupled settings.
    Acquisition values are inversely weighted by costs.

    :param output_dim: Number of objectives.
    :type output_dim: int
    :param evaluation_index: Index of the objective to be evaluated.
    :type evaluation_index: Optional[int]
    :param costs: Costs of evaluating each objective.
    :type costs: Optional[list]
    """

    def __init__(
        self, output_dim: int, evaluation_index: Optional[int] = None, costs: Optional[list] = None
    ) -> None:
        super().__init__()
        self.out_dim = output_dim
        self.evaluation_index = evaluation_index
        self.costs = costs


class SumVarianceAcquisition(AcquisitionStrategy):
    """
    Acquisition function that returns the sum of variances of the objectives assuming
    independent objectives, *i.e.*, trace of covariance matrix.

    :param model: Model to be used for predictions.
    :type model: Model
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sum of variances of the objectives at the given points.

        :param x: Points to evaluate the acquisition function.
        :type x: np.ndarray
        :return: Trace of the covariance matrices at the given points.
        :rtype: np.ndarray
        """
        _, variances = self.model.predict(x)
        return np.sum(np.diagonal(variances, axis1=-2, axis2=-1), axis=-1)


class MaxVarianceDecoupledAcquisition(DecoupledAcquisitionStrategy):
    """
    Decoupled acquisition function that returns the maximum variance in a given objective.

    :param model: Model to be used for predictions.
    :type model: ModelList
    :param evaluation_index: Index of the objective to be evaluated.
    :type evaluation_index: Optional[int]
    :param costs: Costs of evaluating each objective.
    :type costs: Optional[list]
    """

    def __init__(
        self, model: ModelList, evaluation_index: Optional[int] = None, costs: Optional[list] = None
    ) -> None:
        self.model = model
        super().__init__(self.model.output_dim, evaluation_index, costs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the variance in a given objective at the given points.

        :param x: Points to evaluate the acquisition function.
        :type x: np.ndarray
        :return: Corresponding value of the diagonal of the covariance matrices at the given points.
        :rtype: np.ndarray
        """
        if self.evaluation_index is None:
            raise AssertionError("evaluation_index can't be None during forward.")
        _, variances = self.model.predict(x)
        value = np.diagonal(variances, axis1=-2, axis2=-1)[..., self.evaluation_index]
        if self.costs is not None:
            value = value / self.costs[self.evaluation_index]
        return value


class ThompsonEntropyDecoupledAcquisition(DecoupledAcquisitionStrategy):
    r"""
    A novel acquisition function that returns the expected entropy reduction of given points
    being in the Pareto set.

    First, Thompson samples are drawn from the posterior distribution to identify the Pareto set.
    Then, the empirical frequencies of points being in the Pareto set are recorded to
    estimate the entropies. Finally, the mean empirical frequencies of points being in the
    Pareto set given their true value in the evaluation index are calculated to form the
    acquisition value.

    Mathematically, the acquisition function is defined as:

    .. math::

        \alpha_t^i (x) = \frac{H (X | \mathcal{D}_{t-1}) - \mathbb E_{y_i \sim GP^i_t(x)}
        [H (X | \mathcal{D}_{t-1}, y_i)]}{c_i},

    where

    .. math::

        X = \mathbb{I} \{x \in P^*\}.

    :param model: Model to be used for predictions.
    :type model: ModelList
    :param order: Order object to be used for Pareto optimal calculation.
    :type order: Order
    :param evaluation_index: Index of the objective to be evaluated.
    :type evaluation_index: Optional[int]
    :param costs: Costs of evaluating each objective.
    :type costs: Optional[list]
    :param num_thompson_samples: Number of Thompson samples to draw.
    :type num_thompson_samples: int
    """

    def __init__(
        self,
        model: ModelList,
        order: PolyhedralConeOrder,
        evaluation_index: Optional[int] = None,
        costs: Optional[list] = None,
        num_thompson_samples: int = 10,
    ) -> None:
        self.model = model
        super().__init__(self.model.output_dim, evaluation_index, costs)
        self.order = order
        self.num_thompson_samples = num_thompson_samples

        self._clear_cache()

    def _clear_cache(self):
        """
        Clear the cache.
        """
        self._cache_x = None
        self._cache_pareto_mask = None
        self._cache_prior_entropy = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the expected entropy reduction of given points being in the Pareto set.

        :param x: Points to evaluate the acquisition function.
        :type x: np.ndarray
        :return: Expected entropy reductions of Pareto set caused by the given points.
        :rtype: np.ndarray
        """
        if self.evaluation_index is None:
            raise AssertionError("evaluation_index can't be None during forward.")

        # TODO: Model might've been updated for batch selection. That also coincides with
        # a change in array x, but need to be sure.
        if self._cache_x is None or not np.array_equal(self._cache_x, x):
            self._clear_cache()
            self._cache_x = x.copy()
            t_samples = []
            for dim_i in range(self.out_dim):
                t_samples.append(
                    self.model.sample_from_single_posterior(x, dim_i, self.num_thompson_samples)
                )
            t_samples = np.stack(t_samples)
            self._cache_pareto_mask = np.zeros(
                shape=(*[self.num_thompson_samples] * self.out_dim, len(x)), dtype=bool
            )
            for comb in combinations(range(self.num_thompson_samples), r=self.out_dim):
                sample = np.stack(t_samples[np.arange(self.out_dim), np.array(comb)]).T
                self._cache_pareto_mask[comb][self.order.get_pareto_set(sample)] = True

            prior_prob_mean_axes = np.arange(self.out_dim)
            prior_prob = np.mean(self._cache_pareto_mask, axis=tuple(prior_prob_mean_axes))
            self._cache_prior_entropy = binary_entropy(prior_prob)

        posterior_prob_mean_axes = np.arange(self.out_dim)
        posterior_prob_mean_axes = np.delete(posterior_prob_mean_axes, [self.evaluation_index])
        posterior_prob = np.mean(self._cache_pareto_mask, axis=tuple(posterior_prob_mean_axes))
        posterior_entropy = binary_entropy(posterior_prob)
        mean_posterior_entropy = np.mean(posterior_entropy, axis=0)
        value = self._cache_prior_entropy - mean_posterior_entropy

        if self.costs is not None:
            value = value / self.costs[self.evaluation_index]
        return value


class MaxDiagonalAcquisition(AcquisitionStrategy):
    """
    Calculates the length of the diagonals for each point given. Diagonal means the maximum
    distance inside the confidence region. It needs the design space to be a `DiscreteDesignSpace`
    in order to e confidence regions.

    :param design_space: Design space to take confidence regions from.
    :type design_space: DiscreteDesignSpace
    """

    def __init__(self, design_space: DiscreteDesignSpace) -> None:
        super().__init__()
        self.design_space = design_space

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the diameter of the confidence region at the given points.

        :param x: Points to evaluate the acquisition function.
        :type x: np.ndarray
        :return: Length of the diagonals of the confidence regions at the given points.
        :rtype: np.ndarray
        """
        indices = self.design_space.locate_points(x)
        value = np.zeros(len(x))

        for idx, design_i in enumerate(indices):
            value[idx] = self.design_space.confidence_regions[design_i].diagonal()

        return value


def optimize_acqf_discrete(
    acq: AcquisitionStrategy, q: int, choices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize the acquisition function over the given discrete choices.

    :param acq: Acquisition function to be optimized.
    :type acq: AcquisitionStrategy
    :param q: Number of points to select, *i.e.*, batch size.
    :type q: int
    :param choices: Choices to optimize the acquisition function over.
    :type choices: np.ndarray
    :return: Tuple of selected points and their corresponding acquisition values.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    candidate_list, acq_value_list = [], []

    # TODO: Another batch selection method might be updating model at each step.
    # Either a fantasy update or a full update, i.e., adding samples along the way.

    chosen = 0
    while chosen < q:
        acq_values = acq(choices)

        best_idx = np.argmax(acq_values)
        candidate_list.append(choices[best_idx])
        acq_value_list.append(acq_values[best_idx])

        choices = np.concatenate([choices[:best_idx], choices[best_idx + 1 :]])

        chosen += 1

    return np.stack(candidate_list, axis=-2), np.array(acq_value_list)


def optimize_decoupled_acqf_discrete(
    acq: DecoupledAcquisitionStrategy, q: int, choices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize the decoupled acquisition function over the given discrete choices.

    :param acq: Decoupled acquisition function to be optimized.
    :type acq: DecoupledAcquisitionStrategy
    :param q: Number of points to select, *i.e.*, batch size.
    :type q: int
    :param choices: Choices to optimize the acquisition function over.
    :type choices: np.ndarray
    :return: Tuple of selected points, their corresponding acquisition values
        and their corresponding objective indices to evaluate.
    """
    saved_eval_i = acq.evaluation_index

    candidate_list = np.empty((0, choices.shape[-1]))
    acq_values = np.empty(0)
    eval_indices = np.empty(0, dtype=np.int32)
    for eval_i in range(acq.out_dim):
        acq.evaluation_index = eval_i
        curr_candidate_list, curr_acq_values = optimize_acqf_discrete(acq, q, choices)
        candidate_list = np.concatenate([candidate_list, curr_candidate_list], axis=0)
        acq_values = np.concatenate([acq_values, curr_acq_values], axis=0)
        eval_indices = np.concatenate([eval_indices, np.full(q, fill_value=eval_i)], axis=0)

    # Find indices of the highest q elements
    indices = np.argpartition(acq_values, -q)[-q:]
    # Sort these indices by the actual values
    indices = indices[np.argsort(acq_values[indices])[::-1]]

    candidate_list = candidate_list[indices]
    acq_values = acq_values[indices]
    eval_indices = eval_indices[indices]

    acq.evaluation_index = saved_eval_i

    return candidate_list.reshape(q, -1), acq_values, eval_indices

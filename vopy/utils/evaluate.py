import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

from vopy.datasets import Dataset
from vopy.maximization_problem import ContinuousProblem
from vopy.models import Model
from vopy.order import PolyhedralConeOrder
from vopy.utils import generate_sobol_samples, get_delta, get_uncovered_size


def calculate_epsilonF1_score(
    dataset: Dataset,
    order: PolyhedralConeOrder,
    true_indices: np.ndarray,
    pred_indices: np.ndarray,
    epsilon: float,
):
    """
    This method computes the epsilon-F1 score, which is a measure of the accuracy of the predicted
    indices compared to the true indices, considering a specified epsilon value.

    :param dataset: The dataset containing the output data.
    :type dataset: Dataset
    :param order: The order object containing the ordering cone.
    :type order: Order
    :param true_indices: The true indices of the Pareto front.
    :type true_indices: np.ndarray
    :param pred_indices: The predicted indices of the Pareto front.
    :type pred_indices: np.ndarray
    :param epsilon: The epsilon value used for calculating the uncovered size and gap values.
    :type epsilon: float
    :return: The calculated epsilon-F1 score.
    :rtype: float
    """
    indices_of_missed_pareto = list(set(true_indices) - set(pred_indices))

    uncovered_missed_pareto_count = get_uncovered_size(
        dataset.out_data[indices_of_missed_pareto],
        dataset.out_data[pred_indices],
        epsilon,
        order.ordering_cone.W,
    )

    delta_values = get_delta(dataset.out_data, order.ordering_cone.W, order.ordering_cone.alpha)

    true_eps = np.sum(delta_values[np.array(list(pred_indices)).astype(int)] <= epsilon, axis=0)[0]

    tp_eps = true_eps
    fp_eps = len(pred_indices) - true_eps
    f1_eps = (2 * tp_eps) / (2 * tp_eps + fp_eps + uncovered_missed_pareto_count)

    return f1_eps


def calculate_hypervolume_discrepancy_for_model(
    order: PolyhedralConeOrder, problem: ContinuousProblem, model: Model
):
    """
    This method computes the hypervolume discrepancy between the true Pareto front and the
    predicted Pareto front for a given model. It uses Sobol sampling to generate input samples,
    evaluates the true and predicted outputs, and calculates the hypervolumes of the true and
    predicted Pareto fronts.

    :param order: The order object containing the ordering cone and methods to get the Pareto set.
    :type order: Order
    :param problem: The continuous problem to be evaluated.
    :type problem: ContinuousProblem
    :param model: The model trained on the problem, used to predict the outputs.
    :type model: Model
    :return: The logarithm of the hypervolume discrepancy between
        the true and predicted Pareto fronts.
    :rtype: float
    :raises AssertionError: If the hypervolume discrepancy is less than or equal to a threshold.
    """
    x = generate_sobol_samples(problem.in_dim, 2048)  # TODO: magic number

    f = problem.evaluate(x, noisy=False)
    true_pareto_indices = order.get_pareto_set(f)
    y, _ = model.predict(x)
    pred_pareto_indices = order.get_pareto_set(y)

    f_W = f @ order.ordering_cone.W.T
    reference_point = torch.tensor(np.min(f_W, axis=0))
    hypervolume_instance = Hypervolume(reference_point)

    hypervolume_true = hypervolume_instance.compute(torch.tensor(f_W[true_pareto_indices]))
    hypervolume_pred = hypervolume_instance.compute(torch.tensor(f_W[pred_pareto_indices]))

    if hypervolume_true - hypervolume_pred <= 1e-4:  # TODO: magic number
        raise AssertionError("Hypervolumes are the same.")

    log_hv_disc = np.log(hypervolume_true - hypervolume_pred)

    return log_hv_disc

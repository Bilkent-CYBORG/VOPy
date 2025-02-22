import logging
import warnings

import vopy.algorithms
from vopy.datasets import get_dataset_instance
from vopy.order import ConeTheta2DOrder
from vopy.utils import set_seed
from vopy.utils.evaluate import calculate_epsilonF1_score

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


order = ConeTheta2DOrder(cone_degree=100)
fig_cone = order.ordering_cone.plot()

dataset_name = "Fairness"
dataset = get_dataset_instance(dataset_name)
fig_pareto = order.plot_pareto_set(dataset.out_data)

epsilon = 0.01
delta = 0.05
noise_var = epsilon

set_seed(0)

algorithm = vopy.algorithms.PaVeBaGP(
    epsilon=epsilon,
    delta=delta,
    dataset_name=dataset_name,
    order=order,
    noise_var=noise_var,
    conf_contraction=32,
    type="IH",
    batch_size=1,
)

while True:
    is_done = algorithm.run_one_step()

    if is_done:
        break

pred_pareto_indices = sorted(list(algorithm.P))
pareto_indices = order.get_pareto_set(dataset.out_data)

eps_f1 = calculate_epsilonF1_score(dataset, order, pareto_indices, pred_pareto_indices, epsilon)
print(f"epsilon-F1 Score: {eps_f1:.2f}")
print(f"Number of observations: {algorithm.sample_count}")

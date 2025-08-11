import logging

import numpy as np

from vopy.algorithms import VOGP_AD, VOGP_ADOnline
from vopy.order import ComponentwiseOrder, ConeTheta2DOrder
from vopy.maximization_problem import get_continuous_problem


def main():
    np.random.seed(42)

    logging.basicConfig(level=logging.INFO)

    epsilon = 0.1
    delta = 0.05
    noise_var = 0.01
    max_iterations = 20

    problem = get_continuous_problem("BraninCurrin", noise_var)

    order = ComponentwiseOrder(2)
    order = ConeTheta2DOrder(cone_degree=30)

    vogp_ad = VOGP_AD(
        epsilon=epsilon,
        delta=delta,
        problem=problem,
        order=order,
        noise_var=noise_var,
        conf_contraction=32,
        batch_size=1,
    )

    for it in range(max_iterations):
        is_done = vogp_ad.run_one_step()

        if is_done:
            print("VOGP_AD has converged!")
            break

    print(f"{it} iterations completed for VOGP_AD.")
    print()

    vogp_ad_on = VOGP_ADOnline(
        epsilon=epsilon,
        delta=delta,
        problem=problem,
        order=order,
        conf_contraction=8,
        batch_size=1,
        initial_sample_cnt=3,
        reset_on_retrain=True,
    )

    max_iterations = 50
    for it in range(max_iterations):
        is_done = vogp_ad_on.run_one_step()

        if is_done:
            print("VOGP_ADOnline has converged!")
            break

        print()
    print(f"{it} iterations completed for VOGP_ADOnline.")


if __name__ == "__main__":
    main()

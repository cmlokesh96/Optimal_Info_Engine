"""This script compares analytically known marginals/conditionals of the fitted 4d distribution with the analytical counterparts"""
import numpy as np
import time
from matplotlib import pyplot as plt
from sdesim.helpers import load_fitted_distribution
from sdesim.statistics import condition_gaussian
from sdesim.analytics import compute_conditional_xt_dist, compute_conditional_xbt_distribution
from sdesim.constants import BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH
from sdesim.params import params_bath_model
from sdesim.visualization import plot_two_normal_distributions, setup_matplotlib
from sdesim.constants import COMPARISON_P_X_b_T_X_0_PATH, COMPARISON_P_X_T_X_0_PATH

def main():
    setup_matplotlib()
    gaussian_4d = load_fitted_distribution(BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    marginal_x_0_x_t = gaussian_4d.marginal([0,2])
    conditionedX_t_onX_0 = condition_gaussian(marginal_x_0_x_t, [0], -2)
    analytical_x_conditional = compute_conditional_xt_dist(-2,params_bath_model,1)

    plot_two_normal_distributions(analytical_x_conditional,conditionedX_t_onX_0, ["$X_t$ in nm","$P(X_t|\\mathcal{X}_0)$)"], COMPARISON_P_X_T_X_0_PATH, ["Analytical", "Numerical"], title= "Comparison of fitted and analytical $P(X_t|\\mathcal{X}_0)$")

    gaussian_4d = load_fitted_distribution(BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    marginal_x_0_x_b_t = gaussian_4d.marginal([0,3])
    conditionedX_b_t_onX_0 = condition_gaussian(marginal_x_0_x_b_t, [0], 2)
    analytical_x_b_conditional = compute_conditional_xbt_distribution(2,params_bath_model,1)


    plot_two_normal_distributions(analytical_x_b_conditional,conditionedX_b_t_onX_0, ["$X_{b,t}$ in nm","$P(X_{b,t})|\\mathcal{X}_0$)"], COMPARISON_P_X_b_T_X_0_PATH, ["Analytical", "Numerical"], title= "Comparison of fitted and analytical $P(X_{b,t}|\\mathcal{X}_0)$")

if __name__ == "__main__":#Only execute main if program is executed itself, not when imported
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
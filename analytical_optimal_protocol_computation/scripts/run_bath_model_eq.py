"""Starting from a set initial conditions, trajectories in the bath model are simulated until the process is stationary, from that state the 
stationary distribution is obtained by fitting a gaussian which is compared to the respective analytical result"""
import numpy as np

from sdesim.params import params_bath_model
from sdesim.integrators import euler_maruyama_ensemble_bath_model
from sdesim.visualization import setup_matplotlib, plot_trajectory, plot_heatmap_stationary, compare_fitted_and_analytical
from sdesim.statistics import analytical_distribution_eq, symmetric_kl_gaussian
from sdesim.helpers import save_fitted_distribution, reshape_constant_initial_state
from sdesim.constants import SEED_2, BATH_EQ_DIFFERENCE_HEATMAP_PATH, BATH_EQ_FITTED_2D_DISTRIBUTION_PATH, BATH_EQ_HEATMAP_PATH, BATH_EQ_TRAJECTORIES_PATH, TIME_STEP_NON_EQ,FINE_TIME_STEP_NON_EQ
import time


def main():
    setup_matplotlib()

    rng = np.random.default_rng(SEED_2)

    dimension = 2
    noise_amplitude = np.sqrt(np.array([2*params_bath_model.T * params_bath_model.k_B/params_bath_model.gamma, 2 * params_bath_model.T * params_bath_model.k_B/params_bath_model.gamma_b]))
    t_final = 80
    number_of_samples = 10000

    initial_conditions = reshape_constant_initial_state(np.array([0.0, 0.0]), number_of_samples)
    trajectories = euler_maruyama_ensemble_bath_model( dimension, noise_amplitude,initial_conditions, t_final, TIME_STEP_NON_EQ, FINE_TIME_STEP_NON_EQ, number_of_samples, rng, params_bath_model)

    plot_trajectory(trajectories, 33, np.array([1, 2]), BATH_EQ_TRAJECTORIES_PATH, ["$X$", "$X_{\\mathrm{b}}$"])
    fitted_gaussian = plot_heatmap_stationary(10*params_bath_model.tau_b, TIME_STEP_NON_EQ, trajectories, 100, BATH_EQ_HEATMAP_PATH, rng)
    analytical_gaussian = analytical_distribution_eq(params_bath_model, rng)
    compare_fitted_and_analytical(fitted_gaussian, analytical_gaussian, np.array([-4,4]), np.array([-6,6]), BATH_EQ_DIFFERENCE_HEATMAP_PATH)
    print("The symmetric KL Divergence is: ",symmetric_kl_gaussian(fitted_gaussian,analytical_gaussian))
    print(fitted_gaussian.cov)
    print(analytical_gaussian.cov)
    save_fitted_distribution(fitted_gaussian, BATH_EQ_FITTED_2D_DISTRIBUTION_PATH)



if __name__ == "__main__":#Only execute main if program is executed itself, not when imported
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
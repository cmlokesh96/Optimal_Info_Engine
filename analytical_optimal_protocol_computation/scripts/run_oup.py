"""Simulate an ornstein uhlenbeck process as acheck the simulation is working correctly"""
import numpy as np

from sdesim.params import SystemVariables
from sdesim.integrators import euler_maruyama_ensemble_single_particle_in_well
from sdesim.visualization import setup_matplotlib, plot_trajectory, plot_autocorrelation, plot_histogramm_stationary
from sdesim.helpers import reshape_constant_initial_state
import time
from sdesim.constants import SEED_1, OUP_AUTOCORRELATION_PATH,OUP_HISTOGRAM_PATH, OUP_TRAJETORY_PATH


def main():
    setup_matplotlib()

    params = SystemVariables.create(gamma=2, gamma_b = 1, tau_p=1, kappa=1, kappa_b=0.75, T=1, x_thresh=0.5)
    rng = np.random.default_rng(SEED_1)

    dimension = 1
    noise_amplitude = np.array([np.sqrt(2 * params.k_B * params.T/params.gamma)])
    t_final = 20
    time_step = 0.01
    number_of_steps = int(np.ceil(t_final / time_step))
    number_of_samples = 1000

    initial_conditions = reshape_constant_initial_state(np.array([0.0]), number_of_samples)
    trajectories = euler_maruyama_ensemble_single_particle_in_well(dimension, noise_amplitude,initial_conditions, t_final, time_step, number_of_samples, rng, params)

    plot_trajectory(trajectories, 56, np.array([1]), OUP_TRAJETORY_PATH, ["$X$"])
    plot_autocorrelation(trajectories, t_final, 0.01, time_step, 10 * params.tau_p, noise_amplitude, params, OUP_AUTOCORRELATION_PATH)
    fitted_gaussian = plot_histogramm_stationary(10 * params.tau_p, time_step, trajectories, 100,OUP_HISTOGRAM_PATH, rng, f"Stationary distribution of $X$")

#Executes main only if the program is run directly and not if it is imported
if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
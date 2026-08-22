"""trajectories in the bath model are simulated and saved starting from initial conditions sampled from the (analytical)
equilibrium distribution, the 4d gaussian of initial ppositions and positions at a specified time is fitted and saved"""
import numpy as np
import time
from scipy import stats
from sdesim.helpers import load_fitted_distribution, format_trajectories_at_two_times, draw_initial_conditions, save_fitted_distribution
from sdesim.params import params_bath_model
from sdesim.statistics import fit_gaussian, analytical_distribution_eq
from sdesim.visualization import setup_matplotlib, plot_heatmap, plot_autocorrelation_to_init
from sdesim.integrators import euler_maruyama_ensemble_bath_model
from sdesim.constants import SEED_3, BATH_EQ_FITTED_2D_DISTRIBUTION_PATH,BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH, BATH_NEQ_HEATMAP_PATH, BATH_NEQ_AUTOCORRELATION_X_0_X_t_PATH, BATH_NEQ_AUTOCORRELATION_X_0_X_bt_PATH, NON_EQUILIBRIUM_TRAJECTORIES_PATH, TIME_STEP_NON_EQ, FINE_TIME_STEP_NON_EQ
from sdesim.analytics import bath_correlation_C, bath_correlation_C_b

def main():
    setup_matplotlib()

    rng = np.random.default_rng(SEED_3)

    dimension = 2
    noise_amplitude = np.sqrt(np.array([2*params_bath_model.T * params_bath_model.k_B/params_bath_model.gamma, 2 * params_bath_model.T * params_bath_model.k_B/params_bath_model.gamma_b]))
    t_final = 80
    number_of_steps = int(np.ceil(t_final / TIME_STEP_NON_EQ))
    number_of_samples = 500000

    #fitted_2d_gaussian = load_fitted_distribution(BATH_EQ_FITTED_2D_DISTRIBUTION_PATH)
    analytic_2d_gaussian = analytical_distribution_eq(params_bath_model,rng)

    initial_conditions = draw_initial_conditions(analytic_2d_gaussian, number_of_samples, rng)
    trajectories = euler_maruyama_ensemble_bath_model(dimension, noise_amplitude,initial_conditions, t_final, TIME_STEP_NON_EQ, FINE_TIME_STEP_NON_EQ, number_of_samples, rng, params_bath_model)

    init_final_trajectories = format_trajectories_at_two_times(trajectories,0,1,TIME_STEP_NON_EQ)
    
    fitted_4d_gaussian = fit_gaussian(init_final_trajectories, rng)
    print(fitted_4d_gaussian.cov)
    plot_heatmap(1, TIME_STEP_NON_EQ, trajectories, 100, BATH_NEQ_HEATMAP_PATH)
    plot_autocorrelation_to_init(trajectories,t_final,0.1,TIME_STEP_NON_EQ,bath_correlation_C, params_bath_model, BATH_NEQ_AUTOCORRELATION_X_0_X_t_PATH, np.array([1,1]), "Autocorelation $\\langle X(0)X(t)\\rangle$")
    plot_autocorrelation_to_init(trajectories,t_final,0.1,TIME_STEP_NON_EQ,bath_correlation_C_b, params_bath_model, BATH_NEQ_AUTOCORRELATION_X_0_X_bt_PATH, np.array([1,2]), "Autocorelation $\\langle X(0)X_b(t)\\rangle$")

    save_fitted_distribution(fitted_4d_gaussian, BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    np.save(NON_EQUILIBRIUM_TRAJECTORIES_PATH, trajectories)




if __name__ == "__main__":#Only execute main if program is executed itself, not when imported
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
"""Compare the convergence of initial conditions given measurements sampled from trajectories obtained using a brownian dynamics simulation with bootstaping
to the analytical solution for increasing number of simulated trajectories"""
import time
import numpy as np
from sdesim.constants import NON_EQUILIBRIUM_TRAJECTORIES_PATH,TIME_STEP_NON_EQ, SEED_4, SAVE_DIR_2_STATE_INIT, PLOT_DIR_INITIAL_CONDITION_CONVERGENCE, SAVE_DIR_3_STATE_INIT
from sdesim.visualization import plot_bootstrap_convergence_grid
from sdesim.params import params_bath_model
from sdesim.helpers import get_or_compute_initial_positions
from sdesim.analytics import mean_initial_positions_from_analytical

def main():
    trajectories = np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH)
    rng = np.random.default_rng(SEED_4)
    t_second_measurement_values = [0, 1, 80]#smaller then t_final of run_bath_model_eq if run with initial conditions from trajectories
    t_protocol_end_values = [1,3, 5]

    sample_sizes = np.unique(np.round(np.geomspace(200, 500000, 20)).astype(int))

    for t_second_measurement in t_second_measurement_values:

        save_path_colloid = f"{PLOT_DIR_INITIAL_CONDITION_CONVERGENCE}/initial_x_convergence_grid_t2_{t_second_measurement}.png"
        save_path_bath = f"{PLOT_DIR_INITIAL_CONDITION_CONVERGENCE}/initial_x_b_convergence_grid_t2_{t_second_measurement}.png"

        mean_initial_positions = get_or_compute_initial_positions(params_bath_model, t_second_measurement, SAVE_DIR_2_STATE_INIT, mean_initial_positions_from_analytical,method_name="analytical", n_outcomes=2)
        plot_bootstrap_convergence_grid(trajectories,t_second_measurement, TIME_STEP_NON_EQ,params_bath_model, mean_initial_positions, sample_sizes, save_path_colloid,save_path_bath,rng, n_outcomes=2)

if __name__ == "__main__": #Only execute main if program is executed itself, not when imported
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
"""Obtain initial conditions in a chosen manner and optimal protocols given those for different measurement outcome combinations (using two possible outcomes),
looping over different times between measurements and protocol durations. Comparing work of those protocols"""
import numpy as np
import time
from sdesim.constants import NON_EQUILIBRIUM_TRAJECTORIES_PATH, TIME_STEP_NON_EQ, OPTIMAL_PROTOCOL_PATHS, SAVE_DIR_OPTIMAL_PROTOCOLS, PLOT_DIR_OPTIMAL_PROTOCOLS, SAVE_DIR_2_STATE_INIT
from sdesim.params import params_bath_model, energy_scale
from sdesim.statistics import make_distribution_initial_position_estimator, make_trajectory_initial_position_estimator
from sdesim.helpers import get_or_compute_initial_positions
from sdesim.visualization import setup_matplotlib, plot_work_comparison, plot_optimal_trajectories_grid, plot_work_gap, plot_accumulated_work
from sdesim.process_combination import process_combination
from sdesim.analytics import mean_initial_positions_from_analytical


def main():
    setup_matplotlib()
    

    #The times have to be multiples of TIME_STEP_NON_EQ
    t_second_measurement_values = [0, 0.5, 1, 3, 80]#smaller then t_final of run_bath_model_non_eq when using trajectory sampled initial conditions
    t_protocol_end_values = [1,2, 2.5,3, 3.5, 4, 5]
    kBT_scale = energy_scale(params_bath_model)

    #reference initial conditions, can be passed to process combination for reference
    matlab_initial_positions = {
            "m_0=0,m_t=0": {
                "mean_X_t": -6.601709e-9,   # State 00, mean(x_t) in m
                "mean_X_bt": -8.669396e-9,  # State 00, mean(x_b) in m
            },
            "m_0=0,m_t=1": {
                "mean_X_t": 88.764934e-9,   # State 01, mean(x_t) in m
                "mean_X_bt": 78.836656e-9,  # State 01, mean(x_b) in m
            },
            "m_0=1,m_t=0": {
                "mean_X_t": 14.103201e-9,   # State 10, mean(x_t) in m
                "mean_X_bt": 69.505341e-9,  # State 10, mean(x_b) in m
            },
            "m_0=1,m_t=1": {
                "mean_X_t": 93.148047e-9,   # State 11, mean(x_t) in m
                "mean_X_bt": 137.521163e-9, # State 11, mean(x_b) in m
            },
        }

    all_results = {}
    for t_second_measurement in t_second_measurement_values:
        #Per default, the  initial conditions are computed using analytic solution, to switch to sampling from trajectories, or using fitted distributions, comment
        #out one of the following lines and replace mean_initial_positions_from_analytical in get_or_compute_initial with returned function
        # and specify method_name="analytical", "trajectories" or "distribution"
        
        #distribution_initial_condition_function = make_distribution_initial_position_estimator(np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH),TIME_STEP_NON_EQ)
        #trajectory_initial_condition_function = make_trajectory_initial_position_estimator(np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH),TIME_STEP_NON_EQ)
        
        mean_initial_positions = get_or_compute_initial_positions(params_bath_model, t_second_measurement, SAVE_DIR_2_STATE_INIT, mean_initial_positions_from_analytical,method_name="analytical", force_recompute=True)
        print(mean_initial_positions)
        for t_protocol_end in t_protocol_end_values:
            results = process_combination( mean_initial_positions,t_second_measurement, t_protocol_end, params_bath_model, SAVE_DIR_OPTIMAL_PROTOCOLS,PLOT_DIR_OPTIMAL_PROTOCOLS, make_plots=False)#use_matlab_initial_conditions=True)
            all_results.update(results)
            plot_optimal_trajectories_grid(
                t_second_measurement, t_protocol_end,
                save_dir=SAVE_DIR_OPTIMAL_PROTOCOLS,
                save_path=f"{PLOT_DIR_OPTIMAL_PROTOCOLS}/optimal_trajectories_grid_t2_{t_second_measurement}_tf_{t_protocol_end}.png",
                n_outcomes=2, matlab_protocol_dir= "matlab_protocols/2state"
            )
            plot_accumulated_work(
                t_second_measurement, t_protocol_end,
                save_dir=SAVE_DIR_OPTIMAL_PROTOCOLS,
                save_path=f"{PLOT_DIR_OPTIMAL_PROTOCOLS}/accumulated_work_grid_t2_{t_second_measurement}_tf_{t_protocol_end}.png", params =params_bath_model,
                n_outcomes=2
            )

    plot_work_comparison(all_results, params_bath_model, f"{PLOT_DIR_OPTIMAL_PROTOCOLS}/work_comparison.png")
    plot_work_gap(all_results, params_bath_model, f"{PLOT_DIR_OPTIMAL_PROTOCOLS}/work_potential_ratio.png")
    print("\nSummary of mean work over all combinations:")
    for (t2, tf, m_0, m_t), work_summary in all_results.items():
        print(
        f"  t2={t2}, tf={tf}, m_0={m_0}, m_t={m_t}: "
        f"optimal={(work_summary['optimal']/kBT_scale):.4g}, "
        f"optimal_check={(work_summary['optimal_check']/kBT_scale):.4g}, "
        f"jump={(work_summary['jump']/kBT_scale):.4g}"
        f"-V_trap ={(work_summary['-V_trap']/kBT_scale):.4g}"
    )

if __name__ == "__main__": #Only execute main if program is executed itself, not when imported
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
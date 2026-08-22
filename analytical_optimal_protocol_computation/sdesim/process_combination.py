from .analytics import get_cvector, SolutionMatrixCreator, negative_trap_potential, compute_D, mean_work, get_cvector_numeric, compute_work_general_protocol, current_lambda
from .visualization import plot_optimal_trajectories
from. helpers import parse_m_values, compute_optimal_trajectories_array, make_save_path
import numpy as np

def process_combination(mean_initial_positions,t_second_measurement, t_protocol_end, params,
                         save_dir, plot_dir, make_plots=True):
    """Compute optimal protocols and corresponding work (as well as reference works) for all possible measurement outcome combinations at given times, save and return them
    
    Parameters: 
    mean_initial_positions: dictionary containing mean initial conditions for each measurement combination
    t_second_measurement: float, time betweeen measurements
    t_protocol_end: float, duration of the protocol
    params: SystemVariables
    save_dir: string, Path to folder where optimal trajectories and specifics like work are saved
    plot_dir: string, Path to folder where plots of optimal protocols are saved
    make_plots: Boolean, determining whether to make any plots at all

    return: dictionary containing optimal work (computed in two ways) and reference of mean potential energy and jump protocol, saves more information 
    """

    B_creator = SolutionMatrixCreator(params)
    D = compute_D(B_creator, t_protocol_end, params)

    m_keys = list(mean_initial_positions.keys())
    m_combinations = {key: parse_m_values(key) for key in m_keys}

    results = {}

    for key in m_keys:
        m_0, m_t = m_combinations[key]

        mean_X_t = mean_initial_positions[key]["mean_X_t"]
        mean_X_b_t = mean_initial_positions[key]["mean_X_bt"]
        mean_X_t_squared = mean_initial_positions[key]["mean_X_t_squared"]

        if np.isnan(mean_X_t) or np.isnan(mean_X_b_t):
            print(f"Skipping m_0={m_0}, m_t={m_t}: no samples in this category.")
            continue

        c_vector, lambda_f = get_cvector(mean_X_t, mean_X_b_t, D)
        work_optimal = mean_work(D, c_vector, lambda_f)

        #print(f"analytic cvector: {c_vector}, lambdaf: {lambda_f}")
        #Sprint(f"numeric cvector: {get_cvector_numeric(mean_X_t,mean_X_b_t,D)}")

        # cross-check: integrate directly along the optimal trajectory
        def lambda_optimal(t, c_vector=c_vector):
            B = B_creator.current_matrix(t)
            return current_lambda(B, c_vector, params)

        work_optimal_check = compute_work_general_protocol(mean_X_t, mean_X_b_t, lambda_optimal, t_protocol_end, params, lambda_f=lambda_f)

        pot_energy = negative_trap_potential(params, mean_X_t_squared)
        # comparison protocol: jump straight to the mean given measurements and hold
        work_jump = compute_work_general_protocol(mean_X_t, mean_X_b_t, lambda_func=lambda t: mean_X_t, t_protocol_end=t_protocol_end, params=params)

        optimal_trajectories = compute_optimal_trajectories_array(B_creator, t_protocol_end, c_vector, lambda_f)

        work_summary = {
            "optimal": work_optimal,
            "optimal_check": work_optimal_check,
            "jump": work_jump,
            "-V_trap": pot_energy
        }

        save_path = make_save_path(save_dir, t_second_measurement, t_protocol_end, m_0, m_t)
        np.savez(
            save_path,
            optimal_trajectories=optimal_trajectories,
            mean_work=work_optimal,
            mean_work_check=work_optimal_check,
            mean_work_jump=work_jump,
            potential_reference = pot_energy,
            c_vector=c_vector, lambda_f=lambda_f,
            t_second_measurement=t_second_measurement, t_protocol_end=t_protocol_end,
            m_0=m_0, m_t=m_t,
        )

        if make_plots:
            plot_path = make_save_path(plot_dir, t_second_measurement, t_protocol_end, m_0, m_t, extension="png")
            plot_optimal_trajectories(
                optimal_trajectories,
                f"Optimal trajectories $m_0={int(m_0)}$, $m_t={int(m_t)}$, $\\Delta t={t_second_measurement}$, $t_f={t_protocol_end}$",
                plot_path,
            )

        results[(t_second_measurement, t_protocol_end, m_0, m_t)] = work_summary

        print(
            f"t2={t_second_measurement}, t_protocol_end={t_protocol_end}, m_0={m_0}, m_t={m_t}: "
            f"work_optimal={work_optimal:.4g} (check={work_optimal_check:.4g}), "
            f"work_jump={work_jump:.4g} -> saved to {save_path}"
        )

    return results
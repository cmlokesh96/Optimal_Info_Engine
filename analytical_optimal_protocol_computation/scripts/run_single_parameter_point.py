"""
Run the optimal-protocol pipeline for a single, manually specified
parameter combination (kappa, x_thresh, t_second_measurement,
t_protocol_end), and print a detailed per-outcome work summary.

Useful for quickly inspecting one point of interest (e.g. one flagged by
run_optimal_protocol_sweep.py, or just a guess) without running the full
sweep.
"""
import time
from scipy import constants

from sdesim.constants import SAVE_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT, PLOT_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT
from sdesim.params import params_bath_model, energy_scale, SystemVariables
from sdesim.visualization import setup_matplotlib
from sdesim.parameter_sweep import ParameterPoint, run_single_parameter_point


def main():
    setup_matplotlib()
    

    # --- Specify the parameter point to run here ---
    x_thresh_sigma_multiple=1
    params =  SystemVariables.create(kappa=2.4e-6, kappa_b=1e-6, gamma=0.34e-6, gamma_b =15e-6, T=298, k_B= constants.Boltzmann, x_thresh_sigma_multiple=x_thresh_sigma_multiple)
    
    t_second_measurement = 0
    t_protocol_end = 3.0
    n_outcomes = 3
    reference_key = "jump_second_only"  # for the printed ratio: "jump_both", "jump_second_only", or "-V_trap"
    kT = energy_scale(params)

    #Shorthandds
    kappa = params.kappa
    

    point = ParameterPoint(kappa, x_thresh_sigma_multiple, t_second_measurement, t_protocol_end)
    print(f"Running full pipeline for: {point}")

    detailed_results = run_single_parameter_point(
        point, params,
        save_dir=SAVE_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT,
        plot_dir=PLOT_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT,
        n_outcomes=n_outcomes, make_plots=True,
    )

    print("\nDetailed work summary (per outcome):")
    weighted_work_sum, prob_sum = 0.0, 0.0
    weighted_numerator_sum, weighted_denominator_sum = 0.0, 0.0

    for key, summary in detailed_results.items():
        probability = summary["probability"]
        optimal = summary["optimal"]
        jump_both = summary["jump_both"]
        jump_second_only = summary["jump_second_only"]
        neg_V_trap = summary["-V_trap"]
        reference_value = summary[reference_key]

        if reference_value == 0 or reference_value != reference_value:  # nan check
            ratio_str = "n/a"
        else:
            ratio_str = f"{optimal/reference_value:.4g}"

        print(
            f"  {key}: P={probability:.4g}, "
            f"optimal={optimal/kT:.4g} kBT, "
            f"jump_both={jump_both/kT:.4g} kBT, "
            f"jump_second_only={jump_second_only/kT:.4g} kBT, "
            f"-V_trap={neg_V_trap/kT:.4g} kBT, "
            f"per-outcome ratio(optimal/{reference_key})={ratio_str}"
        )

        weighted_work_sum += probability * optimal
        prob_sum += probability
        weighted_numerator_sum += probability * optimal
        weighted_denominator_sum += probability * reference_value

    print("\nProbability-weighted aggregates over outcomes:")
    if prob_sum > 0:
        print(f"  weighted_mean_work     = {weighted_work_sum/prob_sum/kT:.4g} kBT")
    print(f"  weighted_ratio_of_means = {weighted_numerator_sum/weighted_denominator_sum:.4g}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
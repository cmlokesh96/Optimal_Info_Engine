"""
Sweep (kappa, x_thresh, t_second_measurement, t_protocol_end), flag the
parameter combination optimizing a chosen criterion, and re-run/plot that
one combination in full detail.
"""
import time

from sdesim.constants import SAVE_DIR_OPTIMAL_PROTOCOLS, PLOT_DIR_OPTIMAL_PROTOCOLS
from sdesim.params import params_bath_model, energy_scale
from sdesim.visualization import setup_matplotlib
from sdesim.parameter_sweep import (
    SweepAxis, run_parameter_sweep, find_best_parameter_point, run_single_parameter_point,
)


def run_and_report_detail(point, label, params, kT, reference_key="jump_second_only"):
    """
    Re-run the full pipeline for one flagged parameter point and print,
    for each individual (m_0, m_t) outcome, the work quantities and the
    work/reference ratio - plus a sanity-check recomputation of the
    probability-weighted aggregates from those per-outcome rows.
    """
    print(f"\nRunning full pipeline for {label}: {point}")
    detailed_results = run_single_parameter_point(
        point, params,
        save_dir=SAVE_DIR_OPTIMAL_PROTOCOLS, plot_dir=PLOT_DIR_OPTIMAL_PROTOCOLS,
        n_outcomes=2, make_plots=True,
    )

    print(f"\nDetailed work summary for {label} (per outcome):")
    weighted_work_sum, prob_sum = 0.0, 0.0
    weighted_ratio_sum, weighted_ratio_prob_sum = 0.0, 0.0

    for key, summary in detailed_results.items():
        probability = summary["probability"]
        optimal = summary["optimal"]
        jump_both = summary["jump_both"]
        jump_second_only = summary["jump_second_only"]
        neg_V_trap = summary["-V_trap"]

        reference_value = summary[reference_key]
        if reference_value == 0 or reference_value != reference_value:  # nan check
            ratio_str, ratio = "n/a", None
        else:
            ratio = optimal / reference_value
            ratio_str = f"{ratio:.4g}"

        print(
            f"  {key}: P={probability:.4g}, "
            f"optimal={optimal/kT:.4g} kBT, "
            f"jump_both={jump_both/kT:.4g} kBT, "
            f"jump_second_only={jump_second_only/kT:.4g} kBT, "
            f"-V_trap={neg_V_trap/kT:.4g} kBT, "
            f"ratio(optimal/{reference_key})={ratio_str}"
        )

        weighted_work_sum += probability * optimal
        prob_sum += probability
        if ratio is not None:
            weighted_ratio_sum += probability * ratio
            weighted_ratio_prob_sum += probability

    print(f"\nSanity check: recomputed weighted aggregates for {label}:")
    if prob_sum > 0:
        print(f"  weighted_mean_work  = {weighted_work_sum/prob_sum/kT:.4g} kBT")
    if weighted_ratio_prob_sum > 0:
        print(f"  weighted_mean_ratio = {weighted_ratio_sum/weighted_ratio_prob_sum:.4g}")

    return detailed_results


def main():
    setup_matplotlib()
    kT = energy_scale(params_bath_model)

    # Determine which parameters are swept by a SweepAxis object with entries stat and end of the 
    # swept axis and anoptimal parameter for the spacing of the tested values, default 10% of the domain
    # e.g. 11 swept values (initial values and then ten steps to the end)
    # start==end leads to fix the Axis at this value and not sweep it
    kappa_axis = SweepAxis(0.25 * params_bath_model.kappa, 8*params_bath_model.kappa)
    x_thresh_axis = SweepAxis(0.5*params_bath_model.x_thresh, 5*params_bath_model.x_thresh)
    t_second_measurement_axis = SweepAxis(0.1, 8)          
    t_protocol_end_axis = SweepAxis(1.0, 8.0)      

    print("Running sweep...")
    sweep_results = run_parameter_sweep(
        kappa_axis, x_thresh_axis, t_second_measurement_axis, t_protocol_end_axis,
        base_params=params_bath_model, n_outcomes=2,
    )
    print(f"Evaluated {len(sweep_results)} parameter combinations.")

    # --- Flag the best point under two different criteria ---
    best_by_work, best_work_value, _ = find_best_parameter_point(
        sweep_results, metric="weighted_mean_work", work_key="optimal", maximize=False,
        # maximize=False: for work, "best" means *lowest* extracted work.
        # Flip to True if "highest mean work" (as literally stated) is what you want.
    )
    print(f"\nLowest probability-weighted mean optimal work:")
    print(f"  {best_by_work} -> {best_work_value/kT:.4g} kBT")

    best_by_ratio, best_ratio_value, _ = find_best_parameter_point(
        sweep_results, metric="weighted_mean_ratio", reference_key="jump_second_only", maximize=True,
    )
    print(f"\nBest (highest) probability-weighted mean work/jump_second_only ratio:")
    print(f"  {best_by_ratio} -> {best_ratio_value:.4g}")

    # --- Re-run and plot both flagged points in full detail ---
    run_and_report_detail(best_by_work, "the point flagged best by work", params_bath_model, kT)
    run_and_report_detail(best_by_ratio, "the point flagged best by ratio", params_bath_model, kT)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
"""
run_single_parameter_point_cli.py
==================================
Command-line, fully-parameterized twin of run_single_parameter_point.py -
same pipeline (imports run_single_parameter_point() from
sdesim.parameter_sweep unchanged, doesn't touch that script or any other
existing file), but every physical parameter that script hardcodes is a
CLI argument here instead, so a caller (a terminal, run manually with
analytical_optimal_protocol_computation/.venv activated) can generate a
protocol set for an exact, explicit parameter point without editing this
repo's files.

Example (run from analytical_optimal_protocol_computation/, .venv
activated - same "-m scripts.<name>" convention run_single_parameter_point.py
itself needs, since sdesim isn't pip-installed and only resolves via cwd):
    python -m scripts.run_single_parameter_point_cli ^
        --kappa 2.4e-6 --gamma 0.34e-6 --T 298 --x_thresh_sigma_multiple 1 ^
        --t_second_measurement 0.5 --t_protocol_end 3.0 ^
        --save_dir "D:/Data/Aug_26/26-08_Testing_Full_Engine/batch01/analytical_protocols_used"

--save_dir is NOT namespaced by kappa/x_thresh (only by
t_second_measurement/t_protocol_end/m_0/m_t - see
run_single_parameter_point()'s own docstring) - point it at somewhere
specific to this parameter point/batch (not the shared
optimal_protocols_multiple_measurement folder used by
run_single_parameter_point.py/run_parameter_sweep_3state.py) so a later
sweep run there can't silently overwrite these files out from under you.
"""
import argparse
import os
import time

from scipy import constants

from sdesim.params import energy_scale, SystemVariables
from sdesim.visualization import setup_matplotlib
from sdesim.parameter_sweep import ParameterPoint, run_single_parameter_point


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kappa", type=float, required=True, help="trap stiffness, N/m")
    parser.add_argument("--gamma", type=float, required=True, help="drag coefficient, N*s/m")
    parser.add_argument("--kappa_b", type=float, default=1e-6, help="bath-mode stiffness, N/m")
    parser.add_argument("--gamma_b", type=float, default=15e-6, help="bath-mode drag, N*s/m")
    parser.add_argument("--T", type=float, required=True, help="temperature, K")
    parser.add_argument("--x_thresh_sigma_multiple", type=float, required=True,
                        help="decode threshold, in multiples of sigma = sqrt(kB*T/kappa)")
    parser.add_argument("--t_second_measurement", type=float, required=True,
                        help="delta_t_s: spacing between the two decision samples, s")
    parser.add_argument("--t_protocol_end", type=float, required=True,
                        help="protocol_dt_s: duration of the fired protocol waveform, s")
    parser.add_argument("--n_outcomes", type=int, default=3)
    parser.add_argument("--reference_key", default="jump_second_only",
                        choices=["jump_both", "jump_second_only", "-V_trap"],
                        help="which quantity the printed per-outcome ratio is against")
    parser.add_argument("--save_dir", required=True,
                        help="where the .npz protocol files land - NOT namespaced by "
                             "kappa/x_thresh, point this at something specific to this "
                             "parameter point/batch, not the shared solver output folder")
    parser.add_argument("--plot_dir", default=None, help="defaults to save_dir if omitted")
    parser.add_argument("--no_plots", action="store_true",
                        help="skip make_plots (faster, no plot files written)")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_dir = args.plot_dir or args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    setup_matplotlib()

    params = SystemVariables.create(
        kappa=args.kappa, kappa_b=args.kappa_b, gamma=args.gamma, gamma_b=args.gamma_b,
        T=args.T, k_B=constants.Boltzmann, x_thresh_sigma_multiple=args.x_thresh_sigma_multiple,
    )
    kT = energy_scale(params)

    point = ParameterPoint(params.kappa, args.x_thresh_sigma_multiple,
                           args.t_second_measurement, args.t_protocol_end)
    print(f"Running full pipeline for: {point}")
    print(f"save_dir = {args.save_dir}")
    print(f"plot_dir = {plot_dir}")

    detailed_results = run_single_parameter_point(
        point, params,
        save_dir=args.save_dir, plot_dir=plot_dir,
        n_outcomes=args.n_outcomes, make_plots=not args.no_plots,
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
        reference_value = summary[args.reference_key]

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
            f"per-outcome ratio(optimal/{args.reference_key})={ratio_str}"
        )

        weighted_work_sum += probability * optimal
        prob_sum += probability
        weighted_numerator_sum += probability * optimal
        weighted_denominator_sum += probability * reference_value

    print("\nProbability-weighted aggregates over outcomes:")
    if prob_sum > 0:
        print(f"  weighted_mean_work     = {weighted_work_sum/prob_sum/kT:.4g} kBT")
    if weighted_denominator_sum:
        print(f"  weighted_ratio_of_means = {weighted_numerator_sum/weighted_denominator_sum:.4g}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")

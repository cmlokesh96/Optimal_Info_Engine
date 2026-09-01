"""
fit_bath_parameters_from_calib.py
==================================
Fits the tracer drag (gamma) and hidden-bath parameters (kappa_b, gamma_b)
against a real, undisturbed equilibrium recording - the calib_pos_px.npy
files calibrate_trap_center() saves - by matching the MEASURED position
autocorrelation <X(0)X(t)> to sdesim's own closed-form
sdesim.analytics.bath_correlation_C(t, params) via iterative nonlinear
least squares (scipy.optimize.curve_fit).

Trap stiffness kappa is NOT fit here - pass it in already fitted (from
analysis.compute_potential's k_fit_N_per_m, computed in the main notebook/
repo, not this venv). Equilibrium marginal variance alone (which is all
compute_potential can see) carries zero information about gamma/kappa_b/
gamma_b - those only show up in how the position DECORRELATES over time,
which is exactly what this script measures and fits against.

Runs the SAME raw-trace autocorrelation for every lag (not one curve per
decoded (m_0, m_t) state) - by the regression-to-the-mean property of this
linear/Gaussian model, E[X(t)|state=s] = E[X(0)|state=s] * chi(t) for
EVERY state s, i.e. every state-conditioned relaxation curve is just this
same chi(t) shape rescaled by that state's own mean - so the raw-trace
autocorrelation already IS the curve every state's relaxation would trace
out, just estimated far more efficiently (uses every sample, not only the
ones that happen to fall in one state's decode bin).

Anharmonicity check built in: fits TWICE per run - once restricted to the
"harmonic core" (both endpoints of a lagged pair inside U/kT < u_over_kt_cutoff,
same convention/default as analysis.compute_potential's own fit_mask) and
once completely unmasked - so you can see directly, from one run, how much
the non-harmonic tail biases gamma/kappa_b/gamma_b if it's included.

Self-contained (numpy/scipy/sdesim only, same convention as
analytical_protocols.py/run_single_parameter_point_cli.py) - does NOT
import camera.py/analysis.py from the main repo (different venv), so
nm_per_px/x_center_nm/kappa are passed in as plain CLI floats instead.

Example (run from analytical_optimal_protocol_computation/, .venv
activated, same "-m scripts.<name>" convention every other script here
needs - sdesim isn't pip-installed, only resolves via cwd):
    python -m scripts.fit_bath_parameters_from_calib ^
        --calib_folder "D:/Data/Aug_26/Exp_1_20260831_1730/calib" ^
        --nm_per_px 16 --x_center_nm 1800.83 --fps 100 ^
        --kappa 2.1e-6 --T 298 --t_max 2.0 --save_dir "D:/Data/Aug_26/Exp_1_20260831_1730/bath_fit"
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from scipy import constants
from scipy.optimize import curve_fit

from sdesim.analytics import chi
from sdesim.params import SystemVariables
from sdesim.visualization import setup_matplotlib


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib_folder", required=True,
                        help="folder of *_calib_pos_px.npy files, as saved by "
                             "experiment.calibrate_trap_center(savepath=...)")
    parser.add_argument("--nm_per_px", type=float, required=True,
                        help="pixel -> nm scale factor (camera.NM_PER_PX)")
    parser.add_argument("--x_center_nm", type=float, required=True,
                        help="trap center, nm (section 3's x_center_nm)")
    parser.add_argument("--fps", type=float, default=100.0,
                        help="camera frame rate calib_pos_px was recorded at")
    parser.add_argument("--kappa", type=float, required=True,
                        help="ALREADY-fitted trap stiffness, N/m (analysis."
                             "compute_potential's k_fit_N_per_m) - held fixed, not fit here")
    parser.add_argument("--T", type=float, default=298.0, help="temperature, K")
    parser.add_argument("--u_over_kt_cutoff", type=float, default=6.0,
                        help="harmonic-core radius, in U/kT - same convention/default "
                             "as analysis.compute_potential's own fit_mask")
    parser.add_argument("--t_max", type=float, default=2.0,
                        help="largest lag to fit, s")
    parser.add_argument("--n_lags", type=int, default=60,
                        help="number of lag points between 0 and t_max")
    parser.add_argument("--gamma0", type=float, default=0.34e-6,
                        help="initial guess, tracer drag, N*s/m")
    parser.add_argument("--kappa_b0", type=float, default=1e-6,
                        help="initial guess, bath-mode stiffness, N/m")
    parser.add_argument("--gamma_b0", type=float, default=15e-6,
                        help="initial guess, bath-mode drag, N*s/m")
    parser.add_argument("--save_dir", default=None,
                        help="if given, saves the fitted params (.json) and the "
                             "comparison plot (.png) here - otherwise just prints/shows")
    parser.add_argument("--no_plots", action="store_true", help="skip plt.show()/savefig")
    return parser.parse_args()


def load_x_m_per_file(calib_folder, nm_per_px, x_center_nm):
    """
    One array per *_calib_pos_px.npy file, meters, trap-centered - kept
    separate per file (not concatenated) so a lagged pair never straddles
    two unrelated recording runs (column 2/time_s is each file's own
    relative time, not continuous across files - see
    analysis.load_combined_calib_pos_px's docstring).
    """
    paths = sorted(glob.glob(os.path.join(calib_folder, "*_calib_pos_px.npy")))
    if not paths:
        raise FileNotFoundError(f"No *_calib_pos_px.npy files found in {calib_folder!r}")
    x_by_file = []
    for path in paths:
        pos_px = np.load(path)
        x_nm = pos_px[:, 0] * nm_per_px - x_center_nm
        x_by_file.append(x_nm.astype(float) * 1e-9)   # nm -> m (SI, matching params)
    return x_by_file, paths


def accumulate_lag_sums(x_m, lag_frames, x_max_m):
    """
    (sum of x(i)*x(i+lag), count) for one file at one lag, restricted to
    pairs where BOTH endpoints lie inside the harmonic core (|x| < x_max_m,
    the same U/kT < u_over_kt_cutoff region analysis.compute_potential
    fits its own parabola to) - a large excursion at EITHER endpoint is
    excluded, not just the conditioning one, since either can bias the
    lagged product away from what the idealized harmonic/OU model predicts.
    x_max_m=None disables the mask entirely (the unmasked comparison fit).
    """
    if lag_frames == 0:
        a = b = x_m
    else:
        a, b = x_m[:-lag_frames], x_m[lag_frames:]
    valid = np.isfinite(a) & np.isfinite(b)
    if x_max_m is not None:
        valid &= (np.abs(a) < x_max_m) & (np.abs(b) < x_max_m)
    if not np.any(valid):
        return 0.0, 0
    return float(np.sum(a[valid] * b[valid])), int(np.sum(valid))


def measure_correlation_curve(x_by_file, lags_frames, x_max_m):
    """Pooled <X(0)X(t)> and surviving-pair count per lag, across every file."""
    C = np.empty(len(lags_frames))
    N = np.empty(len(lags_frames), dtype=int)
    for li, lag in enumerate(lags_frames):
        total_sum, total_n = 0.0, 0
        for x_m in x_by_file:
            s, n = accumulate_lag_sums(x_m, lag, x_max_m)
            total_sum += s
            total_n += n
        C[li] = total_sum / total_n if total_n else np.nan
        N[li] = total_n
    return C, N


def fit_bath_params(t_lags, C_measured, N, kappa, T, k_B, p0):
    """
    Iteratively refine (gamma, kappa_b, gamma_b) so sdesim's own closed-form
    bath_correlation_C(t, params) matches the measured curve - kappa/T/k_B
    held fixed. Lags with fewer surviving pairs (after the harmonic-core
    mask) are weighted down (sigma ~ 1/sqrt(N), the usual mean-estimator
    scaling) rather than trusted equally to well-sampled ones.

    Returns (popt, perr, model) - popt = [gamma, kappa_b, gamma_b], perr =
    1-sigma parameter uncertainties (sqrt of the covariance diagonal), model
    = the fitted curve function C(t) in raw physical units, m^2 (for
    plotting/residuals).
    """
    C0 = k_B * T / kappa   # = bath_correlation_C(0, params) for ANY (gamma, kappa_b, gamma_b) - chi(0)=1 always, exactly

    def chi_model(t, gamma, kappa_b, gamma_b):
        params = SystemVariables.create(
            gamma=gamma, gamma_b=gamma_b, kappa=kappa, kappa_b=kappa_b,
            T=T, k_B=k_B, x_thresh_sigma_multiple=1.0,   # irrelevant to chi
        )
        return chi(params, t)

    # Fit the DIMENSIONLESS chi(t) = C(t)/C(0), not raw C(t) directly -
    # curve_fit's `trf` method (needed here for the positivity bounds
    # below; the unconstrained `lm` method doesn't support bounds) checks
    # gradient-convergence on an ABSOLUTE, not relative, scale. Raw C(t) is
    # ~1e-15 (m^2, typical trapped-colloid equilibrium variance), which
    # falsely satisfies that check after a single evaluation WITHOUT
    # actually moving away from p0 at all - confirmed against synthetic
    # data with known true parameters (curve_fit silently returned p0
    # unchanged instead of recovering the true values). chi(t) is O(1),
    # which sidesteps this scaling failure entirely.
    sigma = 1.0 / np.sqrt(np.maximum(N, 1))   # same relative lag-weighting either way (chi(0)=1 for all params)
    popt, pcov = curve_fit(
        chi_model, t_lags, C_measured / C0, p0=p0, bounds=(0.0, np.inf),
        sigma=sigma, absolute_sigma=False, maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    def model(t, gamma, kappa_b, gamma_b):
        return chi_model(t, gamma, kappa_b, gamma_b) * C0

    return popt, perr, model


def main():
    args = parse_args()
    setup_matplotlib()
    import matplotlib.pyplot as plt

    k_B = constants.Boltzmann
    x_max_m = np.sqrt(2 * args.u_over_kt_cutoff * k_B * args.T / args.kappa)

    x_by_file, paths = load_x_m_per_file(args.calib_folder, args.nm_per_px, args.x_center_nm)
    n_total = sum(len(x) for x in x_by_file)
    print(f"Loaded {len(paths)} file(s), {n_total} total frames, from {args.calib_folder}")
    print(f"Harmonic core: |x| < {x_max_m*1e9:.1f} nm  (U/kT < {args.u_over_kt_cutoff:g}, kappa = {args.kappa:.4e} N/m)")

    dt = 1.0 / args.fps
    lags_frames = np.unique(np.round(np.linspace(0.0, args.t_max, args.n_lags) / dt).astype(int))
    t_lags = lags_frames * dt

    C_masked, N_masked = measure_correlation_curve(x_by_file, lags_frames, x_max_m)
    C_unmasked, N_unmasked = measure_correlation_curve(x_by_file, lags_frames, None)

    # t=0 carries no information about (gamma, kappa_b, gamma_b) - chi(0)=1
    # always, by construction, regardless of those three - so it's excluded
    # from the fit itself and reported only as a standalone consistency
    # check: measured C(0) should closely match kB*T/kappa (the SAME kappa
    # passed in from compute_potential), independent of anything fit below.
    print(f"\nC(0) cross-check (masked):   measured = {C_masked[0]:.4e} m^2   "
          f"kB*T/kappa = {k_B*args.T/args.kappa:.4e} m^2")

    fit_mask = lags_frames > 0
    p0 = [args.gamma0, args.kappa_b0, args.gamma_b0]

    print("\nFitting harmonic-core-masked curve...")
    popt_m, perr_m, model = fit_bath_params(
        t_lags[fit_mask], C_masked[fit_mask], N_masked[fit_mask], args.kappa, args.T, k_B, p0)

    print("Fitting unmasked (full-range) curve for comparison...")
    popt_u, perr_u, _ = fit_bath_params(
        t_lags[fit_mask], C_unmasked[fit_mask], N_unmasked[fit_mask], args.kappa, args.T, k_B, p0)

    def report(label, popt, perr, C_data):
        gamma, kappa_b, gamma_b = popt
        params = SystemVariables.create(gamma=gamma, gamma_b=gamma_b, kappa=args.kappa,
                                        kappa_b=kappa_b, T=args.T, k_B=k_B)
        residual = C_data[fit_mask] - model(t_lags[fit_mask], *popt)
        rel_rms = float(np.sqrt(np.mean(residual**2)) / C_data[0])
        print(f"\n{label}:")
        print(f"  gamma    = {gamma:.4e} +/- {perr[0]:.2e}  N*s/m")
        print(f"  kappa_b  = {kappa_b:.4e} +/- {perr[1]:.2e}  N/m")
        print(f"  gamma_b  = {gamma_b:.4e} +/- {perr[2]:.2e}  N*s/m")
        print(f"  tau_p = gamma/kappa_b = {params.tau_p:.4e} s   tau_b = gamma_b/kappa_b = {params.tau_b:.4e} s")
        print(f"  relative RMS residual (of C(0)) = {rel_rms:.3%}")
        return rel_rms

    rel_rms_m = report("Harmonic-core-masked fit", popt_m, perr_m, C_masked)
    rel_rms_u = report("Unmasked fit", popt_u, perr_u, C_unmasked)
    print(f"\nNon-harmonic contamination check: masked fit's relative RMS residual is "
          f"{'LOWER' if rel_rms_m < rel_rms_u else 'HIGHER'} than the unmasked fit's "
          f"({rel_rms_m:.3%} vs {rel_rms_u:.3%}) - "
          + ("consistent with the tail biasing the unmasked fit, as expected."
             if rel_rms_m < rel_rms_u else
             "no sign of tail contamination at this cutoff/T/n_lags - the two fits agree."))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        result = {
            "kappa_N_per_m": args.kappa, "T_K": args.T,
            "u_over_kt_cutoff": args.u_over_kt_cutoff,
            "masked": {"gamma": popt_m[0], "kappa_b": popt_m[1], "gamma_b": popt_m[2],
                       "perr": perr_m.tolist(), "rel_rms_residual": rel_rms_m},
            "unmasked": {"gamma": popt_u[0], "kappa_b": popt_u[1], "gamma_b": popt_u[2],
                         "perr": perr_u.tolist(), "rel_rms_residual": rel_rms_u},
        }
        out_path = os.path.join(args.save_dir, "fitted_bath_params.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved fitted params: {out_path}")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t_lags, C_unmasked, "o", ms=4, color="lightgray", label="measured (unmasked)")
        ax.plot(t_lags, C_masked, "o", ms=4, color="steelblue", label="measured (harmonic core)")
        t_fine = np.linspace(t_lags[1], t_lags[-1], 300)
        ax.plot(t_fine, model(t_fine, *popt_u), "--", color="gray", label="fit (unmasked)")
        ax.plot(t_fine, model(t_fine, *popt_m), "-", color="tomato", label="fit (harmonic core)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(r"$\langle X(0)X(t)\rangle$ (m$^2$)")
        ax.set_title(f"Equilibrium relaxation fit ({args.calib_folder})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if args.save_dir:
            fig_path = os.path.join(args.save_dir, "bath_fit.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Saved plot: {fig_path}")
        plt.show()


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")

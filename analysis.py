"""
analysis.py
===========
Pure numpy/matplotlib — no hardware. Offline analysis of recorded particle
trajectories.

The potential U(x) (and any later work/thermodynamic-quantities
calculation) stays in kT units — standard for this kind of analysis, and
means the potential/work numbers don't depend on a temperature assumption.
Stiffness k is the one quantity reported in physical units (N/m) as well,
since "how stiff is the trap" is more useful compared against a real
number than left as a kT-normalized ratio — that conversion needs a
temperature (default 298 K, override via T_K).
"""

import numpy as np
import matplotlib.pyplot as plt

K_BOLTZMANN_J_PER_K = 1.380649e-23   # exact SI value


def _k_over_kT_to_N_per_m(k_over_kT_per_nm2: float, T_K: float) -> float:
    """(k/kT) in 1/nm^2 -> k in N/m, at temperature T_K."""
    kT_J = K_BOLTZMANN_J_PER_K * T_K
    return k_over_kT_per_nm2 * 1e18 * kT_J   # 1/nm^2 -> 1/m^2, then x kT[J] -> N/m


def compute_potential(x_nm: np.ndarray, n_bins: int = 60, T_K: float = 298.0) -> dict:
    """
    Boltzmann-invert a recorded x(t) trace into the trap potential U(x)/kT
    (stays in kT units — no temperature needed for this part), and estimate
    the trap stiffness two ways:

        k_fit_over_kT — from a parabolic fit to U(x)/kT = 0.5*(k/kT)*(x-x0)^2
        k_eq_over_kT  — from equipartition: k/kT = 1 / Var(x)

    Reported alongside each other as a cross-check — a mismatch flags
    anharmonicity, drift, or binning issues rather than a real disagreement.
    Both are also converted to physical stiffness in N/m (k_fit_N_per_m,
    k_eq_N_per_m) using T_K (default 298 K — override for your actual
    experimental temperature); only this conversion depends on T_K, U/kT
    itself does not.

    x_nm should be a real equilibrium stretch (particle undisturbed, no
    protocol firing) — e.g. from run_session_2ch(..., active_states=set()).

    Returns a dict consumed by plot_potential().
    """
    x = np.asarray(x_nm, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("compute_potential: not enough valid (non-NaN) samples")

    counts, edges = np.histogram(x, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    valid = counts > 0
    x_valid = centers[valid]
    P = counts[valid] / (counts.sum() * bin_width)
    U_raw = -np.log(P)

    # Fit on the raw (unshifted) values — an additive constant doesn't
    # affect the quadratic/linear coefficients (stiffness, x0).
    a, b, c = np.polyfit(x_valid, U_raw, 2)
    x0_fit_nm = -b / (2 * a)
    fit_offset = a * x0_fit_nm**2 + b * x0_fit_nm + c   # fit's value at its own vertex

    U_over_kT = U_raw - fit_offset   # shift so the fit's minimum sits at 0
    k_fit_over_kT = 2 * a            # 1/nm^2
    k_eq_over_kT = 1.0 / np.var(x, ddof=1)   # 1/nm^2

    return {
        "bin_centers_nm": x_valid,
        "U_over_kT": U_over_kT,
        "fit_coeffs": (a, b, c),
        "fit_offset": fit_offset,
        "x0_fit_nm": x0_fit_nm,
        "k_fit_over_kT": k_fit_over_kT,
        "k_eq_over_kT": k_eq_over_kT,
        "k_fit_N_per_m": _k_over_kT_to_N_per_m(k_fit_over_kT, T_K),
        "k_eq_N_per_m": _k_over_kT_to_N_per_m(k_eq_over_kT, T_K),
        "T_K": T_K,
        "n_samples": len(x),
    }


def plot_potential(result: dict) -> None:
    """Plot U(x)/kT vs x with the parabolic fit overlaid, per compute_potential()."""
    x = result["bin_centers_nm"]
    U = result["U_over_kT"]
    a, b, c = result["fit_coeffs"]
    offset = result["fit_offset"]

    x_fit = np.linspace(x.min(), x.max(), 200)
    U_fit = a * x_fit**2 + b * x_fit + c - offset

    plt.figure(figsize=(7, 5))
    plt.plot(x, U, "o", ms=4, color="steelblue", label="U/kT = -ln P(x)")
    plt.plot(x_fit, U_fit, "-", color="tomato",
             label=f"parabolic fit (k = {result['k_fit_N_per_m']:.3e} N/m)")
    plt.axvline(result["x0_fit_nm"], color="gray", lw=0.8, ls="--")
    plt.xlabel("x (nm)")
    plt.ylabel("U(x) / kT")
    plt.title(f"Trap potential  (n={result['n_samples']}, T={result['T_K']:.0f} K)\n"
              f"k_fit = {result['k_fit_N_per_m']:.3e} N/m   "
              f"k_eq = {result['k_eq_N_per_m']:.3e} N/m")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Offline self-test: synthetic Gaussian x(t) with known variance —
    # k_fit_over_kT and k_eq_over_kT should both recover 1/variance.
    rng = np.random.default_rng(0)
    true_var_nm2 = 400.0   # e.g. sigma = 20 nm
    x_synth = rng.normal(loc=50.0, scale=np.sqrt(true_var_nm2), size=200_000)

    result = compute_potential(x_synth, n_bins=80)
    expected_k = 1.0 / true_var_nm2
    print(f"Expected k/kT:     {expected_k:.6f} /nm^2")
    print(f"k_fit_over_kT:     {result['k_fit_over_kT']:.6f} /nm^2")
    print(f"k_eq_over_kT:      {result['k_eq_over_kT']:.6f} /nm^2")
    print(f"k_fit (N/m):       {result['k_fit_N_per_m']:.4e}")
    print(f"k_eq  (N/m):       {result['k_eq_N_per_m']:.4e}")
    print(f"x0_fit_nm:         {result['x0_fit_nm']:.2f}  (expected ~50.0)")

    plot_potential(result)

"""
Consistency-check script: cross-validates internally redundant computations
against each other (closed-form vs numeric solve, analytical vs
simulation-derived).

Several checks require files produced by
`run_bath_model_eq.py`/`run_bath_model_non_eq.py`, and are skipped
(not failed) if those files don't exist yet.
"""
import sys
import numpy as np

from sdesim.params import params_bath_model
from sdesim.constants import (
    BATH_EQ_FITTED_2D_DISTRIBUTION_PATH, BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH,
    NON_EQUILIBRIUM_TRAJECTORIES_PATH, TIME_STEP_NON_EQ,
)
from sdesim.helpers import load_fitted_distribution, format_trajectories_at_two_times
from sdesim.analytics import (
    SolutionMatrixCreator, compute_D, get_cvector, get_cvector_numeric, mean_work,
    compute_work_general_protocol, current_lambda, negative_trap_potential,
    compute_conditional_xt_dist, compute_conditional_xbt_distribution,
    compute_c1_analytical_SI, bath_correlation_C, bath_correlation_C_b, compute_c3_analytical_SI
)
from sdesim.statistics import (
    analytical_distribution_eq, symmetric_kl_gaussian, condition_gaussian,
    mean_initial_positions_from_trajectories, compute_c1_from_joint,
    correlation_to_init_time, compute_c3_from_joint
)

FAILURES = []


def check(name, a, b, rtol=1e-5, atol=1e-12):
    """
    Compare two (array-like) quantities and record a pass/fail.

    Parameters
    ----------
    name : str
        Label printed alongside the result, and recorded on failure.
    a, b : array_like
        Values to compare; converted to float ndarrays via `np.asarray`.
    rtol : float, optional
        Relative tolerance passed to `np.allclose`. Default 1e-5.
    atol : float, optional
        Absolute tolerance passed to `np.allclose`. Default 1e-12.

    Returns
    -------
    bool
        True if `a` and `b` agree within tolerance, False otherwise.
        On False, `name` is appended to the module-level `FAILURES` list.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    diff = np.max(np.abs(a - b))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff|={diff:.3e}")
    if not ok:
        FAILURES.append(name)
    return ok


def check_cvector_methods():
    """
    Cross-check the optimal c-vector against two independent solution
    methods: `get_cvector` (closed-form algebraic formulas) and
    `get_cvector_numeric` (generic linear solve of the same stationarity
    condition via `optimize_quadratic_form`).

    Both methods solve the identical optimization problem given the same
    D matrix, so any disagreement indicates an algebra or implementation
    bug rather than a modeling choice.
    """
    print("\n--- c-vector: closed-form (get_cvector) vs numeric solve (get_cvector_numeric) ---")
    B_creator = SolutionMatrixCreator(params_bath_model)
    for c1, c3, t_protocol_end in [(0.0, 0.0, 1.0), (5e-8, -3e-8, 3.0), (-2e-8, 1e-8, 5.0)]:
        D = compute_D(B_creator, t_protocol_end, params_bath_model)
        c_vec_a, lambda_f_a = get_cvector(c1, c3, D)
        c_vec_n, lambda_f_n = get_cvector_numeric(c1, c3, D)
        check(f"c_vector (c1={c1:g}, c3={c3:g}, t_protocol_end={t_protocol_end})", c_vec_a, c_vec_n)
        check(f"lambda_f (c1={c1:g}, c3={c3:g}, t_protocol_end={t_protocol_end})", lambda_f_a, lambda_f_n)


def check_work_methods():
    """
    Cross-check the mean work of the optimal protocol computed two ways:
    the closed-form quadratic form `mean_work(D, c_vector, lambda_f)`, and
    direct deterministic ODE integration along the resulting optimal
    trajectory via `compute_work_general_protocol`.

    Both should agree since the optimal trajectory is fully determined by
    `c_vector`/`lambda_f`; disagreement indicates a bug in either the D
    matrix construction or the ODE cross-check integration.
    """
    print("\n--- mean work: D-matrix quadratic form vs ODE integration along optimal trajectory ---")
    B_creator = SolutionMatrixCreator(params_bath_model)
    for c1, c3, t_protocol_end in [(0.0, 0.0, 1.0), (5e-8, -3e-8, 3.0), (-2e-8, 1e-8, 5.0)]:
        D = compute_D(B_creator, t_protocol_end, params_bath_model)
        c_vec, lambda_f = get_cvector(c1, c3, D)
        work_from_D = mean_work(D, c_vec, lambda_f)

        def lambda_optimal(t, c_vec=c_vec):
            return current_lambda(B_creator.current_matrix(t), c_vec, params_bath_model)

        work_from_ode = compute_work_general_protocol(
            c1, c3, lambda_optimal, t_protocol_end, params_bath_model, lambda_f=lambda_f
        )
        check(f"mean_work (c1={c1:g}, c3={c3:g}, t_protocol_end={t_protocol_end})",
              work_from_D, work_from_ode, rtol=1e-4)


def check_jump_work_identity():
    """
    Verify the exact algebraic identity for a jump-and-hold protocol:
    when lambda(t) = x0 for all t in [0, t_protocol_end], the mean work
    collapses exactly to -V_trap(x0) = -0.5 * kappa * x0**2, independent
    of xb0, t_protocol_end, or the bath dynamics in between (see derivation
    in the accompanying discussion: the lambda-dot term vanishes identically
    for a constant protocol, so the trajectory's interior drops out of the
    work integral).

    This should hold to near machine precision for any x0, xb0,
    t_protocol_end; failure indicates a regression in
    `compute_work_general_protocol` or `negative_trap_potential`.
    """
    print("\n--- jump protocol: work == -V_trap(x0) (exact algebraic identity) ---")
    for x0 in [0.0, 5e-8, -2e-8, 1e-7]:
        for xb0 in [0.0, 3e-8]:          # shouldn't matter - identity holds for any xb0
            for t_protocol_end in [1.0, 3.0, 5.0]:
                work_jump = compute_work_general_protocol(
                    x0, xb0, lambda_func=lambda t: x0, t_protocol_end=t_protocol_end, params=params_bath_model
                )
                pot_energy = negative_trap_potential(params_bath_model, x0**2)
                check(f"jump work vs -V_trap (x0={x0:g}, xb0={xb0:g}, t_protocol_end={t_protocol_end})",
                      work_jump, pot_energy, rtol=1e-6)


def check_equilibrium_distribution():
    """
    Compare the closed-form equilibrium joint distribution of (X, X_b)
    against the Gaussian fitted from simulated stationary-state trajectories.

    Requires `BATH_EQ_FITTED_2D_DISTRIBUTION_PATH` to exist (produced by
    `run_bath_model_eq.py`); skipped, not failed, if missing.

    Checks the symmetric KL divergence against a fixed threshold, plus
    direct mean/covariance comparisons.
    """
    print("\n--- equilibrium distribution: analytical vs fitted ---")
    try:
        fitted = load_fitted_distribution(BATH_EQ_FITTED_2D_DISTRIBUTION_PATH)
    except FileNotFoundError:
        print("  SKIPPED: run `make run_bath_model_eq` first.")
        return

    analytical = analytical_distribution_eq(params_bath_model, None)
    kl = symmetric_kl_gaussian(fitted, analytical)
    threshold = 0.01
    ok = kl < threshold
    print(f"[{'PASS' if ok else 'FAIL'}] symmetric KL(fitted, analytical) = {kl:.4g} (threshold {threshold})")
    if not ok:
        FAILURES.append("equilibrium distribution KL divergence")
    check("equilibrium mean", fitted.mean, analytical.mean, atol=1e-9)
    check("equilibrium covariance", fitted.cov, analytical.cov, rtol=0.05)


def check_conditional_distributions():
    """
    Compare analytical conditional distributions P(X_t | X_0) and
    P(X_bt | X_0) against Gaussians conditioned on the fitted joint
    4D distribution of simulated non-equilibrium trajectories.

    Requires `BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH` to exist (produced by
    `run_bath_model_non_eq.py`); skipped, not failed, if missing.

    Notes
    -----
    `time_between_measurements` must match the t2 value used to build the
    fitted 4D distribution in `run_bath_model_non_eq.py`. Test points for
    x_0 are scaled to `params_bath_model.x_thresh` rather than hardcoded,
    so they remain sensible regardless of the model's physical unit scale.
    """
    print("\n--- conditional distributions: analytical vs fitted (needs run_bath_model_non_eq.py) ---")
    try:
        gaussian_4d = load_fitted_distribution(BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    except FileNotFoundError:
        print("  SKIPPED: run `make run_bath_model_non_eq` first.")
        return

    time_between_measurements = 1.0
    x_thresh = params_bath_model.x_thresh
    for x_0 in [-x_thresh, 0.0, x_thresh, 2 * x_thresh]:
        marginal_x0_xt = gaussian_4d.marginal([0, 2])
        cond_xt_protocol_endit = condition_gaussian(marginal_x0_xt, [0], x_0)
        cond_xt_analytical = compute_conditional_xt_dist(x_0, params_bath_model, time_between_measurements)
        check(f"P(X_t|X_0={x_0:.2e}) mean", cond_xt_protocol_endit.mean, cond_xt_analytical.mean, rtol=0.05)
        check(f"P(X_t|X_0={x_0:.2e}) var", cond_xt_protocol_endit.cov, cond_xt_analytical.cov, rtol=0.1)

        marginal_x0_xbt = gaussian_4d.marginal([0, 3])
        cond_xbt_protocol_endit = condition_gaussian(marginal_x0_xbt, [0], x_0)
        cond_xbt_analytical = compute_conditional_xbt_distribution(x_0, params_bath_model, time_between_measurements)
        check(f"P(X_bt|X_0={x_0:.2e}) mean", cond_xbt_protocol_endit.mean, cond_xbt_analytical.mean, rtol=0.1)
        check(f"P(X_bt|X_0={x_0:.2e}) var", cond_xbt_protocol_endit.cov, cond_xbt_analytical.cov, rtol=0.15)


def check_c1_methods():
    """
    Three-way cross-check of c1 = E[X_t | m_0, m_t]: pure analytical
    integration (compute_c1_analytical_SI, which internally nondimensionalizes
    params_bath_model before integrating and converts the result back to SI
    units), integration over the fitted joint Gaussian (compute_c1_from_joint),
    and empirical binning of raw simulated trajectories
    (mean_initial_positions_from_trajectories).

    Requires `NON_EQUILIBRIUM_TRAJECTORIES_PATH` and
    `BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH` to exist (produced by
    `run_bath_model_non_eq.py`); skipped, not failed, if missing.
    """
    print("\n--- c1 (mean X_t | m_0, m_t): analytical vs fitted-Gaussian integral vs empirical ---")
    try:
        trajectories = np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH)
        gaussian_4d = load_fitted_distribution(BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    except FileNotFoundError:
        print("  SKIPPED: run `make run_bath_model_non_eq` first.")
        return

    time_between_measurements = 1.0
    formatted = format_trajectories_at_two_times(trajectories, 0, time_between_measurements, TIME_STEP_NON_EQ)
    empirical = mean_initial_positions_from_trajectories(formatted, params_bath_model)

    for m_0 in [0, 1]:
        for m_t in [0, 1]:
            c1_analytical = compute_c1_analytical_SI(m_0, m_t, params_bath_model, time_between_measurements)
            c1_from_fit = compute_c1_from_joint(m_0, m_t, params_bath_model, gaussian_4d)
            c1_empirical = empirical[f"m_0={m_0},m_t={m_t}"]["mean_X_t"]
            check(f"c1 (m_0={m_0}, m_t={m_t}) analytical vs from_fit", c1_analytical, c1_from_fit, rtol=0.05)
            check(f"c1 (m_0={m_0}, m_t={m_t}) analytical vs empirical", c1_analytical, c1_empirical, rtol=0.1)

def check_c3_methods():
    """
    Three-way cross-check of c3 = E[X_bt | m_0, m_t]: pure analytical
    (compute_c3_analytical_SI, using the closed-form direct-route Gaussian
    conditional-expectation collapse of the X_bt integral, internally
    nondimensionalized the same way as compute_c1_analytical_SI), integration
    over the fitted joint Gaussian (compute_c3_from_joint), and empirical
    binning of raw simulated trajectories
    (mean_initial_positions_from_trajectories).

    Requires `NON_EQUILIBRIUM_TRAJECTORIES_PATH` and
    `BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH` to exist (produced by
    `run_bath_model_non_eq.py`); skipped, not failed, if missing.
    """
    print("\n--- c3 (mean X_bt | m_0, m_t): analytical vs fitted-Gaussian integral vs empirical ---")
    try:
        trajectories = np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH)
        gaussian_4d = load_fitted_distribution(BATH_NEQ_FITTED_4D_DISTRIBUTION_PATH)
    except FileNotFoundError:
        print("  SKIPPED: run `make run_bath_model_non_eq` first.")
        return

    time_between_measurements = 1.0
    formatted = format_trajectories_at_two_times(trajectories, 0, time_between_measurements, TIME_STEP_NON_EQ)
    empirical = mean_initial_positions_from_trajectories(formatted, params_bath_model)

    for m_0 in [0, 1]:
        for m_t in [0, 1]:
            c3_analytical = compute_c3_analytical_SI(m_0, m_t, params_bath_model, time_between_measurements)
            c3_from_fit = compute_c3_from_joint(m_0, m_t, params_bath_model, gaussian_4d)
            c3_empirical = empirical[f"m_0={m_0},m_t={m_t}"]["mean_X_bt"]
            check(f"c3 (m_0={m_0}, m_t={m_t}) analytical vs from_fit", c3_analytical, c3_from_fit, rtol=0.05)
            check(f"c3 (m_0={m_0}, m_t={m_t}) analytical vs empirical", c3_analytical, c3_empirical, rtol=0.1)


def check_autocorrelations():
    """
    Compare analytical position/bath autocorrelation functions
    (`bath_correlation_C`, `bath_correlation_C_b`) against autocorrelations
    computed directly from simulated non-equilibrium trajectories.

    Requires `NON_EQUILIBRIUM_TRAJECTORIES_PATH` to exist (produced by
    `run_bath_model_non_eq.py`); skipped, not failed, if missing.
    """
    print("\n--- autocorrelation: analytical vs simulated (needs run_bath_model_non_eq.py) ---")
    try:
        trajectories = np.load(NON_EQUILIBRIUM_TRAJECTORIES_PATH)
    except FileNotFoundError:
        print("  SKIPPED: run `make run_bath_model_non_eq` first.")
        return

    for t in [0.0, 1.0, 5.0, 20.0]:
        empirical_xx = correlation_to_init_time(trajectories, t, TIME_STEP_NON_EQ, np.array([1, 1]))
        check(f"<X(0)X(t={t})>", empirical_xx, bath_correlation_C(t, params_bath_model), rtol=0.1)

        empirical_xxb = correlation_to_init_time(trajectories, t, TIME_STEP_NON_EQ, np.array([1, 2]))
        check(f"<X(0)Xb(t={t})>", empirical_xxb, bath_correlation_C_b(t, params_bath_model), rtol=0.15)


def main():
    """
    Run all consistency checks in sequence and exit with a non-zero status
    if any check failed.

    Prints a PASS/FAIL line per check plus a final summary listing every
    failed check by name; intended to be run via `make run_consistency_tests`
    or `python -m scripts.run_consistency_tests`.
    """
    check_cvector_methods()
    check_work_methods()
    check_jump_work_identity()
    check_equilibrium_distribution()
    check_conditional_distributions()
    check_c1_methods()
    check_c3_methods()
    check_autocorrelations()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("All consistency checks passed.")


if __name__ == "__main__":
    main()
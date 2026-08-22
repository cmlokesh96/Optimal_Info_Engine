"""
Quick checks for the `fast` flag on the analytical initial-position
functions:

1. fast=False still matches the original per-outcome dblquad-based
   functions (compute_c1_analytical_SI / compute_c3_analytical_SI /
   compute_second_moment_xt_analytical_SI) closely - i.e. the default
   behavior is unchanged/exact.
2. fast=True gives results close to fast=False (sanity bound on the
   precision/speed tradeoff) and is actually faster.
3. The parameter_sweep caching wrappers key correctly on `fast`, so a
   fast=True call and a fast=False call for the same point never collide
   in the lru_cache.
"""
import time
import numpy as np

from sdesim.params import params_bath_model
from sdesim.analytics import (
    mean_initial_positions_from_analytical,
    mean_second_measurement_positions_from_analytical,
    compute_c1_analytical_SI, compute_c3_analytical_SI,
    compute_second_moment_xt_analytical_SI,
)
from sdesim.parameter_sweep import (
    _cached_mean_initial_positions, _cached_second_measurement_positions,
)

FAILURES = []


def check(name, a, b, rtol=1e-3, atol=1e-15):
    a, b = float(a), float(b)
    ok = np.isclose(a, b, rtol=rtol, atol=atol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: a={a:.6g}, b={b:.6g}")
    if not ok:
        FAILURES.append(name)
    return ok


def check_default_matches_original(t2=1.0, n_outcomes=2):
    """fast=False (the default) should match the original per-outcome functions."""
    print(f"\n--- fast=False vs original per-outcome functions (t2={t2}) ---")
    positions = mean_initial_positions_from_analytical(params_bath_model, t2, n_outcomes)

    for m_0 in range(n_outcomes):
        for m_t in range(n_outcomes):
            key = f"m_0={m_0},m_t={m_t}"
            if np.isnan(positions[key]["mean_X_t"]):
                continue
            c1_ref = compute_c1_analytical_SI(m_0, m_t, params_bath_model, t2, n_outcomes)
            c3_ref = compute_c3_analytical_SI(m_0, m_t, params_bath_model, t2, n_outcomes)
            m2_ref = compute_second_moment_xt_analytical_SI(m_0, m_t, params_bath_model, t2, n_outcomes)

            check(f"mean_X_t ({key})", positions[key]["mean_X_t"], c1_ref)
            check(f"mean_X_bt ({key})", positions[key]["mean_X_bt"], c3_ref)
            check(f"mean_X_t_squared ({key})", positions[key]["mean_X_t_squared"], m2_ref)


def check_fast_close_to_exact(t2=1.0, n_outcomes=2, rtol=0.05):
    """fast=True shouldn't drift far from fast=False - loose bound, just a sanity check."""
    print(f"\n--- fast=True vs fast=False (t2={t2}) ---")
    exact = mean_initial_positions_from_analytical(params_bath_model, t2, n_outcomes, fast=False)
    fast = mean_initial_positions_from_analytical(params_bath_model, t2, n_outcomes, fast=True)

    for key in exact:
        if np.isnan(exact[key]["mean_X_t"]):
            continue
        check(f"mean_X_t fast vs exact ({key})", fast[key]["mean_X_t"], exact[key]["mean_X_t"], rtol=rtol)
        check(f"mean_X_bt fast vs exact ({key})", fast[key]["mean_X_bt"], exact[key]["mean_X_bt"], rtol=rtol)
        check(f"probability fast vs exact ({key})", fast[key]["probability"], exact[key]["probability"], rtol=rtol)


def check_second_measurement_positions(n_outcomes=2):
    print(f"\n--- mean_second_measurement_positions_from_analytical fast=True vs False ---")
    exact = mean_second_measurement_positions_from_analytical(params_bath_model, n_outcomes, fast=False)
    fast = mean_second_measurement_positions_from_analytical(params_bath_model, n_outcomes, fast=True)
    for key in exact:
        if np.isnan(exact[key]["mean_X_t"]):
            continue
        check(f"mean_X_t ({key})", fast[key]["mean_X_t"], exact[key]["mean_X_t"], rtol=0.05)


def check_timing(t2=1.0, n_outcomes=2, n_repeats=3):
    """fast=True should be meaningfully faster than fast=False."""
    print(f"\n--- timing: fast=False vs fast=True ({n_repeats} repeats) ---")

    start = time.perf_counter()
    for _ in range(n_repeats):
        mean_initial_positions_from_analytical(params_bath_model, t2, n_outcomes, fast=False)
    slow_time = (time.perf_counter() - start) / n_repeats

    start = time.perf_counter()
    for _ in range(n_repeats):
        mean_initial_positions_from_analytical(params_bath_model, t2, n_outcomes, fast=True)
    fast_time = (time.perf_counter() - start) / n_repeats

    print(f"  fast=False: {slow_time:.4f}s/call")
    print(f"  fast=True:  {fast_time:.4f}s/call")
    ok = fast_time < slow_time
    print(f"[{'PASS' if ok else 'FAIL'}] fast=True is faster than fast=False")
    if not ok:
        FAILURES.append("fast=True not faster than fast=False")


def check_cache_key_distinguishes_fast(t2=1.0, n_outcomes=2):
    """
    The lru_cache-wrapped sweep helpers must key on `fast`, otherwise a
    fast=True call and a fast=False call for the same point would
    silently return whichever was computed (and cached) first.
    """
    print("\n--- cache key includes `fast` ---")
    _cached_mean_initial_positions.cache_clear()
    exact = _cached_mean_initial_positions(params_bath_model, t2, n_outcomes, 1e-10, False)
    fast = _cached_mean_initial_positions(params_bath_model, t2, n_outcomes, 1e-10, True)

    any_key = next(k for k, v in exact.items() if not np.isnan(v["mean_X_t"]))
    different = not np.isclose(exact[any_key]["mean_X_t"], fast[any_key]["mean_X_t"], rtol=1e-9)
    # They needn't be numerically identical, but they must be independently
    # computed (not the same cached object) - check cache has 2 entries.
    info = _cached_mean_initial_positions.cache_info()
    ok = info.currsize == 2
    print(f"[{'PASS' if ok else 'FAIL'}] cache_info={info} (expected currsize=2)")
    if not ok:
        FAILURES.append("cache key does not distinguish fast=True/False")

    _cached_second_measurement_positions.cache_clear()
    _cached_second_measurement_positions(params_bath_model, n_outcomes, 1e-10, False)
    _cached_second_measurement_positions(params_bath_model, n_outcomes, 1e-10, True)
    info2 = _cached_second_measurement_positions.cache_info()
    ok2 = info2.currsize == 2
    print(f"[{'PASS' if ok2 else 'FAIL'}] cache_info={info2} (expected currsize=2)")
    if not ok2:
        FAILURES.append("second-measurement cache key does not distinguish fast=True/False")


def main():
    check_default_matches_original()
    check_fast_close_to_exact()
    check_second_measurement_positions()
    check_timing()
    check_cache_key_distinguishes_fast()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print("All fast-flag checks passed.")


if __name__ == "__main__":
    main()
"""
Systematic parameter sweep over (kappa, x_thresh, t_second_measurement,
t_protocol_end) for the optimal-protocol pipeline, plus utilities to flag
the "best" combination under a chosen criterion and to re-run/plot exactly
one combination in detail.

Kept deliberately separate from `process_combination`/the run scripts: the
sweep grid, the aggregation metrics, and "run one point in detail" are all
generic across the 2-outcome and 3-outcome measurement conventions, so this
module is shared by both `run_optimal_protocol.py` and
`run_optimal_protocol_3_measurement.py`.
"""
from typing import NamedTuple, Optional
import numpy as np
import functools
import time
from typing import NamedTuple, Optional, Sequence
from sdesim.params import SystemVariables
from sdesim.analytics import (
    SolutionMatrixCreator, compute_D, get_cvector, mean_work, negative_trap_potential,
    mean_initial_positions_from_analytical,
    mean_second_measurement_positions_from_analytical,
    compute_probability_of_outcome_analytical_SI,
)
from sdesim.helpers import parse_m_values, get_or_compute_initial_positions
from sdesim.process_combination import process_combination
from sdesim.visualization import plot_optimal_trajectories_grid, plot_accumulated_work

class SweepAxis(NamedTuple):
    """
    Specification of one swept parameter's domain.
    Two exclusive ways to specify the swept values:
    1. Uniform grid via start/end/step
    Parameters:
        start, end : float
            Inclusive bounds of the sweep domain (order-independent).
        step : float, optional
            Spacing between sample points. If None (default), set to 10% of
            the domain width (`end - start`
    2. Explicit, arbitrarily-spaced values via `values`  construct
       with `SweepAxis.from_values([...])` instead of passing
       start/end/step. When `values` is set, start/end/step are ignored
       entirely by `resolve_sweep_values`.
    """
    start: Optional[float] = None
    end: Optional[float] = None
    step: Optional[float] = None
    values: Optional[Sequence[float]] = None

    @classmethod
    def from_values(cls, values):
        """
        Build a SweepAxis from an explicit, arbitrarily-spaced sequence of
        values, instead of a uniform start/end/step grid.

        Parameters:
        values : sequence of float
            The specified values to sweep over

        Returns:
        SweepAxis
        """
        return cls(values=tuple(values))


def resolve_sweep_values(axis, min_relative_resolution=0.1):
    """
    make a concrete array from an sweep_axis object

    Parameters:
    axis : SweepAxis
    min_relative_resolution : float, optional
        Fraction of (end - start) used as the default step when
        `axis.step` is None

    Returns
    -------
    numpy.ndarray
        Sample points in [start, end], including both endpoints. A
        degenerate axis (start == end) yields the single value `[start]`
        (this is how a parameter is "fixed" rather than swept).
    """

    if axis.values is not None:
        return np.asarray(axis.values, dtype=float)
    start, end, step = axis.start, axis.end, axis.step
    if np.isclose(start, end):
        return np.array([start])
    if step is None:
        step = min_relative_resolution * (end - start)
  
    n_points = int(np.floor((end - start) / step + 1e-9)) + 1 #1e-9 guards for rounding errors due to floating point inaccuracy
    values = start + step * np.arange(n_points)
    if values[-1] < end - 1e-9:
        values = np.append(values, end)
    return values


def make_params_variant(base_params, kappa, x_thresh_sigma_multiple):
    """
    Build a SystemVariables differing from base_params only in
    kappa and x_thresh

    Parameters:
    base_params : SystemVariables
    kappa: float
    x_thresh: float, in units of sqrt(k_B*T/kappa)

    returns:
    SystemVariables
    """
    variant = SystemVariables.create(
        gamma=base_params.gamma, gamma_b=base_params.gamma_b,
        kappa=kappa, kappa_b=base_params.kappa_b,
        T=base_params.T, k_B=base_params.k_B, x_thresh_sigma_multiple= x_thresh_sigma_multiple
    )
    return variant


class ParameterPoint(NamedTuple):
    """One point in the 4D sweep grid."""
    kappa: float
    x_thresh_sigma_multiple: float
    t_second_measurement: float
    t_protocol_end: float


@functools.lru_cache(maxsize=None)
def _cached_second_measurement_positions(params, n_outcomes, probability_threshold, fast=False):
    """
    Cached wrapper around mean_second_measurement_positions_from_analytical.

    The mean position conditioned on a single measurement outcome `m_t`,
    as if no earlier measurement had ever been taken, depends only on
    params and n_outcomes

    Parameters:
    params : SystemVariables
        Hashed as part of the cache key in full
    n_outcomes : int
        Number of measurement outcome categories.
    probability_threshold : float
        Outcomes with marginal probability below this are reported as NaN.
    fast : bool, optional
        Forwarded to mean_second_measurement_positions_from_analytical;

    Returns:
    dict
        As returned by mean_second_measurement_positions_from_analytical.
    """
    return mean_second_measurement_positions_from_analytical(
        params, n_outcomes, probability_threshold, fast=fast
    )


@functools.lru_cache(maxsize=None)
def _cached_mean_initial_positions(params, t_second_measurement, n_outcomes,
                                    probability_threshold, fast=False):
    """
    Cached wrapper around mean_initial_positions_from_analytical.

    This result does mot depend on t_protocol_end, thus cached

    Parameters:
    params : SystemVariables
        Hashed as part of the cache key in full
    t_second_measurement : float
        Time elapsed between the two measurements.
    n_outcomes : int
        Number of measurement outcome categories.
    probability_threshold : float
        Outcome combinations with probability below this are reported as
        NaN (
    fast : bool, optional
        Forwarded to mean_initial_positions_from_analytical; included in
        the cache key so a `fast=True` call is never conflated with a
        `fast=False` (exact) one 

    Returns:
    dict
        As returned by `mean_initial_positions_from_analytical`.
    """
    return mean_initial_positions_from_analytical(
        params, t_second_measurement, n_outcomes, probability_threshold, fast=fast
    )

@functools.lru_cache(maxsize=None)
def _cached_D(kappa, kappa_b, gamma, gamma_b, t_protocol_end):
    """
    Cached (B_creator, D) pair for compute_D.

    compute_D depends only on (kappa, kappa_b, gamma,
    gamma_b, t_protocol_end), but not on x_thresh or t_second_measurement

    T, k_B are irrelevant to compute_D (only kappa/kappa_b/gamma/gamma_b
    enter tau_b, tau_p, k), so they're fixed arbitrarily to 1.0 here.

    Parameters:
    kappa, kappa_b, gamma, gamma_b, t_protocol_end : float

    Returns
    -------
    B_creator : SolutionMatrixCreator
    D : numpy.ndarray, shape (5, 5)
    """
    params = SystemVariables.create(gamma=gamma, gamma_b=gamma_b, kappa=kappa,
                                     kappa_b=kappa_b, T=1.0, k_B=1.0)
    B_creator = SolutionMatrixCreator(params)
    return B_creator, compute_D(B_creator, t_protocol_end, params)

def iter_parameter_grid(kappa_axis, x_thresh_sigma_multiple_axis, t_second_measurement_axis, t_protocol_end_axis):
    """
    Generator that returns a tuple of the four sweep parameters, for iterating over of the four swept axes.

    Fix axes you don't need to vary (start == end)
    to keep number of combinations tractable,
    coarser `step`.

    Yields:
    ParameterPoint
    """
    kappas = resolve_sweep_values(kappa_axis)
    x_thresh_sigma_multiples = resolve_sweep_values(x_thresh_sigma_multiple_axis)
    t2_values = resolve_sweep_values(t_second_measurement_axis)
    tf_values = resolve_sweep_values(t_protocol_end_axis)
    for kappa in kappas:
        for x_thresh_sigma_multiple in x_thresh_sigma_multiples:
            for t2 in t2_values:
                for tf in tf_values:
                    yield ParameterPoint(kappa, x_thresh_sigma_multiple, t2, tf)


def evaluate_parameter_point(point, base_params, n_outcomes=2, probability_threshold=1e-10, fast=True):
    """
    Evaluate the closed-form optimal protocol and work references
    for every measurement-outcome combination at one grid point.

    Parameters:
    point : ParameterPoint
    base_params : SystemVariables
        Only gamma, gamma_b, kappa_b, T, k_B are taken from this; kappa
        and x_thresh come from `point` (see `make_params_variant`).
    n_outcomes : int, optional
        Default 2.
    probability_threshold : float, optional
        Outcomes with P(m_0, m_t) below this are skipped. Default 1e-10.

    Returns:
    dict
        Keys "m_0={m_0},m_t={m_t}" (only outcomes with non-negligible
        propability), each mapping to a dict with:
            "probability" : P(m_0, m_t)
            "optimal" : mean work of the closed-form optimal protocol
            "jump_both" : work of an instantaneous jump to
                E[X_t | m_0, m_t] (conditioned on both measurements)
            "jump_second_only" : work of an instantaneous jump to
                E[X_t | m_t] (conditioned on the second measurement only,
                i.e. as if no first measurement had ever taken place - see
                `mean_second_measurement_positions_from_analytical`)
            "-V_trap" : negative mean potential energy at E[X_t^2 | m_0, m_t]
    """
    params = make_params_variant(base_params, point.kappa, point.x_thresh_sigma_multiple)

    mean_initial_positions = _cached_mean_initial_positions(
        params, point.t_second_measurement, n_outcomes, probability_threshold, fast
    )
    mean_second_measurement_positions = _cached_second_measurement_positions(
        params, n_outcomes, probability_threshold, fast
    )

    B_creator, D = _cached_D(point.kappa, base_params.kappa_b, base_params.gamma,
                             base_params.gamma_b, point.t_protocol_end)

    results = {}
    for key, vals in mean_initial_positions.items():
        mean_X_t, mean_X_bt = vals["mean_X_t"], vals["mean_X_bt"]
        if np.isnan(mean_X_t) or np.isnan(mean_X_bt):
            continue

        m_0, m_t = parse_m_values(key)
        # Reuse the probability already computed as a byproduct of the
        # normalization integral inside mean_initial_positions_from_analytical
        # instead of re-running an equivalent integral from scratch.
        probability = vals["probability"]

        c_vector, lambda_f = get_cvector(mean_X_t, mean_X_bt, D)
        work_optimal = mean_work(D, c_vector, lambda_f)

        work_jump_both = negative_trap_potential(params, mean_X_t**2)

        lambda_second_only = mean_second_measurement_positions[f"m_t={int(m_t)}"]["mean_X_t"]
        work_jump_second_only = (
            np.nan if np.isnan(lambda_second_only) else
            negative_trap_potential(params, lambda_second_only**2)
        )

        pot_energy = negative_trap_potential(params, vals["mean_X_t_squared"])

        results[key] = {
            "probability": probability,
            "optimal": work_optimal,
            "jump_both": work_jump_both,
            "jump_second_only": work_jump_second_only,
            "-V_trap": pot_energy,
        }

    return results

def run_parameter_sweep(kappa_axis, x_thresh_sigma_multiple_axis, t_second_measurement_axis, t_protocol_end_axis,
                         base_params, n_outcomes=2, probability_threshold=1e-10,
                         progress_every=1.0, fast=True):
    """
    Evaluate evaluate_parameter_point over the full grid spanned by the
    four sweep axes.

    Parameters:
    kappa_axis, x_thresh_sigma_multiple_axis, t_second_measurement_axis, t_protocol_end_axis : SweepAxis
    base_params : SystemVariables
    n_outcomes : int, optional
        Default 2.
    probability_threshold : float, optional
        Default 1e-10.
        progress_every : float, optional
        Print a status line whenever cumulative progress has advanced by
        at least this many percentage points since the last printed
        update, Set to `None` to disable progress
        printing entirely.

    Returns:
    dict
        ParameterPoint -> dict, as returned by evaluate_parameter_point.
  
    
    """
    axis_lengths = [
        len(resolve_sweep_values(axis))
        for axis in (kappa_axis, x_thresh_sigma_multiple_axis, t_second_measurement_axis, t_protocol_end_axis)
    ]
    total_points = int(np.prod(axis_lengths))
    print(f"Starting parameter sweep: {' x '.join(map(str, axis_lengths))} = {total_points} points")

    start_time = time.perf_counter()
    last_reported_percent = -progress_every if progress_every is not None else None
    results = {}

    for index, point in enumerate(
        iter_parameter_grid(kappa_axis, x_thresh_sigma_multiple_axis, t_second_measurement_axis, t_protocol_end_axis)
    ):
        results[point] = evaluate_parameter_point(point, base_params, n_outcomes, probability_threshold, fast)

        if progress_every is not None:
            percent_done = 100.0 * (index + 1) / total_points
            if percent_done - last_reported_percent >= progress_every or index + 1 == total_points:
                elapsed = time.perf_counter() - start_time
                rate = (index + 1) / elapsed if elapsed > 0 else float("inf")
                remaining = (total_points - index - 1) / rate if rate > 0 else float("nan")
                print(f"  {index + 1}/{total_points} ({percent_done:.1f}%) done, "
                      f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s, current point: {point}")
                last_reported_percent = percent_done

    print(f"Sweep finished: {total_points} points in {time.perf_counter() - start_time:.1f}s")
    return results


def weighted_mean_work(point_results, work_key="optimal"):
    """
    Probability-weighted mean of one work quantity across measurement
    outcomes, at one parameter point.

    Parameters:
    point_results : dict
        As returned by `evaluate_parameter_point`.
    work_key : str, optional
        One of "optimal", "jump_both", "jump_second_only", "-V_trap".
        Default "optimal".

    Returns:
    float
        sum_i P_i * work_i / sum_i P_i; NaN if `point_results` is empty.
    """
    if not point_results:
        return np.nan
    probabilities = np.array([v["probability"] for v in point_results.values()])
    works = np.array([v[work_key] for v in point_results.values()])
    return float(np.sum(probabilities * works) / np.sum(probabilities))


def weighted_mean_ratio(point_results, reference_key="jump_second_only"):
    """
    Probability-weighted mean of (optimal work / reference work) across
    measurement outcomes, at one parameter point.

    Parameters:
    point_results : dict
    reference_key : str, optional
        One of "jump_both", "jump_second_only", "-V_trap" - lets the
        caller pick which of the three references the ratio is taken
        against. Default "jump_second_only".

    Returns:
    float
        sum_i P_i * (optimal_i / reference_i) / sum_i P_i, over outcomes
        with a finite, nonzero reference; NaN else.
    """
    if not point_results:
        return np.nan
    probabilities, ratios = [], []
    for v in point_results.values():
        reference = v[reference_key]
        if np.isnan(reference) or reference == 0:
            continue
        probabilities.append(v["probability"])
        ratios.append(v["optimal"] / reference)
    if not probabilities:
        return np.nan
    probabilities = np.array(probabilities)
    ratios = np.array(ratios)
    return float(np.sum(probabilities * ratios) / np.sum(probabilities))

def weighted_ratio_of_means(point_results, reference_key="jump_second_only"):
    """
    Ratio of probability-weighted mean optimal work to probability-weighted
    mean reference work.

    metric = (sum_i P_i * optimal_i) / (sum_i P_i * reference_i)

    Parameters:
    point_results : dict
        As returned by evaluate_parameter_point.
    reference_key : str, optional
        One of "jump_both", "jump_second_only", "-V_trap". Default
        "jump_second_only".

    Returns:
    float
        (sum_i P_i * optimal_i) / (sum_i P_i * reference_i); NaN if
        point_results is empty.
    """
    if not point_results:
        return np.nan
    probabilities = np.array([v["probability"] for v in point_results.values()])
    optimals = np.array([v["optimal"] for v in point_results.values()])
    references = np.array([v[reference_key] for v in point_results.values()])

    weighted_numerator = np.sum(probabilities * optimals)
    weighted_denominator = np.sum(probabilities * references)

    return float(weighted_numerator / weighted_denominator)


def find_best_parameter_point(sweep_results, metric="weighted_mean_work",
                               work_key="optimal", reference_key="jump_second_only",
                               maximize=True):
    """
    Flag the parameter combination optimizing a scalar metric aggregated
    across measurement outcomes.

    Only two metrics are supported:
    every example given ("highest mean work weighted by probability",
    "best ratio of work and reference value")

    Parameters:
    sweep_results : dict
        As returned by run_parameter_sweep.
    metric : {"weighted_mean_work", "weighted_mean_ratio"}, optional
        Default "weighted_mean_work".
    work_key : str, optional
        Used only for metric="weighted_mean_work"; one of "optimal",
        "jump_both", "jump_second_only", "-V_trap". Default "optimal".
    reference_key : str, optional
        Used only for metric="weighted_mean_ratio"; one of "jump_both",
        "jump_second_only", "-V_trap" 
        Default "jump_second_only".
    maximize : bool, optional
        If True (default), flag the point with the largest metric value;
        if False, the smallest.

    Returns:
    best_point : ParameterPoint
    best_value : float
    all_values : dict
        ParameterPoint -> metric value for every point in
        sweep_results.
    """
    if metric == "weighted_mean_work":
        metric_fn = lambda r: weighted_mean_work(r, work_key)
    elif metric == "weighted_mean_ratio":
        metric_fn = lambda r: weighted_mean_ratio(r, reference_key)
    elif metric == "weighted_ratio_of_means":
        metric_fn = lambda r: weighted_ratio_of_means(r, reference_key)
    else:
        raise ValueError(f"Unknown metric {metric!r}; choose 'weighted_mean_work', 'weighted_mean_ratio' or 'weighted_ratio_of_means'")

    all_values = {point: metric_fn(results) for point, results in sweep_results.items()}
    valid = {p: v for p, v in all_values.items() if not np.isnan(v)}
    if not valid:
        raise ValueError("No parameter point has a well-defined metric value.")

    best_point = max(valid, key=valid.get) if maximize else min(valid, key=valid.get)
    return best_point, valid[best_point], all_values


def run_single_parameter_point(point, base_params, save_dir, plot_dir, n_outcomes=2,
                                initial_condition_function=mean_initial_positions_from_analytical,
                                method_name="analytical", ic_cache_dir=None, make_plots=True):
    """
    Run (and, by default, save + plot) the optimal-protocol pipeline for
    exactly one parameter combination - the counterpart to
    run_parameter_sweep for inspecting one point in detail.

    Note:
    save_dir/plot_dir are not automatically namespaced by kappa/x_thresh
    (only by t_second_measurement/t_protocol_end/m_0/m_t, as before) 
    Pass distinct `save_dir`/`plot_dir` (e.g. encoding kappa,
    x_thresh yourself) to keep multiple points at same time

    Parameters:
    point : ParameterPoint
    base_params : SystemVariables
    save_dir, plot_dir : str
    n_outcomes : int, optional
        Default 2.
    initial_condition_function : callable, optional
        Any of mean_initial_positions_from_analytical,
        make_trajectory_initial_position_estimator(), or
        make_distribution_initial_position_estimator()`,
         a single point is cheap enough to support all three.
        Default mean_initial_positions_from_analytical.
    method_name : str, optional
        Must match initial_condition_function. Default "analytical".
    ic_cache_dir : str, optional
        Directory for the initial condition CSV cache; defaults to
        save_dir.
    make_plots : bool, optional
        Default True (
        inspecting one point is precisely when plots are wanted).

    Return:
    dict
        Keys "m_0={m_0},m_t={m_t}", each mapping to a dict with
        "probability", "optimal", "jump_both", "jump_second_only",
        "-V_trap" - same shape as one value of `run_parameter_sweep`'s
        return.
    """
    params = make_params_variant(base_params, point.kappa, point.x_thresh_sigma_multiple)
    ic_cache_dir = ic_cache_dir or save_dir

    mean_initial_positions = get_or_compute_initial_positions(
        params, point.t_second_measurement, ic_cache_dir, initial_condition_function,
        method_name=method_name, n_outcomes=n_outcomes,
    )
    # No time dependence here - the pre-protocol marginal is time-invariant;
    # see mean_second_measurement_positions_from_analytical's docstring.
    # Cached (see _cached_second_measurement_positions) since it depends
    # only on (kappa, x_thresh, n_outcomes), not on this point's timing.
    mean_second_measurement_positions = _cached_second_measurement_positions(
        params, n_outcomes, 1e-10,
    )

    work_summaries = process_combination(
        mean_initial_positions, point.t_second_measurement, point.t_protocol_end,
        params, save_dir, plot_dir, make_plots=False,
    )
    if make_plots:
        plot_optimal_trajectories_grid(
            point.t_second_measurement, point.t_protocol_end,
            save_dir=save_dir,
            save_path=f"{plot_dir}/optimal_trajectories_grid_"
                      f"t2_{point.t_second_measurement}_tf_{point.t_protocol_end}.png",
            n_outcomes=n_outcomes, matlab_protocol_dir=None,
        )
        plot_accumulated_work(
            point.t_second_measurement, point.t_protocol_end,
            save_dir=save_dir,
            save_path=f"{plot_dir}/accumulated_work_grid_"
                      f"t2_{point.t_second_measurement}_tf_{point.t_protocol_end}.png",
            params=params, n_outcomes=n_outcomes,
        )

    results = {}
    for (t2, tf, m_0, m_t), summary in work_summaries.items():
        key = f"m_0={int(m_0)},m_t={int(m_t)}"

        lambda_second_only = mean_second_measurement_positions[f"m_t={int(m_t)}"]["mean_X_t"]
        # Closed form: a constant protocol's mean work is exactly
        # -V_trap(lambda) (see check_jump_work_identity), avoiding the
        # solve_ivp cost of compute_work_general_protocol here too.
        work_jump_second_only = (
            np.nan if np.isnan(lambda_second_only) else
            negative_trap_potential(params, lambda_second_only**2)
        )

        probability = compute_probability_of_outcome_analytical_SI(
            m_0, m_t, params, point.t_second_measurement, n_outcomes
        )

        results[key] = {
            "probability": probability,
            "optimal": summary["optimal"],
            "jump_both": summary["jump"],
            "jump_second_only": work_jump_second_only,
            "-V_trap": summary["-V_trap"],
        }

    return results



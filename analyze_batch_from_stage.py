"""
analyze_batch_from_stage.py
============================
Notebook-independent analysis of a saved run_batch_2ch() batch folder -
the same computation as Main_script.ipynb's "4b. Batch-folder analysis"
section, runnable standalone (no camera/DAQ/notebook needed, just the
batch folder's saved files plus the matching analytical protocol files
and calibration already on disk).

Nothing physics-related needs to be remembered or re-entered by hand:
    ao_rate, delta_t_s (t2), protocol_dt_s (tf), fire_00_prob
        - read straight off each selected session's own saved _params.pkl
          (every session already carries these - see protocols_2ch.py::
          SessionParams2Ch). If the selected experiments don't all agree,
          this is FLAGGED and only the largest agreeing group is kept
          (see _check_param_consistency()) rather than silently pooling
          sessions that weren't actually run the same way.
    kappa_N_per_m
        - derived from this batch's own sibling calib/ folder
          (EXPERIMENT_ROOT/calib, where EXPERIMENT_ROOT = dirname(batch_folder) -
          same convention Main_script.ipynb's "parameters cell after 3b"
          uses) via compute_potential() on the pooled *_calib_pos_px.npy
          files - the actual MEASURED trap stiffness for this specific
          experiment. Works for any batch, old or new, as long as its
          calib/ folder is still on disk. Pass kappa_N_per_m explicitly to
          override (e.g. no calib/ folder available, or to force
          k_fit_N_per_m instead of the default k_eq_N_per_m estimate).
    analytical_save_dir
        - this batch's own analytical_protocols_used/ subfolder
          (deterministic from batch_folder alone).
T_K (298.0) is the one thing nothing here measures or saves - override it
explicitly if your experiment actually ran at a different temperature.

Choosing which experiments to pool: pass experiment_select=[1, 3, 4] to
use only experiments 1, 3 and 4 out of a batch of e.g. 5 (1-based, in
discover_session_files()'s own timestamp-sorted order) - e.g. to exclude
a known-bad run from the analysis without touching its saved files.
None (default) pools every saved experiment in the folder.

analyze_batch() returns, per state (m0, mt), direct access to:
    events[state]["x_nm"]   : (n_events_ij, protocol_frames) - real camera x(t), nm
    lambdas[state]          : (protocol_frames,)             - loaded analytical protocol, nm
    W_jump_kT[state]        : (n_events_ij,)                 - the jump's own work, kT
    W_total_kT[state]       : (n_events_ij, protocol_frames) - cumulative TOTAL work (jump + smooth), kT
    events[state]["t_s"]    : (protocol_frames,)             - time axis, s
i.e. x_ij/lambda_ij/W_ij for every state, each carrying all N_ij real
fired events of that state pooled across the selected experiments, each
event protocol_frames samples long (= protocol_time * ao_rate).

Run directly (python analyze_batch_from_stage.py) to get all three of
section 4b's plots (plot_trap_relative_grid, plot_work_split_grid,
plot_stage_vs_protocol_grid), the same printed summaries, and a saved
<BATCH_FOLDER>_events_work.mat for matlab/InfoEngineAnalysis.m. Run it
with `python -i` (or %run in IPython/Jupyter) to keep the returned
variables around afterward, or import analyze_batch() directly:

    from analyze_batch_from_stage import analyze_batch
    events, lambdas, W_jump_kT, W_total_kT, summary = analyze_batch(
        BATCH_FOLDER, experiment_select=[1, 3, 4])

Edit BATCH_FOLDER below to match whichever batch you're analyzing.
"""
import os

from analysis import (
    K_BOLTZMANN_J_PER_K,
    compute_potential,
    load_combined_calib_pos_px,
    extract_protocol_events_from_folder,
    extract_protocol_stage_events_from_folder,
    discover_session_files,
    load_session_files,
    lambda_trap_nm_for_state,
    compute_work_split_kT,
    compute_trigger_energy_reference_kT,
    compute_experimental_probabilities,
    compute_weighted_mean_work_kT,
    compute_weighted_average_kT,
    plot_trap_relative_grid,
    plot_work_split_grid,
    plot_stage_vs_protocol_grid,
    export_for_matlab,
)
from analytical_protocols import (
    load_analytical_protocols,
    load_predicted_work_J,
    load_predicted_probabilities,
)
NM_PER_PX = 16

# ── Edit this to match the batch you're analyzing ──────────────────────────
BATCH_FOLDER = r"D:\Data\Sept_26\Test_Engine_28_09\28-09_Testing_Full_Engine\Exp_1_20260928_1647\batch01"

# Fields every selected experiment's own saved params must agree on before
# being pooled together - see _check_param_consistency().
_CONSISTENCY_FIELDS = ("ao_rate", "delta_t_s", "protocol_dt_s", "fire_00_prob")


def _resolve_prefixes(batch_folder: str, experiment_select) -> list:
    """
    experiment_select: None (use every saved experiment in batch_folder,
    discover_session_files()'s own timestamp-sorted order) or a list of
    1-based indices into that same order, e.g. [1, 3, 4] to pool only
    experiments 1, 3 and 4 out of a batch of 5 - excluding a known-bad run
    (2) from the analysis without touching the saved files themselves.
    """
    all_prefixes = discover_session_files(batch_folder)
    if not all_prefixes:
        raise FileNotFoundError(f"No saved experiments found in {batch_folder}")
    if experiment_select is None:
        return all_prefixes
    bad = [i for i in experiment_select if i < 1 or i > len(all_prefixes)]
    if bad:
        raise IndexError(
            f"experiment_select {bad} out of range - {batch_folder} has "
            f"{len(all_prefixes)} saved experiments (valid: 1..{len(all_prefixes)})"
        )
    return [all_prefixes[i - 1] for i in experiment_select]


def _check_param_consistency(prefixes: list, fields=_CONSISTENCY_FIELDS) -> tuple:
    """
    Loads each prefix's own saved params and groups them by their values
    for `fields`. If every selected experiment agrees, returns all of them
    unchanged. If not, keeps only the LARGEST agreeing group and prints
    exactly which experiments were excluded and what they disagreed on -
    "flag the mismatch, only pool the ones that match", rather than
    silently pooling sessions that weren't actually run the same way.

    Returns
    -------
    (kept_prefixes, agreed_values) - agreed_values is dict[field] -> value,
    the shared value every kept prefix has for each field in `fields`.
    """
    groups = {}   # tuple of field values -> list of prefixes
    for prefix in prefixes:
        params = load_session_files(prefix)["params"]
        key = tuple(getattr(params, f) for f in fields)
        groups.setdefault(key, []).append(prefix)

    if len(groups) == 1:
        (key, kept), = groups.items()
        return kept, dict(zip(fields, key))

    best_key = max(groups, key=lambda k: len(groups[k]))
    print(f"\nWARNING: selected experiments do not all agree on {fields} - "
          f"keeping only the {len(groups[best_key])}/{len(prefixes)} that match:")
    for key, plist in groups.items():
        tag = "KEPT    " if key == best_key else "EXCLUDED"
        for p in plist:
            print(f"  [{tag}] {os.path.basename(p)}: {dict(zip(fields, key))}")
    return groups[best_key], dict(zip(fields, best_key))


def _derive_kappa_T(batch_folder: str, kappa_N_per_m, T_K, nm_per_px: float, n_bins: int) -> tuple:
    """
    kappa_N_per_m=None (default): derive from this batch's own sibling
    calib/ folder - see module docstring. Pass explicitly to override or
    when no calib/ folder is available.
    T_K=None (default): 298.0.
    """
    if T_K is None:
        T_K = 298.0
    if kappa_N_per_m is not None:
        return kappa_N_per_m, T_K

    experiment_root = os.path.dirname(os.path.normpath(batch_folder))
    calib_folder = os.path.join(experiment_root, "calib")
    print(f"kappa_N_per_m not given - deriving from calibration data: {calib_folder}")
    calib_px = load_combined_calib_pos_px(calib_folder)
    x_center_nm = calib_px[:, 0].mean() * nm_per_px
    x_nm_equilibrium = calib_px[:, 0] * nm_per_px - x_center_nm
    potential = compute_potential(x_nm_equilibrium, n_bins=n_bins, T_K=T_K)
    print(f"  k_fit_N_per_m = {potential['k_fit_N_per_m']:.4e}   "
          f"k_eq_N_per_m = {potential['k_eq_N_per_m']:.4e}  (using k_eq_N_per_m)")
    return potential["k_eq_N_per_m"], T_K


def analyze_batch(batch_folder: str = BATCH_FOLDER,
                  experiment_select: list | None = None,
                  kappa_N_per_m: float | None = None,
                  T_K: float | None = None,
                  nm_per_px: float = NM_PER_PX,
                  calib_n_bins: int = 41,
                  make_plots: bool = True,
                  export_mat_path: str | None = None) -> tuple:
    """
    Exactly Main_script.ipynb's 4b section, as a function - see module
    docstring for how every parameter (ao_rate, t2, tf, fire_00_prob,
    kappa_N_per_m) gets resolved without being told by hand.

    Returns
    -------
    (events, lambdas, W_jump_kT, W_total_kT, summary)
        events      : dict[state] -> {"x_nm": (n_events_ij, protocol_frames), "t_s": (protocol_frames,)}
        lambdas     : dict[state] -> (protocol_frames,) - lambda_ij, nm
        W_jump_kT   : dict[state] -> (n_events_ij,) - the jump's own work, kT
        W_total_kT  : dict[state] -> (n_events_ij, protocol_frames) - cumulative TOTAL work, kT
        summary     : dict of the scalar probability-weighted results also printed below
    """
    prefixes = _resolve_prefixes(batch_folder, experiment_select)
    prefixes, agreed = _check_param_consistency(prefixes)
    ao_rate = agreed["ao_rate"]
    t2 = agreed["delta_t_s"]
    tf = agreed["protocol_dt_s"]
    fire_00_prob = agreed["fire_00_prob"]
    print(f"Pooling {len(prefixes)} experiment(s): {[os.path.basename(p) for p in prefixes]}")
    print(f"ao_rate={ao_rate}  t2={t2}  tf={tf}  fire_00_prob={fire_00_prob}  "
          f"(read from the selected sessions' own saved params)")

    kappa_N_per_m, T_K = _derive_kappa_T(batch_folder, kappa_N_per_m, T_K, nm_per_px, calib_n_bins)

    analytical_save_dir = os.path.join(batch_folder, "analytical_protocols_used")

    events = extract_protocol_events_from_folder(batch_folder, prefixes=prefixes)
    for state, ev in events.items():
        print(f"{state}: {ev['x_nm'].shape[0]} real fired events pooled across the selected experiments "
              f"({ev['x_nm'].shape[1]} samples/event)")

    print(f"Loading analytical protocols: {analytical_save_dir}")
    protocols = load_analytical_protocols(analytical_save_dir, t2, tf, ao_rate=ao_rate)

    lambdas = {
        state: lambda_trap_nm_for_state(state, events[state]["t_s"], protocols, tf, ao_rate)
        for state in events
    }

    # compute_work_split_kT() reports the jump's own work separately from
    # the smooth part that follows it (plot_work_split_grid draws the jump
    # as a dashed step) - see its docstring for why it deliberately uses a
    # different landing-frame convention than plot_trap_relative_grid.
    W_jump_kT, W_total_kT = {}, {}
    for state in events:
        W_jump, W_total = compute_work_split_kT(events[state]["x_nm"], lambdas[state],
                                                 kappa_N_per_m, T_K)
        W_jump_kT[state] = W_jump
        W_total_kT[state] = W_total

    for state in events:
        if events[state]["x_nm"].shape[0]:
            print(f"{state}: mean jump work = {W_jump_kT[state].mean():+.2f} kT   "
                  f"mean final work = {W_total_kT[state][:, -1].mean():+.2f} kT   "
                  f"(n={events[state]['x_nm'].shape[0]})")

    # Two reference lines for the work-split plot:
    #  - predicted_work_kT: the analytical solver's own predicted mean work
    #    per state - "what the optimal protocol was designed to extract".
    #  - trigger_reference_kT: -0.5*kappa*<x_trigger>^2/kT, computed
    #    straight from events (no solver dependency) - "what was naively
    #    available at the trigger instant".
    predicted_work_J = load_predicted_work_J(analytical_save_dir, t2, tf)
    kT_J = K_BOLTZMANN_J_PER_K * T_K
    predicted_work_kT = {state: predicted_work_J[state] / kT_J for state in predicted_work_J}

    trigger_reference_kT = compute_trigger_energy_reference_kT(events, kappa_N_per_m, T_K)

    stage_events = extract_protocol_stage_events_from_folder(batch_folder, prefixes=prefixes)
    stage_protocol_nm = {
        state: -lambda_trap_nm_for_state(state, stage_events[state]["t_s"], protocols, tf, ao_rate)
        for state in stage_events
    }

    if make_plots:
        plot_stage_vs_protocol_grid(stage_events, stage_protocol_nm)
        plot_trap_relative_grid(events, lambdas)
        plot_work_split_grid(events, W_jump_kT, W_total_kT, kappa_N_per_m, T_K,
                             predicted_work_kT=predicted_work_kT,
                             trigger_reference_kT=trigger_reference_kT)

    if export_mat_path:
        export_for_matlab(export_mat_path, events, lambdas, W_total_kT, kappa_N_per_m, T_K, tf,
                          jump_works_kT=W_jump_kT,
                          predicted_work_kT=predicted_work_kT,
                          trigger_reference_kT=trigger_reference_kT,
                          stage_events=stage_events,
                          stage_protocol_nm=stage_protocol_nm)

    # Probability-weighted mean work across all states, two ways:
    #  1. sum(exp_prob_ij * mean(w_ij)) - this batch's own real fire-count
    #     fractions (compute_experimental_probabilities()), no solver
    #     dependency. fire_00_prob corrects (0,0)'s count for the fact that
    #     only a fire_00_prob fraction of real (0,0) decodes ever get
    #     fired/counted - without this, (0,0)'s true probability is
    #     massively understated and every other state's is inflated.
    #  2. sum(analytical_prob_ij * mean(w_ij)) - the solver's own predicted
    #     P(m_0,m_t), which only exists on disk if
    #     run_single_parameter_point_cli.py has saved predicted_probabilities.json.
    # Both use W_total_kT (this batch's measured mean work per state), just
    # weighted by a different probability source.
    exp_prob = compute_experimental_probabilities(events, fire_00_prob=fire_00_prob)
    mean_work_exp_weighted_kT = compute_weighted_mean_work_kT(W_total_kT, exp_prob)
    mean_V_exp_weighted_kT = compute_weighted_average_kT(trigger_reference_kT, exp_prob)
    print(f"\nExperimental-probability-weighted mean work: {mean_work_exp_weighted_kT:+.3f} kT")
    print(f"Experimental-probability-weighted mean V:    {mean_V_exp_weighted_kT:+.3f} kT")

    summary = {
        "prefixes": prefixes,
        "ao_rate": ao_rate, "t2": t2, "tf": tf, "fire_00_prob": fire_00_prob,
        "kappa_N_per_m": kappa_N_per_m, "T_K": T_K,
        "mean_work_exp_weighted_kT": mean_work_exp_weighted_kT,
        "mean_V_exp_weighted_kT": mean_V_exp_weighted_kT,
    }
    try:
        analytical_prob = load_predicted_probabilities(analytical_save_dir)
        mean_work_analytical_weighted_kT = compute_weighted_mean_work_kT(W_total_kT, analytical_prob)
        mean_V_analytical_weighted_kT = compute_weighted_average_kT(trigger_reference_kT, analytical_prob)
        print(f"Analytical-probability-weighted mean work:   {mean_work_analytical_weighted_kT:+.3f} kT")
        print(f"Analytical-probability-weighted mean V:      {mean_V_analytical_weighted_kT:+.3f} kT")
        summary["mean_work_analytical_weighted_kT"] = mean_work_analytical_weighted_kT
        summary["mean_V_analytical_weighted_kT"] = mean_V_analytical_weighted_kT
    except FileNotFoundError as e:
        print(f"\nAnalytical-probability-weighted mean work/V: not available yet - {e}")

    return events, lambdas, W_jump_kT, W_total_kT, summary


if __name__ == "__main__":
    analyze_batch(export_mat_path=f"{BATCH_FOLDER}_events_work.mat")

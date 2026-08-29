"""
analytical_protocols.py
========================
Loads the real (non-placeholder) optimal-protocol shapes computed by
analytical_optimal_protocol_computation/scripts/run_single_parameter_point.py
into protocols_2ch.py's (m_0, m_t) -> stage-frame relative-displacement-in-nm
format, resampled onto the session's AO sample grid. Drop-in replacement for
BASE_PROTOCOLS/get_protocol_nm()'s output: pass the result to
split_channels() exactly as get_protocol_nm(state, ...) would be.

Deliberately self-contained (numpy + stdlib only) — does NOT import the
sdesim package (needs scipy/numba, lives in a separate venv). Only the
saved .npz files are shared between the two sides; not the code.

Sign convention (see EXPERIMENT_ARCHITECTURE.md, "Analytical protocols — file
format and loading design"): the analytical lambda(t) is the TRAP position,
lab frame. The stage carries the particle's frame, so the commanded
stage-frame displacement is the negation, applied once here at load time:

    Delta x_stage(t) = -Delta x_trap(t) = -lambda(t)

Worked-example cross-check from that doc: outcome (1,1) (particle
persistently at +x) should push the trap further +x (extraction intent) —
i.e. lambda(t) increasingly positive for (1,1) / increasingly negative for
its mirror (-1,-1), which is what run_single_parameter_point.py's saved
trajectories actually show.
"""
import glob
import os
import numpy as np

# Matches sdesim.constants.POSITION_SCALE — the analytical side works in SI
# meters throughout; nm is what the rest of this codebase (protocols_2ch,
# daq, camera) uses everywhere.
POSITION_SCALE_M_TO_NM = 1e9

# The 8 real (non-no-op) states any complete TWO-measurement protocol set
# must cover.
REAL_STATES = [(1, 1), (1, 0), (0, 1), (-1, 1), (-1, -1), (-1, 0), (0, -1), (1, -1)]

# For t_second_measurement=0 (single-measurement / delta_t=0 sessions - see
# experiment.py::run_session_2ch's single_measurement mode): only the
# diagonal is ever meaningful (m_0 == m_t is a tautology when there's zero
# time between the "two" measurements). This list is ONLY the two REAL
# fired-by-decode states - it deliberately excludes (0,0), which needs no
# separate listing here: (0,0) is always inserted as the same all-zeros
# no-op regardless of which files matched (see below), and
# run_session_2ch's rare-fire mechanism (SessionParams2Ch.fire_00_prob,
# the same unbiased Bernoulli draw as the two-measurement case) applies
# identically and automatically in single-measurement mode too - state ==
# (0,0) there is a real, legitimate decode outcome
# (decode_state_pair(x, x, xth_nm) when |x| <= xth_nm), not a special case.
SINGLE_MEASUREMENT_STATES = [(1, 1), (-1, -1)]


def _fmt(x):
    """Must match analytical_optimal_protocol_computation/sdesim/helpers.py::_fmt exactly - same filenames."""
    return f"{x:g}".replace(".", "p").replace("-", "neg")


def load_analytical_protocols(save_dir, t_second_measurement, t_protocol_end, ao_rate,
                              real_states=None):
    """
    Load every saved (m_0, m_t) optimal protocol for one analytical
    parameter point and resample lambda(t) onto the AO hardware's sample
    grid, in stage-frame nm (ready for protocols_2ch.split_channels()).

    Parameters
    ----------
    save_dir : str
        Directory run_single_parameter_point.py saved into (its
        `save_dir` arg, e.g. SAVE_DIR_OPTIMAL_PROTOCOLS_MULTIPLE_MEASUREMENT).
    t_second_measurement, t_protocol_end : float
        Must match the values that script was actually run with — these
        select the filenames, not a re-derivation of the physics.
    ao_rate : int
        AO hardware sample rate (Hz). Returned waveforms have length
        int(t_protocol_end * ao_rate), matching NICardOutDual's n_samples
        and what split_channels()/ao.load() expect.
    real_states : list[(int, int)], optional
        Which non-(0,0) states to check for and warn about if missing -
        defaults to the module's REAL_STATES (the 8 two-measurement
        states). Pass SINGLE_MEASUREMENT_STATES ([(1,1), (-1,-1)]) when
        t_second_measurement=0 - otherwise this would spuriously warn about
        6 off-diagonal states that were never expected to have files at all.

    Returns
    -------
    dict[(int, int), np.ndarray]
        (0, 0) is always present, as an exact all-zeros no-op array —
        protocols_2ch's convention (no protocol fires on (0,0), regardless
        of whatever near-zero numerical noise the analytical (0,0) file
        actually contains). Any of the 8 real states missing a saved file
        (negligible probability at that parameter point — no file was
        written for it) is simply absent from the dict; check
        `REAL_STATES` against the returned keys to see what's missing.

    Raises
    ------
    FileNotFoundError
        Zero files matched — almost always a t_second_measurement/
        t_protocol_end mismatch against what was actually run.
    """
    n_ao = int(t_protocol_end * ao_rate)
    pattern = os.path.join(
        save_dir,
        f"t2_{_fmt(t_second_measurement)}_tf_{_fmt(t_protocol_end)}_m0_*_mt_*.npz",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No analytical protocol files matched {pattern!r} — check "
            f"t_second_measurement/t_protocol_end match what "
            f"run_single_parameter_point.py was actually run with."
        )

    protocols = {(0, 0): np.zeros(n_ao)}
    t_ao = np.linspace(0.0, t_protocol_end, n_ao)

    for path in paths:
        with np.load(path) as data:
            m_0 = int(round(float(data["m_0"])))
            m_t = int(round(float(data["m_t"])))
            if (m_0, m_t) == (0, 0):
                continue  # keep the exact-zeros no-op above, not this file's numerical noise
            traj = data["optimal_trajectories"]  # rows: [time, X, X_b, lambda] (s, m)

        t_analytical = traj[0]
        lambda_trap_nm = traj[3] * POSITION_SCALE_M_TO_NM
        lambda_resampled_nm = np.interp(t_ao, t_analytical, lambda_trap_nm)

        protocols[(m_0, m_t)] = -lambda_resampled_nm  # trap-frame -> stage-frame

    missing = [s for s in (real_states if real_states is not None else REAL_STATES) if s not in protocols]
    if missing:
        print(f"load_analytical_protocols: no saved file for {missing} "
              f"(negligible probability at this parameter point?) — these "
              f"states will be absent from the returned dict.")

    return protocols


def load_predicted_work_J(save_dir, t_second_measurement, t_protocol_end, real_states=None):
    """
    The analytical solver's own predicted mean work per (m_0, m_t) state -
    each saved .npz's "mean_work" field (Joules, matches
    sdesim.parameter_sweep.run_single_parameter_point()'s "optimal" -
    confirmed against a real saved file: mean_work/kT lines up with this
    codebase's own measured cumulative work to within ~1%), for use as a
    reference line against the real (experimental) work plots - how much
    work the optimal protocol was actually designed to extract for this
    state, independent of anything measured.

    Same file-glob/state-resolution convention as
    load_analytical_protocols() (same save_dir/t_second_measurement/
    t_protocol_end must be used for both, or the reference won't match
    whatever protocols/xth_nm are actually loaded) - (0, 0) is always 0.0
    (no-op state, no work by construction), not read from any file.

    Returns
    -------
    dict[(int, int), float] - predicted mean work, Joules. A state with no
    saved file (matches load_analytical_protocols()'s own "negligible
    probability" case) is simply absent, same as there.
    """
    pattern = os.path.join(
        save_dir,
        f"t2_{_fmt(t_second_measurement)}_tf_{_fmt(t_protocol_end)}_m0_*_mt_*.npz",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No analytical protocol files matched {pattern!r} — check "
            f"t_second_measurement/t_protocol_end match what "
            f"run_single_parameter_point.py was actually run with."
        )

    predicted_work_J = {(0, 0): 0.0}
    for path in paths:
        with np.load(path) as data:
            m_0 = int(round(float(data["m_0"])))
            m_t = int(round(float(data["m_t"])))
            if (m_0, m_t) == (0, 0):
                continue  # keep the exact-zero above, not this file's numerical noise
            predicted_work_J[(m_0, m_t)] = float(data["mean_work"])

    missing = [s for s in (real_states if real_states is not None else REAL_STATES) if s not in predicted_work_J]
    if missing:
        print(f"load_predicted_work_J: no saved file for {missing} - "
              f"these states will be absent from the returned dict.")

    return predicted_work_J

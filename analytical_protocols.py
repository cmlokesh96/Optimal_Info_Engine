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

# The 8 real (non-no-op) states any complete protocol set must cover.
REAL_STATES = [(1, 1), (1, 0), (0, 1), (-1, 1), (-1, -1), (-1, 0), (0, -1), (1, -1)]


def _fmt(x):
    """Must match analytical_optimal_protocol_computation/sdesim/helpers.py::_fmt exactly - same filenames."""
    return f"{x:g}".replace(".", "p").replace("-", "neg")


def load_analytical_protocols(save_dir, t_second_measurement, t_protocol_end, ao_rate):
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

    missing = [s for s in REAL_STATES if s not in protocols]
    if missing:
        print(f"load_analytical_protocols: no saved file for {missing} "
              f"(negligible probability at this parameter point?) — these "
              f"states will be absent from the returned dict.")

    return protocols

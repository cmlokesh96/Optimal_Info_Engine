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

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

    # Symmetric binning: pick a bin width from the data range/n_bins as
    # before, but then lay bin CENTERS at 0, +/-bin_width, +/-2*bin_width,
    # ... (so edges fall at +/-0.5*bin_width, +/-1.5*bin_width, ...) rather
    # than np.histogram's default of splitting [min, max] into n_bins
    # bins with no guarantee any edge/center lands on 0.
    bin_width = (x.max() - x.min()) / n_bins
    n_side = int(np.ceil(max(abs(x.min()), abs(x.max())) / bin_width))
    centers_all = np.arange(-n_side, n_side + 1) * bin_width
    edges = (np.arange(-n_side, n_side + 2) - 0.5) * bin_width

    counts, edges = np.histogram(x, bins=edges)
    centers = centers_all

    valid = counts > 0
    x_valid = centers[valid]
    P = counts[valid] / (counts.sum() * bin_width)
    U_raw = -np.log(P)

    # Fit only the well-sampled bins near the well bottom — tails out past
    # U/kT = 6 are sparse-count bins where -log(P) noise blows up, and a
    # true anharmonic trap deviates from parabolic out there anyway, so
    # including them would bias the quadratic fit away from the harmonic
    # region it's supposed to describe.
    fit_mask = (U_raw - U_raw.min()) < 5.5

    # Fit on the raw (unshifted) values — an additive constant doesn't
    # affect the quadratic/linear coefficients (stiffness, x0).
    a, b, c = np.polyfit(x_valid[fit_mask], U_raw[fit_mask], 2)
    x0_fit_nm = -b / (2 * a)
    fit_offset = a * x0_fit_nm**2 + b * x0_fit_nm + c   # fit's value at its own vertex

    U_over_kT = U_raw - fit_offset   # shift so the fit's minimum sits at 0
    k_fit_over_kT = 2 * a            # 1/nm^2
    x_mean_nm = float(np.mean(x))
    k_eq_over_kT = 1.0 / np.var(x, ddof=1)   # 1/nm^2 — equipartition is about the mean, not x0_fit

    return {
        "bin_centers_nm": x_valid,
        "U_over_kT": U_over_kT,
        "P_density": P,
        "fit_mask": fit_mask,
        "x_mean_nm": x_mean_nm,
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
    """
    Plot the equilibrium distribution P(x) alongside U(x)/kT (with the
    parabolic fit overlaid), per compute_potential(). Points excluded from
    the fit (U/kT >= 5 from the well bottom) are shown hollow/gray in the
    U(x)/kT panel so it's clear how much of the tail the fit ignored.
    """
    x = result["bin_centers_nm"]
    U = result["U_over_kT"]
    P = result["P_density"]
    fit_mask = result["fit_mask"]
    a, b, c = result["fit_coeffs"]
    offset = result["fit_offset"]

    x_fit = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 200)
    U_fit = a * x_fit**2 + b * x_fit + c - offset
    U_eq_fit = 0.5 * result["k_eq_over_kT"] * (x_fit - result["x_mean_nm"])**2

    fig, (ax_p, ax_u) = plt.subplots(1, 2, figsize=(13, 5))

    ax_p.plot(x, P, "o-", ms=4, color="seagreen")
    ax_p.set_xlabel("x (nm)")
    ax_p.set_ylabel("P(x)  (equilibrium distribution)")
    ax_p.set_title(f"Equilibrium distribution (n={result['n_samples']})")
    ax_p.grid(True, alpha=0.3)

    ax_u.plot(x[fit_mask], U[fit_mask], "o", ms=4, color="steelblue",
              label="U/kT = -ln P(x)  (< 5)")
    ax_u.plot(x_fit, U_fit, "-", color="tomato",
              label=f"parabolic fit (k_fit = {result['k_fit_N_per_m']:.3e} N/m)")
    ax_u.plot(x_fit, U_eq_fit, "--", color="blue",
              label=f"equipartition (k_eq = {result['k_eq_N_per_m']:.3e} N/m)")
    ax_u.axvline(result["x0_fit_nm"], color="gray", lw=0.8, ls="--")
    ax_u.set_xlabel("x (nm)")
    ax_u.set_ylabel("U(x) / kT")
    ax_u.set_title(f"Trap potential  (T={result['T_K']:.0f} K)\n"
                   f"k_fit = {result['k_fit_N_per_m']:.3e} N/m   "
                   f"k_eq = {result['k_eq_N_per_m']:.3e} N/m")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compute_equilibrium_state_stats(x_nm: np.ndarray, fps: float, xth_nm: float,
                                    delta_t_s: float) -> dict:
    """
    Model-free empirical P(m_0, m_t) and mean trigger position per state,
    straight from an equilibrium (undisturbed) x(t) trace - decodes every
    (x[i], x[i+offset]) pair the same way run_session_2ch's real decide
    checkpoint does (protocols_2ch.decode_symbol/STATE_TO_CODE), where
    offset is delta_t_s converted to frames at fps.

    The TRIGGER position is x[i+offset] (the CURRENT, second-measurement
    sample), not x[i] - this is the same physical instant
    compute_trigger_energy_reference_kT's x_trigger is (events[state]
    ["x_nm"][:, 0], the raw camera reading at the fire's own decide
    checkpoint frame): experiment.py's decide checkpoint reads x_curr at
    that same frame and immediately fires from it (x_prev, from
    delta_t_s earlier, only contributes to the DECODE, not to where the
    protocol actually launches from).

    Directly comparable to two other estimates of the same per-state
    quantities: analytical_protocols.load_predicted_probabilities()/
    load_predicted_mean_x_trigger_nm() (the solver's own assumptions) and
    a real batch's own compute_experimental_probabilities()/
    compute_trigger_energy_reference_kT()-adjacent x_nm[:, 0].mean() (what
    actually got measured while firing protocols) - a 3-way check with no
    shared assumptions between any two of the three.

    Parameters
    ----------
    x_nm      : 1D equilibrium trace, nm (e.g. combined_calib_px[:, 0] *
        NM_PER_PX - x_center_nm). If this pools multiple separate
        calibration runs concatenated end-to-end (load_combined_calib_pos_px),
        the handful of pairs straddling a run boundary are decoded across
        two unrelated runs - negligible given offset << a single run's
        length, not worth splitting out.
    fps       : camera frame rate x_nm was recorded at.
    xth_nm    : decode threshold - must match whatever xth_nm the real
        session(s) being compared against actually used.
    delta_t_s : spacing between the two decision samples - must match
        ANALYTICAL_T2/DELTA_T_S (0 for a single-measurement comparison,
        where x_prev and x_curr collapse to the same instant, mirroring
        experiment.py's single_measurement convention).

    Returns
    -------
    dict[state] -> {"prob": float, "mean_x_trigger_nm": float, "n": int}
        prob sums to 1 across the 9 states (0.0/nan/0 for a state with no
        occurrences in this trace).
    """
    from protocols_2ch import decode_symbol, STATE_TO_CODE

    x = np.asarray(x_nm, dtype=float)
    offset = max(1, round(delta_t_s * fps)) if delta_t_s > 0 else 0

    if offset == 0:
        x_prev_all, x_curr_all = x, x
    else:
        x_prev_all, x_curr_all = x[:-offset], x[offset:]

    valid = np.isfinite(x_prev_all) & np.isfinite(x_curr_all)
    x_prev_all, x_curr_all = x_prev_all[valid], x_curr_all[valid]

    decode = np.vectorize(decode_symbol, otypes=[int])
    m_0_all = decode(x_prev_all, xth_nm)
    m_t_all = decode(x_curr_all, xth_nm)

    total = len(x_prev_all)
    out = {}
    for state in STATE_TO_CODE:
        mask = (m_0_all == state[0]) & (m_t_all == state[1])
        n = int(mask.sum())
        out[state] = {
            "prob": (n / total) if total else 0.0,
            "mean_x_trigger_nm": float(x_curr_all[mask].mean()) if n else float("nan"),
            "n": n,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
# Per-state protocol-event extraction, cumulative work, plotting, MATLAB export
#
# Camera-vs-theory frame note (load-bearing, see below): the camera measures
# the particle in the LAB frame, which already has the trap's motion baked
# in through the stage - x_cam(t) = X_true(t) - lambda(t), NOT X_true(t)
# alone (X_true is the particle-relative-to-trap coordinate the analytical
# protocols' physics is written in). Rather than reconstructing X_true, the
# work formula below is algebraically simplified to use x_cam directly (see
# compute_cumulative_work_kT's docstring for the one-line derivation).
# ══════════════════════════════════════════════════════════════════════════

def extract_protocol_events(pos: np.ndarray, params, states=None,
                            active_states=None, protocols_loaded=None) -> dict:
    """
    Split a saved/in-memory session's pos[:,3] trigger column into one
    (n_events, protocol_frames) x_nm matrix per requested state - each row
    is one real occurrence of that state actually firing its protocol,
    over [0, protocol_dt_s) measured from the decide/fire checkpoint frame.

    pos[:,3] alone can't prove a code was a FIRE (protocols_2ch.STATE_TO_CODE
    is written at every decide checkpoint, fired or not - see
    run_session_2ch's should_fire). Pass the active_states/protocols_loaded
    that run's run_session_2ch() result (or saved _params.pkl) actually had,
    and this asserts every requested state could really have fired under
    them - not just trusts the code.

    (0,0) is handled separately from the other 8 states throughout: it's
    gated only by params.fire_00_prob (independent of active_states/
    protocols_loaded - see SessionParams2Ch.fire_00_prob), and a fired
    (0,0) is marked with ZERO_FIRE_CODE, not STATE_TO_CODE[(0,0)] (which
    remains the code for an ordinary, non-fired (0,0) checkpoint - still
    the common case, and NOT what this function extracts for (0,0)).

    Parameters
    ----------
    pos     : the session's pos array (n_frames, 4).
    params  : that session's SessionParams2Ch (need .fps, .protocol_frames,
        .fire_00_prob).
    states  : iterable of (m_0, m_t) tuples to extract, or None (default) to
        auto-resolve: all 8 real (non-(0,0)) states, narrowed by
        active_states/protocols_loaded if those aren't None either - i.e.
        active_states=None (the session's own "full engine, every state can
        fire" convention) means every state gets extracted here too. (0,0)
        is included in this auto-resolution only if params.fire_00_prob > 0
        (otherwise it could never have fired, so there's nothing to extract).
    active_states, protocols_loaded : from the run_session_2ch() result dict
        (or the saved _params.pkl) - None means "no restriction" (matches
        run_session_2ch's own None-means-unrestricted convention). Neither
        applies to (0,0) - see above.

    Returns
    -------
    dict[state] -> {"x_nm": (n_events, protocol_frames) ndarray,
                    "t_s": (protocol_frames,) ndarray,
                    "frame_idx": (n_events,) ndarray of fire-start frames}
        A state with zero real occurrences gets an (0, protocol_frames)
        x_nm - not an error, just nothing to average/plot for it.
    """
    states, t_s, protocol_frames, frame_idx_by_state = _resolve_fire_events(
        pos, params, states, active_states, protocols_loaded)

    out = {}
    for state in states:
        frame_idx = frame_idx_by_state[state]
        x_nm = np.empty((len(frame_idx), protocol_frames))
        for i, fi in enumerate(frame_idx):
            x_nm[i] = pos[fi:fi + protocol_frames, 1]

        out[state] = {"x_nm": x_nm, "t_s": t_s, "frame_idx": frame_idx}

    return out


def _resolve_fire_events(pos: np.ndarray, params, states, active_states, protocols_loaded):
    """
    Shared state-resolution/validation and frame_idx lookup behind
    extract_protocol_events() and extract_protocol_stage_events() - see
    extract_protocol_events()'s docstring for the states/active_states/
    protocols_loaded semantics this implements.

    Returns
    -------
    (states, t_s, protocol_frames, frame_idx_by_state) where
    frame_idx_by_state[state] is the (n_events,) array of fire-start
    frames for that state, already dropping events truncated by the
    session ending before protocol_dt_s had time to fully elapse.
    """
    from protocols_2ch import STATE_TO_CODE, ZERO_FIRE_CODE

    real_states = [s for s in STATE_TO_CODE if s != (0, 0)]
    fire_00_prob = getattr(params, "fire_00_prob", 0.0)

    if states is None:
        states = real_states
        if active_states is not None:
            states = [s for s in states if s in active_states]
        if protocols_loaded is not None:
            states = [s for s in states if s in protocols_loaded]
        if fire_00_prob > 0:
            states = states + [(0, 0)]
    else:
        # Explicit states are still validated - an explicitly-requested
        # state that couldn't possibly have fired is almost always a
        # mistake, not something to silently drop.
        for state in states:
            if state == (0, 0):
                if fire_00_prob <= 0:
                    raise ValueError(
                        "(0,0) was requested but this session's "
                        "fire_00_prob was 0 - it could never have fired."
                    )
                continue
            if active_states is not None and state not in active_states:
                raise ValueError(
                    f"{state} was not in this session's active_states "
                    f"({active_states}) - it could never have fired, so its "
                    f"pos[:,3] code (if any) is not a real event."
                )
            if protocols_loaded is not None and state not in protocols_loaded:
                raise ValueError(
                    f"{state} had no analytical protocol loaded in this "
                    f"session (protocols_loaded={protocols_loaded}) - it "
                    f"could never have fired."
                )

    protocol_frames = params.protocol_frames
    t_s = np.arange(protocol_frames) / params.fps
    n_frames = len(pos)

    frame_idx_by_state = {}
    for state in states:
        code = ZERO_FIRE_CODE if state == (0, 0) else STATE_TO_CODE[state]
        frame_idx = np.flatnonzero(pos[:, 3] == code)
        # Drop events truncated by the session ending before protocol_dt_s
        # had time to fully elapse - can't fill a full row for those.
        frame_idx = frame_idx[frame_idx + protocol_frames <= n_frames]
        frame_idx_by_state[state] = frame_idx

    return states, t_s, protocol_frames, frame_idx_by_state


def extract_protocol_stage_events(pos: np.ndarray, DataSync: np.ndarray, params, states=None,
                                  active_states=None, protocols_loaded=None) -> dict:
    """
    Same per-state event windowing as extract_protocol_events(), but pulls
    the physical Stage-channel readback (DataSync[:,1], nm - see
    experiment.py::build_datasync) instead of the camera x - i.e. what the
    stage actually did, independent of camera tracking entirely. Diagnostic
    for comparing against a loaded analytical protocol's stage-frame curve
    (analytical_protocols[state], or -lambda_trap_nm_for_state(...)) without
    touching x_nm/compute_cumulative_work_kT at all.

    Each event's row is relative to that event's OWN pre-fire baseline
    (DataSync[fi, 1], the stage voltage/position it happened to be sitting
    at when this particular fire started - carried over from wherever the
    previous fire's ao1 coarse offset left it) - so stage_nm[..., 0] == 0
    for every row by construction, and later samples show the actual
    stage displacement caused by THIS fire, comparable across events/states
    regardless of each one's arbitrary starting offset.

    Returns
    -------
    dict[state] -> {"stage_nm": (n_events, protocol_frames) ndarray,
                    "t_s": (protocol_frames,) ndarray,
                    "frame_idx": (n_events,) ndarray of fire-start frames}
    """
    states, t_s, protocol_frames, frame_idx_by_state = _resolve_fire_events(
        pos, params, states, active_states, protocols_loaded)

    stage_nm_all = DataSync[:, 1]
    n = min(len(pos), len(DataSync))

    out = {}
    for state in states:
        frame_idx = frame_idx_by_state[state]
        frame_idx = frame_idx[frame_idx + protocol_frames <= n]

        stage_nm = np.empty((len(frame_idx), protocol_frames))
        for i, fi in enumerate(frame_idx):
            stage_nm[i] = stage_nm_all[fi:fi + protocol_frames] - stage_nm_all[fi]

        out[state] = {"stage_nm": stage_nm, "t_s": t_s, "frame_idx": frame_idx}

    return out


# ── Batch-folder loading and cross-experiment pooling ─────────────────────────
# run_batch_2ch() (experiment.py) now saves every experiment of a batch into
# ONE folder (savepath=os.path.join(batch_folder, f"exp{ii}")) instead of
# same-directory siblings sharing only a filename prefix - these three
# functions are what actually pools statistics across all of a batch's
# hour-long experiments (or across several batch folders), rather than
# analyzing one saved file at a time.

def discover_session_files(folder: str) -> list:
    """
    Every saved session's shared file-prefix inside folder (each has a
    matching _pos.npy/_params.pkl pair from save_session()) - sorted so
    exp1, exp2, ... load back in the order they were run.
    """
    pos_files = sorted(glob.glob(os.path.join(folder, "*_pos.npy")))
    return [p[:-len("_pos.npy")] for p in pos_files]


def load_session_files(prefix: str) -> dict:
    """
    Load one saved session's pos + DataSync + params + active_states +
    protocols_loaded + protocols back from its _pos.npy/_DataSync.npy/
    _params.pkl/_protocols.pkl files, given the shared prefix
    discover_session_files() returns (or build one by hand: the exact
    string save_session() used before appending
    "_pos.npy"/"_DataSync.npy"/"_params.pkl"/"_protocols.pkl"). DataSync
    and protocols are each None if their file isn't present (older saves,
    or save_session() called without one).
    """
    pos = np.load(f"{prefix}_pos.npy")
    DataSync = (np.load(f"{prefix}_DataSync.npy")
               if os.path.exists(f"{prefix}_DataSync.npy") else None)
    with open(f"{prefix}_params.pkl", "rb") as f:
        meta = pickle.load(f)
    protocols = None
    if os.path.exists(f"{prefix}_protocols.pkl"):
        with open(f"{prefix}_protocols.pkl", "rb") as f:
            protocols = pickle.load(f)
    return {
        "pos": pos,
        "DataSync": DataSync,
        "params": meta["params"],
        "active_states": meta["active_states"],
        "protocols_loaded": meta["protocols_loaded"],
        "protocols": protocols,
    }


def load_batch_protocols(folder: str) -> dict:
    """
    The (m_0, m_t) -> stage-frame waveform dict every session in a
    run_batch_2ch() batch folder actually fired with, loaded from each
    session's own saved _protocols.pkl (see save_session()'s protocols=
    arg) - makes the batch folder self-contained: no dependency on
    whatever's currently loaded in a notebook or sitting in
    analytical_optimal_protocol_computation's output folder (which can
    silently change - reruns of the solver overwrite same-(t2, tf, m_0,
    m_t) files).

    Every session in folder must have a saved _protocols.pkl (raises
    FileNotFoundError otherwise - older batches predating this save need
    a manually-loaded analytical_protocols dict instead, same as before),
    and if there's more than one session, all their protocols must agree
    exactly (raises ValueError otherwise - run_batch_2ch() only ever
    passes one protocols dict through to every experiment in a batch, so
    a mismatch means these sessions aren't really one batch).
    """
    protocols = None
    for prefix in discover_session_files(folder):
        path = f"{prefix}_protocols.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - load_batch_protocols() needs every "
                f"session in {folder!r} to have a saved _protocols.pkl "
                f"(see save_session()'s protocols= arg)."
            )
        with open(path, "rb") as f:
            this_protocols = pickle.load(f)
        if protocols is None:
            protocols = this_protocols
        elif set(protocols) != set(this_protocols) or not all(
            np.allclose(protocols[s], this_protocols[s]) for s in protocols
        ):
            raise ValueError(
                f"{prefix}'s saved protocols don't match the rest of "
                f"{folder!r} - these sessions weren't fired with the same "
                f"protocol set, so they aren't really one batch."
            )
    return protocols


def load_combined_calib_pos_px(folder: str) -> np.ndarray:
    """
    Stack every *_calib_pos_px.npy file in folder (as saved by
    experiment.calibrate_trap_center(savepath=...), one file per run) into
    one combined array - more pooled frames give a statistically better
    trap-center estimate and a better-sampled potential than any single
    calibration run alone.

    Columns match calibrate_trap_center()'s own calib_pos_px exactly:
    [x_px, y_px, time_s]. Column 2 (time_s) is each run's own relative time
    from ITS OWN start, not a continuous timeline across runs - fine for
    the position-only statistics this feeds (trap-center mean, potential
    histogram), not for time-series plotting spanning multiple runs.
    """
    files = sorted(glob.glob(os.path.join(folder, "*_calib_pos_px.npy")))
    if not files:
        raise FileNotFoundError(f"No *_calib_pos_px.npy files found in {folder}")
    return np.concatenate([np.load(f) for f in files], axis=0)


def _merge_events(combined: dict, new: dict, data_key: str = "x_nm") -> None:
    """Concatenate new's per-state data_key/frame_idx onto combined, in
    place - t_s must agree across everything merged (same protocol_dt_s/
    fps), since a mismatch means these sessions aren't really comparable
    experiments. data_key selects which per-event array is being pooled
    ("x_nm" for extract_protocol_events(), "stage_nm" for
    extract_protocol_stage_events())."""
    for state, data in new.items():
        if state not in combined:
            combined[state] = {data_key: [data[data_key]], "t_s": data["t_s"],
                               "frame_idx": [data["frame_idx"]]}
        else:
            if not np.allclose(combined[state]["t_s"], data["t_s"]):
                raise ValueError(
                    f"Inconsistent protocol timing for {state} across "
                    f"sessions being combined (different protocol_dt_s/fps?) "
                    f"- these experiments aren't directly poolable."
                )
            combined[state][data_key].append(data[data_key])
            combined[state]["frame_idx"].append(data["frame_idx"])


def _finalize_merge(combined: dict, data_key: str = "x_nm") -> dict:
    protocol_frames = None
    for state, data in combined.items():
        protocol_frames = data[data_key][0].shape[1] if data[data_key] else protocol_frames
    out = {}
    for state, data in combined.items():
        arr = (np.concatenate(data[data_key], axis=0) if data[data_key]
              else np.empty((0, protocol_frames or 0)))
        frame_idx = (np.concatenate(data["frame_idx"], axis=0) if data["frame_idx"]
                    else np.empty((0,), dtype=int))
        out[state] = {data_key: arr, "t_s": data["t_s"], "frame_idx": frame_idx}
    return out


def extract_protocol_events_from_folder(folder: str, states=None, prefixes=None) -> dict:
    """
    Load every saved session in folder (one run_batch_2ch() batch's worth)
    and pool their extract_protocol_events() results into one combined
    per-state dict - x_nm rows from every experiment in the folder stacked
    together, so e.g. 8 one-hour experiments' worth of a given state's real
    fired events become one (n_total_events, protocol_frames) matrix instead
    of 8 separate ones. states=None (default) auto-resolves per-session as
    extract_protocol_events() does (each session's own active_states/
    protocols_loaded), then pools across whatever the union of sessions covers.

    prefixes: None (default) pools every session discover_session_files(folder)
    finds, same as before this parameter existed. Pass an explicit subset
    (e.g. a few of discover_session_files(folder)'s own entries) to pool
    only those - excluding a known-bad run from the analysis without
    touching its saved files.
    """
    combined: dict = {}
    for prefix in (prefixes if prefixes is not None else discover_session_files(folder)):
        sess = load_session_files(prefix)
        ev = extract_protocol_events(sess["pos"], sess["params"], states=states,
                                     active_states=sess["active_states"],
                                     protocols_loaded=sess["protocols_loaded"])
        _merge_events(combined, ev)
    return _finalize_merge(combined)


def extract_protocol_events_from_folders(folders, states=None) -> dict:
    """
    Same pooling as extract_protocol_events_from_folder(), across several
    batch folders at once (e.g. combining multiple 8-experiment batches run
    on different days) - each folder's own sessions are still discovered/
    loaded independently via discover_session_files()/load_session_files().
    """
    combined: dict = {}
    for folder in folders:
        for prefix in discover_session_files(folder):
            sess = load_session_files(prefix)
            ev = extract_protocol_events(sess["pos"], sess["params"], states=states,
                                         active_states=sess["active_states"],
                                         protocols_loaded=sess["protocols_loaded"])
            _merge_events(combined, ev)
    return _finalize_merge(combined)


def extract_protocol_stage_events_from_folder(folder: str, states=None, prefixes=None) -> dict:
    """
    Same pooling as extract_protocol_events_from_folder(), but for the
    physical Stage-channel readback (extract_protocol_stage_events()) -
    every session in folder must have a saved DataSync (load_session_files()
    raises KeyError-like access if a session is missing one; older saves
    without DataSync should be pointed at extract_protocol_events_from_folder
    instead, or excluded from folder).

    prefixes: same meaning as extract_protocol_events_from_folder()'s -
    None (default) pools every session in folder; an explicit subset pools
    only those.
    """
    combined: dict = {}
    for prefix in (prefixes if prefixes is not None else discover_session_files(folder)):
        sess = load_session_files(prefix)
        if sess["DataSync"] is None:
            raise FileNotFoundError(
                f"{prefix}_DataSync.npy not found - "
                f"extract_protocol_stage_events_from_folder() needs every "
                f"session in {folder!r} to have a saved DataSync."
            )
        ev = extract_protocol_stage_events(sess["pos"], sess["DataSync"], sess["params"],
                                           states=states,
                                           active_states=sess["active_states"],
                                           protocols_loaded=sess["protocols_loaded"])
        _merge_events(combined, ev, data_key="stage_nm")
    return _finalize_merge(combined, data_key="stage_nm")


def lambda_trap_nm_for_state(state: tuple, t_s: np.ndarray, analytical_protocols: dict,
                             protocol_dt_s: float, ao_rate: int) -> np.ndarray:
    """
    The (single, deterministic) trap-frame lambda(t) for one state, resampled
    onto t_s (the camera/fps grid extract_protocol_events() used) - every
    real event of the same state fires the identical commanded waveform, so
    this is computed once per state, not once per event.

    analytical_protocols[state] is already in STAGE frame (see
    analytical_protocols.py: protocols[(m0,mt)] = -lambda_trap_nm) - negate
    back to get the trap-frame lambda(t) the physics/work formula needs.
    """
    n_ao = int(protocol_dt_s * ao_rate)
    t_ao = np.linspace(0.0, protocol_dt_s, n_ao)
    lambda_trap_nm_ao_grid = -analytical_protocols[state]
    return np.interp(t_s, t_ao, lambda_trap_nm_ao_grid)


def compute_cumulative_work_kT(x_cam_nm: np.ndarray, lambda_trap_nm: np.ndarray,
                               kappa_N_per_m: float, T_K: float) -> np.ndarray:
    """
    Cumulative work W(t)/kT for one or more events of one state, using the
    RAW camera reading directly (no reconstruction of the trap-relative
    particle coordinate needed for the smooth part) - derived as follows.

    The camera sits in the lab frame, same as the (truly stationary) trap -
    but the particle's LAB-frame position already reflects the stage having
    carried it, so what the camera measures is
        x_cam(t) = X_true(t) - lambda(t)
    where X_true is the trap-relative coordinate the potential
    U = (kappa/2)(X_true-lambda)^2 is written in (X_true = x_cam + lambda).

    Standard stochastic-energetics work differential (moving-trap agent):
        dW = (dU/dlambda) dlambda = kappa*(lambda - X_true) dlambda

    Substituting X_true = x_cam + lambda cancels every lambda term for the
    SMOOTH part of the motion:
        dW = kappa*(lambda - x_cam - lambda) dlambda = -kappa * x_cam * dlambda

    The initial jump (lambda_trap_nm[0] is already lambda(0+), the value
    right after the protocol's instantaneous initial jump from lambda(0-)=0
    - see analytical_protocols.py/visualization.py::_plot_lambda_jump) needs
    separate, explicit handling: x_cam_nm[...,0] is recorded from
    run_session_2ch's camera read taken BEFORE that fire's ao.load()/
    ao.start() call, i.e. genuinely PRE-jump (x_cam(0-)) - so x_cam_nm and
    lambda_trap_nm are NOT time-aligned at index 0, and naively diff-ing
    lambda_trap_nm as-is silently drops the jump's own work contribution.

    Since X_true never itself jumps (only lambda does), X_true(0) equals
    x_cam_nm[...,0] exactly (lambda(0-)=0 at that pre-fire instant), which
    gives a closed-form jump work (holding X_true fixed at its pre-fire
    value across the jump, standard stochastic-energetics boundary term):
        W_jump = kappa*(lambda(0+)^2/2 - X_true(0)*lambda(0+))
    and a reconstructed "just after the jump" camera reading
    (unmeasurable directly - camera doesn't sample fast enough to resolve
    the ~1ms jump against its own ~10ms frame period):
        x_cam(0+) = X_true(0) - lambda(0+)
    used as the start of the first smooth trapezoidal step into
    x_cam_nm[...,1]. Every step from index 1 onward is genuinely
    post-jump and time-aligned, so the plain -kappa*x_cam*dlambda
    trapezoidal sum applies there unmodified. (When lambda_trap_nm[0]==0 -
    no jump for this state/protocol - W_jump and the reconstructed value
    both reduce to their no-jump values automatically, no branching needed.)

    Since lambda(t) is a known deterministic curve, not itself stochastic,
    none of this integration has any Ito/Stratonovich ambiguity - it's an
    ordinary Riemann-Stieltjes integral of the real (stochastic) x_cam
    against the known dlambda.

    Parameters
    ----------
    x_cam_nm      : (n_events, protocol_frames) or (protocol_frames,) - raw
                    camera x, nm (same array extract_protocol_events() returns).
    lambda_trap_nm: (protocol_frames,) - from lambda_trap_nm_for_state().
    kappa_N_per_m, T_K : must match the values the analytical protocols were
                    generated with (section 3b's ANALYTICAL_KAPPA_N_PER_M/_T_K).

    Returns
    -------
    W_cum_kT : same shape as x_cam_nm - cumulative work in kT units, W_cum[...,0] = 0.
    """
    x_cam_nm = np.atleast_2d(x_cam_nm)
    x_m = x_cam_nm * 1e-9
    lambda_m = lambda_trap_nm * 1e-9
    kT_J = K_BOLTZMANN_J_PER_K * T_K

    lambda0_m = lambda_m[0]
    X_true_0_m = x_m[:, 0]                              # = x_cam(0-), exact
    jump_work_J = kappa_N_per_m * (lambda0_m**2 / 2 - X_true_0_m * lambda0_m)
    x_cam_0_plus_m = X_true_0_m - lambda0_m             # reconstructed x_cam(0+)

    x_ext_m = x_m.copy()
    x_ext_m[:, 0] = x_cam_0_plus_m                      # only for the smooth integral below

    d_lambda_m = np.diff(lambda_m)                       # (protocol_frames-1,)
    x_mid_m = 0.5 * (x_ext_m[:, :-1] + x_ext_m[:, 1:])   # trapezoidal midpoint
    dW_smooth_J = -kappa_N_per_m * x_mid_m * d_lambda_m  # (n_events, protocol_frames-1)

    W_cum_kT = np.zeros_like(x_m)
    if x_m.shape[1] > 1:
        W_cum_kT[:, 1] = (jump_work_J + dW_smooth_J[:, 0]) / kT_J
    if x_m.shape[1] > 2:
        W_cum_kT[:, 2:] = W_cum_kT[:, 1:2] + np.cumsum(dW_smooth_J[:, 1:], axis=1) / kT_J

    return W_cum_kT.reshape(np.atleast_2d(x_cam_nm).shape if x_cam_nm.ndim > 1
                             else (x_cam_nm.shape[-1],))


def compute_work_split_kT(x_cam_nm: np.ndarray, lambda_trap_nm: np.ndarray,
                          kappa_N_per_m: float, T_K: float) -> tuple:
    """
    Same total as compute_cumulative_work_kT(), but the jump's own
    contribution is computed and returned as an explicit, separate
    quantity instead of being folded into the cumulative curve's first
    step.

    W_jump is the same closed-form boundary term as
    compute_cumulative_work_kT() (X_true held fixed at its pre-fire value
    x_cam_nm[...,0] across the jump), except the jump's target is taken as
    lambda_trap_nm[1] - the first sample that is actually, genuinely
    time-aligned with a real camera reading (x_cam_nm[...,1]) - rather
    than lambda_trap_nm[0] (already lambda(0+), but paired with
    x_cam_nm[...,0]'s PRE-fire reading - see compute_cumulative_work_kT's
    docstring on why those two aren't time-aligned at index 0). Past
    index 1, nothing needs any reconstruction: the smooth part is the
    plain -kappa*x_cam*dlambda trapezoidal sum over indices 1, 2, 3, ...
    directly, real samples only.

    Deliberately uses the protocol's OWN idealized landing frame (index 1),
    not plot_trap_relative_grid()'s index-2 convention (which additionally
    accounts for this hardware's confirmed extra ~1-frame actuation delay,
    verified via the AI Stage-channel readback - see that function's
    docstring). The two are intentionally different: plot_trap_relative_grid
    exists to look visually continuous, and index 2 achieves that; this
    function exists to be compared against the solver's own predicted
    mean_work/mean_work_jump, and index 1 tracks that prediction more
    closely in practice - there's no requirement that the plot and the
    reported work numbers share one convention.

    Parameters/Returns match compute_cumulative_work_kT() except the
    return value is (W_jump_kT, W_total_kT):
        W_jump_kT  : (n_events,) - the jump's own work, kT units.
        W_total_kT : same shape as x_cam_nm - cumulative TOTAL work
                    (jump + smooth), kT units, W_total[...,0] = 0.
    """
    x_cam_nm = np.atleast_2d(x_cam_nm)
    x_m = x_cam_nm * 1e-9
    lambda_m = lambda_trap_nm * 1e-9
    kT_J = K_BOLTZMANN_J_PER_K * T_K

    lambda_jump_target_m = lambda_m[1] if len(lambda_m) > 1 else lambda_m[0]
    X_true_0_m = x_m[:, 0]                              # = x_cam(0-), exact
    jump_work_J = kappa_N_per_m * (lambda_jump_target_m**2 / 2
                                   - X_true_0_m * lambda_jump_target_m)
    W_jump_kT = jump_work_J / kT_J

    W_total_kT = np.zeros_like(x_m)
    if x_m.shape[1] > 1:
        W_total_kT[:, 1] = W_jump_kT
    if x_m.shape[1] > 2:
        d_lambda_m = np.diff(lambda_m[1:])                     # real samples only, from index 1 on
        x_mid_m = 0.5 * (x_m[:, 1:-1] + x_m[:, 2:])            # real samples only, no reconstruction
        dW_smooth_J = -kappa_N_per_m * x_mid_m * d_lambda_m
        W_total_kT[:, 2:] = W_jump_kT[:, None] + np.cumsum(dW_smooth_J, axis=1) / kT_J

    shape = np.atleast_2d(x_cam_nm).shape if x_cam_nm.ndim > 1 else (x_cam_nm.shape[-1],)
    return W_jump_kT, W_total_kT.reshape(shape)


def _grid_dims(n: int) -> tuple[int, int]:
    """Same layout rule as analytical_optimal_protocol_computation's own
    plot_optimal_trajectories_grid: as square as possible, extra panels
    (n not a perfect square) go unused rather than left as blank rows."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _style_grid_axis(ax) -> None:
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
    ax.set_xlabel("$t$ in s")


def plot_x_lambda_grid(events: dict, lambdas: dict) -> None:
    """
    One figure, one panel per state (from events' keys), grid layout as
    square as possible (matches 3c's analytical-protocol grid style:
    gridlines, minor ticks, sci-notation) - mean x(t) (+/- std band) and
    lambda(t) overlaid, nm. States with zero real events get a "no events"
    placeholder panel instead of being silently dropped from the grid.
    """
    states = list(events.keys())
    n = len(states)
    nrows, ncols = _grid_dims(n)
    fig = plt.figure(figsize=(6.5 * ncols, 4.5 * nrows))

    for i, state in enumerate(states):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_grid_axis(ax)
        t_s = events[state]["t_s"]
        x_nm = events[state]["x_nm"]
        n_events = x_nm.shape[0]
        ax.set_title(f"$m_0={state[0]}$, $m_t={state[1]}$  n={n_events}")

        if n_events == 0:
            ax.text(0.5, 0.5, "no events", ha='center', va='center', transform=ax.transAxes)
            continue

        x_mean, x_std = x_nm.mean(axis=0), x_nm.std(axis=0)
        ax.plot(t_s, x_mean, color="tab:blue", label="$\\langle x \\rangle$")
        ax.fill_between(t_s, x_mean - x_std, x_mean + x_std, color="tab:blue", alpha=0.2)
        ax.plot(t_s, lambdas[state], color="tab:green", label="$\\lambda$")
        ax.set_ylabel("Position (nm)")
        ax.legend(fontsize=8)

    fig.suptitle("Per-state protocol events: $x$ and $\\lambda$")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_trap_relative_grid(events: dict, lambdas: dict) -> None:
    """
    Same layout as plot_x_lambda_grid(), but avoids that plot's reference-
    frame artifact: raw x_cam = X_true - lambda rides along with the
    protocol's own initial jump (x_cam jumps too, by construction, not
    because the particle moved - see compute_cumulative_work_kT's
    docstring), which reads as an unphysical plunge right when lambda
    fires. This plots the trap-relative X = x_cam + lambda instead -
    lambda is added from index 2 onward only (matching
    compute_work_split_kT's own jump convention: lambda_trap_nm[2] is the
    first sample that's genuinely time-aligned with a real camera reading
    - this hardware has a confirmed extra ~1-frame delay before the trap
    actually starts moving, verified independently via the AI Stage-
    channel readback, so indices 0 AND 1 are both still pre-jump - so X is
    reconstructed post-jump from index 2 on; indices 0-1 are left as-is,
    lambda(0-)=0 there so no addition is needed) - giving a continuous
    curve showing what the particle actually did relative to the trap.

    lambda(t) is plotted as the plain curve it is (already lambda(0+) from
    its own first sample onward), with its own instantaneous initial jump
    drawn as a separate dashed marker from (t=0, 0) up to (t=0, lambda(0+))
    plus an open circle at the pre-jump rest point - same convention as
    the notebook's own _plot_lambda_jump helper - and, like that helper,
    with no legend entry of its own (it's just an annotation on the
    already-labeled lambda curve, not a separate series).
    """
    states = list(events.keys())
    n = len(states)
    nrows, ncols = _grid_dims(n)
    fig = plt.figure(figsize=(6.5 * ncols, 4.5 * nrows))

    for i, state in enumerate(states):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_grid_axis(ax)
        t_s = events[state]["t_s"]
        x_nm = events[state]["x_nm"]
        n_events = x_nm.shape[0]
        ax.set_title(f"$m_0={state[0]}$, $m_t={state[1]}$  n={n_events}")

        if n_events == 0:
            ax.text(0.5, 0.5, "no events", ha='center', va='center', transform=ax.transAxes)
            continue

        lam = lambdas[state]
        X_true_nm = x_nm.copy()
        if len(lam) > 2:
            X_true_nm[:, 2:] += lam[2:]

        X_mean, X_std = X_true_nm.mean(axis=0), X_true_nm.std(axis=0)
        ax.plot(t_s, X_mean, color="tab:blue", label="$\\langle X \\rangle$")
        ax.fill_between(t_s, X_mean - X_std, X_mean + X_std, color="tab:blue", alpha=0.2)

        ax.plot(t_s, lam, color="tab:green", label="$\\lambda$")
        if len(lam) > 0 and lam[0] != 0:
            ax.plot([t_s[0], t_s[0]], [0.0, lam[0]], color="tab:green", linestyle="--", lw=1.5)
            ax.plot(t_s[0], 0.0, marker="o", ms=5, mfc="white", mec="tab:green", zorder=5)

        ax.set_ylabel("Position (nm)")
        ax.legend(fontsize=8)

    fig.suptitle("Per-state protocol events: trap-relative $X$ and $\\lambda$")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_stage_vs_protocol_grid(stage_events: dict, stage_protocol_nm: dict) -> None:
    """
    Diagnostic grid, same layout as plot_x_lambda_grid(): per state, the
    mean relative Stage-channel readback (+/- std band, from
    extract_protocol_stage_events()/_from_folder() - what the stage
    actually did, zeroed to each event's own pre-fire baseline) overlaid
    against the currently-loaded analytical protocol's stage-frame curve
    (stage_protocol_nm[state] - pass -lambda_trap_nm_for_state(state, t_s,
    analytical_protocols, protocol_dt_s, ao_rate) for each state, or
    equivalently np.interp the raw analytical_protocols[state] AO waveform
    onto t_s - analytical_protocols is already stage-frame, see
    analytical_protocols.py). No camera data (x_nm) involved at all - a
    mismatch here means the loaded protocol doesn't match what this batch's
    hardware actually delivered, independent of any camera-tracking
    question. Purely read-only/diagnostic - doesn't touch or depend on
    compute_cumulative_work_kT or plot_x_lambda_grid.
    """
    states = list(stage_events.keys())
    n = len(states)
    nrows, ncols = _grid_dims(n)
    fig = plt.figure(figsize=(6.5 * ncols, 4.5 * nrows))

    for i, state in enumerate(states):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_grid_axis(ax)
        t_s = stage_events[state]["t_s"]
        stage_nm = stage_events[state]["stage_nm"]
        n_events = stage_nm.shape[0]
        ax.set_title(f"$m_0={state[0]}$, $m_t={state[1]}$  n={n_events}")

        if n_events == 0:
            ax.text(0.5, 0.5, "no events", ha='center', va='center', transform=ax.transAxes)
            continue

        s_mean, s_std = stage_nm.mean(axis=0), stage_nm.std(axis=0)
        ax.plot(t_s, s_mean, color="tab:purple", label="Stage (measured)")
        ax.fill_between(t_s, s_mean - s_std, s_mean + s_std, color="tab:purple", alpha=0.2)
        if state in stage_protocol_nm:
            ax.plot(t_s, stage_protocol_nm[state], color="tab:orange", ls="--",
                   label="protocol (loaded)")
        ax.set_ylabel("Relative stage position (nm)")
        ax.legend(fontsize=8)

    fig.suptitle("Measured Stage readback vs. loaded analytical protocol (stage frame)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_work_grid(events: dict, works_kT: dict, kappa_N_per_m: float, T_K: float) -> None:
    """
    One figure, one panel per state (from events' keys), same grid layout
    as plot_x_lambda_grid() - mean cumulative work(t) (+/- std band), kT.
    """
    states = list(events.keys())
    n = len(states)
    nrows, ncols = _grid_dims(n)
    fig = plt.figure(figsize=(6.5 * ncols, 4.5 * nrows))

    for i, state in enumerate(states):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_grid_axis(ax)
        t_s = events[state]["t_s"]
        W_kT = works_kT[state]
        n_events = W_kT.shape[0]
        ax.set_title(f"$m_0={state[0]}$, $m_t={state[1]}$  n={n_events}")

        if n_events == 0:
            ax.text(0.5, 0.5, "no events", ha='center', va='center', transform=ax.transAxes)
            continue

        W_mean, W_std = W_kT.mean(axis=0), W_kT.std(axis=0)
        ax.plot(t_s, W_mean, color="tab:red", label="$\\langle W \\rangle$")
        ax.fill_between(t_s, W_mean - W_std, W_mean + W_std, color="tab:red", alpha=0.2)
        ax.set_ylabel("$W$ / kT")
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-state cumulative work  (kappa={kappa_N_per_m:.3e} N/m, T={T_K:.0f} K)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def compute_trigger_energy_reference_kT(events: dict, kappa_N_per_m: float, T_K: float) -> dict:
    """
    -0.5*kappa*<x_trigger>^2 / kT per state - a naive "energy available at
    the trigger instant" reference (negative trap potential, matching the
    solver's own "-V_trap" reference key and this codebase's convention of
    negative = work extracted), independent of the analytical solver
    entirely (computed straight from events, i.e. real measured data).

    x_trigger is events[state]["x_nm"][:, 0]: the raw camera reading at
    the fire's own decide/trigger frame - x_cam(0-), recorded before that
    fire's ao.load()/ao.start() call (see compute_cumulative_work_kT's
    docstring). lambda(0-)=0 at that instant (the trap hasn't jumped yet),
    so x_cam(0-) equals the trap-relative coordinate X_true(0-) exactly -
    this is the same position (compared against xth_nm) that got this
    state decoded and fired in the first place (e.g. for (1,1), the mean
    of x_trigger across all its real fires is - by construction - some
    value > xth_nm). Averaged ACROSS events first, then squared (matches
    -0.5*kappa*mean(x)^2 as specified, not -0.5*kappa*mean(x^2)).

    Returns
    -------
    dict[state] -> float, kT units. A state with zero events gets 0.0.
    """
    kT_J = K_BOLTZMANN_J_PER_K * T_K
    reference_kT = {}
    for state, data in events.items():
        x_nm = data["x_nm"]
        if x_nm.shape[0] == 0:
            reference_kT[state] = 0.0
            continue
        x_trigger_m = x_nm[:, 0].mean() * 1e-9
        reference_kT[state] = -0.5 * kappa_N_per_m * x_trigger_m**2 / kT_J
    return reference_kT


def compute_experimental_probabilities(events: dict, fire_00_prob: float | None = None) -> dict:
    """
    Empirical P(m_0, m_t) per state - each state's real fired-event count
    (events[state]["x_nm"].shape[0]) divided by the total across every
    state in events - straight from the batch data, no solver dependency
    at all. Compare against the analytical solver's own predicted
    probabilities (analytical_protocols.load_predicted_probabilities()) to
    see whether the real decode/fire rates matched what was expected.

    (0,0) needs special handling: it's gated by fire_00_prob (an unbiased
    Bernoulli draw made independently every time (0,0) is decoded - see
    SessionParams2Ch.fire_00_prob's docstring and run_session_2ch's
    should_fire_00), so only a fire_00_prob fraction of real (0,0) decodes
    ever produce a countable fired event (extract_protocol_events() only
    extracts ZERO_FIRE_CODE frames, not every ordinary
    STATE_TO_CODE[(0,0)] decode). Every other state fires on every single
    decode (should_fire_real has no such gating), so its fired-event count
    already equals its true decode count. Left uncorrected, (0,0)'s raw
    count massively understates its true decode probability (in practice
    often the single most common decode outcome) and inflates every other
    state's apparent share.

    Pass fire_00_prob (the session's actual SessionParams2Ch.fire_00_prob,
    e.g. section 4's FIRE_00_PROB) to correct for this: (0,0)'s raw fired
    count is divided by fire_00_prob first (estimating its true decode
    count) before normalizing across all states. None (default) skips the
    correction - only appropriate if fire_00_prob was actually 1.0, or if
    events doesn't include (0,0) at all.

    Returns
    -------
    dict[state] -> float, summing to 1.0 across events (0.0 for every
    state if events itself has zero total events).
    """
    counts = {state: float(data["x_nm"].shape[0]) for state, data in events.items()}
    if fire_00_prob is not None and fire_00_prob > 0 and (0, 0) in counts:
        counts[(0, 0)] = counts[(0, 0)] / fire_00_prob
    total = sum(counts.values())
    if total == 0:
        return {state: 0.0 for state in counts}
    return {state: n / total for state, n in counts.items()}


def compute_weighted_average_kT(values_by_state: dict, weights_by_state: dict) -> float:
    """
    Generic probability-weighted average of any per-state scalar (kT
    units): sum_i(weight_i * value_i) / sum_i(weight_i). The shared
    primitive behind compute_weighted_mean_work_kT() (values = each
    state's final mean work) - also usable directly for e.g. the average
    trigger-energy reference:
        compute_weighted_average_kT(trigger_reference_kT,
                                    compute_experimental_probabilities(batch_events, fire_00_prob))
    ("the average V_ij using experimental probabilities").

    values_by_state, weights_by_state: dict[state] -> float. A state
    present in weights_by_state but missing from values_by_state is
    skipped, not zero-valued - it simply doesn't contribute to either sum.

    Returns
    -------
    float, kT units - nan if no state had both a weight and a value.
    """
    numerator, denominator = 0.0, 0.0
    for state, weight in weights_by_state.items():
        if state not in values_by_state:
            continue
        numerator += weight * values_by_state[state]
        denominator += weight
    return numerator / denominator if denominator else float("nan")


def compute_weighted_mean_work_kT(total_by_state: dict, weights_by_state: dict) -> float:
    """
    Probability-weighted mean final work across states:
        sum_i(weight_i * mean(W_total_kT_i[:, -1])) / sum_i(weight_i)
    - the same aggregate run_single_parameter_point_cli.py's own
    "weighted_mean_work" printout computes, but generalized to any weight
    source, e.g.:
        compute_weighted_mean_work_kT(batch_W_total_kT,
                                      compute_experimental_probabilities(batch_events))
        compute_weighted_mean_work_kT(batch_W_total_kT,
                                      analytical_protocols.load_predicted_probabilities(ANALYTICAL_SAVE_DIR))

    total_by_state  : dict[state] -> (n_events, protocol_frames) ndarray,
                     e.g. compute_work_split_kT()'s W_total_kT per state -
                     only the final sample ([:, -1]) is used.
    weights_by_state: dict[state] -> float - need not sum to 1 (normalized
                     internally via the denominator). A state present in
                     weights_by_state but missing from total_by_state (or
                     with zero events there) is skipped, not zero-weighted -
                     it simply doesn't contribute to either sum.

    Returns
    -------
    float, kT units - nan if no state had both a weight and real events.
    """
    final_work_by_state = {
        state: data[:, -1].mean()
        for state, data in total_by_state.items()
        if data.shape[0] > 0
    }
    return compute_weighted_average_kT(final_work_by_state, weights_by_state)


def plot_work_split_grid(events: dict, jump_by_state: dict, total_by_state: dict,
                         kappa_N_per_m: float, T_K: float,
                         predicted_work_kT: dict | None = None,
                         trigger_reference_kT: dict | None = None) -> None:
    """
    Same grid/layout as plot_work_grid(), but from compute_work_split_kT()'s
    output: the jump's own work is drawn as an explicit dashed step (0 at
    t=0 up to <W_jump> at t_s[1], mirroring the notebook's own
    _plot_lambda_jump convention for lambda itself) instead of being
    invisibly folded into the first point of the cumulative curve, so the
    jump's share of the total is visible at a glance. The solid red curve
    is still the cumulative TOTAL (jump + smooth), identical to what
    plot_work_grid(events, total_by_state, ...) would draw.

    jump_by_state[state]  : (n_events,) - compute_work_split_kT()'s W_jump_kT.
    total_by_state[state] : (n_events, protocol_frames) - its W_total_kT.

    Two optional horizontal dashed reference lines, each dict[state] ->
    float (kT units), drawn flat across the full time axis when given:
    predicted_work_kT     : the analytical solver's own predicted mean
                            work for this state (see
                            analytical_protocols.load_predicted_work_J(),
                            /kT_J to convert) - "what the optimal protocol
                            was designed to extract".
    trigger_reference_kT  : compute_trigger_energy_reference_kT()'s
                            -0.5*kappa*<x_trigger>^2/kT - "what was
                            naively available at the trigger instant".
    """
    states = list(events.keys())
    n = len(states)
    nrows, ncols = _grid_dims(n)
    fig = plt.figure(figsize=(6.5 * ncols, 4.5 * nrows))

    for i, state in enumerate(states):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_grid_axis(ax)
        t_s = events[state]["t_s"]
        W_kT = total_by_state[state]
        n_events = W_kT.shape[0]
        ax.set_title(f"$m_0={state[0]}$, $m_t={state[1]}$  n={n_events}")

        if n_events == 0:
            ax.text(0.5, 0.5, "no events", ha='center', va='center', transform=ax.transAxes)
            continue

        W_mean, W_std = W_kT.mean(axis=0), W_kT.std(axis=0)
        W_jump_mean = jump_by_state[state].mean()

        if len(t_s) > 1 and W_jump_mean != 0:
            ax.plot([t_s[0], t_s[1]], [0.0, W_jump_mean], color="tab:orange",
                   linestyle="--", lw=1.5)
            ax.plot(t_s[0], 0.0, marker="o", ms=5, mfc="white", mec="tab:orange", zorder=5)

        ax.plot(t_s, W_mean, color="tab:red", label="$\\langle W_{\\mathrm{total}} \\rangle$")
        #ax.fill_between(t_s, W_mean - W_std, W_mean + W_std, color="tab:red", alpha=0.2)

        if predicted_work_kT is not None and state in predicted_work_kT:
            ax.axhline(predicted_work_kT[state], color="tab:gray", linestyle="--", lw=2,
                      label="predicted (analytical)")
        if trigger_reference_kT is not None and state in trigger_reference_kT:
            ax.axhline(trigger_reference_kT[state], color="tab:brown", linestyle=":", lw=2,
                      label="$\\langle V_{ij}\\rangle$")

        ax.set_ylabel("$W$ / kT")
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-state work, jump vs. total  (kappa={kappa_N_per_m:.3e} N/m, T={T_K:.0f} K)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def export_for_matlab(path: str, events: dict, lambdas: dict, works_kT: dict,
                      kappa_N_per_m: float, T_K: float, protocol_dt_s: float,
                      jump_works_kT: dict | None = None,
                      predicted_work_kT: dict | None = None,
                      trigger_reference_kT: dict | None = None,
                      stage_events: dict | None = None,
                      stage_protocol_nm: dict | None = None) -> None:
    """
    Save the exact arrays plot_x_lambda_grid()/plot_work_grid() plot to one .mat file -
    per state, a struct (MATLAB field names can't hold '(', ',', '-', so
    each state gets a sanitized field name, e.g. (1,1) -> state_1_1,
    (-1,-1) -> state_neg1_neg1) with:
        t_s, x_nm (n_events x protocol_frames), lambda_nm (protocol_frames,),
        W_cum_kT (n_events x protocol_frames), n_events, plus the scalar
        metadata (kappa_N_per_m, T_K, protocol_dt_s) needed to interpret them
        - no need to also ship pos/ai_data; these are the plotted source
        arrays, already in real units (nm, kT), ready to replot directly.

    The remaining params are all optional (None = omit that field entirely,
    matching every caller before this extension existed) and carry what
    matlab/InfoEngineAnalysis.m's newer plots (plotTrapRelative/
    plotWorkSplit/plotStageVsProtocol - the MATLAB equivalents of
    plot_trap_relative_grid/plot_work_split_grid/plot_stage_vs_protocol_grid)
    need on top of the base fields above:
        jump_works_kT        : dict[state] -> (n_events,) W_jump_kT,
                               compute_work_split_kT()'s first return value
                               -> saved as W_jump_kT.
        predicted_work_kT     : dict[state] -> float (kT) -> predicted_work_kT.
        trigger_reference_kT  : dict[state] -> float (kT) -> trigger_reference_kT.
        stage_events          : dict[state] -> {"stage_nm", "t_s"} from
                               extract_protocol_stage_events_from_folder() ->
                               saved as stage_nm/stage_t_s (its own t_s,
                               kept separate from the camera-side t_s above
                               since the two aren't guaranteed to share a
                               sampling grid).
        stage_protocol_nm     : dict[state] -> (protocol_frames,) -> saved
                               as stage_protocol_nm. Only written for a
                               state when stage_events also has it.
    """
    from scipy.io import savemat

    def _field(state):
        return f"state_{state[0]}_{state[1]}".replace("-", "neg")

    mat_dict = {}
    for state in events:
        entry = {
            "m0": state[0], "mt": state[1],
            "t_s": events[state]["t_s"],
            "x_nm": events[state]["x_nm"],
            "lambda_nm": lambdas[state],
            "W_cum_kT": works_kT[state],
            "n_events": events[state]["x_nm"].shape[0],
        }
        if jump_works_kT is not None and state in jump_works_kT:
            entry["W_jump_kT"] = jump_works_kT[state]
        if predicted_work_kT is not None and state in predicted_work_kT:
            entry["predicted_work_kT"] = predicted_work_kT[state]
        if trigger_reference_kT is not None and state in trigger_reference_kT:
            entry["trigger_reference_kT"] = trigger_reference_kT[state]
        if stage_events is not None and state in stage_events:
            entry["stage_nm"] = stage_events[state]["stage_nm"]
            entry["stage_t_s"] = stage_events[state]["t_s"]
            if stage_protocol_nm is not None and state in stage_protocol_nm:
                entry["stage_protocol_nm"] = stage_protocol_nm[state]
        mat_dict[_field(state)] = entry
    mat_dict["kappa_N_per_m"] = kappa_N_per_m
    mat_dict["T_K"] = T_K
    mat_dict["protocol_dt_s"] = protocol_dt_s

    savemat(path, mat_dict)
    print(f"Saved: {path}")


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

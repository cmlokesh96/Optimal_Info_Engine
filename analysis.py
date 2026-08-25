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

    Parameters
    ----------
    pos     : the session's pos array (n_frames, 4).
    params  : that session's SessionParams2Ch (need .fps, .protocol_frames).
    states  : iterable of (m_0, m_t) tuples to extract, or None (default) to
        auto-resolve: all 8 real (non-(0,0)) states, narrowed by
        active_states/protocols_loaded if those aren't None either - i.e.
        active_states=None (the session's own "full engine, every state can
        fire" convention) means every state gets extracted here too.
    active_states, protocols_loaded : from the run_session_2ch() result dict
        (or the saved _params.pkl) - None means "no restriction" (matches
        run_session_2ch's own None-means-unrestricted convention).

    Returns
    -------
    dict[state] -> {"x_nm": (n_events, protocol_frames) ndarray,
                    "t_s": (protocol_frames,) ndarray,
                    "frame_idx": (n_events,) ndarray of fire-start frames}
        A state with zero real occurrences gets an (0, protocol_frames)
        x_nm - not an error, just nothing to average/plot for it.
    """
    from protocols_2ch import STATE_TO_CODE

    real_states = [s for s in STATE_TO_CODE if s != (0, 0)]

    if states is None:
        states = real_states
        if active_states is not None:
            states = [s for s in states if s in active_states]
        if protocols_loaded is not None:
            states = [s for s in states if s in protocols_loaded]
    else:
        # Explicit states are still validated - an explicitly-requested
        # state that couldn't possibly have fired is almost always a
        # mistake, not something to silently drop.
        for state in states:
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

    out = {}
    for state in states:
        code = STATE_TO_CODE[state]
        frame_idx = np.flatnonzero(pos[:, 3] == code)
        # Drop events truncated by the session ending before protocol_dt_s
        # had time to fully elapse - can't fill a full row for those.
        frame_idx = frame_idx[frame_idx + protocol_frames <= n_frames]

        x_nm = np.empty((len(frame_idx), protocol_frames))
        for i, fi in enumerate(frame_idx):
            x_nm[i] = pos[fi:fi + protocol_frames, 1]

        out[state] = {"x_nm": x_nm, "t_s": t_s, "frame_idx": frame_idx}

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
    Load one saved session's pos + params + active_states + protocols_loaded
    back from its _pos.npy/_params.pkl pair, given the shared prefix
    discover_session_files() returns (or build one by hand: the exact string
    save_session() used before appending "_pos.npy"/"_params.pkl").
    """
    pos = np.load(f"{prefix}_pos.npy")
    with open(f"{prefix}_params.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
        "pos": pos,
        "params": meta["params"],
        "active_states": meta["active_states"],
        "protocols_loaded": meta["protocols_loaded"],
    }


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


def _merge_events(combined: dict, new: dict) -> None:
    """Concatenate new's per-state x_nm/frame_idx onto combined, in place -
    t_s must agree across everything merged (same protocol_dt_s/fps), since
    a mismatch means these sessions aren't really comparable experiments."""
    for state, data in new.items():
        if state not in combined:
            combined[state] = {"x_nm": [data["x_nm"]], "t_s": data["t_s"],
                               "frame_idx": [data["frame_idx"]]}
        else:
            if not np.allclose(combined[state]["t_s"], data["t_s"]):
                raise ValueError(
                    f"Inconsistent protocol timing for {state} across "
                    f"sessions being combined (different protocol_dt_s/fps?) "
                    f"- these experiments aren't directly poolable."
                )
            combined[state]["x_nm"].append(data["x_nm"])
            combined[state]["frame_idx"].append(data["frame_idx"])


def _finalize_merge(combined: dict) -> dict:
    protocol_frames = None
    for state, data in combined.items():
        protocol_frames = data["x_nm"][0].shape[1] if data["x_nm"] else protocol_frames
    out = {}
    for state, data in combined.items():
        x_nm = (np.concatenate(data["x_nm"], axis=0) if data["x_nm"]
               else np.empty((0, protocol_frames or 0)))
        frame_idx = (np.concatenate(data["frame_idx"], axis=0) if data["frame_idx"]
                    else np.empty((0,), dtype=int))
        out[state] = {"x_nm": x_nm, "t_s": data["t_s"], "frame_idx": frame_idx}
    return out


def extract_protocol_events_from_folder(folder: str, states=None) -> dict:
    """
    Load every saved session in folder (one run_batch_2ch() batch's worth)
    and pool their extract_protocol_events() results into one combined
    per-state dict - x_nm rows from every experiment in the folder stacked
    together, so e.g. 8 one-hour experiments' worth of a given state's real
    fired events become one (n_total_events, protocol_frames) matrix instead
    of 8 separate ones. states=None (default) auto-resolves per-session as
    extract_protocol_events() does (each session's own active_states/
    protocols_loaded), then pools across whatever the union of sessions covers.
    """
    combined: dict = {}
    for prefix in discover_session_files(folder):
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


def export_for_matlab(path: str, events: dict, lambdas: dict, works_kT: dict,
                      kappa_N_per_m: float, T_K: float, protocol_dt_s: float) -> None:
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
    """
    from scipy.io import savemat

    def _field(state):
        return f"state_{state[0]}_{state[1]}".replace("-", "neg")

    mat_dict = {}
    for state in events:
        mat_dict[_field(state)] = {
            "m0": state[0], "mt": state[1],
            "t_s": events[state]["t_s"],
            "x_nm": events[state]["x_nm"],
            "lambda_nm": lambdas[state],
            "W_cum_kT": works_kT[state],
            "n_events": events[state]["x_nm"].shape[0],
        }
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

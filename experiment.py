"""
experiment.py
=============
The 2-channel / 9-state session stack (run_session_2ch, run_batch_2ch,
ensure_stage_at, calibrate_trap_center, preview_particle, plot_session_2ch)
plus the shared post-session utilities below:

build_datasync() — post-hoc: parse AI stream using CamOut sync signal
save_session()  — save pos, DataSync, params  (mirrors MATLAB save block)

Data structures
---------------
pos : (n_frames, 4)  float64 — see protocols_2ch.py's STATE_TO_CODE /
    CAPTURE_CODE for the trigger-column (col 3) encoding used by the
    current 9-state scheme.

DataSync : (n_frames, 3)  float64
    col 0  time_s    — seconds from first camera frame (via CamOut edges)
    col 1  stage_nm  — low-pass filtered stage position (nm)
    col 2  aom_V     — AOM voltage at each camera frame

Mirrors MATLAB:
    tvec = rising edges of CamOut > 2 V in the AI stream
    DataSync = [t  xwell  aomV]
    save(..., 'pos', 'DataSync', 'pExp')
"""

import os
import time
import pickle
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from protocols_2ch import (
    SessionParams2Ch, decode_state_pair, get_protocol_nm, split_channels,
    STATE_TO_CODE, CODE_TO_STATE, CAPTURE_CODE, ZERO_FIRE_CODE,
    V_CENTER, NM_PER_VOLT,
)
from camera import NM_PER_PX


# ── DataSync builder ───────────────────────────────────────────────────────────
def build_datasync(result: dict,
                   lowpass_hz: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse the continuous AI recording synchronised to camera frames via CamOut.

    Mirrors the MATLAB parsing block:
        a = find(ICardData.CamOut > 2);
        b = diff(a)>1; b=[1;b]; b=find(b==1); tvec = a(b);
        xwell = lowpass(ICardData.Stage, 100, 10000, ...); xwell = xwell*3000;
        t = seconds(ICardData.Time(tvec)); t = t-t(1);
        DataSync = [t  xwell  aomV];

    Returns
    -------
    DataSync : (n_frames, 3)  [time_s, stage_nm, aom_V]
    tvec     : (n_frames,)    AI sample index for each camera frame
    """
    ai_data  = result["ai_data"]
    names    = result["channel_names"]
    ai_rate  = result["params"].ai_rate

    cam_ch   = names.index("CamOut")
    stage_ch = names.index("Stage")
    aom_ch   = names.index("AOM")

    # ── Find camera trigger rising edges in CamOut (mirrors MATLAB tvec) ───────
    cam_out   = ai_data[cam_ch]
    triggered = (cam_out > 2.0).astype(np.int8)
    edges     = np.flatnonzero(np.diff(triggered) > 0) + 1   # rising edges
    tvec      = edges   # AI sample index of each camera frame

    # ── Requested vs recorded — hardware-confirmed frame count check ───────────
    n_requested = getattr(result.get("params"), "n_frames", None)
    n_recorded  = len(tvec)
    if n_requested is not None:
        if n_recorded == n_requested:
            print(f"CamOut: {n_recorded}/{n_requested} frames confirmed on the wire")
        else:
            print(f"WARNING: CamOut shows {n_recorded} triggered frames, "
                  f"{n_requested} were requested "
                  f"({abs(n_requested - n_recorded)} "
                  f"{'missing' if n_recorded < n_requested else 'extra'}) "
                  f"- check for dropped triggers/frames")

    # ── Low-pass filter Stage channel at lowpass_hz ───────────────────────────
    nyq         = ai_rate / 2
    b, a        = butter(N=8, Wn=lowpass_hz / nyq, btype="low")
    stage_filt  = filtfilt(b, a, ai_data[stage_ch])
    stage_nm    = stage_filt * NM_PER_VOLT   # V → nm

    # ── Sync to camera frames ─────────────────────────────────────────────────
    t_s       = tvec / ai_rate
    t_s       = t_s - t_s[0]          # t=0 at first camera frame

    n = min(len(tvec), len(result["pos"]))   # guard against early stop
    DataSync = np.column_stack([
        t_s[:n],
        stage_nm[tvec[:n]],
        ai_data[aom_ch][tvec[:n]],
    ])
    # DataSync columns: [time_s,  stage_nm,  aom_V]

    return DataSync, tvec


# ── Save ───────────────────────────────────────────────────────────────────────
def save_session(result: dict, DataSync: np.ndarray,
                 suffix: str = "", protocols: dict[tuple[int, int], np.ndarray] | None = None) -> None:
    """
    Save pos, DataSync, ai_data, params to files.
    Mirrors MATLAB:
        save([path '_xm_..._Data.mat'], 'pos', 'DataSync', 'pExp')

    params (protocol_dt_s, T_relax_s, xth_nm, delta_t_s, ...), active_states,
    and protocols_loaded (sorted list of states that had a real analytical
    protocol, or None) are pickled together alongside the three arrays — a
    later work/thermodynamics calculation needs params to know, per fire
    (pos[:,3] code + tvec -> AI sample index), where that protocol's window
    ends (+protocol_dt_s*ai_rate samples) and that everything from there
    until the next checkpoint is the waiting window (T_relax_s) where work
    is zero by construction (ao0=0, ao1 constant - the trap doesn't move).
    active_states/protocols_loaded resolve a real ambiguity in pos[:,3]
    alone: a decide checkpoint's code is the DECODED state whether or not
    it fired (see run_session_2ch's should_fire) - only a code for a state
    that was both in active_states (or active_states was None) AND in
    protocols_loaded (or protocols_loaded was None) is proof of an actual
    fire.

    protocols: the actual (m_0, m_t) -> stage-frame waveform dict this
    session fired with (e.g. analytical_protocols, or whatever was passed
    to run_session_2ch()/run_batch_2ch()'s own protocols= arg) - when
    given, pickled alongside params to {base}_protocols.pkl. This is what
    later makes a saved session/batch self-contained: analysis can load
    the EXACT protocol shapes this data was recorded with, independent of
    whatever currently happens to be loaded in a notebook or sitting in
    analytical_optimal_protocol_computation's output folder (which is NOT
    append-only - reruns of the solver silently overwrite same-(t2, tf,
    m_0, m_t) files). None (default) skips saving it - matches the old
    behavior for callers that don't have/need this.

    For .mat compatibility use scipy.io.savemat instead of np.save.
    """
    import datetime
    params  = result["params"]
    ts      = datetime.datetime.now().strftime("%d-%b-%Y-%H.%M.%S")
    base    = f"{params.savepath}{suffix}_{ts}"

    np.save(f"{base}_pos.npy",      result["pos"])
    np.save(f"{base}_DataSync.npy", DataSync)
    np.save(f"{base}_ai_data.npy",  result["ai_data"])
    with open(f"{base}_params.pkl", "wb") as f:
        pickle.dump({
            "params"          : params,
            "active_states"   : result.get("active_states"),
            "protocols_loaded": result.get("protocols_loaded"),
        }, f)
    saved_files = f"{base}_*.npy, {base}_params.pkl"
    if protocols is not None:
        with open(f"{base}_protocols.pkl", "wb") as f:
            pickle.dump(protocols, f)
        saved_files += f", {base}_protocols.pkl"
    print(f"Saved: {saved_files}")


# ══════════════════════════════════════════════════════════════════════════
# 2-channel / 9-state session stack — the real particle, real hardware
#
# Two-timestep, two-threshold 9-state decode (protocols_2ch.py) driving
# ao0 (fine) + ao1 (coarse) through the adder circuit. Calls the real
# BaslerCamera method names (hardwareTrigg() / grab() — raises RuntimeError
# on timeout / find()), not the old MockBaslerCamera-only names a much
# earlier single-channel version of this file used.
# ══════════════════════════════════════════════════════════════════════════

class _LivePreview:
    """
    Background cv2 display thread fed by frames from an EXTERNAL grab loop
    (calibrate_trap_center()'s or run_session_2ch()'s own camera.grab()
    calls) — NOT an independent grab path. A Basler/pypylon camera only
    allows one active grabbing context at a time, so a live view during a
    real acquisition can't use camera.startview()'s own internal grab loop
    concurrently with that acquisition's grab() loop — they'd fight over
    frames. Instead the grab loop calls update() with each frame it already
    retrieved; this just displays whatever's most recent, on its own timer,
    independent of the acquisition rate (so it works the same whether the
    caller is grabbing at 100 fps or 5 fps).
    """

    def __init__(self, window_name: str = "Live"):
        self._window_name = window_name
        self._latest: list = [None, None, None]   # frame, x_nm, y_nm
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, frame, x_nm=None, y_nm=None) -> None:
        with self._lock:
            self._latest[0], self._latest[1], self._latest[2] = frame, x_nm, y_nm

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame, x_nm, y_nm = self._latest
            if frame is not None:
                disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                if disp.ndim == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                if x_nm is not None and not np.isnan(x_nm):
                    cv2.putText(disp, f"x={x_nm:.1f} nm  y={y_nm:.1f} nm",
                               (8, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(self._window_name, disp)
                cv2.waitKey(1)
            time.sleep(1 / 30)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        cv2.destroyWindow(self._window_name)


# ── Independent utilities — each callable on its own, not tied together ────────
def preview_particle(camera, fps: float = 100.0, find_method: str = "fast",
                     find_smooth_sigma: float = 3) -> None:
    """
    Live preview at `fps` (default well above a real session's decision
    cadence) so you can confirm the particle is actually there before/after
    a run. Self-paced — blocks on Enter, not a fixed timeout. Uses camera.py's
    existing free-run preview path (softwareTrigg + startview), independent
    of the hardware-triggered acquisition a real session uses, so it never
    touches — and can't slow down — the session's hot path.

    find_method/find_smooth_sigma: which tracking method draws the cross
    marker, and (for "gradient") how much it pre-blurs the frame — "fast"
    (default) or "gradient", same defaults as camera.find()/startview() so
    what you see here matches what a real session using the same settings
    would actually track. Purely a visual check here; doesn't affect what
    any later camera.find() call in this session uses — pass
    find_method="gradient"/find_smooth_sigma=... to
    calibrate_trap_center()/run_session_2ch()/run_batch_2ch() separately
    for that.
    """
    camera.framerate(fps)
    camera.softwareTrigg()
    camera.startview(particle_cross=True, method=find_method, smooth_sigma=find_smooth_sigma)
    input("Particle visible? Press Enter to continue...")
    camera.stopview()


def calibrate_trap_center(camera, duration_s: float = 30.0,
                          fps: float = 100.0,
                          nm_per_px: float = NM_PER_PX,
                          live_preview: bool = True,
                          savepath: str = "",
                          find_method: str = "fast",
                          find_smooth_sigma: float = 3) -> tuple[float, float, np.ndarray]:
    """
    Independent trap-center measurement — run this once (or whenever you
    want to recheck), then pass its result into
    SessionParams2Ch(x_center_nm=..., y_center_nm=...) for subsequent real
    sessions. Not run automatically inside run_session_2ch.

    Free-runs the camera (software-triggered, like preview_particle) for
    duration_s, averages valid (non-NaN) camera.find() readings — raw
    positions, not centered on anything (see camera.py::find()) — and
    returns that mean as the trap's raw position. Subtracting this from
    every later camera.find() reading (as run_session_2ch already does)
    is what actually centers everything on the trap.

    Also records the raw per-frame position in PIXELS (not just the nm
    average) as pos_px — columns [x_px, y_px, time_s], one row per frame,
    NaN where a frame was dropped or the particle wasn't found — so you can
    directly check tracking quality (dropout rate, jumps, noise) before
    trusting any nm_per_px scale factor; multiply pos_px[:, :2] by your own
    nm_per_px later if you want it in nm instead. x_px/y_px here are
    recovered from camera.find()'s nm output via the same nm_per_px used to
    call it (default: camera.py's own NM_PER_PX), not a second camera call.
    Pass savepath to also write it to "<savepath>_calib_pos_px.npy".

    find_method/find_smooth_sigma are passed straight through to
    camera.find() ("fast" or "gradient" — see camera.py::find()); default
    is unchanged ("fast") so existing calls behave exactly as before.

    live_preview=True (default) shows the live feed the whole time via
    _LivePreview — fed from this function's own grab() loop (see
    _LivePreview's docstring for why it can't just be camera.startview()).

    Returns (x_center_nm, y_center_nm, pos_px).
    """
    camera.framerate(fps)
    camera.softwareTrigg()
    camera.start()

    preview = _LivePreview("Calibrating trap center") if live_preview else None
    if preview:
        preview.start()

    n_frames = max(1, int(duration_s * fps))
    xs, ys = [], []
    pos_px = np.full((n_frames, 3), np.nan)
    timeout_ms = int(2000 / fps) + 2000
    t0 = time.perf_counter()

    try:
        for i in range(n_frames):
            t_now = time.perf_counter() - t0
            try:
                frame = camera.grab(timeout_ms=timeout_ms)
            except RuntimeError:
                pos_px[i, 2] = t_now
                continue
            x_nm, y_nm = camera.find(frame, nm_per_px=nm_per_px,
                                      method=find_method, smooth_sigma=find_smooth_sigma)
            if preview:
                preview.update(frame, x_nm, y_nm)

            pos_px[i] = [x_nm / nm_per_px, y_nm / nm_per_px, t_now]

            if not (np.isnan(x_nm) or np.isnan(y_nm)):
                xs.append(x_nm)
                ys.append(y_nm)
    finally:
        if preview:
            preview.stop()
        camera.stop()

    if not xs:
        raise RuntimeError("calibrate_trap_center: particle never found — check focus/ROI")

    x_center_nm, y_center_nm = float(np.mean(xs)), float(np.mean(ys))
    print(f"Trap center: x={x_center_nm:+.1f} nm  y={y_center_nm:+.1f} nm  "
          f"({len(xs)}/{n_frames} frames had a valid reading)")

    if savepath:
        np.save(f"{savepath}_calib_pos_px.npy", pos_px)
        print(f"Saved: {savepath}_calib_pos_px.npy")

    return x_center_nm, y_center_nm, pos_px


def ensure_stage_at(ao, target_v: float = V_CENTER, speed_nm_per_s: float = 10.0,
                    settle_s: float = 300.0, tol_v: float = 0.05,
                    camera=None, live_preview: bool = False,
                    preview_fps: float = 100.0,
                    aom_gen=None, aom_transit_v: float | None = None,
                    aom_experiment_v: float | None = None,
                    aom_post_ramp_wait_s: float = 10.0) -> float:
    """
    Independent safety check. Reads the current stage voltage; if already
    within tol_v of target_v, does nothing (no needless wait, no AOM
    change either — nothing is moving, so nothing needs protecting).
    Otherwise ramps there slowly via ao.ramp_ao1() (never a bare voltage
    step — "we lose the particle" if you jump it) and waits settle_s
    afterward, with a printed countdown.

    If camera is given and live_preview=True, shows a live view for the
    whole ramp+settle so you're not just staring at a blank screen for up
    to settle_s seconds. Unlike calibrate_trap_center()/run_session_2ch(),
    this function does no grabbing of its own, so — uniquely here — it's
    safe to just use camera.startview()'s own internal grab loop (no
    external loop to conflict with).

    AOM safety (all opt-in — inert unless aom_gen is given): a stronger
    trap during the physical stage move makes it less likely the particle
    is lost while the stage is in motion. If aom_gen is given:
      1. Before ramping — if aom_transit_v is given, set the AOM to it
         (near-max/strong trap for the transit).
      2. Ramp, same as always.
      3. After arriving — if aom_experiment_v is given, wait
         aom_post_ramp_wait_s (let the stage mechanically settle a beat),
         then set the AOM to aom_experiment_v (whatever's right for the
         actual measurement) — BEFORE the settle_s countdown starts, so
         the settle period happens at the experiment voltage, not the
         transit voltage.

    Returns the stage voltage actually settled at (== target_v).
    """
    from daq import read_stage_voltage   # local import: daq.py has no protocols_2ch dependency
    current_v = read_stage_voltage()

    if abs(current_v - target_v) <= tol_v:
        print(f"Stage already at {current_v:.3f} V (target {target_v:.3f} V) - no move needed.")
        return current_v

    if aom_gen is not None and aom_transit_v is not None:
        print(f"Setting AOM to {aom_transit_v:.3f} V for the transit ...")
        aom_gen.set_dc_level(aom_transit_v)

    if camera is not None and live_preview:
        camera.framerate(preview_fps)
        camera.softwareTrigg()
        camera.startview(particle_cross=True)

    try:
        print(f"Stage at {current_v:.3f} V, ramping to {target_v:.3f} V "
              f"at {speed_nm_per_s} nm/s ...")
        ao.ramp_ao1(current_v, target_v, speed_nm_per_s)
        print(f"Arrived at {target_v:.3f} V.")

        if aom_gen is not None and aom_experiment_v is not None:
            print(f"Waiting {aom_post_ramp_wait_s:.0f}s, then setting AOM to "
                  f"{aom_experiment_v:.3f} V ...")
            time.sleep(aom_post_ramp_wait_s)
            aom_gen.set_dc_level(aom_experiment_v)

        print(f"Settling for {settle_s:.0f}s ...")
        remaining = settle_s
        step = 30.0
        while remaining > 0:
            wait = min(step, remaining)
            time.sleep(wait)
            remaining -= wait
            print(f"  ... {remaining:.0f}s remaining")

        print("Settle complete.")
    finally:
        if camera is not None and live_preview:
            camera.stopview()

    return target_v


def run_session_2ch(params: SessionParams2Ch, ao, ai, camera, funcgen,
                    ao1_start_v: float = V_CENTER,
                    active_states: set[tuple[int, int]] | None = None,
                    live_preview: bool = False,
                    protocols: dict[tuple[int, int], np.ndarray] | None = None,
                    find_method: str = "fast",
                    find_smooth_sigma: float = 3) -> dict:
    """
    Camera dry-run session for the 9-state / two-channel scheme. The stage
    is not physically connected yet — x(t) is a freely-diffusing particle,
    not under feedback — but the real camera, real 2-channel AO (ao), and
    real AI (including ai3/AdderOut) are used, so this validates the whole
    software chain end to end before the piezo is wired in.

    Decision-timing model:
        Checkpoints alternate between "capture" (record x_prev only, no
        decision) and "decide" (decode_state_pair(x_prev, x_curr) -> real
        decode, always). (0,0) -> no protocol fired *unless the rare
            fire_00_prob draw hits* (see below); ordinarily, next checkpoint
            is a fresh capture delta_frames later (skips the intermediate
            sample entirely).
        any other state -> protocol fired *only if active_states is None or
            the state is in active_states* (see below); next checkpoint is
            a capture protocol_frames + relax_frames later (the protocol's
            own duration, protocol_dt_s, THEN the T_relax_s equilibration
            wait — the two are sequential/additive, not overlapping), then
            decide delta_frames after that (protocol_dt_s + T_relax_s
            replaces delta_t for the first gap post-firing). If not in
            active_states, treated exactly like (0,0): no fire, next
            checkpoint is a capture delta_frames later.

    params.fire_00_prob (default 0.0): independent of active_states, each
        time (0,0) is decoded it fires (its own flat, zero-displacement
        protocol - a null/control measurement) with this probability, via
        an unbiased Bernoulli draw (not "every Nth occurrence" - avoids any
        alias with periodicity in when (0,0) happens to occur). When it
        fires, pos[:,3] gets ZERO_FIRE_CODE (11), not STATE_TO_CODE[(0,0)]
        (1, an ordinary non-fired (0,0)) - and the SAME post-fire
        protocol_frames + relax_frames cadence as any other fire applies,
        so a full protocol_dt_s-long window of x(t) gets recorded for it.

    active_states: decode_state_pair() ALWAYS runs on the real tracked
        particle — this never overrides/forces a state. It only restricts
        which decoded outcomes are allowed to actually fire a protocol:
            None (default)   -> every non-(0,0) state fires (full engine).
            {(1,1)}          -> fires only when that exact state occurs.
            set()            -> nothing ever fires (pure equilibrium), but
                                 decoding still happens every checkpoint.
        Lets you isolate one (or a few) protocols firing only when that
        state genuinely occurs in the real particle's exploration, so later
        thermodynamic-quantities statistics stay correct (unlike forcing a
        protocol regardless of the particle's actual state, which biases
        them).

    params.delta_t_s == 0.0: single-measurement mode - the capture step is
        skipped entirely, and every checkpoint decides immediately using
        that same frame's x for both "prev" and "curr" (decode_state_pair(
        x, x, xth_nm), always exactly diagonal - only (1,1)/(-1,-1)/(0,0)
        are ever produced). Matches the analytical protocols'
        t_second_measurement=0 case (see analytical_protocols.py) - a real
        two-sample pair doesn't exist when there's zero time between the
        "two" measurements, so this measures the SAME instant twice rather
        than approximating simultaneity with two adjacent camera frames.

    pos[:,3] is written on every checkpoint (decide -> STATE_TO_CODE[state]
    1-9 — always the real decoded state, whether or not it fired, EXCEPT a
    fired (0,0), which gets ZERO_FIRE_CODE (11) instead of code 1 — capture
    -> CAPTURE_CODE), 0 elsewhere, so it can be cross-checked directly
    against the ai3 (AdderOut) recording.

    ao1 is clamped to [params.stage_min_v, params.stage_max_v] during live
    firing (a tighter safety window than the hardware limits) — a warning
    prints if a protocol actually pushes against that boundary.

    Printing is governed independently per category, each throttled by
    params.print_every (every Nth checkpoint WITHIN that category, counted
    separately for fires vs. no-fires): params.print_no_fire (default
    False — never print (0,0)/excluded-state checkpoints) and
    params.print_fire (default True — print fires, throttled same as
    no-fires). The stage-safety-bound WARNING is never throttled/suppressed
    regardless of print_fire. Printed state text is the plain (m_0, m_t)
    tuple, not protocols_2ch.STATE_LABELS — that dict names the PLACEHOLDER
    BASE_PROTOCOLS shapes ("mirror", "-ramp", ...), which is misleading once
    real analytical protocols are loaded (see analytical_protocols.py).
    End-of-session summary reports protocol-fire latency (decision ->
    ao.start()) alongside the usual counts.

    live_preview=True shows a live view fed from this loop's own grab()
    calls (see _LivePreview) — works the same whether params.fps is 100 or
    5, and adds no extra camera calls, so it doesn't slow the hot path.

    Returns the same dict shape as run_session() (pos, ai_data,
    channel_names, params) so build_datasync()/save_session() are reused
    as-is; use plot_session_2ch() for the verification plot.

    protocols: real (non-placeholder) waveforms, as returned by
        analytical_protocols.load_analytical_protocols() — same
        (state -> relative-displacement-in-nm array) shape get_protocol_nm()
        returns, so it's a drop-in replacement. None (default): every fire
        uses get_protocol_nm()/BASE_PROTOCOLS (the placeholder shapes), same
        as before this parameter existed. Given: a fire only happens for a
        state present in this dict (decode_state_pair() still runs and logs
        every checkpoint regardless — only firing is restricted, same
        principle as active_states) - a real, non-(0,0) state missing from
        protocols (e.g. a "negligible probability" state at that analytical
        parameter point) is treated exactly like a state excluded by
        active_states: no fire, logged, not an error. Every array's length
        is checked against protocol_dt_s*ao_rate up front, not per-fire -
        a silent length mismatch would otherwise let a FINITE AO task
        truncate the waveform without any error (see hw_two_channel_test.py's
        notes on that exact failure mode).

    find_method/find_smooth_sigma: passed straight through to every
        camera.find() call in the decision loop ("fast" or "gradient" —
        see camera.py::find()); default is unchanged ("fast").
    """
    if protocols is not None:
        expected_len = int(params.protocol_dt_s * params.ao_rate)
        bad_lengths = {s: len(w) for s, w in protocols.items() if len(w) != expected_len}
        if bad_lengths:
            raise ValueError(
                f"protocols dict has waveforms with the wrong length for "
                f"this session (expected {expected_len} = protocol_dt_s * "
                f"ao_rate = {params.protocol_dt_s} * {params.ao_rate}): "
                f"{bad_lengths}"
            )

    n_frames    = params.n_frames
    frame_dt_ms = int(2000 / params.fps) + 2000

    print(f"Session (2ch dry run): {params.trecord_s/60:.1f} min  "
          f"({n_frames} frames at {params.fps} fps)")
    print(f"delta_t={params.delta_t_s}s ({params.delta_frames} frames)  "
          f"protocol_dt={params.protocol_dt_s}s ({params.protocol_frames} frames)  "
          f"T_relax={params.T_relax_s}s ({params.relax_frames} frames)  "
          f"-> post-fire gap = protocol_dt+T_relax = "
          f"{params.protocol_dt_s + params.T_relax_s}s  "
          f"xth={params.xth_nm} nm\n")

    # ── Configure all hardware once ────────────────────────────────────────────
    funcgen.setup_camera_trigger(params.fps, params.exposure_us, n_frames)
    camera.hardwareTrigg()

    # ── Start continuous AI recording (background thread) ─────────────────────
    ai.start_continuous()
    time.sleep(0.1)

    # ── Start camera, then fire one hardware trigger to start funcgen burst ────
    camera.start()
    ao.hardware_trigger()   # -> funcgen fires -> camera receives fps pulses

    # ── Pre-allocate pos array ─────────────────────────────────────────────────
    pos = np.zeros((n_frames, 4), dtype=np.float64)
    # col: [time_s, x_nm, y_nm, trigger]

    # delta_t_s == 0 -> single-measurement mode: no real "two samples apart"
    # pair exists (t_second_measurement=0 in the analytical protocols is
    # exactly this - the same instant measured "twice"), so every
    # checkpoint decides immediately using that frame's own x for BOTH
    # x_prev and x_curr (true simultaneity - not just two adjacent ~10ms
    # camera frames, which would only approximate it) - the capture step is
    # skipped entirely (is_capture stays False the whole session).
    single_measurement = (params.delta_t_s == 0)

    ao1_offset : float       = ao1_start_v
    x_prev     : float | None = None
    is_capture : bool        = not single_measurement   # first checkpoint just captures x_prev (skipped in single-measurement mode)
    next_frame : int         = 0
    frame_i    : int         = 0

    # Liveness tracking — purely observational (see preview_particle() for
    # the before/after visual check); no auto-pause/abort here, so this
    # never interferes with or slows down the decision loop.
    particle_found_prev : bool = True
    n_lost_frames        : int = 0   # frame grabbed, but no particle found in it
    n_dropped_frames      : int = 0   # camera.grab() itself failed/timed out

    no_fire_count   : int         = 0    # print_every throttle, no-fire decisions only
    fire_count      : int         = 0    # print_every throttle, fire decisions only
    fire_latencies_s: list[float] = []   # decision -> ao.start(), per fire
    rng = np.random.default_rng()        # for the fire_00_prob Bernoulli draw

    # Per-phase timing — purely diagnostic (see the end-of-session summary):
    # answers "is grab/find/fire actually keeping up with 1/fps?" with real
    # numbers instead of guessing.
    grab_latencies_s: list[float] = []
    find_latencies_s: list[float] = []

    preview = _LivePreview("run_session_2ch") if live_preview else None
    if preview:
        preview.start()

    print("Session running - Ctrl+C to stop early\n")
    t0 = time.perf_counter()

    try:
        for frame_i in range(n_frames):

            t_grab0 = time.perf_counter()
            try:
                frame = camera.grab(timeout_ms=frame_dt_ms)
            except RuntimeError:
                frame = None
            grab_latencies_s.append(time.perf_counter() - t_grab0)
            t_now = time.perf_counter() - t0

            if frame is None:
                # No image at all this cycle (dropped/grab failure) — distinct
                # from "got an image, particle not in it" (n_lost_frames
                # below). NaN, not 0.0, so a dropped frame can't silently
                # masquerade as a real position=(0,0) reading downstream
                # (e.g. in compute_potential's histogram).
                pos[frame_i] = [t_now, np.nan, np.nan, 0]
                n_dropped_frames += 1
                continue

            t_find0 = time.perf_counter()
            x_nm, y_nm = camera.find(frame, method=find_method, smooth_sigma=find_smooth_sigma)
            find_latencies_s.append(time.perf_counter() - t_find0)
            x_nm -= params.x_center_nm
            y_nm -= params.y_center_nm
            pos[frame_i] = [t_now, x_nm, y_nm, 0]

            if preview:
                preview.update(frame, x_nm, y_nm)

            found = not np.isnan(x_nm)
            if found != particle_found_prev:
                print(f"  [t={t_now:7.2f}s] "
                      f"{'particle found again' if found else 'WARNING: particle lost (not found)'}")
                particle_found_prev = found
            if not found:
                n_lost_frames += 1

            if frame_i < next_frame:
                continue

            if is_capture:
                x_prev = x_nm
                pos[frame_i, 3] = CAPTURE_CODE
                next_frame = frame_i + params.delta_frames
                is_capture = False
                continue

            # ── Decide checkpoint (real decode, always) ─────────────────────────
            x_curr = x_nm
            if single_measurement:
                x_prev = x_nm   # same instant, both "sides" of the pair - see single_measurement's definition above
            state  = decode_state_pair(x_prev, x_curr, params.xth_nm)
            pos[frame_i, 3] = STATE_TO_CODE[state]

            should_fire_real = (state != (0, 0)) and (
                active_states is None or state in active_states
            ) and (
                protocols is None or state in protocols
            )
            # Independent of active_states/protocols - a separate,
            # probability-gated path so (0,0) never needs to be added to
            # active_states to become eligible. An unbiased Bernoulli draw
            # each time (0,0) is decoded, not "every Nth occurrence" (see
            # SessionParams2Ch.fire_00_prob's docstring for why).
            should_fire_00 = (state == (0, 0)) and (params.fire_00_prob > 0) and (
                rng.random() < params.fire_00_prob
            )
            should_fire = should_fire_real or should_fire_00
            if should_fire_00:
                pos[frame_i, 3] = ZERO_FIRE_CODE

            if not should_fire:
                no_fire_count += 1
                if params.print_no_fire and (no_fire_count % params.print_every == 0):
                    if state == (0, 0):
                        why = ", fire_00 draw missed" if params.fire_00_prob > 0 else ""
                    elif active_states is not None and state not in active_states:
                        why = ", not in active_states"
                    elif protocols is not None and state not in protocols:
                        why = ", no analytical protocol loaded for this state"
                    else:
                        why = ""
                    print(f"  t={t_now:7.2f}s  frame={frame_i:7d}  "
                          f"state={str(state):10s}  "
                          f"x_prev={x_prev:+7.1f}  x_curr={x_curr:+7.1f}  "
                          f"(no fire{why})")
                next_frame = frame_i + params.delta_frames
                is_capture = not single_measurement   # single-measurement mode: next checkpoint decides immediately too
            else:
                t_fire0 = time.perf_counter()

                waveform_nm = (protocols[state] if protocols is not None else
                              get_protocol_nm(state, params.protocol_dt_s, params.ao_rate))
                ao0_wave, ao1_wave, ao1_offset = split_channels(
                    waveform_nm, ao1_offset,
                    ao1_min=params.stage_min_v, ao1_max=params.stage_max_v,
                )

                # Fire AO — non-blocking so camera grab loop continues
                ao.load(ao0_wave, ao1_wave)
                ao.start()

                fire_latencies_s.append(time.perf_counter() - t_fire0)

                at_bound = (abs(ao1_offset - params.stage_min_v) < 1e-6 or
                           abs(ao1_offset - params.stage_max_v) < 1e-6)
                if at_bound:
                    print(f"  WARNING: ao1 hit the stage safety bound "
                          f"[{params.stage_min_v:.3f}, {params.stage_max_v:.3f}] V "
                          f"- clamped to {ao1_offset:.4f} V")

                fire_count += 1
                if params.print_fire and (fire_count % params.print_every == 0):
                    print(f"  t={t_now:7.2f}s  frame={frame_i:7d}  "
                          f"state={str(state):10s}  "
                          f"x_prev={x_prev:+7.1f}  x_curr={x_curr:+7.1f}  "
                          f"-> ao1 offset {ao1_offset:.4f} V")

                # protocol_frames (the fire itself, protocol_dt_s) THEN
                # relax_frames (the T_relax_s equilibration wait) - additive,
                # not overlapping (see SessionParams2Ch.T_relax_s's docstring).
                next_frame = frame_i + params.protocol_frames + params.relax_frames
                is_capture = not single_measurement   # single-measurement mode: next checkpoint decides immediately too

    except KeyboardInterrupt:
        print("\nStopped early.")
        pos = pos[:frame_i]   # trim unused rows
    finally:
        if preview:
            preview.stop()

    # ── Stop hardware ──────────────────────────────────────────────────────────
    camera.stop()
    ao.wait_done(extra_s=10.0)

    print("\nStopping AI recording ...")
    ai_data = ai.stop_and_get()   # (n_channels, n_ai_samples)

    n_decisions = int(np.sum((pos[:, 3] >= 1) & (pos[:, 3] <= 9)))
    lost_frac    = n_lost_frames / len(pos) if len(pos) else 0.0
    dropped_frac = n_dropped_frames / len(pos) if len(pos) else 0.0
    print(f"Done: {n_decisions} decisions  "
          f"| AI samples: {ai_data.shape[1]}  "
          f"({ai_data.shape[1]/params.ai_rate:.1f} s)")
    print(f"     dropped frames (no image): {n_dropped_frames} ({dropped_frac:.1%})  "
          f"| particle not found (image, no particle): {n_lost_frames} ({lost_frac:.1%})")

    frame_budget_ms = 1000.0 / params.fps

    def _stats_ms(samples_s: list[float]) -> str:
        ms = np.array(samples_s) * 1000
        return (f"n={len(ms)}  mean={ms.mean():.2f}ms  median={np.median(ms):.2f}ms  "
               f"p95={np.percentile(ms, 95):.2f}ms  max={ms.max():.2f}ms")

    def _report_budgeted(name: str, samples_s: list[float]) -> None:
        # For work that ISN'T hardware-clock-limited (find/decide/fire): if
        # this exceeds 1/fps, it's eating into the next frame's time slot.
        if not samples_s:
            print(f"     {name}: 0 samples")
            return
        ms = np.array(samples_s) * 1000
        over = int(np.sum(ms > frame_budget_ms))
        print(f"     {name}: {_stats_ms(samples_s)}  | {over} ({over/len(ms):.1%}) "
              f"exceeded the {frame_budget_ms:.1f}ms frame budget (1/fps)")

    print(f"   Per-frame timing (1/fps budget = {frame_budget_ms:.1f}ms):")
    if grab_latencies_s:
        # grab() blocks on the hardware trigger clock, so ~frame_budget_ms
        # is NORMAL (perfectly real-time), not a problem. A grab() time much
        # SHORTER than that is the actual warning sign — it means frames
        # were already buffered up waiting (the loop fell behind and is now
        # draining a backlog), not that grab() itself is slow.
        print(f"     grab() [~{frame_budget_ms:.1f}ms = real-time; much less = "
              f"draining a backlog]: {_stats_ms(grab_latencies_s)}")
    else:
        print("     grab(): 0 samples")
    _report_budgeted("find()", find_latencies_s)
    _report_budgeted("fire (decision->ao.start())", fire_latencies_s)
    print()

    return {
        "pos"             : pos,
        "ai_data"         : ai_data,
        "channel_names"   : ai.channel_names,
        "params"          : params,
        "ao1_final_v"     : ao1_offset,
        # Needed to tell "decoded state X" apart from "decoded state X AND it
        # actually fired" from pos[:,3] alone in post-hoc analysis - a code
        # is only proof of a fire if the state was also allowed to fire here
        # (see should_fire's definition above).
        "active_states"   : active_states,
        "protocols_loaded": sorted(protocols) if protocols is not None else None,
    }


# ── N consecutive experiments ───────────────────────────────────────────────────
def run_batch_2ch(base_kwargs: dict, experiment_overrides: list[dict],
                  ao, ai, camera, funcgen, ao1_start_v: float = V_CENTER,
                  settle_s_between: float | None = None,
                  reset_target_v: float = V_CENTER,
                  reset_speed_nm_per_s: float = 10.0,
                  aom_gen=None, aom_transit_v: float | None = None,
                  aom_experiment_v: float | None = None,
                  aom_post_ramp_wait_s: float = 10.0,
                  protocols: dict[tuple[int, int], np.ndarray] | None = None,
                  find_method: str = "fast",
                  find_smooth_sigma: float = 3) -> list[dict]:
    """
    Run N experiments back-to-back, camera/DAQ staying open the whole time
    (never closed between experiments) — and this is a genuinely blocking,
    sequential loop: each run_session_2ch() call runs to completion before
    the next experiment's SessionParams2Ch is even built, so experiment
    ii+1 never starts until experiment ii is fully done (frame loop
    finished, AI stopped, saved).

    Otherwise lean by design — no preview, no camera/DAQ open/close: those
    stay the caller's explicit calls (preview_particle, ao.ramp_ao1,
    .close()) around this, in whatever order it wants.

    base_kwargs         : SessionParams2Ch fields shared by every experiment
                          (fps, exposure_us, x_center_nm, y_center_nm,
                          protocol_dt_s, T_relax_s, ai_rate, ao_rate, savepath).
                          savepath here is the BATCH FOLDER (created via
                          os.makedirs if it doesn't exist, e.g.
                          r"D:/data/batch01") - not a bare file prefix like
                          run_session_2ch/SessionParams2Ch.savepath normally
                          is elsewhere. Every experiment's own SessionParams2Ch
                          gets savepath=os.path.join(batch_folder, f"exp{ii}")
                          instead, so all N experiments' _pos.npy/_DataSync.npy/
                          _ai_data.npy/_params.pkl files land together inside
                          one folder - not as same-directory siblings sharing
                          only a filename prefix (which is what happened
                          before this - a real gap: nothing grouped a batch's
                          files together for later combined analysis). See
                          analysis.py's extract_protocol_events_from_folder()/
                          _from_folders() for pooling statistics across every
                          experiment in one (or several) such folders.
    experiment_overrides : list of N dicts, each overriding just what varies
                          per experiment (xth_nm, delta_t_s, trecord_s, ...) —
                          mirrors MATLAB pExp.xm=[...]/pExp.tau=[...] without
                          adopting their exact shape.
    ao1_start_v          : ao1 offset the *first* experiment starts from.

    settle_s_between: None (default) — no reset; each later experiment just
        continues from the previous experiment's final ao1, straight away.
        A number — between EVERY pair of experiments, calls
        ensure_stage_at(ao, reset_target_v, reset_speed_nm_per_s,
        settle_s_between) (reads the real stage voltage via
        read_stage_voltage() and only ramps if it isn't already there, same
        as any other ensure_stage_at() call) so each experiment after the
        first starts from the same settled reference position — e.g. "back
        at 5V and re-equilibrated" between every one of the 4 experiments.
        Still opt-in, not a hidden default, consistent with everything else
        in this module.

    aom_gen, aom_transit_v, aom_experiment_v, aom_post_ramp_wait_s: passed
        straight through to that same internal ensure_stage_at() call —
        same AOM safety sequence (strong trap during the transit, wait,
        then experiment voltage before the settle countdown) applied to
        the between-experiment resets, since those are real stage
        transits too. All default to inert (no AOM change) unless
        aom_gen is given, same as ensure_stage_at() itself. Nothing else
        about this function's behavior changes.

    protocols: passed straight through to every run_session_2ch() call —
        see its docstring. None (default): every experiment fires the
        placeholder get_protocol_nm() shapes, same as before this
        parameter existed.

    find_method/find_smooth_sigma: passed straight through to every
        run_session_2ch() call ("fast" or "gradient" — see
        camera.py::find()); default is unchanged ("fast").

    Returns a list of the N run_session_2ch() result dicts, in order.
    """
    n = len(experiment_overrides)
    results = []
    ao1_v = ao1_start_v

    batch_folder = base_kwargs.get("savepath", "")
    if batch_folder:
        os.makedirs(batch_folder, exist_ok=True)

    for ii, overrides in enumerate(experiment_overrides, start=1):
        print(f"\n{'='*60}")
        print(f" Experiment {ii}/{n}")
        print(f"{'='*60}")

        exp_kwargs = dict(base_kwargs)
        if batch_folder:
            exp_kwargs["savepath"] = os.path.join(batch_folder, f"exp{ii}")
        params = SessionParams2Ch(**exp_kwargs, **overrides)

        result = run_session_2ch(params, ao, ai, camera, funcgen, ao1_start_v=ao1_v,
                                 protocols=protocols,
                                 find_method=find_method, find_smooth_sigma=find_smooth_sigma)
        ao1_v = result["ao1_final_v"]

        DataSync, tvec = build_datasync(result)
        save_session(result, DataSync, protocols=protocols)   # exp{ii} is now baked into savepath itself

        n_decisions = int(np.sum((result["pos"][:, 3] >= 1) & (result["pos"][:, 3] <= 9)))
        print(f" Experiment {ii}/{n} done: {n_decisions} decisions  "
              f"| ao1 ended at {ao1_v:.4f} V")

        results.append(result)

        if settle_s_between is not None and ii < n:
            print(f" Resetting to {reset_target_v:.3f} V and settling "
                  f"before experiment {ii+1}/{n} ...")
            ao1_v = ensure_stage_at(
                ao, target_v = reset_target_v, speed_nm_per_s = reset_speed_nm_per_s,
                settle_s = settle_s_between,
                aom_gen = aom_gen, aom_transit_v = aom_transit_v,
                aom_experiment_v = aom_experiment_v,
                aom_post_ramp_wait_s = aom_post_ramp_wait_s,
            )

    return results


# ── Verification plot ──────────────────────────────────────────────────────────
def plot_session_2ch(result: dict, tvec: np.ndarray) -> None:
    """
    Verification plot for run_session_2ch().

    Top panel    — x(t) with +/-xth bands; checkpoint events marked and
                    labeled by decoded state (gray = 00/no-fire, orange =
                    protocol fired).
    Bottom panel — raw AdderOut (ai3, = ao0/10+ao1) trace with each fired
                    protocol's window shaded (located via tvec, the AI
                    sample index per camera frame from build_datasync()) —
                    the direct "trigger state vs. what actually got fired"
                    check.
    """
    pos     = result["pos"]
    params  = result["params"]
    ai_data = result["ai_data"]
    names   = result["channel_names"]

    t_cam = pos[:, 0]
    x_nm  = pos[:, 1]
    trig  = pos[:, 3]

    adder_ch = names.index("AdderOut")
    adder    = ai_data[adder_ch]
    t_ai     = np.arange(len(adder)) / params.ai_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f"2ch camera dry run  ({params.trecord_s/60:.1f} min  |  "
        f"xth={params.xth_nm}nm  dt={params.delta_t_s}s  "
        f"T_relax={params.T_relax_s}s)"
    )

    axes[0].plot(t_cam, x_nm, lw=0.5, color="steelblue", label="x")
    axes[0].axhline( params.xth_nm, color="gray", lw=0.8, ls="--")
    axes[0].axhline(-params.xth_nm, color="gray", lw=0.8, ls="--")
    for fi in np.flatnonzero((trig >= 1) & (trig <= 9)):
        state = CODE_TO_STATE[int(trig[fi])]
        color = "gray" if state == (0, 0) else "orange"
        axes[0].axvline(t_cam[fi], color=color, lw=0.8, alpha=0.6)
        axes[0].text(t_cam[fi], x_nm[fi], str(state),
                     fontsize=6, rotation=90, ha="center", va="bottom")
    axes[0].set_ylabel("x (nm)")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ai, adder, lw=0.5, color="tomato", label="AdderOut (ai3)")
    n_proto_samples = int(params.protocol_dt_s * params.ai_rate)
    y_top = axes[1].get_ylim()[1]
    for fi in np.flatnonzero((trig >= 2) & (trig <= 9)):   # fired states only
        if fi >= len(tvec):
            continue
        idx0 = tvec[fi]
        x0 = idx0 / params.ai_rate
        x1 = (idx0 + n_proto_samples) / params.ai_rate
        axes[1].axvspan(x0, x1, alpha=0.15, color="steelblue", linewidth=0)
        state = CODE_TO_STATE[int(trig[fi])]
        axes[1].text((x0 + x1) / 2, y_top, str(state),
                     fontsize=6, rotation=90, ha="center", va="top")
    axes[1].set_ylabel("AdderOut (V)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

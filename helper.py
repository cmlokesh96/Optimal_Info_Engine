"""
helper.py
=========
Internal diagnostic/plotting helpers used by Main_test_script.ipynb's
standalone hardware-diagnostic sections (9, 10) — pulled out of the
notebook so those sections stay short. Import everything with:

    from helper import *

Leading-underscore names are intentional (they're notebook-internal
scratch helpers, not part of this project's real public API the way
daq.py/experiment.py/protocols_2ch.py are) — `__all__` below is what makes
`import *` still pull them in.
"""
import threading
import time

import numpy as np
import matplotlib.pyplot as plt

from protocols_2ch import NM_PER_VOLT

__all__ = [
    "_stats",
    "_wait_for_ai_sample_count",
    "_sine_waveform_v",
    "_track_during_ramp",
    "_run_waveform_and_track",
    "_fit_ramp",
    "_plot_ramp",
    "_compare_ai1_ai3",
    "plot_fire_session",
]


# ── Section 9a: latency stats ──────────────────────────────────────────────
def _stats(name, arr_ms, budget_ms=None):
    if not arr_ms:
        print(f"  {name}: no samples"); return
    arr = np.array(arr_ms)
    line = (f"  {name}: n={len(arr)}  mean={arr.mean():.3f}ms  median={np.median(arr):.3f}ms  "
           f"p95={np.percentile(arr,95):.3f}ms  max={arr.max():.3f}ms")
    if budget_ms is not None:
        over = int((arr > budget_ms).sum())
        line += f"  | {over} ({over/len(arr):.1%}) over the {budget_ms}ms target"
    print(line)


# ── Section 10: voltage-to-pixel calibration ───────────────────────────────
def _wait_for_ai_sample_count(ai, margin_s=0.05):
    """
    NICardIn's background thread only advances get_sample_count() in
    fixed READ_N chunks, AFTER draining a full chunk - never partway
    through (see daq.py::NICardIn._reader). At ai.rate S/s, filling one
    chunk takes ai.READ_N/ai.rate seconds; querying before that always
    returns 0 (or a stale, up-to-one-chunk-old count), not the true
    real-time sample index. Sleeping only past that point (plus a small
    margin) ensures get_sample_count() reflects reality, so ai_idx0 is a
    trustworthy zero-reference for later t_ai = (index - ai_idx0)/rate
    alignment against ai1/ai3 - a fixed short sleep (e.g. 0.1s) is not
    long enough at typical AI rates/READ_N and produces a systematic
    "flat, then everything appears to start late" offset in every plot
    that uses ai_idx0.
    """
    time.sleep(ai.READ_N / ai.rate + margin_s)


def _sine_waveform_v(amplitude_v, period_s, rate):
    """
    One full period, 0 -> +amplitude_v -> 0 -> -amplitude_v -> 0 — matches
    how a channel actually behaves during a real protocol (starts and ends
    at its resting value), unlike a one-way ramp between two extremes.
    Also sweeps through every voltage magnitude twice (once rising, once
    falling), which doubles the fit's data density and lets you compare
    the two passes for hysteresis. Returns (t, v).
    """
    n = max(2, int(period_s * rate))
    t = np.linspace(0.0, period_s, n)
    return t, amplitude_v * np.sin(2 * np.pi * t / period_s)


def _track_during_ramp(camera, funcgen, ao, fps, exposure_us, duration_s,
                       roi, thres, t0, margin_s=2.0):
    """
    Hardware-triggers the camera for (duration_s + margin_s) at fps and
    tracks every frame, returning (t_rel_s, x_nm) — one pair per grabbed
    frame, t_rel_s measured from t0 (the caller's timestamp for when the
    AO waveform thread was launched, so camera samples line up with the
    commanded-voltage timeline computed from the waveform's own
    parameters). A few ms of thread-start skew between t0 and the
    waveform/camera actually beginning is negligible next to a ramp
    lasting tens of seconds.

    A missed trigger pulse (camera.grab() timeout/failure -> RuntimeError)
    is skipped rather than fatal — printed as a dropped-frame count at the
    end, same as section 9a's diagnostics.
    """
    n_frames = int((duration_s + margin_s) * fps)
    funcgen.setup_camera_trigger(fps, exposure_us, n_frames)
    camera.hardwareTrigg()
    camera.start()
    ao.hardware_trigger()   # pulses Port0/Line0 -> starts the funcgen burst
                            # (same as run_session_2ch: camera.start() then
                            # ao.hardware_trigger() - without this the camera
                            # just waits forever and grab() times out)

    frame_dt_ms = int(2000 / fps) + 2000
    t_rel, x_nm = [], []
    n_dropped = 0
    for _ in range(n_frames):
        try:
            frame = camera.grab(timeout_ms=frame_dt_ms)
        except RuntimeError:
            n_dropped += 1
            continue
        x, y = camera.find(frame, roi=roi, thres=thres)
        t_rel.append(time.perf_counter() - t0)
        x_nm.append(x)

    camera.stop()
    if n_dropped:
        print(f"  ({n_dropped}/{n_frames} frames dropped - grab() failed/timed out)")
    return np.array(t_rel), np.array(x_nm)


def _run_waveform_and_track(camera, funcgen, ao, ai, ao0_wave, ao1_wave, fps, exposure_us,
                            duration_s, roi, thres, margin_s=2.0):
    """
    Plays (ao0_wave, ao1_wave) on a background thread via
    ao.run_custom_waveform() while the main thread tracks the camera via
    _track_during_ramp(), with an AI recording spanning the whole thing.
    Returns (t_rel, x_nm, ai_idx0, ai_data).

    Wrapped in try/finally so wave_thread.join() and ai.stop_and_get()
    ALWAYS run, even if the camera loop raises or the cell is interrupted
    partway through (these waveforms can run for minutes) — otherwise the
    AO thread keeps playing and/or the AI task stays reserved in the
    background, and the very next hardware call anywhere else in the
    notebook (e.g. ensure_stage_at -> read_stage_voltage) fails with a
    device-busy error because ai1 is still held by the orphaned thread.
    join()-ing on the way out means an interrupt here waits for the
    current waveform to actually finish rather than leaving it half-run —
    the same "let it finish, don't abort mid-motion" principle used
    everywhere else in this codebase.
    """
    ai.start_continuous()
    _wait_for_ai_sample_count(ai)
    ai_idx0 = ai.get_sample_count()
    t0 = time.perf_counter()
    wave_thread = threading.Thread(target=ao.run_custom_waveform, args=(ao0_wave, ao1_wave))
    wave_thread.start()
    try:
        t_rel, x_nm = _track_during_ramp(
            camera, funcgen, ao, fps, exposure_us, duration_s, roi, thres, t0, margin_s
        )
    finally:
        wave_thread.join()
        ai_data = ai.stop_and_get()
    return t_rel, x_nm, ai_idx0, ai_data


def _fit_ramp(commanded_v, x_nm):
    """Linear fit x_nm vs. commanded_v; returns (slope_nm_per_v, intercept_nm, r2)."""
    valid = ~np.isnan(x_nm)
    slope, intercept = np.polyfit(commanded_v[valid], x_nm[valid], 1)
    resid = x_nm[valid] - (slope * commanded_v[valid] + intercept)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((x_nm[valid] - x_nm[valid].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {valid.sum()}/{len(x_nm)} frames had a valid particle position")
    return slope, intercept, r2, valid


def _plot_ramp(t_rel, commanded_v, x_nm, valid, slope, intercept, r2, v_lo, v_hi, title, color):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_rel, commanded_v, color="gray", lw=1, label="commanded (V)")
    axb = axes[0].twinx()
    axb.plot(t_rel, x_nm, ".", ms=2, color=color, label="x (nm)")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("commanded (V)"); axb.set_ylabel("x (nm)")
    axes[0].set_title(f"{title}: waveform vs. time")

    axes[1].plot(commanded_v[valid], x_nm[valid], ".", ms=3, alpha=0.4, color=color)
    fit_v = np.array([v_lo, v_hi])
    axes[1].plot(fit_v, slope * fit_v + intercept, "r-", lw=2,
                 label=f"fit: {slope:+.1f} nm/V (R^2={r2:.3f})")
    axes[1].set_xlabel("commanded (V)"); axes[1].set_ylabel("x (nm)")
    axes[1].set_title(f"{title}: x vs. commanded voltage")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def _compare_ai1_ai3(ai_data, ai_channel_names, ai_idx0, ai_rate, duration_s, title,
                     margin_s=2.0, zoom_center_s=None, zoom_window_s=0.05,
                     show_nm=True, nm_per_v=NM_PER_VOLT):
    """
    Directly compares two REAL DAQ recordings over [0, duration_s+margin_s]:
    ai3 ("AdderOut" — the commanded ao0/10+ao1 signal going INTO the stage
    amplifier) vs. ai1 ("Stage" — the amplifier's actual output, i.e. what
    really drives the piezo). Using both as real measurements (rather than
    a value predicted from the waveform's parametric shape) works
    unchanged for any commanded shape — ramp, sine, or a one-off protocol
    handoff test — and isolates the amplifier's own behavior from the
    optics/camera chain the x_nm fit checks.

    ai_idx0 is the exact ai.get_sample_count() reading taken at the same
    t0 the rest of the test's timing is referenced to — see
    _wait_for_ai_sample_count() for why the caller must wait long enough
    before capturing it (get_sample_count() is chunk-quantized, not a
    true real-time counter).

    If zoom_center_s is given, also plots a second, narrow window
    (+/- zoom_window_s) around it — e.g. to inspect a protocol's
    ao0->0 / ao1->new-offset handoff moment for spikes/glitches.
    """
    stage_row = np.asarray(ai_data[ai_channel_names.index("Stage")])
    adder_row = np.asarray(ai_data[ai_channel_names.index("AdderOut")])
    t_ai = (np.arange(len(stage_row)) - ai_idx0) / ai_rate
    keep = (t_ai >= 0) & (t_ai <= duration_s + margin_s)
    t_ai, stage_v, adder_v = t_ai[keep], stage_row[keep], adder_row[keep]

    diff_v = stage_v - adder_v
    print(f"  ai1 (Stage) vs ai3 (AdderOut): mean diff = {diff_v.mean():+.4f} V "
          f"({diff_v.mean() * nm_per_v:+.1f} nm)   "
          f"RMS diff = {np.sqrt(np.mean(diff_v ** 2)):.4f} V "
          f"({np.sqrt(np.mean(diff_v ** 2)) * nm_per_v:.1f} nm)   "
          f"max |diff| = {np.max(np.abs(diff_v)):.4f} V "
          f"({np.max(np.abs(diff_v)) * nm_per_v:.1f} nm)")

    def _plot(t, adder, stage, diff, subtitle):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(t, adder, color="gray", lw=1.2, label="ai3 (AdderOut - commanded, into amp)")
        axes[0].plot(t, stage, color="tab:green", lw=0.9, alpha=0.85, label="ai1 (Stage - amp output)")
        axes[0].set_ylabel("V"); axes[0].set_title(f"{title}{subtitle}"); axes[0].legend()
        axes[1].plot(t, diff, color="tab:red", lw=0.8)
        axes[1].set_ylabel("ai1 - ai3 (V)"); axes[1].set_xlabel("time (s)")

        if show_nm:
            for ax in (axes[0], axes[1]):
                ax_nm = ax.twinx()
                lo_v, hi_v = ax.get_ylim()
                ax_nm.set_ylim(lo_v * nm_per_v, hi_v * nm_per_v)
                ax_nm.set_ylabel(f"nm (assuming {nm_per_v:.0f} nm/V)")

        plt.tight_layout()
        plt.show()

    _plot(t_ai, adder_v, stage_v, diff_v, ": ai1 vs ai3")

    if zoom_center_s is not None:
        zkeep = (t_ai >= zoom_center_s - zoom_window_s) & (t_ai <= zoom_center_s + zoom_window_s)
        _plot(t_ai[zkeep], adder_v[zkeep], stage_v[zkeep], diff_v[zkeep],
             f" (zoom around t={zoom_center_s:.3f}s)")

    return t_ai, stage_v, adder_v, diff_v


# ── Section 10d: analytical-protocol keyboard-fire session ────────────────
def plot_fire_session(ai_data, ai_channel_names, ai_rate, ai_idx0, segments, labels,
                      baseline_v=0.0, nm_per_v=NM_PER_VOLT,
                      title="Analytical protocol fire session"):
    """
    Plot a full continuous ai1 (Stage) vs ai3 (AdderOut) recording spanning
    several separate protocol fires, each one shaded and labeled with its
    state — hw_two_channel_test.py's plot_session() segmented-overlay idea
    (multiple discrete fires on one continuous recording), extended to
    compare the two REAL channels against each other (this notebook's
    established ai1-vs-ai3 pattern, see _compare_ai1_ai3) rather than one
    channel against a value predicted from the waveform's formula.

    Parameters
    ----------
    ai_data, ai_channel_names, ai_rate : as returned by/used with NICardIn.
    ai_idx0 : int
        The ai.get_sample_count() reading taken right before the first
        fire (same t0 the segment start indices below are measured from) —
        see _wait_for_ai_sample_count()'s docstring for why reading this
        too soon silently mislabels the whole time axis.
    segments : list of (idx_start, n_samples)
        Per fire: AI sample index (relative to ai_idx0) the fire started
        at, and how many AO samples it ran for.
    labels : list of str
        One label per segment.
    baseline_v : float, optional
        Subtracted from both ai1 and ai3 before plotting — pass the
        pre-loop AdderOut reading (wherever ao1 happened to be resting
        when the session started) so the y-axis reads "displacement from
        session start" instead of an absolute voltage that depends on
        whatever earlier cell last moved the stage. Default 0.0 (plot the
        raw absolute voltages, unchanged from before).
    """
    stage_row = np.asarray(ai_data[ai_channel_names.index("Stage")]) - baseline_v
    adder_row = np.asarray(ai_data[ai_channel_names.index("AdderOut")]) - baseline_v
    t_ai = (np.arange(len(stage_row)) - ai_idx0) / ai_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(title)

    axes[0].plot(t_ai, adder_row, color="gray", lw=1.0, label="ai3 (AdderOut - commanded)")
    axes[0].plot(t_ai, stage_row, color="tab:green", lw=0.8, alpha=0.85, label="ai1 (Stage - actual)")
    axes[0].set_ylabel("ΔV from start (V)"); axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)

    diff = stage_row - adder_row
    axes[1].plot(t_ai, diff, color="tab:red", lw=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("ai1 - ai3 (V)"); axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)

    y_top = axes[0].get_ylim()[1]
    for (idx_start, n_samples), label in zip(segments, labels):
        x0, x1 = idx_start / ai_rate, (idx_start + n_samples) / ai_rate
        for ax in axes:
            ax.axvspan(x0, x1, alpha=0.10, color="steelblue", linewidth=0)
            ax.axvline(x0, color="gray", lw=0.6, ls="--")
        axes[0].text(
            (x0 + x1) / 2, y_top, label, ha="center", va="top",
            fontsize=7, color="dimgray", rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
        )

    plt.tight_layout()
    plt.show()

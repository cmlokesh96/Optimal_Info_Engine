"""
hw_two_channel_test.py
=======================
Validate the new analog adder circuit: ao0 and ao1 feed a summing circuit
whose output (ao0/10 + ao1) is wired into ai3. This script drives ao0/ao1
as a single synchronized 2-channel AO task and reads ai3 continuously,
comparing measured vs predicted (ao0/10 + ao1) — same pattern as
hw_protocol_test.py (pre-configured AO task reused via stop/write/start,
continuous AI background thread, shaded end-of-session plot), extended to
two AO channels.

ao0 is the fine channel (raw range -10..10 V, attenuated /10 by the
circuit); ao1 is the coarse channel (kept in the existing stage range
1.5..8.5 V for consistency with the rest of the project).

Wiring:
    ao0, ao1  →  adder circuit  →  AI3+   (signal)
    AGND      →  AI7                      (differential negative pair for AI3 on USB-6003)
"""

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

from protocols_2ch import (
    get_protocol_nm, split_channels, STATE_LABELS, STATE_MENU,
    AO0_MIN, AO0_MAX, AO1_MIN, AO1_MAX,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE   = "Dev4"
AO0_CHAN = f"{DEVICE}/ao0"
AO1_CHAN = f"{DEVICE}/ao1"
AI_CHAN  = f"{DEVICE}/ai3"

RATE = 1000

# The AO task's FINITE sample-clock timing is configured ONCE (see manual_loop)
# with samps_per_chan derived from this single duration, then reused for every
# fire — raw DC-hold points, sweep, AND state protocols all share it. They must
# all be the same length: the hardware only ever clocks out exactly
# samps_per_chan samples per FINITE run, so writing a longer waveform than that
# silently truncates it (the tail never plays) — that was the ramp-cuts-off bug.
DURATION_S    = 3.0        # shared fire duration: DC-hold points, sweep, AND protocols
PROTOCOL_DT_S = DURATION_S # protocols use the same duration as everything else
DT_S          = 1.0        # placeholder camera delta-t (no real camera yet) — only
                            # used to time the "00 = hold 2*dt, no fire" wait

AO1_CENTER = 5.0

# ── AO FIRE (uses a pre-configured 2-channel task — no setup overhead per trigger) ──
def fire_ao(ao_task: nidaqmx.Task, ao0_wave: np.ndarray, ao1_wave: np.ndarray,
            duration_s: float, on_start=None) -> None:
    """
    Write and fire a (ao0, ao1) waveform pair on an already-configured
    2-channel AO task. Channel order in the write array must match the
    order the channels were added (ao0 first, ao1 second).
    """
    ao_task.stop()
    ao_task.write(np.vstack([ao0_wave, ao1_wave]), auto_start=False)
    ao_task.start()
    if on_start:
        on_start()
    ao_task.wait_until_done(timeout=duration_s + 5.0)

def hold_waveform(value: float, duration_s: float = DURATION_S) -> np.ndarray:
    return np.full(int(duration_s * RATE), value, dtype=np.float64)

# ── SWEEP PRESET ──────────────────────────────────────────────────────────────
def sweep_points() -> list[tuple[float, float, str]]:
    """(ao0_v, ao1_v, label) test points: fine alone, coarse alone, combined."""
    pts: list[tuple[float, float, str]] = []
    for ao0 in (-10.0, -5.0, 0.0, 5.0, 10.0):
        pts.append((ao0, AO1_CENTER, f"fine {ao0:+.0f}V"))
    for ao1 in (AO1_MIN, (AO1_MIN + AO1_CENTER) / 2, AO1_CENTER,
                (AO1_CENTER + AO1_MAX) / 2, AO1_MAX):
        pts.append((0.0, ao1, f"coarse {ao1:.2f}V"))
    for ao0, ao1 in ((10.0, AO1_MIN), (-10.0, AO1_MAX), (5.0, 6.5), (-5.0, 3.5)):
        pts.append((ao0, ao1, f"combo ({ao0:+.0f},{ao1:.2f})"))
    return pts

# ── SESSION PLOT ──────────────────────────────────────────────────────────────
def plot_session(ai_array: np.ndarray,
                 segments: list[tuple[int, np.ndarray, np.ndarray]],
                 labels: list[str]) -> None:
    """
    ai_array : full continuous AI3 recording for the session (V)
    segments : list of (ai_sample_index_at_ao_start, ao0_wave, ao1_wave)
    labels   : human-readable label per segment

    Top panel    — AI3 continuous + predicted (ao0/10+ao1) overlaid only where active.
    Bottom panel — Error (AI3 − predicted), NaN elsewhere.
    """
    if not segments:
        print("  No data to plot.")
        return

    t_ai = np.arange(len(ai_array)) / RATE

    predicted_trace = np.full(len(ai_array), np.nan)
    for idx_start, ao0_wave, ao1_wave in segments:
        predicted = ao0_wave / 10.0 + ao1_wave
        sl = slice(idx_start, min(idx_start + len(predicted), len(ai_array)))
        predicted_trace[sl] = predicted[: len(predicted_trace[sl])]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Adder circuit: ao0/10 + ao1 → AI3  ({RATE} S/s)")

    axes[0].plot(t_ai, ai_array, color="tomato", lw=0.8, label="AI3 (measured)")
    axes[0].plot(t_ai, predicted_trace, color="steelblue", lw=1.1, ls="--",
                 label="predicted (ao0/10 + ao1)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    err = ai_array - predicted_trace
    axes[1].plot(t_ai, err * 1000, color="darkorchid", lw=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("Error  AI3 − predicted  (mV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    y_top = axes[0].get_ylim()[1]
    for i, (idx_start, ao0_wave, ao1_wave) in enumerate(segments):
        x0 = idx_start / RATE
        x1 = (idx_start + len(ao0_wave)) / RATE
        for ax in axes:
            ax.axvspan(x0, x1, alpha=0.10, color="steelblue", linewidth=0)
            ax.axvline(x0, color="gray", lw=0.6, ls="--")
        axes[0].text(
            (x0 + x1) / 2, y_top, labels[i],
            ha="center", va="top", fontsize=7, color="dimgray", rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
        )

    plt.tight_layout()
    plt.show()

# ── MANUAL LOOP ───────────────────────────────────────────────────────────────
def manual_loop(default_duration: float = DURATION_S) -> None:
    """
    Interactive loop: pre-configured 2-channel AO task, continuous AI3 thread.

        <ao0_v> <ao1_v>   hold both at these DC levels, fire, print predicted vs measured
        sweep             run the preset ao0/ao1 grid automatically
        r                 reset to (0, AO1_CENTER)
        q                 quit + plot
    """
    n = int(default_duration * RATE)

    # ── Continuous AI3 background thread ──────────────────────────────────────
    ai_samples : list[float]        = []
    ai_stop    : threading.Event    = threading.Event()
    session_t0 : list[float | None] = [None]
    READ_N = 100

    def _ai_continuous():
        with nidaqmx.Task() as ai:
            ai.ai_channels.add_ai_voltage_chan(
                AI_CHAN,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10.0,
                max_val=10.0,
            )
            ai.timing.cfg_samp_clk_timing(
                rate=RATE, sample_mode=AcquisitionType.CONTINUOUS
            )
            ai.start()
            session_t0[0] = time.perf_counter()

            while not ai_stop.is_set():
                avail = ai.in_stream.avail_samp_per_chan
                if avail >= READ_N:
                    chunk = ai.read(number_of_samples_per_channel=READ_N, timeout=1.0)
                    ai_samples.extend(chunk)
                else:
                    time.sleep(READ_N / RATE / 4)

            avail = ai.in_stream.avail_samp_per_chan
            if avail > 0:
                ai_samples.extend(
                    ai.read(number_of_samples_per_channel=avail, timeout=2.0)
                )

    ai_thread = threading.Thread(target=_ai_continuous, daemon=True)
    ai_thread.start()
    while session_t0[0] is None:
        time.sleep(0.005)

    # ── Pre-configure 2-channel AO task ONCE ───────────────────────────────────
    with nidaqmx.Task() as ao:
        ao.ao_channels.add_ao_voltage_chan(AO0_CHAN, min_val=AO0_MIN, max_val=AO0_MAX)
        ao.ao_channels.add_ao_voltage_chan(AO1_CHAN, min_val=AO1_MIN, max_val=AO1_MAX)
        ao.timing.cfg_samp_clk_timing(
            rate=RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n
        )

        segments : list[tuple[int, np.ndarray, np.ndarray]] = []
        labels   : list[str]                                = []

        ao1_offset: list[float] = [AO1_CENTER]   # persists across state fires

        print(f"\n{'─'*60}")
        print(f" Adder circuit test - ao0 [{AO0_MIN},{AO0_MAX}] V, "
              f"ao1 [{AO1_MIN},{AO1_MAX}] V")
        print(f" Duration: {default_duration} s  ({n} samples)  RATE: {RATE} S/s")
        print(f" Protocol duration: {PROTOCOL_DT_S} s   dt (placeholder): {DT_S} s")
        print(f"{'─'*60}")
        print("  State select - same call pattern the camera trigger path will")
        print("  use (replace input() with a camera measurement + decode_state_pair()):")
        print(f"    0  {STATE_LABELS[(1,1)]:<24}  4  {STATE_LABELS[(-1,-1)]}")
        print(f"    1  {STATE_LABELS[(1,0)]:<24}  5  {STATE_LABELS[(-1,0)]}")
        print(f"    2  {STATE_LABELS[(0,1)]:<24}  6  {STATE_LABELS[(0,-1)]}")
        print(f"    3  {STATE_LABELS[(-1,1)]:<24}  7  {STATE_LABELS[(1,-1)]}")
        print(f"    8  {STATE_LABELS[(0,0)]}")
        print("  <ao0_v> <ao1_v>   raw point: hold + fire + predicted vs measured")
        print("  sweep             run preset ao0/ao1 grid")
        print(f"  r                 reset to (0, {AO1_CENTER}), ao1 offset -> {AO1_CENTER}")
        print("  q                 quit + show plot")
        print(f"{'─'*60}\n")

        def _fire_point(ao0_v: float, ao1_v: float, label: str, duration_s: float,
                        report: bool) -> None:
            ao0_wave = hold_waveform(ao0_v, duration_s)
            ao1_wave = hold_waveform(ao1_v, duration_s)
            n_pts = len(ao0_wave)

            start_idx: list[int | None] = [None]

            def on_start(t0=session_t0, idx=start_idx):
                idx[0] = int((time.perf_counter() - t0[0]) * RATE)

            fire_ao(ao, ao0_wave, ao1_wave, duration_s, on_start=on_start)

            if start_idx[0] is not None:
                segments.append((start_idx[0], ao0_wave, ao1_wave))
                labels.append(label)

            if report:
                time.sleep(READ_N / RATE + 0.05)   # let the AI thread drain the tail
                predicted = ao0_v / 10.0 + ao1_v
                s0, s1 = start_idx[0], start_idx[0] + n_pts
                measured_window = ai_samples[s0:min(s1, len(ai_samples))]
                if measured_window:
                    measured = float(np.mean(measured_window))
                    err_mv = (measured - predicted) * 1000
                    print(f"  ao0={ao0_v:+.2f}V  ao1={ao1_v:.2f}V  -> "
                          f"predicted={predicted:+.4f}V  measured={measured:+.4f}V  "
                          f"err={err_mv:+.1f} mV")
                else:
                    print("  (no AI samples captured for this window)")

        def _fire_state(key: str) -> None:
            state = STATE_MENU[key]
            label = STATE_LABELS[state]

            if state == (0, 0):
                print(f"  State {label} - no protocol fired, waiting 2*dt "
                      f"({2*DT_S:.1f}s) ...")
                time.sleep(2 * DT_S)
                print("  (would resample here in the real camera loop)\n")
                return

            waveform_nm = get_protocol_nm(state, PROTOCOL_DT_S, RATE)
            ao0_wave, ao1_wave, ao1_new = split_channels(waveform_nm, ao1_offset[0])

            print(f"  State {label}  (ao1 offset {ao1_offset[0]:.4f} V) ...")

            start_idx: list[int | None] = [None]

            def on_start(t0=session_t0, idx=start_idx):
                idx[0] = int((time.perf_counter() - t0[0]) * RATE)

            fire_ao(ao, ao0_wave, ao1_wave, PROTOCOL_DT_S, on_start=on_start)

            if start_idx[0] is not None:
                segments.append((start_idx[0], ao0_wave, ao1_wave))
                labels.append(label)

            ao1_offset[0] = ao1_new
            print(f"  -> ao1 offset now {ao1_offset[0]:.4f} V\n")

        # ── Establish the resting level immediately — ao1 must be at AO1_CENTER
        # from the moment the session starts, not float at whatever ao1 was
        # left at (e.g. 0V) until the first manual command.
        print(f"  Initializing stage: ao0=0V, ao1={AO1_CENTER}V ...")
        _fire_point(0.0, AO1_CENTER, "init", default_duration, report=False)

        while True:
            try:
                raw = input("Point > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                break

            if raw == 'q':
                break

            if raw == 'r':
                print(f"  Reset to (0, {AO1_CENTER}) ...")
                _fire_point(0.0, AO1_CENTER, "reset", default_duration, report=False)
                ao1_offset[0] = AO1_CENTER
                continue

            if raw in STATE_MENU:
                _fire_state(raw)
                continue

            if raw == 'sweep':
                print("  Running preset sweep ...")
                for ao0_v, ao1_v, label in sweep_points():
                    print(f"    {label} ...")
                    _fire_point(ao0_v, ao1_v, label, default_duration, report=True)
                print("  Sweep done.\n")
                continue

            parts = raw.split()
            if len(parts) != 2:
                print("  Enter '<ao0_v> <ao1_v>', 'sweep', 'r', or 'q'.")
                continue

            try:
                ao0_v, ao1_v = float(parts[0]), float(parts[1])
            except ValueError:
                print("  Could not parse two numbers.")
                continue

            if not (AO0_MIN <= ao0_v <= AO0_MAX):
                print(f"  ao0 must be within [{AO0_MIN}, {AO0_MAX}] V.")
                continue
            if not (AO1_MIN <= ao1_v <= AO1_MAX):
                print(f"  ao1 must be within [{AO1_MIN}, {AO1_MAX}] V.")
                continue

            _fire_point(ao0_v, ao1_v, f"({ao0_v:+.2f},{ao1_v:.2f})",
                       default_duration, report=True)

    # AO task closed here by the `with` block

    # ── Stop AI and plot ──────────────────────────────────────────────────────
    ai_stop.set()
    ai_thread.join(timeout=3.0)

    ai_array = np.array(ai_samples)
    print(f"\nSession: {len(segments)} point(s), "
          f"{len(ai_array)} AI3 samples ({len(ai_array)/RATE:.1f} s total)")

    plot_session(ai_array, segments, labels)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─" * 60)
    print(" ao0/ao1 -> adder circuit -> AI3  (USB-6003)")
    print("─" * 60)
    manual_loop(default_duration=DURATION_S)

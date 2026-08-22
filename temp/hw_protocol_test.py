"""
hw_protocol_test.py
====================
Send state-dependent protocols on AO1; record AI3 continuously for the
entire session; plot everything at the end with protocol regions shaded.

AO task is created ONCE at session start — each trigger only calls
stop → write → start on the already-configured task, so there is no
channel/timing setup overhead between protocols.

AI3 runs as a single continuous task — you see the full dynamics of AO1
from session start to quit with no gaps, no margins, no cross-correlation.

Buffer note: nidaqmx maintains a SOFTWARE buffer (much larger than the
2047-sample hardware FIFO). FINITE mode writes the full waveform in one
call; the driver silently refills the hardware FIFO as it drains.

Wiring:
    AO1  →  AI3+   (signal)
    AGND →  AI7    (differential negative pair for AI3 on USB-6003)
"""

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE      = "Dev4"
AO_CHAN     = f"{DEVICE}/ao1"
AI_CHAN     = f"{DEVICE}/ai3"

RATE        = 1000
DURATION_S  = 5.0

V_MIN       = 1.5
V_MAX       = 8.5
V_CENTER    = 5.0
NM_PER_VOLT = 3000.0

# ── PROTOCOL SHAPES ───────────────────────────────────────────────────────────
def make_protocol(duration_s: float, state: int) -> np.ndarray:
    """Return relative displacement in volts from the current stage position."""
    n = int(duration_s * RATE)
    t = np.linspace(0, duration_s, n)
    if state == 0b00:
        rel_nm = np.zeros(n)
    elif state == 0b01:
        rel_nm = np.concatenate([[25.0], np.linspace(25.0, 120.0, n - 1)])
    elif state == 0b10:
        rel_nm = np.concatenate([[-25.0], np.linspace(-25.0, 50.0, n - 1)])
    elif state == 0b11:
        tau = duration_s / 3.0
        rel_nm = 150.0 * (1 - np.exp(-t / tau))
        rel_nm[0] = 0.0
    else:
        raise ValueError(f"Unknown state: {state}")
    return rel_nm / NM_PER_VOLT

# ── AO FIRE (uses a pre-configured task — no setup overhead per trigger) ──────
def fire_ao(ao_task: nidaqmx.Task, waveform: np.ndarray,
            duration_s: float, on_start=None) -> None:
    """
    Write and fire a waveform on an already-configured AO task.

    The task is stopped first (resets it after previous FINITE run), then
    the new waveform is written and the task is started.  on_start() fires
    immediately after ao.start() so the caller can record the exact
    wall-clock moment the first sample hits the wire.

    This is the hot path called on every camera trigger — no Task(),
    no add_ao_voltage_chan(), no cfg_samp_clk_timing() here.
    """
    ao_task.stop()                          # reset after previous FINITE run
    ao_task.write(waveform, auto_start=False)
    ao_task.start()
    if on_start:
        on_start()
    ao_task.wait_until_done(timeout=duration_s + 5.0)

STATE_NAMES = {
    0b00: "00 — hold",
    0b01: "01 — +ramp",
    0b10: "10 — -ramp",
    0b11: "11 — exp",
}

STATE_MAP = {
    '0': 0b00, '00': 0b00,
    '1': 0b01, '01': 0b01,
    '2': 0b10, '10': 0b10,
    '3': 0b11, '11': 0b11,
}

# ── SESSION PLOT ──────────────────────────────────────────────────────────────
def plot_session(ai_array: np.ndarray,
                 ao_segments: list[tuple[int, np.ndarray]],
                 labels: list[str]) -> None:
    """
    ai_array   : full continuous AI3 recording for the session (V)
    ao_segments: list of (ai_sample_index_at_ao_start, waveform)
    labels     : human-readable label per segment

    Top panel   — AI3 continuous + AO1 overlaid only where active.
                  Protocol windows are shaded; idle time is unshaded.
    Bottom panel — Error (AI3 − AO1) only where AO was running (NaN elsewhere).
    """
    if not ao_segments:
        print("  No data to plot.")
        return

    t_ai = np.arange(len(ai_array)) / RATE

    # Sparse AO trace — NaN everywhere AO was idle between protocols
    ao_trace = np.full(len(ai_array), np.nan)
    for idx_start, waveform in ao_segments:
        sl = slice(idx_start, min(idx_start + len(waveform), len(ai_array)))
        ao_trace[sl] = waveform[: len(ao_trace[sl])]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"AO1 → AI3  full session  (continuous AI3, {RATE} S/s)"
    )

    axes[0].plot(t_ai, ai_array, color="tomato",    lw=0.8, label="AI3 (continuous)")
    axes[0].plot(t_ai, ao_trace, color="steelblue", lw=1.1, ls="--",
                 label="AO1 sent (active windows only)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    err = ai_array - ao_trace   # NaN where AO idle → no line drawn
    axes[1].plot(t_ai, err * 1000, color="darkorchid", lw=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("Error  AI3 − AO1  (mV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    y_top = axes[0].get_ylim()[1]
    for i, (idx_start, waveform) in enumerate(ao_segments):
        x0 = idx_start / RATE
        x1 = (idx_start + len(waveform)) / RATE
        for ax in axes:
            ax.axvspan(x0, x1, alpha=0.10, color="steelblue", linewidth=0)
            ax.axvline(x0, color="gray", lw=0.6, ls="--")
        axes[0].text(
            (x0 + x1) / 2, y_top, labels[i],
            ha="center", va="top", fontsize=8, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
        )

    plt.tight_layout()
    plt.show()

# ── MANUAL LOOP ───────────────────────────────────────────────────────────────
def manual_loop(default_duration: float = DURATION_S) -> None:
    """
    Interactive loop with a single pre-configured AO task and a continuous
    AI3 recording thread.

    The AO task is created ONCE before the loop starts:
        channel  → configured once  (add_ao_voltage_chan)
        timing   → configured once  (cfg_samp_clk_timing for n samples)
        per trigger → stop / write / start   (no setup overhead)

    This is the same call pattern the camera trigger path will use —
    replace input() with a camera measurement and decode_state() call.

        0 / 00  hold       1 / 01  +ramp
        2 / 10  -ramp      3 / 11  exp
        r  re-center       q  quit + plot
    """
    n = int(default_duration * RATE)          # samples per protocol (fixed)
    center_wave = np.full(n, V_CENTER)        # re-center waveform (same length)

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

    # ── Pre-configure AO task ONCE — reused for every protocol trigger ─────────
    with nidaqmx.Task() as ao:
        ao.ao_channels.add_ao_voltage_chan(AO_CHAN, min_val=V_MIN, max_val=V_MAX)
        ao.timing.cfg_samp_clk_timing(
            rate=RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n
        )
        # AO task is now ready; fire_ao() only does stop/write/start from here on

        ao_segments     : list[tuple[int, np.ndarray]] = []
        labels          : list[str]                    = []
        current_voltage : float                        = V_CENTER

        print(f"\n{'─'*60}")
        print(f" Manual loop  —  AO task pre-configured, AI3 recording")
        print(f" Duration: {default_duration} s  ({n} samples)  RATE: {RATE} S/s")
        print(f" Stage at {current_voltage:.3f} V  "
              f"({(current_voltage - V_CENTER)*NM_PER_VOLT:+.1f} nm from center)")
        print(f"{'─'*60}")
        print("  0/00 hold  |  1/01 +ramp  |  2/10 -ramp  |  3/11 exp")
        print("  r  re-center   |   q  quit + show plot")
        print(f"{'─'*60}\n")

        while True:
            try:
                raw = input("State > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                break

            if raw == 'q':
                break

            if raw == 'r':
                print(f"  Re-centering to {V_CENTER} V ...")
                # Re-center uses the same n-sample waveform — no timing reconfiguration
                fire_ao(ao, center_wave, default_duration)
                current_voltage = V_CENTER
                print(f"  Stage at {V_CENTER} V  (center)\n")
                continue

            if raw not in STATE_MAP:
                print("  Enter 0–3 (or 00/01/10/11), 'r' to re-center, 'q' to quit.")
                continue

            state = STATE_MAP[raw]
            waveform = np.clip(
                current_voltage + make_protocol(default_duration, state), V_MIN, V_MAX
            )
            print(f"  State {STATE_NAMES[state]}  (from {current_voltage:.3f} V) ...")

            # ── Camera trigger path looks exactly like this block ─────────────
            # measurement = camera.read()
            # state = decode_state(measurement)
            # waveform = np.clip(current_voltage + make_protocol(duration, state), ...)
            ao_start_idx: list[int | None] = [None]

            def on_start(t0=session_t0, idx=ao_start_idx):
                idx[0] = int((time.perf_counter() - t0[0]) * RATE)

            fire_ao(ao, waveform, default_duration, on_start=on_start)
            # ── End of camera trigger path ────────────────────────────────────

            current_voltage = float(waveform[-1])
            if ao_start_idx[0] is not None:
                ao_segments.append((ao_start_idx[0], waveform))
                labels.append(STATE_NAMES[state])

            nm_now = (current_voltage - V_CENTER) * NM_PER_VOLT
            print(f"  → {current_voltage:.4f} V  ({nm_now:+.1f} nm from center)\n")

    # AO task closed here by the `with` block

    # ── Stop AI and plot ──────────────────────────────────────────────────────
    ai_stop.set()
    ai_thread.join(timeout=3.0)

    ai_array = np.array(ai_samples)
    print(f"\nSession: {len(ao_segments)} protocol(s), "
          f"{len(ai_array)} AI3 samples ({len(ai_array)/RATE:.1f} s total)")

    plot_session(ai_array, ao_segments, labels)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─" * 60)
    print(" AO1 → AI3  (USB-6003, pre-configured AO, continuous AI3)")
    print("─" * 60)

    # Show software buffer size before starting the session
    with nidaqmx.Task() as _ao:
        _ao.ao_channels.add_ao_voltage_chan(AO_CHAN, min_val=V_MIN, max_val=V_MAX)
        _ao.timing.cfg_samp_clk_timing(
            rate=RATE, sample_mode=AcquisitionType.FINITE,
            samps_per_chan=int(DURATION_S * RATE)
        )
        sw_buf = _ao.out_stream.output_buf_size
    print(f" HW FIFO: 2047 samples  |  SW buffer: {sw_buf} samples "
          f"({sw_buf/RATE:.1f} s)\n")

    manual_loop(default_duration=DURATION_S)

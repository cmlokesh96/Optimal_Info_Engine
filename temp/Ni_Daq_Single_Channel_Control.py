"""
Ni_Daq_Single_Channel_Control.py
=================================
NI USB-6003: AO0 piezo-stage control with AI3 loopback verification.

Wiring:
    AO0 ──── AI3  (loopback wire — verify here before connecting to stage)
    AO0 ────> Piezo stage (after loopback passes)

Stage:      Piezoconcept LFHS3, 30 µm / 10 V
DAQ:        NI USB-6003, AO ±10 V, 16-bit, 5 kS/s, FIFO 2047 samples
AO range:   1.5 V → 8.5 V  (4.5 µm → 25.5 µm, center 5 V = 15 µm)
"""

import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    MOCK = False
except ImportError:
    MOCK = True
    print("[INFO] nidaqmx not found — running in MOCK mode\n")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE      = "Dev4"
AO_CHAN     = f"{DEVICE}/ao1"
AI_CHAN     = f"{DEVICE}/ai3"

RATE        = 1000          # S/s
CHUNK       = 500           # samples per FIFO write

V_MIN       = 1.5
V_MAX       = 8.5
V_CENTER    = 5.0

NM_PER_VOLT = 3000.0

# ── STATE ─────────────────────────────────────────────────────────────────────
current_voltage = V_CENTER
trigger_queue   = queue.Queue()
THRESHOLD       = 0.0

# ── UNIT HELPERS ──────────────────────────────────────────────────────────────
def nm_to_v(nm: float | np.ndarray) -> float | np.ndarray:
    return nm / NM_PER_VOLT

def v_to_nm(v: float | np.ndarray) -> float | np.ndarray:
    return v * NM_PER_VOLT

def clamp(v: np.ndarray) -> np.ndarray:
    clipped = np.clip(v, V_MIN, V_MAX)
    n = int(np.sum(clipped != v))
    if n:
        print(f"  [WARN] {n} samples clipped to [{V_MIN}, {V_MAX}] V")
    return clipped

# ── PROTOCOLS ─────────────────────────────────────────────────────────────────
def make_protocol(duration_s: float, state: int) -> np.ndarray:
    """Return relative voltage array for the given state, starting at 0 V."""
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

    return nm_to_v(rel_nm)

PROTOCOLS_3S = {s: make_protocol(3.0, s) for s in [0b00, 0b01, 0b10, 0b11]}

# ── AO STREAMING ──────────────────────────────────────────────────────────────
def _ao_mock(waveform: np.ndarray, on_start=None):
    if on_start:
        on_start()
    n_chunks = int(np.ceil(len(waveform) / CHUNK))
    for i in range(n_chunks):
        chunk = waveform[i * CHUNK : (i + 1) * CHUNK]
        time.sleep(len(chunk) / RATE)
        print(f"  [MOCK AO] chunk {i+1}/{n_chunks}  "
              f"{chunk[0]:.4f} V → {chunk[-1]:.4f} V")

def _ao_hw(waveform: np.ndarray, on_start=None):
    """
    Hardware-timed AO streaming. on_start() is called immediately after
    ao.start() so callers can timestamp the true AO output start.
    """
    with nidaqmx.Task() as ao:
        ao.ao_channels.add_ao_voltage_chan(AO_CHAN, min_val=V_MIN, max_val=V_MAX)
        ao.timing.cfg_samp_clk_timing(rate=RATE, sample_mode=AcquisitionType.CONTINUOUS)
        ao.write(waveform[:CHUNK], auto_start=False)
        ao.start()
        if on_start:
            on_start()      # fires right after the AO clock starts
        for i in range(CHUNK, len(waveform), CHUNK):
            ao.write(waveform[i : i + CHUNK])
        time.sleep(CHUNK / RATE + 0.05)   # let last chunk drain

def stream_ao(waveform: np.ndarray, on_start=None):
    if MOCK:
        _ao_mock(waveform, on_start)
    else:
        _ao_hw(waveform, on_start)

# ── PROTOCOL RUNNER (camera loop) ────────────────────────────────────────────
def decode_state(meas1: float, meas2: float) -> int:
    return (int(meas1 > THRESHOLD) << 1) | int(meas2 > THRESHOLD)

def run_protocol(state: int, duration_s: float = 3.0):
    """Build absolute waveform anchored to current_voltage and stream it."""
    global current_voltage
    rel_v = PROTOCOLS_3S[state] if (duration_s == 3.0 and state in PROTOCOLS_3S) \
            else make_protocol(duration_s, state)
    abs_v = clamp(current_voltage + rel_v)
    print(f"  Protocol {state:02b}  {duration_s}s  "
          f"{v_to_nm(abs_v[0]):.1f} nm → {v_to_nm(abs_v[-1]):.1f} nm")
    stream_ao(abs_v)
    current_voltage = abs_v[-1]

def main_loop(protocol_duration: float = 3.0):
    """Camera-triggered experiment loop — blocks waiting on trigger_queue."""
    print(f"Experiment started. Stage at {v_to_nm(current_voltage)/1000:.2f} µm\n")
    while True:
        meas1, meas2 = trigger_queue.get()
        state = decode_state(meas1, meas2)
        print(f"\nTrigger: state={state:02b}  ({meas1:.3f}, {meas2:.3f})")
        run_protocol(state, duration_s=protocol_duration)
        print(f"Stage at {v_to_nm(current_voltage)/1000:.3f} µm")

# ── HEADROOM CHECK ────────────────────────────────────────────────────────────
def check_headroom(margin: float = 1.0):
    if current_voltage > V_MAX - margin:
        print(f"  [WARN] Close to upper rail: {current_voltage:.3f} V (limit {V_MAX} V)")
    elif current_voltage < V_MIN + margin:
        print(f"  [WARN] Close to lower rail: {current_voltage:.3f} V (limit {V_MIN} V)")

# ── SESSION PLOT ──────────────────────────────────────────────────────────────
def plot_session(ai_samples: np.ndarray,
                 ao_segments: list[tuple[int, np.ndarray]],
                 labels: list[str]):
    """
    ai_samples : continuous AI3 array collected since session start (V)
    ao_segments: list of (ai_sample_index_when_ao_started, waveform_array)
    labels     : human-readable label per segment
    """
    if not ao_segments:
        print("  No data to plot.")
        return

    t_ai = np.arange(len(ai_samples)) / RATE

    # Build sparse AO trace — NaN where AO was idle (user at the prompt)
    ao_trace = np.full(len(ai_samples), np.nan)
    for idx_start, waveform in ao_segments:
        idx_end = idx_start + len(waveform)
        sl = slice(idx_start, min(idx_end, len(ai_samples)))
        ao_trace[sl] = waveform[: len(ao_trace[sl])]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("AO0 commanded vs AI3 read-back — manual loop session")

    # ── Voltage panel ─────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(t_ai, ai_samples, color="tomato",    linewidth=0.8, label="AI3 read (V)")
    ax0.plot(t_ai, ao_trace,   color="steelblue", linewidth=1.0,
             linestyle="--", label="AO0 sent (V)")
    ax0.set_ylabel("Voltage (V)")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # ── Error panel ───────────────────────────────────────────────────────
    ax1 = axes[1]
    err = ai_samples - ao_trace              # NaN where AO idle
    ax1.plot(t_ai, err * 1000, color="darkorchid", linewidth=0.7, label="AI3 − AO0")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Error (mV)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)

    # ── Protocol boundary shading + labels ────────────────────────────────
    y_top = ax0.get_ylim()[1]
    for i, (idx_start, waveform) in enumerate(ao_segments):
        x0 = idx_start / RATE
        x1 = (idx_start + len(waveform)) / RATE
        for ax in axes:
            ax.axvspan(x0, x1, alpha=0.07, color="steelblue", linewidth=0)
            ax.axvline(x0, color="gray", linewidth=0.6, linestyle="--")
        ax0.text((x0 + x1) / 2, y_top, labels[i],
                 ha="center", va="top", fontsize=7, color="dimgray")

    plt.tight_layout()
    plt.show()

# ── MANUAL TRIGGER LOOP ───────────────────────────────────────────────────────
def manual_loop(default_duration: float = 3.0):
    """
    Interactive loop. Runs a single continuous AI3 task for the entire session
    so there is no per-protocol start/stop overhead or timing offset.

    AO start index in the AI stream is captured via on_start() callback fired
    immediately after ao.start() — giving sub-millisecond alignment.

    Prompt:
        0/00  hold    1/01  +ramp    2/10  -ramp    3/11  exp
        r  re-center      q  quit + plot
    """
    global current_voltage

    STATE_MAP = {
        '0': 0b00, '00': 0b00,
        '1': 0b01, '01': 0b01,
        '2': 0b10, '10': 0b10,
        '3': 0b11, '11': 0b11,
    }

    # ── Continuous AI3 background reader ──────────────────────────────────
    ai_samples   : list[float] = []
    ai_stop      = threading.Event()
    session_t0   : list[float | None] = [None]   # wall-clock time at AI hardware start

    READ_N = 100    # drain 100 samples per poll (0.1 s at 1 kHz)

    def _ai_continuous():
        if MOCK:
            session_t0[0] = time.perf_counter()
            while not ai_stop.is_set():
                time.sleep(READ_N / RATE)
                ai_samples.extend(np.full(READ_N, V_CENTER).tolist())
            return

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
            session_t0[0] = time.perf_counter()   # t=0 of the AI stream

            while not ai_stop.is_set():
                avail = ai.in_stream.avail_samp_per_chan
                if avail >= READ_N:
                    chunk = ai.read(number_of_samples_per_channel=READ_N, timeout=1.0)
                    ai_samples.extend(chunk)
                else:
                    time.sleep(READ_N / RATE / 4)

            # Drain whatever remains in hardware buffer
            avail = ai.in_stream.avail_samp_per_chan
            if avail > 0:
                chunk = ai.read(number_of_samples_per_channel=avail, timeout=2.0)
                ai_samples.extend(chunk)

    ai_thread = threading.Thread(target=_ai_continuous, daemon=True)
    ai_thread.start()

    # Wait until AI hardware has actually started (session_t0 is set)
    while session_t0[0] is None:
        time.sleep(0.005)

    # ── Session data ──────────────────────────────────────────────────────
    ao_segments : list[tuple[int, np.ndarray]] = []
    labels      : list[str] = []

    print(f"\n{'─'*60}")
    print(f" Manual loop  —  AI3 continuous recording active")
    print(f" Duration: {default_duration} s   Stage: {v_to_nm(current_voltage)/1000:.3f} µm")
    print(f"{'─'*60}")
    print("  0/00 hold  |  1/01 +ramp  |  2/10 -ramp  |  3/11 exp")
    print("  r  re-center   |   q  quit + show plot")
    print(f"{'─'*60}\n")

    while True:
        check_headroom()
        try:
            raw = input("State > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if raw == 'q':
            break

        if raw == 'r':
            print(f"  Re-centering to {v_to_nm(V_CENTER)/1000:.3f} µm")
            current_voltage = V_CENTER
            stream_ao(np.full(10, V_CENTER))
            continue

        if raw not in STATE_MAP:
            print("  Enter 0–3 (or 00/01/10/11), 'r' to re-center, 'q' to quit.")
            continue

        state = STATE_MAP[raw]
        rel_v = PROTOCOLS_3S[state] if (default_duration == 3.0 and state in PROTOCOLS_3S) \
                else make_protocol(default_duration, state)
        abs_v = clamp(current_voltage + rel_v)

        print(f"\n  Protocol {state:02b}  {default_duration}s  "
              f"{v_to_nm(abs_v[0]):.1f} → {v_to_nm(abs_v[-1]):.1f} nm")

        # Capture the AI sample index the instant AO hardware starts outputting.
        # on_start fires immediately after ao.start() inside _ao_hw — the most
        # accurate point we can observe from software.
        ao_start_idx: list[int | None] = [None]

        def on_start():
            elapsed = time.perf_counter() - session_t0[0]
            ao_start_idx[0] = int(elapsed * RATE)

        stream_ao(abs_v, on_start=on_start)

        if ao_start_idx[0] is not None:
            ao_segments.append((ao_start_idx[0], abs_v.copy()))
            labels.append(f"state {state:02b}")

        current_voltage = abs_v[-1]
        print(f"  Stage at {v_to_nm(current_voltage)/1000:.3f} µm\n")

    # ── Stop AI reader and plot ────────────────────────────────────────────
    ai_stop.set()
    ai_thread.join(timeout=3.0)

    ai_array = np.array(ai_samples)
    print(f"\nSession complete — {len(ao_segments)} protocol(s), "
          f"{len(ai_array)} AI samples ({len(ai_array)/RATE:.1f} s).")

    plot_session(ai_array, ao_segments, labels)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "━" * 60)
    print(" PIEZO DAQ — AO0 CONTROL + AI3 LOOPBACK VERIFICATION")
    print(f" Mode  : {'MOCK (no hardware)' if MOCK else 'HARDWARE'}")
    print(f" Device: {DEVICE}   AO: {AO_CHAN}   AI: {AI_CHAN}")
    print("━" * 60)

    # Loopback mode — AO0 wired to AI3:
    manual_loop(default_duration=3.0)

    # Camera-triggered live experiment (after loopback verified):
    # main_loop(protocol_duration=3.0)

"""
sine_loopback_test.py
=====================
Send a sine wave on AO0 and read it back on AI3 (differential).
Plots the sent vs received waveform and the error.

Wiring:
    AO1  →  AI3+  (positive terminal)
    AGND →  AI3-  (negative terminal, differential pair = AI7 on USB-6003)
"""

import time
import threading
import numpy as np
import matplotlib.pyplot as plt

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE    = "Dev4"
AO_CHAN   = f"{DEVICE}/ao1"
AI_CHAN   = f"{DEVICE}/ai3"

RATE      = 1000        # S/s
FREQ      = 2.0         # Hz  — sine frequency
DURATION  = 5.0         # s   — total output duration
AMPLITUDE = 1.5         # V   — sine amplitude
OFFSET    = 5.0         # V   — DC offset keeps output in AO safe range

# ── BUILD WAVEFORM ────────────────────────────────────────────────────────────
n  = int(DURATION * RATE)
t  = np.linspace(0, DURATION, n, endpoint=False)
ao_waveform = OFFSET + AMPLITUDE * np.sin(2 * np.pi * FREQ * t)

print(f"Sine wave: {FREQ} Hz,  ±{AMPLITUDE} V around {OFFSET} V")
print(f"Output range: [{ao_waveform.min():.2f}, {ao_waveform.max():.2f}] V")
print(f"Samples: {n}  ({DURATION} s at {RATE} S/s)\n")

# ── CONTINUOUS AI3 READER (runs from before AO starts until after it ends) ───
ai_samples : list[float] = []
ai_stop    = threading.Event()
ai_t0      : list[float | None] = [None]    # wall-clock time at AI hardware start

def _ai_reader():
    READ_N = 100
    with nidaqmx.Task() as ai:
        ai.ai_channels.add_ai_voltage_chan(
            AI_CHAN,
            # RSE: signal on AI3 pin, return wire to AIGND terminal (most common)
            # DIFF: signal on AI3 pin, return wire to AI7 pin (not AIGND terminal)
            terminal_config=TerminalConfiguration.RSE,
            min_val=0.0,
            max_val=10.0,
        )
        ai.timing.cfg_samp_clk_timing(
            rate=RATE, sample_mode=AcquisitionType.CONTINUOUS
        )
        ai.start()
        ai_t0[0] = time.perf_counter()     # t = 0 of the AI sample stream

        while not ai_stop.is_set():
            avail = ai.in_stream.avail_samp_per_chan
            if avail >= READ_N:
                chunk = ai.read(number_of_samples_per_channel=READ_N, timeout=1.0)
                ai_samples.extend(chunk)
            else:
                time.sleep(READ_N / RATE / 4)

        # Drain remaining hardware buffer
        avail = ai.in_stream.avail_samp_per_chan
        if avail > 0:
            ai_samples.extend(
                ai.read(number_of_samples_per_channel=avail, timeout=2.0)
            )

ai_thread = threading.Thread(target=_ai_reader, daemon=True)
ai_thread.start()

# Wait until AI hardware is actually running
while ai_t0[0] is None:
    time.sleep(0.005)

print("AI3 recording started. Sending sine wave on AO1 ...")

# ── OUTPUT SINE ON AO1 ────────────────────────────────────────────────────────
ao_start_idx : list[int | None] = [None]

with nidaqmx.Task() as ao:
    ao.ao_channels.add_ao_voltage_chan(
        AO_CHAN,
        min_val=OFFSET - AMPLITUDE - 0.1,
        max_val=OFFSET + AMPLITUDE + 0.1,
    )
    ao.timing.cfg_samp_clk_timing(
        rate=RATE,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=n,
    )
    ao.write(ao_waveform, auto_start=False)
    ao.start()

    # Timestamp immediately after ao.start() — this is when the first AO sample
    # hits the wire, so it maps to ai_t0 + ao_start_idx / RATE.
    ao_start_idx[0] = int((time.perf_counter() - ai_t0[0]) * RATE)

    ao.wait_until_done(timeout=DURATION + 5.0)

print("AO1 done. Stopping AI3 reader ...")

# ── STOP AI AND COLLECT ───────────────────────────────────────────────────────
ai_stop.set()
ai_thread.join(timeout=3.0)

ai_array = np.array(ai_samples)
print(f"AI3 samples collected: {len(ai_array)}  ({len(ai_array)/RATE:.2f} s)\n")

# ── EXTRACT ALIGNED WINDOW ────────────────────────────────────────────────────
idx0 = ao_start_idx[0] or 0
idx1 = idx0 + n
ai_window = ai_array[idx0 : min(idx1, len(ai_array))]
ao_window = ao_waveform[: len(ai_window)]
t_window  = np.arange(len(ai_window)) / RATE

# ── STATS ─────────────────────────────────────────────────────────────────────
err    = ai_window - ao_window
rms_mv = np.sqrt(np.mean(err ** 2)) * 1000
peak_mv = np.max(np.abs(err)) * 1000
print(f"RMS error  : {rms_mv:.2f} mV")
print(f"Peak error : {peak_mv:.2f} mV")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle(
    f"Sine loopback: AO1 → AI3 (diff)   "
    f"{FREQ} Hz  ±{AMPLITUDE} V  {RATE} S/s\n"
    f"RMS err = {rms_mv:.2f} mV   Peak err = {peak_mv:.2f} mV"
)

axes[0].plot(t_window, ao_window, color="steelblue", linewidth=1.2, label="AO1 sent")
axes[0].plot(t_window, ai_window, color="tomato",    linewidth=1.0,
             linestyle="--", alpha=0.9, label="AI3 read (diff)")
axes[0].set_ylabel("Voltage (V)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_window, err * 1000, color="darkorchid", linewidth=0.8)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_ylabel("Error  AI3 − AO1  (mV)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
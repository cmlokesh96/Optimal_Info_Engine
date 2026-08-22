# hw_protocol_test.py — Full Code Explanation

## Purpose

This script lets you manually trigger motion protocols on a **Piezoconcept LFHS3 piezo stage** via a **NI USB-6003 DAQ**, and simultaneously verify what the stage actually received by reading the output back on an analog input channel. It is a **loopback test** — AO1 (the output) is physically wired to AI3 (the input) so you can compare what was sent vs. what was received before you ever connect the real stage.

---

## Hardware Setup

```
NI USB-6003
┌─────────────────────────┐
│  AO1  ───────────────────┼──► AI3+  (loopback wire)
│  AGND ───────────────────┼──► AI7   (differential negative pair)
└─────────────────────────┘
```

### Why AI7 for the negative terminal?
The USB-6003 uses **differential (DIFF) input mode** on AI3. In DIFF mode, the DAQ measures the voltage difference between two pins:
- **AI3** = positive terminal
- **AI7** = negative terminal (hardwired pairing on USB-6003)

If you connect the return wire to AIGND instead of AI7, differential mode will not work correctly — the reading will be noisy or wrong.

---

## Constants (CONFIG block)

```python
DEVICE      = "Dev4"         # NI-DAQmx device name assigned by NI MAX
AO_CHAN     = "Dev4/ao1"     # Analog Output channel 1
AI_CHAN     = "Dev4/ai3"     # Analog Input channel 3

RATE        = 1000           # Sample rate: 1000 samples per second (1 kHz)
DURATION_S  = 3.0            # Default protocol length: 3 seconds

V_MIN       = 1.5            # Minimum safe voltage for the stage (= 4.5 µm)
V_MAX       = 8.5            # Maximum safe voltage for the stage (= 25.5 µm)
V_CENTER    = 5.0            # Center/resting voltage (= 15 µm)
NM_PER_VOLT = 3000.0         # Stage calibration: 3000 nm per volt (30 µm / 10 V)
```

### Voltage ↔ Position conversion

The stage moves **3 µm per volt** (3000 nm/V). The full working range is:

| Voltage | Position |
|---------|----------|
| 1.5 V   | 4.5 µm   |
| 5.0 V   | 15 µm (center) |
| 8.5 V   | 25.5 µm  |

The script clips all output to `[V_MIN, V_MAX]` to prevent damaging the stage.

---

## `make_protocol(duration_s, state)` — Protocol Waveform Builder

```python
def make_protocol(duration_s: float, state: int) -> np.ndarray:
    n = int(duration_s * RATE)          # number of samples (e.g. 3.0s × 1000 = 3000)
    t = np.linspace(0, duration_s, n)   # time array [0, 3.0] with 3000 points
    ...
    return rel_nm / NM_PER_VOLT         # returns RELATIVE volts
```

**Important:** This function returns **relative displacement in volts from the current stage position**, NOT an absolute voltage. The caller is responsible for adding `current_voltage` to get the real output. This is what allows the stage to hold its position between protocols.

### The 4 Protocol States

States are encoded as 2-bit binary numbers (00, 01, 10, 11):

#### State `00` — Hold
```python
rel_nm = np.zeros(n)
```
Stage does not move. All samples are 0 nm displacement → adds 0 V to current position. The stage holds wherever it is.

#### State `01` — Positive Ramp (+25 nm → +120 nm)
```python
rel_nm = np.concatenate([[25.0], np.linspace(25.0, 120.0, n - 1)])
```
Stage jumps to +25 nm immediately (sample 0), then linearly ramps up to +120 nm over the full duration.

#### State `10` — Negative Ramp (−25 nm → +50 nm)
```python
rel_nm = np.concatenate([[-25.0], np.linspace(-25.0, 50.0, n - 1)])
```
Stage jumps to −25 nm immediately (sample 0), then linearly ramps up to +50 nm over the full duration. Net movement crosses zero.

#### State `11` — Exponential Rise (+0 nm → +150 nm)
```python
tau    = duration_s / 3.0
rel_nm = 150.0 * (1 - np.exp(-t / tau))
rel_nm[0] = 0.0
```
Stage follows an exponential approach curve toward +150 nm. `tau` is the time constant (= 1 second for a 3-second protocol). After 3τ = 3s, the stage reaches ~95% of 150 nm.

The first sample is forced to 0.0 to avoid a sudden jump at t=0 (the formula gives a non-zero value when evaluated at t=0 due to floating point).

---

## `run_loopback(state, current_v, duration_s)` — Single Protocol Run

This function:
1. Builds the absolute voltage waveform anchored to the current stage position
2. Sends it on AO1
3. Records AI3 simultaneously
4. Aligns AO and AI in time using cross-correlation
5. Returns both arrays for comparison

### Step 1: Build the absolute waveform
```python
waveform = np.clip(current_v + make_protocol(duration_s, state), V_MIN, V_MAX)
```
`current_v` is where the stage is right now. The relative protocol is added to it. `np.clip` enforces the safe voltage limits — any sample outside `[1.5V, 8.5V]` is clamped.

### Step 2: Set up the recording window
```python
n      = len(waveform)          # e.g. 3000 samples
MARGIN = int(0.5 * RATE)        # 500 samples = 500 ms extra on each side
n_ai   = n + 2 * MARGIN         # 3000 + 1000 = 4000 samples recorded by AI
```
AI records **500 ms before and after** the AO waveform. This extra window is needed because the AO and AI tasks start at slightly different times (see Synchronisation section).

### Step 3: Configure and run both tasks
```python
with nidaqmx.Task() as ao, nidaqmx.Task() as ai:
    # AO: FINITE mode, write all 3000 samples at once
    ao.timing.cfg_samp_clk_timing(rate=RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
    ao.write(waveform, auto_start=False)

    # AI: DIFF FINITE, reads 4000 samples
    ai.ai_channels.add_ai_voltage_chan(AI_CHAN, terminal_config=TerminalConfiguration.DIFF, ...)
    ai.timing.cfg_samp_clk_timing(rate=RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n_ai)

    ai.start()   # AI arms first
    ao.start()   # AO starts a few ms later
```

**Why `auto_start=False`?** The full waveform is written to the software buffer before `ao.start()` is called. This lets the driver pre-fill the hardware FIFO before any samples are sent, preventing glitches.

**Why FINITE mode?** In FINITE mode, the driver knows exactly how many samples to generate and stops cleanly. The nidaqmx software buffer is much larger than the 2047-sample hardware FIFO — the driver silently refills the hardware FIFO as it drains, so you can write tens of thousands of samples in a single `ao.write()` call without any manual chunking.

### Step 4: Cross-correlation alignment

```python
ao_ac = waveform - waveform.mean()   # remove DC offset
ai_ac = ai_raw   - ai_raw.mean()

corr   = np.correlate(ai_ac, ao_ac, mode='valid')  # 2*MARGIN+1 = 1001 values
offset = int(np.argmax(corr))
```

Because the USB-6003 has no hardware trigger routing between AO and AI, the two tasks start at slightly different times (a few milliseconds apart). Cross-correlation finds the time offset that makes AO and AI match best.

- `mode='valid'` slides `ao_ac` (shorter) across `ai_ac` (longer), producing `n_ai - n + 1 = 1001` correlation values — one for each candidate offset within the ±500 ms window
- `argmax` finds which offset produces the best match
- The result is used to slice `ai_raw` so it aligns with `waveform`

**State 00 is skipped** — a flat (all-zero-AC) waveform has no features to correlate. The offset is fixed at `MARGIN` (the center of the search window), which means "assume zero delay."

```python
delay_ms = (offset - MARGIN) / RATE * 1000
```
If `offset == MARGIN`, delay = 0 ms (perfect alignment). If `offset > MARGIN`, AI arrived earlier than AO; if `offset < MARGIN`, AO arrived earlier. Typical values on USB-6003 are a few milliseconds.

---

## `STATE_NAMES` and `STATE_MAP` — Input Parsing

```python
STATE_MAP = {
    '0': 0b00, '00': 0b00,   # typing "0" or "00" → state 00 (hold)
    '1': 0b01, '01': 0b01,   # typing "1" or "01" → state 01 (+ramp)
    '2': 0b10, '10': 0b10,   # typing "2" or "10" → state 10 (-ramp)
    '3': 0b11, '11': 0b11,   # typing "3" or "11" → state 11 (exp)
}
```

Accepts both decimal shorthand (`0`, `1`, `2`, `3`) and binary notation (`00`, `01`, `10`, `11`). Both forms refer to the same protocol.

---

## `plot_session(ao_log, ai_log, labels)` — Session Plot

Called at the end of the session. Takes the lists of AO and AI arrays from every protocol that was run.

```
┌─────────────────────────────────────────────────┐
│  Top panel:  AO1 sent (blue) vs AI3 read (red)  │
│              Voltage in V vs time in seconds     │
├─────────────────────────────────────────────────┤
│  Bottom panel: Error = AI3 − AO1 in millivolts  │
└─────────────────────────────────────────────────┘
```

Each protocol segment gets:
- A blue shaded background band
- A dashed gray vertical line at its start
- A small label at the top (e.g. `s01`, `s11`)

All segments are concatenated end-to-end on the time axis (no gaps between them — the plot assumes protocols ran back-to-back, which is what the loopback produces).

---

## `manual_loop(default_duration)` — Interactive Session

The main interactive loop. Keeps the DAQ open for as long as you want to run protocols, one at a time.

### Stage position tracking

```python
current_voltage: float = V_CENTER   # starts at 5.0 V (15 µm)
```

Every time a protocol finishes:
```python
current_voltage = float(ao_w[-1])   # last sample of the waveform = where stage is now
```

This ensures the next protocol starts from where the previous one ended, so the stage never jumps unexpectedly.

### Session flow

```
python hw_protocol_test.py
↓
Buffer info printed
↓
Manual loop starts — current_voltage = V_CENTER = 5.0 V
↓
┌─────────────────────────────────────────────────────┐
│  State > _   ← you type here                       │
│                                                     │
│  0/00 → hold (stage stays put, 3 s)                │
│  1/01 → +ramp (25→120 nm above current, 3 s)       │
│  2/10 → -ramp (-25→+50 nm from current, 3 s)       │
│  3/11 → exp rise (0→150 nm from current, 3 s)      │
│  r    → force stage back to 5.0 V center            │
│  q    → quit and show plot                          │
└─────────────────────────────────────────────────────┘
↓
Each keystroke triggers run_loopback() → DAQ runs → stats printed
↓
On 'q': plot_session() shows all recorded data
```

### Re-center command (`r`)
```python
ao.write(np.full(10, V_CENTER), auto_start=True)
current_voltage = V_CENTER
```
Sends 10 samples of 5.0 V (10 ms at 1 kHz) to gently move the stage to center, then resets the position tracker. Use this if protocols have walked the stage far from center and you want a fresh start.

### Error / interrupt handling
```python
except (EOFError, KeyboardInterrupt):
    print("\nInterrupted.")
    break
```
If you press **Ctrl+C** during the input prompt, the loop exits cleanly and calls `plot_session()` with whatever data has been collected so far.

---

## Entry Point (`__main__`)

```python
if __name__ == "__main__":
    # 1. Open AO task briefly just to read the software buffer size
    with nidaqmx.Task() as _ao:
        ...
        sw_buf = _ao.out_stream.output_buf_size
    print(f" HW FIFO: 2047 samples  |  SW buffer: {sw_buf} samples ...")

    # 2. Start the interactive session
    manual_loop(default_duration=DURATION_S)
```

The buffer size query is informational — it confirms the driver is allocating a software buffer much larger than the 2047-sample hardware FIFO, which is why large waveforms can be written in a single `ao.write()` call.

---

## Key Concepts Summary

### Why a software buffer?
The USB-6003 has a 2047-sample hardware FIFO (a small chip-level buffer). The nidaqmx driver maintains a larger software buffer in RAM. When you call `ao.write(full_waveform)`, the entire waveform goes into the software buffer. The driver's background thread then streams chunks from the software buffer into the hardware FIFO as space becomes available. You never need to manually chunk the data.

### Why cross-correlation instead of hardware trigger?
The USB-6003 cannot internally route the AO start trigger to the AI task — this is a hardware limitation of the device. The workaround is:
1. Start AI slightly before AO (software back-to-back)
2. Record extra samples on both sides (the ±500 ms MARGIN)
3. Cross-correlate to find the true time offset
4. Slice the AI array to align it with the AO array

### Why DIFF mode on AI3?
Differential mode rejects common-mode noise (interference picked up equally on both wires). It is more noise-resistant than RSE (referenced single-ended) mode. In DIFF mode, the measurement is `AI3_pin − AI7_pin`, so any noise coupling onto both wires cancels out.

### Voltage → position calculation
```
position (nm) = (voltage − 0) × NM_PER_VOLT
position (µm) = voltage × 3.0          (3 µm per volt)
```
At `V_CENTER = 5.0 V`: position = 15.0 µm
At `V_MIN = 1.5 V`:    position = 4.5 µm
At `V_MAX = 8.5 V`:    position = 25.5 µm

---

## Typical Console Output

```
───────────────────────────────────────────────────────
 AO1 → AI3 loopback  (USB-6003, DIFF, xcorr-aligned)
───────────────────────────────────────────────────────
 HW FIFO: 2047 samples  |  SW buffer: 16383 samples (16.4 s)

─────────────────────────────────────────────────────
 Manual loop  —  AO1 → AI3  (DIFF, xcorr-aligned)
 Duration: 3.0 s per protocol
 Stage starting at 5.000 V  (+0.0 nm from center)
─────────────────────────────────────────────────────
  0/00 hold  |  1/01 +ramp  |  2/10 -ramp  |  3/11 exp
  r  re-center   |   q  quit + plot

State > 1

  State 01 — +ramp  (from 5.000 V)
    delay correction: +3.2 ms  (+3 samples)
    AO1: [5.0083, 5.0400] V  → now at 5.0400 V
    AI3: [5.0089, 5.0401] V
    RMS=1.23 mV   Peak=3.45 mV

State > q

Session: 1 protocol(s) recorded.
[plot appears]
```

---

## File Structure at a Glance

| Section | Lines | What it does |
|---------|-------|--------------|
| Config constants | 26–37 | Device name, channel names, rate, voltage limits |
| `make_protocol()` | 40–55 | Returns relative voltage waveform for each state |
| `run_loopback()` | 58–115 | Runs one DAQ round-trip and aligns AO/AI |
| `STATE_NAMES`, `STATE_MAP` | 117–129 | Input parsing dictionaries |
| `plot_session()` | 132–171 | End-of-session matplotlib plot |
| `manual_loop()` | 174–241 | Interactive command loop + position tracking |
| `__main__` | 244–259 | Buffer info + launches `manual_loop()` |

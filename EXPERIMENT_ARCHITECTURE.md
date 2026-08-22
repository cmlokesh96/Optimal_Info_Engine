# Optimal Info Engine — Experiment Architecture

## Purpose of this document

Complete context for the two-channel (`ao0` fine + `ao1` coarse) piezo-stage
information-engine control system, and a status/handoff log for continuing
this project in a fresh conversation. Covers hardware, class design, the
9-state protocol scheme, session flow, data structures, what's been
validated on real hardware, and what's still open.

Working directory: `d:\Optimal_Info_Engine\`

This document replaces an earlier version that described a single-channel
(`ao1`-only, 4-state) predecessor system. That system's code has been moved
to `temp\` (see [File Structure](#file-structure)) — it is not part of the
current architecture.

---

## Hardware Inventory

| Device | Role | Interface |
|--------|------|-----------|
| NI USB-6003 (Dev4) | DAQ — `ao0`/`ao1` piezo output, 4-channel AI recording | USB, `nidaqmx` |
| Piezoconcept LFHS3 | Piezo stage — driven by `ao0/10 + ao1` through an analog adder circuit | Analog |
| Basler camera | Image acquisition + real-time particle tracking | GigE Vision, `pypylon` |
| Tektronix AFG3000 (camera-trigger unit) | Function generator — camera hardware-trigger pulse train | USB-VISA, `pyvisa` |
| Tektronix AFG3000 (AOM/laser-power unit) | Second, physically separate function generator — sets laser power via the AOM | USB-VISA, `pyvisa` |

The two Tektronix units are **different physical devices** with different
USB-VISA serial numbers — not two channels of the same box. See
[Two Function Generators](#two-function-generators).

---

## NI USB-6003 — Key Specs

- **`ao0`** (`Dev4/ao0`): fine/correction channel, raw range **-10..10 V**,
  attenuated **/10** by the adder circuit before summing — this is what
  gives it higher effective positioning resolution than `ao1` alone.
- **`ao1`** (`Dev4/ao1`): coarse/offset channel, kept in the stage's
  **1.5..8.5 V** range.
- Combined physical output driving the stage amplifier: **`ao0/10 + ao1`**.
- **AI channels**: 4 channels recorded continuously every session (see
  below) — 16-bit, up to 100 kS/s aggregate.
- **Software buffer**: `nidaqmx` driver refills the hardware FIFO
  automatically; FINITE-mode AO writes the entire waveform at once.
- **`AOIdleOutputBehavior`** (hold-last-value vs. reset-to-zero on task
  stop) is **not supported/settable on this device** — confirmed directly
  (`ch.ao_idle_output_behavior` raises "Specified property is not
  supported"). No software control over this; whatever the hardware does
  by default is what happens.
- **Limitation** (unchanged from the single-channel system): no internal
  routing of `ao/SampleClock`/`ao/StartTrigger` to AI — no hardware-level
  sync between the AO and AI tasks. Synchronization is done post-hoc via
  `CamOut` rising edges (see [Post-hoc Synchronisation](#post-hoc-synchronisation-and-analysis)).

### Wiring

```
ao0, ao1  →  analog adder circuit  →  ao0/10 + ao1  →  stage amplifier input
                                              │
                                    (T-connector)
                                              │
                                        AI3+ (AdderOut)   ← commanded signal, same node as above
                                        AGND → AI7          (DIFF negative pair for AI3 on USB-6003)

stage amplifier output  →  Piezoconcept LFHS3 stage
                        →  AI1+ (Stage)   ← the amplifier's ACTUAL output (DIFF)

Port0/Line0  → Camera-trigger function generator's external trigger input

AI0  (RSE)   — AOM voltage (laser power, from the AOM function generator)
AI1  (DIFF)  — Stage — amplifier output (see note below)
AI2  (RSE)   — CamOut — camera Line3 ExposureActive strobe
AI3  (DIFF)  — AdderOut — ao0/10 + ao1, commanded signal into the amplifier
```

**`ai1` (Stage) vs. `ai3` (AdderOut) are on different sides of the stage
amplifier and are NOT expected to read the same voltage**, even at rest —
the amplifier has its own gain/offset. This distinction cost real debugging
time (see [Lessons From Calibration](#lessons-from-calibration-testing)):
anywhere new code needs to know "what am I currently commanding," read
`ai3`; anywhere it needs "where is the stage physically," read `ai1`.

### Voltage ↔ Position (Piezoconcept LFHS3)
```
Calibration (assumed, not yet independently re-verified — see Pending Work):
NM_PER_VOLT = 3000 nm / V  (30 µm / 10 V), applies to ao1/Stage directly.
ao0's effective calibration is NM_PER_VOLT/10 (≈300 nm/V), since it's
attenuated /10 by the adder before summing.

V_MIN    = 1.5 V  →   4.5 µm
V_CENTER = 5.0 V  →  15.0 µm  (resting)
V_MAX    = 8.5 V  →  25.5 µm

Live safety bound during protocol firing (SessionParams2Ch.stage_min_v/
stage_max_v, ±10 µm around center — narrower than the hardware limits
above): 1.667 V .. 8.333 V (≈5..25 µm). Does not apply to the deliberate
shutdown ramp to 0 V.
```

---

## Two Function Generators

Both are Tektronix AFG3000 units, both driven the same way (`pyvisa` SCPI
over USB-VISA), both wrapped by the **same** `FunctionGenerator` class in
`function_gen.py` — you just open two instances, one per physical unit's
VISA ID.

### Camera-trigger unit
**USB ID**: `USB0::0x0699::0x0358::C018956::INSTR`
**Role**: Channel 2 — pulse train at `fps`, hardware-triggers the Basler
camera. Configured once per session via `setup_camera_trigger(fps,
exposure_us, n_frames)` (`n_frames = int(fps * trecord_s)`, burst mode).
A single digital pulse from `NICardOutDual.hardware_trigger()` (Port0/
Line0) starts the whole burst.

### AOM/laser-power unit
**USB ID**: `USB0::0x0699::0x0358::C020332::INSTR`
**Role**: Sets laser power (via the AOM) as a DC level. Channel 1's old
role (a pulse/burst signal, historically used to help control the stage
before the DAQ took over that job) is obsolete and not implemented in
Python — only the DC/AOM-power block was ported.

```python
class FunctionGenerator:
    def __init__(self, usb_id=USB_ID)
    def setup_camera_trigger(self, fps, exposure_us, n_frames, ch=2)
        # Camera-trigger unit only.
    def set_dc_level(self, voltage, ch=2)
        # AOM unit — Output Off -> Waveform=DC -> Low/High level
        # (voltage +/- 0.1V) -> Output On. Mirrors the original MATLAB
        # FunctionGen_shutter_stage AOM-power block exactly.
    def output_on(self, ch=2)
    def output_off(self, ch=2)
    def close(self)
```

**AOM safety usage** — a stronger trap (near-max AOM voltage) is used
while the stage is physically moving, to reduce the chance of losing the
particle in transit; it's lowered to the experiment-appropriate level once
settled. Built into `ensure_stage_at()` (see below) as opt-in params
(`aom_gen`, `aom_transit_v`, `aom_experiment_v`, `aom_post_ramp_wait_s`),
threaded through to `run_batch_2ch()`'s between-experiment resets too.
Sequence when `aom_gen` is given and the stage actually needs to move: set
AOM to `aom_transit_v` → ramp → arrive → wait
`aom_post_ramp_wait_s` (default 10 s) → set AOM to `aom_experiment_v` →
*then* the usual settle countdown.

---

## Basler Camera

**Python library**: `pypylon` (Basler Pylon SDK)

### Hardware lines
```
Line 2 (Input)  — Hardware trigger input, rising edge from the camera-trigger funcgen
Line 3 (Output) — ExposureActive strobe, inverted → feeds AI2 (CamOut channel)
```

### Python class — `camera.py`
```python
class BaslerCamera:
    def __init__(self, exposure_us=500.0)
    def exposure(self, value)
    def gamma(self, value)
    def framerate(self, fps)
    def gain(self, gain_db)
    def roi(self, x_offset, y_offset, width, height)
    def full_roi(self)
    def pixel_format(self, fmt="Mono8")

    def hardwareTrigg(self)      # Line2 input, rising edge
    def softwareTrigg(self)      # free-run, for preview/calibration
    def startview(self, particle_cross=False)   # live cv2 preview, own grab loop
    def stopview(self)

    def start(self)              # StartGrabbing
    def stop(self)
    def close(self)

    def grab(self, timeout_ms=2000) -> np.ndarray
        # HOT PATH. Always raises RuntimeError on failure (a missed
        # trigger surfaces as genicam.TimeoutException from
        # RetrieveResult itself — caught and re-raised here so every
        # caller only ever needs to catch one exception type).

    def find(self, frame, nm_per_px=NM_PER_PX, roi=96, thres=THRES) -> (x_nm, y_nm)
        # Raw sub-pixel position via find_particle_fast() — NOT centered
        # on anything (not the ROI, not the trap); (nan, nan) if nothing
        # found above threshold. Centering (subtract a calibrated trap
        # center) is the caller's job, done once downstream, not here.
        # Keeps internal windowed-search state (find_particle_fast's
        # `prev`) so consecutive calls only blur a small region around
        # the last detection — much faster than a full-frame search.

    def show(self, frame, x_nm=None, y_nm=None)
```

`find_particle_fast(im, prev=None, roi=96, thres=THRES, matlab_indexing=True)`
is a module-level function `find()` calls internally — windowed Gaussian-
kernel search using the previous detection's position, with automatic
full-frame fallback if the result lands on the edge of the search window
(drifted out) or `prev` isn't available (first call, or after a loss).

---

## Protocol States — 9-state, two-timestep decode

Two consecutive thresholded camera samples `x(t-δt)`, `x(t)`, each in
`{-1, 0, 1}`, form a 9-state pair. `decode_symbol()`/`decode_state_pair()`
in `protocols_2ch.py`:

```python
def decode_symbol(x_nm, xth_nm) -> int          # threshold into {-1, 0, 1}
def decode_state_pair(x_prev_nm, x_curr_nm, xth_nm) -> tuple[int, int]
```

- `(0, 0)` — no-op: no protocol fires, wait `2·δt` and resample.
- The other 8 states form 4 mirror-symmetric pairs — only 4 base protocol
  shapes are stored (`BASE_PROTOCOLS`, currently **placeholders** —
  simple ramps/exponentials, not yet real optimized shapes); the mirrored
  group negates the matching base shape:

| Group A (stored) | Group B (mirror = negated A) |
|-------------------|-------------------------------|
| `(1,1)` — exp | `(-1,-1)` — mirror exp |
| `(1,0)` — -ramp | `(-1,0)` — mirror -ramp |
| `(0,1)` — +ramp | `(0,-1)` — mirror +ramp |
| `(-1,1)` — -jump-exp | `(1,-1)` — mirror -jump-exp |

`get_protocol_nm(state, duration_s, rate)` returns the **relative
displacement in nm** for a state (zeros for `(0,0)`).

### Channel split — only `ao0` fires

```python
def split_channels(waveform_nm, ao1_offset_v,
                   ao0_min=AO0_MIN, ao0_max=AO0_MAX,
                   ao1_min=AO1_MIN, ao1_max=AO1_MAX
                   ) -> (ao0_waveform_v, ao1_waveform_v, ao1_new_offset_v)
```

`ao0` carries the actual excursion for the protocol's duration, then resets
to 0 on the **last sample**; `ao1` holds `ao1_offset_v` constant for the
whole run and only changes on that same last sample, absorbing the net
displacement — so `ao0/10 + ao1` stays continuous across the reset, and the
next fire starts from `ao0 = 0` at the new `ao1` baseline. **`ao0`'s
waveform depends only on `(state, duration_s, rate)`** — never on the live
`ao1` offset — confirmed by reading `split_channels()`, and the direct
motivation for the still-pending file-backed-protocol work (see
[Pending Work](#pending-work)): the same 8 non-`(0,0)` waveforms can be
loaded from file once and reused for every fire, only the `ao1` offset
changes call to call, tracked purely as Python state (never re-read from
hardware mid-session — see [Lessons](#lessons-from-calibration-testing)).

### Manual test menu (`STATE_MENU`, for REPL-style hardware tests)
```
0:(1,1)  1:(1,0)  2:(0,1)  3:(-1,1)  4:(-1,-1)  5:(-1,0)  6:(0,-1)  7:(1,-1)  8:(0,0)
```

---

## Session Data Structures

### `pos` — (n_frames, 4) float64, built frame by frame

| Column | Name | Description |
|--------|------|-------------|
| 0 | `time_s` | Seconds from session start |
| 1 | `x_nm` | Particle x position (nm), trap-center-relative |
| 2 | `y_nm` | Particle y position (nm), trap-center-relative |
| 3 | `trigger` | 0 = not a checkpoint frame; else `STATE_TO_CODE` (1-9) or `CAPTURE_CODE` (10) |

`STATE_TO_CODE`/`CODE_TO_STATE`/`CAPTURE_CODE` are in `protocols_2ch.py`.
Written on **every** checkpoint (decide *or* capture-only), so it can be
cross-checked directly against the `AdderOut` (`ai3`) trace.

### `DataSync` — (n_frames, 3) float64, built post-hoc — unchanged shape
from the single-channel system: `[time_s, stage_nm, aom_V]`, `stage_nm`
from the low-pass-filtered `Stage` (`ai1`) channel. `build_datasync()`
(`experiment.py`) is otherwise unchanged.

### `ai_data` — (4, n_ai_samples) float64 — continuous AI recording

Row order (`NICardIn.CHANNELS`, `daq.py`): **`AOM, Stage, CamOut,
AdderOut`** — one more channel than the single-channel system's 3
(`AdderOut` is new).

---

## File Structure

```
d:\Optimal_Info_Engine\
│
├── camera.py            ← BaslerCamera, find_particle_fast()
├── daq.py                ← NICardOutDual, NICardIn, read_stage_voltage(),
│                            read_ai_channel()
├── function_gen.py       ← FunctionGenerator (used for BOTH physical units)
├── protocols_2ch.py       ← Pure numpy, no hardware. 9-state decode/
│                            protocol logic, SessionParams2Ch
├── analytical_protocols.py ← load_analytical_protocols(): reads
│                            run_single_parameter_point.py's saved .npz
│                            files into protocols_2ch's (m_0,m_t) -> nm
│                            format, resampled to the AO grid, trap->stage
│                            sign negation applied. numpy-only, does not
│                            import sdesim (keeps the scipy/numba analytical
│                            venv decoupled from the hardware-control side).
├── helper.py              ← Internal diagnostic/plotting helpers for
│                            Main_test_script.ipynb sections 9/10 (latency
│                            stats, calibration-sweep tracking/fitting,
│                            ai1-vs-ai3 comparison, the 10d fire-session
│                            plot) — `from helper import *` (imported once,
│                            section 2) keeps those notebook sections short.
├── experiment.py          ← run_session_2ch(), run_batch_2ch(),
│                            ensure_stage_at(), calibrate_trap_center(),
│                            preview_particle(), plot_session_2ch(),
│                            build_datasync(), save_session()
├── analysis.py            ← compute_potential(), plot_potential() (kT units
│                            + physical N/m stiffness)
│
├── Main_test_script.ipynb ← the real, current experiment notebook —
│                            sections 1-10, see Session Flow below
│
├── EXPERIMENT_ARCHITECTURE.md   ← this file
├── CAMERA_INTERFACE.md          ← BaslerCamera usage examples (still accurate)
│
├── test_data\             ← recorded session output (untouched by this cleanup)
├── analytical_optimal_protocol_computation\
│                           ← SEPARATE work area: computing the real
│                             (non-placeholder) protocol shapes. Not yet
│                             integrated with the files above — see
│                             Pending Work.
│
└── temp\                  ← archived, superseded-by-the-above files (moved
                              here 2026-08-19, not deleted): the old single-
                              channel protocols.py/experiment.run_session()
                              stack and its docs, MockBaslerCamera-based
                              tests, and one-off/already-validated hardware
                              diagnostic scripts (hw_two_channel_test.py,
                              test_camera_2ch.py, run_batch_2ch_example.py,
                              sine_loopback_test.py, run_on_video.py,
                              track_compare.py, Ni_Daq_Single_Channel_Control.py).
                              Nothing in the active file list above imports
                              from anything in temp\.
```

**Note**: `protocols.py` (the old single-channel module) was a real,
load-bearing import of `daq.py`/`experiment.py` until this cleanup — both
now import `NM_PER_VOLT`/`V_CENTER` from `protocols_2ch.py` instead, and
the dead single-channel `run_session()`/`plot_session()` functions (which
needed the rest of `protocols.py`'s exports, and were already established
as never having actually run against real hardware) were deleted from
`experiment.py`. `protocols.py` has no remaining active dependents.

---

## Python Library Dependencies

| Purpose | Library |
|---------|---------|
| NI DAQ | `nidaqmx` |
| Function generators | `pyvisa` |
| Basler camera | `pypylon`, `opencv-python` (`cv2`) |
| Signal filtering | `scipy` |
| Numerics | `numpy` |
| Plotting | `matplotlib` |

---

## DAQ Class Interfaces — `daq.py`

### `NICardOutDual`
```python
class NICardOutDual:
    def __init__(self, n_samples, rate=1000,
                ao0_min=-10.0, ao0_max=10.0, ao1_min=1.5, ao1_max=8.5)
        # ONE task, both ao0 and ao1 added as channels, ONE shared sample
        # clock. Every write()/start() re-arms BOTH channels together —
        # there is no way to touch only one within a single task (a real
        # nidaqmx API constraint, not a choice — see Lessons).
        # DO task: Port0/Line0 hardware trigger, same as before.

    def load(self, ao0_waveform, ao1_waveform)   # stop + write, don't start
    def start(self)                              # non-blocking
    def wait_done(self, extra_s=5.0)
    def hardware_trigger(self)                   # pulse Port0/Line0
    def set_dc(self, ao0_v, ao1_v)                # hold both, blocking

    def ramp_ao1(self, from_v, to_v, speed_nm_per_s=10.0)
    def ramp_ao0(self, ao1_hold_v, from_v, to_v, speed_nm_per_s=10.0)
    def run_custom_waveform(self, ao0_wave, ao1_wave)
        # Generalizes the two ramp_*() methods to an arbitrary waveform
        # shape (e.g. a sine sweep) — all three share one internal
        # _run_waveform() helper (temporarily reconfigures the clock for
        # a longer/shorter run, then restores standard protocol-length
        # timing so load()/start() keep working unchanged afterward).

    def close(self)
```

### `NICardIn`
```python
class NICardIn:
    CHANNELS = [AOM (ai0,RSE), Stage (ai1,DIFF), CamOut (ai2,RSE),
               AdderOut (ai3,DIFF)]
    READ_N = 1000   # samples drained per poll

    def __init__(self, rate=3000)
    def start_continuous(self)      # resets accumulated data — safe to
                                    # call again on an already-used instance
                                    # (e.g. between run_batch_2ch experiments)
    def get_sample_count(self) -> int
        # CHUNK-QUANTIZED, not a true real-time counter — only advances
        # by READ_N (1000) AFTER a full chunk is drained. Querying before
        # one READ_N/rate has elapsed since start_continuous() always
        # returns 0. See Lessons — this bit real code in this session.
    def stop_and_get(self) -> np.ndarray   # (4, n_samples)
    def close(self)                        # safety net if stop_and_get()
                                           # was never reached
```

### Module-level
```python
def read_stage_voltage(rate=1000, n_samples=200) -> float
    # One-shot read of ai1 (Stage) — the amplifier's ACTUAL output.
def read_ai_channel(name, rate=1000, n_samples=200) -> float
    # Generalizes the above to any of the 4 channels by name.
    # read_stage_voltage() is now a thin wrapper: read_ai_channel("Stage").
```

---

## Session Flow — `run_session_2ch()`

Camera runs continuously at `fps` (hardware-triggered). Every frame's
`(x, y)` is recorded into `pos`. A separate checkpoint cadence walks
forward in frame-index space:

- `delta_frames = round(δt * fps)`, `relax_frames = round(T_relax_s * fps)`.
- **capture** checkpoint: record `x_prev` only, no decision.
- **decide** checkpoint: have `x_prev` and `x_curr` → `decode_state_pair()` → act.
- Normal flow: capture → (`delta_frames` later) → decide → capture → decide → ...
- On `(0,0)`, or any decoded state **not** in `active_states` (if given):
  no fire; next checkpoint is a fresh **capture** `delta_frames` later.
- On a real fire: next checkpoint is a **capture** `relax_frames` later
  (`T_relax_s` replaces `δt` for the first gap after a firing), then normal
  `δt` cadence resumes. Constraint (asserted in `SessionParams2Ch`):
  `T_relax_s >= protocol_dt_s`.
- `active_states=None` (default): every non-`(0,0)` state fires — full
  9-state engine. `active_states={...}`: decode still runs on the real
  trajectory always (so statistics stay valid for later thermodynamic-
  quantity calculations), but only fires when the real decoded state is in
  the set — used for single-protocol validation and equilibrium
  characterization (`active_states=set()` → never fires, pure decode).

```
SESSION START
├── funcgen.setup_camera_trigger(fps, exposure_us, n_frames)
├── camera.hardwareTrigg()
├── ai.start_continuous()
├── camera.start()
├── ao.hardware_trigger()      → funcgen fires → camera receives fps pulses
│
│   ┌── CHECKPOINT LOOP (n_frames iterations) ──────────────────────────┐
│   │   frame = camera.grab(timeout_ms=...)    (RuntimeError on miss)   │
│   │   x_nm, y_nm = camera.find(frame) - x_center_nm/y_center_nm       │
│   │   pos[frame_i] = [t, x_nm, y_nm, 0]                               │
│   │   at a checkpoint: capture x_prev, OR decide via                  │
│   │     decode_state_pair() → (if active) get_protocol_nm() →         │
│   │     split_channels() → ao.load() → ao.start() (non-blocking)      │
│   │   pos[frame_i, 3] = STATE_TO_CODE[state] or CAPTURE_CODE          │
│   └─────────────────────────────────────────────────────────────────┘
│
├── camera.stop(); ai_data = ai.stop_and_get()
└── returns {pos, ai_data, channel_names, params, ao1_final_v}
```

`run_batch_2ch(base_kwargs, experiment_overrides, ao, ai, camera, funcgen,
...)` runs N of these back-to-back, camera/DAQ staying open the whole time
— genuinely blocking/sequential, `ao1` carries continuously from one
experiment's end to the next's start unless `settle_s_between` is given
(then it resets to `reset_target_v` and re-settles between every pair,
with the same AOM safety sequence as `ensure_stage_at()` if `aom_gen` is
passed through).

`ensure_stage_at(ao, target_v=V_CENTER, speed_nm_per_s=10.0, settle_s=300.0,
...)` — independent safety utility, not run automatically by anything: no
bare voltage steps, ever ("we lose the particle" if you jump it). Only
moves/settles if not already within `tol_v` of `target_v`.

---

## Post-hoc Synchronisation and Analysis

Unchanged mechanism from the single-channel system — `CamOut` (`ai2`)
rising edges give the exact AI sample index for every camera frame,
hardware-accurate, no software-timestamp jitter:

```python
DataSync, tvec = build_datasync(result)
# tvec[i] = AI sample index where camera frame i was captured
trigger_frames = np.where((pos[:, 3] >= 1) & (pos[:, 3] <= 9))[0]  # real fires only
```

---

## `SessionParams2Ch` — `protocols_2ch.py`

```python
@dataclass
class SessionParams2Ch:
    trecord_s     : float
    fps           : float
    exposure_us   : float
    xth_nm        : float = 50.0    # decode_symbol() threshold
    delta_t_s     : float = 1.0     # spacing between decision samples
    T_relax_s     : float = 10.0    # wait after firing before resuming
    protocol_dt_s : float = 3.0     # duration of a fired protocol waveform
    x_center_nm   : float = 0.0     # trap center (raw camera.find() units)
    y_center_nm   : float = 0.0
    stage_min_v   : float = 1.667   # ao1 safety bound, live firing only
    stage_max_v   : float = 8.333
    print_every   : int   = 1       # throttle no-fire checkpoint logging
    print_no_fire : bool  = True
    ai_rate       : int   = 3000
    ao_rate       : int   = 1000
    savepath      : str   = ""

    # Derived: n_frames, delta_frames, relax_frames
    # __post_init__ asserts T_relax_s >= protocol_dt_s
```

---

## Trap Characterization — `analysis.py`

```python
def compute_potential(x_nm, n_bins=60, T_K=298.0) -> dict
    # Boltzmann-inverts x(t) into U(x)/kT (kT units, no temperature
    # needed for this part) via histogram + parabolic fit near the
    # minimum. Returns k_fit_over_kT and k_eq_over_kT (equipartition,
    # 1/Var(x)) as a cross-check, PLUS k_fit_N_per_m/k_eq_N_per_m (only
    # these use T_K). Run on a real equilibrium stretch, e.g.
    # run_session_2ch(..., active_states=set())'s pos[:,1].
def plot_potential(result) -> None
```

---

## What's Been Validated on Real Hardware

- Adder circuit (`ao0/10 + ao1` → `ai3`) — manual sweep, predicted vs.
  measured within noise.
- Camera → decode → fire → `ai3` readback, full checkpoint loop, hardware
  trigger latency and frame-drop behavior characterized (grab()+find()
  well under the frame budget at 100-180 fps).
- `run_batch_2ch` N-experiment batches, `NICardIn` accumulation bug fixed
  (was carrying data across experiments), CamOut-vs-requested frame-count
  cross-check.
- `ensure_stage_at`'s slow ramp + settle, now with AOM safety integration.
- Section 10 (notebook) — voltage-to-pixel calibration Step 1: sine sweeps
  on `ao0` and `ao1` independently, direct `ai1`-vs-`ai3` real-trace
  comparison, and a protocol-handoff spike check using the exact real
  firing path. See [Lessons](#lessons-from-calibration-testing) — the
  actual sign/slope numbers from these runs are the open item, not yet
  recorded here.

## Lessons From Calibration Testing

Worth remembering before extending any of this further:

1. **`ai1` (Stage) ≠ `ai3` (AdderOut)**, even at rest — different sides of
   the amplifier, different gain/offset. Any code that needs to know "what
   voltage am I currently commanding" (e.g. to seed a new waveform's
   baseline so it continues smoothly from where the physical signal
   already is) must read `ai3`, not `ai1`. Reading `ai1` for this
   introduced a real, reproducible voltage step at the start of several
   calibration test waveforms — fixed by switching the reference channel,
   not by adding a correction offset.
2. **`NICardIn.get_sample_count()` is chunk-quantized** (`READ_N=1000`),
   not a true real-time counter. Capturing it too soon after
   `start_continuous()` (before one `READ_N/rate` has elapsed) reads a
   stale `0`, silently mislabeling an entire plot's time axis. Any new
   code using it as a `t=0` reference must wait past that first.
3. **`AOIdleOutputBehavior` is not supported on this USB-6003** — confirmed
   directly, not assumed. There is no software lever for "hold vs. reset"
   AO idle behavior on this hardware.
4. A shared `NICardOutDual` task means **every** `load()`/`start()`
   re-arms *both* `ao0` and `ao1` together — there's no way to fire `ao0`
   alone while leaving `ao1`'s task genuinely untouched, short of
   splitting into two independent `nidaqmx.Task`s (untested whether this
   hardware even supports two simultaneous AO tasks — not pursued, since
   the `ai1`-vs-`ai3` fix above was sufficient in practice).
5. Camera `grab()` timeouts raise `genicam.TimeoutException` directly from
   `RetrieveResult`, not the documented `RuntimeError` — now caught and
   converted at the source in `camera.py`.
6. Always call `ao.hardware_trigger()` after `camera.start()` for any new
   hardware-triggered camera loop — it's the pulse that actually starts
   the funcgen's burst; omitting it just hangs `grab()` until timeout.
7. Any script that calls `ai.start_continuous()` must guarantee
   `ai.stop_and_get()` (or `.close()`) runs even on exception/interrupt —
   otherwise the background thread's `nidaqmx.Task` keeps `ai0-ai3`
   reserved, and the next hardware call anywhere in the process fails
   with a device-busy error. Same for any un-joined AO background thread.

---

## Pending Work

This is the actual next-conversation starting point — start a **fresh
conversation** for the analytical-protocols work below; it's a large,
separate research codebase and doesn't need this whole session's history.

### Section 10a — real run, and a reframing worth checking first

First real ao0 sine sweep result: measured slope **+54.5 nm/V**
(R²=0.70, N≈5000), vs. the assumed `NM_PER_VOLT/10 = 300 nm/V` — about
**5.5× smaller**. The R²=0.70 is expected and doesn't undermine the slope:
the particle is trap-bound, not rigidly driven, so at 10 nm/s (far slower
than the trap's relaxation time) it tracks the commanded position in
quasi-static equilibrium — the scatter is thermal noise (~50-90 nm std,
comparable to this test's ~109 nm peak driven excursion) superimposed on
a still-valid mean, not a broken calibration.

**The 5.5× discrepancy itself may not be a DAQ/piezo issue at all** —
`camera.py::NM_PER_PX = 16` is explicitly commented `# update after
calibration` and has never actually been calibrated. `ai1` is a pure
voltage measurement (no optics involved), so if `NM_PER_VOLT = 3000 nm/V`
(manufacturer spec) is trusted, working backward from the measured 54.5
nm/V (which used the placeholder 16 nm/px) implies a real pixel scale of
`16 × (300/54.5) ≈ 88 nm/px`. **Before trusting any nm/V number from
section 10, rerun the fit using raw pixels (`x_px`, pre-`NM_PER_PX`)
against the known stage displacement from `ai1 × NM_PER_VOLT`** — this
uses only trusted quantities and may turn out to be a camera pixel-scale
calibration rather than a piezo one. 10b (ao1, no `/10` attenuation
in the way) is the natural cross-check: if its slope comes out close to
3000 nm/V, the discrepancy is isolated to the `/10` stage/camera path,
not the base piezo calibration.

### Analytical protocols — file format and loading design

**Status: loading is now integrated (2026-08-19)** — the design below was
implemented as `analytical_protocols.py::load_analytical_protocols()`, and
exercised from a new notebook section (10d, `Main_test_script.ipynb`): a
short keyboard loop (same `STATE_MENU` keys as `hw_two_channel_test.py`)
that loads all 9 states from a `run_single_parameter_point.py` output
directory and fires the real shape for the chosen state on `ao0`. `ai1`
(Stage) and `ai3` (AdderOut) are recorded continuously for the whole
keyboard session; on quitting the loop, `helper.plot_fire_session()` plots
both real channels together with every fire shaded/labeled — same
segmented-overlay idea as `hw_two_channel_test.py`'s `plot_session()`,
extended to real `ai1`-vs-`ai3` comparison instead of one channel against a
formula-predicted value. **Not yet run against real
hardware** — that's the immediate next step, same verification pattern as
section 10 (compare `ai3` commanded vs `ai1` actual stage response vs real
camera tracking).

`scripts/run_single_parameter_point.py` is confirmed as the right entry
point: with `n_outcomes=3`, its `(m_0, m_t) ∈ {-1,0,1}²` keys are an exact
match for `decode_state_pair()`'s 9-state system — no remapping needed.
`run_parameter_sweep.py`/`run_parameter_sweep_3state.py` remain for sweeping
many points to find a good parameter point in the first place; once a point
is chosen, `run_single_parameter_point.py` is what actually generates the
files `load_analytical_protocols()` reads.

**One assumption from the paragraphs below turned out to be wrong, in a way
that simplified things**: real saved trajectories have `lambda(0) = 0`
*exactly*, for all 9 states (checked directly against the saved `.npz`
files) — unlike the current placeholder `_shape_m1_1`, which stylizes an
instantaneous jump at `t=0`. The true optimal protocol computed here has no
such jump; the placeholders' jump was a guess, not a feature to preserve.

**Sign convention, confirmed with a worked example (state `(1,1)`,
`"11 (exp)"`)**: particle persistently at `+x` past `xth` → extraction
intent is to move the *trap* toward the particle, `+x`, over `t_ext`
(`protocol_dt_s`). The stage↔trap negation is always-true (stage carries
the particle's frame; trap is fixed in the lab):

```
Δx_trap(t)   = +∫v(t')dt'                    (trap moves toward the particle)
Δx_stage(t)  = -Δx_trap(t)                    (always-true negation)
ΔV_stage(t)  = Δx_stage(t) / NM_PER_VOLT       (once the calibration above is settled)
```

**This means the placeholder `_shape_11` was indeed backwards**: it's
`"0 → +amplitude"` (positive-going), but a `(1,1)` protocol taken as
trap-frame intent should produce a *negative*-going stage voltage (per the
derivation above) — confirmed directly against the real saved data
(`(1,1)`'s `lambda(t)` ends at `+126 nm` trap-frame, i.e. `-126 nm`
stage-frame). `split_channels()` itself is unchanged and still applies no
sign negation (`displacement_v = waveform_nm / NM_PER_VOLT`, direct) — the
negation is applied once by the loader instead (see below), not inside
`split_channels()`, so the placeholder path (whatever still calls
`get_protocol_nm()` directly) is untouched and stays exactly as backwards
as before unless/until it's switched over to the loaded dict too.

**Implemented file format**: `analytical_protocols.load_analytical_protocols()`
reads each `(m_0, m_t)`'s saved `optimal_trajectories` row 3 (`lambda(t)`,
meters, lab/trap frame), resamples it onto the AO grid (`np.interp`, since
the analytical side saves 1001 points over `t_protocol_end` regardless of
`ao_rate`), converts to nm, and negates once (trap→stage) — landing in
**nm, stage-frame, at `ao0`'s pre-×10 scale**, i.e. exactly
`protocols_2ch.get_protocol_nm()`'s return convention (not V, and not yet
×10'd — that deliberately stays inside `split_channels()`, matching the
existing placeholder path so both sources are drop-in interchangeable).
Negation happens at load time, once per session, not at fire time — so the
hot path is unchanged: dict lookup + `split_channels()` (clip + last-sample
ao1-handoff) + `ao.load()` + `ao.start()`, same ~7-9 ms fire latency.
`(0, 0)` is always returned as an exact all-zeros array regardless of the
saved file's numerical noise, matching the no-op convention. Wired into
`Main_test_script.ipynb` section 10d as a short keyboard-fire loop, same
`STATE_MENU` keys as `hw_two_channel_test.py` — **not yet run against real
hardware**; that, plus verifying via `ai3`/`ai1`/camera tracking (same
pattern as section 10), is the next actual step.

---

## Constants Reference

```python
# NI USB-6003
DEVICE      = "Dev4"
AO0_CHAN    = "Dev4/ao0"      # fine, raw -10..10 V
AO1_CHAN    = "Dev4/ao1"      # coarse, 1.5..8.5 V
DO_CHAN     = "Dev4/port0/line0"   # hardware trigger to camera-trigger funcgen
AI_RATE     = 3000            # S/s (session default)
AO_RATE     = 1000            # S/s (session default)

# Stage (Piezoconcept LFHS3)
V_MIN       = 1.5             # V  →  4.5 µm
V_MAX       = 8.5             # V  → 25.5 µm
V_CENTER    = 5.0             # V  → 15.0 µm
NM_PER_VOLT = 3000.0          # nm/V (ao1/Stage; ao0 effective ≈ NM_PER_VOLT/10)
STAGE_SAFETY_MIN_V = 1.667    # live-firing bound (~5 µm)
STAGE_SAFETY_MAX_V = 8.333    # live-firing bound (~25 µm)

# Function generators (Tektronix AFG3000 — TWO physical units)
FG_CAMERA_USB_ID = "USB0::0x0699::0x0358::C018956::INSTR"
FG_AOM_USB_ID    = "USB0::0x0699::0x0358::C020332::INSTR"
FG_CH_CAM        = 2          # camera-trigger unit, CH2 drives camera trigger
FG_CH_AOM        = 2          # AOM unit, CH2 (its CH1 stage-pulse role is obsolete)

# Camera (Basler)
CAM_TRIGGER_LINE = "Line2"    # hardware trigger input
CAM_STROBE_LINE  = "Line3"    # ExposureActive output → AI2 CamOut
NM_PER_PX         = 16        # update after calibration
```

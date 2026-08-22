# System Ready for Testing

## Status: Complete Mock Implementation Validated ✓

All core components are implemented and tested with synthetic data.

---

## Validated Components

### 1. **Session Management** (`experiment.py`)
- `run_session()`: Executes full recording session with position-based triggers
  - ✓ Continuous AI recording in background
  - ✓ Position tracking per frame
  - ✓ Trigger detection (position > threshold)
  - ✓ State decoding from (x, y) position
  - ✓ Waveform firing on trigger
  - ✓ Pos array: (n_frames, 4) with [time_s, x_nm, y_nm, trigger]
  - ✓ Trigger encoding: 0=no event, 1–4 = state pulse height

- `build_datasync()`: Post-hoc synchronization via CamOut edges
  - ✓ CamOut rising edge detection (> 2V)
  - ✓ Low-pass filtering of Stage channel (100 Hz)
  - ✓ DataSync array: (n_frames, 3) with [time_s, stage_nm, aom_V]

- `save_session()`: I/O for offline analysis
  - ✓ Saves pos, DataSync, ai_data as .npy files
  - ✓ Timestamp-based filenames

### 2. **Hardware Abstraction** (DAQ, Camera, Function Generator)
- `daq.py`: NI USB-6003 multi-channel interface
  - ✓ NICardOut: AO1 + digital trigger (pre-configured once, reused per trigger)
  - ✓ NICardIn: Continuous multi-channel AI with background thread

- `camera.py`: Basler camera hardware-triggered interface
  - ✓ Hardware trigger on Line2 (rising edge from function generator)
  - ✓ Frame grabbing: one-at-a-time streaming
  - ✓ Particle tracking: centroid-based (ready for Gaussian fit)
  - ✓ Line3 ExposureActive strobe output → AI2 (CamOut)

- `function_gen.py`: Tektronix AFG3000 SCPI interface
  - ✓ Burst mode configuration (n_frames pulses at fps)
  - ✓ Output control (on/off)

### 3. **Protocol Definitions** (`protocols.py`)
- ✓ SessionParams: trecord_s, protocol_dt_s, fps, trigger_nm, etc.
- ✓ State decoding: (x, y) → 2-bit state (00, 01, 10, 11)
- ✓ Waveform building: relative displacement + current voltage → absolute AO output
- ✓ Trigger condition: |position| > threshold_nm

### 4. **Mock Hardware** (`mock_hardware.py`)
- ✓ MockFunctionGenerator: no-op SCPI interface
- ✓ MockBaslerCamera: synthetic Lissajous particle trajectory
- ✓ MockNICardOut: waveform logging
- ✓ MockNICardIn: synthetic AI data (Stage DC shift, constant AOM, CamOut pulses)
  - Generates data matching session duration
  - CamOut pulses at specified fps for proper synchronization

---

## Test Results (10-second session, 100 fps)

```
Session config:    10.0 s, 1000 frames, 100 fps
Triggers fired:    5 state-dependent protocols
Pos array shape:   (1000, 4)  [time_s, x_nm, y_nm, trigger]
DataSync shape:    (999, 3)   [time_s, stage_nm, aom_V]
AI samples:        30,000     (10 s × 3000 S/s)
Files saved:       pos, DataSync, ai_data (.npy)
Sync accuracy:     CamOut edges detected correctly
```

---

## Architecture Overview

```
Camera Frame Loop
  ├─ grab_single() → raw frame
  ├─ track_particle() → (x_nm, y_nm)
  ├─ check trigger condition
  │  └─ if |pos| > threshold
  │     ├─ decode_state(pos) → 2-bit state
  │     ├─ build_waveform(state, current_v) → AO samples
  │     └─ fire AO (pre-configured task: stop→write→start)
  ├─ store pos[i] = [t, x, y, trigger_encoding]
  └─ repeat n_frames times

Background: Continuous AI recording (Stage, AOM, CamOut)

Post-hoc:
  ├─ Find CamOut edges in AI stream
  ├─ Sync Stage + AOM to camera frame times
  └─ DataSync = [t_sync, stage_filt, aom_V]
```

---

## Real Hardware: Next Steps

1. **Connect hardware** (camera, DAQ, function generator, piezo stage)
2. **Update `__main__` block** in any test script to use real hardware classes instead of mocks
3. **Run session** with `run_session(params, ao, ai, camera, funcgen)`
4. **Analyze data** with build_datasync() and plot_session()

Example:
```python
from daq import NICardOut, NICardIn
from camera import BaslerCamera
from function_gen import FunctionGenerator

ao = NICardOut(...)
ai = NICardIn(...)
camera = BaslerCamera(...)
funcgen = FunctionGenerator(...)

result = run_session(params, ao, ai, camera, funcgen)
DataSync, _ = build_datasync(result)
save_session(result, DataSync)
```

---

## Data Structures

### pos (n_frames, 4)
| Column | Name    | Type    | Notes                          |
|--------|---------|---------|--------------------------------|
| 0      | time_s  | float64 | Seconds from session start     |
| 1      | x_nm    | float64 | X position (nm)                |
| 2      | y_nm    | float64 | Y position (nm)                |
| 3      | trigger | float64 | 0=no event, 1–4=state pulse   |

### DataSync (n_frames, 3)
| Column | Name     | Type    | Notes                          |
|--------|----------|---------|--------------------------------|
| 0      | time_s   | float64 | Seconds from first camera frame|
| 1      | stage_nm | float64 | Stage position (nm, filtered)  |
| 2      | aom_V    | float64 | AOM voltage (V)                |

### AI data (n_channels, n_ai_samples)
Channels (in order): [AOM, Stage, CamOut]

---

## Known Limitations

- **Particle tracking**: Currently centroid-based. Consider Gaussian fit for sub-pixel accuracy.
- **Mock vs. Real timing**: Mock hardware runs instantaneously; real hardware will introduce realistic delays.
- **DAQ buffer**: Mock uses default size. For very long protocols, verify hardware buffer (see `hw_protocol_test.py` for buffer diagnostics).

---

## Files Summary

| File | Purpose |
|------|---------|
| `protocols.py` | State definitions, waveform builders, session params |
| `daq.py` | NI USB-6003 AO/AI interface (pre-configured AO task) |
| `camera.py` | Basler camera hardware-triggered interface |
| `function_gen.py` | Tektronix AFG3000 SCPI control |
| `experiment.py` | Session runner, DataSync builder, save/plot utilities |
| `mock_hardware.py` | Synthetic hardware for offline testing |
| `test_session_mock.py` | Validation test (all components working) |
| `hw_protocol_test.py` | Manual loop for AO/AI testing (raw voltage loopback) |
| `EXPERIMENT_ARCHITECTURE.md` | Hardware spec, wiring, API details |

---

## Running the Full System

```bash
# Test with mock hardware (no devices required)
python test_session_mock.py

# Manual loop with real hardware (AO→AI loopback)
python hw_protocol_test.py

# Full session with real hardware
python session_runner.py  # (to be created for final integration)
```

---

**Last validated:** 2026-06-26  
**Architecture:** Session-based (1 long recording, position-triggered protocols)  
**Status:** Ready for real hardware integration

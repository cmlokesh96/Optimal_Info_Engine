# BaslerCamera Python Interface (MATLAB API Equivalent)

## Overview
The Python `BaslerCamera` class now mirrors the MATLAB `BaslerCamera_L` interface with live preview, recording, and hardware trigger support.

---

## Usage Examples

### Live Preview
```python
from camera import BaslerCamera

cam = BaslerCamera(exposure_us=10000)
cam.startview()      # Opens live video window
# ... watch particle ...
cam.stopview()       # Close window
cam.close()          # Cleanup
```

### Hardware-Triggered Acquisition
```python
cam = BaslerCamera(exposure_us=10000)
cam.nframe(1000)                # 1000 frames per trigger
cam.framerate(100)              # 100 fps
cam.hardwareTrigg()             # Configure Line2 input
cam.exposure('Once')            # Manual exposure per trigger
cam.start()                      # Arm camera
# ... trigger will cause frames to arrive ...
frame = cam.grab_single()       # Get one frame
cam.stop()
cam.close()
```

### Region of Interest (ROI)
```python
cam = BaslerCamera()
cam.roi(width=256, height=256, x_offset=128, y_offset=128)
cam.startview()      # Preview with ROI applied
```

### Recording Video
```python
cam = BaslerCamera(exposure_us=10000)
cam.savepath("./output.avi", fps=100)  # Set save path
cam.record()         # Start recording
cam.start()
# ... grab frames ...
while True:
    frame = cam.grab_single()
    if frame is None:
        break
cam.stop()
cam.close()          # Video saved automatically
```

### Exposure Control
```python
cam = BaslerCamera()

# Set exposure time (microseconds)
cam.exposure(10000)              # 10 ms exposure

# Set exposure mode (MATLAB compatibility)
cam.exposure('Once')             # Single shot
cam.exposure('Continuous')       # Auto exposure
```

### Software Trigger
```python
cam = BaslerCamera(exposure_us=10000)
cam.softwareTrigg()              # Software trigger mode
cam.framerate(30)                # 30 fps
cam.start()
frame = cam.grab_single()        # Next software trigger
```

---

## API Reference

### Initialization
```python
cam = BaslerCamera(exposure_us=10000)
```
- **exposure_us**: Initial exposure time in microseconds (default: 10000)

### Live Preview
```python
cam.startview()      # Start live preview window
cam.stopview()       # Stop and close window
```

### Frame Configuration
```python
cam.nframe(nf)                   # Set frames per trigger
cam.framerate(fps)               # Set acquisition rate (fps)
cam.roi(width, height, x_offset, y_offset)  # Set region of interest
```

### Trigger Modes
```python
cam.hardwareTrigg()              # Line2 input, rising edge
cam.softwareTrigg()              # Software-triggered mode
cam.setup_hardware_trigger()     # Alias for hardwareTrigg()
```

### Exposure Control
```python
cam.exposure(10000)              # Set time (µs)
cam.exposure('Once')             # Auto mode: single shot
cam.exposure('Continuous')       # Auto mode: continuous
cam.set_exposure(time_us)        # Alias for exposure(numeric)
```

### Recording
```python
cam.savepath(path, fps=30)       # Set save path for video
cam.record()                     # Start recording (memory or disk)
cam.stop()                       # Stop recording, close file
```

### Acquisition
```python
cam.start()                      # Arm camera for acquisition
frame = cam.grab_single(timeout_ms=2000)  # Get one frame (H,W) uint8
cam.stop()                       # Stop acquisition
```

### Particle Tracking
```python
x_nm, y_nm = cam.track_particle(frame, nm_per_px=100.0)
```
- Returns raw particle position in nanometers (pixel (0,0) = top-left of the
  current frame/ROI) — NOT centered on anything; subtract a calibrated trap
  center downstream for a trap-relative reading
- **nm_per_px**: Pixel-to-nanometer scale factor (default: 100 nm/pixel)

### Cleanup
```python
cam.close()                      # Close camera, cleanup resources
```

---

## MATLAB to Python Mapping

| MATLAB Method | Python Method | Notes |
|---|---|---|
| `Cam = BaslerCamera_L()` | `cam = BaslerCamera()` | Constructor |
| `Cam.startview()` | `cam.startview()` | Live preview window |
| `Cam.stopview()` | `cam.stopview()` | Close preview |
| `Cam.exposure('Once')` | `cam.exposure('Once')` | Exposure mode |
| `Cam.exposure(10000)` | `cam.exposure(10000)` | Exposure time |
| `Cam.nframe(1000)` | `cam.nframe(1000)` | Frames per trigger |
| `Cam.framerate(100)` | `cam.framerate(100)` | Frame rate (fps) |
| `Cam.hardwareTrigg()` | `cam.hardwareTrigg()` | Hardware trigger |
| `Cam.softwareTrigg()` | `cam.softwareTrigg()` | Software trigger |
| `Cam.ROI(w,h,x,y)` | `cam.roi(w, h, x, y)` | Region of interest |
| `Cam.savepath(path)` | `cam.savepath(path, fps)` | Video save path |
| `Cam.record()` | `cam.record()` | Start recording |
| `Cam.stop()` | `cam.stop()` | Stop recording |

---

## Troubleshooting

### Error: "Device is exclusively opened by another client"

**Cause:** Camera is locked by another process.

**Solutions:**
1. **Close Basler Pylon Viewer** if running
2. **Restart Python kernel** (Jupyter: Kernel → Restart)
3. **Check for other Python instances** (they may have the camera open)
4. **Unplug/replug USB cable**

**In Jupyter:**
```python
# Restart kernel before running:
import sys
# (This clears all previous camera references)
```

### Live Preview Window Not Showing

**Cause:** OpenCV display issues on Windows or SSH sessions.

**Solution:** Skip preview for headless operation:
```python
# Use grab_single() directly without startview()
cam.start()
frame = cam.grab_single()  # Get frame directly
```

### Recording to Disk Fails

**Cause:** Invalid path or insufficient permissions.

**Solution:** Ensure directory exists:
```python
import os
os.makedirs("./output", exist_ok=True)
cam.savepath("./output/video.avi", fps=100)
```

---

## Implementation Details

### Live Preview
- Runs in background thread
- Uses OpenCV for display (cv2.imshow)
- Press 'q' in window to stop
- Automatic cleanup on stopview()

### Recording
- **Memory mode**: Frames buffered in RAM (self._frame_buffer)
- **Disk mode**: Direct write to video file via OpenCV
- **Frame count**: Tracked in self._frame_count

### Exposure Modes
- **'Once'**: Single exposure per trigger (manual)
- **'Continuous'**: Auto exposure (continuously adjusted)

---

## Integration with Experiment System

The expanded camera integrates seamlessly with `experiment.py`:

```python
from camera import BaslerCamera
from protocols import SessionParams
from experiment import run_session
from daq import NICardOut, NICardIn
from function_gen import FunctionGenerator

params = SessionParams(...)

# Initialize hardware
ao = NICardOut(...)
ai = NICardIn(...)
camera = BaslerCamera(exposure_us=params.exposure_us)
funcgen = FunctionGenerator()

# Run session with live preview capability
# camera.startview()  # Optional: watch live
result = run_session(params, ao, ai, camera, funcgen)
# camera.stopview()   # Optional: stop watching

# Analysis
from experiment import build_datasync, plot_session, save_session
DataSync, _ = build_datasync(result)
save_session(result, DataSync)
plot_session(result, DataSync)
```

---

**Status:** Ready for real hardware testing  
**Last updated:** 2026-06-26

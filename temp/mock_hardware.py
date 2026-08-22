"""
mock_hardware.py
================
Mock DAQ, camera, and function generator for offline testing.
Allows testing the full session flow without hardware connected.
"""

import numpy as np
import time


class MockFunctionGenerator:
    """Mock AFG3000 — no-op SCPI interface."""
    def __init__(self, usb_id=None):
        pass
    def setup_camera_trigger(self, fps, exposure_us, n_frames, ch=2):
        print(f"[MOCK FuncGen] Setup: {n_frames} frames at {fps} fps, "
              f"exposure={exposure_us}µs")
    def output_on(self, ch=2):
        pass
    def output_off(self, ch=2):
        pass
    def query(self, cmd):
        return "OK"
    def close(self):
        pass


class MockBaslerCamera:
    """Mock Basler camera — generates synthetic particle trajectories."""
    def __init__(self, exposure_us):
        self.exposure_us = exposure_us
        self._frame_i = 0

    def setup_hardware_trigger(self):
        pass

    def start(self):
        self._frame_i = 0

    def grab_single(self, timeout_ms=2000) -> np.ndarray:
        """Return a synthetic frame (all ones for tracking test)."""
        frame = np.ones((256, 256), dtype=np.uint8) * 128
        self._frame_i += 1
        return frame

    def track_particle(self, frame, nm_per_px=100.0) -> tuple[float, float]:
        """
        Synthetic particle: Lissajous trajectory
        x(t) = A sin(ωt), y(t) = B sin(2ωt)
        """
        t = self._frame_i / 100.0   # t in arbitrary units
        x_nm = 100.0 * np.sin(2 * np.pi * 0.05 * t)
        y_nm = 80.0 * np.sin(2 * np.pi * 0.07 * t)
        return float(x_nm), float(y_nm)

    def stop(self):
        pass

    def set_roi(self, *args):
        pass

    def set_exposure(self, exposure_us):
        self.exposure_us = exposure_us

    def close(self):
        pass


class MockNICardOut:
    """Mock NICardOut — logs waveform fire events."""
    def __init__(self, n_samples, rate=1000, v_min=1.5, v_max=8.5):
        self._n = n_samples
        self._rate = rate
        self.fired = []   # log of (state, waveform) tuples

    def load(self, waveform):
        pass

    def start(self):
        pass

    def wait_done(self, extra_s=5.0):
        time.sleep(0.01)  # simulate short wait

    def hardware_trigger(self):
        pass

    def set_dc(self, voltage):
        pass

    def close(self):
        pass


class MockNICardIn:
    """Mock NICardIn — generates synthetic AI data (Stage + AOM + CamOut)."""
    def __init__(self, rate=3000, session_s=None):
        self.rate = rate
        self.session_s = session_s or 10.0   # default 10 seconds if not specified
        self.channel_names = ["AOM", "Stage", "CamOut"]
        self._data = [[], [], []]
        self._sample_count = 0

    def start_continuous(self):
        print(f"[MOCK NICardIn] Start continuous recording at {self.rate} S/s")

    def get_sample_count(self) -> int:
        return self._sample_count

    def stop_and_get(self) -> np.ndarray:
        """Return synthetic AI data matching configured session duration."""
        print(f"[MOCK NICardIn] Generating synthetic data...")
        # Generate data for the configured session duration
        n_samples = int(self.session_s * self.rate)

        # Synthetic Stage (piezo) — slow DC shifts
        t = np.arange(n_samples) / self.rate
        stage_v = 5.0 + 0.5 * np.sin(2 * np.pi * t / 600)  # 10 minute oscillation

        # Synthetic AOM — constant ~3V
        aom_v = np.ones(n_samples) * 3.0

        # Synthetic CamOut — camera trigger pulses (one per frame at fps)
        cam_out = np.zeros(n_samples)
        pulse_width = int(0.001 * self.rate)    # 1 ms pulse width
        pulse_period = int(1.0 / 100 * self.rate)  # 10 ms between pulses (100 fps)
        for i in range(0, n_samples, pulse_period):
            if i + pulse_width < n_samples:
                cam_out[i:i+pulse_width] = 5.0

        data = np.array([aom_v, stage_v, cam_out])
        print(f"  Generated {n_samples} samples, {n_samples/self.rate/3600:.2f} h")
        return data

    def close(self):
        pass

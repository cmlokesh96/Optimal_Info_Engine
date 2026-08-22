"""
test_session_mock.py
====================
Test the full session flow using mock hardware (no real devices required).
Validates pos and DataSync data structures, trigger encoding, and save I/O.
"""

import numpy as np
import matplotlib.pyplot as plt

from protocols import SessionParams
from experiment import run_session, build_datasync, save_session, plot_session
from mock_hardware import (
    MockFunctionGenerator, MockBaslerCamera, MockNICardOut, MockNICardIn
)


def test_mock_session():
    """Run a mini session with mock hardware (10 sec, 100 fps)."""
    print("=" * 60)
    print(" Test: Mock Hardware Session")
    print("=" * 60)

    # ── Session parameters ─────────────────────────────────────────────────────
    params = SessionParams(
        trecord_s=10.0,          # short test duration
        protocol_dt_s=2.0,       # 2 sec per protocol
        fps=100,
        exposure_us=10000,
        trigger_nm=50.0,         # trigger threshold
        ai_rate=3000,
        ao_rate=5000,
        savepath="./test_data/session"
    )

    n_frames = int(params.fps * params.trecord_s)
    print(f"\nSession config:")
    print(f"  Duration: {params.trecord_s} s")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {params.fps}")
    print(f"  Trigger threshold: {params.trigger_nm} nm\n")

    # ── Initialize mock hardware ───────────────────────────────────────────────
    funcgen  = MockFunctionGenerator()
    camera   = MockBaslerCamera(exposure_us=params.exposure_us)
    ao       = MockNICardOut(
        n_samples=int(params.protocol_dt_s * params.ao_rate),
        rate=params.ao_rate
    )
    ai       = MockNICardIn(rate=params.ai_rate, session_s=params.trecord_s)

    # ── Run session ────────────────────────────────────────────────────────────
    result = run_session(params, ao, ai, camera, funcgen)

    pos = result["pos"]
    print(f"\nSession complete:")
    print(f"  Pos shape: {pos.shape}")
    print(f"  Triggers fired: {int(np.sum(pos[:, 3] > 0))}")

    # ── Validate pos structure ─────────────────────────────────────────────────
    print(f"\nPos array validation:")
    print(f"  Time range: {pos[:, 0].min():.3f}–{pos[:, 0].max():.3f} s")
    print(f"  X range: {pos[:, 1].min():.1f}–{pos[:, 1].max():.1f} nm")
    print(f"  Y range: {pos[:, 2].min():.1f}–{pos[:, 2].max():.1f} nm")
    print(f"  Trigger values: {np.unique(pos[:, 3])}")

    if np.any(pos[:, 3] > 0):
        trig_frames = np.flatnonzero(pos[:, 3] > 0)
        print(f"  Trigger frames: {trig_frames.tolist()}")
        print(f"  Trigger states: {pos[trig_frames, 3].astype(int).tolist()}")

    # ── Build DataSync ─────────────────────────────────────────────────────────
    print(f"\nBuilding DataSync...")
    DataSync, tvec = build_datasync(result, lowpass_hz=100.0)

    print(f"  DataSync shape: {DataSync.shape}")
    print(f"  Time range: {DataSync[:, 0].min():.3f}–{DataSync[:, 0].max():.3f} s")
    print(f"  Stage range: {DataSync[:, 1].min():.1f}–{DataSync[:, 1].max():.1f} nm")
    print(f"  AOM range: {DataSync[:, 2].min():.3f}–{DataSync[:, 2].max():.3f} V")

    # ── Test save ──────────────────────────────────────────────────────────────
    print(f"\nTesting save_session()...")
    try:
        import os
        os.makedirs("./test_data", exist_ok=True)
        save_session(result, DataSync, suffix="mock_test")
        print(f"  [OK] Save successful")
    except Exception as e:
        print(f"  [FAIL] Save failed: {e}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots...")
    plot_session(result, DataSync)

    return result, DataSync


if __name__ == "__main__":
    test_mock_session()

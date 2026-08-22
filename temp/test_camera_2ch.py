"""
test_camera_2ch.py
===================
Real-hardware dry run of the 9-state / two-channel decision loop
(protocols_2ch.py + experiment.py::run_session_2ch).

Camera, function generator, and DAQ are all REAL; the piezo stage is NOT
physically connected yet, so x(t) is just a freely-diffusing particle in
the trap (no feedback effect from firing protocols). This still validates
the whole software chain end to end before the stage is wired in:
    camera -> track -> threshold/decode -> protocol lookup -> ao0/ao1 fire
    -> ai3 (AdderOut) readback, cross-checked against the state recorded
    in pos[:,3].

See EXPERIMENT_ARCHITECTURE.md for wiring and the harmonic-pondering-lantern
plan for the decision-timing model (capture/decide checkpoints, 00 -> skip
one checkpoint, fire -> wait T_relax).
"""

import os

from protocols_2ch import SessionParams2Ch, V_CENTER
from daq import NICardOutDual, NICardIn
from function_gen import FunctionGenerator
from camera import BaslerCamera
from experiment import run_session_2ch, build_datasync, save_session, plot_session_2ch


def main():
    params = SessionParams2Ch(
        trecord_s     = 120.0,   # short first dry run
        fps           = 100.0,
        exposure_us   = 500.0,
        xth_nm        = 50.0,
        delta_t_s     = 1.0,
        T_relax_s     = 10.0,
        protocol_dt_s = 3.0,
        savepath      = "./test_data/cam2ch",
    )

    # Every resource below opens a real hardware handle (NI-DAQmx task, camera
    # device, VISA session). If a later one fails to open, the earlier ones
    # must still be closed — otherwise they leak (orphaned DAQ tasks, camera
    # left exclusively locked) and the next run fails too. So construction
    # AND cleanup both live inside the same try/finally.
    ao = ai = camera = funcgen = None
    try:
        ao      = NICardOutDual(
            n_samples=int(params.protocol_dt_s * params.ao_rate), rate=params.ao_rate
        )
        ai      = NICardIn(rate=params.ai_rate)
        camera  = BaslerCamera(exposure_us=params.exposure_us)
        funcgen = FunctionGenerator()

        result = run_session_2ch(
            params, ao, ai, camera, funcgen, ao1_start_v=V_CENTER
        )

        DataSync, tvec = build_datasync(result)

        os.makedirs("./test_data", exist_ok=True)
        save_session(result, DataSync, suffix="_2ch")

        plot_session_2ch(result, tvec)

    finally:
        for resource in (ao, ai, camera, funcgen):
            if resource is not None:
                try:
                    resource.close()
                except Exception as e:
                    print(f"  warning: failed to close {resource!r}: {e}")


if __name__ == "__main__":
    main()

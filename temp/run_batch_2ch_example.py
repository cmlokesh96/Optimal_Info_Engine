"""
run_batch_2ch_example.py
=========================
Illustrative N-experiment batch using the independent building blocks —
preview_particle, ensure_stage_at, run_batch_2ch, ao.ramp_ao1 — composed
explicitly here rather than bundled into one big orchestrator. Each piece
is independently callable; edit/reorder/drop any of them as needed.

Numbers below (trecord_s, xth_nm, ...) are placeholders — edit for your
actual run. x_center_nm/y_center_nm should come from a separate
calibrate_trap_center() run first (see its docstring in experiment.py),
not from this script.
"""

from protocols_2ch import V_CENTER
from daq import NICardOutDual, NICardIn, read_stage_voltage
from function_gen import FunctionGenerator
from camera import BaslerCamera
from experiment import preview_particle, ensure_stage_at, run_batch_2ch


def main():
    # Paste the result of a separate calibrate_trap_center() run here.
    x_center_nm, y_center_nm = 0.0, 0.0

    ao_rate, ai_rate = 1000, 3000

    base_kwargs = dict(
        fps           = 100.0,
        exposure_us   = 500.0,
        x_center_nm   = x_center_nm,
        y_center_nm   = y_center_nm,
        protocol_dt_s = 3.0,
        T_relax_s     = 10.0,
        ai_rate       = ai_rate,
        ao_rate       = ao_rate,
        savepath      = "./test_data/batch2ch",
    )

    # 4 illustrative experiments, short trecord_s for a first dry run —
    # replace with your real per-experiment xth_nm/delta_t_s/trecord_s.
    experiment_overrides = [
        dict(xth_nm=70.0, delta_t_s=1.0, trecord_s=60.0)
        for _ in range(4)
    ]

    ao = ai = camera = funcgen = None
    try:
        ao      = NICardOutDual(
            n_samples=int(base_kwargs["protocol_dt_s"] * ao_rate), rate=ao_rate
        )
        ai      = NICardIn(rate=ai_rate)
        camera  = BaslerCamera(exposure_us=base_kwargs["exposure_us"])
        funcgen = FunctionGenerator()

        camera.hardwareTrigg()

        ensure_stage_at(ao, V_CENTER)     # independent: skips the wait if already there
        preview_particle(camera, 100.0)   # independent

        run_batch_2ch(
            base_kwargs, experiment_overrides, ao, ai, camera, funcgen,
            ao1_start_v=V_CENTER,
        )

        preview_particle(camera, 100.0)   # independent

        # Only if you're powering down now — independent, your call:
        # current_v = read_stage_voltage()
        # ao.ramp_ao1(current_v, 0.0, speed_nm_per_s=10.0)

    finally:
        for resource in (ao, ai, camera, funcgen):
            if resource is not None:
                try:
                    resource.close()
                except Exception as e:
                    print(f"  warning: failed to close {resource!r}: {e}")


if __name__ == "__main__":
    main()

"""
analyze_batch_from_stage.py
============================
Notebook-independent analysis of a saved run_batch_2ch() batch folder.

Two diagnostic grids, both from the batch folder alone plus the currently-
loaded analytical protocol:

1. plot_stage_vs_protocol_grid - measured Stage-channel readback (ground
   truth, what the hardware actually delivered) vs. the loaded analytical
   protocol's stage-frame curve. A mismatch here means the loaded protocol
   doesn't match what this batch's hardware actually fired.

2. plot_trap_relative_grid - trap-relative X(t) = x_cam + lambda(t)
   (avoids the reference-frame plunge raw x_cam shows right at the jump)
   overlaid with lambda(t), its own instantaneous jump drawn as a dashed
   marker.

Work is computed with the analytical protocol (not the measured stage),
via compute_work_split_kT() - the jump's own contribution is reported and
plotted separately from the smooth part that follows it, using
lambda_trap_nm[1] (the first sample genuinely time-aligned with a real
camera reading) as the jump's target, so nothing past that point needs
any reconstruction.

Still depends on analytical_optimal_protocol_computation/'s currently-
saved protocol files for lambda(t) itself, until those get saved
per-batch (see the "save protocols per batch" plan this investigation
led to) - kappa/T and the protocol-timing constants below must match
what was actually used for BATCH_FOLDER.

Run directly:  python analyze_batch_from_stage.py
"""
import numpy as np

from analysis import (
    extract_protocol_events_from_folder,
    extract_protocol_stage_events_from_folder,
    compute_work_split_kT,
    lambda_trap_nm_for_state,
    plot_trap_relative_grid,
    plot_stage_vs_protocol_grid,
    plot_work_split_grid,
)
from analytical_protocols import load_analytical_protocols

BATCH_FOLDER  = r"D:\Data\Aug_26\26-08_Testing_Full_Engine\batch01"
KAPPA_N_PER_M = 2.4e-6
T_K           = 298.0

ANALYTICAL_SAVE_DIR = r"analytical_optimal_protocol_computation/parameters/optimal_protocols_multiple_measurement"
ANALYTICAL_T2 = 0.5
ANALYTICAL_TF = 3.0
AO_RATE       = 1000


def main():
    print(f"Loading batch: {BATCH_FOLDER}")
    x_events = extract_protocol_events_from_folder(BATCH_FOLDER)
    stage_events = extract_protocol_stage_events_from_folder(BATCH_FOLDER)

    print(f"Loading theoretical protocols: {ANALYTICAL_SAVE_DIR}")
    protocols = load_analytical_protocols(ANALYTICAL_SAVE_DIR, t_second_measurement=ANALYTICAL_T2,
                                          t_protocol_end=ANALYTICAL_TF, ao_rate=AO_RATE)

    lambda_theory_nm = {
        state: lambda_trap_nm_for_state(state, x_events[state]["t_s"], protocols,
                                        ANALYTICAL_TF, AO_RATE)
        for state in x_events if state in protocols
    }
    x_events = {s: x_events[s] for s in lambda_theory_nm}
    stage_events = {s: stage_events[s] for s in lambda_theory_nm if s in stage_events}
    stage_protocol_nm = {state: -lambda_theory_nm[state] for state in lambda_theory_nm}

    W_jump_by_state, W_total_by_state = {}, {}
    for state, lam in lambda_theory_nm.items():
        W_jump, W_total = compute_work_split_kT(x_events[state]["x_nm"], lam, KAPPA_N_PER_M, T_K)
        W_jump_by_state[state] = W_jump
        W_total_by_state[state] = W_total

    print(f"\n{'state':>10}  {'n_events':>9}  {'<W_jump>/kT':>12}  "
          f"{'<W_total>/kT':>14}  {'std':>8}")
    for state in sorted(W_total_by_state, key=lambda s: (s[0], s[1])):
        n = W_total_by_state[state].shape[0]
        if n == 0:
            print(f"{str(state):>10}  {n:9d}  {'--':>12}  {'--':>14}  {'--':>8}")
            continue
        W_final = W_total_by_state[state][:, -1]
        print(f"{str(state):>10}  {n:9d}  {W_jump_by_state[state].mean():12.3f}  "
              f"{W_final.mean():14.3f}  {W_final.std():8.3f}")

    plot_stage_vs_protocol_grid(stage_events, stage_protocol_nm)
    plot_trap_relative_grid(x_events, lambda_theory_nm)
    plot_work_split_grid(x_events, W_jump_by_state, W_total_by_state, KAPPA_N_PER_M, T_K)


if __name__ == "__main__":
    main()

"""
protocols_2ch.py
=================
Pure numpy — no hardware. 9-state decode/protocol logic for the two-channel
(ao0 fine + ao1 coarse) actuation scheme, kept separate from protocols.py
so the existing single-channel 4-state session code (daq.py, experiment.py)
is untouched.

State model
-----------
Each decision looks at two consecutive thresholded camera samples
x(t-δt), x(t), each in {-1, 0, 1} → a 9-state pair.
    (0, 0)            → no-op: caller should wait 2·δt and resample,
                          no protocol is fired.
    the other 8 states form 4 mirror-symmetric pairs:
        group A: (1,1) (1,0) (0,1) (-1,1)          — 4 stored base protocols
        group B: (-1,-1) (-1,0) (0,-1) (1,-1)      — group A negated elementwise
    get_protocol_nm() resolves group B by negating the matching group-A waveform.

Channel split
-------------
A protocol waveform is a *relative displacement in nm* (same convention as
protocols.py::make_protocol). split_channels() converts that into:
    ao0 — fine channel, does the actual excursion during the firing, then
          resets to 0 on the very last sample.
    ao1 — coarse channel, held constant during the firing, then absorbs the
          net excursion on the last sample so the physical output
          (ao0/10 + ao1) is continuous across the reset and the next
          firing starts from ao0=0 at the new ao1 offset.

The 4 base waveform shapes below are PLACEHOLDERS (simple ramps/exponential)
— swap in the real optimized-protocol shapes once available. The framework
(decode / lookup / mirror / channel-split) does not hardcode any physics.
"""

from dataclasses import dataclass
import numpy as np

#from protocols import NM_PER_VOLT, V_MIN, V_MAX, V_CENTER  # noqa: F401  (re-exported)

# ── Stage constants ────────────────────────────────────────────────────────────
V_MIN       = 1.5       # V  →  4.5 µm
V_MAX       = 8.5       # V  → 25.5 µm
V_CENTER    = 5.0       # V  → 15.0 µm  (resting)
NM_PER_VOLT = 3000.0    # nm per volt

# ── Channel ranges ──────────────────────────────────────────────────────────────
AO0_MIN, AO0_MAX = -10.0, 10.0     # fine channel, raw (before the circuit's /10)
AO1_MIN, AO1_MAX = V_MIN, V_MAX    # coarse channel — same span as the stage

PLACEHOLDER_AMPLITUDE_NM = 100.0   # tune once real protocol shapes are known

# ── Session parameters for the 9-state / two-channel scheme ──────────────────
@dataclass
class SessionParams2Ch:
    """Parallels protocols.py::SessionParams, for the 9-state 2-channel scheme."""
    trecord_s     : float          # total session duration (s)
    fps           : float          # camera frame rate (Hz)
    exposure_us   : float          # camera exposure time (us)
    xth_nm        : float = 50.0   # decode_symbol() threshold (nm)
    delta_t_s     : float = 1.0    # spacing between the two decision samples (s).
                                    # 0.0 triggers single-measurement mode in
                                    # run_session_2ch(): no capture step, every
                                    # checkpoint decides immediately using the
                                    # SAME sample for both "prev" and "curr"
                                    # (true simultaneity, matching the analytical
                                    # protocols' t_second_measurement=0 - only
                                    # the diagonal states (1,1)/(-1,-1)/(0,0)
                                    # are ever decoded this way)
    T_relax_s     : float = 10.0   # equilibration wait AFTER the fired protocol
                                    # finishes, before resuming decisions (s) -
                                    # does NOT include protocol_dt_s; the total
                                    # gap between a fire and the next decide
                                    # checkpoint is protocol_dt_s + T_relax_s
                                    # (see run_session_2ch's use of
                                    # protocol_frames + relax_frames)
    protocol_dt_s : float = 3.0    # duration of a fired protocol waveform (s)
    x_center_nm   : float = 0.0    # trap center, x (nm, raw camera.find() units,
                                    # NOT image-center-relative) — see
                                    # experiment.py::calibrate_trap_center()
    y_center_nm   : float = 0.0    # trap center, y (nm, raw camera.find() units)
    stage_min_v   : float = 1.667  # ao1 safety bound, live firing only (~5 um)
    stage_max_v   : float = 8.333  # ao1 safety bound, live firing only (~25 um)
                                    # (5-25 um = +/-10 um around the 15 um/5V
                                    # center; does NOT apply to a deliberate
                                    # ao.ramp_ao1(..., 0.0, ...) shutdown move)
    print_every   : int   = 1      # throttle: only print every Nth checkpoint
                                    # WITHIN each category (fire, no-fire) -
                                    # the two categories are counted and
                                    # throttled independently
    print_no_fire : bool  = False   # False = never print (0,0)/excluded-state
                                    # checkpoints at all (real experiments: only
                                    # fires are interesting); True keeps today's
                                    # behavior (needed e.g. for active_states=set()
                                    # equilibrium runs, where every decision is a
                                    # no-fire and this is the only console signal
                                    # that decoding is actually running)
    print_fire    : bool  = True   # False = never print fire checkpoints either
                                    # (e.g. a long batch run where you only want
                                    # the end-of-session summary); True (default)
                                    # prints fires, throttled by print_every same
                                    # as no-fire lines are
    ai_rate       : int   = 3000   # AI sampling rate (S/s)
    ao_rate       : int   = 1000   # AO sampling rate (S/s)
    savepath      : str   = ""     # path prefix for saved files
    fire_00_prob  : float = 0.0    # probability of firing the (0,0) protocol
                                    # (a flat, zero-displacement waveform - a
                                    # null/control measurement) each time
                                    # (0,0) is decoded, independent of
                                    # active_states - an unbiased Bernoulli
                                    # thinning (not "every Nth occurrence",
                                    # which could alias with any periodicity
                                    # in when (0,0) happens to occur). 0.0
                                    # (default) = never, today's behavior
                                    # unchanged. When it fires, pos[:,3] gets
                                    # ZERO_FIRE_CODE (not the usual (0,0)
                                    # code) so it's distinguishable from an
                                    # ordinary, non-fired (0,0) checkpoint.

    def __post_init__(self):
        # No T_relax_s >= protocol_dt_s constraint needed here anymore - the
        # next possible firing is always protocol_frames + relax_frames after
        # a fire (see run_session_2ch), so the waveform always finishes
        # before then regardless of how small T_relax_s is.
        assert self.T_relax_s >= 0, f"T_relax_s ({self.T_relax_s}) must be >= 0"

    @property
    def n_frames(self) -> int:
        return int(self.trecord_s * self.fps)

    @property
    def delta_frames(self) -> int:
        return max(1, round(self.delta_t_s * self.fps))

    @property
    def relax_frames(self) -> int:
        return max(1, round(self.T_relax_s * self.fps))

    @property
    def protocol_frames(self) -> int:
        """Frames the fired protocol itself occupies (protocol_dt_s at fps)."""
        return max(1, round(self.protocol_dt_s * self.fps))

# ── Symbol / state decoding ───────────────────────────────────────────────────
def decode_symbol(x_nm: float, xth_nm: float) -> int:
    """Threshold one position sample into {-1, 0, 1}."""
    if x_nm < -xth_nm:
        return -1
    if x_nm > xth_nm:
        return 1
    return 0

def decode_state_pair(x_prev_nm: float, x_curr_nm: float, xth_nm: float) -> tuple[int, int]:
    """Map two consecutive thresholded samples to a 9-state pair."""
    return decode_symbol(x_prev_nm, xth_nm), decode_symbol(x_curr_nm, xth_nm)

# ── State labels (for plotting / logging) ─────────────────────────────────────
STATE_LABELS = {
    (0, 0): "00 (hold, 2dt, no fire)",
    (1, 1): "11 (exp)", (1, 0): "10 (-ramp)",
    (0, 1): "01 (+ramp)", (-1, 1): "-11 (-jump-exp)",
    (-1, -1): "-1-1 (mirror exp)", (-1, 0): "-10 (mirror -ramp)",
    (0, -1): "0-1 (mirror +ramp)", (1, -1): "1-1 (mirror -jump-exp)",
}

# ── pos[:,3] trigger-column encoding: all 9 states → codes 1-9, 0 = not a
# checkpoint frame. Written at EVERY checkpoint (decide or capture-only), so
# it can be directly cross-checked against the ai3 recording. ─────────────
STATE_TO_CODE = {
    (0, 0): 1,
    (1, 1): 2, (1, 0): 3, (0, 1): 4, (-1, 1): 5,
    (-1, -1): 6, (-1, 0): 7, (0, -1): 8, (1, -1): 9,
}
CODE_TO_STATE = {code: state for state, code in STATE_TO_CODE.items()}
CAPTURE_CODE = 10   # checkpoint that only recorded x_prev, no decision made
ZERO_FIRE_CODE = 11 # decoded (0,0) AND it fired (the rare null/control
                    # measurement, see SessionParams2Ch.fire_00_prob) -
                    # distinct from STATE_TO_CODE[(0,0)]=1 (an ordinary,
                    # non-fired (0,0) checkpoint, still the common case)

# ── Manual-test menu: digit key → state tuple ─────────────────────────────────
# 0-7 = the 8 real (non-00) states, in the order they were first specified
# (group A then group B); 8 = 00 (hold, no fire).
STATE_MENU = {
    '0': (1, 1),
    '1': (1, 0),
    '2': (0, 1),
    '3': (-1, 1),
    '4': (-1, -1),
    '5': (-1, 0),
    '6': (0, -1),
    '7': (1, -1),
    '8': (0, 0),
}

# ── Base protocol shapes (group A) — placeholders, relative nm vs time ───────
def _shape_11(duration_s: float, rate: int) -> np.ndarray:
    """11 — exp: 0 → +amplitude, exponential rise."""
    n = int(duration_s * rate)
    t = np.linspace(0, duration_s, n)
    tau = duration_s / 3.0
    rel = PLACEHOLDER_AMPLITUDE_NM * (1 - np.exp(-t / tau))
    rel[0] = 0.0
    return rel

def _shape_10(duration_s: float, rate: int) -> np.ndarray:
    """10 — -ramp: 0 → -0.6*amplitude, linear."""
    n = int(duration_s * rate)
    return np.linspace(0.0, -0.6 * PLACEHOLDER_AMPLITUDE_NM, n)

def _shape_01(duration_s: float, rate: int) -> np.ndarray:
    """01 — +ramp: 0 → +0.6*amplitude, linear."""
    n = int(duration_s * rate)
    return np.linspace(0.0, 0.6 * PLACEHOLDER_AMPLITUDE_NM, n)

def _shape_m1_1(duration_s: float, rate: int) -> np.ndarray:
    """-11 — -jump-exp: immediate negative jump, then exponential further negative."""
    n = int(duration_s * rate)
    t = np.linspace(0, duration_s, n)
    tau = duration_s / 3.0
    jump_nm = 0.25 * PLACEHOLDER_AMPLITUDE_NM
    rel = -jump_nm - (PLACEHOLDER_AMPLITUDE_NM - jump_nm) * (1 - np.exp(-t / tau))
    rel[0] = -jump_nm
    return rel

BASE_PROTOCOLS = {
    (1, 1): _shape_11,
    (1, 0): _shape_10,
    (0, 1): _shape_01,
    (-1, 1): _shape_m1_1,
}

# ── Protocol lookup (resolves group B via mirroring) ──────────────────────────
def get_protocol_nm(state: tuple[int, int], duration_s: float,
                    rate: int = 1000) -> np.ndarray:
    """Relative displacement in nm for a 9-state pair. (0,0) → zeros (no-op)."""
    if state == (0, 0):
        return np.zeros(int(duration_s * rate))
    if state in BASE_PROTOCOLS:
        return BASE_PROTOCOLS[state](duration_s, rate)
    mirror = (-state[0], -state[1])
    if mirror in BASE_PROTOCOLS:
        return -BASE_PROTOCOLS[mirror](duration_s, rate)
    raise ValueError(f"Unknown state: {state}")

# ── Fine/coarse channel split ──────────────────────────────────────────────────
def split_channels(waveform_nm: np.ndarray, ao1_offset_v: float,
                   ao0_min: float = AO0_MIN, ao0_max: float = AO0_MAX,
                   ao1_min: float = AO1_MIN, ao1_max: float = AO1_MAX
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Split a relative-displacement (nm) protocol waveform into ao0 (fine) and
    ao1 (coarse) channels.

    ao0 carries the excursion (displacement_v * 10, clipped to its raw
    range) and is reset to 0 on the last sample; ao1 holds ao1_offset_v for
    the whole run and then absorbs the *actual* (post-clip) excursion at the
    last sample, so ao0/10 + ao1 is continuous across the reset even if ao0
    clipped. ao1_new_offset_v is also clamped to [ao1_min, ao1_max] as a
    final safety bound (only engages at the extreme edge of stage travel;
    the caller should treat a clamp here as "protocol tried to move past the
    stage limit").

    Returns (ao0_waveform_v, ao1_waveform_v, ao1_new_offset_v).
    """
    displacement_v = waveform_nm / NM_PER_VOLT
    ao0_wave = np.clip(displacement_v * 10.0, ao0_min, ao0_max)
    ao1_wave = np.full(len(waveform_nm), ao1_offset_v, dtype=np.float64)

    last_displacement_v = ao0_wave[-1] / 10.0   # post-clip, for continuity
    ao0_wave[-1] = 0.0
    ao1_new_offset_v = np.clip(ao1_offset_v + last_displacement_v, ao1_min, ao1_max)
    ao1_wave[-1] = ao1_new_offset_v

    return ao0_wave, ao1_wave, ao1_new_offset_v

# ── Offline self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    duration_s, rate = 3.0, 1000
    t = np.linspace(0, duration_s, int(duration_s * rate))

    group_a = [(1, 1), (1, 0), (0, 1), (-1, 1)]
    group_b = [(-1, -1), (-1, 0), (0, -1), (1, -1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for state in group_a:
        axes[0].plot(t, get_protocol_nm(state, duration_s, rate),
                     label=STATE_LABELS[state])
    axes[0].set_title("Group A (base protocols)")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Relative displacement (nm)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for state in group_b:
        axes[1].plot(t, get_protocol_nm(state, duration_s, rate),
                     label=STATE_LABELS[state])
    axes[1].set_title("Group B (mirrored)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nWorked example - split_channels() continuity check:")
    waveform_nm = get_protocol_nm((1, 1), duration_s, rate)
    ao1_start = 5.0
    ao0_wave, ao1_wave, ao1_new = split_channels(waveform_nm, ao1_start)
    combined_before_last = ao0_wave[-2] / 10.0 + ao1_wave[-2]
    combined_at_last      = ao0_wave[-1] / 10.0 + ao1_wave[-1]
    print(f"  ao1 offset:  {ao1_start:.4f} V  ->  {ao1_new:.4f} V")
    print(f"  ao0 last two samples: [{ao0_wave[-2]:.4f}, {ao0_wave[-1]:.4f}] V "
          f"(resets to 0)")
    print(f"  combined output just before reset: {combined_before_last:.4f} V")
    print(f"  combined output at reset sample:   {combined_at_last:.4f} V "
          f"(should match - continuous handoff)")

"""
protocols.py
============
Pure numpy — no hardware. Waveform shapes, state decoding, session parameters.
Testable fully offline.
"""

from dataclasses import dataclass, field
import numpy as np

# ── Stage constants ────────────────────────────────────────────────────────────
V_MIN       = 1.5       # V  →  4.5 µm
V_MAX       = 8.5       # V  → 25.5 µm
V_CENTER    = 5.0       # V  → 15.0 µm  (resting)
NM_PER_VOLT = 3000.0    # nm per volt

# ── Session parameters (equivalent to pExp in MATLAB) ─────────────────────────
@dataclass
class SessionParams:
    trecord_s    : float          # total session duration (s)  e.g. 3600*2
    protocol_dt_s: float          # duration of one AO protocol waveform (s)  e.g. 3.0
    fps          : float          # camera frame rate (Hz)
    exposure_us  : float          # camera exposure time (µs)
    trigger_nm   : float = 50.0   # particle position threshold to fire a protocol (nm)
    ai_rate      : int   = 3000   # AI sampling rate (S/s)
    ao_rate      : int   = 1000   # AO sampling rate (S/s)
    savepath     : str   = ""     # path prefix for saved files

    @property
    def n_frames(self) -> int:
        """Total camera frames for the full session."""
        return int(self.trecord_s * self.fps)

    @property
    def n_protocol_samples_ao(self) -> int:
        return int(self.protocol_dt_s * self.ao_rate)

    @property
    def n_protocol_samples_ai(self) -> int:
        return int(self.protocol_dt_s * self.ai_rate)

# ── Protocol state names ───────────────────────────────────────────────────────
STATE_NAMES = {
    0b00: "hold",
    0b01: "+ramp",
    0b10: "-ramp",
    0b11: "exp",
}

# ── Waveform shapes ────────────────────────────────────────────────────────────
def make_protocol(protocol_dt_s: float, state: int,
                  ao_rate: int = 1000) -> np.ndarray:
    """
    Relative displacement in volts from current stage position.
    Caller adds current_voltage to get absolute AO output.
    """
    n = int(protocol_dt_s * ao_rate)
    t = np.linspace(0, protocol_dt_s, n)

    if state == 0b00:
        rel_nm = np.zeros(n)
    elif state == 0b01:
        rel_nm = np.concatenate([[25.0], np.linspace(25.0, 120.0, n - 1)])
    elif state == 0b10:
        rel_nm = np.concatenate([[-25.0], np.linspace(-25.0, 50.0, n - 1)])
    elif state == 0b11:
        tau = protocol_dt_s / 3.0
        rel_nm = 150.0 * (1 - np.exp(-t / tau))
        rel_nm[0] = 0.0
    else:
        raise ValueError(f"Unknown state: {state}")

    return rel_nm / NM_PER_VOLT

def build_waveform(state: int, current_v: float,
                   protocol_dt_s: float, ao_rate: int = 1000) -> np.ndarray:
    """Absolute voltage waveform, clipped to safe stage range."""
    return np.clip(
        current_v + make_protocol(protocol_dt_s, state, ao_rate),
        V_MIN, V_MAX,
    )

# ── Trigger condition ──────────────────────────────────────────────────────────
def decode_state(position_nm: tuple[float, float]) -> int:
    """
    Map particle (x, y) position to protocol state.
    Adjust thresholds to your experiment geometry.
    """
    x_nm, y_nm = position_nm
    return (int(x_nm > 0) << 1) | int(y_nm > 0)

def should_trigger(position_nm: tuple[float, float],
                   trigger_nm: float) -> bool:
    """
    Fire a protocol when the particle is displaced beyond trigger_nm from origin.
    Replace with your specific trigger condition.
    """
    x, y = position_nm
    return np.sqrt(x**2 + y**2) > trigger_nm

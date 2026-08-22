"""
daq.py
======
NICardOut  — AO1 (piezo voltage) + Port0/Line0 (hardware trigger pulse to funcgen)
NICardIn   — Multi-channel continuous AI recording with sample-count tracking

NICardOut:  task configured once; per-trigger cost = stop / write / start
NICardIn:   background thread drains hardware buffer every READ_N samples;
            get_sample_count() returns exact AI sample index at any moment
            (used to timestamp each trigger in the AI stream for post-hoc slicing)
"""

import threading
import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

from protocols_2ch import NM_PER_VOLT

DEVICE = "Dev4"


class NICardOut:
    """
    AO1  — piezo stage voltage (pre-configured FINITE task, reused per trigger)
    Port0/Line0 — digital pulse to function generator  (MATLAB OCard.hardwaretrigg())
    """

    def __init__(self, n_samples: int, rate: int = 1000,
                 v_min: float = 1.5, v_max: float = 8.5):
        self._rate     = rate
        self._n        = n_samples
        self._duration = n_samples / rate

        # AO task — channel + timing configured once for the whole session
        self._ao = nidaqmx.Task()
        self._ao.ao_channels.add_ao_voltage_chan(
            f"{DEVICE}/ao1", min_val=v_min, max_val=v_max
        )
        self._ao.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )

        # DO task — single line for hardware trigger
        self._do = nidaqmx.Task()
        self._do.do_channels.add_do_chan(f"{DEVICE}/port0/line0")

    def load(self, waveform: np.ndarray) -> None:
        """Pre-load waveform into the AO buffer without starting."""
        self._ao.stop()
        self._ao.write(waveform, auto_start=False)

    def start(self) -> None:
        """Start AO output — non-blocking, runs in parallel with camera + AI."""
        self._ao.start()

    def wait_done(self, extra_s: float = 5.0) -> None:
        """Block until the FINITE waveform has finished playing."""
        self._ao.wait_until_done(timeout=self._duration + extra_s)

    def hardware_trigger(self) -> None:
        """
        Pulse Port0/Line0 HIGH then LOW.
        Triggers the function generator burst → camera starts.
        Equivalent to MATLAB OCard.hardwaretrigg().
        """
        self._do.write(True)
        self._do.write(False)

    def set_dc(self, voltage: float) -> None:
        """Hold stage at a fixed voltage (park / re-center between protocols)."""
        v = np.full(self._n, voltage, dtype=np.float64)
        self._ao.stop()
        self._ao.write(v, auto_start=True)
        self._ao.wait_until_done(timeout=self._duration + 5.0)

    def close(self) -> None:
        self._ao.close()
        self._do.close()


class NICardOutDual:
    """
    ao0 (fine) + ao1 (coarse) — piezo stage voltage via the two-channel
    adder circuit (ao0/10 + ao1). One pre-configured FINITE task, shared
    sample clock, reused per trigger via stop/write/start — same interface
    as NICardOut, extended to two channels. Kept as a separate class (not a
    rewrite of NICardOut) so the existing single-channel session code keeps
    working unmodified.

    Port0/Line0 — digital pulse to function generator, same as NICardOut.
    """

    def __init__(self, n_samples: int, rate: int = 1000,
                 ao0_min: float = -10.0, ao0_max: float = 10.0,
                 ao1_min: float = 1.5, ao1_max: float = 8.5):
        self._rate     = rate
        self._n        = n_samples
        self._duration = n_samples / rate

        # AO task — both channels + timing configured once for the whole session.
        # Write order must match add order: ao0 first, ao1 second.
        self._ao = nidaqmx.Task()
        self._ao.ao_channels.add_ao_voltage_chan(
            f"{DEVICE}/ao0", min_val=ao0_min, max_val=ao0_max
        )
        self._ao.ao_channels.add_ao_voltage_chan(
            f"{DEVICE}/ao1", min_val=ao1_min, max_val=ao1_max
        )
        self._ao.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )

        # DO task — single line for hardware trigger
        self._do = nidaqmx.Task()
        self._do.do_channels.add_do_chan(f"{DEVICE}/port0/line0")

    def load(self, ao0_waveform: np.ndarray, ao1_waveform: np.ndarray) -> None:
        """Pre-load an (ao0, ao1) waveform pair into the AO buffer without starting."""
        self._ao.stop()
        self._ao.write(np.vstack([ao0_waveform, ao1_waveform]), auto_start=False)

    def start(self) -> None:
        """Start AO output — non-blocking, runs in parallel with camera + AI."""
        self._ao.start()

    def wait_done(self, extra_s: float = 5.0) -> None:
        """Block until the FINITE waveform has finished playing."""
        self._ao.wait_until_done(timeout=self._duration + extra_s)

    def hardware_trigger(self) -> None:
        """Pulse Port0/Line0 HIGH then LOW — triggers the function generator burst."""
        self._do.write(True)
        self._do.write(False)

    def set_dc(self, ao0_v: float, ao1_v: float) -> None:
        """Hold both channels at fixed voltages (park between sessions)."""
        ao0 = np.full(self._n, ao0_v, dtype=np.float64)
        ao1 = np.full(self._n, ao1_v, dtype=np.float64)
        self._ao.stop()
        self._ao.write(np.vstack([ao0, ao1]), auto_start=True)
        self._ao.wait_until_done(timeout=self._duration + 5.0)

    def _run_waveform(self, ao0_wave: np.ndarray, ao1_wave: np.ndarray) -> None:
        """
        Shared machinery for ramp_ao1() / ramp_ao0() / run_custom_waveform():
        temporarily reconfigures this task's own sample-clock timing to
        len(ao0_wave) samples (stop -> reconfigure -> write -> start ->
        wait), then restores the standard protocol-length timing so
        load()/start() keep working unchanged afterward. Reuses the same
        task/channel reservation rather than opening a second one. Blocks
        until the waveform finishes playing.
        """
        n = len(ao0_wave)
        duration_s = n / self._rate

        self._ao.stop()
        self._ao.timing.cfg_samp_clk_timing(
            rate=self._rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=n
        )
        self._ao.write(np.vstack([ao0_wave, ao1_wave]), auto_start=False)
        self._ao.start()
        self._ao.wait_until_done(timeout=duration_s + 10.0)

        # Restore standard protocol-length timing for subsequent load()/start()
        self._ao.stop()
        self._ao.timing.cfg_samp_clk_timing(
            rate=self._rate, sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self._n,
        )

    def ramp_ao1(self, from_v: float, to_v: float,
                 speed_nm_per_s: float = 10.0) -> None:
        """
        Slowly ramp ao1 from from_v to to_v at speed_nm_per_s (converted via
        NM_PER_VOLT), ao0 held at 0 throughout — for moves where a bare
        voltage step would be unsafe (e.g. risks losing a trapped particle).
        Blocks until the move completes. See _run_waveform() for the
        reconfigure/restore mechanics.
        """
        distance_nm = abs(to_v - from_v) * NM_PER_VOLT
        duration_s  = max(distance_nm / speed_nm_per_s, 0.001)
        n = max(2, int(duration_s * self._rate))

        ao0 = np.zeros(n, dtype=np.float64)
        ao1 = np.linspace(from_v, to_v, n)
        self._run_waveform(ao0, ao1)

    def ramp_ao0(self, ao1_hold_v: float, from_v: float, to_v: float,
                speed_nm_per_s: float = 10.0) -> None:
        """
        Slowly ramp ao0 from from_v to to_v, ao1 held constant at
        ao1_hold_v throughout (the caller's current resting voltage —
        NOT forced to a fixed default, so the stage doesn't jump).
        Mirrors ramp_ao1() exactly, with the varying/held channels
        swapped.

        speed_nm_per_s only paces how slowly the ramp runs (via the
        nominal NM_PER_VOLT/10 attenuated-channel estimate) — it does not
        assume ao0's real transfer function, which is what this ramp is
        typically used to measure in the first place.
        """
        distance_nm = abs(to_v - from_v) * (NM_PER_VOLT / 10.0)
        duration_s  = max(distance_nm / speed_nm_per_s, 0.001)
        n = max(2, int(duration_s * self._rate))

        ao0 = np.linspace(from_v, to_v, n)
        ao1 = np.full(n, ao1_hold_v, dtype=np.float64)
        self._run_waveform(ao0, ao1)

    def run_custom_waveform(self, ao0_wave: np.ndarray, ao1_wave: np.ndarray) -> None:
        """
        Play an arbitrary-shaped (ao0, ao1) waveform pair (e.g. a sine
        sweep, not just a straight ramp) — generalizes ramp_ao0()/
        ramp_ao1() to any shape. Both arrays must be the same length;
        duration is len(ao0_wave)/rate. Blocks until finished, then
        restores standard protocol-length timing (see _run_waveform()).
        """
        if len(ao0_wave) != len(ao1_wave):
            raise ValueError("ao0_wave and ao1_wave must be the same length")
        self._run_waveform(np.asarray(ao0_wave, dtype=np.float64),
                           np.asarray(ao1_wave, dtype=np.float64))

    def close(self) -> None:
        self._ao.close()
        self._do.close()


def read_ai_channel(name: str, rate: int = 1000, n_samples: int = 200) -> float:
    """
    One-shot short read (mean of n_samples at rate S/s) of a single AI
    channel, by name — one of NICardIn.CHANNELS: "AOM", "Stage", "CamOut",
    "AdderOut". Independent of NICardIn's continuous background recording
    (no session/thread needed) — for a quick single-value check, e.g.
    "what is ai3 sitting at right now".
    """
    ch = next((c for c in NICardIn.CHANNELS if c["name"] == name), None)
    if ch is None:
        valid = [c["name"] for c in NICardIn.CHANNELS]
        raise ValueError(f"Unknown AI channel {name!r} - choose from {valid}")

    with nidaqmx.Task() as ai:
        ai.ai_channels.add_ai_voltage_chan(
            ch["chan"], terminal_config=ch["term"], min_val=-10.0, max_val=10.0,
        )
        ai.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples
        )
        ai.start()
        samples = ai.read(number_of_samples_per_channel=n_samples, timeout=5.0)
    return float(np.mean(samples))


def read_stage_voltage(rate: int = 1000, n_samples: int = 200) -> float:
    """
    One-shot short read of the Stage (ai1) channel to find out what ao1 is
    currently sitting at — the DAQ device doesn't otherwise expose this
    (e.g. it could be left over from a prior process), so this is how a
    caller decides whether ramp_ao1() is even needed before starting.
    Thin wrapper around read_ai_channel("Stage", ...).
    """
    return read_ai_channel("Stage", rate, n_samples)


class NICardIn:
    """
    Continuous multi-channel AI recording for an entire session.

    A background thread drains the hardware buffer every READ_N samples
    and appends to an in-memory list.  get_sample_count() returns the
    number of samples collected so far — call this immediately when a
    trigger fires to record the exact AI sample index for post-hoc slicing.

    Channel order: AOM (ai0, RSE), Stage (ai1, DIFF), CamOut (ai2, RSE),
    AdderOut (ai3, DIFF — the ao0/10+ao1 adder-circuit output).
    """

    READ_N = 1000   # samples drained per poll (= 1/3 s at 3000 S/s)

    CHANNELS = [
        {"name": "AOM",      "chan": f"{DEVICE}/ai0", "term": TerminalConfiguration.RSE},
        {"name": "Stage",    "chan": f"{DEVICE}/ai1", "term": TerminalConfiguration.DIFF},
        {"name": "CamOut",   "chan": f"{DEVICE}/ai2", "term": TerminalConfiguration.RSE},
        {"name": "AdderOut", "chan": f"{DEVICE}/ai3", "term": TerminalConfiguration.DIFF},
    ]

    def __init__(self, rate: int = 3000):
        self.rate          = rate
        self.channel_names = [ch["name"] for ch in self.CHANNELS]
        self._data         : list[list] = [[] for _ in self.CHANNELS]
        self._sample_count : int        = 0
        self._lock         = threading.Lock()
        self._stop         = threading.Event()
        self._thread       : threading.Thread | None = None

    def start_continuous(self) -> None:
        """
        Start background AI recording. Call once per session/experiment.

        Resets any samples retained from a previous start_continuous() /
        stop_and_get() cycle — required when the same NICardIn instance is
        reused across multiple experiments (e.g. run_batch_2ch): without
        this, each new experiment's ai_data silently accumulated on top of
        every prior experiment's samples instead of starting fresh.
        """
        with self._lock:
            self._data = [[] for _ in self.CHANNELS]
            self._sample_count = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        with nidaqmx.Task() as ai:
            for ch in self.CHANNELS:
                ai.ai_channels.add_ai_voltage_chan(
                    ch["chan"],
                    name_to_assign_to_channel=ch["name"],
                    terminal_config=ch["term"],
                    min_val=-10.0,
                    max_val=10.0,
                )
            ai.timing.cfg_samp_clk_timing(
                rate=self.rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
            ai.start()

            while not self._stop.is_set():
                avail = ai.in_stream.avail_samp_per_chan
                if avail >= self.READ_N:
                    chunk = ai.read(
                        number_of_samples_per_channel=self.READ_N,
                        timeout=2.0,
                    )
                    with self._lock:
                        for i, ch_data in enumerate(chunk):
                            self._data[i].extend(ch_data)
                        self._sample_count += self.READ_N
                else:
                    time.sleep(self.READ_N / self.rate / 4)

            # Drain remaining samples when stopping
            avail = ai.in_stream.avail_samp_per_chan
            if avail > 0:
                chunk = ai.read(
                    number_of_samples_per_channel=avail,
                    timeout=2.0,
                )
                with self._lock:
                    for i, ch_data in enumerate(chunk):
                        self._data[i].extend(ch_data)
                    self._sample_count += avail

    def get_sample_count(self) -> int:
        """
        Current number of AI samples collected.
        Call immediately when a trigger fires — this is the AI sample index
        that maps the trigger event into the continuous AI recording.
        """
        with self._lock:
            return self._sample_count

    def close(self) -> None:
        """
        No persistent task to close — the AI task lives inside the
        background thread's `with nidaqmx.Task()` block and closes itself
        when the thread exits. This is a safety net for callers that clean
        up all their hardware resources uniformly (e.g. a session that
        failed before stop_and_get() was ever called): stops the thread if
        it's still running.
        """
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=5.0)

    def stop_and_get(self) -> np.ndarray:
        """
        Stop background recording and return all data.
        Returns shape (n_channels, n_samples).
        Row order: AOM, Stage, CamOut.
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._lock:
            return np.array(self._data)

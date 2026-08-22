"""
function_gen.py
===============
Tektronix AFG3000 via SCPI over USB-VISA.
Equivalent to MATLAB FunctionGenCam_L.

setup_camera_trigger() configures CH2 for the FULL session:
  n_frames = fps * trecord_s  pulses in one burst.
After NICardOut.hardware_trigger() fires, the funcgen runs autonomously
for the entire session — no further interaction needed.
"""

import pyvisa

USB_ID = "USB0::0x0699::0x0358::C018956::INSTR"


class FunctionGenerator:
    """Tektronix AFG3000 SCPI wrapper."""

    def __init__(self, usb_id: str = USB_ID):
        rm = pyvisa.ResourceManager()
        self.dev = rm.open_resource(usb_id)
        self.dev.timeout = 5000     # ms

    def setup_camera_trigger(self, fps: float, exposure_us: float,
                              n_frames: int, ch: int = 2) -> None:
        """
        Configure CH2 as a pulse train for camera triggering for the full session.
        n_frames = int(fps * trecord_s)  — total frames for the whole session.

        Burst mode: fires n_frames pulses after one external trigger from DAQ.
        Mirrors the CH2 block in BaslerCamera_L.setupLive().
        """
        width_s = exposure_us * 1e-6

        cmds = [
            f":OUTPut{ch}:STATe OFF",
            f"*RST",                                                    # ← full reset first
            f":OUTPut{ch}:IMPedance HIGHz",                            # ← impedance before levels
            f":SOURce{ch}:FUNCtion:SHAPe PULSe",
            f":SOURce{ch}:FREQuency:FIXed {fps:.6f}",
            f":SOURce{ch}:PULSe:WIDTh {width_s:.9f}",
            f":SOURce{ch}:PULSe:LEADing 7.0E-9",
            f":SOURce{ch}:PULSe:TRAiling 7.0E-9",
            f":SOURce{ch}:VOLTage:LEVel:IMMediate:AMPLitude 5.0",      # ← levels after impedance
            f":SOURce{ch}:VOLTage:LEVel:IMMediate:OFFSet 2.5",
            f":SOURce{ch}:BURSt:STATe ON",
            f":SOURce{ch}:BURSt:NCYCles {int(n_frames)}",
            ":TRIGger:SEQuence:SOURce EXTernal",
            ":TRIGger:SEQuence:SLOPe POSitive",
            f":OUTPut{ch}:STATe ON",
        ]
        for cmd in cmds:
            self.dev.write(cmd)

    def set_dc_level(self, voltage: float, ch: int = 2) -> None:
        """
        Configure `ch` as a DC output at `voltage`. Mirrors the MATLAB
        FunctionGen_shutter_stage AOM-power block exactly (Output Off ->
        Waveform=DC -> Low/High level -> Output On) rather than switching
        to a different SCPI DC mechanism (e.g. VOLTage:OFFSet) not
        verified against this hardware. Low/High are set to voltage
        +/-0.1V, same as that MATLAB code.
        """
        cmds = [
            f":OUTPut{ch}:STATe OFF",
            f":SOURce{ch}:FUNCtion:SHAPe DC",
            f":SOURce{ch}:VOLTage:LEVel:IMMediate:LOW {voltage - 0.1:.4f}",
            f":SOURce{ch}:VOLTage:LEVel:IMMediate:HIGH {voltage + 0.1:.4f}",
            f":OUTPut{ch}:STATe ON",
        ]
        for cmd in cmds:
            self.dev.write(cmd)

    def output_on(self, ch: int = 2) -> None:
        self.dev.write(f":OUTPut{ch}:STATe ON")

    def output_off(self, ch: int = 2) -> None:
        self.dev.write(f":OUTPut{ch}:STATe OFF")

    def query(self, cmd: str) -> str:
        return self.dev.query(cmd)

    def close(self) -> None:
        self.dev.close()

"""
camera.py  —  BaslerCamera
==========================
Python equivalent of BaslerCamera_L.m

Mirrors MATLAB usage exactly:

    cam = BaslerCamera()          # open once, stays open
    cam.exposure(500)
    cam.roi(32, 32, 256, 256)     # [xOffset, yOffset, width, height]  same as MATLAB ROIPosition
    cam.framerate(100)
    cam.startview()               # live preview window, background thread
    cam.stopview()                # close preview

    cam.hardwareTrigg()           # configure Line2 trigger
    cam.start()                   # begin grabbing
    frame      = cam.grab()       # one frame per trigger pulse
    x, y       = cam.find(frame)  # sub-pixel particle position in nm
    cam.stop()

    cam.close()                   # call when completely done
"""

import threading
import numpy as np
import cv2
from pypylon import pylon, genicam
from scipy.ndimage import gaussian_filter


NM_PER_PX = 16   # update after calibration

# ---------------------------------------------------------------------------
# Parameters, same names as in Track.m
# ---------------------------------------------------------------------------

GAUSSIAN_KERNEL_SIZE = 60     # gaussianKernelSize
THRES = 50.0                  # thres
SIZE = 9                      # size   (must be odd)

def _build_kernel(n=GAUSSIAN_KERNEL_SIZE):
    """Track.m lines:
           r2 = linspace(-1,1,n).^2
           R2 = repmat(r2,n,1) + repmat(r2',1,n)
           filterKernel = exp(-R2)/sum(sum(exp(-R2)))

    exp(-(x^2+y^2)) = exp(-x^2) * exp(-y^2), so the 2D kernel is just the
    outer product of a 1D kernel with itself. That lets us filter rows and
    columns separately, which is what makes it fast.

    The width in pixels is (n-1)/(2*sqrt(2)):  n=60 -> sigma = 20.9 px.
    """
    g = np.exp(-np.linspace(-1.0, 1.0, n) ** 2)
    return (g / g.sum()).astype(np.float32)


K1D = _build_kernel()

# MATLAB puts the kernel origin at floor((n+1)/2). For an even n like 60 that
# is half a pixel off the true centre, so every position Track.m reports is
# shifted by ~0.5 px. OpenCV's default origin for a 60-tap kernel is a
# different pixel again, so it has to be set by hand.
ANCHOR = (GAUSSIAN_KERNEL_SIZE + 1) // 2 - 1


# ── Particle detection ────────────────────────────────────────────────────────
def _imfilter(im):
    """MATLAB: imfilter(double(frame), filterKernel).

    imfilter pads with zeros by default, hence BORDER_CONSTANT.
    """
    return cv2.sepFilter2D(im, cv2.CV_32F, K1D, K1D,
                           anchor=(ANCHOR, ANCHOR),
                           borderType=cv2.BORDER_CONSTANT)

def _smooth_window(im, cx, cy, half):
    """Blur only a window around (cx, cy), but grow the crop by the full
    kernel length first so the result inside the window is identical to
    blurring the whole frame. Anything outside the image is filled with zeros,
    which is what imfilter does.

    Returns the blurred window and the image coordinates of its top-left pixel.
    """
    n = GAUSSIAN_KERNEL_SIZE
    nr, nc = im.shape
    x0, x1 = cx - half - n, cx + half + n + 1
    y0, y1 = cy - half - n, cy + half + n + 1
    sx0, sy0 = max(0, x0), max(0, y0)
    crop = im[sy0:min(nr, y1), sx0:min(nc, x1)]
    if crop.shape != (y1 - y0, x1 - x0):
        pad = np.zeros((y1 - y0, x1 - x0), np.float32)
        pad[sy0 - y0:sy0 - y0 + crop.shape[0], sx0 - x0:sx0 - x0 + crop.shape[1]] = crop
        crop = pad
    sm = _imfilter(crop)
    return sm[n:sm.shape[0] - n, n:sm.shape[1] - n], x0 + n, y0 + n


def find_particle_fast(im, prev=None, roi=96, thres=THRES, matlab_indexing=True):
    """Drop-in replacement for find_particle. Returns (x, y), same order as
    Track.m, or (nan, nan) when nothing is above threshold.

    Pass the previous frame's result as `prev` so only a small window has to be
    blurred. On the first frame, or after a loss, pass None and it searches the
    whole frame.
    """
    if im is None or im.ndim != 2 or im.size == 0:
        return np.nan, np.nan
    im = im.astype(np.float32, copy=False)

    if prev is None or not np.isfinite(prev).all():
        sm, ox, oy = _imfilter(im), 0, 0
    else:
        px = int(round(prev[0])) - (1 if matlab_indexing else 0)
        py = int(round(prev[1])) - (1 if matlab_indexing else 0)
        sm, ox, oy = _smooth_window(im, px, py, roi // 2)

    yl, xl = np.unravel_index(np.argmax(sm), sm.shape)
    if sm[yl, xl] <= thres:                       # bead lost, like Track.m
        return np.nan, np.nan
    if xl < 2 or xl > sm.shape[1] - 3 or yl < 2 or yl > sm.shape[0] - 3:
        if prev is not None:                      # drifted out of the window
            return find_particle_fast(im, None, roi, thres, matlab_indexing)
        return np.nan, np.nan

    # same 5-point parabola fit your colleague already uses
    c1 = np.array([1, -8, 0, 8, -1], dtype=np.float32)
    c2 = np.array([-1, 16, -30, 16, -1], dtype=np.float32)
    xw = sm[yl, xl - 2:xl + 3]; yw = sm[yl - 2:yl + 3, xl]
    dx, d2x = float(xw @ c1), float(xw @ c2)
    dy, d2y = float(yw @ c1), float(yw @ c2)
    xf = xl if d2x >= 0 else xl - dx / d2x
    yf = yl if d2y >= 0 else yl - dy / d2y

    x, y = xf + ox, yf + oy
    return (x + 1.0, y + 1.0) if matlab_indexing else (x, y)

# def _find_particle(im, sigma=2.0, crop_r = 15):
#     if im is None or im.ndim != 2 or im.size == 0:
#         return np.nan, np.nan
#     y0, x0 = np.unravel_index(np.argmax(im), im.shape)
#     ys = max(0, y0 - crop_r);  ye = min(im.shape[0], y0 + crop_r + 1)
#     xs = max(0, x0 - crop_r);  xe = min(im.shape[1], x0 + crop_r + 1)
#     crop = np.asarray(im[ys:ye, xs:xe], dtype=np.float32)
#     if crop.size == 0:
#         return np.nan, np.nan
#     sm = gaussian_filter(crop, sigma=sigma)
#     yl, xl = np.unravel_index(np.argmax(sm), sm.shape)
#     if xl < 2 or xl > sm.shape[1]-3 or yl < 2 or yl > sm.shape[0]-3:
#         return float(ys+yl), float(xs+xl)
#     c1 = np.array([ 1,-8, 0, 8,-1], dtype=np.float32)
#     c2 = np.array([-1,16,-30,16,-1], dtype=np.float32)
#     xw = sm[yl, xl-2:xl+3];  yw = sm[yl-2:yl+3, xl]
#     dx, d2x = np.dot(xw,c1), np.dot(xw,c2)
#     dy, d2y = np.dot(yw,c1), np.dot(yw,c2)
#     xf = float(xl) if d2x >= 0 else float(xl) - dx/d2x
#     yf = float(yl) if d2y >= 0 else float(yl) - dy/d2y
#     return float(ys+yf), float(xs+xf)


# ── BaslerCamera ──────────────────────────────────────────────────────────────

class BaslerCamera:

    def __init__(self, exposure_us: float = 500.0):
        self._cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self._cam.Open()

        # Line3 → ExposureActive strobe → AI2 CamOut  (same as MATLAB)
        self._cam.LineSelector.Value = "Line3"
        self._cam.LineMode.Value     = "Output"
        self._cam.LineSource.Value   = "ExposureActive"
        self._cam.LineInverter.Value = True          # same as MATLAB LineInverter=True

        self._cam.AcquisitionMode.Value = "Continuous"
        self._cam.MaxNumBuffer          = 10

        self._preview_thread = None
        self._preview_stop   = threading.Event()
        self._preview_fps    = 30.0
        self._preview_track_particle = False

        # find()'s windowed search state (find_particle_fast's `prev`) — lets
        # consecutive find() calls only blur a small region around the last
        # detection instead of the whole frame. NaN means "search whole
        # frame" (first call, or after losing the particle).
        self._prev_particle_px = (np.nan, np.nan)

        self.exposure(exposure_us)
        print("[Camera] Opened")

    # ── Parameters — mirror MATLAB device.X = value ──────────────────────────

    def exposure(self, value):
        """
        exposure(500)       → set exposure time in µs
        exposure('Once')    → auto exposure once
        exposure('Continuous') → continuous auto
        """
        if isinstance(value, str):
            self._cam.ExposureAuto.Value = value
        else:
            self._cam.ExposureAuto.Value  = "Off"
            self._cam.ExposureTime.Value  = float(value)
        print(f"[Camera] Exposure = {value}")


    def gamma(self, value: float):
        """Gamma correction. Typical range 0.0-3.99. Set to 1.0 to disable."""
        # Some Basler models expose Gamma directly but do not provide GammaEnable.
        nodemap = self._cam.GetNodeMap()

        try:
            gamma_enable_node = nodemap.GetNode("GammaEnable")
        except genicam.LogicalErrorException:
            gamma_enable_node = None
        if gamma_enable_node is not None and genicam.IsWritable(gamma_enable_node):
            gamma_enable_node.SetValue(True)

        try:
            gamma_node = nodemap.GetNode("Gamma")
        except genicam.LogicalErrorException:
            gamma_node = None
        if gamma_node is None or not genicam.IsWritable(gamma_node):
            print("[Camera] Gamma control not available on this camera")
            return

        gamma_node.SetValue(float(value))
        print(f"[Camera] Gamma = {value}")

    def framerate(self, fps: float):
        """Set free-run frame rate. Mirrors MATLAB obj.device.AcquisitionFrameRate = fps."""
        self._cam.AcquisitionFrameRateEnable.Value = True
        self._cam.AcquisitionFrameRate.Value       = float(fps)
        self._preview_fps = float(fps)
        print(f"[Camera] Framerate = {fps} fps")

    def gain(self, gain_db: float):
        self._cam.Gain.Value = float(gain_db)
        print(f"[Camera] Gain = {gain_db} dB")

    @staticmethod
    def _snap_int_node(node, value: int, mode: str = "nearest") -> int:
        """Snap an integer value to the node's [Min, Max] range and Inc grid."""
        vmin = int(node.GetMin())
        vmax = int(node.GetMax())
        inc = int(node.GetInc())

        v = int(round(value))
        v = max(vmin, min(v, vmax))

        if inc <= 1:
            return v

        if mode == "floor":
            steps = (v - vmin) // inc
        elif mode == "ceil":
            steps = (v - vmin + inc - 1) // inc
        else:
            steps = int(round((v - vmin) / inc))

        snapped = vmin + steps * inc
        snapped = max(vmin, min(snapped, vmax))
        return int(snapped)

    def roi(self, x_offset: int, y_offset: int, width: int, height: int):
        """
        roi(xOffset, yOffset, width, height)
        Same argument order as MATLAB:  vid.ROIPosition = [xOffset yOffset width height]
        Safe to call during preview — restarts grab transparently.
        """
        was = self._cam.IsGrabbing()
        if was:
            self._cam.StopGrabbing()

        # Width/Height limits are easiest to satisfy with offsets at zero.
        self._cam.OffsetX.Value = 0
        self._cam.OffsetY.Value = 0

        w = self._snap_int_node(self._cam.Width, width, mode="nearest")
        h = self._snap_int_node(self._cam.Height, height, mode="nearest")
        self._cam.Width.Value = w
        self._cam.Height.Value = h

        # Offset constraints depend on the selected ROI size.
        ox = self._snap_int_node(self._cam.OffsetX, x_offset, mode="nearest")
        oy = self._snap_int_node(self._cam.OffsetY, y_offset, mode="nearest")
        self._cam.OffsetX.Value = ox
        self._cam.OffsetY.Value = oy

        if was:
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(
            f"[Camera] ROI  {w}x{h}  offset ({ox},{oy})"
            f"  [requested {width}x{height} @ ({x_offset},{y_offset})]"
        )

    def full_roi(self):
        """Reset to full sensor. Equivalent to MATLAB ROI(1800,1500,0,0)."""
        # Read sensor max without touching OffsetX/Y — safe on open camera
        w = self._cam.Width.GetMax()
        h = self._cam.Height.GetMax()
        self.roi(0, 0, w, h)
        print(f"[Camera] Full sensor {w}×{h}")


    def pixel_format(self, fmt: str = "Mono8"):
        self._cam.PixelFormat.Value = fmt
        print(f"[Camera] Pixel format = {fmt}")

    # ── Trigger modes — mirror MATLAB exactly ────────────────────────────────

    def hardwareTrigg(self):
        """
        Hardware trigger on Line2 rising edge.
        Mirrors MATLAB obj.hardwareTrigg()

        Also sets TriggerOverlap to ReadOut/PreviousFrame if the camera
        supports it. The default (Off) makes the camera refuse the next
        trigger until the PREVIOUS frame's entire exposure+readout+transfer
        cycle has fully finished — capping hardware-triggered throughput at
        that full-cycle rate even though the sensor can go much faster in
        free-run (e.g. 200 fps free-run vs ~100 fps hardware-triggered on
        the same ROI is the textbook symptom). Not all models expose this
        node, so this is best-effort and won't raise if unsupported.
        """
        self._cam.AcquisitionFrameRateEnable.Value = False
        self._cam.LineSelector.Value   = "Line2"
        self._cam.LineMode.Value       = "Input"
        self._cam.TriggerMode.Value    = "On"
        self._cam.TriggerSource.Value  = "Line2"
        self._cam.TriggerActivation.Value = "RisingEdge"
        self._cam.TriggerSelector.Value   = "FrameStart"

        print("[Camera] Hardware trigger (Line2, rising edge)")

    def softwareTrigg(self):
        """
        Software / free-run trigger.
        Mirrors MATLAB obj.softwareTrigg()
        """
        self._cam.AcquisitionFrameRateEnable.Value = True
        self._cam.LineSelector.Value  = "Line2"
        self._cam.TriggerMode.Value   = "Off"
        self._cam.TriggerSource.Value = "Software"
        self._cam.TriggerSelector.Value = "FrameStart"
        print("[Camera] Software trigger")

    # ── Preview — mirrors MATLAB startview/stopview ───────────────────────────

    def startview(self, particle_cross: bool = False):
        """
        Open live preview window in background thread. Returns immediately.
        Mirrors MATLAB obj.startview() / preview(obj.vid)
        Call roi(), exposure(), framerate() freely while preview runs.

        Args:
            particle_cross: Draw a cross marker at detected particle location.
        """
        if self._preview_thread and self._preview_thread.is_alive():
            print("[Camera] Preview already running")
            return
        self._preview_track_particle = bool(particle_cross)
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True
        )
        self._preview_thread.start()
        print(f"[Camera] Preview started (particle_cross={self._preview_track_particle})")

    def stopview(self):
        """
        Stop preview window.
        Mirrors MATLAB obj.stopview() / stoppreview(obj.vid)
        """
        self._preview_stop.set()
        if self._preview_thread:
            self._preview_thread.join(timeout=3)
        cv2.destroyAllWindows()
        print("[Camera] Preview stopped")

    def _preview_loop(self):
        if self._cam.IsGrabbing():
            self._cam.StopGrabbing()
        # free-run for preview
        self._cam.TriggerMode.Value                = "Off"
        self._cam.AcquisitionFrameRateEnable.Value = True
        self._cam.AcquisitionFrameRate.Value       = self._preview_fps
        self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        while not self._preview_stop.is_set():
            res = self._cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if res and res.GrabSucceeded():
                frame = res.Array
                disp  = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if frame.ndim == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

                if self._preview_track_particle:
                    # Independent of find()'s own tracking state — preview is
                    # just a visual check, not the real measurement path — so
                    # this always does a fresh whole-frame search (prev=None).
                    x_px, y_px = find_particle_fast(frame, prev=None, matlab_indexing=False)
                    if not np.isnan(x_px):
                        px = int(round(x_px))
                        py = int(round(y_px))
                        cv2.drawMarker(
                            disp,
                            (px, py),
                            (0, 255, 0),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=16,
                            thickness=1,
                            line_type=cv2.LINE_AA,
                        )
                        
                cv2.imshow("Camera", disp)
                cv2.waitKey(1)
            if res:
                res.Release()

        self._cam.StopGrabbing()

    # ── Session control ───────────────────────────────────────────────────────

    def start(self):
        """Start grabbing. Mirrors MATLAB start(obj.vid)."""
        self._cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        print("[Camera] Started")

    def stop(self):
        """Stop grabbing. Mirrors MATLAB stop(obj.vid)."""
        if self._cam.IsGrabbing():
            self._cam.StopGrabbing()
        print("[Camera] Stopped")

    def close(self):
        """Close camera completely. Recreate object to reopen."""
        self.stopview()
        self.stop()
        if self._cam.IsOpen():
            self._cam.Close()
        cv2.destroyAllWindows()
        print("[Camera] Closed")

    # ── Grab + detect — hot path ──────────────────────────────────────────────

    def grab(self, timeout_ms: int = 2000) -> np.ndarray:
        """
        Grab one frame. Blocks until next hardware trigger pulse.
        Mirrors MATLAB getdata(obj.vid, 1)

        Always raises RuntimeError on failure (never a raw pylon/genicam
        exception) — a missed trigger pulse surfaces as
        genicam.TimeoutException from RetrieveResult itself, which is not
        a RuntimeError subclass, so it's caught and re-raised here to keep
        one exception type for every caller to catch.
        """
        try:
            res = self._cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        except genicam.TimeoutException as e:
            raise RuntimeError(f"[Camera] Grab timed out: {e}") from e
        if not res.GrabSucceeded():
            res.Release()
            raise RuntimeError(f"[Camera] Grab failed: {res.GetErrorDescription()}")
        frame = res.Array.copy()
        res.Release()
        return frame

    def find(self, frame: np.ndarray, nm_per_px: float = NM_PER_PX,
            roi: int = 96, thres: float = THRES):
        """
        Raw sub-pixel particle position, scaled to nm via nm_per_px — NOT
        centered on anything (not the ROI, not the trap). Pixel (0,0) is
        the top-left corner of the current frame/ROI, so this value shifts
        if the ROI offset changes, but is otherwise a direct, unambiguous
        reading with no hidden reference point.

        Centering (e.g. subtracting a calibrated trap center) is the
        caller's job — see experiment.py::calibrate_trap_center() /
        run_session_2ch(), which do exactly that for every other
        calculation (decode, potential analysis, ...) downstream of this.

        roi/thres are find_particle_fast()'s windowed-search size and
        detection threshold — exposed here so they're tunable per-call
        without editing this file. If the particle's frame-to-frame motion
        often exceeds roi/2, detections land near the window edge and
        find_particle_fast falls back to a full-frame search for that
        frame (much slower) — widening roi avoids that at some extra
        per-frame cost from the larger window itself.

        Returns (x_nm, y_nm).
        """
        x_px, y_px = find_particle_fast(
            frame, prev=self._prev_particle_px, roi=roi, thres=thres,
            matlab_indexing=False,
        )
        self._prev_particle_px = (x_px, y_px)
        if np.isnan(x_px):
            return np.nan, np.nan
        return x_px * nm_per_px, y_px * nm_per_px

    def show(self, frame: np.ndarray, x_nm=None, y_nm=None):
        """
        Display one frame with optional overlay. Non-blocking.
        Call inside experiment loop every frame or every Nth frame.
        """
        disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if frame.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        if x_nm is not None and not np.isnan(x_nm):
            cv2.putText(disp, f"x={x_nm:.1f} nm  y={y_nm:.1f} nm",
                        (8, disp.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("Camera", disp)
        cv2.waitKey(1)
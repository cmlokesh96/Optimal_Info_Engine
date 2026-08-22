"""
track_compare.py
================

Three functions, in this order:

    matlab_track(frame)              line-by-line translation of Track.m
    find_particle(im)                your colleague's function, unchanged
    find_particle_fast(im, prev)     the suggested replacement

plus compare_video(path) which runs all three on the same movie and prints
how far apart they are.

Only numpy + opencv are needed (scipy just for find_particle).

    python3 track_compare.py                 # self test on synthetic frames
    python3 track_compare.py my_movie.avi    # compare on real data
"""

import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parameters, same names as in Track.m
# ---------------------------------------------------------------------------

GAUSSIAN_KERNEL_SIZE = 60     # gaussianKernelSize
THRES = 80.0                  # thres
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


def _imfilter(im):
    """MATLAB: imfilter(double(frame), filterKernel).

    imfilter pads with zeros by default, hence BORDER_CONSTANT.
    """
    return cv2.sepFilter2D(im, cv2.CV_32F, K1D, K1D,
                           anchor=(ANCHOR, ANCHOR),
                           borderType=cv2.BORDER_CONSTANT)


# Track.m: aleatoire = 0.0001*rand(xs,ys), made once and reused every frame.
# It exists to break ties when the bead is saturated and many pixels share the
# same value.
_NOISE = {}


def _dither(shape):
    if shape not in _NOISE:
        rng = np.random.default_rng(0)
        _NOISE[shape] = (1e-4 * rng.random(shape)).astype(np.float32)
    return _NOISE[shape]


# ---------------------------------------------------------------------------
# 1. Track.m, translated
# ---------------------------------------------------------------------------


def pkfnd(im, th=THRES, sz=SIZE):
    """pkfnd.m: pixels above `th` that are >= all 8 neighbours, at least `sz`
    from the border, and the brightest within an `sz` box.
    Returns integer [x, y] pairs (column, row), 0-based."""
    above = im > th
    if not above.any():
        return np.zeros((0, 2), np.int32)

    neighbourhood_max = cv2.dilate(im, np.ones((3, 3), np.uint8))
    peaks = above & (im >= neighbourhood_max)
    peaks[0, :] = peaks[-1, :] = peaks[:, 0] = peaks[:, -1] = False

    ys, xs = np.nonzero(peaks)
    nr, nc = im.shape
    keep = (xs > sz) & (xs < nc - sz) & (ys > sz) & (ys < nr - sz)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return np.zeros((0, 2), np.int32)

    if ys.size > 1:                       # keep the brightest in each sz box
        sparse = np.zeros_like(im)
        sparse[ys, xs] = im[ys, xs]
        box = np.ones((2 * (sz // 2) + 1, 2 * (sz // 2) + 1), np.uint8)
        keep = sparse[ys, xs] >= cv2.dilate(sparse, box)[ys, xs]
        ys, xs = ys[keep], xs[keep]

    return np.stack([xs, ys], 1).astype(np.int32)


def cntrd(im, pks, sz=SIZE):
    """cntrd.m: intensity-weighted average inside a circle of diameter `sz`
    centred on each peak pixel. Returns float [x, y] pairs, 0-based."""
    if pks.size == 0:
        return np.zeros((0, 2))
    rad = sz // 2
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    mask = (np.hypot(xx, yy) < sz / 2.0).astype(np.float32)
    coord = np.arange(-rad, rad + 1, dtype=np.float32)
    nr, nc = im.shape

    out = []
    for x, y in pks:
        if not (rad <= x < nc - rad and rad <= y < nr - rad):
            continue
        w = im[y - rad:y + rad + 1, x - rad:x + rad + 1] * mask
        total = float(w.sum())
        if total <= 0:
            continue
        out.append((x + float((w.sum(0) * coord).sum() / total),
                    y + float((w.sum(1) * coord).sum() / total)))
    return np.asarray(out).reshape(-1, 2)


def matlab_track(frame, matlab_indexing=True):
    """Exactly what Track.m does to one frame. Returns (x, y) or (nan, nan).

    matlab_indexing=True adds 1 to both coordinates so the numbers match your
    saved .mat files (MATLAB counts pixels from 1, Python from 0).
    """
    if frame.ndim == 3:
        frame = frame.mean(2)             # Track.m has frame(:,:,3) commented out
    im = frame.astype(np.float32, copy=False) + _dither(frame.shape)
    sm = _imfilter(im)
    pks = pkfnd(sm)
    cnt = cntrd(sm, pks)
    if cnt.shape[0] == 0:
        return np.nan, np.nan
    if cnt.shape[0] > 1:                  # brightest one
        cnt = cnt[[int(np.argmax([sm[int(round(y)), int(round(x))] for x, y in cnt]))]]
    x, y = cnt[0]
    return (x + 1.0, y + 1.0) if matlab_indexing else (x, y)


# ---------------------------------------------------------------------------
# 2. Your colleague's function, unchanged
# ---------------------------------------------------------------------------


def find_particle(im, sigma=2.0, crop_r=15):
    """Returns (row, col) = (y, x). Note the order is the opposite of
    Track.m's, which returns [x, y]."""
    from scipy.ndimage import gaussian_filter

    if im is None or im.ndim != 2 or im.size == 0:
        return np.nan, np.nan
    y0, x0 = np.unravel_index(np.argmax(im), im.shape)
    ys = max(0, y0 - crop_r); ye = min(im.shape[0], y0 + crop_r + 1)
    xs = max(0, x0 - crop_r); xe = min(im.shape[1], x0 + crop_r + 1)
    crop = np.asarray(im[ys:ye, xs:xe], dtype=np.float32)
    if crop.size == 0:
        return np.nan, np.nan
    sm = gaussian_filter(crop, sigma=sigma)
    yl, xl = np.unravel_index(np.argmax(sm), sm.shape)
    if xl < 2 or xl > sm.shape[1] - 3 or yl < 2 or yl > sm.shape[0] - 3:
        return float(ys + yl), float(xs + xl)
    c1 = np.array([1, -8, 0, 8, -1], dtype=np.float32)
    c2 = np.array([-1, 16, -30, 16, -1], dtype=np.float32)
    xw = sm[yl, xl - 2:xl + 3]; yw = sm[yl - 2:yl + 3, xl]
    dx, d2x = np.dot(xw, c1), np.dot(xw, c2)
    dy, d2y = np.dot(yw, c1), np.dot(yw, c2)
    xf = float(xl) if d2x >= 0 else float(xl) - dx / d2x
    yf = float(yl) if d2y >= 0 else float(yl) - dy / d2y
    return float(ys + yf), float(xs + xf)


# ---------------------------------------------------------------------------
# 3. Suggested replacement: MATLAB's image processing, colleague's speed
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 4. Comparison
# ---------------------------------------------------------------------------


def _report(name, a, b):
    """a, b are (N, 2) arrays of [x, y]. Prints how they differ."""
    d = a - b
    ok = np.isfinite(d).all(1)
    if ok.sum() == 0:
        print(f"{name}: no frames in common")
        return
    m = np.nanmean(d[ok], 0)
    s = np.nanstd(d[ok], 0)
    print(f"{name}")
    print(f"   frames         : {ok.sum()}")
    print(f"   mean offset    : x {m[0]:+7.3f}   y {m[1]:+7.3f}  px   "
          f"(constant -> can be calibrated away)")
    print(f"   scatter        : x {s[0]:7.3f}   y {s[1]:7.3f}  px   "
          f"(real disagreement)")


def compare_video(path, n=500):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    ml, cw, fast = [], [], []
    prev = None
    t = {"matlab": 0.0, "colleague": 0.0, "fast": 0.0}
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame.mean(2).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)

        t0 = time.perf_counter(); ml.append(matlab_track(gray)); t["matlab"] += time.perf_counter() - t0
        t0 = time.perf_counter(); ry, rx = find_particle(gray); t["colleague"] += time.perf_counter() - t0
        cw.append((rx, ry))                        # swapped into [x, y] order
        t0 = time.perf_counter(); p = find_particle_fast(gray, prev); t["fast"] += time.perf_counter() - t0
        fast.append(p)
        prev = p
    cap.release()

    ml, cw, fast = np.array(ml), np.array(cw), np.array(fast)
    print()
    _report("Track.m  vs  find_particle", ml, cw)
    _report("Track.m  vs  find_particle_fast", ml, fast)
    print("\nspeed (ms per frame)")
    for k, v in t.items():
        print(f"   {k:10s}: {v / max(1, len(ml)) * 1e3:6.2f}")
    return ml, cw, fast


# ---------------------------------------------------------------------------
# 5. Self test on synthetic frames with a known bead position
# ---------------------------------------------------------------------------


def _synth(x=300.0, y=250.0, nr=512, nc=640, width=18.0, amp=200, bg=30, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:nr, 0:nc]
    im = bg + amp * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * width ** 2)))
    return np.clip(im + rng.normal(0, 1.5, im.shape), 0, 255).astype(np.float32)


def self_test():
    xs = np.arange(300.0, 302.01, 0.2)
    print("bead moved in known 0.2 px steps; each method should follow it 1:1\n")
    print("  true      Track.m   find_particle   find_particle_fast")
    a, b, c = [], [], []
    prev = None
    for x in xs:
        f = _synth(x=x)
        mx, _ = matlab_track(f, matlab_indexing=False)
        _, cx = find_particle(f)
        p = find_particle_fast(f, prev, matlab_indexing=False)
        prev = p
        a.append(mx); b.append(cx); c.append(p[0])
        print(f"  {x:7.2f}  {mx:9.3f}  {cx:13.3f}  {p[0]:18.3f}")
    print("\n  method               slope   scatter    mean offset")
    for name, v in [("Track.m", a), ("find_particle", b), ("find_particle_fast", c)]:
        v = np.array(v); off = float(np.mean(v - xs))
        print(f"  {name:20s} {np.polyfit(xs, v, 1)[0]:5.3f}   {np.std(v - xs - off):7.3f}   {off:+7.3f} px")

    f = _synth()
    print("\nspeed (ms per frame)")
    for name, fn in [("Track.m full frame", lambda: matlab_track(f)),
                     ("find_particle", lambda: find_particle(f)),
                     ("find_particle_fast", lambda: find_particle_fast(f, (301.0, 251.0)))]:
        fn()
        t0 = time.perf_counter()
        for _ in range(200):
            fn()
        print(f"   {name:22s}: {(time.perf_counter() - t0) / 200 * 1e3:6.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_video(sys.argv[1])
    else:
        self_test()

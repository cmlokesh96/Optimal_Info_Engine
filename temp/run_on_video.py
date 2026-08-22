"""
run_on_video.py
===============

Loads your .avi and runs all three trackers on it.

Put this next to track_compare.py, then:

  Step 1 - check one frame first (equivalent of Track.m's checkmode):

      python3 run_on_video.py movie.avi --check 400

  This prints the brightness numbers and saves check_frame.png so you can see
  whether the blur and the threshold are behaving. Do this before Step 2.

  Step 2 - run the whole movie:

      python3 run_on_video.py movie.avi

  This writes positions.csv (one row per frame, all three methods) and
  comparison.png, and prints how far apart the methods are.

  Options:
      --n 2000          only process the first 2000 frames
      --thres 60        override the threshold
      --start 100       skip the first 100 frames
"""

import argparse
import os
import sys

import cv2
import numpy as np

import track_compare as tc


# ---------------------------------------------------------------------------


def open_video(path):
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(
            f"OpenCV cannot open {path}.\n"
            "The codec is probably not supported. Easiest fix is to convert once:\n"
            "    ffmpeg -i movie.avi -pix_fmt gray -vcodec rawvideo movie_raw.avi\n"
            "and run this script on the converted file."
        )
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{os.path.basename(path)}: {n} frames, {w}x{h}, {fps:.1f} fps reported")
    return cap, n


def to_gray(frame):
    """Track.m loads the frame and (with frame(:,:,3) commented out) filters
    whatever comes back. For a grayscale-encoded avi all channels are equal."""
    if frame.ndim == 3:
        return frame.mean(2).astype(np.float32)
    return frame.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1: inspect a single frame
# ---------------------------------------------------------------------------


def check_frame(path, index, thres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap, n = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read frame {index}")

    gray = to_gray(frame)
    smooth = tc._imfilter(gray + tc._dither(gray.shape))

    print(f"\nframe {index}")
    print(f"  raw pixel values      : min {gray.min():6.1f}  max {gray.max():6.1f}")
    print(f"  after the 60 px blur  : min {smooth.min():6.1f}  max {smooth.max():6.1f}")
    print(f"  threshold in use      : {thres}")
    if smooth.max() <= thres:
        print("  --> NOTHING is above threshold. Track.m would return NaN here.")
        print("      Either the threshold is too high for this movie, or the bead")
        print("      is darker than the background (pkfnd only finds bright peaks).")
    peaks = tc.pkfnd(smooth, thres, tc.SIZE)
    print(f"  peaks found by pkfnd  : {len(peaks)}")

    mx, my = tc.matlab_track(gray)
    cy, cx = tc.find_particle(gray)
    fx, fy = tc.find_particle_fast(gray, None)
    print(f"\n  Track.m equivalent    : x {mx:9.3f}   y {my:9.3f}")
    print(f"  find_particle         : x {cx:9.3f}   y {cy:9.3f}")
    print(f"  find_particle_fast    : x {fx:9.3f}   y {fy:9.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("raw frame")
    ax[1].imshow(smooth, cmap="gray"); ax[1].set_title(f"after 60 px blur (sigma 20.9 px)")
    for a in ax:
        a.plot(mx - 1, my - 1, "rx", ms=12, mew=2, label="Track.m")
        a.plot(cx, cy, "b+", ms=12, mew=2, label="find_particle")
        a.plot(fx - 1, fy - 1, "yo", ms=9, mfc="none", mew=2, label="find_particle_fast")
    ax[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("check_frame.png", dpi=120)
    print("\nsaved check_frame.png")


# ---------------------------------------------------------------------------
# Step 2: run the whole movie
# ---------------------------------------------------------------------------


def run_all(path, nmax, start, thres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap, ntot = open_video(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    limit = ntot if nmax is None else min(ntot, start + nmax)

    ml, cw, fa = [], [], []
    prev = None
    i = start
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame)
        ml.append(tc.matlab_track(gray))
        ry, rx = tc.find_particle(gray)
        cw.append((rx, ry))                      # put into (x, y) order
        prev = tc.find_particle_fast(gray, prev, thres=thres)
        fa.append(prev)
        i += 1
        if (i - start) % 500 == 0:
            print(f"  {i - start} frames", flush=True)
    cap.release()

    ml, cw, fa = np.array(ml), np.array(cw), np.array(fa)
    frames = np.arange(start, start + len(ml))

    np.savetxt(
        "positions.csv",
        np.column_stack([frames, ml, cw, fa]),
        delimiter=",", fmt="%.4f",
        header="frame,matlab_x,matlab_y,colleague_x,colleague_y,fast_x,fast_y",
        comments="",
    )
    print(f"\nsaved positions.csv ({len(ml)} frames)")

    print()
    tc._report("Track.m  vs  find_particle", ml, cw)
    tc._report("Track.m  vs  find_particle_fast", ml, fa)

    for name, v in [("Track.m", ml), ("find_particle", cw), ("find_particle_fast", fa)]:
        lost = (~np.isfinite(v).all(1)).sum()
        print(f"  {name:20s}: {lost} frames with no bead, "
              f"x std {np.nanstd(v[:, 0]):.3f} px")

    fig = plt.figure(figsize=(11, 9))
    ax = [fig.add_subplot(3, 1, 1)]
    ax.append(fig.add_subplot(3, 1, 2, sharex=ax[0]))   # frame number axis
    ax.append(fig.add_subplot(3, 1, 3))                 # histogram, own axis
    for v, lbl in [(ml, "Track.m"), (cw, "find_particle"), (fa, "find_particle_fast")]:
        ax[0].plot(frames, v[:, 0], lw=0.8, label=lbl)
    ax[0].set_ylabel("x (px)"); ax[0].legend(); ax[0].set_title("x position")

    ax[1].plot(frames, ml[:, 0] - cw[:, 0], lw=0.8, label="Track.m - find_particle")
    ax[1].plot(frames, ml[:, 0] - fa[:, 0], lw=0.8, label="Track.m - find_particle_fast")
    ax[1].set_ylabel("difference in x (px)"); ax[1].set_xlabel("frame"); ax[1].legend()

    for v, lbl in [(ml, "Track.m"), (cw, "find_particle"), (fa, "find_particle_fast")]:
        frac = np.mod(v[np.isfinite(v[:, 0]), 0], 1.0)
        ax[2].hist(frac, bins=50, range=(0, 1), histtype="step", lw=1.2, label=lbl)
    ax[2].set_xlabel("fractional part of x")
    ax[2].set_ylabel("count")
    ax[2].set_title("flat = healthy sub-pixel, spikes = locking to whole pixels")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig("comparison.png", dpi=120)
    print("saved comparison.png")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--check", type=int, default=None, metavar="FRAME")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--thres", type=float, default=tc.THRES)
    a = p.parse_args()

    tc.THRES = a.thres
    if a.check is not None:
        check_frame(a.video, a.check, a.thres)
    else:
        run_all(a.video, a.n, a.start, a.thres)

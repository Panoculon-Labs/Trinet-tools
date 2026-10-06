# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Per-take stereo alignment: estimate and remove a constant epipolar offset.

Stereo housings move — handling, temperature, a lens swap — and a calibration
describes the geometry only as it was on calibration day. The dominant error
mode is a CONSTANT vertical offset between the rectified eyes (relative pitch
of one camera / principal-point shift), which silently ruins row-search stereo
matching and degrades VIO stereo tracking.

This module measures that residual from a take itself (median vertical
parallax of features tracked from the left to the right rectified eye, with
sub-pixel optical flow, over pairs spread across the take) and can fold it
into the right camera's principal point. Usage:

    from trinet_tools.stereo_align import (rectification, auto_align,
                                           measure_y_offset, calibration_verdict)

    rect = rectification(calib)                      # maps from a calibration
    m = measure_y_offset(mp4_l, mp4_r, pairs, rect)  # field check (dict)
    calibration_verdict(m["offset_px"])              # "ok" / "check" / "recalibrate"
    rect, shift = auto_align(mp4_l, mp4_r, pairs, calib)   # + per-take fix

Healthy units read a few tenths of a pixel to about one pixel; a residual of
more than ~3 px means the mount has moved since calibration — the correction
keeps depth working, but recalibrating restores fully trustworthy metric
geometry (a large shift can also carry smaller uncorrected components: roll,
focal change if a lens was touched). `scripts/check_calibration.py` wraps the
measurement as a field check.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class Rectification:
    """Fisheye stereo rectification derived from a Trinet calibration dict."""

    def __init__(self, calib: dict, size=(1920, 1080), cy1_shift: float = 0.0):
        def KD(cam):
            it = cam["intrinsics"]
            K = np.array([[it["fx"], 0, it["cx"]],
                          [0, it["fy"], it["cy"]], [0, 0, 1]])
            D = np.array((list(it["distortion"]) + [0.0] * 4)[:4],
                         dtype=np.float64).reshape(-1, 1)
            return K, D

        self.calib, self.size, self.cy1_shift = calib, size, cy1_shift
        K0, D0 = KD(calib["cameras"][0])
        K1, D1 = KD(calib["cameras"][1])
        K1 = K1.copy()
        K1[1, 2] += cy1_shift
        T10 = np.array(calib["T_cam1_cam0"], dtype=np.float64)
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            K0, D0, K1, D1, size, T10[:3, :3], T10[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
        self.projection = "opencv"
        if not np.isfinite(self.P1[0, 0]) or self.P1[0, 0] < 1.0:
            # OpenCV sizes the rectified focal from the undistorted image
            # boundary. On a lens that follows the ideal equidistant fisheye
            # curve closely (the global-shutter head: k1 ~ 0.03 vs ~0.15 on
            # the fleet lens, fx ~ 624 at 1920x1080 — still a strongly curved
            # fisheye, just with little polynomial correction on top) the
            # frame corners lie beyond 90 deg off-axis, those boundary points
            # blow up, and stereoRectify returns fx = 0 — every pixel maps to
            # one point. The rotations R1/R2 come from T only and stay valid, so
            # build the pinhole projection ourselves: a fixed 0.85x of the
            # mean fisheye focal (the ratio OpenCV picks on the fleet lens),
            # principal point at the frame centre, ZERO_DISPARITY baseline.
            f = 0.85 * 0.5 * (float(K0[0, 0]) + float(K1[0, 0]))
            b = float(np.linalg.norm(T10[:3, 3]))
            P = np.array([[f, 0.0, size[0] / 2.0, 0.0],
                          [0.0, f, size[1] / 2.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]])
            self.P1 = P
            self.P2 = P.copy()
            self.P2[0, 3] = -f * b
            self.projection = f"manual pinhole f={f:.1f}"
        self.map_l = cv2.fisheye.initUndistortRectifyMap(
            K0, D0, self.R1, self.P1, size, cv2.CV_16SC2)
        self.map_r = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, self.R2, self.P2, size, cv2.CV_16SC2)
        self.fx = float(self.P2[0, 0])
        self.baseline_m = abs(float(self.P2[0, 3]) / self.fx)

    def remap(self, img, eye: str):
        m = self.map_l if eye in ("l", "L", 0) else self.map_r
        return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR)


def rectification(calib: dict, size=(1920, 1080)) -> Rectification:
    return Rectification(calib, size)


# Field-check thresholds on |vertical offset| between the rectified eyes, in
# pixels at 1920x1080. Set from the published sample recordings (healthy units
# read ~0.5-1 px) with headroom; see scripts/check_calibration.py.
OK_MAX_PX = 1.5
CHECK_MAX_PX = 3.0
MIN_POINTS = 200          # fewer tracked points -> "inconclusive"


def pair_y_offsets(rl, rr, max_corners: int = 1500) -> np.ndarray:
    """Vertical parallax (yL - yR) of features tracked from the rectified left
    image `rl` into the rectified right image `rr` (both 8-bit grayscale).

    Corners are tracked with pyramidal Lucas-Kanade (sub-pixel), checked by
    tracking back (forward-backward error < 0.5 px), and kept only with a
    plausible stereo disparity (0 < xL - xR < 300 px) and |dy| < 60 px."""
    pts = cv2.goodFeaturesToTrack(rl, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) == 0:
        return np.empty(0)
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(rl, rr, pts, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(rr, rl, p1, None, **lk)
    fb = np.linalg.norm((pts - p0b).reshape(-1, 2), axis=1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
    a, b = pts.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    dx, dy = a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]
    keep = (dx > 0) & (dx < 300) & (np.abs(dy) < 60)
    return dy[keep]


def measure_y_offset(mp4_l, mp4_r, pairs, rect: Rectification,
                     samples: int = 15) -> dict:
    """Measure the vertical offset between the rectified eyes over [samples]
    frame pairs spread across the take.

    Returns {"offset_px": median yL - yR or None, "spread_px": interquartile
    range of the per-frame medians, "points": tracked points used,
    "frames": frame pairs that contributed}. offset_px is None (inconclusive)
    when fewer than MIN_POINTS points could be tracked — e.g. a featureless or
    dark scene."""
    caps = (cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r)))
    all_dy, per_frame = [], []
    for k in np.linspace(0.05, 0.95, samples):
        il, ir, _ = pairs[int(k * (len(pairs) - 1))]
        caps[0].set(cv2.CAP_PROP_POS_FRAMES, il)
        caps[1].set(cv2.CAP_PROP_POS_FRAMES, ir)
        okl, L = caps[0].read()
        okr, R = caps[1].read()
        if not (okl and okr):
            continue
        rl = rect.remap(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), "L")
        rr = rect.remap(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), "R")
        dys = pair_y_offsets(rl, rr)
        if len(dys) >= 20:
            all_dy.append(dys)
            per_frame.append(float(np.median(dys)))
    for c in caps:
        c.release()
    n = int(sum(len(d) for d in all_dy))
    if n < MIN_POINTS:
        return {"offset_px": None, "spread_px": None, "points": n, "frames": len(per_frame)}
    pf = np.array(per_frame)
    return {"offset_px": float(np.median(np.concatenate(all_dy))),
            "spread_px": float(np.subtract(*np.percentile(pf, [75, 25]))),
            "points": n, "frames": len(per_frame)}


def calibration_verdict(offset_px: float | None) -> str:
    """'ok' / 'check' / 'recalibrate' from a measured vertical offset, or
    'inconclusive' when it could not be measured."""
    if offset_px is None:
        return "inconclusive"
    a = abs(offset_px)
    return "ok" if a <= OK_MAX_PX else ("check" if a <= CHECK_MAX_PX else "recalibrate")


def rect_y_offset(mp4_l, mp4_r, pairs, rect: Rectification,
                  samples: int = 15) -> float | None:
    """Median vertical parallax (yL - yR) over [samples] rectified pairs spread
    across the take. ~0 for healthy geometry; None when there aren't enough
    tracked points to trust. See measure_y_offset for the details."""
    return measure_y_offset(mp4_l, mp4_r, pairs, rect, samples)["offset_px"]


def auto_align(mp4_l, mp4_r, pairs, calib: dict, size=(1920, 1080),
               rounds: int = 3, tol_px: float = 0.3):
    """Measure and remove the constant vertical offset for this take.

    Returns (Rectification, shift_px). shift_px is the cy1 correction that
    zeroed the residual (0.0 when the calibration already fits)."""
    rect = Rectification(calib, size)
    shift = 0.0
    for _ in range(rounds):
        dy = rect_y_offset(mp4_l, mp4_r, pairs, rect)
        if dy is None or abs(dy) < tol_px:
            break
        # dy = yL - yR: negative dy = right image content sits lower; raising
        # cy1 lifts it. (Direction verified empirically — the wrong sign
        # doubles the residual.)
        shift -= dy
        rect = Rectification(calib, size, cy1_shift=shift)
    return rect, shift

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Per-take stereo alignment: estimate and remove a constant epipolar offset.

Stereo housings move — handling, temperature, a lens swap — and a calibration
describes the geometry only as it was on calibration day. The dominant error
mode is a CONSTANT vertical offset between the rectified eyes (relative pitch
of one camera / principal-point shift), which silently ruins row-search stereo
matching and degrades VIO stereo tracking.

This module measures that residual from a take itself (median vertical
parallax of ORB matches over rectified sample pairs) and folds it into the
right camera's principal point. Usage:

    from trinet_tools.stereo_align import rectification, auto_align

    rect = rectification(calib)                      # maps from a calibration
    rect, shift = auto_align(mp4_l, mp4_r, pairs, calib)   # + per-take fix

A |shift| of more than a pixel or two means the mount has moved since
calibration — the correction keeps depth working, but recalibrating restores
fully trustworthy metric geometry (a large shift can also carry smaller
uncorrected components: roll, focal change if a lens was touched).
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


def rect_y_offset(mp4_l, mp4_r, pairs, rect: Rectification,
                  samples: int = 5) -> float | None:
    """Median vertical parallax (yL - yR) of ORB matches over [samples]
    rectified pairs spread across the take. ~0 for healthy geometry;
    None when there aren't enough matches to trust."""
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    caps = (cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r)))
    dys = []
    for k in np.linspace(0.1, 0.9, samples):
        il, ir, _ = pairs[int(k * (len(pairs) - 1))]
        caps[0].set(cv2.CAP_PROP_POS_FRAMES, il)
        caps[1].set(cv2.CAP_PROP_POS_FRAMES, ir)
        okl, L = caps[0].read()
        okr, R = caps[1].read()
        if not (okl and okr):
            continue
        rl = rect.remap(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), "L")
        rr = rect.remap(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), "R")
        kL, dL = orb.detectAndCompute(rl, None)
        kR, dR = orb.detectAndCompute(rr, None)
        if dL is None or dR is None:
            continue
        for m in bf.match(dL, dR):
            dx = kL[m.queryIdx].pt[0] - kR[m.trainIdx].pt[0]
            dy = kL[m.queryIdx].pt[1] - kR[m.trainIdx].pt[1]
            if 0 < dx < 300 and abs(dy) < 60:
                dys.append(dy)
    for c in caps:
        c.release()
    return float(np.median(dys)) if len(dys) > 100 else None


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

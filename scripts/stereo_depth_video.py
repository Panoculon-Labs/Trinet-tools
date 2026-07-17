#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Stereo depth video for Trinet stereo recordings — from the two MP4s alone.

Layout: [ LEFT RGB | RIGHT RGB ] rectified on top, metric depth below (raw
SGBM, or raw + WLS-filtered side by side with --wls), an optional scrolling
gyro|accel strip (--imu), and a depth legend.

Everything the renderer needs beyond the two video files rides inside them
(the embedded TMF metadata: per-frame timing for exact L/R pairing, the IMU
stream, and the camera calibration the device stores). Sidecar files are used
when present; `--calibration` overrides the embedded blob with a
calibration.json or a TBLC .bin.

Usage:
    python3 scripts/stereo_depth_video.py TAKE_PREFIX OUT.mp4 \
        [--wls] [--imu] [--scale 0.6667] [--num-disp 128] \
        [--min-depth 0.25] [--max-depth 6.0] [--ema 0.5] \
        [--match-scale 1.0] [--calibration CALIB]

TAKE_PREFIX names `<prefix>_L.mp4` + `<prefix>_R.mp4`. `--wls` needs
opencv-contrib-python (cv2.ximgproc); the tool says so and continues raw-only
if it's missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

for _p in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from trinet_tools import calib_blob                           # noqa: E402
from trinet_tools.reader import read_imu, read_vts            # noqa: E402
from trinet_tools.stereo_align import Rectification, auto_align  # noqa: E402
from trinet_tools.tmf import read_tmf                         # noqa: E402


# ----------------------------------------------------------- recording IO --
def load_take(prefix: Path, workdir: Path, calib_override: Path | None):
    """(mp4_l, mp4_r, vts_l, vts_r, imu, calib_dict) — sidecars when present,
    embedded TMF otherwise; calibration from override file, else the MP4."""
    mp4_l, mp4_r = Path(f"{prefix}_L.mp4"), Path(f"{prefix}_R.mp4")
    for p in (mp4_l, mp4_r):
        if not p.exists():
            sys.exit(f"missing {p}")

    recs = {}

    def side(path: Path, eye: str):
        vts_p = Path(f"{prefix}_{eye}.vts")
        if not vts_p.exists():
            rec = recs.setdefault(eye, read_tmf(path))
            vts_p = workdir / f"{eye}.vts"
            data = rec.vts_bytes()
            if not data:
                sys.exit(f"{path}: no frame-timing metadata (TMF) and no {vts_p.name}")
            vts_p.write_bytes(data)
        return read_vts(str(vts_p))

    vts_l, vts_r = side(mp4_l, "L"), side(mp4_r, "R")

    imu_p = Path(f"{prefix}.imu")
    if not imu_p.exists():
        rec = recs.setdefault("L", read_tmf(mp4_l))
        data = rec.imu_bytes()
        imu_p = workdir / "take.imu"
        if data:
            imu_p.write_bytes(data)
        else:
            imu_p = None
    imu = read_imu(str(imu_p)) if imu_p else None

    calib = None
    if calib_override:
        raw = Path(calib_override).read_bytes()
        try:
            calib = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            calib = calib_blob.unpack(raw)
    else:
        rec = recs.setdefault("L", read_tmf(mp4_l))
        if rec.calib_blob:
            calib = calib_blob.unpack(rec.calib_blob)
    if calib is None or "cameras" not in calib or len(calib["cameras"]) < 2:
        sys.exit("no stereo calibration found — the camera stores one after a "
                 "calibration upload (it is embedded in every recording); or "
                 "pass --calibration calibration.json/.bin")
    return mp4_l, mp4_r, vts_l, vts_r, imu, calib




def pair_frames(vts_l, vts_r, max_skew_frac: float = 0.5):
    """L/R frame association by start-of-frame timestamp. The eyes are
    frame-synchronized in hardware (microsecond deltas), so nearest-neighbour
    with a half-period gate is exact."""
    tl = vts_l.sof_timestamps_ns.astype(np.int64)
    tr = vts_r.sof_timestamps_ns.astype(np.int64)
    gate = int(np.median(np.diff(tl)) * max_skew_frac)
    out, j = [], 0
    for i, t in enumerate(tl):
        while j + 1 < len(tr) and abs(int(tr[j + 1]) - int(t)) < abs(int(tr[j]) - int(t)):
            j += 1
        if abs(int(tr[j]) - int(t)) <= gate:
            out.append((i, j, int(t)))
    return out


# ------------------------------------------------------------- rendering --
def _label(img, txt, x, y, scale=0.8):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)


class ImuStrip:
    """Scrolling gyro | accel strip: a +-window_s window around the current
    frame, cursor centered, y-range fixed to the take's extremes."""

    COLORS = [(80, 80, 255), (80, 220, 80), (255, 160, 60)]   # x, y, z (BGR)

    def __init__(self, imu, width, h=220, window_s=2.5):
        self.t = (imu.timestamps_ns.astype(np.int64)
                  - int(imu.timestamps_ns[0])) / 1e9
        self.t0_ns = int(imu.timestamps_ns[0])
        self.gyro = np.degrees(imu.gyro)
        self.accel = imu.accel
        self.w, self.h, self.win = width, h, window_s
        self.gy_lim = max(60.0, float(np.abs(self.gyro).max()) * 1.05)
        self.ac_lim = max(15.0, float(np.abs(self.accel).max()) * 1.05)

    def _panel(self, data, lim, t_now, pw, title, unit):
        img = np.full((self.h, pw, 3), 16, np.uint8)
        lo, hi = t_now - self.win, t_now + self.win
        cv2.line(img, (0, self.h // 2), (pw, self.h // 2), (60, 60, 60), 1)
        m = (self.t >= lo) & (self.t <= hi)
        if m.any():
            xs = ((self.t[m] - lo) / (2 * self.win) * (pw - 1)).astype(np.int32)
            for a in range(3):
                ys = ((1 - (data[m, a] / lim + 1) / 2) * (self.h - 1)).astype(np.int32)
                pts = np.stack([xs, np.clip(ys, 0, self.h - 1)], axis=1)
                cv2.polylines(img, [pts], False, self.COLORS[a], 1, cv2.LINE_AA)
        cv2.line(img, (pw // 2, 0), (pw // 2, self.h), (200, 200, 200), 1)
        _label(img, f"{title} [+-{lim:.0f} {unit}]", 10, 24, scale=0.6)
        return img

    def render(self, sof_ns):
        t_now = (int(sof_ns) - self.t0_ns) / 1e9
        pw = self.w // 2
        strip = np.hstack([
            self._panel(self.gyro, self.gy_lim, t_now, pw, "gyro", "deg/s"),
            self._panel(self.accel, self.ac_lim, t_now, self.w - pw,
                        "accel", "m/s^2"),
        ])
        cv2.line(strip, (pw, 0), (pw, self.h), (90, 90, 90), 1)
        return strip


class DepthColorizer:
    """disparity -> metric depth -> EMA-smoothed inverse-depth turbo cmap."""

    def __init__(self, fx, baseline, min_depth, max_depth, ema):
        self.fx, self.baseline = fx, baseline
        self.lo, self.hi = 1.0 / max_depth, 1.0 / min_depth
        self.min_depth, self.max_depth = min_depth, max_depth
        self.ema, self.state = ema, None

    def __call__(self, disp):
        valid = disp > 0.5
        depth = np.zeros_like(disp)
        depth[valid] = self.fx * self.baseline / disp[valid]
        in_range = valid & (depth >= self.min_depth) & (depth <= self.max_depth)
        inv = np.zeros_like(depth)
        inv[in_range] = 1.0 / depth[in_range]
        if self.ema > 0:
            if self.state is None:
                self.state = inv.copy()
            else:
                both = in_range & (self.state > 0)
                self.state[both] = (self.ema * self.state[both]
                                    + (1 - self.ema) * inv[both])
                fresh = in_range & (self.state <= 0)
                self.state[fresh] = inv[fresh]
                self.state[~in_range] = 0
            inv = self.state
            in_range = inv > 0
        norm = np.clip((inv - self.lo) / (self.hi - self.lo), 0, 1)
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        cmap[~in_range] = (24, 24, 24)
        return cmap


def make_legend(W, min_depth, max_depth, h=44):
    legend = np.zeros((h, W, 3), np.uint8)
    bar = np.linspace(1, 0, W // 2, dtype=np.float32)
    grad = cv2.applyColorMap((bar * 255).astype(np.uint8).reshape(1, -1),
                             cv2.COLORMAP_TURBO)
    x0 = W // 2 - W // 4
    legend[10:34, x0:x0 + W // 2] = grad
    mid = 2 / (1 / min_depth + 1 / max_depth)
    for frac, label in ((0.0, f"{min_depth:.1f}m"),
                        (0.5, f"{mid:.1f}m"), (1.0, f"{max_depth:.1f}m")):
        x = int(x0 + frac * (W // 2))
        cv2.putText(legend, label, (min(max(4, x - 30), W - 90), 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return legend


def compose_canvas(left, right, panels, legend, pw, ph, imu_strip=None):
    W = pw * 2
    top = np.hstack([cv2.resize(left, (pw, ph)), cv2.resize(right, (pw, ph))])
    bottom = np.zeros((ph, W, 3), np.uint8)
    xs = [(W - pw) // 2] if len(panels) == 1 else [0, pw]
    for (cmap, _), x in zip(panels, xs):
        bottom[:, x:x + pw] = cv2.resize(cmap, (pw, ph))
    rows = [top, bottom]
    if imu_strip is not None:
        rows.append(imu_strip)
    rows.append(legend)
    frame = np.vstack(rows)
    _label(frame, "LEFT RGB (rectified)", 12, 30)
    _label(frame, "RIGHT RGB (rectified)", pw + 12, 30)
    for (_, name), x in zip(panels, xs):
        _label(frame, name, x + 12, ph + 34)
    _label(frame, "Panoculon Labs", 16, ph * 2 - 24, scale=1.5)
    return frame


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", help="take prefix (expects <prefix>_L.mp4/_R.mp4)")
    ap.add_argument("out")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration.json or TBLC .bin (default: the blob "
                         "embedded in the recording)")
    ap.add_argument("--scale", type=float, default=2.0 / 3.0)
    ap.add_argument("--num-disp", type=int, default=128)
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--wls", action="store_true",
                    help="add a WLS-filtered depth panel (needs "
                         "opencv-contrib-python)")
    ap.add_argument("--imu", action="store_true",
                    help="scrolling gyro|accel strip under the depth panels")
    ap.add_argument("--match-scale", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--no-auto-align", action="store_true",
                    help="disable the per-take vertical auto-alignment")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="render from this many seconds into the take")
    ap.add_argument("--end-s", type=float, default=0.0,
                    help="stop after this many seconds (0 = full take)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="trinet_depth_") as td:
        workdir = Path(td)
        mp4_l, mp4_r, vts_l, vts_r, imu, calib = load_take(
            Path(args.take), workdir, args.calibration)

    pairs = pair_frames(vts_l, vts_r)
    if args.start_s > 0 or args.end_s > 0:
        t0 = pairs[0][2]
        lo = t0 + int(args.start_s * 1e9)
        hi = t0 + int(args.end_s * 1e9) if args.end_s > 0 else pairs[-1][2] + 1
        pairs = [p for p in pairs if lo <= p[2] <= hi]
    print(f"{len(pairs)} synchronized pairs")
    if args.no_auto_align:
        rect = Rectification(calib)
    else:
        rect = Rectification(calib)
        rect, shift = auto_align(mp4_l, mp4_r, pairs, calib)
        if abs(shift) > 0.5:
            print(f"[auto-align] residual vertical offset {shift:+.1f} px "
                  f"absorbed into rectification (stereo mount has moved "
                  f"since calibration — consider recalibrating)")
    m0, m1 = rect.map_l, rect.map_r
    fx, baseline = rect.fx, rect.baseline_m
    print(f"baseline {baseline*1000:.1f} mm (from "
          f"{'--calibration' if args.calibration else 'embedded calibration'})")

    nd = (args.num_disp + 15) // 16 * 16
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=5,
        P1=8 * 3 * 5 * 5, P2=32 * 3 * 5 * 5,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=120, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    wls = matcher_r = None
    if args.wls:
        if not hasattr(cv2, "ximgproc"):
            print("[warn] cv2.ximgproc missing (pip install "
                  "opencv-contrib-python) — rendering raw SGBM only",
                  file=sys.stderr)
        else:
            matcher_r = cv2.ximgproc.createRightMatcher(sgbm)
            wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
            wls.setLambda(8000.0)
            wls.setSigmaColor(1.2)

    pw, ph = int(1920 * args.scale) & ~1, int(1080 * args.scale) & ~1
    strip = ImuStrip(imu, pw * 2) if (args.imu and imu) else None
    if args.imu and imu is None:
        print("[warn] --imu: no inertial data found in the recording",
              file=sys.stderr)
    legend_h = 44
    W = pw * 2
    H = ph * 2 + legend_h + (strip.h if strip else 0)
    legend = make_legend(W, args.min_depth, args.max_depth, legend_h)
    color_raw = DepthColorizer(fx, baseline, args.min_depth, args.max_depth, args.ema)
    color_wls = DepthColorizer(fx, baseline, args.min_depth, args.max_depth, args.ema)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    cap_l, cap_r = cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r))
    cur_l = cur_r = -1
    frame_l = frame_r = None
    n = 0
    for il, ir, ts in pairs:
        while cur_l < il:
            ok, frame_l = cap_l.read()
            if not ok:
                break
            cur_l += 1
        while cur_r < ir:
            ok, frame_r = cap_r.read()
            if not ok:
                break
            cur_r += 1
        if cur_l != il or cur_r != ir:
            break

        left = cv2.remap(frame_l, m0[0], m0[1], cv2.INTER_LINEAR)
        right = cv2.remap(frame_r, m1[0], m1[1], cv2.INTER_LINEAR)

        ms = args.match_scale
        if ms != 1.0:
            mw, mh = int(1920 * ms) & ~1, int(1080 * ms) & ~1
            ml = cv2.resize(left, (mw, mh), interpolation=cv2.INTER_AREA)
            mr = cv2.resize(right, (mw, mh), interpolation=cv2.INTER_AREA)
        else:
            ml, mr = left, right
        gl = cv2.cvtColor(ml, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(mr, cv2.COLOR_BGR2GRAY)
        disp16 = sgbm.compute(gl, gr)

        def to_fullres(d):
            if ms != 1.0:
                d = cv2.resize(d, (1920, 1080), interpolation=cv2.INTER_LINEAR) / ms
            return d

        disp_raw = to_fullres(cv2.medianBlur(disp16.astype(np.float32) / 16.0, 5))
        panels = [(color_raw(disp_raw), "depth - SGBM")]
        if wls is not None:
            disp16_r = matcher_r.compute(gr, gl)
            d16f = wls.filter(disp16, ml, disparity_map_right=disp16_r)
            panels.append((color_wls(to_fullres(d16f.astype(np.float32) / 16.0)),
                           "depth - SGBM + WLS"))

        ff.stdin.write(compose_canvas(
            left, right, panels, legend, pw, ph,
            imu_strip=(strip.render(ts) if strip else None)).tobytes())
        n += 1
        if n % 100 == 0:
            print(f"{n}/{len(pairs)} frames")

    ff.stdin.close()
    ff.wait()
    cap_l.release()
    cap_r.release()
    print(f"wrote {args.out}: {n} frames @ {args.fps:g} fps ({n/args.fps:.1f}s)")


if __name__ == "__main__":
    main()

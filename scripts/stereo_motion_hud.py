#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Side-by-side stereo playback with a motion HUD — grid overlay, live
rotation rates, and the predicted rolling-shutter shear per frame.

The camera's sensor is a rolling shutter: rows are read out over a finite
window, so rotating the camera shears the image geometry within each frame
(distinct from motion BLUR, which is set by exposure). This tool makes that
visible and quantified: a fixed reference grid over both eyes, the take's
gyro rates at every frame, and the first-order shear prediction

    shear_px ≈ fx * omega_yaw[rad/s] * t_readout

Because the eyes are frame-synchronized in hardware, the shear is common-mode
between them — stereo geometry stays consistent even while both eyes shear
against the world (verify with scripts/stereo_depth_video.py during fast
rotation).

All inputs come from the recording itself: IMU + per-frame timing from the
embedded metadata (sidecars when present), fx from the embedded calibration
(falls back to a nominal value with a warning). Readout time comes from the
per-frame metadata when the recording carries it, else --readout-ms.

Usage:
    python3 scripts/stereo_motion_hud.py TAKE_PREFIX OUT.mp4 \
        [--start-s 0] [--end-s 0] [--speed 0.5] [--readout-ms 31.8]
"""

from __future__ import annotations

import argparse
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
from trinet_tools.tmf import read_tmf                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", help="take prefix (expects <prefix>_L.mp4/_R.mp4)")
    ap.add_argument("out")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=0.0, help="0 = full take")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="playback speed factor (default 0.5 = half speed)")
    ap.add_argument("--readout-ms", type=float, default=31.8,
                    help="sensor readout time when the recording doesn't "
                         "carry per-frame readout metadata")
    ap.add_argument("--grid-px", type=int, default=160)
    args = ap.parse_args()

    prefix = Path(args.take)
    mp4_l, mp4_r = Path(f"{prefix}_L.mp4"), Path(f"{prefix}_R.mp4")
    for p in (mp4_l, mp4_r):
        if not p.exists():
            sys.exit(f"missing {p}")

    with tempfile.TemporaryDirectory(prefix="trinet_hud_") as td:
        td = Path(td)
        rec = read_tmf(mp4_l)
        vts_p = Path(f"{prefix}_L.vts")
        if not vts_p.exists():
            (td / "L.vts").write_bytes(rec.vts_bytes() or b"")
            vts_p = td / "L.vts"
        vts = read_vts(str(vts_p))
        imu_p = Path(f"{prefix}.imu")
        if not imu_p.exists():
            (td / "t.imu").write_bytes(rec.imu_bytes() or b"")
            imu_p = td / "t.imu"
        imu = read_imu(str(imu_p))

    fx = 589.0
    if rec.calib_blob:
        fx = float(calib_blob.unpack(rec.calib_blob)["cameras"][0]["intrinsics"]["fx"])
    else:
        print(f"[warn] no embedded calibration — using nominal fx={fx}",
              file=sys.stderr)

    # Per-frame readout: .vts v5+ records it; else the CLI value.
    t_read = args.readout_ms / 1e3
    ro = getattr(vts, "readout_time_us", None)
    if ro is not None and np.any(np.asarray(ro) > 0):
        t_read = float(np.median(np.asarray(ro)[np.asarray(ro) > 0])) / 1e6
        print(f"readout from recording: {t_read*1e3:.1f} ms")

    sof = vts.sof_timestamps_ns.astype(np.int64)
    ts = imu.timestamps_ns.astype(np.int64)
    w_dps = np.degrees(imu.gyro)
    rates = np.zeros((len(sof), 3))
    for i, s in enumerate(sof):
        m = (ts >= s) & (ts <= s + int(t_read * 1e9))
        if m.any():
            rates[i] = np.abs(w_dps[m]).mean(axis=0)

    t0 = sof[0]
    f0 = int(np.searchsorted(sof, t0 + int(args.start_s * 1e9)))
    f1 = len(sof) if args.end_s <= 0 else \
        int(np.searchsorted(sof, t0 + int(args.end_s * 1e9)))

    W, H, HUD = 1920, 540, 60
    fps = 30.0 * args.speed
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + HUD}",
         "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

    cl, cr = cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r))
    cl.set(cv2.CAP_PROP_POS_FRAMES, f0)
    cr.set(cv2.CAP_PROP_POS_FRAMES, f0)
    n = 0
    for fr in range(f0, f1):
        okl, L = cl.read()
        okr, R = cr.read()
        if not (okl and okr):
            break
        L = cv2.resize(L, (W // 2, H))
        R = cv2.resize(R, (W // 2, H))
        canvas = np.zeros((H + HUD, W, 3), np.uint8)
        canvas[HUD:, :W // 2] = L
        canvas[HUD:, W // 2:] = R
        step = max(40, int(args.grid_px * (H / 1080)))
        for half in (0, W // 2):
            for x in range(step, W // 2, step):
                cv2.line(canvas, (half + x, HUD), (half + x, H + HUD),
                         (0, 255, 255), 1)
        cv2.line(canvas, (W // 2, HUD), (W // 2, H + HUD), (255, 255, 255), 2)
        p, y, r = rates[fr]
        shear = fx * np.radians(rates[fr][1]) * t_read
        cv2.putText(canvas,
                    f"frame {fr}  pitch {p:5.0f}  yaw {y:5.0f}  roll {r:5.0f} deg/s"
                    f"   predicted shear {shear:5.1f} px   {args.speed:g}x speed",
                    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        bar = int(min(rates[fr][1] / 300.0, 1.0) * 380)
        cv2.rectangle(canvas, (1520, 20), (1520 + bar, 44), (0, 128, 255), -1)
        ff.stdin.write(canvas.tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    cl.release()
    cr.release()
    print(f"wrote {args.out}: {n} frames")


if __name__ == "__main__":
    main()

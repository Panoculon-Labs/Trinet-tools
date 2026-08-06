#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Single-view stereo visualization: L|R feed + depth + orientation.

One video, four channels of the take at once:

  * **Left | right camera feed** — the two eyes side by side. Native
    fisheye by default; `--rectified` shows the epipolar-aligned pair the
    depth panel is actually computed from.
  * **SGBM + WLS depth** — the dense, edge-preserving disparity (metric,
    from the same calibration), turbo-mapped over an inverse-depth ramp.
  * **IMU orientation** — Madgwick attitude from the take's own inertial
    stream, drawn as a 3D body frame plus an artificial horizon and
    roll/pitch/yaw readout.
  * **Audio** — copied from the recording and muxed onto the render.

Usage:
    python3 scripts/stereo_stitch_viz.py TAKE_PREFIX OUT.mp4 \
        --calibration CALIB [--rectified] [--width 1280] \
        [--min-depth 0.25] [--max-depth 5.0] [--stride 1] [--no-audio]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

for _p in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from trinet_tools import calib_blob                              # noqa: E402
from trinet_tools.madgwick import run_madgwick                   # noqa: E402
from trinet_tools.reader import read_imu, read_vts               # noqa: E402
from trinet_tools.stereo_align import Rectification              # noqa: E402


# ---------------------------------------------------------- orientation --
def quat_to_R(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_attitude(R_wc, up, ref_heading):
    """Roll/pitch/yaw of the CAMERA against a gravity-defined horizon.

    Madgwick returns the attitude of the IMU in its own gravity-aligned
    world; on this rig the IMU is mounted ~90 deg off the optical axis, so
    reading Euler angles straight off it shows a permanently inverted
    horizon. Composing with R_imu_cam0 and measuring against the observed
    gravity direction gives angles a viewer can read: pitch 0 = looking at
    the horizon, roll 0 = level.
    """
    fwd = R_wc @ np.array([0.0, 0.0, 1.0])
    right = R_wc @ np.array([1.0, 0.0, 0.0])
    pitch = math.degrees(math.asin(float(np.clip(np.dot(fwd, up), -1, 1))))
    h = fwd - np.dot(fwd, up) * up
    hn = np.linalg.norm(h)
    if hn < 1e-6:                      # looking straight up/down
        return 0.0, pitch, 0.0
    h = h / hn
    hr = np.cross(h, up)
    hr /= max(np.linalg.norm(hr), 1e-9)
    roll = math.degrees(math.atan2(float(np.dot(np.cross(hr, right), h)),
                                   float(np.dot(hr, right))))
    yaw = math.degrees(math.atan2(float(np.dot(h, ref_heading[1])),
                                  float(np.dot(h, ref_heading[0]))))
    return roll, pitch, yaw


def draw_orientation(size, R, roll, pitch, yaw, gyro_dps):
    """3D body axes + artificial horizon + numeric attitude."""
    Wp, Hp = size
    img = np.full((Hp, Wp, 3), 18, np.uint8)

    # --- artificial horizon (top half) ---
    cx, cy, rad = Wp // 2, int(Hp * 0.33), int(min(Wp, Hp) * 0.26)
    horizon = np.zeros((2 * rad, 2 * rad, 3), np.uint8)
    off = int(np.clip(pitch / 60.0, -1, 1) * rad)
    horizon[:rad + off] = (92, 62, 30)          # sky (BGR)
    horizon[rad + off:] = (38, 68, 96)          # ground
    cv2.line(horizon, (0, rad + off), (2 * rad, rad + off), (230, 230, 230), 2)
    M = cv2.getRotationMatrix2D((rad, rad), roll, 1.0)
    horizon = cv2.warpAffine(horizon, M, (2 * rad, 2 * rad))
    mask = np.zeros((2 * rad, 2 * rad), np.uint8)
    cv2.circle(mask, (rad, rad), rad - 1, 255, -1)
    roi = img[cy - rad:cy + rad, cx - rad:cx + rad]
    roi[mask > 0] = horizon[mask > 0]
    cv2.circle(img, (cx, cy), rad - 1, (140, 150, 160), 2)
    cv2.line(img, (cx - 22, cy), (cx - 6, cy), (0, 230, 255), 2)
    cv2.line(img, (cx + 6, cy), (cx + 22, cy), (0, 230, 255), 2)
    cv2.circle(img, (cx, cy), 3, (0, 230, 255), -1)

    # --- 3D body axes (bottom half) ---
    ox, oy = Wp // 2, int(Hp * 0.74)
    f = min(Wp, Hp) * 0.30
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    # view: look along -Y_world with Z up, so yaw spins visibly
    V = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    cols = [(80, 80, 255), (90, 220, 90), (255, 170, 60)]
    names = ["x", "y", "z"]
    proj = []
    for a in axes:
        p = V @ (R @ a)
        proj.append((ox + int(p[0] * f), oy - int(p[1] * f), p[2]))
    for (px, py, depth), c, nm in sorted(zip(proj, cols, names),
                                         key=lambda t: t[0][2]):
        cv2.line(img, (ox, oy), (px, py), c, 3, cv2.LINE_AA)
        cv2.circle(img, (px, py), 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, nm, (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1, cv2.LINE_AA)

    def lab(txt, x0, y0, scale=0.52, col=(215, 225, 235)):
        cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, 1, cv2.LINE_AA)
    lab("IMU ORIENTATION (Madgwick)", 12, 22, 0.5, (150, 200, 220))
    lab(f"roll  {roll:+7.1f}", 12, Hp - 54)
    lab(f"pitch {pitch:+7.1f}", 12, Hp - 34)
    lab(f"yaw   {yaw:+7.1f}", 12, Hp - 14)
    lab(f"|w| {np.linalg.norm(gyro_dps):6.1f} d/s", Wp - 150, Hp - 14, 0.5,
        (150, 200, 220))
    return img


def sanitize_mag(mag):
    """Repair magnetometer dropouts before they reach Madgwick.

    The v5 IMU occasionally emits an all-zero magnetometer sample. Madgwick's
    9-DOF update normalises that vector, so a single dropout divides by zero
    and NaNs the quaternion for the WHOLE run (observed: 50 bad rows out of
    48822 destroyed every attitude in the take). Hold the last valid reading
    across dropouts; if the mag is mostly unusable, return None so the caller
    falls back to 6-DOF.
    """
    if mag is None:
        return None
    m = np.asarray(mag, dtype=np.float64).copy()
    good = np.isfinite(m).all(axis=1) & (np.linalg.norm(m, axis=1) > 1e-6)
    if good.sum() < 0.5 * len(m):
        return None
    if not good.all():
        idx = np.maximum.accumulate(np.where(good, np.arange(len(m)), -1))
        idx[idx < 0] = int(good.argmax())          # leading dropouts
        m = m[idx]
        print(f"  [imu] repaired {int((~good).sum())} magnetometer dropouts")
    return m


def colorize_depth(disp, fx, baseline, lo, hi):
    depth = np.zeros_like(disp)
    good = disp > 0.5
    depth[good] = fx * baseline / disp[good]
    inr = good & (depth >= lo) & (depth <= hi)
    inv = np.zeros_like(depth)
    inv[inr] = 1.0 / depth[inr]
    a, b = 1.0 / hi, 1.0 / lo
    norm = np.clip((inv - a) / (b - a), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[~inr] = (26, 26, 26)
    return cm


def label(img, txt, x, y, scale=0.62, col=(255, 255, 255)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2,
                cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("out")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1280,
                    help="output width; each eye gets half of it")
    ap.add_argument("--rectified", action="store_true",
                    help="show the rectified pair instead of raw fisheye")
    ap.add_argument("--depth-width", type=int, default=1280,
                    help="rectified width used for SGBM+WLS")
    ap.add_argument("--num-disp", type=int, default=128)
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    raw = args.calibration.read_bytes()
    try:
        calib = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        calib = calib_blob.unpack(raw)

    take = Path(args.take)
    mp4_l, mp4_r = Path(f"{take}_L.mp4"), Path(f"{take}_R.mp4")
    vts_l = read_vts(f"{take}_L.vts")
    vts_r = read_vts(f"{take}_R.vts")
    imu = read_imu(f"{take}.imu")

    tl = vts_l.sof_timestamps_ns.astype(np.int64)
    tr = vts_r.sof_timestamps_ns.astype(np.int64)
    gate = int(np.median(np.diff(tl)) * 0.5)
    pairs, j = [], 0
    for i, t in enumerate(tl):
        while j + 1 < len(tr) and abs(int(tr[j + 1]) - int(t)) < abs(int(tr[j]) - int(t)):
            j += 1
        if abs(int(tr[j]) - int(t)) <= gate:
            pairs.append((i, j, int(t)))
    pairs = pairs[::max(1, args.stride)]
    print(f"{len(pairs)} synchronized pairs")

    # --- orientation for the whole take, sampled per frame ---
    ts_imu = np.asarray(imu.timestamps_ns, dtype=np.float64) / 1e9
    mag = sanitize_mag(getattr(imu, "mag", None))
    quats = run_madgwick(imu.accel, imu.gyro, mag, ts_imu,
                         use_mag=mag is not None)
    if not np.isfinite(quats).all():
        print("  [imu] 9-DOF attitude went non-finite - falling back to 6-DOF",
              file=sys.stderr)
        mag = None
        quats = run_madgwick(imu.accel, imu.gyro, None, ts_imu, use_mag=False)
    if not np.isfinite(quats).all():
        sys.exit("attitude solution is non-finite even in 6-DOF")
    gyro_dps = np.degrees(np.asarray(imu.gyro))
    print(f"Madgwick attitude over {len(quats)} IMU samples "
          f"({'9-DOF' if mag is not None else '6-DOF'})")

    # IMU -> camera, and the world "up" that Madgwick's frame actually uses
    # (measured, not assumed: a static accelerometer reads +g along up).
    R_cam_imu = np.array(calib["T_cam0_imu"], dtype=np.float64)[:3, :3]
    R_imu_cam = R_cam_imu.T
    g_imu = np.asarray(imu.accel, dtype=np.float64).mean(axis=0)
    g_imu /= max(np.linalg.norm(g_imu), 1e-9)
    ups = np.stack([quat_to_R(q) @ g_imu for q in quats[::37]])
    up = ups.mean(axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    R_wc0 = quat_to_R(quats[0]) @ R_imu_cam
    f0 = R_wc0 @ np.array([0.0, 0.0, 1.0])
    f0 = f0 - np.dot(f0, up) * up
    if np.linalg.norm(f0) < 1e-6:
        f0 = np.cross(up, [1.0, 0.0, 0.0])
    f0 /= max(np.linalg.norm(f0), 1e-9)
    ref = (f0, np.cross(up, f0))
    print(f"world up (measured) {np.round(up, 3)}; "
          f"camera pitched {math.degrees(math.asin(float(np.clip(np.dot(R_wc0 @ [0,0,1.], up), -1, 1)))):+.0f} deg at t0")

    PW = args.width
    EW = (PW // 2) & ~1                      # per-eye width
    EH = int(round(EW * 1080 / 1920)) & ~1
    PH = EH
    print(f"eyes {EW}x{EH} side by side "
          f"({'rectified' if args.rectified else 'native fisheye'})")

    rect = Rectification(calib)
    DW = args.depth_width
    DH = int(round(DW * 1080 / 1920)) & ~1
    fx_d = rect.fx * (DW / 1920.0)
    nd = (args.num_disp + 15) // 16 * 16
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=5,
        P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=120, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    if not hasattr(cv2, "ximgproc"):
        sys.exit("needs opencv-contrib-python for the WLS filter")
    mr = cv2.ximgproc.createRightMatcher(sgbm)
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.2)

    # --- canvas: L|R on top, depth + orientation below ---
    OW = EW * 2
    dep_w = int(OW * 0.62) & ~1
    ori_w = OW - dep_w
    bot_h = int(dep_w * DH / DW) & ~1
    legend_h = 40
    OH = PH + bot_h + legend_h

    legend = np.zeros((legend_h, OW, 3), np.uint8)
    bar = cv2.applyColorMap(
        np.linspace(255, 0, OW // 2, dtype=np.uint8).reshape(1, -1),
        cv2.COLORMAP_TURBO)
    x0 = OW // 4
    legend[10:30, x0:x0 + OW // 2] = bar
    mid = 2 / (1 / args.min_depth + 1 / args.max_depth)
    for frac, txt in ((0.0, f"{args.min_depth:.2f} m"), (0.5, f"{mid:.1f} m"),
                      (1.0, f"{args.max_depth:.1f} m")):
        xx = int(x0 + frac * (OW // 2))
        cv2.putText(legend, txt, (min(max(4, xx - 28), OW - 80), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    tmp_video = str(Path(args.out).with_suffix(".video.mp4"))
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
         "-r", str(args.fps / max(1, args.stride)), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", tmp_video], stdin=subprocess.PIPE)

    caps = (cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r)))
    cur, frame = [-1, -1], [None, None]
    n = 0
    for il, ir, ts in pairs:
        for k, want in ((0, il), (1, ir)):
            while cur[k] < want:
                ok, frame[k] = caps[k].read()
                if not ok:
                    break
                cur[k] += 1
        if cur[0] != il or cur[1] != ir:
            break

        if args.rectified:
            vl = rect.remap(frame[0], "l")
            vr = rect.remap(frame[1], "r")
        else:
            vl, vr = frame[0], frame[1]
        top = np.hstack([cv2.resize(vl, (EW, EH)), cv2.resize(vr, (EW, EH))])
        cv2.line(top, (EW, 0), (EW, EH), (60, 60, 60), 1)

        rl = cv2.resize(rect.remap(frame[0], "l"), (DW, DH))
        rr = cv2.resize(rect.remap(frame[1], "r"), (DW, DH))
        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        d16 = sgbm.compute(gl, gr)
        d16r = mr.compute(gr, gl)
        disp = wls.filter(d16, gl, disparity_map_right=d16r).astype(np.float32) / 16.0
        dep = colorize_depth(disp, fx_d, rect.baseline_m,
                             args.min_depth, args.max_depth)

        k = int(np.searchsorted(imu.timestamps_ns, ts))
        k = max(0, min(k, len(quats) - 1))
        R_wc = quat_to_R(quats[k]) @ R_imu_cam
        roll, pitch, yaw = camera_attitude(R_wc, up, ref)
        ori = draw_orientation((ori_w, bot_h), R_wc, roll, pitch, yaw,
                               gyro_dps[k])

        canvas = np.zeros((OH, OW, 3), np.uint8)
        canvas[:PH] = top
        canvas[PH:PH + bot_h, :dep_w] = cv2.resize(dep, (dep_w, bot_h))
        canvas[PH:PH + bot_h, dep_w:] = ori
        canvas[PH + bot_h:] = legend
        tag = "RECTIFIED" if args.rectified else ""
        label(canvas, f"LEFT {tag}".strip(), 14, 30)
        label(canvas, f"RIGHT {tag}".strip(), EW + 14, 30)
        label(canvas, "SGBM + WLS DEPTH", 14, PH + 28)
        label(canvas, f"t {(ts - pairs[0][2]) / 1e9:6.2f} s", OW - 150, 30, 0.55)
        label(canvas, "Panoculon Labs", 14, PH - 12, 0.62, (235, 235, 235))
        ff.stdin.write(canvas.tobytes())
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(pairs)}")

    for c in caps:
        c.release()
    ff.stdin.close()
    ff.wait()

    if args.no_audio:
        Path(tmp_video).replace(args.out)
    else:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", tmp_video, "-i", str(mp4_l),
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", str(args.out)])
        if r.returncode == 0:
            Path(tmp_video).unlink(missing_ok=True)
        else:
            print("audio mux failed — keeping silent render", file=sys.stderr)
            Path(tmp_video).replace(args.out)
    print(f"wrote {args.out}: {n} frames, {OW}x{OH}")


if __name__ == "__main__":
    main()

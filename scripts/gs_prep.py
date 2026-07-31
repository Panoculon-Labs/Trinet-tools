#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Prepare a Trinet stereo take for Gaussian-splatting training.

Selects sharp, motion-calm, spatially spread keyframes, rectifies the LEFT
eye to a pinhole view, and writes a COLMAP text model (cameras/images/
points3D) using the SLAM/VIO trajectory for poses and the fused TSDF map as
the seed point cloud — no SfM run needed; poses and scale are already metric.

Usage:
    python3 scripts/gs_prep.py TAKE_PREFIX TRAJ.txt MAP.ply OUT_DIR \
        [--calibration CALIB] [--num-frames 220] [--max-gyro-dps 20] \
        [--width 1280]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

for _p in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from trinet_tools import calib_blob                              # noqa: E402
from trinet_tools.reader import read_imu, read_vts               # noqa: E402
from trinet_tools.stereo_align import Rectification              # noqa: E402


def quat_from_R(R):
    """COLMAP-order quaternion (w, x, y, z) from a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, \
            (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, \
            (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, \
            0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, \
            (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return w, x, y, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("traj")
    ap.add_argument("map_ply")
    ap.add_argument("out", type=Path)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=220)
    ap.add_argument("--max-gyro-dps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--seed-points", type=int, default=120000)
    args = ap.parse_args()

    import json
    raw = Path(args.calibration).read_bytes()
    try:
        calib = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        calib = calib_blob.unpack(raw)

    vts_l = read_vts(f"{args.take}_L.vts")
    imu = read_imu(f"{args.take}.imu")
    traj = np.loadtxt(args.traj, skiprows=1)
    traj_t = traj[:, 0]

    rect = Rectification(calib)
    W = args.width
    H = int(round(1080 * W / 1920)) & ~1
    sx, sy = W / 1920.0, H / 1080.0
    fx, fy = rect.P1[0, 0] * sx, rect.P1[1, 1] * sy
    cx, cy = rect.P1[0, 2] * sx, rect.P1[1, 2] * sy

    # cam0-rect from IMU/body: T_rect_i = R1h @ T_cam0_imu (as map fusion)
    R1h = np.eye(4)
    R1h[:3, :3] = rect.R1
    T_rect_i = R1h @ np.array(calib["T_cam0_imu"])

    # --- keyframe selection: calm + pose available + evenly spread ---
    ts_all = vts_l.sof_timestamps_ns.astype(np.int64)
    ti = np.asarray(imu.timestamps_ns, dtype=np.int64)
    w = np.degrees(np.linalg.norm(imu.gyro, axis=1))
    cand = []
    for idx, t in enumerate(ts_all):
        t_s = t / 1e9
        k = int(np.searchsorted(traj_t, t_s))
        if k <= 0 or k >= len(traj):
            continue
        if not (traj_t[k - 1] - 0.02 <= t_s <= traj_t[k] + 0.02):
            continue
        m = (ti >= t) & (ti <= t + int(33e6))
        if m.any() and float(w[m].mean()) <= args.max_gyro_dps:
            cand.append((idx, t, k))
    step = max(1, len(cand) // args.num_frames)
    sel = cand[::step][:args.num_frames]
    print(f"{len(cand)} calm candidates -> {len(sel)} keyframes")

    img_dir = args.out / "images"
    sparse = args.out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(f"{args.take}_L.mp4")
    cur, frame = -1, None
    images_txt = []
    n = 0
    for idx, t, k in sel:
        while cur < idx:
            ok, frame = cap.read()
            if not ok:
                break
            cur += 1
        if cur != idx:
            break
        rl = rect.remap(frame, "l")
        rl = cv2.resize(rl, (W, H))
        name = f"{t}.png"
        cv2.imwrite(str(img_dir / name), rl)

        a = (t / 1e9 - traj_t[k - 1]) / max(traj_t[k] - traj_t[k - 1], 1e-9)
        pos = (1 - a) * traj[k - 1, 1:4] + a * traj[k, 1:4]
        kq = k if a > 0.5 else k - 1
        x, y, z, qw_ = traj[kq, 4], traj[kq, 5], traj[kq, 6], traj[kq, 7]
        Rq = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * qw_), 2 * (x * z + y * qw_)],
            [2 * (x * y + z * qw_), 1 - 2 * (x * x + z * z), 2 * (y * z - x * qw_)],
            [2 * (x * z - y * qw_), 2 * (y * z + x * qw_), 1 - 2 * (x * x + y * y)],
        ])
        T_G_I = np.eye(4)
        T_G_I[:3, :3] = Rq
        T_G_I[:3, 3] = pos
        T_G_cam = T_G_I @ np.linalg.inv(T_rect_i)
        T_cam_G = np.linalg.inv(T_G_cam)
        qw, qx, qy, qz = quat_from_R(T_cam_G[:3, :3])
        tx, ty, tz = T_cam_G[:3, 3]
        n += 1
        images_txt.append(
            f"{n} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.6f} {ty:.6f} {tz:.6f} 1 {name}\n")
    cap.release()
    print(f"wrote {n} rectified keyframes at {W}x{H}")

    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    (sparse / "images.txt").write_text("".join(images_txt))

    import open3d as o3d
    src = o3d.io.read_triangle_mesh(args.map_ply)
    P = np.asarray(src.vertices)
    C = np.asarray(src.vertex_colors)
    if len(P) > args.seed_points:
        pick = np.random.default_rng(0).choice(len(P), args.seed_points,
                                               replace=False)
        P, C = P[pick], C[pick]
    with open(sparse / "points3D.txt", "w") as f:
        for i, (p, c) in enumerate(zip(P, np.clip(C * 255, 0, 255))):
            f.write(f"{i + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")
    print(f"seeded {len(P)} points from {args.map_ply}")
    print(f"COLMAP model at {sparse}")


if __name__ == "__main__":
    main()

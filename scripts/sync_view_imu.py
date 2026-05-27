#!/usr/bin/env python3
"""
Multi-camera synced viewer WITH per-camera Madgwick orientation.

The N camera panels are placed on the shared master clock (cross-camera time
sync); under each panel a 3D orientation gizmo shows that device's attitude
from Madgwick fusion, on the SAME timeline. The IMU is rotated into the camera
frame with the calibration extrinsic R_cam_imu (a board-mounting property common
to all units) and de-biased, as produced by the Trinet-Calibration pipeline, so
the gizmo is the camera's orientation.

Madgwick is the 6-DOF (accel+gyro) variant, implemented in numpy (no ahrs dep);
gyro bias is removed first so yaw drift is bounded over a clip.

Reuses CameraStream (streaming decode + global clock) from sync_view; the
orientation gizmo style matches visualize.py.

Usage:
    python scripts/sync_view_imu.py LEFT HEAD RIGHT --imu CALIB.json -o out.mp4
    (HEAD = the group master; it is auto-detected as the offset reference)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sync_view import (CameraStream, resolve_recording, _finalize, _put,
                       HEADER_H, LABEL_H, PANEL_GAP, BG, FG, ACCENT, WARN)
from trinet_tools.reader import read_imu

BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # X red, Y green, Z blue (BGR)


def madgwick_imu(accel, gyro, ts_s, beta=0.08):
    """6-DOF Madgwick (accel+gyro), numpy. Returns (N,4) xyzw unit quaternions.
    accel (N,3) m/s2, gyro (N,3) rad/s, ts_s (N,) seconds."""
    n = len(ts_s)
    if n == 0:
        return np.array([[0.0, 0.0, 0.0, 1.0]])
    # Seed level attitude from the first accel (roll/pitch; yaw=0).
    a0 = accel[0]
    if np.linalg.norm(a0) > 1e-6:
        roll = np.arctan2(a0[1], a0[2])
        pitch = np.arctan2(-a0[0], np.hypot(a0[1], a0[2]))
        q = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()  # xyzw
        q = np.array([q[3], q[0], q[1], q[2]])  # -> wxyz
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.zeros((n, 4))
    out[0] = [q[1], q[2], q[3], q[0]]  # xyzw
    dt_all = np.diff(ts_s)
    for i in range(1, n):
        dt = dt_all[i - 1]
        if dt <= 0 or dt > 0.5:
            out[i] = out[i - 1]
            continue
        q0, q1, q2, q3 = q
        gx, gy, gz = gyro[i]
        ax, ay, az = accel[i]
        # gyro rate of change of quaternion
        qd0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qd1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qd2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qd3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)
        nrm = np.sqrt(ax * ax + ay * ay + az * az)
        if nrm > 1e-9:
            ax, ay, az = ax / nrm, ay / nrm, az / nrm
            _2q0, _2q1, _2q2, _2q3 = 2 * q0, 2 * q1, 2 * q2, 2 * q3
            _4q0, _4q1, _4q2 = 4 * q0, 4 * q1, 4 * q2
            _8q1, _8q2 = 8 * q1, 8 * q2
            q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
            s1 = _4q1 * q3q3 - _2q3 * ax + 4 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
            s2 = 4 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
            s3 = 4 * q1q1 * q3 - _2q1 * ax + 4 * q2q2 * q3 - _2q2 * ay
            sn = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if sn > 1e-9:
                qd0 -= beta * s0 / sn
                qd1 -= beta * s1 / sn
                qd2 -= beta * s2 / sn
                qd3 -= beta * s3 / sn
        q = q + np.array([qd0, qd1, qd2, qd3]) * dt
        q = q / np.linalg.norm(q)
        out[i] = [q[1], q[2], q[3], q[0]]  # xyzw
    return out


def draw_orientation(canvas, x0, y0, w, h, quat, title):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    cv2.putText(canvas, title, (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)
    rot = Rotation.from_quat(quat).as_matrix()
    cx, cy = x0 + w // 2, y0 + h // 2 + 8
    scale = min(w, h) * 0.32
    view = np.array([[1, 0, 0], [0, 0.85, -0.53], [0, 0.53, 0.85]])
    comb = view @ rot
    for i in range(3):
        d = comb[:, i] * scale
        ex, ey = int(cx + d[0]), int(cy - d[1])
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), AXIS_COLORS[i], 2, tipLength=0.15)
        cv2.putText(canvas, "XYZ"[i], (ex + 4, ey - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, AXIS_COLORS[i], 1)


def load_calib(path):
    d = json.load(open(path))
    e, im = d["extrinsics"], d["imu"]
    return (np.array(e["R_cam_imu"], float),
            float(e.get("timeshift_cam_imu_sec", 0.0)),
            np.array(im.get("gyro_bias_rad_s", [0, 0, 0]), float),
            np.array(im.get("accel_bias_m_s2", [0, 0, 0]), float))


def prep_orientation(loaded, R, timeshift_s, gyro_bias, accel_bias, target_hz=120.0):
    """Madgwick orientation in the CAMERA frame vs master-global time.
    Returns (t_global_ns sorted, quats xyzw) or None."""
    R_cam = Rotation.from_matrix(R)
    ts_g, qs = [], []
    tshift_ns = timeshift_s * 1e9
    for mp4, vts in loaded:
        imu_path = mp4[:-4] + ".imu" if mp4.endswith(".mp4") else mp4 + ".imu"
        if not os.path.exists(imu_path) or vts is None or len(vts.sof_timestamps_ns) == 0:
            continue
        imu = read_imu(imu_path)
        if imu.num_samples < 2:
            continue
        rate = imu.actual_rate_hz or 562.0
        step = max(1, int(round(rate / target_hz)))
        acc = (imu.accel[::step].astype(np.float64) - accel_bias)
        gyr = (imu.gyro[::step].astype(np.float64) - gyro_bias)
        tns = imu.timestamps_ns[::step].astype(np.float64)
        q_imu = madgwick_imu(acc, gyr, (tns - tns[0]) / 1e9)
        # express as CAMERA orientation: R_world_cam = R_world_imu @ R_cam_imu^T
        q_cam = (Rotation.from_quat(q_imu) * R_cam.inv()).as_quat()
        off = float(vts.header.master_clock_offset_ns)
        skew = float(vts.header.clock_skew_ppb)
        ref = float(vts.sof_timestamps_ns[0])
        t = tns - tshift_ns
        g = t + off + skew * (t - ref) / 1e9
        ts_g.append(g); qs.append(q_cam)
    if not ts_g:
        return None
    t = np.concatenate(ts_g); q = np.concatenate(qs)
    order = np.argsort(t)
    return t[order].astype(np.int64), q[order]


def main():
    ap = argparse.ArgumentParser(description="Multi-camera synced viewer + per-camera Madgwick orientation")
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--imu", required=True, help="calibration.json with extrinsics.R_cam_imu")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--fps", type=float, default=29.65)
    ap.add_argument("--height", type=int, default=360, help="video panel height px")
    ap.add_argument("--orient-h", type=int, default=300, help="orientation gizmo height px")
    ap.add_argument("--rotate180", default="",
                    help="comma list of 0-based panel indices whose video to rotate 180 "
                         "(e.g. inverted-mounted wrist cams): --rotate180 0,2")
    ap.add_argument("--watermark", default="", help="bottom-right watermark text")
    ap.add_argument("--no-sync-info", action="store_true",
                    help="hide the time-sync readouts (cross-camera offset, per-panel "
                         "offset, and the 'master clock' annotation); keeps a plain "
                         "elapsed-time clock, the camera labels, and orientation gizmos")
    args = ap.parse_args()
    rotate_set = {int(i) for i in args.rotate180.split(",") if i.strip() != ""}

    R, timeshift_s, gyro_bias, accel_bias = load_calib(args.imu)

    cams, oris = [], []
    for r in args.recordings:
        label, loaded, meta = resolve_recording(r)
        cams.append(CameraStream(label, loaded, meta))
        print(f"  fusing orientation: {label} ...", flush=True)
        oris.append(prep_orientation(loaded, R, timeshift_s, gyro_bias, accel_bias))

    master_idx = next((i for i, c in enumerate(cams) if (c.meta or {}).get("role") == "master"), 0)
    t0 = max(c.t_start for c in cams)
    t1 = min(c.t_end for c in cams)
    if t1 <= t0:
        sys.exit("Recordings do not overlap on the shared clock (same session?)")
    dt_ns = int(1e9 / args.fps)

    panel_w = []
    for c in cams:
        cap = cv2.VideoCapture(c._segs[0])
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        cap.release()
        panel_w.append(max(1, int(round(args.height * w / h))))
    total_w = sum(panel_w) + PANEL_GAP * (len(cams) - 1)
    out_h = HEADER_H + args.height + LABEL_H + args.orient_h

    out_path = args.output
    tmp = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (total_w, out_h))
    if not writer.isOpened():
        sys.exit(f"Could not open VideoWriter for {tmp}")

    n_out = int((t1 - t0) // dt_ns) + 1
    print(f"Rendering {n_out} frames @ {args.fps} fps over {(t1-t0)/1e9:.1f}s, "
          f"{len(cams)} cams + Madgwick orientation ({total_w}x{out_h})")

    for k in range(n_out):
        T = t0 + k * dt_ns
        tc = (T - t0) / 1e9
        canvas = np.full((out_h, total_w, 3), BG, np.uint8)

        frames, gts = [], []
        for c in cams:
            fr, resid = c.frame_for(T)
            frames.append(fr); gts.append(T + resid)
        ref_g = gts[master_idx]
        spread_ms = (max(gts) - min(gts)) / 1e6

        x = 0
        oy = HEADER_H + args.height + LABEL_H
        for ci, c in enumerate(cams):
            pw = panel_w[ci]
            if frames[ci] is not None:
                fr_disp = cv2.rotate(frames[ci], cv2.ROTATE_180) if ci in rotate_set else frames[ci]
                cv2.resize(fr_disp, (pw, args.height),
                           dst=canvas[HEADER_H:HEADER_H + args.height, x:x + pw],
                           interpolation=cv2.INTER_AREA)
            ly = HEADER_H + args.height
            cv2.rectangle(canvas, (x, ly), (x + pw, ly + LABEL_H), (40, 40, 40), -1)
            _put(canvas, c.label, (x + 6, ly + 17), 0.5, FG)
            if not args.no_sync_info:
                off = (gts[ci] - ref_g) / 1e6
                _put(canvas, "ref" if ci == master_idx else f"{off:+.1f}ms",
                     (x + pw - 78, ly + 17), 0.45, ACCENT if abs(off) < 2 else WARN)

            ori = oris[ci]
            if ori is not None:
                t_arr, q_arr = ori
                j = int(np.searchsorted(t_arr, T))
                if j >= len(t_arr): j = len(t_arr) - 1
                elif j > 0 and (T - t_arr[j - 1]) <= (t_arr[j] - T): j -= 1
                draw_orientation(canvas, x, oy, pw, args.orient_h, q_arr[j],
                                 "Madgwick orientation (cam-frame)")
            x += pw + PANEL_GAP

        cv2.rectangle(canvas, (0, 0), (total_w, HEADER_H), (40, 40, 40), -1)
        if args.no_sync_info:
            _put(canvas, f"t = {tc:8.3f} s", (8, 23), 0.6, FG)
        else:
            _put(canvas, f"t = {tc:8.3f} s   (master clock)", (8, 23), 0.6, FG)
            _put(canvas, f"cross-cam {spread_ms:+5.2f} ms", (total_w - 230, 23), 0.55,
                 ACCENT if spread_ms < 2 else WARN)

        if args.watermark:
            (tw, th), _ = cv2.getTextSize(args.watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            wx, wy = total_w - tw - 14, out_h - 12
            cv2.putText(canvas, args.watermark, (wx + 1, wy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, args.watermark, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(canvas)
        if k % 200 == 0:
            print(f"  {k}/{n_out}", end="\r", flush=True)

    for c in cams:
        c.release()
    writer.release()
    _finalize(tmp, out_path, args.fps)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

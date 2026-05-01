#!/usr/bin/env python3
"""
Trinet Recording Visualizer

Renders synchronized video + IMU data as a composite MP4 video.
Orientation defaults to Madgwick MARG fusion (accel + gyro + mag); use
``--orientation gyro`` for gyro-only integration.

Usage:
    python visualize_recording.py captures/recording.mp4
    python visualize_recording.py captures/recording.mp4 -o output.mp4
    python visualize_recording.py captures/recording.mp4 --orientation gyro
    python visualize_recording.py captures/recording.mp4 --plots orientation,accel,gyro,sync_delay

Reads: recording.mp4 + recording.imu + recording.vts
Outputs: recording_viz.mp4 (or custom path with -o)
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.reader import (read_imu, read_vts,
                                  interpolate_imu_to_frames,
                                  get_per_frame_fsync_delay_us)
from trinet_tools.madgwick import run_madgwick

OUT_W, OUT_H = 1920, 1080
VIDEO_W = 1280
PANEL_W = OUT_W - VIDEO_W
PANEL_H = OUT_H

BG_COLOR = (30, 30, 30)
GRID_COLOR = (60, 60, 60)
TEXT_COLOR = (200, 200, 200)
CURSOR_COLOR = (255, 255, 255)
COLORS_RGB = [(66, 133, 244), (52, 168, 83), (234, 67, 53)]
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def integrate_orientation(gyro, timestamps_s):
    """Integrate gyroscope to quaternions (post-processing). Returns (N,4) xyzw array."""
    n = len(timestamps_s)
    if n == 0:
        return np.array([[0, 0, 0, 1]])

    max_samples = 50000
    if n > max_samples:
        step = n // max_samples
        idx = np.arange(0, n, step)
        g, ts = gyro[idx], timestamps_s[idx]
    else:
        idx, g, ts = None, gyro, timestamps_s

    m = len(ts)
    q = np.zeros((m, 4))
    q[0] = [0, 0, 0, 1]
    dt = np.diff(ts)

    for i in range(1, m):
        d = dt[i - 1]
        if d <= 0 or d > 0.5:
            q[i] = q[i - 1]
            continue
        rv = g[i] * d
        a = np.linalg.norm(rv)
        if a < 1e-10:
            q[i] = q[i - 1]
            continue
        q[i] = (Rotation.from_quat(q[i - 1]) * Rotation.from_rotvec(rv)).as_quat()

    if idx is not None:
        full = np.zeros((n, 4))
        for j in range(n):
            k = np.searchsorted(idx, j, side="right") - 1
            full[j] = q[max(k, 0)]
        return full
    return q


def draw_chart(canvas, x0, y0, w, h, t_arr, data_arr, t_center, t_window,
               title, ylabel, labels, colors):
    """Draw a scrolling time-series chart."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)

    margin_l, margin_r, margin_t, margin_b = 55, 10, 22, 25
    cx0 = x0 + margin_l
    cy0 = y0 + margin_t
    cw = w - margin_l - margin_r
    ch = h - margin_t - margin_b

    cv2.putText(canvas, title, (cx0, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    t_lo = max(0, t_center - t_window)
    t_hi = t_center + t_window

    mask = (t_arr >= t_lo) & (t_arr <= t_hi)
    tw = t_arr[mask]
    if len(tw) < 2:
        return

    vals = data_arr[mask]
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return
    y_min = float(np.nanmin(np.where(finite, vals, np.nan))) - 0.5
    y_max = float(np.nanmax(np.where(finite, vals, np.nan))) + 0.5
    if abs(y_max - y_min) < 1.0:
        mid = (y_max + y_min) / 2
        y_min, y_max = mid - 1, mid + 1

    def to_px(t_val, y_val):
        px = int(cx0 + (t_val - t_lo) / (t_hi - t_lo) * cw)
        py = int(cy0 + ch - (y_val - y_min) / (y_max - y_min) * ch)
        return px, py

    for frac in (0.25, 0.5, 0.75):
        gy = int(cy0 + ch * frac)
        cv2.line(canvas, (cx0, gy), (cx0 + cw, gy), GRID_COLOR, 1)

    for frac, label_frac in ((0.0, y_max), (0.5, (y_max + y_min) / 2), (1.0, y_min)):
        gy = int(cy0 + ch * frac)
        cv2.putText(canvas, f"{label_frac:.1f}", (x0 + 2, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_COLOR, 1)

    cv2.putText(canvas, ylabel, (x0 + 2, cy0 + ch // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    step = max(1, len(tw) // cw)
    for ch_idx in range(vals.shape[1]):
        color = colors[ch_idx % len(colors)]
        pts_x = tw[::step]
        pts_y = vals[::step, ch_idx]
        fin = np.isfinite(pts_x) & np.isfinite(pts_y)
        i = 0
        while i < len(pts_x):
            seg = []
            while i < len(pts_x) and fin[i]:
                seg.append(to_px(float(pts_x[i]), float(pts_y[i])))
                i += 1
            if len(seg) > 1:
                pts_np = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts_np], False, color, 1, cv2.LINE_AA)
            while i < len(pts_x) and not fin[i]:
                i += 1

    cx_px = int(cx0 + (t_center - t_lo) / (t_hi - t_lo) * cw)
    cv2.line(canvas, (cx_px, cy0), (cx_px, cy0 + ch), CURSOR_COLOR, 1)

    for i, lbl in enumerate(labels):
        lx = cx0 + cw - 100 + i * 35
        cv2.putText(canvas, lbl, (lx, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i % len(colors)], 1)

    cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), (80, 80, 80), 1)


def draw_orientation(canvas, x0, y0, w, h, quat, title="Orientation"):
    """Draw 3D orientation axes projected to 2D."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    rot = Rotation.from_quat(quat).as_matrix()
    cx, cy = x0 + w // 2, y0 + h // 2 + 10
    scale = min(w, h) * 0.3

    view_rot = np.array([[1, 0, 0], [0, 0.85, -0.53], [0, 0.53, 0.85]])
    combined = view_rot @ rot

    axis_labels = ["X", "Y", "Z"]
    for i in range(3):
        direction = combined[:, i] * scale
        ex = int(cx + direction[0])
        ey = int(cy - direction[1])
        color = AXIS_COLORS[i]
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 2, tipLength=0.15)
        cv2.putText(canvas, axis_labels[i], (ex + 5, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    parser = argparse.ArgumentParser(description="Render recording + IMU as visualization video")
    parser.add_argument("recording", help="Path to .mp4 (or base name)")
    parser.add_argument("-o", "--output", help="Output video path (default: <base>_viz.mp4)")
    parser.add_argument("--fps", type=float, default=0, help="Output FPS (default: match source)")
    parser.add_argument("--window", type=float, default=10.0, help="Chart time window +/-seconds")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds (0 = full)")
    parser.add_argument("--plots", type=str, default="orientation,accel,gyro,mag,sync_delay",
                        help="Comma-separated: orientation,accel,gyro,mag,temp,sync_delay")
    parser.add_argument(
        "--orientation",
        choices=("madgwick", "gyro"),
        default="madgwick",
        help="Attitude source: Madgwick MARG (default) or gyro-only integration",
    )
    args = parser.parse_args()

    active_plots = [p.strip() for p in args.plots.split(",") if p.strip()]

    base = args.recording
    for ext in (".mp4", ".imu", ".vts"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    video_path = f"{base}.mp4"
    imu_path = f"{base}.imu"
    vts_path = f"{base}.vts"
    out_path = args.output or f"{base}_viz.mp4"
    T_WINDOW = args.window

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    if not Path(imu_path).exists():
        print(f"Error: IMU file not found: {imu_path}")
        sys.exit(1)

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        sys.exit(1)

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120:
        fps = 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Loading IMU: {imu_path}")
    imu = read_imu(imu_path)

    vts = None
    if Path(vts_path).exists():
        print(f"Loading VTS: {vts_path}")
        vts = read_vts(vts_path)

    if vts is not None and vts.num_frames > cv_frames:
        total_frames = vts.num_frames
    elif cv_frames > 0:
        total_frames = cv_frames
    else:
        total_frames = max(1, int(imu.timestamps_s[-1] * fps)) if imu.num_samples > 0 else 1

    print(f"\n--- Recording Summary ---")
    print(f"  Video:  {src_w}x{src_h} @ {fps:.1f} fps, {total_frames} frames")
    print(f"  IMU:    {imu.num_samples:,} samples, {imu.duration_s:.2f}s, "
          f"{imu.actual_rate_hz:.1f} Hz actual")
    if vts is not None:
        actual_fps = vts.num_frames / max(vts.duration_s, 0.001)
        print(f"  VTS:    {vts.num_frames:,} frames, {vts.duration_s:.2f}s, {actual_fps:.1f} fps")

    if vts is not None and vts.num_frames > 0:
        frame_imu = interpolate_imu_to_frames(imu, vts)
    else:
        n = total_frames
        frame_imu = {
            "accel": np.zeros((n, 3), dtype=np.float32),
            "gyro": np.zeros((n, 3), dtype=np.float32),
            "mag": np.zeros((n, 3), dtype=np.float32),
        }
    frame_fsync_ts = frame_imu.get("fsync_frame_ts_ns")

    fsync_onset_idx = None
    if imu.fsync_delay_us is not None and imu.header.fsync_enabled:
        fd = imu.fsync_delay_us.astype(np.float64)
        prev = fd[0]
        for _si in range(1, len(fd)):
            if abs(fd[_si] - prev) > 2.0:
                fsync_onset_idx = _si
                break
            prev = fd[_si]
        if fsync_onset_idx is not None:
            onset_s = (imu.timestamps_ns[fsync_onset_idx] - imu.timestamps_ns[0]) / 1e9
            print(f"  FSYNC onset at IMU sample {fsync_onset_idx} "
                  f"({onset_s:.2f}s into stream, {fsync_onset_idx} pre-roll samples)")

    if args.orientation == "madgwick":
        print("\nMadgwick fusion (accel + gyro + mag)...")
        orientations = run_madgwick(
            imu.accel, imu.gyro, imu.mag, imu.timestamps_s.astype(np.float64), use_mag=True
        )
        orientation_title = "Orientation (Madgwick MARG)"
    else:
        print("\nIntegrating orientation from gyroscope...")
        orientations = integrate_orientation(imu.gyro, imu.timestamps_s)
        orientation_title = "Orientation (gyro-integrated)"

    gyro_deg = np.degrees(imu.gyro)

    if vts is not None and vts.num_frames > 0 and frame_fsync_ts is not None and len(frame_fsync_ts) == vts.num_frames:
        frame_ts_ns = frame_fsync_ts
    elif vts is not None and vts.num_frames > 0:
        frame_ts_ns = vts.best_timestamps_ns
    else:
        frame_ts_ns = None

    if fsync_onset_idx is not None:
        t0_ns = int(imu.timestamps_ns[fsync_onset_idx])
        print(f"  Timeline t0: FSYNC onset (imu index {fsync_onset_idx})")
    elif frame_ts_ns is not None and imu.num_samples > 0:
        t0_ns = min(int(imu.timestamps_ns[0]), int(frame_ts_ns[0]))
    elif imu.num_samples > 0:
        t0_ns = int(imu.timestamps_ns[0])
    else:
        t0_ns = 0
    t_imu = (imu.timestamps_ns.astype(np.float64) - t0_ns) / 1e9

    per_frame_delay_us = None
    sync_delay_frame_times = None
    per_frame_delay_plot = None
    if vts is not None and vts.num_frames > 0:
        per_frame_delay_us = get_per_frame_fsync_delay_us(imu, vts)
        if per_frame_delay_us is not None:
            sync_delay_frame_times = np.zeros(len(per_frame_delay_us))
            if frame_fsync_ts is not None and len(frame_fsync_ts) == len(per_frame_delay_us):
                for si in range(len(per_frame_delay_us)):
                    sync_delay_frame_times[si] = (int(frame_fsync_ts[si]) - t0_ns) / 1e9
            else:
                for si in range(len(per_frame_delay_us)):
                    sync_delay_frame_times[si] = (int(vts.best_timestamps_ns[si]) - t0_ns) / 1e9
            per_frame_delay_plot = per_frame_delay_us.astype(np.float64).copy()
            if fsync_onset_idx is not None:
                t_onset = int(imu.timestamps_ns[fsync_onset_idx])
                vref = vts.best_timestamps_ns.astype(np.uint64)
                for si in range(len(per_frame_delay_plot)):
                    if int(vref[si]) < t_onset:
                        per_frame_delay_plot[si] = np.nan
            print(f"  Sync Delay: mean={np.nanmean(per_frame_delay_us):.0f} us  "
                  f"median={np.nanmedian(per_frame_delay_us):.0f} us  "
                  f"max={np.nanmax(per_frame_delay_us):.0f} us")

    frame_times = np.zeros(total_frames)
    frame_imu_idx = np.zeros(total_frames, dtype=np.int64)
    if fsync_onset_idx is not None:
        for fi in range(total_frames):
            frame_times[fi] = fi / fps
            target_ns = np.uint64(t0_ns + int(fi / fps * 1e9))
            frame_imu_idx[fi] = min(
                int(np.searchsorted(imu.timestamps_ns, target_ns)),
                imu.num_samples - 1
            )
    else:
        for fi in range(total_frames):
            if frame_ts_ns is not None and fi < len(frame_ts_ns):
                frame_times[fi] = (int(frame_ts_ns[fi]) - t0_ns) / 1e9
                target_ns = frame_ts_ns[fi]
                frame_imu_idx[fi] = min(
                    int(np.searchsorted(imu.timestamps_ns, target_ns)),
                    imu.num_samples - 1
                )
            else:
                frame_times[fi] = fi / fps
                frame_imu_idx[fi] = int(fi / max(total_frames, 1) * (imu.num_samples - 1))

    if per_frame_delay_us is None:
        active_plots = [p for p in active_plots if p != "sync_delay"]
    if imu.temp_c is None:
        active_plots = [p for p in active_plots if p != "temp"]

    start_frame = 0
    end_frame = total_frames
    if args.start > 0:
        start_frame = max(0, min(int(args.start * fps), total_frames - 1))
    if args.end > 0:
        end_frame = max(start_frame + 1, min(int(args.end * fps), total_frames))
    render_count = end_frame - start_frame

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        print(f"Error: Cannot create output video {out_path}")
        sys.exit(1)

    print(f"\nDecoding video frames via ffmpeg...")
    decoded_frames = []
    last_good = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    frame_nbytes = src_w * src_h * 3

    ffmpeg_cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an", "-sn",
        "pipe:1",
    ]
    ff_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while len(decoded_frames) < total_frames:
        raw = ff_proc.stdout.read(frame_nbytes)
        if len(raw) < frame_nbytes:
            break
        vframe = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
        last_good = vframe
        decoded_frames.append(vframe)

    ff_proc.stdout.close()
    ff_proc.wait()

    while len(decoded_frames) < total_frames:
        decoded_frames.append(last_good.copy())

    print(f"  Decoded {len(decoded_frames)} frames")

    if start_frame > 0 or end_frame < total_frames:
        print(f"Rendering frames {start_frame}-{end_frame} "
              f"({render_count} frames, {render_count / fps:.1f}s) -> {out_path}")
    else:
        print(f"Rendering {total_frames} frames -> {out_path}")

    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    report_interval = max(1, render_count // 100)

    for fi in range(start_frame, end_frame):
        rendered = fi - start_frame
        if rendered % report_interval == 0 or fi == end_frame - 1:
            pct = (rendered + 1) / render_count * 100
            print(f"\r  [{pct:5.1f}%] Frame {rendered + 1}/{render_count}", end="", flush=True)

        vframe = decoded_frames[fi]

        scale_v = min(VIDEO_W / src_w, OUT_H / src_h)
        new_w = int(src_w * scale_v)
        new_h = int(src_h * scale_v)
        vframe_scaled = cv2.resize(vframe, (new_w, new_h))

        canvas[:] = BG_COLOR

        vx = (VIDEO_W - new_w) // 2
        vy = (OUT_H - new_h) // 2
        canvas[vy:vy + new_h, vx:vx + new_w] = vframe_scaled

        t_s = frame_times[fi]
        imu_i = frame_imu_idx[fi]
        fii = max(0, min(fi, len(frame_imu['accel']) - 1))

        overlay_lines = [
            f"Frame: {fi}/{total_frames}  t={t_s:.2f}s",
            f"IMU: {imu.actual_rate_hz:.0f} Hz  |  Cam: {fps:.0f} fps",
            f"Accel: [{frame_imu['accel'][fii, 0]:.1f}, "
            f"{frame_imu['accel'][fii, 1]:.1f}, "
            f"{frame_imu['accel'][fii, 2]:.1f}] m/s^2",
            f"Gyro:  [{np.degrees(frame_imu['gyro'][fii, 0]):.1f}, "
            f"{np.degrees(frame_imu['gyro'][fii, 1]):.1f}, "
            f"{np.degrees(frame_imu['gyro'][fii, 2]):.1f}] deg/s",
        ]
        if per_frame_delay_us is not None and fi < len(per_frame_delay_us):
            overlay_lines.append(f"Sync: {per_frame_delay_us[fi]:.0f} us")
        for i, line in enumerate(overlay_lines):
            cv2.putText(canvas, line, (vx + 10, vy + 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        n_charts = max(len(active_plots), 1)
        ch = OUT_H // n_charts
        slot = 0
        for pname in active_plots:
            y_off = slot * ch
            if pname == "orientation":
                draw_orientation(canvas, VIDEO_W, y_off, PANEL_W, ch,
                                 orientations[imu_i], orientation_title)
            elif pname == "accel":
                draw_chart(canvas, VIDEO_W, y_off, PANEL_W, ch,
                           t_imu, imu.accel, t_s, T_WINDOW,
                           "Accelerometer", "m/s^2", ["X", "Y", "Z"], COLORS_RGB)
            elif pname == "gyro":
                draw_chart(canvas, VIDEO_W, y_off, PANEL_W, ch,
                           t_imu, gyro_deg, t_s, T_WINDOW,
                           "Gyroscope", "deg/s", ["X", "Y", "Z"], COLORS_RGB)
            elif pname == "mag":
                draw_chart(canvas, VIDEO_W, y_off, PANEL_W, ch,
                           t_imu, imu.mag, t_s, T_WINDOW,
                           "Magnetometer", "uT", ["X", "Y", "Z"], COLORS_RGB)
            elif pname == "temp" and imu.temp_c is not None:
                draw_chart(canvas, VIDEO_W, y_off, PANEL_W, ch,
                           t_imu, imu.temp_c, t_s, T_WINDOW,
                           "Temperature", "C", [""], [(180, 180, 180)])
            elif pname == "sync_delay" and per_frame_delay_plot is not None:
                title_sd = (f"Sync Delay  "
                            f"mean={np.nanmean(per_frame_delay_us):.0f}  "
                            f"med={np.nanmedian(per_frame_delay_us):.0f}  "
                            f"max={np.nanmax(per_frame_delay_us):.0f} us")
                if fsync_onset_idx is not None:
                    title_sd += "  (pre-sync masked)"
                draw_chart(canvas, VIDEO_W, y_off, PANEL_W, ch,
                           sync_delay_frame_times, per_frame_delay_plot, t_s, T_WINDOW,
                           title_sd, "us", [""], [(100, 220, 255)])
            slot += 1

        writer.write(canvas)

    writer.release()
    decoded_frames.clear()
    print(f"\n\nDone! Output: {out_path}")
    print(f"  Duration: {render_count / fps:.1f}s @ {fps:.0f} fps")


if __name__ == "__main__":
    main()

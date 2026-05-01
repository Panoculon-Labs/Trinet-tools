#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet calibration toolkit.
"""
Extract TRIMU IMU SEI payloads from a Trinet MP4 recording and write
TRIMU001 v3 (.imu) + TRIVTS01 v2 (.vts) sidecars, plus a copy of the
video as video.mp4. The output folder layout matches what
tools/calibrate.py, calibrate_kalibr.py, and calibrate_viz.py
already consume.

Usage:
    python3 extract_sei_sidecars.py input.mp4 --out folder/

The MP4 must contain SEI user_data_unregistered NALs carrying the
TRINETIMUSEI UUID as written by the Trinet camera.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

TRIMU_UUID = bytes([
    0x54, 0x52, 0x49, 0x4E, 0x45, 0x54, 0x49, 0x4D,
    0x55, 0x53, 0x45, 0x49, 0x00, 0x01, 0x00, 0x00,
])

SEI_TYPE_USER_DATA_UNREGISTERED = 5
H264_NAL_TYPE_SEI = 6
IMU_SAMPLE_SIZE_V3 = 80


def ffmpeg_extract_annexb(mp4: Path, out_h264: Path) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(mp4),
        "-c", "copy", "-bsf:v", "h264_mp4toannexb",
        "-f", "h264", str(out_h264),
    ]
    subprocess.check_call(cmd)


def ffprobe_packet_pts_us(mp4: Path) -> list[int]:
    """Return list of packet PTS in microseconds (one per encoded frame)."""
    cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=print_section=0",
        str(mp4),
    ]
    out = subprocess.check_output(cmd, text=True)
    pts_us: list[int] = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            t = float(s)
        except ValueError:
            continue
        pts_us.append(int(round(t * 1e6)))
    return pts_us


def ffprobe_fps(mp4: Path) -> float:
    cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "csv=print_section=0",
        str(mp4),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        n, d = out.split("/")
        return float(n) / max(float(d), 1.0)
    return float(out or 30.0)


def split_nal_units(data: bytes):
    """Yield (offset_after_startcode, nal_length) for each NAL in an Annex-B stream."""
    starts = []
    i, n = 0, len(data)
    while i + 2 < n:
        if data[i] == 0 and data[i + 1] == 0:
            if i + 3 < n and data[i + 2] == 0 and data[i + 3] == 1:
                starts.append((i, 4)); i += 4; continue
            if data[i + 2] == 1:
                starts.append((i, 3)); i += 3; continue
        i += 1
    for idx, (off, sc_len) in enumerate(starts):
        s = off + sc_len
        e = starts[idx + 1][0] if idx + 1 < len(starts) else n
        while e > s and data[e - 1] == 0:
            e -= 1
        if e > s:
            yield s, e - s


def remove_emulation_prevention(raw: bytes) -> bytes:
    out = bytearray(len(raw))
    oi = 0
    i, n = 0, len(raw)
    while i < n:
        if i + 2 < n and raw[i] == 0 and raw[i + 1] == 0 and raw[i + 2] == 3:
            out[oi] = 0; oi += 1
            out[oi] = 0; oi += 1
            i += 3
        else:
            out[oi] = raw[i]; oi += 1
            i += 1
    return bytes(out[:oi])


def decode_trimu_sei(nal: bytes):
    """Return (samples_bytes, accel_fs, gyro_fs, version, num_samples) or None."""
    raw = remove_emulation_prevention(nal)
    pos = 1  # skip NAL header
    n = len(raw)
    while pos < n - 1:
        payload_type = 0
        while pos < n and raw[pos] == 0xFF:
            payload_type += 255; pos += 1
        if pos >= n:
            return None
        payload_type += raw[pos]; pos += 1

        payload_size = 0
        while pos < n and raw[pos] == 0xFF:
            payload_size += 255; pos += 1
        if pos >= n:
            return None
        payload_size += raw[pos]; pos += 1

        if pos + payload_size > n:
            return None

        payload = raw[pos:pos + payload_size]
        pos += payload_size

        if payload_type != SEI_TYPE_USER_DATA_UNREGISTERED:
            continue
        if len(payload) < 23 or payload[:16] != TRIMU_UUID:
            continue
        version = payload[16]
        num_samples = int.from_bytes(payload[17:19], "little")
        accel_fs = int.from_bytes(payload[19:21], "little")
        gyro_fs = int.from_bytes(payload[21:23], "little")
        samples = payload[23:23 + num_samples * IMU_SAMPLE_SIZE_V3]
        if len(samples) < num_samples * IMU_SAMPLE_SIZE_V3:
            return None
        return samples, accel_fs, gyro_fs, version, num_samples
    return None


def extract(mp4: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] source: {mp4}")
    print(f"[extract] output: {out_dir}")

    # 1. Annex-B bitstream
    h264_path = out_dir / "_stream.h264"
    ffmpeg_extract_annexb(mp4, h264_path)
    pts_us = ffprobe_packet_pts_us(mp4)
    fps = ffprobe_fps(mp4)
    print(f"[extract] fps={fps:.2f}  packets={len(pts_us)}")

    with open(h264_path, "rb") as f:
        bitstream = f.read()

    per_frame_samples: list[bytes] = []
    per_frame_first_sample_ts: list[int] = []
    per_frame_first_sample_fsync_us: list[float] = []
    pending_sample_bytes = bytearray()
    pending_first_ts = 0
    pending_first_fsync_us = 0.0
    pending_has = False

    accel_fs = 0
    gyro_fs = 0
    imu_version = 3
    fsync_seen = False

    frame_idx = -1
    for off, length in split_nal_units(bitstream):
        header = bitstream[off]
        nal_type = header & 0x1F
        nal_bytes = bytes(bitstream[off:off + length])
        if nal_type == H264_NAL_TYPE_SEI:
            result = decode_trimu_sei(nal_bytes)
            if result is not None:
                samples_bytes, a_fs, g_fs, ver, nsamp = result
                accel_fs = a_fs
                gyro_fs = g_fs
                imu_version = max(imu_version, ver)
                if nsamp > 0:
                    first_ts = struct.unpack_from("<Q", samples_bytes, 0)[0]
                    first_fsync_us = struct.unpack_from(
                        "<f", samples_bytes, 8 + 3 * 4 + 3 * 4 + 3 * 4 + 4 + 4 * 4 + 3 * 4
                    )[0]
                    if first_fsync_us > 0:
                        fsync_seen = True
                    if not pending_has:
                        pending_first_ts = first_ts
                        pending_first_fsync_us = first_fsync_us
                        pending_has = True
                    pending_sample_bytes.extend(samples_bytes)
        elif 1 <= nal_type <= 5:
            # VCL NAL => new frame. Flush pending SEI samples to this frame.
            frame_idx += 1
            if pending_has:
                per_frame_samples.append(bytes(pending_sample_bytes))
                per_frame_first_sample_ts.append(pending_first_ts)
                per_frame_first_sample_fsync_us.append(pending_first_fsync_us)
            else:
                per_frame_samples.append(b"")
                per_frame_first_sample_ts.append(0)
                per_frame_first_sample_fsync_us.append(0.0)
            pending_sample_bytes = bytearray()
            pending_has = False

    total_samples = sum(len(b) // IMU_SAMPLE_SIZE_V3 for b in per_frame_samples)
    print(f"[extract] frames={len(per_frame_samples)}  imu_samples={total_samples}  fsync={'yes' if fsync_seen else 'no'}")

    # 2. Write imu.bin (TRIMU001 v3)
    start_time_ns = next((ts for ts in per_frame_first_sample_ts if ts > 0), 0)
    video_start_ns = start_time_ns

    imu_rate_hz = 0
    # Estimate from median interval between adjacent sample timestamps.
    all_ts: list[int] = []
    for blob in per_frame_samples:
        for k in range(len(blob) // IMU_SAMPLE_SIZE_V3):
            ts = struct.unpack_from("<Q", blob, k * IMU_SAMPLE_SIZE_V3)[0]
            all_ts.append(ts)
    if len(all_ts) > 10:
        diffs = [b - a for a, b in zip(all_ts, all_ts[1:]) if b > a]
        if diffs:
            diffs.sort()
            med = diffs[len(diffs) // 2]
            if med > 0:
                imu_rate_hz = int(round(1e9 / med))

    flags = 1 if fsync_seen else 0
    header = struct.pack(
        "<8sIIHHQQI24s",
        b"TRIMU001",
        imu_version,
        imu_rate_hz,
        accel_fs,
        gyro_fs,
        start_time_ns,
        video_start_ns,
        flags,
        b"\x00" * 24,
    )
    imu_path = out_dir / "imu.bin"
    with open(imu_path, "wb") as f:
        f.write(header)
        # Samples must be strictly monotonic; read_imu() will enforce that,
        # but we deliver them in arrival order (already monotonic here).
        for blob in per_frame_samples:
            f.write(blob)
    print(f"[extract] wrote {imu_path} ({imu_path.stat().st_size} bytes, rate≈{imu_rate_hz} Hz)")

    # 3. Write frames.bin (TRIVTS01 v2)
    vts_header = struct.pack(
        "<8sII16s",
        b"TRIVTS01",
        2,  # version
        int(round(fps * 1000)),
        b"\x00" * 16,
    )
    vts_path = out_dir / "frames.bin"
    n_frames = len(per_frame_samples)
    if len(pts_us) < n_frames:
        # Pad using nominal fps if ffprobe missed some packets.
        step_us = int(round(1e6 / max(fps, 1.0)))
        last = pts_us[-1] if pts_us else 0
        while len(pts_us) < n_frames:
            last += step_us
            pts_us.append(last)

    with open(vts_path, "wb") as f:
        f.write(vts_header)
        for i in range(n_frames):
            first_ts = per_frame_first_sample_ts[i]
            fsync_us = per_frame_first_sample_fsync_us[i]
            sof_ns = 0
            if first_ts > 0:
                sof_ns = int(first_ts - fsync_us * 1000.0)
            entry = struct.pack(
                "<IQIQ",
                i,              # frame_number
                sof_ns,         # sof_timestamp_ns
                i,              # venc_seq (we don't have the encoder seq; use idx)
                int(pts_us[i]),
            )
            f.write(entry)
    print(f"[extract] wrote {vts_path} ({vts_path.stat().st_size} bytes, {n_frames} entries)")

    # 4. Build a clean video.mp4 that OpenCV can decode end-to-end.
    #    Android-wrapped Trinet MP4s often have leading access units that
    #    libavcodec's H.264 decoder refuses, causing `cv2.VideoCapture.read()`
    #    to bail after frame 1. Re-encode with libx264 to normalize the stream.
    video_dst = out_dir / "video.mp4"
    tmp_mp4 = out_dir / "_video_clean.mp4"
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(mp4),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(tmp_mp4),
    ])
    tmp_mp4.replace(video_dst)

    # 5. Count actually-decodable frames and drop leading VTS entries
    #    that were discarded during re-encode so frame 0 aligns.
    import cv2  # imported late to avoid an unnecessary dependency on --help
    cap = cv2.VideoCapture(str(video_dst))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    drop = n_frames - decoded
    if drop > 0:
        print(f"[extract] libx264 dropped {drop} leading frame(s); re-aligning frames.bin")
        with open(vts_path, "rb") as f:
            vts_data = f.read()
        entry_size = 24
        body = vts_data[32:]
        new_body = bytearray()
        for i in range(decoded):
            src = body[(i + drop) * entry_size:(i + drop + 1) * entry_size]
            _, sof, _, pts = struct.unpack("<IQIQ", src)
            new_body += struct.pack("<IQIQ", i, sof, i, pts)
        with open(vts_path, "wb") as f:
            f.write(vts_data[:32])
            f.write(new_body)
    print(f"[extract] wrote {video_dst} ({decoded} decodable frames)")

    # 6. Cleanup
    try:
        h264_path.unlink()
    except OSError:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("mp4", type=Path, help="Trinet MP4 with SEI IMU payloads")
    p.add_argument("--out", type=Path, required=True, help="Output folder")
    args = p.parse_args(argv)
    if not args.mp4.exists():
        print(f"error: {args.mp4} not found", file=sys.stderr)
        return 2
    extract(args.mp4, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

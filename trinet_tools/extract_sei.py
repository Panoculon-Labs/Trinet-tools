#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet calibration toolkit.
"""
Extract TRIMU IMU SEI payloads from a Trinet MP4 recording and write
TRIMU001 (.imu) + TRIVTS01 (.vts) sidecars, plus a copy of the
video as video.mp4. The .imu version tracks whatever the camera embedded
(v3/v4 cameras carry a per-sample frame-sync delay; v5 cameras carry live
magnetometer data + a mag_age_us timestamp in the same trailing float; v6
cameras add a per-frame mid-exposure timing block). The `.vts` is written as
TRIVTS01 v4 (sof + exposure + readout) for v6 cameras, else v2. The
output folder layout matches what tools/calibrate.py, calibrate_kalibr.py,
and calibrate_viz.py already consume.

Usage:
    python3 extract_sei_sidecars.py input.mp4 --out folder/

The MP4 must contain SEI user_data_unregistered NALs carrying the
TRINETIMUSEI UUID as written by the Trinet camera.
"""

from __future__ import annotations

import argparse
import json
import os
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

# "TRINETAAC" — v4+ cameras embed stereo AAC-LC in its own
# user_data_unregistered SEI NAL alongside the IMU SEI. Older cameras never
# emit it, so an absent track is normal rather than an error. Layout after the
# UUID: version(1) sample_rate(4 LE) channels(1) num_frames(2 LE), then per
# frame pts_us(8 LE, CLOCK_MONOTONIC - the same clock as the IMU SEI's
# frame_sof_ts_ns) len(2 LE) adts[len].
TRINET_AAC_UUID = bytes([
    0x54, 0x52, 0x49, 0x4E, 0x45, 0x54, 0x41, 0x41,
    0x43, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
])
AAC_SEI_HEADER_SIZE = 16 + 1 + 4 + 1 + 2
H264_NAL_TYPE_SEI = 6
IMU_SAMPLE_SIZE_V3 = 80
# Byte offset of the trailing float32 within an 80-byte sample:
#   ts(8) + accel(12) + gyro(12) + mag(12) + temp(4) + quat(16) + lin_accel(12) = 76
# It is fsync_delay_us on v3/v4 cameras and mag_age_us on v5 cameras.
SAMPLE_TRAILING_FLOAT_OFFSET = 8 + 3 * 4 + 3 * 4 + 3 * 4 + 4 + 4 * 4 + 3 * 4
IMU_HDR_FLAG_FSYNC = 0x01
IMU_HDR_FLAG_MAG = 0x02


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


def ffprobe_decode_map(mp4: Path) -> list[int]:
    """Decode the video stream and return, for each decoded frame in order,
    the index of the source packet it came from (matched by byte position).

    This is the exact frame<->packet correspondence: if the decoder skips
    undecodable packets (e.g. a capture that joined mid-GOP), the skipped
    packets simply never appear in the list. Never infer the mapping from
    frame counts -- assuming a count shortfall sits at the head is what
    silently time-shifted every frame of 4-cam rig captures by ~2.5 s.
    """
    out = subprocess.check_output(
        ["ffprobe", "-hide_banner", "-loglevel", "error",
         "-select_streams", "v:0",
         "-show_packets", "-show_frames",
         "-show_entries", "packet=pos:frame=pkt_pos",
         "-of", "json", str(mp4)])
    items = json.loads(out).get("packets_and_frames", [])
    pkt_pos: list[int] = []
    frame_pos: list[int] = []
    for it in items:
        if it.get("type") == "packet" and "pos" in it:
            pkt_pos.append(int(it["pos"]))
        elif it.get("type") == "frame" and "pkt_pos" in it:
            frame_pos.append(int(it["pkt_pos"]))
    idx_of = {pos: i for i, pos in enumerate(pkt_pos)}
    return [idx_of[p] for p in frame_pos if p in idx_of]


def ffprobe_packet_count(mp4: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-hide_banner", "-loglevel", "error",
         "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets",
         "-of", "csv=print_section=0", str(mp4)], text=True)
    return int(out.strip() or 0)


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
    # C-speed scan: find every 00 00 01, then classify whether a leading 00
    # makes it a 4-byte start code. Semantically identical to the original
    # byte-at-a-time walk (which cost seconds on 100 MB streams).
    starts = []
    i, n = 0, len(data)
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        if j > 0 and data[j - 1] == 0:
            starts.append((j - 1, 4))
        else:
            starts.append((j, 3))
        i = j + 3
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


def decode_trinet_aac_sei(nal: bytes):
    """Return (sample_rate, channels, [(pts_us, adts_bytes)]) or None."""
    raw = remove_emulation_prevention(nal)
    idx = raw.find(TRINET_AAC_UUID[:9])          # match the ASCII prefix
    if idx < 0 or idx + AAC_SEI_HEADER_SIZE > len(raw):
        return None
    p = raw[idx:]
    sample_rate, = struct.unpack_from("<I", p, 17)
    channels = p[21]
    num_frames, = struct.unpack_from("<H", p, 22)
    out = []
    pos = AAC_SEI_HEADER_SIZE
    for _ in range(num_frames):
        if pos + 10 > len(p):
            break
        pts_us, = struct.unpack_from("<Q", p, pos)
        ln, = struct.unpack_from("<H", p, pos + 8)
        pos += 10
        if pos + ln > len(p):
            break
        out.append((pts_us, p[pos:pos + ln]))
        pos += ln
    if not out:
        return None
    return sample_rate, channels, out


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
        # v6 inserts a per-frame timing block between gyro_fs and the samples:
        #   frame_sof_ts_ns(8) + exposure_us(4) + timing_flags(1) + readout_time_us(4)
        #   = 17 bytes → samples start at 23 + 17 = 40.
        # Samples remain the 80-byte v5 layout.
        sample_base = 40 if version >= 6 else 23
        if len(payload) < sample_base:
            continue
        # v6 per-frame timing block at payload[23:40]: mid-exposure frame time +
        # applied exposure + rolling-shutter readout span + timing flags. None for
        # v<6 cameras. timing_flags (1 byte here; uint32 in the .vts) carries
        # MID_EXPOSURE/EXPOSURE_VALID/READOUT_VALID/FRAME_CENTERED — FRAME_CENTERED
        # (0x08) marks frame_sof_ts_ns as referencing the MIDDLE row. The byte is
        # passed through verbatim into the .vts entry below, preserving every bit.
        timing = None
        if version >= 6:
            frame_sof_ts_ns = int.from_bytes(payload[23:31], "little")
            exposure_us = int.from_bytes(payload[31:35], "little")
            timing_flags = payload[35]
            readout_time_us = int.from_bytes(payload[36:40], "little")
            timing = (frame_sof_ts_ns, exposure_us, timing_flags, readout_time_us)
        samples = payload[sample_base:sample_base + num_samples * IMU_SAMPLE_SIZE_V3]
        if len(samples) < num_samples * IMU_SAMPLE_SIZE_V3:
            return None
        return samples, accel_fs, gyro_fs, version, num_samples, timing
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
    per_frame_timing: list = []          # v6: (sof_ns, exposure_us, flags, readout_us) | None
    pending_sample_bytes = bytearray()
    pending_first_ts = 0
    pending_first_fsync_us = 0.0
    pending_timing = None
    pending_has = False

    accel_fs = 0
    gyro_fs = 0
    imu_version = 3
    fsync_seen = False
    mag_seen = False

    # ADTS frames in stream order, with their device-clock PTS (us, same
    # CLOCK_MONOTONIC base as the IMU SEI's frame_sof_ts_ns).
    aac_pts_frames: list[tuple[int, bytes]] = []
    aac_rate = aac_channels = 0

    frame_idx = -1
    for off, length in split_nal_units(bitstream):
        header = bitstream[off]
        nal_type = header & 0x1F
        nal_bytes = bytes(bitstream[off:off + length])
        if nal_type == H264_NAL_TYPE_SEI:
            audio = decode_trinet_aac_sei(nal_bytes)
            if audio is not None:
                aac_rate, aac_channels, aframes = audio
                aac_pts_frames.extend(aframes)
            result = decode_trimu_sei(nal_bytes)
            if result is not None:
                samples_bytes, a_fs, g_fs, ver, nsamp, timing = result
                accel_fs = a_fs
                gyro_fs = g_fs
                imu_version = max(imu_version, ver)
                if nsamp > 0:
                    first_ts = struct.unpack_from("<Q", samples_bytes, 0)[0]
                    trailing_f = struct.unpack_from(
                        "<f", samples_bytes, SAMPLE_TRAILING_FLOAT_OFFSET
                    )[0]
                    # The trailing float is fsync_delay_us on v3/v4 cameras and
                    # mag_age_us on v5 cameras (which have no frame-sync delay).
                    if ver >= 5:
                        mag_seen = True
                        first_fsync_us = 0.0
                    else:
                        first_fsync_us = trailing_f
                        if first_fsync_us > 0:
                            fsync_seen = True
                    if not pending_has:
                        pending_first_ts = first_ts
                        pending_first_fsync_us = first_fsync_us
                        pending_timing = timing
                        pending_has = True
                    pending_sample_bytes.extend(samples_bytes)
        elif 1 <= nal_type <= 5:
            # VCL NAL => new frame. Flush pending SEI samples to this frame.
            frame_idx += 1
            if pending_has:
                per_frame_samples.append(bytes(pending_sample_bytes))
                per_frame_first_sample_ts.append(pending_first_ts)
                per_frame_first_sample_fsync_us.append(pending_first_fsync_us)
                per_frame_timing.append(pending_timing)
            else:
                per_frame_samples.append(b"")
                per_frame_first_sample_ts.append(0)
                per_frame_first_sample_fsync_us.append(0.0)
                per_frame_timing.append(None)
            pending_sample_bytes = bytearray()
            pending_timing = None
            pending_has = False

    total_samples = sum(len(b) // IMU_SAMPLE_SIZE_V3 for b in per_frame_samples)
    print(f"[extract] frames={len(per_frame_samples)}  imu_samples={total_samples}  "
          f"version={imu_version}  fsync={'yes' if fsync_seen else 'no'}  mag={'yes' if mag_seen else 'no'}")

    # 2. Write imu.bin (TRIMU001 — version mirrors what the camera embedded)
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

    flags = (IMU_HDR_FLAG_FSYNC if fsync_seen else 0) | (IMU_HDR_FLAG_MAG if mag_seen else 0)
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

    # 3. Session-start backlog trim + exact frame<->packet mapping.
    #
    #    At STREAMON the camera flushes the encoded-frame queue left over from
    #    the previous streaming session: ~1-3 s of frames whose SEI sof is
    #    correct *for their old content* and therefore older than the live IMU
    #    samples delivered alongside them by the whole idle gap (seconds to
    #    hours). They are stale pictures and put a giant time gap at the head
    #    of the frame timeline, so drop every frame up to the last stale one.
    #    The IMU samples embedded in their SEIs are current and were all kept
    #    in imu.bin above. (Detectable on v6 cameras only -- older SEI
    #    versions carry no per-frame sof to compare against.)
    n_frames_total = len(per_frame_samples)
    STALE_SOF_NS = 500_000_000          # sof lagging its own samples by >0.5 s
    head_trim = 0
    for i in range(min(n_frames_total, int(15 * max(fps, 1.0)))):
        timing = per_frame_timing[i]
        first_ts = per_frame_first_sample_ts[i]
        if (timing is not None and first_ts > 0
                and first_ts - int(timing[0]) > STALE_SOF_NS):
            head_trim = i + 1
    if head_trim:
        print(f"[extract] dropping {head_trim} stale head frame(s) "
              f"(previous session's backlog, flushed at stream start)")

    #    Exact mapping of decoded frames to source packets (by byte position):
    #    the re-encode below consumes the decoder output in order, so output
    #    frame n came from packet dec2pkt[n0 + n].
    dec2pkt = ffprobe_decode_map(mp4)
    if len(dec2pkt) < len(pts_us):
        print(f"[extract] decoder skipped {len(pts_us) - len(dec2pkt)} "
              f"undecodable packet(s)")
    n0 = next((n for n, p in enumerate(dec2pkt) if p >= head_trim),
              len(dec2pkt))
    kept_pkts = dec2pkt[n0:]

    if len(pts_us) < n_frames_total:
        # Pad using nominal fps if ffprobe missed some packets.
        step_us = int(round(1e6 / max(fps, 1.0)))
        last = pts_us[-1] if pts_us else 0
        while len(pts_us) < n_frames_total:
            last += step_us
            pts_us.append(last)

    # 4. Build a clean video.mp4 that OpenCV can decode end-to-end, with the
    #    stale head frames dropped. Android-wrapped Trinet MP4s often have
    #    leading access units that libavcodec's H.264 decoder refuses, causing
    #    `cv2.VideoCapture.read()` to bail after frame 1. Re-encode with
    #    libx264 to normalize the stream; -fps_mode passthrough so the
    #    encoder/muxer never drops or duplicates frames on its own.
    video_dst = out_dir / "video.mp4"
    tmp_mp4 = out_dir / "_video_clean.mp4"

    # The camera's audio rides in TRINETAAC SEI NALs, not in an audio track,
    # and this re-encode strips every SEI — so the ADTS frames collected above
    # are written out and muxed back in as a real track. Audio capture starts
    # a few seconds after video, so the track is placed at its true offset on
    # the shared device clock. NEVER pass -shortest here: it truncated the
    # audio/video length difference off the *video tail*, and the old
    # count-based realign then mis-attributed the loss to the head, silently
    # time-shifting the whole frame timeline by seconds.
    aac_path = out_dir / "_audio.aac"
    have_audio = bool(aac_pts_frames)
    audio_delay_s = 0.0
    if have_audio:
        # Device-clock time of the first kept video frame.
        ref_ns = 0
        for i in (kept_pkts or range(n_frames_total)):
            if i >= n_frames_total:
                break
            t = per_frame_timing[i]
            if t is not None and t[0] > 0:
                ref_ns = int(t[0])
                break
            if per_frame_first_sample_ts[i] > 0:
                ref_ns = per_frame_first_sample_ts[i]
                break
        if ref_ns:
            # Drop audio from before the kept video start, then place the
            # remainder at its residual offset (device clocks cancel).
            k0 = 0
            while (k0 < len(aac_pts_frames)
                   and aac_pts_frames[k0][0] * 1000 < ref_ns):
                k0 += 1
            aac_pts_frames = aac_pts_frames[k0:]
            have_audio = bool(aac_pts_frames)
            if have_audio:
                # The re-encode synthesizes a video timeline starting at 0, so
                # the audio offset is just the device-clock difference.
                audio_delay_s = (aac_pts_frames[0][0] * 1000 - ref_ns) / 1e9
    if have_audio:
        with open(aac_path, "wb") as f:
            for _pts, a in aac_pts_frames:
                f.write(a)
        print(f"[extract] audio: {len(aac_pts_frames)} AAC frames, "
              f"{aac_rate} Hz, {aac_channels} ch, +{audio_delay_s:.3f} s offset")

    # The capture's container PTS are USB *arrival* times: the session-start
    # backlog arrives as a wire-speed burst with duplicated/non-monotonic PTS
    # that make downstream muxers and decoders reorder frames unpredictably.
    # Synthesize a clean uniform timeline in encode order instead (setpts) --
    # every consumer of video.mp4 is frame-indexed and gets real per-frame
    # times from frames.bin, so container timestamps only need to be sane.
    # The frame period comes from the camera's own sof deltas when available.
    true_fps = fps
    sof_kept = [int(per_frame_timing[i][0]) for i in kept_pkts
                if i < n_frames_total and per_frame_timing[i] is not None]
    sof_d = sorted(b - a for a, b in zip(sof_kept, sof_kept[1:]) if b > a)
    if sof_d:
        med = sof_d[len(sof_d) // 2]
        if med > 0 and 1.0 <= 1e9 / med <= 240.0:
            true_fps = 1e9 / med

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4)]
    if have_audio:
        cmd += ["-itsoffset", f"{audio_delay_s:.6f}", "-i", str(aac_path)]
    # Encoder settings are part of the calibration chain: recovered fx moves
    # 2-4 px under ANY re-encode change (superfast/crf16, veryfast/crf17 and
    # h264_vaapi all A/B-failed at a <1 px bar on 2026-08-12 — SUBPIX corners
    # integrate the compression noise). Do not touch preset/crf/threads
    # without an A/B against a stored capture; the real fix is detecting
    # corners on the original stream so the re-encode leaves the accuracy
    # chain entirely.
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-fps_mode:v", "passthrough"]
    filters = []
    if n0 > 0:
        filters.append(f"select=gte(n\\,{n0})")
    filters.append(f"setpts=N/({true_fps:.6f}*TB)")
    cmd += ["-vf", ",".join(filters)]
    if have_audio:
        # -c:a copy keeps the camera's AAC bit-exact; no second lossy pass.
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [str(tmp_mp4)]
    subprocess.check_call(cmd)
    tmp_mp4.replace(video_dst)
    aac_path.unlink(missing_ok=True)

    # 5. Verify the produced frame count against the mapping, then write
    #    frames.bin with one entry per video.mp4 frame, in the same order.
    #    v6 cameras carry a per-frame mid-exposure timing block -> emit
    #    TRIVTS01 v4 (sof + exposure + flags + readout). Older cameras ->
    #    TRIVTS01 v2 (sof from the frame-sync delay, or 0 / PTS-fallback
    #    on v5).
    out_n = ffprobe_packet_count(video_dst)
    if out_n != len(kept_pkts):
        print(f"[extract] WARNING: re-encode produced {out_n} frames, "
              f"expected {len(kept_pkts)} -- truncating the map at the tail")
        kept_pkts = kept_pkts[:out_n]

    use_v6 = imu_version >= 6
    vts_version = 4 if use_v6 else 2
    vts_header = struct.pack(
        "<8sII16s", b"TRIVTS01", vts_version, int(round(fps * 1000)), b"\x00" * 16,
    )
    vts_path = out_dir / "frames.bin"
    with open(vts_path, "wb") as f:
        f.write(vts_header)
        for i, pkt in enumerate(kept_pkts):
            timing = per_frame_timing[pkt] if pkt < n_frames_total else None
            pts_i = int(pts_us[pkt]) if pkt < len(pts_us) else 0
            if use_v6 and timing is not None:
                sof_ns, exposure_us, flags, readout_us = timing
                # v4 entry: v2 fields + exposure_us, entry_flags, readout_time_us.
                # sof is the device mid-exposure frame time (already exposure-centred).
                entry = struct.pack("<IQIQIII", i, int(sof_ns), i, pts_i,
                                    int(exposure_us), int(flags), int(readout_us))
            elif use_v6:
                entry = struct.pack("<IQIQIII", i, 0, i, pts_i, 0, 0, 0)
            else:
                first_ts = per_frame_first_sample_ts[pkt] if pkt < n_frames_total else 0
                fsync_us = (per_frame_first_sample_fsync_us[pkt]
                            if pkt < n_frames_total else 0.0)
                # v3/v4 cameras: sof = first sample time - frame-sync delay. v5
                # cameras have no frame-sync delay -> sof=0, readers fall back to PTS.
                sof_ns = (int(first_ts - fsync_us * 1000.0)
                          if (imu_version < 5 and first_ts > 0) else 0)
                entry = struct.pack("<IQIQ", i, sof_ns, i, pts_i)
            f.write(entry)
    print(f"[extract] wrote {vts_path} ({vts_path.stat().st_size} bytes, "
          f"{len(kept_pkts)} entries, TRIVTS01 v{vts_version})")

    #    Decodability sanity check (verification only — NEVER used to shift
    #    the timeline; the frame<->entry mapping comes from ffprobe_decode_map).
    #    Costs a third full decode (~7 s/unit), and the calibration pipeline
    #    re-decodes this file minutes later anyway (a truncated write shows
    #    up there as "0 usable frames") — so it's opt-in for debugging.
    if os.environ.get("EXTRACT_VERIFY") == "1":
        import cv2  # imported late to avoid an unnecessary dependency
        cap = cv2.VideoCapture(str(video_dst))
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        if decoded != len(kept_pkts):
            print(f"[extract] WARNING: OpenCV decodes only "
                  f"{decoded}/{len(kept_pkts)} frames of {video_dst.name}")
        print(f"[extract] wrote {video_dst} ({decoded} decodable frames)")
    else:
        print(f"[extract] wrote {video_dst} "
              f"({len(kept_pkts)} frames, verify skipped)")

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

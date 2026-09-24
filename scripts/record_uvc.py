#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet calibration toolkit.
"""
Record a Trinet camera over USB-UVC on Linux and write a recording folder that
matches what the Android SDK produces.

    <devid8>_recording_<YYYYmmdd_HHMMSS>/
        video.mp4     H.264, stream-copied from the camera (never re-encoded)
        imu.bin       TRIMU001 sidecar
        frames.bin    TRIVTS01 sidecar
        meta.json     device / video metadata

Capture is the only thing this script does itself; the SEI -> sidecar work is
delegated to trinet_tools.extract_sei.extract(), which already understands the
v3/v4/v5/v6 SEI layouts and writes the folder layout the rest of the toolkit
(calibrate.py, sync_view.py, visualize.py, reader.py) consumes.

Transport note: firmware >= 0.3.1 streams over USB *bulk* rather than
isochronous. Linux uvcvideo handles bulk natively — nothing special is needed
here — but see --list if the camera is not picked up automatically.

Stereo cameras (Trinet Pro Stereo, rolling shutter v5, and Pro Stereo GS,
global shutter v6) carry both eyes in one side-by-side 3840x1080 frame on the
same USB function and PID as the mono camera. The frame size is therefore
read from what the camera advertises, widest first, exactly as the Android
SDK does. It is never assumed. meta.json then records video.layout = "sbs"
(left half = physically left eye) and video.shutter = "global" | "rolling",
matching the SDK's keys.

Usage:
    python3 scripts/record_uvc.py                    # auto-detect, until Ctrl-C
    python3 scripts/record_uvc.py -d 60              # fixed 60 s
    python3 scripts/record_uvc.py -o ~/captures      # parent directory
    python3 scripts/record_uvc.py --device /dev/video2
    python3 scripts/record_uvc.py --list             # show Trinet video nodes
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trinet_tools.extract_sei import extract  # noqa: E402
from trinet_tools.reader import read_vts  # noqa: E402

USB_VID, USB_PID = 0x2207, 0x0016
DEFAULT_FPS = 30


# ---------------------------------------------------------------------------
#  Device discovery
# ---------------------------------------------------------------------------
def _usb_ids_for(node: str):
    """Walk sysfs up from a video node to its USB device's VID/PID/serial."""
    path = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(node)}/device")
    for _ in range(6):
        vid_p, pid_p = os.path.join(path, "idVendor"), os.path.join(path, "idProduct")
        if os.path.exists(vid_p) and os.path.exists(pid_p):
            try:
                vid = int(Path(vid_p).read_text().strip(), 16)
                pid = int(Path(pid_p).read_text().strip(), 16)
            except (OSError, ValueError):
                return None
            ser_p = os.path.join(path, "serial")
            serial = Path(ser_p).read_text().strip() if os.path.exists(ser_p) else ""
            return vid, pid, serial
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def _does_h264_capture(node: str) -> bool:
    """A UVC camera exposes several video nodes; only some do H.264 capture.

    Must use --list-formats: --all reports the capability bits ("Video
    Capture") but does NOT enumerate pixel formats, so it can never show
    H264. On a Trinet the metadata node also reports "Video Capture" with an
    empty format list, which is exactly what this has to filter out.
    """
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "H264" in out.upper()


def h264_sizes(formats_ext: str) -> list[tuple[int, int]]:
    """Frame sizes listed under the H264 format in `v4l2-ctl --list-formats-ext`
    output, widest first.

    Only the H264 section counts. The stereo camera advertises MJPEG *first* at
    the same size, and MJPEG carries no SEI (so no IMU), so it must never be
    what we pick a size from by accident.
    """
    sizes, in_h264 = [], False
    for line in formats_ext.splitlines():
        fmt = re.search(r"\[\d+\]:\s*'(\w+)'", line)
        if fmt:
            in_h264 = fmt.group(1).upper() == "H264"
            continue
        m = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if in_h264 and m:
            wh = (int(m.group(1)), int(m.group(2)))
            if wh not in sizes:
                sizes.append(wh)
    return sorted(sizes, key=lambda wh: (wh[0] * wh[1], wh[0]), reverse=True)


def advertised_h264_size(node: str):
    """(w, h) the camera advertises for H.264, widest first; None if unreadable."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sizes = h264_sizes(out)
    return sizes[0] if sizes else None


def layout_of(width: int, height: int):
    """"sbs" for a side-by-side stereo frame, else None (mono omits the key).

    Same rule as the SDK's StreamLayout.of: a stereo pair is 2:1 wider than a
    mono frame of the same height, so >= 3:1 separates it from every single-eye
    format with room to spare, 21:9 crops included.
    """
    return "sbs" if height > 0 and width >= height * 3 else None


def find_trinet_nodes() -> list[tuple[str, str]]:
    """[(node, serial)] for Trinet H.264 capture nodes, lowest node first."""
    out = []
    for node in sorted(glob.glob("/dev/video*"),
                       key=lambda n: int(re.sub(r"\D", "", n) or 0)):
        ids = _usb_ids_for(node)
        if not ids:
            continue
        vid, pid, serial = ids
        if (vid, pid) == (USB_VID, USB_PID) and _does_h264_capture(node):
            out.append((node, serial))
    return out


# ---------------------------------------------------------------------------
#  Capture
# ---------------------------------------------------------------------------
def record(node: str, dst: Path, width: int, height: int, fps: int,
           duration: int) -> None:
    """Stream-copy H.264 from the camera into an MP4.

    -c copy is deliberate: the camera's H.264 carries the IMU SEI NALs, and any
    re-encode would strip them.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "v4l2", "-input_format", "h264",
           "-video_size", f"{width}x{height}", "-framerate", str(fps),
           "-i", node]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-c", "copy", "-y", str(dst)]

    how = f"{duration}s" if duration else "until Ctrl-C"
    print(f"[record] {node} -> {width}x{height}@{fps} H.264 ({how})")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[record] stopping ...")
        proc.send_signal(signal.SIGINT)   # let ffmpeg finalise the container
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def shutter_of(frames_bin: Path):
    """"global" / "rolling" from the recording's own frame timing, else None.

    Uses the reader's VtsData.is_global_shutter / is_rolling_shutter, the same
    rule the SDK writes. It comes from the stream, not the camera's generation,
    because an unprovisioned unit reports none. Older firmware declares neither,
    and the key is then omitted.
    """
    if not frames_bin.exists():
        return None
    try:
        vts = read_vts(str(frames_bin))
    except Exception:  # a malformed sidecar must not fail an otherwise good take
        return None
    if vts.is_global_shutter:
        return "global"
    if vts.is_rolling_shutter:
        return "rolling"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--outdir", default="captures",
                    help="parent directory for the recording folder (default: captures)")
    ap.add_argument("-d", "--duration", type=int, default=0,
                    help="seconds to record (0 = until Ctrl-C)")
    ap.add_argument("--device", help="video node, e.g. /dev/video0 (default: auto-detect)")
    ap.add_argument("--width", type=int, default=0,
                    help="frame width (default: widest H.264 size the camera advertises)")
    ap.add_argument("--height", type=int, default=0,
                    help="frame height (default: from the advertised H.264 size)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--list", action="store_true", help="list Trinet nodes and exit")
    args = ap.parse_args(argv)

    nodes = find_trinet_nodes()

    if args.list:
        if not nodes:
            print("no Trinet UVC capture nodes found")
            return 1
        for node, serial in nodes:
            print(f"{node}  serial={serial or '(none)'}")
        return 0

    if args.device:
        node = args.device
        ids = _usb_ids_for(node)
        serial = ids[2] if ids else ""
    elif nodes:
        node, serial = nodes[0]
    else:
        print(f"error: no Trinet camera found (looked for {USB_VID:04x}:{USB_PID:04x} "
              "with an H.264 capture node).\n"
              "       Check it is plugged in and in UVC mode; try --list.",
              file=sys.stderr)
        return 2

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = advertised_h264_size(node) or (1920, 1080)
    layout = layout_of(width, height)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dev8 = (serial or "")[:8]
    folder_id = f"{dev8}_recording_{ts}" if dev8 else f"recording_{ts}"
    out_dir = Path(args.outdir) / folder_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_mp4 = out_dir / "_capture.mp4"
    record(node, raw_mp4, width, height, args.fps, args.duration)

    if not raw_mp4.exists() or raw_mp4.stat().st_size == 0:
        print("error: capture produced no data.\n"
              "       If this was the first open after the camera booted, "
              "unplug/replug and retry.", file=sys.stderr)
        return 3
    print(f"[record] captured {raw_mp4.stat().st_size / 1e6:.1f} MB")

    # SEI -> sidecars. Writes video.mp4 / imu.bin / frames.bin into out_dir.
    extract(raw_mp4, out_dir)

    video = {"width": width, "height": height, "fps": args.fps, "codec": "h264"}
    if layout:
        video["layout"] = layout
    shutter = shutter_of(out_dir / "frames.bin")
    if shutter:
        video["shutter"] = shutter
    (out_dir / "meta.json").write_text(json.dumps({
        "id": folder_id,
        "created_at_epoch_ms": int(time.time() * 1000),
        "device": {"vendor_id": USB_VID, "product_id": USB_PID,
                   "serial": serial or None},
        "video": video,
        "source": "trinet-tools/record_uvc.py",
    }, indent=2))

    raw_mp4.unlink(missing_ok=True)
    for leftover in ("_stream.h264", "_video_clean.mp4"):
        (out_dir / leftover).unlink(missing_ok=True)

    print(f"\n[ok] {out_dir}")
    for name in ("video.mp4", "imu.bin", "frames.bin", "meta.json"):
        p = out_dir / name
        if p.exists():
            print(f"     {name:<12} {p.stat().st_size:>12,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())

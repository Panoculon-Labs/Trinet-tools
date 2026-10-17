#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Get a Trinet camera's own calibration, as calibration.json.

Three ways, depending on what you have:

    # 1. From any recording the camera made (card or app) — works anywhere
    python3 scripts/get_calibration.py --mp4 take0001_L.mp4

    # 2. Over USB, camera in USB webcam mode (no memory card) — Linux
    python3 scripts/get_calibration.py --uvc              # first Trinet camera found
    python3 scripts/get_calibration.py --uvc /dev/video2

    # 3. Over USB networking, camera in network (NCM / iPhone) mode
    python3 scripts/get_calibration.py --ncm              # first camera found
    python3 scripts/get_calibration.py --ncm 172.32.18.70

Writes calibration.json (or -o PATH) — accepted by every toolkit command that
takes --calibration — and, with --bin PATH, the raw blob as stored on the camera.
Exit status: 0 written, 1 no calibration stored, 2 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trinet_tools.device_calibration import (CalibrationError, from_mp4,  # noqa: E402
                                             read_ncm, read_uvc, to_calibration_json)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mp4", type=Path, help="a recording from the camera (either eye of a stereo take)")
    src.add_argument("--uvc", nargs="?", const="", metavar="DEVICE",
                     help="camera in USB webcam mode (Linux); optional /dev/videoN")
    src.add_argument("--ncm", nargs="?", const="", metavar="IP",
                     help="camera in network (NCM) mode; optional camera IP")
    ap.add_argument("-o", "--output", type=Path, default=Path("calibration.json"),
                    help="where to write the JSON (default: calibration.json)")
    ap.add_argument("--bin", type=Path, help="also save the raw calibration blob here")
    args = ap.parse_args(argv)

    try:
        if args.mp4:
            source, blob = str(args.mp4), from_mp4(args.mp4)
        elif args.uvc is not None:
            source, blob = f"USB {args.uvc or '(auto)'}", read_uvc(args.uvc or None)
        else:
            source, blob = f"network {args.ncm or '(auto)'}", read_ncm(args.ncm or None)
    except (CalibrationError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not blob:
        print(f"no calibration stored ({source}). A recording made before the camera was "
              "calibrated, or re-encoded, does not carry one.", file=sys.stderr)
        return 1

    try:
        calib = to_calibration_json(blob)
    except Exception as e:  # corrupt / unknown blob
        print(f"error: could not decode the calibration ({e})", file=sys.stderr)
        return 2

    args.output.write_text(json.dumps(calib, indent=2) + "\n")
    if args.bin:
        args.bin.write_bytes(blob)
    kind = "stereo" if len(calib.get("cameras", [])) >= 2 else "single-camera"
    print(f"{kind} calibration ({len(blob)} bytes) from {source} -> {args.output}"
          + (f" (+ {args.bin})" if args.bin else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

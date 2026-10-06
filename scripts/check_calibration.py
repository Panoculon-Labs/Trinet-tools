#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Field check: does a stereo camera's calibration still fit?

Measures the vertical offset between the two rectified eyes on an ordinary
recording — no calibration target needed — and prints a verdict:

    ok           |offset| <= 1.5 px   calibration fits
    check        1.5 .. 3 px          re-check with another take (textured scene)
    recalibrate  > 3 px               the stereo mount has moved since calibration
    inconclusive                      too few features (dark / featureless scene)

Uses the calibration embedded in the recording, or --calibration
(calibration.json or a TBLC .bin). Record a few seconds of a well-lit, textured
scene at 1-5 m (a desk, shelves, a room) — not a blank wall or the sky.

Usage:
    python3 scripts/check_calibration.py TAKE_PREFIX [--calibration CALIB] [--json]

TAKE_PREFIX names `<prefix>_L.mp4` + `<prefix>_R.mp4`.
Exit status: 0 ok, 1 check, 2 recalibrate, 3 inconclusive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Trinet MP4s carry an extra metadata track; when seeking, OpenCV's FFmpeg
# reader can exhaust its default packet-read budget and print a warning per
# seek. Raise the budget (read at capture time) so the output stays clean.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "100000")

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_depth_video import load_take, pair_frames             # noqa: E402
from trinet_tools.stereo_align import (CHECK_MAX_PX, OK_MAX_PX,    # noqa: E402
                                       Rectification, calibration_verdict,
                                       measure_y_offset)

MESSAGES = {
    "ok": f"OK — the calibration fits (|offset| <= {OK_MAX_PX} px).",
    "check": (f"CHECK — offset between {OK_MAX_PX} and {CHECK_MAX_PX} px. Re-run on another take of a "
              "well-lit, textured scene; if it stays in this range, plan a recalibration."),
    "recalibrate": (f"RECALIBRATE — offset above {CHECK_MAX_PX} px: the stereo mount has moved since "
                    "calibration. Depth from this camera will be degraded until it is recalibrated."),
    "inconclusive": ("INCONCLUSIVE — not enough features to measure. Record a few seconds of a "
                     "well-lit, textured scene and run the check again."),
}
EXIT = {"ok": 0, "check": 1, "recalibrate": 2, "inconclusive": 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("take", help="take prefix, e.g. card/Trinet/recording/take0002")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration.json or TBLC .bin (default: embedded in the recording)")
    ap.add_argument("--samples", type=int, default=15, help="frame pairs to sample (default 15)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="trinet_calcheck_") as td:
        mp4_l, mp4_r, vts_l, vts_r, _imu, calib = load_take(Path(args.take), Path(td), args.calibration)
        pairs = pair_frames(vts_l, vts_r)
        m = measure_y_offset(mp4_l, mp4_r, pairs, Rectification(calib), samples=args.samples)

    verdict = calibration_verdict(m["offset_px"])
    if args.json:
        print(json.dumps({"take": args.take, "verdict": verdict,
                          "calibration": str(args.calibration) if args.calibration else "embedded",
                          **m}, indent=2))
        return EXIT[verdict]

    src = str(args.calibration) if args.calibration else "embedded in the recording"
    print(f"take:        {args.take}")
    print(f"calibration: {src}")
    print(f"sampled:     {m['frames']} frame pairs, {m['points']} tracked points")
    if m["offset_px"] is not None:
        print(f"offset:      {m['offset_px']:+.2f} px vertical between the eyes "
              f"(spread across the take {m['spread_px']:.2f} px)")
    print(MESSAGES[verdict])
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())

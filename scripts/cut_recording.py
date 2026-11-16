#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""
Cut a Trinet recording (one camera or a whole wrist kit) to a time segment that
is itself a valid Trinet recording: the same file names, the video copied frame
for frame (no re-encoding), the sound kept and in step, and the .vts / .imu /
.tel sidecars and the embedded metadata track (TMF) cut to match.

    # 12.4 s to 40 s into the take (seconds from the first frame of any camera)
    python scripts/cut_recording.py take_folder/ --from 12.4 --to 40 -o segment/

    # the same window as absolute master-clock nanoseconds (as Studio passes it)
    python scripts/cut_recording.py take_folder/ --t0-ns 298318745942 --t1-ns 325918745942 -o segment/

    # one camera: pass its files (or its folder)
    python scripts/cut_recording.py Mono-Left/grp1_aaaa_1.* --from 60 --to 120 -o seg/

Every camera starts at its own last keyframe at or before the start (up to one
keyframe interval, about a second, early: cameras' keyframes are not aligned)
and ends at its last frame before the end. Frame times on the master clock are
unchanged, so the cut cameras stay in sync. Originals are never modified.
A fragmented recording (the camera did not finish writing it) must first be
repaired with scripts/repair_recordings.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trinet_tools.cut import CutError, collect_files, cut_recording, group_files  # noqa: E402
from trinet_tools.reader import read_vts  # noqa: E402


def take_start_ns(src) -> int:
    """Master-clock time of the earliest frame of any camera of the recording."""
    paths, _root = collect_files(src)
    starts = []
    for cam in group_files(paths):
        for v in cam.videos:
            g = read_vts(str(v.vts)).global_sof_ns()
            g = g[g > 0]
            if len(g):
                starts.append(int(g.min()))
    if not starts:
        raise CutError("no frame times found")
    return min(starts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("src", nargs="+", help="the recording's folder, or its files")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--from", dest="start_s", type=float, help="start, seconds into the take")
    ap.add_argument("--to", dest="end_s", type=float, help="end, seconds into the take")
    ap.add_argument("--t0-ns", type=int, help="start on the master clock (ns)")
    ap.add_argument("--t1-ns", type=int, help="end on the master clock (ns)")
    ap.add_argument("--hw-generation", help="camera generation to record in the cut's metadata (e.g. v4)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    src = args.src[0] if len(args.src) == 1 and Path(args.src[0]).is_dir() else args.src
    try:
        if args.t0_ns is not None and args.t1_ns is not None:
            t0, t1 = args.t0_ns, args.t1_ns
        elif args.start_s is not None and args.end_s is not None:
            base = take_start_ns(src)
            t0, t1 = base + round(args.start_s * 1e9), base + round(args.end_s * 1e9)
        else:
            ap.error("give --from/--to (seconds) or --t0-ns/--t1-ns")
        tic = time.time()
        report = cut_recording(src, t0, t1, args.out, hw_generation=args.hw_generation)
        took = time.time() - tic
    except CutError as e:
        print(f"cannot cut: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for cam in report["cameras"]:
        for v in cam["videos"]:
            if not v.get("frames"):
                print(f"{cam['folder']}/{v['file']}: no frames in the window (left out)")
                continue
            print(f"{cam['folder']}/{v['file']}: frames {v['first_frame']}-{v['last_frame']} "
                  f"({v['frames']}), starts {v['starts_before_s']:.3f} s before the asked start, "
                  f"{v['bytes'] / 1e6:.1f} MB")
    print(f"done in {took:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

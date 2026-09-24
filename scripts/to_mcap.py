#!/usr/bin/env python3
"""Convert a Trinet recording to MCAP (Foxglove / ROS 2).

    python scripts/to_mcap.py /data/take0004            # stereo: take0004_L/_R.mp4 ...
    python scripts/to_mcap.py /data/take0004_L.mp4 -o out/take0004.mcap
    python scripts/to_mcap.py /data/recording4_1.mp4    # single camera

Works for stereo and single-camera recordings, global and rolling shutter,
H.264 and H.265. Video is copied frame by frame, not re-encoded. See
trinet_tools/mcap_export.py for the topic list and docs/mcap_export.md for
details.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trinet_tools.mcap_export import discover, export, shutter_info  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="any file of the recording (an .mp4 or sidecar) or dir/basename")
    ap.add_argument("-o", "--output", help="output .mcap (default: <basename>.mcap next to the input)")
    ap.add_argument("--calibration", help="calibration JSON to use instead of the one embedded in the MP4")
    ap.add_argument("--local-clock", action="store_true",
                    help="keep this camera's own clock even if the take was synced to a master camera")
    ap.add_argument("--apply-timeshift", action="store_true",
                    help="shift camera times by the calibrated camera-IMU timeshift (t_imu = t_cam + ts)")
    ap.add_argument("--compression", choices=["zstd", "lz4", "none"], default="zstd")
    args = ap.parse_args(argv)

    rec = discover(args.recording, args.calibration)
    src = Path(args.recording)
    out = Path(args.output) if args.output else src.parent / f"{rec.base}.mcap"
    print(f"recording  {rec.base}: {len(rec.cameras)} camera(s)")
    for c in rec.cameras:
        sh, ro, centred = shutter_info(c.vts)
        extra = f", readout {ro} us{', middle-row timestamps' if centred else ''}" if sh == "rolling" else ""
        print(f"  {c.name}  {c.mp4.name}  {c.codec} {c.width}x{c.height}  {c.vts.num_frames} frames  {sh} shutter{extra}")
    if rec.imu is not None:
        print(f"  imu   {rec.imu.num_samples} samples @ {rec.imu.actual_rate_hz:.0f} Hz")
    print(f"  calibration: {'yes' if rec.calibration else 'no'}")

    t0 = time.time()
    def progress(i, n):
        print(f"\r  writing  {i}/{n} frames", end="", flush=True)
    st = export(rec, str(out), use_global_clock=not args.local_clock,
                apply_timeshift=args.apply_timeshift, compression=args.compression,
                progress=progress)
    print(f"\r  wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f} s")
    print("  frames " + ", ".join(f"{k}={v}" for k, v in st.frames.items())
          + f"; imu={st.imu_samples}; mag={st.mag_samples}")
    for n in st.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

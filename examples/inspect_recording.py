#!/usr/bin/env python3
"""
Inspect a Trinet recording from the shell.

Prints a one-screen summary of what's in a recording: number of samples,
actual sample rate, accel/gyro full-scale, frame-sync state, video frame
count, and the device_id (or "(pre-v4 recording)" if the recording was
made before device-id support).

Usage:
    python examples/inspect_recording.py path/to/recording.imu
    python examples/inspect_recording.py path/to/recording.mp4    # also reads .imu/.vts
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.reader import read_imu, read_vts


def resolve_triple(arg: str):
    """Given any one of recording.mp4 / recording.imu / recording.vts /
    recording (no extension), return (mp4_path, imu_path, vts_path)."""
    p = Path(arg)
    base = p.with_suffix("") if p.suffix in {".mp4", ".imu", ".vts"} else p
    return (
        base.with_suffix(".mp4"),
        base.with_suffix(".imu"),
        base.with_suffix(".vts"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("recording", help="Path to recording (.mp4 / .imu / .vts / base name)")
    args = ap.parse_args()

    mp4, imu_path, vts_path = resolve_triple(args.recording)

    print(f"Recording: {mp4.parent / mp4.stem}")
    print()

    if imu_path.exists():
        imu = read_imu(str(imu_path))
        h = imu.header
        print(f"  .imu  : {imu_path.name}")
        print(f"          version       = {h.version}")
        print(f"          device_id     = {h.device_id_hex or '(pre-v4 recording)'}")
        print(f"          samples       = {imu.num_samples}")
        print(f"          duration      = {imu.duration_s:.3f} s")
        print(f"          actual rate   = {imu.actual_rate_hz:.1f} Hz "
              f"(nominal {h.sample_rate_hz} Hz)")
        print(f"          accel range   = {h.accel_fs_name}")
        print(f"          gyro range    = {h.gyro_fs_name}")
        print(f"          fsync         = {'on' if h.fsync_enabled else 'off'}")
        if imu.num_samples > 0:
            import numpy as np
            mag = float(np.mean(np.linalg.norm(imu.accel, axis=1)))
            print(f"          mean |accel|  = {mag:.3f} m/s² "
                  f"(stationary unit reads ~9.8)")
        print()
    else:
        print(f"  .imu  : (not found at {imu_path})")
        print()

    if vts_path.exists():
        vts = read_vts(str(vts_path))
        print(f"  .vts  : {vts_path.name}")
        print(f"          version       = {vts.header.version}")
        print(f"          frame rate    = {vts.header.fps:.2f} fps (configured)")
        print(f"          frames        = {len(vts.frame_numbers)}")
        print()
    else:
        print(f"  .vts  : (not found at {vts_path})")
        print()

    if mp4.exists():
        size_mb = mp4.stat().st_size / (1024 * 1024)
        print(f"  .mp4  : {mp4.name}  ({size_mb:.1f} MB)")
    else:
        print(f"  .mp4  : (not found at {mp4})")


if __name__ == "__main__":
    main()

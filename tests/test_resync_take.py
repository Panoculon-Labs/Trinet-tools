# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""resync_take.py on a synthetic long take: one master + two slaves whose clocks
drift (constant rate plus a slow thermal wander) far past half a frame, with
frame-phase jitter, mixed exposures and dropped frames on every camera."""
import csv
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from trinet_tools.reader import read_imu, read_vts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resync_take.py"

PERIOD = 33_726_000           # ns, ~29.65 fps
HOURS = 2.0
VTS_DT = np.dtype([("fn", "<u4"), ("sof", "<u8"), ("seq", "<u4"), ("pts", "<u8"),
                   ("exp", "<u4"), ("fl", "<u4"), ("ro", "<u4")])


def _write_take(d, rng):
    n = int(HOURS * 3600e9 / PERIOD)
    k = np.arange(n)
    start = 5_000_000_000 + k * PERIOD                  # frame starts on the master clock
    truth = {}

    def write(name, role, local_start, exp, hdr_off, flags, drop):
        keep = np.ones(n, bool)
        keep[drop] = False
        m = int(keep.sum())
        e = np.zeros(m, VTS_DT)
        e["fn"] = np.arange(m)
        e["sof"] = local_start[keep] - exp[keep] * 500  # stamped at mid-exposure
        e["seq"] = np.arange(m)
        e["pts"] = e["sof"] // 1000
        e["exp"] = exp[keep]
        e["fl"] = 0x07
        e["ro"] = 26000
        (d / f"{name}.vts").write_bytes(
            struct.pack("<8sIIqiHH", b"TRIVTS01", 4, 30000, hdr_off, 0, 150, flags) + e.tobytes())
        # 400 Hz IMU on the same local clock; accel x carries a sample counter
        t = np.arange(int(e["sof"][0]) - 10**8, int(e["sof"][-1]) + 10**8, 2_500_000, dtype=np.int64)
        rec = np.zeros(len(t), np.dtype([("t", "<u8"), ("f", "<f4", (18,))]))
        rec["t"] = t
        rec["f"][:, 0] = np.arange(len(t)) % 1000
        hdr = struct.pack("<8sIIHHQQI", b"TRIMU001", 5, 400, 2, 2, int(t[0]), int(e["sof"][0]), 2)
        (d / f"{name}.imu").write_bytes(hdr + bytes(64 - len(hdr)) + rec.tobytes())
        (d / f"{name}.json").write_text(json.dumps(
            {"session": 7, "segment": 1, "role": role, "device_tag": name,
             "master_clock_offset_ns": hdr_off, "clock_skew_ppb": 27000}))
        return keep

    write("head", "master", start, np.full(n, 10000), 0, 3, rng.choice(n, 40, replace=False))
    t_h = k * PERIOD / 3.6e12
    for name, ppm, off0 in (("left", 9.0, 150_000_000), ("right", -6.0, -80_000_000)):
        off = off0 + ppm * 1e-6 * (k * PERIOD) + 3e6 * np.sin(2 * np.pi * t_h / 1.5)
        local = (start - off + rng.normal(0, 0.8e6, n)).astype(np.int64)
        exp = np.where(rng.random(n) < 0.15, 20000, 10000)
        keep = write(name, "slave", local, exp, off0, 1, rng.choice(n, 60, replace=False))
        truth[name] = off[keep]
    return truth


def test_long_take_recovers_master_clock(tmp_path):
    rng = np.random.default_rng(3)
    src, out = tmp_path / "take", tmp_path / "out"
    src.mkdir()
    truth = _write_take(src, rng)
    before = {p.name: p.read_bytes() for p in src.iterdir()}

    p = subprocess.run([sys.executable, str(SCRIPT), "--auto", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # originals untouched
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before

    for name in ("left", "right"):
        a, b = read_vts(str(src / f"{name}.vts")), read_vts(str(out / f"{name}.vts"))
        shift = b.best_timestamps_ns.astype(np.int64) - a.best_timestamps_ns.astype(np.int64)
        err_ms = (shift - truth[name]) / 1e6
        assert np.percentile(np.abs(err_ms), 99) < 0.2, f"{name}: offset not recovered"
        # the header was hundreds of ms stale by the end; the output needs no offset at all
        assert b.header.master_clock_offset_ns == 0 and b.header.clock_skew_ppb == 0
        assert np.array_equal(b.global_sof_ns(), b.best_timestamps_ns.astype(np.int64))
        assert np.array_equal(a.exposure_us, b.exposure_us)

        # IMU moved by exactly the same amount as the frames around it
        ia, ib = read_imu(str(src / f"{name}.imu")), read_imu(str(out / f"{name}.imu"))
        ishift = ib.timestamps_ns.astype(np.int64) - ia.timestamps_ns.astype(np.int64)
        at_frames = np.interp(a.best_timestamps_ns.astype(float), ia.timestamps_ns.astype(float), ishift)
        assert np.abs(at_frames - shift).max() < 1000
        assert np.array_equal(ia.accel, ib.accel)

        meta = json.loads((out / f"{name}.json").read_text())
        assert meta["master_clock_offset_ns"] == 0 and meta["clock_skew_ppb"] == 0

    report = json.loads((out / "7_1_resync.json").read_text())
    assert abs(report["cameras"]["left"]["clock_rate_ppm"] - 9.0) < 0.5
    assert report["cameras"]["left"]["old_timeline_error_max_ms"] > 50

    with open(out / "7_1_frames.csv") as f:
        rows = list(csv.DictReader(f))
    d = np.array([float(r["left_minus_master_ms"]) for r in rows if r["left_minus_master_ms"]])
    assert len(d) > 0.99 * len(rows)
    assert abs(np.median(d)) < 0.2


def test_scan_flags_damaged_packets(tmp_path):
    spec = importlib.util.spec_from_file_location("resync_take", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = (5).to_bytes(4, "big") + b"\x41abcd" + (3).to_bytes(4, "big") + b"\x01xy"
    blob = good + b"\xb1\x25\x33\x6d" + good[4:] + good      # middle packet: corrupted length prefix
    f = tmp_path / "v.bin"
    f.write_bytes(blob)
    orig = mod.subprocess.run

    class _Out:
        stdout = "\n".join(f"size={len(good)}|pos={i * len(good)}" for i in range(3))

    mod.subprocess.run = lambda *a, **k: _Out()
    try:
        n, damaged = mod.scan_mp4(f)
    finally:
        mod.subprocess.run = orig
    assert n == 3 and damaged == [1]

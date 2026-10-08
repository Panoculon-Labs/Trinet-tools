#!/usr/bin/env python3
"""
Re-synchronise the timestamps of a multi-camera take onto the master's clock.

Why this exists
---------------
Each camera stamps its frames and IMU samples with its own local clock. A
synced slave's .vts header (format v3/v4) stores ONE local->master clock offset,
measured when the file was opened, and readers extend it across the whole file:

    global = sof + header.offset (+ header.skew * dt)

Two independent crystals run a few parts-per-million apart, so over a long take
that single offset goes stale: ~2-4 ppm is 5-15 ms per hour, and it keeps
growing. The frames themselves do NOT drift -- while the cameras are radio-
synced, each slave holds its frame start on the master's frame grid -- only the
recorded timeline does.

Because the frames stay locked, the true clock relationship can be measured
from the recording itself: pair every slave frame with the master frame it was
locked to, and the time difference between the two is the live clock offset at
that moment. This script fits that offset across the take (smoothly, so it also
follows slow rate changes as the units warm up) and rewrites every slave
timestamp -- frames AND IMU samples -- onto the master's clock. After that, a
timestamp means the same instant on every camera, for the whole recording.

What it writes (into --out, original files are never modified)
--------------------------------------------------------------
  <rec>.vts    frame times on the master clock (header offset = 0, skew = 0),
               same binary format as the input, so every existing tool reads it
  <rec>.imu    IMU sample times on the master clock, same binary format
  <rec>.mp4    symlink to the original video, so sync_view.py etc. can be
               pointed at the output folder directly
  <rec>.json   the recording's .json sidecar with offset/skew zeroed to match
  <session>_frames.csv   one row per master frame: the matching frame index of
               every camera and its corrected time
  <session>_resync.json  per-camera report: clock rate, fit residual, frame
               pairing statistics, and how far the old header timeline had drifted
  <session>_resync.png   (with --plot) before/after drift plot

Frame times keep their original meaning (mid-exposure when the file says so):
only the clock they are expressed in changes. Within one camera, IMU and frames
are shifted by the same amount at every instant, so per-camera IMU<->frame
alignment is untouched.

Usage
-----
    # explicit recordings (a .mp4/.vts path or a base name); master auto-detected
    python scripts/resync_take.py head/grp1_aaaa_1 left/grp1_bbbb_1 right/grp1_cccc_1 -o resynced/

    # every take in a folder tree, grouped by session id from the .json sidecars
    python scripts/resync_take.py --auto /path/to/recordings -o resynced/ --plot

Recordings that already carry a per-frame clock offset (.vts v5) do not need
this; the script still accepts them and uses those offsets directly.
"""

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.reader import (  # noqa: E402
    IMU_HEADER_SIZE, IMU_SAMPLE_SIZE_V1, IMU_SAMPLE_SIZE_V2, IMU_SAMPLE_SIZE_V3,
    TIMING_EXPOSURE_VALID, TIMING_MID_EXPOSURE, VTS_ENTRY_SIZE_V1, VTS_ENTRY_SIZE_V2,
    VTS_ENTRY_SIZE_V4, VTS_ENTRY_SIZE_V5, VTS_HEADER_SIZE, VTS_SYNC_FLAG_IS_MASTER,
    VTS_SYNC_FLAG_SYNCED, read_vts,
)

BIN_S = 60.0          # offset-model knot spacing (s); drift per bin is ~µs
TRACK_S = 20.0        # pairing window (s) while tracking the offset through the take
OUTLIER_MS = 3.0      # pairs further than this from the local median are not used in the fit


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------
class Recording:
    def __init__(self, arg):
        p = Path(arg)
        base = p.with_suffix("") if p.suffix in (".mp4", ".vts", ".imu", ".json") else p
        self.base = base
        self.name = base.name
        self.vts_path = base.with_suffix(".vts")
        if not self.vts_path.exists():
            raise SystemExit(f"{self.vts_path} not found")
        self.imu_path = base.with_suffix(".imu") if base.with_suffix(".imu").exists() else None
        self.mp4_path = base.with_suffix(".mp4") if base.with_suffix(".mp4").exists() else None
        self.json_path = base.with_suffix(".json") if base.with_suffix(".json").exists() else None
        self.meta = {}
        if self.json_path:
            try:
                self.meta = json.loads(self.json_path.read_text())
            except ValueError:
                pass
        self.vts = read_vts(str(self.vts_path))
        self.label = self.meta.get("device_tag") or self.name
        self.is_master = bool(self.vts.header.sync_flags & VTS_SYNC_FLAG_IS_MASTER) \
            or self.meta.get("role") == "master"
        self.stamp_ns = self.vts.best_timestamps_ns.astype(np.int64)

    def frame_start_ns(self):
        """Start-of-frame time: undo the on-device mid-exposure shift. The frame
        lock holds frame STARTS together, so this is the quantity that is equal
        across cameras even when their exposures differ."""
        t = self.stamp_ns.copy()
        v = self.vts
        if v.timing_flags is not None and v.exposure_us is not None and len(t):
            fl = v.timing_flags.astype(np.uint32)
            m = ((fl & TIMING_MID_EXPOSURE) != 0) & ((fl & TIMING_EXPOSURE_VALID) != 0)
            t[m] += v.exposure_us[m].astype(np.int64) * 500  # + exposure/2, µs -> ns
        return t


# ---------------------------------------------------------------------------
#  Offset model: local slave clock -> master clock
# ---------------------------------------------------------------------------
def track_pairs(slave_t, master_t, start_offset_ns, period_ns):
    """Pair each slave frame with the master frame it is locked to.

    Walks through the take in TRACK_S windows, predicting the offset from the
    previous window, so the pairing follows the drift however long the take is
    (a fixed offset would pair with the wrong frame once the drift passes half a
    frame). Returns (slave_idx, master_idx, offset_ns) for the accepted pairs.
    """
    win = int(TRACK_S * 1e9)
    t0 = slave_t[0]
    off, rate = float(start_offset_ns), 0.0     # rate in ns of offset per ns of time
    prev = None
    s_all, m_all, d_all = [], [], []
    w = 0
    while True:
        lo, hi = t0 + w * win, t0 + (w + 1) * win
        i = np.nonzero((slave_t >= lo) & (slave_t < hi))[0]
        if lo > slave_t[-1]:
            break
        w += 1
        if i.size == 0:
            continue
        tc = slave_t[i].astype(np.float64)
        pred = off + rate * (tc - tc[0]) if prev is None else prev[1] + rate * (tc - prev[0])
        g = tc + pred
        j = np.clip(np.searchsorted(master_t, g), 1, len(master_t) - 1)
        j -= (g - master_t[j - 1]) < (master_t[j] - g)
        d = master_t[j].astype(np.float64) - tc
        r = d - pred
        ok = np.abs(r) < period_ns / 2
        if ok.sum() < 5:
            continue
        med_r = np.median(r[ok])
        keep = ok & (np.abs(r - med_r) < OUTLIER_MS * 1e6)
        s_all.append(i[keep]); m_all.append(j[keep]); d_all.append(d[keep])
        mid_t = float(np.median(tc[keep])) if keep.any() else float(np.median(tc))
        mid_d = float(np.median(d[keep])) if keep.any() else float(np.median(d))
        if prev is not None and mid_t > prev[0]:
            rate = 0.7 * rate + 0.3 * (mid_d - prev[1]) / (mid_t - prev[0])
        prev = (mid_t, mid_d)
    if not s_all:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(s_all), np.concatenate(m_all), np.concatenate(d_all)


class OffsetModel:
    """Piecewise-linear offset(t_local) through BIN_S medians, with the global
    straight-line fit used to extrapolate past the ends."""

    def __init__(self, t_local, offset_ns):
        x = t_local.astype(np.float64)
        y = offset_ns.astype(np.float64)
        self.x0 = x[0]
        xs = (x - self.x0) / 1e9
        self.lin = np.polyfit(xs, y, 1)
        b = np.floor(xs / BIN_S).astype(int)
        kx, ky = [], []
        for k in np.unique(b):
            m = b == k
            if m.sum() >= 30:
                kx.append(np.median(xs[m])); ky.append(np.median(y[m]))
        self.kx, self.ky = np.array(kx), np.array(ky)
        resid = y - self(t_local)
        self.resid_std_us = float(np.std(resid) / 1e3)
        self.resid_p99_us = float(np.percentile(np.abs(resid), 99) / 1e3)
        self.rate_ppm = float(self.lin[0] / 1e3)   # ns per s -> ppm

    def __call__(self, t_local):
        xs = (np.asarray(t_local, dtype=np.float64) - self.x0) / 1e9
        if len(self.kx) < 2:
            return np.polyval(self.lin, xs)
        y = np.interp(xs, self.kx, self.ky)
        lo, hi = xs < self.kx[0], xs > self.kx[-1]
        slope = self.lin[0]
        y[lo] = self.ky[0] + slope * (xs[lo] - self.kx[0])
        y[hi] = self.ky[-1] + slope * (xs[hi] - self.kx[-1])
        return y

    def to_master(self, t_local):
        t = np.asarray(t_local, dtype=np.int64)
        return t + np.round(self(t)).astype(np.int64)


# ---------------------------------------------------------------------------
#  Writers (byte-level patch: the output keeps the exact input layout)
# ---------------------------------------------------------------------------
def write_vts(rec, out_path, new_stamp_ns, is_master, quality_us):
    raw = bytearray(rec.vts_path.read_bytes())
    ver = struct.unpack_from("<I", raw, 8)[0]
    size = {1: VTS_ENTRY_SIZE_V1, 2: VTS_ENTRY_SIZE_V2, 3: VTS_ENTRY_SIZE_V2,
            4: VTS_ENTRY_SIZE_V4}.get(ver, VTS_ENTRY_SIZE_V5)
    n = (len(raw) - VTS_HEADER_SIZE) // size
    assert n == len(new_stamp_ns), "frame count mismatch while writing .vts"
    body = np.frombuffer(raw, dtype=np.uint8, count=n * size, offset=VTS_HEADER_SIZE).reshape(n, size).copy()
    body[:, 4:12] = new_stamp_ns.astype("<u8").view(np.uint8).reshape(n, 8)   # sof_timestamp_ns
    if size == VTS_ENTRY_SIZE_V5:
        body[:, 36:44] = 0                                                  # per-frame offset now folded in
    raw[VTS_HEADER_SIZE:VTS_HEADER_SIZE + n * size] = body.tobytes()
    flags = VTS_SYNC_FLAG_SYNCED | (VTS_SYNC_FLAG_IS_MASTER if is_master else 0)
    struct.pack_into("<qiHH", raw, 16, 0, 0, min(int(round(quality_us)), 65535), flags)
    Path(out_path).write_bytes(bytes(raw))


def write_imu(rec, out_path, model):
    raw = bytearray(rec.imu_path.read_bytes())
    ver = struct.unpack_from("<I", raw, 8)[0]
    size = IMU_SAMPLE_SIZE_V3 if ver >= 3 else IMU_SAMPLE_SIZE_V2 if ver == 2 else IMU_SAMPLE_SIZE_V1
    n = (len(raw) - IMU_HEADER_SIZE) // size
    body = np.frombuffer(raw, dtype=np.uint8, count=n * size, offset=IMU_HEADER_SIZE).reshape(n, size).copy()
    ts = body[:, 0:8].copy().view("<u8").reshape(n).astype(np.int64)
    body[:, 0:8] = model.to_master(ts).astype("<u8").view(np.uint8).reshape(n, 8)
    raw[IMU_HEADER_SIZE:IMU_HEADER_SIZE + n * size] = body.tobytes()
    for off in (20, 28):                      # start_time_ns, video_start_ns
        v = struct.unpack_from("<Q", raw, off)[0]
        if v:
            struct.pack_into("<Q", raw, off, int(model.to_master(np.array([v]))[0]))
    Path(out_path).write_bytes(bytes(raw))
    return ts, n


def link_or_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def scan_mp4(path):
    """Walk every video packet's length-prefixed NAL structure straight from the
    container index (no decoding, ~1 s per hour of video). Returns
    (frame_count, damaged_frame_indices), or (None, None) without ffprobe.

    A damaged packet is one whose bytes do not parse as NAL units -- typically a
    block of the file that was corrupted on the storage card or in a copy.
    Decoders (including OpenCV) drop or stall on these, which silently shifts
    every later frame against its timestamp in tools that count decoded frames.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "packet=pos,size", "-of", "compact=p=0", str(path)],
            capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    pk = []
    for line in out.stdout.splitlines():   # "size=N|pos=M" -- ffprobe's own field order
        kv = dict(x.split("=", 1) for x in line.split("|") if "=" in x)
        if kv.get("pos", "N/A").isdigit() and kv.get("size", "").isdigit():
            pk.append((int(kv["pos"]), int(kv["size"])))
    damaged = []
    with open(path, "rb") as f:
        for i, (pos, size) in enumerate(pk):
            f.seek(pos)
            buf = f.read(size)
            o = 0
            while o + 4 < len(buf):
                n = int.from_bytes(buf[o:o + 4], "big")
                if n == 0 or o + 4 + n > len(buf) or buf[o + 4] & 0x80:
                    break
                o += 4 + n
            if o != len(buf):
                damaged.append(i)
    return len(pk), damaged


# ---------------------------------------------------------------------------
def resync(recs, out_dir, plot=False, check_video=False, session=None):
    masters = [r for r in recs if r.is_master]
    if len(masters) != 1:
        raise SystemExit(f"need exactly one master in the take, found {len(masters)} "
                         f"({', '.join(r.label for r in masters) or 'none'}); use --master")
    master = masters[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or master.meta.get("session") or master.name
    m_start = master.frame_start_ns()
    period = float(np.median(np.diff(m_start)))
    report = {"session": session, "master": master.label, "cameras": {}}

    corrected = {master.label: master.stamp_ns}
    write_vts(master, out_dir / master.vts_path.name, master.stamp_ns, True, 0)
    if master.imu_path:
        shutil.copy2(master.imu_path, out_dir / master.imu_path.name)
    report["cameras"][master.label] = {"role": "master", "frames": len(master.stamp_ns)}
    plots = []

    for rec in recs:
        if rec is master:
            continue
        v = rec.vts
        s_start = rec.frame_start_ns()
        info = {"role": "slave", "frames": len(s_start)}
        if not v.header.synced:
            print(f"  {rec.label}: header says NOT synced -- pairing from raw clocks is unreliable; skipped")
            info["skipped"] = "not synced"
            report["cameras"][rec.label] = info
            continue
        if v.frame_offset_ns is not None and len(v.frame_offset_ns) and np.any(v.frame_offset_ns):
            si = np.arange(len(s_start))
            mi = None
            offs = v.frame_offset_ns.astype(np.float64)
            info["source"] = "per-frame offsets (v5)"
        else:
            si, mi, offs = track_pairs(s_start, m_start, v.header.master_clock_offset_ns, period)
            if len(si) < 100:
                raise SystemExit(f"{rec.label}: could not pair frames with the master")
            info["source"] = "frame pairing"
            info["paired_frames"] = int(len(si))
            info["paired_fraction"] = round(len(si) / len(s_start), 4)
            info["frame_index_shift"] = {str(int(k)): int(c) for k, c in
                                         zip(*np.unique(mi - si, return_counts=True))}
        model = OffsetModel(s_start[si], offs)
        new_stamp = model.to_master(rec.stamp_ns)
        corrected[rec.label] = new_stamp

        # How far the old single-offset timeline had drifted from the fitted one.
        hdr_g = v.global_sof_ns().astype(np.int64)
        drift_ms = (hdr_g - new_stamp) / 1e6
        info.update({
            "header_offset_ms": v.header.master_clock_offset_ns / 1e6,
            "fitted_offset_start_ms": float(model(s_start[:1])[0] / 1e6),
            "fitted_offset_end_ms": float(model(s_start[-1:])[0] / 1e6),
            "clock_rate_ppm": round(model.rate_ppm, 3),
            "fit_residual_std_ms": round(model.resid_std_us / 1e3, 3),
            "fit_residual_p99_ms": round(model.resid_p99_us / 1e3, 3),
            "old_timeline_error_end_ms": round(float(drift_ms[-1]), 3),
            "old_timeline_error_max_ms": round(float(np.abs(drift_ms).max()), 3),
        })
        write_vts(rec, out_dir / rec.vts_path.name, new_stamp, False, model.resid_std_us)
        if rec.imu_path:
            _, n = write_imu(rec, out_dir / rec.imu_path.name, model)
            info["imu_samples"] = int(n)
        report["cameras"][rec.label] = info
        if mi is not None:
            v_hdr = v.header.master_clock_offset_ns
            plots.append((rec.label, s_start[si], s_start[si] + v_hdr, model.to_master(s_start[si]), m_start[mi]))
        print(f"  {rec.label}: rate {model.rate_ppm:+.2f} ppm, fit residual {model.resid_std_us/1e3:.2f} ms, "
              f"old timeline off by {drift_ms[-1]:+.1f} ms at the end -> corrected")

    for rec in recs:
        if rec.mp4_path:
            link_or_copy(rec.mp4_path, out_dir / rec.mp4_path.name)
        if rec.json_path:
            # The .json mirrors the .vts sync header; it must describe the
            # corrected files, or a reader applying its offset/skew would undo
            # the re-sync.
            meta = dict(rec.meta)
            meta.update({"master_clock_offset_ns": 0, "clock_skew_ppb": 0,
                         "resynced_to_master": master.label})
            (out_dir / rec.json_path.name).write_text(json.dumps(meta, indent=2) + "\n")
        if check_video and rec.mp4_path:
            n, damaged = scan_mp4(rec.mp4_path)
            c = report["cameras"].setdefault(rec.label, {})
            c["mp4_frames"] = n
            if n is None:
                print(f"  ({rec.label}: ffprobe not found, video not checked)")
                continue
            nv = len(rec.stamp_ns)
            # The recorder closes the .vts a moment before the encoder drains, so
            # a few trailing video frames without timestamps are normal; frame i
            # of the video is still .vts entry i. Fewer video frames is not.
            if n < nv:
                c["warning_count"] = (f"video has {n} frames, fewer than the {nv} in the .vts")
                print(f"  WARNING {rec.label}: {c['warning_count']}")
            inner = [i for i in damaged if i < n - 1]      # last packet is often cut at stop
            if inner:
                c["damaged_frames"] = inner
                c["warning_damage"] = (
                    f"{len(inner)} damaged video frames between {inner[0] / (1e9 / period) / 60:.1f} and "
                    f"{inner[-1] / (1e9 / period) / 60:.1f} min -- decoders drop or stall on these, so tools that "
                    f"count decoded frames (sync_view.py) fall out of step after the first one. "
                    f"Re-copy this file from the original card if possible.")
                print(f"  WARNING {rec.label}: {c['warning_damage']}")

    # Frame matching table on the master clock.
    csv_path = out_dir / f"{session}_frames.csv"
    order = [master.label] + [r.label for r in recs if r is not master and r.label in corrected]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["master_frame", "master_time_ns"] +
                   [x for lab in order[1:] for x in (f"{lab}_frame", f"{lab}_time_ns", f"{lab}_minus_master_ms")])
        mt = corrected[master.label]
        cols = []
        for lab in order[1:]:
            t = corrected[lab]
            j = np.clip(np.searchsorted(t, mt), 1, len(t) - 1)
            j -= (mt - t[j - 1]) < (t[j] - mt)
            d = (t[j] - mt) / 1e6
            valid = np.abs(d) < period / 2e6
            cols.append((j, t[j], d, valid))
        for i in range(len(mt)):
            row = [i, int(mt[i])]
            for j, t, d, valid in cols:
                row += [int(j[i]), int(t[i]), round(float(d[i]), 3)] if valid[i] else ["", "", ""]
            w.writerow(row)

    (out_dir / f"{session}_resync.json").write_text(json.dumps(report, indent=2))
    if plot and plots:
        _plot(plots, master, out_dir / f"{session}_resync.png")
    print(f"  wrote {out_dir}/ ({csv_path.name}, {session}_resync.json)")
    return report


def _plot(plots, master, path):
    """Before/after: per-frame (slave - master) frame-start difference on the
    master clock, using the header's single offset vs the fitted clock model.
    Frame starts are compared because that is what the cameras hold together;
    10 s medians show the trend, the faint band is every frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(plots), 1, figsize=(11, 3.2 * len(plots) + 0.6), sharex=True, squeeze=False)
    t_ref = min(int(p[4][0]) for p in plots)
    for row, (label, _, before, after, m) in zip(ax[:, 0], plots):
        t_min = (m - t_ref) / 6e10
        for y, name, col in (((before - m) / 1e6, "before (single offset from the file header)", "tab:red"),
                             ((after - m) / 1e6, "after re-sync", "tab:green")):
            row.plot(t_min, y, ",", color=col, alpha=0.15)
            b = np.floor(t_min * 6).astype(int)               # 10 s bins
            ks = np.unique(b)
            med = [np.median(y[b == k]) for k in ks]
            row.plot((ks + 0.5) / 6, med, color=col, lw=1.8, label=name)
        row.axhline(0, color="k", lw=0.6)
        row.set_ylabel("camera - master [ms]")
        row.set_title(f"{label} vs master {master.label}", fontsize=10)
        row.grid(alpha=0.3)
        row.legend(fontsize=8, loc="lower left")
    ax[-1, 0].set_xlabel("elapsed [min]")
    fig.suptitle("Frame timing on the master clock, before and after re-sync")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def auto_groups(folder):
    groups = {}
    for js in Path(folder).rglob("*.json"):
        try:
            meta = json.loads(js.read_text())
        except (OSError, ValueError):
            continue
        if "session" in meta and js.with_suffix(".vts").exists():
            key = (meta["session"], meta.get("segment", 1))
            groups.setdefault(key, []).append(str(js.with_suffix("")))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("recordings", nargs="*", help=".mp4/.vts path or base name of each camera in ONE take")
    ap.add_argument("--auto", metavar="DIR", help="find and re-sync every take under DIR (by .json session id)")
    ap.add_argument("-o", "--out", required=True, help="output folder (originals are never modified)")
    ap.add_argument("--master", help="device tag or file name of the master, if it cannot be auto-detected")
    ap.add_argument("--plot", action="store_true", help="also write a before/after drift plot")
    ap.add_argument("--check-video", action="store_true",
                    help="check each .mp4 for damaged frames and a frame count that fits its .vts (needs ffprobe)")
    args = ap.parse_args()

    takes = []
    if args.auto:
        for (sess, seg), members in sorted(auto_groups(args.auto).items()):
            if len(members) > 1:
                takes.append((f"{sess}_{seg}", members))
    if args.recordings:
        takes.append((None, args.recordings))
    if not takes:
        ap.error("no recordings given (pass files, or --auto DIR)")

    for name, members in takes:
        recs = [Recording(m) for m in members]
        if args.master:
            for r in recs:
                r.is_master = args.master in (r.label, r.name)
        print(f"take {name or recs[0].name}: {', '.join(r.label for r in recs)}")
        out = Path(args.out) / name if (name and len(takes) > 1) else Path(args.out)
        resync(recs, out, plot=args.plot, check_video=args.check_video, session=name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""
Quality check for Trinet recordings: frame drops, frame rate, duration,
bitrate, motion data and — for Wrist Kit takes — camera-to-camera sync.

Works on a single recording, a stereo take, or a whole kit:

    # one recording (a .mp4, .vts or base name)
    python scripts/check_recording.py path/to/recording1_1.mp4
    python scripts/check_recording.py path/to/take0001_L.mp4      # both eyes

    # a kit: one folder per camera; every take found in them is matched by its
    # grp number and checked together
    python scripts/check_recording.py my-session/head my-session/left my-session/right

    # only one take, and a machine-readable report
    python scripts/check_recording.py my-session/head my-session/left my-session/right \
        --take grp143024 --json report.json

What it measures, per video stream (each eye of a stereo camera is a stream):

  frames      frames in the .vts, and in the .mp4 when ffprobe is available
  dropped     frames missing from the take, counted from intervals longer than
              1.5 frame periods. The first second is reported separately
              ("start-up"): timestamps can step there while a kit camera joins
              the kit's clock, and that is not a loss.
  fps         measured from the frame timestamps. Kit takes run at about
              29.7 fps on purpose (see below); single cameras at 30.
  duration    first to last frame, and the .mp4's own duration
  bitrate     average video bitrate of the .mp4
  motion      .imu sample count, rate and the longest gap
  sync        kit takes only. "synced" says the camera's timestamps are on the
              kit's shared clock. "Frame alignment" is how far apart each frame
              was captured from the matching frame of the kit's master camera
              (median, 95th percentile, maximum); current firmware holds this
              well under 1 ms. A larger value means frames were captured at
              slightly different instants -- their timestamps still say exactly
              when, so data can still be lined up by time. The first 2 s of a
              take are left out while cameras lock on (--skip).

Kit frame rate: in a Wrist Kit every camera records at the same, slightly
lower rate — about 29.7 fps — so each one can hold its frames in step with the
kit's master camera. This is intentional; a camera recording on its own runs at
30 fps.

Exit status: 0 if every check passed, 1 if any check warned or failed, so the
script can gate a batch.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.reader import read_imu, read_vts  # noqa: E402
from trinet_tools.tmf import read_tmf  # noqa: E402

SOLO_FPS = 30.0
STARTUP_S = 1.0               # start-of-take window reported separately for drops
KIT_FPS = (29.5, 29.9)        # expected kit rate band
SYNC_OK_MS = 1.0              # frame alignment, 95th percentile within this -> OK
GRP_RE = re.compile(r"^(grp\d+)_")


# ---------------------------------------------------------------------------
#  Finding recordings
# ---------------------------------------------------------------------------
def _base(path: Path) -> Path:
    return path.with_suffix("") if path.suffix in (".mp4", ".vts", ".imu", ".json", ".tel") else path


def _camera_key(base: Path):
    """(take key, camera base) for a stream base. Stereo eyes share a camera base."""
    name = base.name
    cam = name[:-2] if name.endswith(("_L", "_R")) else name
    m = GRP_RE.match(name)
    return (m.group(1) if m else None), base.with_name(cam)


def discover(args_paths):
    """Return {camera_base: [stream bases]} for every recording under the given paths."""
    streams = []
    for a in args_paths:
        p = Path(a)
        if p.is_dir():
            streams += [v.with_suffix("") for v in sorted(p.glob("*.vts"))]
        else:
            b = _base(p)
            if not b.with_suffix(".vts").exists():
                raise SystemExit(f"{b.with_suffix('.vts')} not found (the .vts timestamps are needed)")
            streams.append(b)
            # a stereo eye given on its own: check the other eye too
            if b.name.endswith(("_L", "_R")):
                other = b.with_name(b.name[:-1] + ("R" if b.name.endswith("L") else "L"))
                if other.with_suffix(".vts").exists():
                    streams.append(other)
    cams = {}
    for s in dict.fromkeys(streams):
        cams.setdefault(_camera_key(s)[1], []).append(s)
    return cams


def group_takes(cams, only_take=None):
    """[(take name, [camera bases])]: kit takes grouped by grp number, others alone."""
    takes = {}
    for cam in cams:
        take, _ = _camera_key(cams[cam][0])
        key = take or str(cam)
        takes.setdefault(key, []).append(cam)
    out = sorted(takes.items())
    if only_take:
        out = [t for t in out if t[0] == only_take]
        if not out:
            raise SystemExit(f"take {only_take} not found")
    return out


# ---------------------------------------------------------------------------
#  Per-stream measurements
# ---------------------------------------------------------------------------
def ffprobe(mp4: Path):
    if not shutil.which("ffprobe") or not mp4.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
             "format=duration,size:stream=codec_type,codec_name,width,height,"
             "nb_read_packets,sample_rate,bit_rate", "-of", "json", str(mp4)],
            capture_output=True, text=True, timeout=600)
        return json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, ValueError):
        return None


def frame_stats(vts):
    t = vts.best_timestamps_ns.astype(np.int64)
    t = t[t > 0]
    if len(t) < 2:
        return {"frames": int(len(t))}
    d = np.diff(t)
    period = float(np.median(d))
    span = float(t[-1] - t[0])
    missing = np.where(d > 1.5 * period, np.rint(d / period) - 1, 0).astype(int)
    early = t[1:] - t[0] < STARTUP_S * 1e9
    later = d[~early]
    return {
        "frames": int(len(t)),
        "duration_s": round(span / 1e9, 3),
        "nominal_fps": vts.header.fps,
        "fps": round((len(t) - 1) / (span / 1e9), 3),
        "frame_period_ms": round(period / 1e6, 3),
        "dropped": int(missing[~early].sum()),
        "longest_interval_ms": round(float(later.max()) / 1e6, 2) if len(later) else None,
        "startup_irregular": int(missing[early].sum() + (d[early] < 0.5 * period).sum()),
    }


def stream_report(base: Path):
    vts = read_vts(str(base.with_suffix(".vts")))
    r = {"file": base.name, **frame_stats(vts)}
    probe = ffprobe(base.with_suffix(".mp4"))
    if probe:
        fmt = probe.get("format", {})
        v = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
        a = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
        dur = float(fmt.get("duration") or 0)
        r["mp4"] = {
            "codec": v.get("codec_name"),
            "resolution": f"{v.get('width')}x{v.get('height')}" if v else None,
            "frames": int(v["nb_read_packets"]) if v.get("nb_read_packets") else None,
            "duration_s": round(dur, 3) if dur else None,
            "video_bitrate_mbps": round(int(v["bit_rate"]) / 1e6, 2) if v.get("bit_rate")
            else (round(int(fmt["size"]) * 8 / dur / 1e6, 2) if dur and fmt.get("size") else None),
            "audio": f"{a.get('codec_name')} {int(a.get('sample_rate', 0)) / 1000:g} kHz" if a else None,
        }
    elif base.with_suffix(".mp4").exists():
        r["mp4"] = {"note": "install ffprobe for codec, bitrate and mp4 frame count"}
    else:
        r["mp4"] = {"note": "no .mp4 next to the .vts"}
    # motion data and frame timestamps carried inside the .mp4 (TMF)
    try:
        r["embedded_data"] = bool(read_tmf(base.with_suffix(".mp4")).vts_entries)
    except Exception:
        r["embedded_data"] = False
    return r, vts


def imu_report(cam_base: Path, stream_bases):
    for cand in [cam_base.with_suffix(".imu")] + [s.with_suffix(".imu") for s in stream_bases]:
        if cand.exists():
            imu = read_imu(str(cand))
            t = imu.timestamps_ns.astype(np.int64)
            if len(t) < 2:
                return {"file": cand.name, "samples": int(len(t))}
            d = np.diff(t)
            return {"file": cand.name, "samples": int(len(t)),
                    "rate_hz": round(imu.actual_rate_hz, 1),
                    "nominal_hz": imu.header.sample_rate_hz,
                    "duration_s": round(imu.duration_s, 3),
                    "longest_gap_ms": round(float(d.max()) / 1e6, 2),
                    "gaps_over_10ms": int((d > 10_000_000).sum())}
    return None


def role_of(vts, meta):
    if meta and meta.get("role") in ("master", "slave"):
        return meta["role"]
    if vts.header.is_master:
        return "master"
    return "slave" if vts.synced else "solo"


# ---------------------------------------------------------------------------
#  Sync between cameras of a kit take
# ---------------------------------------------------------------------------
def sync_stats(ref_g, g, mask, skip_ns):
    """Signed distance (ms) from each frame to the nearest reference frame."""
    t0 = max(ref_g[0], g[0]) + skip_ns
    sel = (g >= t0) & (g <= ref_g[-1]) & mask
    g = g[sel]
    if len(g) < 10:
        return None
    i = np.clip(np.searchsorted(ref_g, g), 1, len(ref_g) - 1)
    near = np.where(np.abs(ref_g[i - 1] - g) <= np.abs(ref_g[i] - g), ref_g[i - 1], ref_g[i])
    dms = (g - near) / 1e6
    a = np.abs(dms)
    return {"frames": int(len(dms)), "median_ms": round(float(np.median(dms)), 3),
            "p95_abs_ms": round(float(np.percentile(a, 95)), 3),
            "max_abs_ms": round(float(a.max()), 3)}


def overlap_s(gs):
    lo, hi = max(g[0] for g in gs), min(g[-1] for g in gs)
    return round(max(0, hi - lo) / 1e9, 3)


# ---------------------------------------------------------------------------
#  Verdicts
# ---------------------------------------------------------------------------
def verdicts(take):
    out = []
    kit = take["kit"]
    multi = len(take["cameras"]) > 1          # alignment needs a second camera to compare with
    for cam in take["cameras"]:
        for s in cam["streams"]:
            tag = f"{cam['name']} {s['file'][-2:]}" if len(cam["streams"]) > 1 else cam["name"]
            if s.get("dropped"):
                out.append(("WARN", f"{tag}: {s['dropped']} dropped frame(s)"))
            fps = s.get("fps")
            if fps:
                if kit and not (KIT_FPS[0] <= fps <= KIT_FPS[1]):
                    out.append(("WARN", f"{tag}: {fps} fps, outside the kit rate {KIT_FPS[0]}–{KIT_FPS[1]}"))
                if not kit and abs(fps - SOLO_FPS) > 0.1:
                    out.append(("WARN", f"{tag}: {fps} fps, expected {SOLO_FPS:g}"))
            mp4 = s.get("mp4", {})
            if mp4.get("frames") is not None and mp4["frames"] != s.get("frames"):
                out.append(("WARN", f"{tag}: .mp4 has {mp4['frames']} frames, .vts {s.get('frames')}"))
        imu = cam.get("imu")
        if imu is None:
            out.append(("WARN", f"{cam['name']}: no .imu motion data"))
        elif imu.get("gaps_over_10ms"):
            out.append(("WARN", f"{cam['name']}: {imu['gaps_over_10ms']} motion-data gap(s) over 10 ms"))
        if kit and cam["role"] != "master":
            sy = cam.get("sync")
            if not cam["synced"]:
                out.append(("FAIL", f"{cam['name']}: not synced to the kit for this take"))
            elif not multi or cam["name"] == take.get("reference"):
                pass
            elif sy is None:
                out.append(("WARN", f"{cam['name']}: too little overlap to measure frame alignment"))
            elif sy["p95_abs_ms"] > SYNC_OK_MS:
                out.append(("WARN", f"{cam['name']}: frame alignment p95 {sy['p95_abs_ms']} ms "
                                    f"(over {SYNC_OK_MS:g} ms)"))
    if multi and not any(c["role"] == "master" for c in take["cameras"]):
        out.append(("WARN", "kit master camera not among the files; alignment measured against "
                            f"{take['cameras'][0]['name']}"))
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def check_take(name, cam_bases, cams, skip_s):
    take = {"take": name, "cameras": []}
    loaded = []
    for cb in sorted(cam_bases, key=str):
        streams = sorted(cams[cb], key=str)
        meta = None
        js = cb.with_suffix(".json") if cb.with_suffix(".json").exists() else streams[0].with_suffix(".json")
        if js.exists():
            try:
                meta = json.loads(js.read_text())
            except ValueError:
                meta = None
        reps, vtss = zip(*(stream_report(s) for s in streams))
        label = cb.parent.name + "/" + cb.name if cb.parent.name else cb.name
        cam = {"name": label, "role": role_of(vtss[0], meta), "synced": bool(vtss[0].synced),
               "self_reported_sync_us": vtss[0].header.sync_quality_us,
               "streams": list(reps), "imu": imu_report(cb, streams)}
        if meta and "group" in meta:
            cam["kit_group"] = meta["group"]
        take["cameras"].append(cam)
        loaded.append((cam, vtss))

    take["kit"] = len(loaded) > 1 or any(c["role"] in ("master", "slave") for c, _ in loaded)
    if len(loaded) > 1:
        ref_i = next((i for i, (c, _) in enumerate(loaded) if c["role"] == "master"), 0)
        ref_g = loaded[ref_i][1][0].global_sof_ns()
        take["reference"] = loaded[ref_i][0]["name"]
        take["overlap_s"] = overlap_s([v[0].global_sof_ns() for _, v in loaded])
        for i, (cam, vtss) in enumerate(loaded):
            if i == ref_i:
                continue
            v = vtss[0]
            mask = v.phase_locked_mask
            if len(mask) != v.num_frames:
                mask = np.ones(v.num_frames, bool)
            g = v.global_sof_ns()
            ok = v.best_timestamps_ns > 0
            cam["sync"] = sync_stats(ref_g, g[ok], mask[ok], int(skip_s * 1e9))
    # left/right eye of a stereo camera
    for cam, vtss in loaded:
        if len(vtss) == 2:
            a, b = (v.best_timestamps_ns.astype(np.int64) for v in vtss)
            cam["eyes"] = sync_stats(a[a > 0], b[b > 0], np.ones(int((b > 0).sum()), bool), 0)
    take["checks"] = verdicts(take)
    return take


def fmt(x, unit=""):
    return "—" if x is None else f"{x}{unit}"


def print_take(t):
    kind = "kit take" if t["kit"] else "recording"
    print(f"\n=== {t['take']}  ({kind}, {len(t['cameras'])} camera{'s' if len(t['cameras']) > 1 else ''})")
    if t.get("overlap_s") is not None:
        print(f"    all cameras recording together: {t['overlap_s']} s; sync reference: {t['reference']}")
    for c in t["cameras"]:
        grp = f", kit {c['kit_group']}" if "kit_group" in c else ""
        print(f"\n  {c['name']}  [{c['role']}{grp}]")
        for s in c["streams"]:
            m = s.get("mp4", {})
            print(f"    {s['file']}")
            start = f", start-up irregularities {s['startup_irregular']}" if s.get("startup_irregular") else ""
            print(f"      frames {s.get('frames')} (mp4 {fmt(m.get('frames'))}), dropped {fmt(s.get('dropped'))}, "
                  f"longest interval {fmt(s.get('longest_interval_ms'), ' ms')}{start}")
            print(f"      {fmt(s.get('fps'))} fps measured ({fmt(s.get('nominal_fps'))} nominal), "
                  f"duration {fmt(s.get('duration_s'), ' s')} (mp4 {fmt(m.get('duration_s'), ' s')})")
            if "note" in m:
                print(f"      {m['note']}")
            else:
                print(f"      {fmt(m.get('codec'))} {fmt(m.get('resolution'))}, "
                      f"{fmt(m.get('video_bitrate_mbps'), ' Mbps')}, audio {fmt(m.get('audio'))}, "
                      f"data embedded in mp4 {'yes' if s.get('embedded_data') else 'no'}")
        imu = c.get("imu")
        if imu:
            print(f"    motion {imu.get('samples')} samples @ {fmt(imu.get('rate_hz'), ' Hz')}, "
                  f"longest gap {fmt(imu.get('longest_gap_ms'), ' ms')}")
        if c.get("eyes"):
            e = c["eyes"]
            print(f"    left/right eyes: median {e['median_ms']} ms, p95 {e['p95_abs_ms']} ms")
        if c.get("sync"):
            sy = c["sync"]
            print(f"    synced, frame alignment to reference: median {sy['median_ms']} ms, p95 {sy['p95_abs_ms']} ms, "
                  f"max {sy['max_abs_ms']} ms over {sy['frames']} frames")
        elif t["kit"] and c["name"] == t.get("reference"):
            print(f"    synced: {'yes' if c['synced'] else 'NO'} (alignment reference)")
        elif t["kit"] and c["role"] != "master":
            print(f"    synced: {'yes' if c['synced'] else 'NO'}")
    print()
    if not t["checks"]:
        print("  OK — all checks passed")
    for level, msg in t["checks"]:
        print(f"  {level}: {msg}")
    if t["kit"]:
        print("  note: kit takes run at about 29.7 fps by design (single cameras: 30 fps)")
    if t["kit"] and len(t["cameras"]) == 1:
        print("  note: give the other kit cameras' files or folders too to measure frame alignment")


def main():
    ap = argparse.ArgumentParser(
        description="Check Trinet recordings: frame drops, fps, duration, bitrate, motion data "
                    "and Wrist Kit sync")
    ap.add_argument("paths", nargs="+",
                    help="recordings (.mp4/.vts/base name) or folders (one per kit camera)")
    ap.add_argument("--take", help="only this take, e.g. grp143024")
    ap.add_argument("--skip", type=float, default=2.0,
                    help="seconds at the start of a kit take left out of sync figures (default 2)")
    ap.add_argument("--json", metavar="FILE", help="also write the full report as JSON")
    args = ap.parse_args()

    cams = discover(args.paths)
    if not cams:
        raise SystemExit("no recordings (.vts files) found")
    takes = [check_take(n, cb, cams, args.skip) for n, cb in group_takes(cams, args.take)]
    for t in takes:
        print_take(t)
    if args.json:
        Path(args.json).write_text(json.dumps(takes, indent=2))
        print(f"\nReport written to {args.json}")
    sys.exit(1 if any(t["checks"] for t in takes) else 0)


if __name__ == "__main__":
    main()

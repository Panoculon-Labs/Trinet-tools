#!/usr/bin/env python3
"""
Trinet multi-camera synced side-by-side viewer.

Takes two or more recordings of the *same take* (e.g. a head + two wrist
cameras) and renders them side by side, aligned on the shared master clock so
the same instant lines up across panels. Alignment uses each recording's per-
frame global time:

    global_sof_ns = sof_timestamp_ns + master_clock_offset_ns + skew·Δt

(the offset/skew live in each .vts v3 header, written on-device). For every
output frame at time T we pick each camera's frame whose global time is nearest
T, so cameras that free-run at slightly different phases still stay locked to a
common timeline. Each panel is labelled with its role/device and the residual
(how far its shown frame sits from T — bounded by the ~33 ms frame phase, which
clock sync does NOT remove; it only tells you the true capture time).

Usage:
    # explicit list of recordings (each: a .mp4, a base name, or a chunk dir)
    python scripts/sync_view.py head.mp4 wristL.mp4 wristR.mp4 -o take.mp4

    # auto-group every recording in a folder by its session id (.json sidecars)
    python scripts/sync_view.py --auto /mnt/sdcard/Trinet -o take.mp4

    # live preview instead of writing a file
    python scripts/sync_view.py head.mp4 wristL.mp4 --show

Reads:  <rec>.mp4 + <rec>.vts (+ optional <rec>.json for labels), or a chunk
        directory containing partNNN.{mp4,vts}.
Outputs: a side-by-side .mp4 (default <first>_sync.mp4), or a live window.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.reader import read_vts  # noqa: E402

HEADER_H = 34          # px, top global-time bar
LABEL_H = 26           # px, per-panel label strip
PANEL_GAP = 4          # px between panels
BG = (24, 24, 24)
FG = (235, 235, 235)
ACCENT = (90, 200, 90)
WARN = (70, 200, 255)


# ---------------------------------------------------------------------------
#  Resolving a "recording" argument into ordered (mp4, vts) segments + meta
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve_recording(arg: str):
    """
    Return (label, [(mp4_path, VtsData), ...], meta_dict).

    Accepts a .mp4 path, a base name (no extension), or a chunk directory that
    contains partNNN.mp4 / partNNN.vts.
    """
    p = Path(arg)
    segments = []          # list of (mp4_path, vts_path)
    meta = None

    if p.is_dir():
        parts = sorted(p.glob("part*.mp4"))
        if not parts:
            raise ValueError(f"{p}: directory has no part*.mp4 chunks")
        for mp4 in parts:
            vts = mp4.with_suffix(".vts")
            if vts.exists():
                segments.append((mp4, vts))
        meta = _load_json(p.with_suffix(".json")) or _load_json(p / "meta.json")
        label_base = p.name
    else:
        base = p if p.suffix == "" else p.with_suffix("")
        mp4 = base.with_suffix(".mp4")
        vts = base.with_suffix(".vts")
        if not mp4.exists():
            raise ValueError(f"{mp4} not found")
        if not vts.exists():
            raise ValueError(f"{vts} not found (cannot place on the shared clock)")
        segments.append((mp4, vts))
        meta = _load_json(base.with_suffix(".json"))
        label_base = base.name

    loaded = [(str(mp4), read_vts(str(vts))) for mp4, vts in segments]

    # Build a readable label: prefer role + device tag from the .json sidecar.
    if meta:
        role = meta.get("role", "")
        tag = meta.get("device_tag", "")
        label = " ".join(x for x in (role, tag) if x) or label_base
    else:
        label = label_base
    return label, loaded, meta


# ---------------------------------------------------------------------------
#  Per-camera forward-only frame reader on the global timeline
# ---------------------------------------------------------------------------
class CameraStream:
    def __init__(self, label, loaded, meta):
        self.label = label
        self.meta = meta or {}
        self.synced = all(v.synced for _, v in loaded)
        self.quality_us = max((v.header.sync_quality_us for _, v in loaded), default=0)

        self._segs = []         # list of mp4 paths
        gns, smap = [], []      # flat global-ns, parallel (seg_idx, local_idx)
        for si, (mp4, vts) in enumerate(loaded):
            g = vts.global_sof_ns()
            self._segs.append(mp4)
            for li in range(len(g)):
                gns.append(int(g[li]))
                smap.append((si, li))
        order = np.argsort(np.array(gns, dtype=np.int64), kind="stable")
        self.global_ns = np.array(gns, dtype=np.int64)[order]
        self._smap = [smap[i] for i in order]

        self._cap = None
        self._cur_seg = -1
        self._cur_local = -1
        self._frame = None

    @property
    def t_start(self):
        return int(self.global_ns[0])

    @property
    def t_end(self):
        return int(self.global_ns[-1])

    def _open(self, si):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._segs[si])
        self._cur_seg = si
        self._cur_local = -1

    def frame_for(self, target_ns):
        """Nearest frame to target_ns; returns (bgr_or_None, residual_ns)."""
        idx = int(np.searchsorted(self.global_ns, target_ns))
        if idx >= len(self.global_ns):
            idx = len(self.global_ns) - 1
        elif idx > 0 and (target_ns - self.global_ns[idx - 1]) <= (self.global_ns[idx] - target_ns):
            idx -= 1
        seg, local = self._smap[idx]
        if seg != self._cur_seg:
            self._open(seg)
        # forward-decode to the target local frame (output time is monotonic)
        while self._cur_local < local:
            ok, fr = self._cap.read()
            if not ok:
                break
            self._cur_local += 1
            self._frame = fr
        return self._frame, int(self.global_ns[idx] - target_ns)

    def release(self):
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------
def _put(img, text, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(cams, args):
    # Common timeline = overlap of all cameras on the global clock.
    t0 = max(c.t_start for c in cams)
    t1 = min(c.t_end for c in cams)
    if t1 <= t0:
        raise SystemExit("Recordings do not overlap on the shared clock — "
                         "are they the same take? (check session ids)")
    dt_ns = int(1e9 / args.fps)

    # Panel geometry: scale each camera to a common height, keep aspect.
    panel_w = []
    for c in cams:
        cap = cv2.VideoCapture(c._segs[0])
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        cap.release()
        panel_w.append(max(1, int(round(args.height * w / h))))
    total_w = sum(panel_w) + PANEL_GAP * (len(cams) - 1)
    out_h = HEADER_H + args.height + LABEL_H

    writer = None
    if not args.show:
        out_path = args.output or (Path(cams[0]._segs[0]).with_name(
            Path(cams[0]._segs[0]).stem + "_sync.mp4"))
        tmp_path = str(out_path) + ".raw.mp4"
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (total_w, out_h))
        if not writer.isOpened():
            raise SystemExit(f"Could not open VideoWriter for {tmp_path}")

    n_out = int((t1 - t0) // dt_ns) + 1
    print(f"Rendering {n_out} frames @ {args.fps} fps over {(t1 - t0)/1e9:.1f}s "
          f"of overlap across {len(cams)} cameras")

    for k in range(n_out):
        T = t0 + k * dt_ns
        canvas = np.full((out_h, total_w, 3), BG, np.uint8)

        # Pass 1: fetch each camera's nearest frame AND its true global SoF time.
        # frame_for returns resid = (shown frame's global time) - T, so the shown
        # frame's global time is T + resid. We compare cameras to EACH OTHER, not
        # to the playback grid T: when the cameras' capture rate differs slightly
        # from the output fps, the grid beats against capture so the per-camera
        # distance-to-grid sweeps a full frame — but that's a render artifact, not
        # the cross-camera sync. The cross-camera offset is the DIFFERENCE of the
        # shown frames' global times, which is what actually matters.
        frames, gts = [], []
        for c in cams:
            fr, resid_ns = c.frame_for(T)
            frames.append(fr)
            gts.append(T + resid_ns)
        ref = gts[0]                                  # camera 0 (master) = reference
        cross_ms = [(g - ref) / 1e6 for g in gts]
        spread_ms = (max(gts) - min(gts)) / 1e6       # true cross-camera simultaneity error

        x = 0
        for ci, c in enumerate(cams):
            pw = panel_w[ci]
            cell = canvas[HEADER_H:HEADER_H + args.height, x:x + pw]
            if frames[ci] is not None:
                cv2.resize(frames[ci], (pw, args.height), dst=cell, interpolation=cv2.INTER_AREA)
            # per-panel label strip: offset from the reference (master) camera.
            ly = HEADER_H + args.height
            cv2.rectangle(canvas, (x, ly), (x + pw, ly + LABEL_H), (40, 40, 40), -1)
            _put(canvas, f"{c.label}", (x + 6, ly + 17), 0.5, FG)
            off = cross_ms[ci]
            lbl = "ref" if ci == 0 else f"{off:+.1f}ms"
            rc = ACCENT if abs(off) < 2.0 else WARN
            _put(canvas, lbl, (x + pw - 78, ly + 17), 0.45, rc)
            x += pw + PANEL_GAP

        # header: elapsed global time + TRUE cross-camera offset (vs master),
        # NOT the distance-to-playback-grid (which beats with the capture rate).
        cv2.rectangle(canvas, (0, 0), (total_w, HEADER_H), (40, 40, 40), -1)
        _put(canvas, f"t = {(T - t0)/1e9:8.3f} s   (master clock)", (8, 23), 0.6, FG)
        q = max((c.quality_us for c in cams), default=0)
        _put(canvas, f"sync ~{q} us  |  cross-cam {spread_ms:+5.2f} ms",
             (total_w - 360, 23), 0.5, ACCENT if spread_ms < 2.0 else WARN)

        if args.show:
            cv2.imshow("trinet sync view", canvas)
            if cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF in (27, ord("q")):
                break
        else:
            writer.write(canvas)
        if k % 100 == 0:
            print(f"  {k}/{n_out}", end="\r", flush=True)

    for c in cams:
        c.release()
    if writer is not None:
        writer.release()
        _finalize(tmp_path, str(out_path), args.fps)
        print(f"\nWrote {out_path}")
    else:
        cv2.destroyAllWindows()


def _finalize(tmp_path, out_path, fps):
    """Re-encode to H.264 with ffmpeg if available (broadly playable); else keep raw."""
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps), "-i", tmp_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
            capture_output=True)
        if r.returncode == 0:
            os.remove(tmp_path)
            return
        print("  (ffmpeg re-encode failed; keeping raw mp4v output)")
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
#  Auto-grouping a folder by session id
# ---------------------------------------------------------------------------
def auto_group(folder: str):
    """Group recordings in a folder by their session id (from .json sidecars)."""
    folder = Path(folder)
    groups = {}
    for js in folder.glob("*.json"):
        meta = _load_json(js)
        if not meta or "session" not in meta:
            continue
        rec = js.with_suffix("")           # the recording base (or chunk dir)
        target = rec if (rec.with_suffix(".mp4").exists() or rec.is_dir()) else None
        if target is None:
            continue
        groups.setdefault(meta["session"], []).append(str(target))
    if not groups:
        raise SystemExit(f"No grouped recordings (with .json session ids) under {folder}")
    # pick the session with the most cameras (or the newest)
    session = max(groups, key=lambda s: (len(groups[s]), s))
    members = sorted(groups[session])
    print(f"Auto-grouped session {session}: {len(members)} cameras")
    for m in members:
        print(f"  - {m}")
    return members


def main():
    ap = argparse.ArgumentParser(description="Side-by-side viewer for synced Trinet recordings")
    ap.add_argument("recordings", nargs="*", help="2+ recordings (.mp4 / base name / chunk dir)")
    ap.add_argument("--auto", metavar="DIR", help="auto-group all recordings in DIR by session id")
    ap.add_argument("-o", "--output", help="output .mp4 (default <first>_sync.mp4)")
    ap.add_argument("--fps", type=float, default=30.0, help="output fps (default 30)")
    ap.add_argument("--height", type=int, default=480, help="panel height px (default 480)")
    ap.add_argument("--show", action="store_true", help="live preview instead of writing a file")
    args = ap.parse_args()

    recs = auto_group(args.auto) if args.auto else args.recordings
    if len(recs) < 2:
        ap.error("need at least 2 recordings (or --auto DIR with a multi-camera session)")

    cams = []
    for r in recs:
        label, loaded, meta = resolve_recording(r)
        cams.append(CameraStream(label, loaded, meta))

    # Sync report
    print("\nCameras:")
    for c in cams:
        if c.synced:
            print(f"  {c.label:24s} synced, ~{c.quality_us} us, "
                  f"{len(c.global_ns)} frames")
        else:
            print(f"  {c.label:24s} NOT synced (no v3 offset) — aligning by raw "
                  f"clock; cross-camera accuracy not guaranteed")
    render(cams, args)


if __name__ == "__main__":
    main()

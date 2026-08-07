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

    # with a per-camera IMU strip (accel + gyro) under each panel, and the
    # master's audio track carried through
    python scripts/sync_view.py head.mp4 wristL.mp4 --imu --audio master -o take.mp4

    # live preview instead of writing a file
    python scripts/sync_view.py head.mp4 wristL.mp4 --show

Reads:  <rec>.mp4 + <rec>.vts (+ optional <rec>.imu for --imu, optional
        <rec>.json for labels), or a chunk directory containing
        partNNN.{mp4,vts,imu}.
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
from trinet_tools.reader import read_imu, read_vts  # noqa: E402

HEADER_H = 34          # px, top global-time bar
LABEL_H = 26           # px, per-panel label strip
PANEL_GAP = 4          # px between panels
BG = (24, 24, 24)
FG = (235, 235, 235)
ACCENT = (90, 200, 90)
WARN = (70, 200, 255)
GRID = (58, 58, 58)
CURSOR = (200, 200, 200)
AXIS_COLORS = [(66, 133, 244), (52, 168, 83), (60, 76, 231)]   # X, Y, Z (BGR)


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

        self._loaded = loaded    # [(mp4_path, VtsData)] — kept for the IMU mapping
        self._cap = None
        self._cur_seg = -1
        self._cur_local = -1
        self._frame = None

        # Filled in by load_imu(); empty until then (and if no .imu sidecar).
        self.imu_ns = np.zeros(0, dtype=np.int64)
        self.imu_accel = np.zeros((0, 3), dtype=np.float32)
        self.imu_gyro = np.zeros((0, 3), dtype=np.float32)
        self.imu_rate_hz = 0.0

    def load_imu(self):
        """
        Load the .imu sidecar(s) and put the samples on the SAME global clock as
        the frames, so an inertial event and the frame that saw it line up on one
        timeline across cameras:

            global = ts + master_clock_offset_ns + skew_ppb * (ts - sof0) / 1e9

        ``sof0`` is the segment's first frame timestamp — the same skew origin
        VtsData.global_sof_ns() uses, so IMU and video cannot drift apart by
        construction. Returns True if any samples were loaded.
        """
        ns, acc, gyr = [], [], []
        for mp4, vts in self._loaded:
            imu_path = Path(mp4).with_suffix(".imu")
            if not imu_path.exists():
                continue
            try:
                imu = read_imu(str(imu_path))
            except (OSError, ValueError) as exc:
                print(f"  ({self.label}: cannot read {imu_path.name}: {exc})")
                continue
            t = imu.timestamps_ns.astype(np.int64)
            hdr = vts.header
            sof = vts.best_timestamps_ns.astype(np.int64)
            if hdr.synced and sof.size:
                off = np.int64(hdr.master_clock_offset_ns)
                skew = np.int64(hdr.clock_skew_ppb)
                t = t + off + (skew * (t - np.int64(sof[0]))) // np.int64(1_000_000_000)
            ns.append(t)
            acc.append(np.asarray(imu.accel, dtype=np.float32))
            gyr.append(np.asarray(imu.gyro, dtype=np.float32))
        if not ns:
            return False
        t = np.concatenate(ns)
        order = np.argsort(t, kind="stable")
        self.imu_ns = t[order]
        self.imu_accel = np.concatenate(acc)[order]
        self.imu_gyro = np.concatenate(gyr)[order]
        span_s = (self.imu_ns[-1] - self.imu_ns[0]) / 1e9
        self.imu_rate_hz = (len(self.imu_ns) - 1) / span_s if span_s > 0 else 0.0
        return True

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


def _draw_trace(canvas, x0, y0, w, h, t_s, vals, t_center, t_window,
                title, unit, y_span=None):
    """
    Scrolling 3-axis strip chart: a ±t_window view of `vals` (N,3) sampled at
    `t_s` seconds, with a cursor at t_center. Y is auto-scaled over the visible
    window unless `y_span` pins it (pinning keeps panels comparable between
    cameras — the wrist that swung hardest is then the one with the tallest
    trace, not just the one with the tightest autoscale).
    """
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (32, 32, 32), -1)
    ml, mr, mt, mb = 34, 6, 15, 4
    cx, cy = x0 + ml, y0 + mt
    cw, chh = w - ml - mr, h - mt - mb
    if cw < 8 or chh < 8:
        return
    lo, hi = t_center - t_window, t_center + t_window
    m = (t_s >= lo) & (t_s <= hi)
    _put(canvas, title, (x0 + 3, y0 + 11), 0.33, (150, 150, 150))

    if y_span is not None:
        y_lo, y_hi = y_span
    elif np.any(m):
        v = vals[m]
        y_lo, y_hi = float(np.nanmin(v)), float(np.nanmax(v))
        pad = max(0.05 * (y_hi - y_lo), 1e-3)
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        y_lo, y_hi = -1.0, 1.0
    if y_hi - y_lo < 1e-6:
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5

    for frac in (0.0, 0.5, 1.0):
        gy = int(cy + chh * frac)
        cv2.line(canvas, (cx, gy), (cx + cw, gy), GRID, 1)
        _put(canvas, f"{y_hi - frac * (y_hi - y_lo):+.0f}", (x0 + 1, gy + 4), 0.28, (130, 130, 130))
    _put(canvas, unit, (x0 + cw + ml - 34, y0 + 11), 0.28, (130, 130, 130))

    if np.any(m):
        tw = t_s[m]
        vw = vals[m]
        step = max(1, len(tw) // max(cw, 1))
        px = (cx + (tw[::step] - lo) / (hi - lo) * cw).astype(np.int32)
        for ax in range(vw.shape[1]):
            py = (cy + chh - (vw[::step, ax] - y_lo) / (y_hi - y_lo) * chh)
            py = np.clip(py, cy, cy + chh).astype(np.int32)
            cv2.polylines(canvas, [np.stack([px, py], axis=1).reshape(-1, 1, 2)],
                          False, AXIS_COLORS[ax % 3], 1, cv2.LINE_AA)

    cxp = int(cx + (t_center - lo) / (hi - lo) * cw)
    cv2.line(canvas, (cxp, cy), (cxp, cy + chh), CURSOR, 1)
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + chh), (70, 70, 70), 1)


def _audio_sources(cams, t0, master_idx, choice):
    """
    Resolve --audio into [(mp4_path, skip_s), ...] for the muxer.

    Each camera's MP4 starts at its own first frame, but the output starts at
    t0 (the head of the shared overlap), so every source is seeked forward by
    (t0 - camera start) to land on the common timeline.
    """
    choice = (choice or "none").strip().lower()
    if choice in ("", "none", "off"):
        return None
    if choice == "master":
        picks = [master_idx]
    elif choice in ("mix", "all"):
        picks = list(range(len(cams)))
    else:
        try:
            picks = [int(choice)]
        except ValueError:
            raise SystemExit(f"--audio: expected none|master|mix|<panel index>, got {choice!r}")
        if not 0 <= picks[0] < len(cams):
            raise SystemExit(f"--audio: panel index {picks[0]} out of range "
                             f"(0..{len(cams) - 1})")

    out = []
    for i in picks:
        c = cams[i]
        # Only the first segment is used: chunked takes would need a concat
        # graph, which is not worth it for a preview soundtrack.
        src = c._segs[0]
        if not has_audio(src):
            print(f"  (no audio stream in {Path(src).name} — skipping)")
            continue
        out.append((src, (t0 - c.t_start) / 1e9))
    if not out:
        print("  (no usable audio sources — rendering silent)")
        return None
    print(f"  audio: {', '.join(Path(s).name for s, _ in out)}"
          f"{' (mixed)' if len(out) > 1 else ''}")
    return out


def _shared_span(arrays, pct=99.0, pad_frac=0.08):
    """
    One (lo, hi) y-range covering every camera's samples, for comparability.

    Scaled to a percentile band rather than the outright min/max: a single
    knock against a wrist unit is 10x the amplitude of everything else in the
    take, and scaling to it would flatten the whole strip into a straight line.
    Samples beyond the band clip at the chart edge, which still reads as "off
    the scale" without destroying the resolution of ordinary motion.
    """
    vals = [a for a in arrays if a.size]
    if not vals:
        return None
    stacked = np.concatenate([a.reshape(-1) for a in vals])
    stacked = stacked[np.isfinite(stacked)]
    if not stacked.size:
        return None
    lo = float(np.percentile(stacked, 100.0 - pct))
    hi = float(np.percentile(stacked, pct))
    pad = max(pad_frac * (hi - lo), 1e-3)
    return lo - pad, hi + pad


def render(cams, args):
    rotate_set = getattr(args, "rotate_set", set())
    # Common timeline = overlap of all cameras on the global clock.
    t0 = max(c.t_start for c in cams)
    t1 = min(c.t_end for c in cams)
    if t1 <= t0:
        raise SystemExit("Recordings do not overlap on the shared clock — "
                         "are they the same take? (check session ids)")
    dt_ns = int(1e9 / args.fps)

    # Offset reference = the group master (the camera whose clock the others are
    # mapped onto), wherever it sits in the panel order. Falls back to the first
    # camera if no master is identifiable. This lets the master sit anywhere
    # (e.g. head cam in the centre) and still read "ref".
    master_idx = next(
        (i for i, c in enumerate(cams) if (c.meta or {}).get("role") == "master"),
        0)

    # Panel geometry: scale each camera to a common height, keep aspect.
    panel_w = []
    for c in cams:
        cap = cv2.VideoCapture(c._segs[0])
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        cap.release()
        panel_w.append(max(1, int(round(args.height * w / h))))
    total_w = sum(panel_w) + PANEL_GAP * (len(cams) - 1)

    # IMU strips: one accel + one gyro chart under every panel, all cameras
    # sharing one y-scale per quantity so the traces are comparable panel to
    # panel. Times are seconds on the master clock, same origin as the header.
    imu_h = args.imu_h if getattr(args, "imu", False) else 0
    imu_t_s, acc_span, gyr_span = [], None, None
    if imu_h:
        have = [c.load_imu() for c in cams]
        if not any(have):
            print("  (no .imu sidecars found next to these recordings — "
                  "rendering without IMU strips)")
            imu_h = 0
        else:
            for c in cams:
                imu_t_s.append((c.imu_ns - t0).astype(np.float64) / 1e9
                               if len(c.imu_ns) else np.zeros(0))
            in_win = [(c.imu_ns >= t0) & (c.imu_ns <= t1) for c in cams]
            acc_span = _shared_span([c.imu_accel[m] for c, m in zip(cams, in_win)])
            gyr_span = _shared_span([c.imu_gyro[m] for c, m in zip(cams, in_win)])
            for c in cams:
                if len(c.imu_ns):
                    print(f"  IMU {c.label:24s} {len(c.imu_ns)} samples @ "
                          f"{c.imu_rate_hz:.1f} Hz")
                else:
                    print(f"  IMU {c.label:24s} (none)")
    out_h = HEADER_H + args.height + LABEL_H + imu_h

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
        ref = gts[master_idx]                         # group master = reference
        cross_ms = [(g - ref) / 1e6 for g in gts]
        spread_ms = (max(gts) - min(gts)) / 1e6       # true cross-camera simultaneity error

        x = 0
        for ci, c in enumerate(cams):
            pw = panel_w[ci]
            cell = canvas[HEADER_H:HEADER_H + args.height, x:x + pw]
            if frames[ci] is not None:
                fr_disp = (cv2.rotate(frames[ci], cv2.ROTATE_180)
                           if ci in rotate_set else frames[ci])
                cv2.resize(fr_disp, (pw, args.height), dst=cell, interpolation=cv2.INTER_AREA)
            # per-panel label strip: offset from the reference (master) camera.
            ly = HEADER_H + args.height
            cv2.rectangle(canvas, (x, ly), (x + pw, ly + LABEL_H), (40, 40, 40), -1)
            _put(canvas, f"{c.label}", (x + 6, ly + 17), 0.5, FG)
            off = cross_ms[ci]
            lbl = "ref" if ci == master_idx else f"{off:+.1f}ms"
            rc = ACCENT if abs(off) < 2.0 else WARN
            _put(canvas, lbl, (x + pw - 78, ly + 17), 0.45, rc)

            # IMU strip: accel | gyro, side by side, cursor at the current time.
            if imu_h and len(c.imu_ns):
                iy = ly + LABEL_H
                half = (pw - PANEL_GAP) // 2
                tc = (T - t0) / 1e9
                _draw_trace(canvas, x, iy, half, imu_h, imu_t_s[ci], c.imu_accel,
                            tc, args.imu_window, "accel xyz", "m/s2", acc_span)
                _draw_trace(canvas, x + half + PANEL_GAP, iy, pw - half - PANEL_GAP,
                            imu_h, imu_t_s[ci], c.imu_gyro, tc, args.imu_window,
                            "gyro xyz", "rad/s", gyr_span)
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
        audio = _audio_sources(cams, t0, master_idx, getattr(args, "audio", "none"))
        _finalize(tmp_path, str(out_path), args.fps, audio=audio)
        print(f"\nWrote {out_path}")
    else:
        cv2.destroyAllWindows()


def has_audio(path):
    """True if `path` carries at least one audio stream (needs ffprobe)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def _finalize(tmp_path, out_path, fps, audio=None):
    """
    Re-encode to H.264 with ffmpeg if available (broadly playable); else keep raw.

    `audio`, when given, is a list of (mp4_path, skip_s) to carry across: each
    source is seeked `skip_s` into so its samples start at the output's t=0
    (the head of the cameras' shared overlap). Several sources are mixed. Audio
    rides the MP4's own a/v alignment, not the hardware frame timestamps, so
    treat it as ~frame-accurate, not as a sync reference.
    """
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps), "-i", tmp_path]
        for src, skip_s in (audio or []):
            cmd += ["-ss", f"{max(0.0, skip_s):.6f}", "-i", str(src)]
        cmd += ["-map", "0:v", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
        if audio:
            if len(audio) == 1:
                cmd += ["-map", "1:a"]
            else:
                mix = "".join(f"[{i + 1}:a]" for i in range(len(audio)))
                cmd += ["-filter_complex",
                        f"{mix}amix=inputs={len(audio)}:duration=shortest:normalize=0[aout]",
                        "-map", "[aout]"]
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd.append(out_path)
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            os.remove(tmp_path)
            return
        if audio:
            print("  (ffmpeg mux with audio failed; retrying video-only)")
            return _finalize(tmp_path, out_path, fps, audio=None)
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
    ap.add_argument("--rotate180", default="",
                    help="comma list of 0-based panel indices whose video to rotate 180 "
                         "(e.g. inverted-mounted wrist cams): --rotate180 0,2")
    ap.add_argument("--imu", action="store_true",
                    help="draw each camera's accel/gyro under its panel, on the "
                         "same shared clock (reads the .imu sidecars)")
    ap.add_argument("--imu-h", type=int, default=150,
                    help="height px of the IMU strip under each panel (default 150)")
    ap.add_argument("--imu-window", type=float, default=2.0,
                    help="IMU strip half-width in seconds (default 2.0)")
    ap.add_argument("--audio", default="none",
                    help="carry audio through: none (default), master, mix, or a "
                         "0-based panel index")
    args = ap.parse_args()
    args.rotate_set = {int(i) for i in args.rotate180.split(",") if i.strip() != ""}

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

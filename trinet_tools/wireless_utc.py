#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Put memory-card recordings on UTC using the wireless status log.

A Trinet camera keeps no wall clock: its frame timestamps count from power-on.
While it records to its own card it broadcasts its status (including its clock)
over Bluetooth LE; the Trinet Android SDK logs those broadcasts together with the
phone's clock. This module joins the two: it refits the camera-to-phone clock
relation from the logged bucket minima, finds which logged camera boot each
recording belongs to, and maps every frame to UTC with an uncertainty.

Pipeline (each step is a public function):

    log   = wireless_log.load_logs(paths)
    runs  = build_runs(log)                       # merged segments per camera boot
    takes = discover_takes(dirs)                  # recordings on the cards
    res   = match_takes(takes, runs, log, cfg)    # which run each take belongs to
    rows  = [r for t in res for r in take_to_utc(t, cfg).rows]

``scripts/wireless_utc.py`` is the command-line front end; docs/wireless_utc.md
explains the method, the accuracy figures and the recommended field workflow.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .reader import (IMU_HEADER_SIZE, TIMING_EXPOSURE_VALID, TIMING_FRAME_CENTERED,
                     TIMING_MID_EXPOSURE, TIMING_READOUT_VALID, VtsData, read_vts)
from .tmf import read_tmf, read_tmf_meta
from .wireless_log import BootKey, ClockRef, Event, Segment, SegKey, WirelessLog

TOOL_NAME = "trinet-wireless-utc"
TOOL_VERSION = "1.0.0"
SIDECAR_SCHEMA = "trinet-take-utc/1"
REPORT_SCHEMA = "trinet-wireless-utc/1"

NS_PER_MS = 1_000_000
TWO32_MS = 1 << 32
TIMING_OFFSET_STALE = 0x20       # .vts entry flag: kit offset carried forward, not fresh

# Tunables (documented in docs/wireless_utc.md).
PHONE_STEP_MS = 20.0             # a jump of utc-elapsed larger than this is a clock step
PHONE_HOLD_PPM = 50.0            # phone clock drift assumed beyond the clock references
PHONE_STEP_WINDOW_S = 1800.0     # steps this close to a take add to its absolute sigma
PHONE_SIGMA_MS = {"network": 25.0, "system": 100.0, "corrected": 100.0}
MERGE_TOL_MS = 20.0              # segments of one camera boot must agree this well
EXTRAP_PPM = 2.0                 # camera-vs-phone drift change assumed outside the data
LIVE_EXTRAP_PPM = 5.0            # the phone's 2-minute live fit extrapolates worse
EDGE_TOL_MS = 2.0                # advertised edge vs file first/last frame
KIT_MATE_TOL_MS = 2.0
BOOT_CLUSTER_TOL_S = 5.0         # candidates closer than this are the same camera boot
HIGH_SIGMA_MS = 5.0
MEDIUM_SIGMA_MS = 50.0
HIGH_EXTRAP_S = 60.0
LATENCY_BIAS_MAX_MS = 1.5
STALE_SIGMA_MS = 2.0

CONFIDENCE_ORDER = {"unmatched": 0, "ambiguous": 0, "low": 1, "medium": 2, "high": 3}


# ===========================================================================
#  Port of the SDK live fit (DeviceClockFit) — parity with the phone
# ===========================================================================

def _round_half_up(x: float) -> int:
    """Kotlin/Java ``roundToLong`` (Math.round): floor(x + 0.5)."""
    return int(math.floor(x + 0.5))


class DeviceClockFit:
    """Python port of the Android SDK's live clock fit.

    Keeps the minimum ``rx_elapsed_ns - device_ms*1e6`` per ``bucket_ms`` of
    camera time, the last ``max_buckets`` buckets, and a least-squares line
    through them (skew clamped to ±``max_skew_ppm``). Used for the ``live`` fit
    mode and to check parity with the phone (tests/test_fit_vector.py).
    """

    U32 = 0xFFFF_FFFF

    def __init__(self, bucket_ms: int = 5000, max_buckets: int = 24,
                 max_skew_ppm: float = 200.0, jump_reset_ms: int = 10_000):
        self.bucket_ms = int(bucket_ms)
        self.max_buckets = int(max_buckets)
        self.max_skew_ppm = float(max_skew_ppm)
        self.jump_reset_ms = int(jump_reset_ms)
        self.reset()

    def reset(self):
        self.buckets: Dict[int, List[int]] = {}      # key -> [x_ms, min_offset_ns]
        self.samples = 0
        self.last_ms: Optional[int] = None
        self.first_ms = 0
        self.boot_nonce: Optional[int] = None
        self.timebase_is_master = False
        self.ref_ms = 0
        self.intercept_ns = 0
        self.slope = 0.0
        self.residual_ms = 0.0

    def unwrap_near(self, raw_ms: int) -> int:
        raw = int(raw_ms) & self.U32
        ref = self.last_ms
        if ref is None:
            return raw
        cand = (ref & ~self.U32) + raw
        half = 1 << 31
        if cand - ref > half:
            cand -= 1 << 32
        elif ref - cand > half:
            cand += 1 << 32
        return cand

    def to_elapsed_ns(self, device_ns: int) -> Optional[int]:
        if self.samples == 0:
            return None
        return int(device_ns) + self.intercept_ns + _round_half_up(
            self.slope * float(int(device_ns) - self.ref_ms * NS_PER_MS))

    def add_sample(self, device_ms_raw: int, rx_elapsed_ns: int, boot_nonce: int = 0,
                   timebase_is_master: bool = False) -> Tuple[int, bool]:
        did_reset = False
        if self.samples > 0 and (boot_nonce != self.boot_nonce
                                 or timebase_is_master != self.timebase_is_master):
            self.reset()
            did_reset = True
        x = self.unwrap_near(device_ms_raw)
        if self.samples > 0:
            pred = self.to_elapsed_ns(x * NS_PER_MS)
            if abs(int(rx_elapsed_ns) - pred) > self.jump_reset_ms * NS_PER_MS:
                self.reset()
                did_reset = True
                x = int(device_ms_raw) & self.U32
        if self.samples == 0:
            self.boot_nonce = boot_nonce
            self.timebase_is_master = timebase_is_master
            self.first_ms = x
        self.samples += 1
        if self.last_ms is None or x > self.last_ms:
            self.last_ms = x
        off = int(rx_elapsed_ns) - x * NS_PER_MS
        key = x // self.bucket_ms
        b = self.buckets.get(key)
        if b is None:
            self.buckets[key] = [x, off]
            while len(self.buckets) > self.max_buckets:
                del self.buckets[min(self.buckets)]
        elif off < b[1]:
            b[0], b[1] = x, off
        self._refit()
        return x, did_reset

    def _refit(self):
        pts = [self.buckets[k] for k in sorted(self.buckets)]
        if len(pts) < 2:
            m = min(pts, key=lambda p: p[1])
            self.ref_ms, self.intercept_ns, self.slope, self.residual_ms = m[0], m[1], 0.0, 0.0
            return
        r = fit_line_like_sdk([p[0] for p in pts], [p[1] for p in pts], self.max_skew_ppm)
        self.ref_ms, self.intercept_ns, self.slope, self.residual_ms = r

    @property
    def params(self) -> Optional[dict]:
        if self.samples == 0:
            return None
        return {"ref_device_ns": self.ref_ms * NS_PER_MS, "offset_at_ref_ns": self.intercept_ns,
                "skew_ppm": self.slope * 1e6, "residual_ms": self.residual_ms,
                "buckets": len(self.buckets), "samples": self.samples,
                "first_device_ms": self.first_ms, "last_device_ms": self.last_ms}


def fit_line_like_sdk(xs_ms: Sequence[int], ys_ns: Sequence[int],
                      max_skew_ppm: float = 200.0) -> Tuple[int, int, float, float]:
    """The SDK's bucket line fit, bit-for-bit where Kotlin doubles allow.
    Returns ``(ref_ms, offset_at_ref_ns, skew, residual_ms)``."""
    x0, y0 = int(xs_ms[0]), int(ys_ns[0])
    n = float(len(xs_ms))
    sx = sy = 0.0
    for x, y in zip(xs_ms, ys_ns):
        sx += float(int(x) - x0)
        sy += float(int(y) - y0) / NS_PER_MS
    mx, my = sx / n, sy / n
    sxx = sxy = 0.0
    for x, y in zip(xs_ms, ys_ns):
        dx = float(int(x) - x0) - mx
        dy = float(int(y) - y0) / NS_PER_MS - my
        sxx += dx * dx
        sxy += dx * dy
    s = sxy / sxx if sxx > 0 else 0.0
    lim = max_skew_ppm * 1e-6
    s = min(max(s, -lim), lim)
    ref = x0 + int(math.floor(mx))
    icpt_ms = my + s * (float(ref - x0) - mx)
    icpt_ns = y0 + _round_half_up(icpt_ms * NS_PER_MS)
    res = 0.0
    if len(xs_ms) >= 3:
        ss = 0.0
        for x, y in zip(xs_ms, ys_ns):
            dx = float(int(x) - x0) - mx
            dy = float(int(y) - y0) / NS_PER_MS - my
            rr = dy - s * dx
            ss += rr * rr
        res = math.sqrt(ss / len(xs_ms))
    return ref, icpt_ns, s, res


# ===========================================================================
#  Phone clock: phone elapsed time -> UTC
# ===========================================================================

@dataclass
class PhoneStep:
    elapsed_ns: int          # midpoint between the two references around the step
    delta_ms: float          # change of (utc - elapsed)
    reason: str = "jump"


class PhoneClock:
    """Maps one phone boot's ``elapsedRealtime`` nanoseconds to UTC nanoseconds.

    ``policy``: ``best`` (SNTP, else network time, else the system clock),
    ``system``, ``network``, ``sntp`` or ``corrected`` (system clock with every
    later step applied backwards, i.e. trusting the clock as it was last set).
    """

    def __init__(self, refs: List[ClockRef], policy: str = "best"):
        self.refs = sorted(refs, key=lambda r: r.elapsed_ns)
        self.policy = policy
        self.notes: List[str] = []
        self.steps: List[PhoneStep] = []
        self.source = "none"
        self._pieces: List[Tuple[np.ndarray, np.ndarray]] = []   # (elapsed, offset) int64
        self._line = None       # (e_ref, o_ref, slope, e_lo, e_hi)
        self._src_sigma = 0.0
        if not self.refs:
            return
        self._build_system()
        want = policy
        if policy == "best":
            if any(r.sntp_utc_ms is not None and r.sntp_rtt_ms is not None for r in self.refs):
                want = "sntp"
            elif any(r.network_utc_ms is not None for r in self.refs):
                want = "network"
            else:
                want = "system"
        if want == "sntp" and not self._build_sntp():
            self.notes.append("no SNTP references in this phone boot; using the system clock")
            want = "system"
        if want == "network" and not self._build_network():
            self.notes.append("no network-time references in this phone boot; using the system clock")
            want = "system"
        if want == "corrected":
            self._apply_corrected()
        if want in ("system", "corrected"):
            self._src_sigma = PHONE_SIGMA_MS[want]
        self.source = want

    # -- construction -------------------------------------------------------
    def _build_system(self):
        el = np.array([r.elapsed_ns for r in self.refs], dtype=np.int64)
        off = np.array([r.utc_ms * NS_PER_MS - r.elapsed_ns for r in self.refs], dtype=np.int64)
        start = 0
        for i in range(1, len(el)):
            d_ms = (off[i] - off[i - 1]) / NS_PER_MS
            if abs(d_ms) > PHONE_STEP_MS or self.refs[i].reason == "time_set":
                self.steps.append(PhoneStep(int((el[i - 1] + el[i]) // 2), float(d_ms),
                                            "time_set" if self.refs[i].reason == "time_set" else "jump"))
                self._pieces.append((el[start:i], off[start:i]))
                start = i
        self._pieces.append((el[start:], off[start:]))

    def _apply_corrected(self):
        pieces = []
        for i, (e, o) in enumerate(self._pieces):
            later = sum(s.delta_ms for s in self.steps[i:])
            pieces.append((e, o + np.int64(round(later * NS_PER_MS))))
        self._pieces = pieces

    def _fit_line(self, el: np.ndarray, off: np.ndarray, w: Optional[np.ndarray] = None):
        e0, o0 = int(el[0]), int(off[0])
        x = (el - e0).astype(np.float64) / 1e9          # s
        y = (off - o0).astype(np.float64) / 1e6         # ms
        w = np.ones_like(x) if w is None else w
        if len(x) >= 3 and (x[-1] - x[0]) >= 600.0:
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            sxx = np.sum(w * (x - xm) ** 2)
            b = float(np.sum(w * (x - xm) * (y - ym)) / sxx) if sxx > 0 else 0.0
        else:
            xm, ym, b = float(np.average(x, weights=w)), float(np.average(y, weights=w)), 0.0
        self._line = (e0 + int(xm * 1e9), o0 + int(round(ym * 1e6)), b, int(el[0]), int(el[-1]))

    def _build_sntp(self) -> bool:
        rs = [r for r in self.refs if r.sntp_utc_ms is not None and r.sntp_rtt_ms is not None]
        if not rs:
            return False
        rtt = np.array([r.sntp_rtt_ms for r in rs], dtype=np.float64)
        thr = max(2.0 * rtt.min(), rtt.min() + 5.0)
        keep = [r for r, t in zip(rs, rtt) if t <= thr]
        el = np.array([r.elapsed_ns for r in keep], dtype=np.int64)
        off = np.array([r.sntp_utc_ms * NS_PER_MS - r.elapsed_ns for r in keep], dtype=np.int64)
        kr = np.array([r.sntp_rtt_ms for r in keep], dtype=np.float64)
        self._fit_line(el, off, 1.0 / np.maximum(kr, 1.0) ** 2)
        self._src_sigma = max(float(np.median(kr)) / 2.0, 0.5)
        return True

    def _build_network(self) -> bool:
        rs = [r for r in self.refs if r.network_utc_ms is not None]
        if not rs:
            return False
        el = np.array([r.elapsed_ns for r in rs], dtype=np.int64)
        off = np.array([r.network_utc_ms * NS_PER_MS - r.elapsed_ns for r in rs], dtype=np.int64)
        self._fit_line(el, off)
        self._src_sigma = PHONE_SIGMA_MS["network"]
        return True

    # -- evaluation -----------------------------------------------------------
    @property
    def usable(self) -> bool:
        return self.source != "none"

    def offset_ns(self, elapsed_ns) -> np.ndarray:
        """``utc_ns - elapsed_ns`` at each phone elapsed time (int64)."""
        e = np.atleast_1d(np.asarray(elapsed_ns, dtype=np.int64))
        if self._line is not None and self.source in ("sntp", "network"):
            er, orf, b, _, _ = self._line
            dt_s = (e - er).astype(np.float64) / 1e9
            return orf + np.round(dt_s * b * 1e6).astype(np.int64)
        out = np.empty(len(e), dtype=np.int64)
        dist = np.stack([np.where(e < pe[0], pe[0] - e, np.where(e > pe[-1], e - pe[-1], 0))
                         for pe, _ in self._pieces])
        which = np.argmin(dist, axis=0)
        for i, (pe, po) in enumerate(self._pieces):
            sel = which == i
            if not sel.any():
                continue
            base, e0 = int(po[0]), int(pe[0])
            rel = np.interp((e[sel] - e0).astype(np.float64), (pe - e0).astype(np.float64),
                            (po - base).astype(np.float64))
            out[sel] = base + np.round(rel).astype(np.int64)
        return out

    def utc_ns(self, elapsed_ns) -> np.ndarray:
        e = np.atleast_1d(np.asarray(elapsed_ns, dtype=np.int64))
        return e + self.offset_ns(e)

    def _piece_for(self, e: int):
        best, bd = None, None
        for pe, po in self._pieces:
            d = 0 if pe[0] <= e <= pe[-1] else min(abs(e - int(pe[0])), abs(e - int(pe[-1])))
            if bd is None or d < bd:
                best, bd = (pe, po), d
        return best

    def sigma_ms(self, elapsed_ns: int) -> float:
        """1-sigma error of UTC at ``elapsed_ns`` (source + drift away from references)."""
        e = int(elapsed_ns)
        if self._line is not None and self.source in ("sntp", "network"):
            lo, hi = self._line[3], self._line[4]
        else:
            pe, _ = self._piece_for(e)
            lo, hi = int(pe[0]), int(pe[-1])
        d_s = (lo - e) / 1e9 if e < lo else (e - hi) / 1e9 if e > hi else 0.0
        return self._src_sigma + PHONE_HOLD_PPM * 1e-6 * d_s * 1e3

    def steps_near(self, e0: int, e1: int, window_s: float = PHONE_STEP_WINDOW_S) -> List[PhoneStep]:
        w = int(window_s * 1e9)
        return [s for s in self.steps if e0 - w <= s.elapsed_ns <= e1 + w]

    def step_penalty_ms(self, e0: int, e1: int) -> float:
        if self.source not in ("system", "corrected"):
            return 0.0
        return float(sum(abs(s.delta_ms) for s in self.steps_near(e0, e1)))


# ===========================================================================
#  Clock runs (merged segments of one camera boot) and the refit
# ===========================================================================

@dataclass(eq=False)
class ClockRun:
    id: int
    store_id: str
    phone_boot: int
    unit_id: str
    boot_nonce: int
    timebase_is_master: bool
    segments: List[Segment] = field(default_factory=list)
    seg_shift: Dict[SegKey, int] = field(default_factory=dict)   # k: +k*2^32 ms into run domain
    x_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    y_ns: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    n: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def boot_key(self) -> BootKey:
        return (self.store_id, self.phone_boot)

    @property
    def label(self) -> str:
        return f"{self.store_id}:" + ",".join(str(s.id) for s in self.segments)

    @property
    def group_id(self) -> Optional[int]:
        for s in self.segments:
            if s.group_id:
                return s.group_id
        return None

    @property
    def span_ms(self) -> Tuple[int, int]:
        lo = hi = None
        for s in self.segments:
            a, b = s.span_ms
            k = self.seg_shift.get(s.key, 0) * TWO32_MS
            if a is not None:
                lo = a + k if lo is None else min(lo, a + k)
            if b is not None:
                hi = b + k if hi is None else max(hi, b + k)
        return (lo or 0, hi if hi is not None else (lo or 0))

    @property
    def take_range(self) -> Tuple[Optional[int], Optional[int]]:
        lo = [s.take_min for s in self.segments if s.take_min is not None]
        hi = [s.take_max for s in self.segments if s.take_max is not None]
        return (min(lo) if lo else None, max(hi) if hi else None)

    def has_live(self) -> bool:
        return any(s.live_fit is not None for s in self.segments)


def _quick_line(x: np.ndarray, y: np.ndarray):
    """(x0, y0, slope) of an OLS line through bucket minima (y relative, ms)."""
    x0, y0 = int(x[0]), int(y[0])
    if len(x) < 2:
        return x0, y0, 0.0
    X = (x - x0).astype(np.float64)
    Y = (y - y0).astype(np.float64) / NS_PER_MS
    b = np.polyfit(X, Y, 1)[0] if X[-1] > X[0] else 0.0
    return x0, y0, float(b)


def _pred_offset_ns(run: ClockRun, x_ms: float) -> Optional[Tuple[float, int]]:
    """Rough offset of ``run`` at camera time ``x_ms`` and the tolerance to use."""
    if len(run.x_ms):
        m = min(len(run.x_ms), 200)
        xs, ys = run.x_ms[-m:], run.y_ns[-m:]
        x0, y0, b = _quick_line(xs, ys)
        gap_ms = max(0.0, x_ms - float(xs[-1]), float(xs[0]) - x_ms)
        ppm = EXTRAP_PPM if len(xs) >= 2 else PHONE_HOLD_PPM
        return y0 + (x_ms - x0) * b * NS_PER_MS, MERGE_TOL_MS + ppm * 1e-6 * gap_ms
    return None


def build_runs(log: WirelessLog) -> List[ClockRun]:
    """Merge the log's segments into clock runs: same store, phone boot, unit,
    boot nonce and timebase, and offsets that agree within 20 ms (a repeated
    nonce after a camera reboot does not agree and stays separate). Handles a
    camera clock that the phone unwrapped differently across a logger restart
    (u32 wrap) by trying ±2^32 ms shifts."""
    groups: Dict[tuple, List[Segment]] = {}
    for s in log.segments:
        groups.setdefault((s.store_id, s.phone_boot, s.unit_id, s.boot_nonce,
                           s.timebase_is_master), []).append(s)
    runs: List[ClockRun] = []
    for key, segs in groups.items():
        segs = sorted(segs, key=lambda s: (s.first_rx_elapsed_ns if s.first_rx_elapsed_ns is not None
                                           else 0, s.span_ms[0] or 0))
        grp_runs: List[ClockRun] = []
        for s in segs:
            placed = False
            if len(s.device_ms):
                m = min(3, len(s.device_ms))
                j = int(np.argmin(s.min_offset_ns[:m]))
                sx, sy = int(s.device_ms[j]), int(s.min_offset_ns[j])
                for r in reversed(grp_runs):
                    for k in (0, 1, -1, 2):
                        xk = sx + k * TWO32_MS
                        yk = sy - k * TWO32_MS * NS_PER_MS
                        p = _pred_offset_ns(r, float(xk))
                        if p is None:
                            continue
                        if abs(yk - p[0]) / NS_PER_MS <= p[1]:
                            _add_segment(r, s, k)
                            placed = True
                            break
                    if placed:
                        break
            if not placed:
                r = ClockRun(id=0, store_id=key[0], phone_boot=key[1], unit_id=key[2],
                             boot_nonce=key[3], timebase_is_master=key[4])
                _add_segment(r, s, 0)
                grp_runs.append(r)
        runs.extend(grp_runs)
    runs.sort(key=lambda r: (r.unit_id, r.store_id, r.phone_boot, r.span_ms[0]))
    for i, r in enumerate(runs, 1):
        r.id = i
    return runs


def _add_segment(run: ClockRun, s: Segment, k: int):
    run.segments.append(s)
    run.seg_shift[s.key] = k
    if not len(s.device_ms):
        return
    x = s.device_ms + np.int64(k * TWO32_MS)
    y = s.min_offset_ns - np.int64(k * TWO32_MS * NS_PER_MS)
    X = np.concatenate([run.x_ms, x])
    Y = np.concatenate([run.y_ns, y])
    N = np.concatenate([run.n, s.n])
    # Buckets of the same 5 s key from overlapping segments: keep the lower offset.
    order = np.lexsort((Y, X // 5000))
    X, Y, N = X[order], Y[order], N[order]
    keys = X // 5000
    first = np.ones(len(X), dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    o2 = np.argsort(X[first], kind="stable")
    run.x_ms, run.y_ns, run.n = X[first][o2], Y[first][o2], N[first][o2]


# ---------------------------------------------------------------------------
#  Refit
# ---------------------------------------------------------------------------

def _clean_mask(X: np.ndarray, Y: np.ndarray, iters: int = 3) -> Tuple[np.ndarray, float]:
    """One-sided outlier rejection of bucket minima: a bucket whose least-delayed
    advert was still late sits ABOVE the envelope. Residuals are taken against a
    local reference (a running median of neighbouring buckets), so genuine clock
    curvature is not mistaken for outliers."""
    n = len(X)
    mask = np.ones(n, dtype=bool)
    if n < 5:
        return mask, 0.0
    r = np.zeros(n)
    for _ in range(iters):
        idx = np.flatnonzero(mask)
        # Detrend with the line through the kept buckets, then take residuals
        # against a running median (13 buckets) of the detrended values.
        b, a = np.polyfit(X[idx], Y[idx], 1)
        D = Y - (a + b * X)
        xs, ds = X[idx], D[idx]
        h = 6
        ref = np.empty(n)
        for i in range(n):
            j = int(np.searchsorted(xs, X[i]))
            lo, hi = max(0, j - h), min(len(xs), j + h + 1)
            ref[i] = np.median(ds[lo:hi])
        r = D - ref
        med = np.median(r[mask])
        s = 1.4826 * np.median(np.abs(r[mask] - med))
        s = max(s, 0.05)                                    # ms floor
        new = r - med <= 4.0 * s
        if np.array_equal(new, mask):
            break
        mask = new
    return mask, float(np.sqrt(np.mean((r[mask] - np.mean(r[mask])) ** 2)))


class ClockModel:
    """Camera time (ms, run domain) -> phone-elapsed offset (ns) with a 1-sigma.

    ``mode``: ``local`` (LOESS, degree 1, tricube weights over ±``window_s``,
    doubled until at least ``min_buckets`` buckets), ``global`` (one line) or
    ``live`` (the phone's own export-time fit).
    """

    def __init__(self, run: ClockRun, mode: str = "local", window_s: float = 600.0,
                 min_buckets: int = 8, bucket_ms: int = 5000):
        self.run = run
        self.notes: List[str] = []
        self.window_ms = float(window_s) * 1000.0
        self.min_buckets = int(min_buckets)
        self.bucket_ms = bucket_ms
        if mode != "live" and len(run.x_ms) == 0:
            if run.has_live():
                self.notes.append("log has no bucket minima for this run; using the phone's live fit")
                mode = "live"
            else:
                raise ValueError("run has neither bucket minima nor a live fit")
        self.mode = mode
        if len(run.x_ms):
            self.x0 = int(run.x_ms[0])
            self.y0 = int(run.y_ns[0])
            X = (run.x_ms - self.x0).astype(np.float64)
            Y = (run.y_ns - self.y0).astype(np.float64) / NS_PER_MS
            self.mask, local_rms = _clean_mask(X, Y)
            self.X, self.Y = X[self.mask], Y[self.mask]
            self.rejected = int((~self.mask).sum())
            g = self._ols(self.X, self.Y)
            self.global_line = g
            # RMS scatter of the kept bucket minima about the model in use.
            self.residual_ms = g[3] if mode == "global" or len(self.X) < 5 else local_rms
        else:
            self.x0 = self.y0 = 0
            self.X = self.Y = np.zeros(0)
            self.rejected = 0
            self.global_line = None
            lf = [s.live_fit for s in run.segments if s.live_fit is not None]
            self.residual_ms = max(f.residual_ms for f in lf) if lf else 0.0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _ols(X, Y):
        n = len(X)
        if n == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0)
        xm, ym = float(X.mean()), float(Y.mean())
        sxx = float(np.sum((X - xm) ** 2))
        b = float(np.sum((X - xm) * (Y - ym)) / sxx) if sxx > 0 else 0.0
        r = Y - (ym + b * (X - xm))
        s = float(np.sqrt(np.sum(r ** 2) / max(n - 2, 1))) if n > 2 else 0.3
        return (xm, ym, b, s, sxx, n)

    @property
    def observed_ms(self) -> Tuple[float, float]:
        """Run-domain camera-time range the fit is supported by."""
        if self.mode == "live":
            spans = []
            for s in self.run.segments:
                if s.live_fit is None:
                    continue
                lo, hi = self._live_window(s)
                spans.append((lo, hi))
            return (min(a for a, _ in spans), max(b for _, b in spans))
        return (self.x0 + float(self.X[0]), self.x0 + float(self.X[-1])) if len(self.X) else (0.0, 0.0)

    def _live_window(self, s: Segment) -> Tuple[float, float]:
        k = self.run.seg_shift.get(s.key, 0) * TWO32_MS
        lo, hi = s.span_ms
        hi = (hi if hi is not None else 0) + k
        lo = (lo if lo is not None else hi - k) + k
        nb = s.live_fit.buckets or 1
        return (max(float(lo), float(hi) - nb * self.bucket_ms), float(hi))

    def _local_line(self, t: float) -> Tuple[float, float, float, float, float, float, int]:
        """Weighted local line around t (relative ms): (xm, ym, b, s2, n_eff, sxx_eff, used)."""
        X, Y = self.X, self.Y
        n = len(X)
        W = self.window_ms
        full = max(X[-1] - X[0], 1.0)
        while True:
            lo = int(np.searchsorted(X, t - W, "left"))
            hi = int(np.searchsorted(X, t + W, "right"))
            if hi - lo >= min(self.min_buckets, n) or W > 4 * full:
                break
            W *= 2.0
        xs, ys = X[lo:hi], Y[lo:hi]
        w = (1.0 - np.clip(np.abs(xs - t) / (W * 1.0001), 0, 1) ** 3) ** 3
        sw = float(w.sum())
        if len(xs) < 2 or sw <= 0:
            return self._global_line_tuple()
        xm = float(np.sum(w * xs) / sw)
        ym = float(np.sum(w * ys) / sw)
        sxx = float(np.sum(w * (xs - xm) ** 2))
        b = float(np.sum(w * (xs - xm) * (ys - ym)) / sxx) if sxx > 0 else 0.0
        r = ys - (ym + b * (xs - xm))
        n_eff = sw ** 2 / float(np.sum(w ** 2))
        s2 = float(np.sum(w * r ** 2) / sw) * n_eff / max(n_eff - 2.0, 1.0)
        return xm, ym, b, s2, n_eff, sxx / sw * n_eff, len(xs)

    def _global_line_tuple(self):
        xm, ym, b, s, sxx, n = self.global_line
        return xm, ym, b, s ** 2, float(max(n, 1)), sxx, n

    @staticmethod
    def _line_at(line, t: float) -> Tuple[float, float, int]:
        xm, ym, b, s2, n_eff, sxx_eff, used = line
        if used < 2:
            return ym, 0.5, used
        var = s2 * (1.0 / n_eff + ((t - xm) ** 2 / sxx_eff if sxx_eff > 0 else 0.0))
        return ym + b * (t - xm), math.sqrt(max(var, 0.0)), used

    # -- public ---------------------------------------------------------------
    def eval(self, t_ms) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate at run-domain camera times (ms, float). Returns
        ``(offset_ns int64, sigma_ms, extrapolation_ms, buckets_used)``.

        Outside the observed range the line of the edge window is extrapolated
        and ``(d * 2 ppm)^2`` is added to the variance (d = distance)."""
        t = np.atleast_1d(np.asarray(t_ms, dtype=np.float64))
        off = np.empty(len(t), dtype=np.int64)
        sig = np.empty(len(t))
        ext = np.empty(len(t))
        used = np.empty(len(t), dtype=np.int64)
        lo, hi = self.observed_ms
        for i, ti in enumerate(t):
            d = lo - ti if ti < lo else ti - hi if ti > hi else 0.0
            ext[i] = d
            if self.mode == "live":
                off[i], sig[i], used[i] = self._eval_live(ti)
                continue
            tr = ti - self.x0
            if len(self.X) < 2:
                v, s, u = (float(self.Y[0]) if len(self.Y) else 0.0), 0.5, len(self.X)
            elif self.mode == "global":
                v, s, u = self._line_at(self._global_line_tuple(), tr)
            else:
                te = min(max(tr, float(self.X[0])), float(self.X[-1]))
                v, s, u = self._line_at(self._local_line(te), tr)
            ppm = EXTRAP_PPM if len(self.X) >= 2 else PHONE_HOLD_PPM
            sig[i] = math.sqrt(s ** 2 + (ppm * 1e-6 * d) ** 2)
            off[i] = self.y0 + int(round(v * NS_PER_MS))
            used[i] = u
        return off, sig, ext, used

    def _eval_live(self, t: float) -> Tuple[int, float, int]:
        best = None
        for s in self.run.segments:
            if s.live_fit is None:
                continue
            lo, hi = self._live_window(s)
            d = lo - t if t < lo else t - hi if t > hi else 0.0
            if best is None or d < best[0]:
                best = (d, s)
        d, s = best
        k = self.run.seg_shift.get(s.key, 0) * TWO32_MS
        lf = s.live_fit
        t_seg_ns = int(round((t - k) * NS_PER_MS))
        el = int(lf.elapsed_ns(np.array([t_seg_ns]))[0])
        # offset in run domain: el - t_run_ns ; t_run_ns = t_seg_ns + k*1e6
        off = el - (t_seg_ns + k * NS_PER_MS)
        sig = math.sqrt(max(lf.residual_ms, 0.3) ** 2 + (LIVE_EXTRAP_PPM * 1e-6 * d) ** 2)
        return off, sig, lf.buckets

    def eval_grid(self, t_ms: np.ndarray, step_ms: float = 1000.0):
        """Per-frame evaluation: the model on a 1 s grid, linearly interpolated."""
        t = np.asarray(t_ms, dtype=np.float64)
        if len(t) == 0:
            return np.zeros(0, np.int64), np.zeros(0)
        a, b = float(t.min()), float(t.max())
        g = np.arange(a, b + step_ms, step_ms)
        if g[-1] < b:
            g = np.append(g, b)
        off, sig, _, _ = self.eval(g)
        base = int(off[0])
        rel = np.interp(t, g, (off - base).astype(np.float64))
        return base + np.round(rel).astype(np.int64), np.interp(t, g, sig)


def refit(run: ClockRun, mode: str = "local", window_s: float = 600.0,
          bucket_ms: int = 5000) -> ClockModel:
    """Fit a :class:`ClockModel` to a run's bucket minima."""
    return ClockModel(run, mode=mode, window_s=window_s, bucket_ms=bucket_ms)


# ===========================================================================
#  Take discovery
# ===========================================================================

STEREO_RE = re.compile(
    r"^(?:grp(?P<session>\d+)_(?P<dev8>[0-9a-f]{8})_)?take(?P<num>\d{4})(?:_(?P<eye>[LR]))?"
    r"\.(?P<ext>mp4|vts|imu|tel)$")
MONO_RE = re.compile(
    r"^(?:grp(?P<session>\d+)_(?P<dev8>[0-9a-f]{8})_(?P<kg>\d+)|recording(?P<s>\d+)_(?P<g>\d+))"
    r"\.(?P<ext>mp4|vts|imu|tel|json)$")
MONO_DIR_RE = re.compile(
    r"^(?:grp(?P<session>\d+)_(?P<dev8>[0-9a-f]{8})_(?P<kg>\d+)|recording(?P<s>\d+)_(?P<g>\d+))$")
PART_RE = re.compile(r"^part(?P<p>\d{3,})\.(?P<ext>mp4|vts|imu|tel|json)$")


@dataclass(eq=False)
class TakeFiles:
    """The files of one take (one recording on one camera)."""
    take_path: str                     # directory + base name, no eye suffix or extension
    layout: str                        # stereo | mono | mono_parts
    take_number: int                   # the number the camera advertises for this take
    kit_session: Optional[int] = None
    kit_dev8: Optional[str] = None
    segment: Optional[int] = None      # mono: the <G> of recording<S>_<G>
    mp4: Dict[str, List[Path]] = field(default_factory=dict)    # eye ('' = mono) -> parts
    vts: Dict[str, List[Optional[Path]]] = field(default_factory=dict)
    imu: Optional[Path] = None
    json: Optional[Path] = None

    @property
    def name(self) -> str:
        return Path(self.take_path).name

    @property
    def eyes(self) -> List[str]:
        return sorted(set(self.mp4) | set(self.vts))


def _walk_files(root: Path):
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        yield Path(dp), sorted(f for f in fns if not f.startswith("."))


def discover_takes(dirs: Iterable) -> List[TakeFiles]:
    """Find every take under the given directories (recursively).

    Stereo cameras: ``takeNNNN_L.mp4``/``_R.mp4`` + ``_L.vts``/``_R.vts`` +
    ``takeNNNN.imu``; kit takes ``grp<session>_<dev8>_takeNNNN_*``. Mono cameras:
    ``recording<S>_<G>.*`` (advertised take number = S), chunked
    ``recording<S>_<G>/partNNN.*``, kit ``grp<session>_<dev8>_<G>.*``
    (advertised take number = G)."""
    takes: Dict[Tuple[str, str], TakeFiles] = {}
    for root in dirs:
        root = Path(root)
        if root.is_file():
            root = root.parent
        for d, files in _walk_files(root):
            dm = MONO_DIR_RE.match(d.name)
            parts = sorted(((int(m.group("p")), m.group("ext"), f) for f in files
                            if (m := PART_RE.match(f))))
            if dm and parts:
                tf = _mono_take(d.parent, d.name, dm, "mono_parts", takes)
                tf.layout = "mono_parts"
                byp: Dict[int, Dict[str, Path]] = {}
                for p, ext, f in parts:
                    byp.setdefault(p, {})[ext] = d / f
                for p in sorted(byp):
                    ex = byp[p]
                    if "mp4" in ex or "vts" in ex:
                        tf.mp4.setdefault("", []).append(ex.get("mp4"))
                        tf.vts.setdefault("", []).append(ex.get("vts"))
                    if tf.imu is None and "imu" in ex:
                        tf.imu = ex["imu"]
                    if tf.json is None and "json" in ex:
                        tf.json = ex["json"]
                continue
            for f in files:
                m = STEREO_RE.match(f)
                if m:
                    session = m.group("session")
                    dev8 = m.group("dev8")
                    num = int(m.group("num"))
                    base = (f"grp{session}_{dev8}_" if session else "") + f"take{m.group('num')}"
                    key = (str(d), base)
                    tf = takes.get(key)
                    if tf is None:
                        tf = takes[key] = TakeFiles(str(d / base), "stereo", num,
                                                    int(session) if session else None, dev8)
                    eye = m.group("eye") or ""
                    ext = m.group("ext")
                    if ext == "mp4":
                        tf.mp4[eye] = [d / f]
                    elif ext == "vts":
                        tf.vts[eye] = [d / f]
                    elif ext == "imu":
                        tf.imu = d / f
                    continue
                m = MONO_RE.match(f)
                if m:
                    base = f[: -len(m.group("ext")) - 1]
                    tf = _mono_take(d, base, m, "mono", takes)
                    ext = m.group("ext")
                    if ext == "mp4":
                        tf.mp4[""] = [d / f]
                    elif ext == "vts":
                        tf.vts[""] = [d / f]
                    elif ext == "imu":
                        tf.imu = d / f
                    elif ext == "json":
                        tf.json = d / f
    out = []
    for tf in takes.values():
        # Align vts lists with mp4 lists per eye.
        for eye in tf.eyes:
            mp4s = tf.mp4.get(eye) or []
            vtss = tf.vts.get(eye) or []
            n = max(len(mp4s), len(vtss))
            tf.mp4[eye] = (list(mp4s) + [None] * n)[:n]
            tf.vts[eye] = (list(vtss) + [None] * n)[:n]
        if tf.eyes:
            out.append(tf)
    out.sort(key=lambda t: t.take_path)
    return out


def _mono_take(d: Path, base: str, m, layout: str, takes) -> TakeFiles:
    key = (str(d), base)
    tf = takes.get(key)
    if tf is not None:
        return tf
    session = m.group("session")
    if session:
        g = int(m.group("kg"))
        tf = TakeFiles(str(d / base), layout, g, int(session), m.group("dev8"), segment=g)
    else:
        tf = TakeFiles(str(d / base), layout, int(m.group("s")), segment=int(m.group("g")))
    takes[key] = tf
    return tf


# ---------------------------------------------------------------------------
#  Loading a take's timing
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class EyeTimeline:
    eye: str
    frame_numbers: np.ndarray
    sof_ns: np.ndarray                  # this camera's own clock (int64)
    global_ns: np.ndarray               # kit master clock (== sof_ns when not synced)
    synced: bool
    is_master: bool
    sync_quality_us: int
    stale: np.ndarray                   # bool, kit offset carried forward
    raw_shift_first_ns: Optional[int]   # add to sof to reach the raw start-of-frame
    raw_shift_last_ns: Optional[int]
    widen_ns: int                       # extra edge tolerance when the shift is unknown
    vts_source: str                     # sidecar | embedded

    @property
    def frames(self) -> int:
        return len(self.sof_ns)

    def timeline(self, which: str) -> np.ndarray:
        return self.global_ns if which == "kit_master" else self.sof_ns


@dataclass(eq=False)
class TakeInfo:
    files: TakeFiles
    unit_id: Optional[str]
    unit_id_source: str
    boot_id: Optional[str]
    boot_id_source: Optional[str]
    eyes: Dict[str, EyeTimeline]
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def nonce(self) -> Optional[int]:
        return int(self.boot_id[:2], 16) if self.boot_id else None

    @property
    def primary(self) -> Optional[EyeTimeline]:
        for e in ("L", "", "R"):
            if e in self.eyes and self.eyes[e].frames:
                return self.eyes[e]
        return None

    @property
    def synced(self) -> bool:
        p = self.primary
        return bool(p and p.synced)

    @property
    def is_master(self) -> bool:
        p = self.primary
        return bool(p and p.is_master)


def read_imu_device_id(path) -> str:
    """Device id (hex) from an .imu header, reading only the 64-byte header."""
    with open(path, "rb") as f:
        h = f.read(IMU_HEADER_SIZE)
    if len(h) < IMU_HEADER_SIZE or h[:8].rstrip(b"\x00") != b"TRIMU001":
        return ""
    ver = struct.unpack_from("<I", h, 8)[0]
    did = h[40:56] if ver >= 3 else h[36:52]
    return did.hex() if any(did) else ""


def _vts_from_mp4(mp4: Path) -> Optional[VtsData]:
    data = read_tmf(mp4).vts_bytes()
    if not data:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".vts")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return read_vts(tmp)
    finally:
        os.unlink(tmp)


def _raw_shift(vts: VtsData, i: int) -> Tuple[Optional[int], int]:
    """ns to ADD to the .vts timestamp of frame i to get the raw start-of-frame
    that older firmware advertised; (None, widen) when it cannot be rebuilt."""
    if vts.timing_flags is None or not len(vts.timing_flags):
        return 0, 0                          # pre-v4: .vts already holds the raw SoF
    fl = int(vts.timing_flags[i])
    if not fl & TIMING_MID_EXPOSURE:
        return 0, 0
    exp = int(vts.exposure_us[i]) if vts.exposure_us is not None else 0
    ro = int(vts.readout_time_us[i]) if vts.readout_time_us is not None else 0
    if not (fl & TIMING_EXPOSURE_VALID) or exp == 0:
        return None, 20 * NS_PER_MS
    shift = exp * 1000 // 2
    if fl & TIMING_FRAME_CENTERED:
        if not (fl & TIMING_READOUT_VALID):
            return None, (exp * 1000 // 2) + 20 * NS_PER_MS
        shift -= ro * 1000 // 2
    return shift, 0


def _eye_timeline(eye: str, vts_list: List[VtsData], source: str) -> EyeTimeline:
    fn = np.concatenate([v.frame_numbers.astype(np.int64) for v in vts_list])
    sof = np.concatenate([v.best_timestamps_ns.astype(np.int64) for v in vts_list])
    glb = np.concatenate([v.global_sof_ns().astype(np.int64) for v in vts_list])
    stale = np.concatenate([
        (v.timing_flags.astype(np.uint32) & TIMING_OFFSET_STALE) != 0 if v.timing_flags is not None
        else np.zeros(v.num_frames, dtype=bool) for v in vts_list])
    synced = any(v.synced or (v.frame_offset_ns is not None and bool(np.any(v.frame_offset_ns)))
                 for v in vts_list)
    is_master = any(v.header.is_master for v in vts_list)
    first, last = vts_list[0], vts_list[-1]
    rs0, w0 = _raw_shift(first, 0) if first.num_frames else (None, 0)
    rs1, w1 = _raw_shift(last, last.num_frames - 1) if last.num_frames else (None, 0)
    q = max(int(v.header.sync_quality_us) for v in vts_list)
    return EyeTimeline(eye, fn, sof, glb, synced, is_master, q, stale, rs0, rs1,
                       max(w0, w1), source)


def load_take(tf: TakeFiles, default_unit: Optional[str] = None) -> TakeInfo:
    """Read what matching needs: frame timelines per eye, unit id and boot id."""
    notes: List[str] = []
    meta = None
    for eye in tf.eyes:
        for mp4 in tf.mp4.get(eye) or []:
            if mp4 is None:
                continue
            try:
                meta = read_tmf_meta(mp4)
            except (OSError, ValueError) as e:
                notes.append(f"{mp4.name}: no readable metadata ({e})")
            break
        if meta is not None:
            break
    meta = meta or {}
    # --- unit id (priority: embedded metadata > .imu > .json > kit prefix > --unit)
    ids: List[Tuple[str, str]] = []
    if meta.get("device_id"):
        ids.append(("metadata", str(meta["device_id"]).lower()[:8]))
    if tf.imu is not None:
        try:
            d = read_imu_device_id(tf.imu)
            if d:
                ids.append(("imu", d[:8]))
        except OSError:
            pass
    js = {}
    if tf.json is not None:
        try:
            js = json.loads(Path(tf.json).read_text())
            did = js.get("device_id") or js.get("device_tag")
            if did:
                ids.append(("json", str(did).lower()[:8]))
        except (OSError, ValueError):
            js = {}
    if tf.kit_dev8:
        ids.append(("kit_prefix", tf.kit_dev8))
    if default_unit:
        ids.append(("--unit", default_unit.lower()[:8]))
    unit, src = (ids[0][1], ids[0][0]) if ids else (None, "none")
    others = {v for srcname, v in ids if v != unit and srcname != "--unit"}
    if others:
        notes.append(f"unit id conflict: using {unit} ({src}); also saw " + ", ".join(sorted(others)))
    # --- boot id
    boot_id = meta.get("boot_id")
    if boot_id:
        boot_id = str(boot_id).replace("-", "").lower()
        if not re.fullmatch(r"[0-9a-f]{2,}", boot_id):
            notes.append(f"ignoring malformed boot_id {boot_id!r}")
            boot_id = None
    boot_src = meta.get("boot_id_source") if boot_id else None
    # --- timelines
    eyes: Dict[str, EyeTimeline] = {}
    err = None
    for eye in tf.eyes:
        vl, source = [], "sidecar"
        for mp4, vp in zip(tf.mp4.get(eye) or [], tf.vts.get(eye) or []):
            v = None
            if vp is not None:
                try:
                    v = read_vts(vp)
                except (OSError, ValueError) as e:
                    notes.append(f"{Path(vp).name}: unreadable ({e})")
            if v is None and mp4 is not None:
                try:
                    v = _vts_from_mp4(mp4)
                    source = "embedded"
                except (OSError, ValueError) as e:
                    notes.append(f"{mp4.name}: no embedded frame timing ({e})")
            if v is not None and v.num_frames:
                vl.append(v)
        if vl:
            eyes[eye] = _eye_timeline(eye, vl, source)
    if not any(e.frames for e in eyes.values()):
        err = "no_vts"
    ti = TakeInfo(tf, unit, src, boot_id, boot_src, eyes, notes, err)
    if js.get("role") == "master":
        for e in eyes.values():
            e.is_master = True
    return ti


# ===========================================================================
#  Matching
# ===========================================================================

@dataclass
class MatchConfig:
    fit: str = "local"
    window_s: float = 600.0
    max_extrapolation_s: float = 1800.0
    phone_utc: str = "best"
    latency_correction_ms: float = 0.0
    min_confidence: str = "low"
    picks: Dict[str, str] = field(default_factory=dict)     # take path -> "STORE:SEGMENT"
    unit: Optional[str] = None


@dataclass(eq=False)
class Candidate:
    run: ClockRun
    k: int
    timeline: str                      # local | kit_master
    utc_first_ns: int = 0
    rel_sigma_ms: float = 0.0
    extrap_ms: float = 0.0
    edge_ok: bool = False


@dataclass(eq=False)
class TakeMatch:
    take: TakeInfo
    status: str = "unmatched"          # resolved | ambiguous | unmatched | below_min_confidence
    method: Optional[str] = None       # boot_id | edge | range | kit_mate | picked
    confidence: str = "unmatched"
    reason: Optional[str] = None
    chosen: Optional[Candidate] = None
    candidates: List[Candidate] = field(default_factory=list)
    cross_run_delta_ms: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    clusters: List[List[Candidate]] = field(default_factory=list, repr=False)
    ctx: Optional["_Ctx"] = field(default=None, repr=False)


class _Ctx:
    """Shared state for matching: runs, fitted models and phone clocks."""

    def __init__(self, runs: List[ClockRun], log: WirelessLog, cfg: MatchConfig):
        self.runs = runs
        self.log = log
        self.cfg = cfg
        self.by_unit: Dict[str, List[ClockRun]] = {}
        for r in runs:
            self.by_unit.setdefault(r.unit_id, []).append(r)
        self.seg_run: Dict[SegKey, ClockRun] = {s.key: r for r in runs for s in r.segments}
        self._models: Dict[int, ClockModel] = {}
        self._clocks: Dict[BootKey, PhoneClock] = {}
        bucket = {f.store_id: f.bucket_ms for f in log.files}
        self.bucket_ms = bucket

    def model(self, run: ClockRun) -> ClockModel:
        m = self._models.get(run.id)
        if m is None:
            m = self._models[run.id] = ClockModel(run, self.cfg.fit, self.cfg.window_s,
                                                  bucket_ms=self.bucket_ms.get(run.store_id, 5000))
        return m

    def clock(self, key: BootKey) -> PhoneClock:
        c = self._clocks.get(key)
        if c is None:
            c = self._clocks[key] = PhoneClock(self.log.refs_for_boot(key), self.cfg.phone_utc)
        return c


def _applicable_timeline(take: TakeInfo, run: ClockRun) -> Optional[str]:
    if not run.timebase_is_master:
        return "local"
    if take.is_master or take.synced:
        return "kit_master"
    return None


def _wrap_k(t0_ms: float, t1_ms: float, lo: float, hi: float) -> Optional[int]:
    for k in (0, 1, 2, 3, -1):
        a, b = t0_ms - k * TWO32_MS, t1_ms - k * TWO32_MS
        if lo <= a and b <= hi:
            return k
    return None


def _nearest_k(t_ms: float, ref_ms: float) -> int:
    return int(round((t_ms - ref_ms) / TWO32_MS))


def _evaluate(ctx: _Ctx, take: TakeInfo, cand: Candidate, eye: Optional[EyeTimeline] = None,
              per_frame: bool = False) -> dict:
    """UTC of the first/last frame of ``eye`` through ``cand``, with sigmas."""
    eye = eye or take.primary
    tl = eye.timeline(cand.timeline)
    model = ctx.model(cand.run)
    clock = ctx.clock(cand.run.boot_key)
    kns = cand.k * TWO32_MS * NS_PER_MS
    ends = np.array([tl[0], tl[-1]], dtype=np.int64) - kns
    off, sig, ext, used = model.eval(ends.astype(np.float64) / NS_PER_MS)
    el = ends + off
    corr = int(round(ctx.cfg.latency_correction_ms * NS_PER_MS))
    utc = clock.utc_ns(el) - corr
    sig_tb = 0.0
    if cand.timeline == "kit_master" and not eye.is_master:
        sig_tb = eye.sync_quality_us / 1000.0
    stale_n = int(eye.stale.sum())
    if stale_n and cand.timeline == "kit_master":
        sig_tb = math.sqrt(sig_tb ** 2 + STALE_SIGMA_MS ** 2)
    rel = [math.sqrt(float(s) ** 2 + sig_tb ** 2) for s in sig]
    steps = clock.steps_near(int(el[0]), int(el[1]))
    ph = [clock.sigma_ms(int(e)) + clock.step_penalty_ms(int(el[0]), int(el[1])) for e in el]
    absg = [math.sqrt(r ** 2 + p ** 2) for r, p in zip(rel, ph)]
    out = {
        "utc_first_ns": int(utc[0]), "utc_last_ns": int(utc[1]),
        "elapsed_first_ns": int(el[0]),
        "rel_sigma_ms": max(rel), "abs_sigma_ms": max(absg),
        "extrap_ms": float(max(ext)), "fit_residual_ms": float(model.residual_ms),
        "buckets_used": int(min(used)), "phone_source": clock.source,
        "steps": steps, "stale_frames": stale_n, "model_notes": model.notes + clock.notes,
    }
    if per_frame:
        t = (tl - kns).astype(np.float64) / NS_PER_MS
        o, s = model.eval_grid(t)
        e_all = (tl - kns) + o
        u_all = clock.utc_ns(e_all) - corr
        out["frames"] = {"frame_number": eye.frame_numbers, "sof_ns": eye.sof_ns,
                         "timeline_ns": tl, "phone_elapsed_ns": e_all, "utc_ns": u_all,
                         "rel_sigma_ms": np.sqrt(s ** 2 + sig_tb ** 2)}
    return out


def _fill(ctx: _Ctx, take: TakeInfo, c: Candidate) -> Candidate:
    r = _evaluate(ctx, take, c)
    c.utc_first_ns = r["utc_first_ns"]
    c.rel_sigma_ms = r["rel_sigma_ms"]
    c.extrap_ms = r["extrap_ms"]
    return c


def _cluster(cands: List[Candidate], tol_ns: int) -> List[List[Candidate]]:
    cs = sorted(cands, key=lambda c: c.utc_first_ns)
    out: List[List[Candidate]] = []
    for c in cs:
        if out and c.utc_first_ns - out[-1][-1].utc_first_ns <= tol_ns:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def _covering(ctx: _Ctx, take: TakeInfo, run: ClockRun) -> Optional[Candidate]:
    tlname = _applicable_timeline(take, run)
    if tlname is None:
        return None
    tl = take.primary.timeline(tlname)
    M = ctx.cfg.max_extrapolation_s * 1000.0
    lo, hi = run.span_ms
    k = _wrap_k(tl[0] / NS_PER_MS, tl[-1] / NS_PER_MS, lo - M, hi + M)
    if k is None:
        return None
    try:
        ctx.model(run)
    except ValueError:
        return None
    if not ctx.clock(run.boot_key).usable:
        return None
    return _fill(ctx, take, Candidate(run, k, tlname))


def _edge_match(take: TakeInfo, ev: Event, tl: np.ndarray, ev_run_ms: float,
                eye: EyeTimeline) -> Tuple[bool, int]:
    """Does the advertised edge (run-domain ms) match the take's first/last frame?"""
    if ev.kind == "started":
        t, rs = int(tl[0]), eye.raw_shift_first_ns
    elif ev.kind == "stopped":
        t, rs = int(tl[-1]), eye.raw_shift_last_ns
    else:
        return False, 0
    k = _nearest_k(t / NS_PER_MS, ev_run_ms)
    tt = t - k * TWO32_MS * NS_PER_MS
    e_ns = ev_run_ms * NS_PER_MS
    tol = EDGE_TOL_MS * NS_PER_MS
    if abs(e_ns - tt) <= tol:
        return True, k
    if take.boot_id is None:                  # older firmware advertised the raw SoF
        if rs is not None and rs != 0 and abs(e_ns - (tt + rs)) <= tol:
            return True, k
        if rs is None and abs(e_ns - tt) <= tol + eye.widen_ns:
            return True, k
    return False, k


def _events_for(ctx: _Ctx, unit: str, num: int) -> List[Event]:
    return [e for e in ctx.log.events if e.unit_id == unit and e.take_number == num
            and e.kind in ("started", "stopped") and e.device_ms is not None]


def _runs_for_event(ctx: _Ctx, ev: Event) -> List[ClockRun]:
    if ev.seg_key is not None and ev.seg_key in ctx.seg_run:
        return [ctx.seg_run[ev.seg_key]]
    return [r for r in ctx.by_unit.get(ev.unit_id, [])
            if r.store_id == ev.store_id and r.boot_nonce == ev.boot_nonce]


def _edge_candidates(ctx: _Ctx, take: TakeInfo, runs: Optional[List[ClockRun]] = None
                     ) -> List[Candidate]:
    out: Dict[Tuple[int, int, str], Candidate] = {}
    eye = take.primary
    for ev in _events_for(ctx, take.unit_id, take.files.take_number):
        for run in _runs_for_event(ctx, ev):
            if runs is not None and run not in runs:
                continue
            tlname = _applicable_timeline(take, run)
            if tlname is None:
                continue
            shift = 0
            if ev.seg_key is not None:
                shift = run.seg_shift.get(ev.seg_key, 0)
            ev_ms = float(ev.device_ms + shift * TWO32_MS)
            ok, k = _edge_match(take, ev, eye.timeline(tlname), ev_ms, eye)
            if not ok:
                continue
            try:
                ctx.model(run)
            except ValueError:
                continue
            if not ctx.clock(run.boot_key).usable:
                continue
            key = (run.id, k, tlname)
            if key not in out:
                c = _fill(ctx, take, Candidate(run, k, tlname, edge_ok=True))
                out[key] = c
    return list(out.values())


def _pick_best(cluster: List[Candidate]) -> Tuple[Candidate, Optional[float]]:
    best = min(cluster, key=lambda c: c.rel_sigma_ms)
    delta = None
    if len(cluster) > 1:
        u = [c.utc_first_ns for c in cluster]
        delta = (max(u) - min(u)) / NS_PER_MS
    return best, delta


def _resolve(m: TakeMatch, method: str, cluster: List[Candidate]):
    best, delta = _pick_best(cluster)
    m.status, m.method, m.chosen, m.cross_run_delta_ms = "resolved", method, best, delta


def _tier1(ctx: _Ctx, take: TakeInfo, m: TakeMatch):
    runs = ctx.by_unit.get(take.unit_id, [])
    same = [r for r in runs if r.boot_nonce == take.nonce]
    if not same:
        m.reason = "boot_not_observed"
        return
    cands = [c for r in same if (c := _covering(ctx, take, r)) is not None]
    if not cands:
        if all(_applicable_timeline(take, r) is None for r in same):
            m.reason = "timebase_inconsistent"
        else:
            m.reason = "no_covering_run"
        return
    clusters = _cluster(cands, int(BOOT_CLUSTER_TOL_S * 1e9))
    m.candidates = cands
    if len(clusters) == 1:
        _resolve(m, "boot_id", clusters[0])
        return
    # Nonce collision (different camera boots sharing the one-byte nonce):
    # the advertised start/stop edge decides.
    edge = _edge_candidates(ctx, take, same)
    edge_runs = {c.run.id for c in edge}
    hit = [cl for cl in clusters if any(c.run.id in edge_runs for c in cl)]
    if len(hit) == 1:
        _resolve(m, "boot_id", hit[0])
        m.notes.append("boot nonce seen in several camera boots; resolved by the recording edge")
        return
    m.status = "ambiguous"
    m.reason = "nonce_collision"
    m.clusters = clusters           # for boot-group consensus


def _tier2(ctx: _Ctx, take: TakeInfo, m: TakeMatch) -> bool:
    cands = _edge_candidates(ctx, take)
    if not cands:
        return False
    clusters = _cluster(cands, int(BOOT_CLUSTER_TOL_S * 1e9))
    m.candidates = cands
    if len(clusters) == 1:
        _resolve(m, "edge", clusters[0])
        return True
    m.status, m.reason = "ambiguous", "several_edges_match"
    return True


def _tier3(ctx: _Ctx, take: TakeInfo, m: TakeMatch):
    runs = ctx.by_unit.get(take.unit_id, [])
    cands = [c for r in runs if (c := _covering(ctx, take, r)) is not None]
    if not cands:
        if runs and all(_applicable_timeline(take, r) is None for r in runs):
            m.reason = "timebase_inconsistent"
        else:
            m.reason = "no_covering_run"
        return
    num = take.files.take_number
    sup = [c for c in cands if c.run.take_range[0] is not None
           and c.run.take_range[0] <= num <= c.run.take_range[1]]
    if sup:
        cands = sup
    clusters = _cluster(cands, int(BOOT_CLUSTER_TOL_S * 1e9))
    m.candidates = cands
    if len(clusters) == 1:
        _resolve(m, "range", clusters[0])
        m.notes.append("matched by time range only (no boot id, no advertised edge)")
        return
    m.status, m.reason = "ambiguous", "several_runs_cover_take"


def _tier4(ctx: _Ctx, take: TakeInfo, m: TakeMatch, done: List[TakeMatch]):
    if take.files.kit_session is None or not (take.synced or take.is_master):
        return False
    cands = []
    for o in done:
        ot = o.take
        if (o.status != "resolved" or o.method not in ("boot_id", "edge") or ot is take
                or ot.files.kit_session != take.files.kit_session or ot.unit_id == take.unit_id):
            continue
        c0 = o.chosen
        maps_master = c0.timeline == "kit_master" or (c0.timeline == "local" and ot.is_master)
        if not maps_master:
            continue
        tl = take.primary.timeline("kit_master")
        M = ctx.cfg.max_extrapolation_s * 1000.0
        lo, hi = c0.run.span_ms
        k = _wrap_k(tl[0] / NS_PER_MS, tl[-1] / NS_PER_MS, lo - M, hi + M)
        if k is None:
            continue
        cands.append(_fill(ctx, take, Candidate(c0.run, k, "kit_master")))
    if not cands:
        return False
    clusters = _cluster(cands, int(KIT_MATE_TOL_MS * NS_PER_MS))
    m.candidates = cands
    if len(clusters) == 1:
        _resolve(m, "kit_mate", clusters[0])
        m.notes.append("placed through kit-mates of the same session (this camera was not heard)")
    else:
        m.status, m.reason = "ambiguous", "kit_mates_disagree"
    return True


def _apply_pick(ctx: _Ctx, take: TakeInfo, m: TakeMatch, spec: str) -> bool:
    store, _, seg = spec.rpartition(":")
    try:
        seg_id = int(seg)
    except ValueError:
        m.notes.append(f"--pick {spec!r}: expected STORE:SEGMENT")
        return False
    for key, run in ctx.seg_run.items():
        if key[1] == seg_id and (not store or key[0].startswith(store)):
            tlname = _applicable_timeline(take, run) or "local"
            tl = take.primary.timeline(tlname)
            lo, hi = run.span_ms
            k = max(0, _nearest_k(tl[0] / NS_PER_MS, (lo + hi) / 2.0))
            if not ctx.clock(run.boot_key).usable:
                m.notes.append(f"--pick {spec!r}: no phone clock for that phone boot")
                return False
            c = _fill(ctx, take, Candidate(run, k, tlname))
            m.status, m.method, m.chosen, m.reason = "resolved", "picked", c, None
            m.notes.append(f"run chosen by --pick {spec}")
            return True
    m.notes.append(f"--pick {spec!r}: no such segment in the log")
    return False


def _confidence(m: TakeMatch) -> str:
    if m.status != "resolved":
        return m.status
    c = m.chosen
    if m.method in ("range", "picked"):
        return "low"
    if m.method == "kit_mate":
        return "medium" if c.rel_sigma_ms < MEDIUM_SIGMA_MS else "low"
    if c.extrap_ms / 1000.0 <= HIGH_EXTRAP_S and c.rel_sigma_ms < HIGH_SIGMA_MS:
        return "high"
    if c.rel_sigma_ms < MEDIUM_SIGMA_MS:
        return "medium"
    return "low"


def _find_pick(cfg: MatchConfig, take: TakeInfo) -> Optional[str]:
    tp = take.files.take_path
    for k, v in cfg.picks.items():
        kk = k.rstrip("/")
        for suf in ("_L.mp4", "_R.mp4", ".mp4", ".vts", ".imu", "_L", "_R"):
            if kk.endswith(suf):
                kk = kk[: -len(suf)]
                break
        if tp == kk or tp.endswith("/" + kk) or Path(tp).name == kk \
                or os.path.abspath(kk) == os.path.abspath(tp):
            return v
    return None


def match_takes(takes: List[TakeInfo], runs: List[ClockRun], log: WirelessLog,
                cfg: Optional[MatchConfig] = None) -> List[TakeMatch]:
    """Decide, per take, which clock run (camera boot as heard by a phone) it
    belongs to. Tiers: 1 boot id, 2 advertised edge, 3 time range, 4 kit-mate."""
    cfg = cfg or MatchConfig()
    ctx = _Ctx(runs, log, cfg)
    results: List[TakeMatch] = []
    for take in takes:
        m = TakeMatch(take, notes=list(take.notes), ctx=ctx)
        results.append(m)
        if take.error or take.primary is None:
            m.reason = take.error or "no_vts"
            continue
        pick = _find_pick(cfg, take)
        if pick and _apply_pick(ctx, take, m, pick):
            continue
        if take.unit_id is None:
            m.reason = "no_unit_id"
            continue
        if not ctx.by_unit.get(take.unit_id):
            m.reason = "no_runs_for_unit"
            continue
        if take.boot_id:
            _tier1(ctx, take, m)
            continue
        if _tier2(ctx, take, m):
            continue
        _tier3(ctx, take, m)
    # Boot-group consensus for nonce collisions: all takes with the same full
    # boot id come from the same camera boot.
    for m in results:
        if m.reason != "nonce_collision":
            continue
        sib_runs = {s.chosen.run.id for s in results if s is not m and s.status == "resolved"
                    and s.take.boot_id == m.take.boot_id and s.take.unit_id == m.take.unit_id}
        hit = [cl for cl in m.clusters if any(c.run.id in sib_runs for c in cl)]
        if len(hit) == 1:
            _resolve(m, "boot_id", hit[0])
            m.reason = None
            m.notes.append("boot nonce seen in several camera boots; resolved by other takes "
                           "of the same boot")
    # Kit-mates for cameras the phone never heard.
    for m in results:
        if m.reason == "no_runs_for_unit":
            if _tier4(ctx, m.take, m, results):
                if m.status == "resolved":
                    m.reason = None
    for m in results:
        m.confidence = _confidence(m)
        if m.status == "resolved" and CONFIDENCE_ORDER[m.confidence] < CONFIDENCE_ORDER[cfg.min_confidence]:
            m.status = "below_min_confidence"
    return results


# ===========================================================================
#  Per take x eye results
# ===========================================================================

def utc_iso(ns: Optional[int]) -> Optional[str]:
    if ns is None:
        return None
    s, rem = divmod(int(ns), 1_000_000_000)
    t = _dt.datetime.fromtimestamp(s, tz=_dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{rem // 1000:06d}Z"


ROW_FIELDS = [
    "take_path", "unit_id", "take_number", "kit_session", "eye", "frames", "boot_id",
    "boot_id_source", "status", "match_method", "confidence", "log_file", "store_id",
    "run_segments", "timeline", "utc_first_frame_ns", "utc_first_frame_iso", "utc_last_frame_ns",
    "utc_last_frame_iso", "duration_s", "rel_sigma_ms", "abs_sigma_ms", "latency_bias_ms",
    "extrapolation_s", "fit_residual_ms", "fit_buckets_used", "phone_utc_source",
    "phone_clock_steps", "cross_run_delta_ms", "notes",
]


@dataclass(eq=False)
class TakeUtc:
    match: TakeMatch
    rows: List[dict]
    frames: Dict[str, dict] = field(default_factory=dict)     # eye -> per-frame arrays


def take_to_utc(m: TakeMatch, ctx: Optional[_Ctx] = None, per_frame: bool = False) -> TakeUtc:
    """Build the output rows (one per eye) of a matched take."""
    ctx = ctx or m.ctx
    take = m.take
    rows, frames = [], {}
    c = m.chosen
    lat = ctx.cfg.latency_correction_ms if ctx else 0.0
    eyes = [e for e in take.eyes.values() if e.frames] or [None]
    for eye in eyes:
        notes = list(m.notes)
        row = {k: None for k in ROW_FIELDS}
        row.update({
            "take_path": take.files.take_path, "unit_id": take.unit_id,
            "take_number": take.files.take_number, "kit_session": take.files.kit_session,
            "eye": eye.eye if eye else "", "frames": eye.frames if eye else 0,
            "boot_id": take.boot_id, "boot_id_source": take.boot_id_source,
            "status": m.status, "match_method": m.method, "confidence": m.confidence,
            "latency_bias_ms": [round(0.0 - lat, 3), round(LATENCY_BIAS_MAX_MS - lat, 3)],
        })
        if m.reason:
            notes.insert(0, m.reason)
        if m.status == "ambiguous" and m.candidates:
            notes.append("candidates: " + "; ".join(
                f"{cc.run.label} -> {utc_iso(cc.utc_first_ns)}" for cc in
                sorted(m.candidates, key=lambda x: x.utc_first_ns)))
        if c is not None and eye is not None and m.status in ("resolved", "below_min_confidence"):
            r = _evaluate(ctx, take, c, eye, per_frame=per_frame)
            files = sorted({s.log_file for s in c.run.segments})
            row.update({
                "log_file": ";".join(files), "store_id": c.run.store_id,
                "run_segments": ",".join(str(s.id) for s in c.run.segments),
                "timeline": c.timeline,
                "utc_first_frame_ns": r["utc_first_ns"], "utc_first_frame_iso": utc_iso(r["utc_first_ns"]),
                "utc_last_frame_ns": r["utc_last_ns"], "utc_last_frame_iso": utc_iso(r["utc_last_ns"]),
                "duration_s": round((r["utc_last_ns"] - r["utc_first_ns"]) / 1e9, 6),
                "rel_sigma_ms": round(r["rel_sigma_ms"], 3), "abs_sigma_ms": round(r["abs_sigma_ms"], 3),
                "extrapolation_s": round(r["extrap_ms"] / 1000.0, 1),
                "fit_residual_ms": round(r["fit_residual_ms"], 3),
                "fit_buckets_used": r["buckets_used"], "phone_utc_source": r["phone_source"],
                "phone_clock_steps": [round(s.delta_ms, 1) for s in r["steps"]],
                "cross_run_delta_ms": (round(m.cross_run_delta_ms, 3)
                                       if m.cross_run_delta_ms is not None else None),
            })
            notes.extend(n for n in r["model_notes"] if n not in notes)
            if r["stale_frames"] and c.timeline == "kit_master":
                notes.append(f"{r['stale_frames']} frames carry a carried-forward kit offset")
            if r["steps"]:
                notes.append("phone clock stepped near this take")
            if eye.vts_source == "embedded":
                notes.append("frame timing read from the MP4 (no .vts sidecar)")
            if per_frame:
                frames[eye.eye] = r["frames"]
        row["notes"] = notes
        rows.append(row)
    return TakeUtc(m, rows, frames)


def device_to_utc_ns(run: ClockRun, device_ns, log: WirelessLog, cfg: Optional[MatchConfig] = None,
                     k: int = 0) -> np.ndarray:
    """Map camera-clock nanoseconds (on ``run``'s timebase) straight to UTC ns."""
    cfg = cfg or MatchConfig()
    model = ClockModel(run, cfg.fit, cfg.window_s)
    clock = PhoneClock(log.refs_for_boot(run.boot_key), cfg.phone_utc)
    d = np.atleast_1d(np.asarray(device_ns, dtype=np.int64)) - k * TWO32_MS * NS_PER_MS
    off, _ = model.eval_grid(d.astype(np.float64) / NS_PER_MS)
    return clock.utc_ns(d + off) - int(round(cfg.latency_correction_ms * NS_PER_MS))


# ===========================================================================
#  Kit consistency, summary, writers, inspect
# ===========================================================================

def kit_consistency(results: List[TakeUtc]) -> List[dict]:
    """Per kit session: how well the cameras agree on UTC.

    A kit take is the same moment filmed by several cameras whose files carry
    the kit master's clock. For each resolved camera the UTC of one common
    master-clock instant (the earliest first frame of the group) is
    ``utc(first frame) + (common - master-clock time of first frame)``; the
    spread of that across cameras is the cross-camera UTC disagreement and
    should lie within the combined sigmas. Cameras without a kit offset fall
    back to the spread of first-frame UTC, which also includes genuine
    start-time differences. Takes of one session further apart than 60 s on
    the master clock are treated as separate groups."""
    by: Dict[int, List[Tuple[TakeUtc, dict]]] = {}
    for t in results:
        s = t.match.take.files.kit_session
        if s is None:
            continue
        for r in t.rows:
            if r["status"] == "resolved" and r["eye"] in ("", "L"):
                by.setdefault(s, []).append((t, r))
    out = []
    for s, lst in sorted(by.items()):
        items = []
        for t, r in lst:
            eye = t.match.take.eyes.get(r["eye"])
            items.append((int(eye.global_ns[0]) if eye is not None else 0, t, r, eye))
        items.sort(key=lambda x: x[0])
        groups: List[list] = []
        for it in items:
            if groups and it[0] - groups[-1][-1][0] <= 60 * 1_000_000_000:
                groups[-1].append(it)
            else:
                groups.append([it])
        for g in groups:
            units = sorted({r["unit_id"] for _, _, r, _ in g})
            if len(units) < 2:
                continue
            common = g[0][0]
            on_master = all(e is not None and (e.synced or e.is_master) for _, _, _, e in g)
            if on_master:
                vals = [r["utc_first_frame_ns"] + (common - g0) for g0, _, r, _ in g]
                kind = "master_clock"
            else:
                vals = [r["utc_first_frame_ns"] for _, _, r, _ in g]
                kind = "first_frame"
            out.append({"kit_session": s, "units": units,
                        "takes": [Path(r["take_path"]).name for _, _, r, _ in g],
                        "spread_ms": round((max(vals) - min(vals)) / NS_PER_MS, 3), "basis": kind,
                        "max_abs_sigma_ms": max(r["abs_sigma_ms"] or 0 for _, _, r, _ in g),
                        "max_rel_sigma_ms": max(r["rel_sigma_ms"] or 0 for _, _, r, _ in g)})
    return out


def summarize(results: List[TakeUtc], kits: List[dict]) -> List[str]:
    rows = [r for t in results for r in t.rows]
    lines = [f"{len(results)} take(s), {len(rows)} row(s)"]
    st: Dict[str, int] = {}
    cf: Dict[str, int] = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
        if r["status"] == "resolved":
            cf[r["confidence"]] = cf.get(r["confidence"], 0) + 1
    lines.append("  status:     " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    if cf:
        lines.append("  confidence: " + ", ".join(f"{k}={cf[k]}" for k in ("high", "medium", "low") if k in cf))
    res = [r for r in rows if r["rel_sigma_ms"] is not None]
    if res:
        w = max(res, key=lambda r: r["abs_sigma_ms"])
        lines.append(f"  worst sigma: rel {max(r['rel_sigma_ms'] for r in res):.2f} ms, "
                     f"abs {w['abs_sigma_ms']:.2f} ms ({Path(w['take_path']).name}"
                     f"{'_' + w['eye'] if w['eye'] else ''})")
    lines.append("")
    for r in rows:
        name = Path(r["take_path"]).name + (f"_{r['eye']}" if r["eye"] else "")
        if r["status"] in ("resolved", "below_min_confidence"):
            lines.append(f"  {name:<34} {r['unit_id'] or '?':<8}  {r['utc_first_frame_iso']}  "
                         f"{r['duration_s']:>8.2f} s  {r['match_method']:<8} {r['confidence']:<6} "
                         f"rel {r['rel_sigma_ms']:.2f} ms  abs {r['abs_sigma_ms']:.1f} ms")
        else:
            lines.append(f"  {name:<34} {r['unit_id'] or '?':<8}  {r['status'].upper()}: "
                         + "; ".join(str(n) for n in r["notes"][:2]))
    for k in kits:
        lines.append(f"  kit session {k['kit_session']} ({', '.join(k['takes'])}): "
                     f"{len(k['units'])} cameras agree within {k['spread_ms']:.2f} ms "
                     f"({k['basis']}; max rel sigma {k['max_rel_sigma_ms']:.2f} ms)")
    return lines


def _csv_value(v):
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def write_csv(path, results: List[TakeUtc]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for t in results:
            for r in t.rows:
                row = dict(r)
                row["notes"] = " | ".join(str(n) for n in r["notes"])
                row["phone_clock_steps"] = ";".join(str(x) for x in (r["phone_clock_steps"] or []))
                lb = r["latency_bias_ms"]
                row["latency_bias_ms"] = f"{lb[0]}..{lb[1]}"
                w.writerow({k: _csv_value(v) for k, v in row.items()})


def provenance(log: WirelessLog, params: dict) -> dict:
    return {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "generated_utc": utc_iso(int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1e9)),
        "log_files": [{"path": f.path, "sha256": f.sha256, "version": f.version,
                       "store_id": f.store_id} for f in log.files],
        "params": params,
    }


def write_json(path, results: List[TakeUtc], kits: List[dict], prov: dict):
    doc = {"schema": REPORT_SCHEMA, **prov, "kits": kits,
           "takes": [r for t in results for r in t.rows]}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def write_per_frame(outdir, t: TakeUtc) -> List[Path]:
    out = []
    for eye, fr in t.frames.items():
        name = t.match.take.files.name + (f"_{eye}" if eye else "")
        p = Path(outdir) / f"{name}.utc.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_number", "sof_ns", "timeline_ns", "phone_elapsed_ns", "utc_ns",
                        "rel_sigma_ms"])
            for i in range(len(fr["utc_ns"])):
                w.writerow([int(fr["frame_number"][i]), int(fr["sof_ns"][i]), int(fr["timeline_ns"][i]),
                            int(fr["phone_elapsed_ns"][i]), int(fr["utc_ns"][i]),
                            f"{float(fr['rel_sigma_ms'][i]):.3f}"])
        out.append(p)
    return out


def sidecar_path(t: TakeUtc) -> Path:
    return Path(t.match.take.files.take_path + ".utc.json")


def write_sidecar(t: TakeUtc, prov: dict, force: bool = False) -> Tuple[Path, bool]:
    """Write ``<take>.utc.json`` next to the recording. Never overwrites an
    existing file unless ``force``. Returns (path, written)."""
    p = sidecar_path(t)
    if p.exists() and not force:
        return p, False
    take = t.match.take
    doc = {
        "schema": SIDECAR_SCHEMA, "take": take.files.name, "unit_id": take.unit_id,
        "take_number": take.files.take_number, "kit_session": take.files.kit_session,
        "boot_id": take.boot_id,
        "eyes": {(r["eye"] or "mono"): {k: r[k] for k in ROW_FIELDS if k not in (
            "take_path", "unit_id", "take_number", "kit_session", "boot_id")} for r in t.rows},
        "provenance": prov,
    }
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p, True


def inspect_log(log: WirelessLog, cfg: Optional[MatchConfig] = None) -> List[str]:
    """Human-readable dump of a log: units, runs, spans, residuals, clock refs, steps."""
    cfg = cfg or MatchConfig()
    runs = build_runs(log)
    L = []
    for f in log.files:
        L.append(f"log {f.path}: v{f.version} store {f.store_id} sdk {f.sdk_version or '?'} "
                 f"phone {f.device_model or '?'}{'' if f.complete else ' (TRUNCATED)'}")
    for w in log.warnings:
        L.append(f"  warning: {w}")
    L.append("")
    L.append(f"{len(log.units)} unit(s), {len(log.segments)} segment(s) -> {len(runs)} run(s), "
             f"{len(log.events)} event(s)")
    for uid in sorted(log.units):
        u = log.units[uid]
        ident = (f"  {u.board} {u.hw_generation or ''} fw {u.fw_version or '?'} ({u.build or '?'})"
                 if u.board else "  identity unknown")
        L.append(f"unit {uid}  label={u.label!r} group={u.group_id} role={u.last_role}{ident}")
        for r in (x for x in runs if x.unit_id == uid):
            lo, hi = r.span_ms
            dur_h = max((hi - lo) / 3.6e6, 1e-9)
            try:
                m = ClockModel(r, cfg.fit, cfg.window_s)
                g = m.global_line
                skew = f"{-g[2] * 1e6:+.2f} ppm" if g else "live"   # camera rate vs phone
                res = f"{m.residual_ms:.3f} ms"
                rej = m.rejected
            except ValueError:
                skew, res, rej = "?", "?", 0
            L.append(f"  run {r.id} [{r.label}] phone boot {r.phone_boot} nonce 0x{r.boot_nonce & 0xff:02x} "
                     f"{'master-timebase' if r.timebase_is_master else 'own clock'}")
            L.append(f"      camera {lo / 1000:.1f}..{hi / 1000:.1f} s ({(hi - lo) / 60000:.1f} min), "
                     f"{len(r.x_ms)} buckets ({len(r.x_ms) / dur_h:.0f}/h), skew {skew}, "
                     f"residual {res}, rejected {rej}, takes {r.take_range}")
    L.append("")
    for key in sorted({r.boot_key for r in log.clock_refs}):
        c = PhoneClock(log.refs_for_boot(key), cfg.phone_utc)
        refs = log.refs_for_boot(key)
        L.append(f"phone boot {key[0]}:{key[1]}: {len(refs)} clock refs, source {c.source}, "
                 f"{sum(r.network_utc_ms is not None for r in refs)} network, "
                 f"{sum(r.sntp_utc_ms is not None for r in refs)} SNTP, {len(c.steps)} step(s)")
        for s in c.steps:
            L.append(f"    step {s.delta_ms:+.1f} ms at elapsed {s.elapsed_ns / 1e9:.1f} s ({s.reason})")
    return L


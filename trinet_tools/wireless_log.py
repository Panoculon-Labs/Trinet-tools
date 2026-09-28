#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Reader for the wireless status log exported by the Trinet Android SDK.

The format is specified in docs/wireless_log_format.md: v2 is JSON Lines
(optionally gzip-compressed), the legacy v1 export is one JSON object.

    from trinet_tools.wireless_log import load_logs
    log = load_logs(["phone_a.jsonl.gz", "phone_b.jsonl"])
    for seg in log.segments:
        print(seg.unit_id, seg.boot_nonce, len(seg.device_ms), "buckets")

Several exports (of the same phone, or of several phones) merge into one
:class:`WirelessLog`. Records from the same phone's history store are
de-duplicated by ``(store_id, id)``; everything that is scoped to a phone boot
is keyed by ``(store_id, phone_boot)`` because phone-boot ids are only unique
within one store.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

FORMAT_V2 = "trinet-wireless-log"
FORMAT_V1 = "trinet-wireless-status-log"
DEFAULT_BUCKET_MS = 5000

BootKey = Tuple[str, int]          # (store_id, phone_boot id)
SegKey = Tuple[str, int]           # (store_id, segment id)


# ---------------------------------------------------------------------------
#  Records
# ---------------------------------------------------------------------------

@dataclass
class PhoneBoot:
    store_id: str
    id: int
    boot_count: Optional[int] = None
    first_elapsed_ns: Optional[int] = None
    first_utc_ms: Optional[int] = None

    @property
    def key(self) -> BootKey:
        return (self.store_id, self.id)


@dataclass
class MonitorRun:
    store_id: str
    id: int
    phone_boot: int
    start_elapsed_ns: Optional[int] = None
    start_utc_ms: Optional[int] = None
    stop_elapsed_ns: Optional[int] = None
    stop_utc_ms: Optional[int] = None


@dataclass
class ClockRef:
    store_id: str
    phone_boot: int
    elapsed_ns: int
    utc_ms: int
    read_span_ns: int = 0
    network_utc_ms: Optional[int] = None
    sntp_utc_ms: Optional[int] = None
    sntp_rtt_ms: Optional[float] = None
    reason: str = "periodic"

    @property
    def boot_key(self) -> BootKey:
        return (self.store_id, self.phone_boot)


@dataclass
class UnitInfo:
    unit_id: str
    label: Optional[str] = None
    group_id: Optional[int] = None
    last_role: Optional[str] = None
    last_address: Optional[str] = None
    # From the camera's identity broadcast (firmware 0.5.9+); None when unknown.
    board: Optional[str] = None          # pro_mono | pro_stereo | pro_stereo_gs
    hw_generation: Optional[str] = None  # e.g. "v6"
    fw_version: Optional[str] = None     # e.g. "0.5.9"
    build: Optional[str] = None          # shipping | dev


@dataclass
class LiveFit:
    """The phone's own clock fit at export time:
    ``phone_elapsed_ns = offset_at_ref_ns + device_ns + skew_ppm*1e-6*(device_ns - ref_device_ns)``."""
    ref_device_ns: int
    offset_at_ref_ns: int
    skew_ppm: float
    residual_ms: float = 0.0
    buckets: int = 0

    def elapsed_ns(self, device_ns):
        d = np.asarray(device_ns, dtype=np.int64)
        corr = np.round(self.skew_ppm * 1e-6 * (d - self.ref_device_ns).astype(np.float64)).astype(np.int64)
        return d + np.int64(self.offset_at_ref_ns) + corr


@dataclass(eq=False)
class Segment:
    """One continuous run of one camera clock as heard by one phone boot."""
    store_id: str
    id: int
    unit_id: str
    phone_boot: int
    boot_nonce: int
    timebase_is_master: bool = False
    group_id: Optional[int] = None
    role: Optional[str] = None
    reset_reason: str = "first"
    first_device_ms: Optional[int] = None
    last_device_ms: Optional[int] = None
    first_rx_elapsed_ns: Optional[int] = None
    last_rx_elapsed_ns: Optional[int] = None
    samples: int = 0
    take_min: Optional[int] = None
    take_max: Optional[int] = None
    rec_count_min: Optional[int] = None
    rec_count_max: Optional[int] = None
    live_fit: Optional[LiveFit] = None
    # Bucket minima (columnar, sorted by device_ms).
    device_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    min_offset_ns: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    n: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    log_file: str = ""

    @property
    def key(self) -> SegKey:
        return (self.store_id, self.id)

    @property
    def boot_key(self) -> BootKey:
        return (self.store_id, self.phone_boot)

    @property
    def span_ms(self) -> Tuple[Optional[int], Optional[int]]:
        """Camera-time span seen by the phone (ms, as logged)."""
        lo = self.first_device_ms
        hi = self.last_device_ms
        if len(self.device_ms):
            lo = int(self.device_ms[0]) if lo is None else min(lo, int(self.device_ms[0]))
            hi = int(self.device_ms[-1]) if hi is None else max(hi, int(self.device_ms[-1]))
        return lo, hi


@dataclass
class Event:
    store_id: str
    id: Optional[int]
    unit_id: str
    kind: str                         # started | stopped | abnormal_stop
    take_number: Optional[int] = None
    boot_nonce: Optional[int] = None
    event_seq: Optional[int] = None
    device_ms: Optional[int] = None
    missed_edges: int = 0
    segment: Optional[int] = None
    phone_boot: Optional[int] = None
    detected_elapsed_ns: Optional[int] = None
    utc_ms_live: Optional[int] = None

    @property
    def seg_key(self) -> Optional[SegKey]:
        return None if self.segment is None else (self.store_id, self.segment)


@dataclass
class TakeRecord:
    store_id: str
    unit_id: str
    segment: Optional[int]
    take_number: Optional[int]
    boot_nonce: Optional[int]
    pairing: str
    start: Optional[dict] = None
    stop: Optional[dict] = None


@dataclass
class Sighting:
    store_id: str
    unit_id: str
    phone_boot: int
    window_start_elapsed_ns: int
    adverts: int = 0
    rssi_min: Optional[int] = None
    rssi_max: Optional[int] = None
    rssi_mean: Optional[float] = None
    gap_max_ms: Optional[int] = None


@dataclass
class LogFile:
    path: str
    sha256: str
    version: int
    store_id: str
    sdk_version: Optional[str] = None
    device_model: Optional[str] = None
    exported_utc_ms: Optional[int] = None
    bucket_ms: int = DEFAULT_BUCKET_MS
    complete: bool = True


@dataclass
class WirelessLog:
    files: List[LogFile] = field(default_factory=list)
    phone_boots: Dict[BootKey, PhoneBoot] = field(default_factory=dict)
    monitor_runs: List[MonitorRun] = field(default_factory=list)
    clock_refs: List[ClockRef] = field(default_factory=list)
    units: Dict[str, UnitInfo] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    takes: List[TakeRecord] = field(default_factory=list)
    sightings: List[Sighting] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def refs_for_boot(self, key: BootKey) -> List[ClockRef]:
        return sorted((r for r in self.clock_refs if r.boot_key == key), key=lambda r: r.elapsed_ns)

    def segment(self, key: SegKey) -> Optional[Segment]:
        for s in self.segments:
            if s.key == key:
                return s
        return None


# ---------------------------------------------------------------------------
#  Parsing helpers
# ---------------------------------------------------------------------------

def _int(v, default=None):
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _read_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
                return g.read()
        except (EOFError, OSError):
            # Truncated gzip stream: salvage what decompresses.
            d = gzip.GzipFile(fileobj=io.BytesIO(raw))
            out = bytearray()
            try:
                while True:
                    chunk = d.read(65536)
                    if not chunk:
                        break
                    out += chunk
            except (EOFError, OSError):
                pass
            return bytes(out)
    return raw


def _role_name(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v.lower()
    return {0: "unpaired", 1: "master", 2: "slave"}.get(_int(v), str(v))


def _parse_live_fit(d) -> Optional[LiveFit]:
    if not isinstance(d, dict):
        return None
    if d.get("offset_at_ref_ns") is None:
        return None
    return LiveFit(
        ref_device_ns=_int(d.get("ref_device_ns"), 0),
        offset_at_ref_ns=_int(d.get("offset_at_ref_ns"), 0),
        skew_ppm=_float(d.get("skew_ppm"), 0.0),
        residual_ms=_float(d.get("residual_ms"), 0.0) or 0.0,
        buckets=_int(d.get("buckets"), 0) or 0,
    )


def _parse_segment(o: dict, store_id: str, log_file: str) -> Segment:
    b = o.get("buckets") or {}
    x = np.asarray(b.get("device_ms") or [], dtype=np.int64)
    y = np.asarray(b.get("min_offset_ns") or [], dtype=np.int64)
    n = np.asarray(b.get("n") or [1] * len(x), dtype=np.int64)
    m = min(len(x), len(y))
    x, y = x[:m], y[:m]
    n = n[:m] if len(n) >= m else np.ones(m, np.int64)
    order = np.argsort(x, kind="stable")
    return Segment(
        store_id=store_id, id=_int(o.get("id"), 0), unit_id=str(o.get("unit_id", "")).lower(),
        phone_boot=_int(o.get("phone_boot"), 0), boot_nonce=_int(o.get("boot_nonce"), -1),
        timebase_is_master=bool(o.get("timebase_is_master", False)),
        group_id=_int(o.get("group_id")), role=_role_name(o.get("role")),
        reset_reason=str(o.get("reset_reason", "first")),
        first_device_ms=_int(o.get("first_device_ms")), last_device_ms=_int(o.get("last_device_ms")),
        first_rx_elapsed_ns=_int(o.get("first_rx_elapsed_ns")),
        last_rx_elapsed_ns=_int(o.get("last_rx_elapsed_ns")),
        samples=_int(o.get("samples"), 0) or 0,
        take_min=_int(o.get("take_min")), take_max=_int(o.get("take_max")),
        rec_count_min=_int(o.get("rec_count_min")), rec_count_max=_int(o.get("rec_count_max")),
        live_fit=_parse_live_fit(o.get("live_fit")),
        device_ms=x[order], min_offset_ns=y[order], n=n[order], log_file=log_file,
    )


def _parse_v2(text: str, path: str, sha: str, log: WirelessLog) -> LogFile:
    lines = text.splitlines()
    header = None
    store_id = None
    saw_end = False
    count = 0
    lf = None
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            log.warnings.append(f"{path}:{lineno}: unreadable line (truncated file?) — ignored")
            continue
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if header is None:
            if t != "header":
                raise ValueError(f"{path}: first record is not a header")
            header = o
            ver = _int(o.get("version"), 2)
            if ver != 2:
                log.warnings.append(f"{path}: log version {ver}; reading as v2")
            store_id = str(o.get("store_id") or f"file-{sha[:12]}")
            lf = LogFile(path=path, sha256=sha, version=2, store_id=store_id,
                         sdk_version=o.get("sdk_version"), device_model=o.get("device_model"),
                         exported_utc_ms=_int(o.get("exported_utc_ms")),
                         bucket_ms=_int(o.get("bucket_ms"), DEFAULT_BUCKET_MS) or DEFAULT_BUCKET_MS)
            count += 1          # the header counts: end.records = every line before "end"
            continue
        count += 1
        if t == "phone_boot":
            pb = PhoneBoot(store_id, _int(o.get("id"), 0), _int(o.get("boot_count")),
                           _int(o.get("first_elapsed_ns")), _int(o.get("first_utc_ms")))
            log.phone_boots.setdefault(pb.key, pb)
        elif t == "monitor_run":
            log.monitor_runs.append(MonitorRun(
                store_id, _int(o.get("id"), 0), _int(o.get("phone_boot"), 0),
                _int(o.get("start_elapsed_ns")), _int(o.get("start_utc_ms")),
                _int(o.get("stop_elapsed_ns")), _int(o.get("stop_utc_ms"))))
        elif t == "clock_ref":
            if o.get("elapsed_ns") is None or o.get("utc_ms") is None:
                continue
            log.clock_refs.append(ClockRef(
                store_id, _int(o.get("phone_boot"), 0), _int(o["elapsed_ns"]), _int(o["utc_ms"]),
                _int(o.get("read_span_ns"), 0) or 0, _int(o.get("network_utc_ms")),
                _int(o.get("sntp_utc_ms")), _float(o.get("sntp_rtt_ms")),
                str(o.get("reason", "periodic"))))
        elif t == "unit":
            uid = str(o.get("unit_id", "")).lower()
            if uid:
                u = log.units.get(uid) or UnitInfo(uid)
                u.label = o.get("label") if o.get("label") is not None else u.label
                u.group_id = _int(o.get("group_id"), u.group_id)
                u.last_role = _role_name(o.get("last_role")) or u.last_role
                u.last_address = o.get("last_address") or u.last_address
                for key in ("board", "hw_generation", "fw_version", "build"):
                    if o.get(key) is not None:
                        setattr(u, key, str(o.get(key)))
                log.units[uid] = u
        elif t == "segment":
            log.segments.append(_parse_segment(o, store_id, path))
        elif t == "event":
            log.events.append(Event(
                store_id, _int(o.get("id")), str(o.get("unit_id", "")).lower(), str(o.get("kind", "")),
                _int(o.get("take_number")), _int(o.get("boot_nonce")), _int(o.get("event_seq")),
                _int(o.get("device_ms")), _int(o.get("missed_edges"), 0) or 0, _int(o.get("segment")),
                _int(o.get("phone_boot")), _int(o.get("detected_elapsed_ns")), _int(o.get("utc_ms_live"))))
        elif t == "take":
            log.takes.append(TakeRecord(
                store_id, str(o.get("unit_id", "")).lower(), _int(o.get("segment")),
                _int(o.get("take_number")), _int(o.get("boot_nonce")), str(o.get("pairing", "")),
                o.get("start") if isinstance(o.get("start"), dict) else None,
                o.get("stop") if isinstance(o.get("stop"), dict) else None))
        elif t == "sighting":
            log.sightings.append(Sighting(
                store_id, str(o.get("unit_id", "")).lower(), _int(o.get("phone_boot"), 0),
                _int(o.get("window_start_elapsed_ns"), 0), _int(o.get("adverts"), 0) or 0,
                _int(o.get("rssi_min")), _int(o.get("rssi_max")), _float(o.get("rssi_mean")),
                _int(o.get("gap_max_ms"))))
        elif t == "end":
            saw_end = True
            count -= 1
            declared = _int(o.get("records"))
            if declared is not None and declared != count:
                log.warnings.append(f"{path}: end record declares {declared} records, read {count}")
        # unknown types are ignored (forward compatibility)
    if lf is None:
        raise ValueError(f"{path}: empty log")
    if not saw_end:
        lf.complete = False
        log.warnings.append(f"{path}: no end record — the file is truncated; using what was read")
    return lf


def _parse_v1(o: dict, path: str, sha: str, log: WirelessLog) -> LogFile:
    """Legacy v1: live fits only, one phone reference, one implicit phone boot."""
    store_id = f"v1-{sha[:12]}"
    lf = LogFile(path=path, sha256=sha, version=1, store_id=store_id,
                 exported_utc_ms=_int(o.get("exported_utc_ms")))
    ref = o.get("phone_reference") or {}
    if ref.get("elapsed_realtime_ns") is not None and ref.get("utc_ms") is not None:
        log.clock_refs.append(ClockRef(store_id, 0, _int(ref["elapsed_realtime_ns"]),
                                       _int(ref["utc_ms"]), reason="export"))
    log.phone_boots.setdefault((store_id, 0), PhoneBoot(store_id, 0))
    seg_id = 0
    for u in o.get("units") or []:
        uid = str(u.get("unit_id", "")).lower()
        if not uid:
            continue
        info = log.units.get(uid) or UnitInfo(uid)
        info.group_id = _int(u.get("group_low"), info.group_id)
        info.last_role = _role_name(u.get("role")) or info.last_role
        info.last_address = u.get("address") or info.last_address
        log.units[uid] = info
        for f in u.get("fits") or []:
            seg_id += 1
            lfit = _parse_live_fit(f)
            log.segments.append(Segment(
                store_id=store_id, id=seg_id, unit_id=uid, phone_boot=0,
                boot_nonce=_int(f.get("boot_nonce"), _int(u.get("boot_nonce"), -1)),
                timebase_is_master=bool(f.get("timebase_is_master", False)),
                group_id=info.group_id, role=info.last_role, reset_reason="first",
                first_device_ms=_int(f.get("first_device_ms")),
                last_device_ms=_int(f.get("last_device_ms")),
                samples=_int(f.get("samples"), 0) or 0, live_fit=lfit, log_file=path))
    for i, e in enumerate(o.get("events") or []):
        log.events.append(Event(
            store_id, i + 1, str(e.get("unit_id", "")).lower(), str(e.get("type") or e.get("kind") or ""),
            _int(e.get("take_number")), _int(e.get("boot_nonce")), _int(e.get("event_seq")),
            _int(e.get("device_ms")), _int(e.get("missed_edges"), 0) or 0, None, 0,
            _int(e.get("detected_elapsed_ns")),
            _int(e.get("utc_ms_at_detection"), _int(e.get("utc_ms")))))
    log.warnings.append(f"{path}: legacy v1 log — no bucket minima, only the phone's live fit "
                        "is available (lower accuracy)")
    return lf


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_log(path, into: Optional[WirelessLog] = None) -> WirelessLog:
    """Parse one log file (v1 JSON, v2 JSONL, either optionally gzip) into ``into``."""
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = _read_bytes(p)
    text = data.decode("utf-8", errors="replace")
    log = into if into is not None else WirelessLog()
    obj = None
    try:
        # A v1 export is one JSON document (possibly pretty-printed); a v2 JSONL
        # file with more than one line fails this parse and falls through.
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and obj.get("type") != "header" and (
            _int(obj.get("version")) == 1 or obj.get("format") == FORMAT_V1):
        lf = _parse_v1(obj, str(p), sha, log)
    else:
        lf = _parse_v2(text, str(p), sha, log)
    log.files.append(lf)
    return log


def _merge_dedup(log: WirelessLog) -> WirelessLog:
    """Drop records duplicated across exports of the same store."""
    segs: Dict[SegKey, Segment] = {}
    for s in log.segments:
        old = segs.get(s.key)
        # A later export of the same store holds a superset: keep the richer copy.
        if old is None or (len(s.device_ms), s.samples) > (len(old.device_ms), old.samples):
            segs[s.key] = s
    log.segments = sorted(segs.values(), key=lambda s: (s.unit_id, s.store_id, s.phone_boot,
                                                         s.span_ms[0] if s.span_ms[0] is not None else 0))
    seen = set()
    refs = []
    for r in log.clock_refs:
        k = (r.store_id, r.phone_boot, r.elapsed_ns, r.reason)
        if k not in seen:
            seen.add(k)
            refs.append(r)
    log.clock_refs = refs
    ev: Dict[Tuple, Event] = {}
    for e in log.events:
        k = (e.store_id, e.id) if e.id is not None else (e.store_id, e.unit_id, e.kind, e.device_ms)
        ev.setdefault(k, e)
    log.events = list(ev.values())
    tk: Dict[Tuple, TakeRecord] = {}
    for t in log.takes:
        sid = (t.start or {}).get("event_id"), (t.stop or {}).get("event_id")
        tk[(t.store_id, t.unit_id, t.segment, t.take_number, sid)] = t
    log.takes = list(tk.values())
    sg: Dict[Tuple, Sighting] = {}
    for s in log.sightings:
        sg.setdefault((s.store_id, s.unit_id, s.phone_boot, s.window_start_elapsed_ns), s)
    log.sightings = list(sg.values())
    mr: Dict[Tuple, MonitorRun] = {}
    for m in log.monitor_runs:
        mr[(m.store_id, m.id)] = m
    log.monitor_runs = list(mr.values())
    return log


def load_logs(paths: Iterable) -> WirelessLog:
    """Load and merge several exports (any mix of v1 / v2 / gzip)."""
    log = WirelessLog()
    for p in paths:
        load_log(p, log)
    return _merge_dedup(log)

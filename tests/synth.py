# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Synthetic cameras, phones, wireless logs and recordings for the tests.

Nothing binary is committed: every test builds its inputs here.

Time axis: ``t`` = seconds of the phone's elapsed clock (taken as true time).
A camera's clock runs at ``1 + (skew + curvature*age_h)*1e-6`` of that; the
phone's wall clock is true UTC plus an error and optional steps. Adverts arrive
``0.4 ms + Exp(1.5 ms)`` late, 2 % of them a further 50 ms late.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trinet_tools.reader import (TIMING_EXPOSURE_VALID, TIMING_FRAME_CENTERED, TIMING_MID_EXPOSURE,
                                 TIMING_READOUT_VALID, VTS_ENTRY_FMT_V4, VTS_ENTRY_FMT_V5,
                                 VTS_SYNC_FLAG_IS_MASTER, VTS_SYNC_FLAG_SYNCED)
from trinet_tools.wireless_utc import fit_line_like_sdk

NS = 1_000_000_000
MS = 1_000_000
UTC0_MS = 1_790_000_000_000
TWO32_MS = 1 << 32


# ---------------------------------------------------------------------------
#  Clocks
# ---------------------------------------------------------------------------

@dataclass
class SimCamera:
    unit_id: str = "a1b2c3d4"
    skew_ppm: float = -30.0
    curvature_ppm_per_h: float = 0.0
    boot_nonce: int = 0x3C
    boot_t: float = -600.0                   # phone time (s) at camera power-on
    device_id: Optional[str] = None
    uptime0_ms: int = 0                      # camera clock at power-on (simulate long uptime)

    def __post_init__(self):
        if self.device_id is None:
            self.device_id = self.unit_id + "5da17f71e411e8f1a8103a1558"[:24]

    @property
    def boot_id(self) -> str:
        """32 hex; the first byte is the advertised nonce, the rest unique per boot."""
        tail = hashlib.sha256(f"{self.unit_id}/{self.boot_t}/{self.uptime0_ms}".encode()).hexdigest()
        return f"{self.boot_nonce:02x}" + tail[:30]

    def device_ns(self, t) -> np.ndarray:
        dt = np.asarray(t, dtype=np.float64) - self.boot_t
        d = dt + 1e-6 * (self.skew_ppm * dt + self.curvature_ppm_per_h * dt * dt / 7200.0)
        return (np.round(d * 1e9) + self.uptime0_ms * MS).astype(np.int64)

    def t_of_device(self, dev_ns) -> np.ndarray:
        d = (np.asarray(dev_ns, dtype=np.int64) - self.uptime0_ms * MS).astype(np.float64) / 1e9
        dt = d.copy()
        for _ in range(4):
            f = dt + 1e-6 * (self.skew_ppm * dt + self.curvature_ppm_per_h * dt * dt / 7200.0) - d
            fp = 1 + 1e-6 * (self.skew_ppm + self.curvature_ppm_per_h * dt / 3600.0)
            dt = dt - f / fp
        return dt + self.boot_t


@dataclass
class SimPhone:
    seed: int = 1
    elapsed0_ns: int = 3_600 * NS            # phone elapsed at t = 0
    sys_err_ms: float = 0.0                  # system clock error (constant)
    steps: Sequence[Tuple[float, float]] = ()   # (t_s, delta_ms) system clock steps
    network: bool = False
    network_err_ms: float = 0.0
    sntp_rtt_ms: Optional[float] = None
    latency_min_ms: float = 0.4
    latency_exp_ms: float = 1.5
    outlier_frac: float = 0.02
    outlier_ms: float = 50.0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def elapsed_ns(self, t) -> np.ndarray:
        return (np.round(np.asarray(t, dtype=np.float64) * 1e9) + self.elapsed0_ns).astype(np.int64)

    def true_utc_ns(self, t) -> np.ndarray:
        return (np.round(np.asarray(t, dtype=np.float64) * 1e9) + UTC0_MS * MS).astype(np.int64)

    def system_err_ms(self, t: float) -> float:
        return self.sys_err_ms + sum(d for ts, d in self.steps if t >= ts)

    def latency_ns(self, n: int) -> np.ndarray:
        lat = self.latency_min_ms + self.rng.exponential(self.latency_exp_ms, n)
        out = self.rng.random(n) < self.outlier_frac
        lat[out] += self.outlier_ms
        return np.round(lat * MS).astype(np.int64)


# ---------------------------------------------------------------------------
#  Wireless log builder
# ---------------------------------------------------------------------------

def bucket_minima(dev_ms: np.ndarray, rx_ns: np.ndarray, bucket_ms: int = 5000):
    off = rx_ns - dev_ms * MS
    keys = dev_ms // bucket_ms
    order = np.lexsort((off, keys))
    k, o, x = keys[order], off[order], dev_ms[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    counts = np.diff(np.append(np.flatnonzero(first), len(k)))
    return x[first], o[first], counts


@dataclass
class LogBuilder:
    phone: SimPhone
    store_id: str = "store-a"
    phone_boot: int = 1
    records: List[dict] = field(default_factory=list)
    _seg: int = 0
    _ev: int = 0
    segments: List[dict] = field(default_factory=list)
    units: Dict[str, dict] = field(default_factory=dict)
    samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def clock_refs(self, t0: float, t1: float, every_s: float = 60.0, reason="periodic"):
        ts = np.arange(t0, t1 + 1e-9, every_s)
        for t in ts:
            el = int(self.phone.elapsed_ns(t))
            true_ms = int(self.phone.true_utc_ns(t) // MS)
            r = {"type": "clock_ref", "phone_boot": self.phone_boot, "elapsed_ns": el,
                 "read_span_ns": 4000, "utc_ms": int(round(true_ms + self.phone.system_err_ms(t))),
                 "network_utc_ms": None, "sntp_utc_ms": None, "sntp_rtt_ms": None, "reason": reason}
            if self.phone.network:
                r["network_utc_ms"] = int(round(true_ms + self.phone.network_err_ms))
            if self.phone.sntp_rtt_ms is not None:
                rtt = self.phone.sntp_rtt_ms * (1 + self.phone.rng.random())
                r["sntp_utc_ms"] = true_ms + int(round(self.phone.rng.uniform(-rtt / 2, rtt / 2) * 0.2))
                r["sntp_rtt_ms"] = round(rtt, 1)
            self.records.append(r)
        for ts_, d in self.phone.steps:
            if t0 <= ts_ <= t1:
                el = int(self.phone.elapsed_ns(ts_ + 0.001))
                self.records.append({"type": "clock_ref", "phone_boot": self.phone_boot,
                                     "elapsed_ns": el, "read_span_ns": 4000,
                                     "utc_ms": int(self.phone.true_utc_ns(ts_ + 0.001) // MS
                                                   + round(self.phone.system_err_ms(ts_ + 0.001))),
                                     "reason": "time_set"})

    def listen(self, cam: SimCamera, t0: float, t1: float, *, rate_hz: float = 4.0,
               clock: Optional[SimCamera] = None, timebase_is_master: bool = False,
               take_min: Optional[int] = None, take_max: Optional[int] = None,
               group_id: int = 0, role: str = "unpaired", log_wrap_k: int = 0,
               reset_reason: str = "first") -> int:
        """Hear ``cam`` from t0 to t1; the advertised clock is ``clock`` (the kit
        master for a slave in the master timebase). ``log_wrap_k`` subtracts
        k*2^32 ms from the logged camera time (phone unwrapped from the raw u32)."""
        clk = clock or cam
        n = int((t1 - t0) * rate_hz)
        t = t0 + np.arange(n) / rate_hz + self.phone.rng.uniform(0, 1.0 / rate_hz, n)
        dev_ms = clk.device_ns(t) // MS - log_wrap_k * TWO32_MS
        rx = self.phone.elapsed_ns(t) + self.phone.latency_ns(n)
        x, y, c = bucket_minima(dev_ms, rx)
        self._seg += 1
        sid = self._seg
        live_n = min(24, len(x))
        ref, icpt, skew, res = fit_line_like_sdk(list(x[-live_n:]), list(y[-live_n:])) \
            if live_n >= 2 else (int(x[-1]), int(y[-1]), 0.0, 0.0)
        seg = {"type": "segment", "id": sid, "unit_id": cam.unit_id, "phone_boot": self.phone_boot,
               "boot_nonce": clk.boot_nonce if clock is None else cam.boot_nonce,
               "timebase_is_master": timebase_is_master, "group_id": group_id, "role": role,
               "reset_reason": reset_reason,
               "first_device_ms": int(dev_ms[0]), "last_device_ms": int(dev_ms[-1]),
               "first_rx_elapsed_ns": int(rx[0]), "last_rx_elapsed_ns": int(rx[-1]),
               "samples": n, "take_min": take_min, "take_max": take_max,
               "rec_count_min": None, "rec_count_max": None,
               "live_fit": {"ref_device_ns": ref * MS, "offset_at_ref_ns": icpt, "skew_ppm": skew * 1e6,
                            "residual_ms": res, "buckets": live_n},
               "buckets": {"device_ms": [int(v) for v in x], "min_offset_ns": [int(v) for v in y],
                           "n": [int(v) for v in c]}}
        self.segments.append(seg)
        self.samples[sid] = (dev_ms, rx)
        self.units.setdefault(cam.unit_id, {"type": "unit", "unit_id": cam.unit_id, "label": None,
                                            "group_id": group_id, "last_role": role,
                                            "last_address": "D2:54:00:00:00:01"})
        return sid

    def event(self, cam: SimCamera, kind: str, take_number: int, device_ms: int, seg: int,
              t_detect: float, log_wrap_k: int = 0):
        self._ev += 1
        self.records.append({
            "type": "event", "id": self._ev, "segment": seg, "unit_id": cam.unit_id, "kind": kind,
            "take_number": take_number, "boot_nonce": cam.boot_nonce, "event_seq": self._ev,
            "device_ms": int(device_ms) - log_wrap_k * TWO32_MS, "missed_edges": 0,
            "phone_boot": self.phone_boot, "detected_elapsed_ns": int(self.phone.elapsed_ns(t_detect)),
            "utc_ms_live": None})

    def take_events(self, cam: SimCamera, take: "SimTake", seg: int, log_wrap_k: int = 0,
                    old_firmware: bool = False, stop: bool = True):
        first = take.raw_ns[0] if old_firmware else take.sof_ns[0]
        last = take.raw_ns[-1] if old_firmware else take.sof_ns[-1]
        if take.adv_clock_ns is not None:
            first = take.adv_clock_ns[0] + (take.raw_ns[0] - take.sof_ns[0] if old_firmware else 0)
            last = take.adv_clock_ns[-1] + (take.raw_ns[-1] - take.sof_ns[-1] if old_firmware else 0)
        self.event(cam, "started", take.take_number, first // MS, seg, take.t[0] + 0.5, log_wrap_k)
        if stop:
            self.event(cam, "stopped", take.take_number, last // MS, seg, take.t[-1] + 0.5, log_wrap_k)

    def all_records(self, header_extra: Optional[dict] = None) -> List[dict]:
        hdr = {"type": "header", "format": "trinet-wireless-log", "version": 2,
               "store_id": self.store_id, "sdk_version": "0.5.3", "device_model": "SimPhone",
               "exported_utc_ms": UTC0_MS + 86_400_000, "exported_elapsed_ns": 0, "bucket_ms": 5000,
               "scope": {"units": None, "groups": None, "from_utc_ms": None, "to_utc_ms": None}}
        hdr.update(header_extra or {})
        body = [{"type": "phone_boot", "id": self.phone_boot, "boot_count": 412,
                 "first_elapsed_ns": self.phone.elapsed0_ns, "first_utc_ms": UTC0_MS}]
        body += list(self.units.values())
        body += [r for r in self.records if r["type"] == "clock_ref"]
        body += self.segments
        body += [r for r in self.records if r["type"] != "clock_ref"]
        return [hdr] + body + [{"type": "end", "records": len(body) + 1}]   # header counts

    def write_jsonl(self, path, gz: bool = False, truncate: bool = False) -> Path:
        recs = self.all_records()
        if truncate:
            recs = recs[:-1]
        text = "\n".join(json.dumps(r) for r in recs) + "\n"
        if truncate:
            text = text + '{"type":"event","id":99'       # a torn last line
        p = Path(path)
        if gz:
            p.write_bytes(gzip.compress(text.encode()))
        else:
            p.write_text(text)
        return p

    def write_v1(self, path) -> Path:
        units = []
        for uid in self.units:
            fits = []
            for s in self.segments:
                if s["unit_id"] != uid:
                    continue
                lf = s["live_fit"]
                fits.append({"current": True, "boot_nonce": s["boot_nonce"],
                             "timebase_is_master": s["timebase_is_master"],
                             "a_ns": 0.0, "b": 1.0, **{k: lf[k] for k in
                                                        ("ref_device_ns", "offset_at_ref_ns", "skew_ppm",
                                                         "residual_ms", "buckets")},
                             "samples": s["samples"], "first_device_ms": s["first_device_ms"],
                             "last_device_ms": s["last_device_ms"]})
            units.append({"unit_id": uid, "address": "D2:54:00:00:00:01", "boot_nonce": fits[-1]["boot_nonce"],
                          "role": "unpaired", "group_low": 0, "timebase_is_master": False,
                          "recording": False, "take_number": 0, "recordings_on_card": 0,
                          "last_seen_elapsed_ns": 0, "fits": fits})
        refs = [r for r in self.records if r["type"] == "clock_ref"]
        ref = refs[-1]
        evs = [{"type": r["kind"], "unit_id": r["unit_id"], "take_number": r["take_number"],
                "boot_nonce": r["boot_nonce"], "event_seq": r["event_seq"], "device_ms": r["device_ms"],
                "missed_edges": 0, "detected_elapsed_ns": r["detected_elapsed_ns"],
                "utc_ms_at_detection": None, "utc_ms": None}
               for r in self.records if r["type"] == "event"]
        doc = {"format": "trinet-wireless-status-log", "version": 1, "exported_utc_ms": ref["utc_ms"],
               "phone_reference": {"elapsed_realtime_ns": ref["elapsed_ns"], "utc_ms": ref["utc_ms"]},
               "how_to_apply": "...", "units": units, "events": evs}
        p = Path(path)
        p.write_text(json.dumps(doc, indent=1))
        return p


# ---------------------------------------------------------------------------
#  Recordings
# ---------------------------------------------------------------------------

@dataclass
class SimTake:
    take_number: int
    t: np.ndarray                  # phone time of each frame's .vts instant
    sof_ns: np.ndarray             # .vts timestamps (camera's own clock)
    raw_ns: np.ndarray             # raw start-of-frame (what old firmware advertised)
    true_utc_ns: np.ndarray        # truth for each .vts instant
    global_ns: Optional[np.ndarray] = None       # kit master clock (slaves)
    adv_clock_ns: Optional[np.ndarray] = None    # the clock the edge is advertised in
    exposure_us: int = 8000
    readout_us: int = 20000


def make_take(phone: SimPhone, cam: SimCamera, take_number: int, t_start: float, dur_s: float,
              fps: float = 30.0, exposure_us: int = 8000, readout_us: int = 20000,
              master: Optional[SimCamera] = None) -> SimTake:
    n = max(2, int(dur_s * fps))
    t = t_start + np.arange(n) / fps
    sof = cam.device_ns(t)
    shift = -exposure_us * 1000 // 2 + readout_us * 1000 // 2        # frame-centred mid-exposure
    raw = sof - shift
    glb = master.device_ns(t) if master is not None else None
    return SimTake(take_number, t, sof, raw, phone.true_utc_ns(t), glb,
                   glb if master is not None else None, exposure_us, readout_us)


def write_vts(path, sof_ns, *, version: int = 4, fps_milli: int = 30000, exposure_us: int = 8000,
              readout_us: int = 20000, centered: bool = True, mid_exposure: bool = True,
              sync: Tuple[int, int, int, int] = (0, 0, 0, 0), frame_offset_ns=None,
              extra_flags=None) -> Path:
    sof_ns = np.asarray(sof_ns, dtype=np.int64)
    reserved = struct.pack("<qiHH", *sync) if version >= 3 else b"\x00" * 16
    out = bytearray(struct.pack("<8sII16s", b"TRIVTS01", version, fps_milli, reserved))
    fl = 0
    if mid_exposure:
        fl |= TIMING_MID_EXPOSURE | TIMING_EXPOSURE_VALID | TIMING_READOUT_VALID
        if centered:
            fl |= TIMING_FRAME_CENTERED
    for i, s in enumerate(sof_ns):
        f = fl | (int(extra_flags[i]) if extra_flags is not None else 0)
        if version >= 5:
            fo = int(frame_offset_ns[i]) if frame_offset_ns is not None else 0
            out += struct.pack(VTS_ENTRY_FMT_V5, i, int(s), i, int(s) // 1000, exposure_us, f,
                               readout_us, fo)
        elif version == 4:
            out += struct.pack(VTS_ENTRY_FMT_V4, i, int(s), i, int(s) // 1000, exposure_us, f, readout_us)
        else:
            out += struct.pack("<IQIQ", i, int(s), i, int(s) // 1000)
    p = Path(path)
    p.write_bytes(bytes(out))
    return p


def write_imu_header(path, device_id_hex: str, version: int = 5) -> Path:
    did = bytes.fromhex(device_id_hex)[:16].ljust(16, b"\x00")
    hdr = struct.pack("<8sIIHHQQI24s", b"TRIMU001", version, 400, 1, 2, 0, 0, 0x02, did + b"\x00" * 8)
    p = Path(path)
    p.write_bytes(hdr)
    return p


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def write_tmfm_mp4(path, meta: Optional[dict], stale_meta: Optional[dict] = None,
                   mdat_bytes: int = 4096) -> Path:
    """A minimal MP4 carrying only moov/udta/tmfm. With ``stale_meta`` an older
    moov precedes the mdat and the real one is appended at EOF (as flatten does)."""
    def moov(m):
        udta = _box(b"udta", _box(b"tmfm", json.dumps(m).encode()) if m is not None else b"")
        return _box(b"moov", _box(b"mvhd", b"\x00" * 100) + udta)
    data = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
    if stale_meta is not None:
        data += moov(stale_meta)
    data += _box(b"mdat", b"\x00" * mdat_bytes)
    data += moov(meta)
    p = Path(path)
    p.write_bytes(data)
    return p


def write_stereo_take(folder, take: SimTake, cam: SimCamera, *, boot_id: Optional[str] = "auto",
                      version: int = 4, prefix: str = "", eyes=("L", "R"), vts_sidecar: bool = True,
                      master: bool = False, sync_quality_us: int = 150,
                      boot_id_source: str = "live") -> Path:
    """Write ``[prefix]takeNNNN_{L,R}.{mp4,vts}`` + ``takeNNNN.imu``. Returns the take path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}take{take.take_number:04d}"
    meta = {"tmf_schema": 1, "device_id": cam.device_id, "codec": "h265"}
    if boot_id == "auto":
        boot_id = cam.boot_id
    if boot_id:
        meta["boot_id"] = boot_id
        meta["boot_id_source"] = boot_id_source
    fo = None
    sync = (0, 0, 0, 0)
    if take.global_ns is not None:
        fo = take.global_ns - take.sof_ns
        sync = (int(fo[-1]), 0, sync_quality_us, VTS_SYNC_FLAG_SYNCED)
        version = 5
    elif master:
        sync = (0, 0, 0, VTS_SYNC_FLAG_SYNCED | VTS_SYNC_FLAG_IS_MASTER)
    for eye in eyes:
        write_tmfm_mp4(folder / f"{base}_{eye}.mp4", dict(meta, eye=eye))
        if vts_sidecar:
            write_vts(folder / f"{base}_{eye}.vts", take.sof_ns, version=version,
                      exposure_us=take.exposure_us, readout_us=take.readout_us, sync=sync,
                      frame_offset_ns=fo)
    write_imu_header(folder / f"{base}.imu", cam.device_id)
    return folder / base


def write_mono_take(folder, take: SimTake, cam: SimCamera, base: str, *,
                    boot_id: Optional[str] = "auto", parts: int = 0) -> Path:
    """Mono layout ``<base>.{mp4,vts,imu,json}`` or chunked ``<base>/partNNN.*``."""
    folder = Path(folder)
    meta = {"tmf_schema": 1, "device_id": cam.device_id, "codec": "h264"}
    if boot_id == "auto":
        boot_id = cam.boot_id
    if boot_id:
        meta["boot_id"] = boot_id
        meta["boot_id_source"] = "live"
    if parts:
        d = folder / base
        d.mkdir(parents=True, exist_ok=True)
        chunks = np.array_split(np.arange(len(take.sof_ns)), parts)
        for i, idx in enumerate(chunks, 1):
            write_tmfm_mp4(d / f"part{i:03d}.mp4", meta)
            write_vts(d / f"part{i:03d}.vts", take.sof_ns[idx], exposure_us=take.exposure_us,
                      readout_us=take.readout_us)
            write_imu_header(d / f"part{i:03d}.imu", cam.device_id)
        return folder / base
    folder.mkdir(parents=True, exist_ok=True)
    write_tmfm_mp4(folder / f"{base}.mp4", meta)
    write_vts(folder / f"{base}.vts", take.sof_ns, exposure_us=take.exposure_us,
              readout_us=take.readout_us)
    write_imu_header(folder / f"{base}.imu", cam.device_id)
    (folder / f"{base}.json").write_text(json.dumps({
        "format": "trinet-recording-meta/1", "session": 0, "group": 0, "role": "unpaired",
        "device_id": cam.device_id, "device_tag": cam.device_id[:8], "segment": 1}))
    return folder / base


# ---------------------------------------------------------------------------
#  A ready-made scenario
# ---------------------------------------------------------------------------

def basic_scenario(tmp: Path, *, seed: int = 7, boot_id: bool = True, old_firmware: bool = False,
                   events: bool = True, hours: float = 1.0, take_at_s: float = 1200.0,
                   take_dur_s: float = 20.0, phone: Optional[SimPhone] = None,
                   cam: Optional[SimCamera] = None) -> dict:
    """One camera heard for ``hours``, one stereo take in the middle."""
    phone = phone or SimPhone(seed=seed)
    cam = cam or SimCamera()
    lb = LogBuilder(phone)
    T = hours * 3600.0
    lb.clock_refs(0.0, T)
    seg = lb.listen(cam, 0.0, T, take_min=1, take_max=1)
    take = make_take(phone, cam, 1, take_at_s, take_dur_s)
    if events:
        lb.take_events(cam, take, seg, old_firmware=old_firmware)
    card = tmp / "card" / "Trinet" / "recording"
    tp = write_stereo_take(card, take, cam, boot_id="auto" if boot_id else None,
                           version=4)
    log = lb.write_jsonl(tmp / "phone.jsonl")
    return {"phone": phone, "cam": cam, "lb": lb, "take": take, "take_path": tp, "log": log,
            "card": tmp / "card", "seg": seg}

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""TMF (Trinet Metadata Format) writer: the host-side twin of the camera's
``tmf_write.c``. It folds ``.imu`` / ``.vts`` / ``.tel`` sidecar images into the
payloads of the MP4's timed-metadata track (sample entry ``tmfd``) and builds
the ``udta/tmfm`` recording-summary JSON, byte for byte as the camera does
(see docs/data_formats.md, "Embedded metadata track (TMF)"; the reader is
``trinet_tools.tmf``).

    from trinet_tools.tmf_write import encode, build_meta_json
    chunks = encode(imu_bytes, vts_bytes, tel_bytes)    # [TmfChunk(data, dur_ms)]
    meta = build_meta_json(MetaStrings(device_id="..."), imu_bytes, vts_bytes, tel_bytes)

Layout, as on the camera:
  - one track sample = one ``DEVC`` KLV container covering 1 s, chunk walls at
    fixed 1 s steps from t0 = the earliest of the first IMU sample, the first
    (non-zero) frame time and the first (non-zero) telemetry record;
  - the raw sidecar headers (``IHDR`` / ``VHDR`` / ``THDR``) ride in chunk 0;
  - IMU columns as typed streams (TSNS ACCL GYRO MAGN TMPC QUAT LACC + MAGA on
    ``.imu`` v5+ or FSYN on v3/v4); frames as raw ``.vts`` entries (``TFRM``,
    struct size = the file's entry size: 12/24/36/44); telemetry as raw ``.tel``
    records (``TSOC``);
  - per-chunk durations in milliseconds (the track timescale is 1000).

Pure Python + numpy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

IMU_MAGIC = b"TRIMU001"
VTS_MAGIC = b"TRIVTS01"
TEL_MAGIC = b"TRTEL01\x00"
IMU_HEADER_SIZE = 64
IMU_SAMPLE_SIZE = 80          # the TMF columns are the 80-byte (v3+) sample
VTS_HEADER_SIZE = 32
TEL_HEADER_SIZE = 32
TEL_RECORD_SIZE = 24
CHUNK_NS = 1_000_000_000
MAX_GAPS = 64


def vts_entry_size(version: int) -> int:
    """Entry size of a ``.vts`` version (as the camera reads it)."""
    if version == 1:
        return 12
    if version in (2, 3):
        return 24
    if version == 4:
        return 36
    return 44


@dataclass
class TmfChunk:
    data: bytes
    dur_ms: int


# --------------------------------------------------------------------------- KLV
def _pad4(b: bytearray) -> None:
    while len(b) & 3:
        b.append(0)


def _klv_hdr(b: bytearray, key: bytes, typ: int, ssize: int, repeat: int) -> None:
    b += key
    b += bytes((typ, ssize))
    b += struct.pack("<H", repeat & 0xFFFF)


def _klv_item(b: bytearray, key: bytes, typ: int, ssize: int, repeat: int, payload) -> None:
    _klv_hdr(b, key, typ, ssize, repeat)
    b += payload
    _pad4(b)


def _klv_cstr(b: bytearray, key: bytes, s: str) -> None:
    raw = s.encode()[:0xFFFF]
    _klv_item(b, key, ord("c"), 1, len(raw), raw)


def _klv_u32(b: bytearray, key: bytes, v: int) -> None:
    _klv_item(b, key, ord("L"), 4, 1, struct.pack("<I", v))


def _klv_u64(b: bytearray, key: bytes, v: int) -> None:
    _klv_item(b, key, ord("Q"), 8, 1, struct.pack("<Q", v))


def _klv_open(b: bytearray, key: bytes) -> int:
    _klv_hdr(b, key, 0, 4, 0)
    return len(b)


def _klv_close(b: bytearray, start: int) -> None:
    words = (len(b) - start) // 4
    if words > 0xFFFF:
        raise ValueError("TMF chunk too large (more than 256 KiB in one second)")
    struct.pack_into("<H", b, start - 2, words)


def _emit_strm(b: bytearray, key: bytes, name: str, unit: str | None, typ: str, ssize: int,
               first_ts: int, data: bytes) -> None:
    if not data:
        return
    s = _klv_open(b, b"STRM")
    _klv_cstr(b, b"STNM", name)
    if unit:
        _klv_cstr(b, b"SIUN", unit)
    _klv_u64(b, b"STMP", first_ts)
    repeat = min(len(data) // ssize, 0xFFFF)
    _klv_item(b, key, ord(typ), ssize, repeat, data[:repeat * ssize])
    _klv_close(b, s)


# --------------------------------------------------------------------------- views
@dataclass
class _Imu:
    header: bytes
    version: int
    rate_hz: int
    rows: np.ndarray            # (n, 80) uint8
    ts: np.ndarray              # (n,) uint64


@dataclass
class _Vts:
    header: bytes
    version: int
    frame_rate_milli: int
    entry_size: int
    entries: bytes
    ts: np.ndarray              # (n,) uint64

    @property
    def n(self) -> int:
        return len(self.ts)


@dataclass
class _Tel:
    header: bytes
    recs: bytes
    ts: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ts)


def _imu_view(buf: bytes | None) -> _Imu | None:
    if not buf or len(buf) < IMU_HEADER_SIZE or buf[:8] != IMU_MAGIC:
        return None
    version, rate = struct.unpack_from("<II", buf, 8)
    n = (len(buf) - IMU_HEADER_SIZE) // IMU_SAMPLE_SIZE
    rows = np.frombuffer(buf, np.uint8, n * IMU_SAMPLE_SIZE, IMU_HEADER_SIZE).reshape(n, IMU_SAMPLE_SIZE)
    ts = rows[:, :8].copy().view("<u8").reshape(n)
    return _Imu(bytes(buf[:IMU_HEADER_SIZE]), version, rate, rows, ts)


def _vts_view(buf: bytes | None) -> _Vts | None:
    if not buf or len(buf) < VTS_HEADER_SIZE or buf[:8] != VTS_MAGIC:
        return None
    version, fr = struct.unpack_from("<II", buf, 8)
    es = vts_entry_size(version)
    n = (len(buf) - VTS_HEADER_SIZE) // es
    body = bytes(buf[VTS_HEADER_SIZE:VTS_HEADER_SIZE + n * es])
    ts = np.frombuffer(body, np.uint8).reshape(n, es)[:, 4:12].copy().view("<u8").reshape(n) if n else \
        np.zeros(0, np.uint64)
    return _Vts(bytes(buf[:VTS_HEADER_SIZE]), version, fr, es, body, ts)


def _tel_view(buf: bytes | None) -> _Tel | None:
    if not buf or len(buf) < TEL_HEADER_SIZE or buf[:8] != TEL_MAGIC:
        return None
    n = (len(buf) - TEL_HEADER_SIZE) // TEL_RECORD_SIZE
    recs = bytes(buf[TEL_HEADER_SIZE:TEL_HEADER_SIZE + n * TEL_RECORD_SIZE])
    ts = np.frombuffer(recs, np.uint8).reshape(n, TEL_RECORD_SIZE)[:, :8].copy().view("<u8").reshape(n) if n \
        else np.zeros(0, np.uint64)
    return _Tel(bytes(buf[:TEL_HEADER_SIZE]), recs, ts)


def _cursor_ends(ts: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """End index of each chunk for the camera's cursor walk (``while ts[i] < wall: i++``),
    exact for non-monotonic timestamps too: every element the walk consumed is below
    the current wall, so the stop index is the first position where the running
    maximum reaches it."""
    if not len(ts):
        return np.zeros(len(walls), np.int64)
    run = np.maximum.accumulate(ts.astype(np.uint64))
    return np.searchsorted(run, walls.astype(np.uint64), side="left").astype(np.int64)


# --------------------------------------------------------------------------- encode
_IMU_STREAMS = (  # key, name, unit, type, struct size, byte slice of the 80-byte row
    (b"TSNS", "sample timestamps", "ns", "Q", 8, slice(0, 8)),
    (b"ACCL", "accelerometer", "m/s2", "f", 12, slice(8, 20)),
    (b"GYRO", "gyroscope", "rad/s", "f", 12, slice(20, 32)),
    (b"MAGN", "magnetometer", "uT", "f", 12, slice(32, 44)),
    (b"TMPC", "IMU temperature", "degC", "f", 4, slice(44, 48)),
    (b"QUAT", "orientation xyzw", None, "f", 16, slice(48, 64)),
    (b"LACC", "linear acceleration", "m/s2", "f", 12, slice(64, 76)),
)


def encode(imu: bytes | None, vts: bytes | None, tel: bytes | None = None) -> list[TmfChunk]:
    """TMF track samples (``DEVC`` payloads + durations in ms) from whole sidecar images.
    Any input may be missing; at least one of ``imu`` / ``vts`` must hold data."""
    iv = _imu_view(imu)
    vv = _vts_view(vts)
    tv = _tel_view(tel)
    if iv is not None and iv.version < 3:
        iv = None                # v1/v2 samples are not 80 bytes; the camera never folds them
    have_imu = iv is not None and len(iv.ts) > 0
    have_vts = vv is not None and vv.n > 0
    have_tel = tv is not None and tv.n > 0
    if not have_imu and not have_vts:
        raise ValueError("TMF needs IMU samples or frame times")

    big = (1 << 64) - 1
    t0, t1 = big, 0
    if have_imu:
        t0 = min(t0, int(iv.ts[0]))
        t1 = max(t1, int(iv.ts[-1]))
    if have_vts:
        f0, f1 = int(vv.ts[0]), int(vv.ts[-1])
        if f0 and f0 < t0:
            t0 = f0
        t1 = max(t1, f1)
    if have_tel:
        e0, e1 = int(tv.ts[0]), int(tv.ts[-1])
        if e0 and e0 < t0:
            t0 = e0
        t1 = max(t1, e1)
    if t0 == big or t1 < t0:
        raise ValueError("TMF: no usable timestamps")

    nchunks = (t1 - t0) // CHUNK_NS + 1
    walls = np.array([t0 + (c + 1) * CHUNK_NS for c in range(nchunks)], dtype=np.uint64)
    imu_end = _cursor_ends(iv.ts, walls) if have_imu else None
    vts_end = _cursor_ends(vv.ts, walls) if have_vts else None
    tel_end = _cursor_ends(tv.ts, walls) if have_tel else None
    if have_imu:
        cols = {key: np.ascontiguousarray(iv.rows[:, sl]) for key, _n, _u, _t, _s, sl in _IMU_STREAMS}
        cols[b"MAGA"] = np.ascontiguousarray(iv.rows[:, 76:80])
        tail = (b"MAGA", "mag age") if iv.version >= 5 else (b"FSYN", "fsync delay")

    out: list[TmfChunk] = []
    si = fi = ei = 0
    for ci in range(nchunks):
        b = bytearray()
        devc = _klv_open(b, b"DEVC")
        _klv_cstr(b, b"DVNM", "Trinet")
        _klv_u32(b, b"DVID", 1)
        _klv_u64(b, b"TICK", t0 + ci * CHUNK_NS)
        if ci == 0:
            if have_imu:
                _klv_item(b, b"IHDR", ord("B"), 1, IMU_HEADER_SIZE, iv.header)
            if have_vts:
                _klv_item(b, b"VHDR", ord("B"), 1, VTS_HEADER_SIZE, vv.header)
            if have_tel:
                _klv_item(b, b"THDR", ord("B"), 1, TEL_HEADER_SIZE, tv.header)
        if have_imu:
            first, si = si, max(si, int(imu_end[ci]))
            if si > first:
                ts0 = int(iv.ts[first])
                for key, name, unit, typ, ssize, _sl in _IMU_STREAMS:
                    _emit_strm(b, key, name, unit, typ, ssize, ts0, cols[key][first:si].tobytes())
                _emit_strm(b, tail[0], tail[1], "us", "f", 4, ts0, cols[b"MAGA"][first:si].tobytes())
        if have_vts:
            first, fi = fi, max(fi, int(vts_end[ci]))
            if fi > first:
                s = _klv_open(b, b"STRM")
                _klv_cstr(b, b"STNM", "frame timing (vts entries)")
                _klv_u64(b, b"STMP", int(vv.ts[first]))
                cnt = min(fi - first, 0xFFFF)
                es = vv.entry_size
                _klv_item(b, b"TFRM", ord("B"), es, cnt, vv.entries[first * es:(first + cnt) * es])
                _klv_close(b, s)
        if have_tel:
            first, ei = ei, max(ei, int(tel_end[ci]))
            if ei > first:
                s = _klv_open(b, b"STRM")
                _klv_cstr(b, b"STNM", "SoC thermal telemetry (tel records)")
                _klv_u64(b, b"STMP", int(tv.ts[first]))
                cnt = ei - first
                _klv_item(b, b"TSOC", ord("B"), TEL_RECORD_SIZE, cnt,
                          tv.recs[first * TEL_RECORD_SIZE:ei * TEL_RECORD_SIZE])
                _klv_close(b, s)
        _klv_close(b, devc)

        cstart = t0 + ci * CHUNK_NS
        cend = int(walls[ci]) if ci + 1 < nchunks else t1 + 1
        dur = (cend - cstart + 500_000) // 1_000_000
        out.append(TmfChunk(bytes(b), max(int(dur), 1)))
    return out


# --------------------------------------------------------------------------- tmfm
@dataclass
class MetaStrings:
    """Identity strings of the ``tmfm`` JSON (None = omitted), in the camera's order."""
    device_id: str | None = None
    fw_version: str | None = None
    hw_generation: str | None = None
    eye: str | None = None
    pair_file: str | None = None
    codec: str | None = None
    boot_id: str | None = None
    boot_id_source: str | None = None
    source: str = "sd"
    schema: int = 2
    generator: str = "trinet-mp4-finalize"
    extra: dict = field(default_factory=dict)   # more keys (JSON values), written before "generator"


def _json_str(key: str, val: str) -> str:
    out = []
    for ch in val:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) >= 0x20:
            out.append(ch)
    return f'"{key}":"{"".join(out)}",'


def drops_summary(vts: bytes | None) -> dict | None:
    """The camera's frame-continuity summary of a ``.vts`` image (the ``drops`` block):
    gaps of more than 1.5x the median frame interval, or encoder sequence jumps."""
    text = _drops_json(_vts_view(vts))
    if text is None:
        return None
    import json
    return json.loads("{" + text.rstrip(",") + "}")["drops"]


def _drops_json(vv: _Vts | None) -> str | None:
    if vv is None or vv.n < 3 or vv.frame_rate_milli <= 0:
        return None
    n = vv.n
    nominal_ms = 1000000.0 / float(vv.frame_rate_milli)
    have_seq = vv.version >= 2 and vv.entry_size >= 24
    ts = [int(x) for x in vv.ts]

    want = min(n - 1, 4096)
    step = max((n - 1) // want, 1)
    sample = []
    i = 1
    while i < n and len(sample) < want:
        a, c = ts[i - 1], ts[i]
        if c > a:
            sample.append(float(c - a) / 1.0e6)
        i += step
    sample.sort()
    med_ms = sample[len(sample) // 2] if sample else 0.0
    if med_ms <= 0.0:
        med_ms = nominal_ms

    es = vv.entry_size
    seqs = [struct.unpack_from("<I", vv.entries, k * es + 12)[0] for k in range(n)] if have_seq else None
    gap_count = listed = 0
    dropped = 0
    gaps = []
    prev_seq = seqs[0] if have_seq else 0
    for i in range(1, n):
        ta, tc = ts[i - 1], ts[i]
        seq = seqs[i] if have_seq else 0
        monotonic = tc > ta
        d_ms = float(tc - ta) / 1e6 if monotonic else 0.0
        timing_gap = monotonic and d_ms > med_ms * 1.5
        seq_gap = have_seq and seq > ((prev_seq + 1) & 0xFFFFFFFF)
        if timing_gap or seq_gap:
            miss = 0
            if timing_gap:
                k = int(d_ms / med_ms + 0.5) - 1
                if k > 0:
                    miss = k
            if seq_gap:
                sk = (seq - prev_seq - 1) & 0xFFFFFFFF
                miss = max(miss, sk)
            gap_count += 1
            dropped += miss
            if listed < MAX_GAPS:
                gaps.append('{"after_frame":%d,"ms":%.1f}' % (i - 1, d_ms))
                listed += 1
        prev_seq = seq
    return ('"drops":{"nominal_ms":%.2f,"measured_ms":%.3f,"recorded":%d,"dropped":%d,"expected":%d,'
            '"gap_count":%d,"gaps":[' % (nominal_ms, med_ms, n, dropped, n + dropped, gap_count)
            + ",".join(gaps) + "]},")


def build_meta_json(s: MetaStrings | None, imu: bytes | None, vts: bytes | None,
                    tel: bytes | None = None) -> bytes:
    """The ``udta/tmfm`` JSON as the camera writes it (``tmf_build_meta_json``)."""
    import json
    s = s or MetaStrings()
    iv = _imu_view(imu)
    vv = _vts_view(vts)
    tv = _tel_view(tel)
    have_tel = tv is not None and tv.n > 0
    parts = ['{"tmf_schema":%d,' % s.schema, _json_str("source", s.source)]
    for key in ("device_id", "fw_version", "hw_generation", "eye", "pair_file", "codec", "boot_id"):
        v = getattr(s, key)
        if v:
            parts.append(_json_str(key, v))
    if s.boot_id and s.boot_id_source:
        parts.append(_json_str("boot_id_source", s.boot_id_source))
    if iv is not None:
        parts.append('"imu_version":%d,"imu_rate_hz":%d,' % (iv.version, iv.rate_hz))
    if vv is not None:
        parts.append('"vts_version":%d,' % vv.version)
        if vv.version >= 3:
            off, skew, qual, flags = struct.unpack_from("<qiHH", vv.header, 16)
            parts.append('"sync":{"offset_ns":%d,"skew_ppb":%d,"quality_us":%d,"flags":%d},'
                         % (off, skew, qual, flags))
        drops = _drops_json(vv)
        if drops:
            parts.append(drops)
    if have_tel:
        mc = np.frombuffer(tv.recs, np.uint8).reshape(tv.n, TEL_RECORD_SIZE)[:, 8:12].copy().view("<i4")
        parts.append('"thermal":{"min_c":%.1f,"max_c":%.1f,"samples":%d},'
                     % (int(mc.min()) / 1000.0, int(mc.max()) / 1000.0, tv.n))
    for key, val in s.extra.items():
        parts.append(json.dumps(key) + ":" + json.dumps(val, separators=(",", ":")) + ",")
    parts.append(_json_str("generator", s.generator)[:-1] + "}")
    return "".join(parts).encode()


def meta_strings_from(meta: dict | None) -> MetaStrings:
    """The identity strings of an existing ``tmfm`` dict (to rebuild it after an edit)."""
    meta = meta or {}
    known = {"tmf_schema", "source", "device_id", "fw_version", "hw_generation", "eye", "pair_file",
             "codec", "boot_id", "boot_id_source", "imu_version", "imu_rate_hz", "vts_version", "sync",
             "drops", "thermal", "generator"}
    return MetaStrings(
        device_id=meta.get("device_id"), fw_version=meta.get("fw_version"),
        hw_generation=meta.get("hw_generation"), eye=meta.get("eye"), pair_file=meta.get("pair_file"),
        codec=meta.get("codec"), boot_id=meta.get("boot_id"), boot_id_source=meta.get("boot_id_source"),
        source=str(meta.get("source") or "sd"), schema=int(meta.get("tmf_schema") or 2),
        generator=str(meta.get("generator") or "trinet-mp4-finalize"),
        extra={k: v for k, v in meta.items() if k not in known})

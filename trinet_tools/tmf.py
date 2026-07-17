#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""TMF (Trinet Metadata Format) reader — extract embedded metadata from a
Trinet MP4 (see docs/data_formats.md, "Embedded metadata track (TMF)").

A finalized Trinet recording carries:
  - a timed metadata track (sample entry 'tmfd', handler "Trinet TMF") whose
    samples are KLV 'DEVC' payloads: IMU sensor streams + frame timing,
  - moov/udta boxes: 'tmfm' (take meta JSON incl. frame-drop summary) and
    'tmfc' (the raw TBLC calibration blob).

The KLV columns are exactly the .imu sample fields and the raw .vts entries,
plus the raw sidecar headers (IHDR/VHDR), so this module reconstructs
byte-identical `.imu` / `.vts` sidecars.

CLI:
  python3 -m trinet_tools.tmf info    take0001_L.mp4
  python3 -m trinet_tools.tmf extract take0001_L.mp4 [--out DIR]

Library:
  from trinet_tools.tmf import read_tmf
  t = read_tmf("take0001_L.mp4")
  t.meta            # dict from tmfm JSON (or None)
  t.calib_blob      # bytes (TBLC) or None
  t.imu_bytes()     # full .imu file image (header + samples) or None
  t.vts_bytes()     # full .vts file image or None
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"dinf"}

# IMU sample column layout (80 B rows) — mirrors imu_capture.h imu_sample_t.
_IMU_COLS = [  # (fourcc, byte width)
    (b"TSNS", 8), (b"ACCL", 12), (b"GYRO", 12), (b"MAGN", 12),
    (b"TMPC", 4), (b"QUAT", 16), (b"LACC", 12), (b"MAGA", 4), (b"FSYN", 4),
]


def _walk(buf: bytes, start: int, end: int):
    o = start
    while o + 8 <= end:
        (sz,) = struct.unpack_from(">I", buf, o)
        hdr = 8
        if sz == 1:
            (sz,) = struct.unpack_from(">Q", buf, o + 8)
            hdr = 16
        elif sz == 0:
            sz = end - o
        if sz < hdr or o + sz > end:
            break
        yield buf[o + 4:o + 8], o, sz, hdr
        o += sz


def _find(buf: bytes, start: int, end: int, name: bytes):
    for kind, o, sz, hdr in _walk(buf, start, end):
        if kind == name:
            return o, sz, hdr
        if kind in _CONTAINERS:
            r = _find(buf, o + hdr, o + sz, name)
            if r:
                return r
    return None


def _klv_items(buf: bytes, start: int, end: int):
    """Yield (key, type, struct_size, repeat, payload) of one KLV level."""
    o = start
    while o + 8 <= end:
        key = buf[o:o + 4]
        typ, ssize = buf[o + 4], buf[o + 5]
        (repeat,) = struct.unpack_from("<H", buf, o + 6)
        plen = ssize * repeat if typ else repeat * 4
        pad = (4 - plen % 4) % 4
        yield key, typ, ssize, repeat, buf[o + 8:o + 8 + plen]
        o += 8 + plen + pad


@dataclass
class TmfRecording:
    path: str
    meta: dict | None = None
    calib_blob: bytes | None = None
    imu_header: bytes | None = None
    vts_header: bytes | None = None
    tel_header: bytes | None = None
    imu_cols: dict = field(default_factory=dict)   # fourcc -> bytearray
    vts_entries: bytearray = field(default_factory=bytearray)
    vts_entry_size: int = 0
    tel_records: bytearray = field(default_factory=bytearray)
    chunk_count: int = 0

    @property
    def imu_sample_count(self) -> int:
        return len(self.imu_cols.get(b"TSNS", b"")) // 8

    @property
    def vts_entry_count(self) -> int:
        return len(self.vts_entries) // self.vts_entry_size \
            if self.vts_entry_size else 0

    def imu_bytes(self) -> bytes | None:
        """Reconstruct the byte-identical .imu file image."""
        if self.imu_header is None:
            return None
        n = self.imu_sample_count
        out = bytearray(self.imu_header)
        cols = self.imu_cols
        for i in range(n):
            for key, w in _IMU_COLS:
                col = cols.get(key)
                if col is None:
                    if key in (b"MAGA", b"FSYN"):
                        continue        # the other alias carried this slot
                    col = bytes(w * n)  # absent column = zeros
                out += col[i * w:(i + 1) * w]
        return bytes(out)

    def vts_bytes(self) -> bytes | None:
        if self.vts_header is None:
            return None
        return bytes(self.vts_header) + bytes(self.vts_entries)

    @property
    def tel_record_count(self) -> int:
        return len(self.tel_records) // 24

    def tel_bytes(self) -> bytes | None:
        """Reconstruct the .tel (TRTEL01) thermal telemetry sidecar.
        The header's record_count field (offset 16) is refreshed since the
        on-device writer updates it at close."""
        if self.tel_header is None:
            return None
        hdr = bytearray(self.tel_header)
        struct.pack_into("<I", hdr, 16, self.tel_record_count)
        return bytes(hdr) + bytes(self.tel_records)


def read_tmf(path: str | Path) -> TmfRecording:
    data = Path(path).read_bytes()
    rec = TmfRecording(path=str(path))

    # Last moov wins (flatten appends the rebuilt moov at EOF).
    moovs = [(o, sz, hdr) for k, o, sz, hdr in _walk(data, 0, len(data))
             if k == b"moov"]
    if not moovs:
        raise ValueError(f"{path}: no moov box (truncated recording?)")
    mo, msz, mhdr = moovs[-1]

    # --- udta: tmfm / tmfc ---
    udta = _find(data, mo + mhdr, mo + msz, b"udta")
    if udta:
        uo, usz, uhdr = udta
        for kind, o, sz, hdr in _walk(data, uo + uhdr, uo + usz):
            body = data[o + hdr:o + sz]
            if kind == b"tmfm":
                try:
                    rec.meta = json.loads(body)
                except json.JSONDecodeError:
                    rec.meta = None
            elif kind == b"tmfc":
                rec.calib_blob = bytes(body)

    # --- locate the tmfd track ---
    tmfd_trak = None
    for kind, o, sz, hdr in _walk(data, mo + mhdr, mo + msz):
        if kind != b"trak":
            continue
        stsd = _find(data, o + hdr, o + sz, b"stsd")
        if stsd and b"tmfd" in data[stsd[0]:stsd[0] + stsd[1]]:
            tmfd_trak = (o, sz, hdr)
            break
    if not tmfd_trak:
        return rec     # udta-only file (or no TMF at all)

    to, tsz, thdr = tmfd_trak
    stsz = _find(data, to + thdr, to + tsz, b"stsz")
    stco = _find(data, to + thdr, to + tsz, b"stco")
    co64 = _find(data, to + thdr, to + tsz, b"co64")
    if not stsz or not (stco or co64):
        raise ValueError(f"{path}: tmfd track missing sample tables")

    (fixed,) = struct.unpack_from(">I", data, stsz[0] + 12)
    (n,) = struct.unpack_from(">I", data, stsz[0] + 16)
    sizes = [fixed] * n if fixed else [
        struct.unpack_from(">I", data, stsz[0] + 20 + 4 * i)[0]
        for i in range(n)]
    if co64:
        offs = [struct.unpack_from(">Q", data, co64[0] + 16 + 8 * i)[0]
                for i in range(n)]
    else:
        offs = [struct.unpack_from(">I", data, stco[0] + 16 + 4 * i)[0]
                for i in range(n)]

    # --- decode every DEVC payload (recursive absolute-offset KLV walk) ---
    def scan(o: int, end: int):
        while o + 8 <= end:
            key = data[o:o + 4]
            typ, ssize = data[o + 4], data[o + 5]
            (repeat,) = struct.unpack_from("<H", data, o + 6)
            plen = ssize * repeat if typ else repeat * 4
            if key == b"IHDR" and rec.imu_header is None:
                rec.imu_header = data[o + 8:o + 8 + plen]
            elif key == b"VHDR" and rec.vts_header is None:
                rec.vts_header = data[o + 8:o + 8 + plen]
            elif key == b"THDR" and rec.tel_header is None:
                rec.tel_header = data[o + 8:o + 8 + plen]
            elif key == b"TSOC":
                rec.tel_records += data[o + 8:o + 8 + plen]
            elif key == b"TFRM":
                rec.vts_entry_size = ssize
                rec.vts_entries += data[o + 8:o + 8 + plen]
            elif key in dict(_IMU_COLS):
                rec.imu_cols.setdefault(key, bytearray())
                rec.imu_cols[key] += data[o + 8:o + 8 + plen]
            if typ == 0:
                scan(o + 8, o + 8 + plen)
            o += 8 + plen + (4 - plen % 4) % 4

    for off, size in zip(offs, sizes):
        rec.chunk_count += 1
        scan(off, off + size)

    return rec


# ------------------------------------------------------------------ CLI ----
def _cmd_info(path: str) -> int:
    rec = read_tmf(path)
    print(f"{path}")
    print(f"  TMF chunks:   {rec.chunk_count}")
    print(f"  IMU samples:  {rec.imu_sample_count}")
    print(f"  vts entries:  {rec.vts_entry_count} "
          f"(entry size {rec.vts_entry_size})")
    print(f"  tel records:  {rec.tel_record_count}")
    print(f"  calib blob:   {len(rec.calib_blob) if rec.calib_blob else 0} B")
    if rec.meta:
        print("  meta:")
        print("    " + json.dumps(rec.meta, indent=2).replace("\n", "\n    "))
    return 0


def _cmd_extract(path: str, out_dir: str | None) -> int:
    rec = read_tmf(path)
    base = Path(path)
    out = Path(out_dir) if out_dir else base.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = base.stem

    wrote = []
    imu = rec.imu_bytes()
    if imu:
        p = out / f"{stem}.imu"
        p.write_bytes(imu)
        wrote.append(f"{p} ({rec.imu_sample_count} samples)")
    vts = rec.vts_bytes()
    if vts:
        p = out / f"{stem}.vts"
        p.write_bytes(vts)
        wrote.append(f"{p} ({rec.vts_entry_count} frames)")
    tel = rec.tel_bytes()
    if tel:
        p = out / f"{stem}.tel"
        p.write_bytes(tel)
        wrote.append(f"{p} ({rec.tel_record_count} thermal records)")
    if rec.calib_blob:
        p = out / f"{stem}.calib.bin"
        p.write_bytes(rec.calib_blob)
        wrote.append(str(p))
    if rec.meta is not None:
        p = out / f"{stem}.meta.json"
        p.write_text(json.dumps(rec.meta, indent=2))
        wrote.append(str(p))

    if not wrote:
        print(f"{path}: no TMF metadata found", file=sys.stderr)
        return 1
    for w in wrote:
        print(f"wrote {w}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "info":
        return _cmd_info(argv[1])
    if len(argv) >= 2 and argv[0] == "extract":
        out = None
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
        return _cmd_extract(argv[1], out)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

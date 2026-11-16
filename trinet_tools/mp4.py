# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Minimal ISO-BMFF (MP4) reading and writing for stream-copy edits of Trinet
recordings: parse a flat (non-fragmented) movie's tracks into per-sample tables
and write a new movie (moov first) whose samples are byte ranges copied from
the source. No decoding, no re-encoding. Pure Python + numpy.

    m = read_movie("take_L.mp4")
    v = m.track("vide")          # Track: sizes, offsets, durations, sync flags ...
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"dinf", b"edts", b"mvex"}


class Mp4Error(ValueError):
    """The file cannot be edited as asked (plain-language message)."""


# --------------------------------------------------------------------------- reading
def iter_boxes(buf, start: int = 0, end: int | None = None):
    """Yield (type, offset, size, header_len) for the boxes of ``buf[start:end]``."""
    end = len(buf) if end is None else end
    o = start
    while o + 8 <= end:
        (sz,) = struct.unpack_from(">I", buf, o)
        hdr = 8
        if sz == 1:
            if o + 16 > end:
                break
            (sz,) = struct.unpack_from(">Q", buf, o + 8)
            hdr = 16
        elif sz == 0:
            sz = end - o
        if sz < hdr or o + sz > end:
            break
        yield bytes(buf[o + 4:o + 8]), o, sz, hdr
        o += sz


def find_box(buf, start: int, end: int, kind: bytes):
    for k, o, sz, hdr in iter_boxes(buf, start, end):
        if k == kind:
            return o, sz, hdr
    return None


def file_boxes(f, start: int, end: int):
    """Top-level boxes of an open file, reading only headers."""
    o = start
    while o + 8 <= end:
        f.seek(o)
        h = f.read(16)
        if len(h) < 8:
            break
        (sz,) = struct.unpack_from(">I", h, 0)
        hdr = 8
        if sz == 1:
            if len(h) < 16:
                break
            (sz,) = struct.unpack_from(">Q", h, 8)
            hdr = 16
        elif sz == 0:
            sz = end - o
        if sz < hdr or o + sz > end:
            break
        yield h[4:8], o, sz, hdr
        o += sz


@dataclass
class Track:
    track_id: int
    handler: bytes                 # b"vide", b"soun", b"meta", ...
    timescale: int
    tkhd: bytes                    # whole boxes, verbatim
    mdhd: bytes
    hdlr: bytes
    mhd: bytes                     # vmhd / smhd / nmhd / ...
    dinf: bytes
    stsd: bytes
    sample_entry: bytes            # fourcc of the (single) sample description
    sizes: np.ndarray              # (n,) int64
    offsets: np.ndarray            # (n,) int64, absolute file offsets
    durations: np.ndarray          # (n,) int64, media timescale
    sync: np.ndarray | None        # (n,) bool, None = every sample is a sync sample (no stss)
    has_composition_offsets: bool = False
    edits: list = field(default_factory=list)   # [(segment_duration (movie ts), media_time, rate)]

    @property
    def n(self) -> int:
        return len(self.sizes)

    @property
    def dts(self) -> np.ndarray:
        """Decode time of each sample (media timescale)."""
        return np.concatenate(([0], np.cumsum(self.durations)[:-1])).astype(np.int64) if self.n else \
            np.zeros(0, np.int64)

    @property
    def media_duration(self) -> int:
        return int(self.durations.sum())

    def keyframes(self) -> np.ndarray:
        return np.flatnonzero(self.sync) if self.sync is not None else np.arange(self.n)


@dataclass
class Movie:
    path: Path
    ftyp: bytes
    mvhd: bytes
    movie_timescale: int
    tracks: list[Track]
    udta_children: list[tuple[bytes, bytes]]    # (type, whole box) in order
    file_size: int

    def track(self, handler: bytes | str) -> Track | None:
        h = handler.encode() if isinstance(handler, str) else handler
        found = [t for t in self.tracks if t.handler == h]
        return found[0] if found else None

    def tmf_track(self) -> Track | None:
        for t in self.tracks:
            if t.sample_entry == b"tmfd":
                return t
        return None

    def udta(self, kind: bytes) -> bytes | None:
        """Body of the first ``moov/udta/<kind>`` box."""
        for k, box in self.udta_children:
            if k == kind:
                hdr = 16 if struct.unpack_from(">I", box, 0)[0] == 1 else 8
                return box[hdr:]
        return None


def _fullbox_u32(buf, body: int, v0: int, v1: int) -> int:
    return struct.unpack_from(">I", buf, body + (v1 if buf[body] == 1 else v0))[0]


def _u32s(buf, o: int, n: int) -> np.ndarray:
    return np.frombuffer(bytes(buf[o:o + 4 * n]), ">u4").astype(np.int64)


def _parse_trak(m: bytes, to: int, tsz: int, thdr: int, path) -> Track:
    s, e = to + thdr, to + tsz

    def need(start, end, kind):
        r = find_box(m, start, end, kind)
        if not r:
            raise Mp4Error(f"{Path(path).name}: a track has no {kind.decode()} box")
        return r

    ko, ksz, khdr = need(s, e, b"tkhd")
    tkhd = m[ko:ko + ksz]
    track_id = _fullbox_u32(m, ko + khdr, 12, 20)
    edits = []
    ed = find_box(m, s, e, b"edts")
    if ed:
        el = find_box(m, ed[0] + ed[2], ed[0] + ed[1], b"elst")
        if el:
            b = el[0] + el[2]
            ver = m[b]
            (cnt,) = struct.unpack_from(">I", m, b + 4)
            p = b + 8
            for _ in range(cnt):
                if ver == 1:
                    dur, mt, ri, rf = struct.unpack_from(">QqhH", m, p)
                    p += 20
                else:
                    dur, mt, ri, rf = struct.unpack_from(">IihH", m, p)
                    p += 12
                edits.append((dur, mt, ri + rf / 65536.0))
    mo, msz, mhdr = need(s, e, b"mdia")
    do, dsz, dhdr = need(mo + mhdr, mo + msz, b"mdhd")
    mdhd = m[do:do + dsz]
    timescale = _fullbox_u32(m, do + dhdr, 12, 20)
    ho, hsz, hhdr = need(mo + mhdr, mo + msz, b"hdlr")
    hdlr = m[ho:ho + hsz]
    handler = bytes(m[ho + hhdr + 8:ho + hhdr + 12])
    io, isz, ihdr = need(mo + mhdr, mo + msz, b"minf")
    mhd = b""
    for k in (b"vmhd", b"smhd", b"nmhd", b"sthd", b"hmhd", b"gmhd"):
        r = find_box(m, io + ihdr, io + isz, k)
        if r:
            mhd = m[r[0]:r[0] + r[1]]
            break
    r = find_box(m, io + ihdr, io + isz, b"dinf")
    dinf = m[r[0]:r[0] + r[1]] if r else b""
    bo, bsz, bhdr = need(io + ihdr, io + isz, b"stbl")
    st = {k: (o, sz, hdr) for k, o, sz, hdr in iter_boxes(m, bo + bhdr, bo + bsz)}
    if b"stsd" not in st or b"stsz" not in st and b"stz2" not in st:
        raise Mp4Error(f"{Path(path).name}: a track has no sample tables")
    so, ssz, shdr = st[b"stsd"]
    stsd = m[so:so + ssz]
    (n_desc,) = struct.unpack_from(">I", m, so + shdr + 4)
    entry = bytes(m[so + shdr + 12:so + shdr + 16]) if n_desc else b""

    # sample sizes
    if b"stsz" in st:
        o = st[b"stsz"][0] + st[b"stsz"][2]
        fixed, n = struct.unpack_from(">II", m, o + 4)
        sizes = np.full(n, fixed, np.int64) if fixed else _u32s(m, o + 12, n)
    else:
        raise Mp4Error(f"{Path(path).name}: compact sample sizes (stz2) are not supported")
    # chunk offsets
    if b"stco" in st:
        o = st[b"stco"][0] + st[b"stco"][2]
        (nc,) = struct.unpack_from(">I", m, o + 4)
        chunk_off = _u32s(m, o + 8, nc)
    elif b"co64" in st:
        o = st[b"co64"][0] + st[b"co64"][2]
        (nc,) = struct.unpack_from(">I", m, o + 4)
        chunk_off = np.frombuffer(bytes(m[o + 8:o + 8 + 8 * nc]), ">u8").astype(np.int64)
    else:
        raise Mp4Error(f"{Path(path).name}: a track has no chunk offsets")
    # sample-to-chunk
    o = st[b"stsc"][0] + st[b"stsc"][2] if b"stsc" in st else None
    if o is None:
        raise Mp4Error(f"{Path(path).name}: a track has no sample-to-chunk table")
    (ne,) = struct.unpack_from(">I", m, o + 4)
    ent = _u32s(m, o + 8, 3 * ne).reshape(ne, 3) if ne else np.zeros((0, 3), np.int64)
    if ne and np.any(ent[:, 2] != ent[0, 2]):
        raise Mp4Error(f"{Path(path).name}: a track changes its sample description midway")
    spc = np.zeros(nc, np.int64)
    for i in range(ne):
        first = int(ent[i, 0]) - 1
        last = int(ent[i + 1, 0]) - 1 if i + 1 < ne else nc
        spc[first:last] = ent[i, 1]
    n = len(sizes)
    if int(spc.sum()) < n:
        raise Mp4Error(f"{Path(path).name}: sample tables disagree (chunks hold fewer samples than listed)")
    chunk_of = np.repeat(np.arange(nc), spc)[:n]
    csum = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    first_in_chunk = np.concatenate(([0], np.cumsum(spc)[:-1]))
    offsets = chunk_off[chunk_of] + (csum - csum[first_in_chunk[chunk_of]]) if n else np.zeros(0, np.int64)
    # decode durations
    durations = np.zeros(n, np.int64)
    if b"stts" in st:
        o = st[b"stts"][0] + st[b"stts"][2]
        (ne,) = struct.unpack_from(">I", m, o + 4)
        tt = _u32s(m, o + 8, 2 * ne).reshape(ne, 2) if ne else np.zeros((0, 2), np.int64)
        d = np.repeat(tt[:, 1], tt[:, 0])[:n]
        durations[:len(d)] = d
    # sync samples
    sync = None
    if b"stss" in st:
        o = st[b"stss"][0] + st[b"stss"][2]
        (ne,) = struct.unpack_from(">I", m, o + 4)
        idx = _u32s(m, o + 8, ne) - 1
        sync = np.zeros(n, bool)
        sync[idx[(idx >= 0) & (idx < n)]] = True
    comp = False
    if b"ctts" in st:
        o = st[b"ctts"][0] + st[b"ctts"][2]
        (ne,) = struct.unpack_from(">I", m, o + 4)
        tt = np.frombuffer(bytes(m[o + 8:o + 8 + 8 * ne]), ">i4").reshape(ne, 2) if ne else np.zeros((0, 2))
        comp = bool(np.any(tt[:, 1] != 0))
    return Track(track_id, handler, timescale, bytes(tkhd), bytes(mdhd), bytes(hdlr), bytes(mhd), bytes(dinf),
                 bytes(stsd), entry, sizes, offsets.astype(np.int64), durations, sync, comp, edits)


def read_movie(path: str | Path) -> Movie:
    """Parse a flat MP4 (moov + mdat). Raises Mp4Error for fragmented or unreadable files."""
    path = Path(path)
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        top = list(file_boxes(f, 0, size))
        kinds = [k for k, *_ in top]
        if b"moof" in kinds:
            raise Mp4Error(f"{path.name}: the recording is still in fragments (the camera did not finish "
                           "writing it); rebuild its index first with scripts/repair_recordings.py")
        moovs = [(o, sz) for k, o, sz, _h in top if k == b"moov"]
        if not moovs:
            raise Mp4Error(f"{path.name}: no movie index (moov) - the file is incomplete")
        ftyp = b""
        for k, o, sz, _h in top:
            if k == b"ftyp":
                f.seek(o)
                ftyp = f.read(sz)
                break
        mo, msz = moovs[-1]       # the camera's flatten appends the rebuilt moov at the end
        f.seek(mo)
        m = f.read(msz)
    mhdr = 16 if struct.unpack_from(">I", m, 0)[0] == 1 else 8
    if find_box(m, mhdr, len(m), b"mvex"):
        raise Mp4Error(f"{path.name}: the recording is fragmented; rebuild its index first with "
                       "scripts/repair_recordings.py")
    r = find_box(m, mhdr, len(m), b"mvhd")
    if not r:
        raise Mp4Error(f"{path.name}: no movie header")
    mvhd = m[r[0]:r[0] + r[1]]
    movie_ts = _fullbox_u32(m, r[0] + r[2], 12, 20)
    tracks = [_parse_trak(m, o, sz, hdr, path) for k, o, sz, hdr in iter_boxes(m, mhdr, len(m)) if k == b"trak"]
    udta = []
    r = find_box(m, mhdr, len(m), b"udta")
    if r:
        udta = [(k, bytes(m[o:o + sz])) for k, o, sz, _h in iter_boxes(m, r[0] + r[2], r[0] + r[1])]
    return Movie(path, bytes(ftyp), bytes(mvhd), movie_ts, tracks, udta, size)


def read_samples(path, track: Track, idx=None) -> list[bytes]:
    """The bytes of some samples of a track (all when ``idx`` is None)."""
    idx = range(track.n) if idx is None else idx
    out = []
    with open(path, "rb") as f:
        for i in idx:
            f.seek(int(track.offsets[i]))
            out.append(f.read(int(track.sizes[i])))
    return out


# --------------------------------------------------------------------------- writing
def box(kind: bytes, *parts: bytes) -> bytes:
    body = b"".join(parts)
    if len(body) + 8 > 0xFFFFFFFF:
        return struct.pack(">I4sQ", 1, kind, len(body) + 16) + body
    return struct.pack(">I4s", len(body) + 8, kind) + body


def fullbox(kind: bytes, version: int, flags: int, *parts: bytes) -> bytes:
    return box(kind, struct.pack(">I", (version << 24) | flags), *parts)


def patch_duration(b: bytes, dur: int, off_v0: int, off_v1: int, next_track_id: int | None = None) -> bytes:
    """Copy of a FullBox (mvhd/tkhd/mdhd) with its duration (and mvhd next_track_ID) replaced."""
    out = bytearray(b)
    hdr = 16 if struct.unpack_from(">I", out, 0)[0] == 1 else 8
    ver = out[hdr]
    if ver == 1:
        struct.pack_into(">Q", out, hdr + off_v1, dur)
    else:
        if dur > 0xFFFFFFFF:
            raise Mp4Error("duration too long for a version-0 header")
        struct.pack_into(">I", out, hdr + off_v0, dur)
    if next_track_id is not None:
        struct.pack_into(">I", out, hdr + (108 if ver == 1 else 96), next_track_id)
    return bytes(out)


def stbl(stsd: bytes, durations: np.ndarray, sizes: np.ndarray, offsets: np.ndarray,
         sync: np.ndarray | None, use64: bool) -> bytes:
    """A sample table with one sample per chunk."""
    n = len(sizes)
    # stts: run-length encoded durations
    if n:
        d = np.asarray(durations, np.int64)
        change = np.flatnonzero(np.diff(d)) + 1
        starts = np.concatenate(([0], change))
        runs = np.diff(np.concatenate((starts, [n])))
        tt = np.stack([runs, d[starts]], 1).astype(">u4").tobytes()
        nruns = len(starts)
    else:
        tt, nruns = b"", 0
    parts = [stsd, fullbox(b"stts", 0, 0, struct.pack(">I", nruns), tt)]
    if sync is not None and not bool(np.all(sync)):
        idx = (np.flatnonzero(sync) + 1).astype(">u4")
        parts.append(fullbox(b"stss", 0, 0, struct.pack(">I", len(idx)), idx.tobytes()))
    parts.append(fullbox(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1) if n else struct.pack(">I", 0)))
    parts.append(fullbox(b"stsz", 0, 0, struct.pack(">II", 0, n), np.asarray(sizes).astype(">u4").tobytes()))
    if use64:
        parts.append(fullbox(b"co64", 0, 0, struct.pack(">I", n), np.asarray(offsets).astype(">u8").tobytes()))
    else:
        parts.append(fullbox(b"stco", 0, 0, struct.pack(">I", n), np.asarray(offsets).astype(">u4").tobytes()))
    return box(b"stbl", *parts)


def elst(edits: list[tuple[int, int]]) -> bytes:
    """An edit list box from [(segment_duration in movie ts, media_time or -1)], rate 1."""
    v1 = any(d > 0xFFFFFFFF or abs(t) > 0x7FFFFFFF for d, t in edits)
    body = b"".join(struct.pack(">QqhH" if v1 else ">IihH", d, t, 1, 0) for d, t in edits)
    return box(b"edts", fullbox(b"elst", 1 if v1 else 0, 0, struct.pack(">I", len(edits)), body))


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def copy_ranges(src_fd: int, dst_fd: int, ranges, bufsize: int = 8 << 20) -> None:
    """Append byte ranges [(offset, length)] of file descriptor ``src_fd`` at the current
    position of ``dst_fd`` (kernel copy when the platform allows, else read + write)."""
    fast = hasattr(os, "copy_file_range")
    for off, length in ranges:
        off, length = int(off), int(length)
        while length > 0:
            n = 0
            if fast:
                try:
                    n = os.copy_file_range(src_fd, dst_fd, length, off)
                except OSError:
                    fast = False
                    n = 0
            if not n:
                chunk = os.pread(src_fd, min(length, bufsize), off)
                if not chunk:
                    raise Mp4Error("the source video ended early (truncated file)")
                write_all(dst_fd, chunk)
                n = len(chunk)
            off += n
            length -= n

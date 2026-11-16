# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Cut a Trinet recording to a time segment that is itself a valid Trinet recording.

    from trinet_tools.cut import cut_recording, cut_camera
    report = cut_recording("take_folder/", t0_ns, t1_ns, "out/")     # every camera of a kit
    report = cut_camera(["…_L.mp4", "…_R.mp4", "….imu", …], t0_ns, t1_ns, "out/")

Times are on the take's **master clock** (the clock ``reader.read_vts(...).global_sof_ns()``
puts every camera's frames on; the head camera of a wrist kit is the master), in ns.

Per camera:
  * **video** — stream copy (no re-encoding: the cut is bit-for-bit the original
    frames). Each camera starts at ITS OWN last keyframe at or before ``t0`` (at most
    one keyframe interval, ~1 s, early: cameras' keyframes are not aligned) and ends at
    its last frame before ``t1``. The sample tables are rebuilt (one sample per chunk),
    the movie index is written first (fast start).
  * **audio** — the AAC frames covering the video (plus one frame of decoder pre-roll),
    with an edit list that keeps sound and picture aligned to the audio sample.
  * **embedded metadata (TMF)** — the ``tmfd`` track is rebuilt from the cut sidecars
    (``trinet_tools.tmf_write``, the camera's own encoder), ``udta/tmfm`` regenerated for
    the cut (frame drops / sync of the cut, plus ``cut: {source, first_frame, …}``),
    ``udta/tmfc`` (calibration) copied. A source without TMF (e.g. a re-muxed file) gets
    one built from its sidecars.
  * **.vts** — the header and the kept frames' entries; master-clock frame times are
    unchanged (v5: per-frame offsets; v3/v4: the clock-skew reference moves to the new
    first frame, so the header offset is re-based — exact when the skew is 0, within
    1 ns otherwise). Frame numbers are rebased to 0 when they counted frames.
  * **.imu / .tel** — samples from 1 s before the first kept frame to 1 s after the
    last; ``.imu`` header ``start_time_ns`` = first kept sample, ``video_start_ns``
    moved with the first frame (as the camera's file rotation writes them); ``.tel``
    ``record_count`` updated.
  * ``.json`` sidecar and calibration files are copied unchanged. File names never change.

Refuses (``CutError`` with a plain message) fragmented recordings, B-frames, and tracks
that change their sample description midway; rebuild a fragmented file's index first
with ``scripts/repair_recordings.py``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import mp4 as M
from . import tmf_write as TW
from .reader import read_vts

NS = 1_000_000_000
PAD_NS = NS                       # .imu / .tel kept around the frames
VTS_HEADER = 32
IMU_HEADER = 64
TEL_HEADER = 32
TEL_RECORD = 24
OFFSET_LIMIT = 0xFFFFFFFF          # beyond: 64-bit chunk offsets and a large mdat header
CODEC_NAMES = {b"avc1": "h264", b"avc3": "h264", b"hvc1": "h265", b"hev1": "h265"}


class CutError(ValueError):
    """A recording that cannot be cut as asked (the message is plain language)."""


# --------------------------------------------------------------------------- files
@dataclass
class VideoFiles:
    mp4: Path
    vts: Path | None
    eye: str | None                  # "L" / "R" for a stereo camera


@dataclass
class CameraFiles:
    base: str                        # grp<S>_<dev>_take0001 (stereo) / grp<S>_<dev>_1 (mono)
    folder: Path
    videos: list[VideoFiles]
    imu: Path | None = None
    tel: Path | None = None
    json: Path | None = None
    extras: list[Path] = field(default_factory=list)   # calibration files, copied unchanged

    @property
    def device_tag(self) -> str | None:
        m = re.match(r"^grp\d+_([0-9a-fA-F]{8})_", self.base)
        return m.group(1).lower() if m else None


_STEREO = re.compile(r"^(?P<base>.+)_(?P<eye>[LR])$")


def group_files(paths) -> list[CameraFiles]:
    """Group a recording's files into cameras by base name: ``<base>_L/_R.mp4`` + ``.vts``
    and a shared ``<base>.imu`` / ``.tel`` (stereo), or ``<base>.mp4/.vts/.imu/.tel/.json``
    (mono). Other ``.json`` files (calibrations) go with the camera whose device tag their
    name carries, or with every camera of their folder when it names none."""
    paths = [Path(p) for p in paths]
    files = {p.name: p for p in paths if p.is_file()}
    cams: dict[tuple[Path, str], CameraFiles] = {}
    for p in sorted(files.values()):
        if p.suffix.lower() != ".mp4":
            continue
        m = _STEREO.match(p.stem)
        stereo = bool(m) and (p.parent / f"{p.stem}.vts").exists() and \
            any((p.parent / f"{m.group('base')}{s}").exists() for s in (".imu", "_L.mp4", "_R.mp4"))
        base = m.group("base") if stereo else p.stem
        vts = p.with_suffix(".vts")
        if vts.name not in files or files[vts.name].parent != p.parent:
            continue                     # not a Trinet camera file (e.g. a vendor preview video)
        cam = cams.setdefault((p.parent, base), CameraFiles(base, p.parent, []))
        cam.videos.append(VideoFiles(p, vts, m.group("eye") if stereo else None))
    used = {v.mp4 for c in cams.values() for v in c.videos} | {v.vts for c in cams.values() for v in c.videos}
    for (folder, base), cam in cams.items():
        cam.videos.sort(key=lambda v: (v.eye or ""))
        for suf, attr in ((".imu", "imu"), (".tel", "tel"), (".json", "json")):
            p = folder / f"{base}{suf}"
            if p.name in files and files[p.name].parent == folder:
                setattr(cam, attr, files[p.name])
                used.add(files[p.name])
    for p in sorted(files.values()):
        if p in used or p.suffix.lower() != ".json":
            continue
        tagged = [c for c in cams.values() if c.device_tag and c.device_tag in p.name.lower()]
        targets = tagged or [c for c in cams.values() if c.folder == p.parent]
        for c in targets:
            c.extras.append(p)
    return [cams[k] for k in sorted(cams, key=lambda k: (str(k[0]), k[1]))]


# --------------------------------------------------------------------------- sidecars
def _imu_sample_size(version: int) -> int:
    return 44 if version == 1 else 76 if version == 2 else 80


def _imu_ts(data: bytes) -> tuple[np.ndarray, int]:
    if len(data) < IMU_HEADER or data[:8] != b"TRIMU001":
        raise CutError("the .imu file is not a Trinet IMU file")
    ss = _imu_sample_size(struct.unpack_from("<I", data, 8)[0])
    n = (len(data) - IMU_HEADER) // ss
    rows = np.frombuffer(data, np.uint8, n * ss, IMU_HEADER).reshape(n, ss)
    return rows[:, :8].copy().view("<u8").reshape(n).astype(np.int64), ss


def cut_imu(data: bytes, lo_ns: int, hi_ns: int, first_frame_shift_ns: int | None) -> bytes:
    """``.imu`` samples with ``lo <= t <= hi``; header ``start_time_ns`` = the first kept
    sample, ``video_start_ns`` moved by ``first_frame_shift_ns`` (kept 0 when unknown)."""
    ts, ss = _imu_ts(data)
    keep = np.flatnonzero((ts >= lo_ns) & (ts <= hi_ns))
    out = bytearray(data[:IMU_HEADER])
    if len(keep):
        a, b = int(keep[0]), int(keep[-1]) + 1
        out += data[IMU_HEADER + a * ss:IMU_HEADER + b * ss]
        struct.pack_into("<Q", out, 20, int(ts[a]))
    vs = struct.unpack_from("<Q", out, 28)[0]
    if vs and first_frame_shift_ns is not None:
        struct.pack_into("<Q", out, 28, max(int(vs) + int(first_frame_shift_ns), 0))
    return bytes(out)


def cut_tel(data: bytes, lo_ns: int, hi_ns: int) -> bytes:
    if len(data) < TEL_HEADER or data[:7] != b"TRTEL01":
        raise CutError("the .tel file is not a Trinet telemetry file")
    n = (len(data) - TEL_HEADER) // TEL_RECORD
    rows = np.frombuffer(data, np.uint8, n * TEL_RECORD, TEL_HEADER).reshape(n, TEL_RECORD)
    ts = rows[:, :8].copy().view("<u8").reshape(n).astype(np.int64)
    keep = np.flatnonzero((ts >= lo_ns) & (ts <= hi_ns))
    out = bytearray(data[:TEL_HEADER])
    if len(keep):
        out += rows[int(keep[0]):int(keep[-1]) + 1].tobytes()
    struct.pack_into("<I", out, 16, (len(out) - TEL_HEADER) // TEL_RECORD)
    return bytes(out)


def _best_ns(entries: np.ndarray, version: int) -> np.ndarray:
    """The reader's ``best_timestamps_ns`` of raw entries (sof, else the encoder PTS)."""
    sof = entries[:, 4:12].copy().view("<u8").reshape(-1).astype(np.int64)
    if version >= 2 and not np.any(sof != 0):
        return entries[:, 16:24].copy().view("<u8").reshape(-1).astype(np.int64) * 1000
    return sof


def cut_vts(data: bytes, k0: int, k1: int) -> bytes:
    """``.vts`` of frames ``[k0, k1)`` whose master-clock times equal the original's."""
    if len(data) < VTS_HEADER or data[:8] != b"TRIVTS01":
        raise CutError("the .vts file is not a Trinet frame-time file")
    ver = struct.unpack_from("<I", data, 8)[0]
    es = TW.vts_entry_size(ver)
    n = (len(data) - VTS_HEADER) // es
    ent = np.frombuffer(data, np.uint8, n * es, VTS_HEADER).reshape(n, es)
    hdr = bytearray(data[:VTS_HEADER])
    kept = ent[k0:k1].copy()
    fn = ent[:, 0:4].copy().view("<u4").reshape(n)
    if k0 and n and np.array_equal(fn, np.arange(n, dtype=np.uint32)):
        kept[:, 0:4] = np.arange(k1 - k0, dtype="<u4").view(np.uint8).reshape(-1, 4)
    if 3 <= ver < 5 and len(kept):
        off, skew, qual, flags = struct.unpack_from("<qiHH", hdr, 16)
        if flags & 1 and skew:
            best = _best_ns(ent, ver)
            best_cut = _best_ns(kept, ver)
            d = int(best_cut[0]) - int(best[0])
            struct.pack_into("<q", hdr, 16, off + (skew * d) // NS)
    return bytes(hdr) + kept.tobytes()


# --------------------------------------------------------------------------- frames
@dataclass
class _Video:
    files: VideoFiles
    movie: M.Movie
    video: M.Track
    vts: bytes
    g: np.ndarray                  # master-clock time of each frame (ns)
    local: np.ndarray              # the camera's own clock (sof)
    k0: int = 0
    k1: int = 0


def _plan_video(vf: VideoFiles, t0: int, t1: int) -> _Video:
    try:
        movie = M.read_movie(vf.mp4)
    except M.Mp4Error as e:
        raise CutError(str(e)) from None
    vids = [t for t in movie.tracks if t.handler == b"vide"]
    if len(vids) != 1:
        raise CutError(f"{vf.mp4.name}: expected one video track, found {len(vids)}")
    v = vids[0]
    if v.has_composition_offsets:
        raise CutError(f"{vf.mp4.name}: the video uses B-frames (reordered frames); it cannot be cut "
                       "without re-encoding")
    if vf.vts is None:
        raise CutError(f"{vf.mp4.name}: its frame-time file (.vts) is missing")
    data = vf.vts.read_bytes()
    vd = read_vts(str(vf.vts))
    g = vd.global_sof_ns().astype(np.int64)
    local = vd.best_timestamps_ns.astype(np.int64)
    nf = min(v.n, len(g))
    out = _Video(vf, movie, v, data, g[:nf], local[:nf])
    if nf == 0:
        return out
    run = np.maximum.accumulate(out.g)
    keys = v.keyframes()
    keys = keys[keys < nf]
    if not len(keys):
        raise CutError(f"{vf.mp4.name}: the video has no keyframe")
    before = keys[run[keys] <= t0]
    k0 = int(before[-1]) if len(before) else int(keys[0])
    k1 = int(np.searchsorted(run, t1, side="left"))
    if int(np.searchsorted(run, t0, side="left")) >= k1:
        return out                       # no frame of this camera in [t0, t1): nothing to cut
    out.k0, out.k1 = k0, max(k1, k0)
    return out


# --------------------------------------------------------------------------- mp4
def _presentation_offset(t: M.Track, movie_ts: int, name: str) -> tuple[Fraction, Fraction]:
    """(empty edit, media time) of a track's edit list in seconds: a sample at media time
    T is presented at ``T - media + empty``."""
    if not t.edits:
        return Fraction(0), Fraction(0)
    empty = Fraction(0)
    normal = []
    for dur, mt, rate in t.edits:
        if mt == -1:
            if normal:
                raise CutError(f"{name}: unsupported edit list (a gap in the middle)")
            empty += Fraction(dur, movie_ts)
        else:
            if rate != 1:
                raise CutError(f"{name}: unsupported edit list (speed change)")
            normal.append(mt)
    if len(normal) > 1:
        raise CutError(f"{name}: unsupported edit list (several pieces)")
    return empty, Fraction(normal[0] if normal else 0, t.timescale)


def _synth_tmf_boxes(track_id: int) -> dict:
    """tkhd/mdhd/hdlr/nmhd/dinf/stsd of a TMF track, as the camera's flatten writes them."""
    tkhd = M.fullbox(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, track_id, 0, 0),   # times, id, -, duration
                     bytes(16),                                                        # reserved, layer, volume
                     struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000), bytes(8))
    mdhd = M.fullbox(b"mdhd", 0, 0, struct.pack(">IIIII", 0, 0, 1000, 0, 0x55C40000))
    hdlr = M.fullbox(b"hdlr", 0, 0, struct.pack(">I", 0), b"meta", bytes(12), b"Trinet TMF\x00")
    nmhd = M.fullbox(b"nmhd", 0, 0)
    dinf = M.box(b"dinf", M.fullbox(b"dref", 0, 0, struct.pack(">I", 1), M.fullbox(b"url ", 0, 1)))
    stsd = M.fullbox(b"stsd", 0, 0, struct.pack(">I", 1), M.box(b"tmfd", bytes(6), struct.pack(">H", 1)))
    return dict(tkhd=tkhd, mdhd=mdhd, hdlr=hdlr, mhd=nmhd, dinf=dinf, stsd=stsd)


def _codec(entry: bytes) -> str | None:
    return CODEC_NAMES.get(entry)


def write_cut_mp4(pv: _Video, tmf_chunks: list[TW.TmfChunk], tmfm: bytes, out_path: Path) -> dict:
    """Write the cut MP4 of ``pv`` (frames ``[k0, k1)``) with the given TMF payloads and
    ``tmfm`` JSON. Returns details (sizes, audio alignment, dropped tracks)."""
    movie, v, name = pv.movie, pv.video, pv.files.mp4.name
    tsm = movie.movie_timescale
    k0, k1 = pv.k0, pv.k1
    if v.sync is not None and not v.sync[k0]:
        raise CutError(f"{name}: internal error: the cut does not start on a keyframe")
    v_idx = np.arange(k0, k1)
    v_dts = v.dts
    v_dur = v.durations[k0:k1]
    ev, mv = _presentation_offset(v, tsm, name)
    pv0 = Fraction(int(v_dts[k0]), v.timescale) - mv + ev      # original presentation time of frame k0
    video_secs = Fraction(int(v_dur.sum()), v.timescale)

    info: dict = {"dropped_tracks": [], "audio": None}
    audio = None
    for t in movie.tracks:
        if t.handler == b"soun" and audio is None:
            audio = t
        elif t is not v and t.sample_entry != b"tmfd":
            info["dropped_tracks"].append(t.handler.decode("latin1"))   # e.g. a second sound track
    a_plan = None
    if audio is not None and audio.n:
        if audio.has_composition_offsets:
            raise CutError(f"{name}: the sound track reorders its samples; it cannot be cut")
        ea, ma = _presentation_offset(audio, tsm, name)
        tsa = audio.timescale
        xa = (pv0 - ea + ma) * tsa                    # audio media time at the cut's time 0
        xa_end = xa + video_secs * tsa
        a_dts = audio.dts
        j = int(np.searchsorted(a_dts, math.floor(xa), side="right")) - 1
        j0 = max(j - 1, 0)                            # one frame of decoder pre-roll
        j1 = int(np.searchsorted(a_dts, math.ceil(xa_end), side="left"))
        if j1 > j0 and j0 < audio.n:
            start = int(a_dts[j0])
            media_dur = int(audio.durations[j0:j1].sum())
            xr = round(xa)
            if xr >= start:
                empty_mov, media_time = 0, xr - start
            else:
                empty_mov, media_time = round(Fraction(start - xa, tsa) * tsm), 0
            seg = max((media_dur - media_time) * tsm // tsa, 0)
            edits = ([(empty_mov, -1)] if empty_mov else []) + [(seg, media_time)]
            a_plan = (audio, np.arange(j0, j1), edits, media_dur)
            info["audio"] = {"first_sample": j0, "samples": j1 - j0, "media_time": media_time,
                             "empty_ms": empty_mov * 1000 / tsm,
                             "error_ms": float((xr - xa) / tsa * 1000) if xr >= start else 0.0}

    # sample data: copied in the source's own order (keeps its interleaving), runs coalesced
    src_off = [v.offsets[v_idx]]
    src_len = [v.sizes[v_idx]]
    if a_plan:
        src_off.append(a_plan[0].offsets[a_plan[1]])
        src_len.append(a_plan[0].sizes[a_plan[1]])
    offs = np.concatenate(src_off).astype(np.int64)
    lens = np.concatenate(src_len).astype(np.int64)
    order = np.argsort(offs, kind="stable")
    so, sl = offs[order], lens[order]
    if len(so) > 1 and np.any(so[1:] < so[:-1] + sl[:-1]):
        raise CutError(f"{name}: samples overlap in the file; it cannot be cut")
    rel = np.empty(len(order), np.int64)
    rel[order] = np.concatenate(([0], np.cumsum(sl)[:-1]))
    media_bytes = int(sl.sum())
    breaks = np.flatnonzero(so[1:] != so[:-1] + sl[:-1]) + 1 if len(so) > 1 else np.zeros(0, int)
    rstart = np.concatenate(([0], breaks)) if len(so) else np.zeros(0, int)
    rend = np.concatenate((breaks, [len(so)])) if len(so) else np.zeros(0, int)
    ranges = [(int(so[a]), int(so[b - 1] + sl[b - 1] - so[a])) for a, b in zip(rstart, rend)]
    tmf_sizes = np.array([len(c.data) for c in tmf_chunks], np.int64)
    tmf_rel = media_bytes + np.concatenate(([0], np.cumsum(tmf_sizes)[:-1])) if len(tmf_sizes) else \
        np.zeros(0, np.int64)
    data_bytes = media_bytes + int(tmf_sizes.sum())

    # track ids / boxes
    tmf_src = movie.tmf_track()
    ids = [t.track_id for t in movie.tracks]
    tmf_id = tmf_src.track_id if tmf_src is not None else max(ids, default=0) + 1
    next_id = max(ids + [tmf_id]) + 1
    tmf_boxes = (dict(tkhd=tmf_src.tkhd, mdhd=tmf_src.mdhd, hdlr=tmf_src.hdlr, mhd=tmf_src.mhd,
                      dinf=tmf_src.dinf, stsd=tmf_src.stsd) if tmf_src is not None else _synth_tmf_boxes(tmf_id))
    n_v = len(v_idx)
    v_rel = rel[:n_v]
    a_rel = rel[n_v:]

    def trak(src_boxes: dict, timescale: int, durations, sizes, offsets, sync, use64, edits=None) -> bytes:
        media_dur = int(np.sum(durations))
        if edits:
            track_dur = sum(d for d, _t in edits)
        else:
            track_dur = media_dur * tsm // timescale
        tkhd = M.patch_duration(src_boxes["tkhd"], track_dur, 20, 28)
        mdhd = M.patch_duration(src_boxes["mdhd"], media_dur, 16, 24)
        minf = M.box(b"minf", src_boxes["mhd"], src_boxes["dinf"],
                     M.stbl(src_boxes["stsd"], durations, sizes, offsets, sync, use64))
        parts = [tkhd] + ([M.elst(edits)] if edits else []) + [M.box(b"mdia", mdhd, src_boxes["hdlr"], minf)]
        return M.box(b"trak", *parts), track_dur

    def moov(base: int, use64: bool) -> bytes:
        traks, durs = [], []
        for t in movie.tracks:
            if t is v:
                sync = v.sync[v_idx] if v.sync is not None else None
                b, d = trak(dict(tkhd=t.tkhd, mdhd=t.mdhd, hdlr=t.hdlr, mhd=t.mhd, dinf=t.dinf, stsd=t.stsd),
                            t.timescale, v_dur, v.sizes[v_idx], base + v_rel, sync, use64)
            elif a_plan and t is a_plan[0]:
                idx = a_plan[1]
                b, d = trak(dict(tkhd=t.tkhd, mdhd=t.mdhd, hdlr=t.hdlr, mhd=t.mhd, dinf=t.dinf, stsd=t.stsd),
                            t.timescale, t.durations[idx], t.sizes[idx], base + a_rel, None, use64, a_plan[2])
            elif t is tmf_src:
                continue
            else:
                continue
            traks.append(b)
            durs.append(d)
        if tmf_chunks:
            b, d = trak(tmf_boxes, 1000, np.array([c.dur_ms for c in tmf_chunks], np.int64), tmf_sizes,
                        base + tmf_rel, None, use64)
            traks.append(b)
            durs.append(d)
        mvhd = M.patch_duration(movie.mvhd, max(durs, default=0), 16, 24, next_track_id=next_id)
        udta_parts = [bx for k, bx in movie.udta_children if k not in (b"tmfm", b"tmfc")]
        udta_parts.append(M.box(b"tmfm", tmfm))
        tmfc = movie.udta(b"tmfc")
        if tmfc:
            udta_parts.append(M.box(b"tmfc", tmfc))
        return M.box(b"moov", mvhd, *traks, M.box(b"udta", *udta_parts))

    ftyp = movie.ftyp
    probe = moov(0, False)
    use64 = len(ftyp) + len(probe) + 16 + data_bytes > OFFSET_LIMIT
    mdat_hdr_len = 16 if data_bytes + 8 > OFFSET_LIMIT else 8
    head_len = len(ftyp) + len(moov(0, use64)) + mdat_hdr_len
    moov_b = moov(head_len, use64)
    assert len(ftyp) + len(moov_b) + mdat_hdr_len == head_len
    mdat_hdr = struct.pack(">I4sQ", 1, b"mdat", data_bytes + 16) if mdat_hdr_len == 16 else \
        struct.pack(">I4s", data_bytes + 8, b"mdat")

    tmp = out_path.with_name(out_path.name + ".part")
    expect = head_len + data_bytes
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            M.write_all(fd, ftyp + moov_b + mdat_hdr)
            sfd = os.open(pv.files.mp4, os.O_RDONLY)
            try:
                M.copy_ranges(sfd, fd, ranges)
            finally:
                os.close(sfd)
            for c in tmf_chunks:
                M.write_all(fd, c.data)
            size = os.lseek(fd, 0, os.SEEK_CUR)
        finally:
            os.close(fd)
        if size != expect:
            raise CutError(f"{name}: wrote {size} bytes, expected {expect}")
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    info.update(bytes=size, media_bytes=media_bytes, tmf_chunks=len(tmf_chunks), co64=use64,
                video_seconds=float(video_secs), copy_ranges=len(ranges))
    return info


# --------------------------------------------------------------------------- cameras
def _meta_strings(pv: _Video, cam: CameraFiles, imu: bytes | None, hw_generation: str | None,
                  extra: dict) -> TW.MetaStrings:
    raw = pv.movie.udta(b"tmfm")
    meta = None
    if raw:
        try:
            meta = json.loads(raw)
        except ValueError:
            meta = None
    s = TW.meta_strings_from(meta)
    if meta is None:
        s.codec = _codec(pv.video.sample_entry)
        if pv.files.eye:
            s.eye = pv.files.eye
            other = "R" if pv.files.eye == "L" else "L"
            pair = pv.files.mp4.with_name(f"{cam.base}_{other}.mp4")
            s.pair_file = pair.name
    if not s.device_id:
        dev = imu[40:56] if imu and len(imu) >= 56 and struct.unpack_from("<I", imu, 8)[0] >= 3 else b""
        if dev and any(dev):
            s.device_id = dev.hex()
        elif cam.json is not None:
            try:
                s.device_id = json.loads(cam.json.read_text()).get("device_id") or None
            except (OSError, ValueError):
                pass
    if hw_generation and not s.hw_generation:
        s.hw_generation = hw_generation          # the camera's own value, when it wrote one, wins
    s.generator = "trinet-tools-cut"
    s.extra = {**s.extra, **extra}
    return s


def cut_camera(files, t0_ns: int, t1_ns: int, out_dir, hw_generation: str | None = None) -> dict:
    """Cut one camera (all its files: video(s), .vts, .imu, .tel, .json, calibration) to
    ``[t0_ns, t1_ns)`` on the master clock, writing files of the same names into ``out_dir``.

    Returns a report: per video the kept frames, their master-clock times and how far
    before ``t0`` the camera's cut starts; ``frames == 0`` when the camera recorded
    nothing in the window (then nothing is written)."""
    t0_ns, t1_ns = int(t0_ns), int(t1_ns)
    if t1_ns <= t0_ns:
        raise CutError("the segment ends before it starts")
    cams = files if isinstance(files, CameraFiles) else None
    if cams is None:
        found = group_files(files)
        if len(found) != 1:
            raise CutError(f"expected the files of one camera, found {len(found)} cameras")
        cams = found[0]
    cam = cams
    out_dir = Path(out_dir)
    plans = [_plan_video(vf, t0_ns, t1_ns) for vf in cam.videos]
    report = {"camera": cam.base, "device_tag": cam.device_tag, "videos": [], "files": []}
    live = [p for p in plans if p.k1 > p.k0]
    if not live:
        report["frames"] = 0
        for p in plans:
            report["videos"].append({"file": p.files.mp4.name, "frames": 0})
        return report
    out_dir.mkdir(parents=True, exist_ok=True)

    # sidecar window on the camera's own clock
    seen = np.concatenate([p.local[p.k0:p.k1] for p in live])
    seen = seen[seen > 0]
    if not len(seen):
        raise CutError(f"{cam.base}: the kept frames have no capture times")
    lo, hi = int(seen.min()) - PAD_NS, int(seen.max()) + PAD_NS
    first_now = min(int(p.local[p.k0]) for p in live)
    first_was = min(int(p.local[0]) for p in plans if len(p.local))
    imu = cam.imu.read_bytes() if cam.imu else None
    tel = cam.tel.read_bytes() if cam.tel else None
    cut_i = cut_imu(imu, lo, hi, first_now - first_was if first_was > 0 else None) if imu else None
    cut_t = cut_tel(tel, lo, hi) if tel else None
    if cut_i is not None:
        (out_dir / cam.imu.name).write_bytes(cut_i)
        report["files"].append(cam.imu.name)
        report["imu_samples"] = (len(cut_i) - IMU_HEADER) // _imu_sample_size(struct.unpack_from("<I", cut_i, 8)[0])
    if cut_t is not None:
        (out_dir / cam.tel.name).write_bytes(cut_t)
        report["files"].append(cam.tel.name)
        report["tel_records"] = (len(cut_t) - TEL_HEADER) // TEL_RECORD

    for p in plans:
        name = p.files.mp4.name
        if p.k1 <= p.k0:
            report["videos"].append({"file": name, "frames": 0})
            continue
        vts_cut = cut_vts(p.vts, p.k0, p.k1)
        (out_dir / p.files.vts.name).write_bytes(vts_cut)
        try:
            chunks = TW.encode(cut_i if cut_i and len(cut_i) > IMU_HEADER else None, vts_cut, cut_t)
        except ValueError:
            chunks = []
        extra = {"cut": {"source": name, "first_frame": p.k0, "last_frame": p.k1 - 1,
                         "t0_ns": t0_ns, "t1_ns": t1_ns}}
        ms = _meta_strings(p, cam, imu, hw_generation, extra)
        tmfm = TW.build_meta_json(ms, cut_i, vts_cut, cut_t)
        info = write_cut_mp4(p, chunks, tmfm, out_dir / name)
        report["files"] += [name, p.files.vts.name]
        g0, g1 = int(p.g[p.k0]), int(p.g[p.k1 - 1])
        report["videos"].append({
            "file": name, "eye": p.files.eye, "frames": p.k1 - p.k0, "of_frames": len(p.g),
            "first_frame": p.k0, "last_frame": p.k1 - 1, "first_ns": g0, "last_ns": g1,
            "starts_before_s": (t0_ns - g0) / NS, **info})
    for p in [cam.json, *cam.extras]:
        if p is not None:
            shutil.copy2(p, out_dir / p.name)
            report["files"].append(p.name)
    report["frames"] = sum(v.get("frames", 0) for v in report["videos"])
    return report


def collect_files(src) -> tuple[list[Path], Path]:
    if isinstance(src, (str, os.PathLike)) and Path(src).is_dir():
        root = Path(src)
        return sorted(p for p in root.rglob("*") if p.is_file()), root
    paths = [Path(p) for p in (src if not isinstance(src, (str, os.PathLike)) else [src])]
    root = Path(os.path.commonpath([str(p.parent.resolve()) for p in paths])) if paths else Path(".")
    return paths, root


def cut_recording(src, t0_ns: int, t1_ns: int, out_dir, hw_generation=None) -> dict:
    """Cut every camera of a recording (a folder, searched recursively, or a list of files)
    to the same master-clock window. Each camera's files keep their names and their folder
    relative to the recording's root. ``hw_generation``: one string for every camera or
    ``{device_tag: generation}`` (firmware spelling, e.g. ``v4``), recorded in a cut whose
    source did not say. Cameras with no frames in the window are left out."""
    paths, root = collect_files(src)
    cams = group_files(paths)
    if not cams:
        raise CutError("no Trinet recording found (no .mp4 with its .vts)")
    out_dir = Path(out_dir)
    report = {"t0_ns": int(t0_ns), "t1_ns": int(t1_ns), "cameras": []}
    for cam in cams:
        try:
            rel = cam.folder.resolve().relative_to(root.resolve())
        except ValueError:
            rel = Path(".")
        gen = hw_generation.get(cam.device_tag) if isinstance(hw_generation, dict) else hw_generation
        r = cut_camera(cam, t0_ns, t1_ns, out_dir / rel, gen)
        r["folder"] = str(rel)
        report["cameras"].append(r)
    return report

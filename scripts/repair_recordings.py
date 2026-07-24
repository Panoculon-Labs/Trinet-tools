#!/usr/bin/env python3
"""
Repair Trinet recordings that show only ~1 second in some players/uploaders.

Some Trinet camera firmware writes each MP4 as a *fragmented* MP4: the index at
the front of the file (the `moov` atom) describes only the first second of
video, and the rest of the recording lives in per-second "movie fragments".
Capable players (VLC, QuickTime, ffmpeg, most phones) read the fragments and
show the full clip — but **naive readers** (some web uploaders, WhatsApp's
transcoder, a few editors) read only the front index and so see a recording as
"1 second long", or refuse it as truncated/corrupt.

This tool rewrites those files into a **standard, single-index MP4** that every
player and uploader accepts. It:

  * scans a folder for `.mp4` files,
  * leaves already-standard files untouched,
  * for fragmented ones, builds one complete index covering the whole clip and
    appends it, then retires the old fragment index — **the media (video/audio)
    bytes are never moved or re-encoded**, so it's fast and lossless, and
  * is crash-safe: at every moment the file on disk is a valid MP4 (either the
    original fragmented form or the repaired form), so an interruption can't
    destroy a recording.

Audio (if present) is preserved as a second track.

Pure standard-library Python 3 — no dependencies, no ffmpeg required.

Usage:
    python3 repair_recordings.py /path/to/folder
    python3 repair_recordings.py /path/to/folder --recursive
    python3 repair_recordings.py /path/to/folder --backup      # keep <name>.mp4.bak
    python3 repair_recordings.py /path/to/folder --dry-run      # report only
"""

import argparse
import os
import struct
import sys

# --------------------------------------------------------------------------- #
# Big-endian readers
# --------------------------------------------------------------------------- #
def _u32(b, o): return struct.unpack_from(">I", b, o)[0]
def _u64(b, o): return struct.unpack_from(">Q", b, o)[0]
def _s32(b, o): return struct.unpack_from(">i", b, o)[0]

# ISO/IEC 14496-12 sample_flags bit: sample_is_non_sync_sample
_SF_NON_SYNC = 0x00010000

# tfhd flags
_TFHD_BASE_DATA_OFFSET = 0x000001
_TFHD_SAMPLE_DESC_INDEX = 0x000002
_TFHD_DEFAULT_DURATION = 0x000008
_TFHD_DEFAULT_SIZE = 0x000010
_TFHD_DEFAULT_FLAGS = 0x000020

# trun flags
_TRUN_DATA_OFFSET = 0x000001
_TRUN_FIRST_SAMPLE_FLAGS = 0x000004
_TRUN_SAMPLE_DURATION = 0x000100
_TRUN_SAMPLE_SIZE = 0x000200
_TRUN_SAMPLE_FLAGS = 0x000400
_TRUN_SAMPLE_CTS = 0x000800


def _iter_boxes(buf, start, end):
    """Yield (type:bytes, offset, size, header_len) for each box in [start,end)."""
    o = start
    while o + 8 <= end:
        size = _u32(buf, o)
        hdr = 8
        if size == 1:
            if o + 16 > end:
                break
            size = _u64(buf, o + 8)
            hdr = 16
        elif size == 0:
            size = end - o
        if size < hdr or o + size > end:
            break
        yield buf[o + 4:o + 8], o, size, hdr
        o += size


def _find(buf, start, end, typ):
    for t, o, s, h in _iter_boxes(buf, start, end):
        if t == typ:
            return o, s, h
    return None


def _find_all(buf, start, end, typ):
    return [(o, s, h) for t, o, s, h in _iter_boxes(buf, start, end) if t == typ]


# --------------------------------------------------------------------------- #
# Output box builder
# --------------------------------------------------------------------------- #
class _Buf:
    def __init__(self):
        self.a = bytearray()

    def u32(self, v): self.a += struct.pack(">I", v & 0xFFFFFFFF)
    def u64(self, v): self.a += struct.pack(">Q", v & 0xFFFFFFFFFFFFFFFF)
    def raw(self, b): self.a += b

    def box_begin(self, typ):
        at = len(self.a)
        self.u32(0)            # size placeholder
        self.a += typ
        return at

    def box_end(self, at):
        struct.pack_into(">I", self.a, at, len(self.a) - at)


def _copy_patch_dur(out, src, boff, bsz, dur_off_v0, dur_off_v1, dur):
    """Append src[boff:boff+bsz] then overwrite its FullBox duration field."""
    dst = len(out.a)
    out.raw(src[boff:boff + bsz])
    ver = src[boff + 8]
    if ver == 1:
        struct.pack_into(">Q", out.a, dst + 8 + dur_off_v1, dur & 0xFFFFFFFFFFFFFFFF)
    else:
        struct.pack_into(">I", out.a, dst + 8 + dur_off_v0, dur & 0xFFFFFFFF)


# --------------------------------------------------------------------------- #
# Per-track parse + sample expansion
# --------------------------------------------------------------------------- #
class _Track:
    __slots__ = ("track_id", "mdhd_ts", "tkhd", "mdhd", "hdlr", "mh", "dinf",
                 "stsd", "stts", "stsz", "stsc", "stco", "stco64", "stss",
                 "stsz_sample_size", "stsz_cnt", "stts_cnt", "stsc_cnt",
                 "stco_cnt", "stss_cnt", "trex_dur", "trex_size", "trex_flags",
                 "samples")

    def __init__(self):
        self.stss = None
        self.trex_dur = self.trex_size = self.trex_flags = 0
        self.samples = []   # list of (offset, size, dur, key)


def _parse_track(m, to, ts):
    t = _Track()
    tkhd = _find(m, to + 8, to + ts, b"tkhd")
    if not tkhd:
        return None
    t.tkhd = (tkhd[0], tkhd[1])
    ver = m[tkhd[0] + 8]
    t.track_id = _u32(m, tkhd[0] + 8 + (20 if ver == 1 else 12))

    mdia = _find(m, to + 8, to + ts, b"mdia")
    if not mdia:
        return None
    mdhd = _find(m, mdia[0] + 8, mdia[0] + mdia[1], b"mdhd")
    if not mdhd:
        return None
    t.mdhd = (mdhd[0], mdhd[1])
    mver = m[mdhd[0] + 8]
    t.mdhd_ts = _u32(m, mdhd[0] + 8 + (20 if mver == 1 else 12))

    hdlr = _find(m, mdia[0] + 8, mdia[0] + mdia[1], b"hdlr")
    if not hdlr:
        return None
    t.hdlr = (hdlr[0], hdlr[1])

    minf = _find(m, mdia[0] + 8, mdia[0] + mdia[1], b"minf")
    if not minf:
        return None
    mh = (_find(m, minf[0] + 8, minf[0] + minf[1], b"vmhd")
          or _find(m, minf[0] + 8, minf[0] + minf[1], b"smhd")
          or _find(m, minf[0] + 8, minf[0] + minf[1], b"nmhd"))
    if not mh:
        return None
    t.mh = (mh[0], mh[1])
    dinf = _find(m, minf[0] + 8, minf[0] + minf[1], b"dinf")
    if not dinf:
        return None
    t.dinf = (dinf[0], dinf[1])

    stbl = _find(m, minf[0] + 8, minf[0] + minf[1], b"stbl")
    if not stbl:
        return None
    s0, s1 = stbl[0] + 8, stbl[0] + stbl[1]
    stsd = _find(m, s0, s1, b"stsd")
    if not stsd:
        return None
    t.stsd = (stsd[0], stsd[1])

    b = _find(m, s0, s1, b"stts")
    if b:
        t.stts = b[0] + b[2]; t.stts_cnt = _u32(m, b[0] + b[2] + 4)
    else:
        t.stts = None; t.stts_cnt = 0
    b = _find(m, s0, s1, b"stsz")
    if not b:
        return None
    t.stsz = b[0] + b[2]
    t.stsz_sample_size = _u32(m, b[0] + b[2] + 4)
    t.stsz_cnt = _u32(m, b[0] + b[2] + 8)
    b = _find(m, s0, s1, b"stsc")
    if not b:
        return None
    t.stsc = b[0] + b[2]; t.stsc_cnt = _u32(m, b[0] + b[2] + 4)
    b = _find(m, s0, s1, b"stco")
    if b:
        t.stco = b[0] + b[2]; t.stco_cnt = _u32(m, b[0] + b[2] + 4); t.stco64 = False
    else:
        b = _find(m, s0, s1, b"co64")
        if not b:
            return None
        t.stco = b[0] + b[2]; t.stco_cnt = _u32(m, b[0] + b[2] + 4); t.stco64 = True
    b = _find(m, s0, s1, b"stss")
    if b:
        t.stss = b[0] + b[2]; t.stss_cnt = _u32(m, b[0] + b[2] + 4)
    return t


def _expand_base(m, t):
    """Expand a track's base-moov stbl into (off,size,dur,key) samples."""
    n = t.stsz_cnt
    if n == 0:
        return
    stss_i = 0
    next_key = _u32(m, t.stss + 8) if (t.stss and t.stss_cnt > 0) else 0
    stts_i = 0
    stts_remain = _u32(m, t.stts + 8) if t.stts_cnt else n
    stts_delta = _u32(m, t.stts + 12) if t.stts_cnt else 0
    stsc_i = 0
    stsc_spc = _u32(m, t.stsc + 12)
    stsc_next_first = _u32(m, t.stsc + 8 + 12) if t.stsc_cnt > 1 else 0xFFFFFFFF
    sample = 0
    chunk = 1
    while sample < n and chunk <= t.stco_cnt:
        while t.stsc_cnt > stsc_i + 1 and chunk >= stsc_next_first:
            stsc_i += 1
            stsc_spc = _u32(m, t.stsc + 12 + stsc_i * 12)
            stsc_next_first = (_u32(m, t.stsc + 8 + (stsc_i + 1) * 12)
                               if t.stsc_cnt > stsc_i + 1 else 0xFFFFFFFF)
        if t.stco64:
            run = _u64(m, t.stco + 8 + (chunk - 1) * 8)
        else:
            run = _u32(m, t.stco + 8 + (chunk - 1) * 4)
        k = 0
        while k < stsc_spc and sample < n:
            ssz = (t.stsz_sample_size if t.stsz_sample_size
                   else _u32(m, t.stsz + 12 + sample * 4))
            while stts_remain == 0 and t.stts_cnt > stts_i + 1:
                stts_i += 1
                stts_remain = _u32(m, t.stts + 8 + stts_i * 8)
                stts_delta = _u32(m, t.stts + 12 + stts_i * 8)
            dur = stts_delta
            if stts_remain:
                stts_remain -= 1
            if t.stss is None:
                key = 1
            else:
                key = 0
                if next_key == sample + 1:
                    key = 1
                    stss_i += 1
                    next_key = (_u32(m, t.stss + 8 + stss_i * 4)
                                if stss_i < t.stss_cnt else 0)
            t.samples.append((run, ssz, dur, key))
            run += ssz
            k += 1
            sample += 1
        chunk += 1


def _expand_moof(mbuf, mfoff, tracks_by_id):
    """Route each traf's samples in one moof to its track by track_ID."""
    n = len(mbuf)
    for traf_o, traf_s, _ in _find_all(mbuf, 8, n, b"traf"):
        tfhd = _find(mbuf, traf_o + 8, traf_o + traf_s, b"tfhd")
        if not tfhd:
            continue
        tf_flags = _u32(mbuf, tfhd[0] + 8) & 0xFFFFFF
        track_id = _u32(mbuf, tfhd[0] + 8 + 4)
        t = tracks_by_id.get(track_id)
        p = tfhd[0] + 8 + 4 + 4
        def_dur = t.trex_dur if t else 0
        def_size = t.trex_size if t else 0
        def_flags = t.trex_flags if t else 0
        base_off = mfoff
        if tf_flags & _TFHD_BASE_DATA_OFFSET:
            base_off = _u64(mbuf, p); p += 8
        if tf_flags & _TFHD_SAMPLE_DESC_INDEX:
            p += 4
        if tf_flags & _TFHD_DEFAULT_DURATION:
            def_dur = _u32(mbuf, p); p += 4
        if tf_flags & _TFHD_DEFAULT_SIZE:
            def_size = _u32(mbuf, p); p += 4
        if tf_flags & _TFHD_DEFAULT_FLAGS:
            def_flags = _u32(mbuf, p); p += 4

        for trun_o, trun_s, _ in _find_all(mbuf, traf_o + 8, traf_o + traf_s, b"trun"):
            tr_flags = _u32(mbuf, trun_o + 8) & 0xFFFFFF
            count = _u32(mbuf, trun_o + 12)
            rp = trun_o + 16
            data_off = 0
            first_flags = 0
            have_first = False
            if tr_flags & _TRUN_DATA_OFFSET:
                data_off = _s32(mbuf, rp); rp += 4
            if tr_flags & _TRUN_FIRST_SAMPLE_FLAGS:
                first_flags = _u32(mbuf, rp); rp += 4; have_first = True
            run = base_off + data_off
            for i in range(count):
                dur, ssz, sflags = def_dur, def_size, def_flags
                if tr_flags & _TRUN_SAMPLE_DURATION:
                    dur = _u32(mbuf, rp); rp += 4
                if tr_flags & _TRUN_SAMPLE_SIZE:
                    ssz = _u32(mbuf, rp); rp += 4
                if tr_flags & _TRUN_SAMPLE_FLAGS:
                    sflags = _u32(mbuf, rp); rp += 4
                if tr_flags & _TRUN_SAMPLE_CTS:
                    rp += 4  # ignored: no B-frames
                if i == 0 and have_first:
                    sflags = first_flags
                key = 0 if (sflags & _SF_NON_SYNC) else 1
                if t is not None:
                    t.samples.append((run, ssz, dur, key))
                run += ssz


# --------------------------------------------------------------------------- #
# Build the new complete moov
# --------------------------------------------------------------------------- #
def _build_stbl(out, m, t):
    max_off = max((s[0] for s in t.samples), default=0)
    any_nonkey = any(not s[3] for s in t.samples)
    use64 = max_off > 0xFFFFFFFF

    stbl = out.box_begin(b"stbl")
    out.raw(m[t.stsd[0]:t.stsd[0] + t.stsd[1]])   # stsd verbatim (avcC / esds)

    # stts (run-length)
    b = out.box_begin(b"stts")
    out.u32(0)
    cnt_at = len(out.a); out.u32(0)
    entries = 0
    i = 0
    nsamp = len(t.samples)
    while i < nsamp:
        dur = t.samples[i][2]; run = 0
        while i < nsamp and t.samples[i][2] == dur:
            run += 1; i += 1
        out.u32(run); out.u32(dur); entries += 1
    struct.pack_into(">I", out.a, cnt_at, entries)
    out.box_end(b)

    # stss only when there are non-sync samples (video). Audio is all-sync.
    if any_nonkey:
        b = out.box_begin(b"stss")
        out.u32(0)
        cnt_at = len(out.a); out.u32(0)
        entries = 0
        for idx, s in enumerate(t.samples):
            if s[3]:
                out.u32(idx + 1); entries += 1
        struct.pack_into(">I", out.a, cnt_at, entries)
        out.box_end(b)

    # stsc: one sample per chunk
    b = out.box_begin(b"stsc")
    out.u32(0); out.u32(1); out.u32(1); out.u32(1); out.u32(1)
    out.box_end(b)

    # stsz
    b = out.box_begin(b"stsz")
    out.u32(0); out.u32(0); out.u32(nsamp)
    for s in t.samples:
        out.u32(s[1])
    out.box_end(b)

    # stco / co64
    b = out.box_begin(b"co64" if use64 else b"stco")
    out.u32(0); out.u32(nsamp)
    for s in t.samples:
        out.u64(s[0]) if use64 else out.u32(s[0])
    out.box_end(b)

    out.box_end(stbl)


def _build_moov(m, mvhd, mvhd_ts, tracks, udta):
    out = _Buf()
    movie_dur = 0
    for t in tracks:
        media_dur = sum(s[2] for s in t.samples)
        md = (media_dur * mvhd_ts) // t.mdhd_ts if t.mdhd_ts else media_dur
        movie_dur = max(movie_dur, md)

    moov = out.box_begin(b"moov")
    _copy_patch_dur(out, m, mvhd[0], mvhd[1], 16, 24, movie_dur)
    for t in tracks:
        media_dur = sum(s[2] for s in t.samples)
        track_movie_dur = (media_dur * mvhd_ts) // t.mdhd_ts if t.mdhd_ts else media_dur
        trak = out.box_begin(b"trak")
        _copy_patch_dur(out, m, t.tkhd[0], t.tkhd[1], 20, 28, track_movie_dur)
        mdia = out.box_begin(b"mdia")
        _copy_patch_dur(out, m, t.mdhd[0], t.mdhd[1], 16, 24, media_dur)
        out.raw(m[t.hdlr[0]:t.hdlr[0] + t.hdlr[1]])
        minf = out.box_begin(b"minf")
        out.raw(m[t.mh[0]:t.mh[0] + t.mh[1]])
        out.raw(m[t.dinf[0]:t.dinf[0] + t.dinf[1]])
        _build_stbl(out, m, t)
        out.box_end(minf)
        out.box_end(mdia)
        out.box_end(trak)
    if udta:
        out.raw(m[udta[0]:udta[0] + udta[1]])
    out.box_end(moov)
    return bytes(out.a)


# --------------------------------------------------------------------------- #
# Top-level flatten
# --------------------------------------------------------------------------- #
class RepairResult:
    OK = "repaired"
    ALREADY_FLAT = "already standard"
    SKIP = "skipped"
    ERROR = "error"


def flatten_file(path, backup=False, dry_run=False, log=print):
    """Flatten one fragmented MP4 in place. Returns (RepairResult, detail)."""
    try:
        fsize = os.path.getsize(path)
    except OSError as e:
        return RepairResult.ERROR, str(e)
    if fsize < 16:
        return RepairResult.SKIP, "not an MP4"

    moov_off = moov_sz = None
    moof_offs = []
    mfra_off = None
    max_moof = 0
    try:
        with open(path, "rb") as f:
            o = 0
            while o + 8 <= fsize:
                f.seek(o)
                h = f.read(16)
                if len(h) < 8:
                    break
                sz = _u32(h, 0); hdr = 8
                if sz == 1:
                    sz = _u64(h, 8); hdr = 16
                elif sz == 0:
                    sz = fsize - o
                if sz < hdr or o + sz > fsize:
                    return RepairResult.SKIP, "malformed box (not a clean MP4)"
                typ = h[4:8]
                if typ == b"moov":
                    moov_off, moov_sz = o, sz
                elif typ == b"moof":
                    moof_offs.append(o)
                    max_moof = max(max_moof, sz)
                elif typ == b"mfra":
                    mfra_off = o
                o += sz
    except OSError as e:
        return RepairResult.ERROR, str(e)

    if moov_sz is None:
        return RepairResult.SKIP, "no moov (not an MP4)"
    if not moof_offs:
        return RepairResult.ALREADY_FLAT, ""

    if dry_run:
        return RepairResult.OK, "would repair (%d fragments)" % len(moof_offs)

    try:
        with open(path, "rb") as f:
            f.seek(moov_off)
            m = f.read(moov_sz)
            # Parse moov
            mvhd = _find(m, 8, moov_sz, b"mvhd")
            if not mvhd:
                return RepairResult.SKIP, "no mvhd"
            mver = m[mvhd[0] + 8]
            mvhd_ts = _u32(m, mvhd[0] + 8 + (20 if mver == 1 else 12))
            tracks = []
            for to, tsz, _ in _find_all(m, 8, moov_sz, b"trak"):
                t = _parse_track(m, to, tsz)
                if t is None:
                    return RepairResult.SKIP, "unparseable track"
                tracks.append(t)
            if not tracks:
                return RepairResult.SKIP, "no tracks"
            by_id = {t.track_id: t for t in tracks}
            # trex defaults
            mvex = _find(m, 8, moov_sz, b"mvex")
            if mvex:
                for tx, txs, txh in _find_all(m, mvex[0] + 8, mvex[0] + mvex[1], b"trex"):
                    tid = _u32(m, tx + txh + 4)
                    t = by_id.get(tid)
                    if t:
                        t.trex_dur = _u32(m, tx + txh + 12)
                        t.trex_size = _u32(m, tx + txh + 16)
                        t.trex_flags = _u32(m, tx + txh + 20)
            udta = _find(m, 8, moov_sz, b"udta")

            # Base samples
            for t in tracks:
                _expand_base(m, t)
            # Fragment samples
            for mo in moof_offs:
                f.seek(mo)
                h = f.read(16)
                sz = _u32(h, 0)
                if sz == 1:
                    sz = _u64(h, 8)
                if sz < 8 or sz > max_moof:
                    return RepairResult.SKIP, "bad fragment"
                f.seek(mo)
                mbuf = f.read(sz)
                _expand_moof(mbuf, mo, by_id)

        total = sum(len(t.samples) for t in tracks)
        if total == 0:
            return RepairResult.SKIP, "zero samples"

        new_moov = _build_moov(m, mvhd, mvhd_ts, tracks, udta)
    except (OSError, struct.error, IndexError) as e:
        return RepairResult.ERROR, "parse failure: %s" % e

    if backup:
        bak = path + ".bak"
        if not os.path.exists(bak):
            try:
                import shutil
                shutil.copy2(path, bak)
            except OSError as e:
                return RepairResult.ERROR, "backup failed: %s" % e

    # --- crash-safe write ---------------------------------------------------
    # 1. Append the new complete moov at EOF and fsync. Until step 2 the
    #    original front moov is authoritative → file stays a valid fragmented
    #    MP4 if interrupted here.
    # 2. Rename the stale front moov to 'free' (its mvex disappears with it, so
    #    the moofs become inert). After this fsync the only moov is the new one.
    # 3. Retire the moof/mfra boxes to 'free' too (belt-and-suspenders).
    try:
        with open(path, "r+b") as f:
            f.seek(fsize)
            f.write(new_moov)
            f.flush(); os.fsync(f.fileno())

            f.seek(moov_off + 4); f.write(b"free")
            f.flush(); os.fsync(f.fileno())

            for mo in moof_offs:
                f.seek(mo + 4); f.write(b"free")
            if mfra_off is not None:
                f.seek(mfra_off + 4); f.write(b"free")
            f.flush(); os.fsync(f.fileno())
    except OSError as e:
        return RepairResult.ERROR, "write failed: %s" % e

    return RepairResult.OK, "%d track(s), %d samples" % (
        len(tracks), sum(len(t.samples) for t in tracks))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Repair Trinet recordings that show only ~1 second in some "
                    "players/uploaders (fragmented MP4 → standard MP4).")
    ap.add_argument("folder", help="folder containing .mp4 recordings to repair")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="also process .mp4 files in subfolders")
    ap.add_argument("--backup", action="store_true",
                    help="keep a copy of each modified file as <name>.mp4.bak")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without modifying any file")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.folder):
        print("error: not a folder: %s" % args.folder, file=sys.stderr)
        return 2

    mp4s = []
    if args.recursive:
        for root, _dirs, files in os.walk(args.folder):
            for name in files:
                if name.lower().endswith(".mp4") and not name.startswith("._"):
                    mp4s.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(args.folder)):
            p = os.path.join(args.folder, name)
            if (os.path.isfile(p) and name.lower().endswith(".mp4")
                    and not name.startswith("._")):
                mp4s.append(p)
    mp4s.sort()

    if not mp4s:
        print("No .mp4 files found in %s" % args.folder)
        return 0

    counts = {RepairResult.OK: 0, RepairResult.ALREADY_FLAT: 0,
              RepairResult.SKIP: 0, RepairResult.ERROR: 0}
    for p in mp4s:
        res, detail = flatten_file(p, backup=args.backup, dry_run=args.dry_run)
        counts[res] = counts.get(res, 0) + 1
        rel = os.path.relpath(p, args.folder)
        tag = {RepairResult.OK: "FIXED " if not args.dry_run else "WOULD-FIX",
               RepairResult.ALREADY_FLAT: "ok    ",
               RepairResult.SKIP: "skip  ",
               RepairResult.ERROR: "ERROR "}[res]
        print("  [%s] %s%s" % (tag, rel, (" — " + detail) if detail else ""))

    print("\n%s: %d repaired, %d already standard, %d skipped, %d errors" % (
        "Dry run" if args.dry_run else "Done",
        counts[RepairResult.OK], counts[RepairResult.ALREADY_FLAT],
        counts[RepairResult.SKIP], counts[RepairResult.ERROR]))
    return 1 if counts[RepairResult.ERROR] else 0


if __name__ == "__main__":
    sys.exit(main())

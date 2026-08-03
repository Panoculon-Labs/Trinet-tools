#!/usr/bin/env python3
"""Ingest a Trinet SD card into per-clip delivery ZIPs.

Reads every recording on a camera's SD card, attaches the collection metadata a
delivery program requires, and writes one upload-ready ZIP per clip:

    <collector>_<date>_<device-tag>_<base>.zip
        <base>.mp4          video
        <base>.imu          inertial samples
        <base>.vts          per-frame video timestamps
        <base>.json         the camera's own recording sidecar (when present)
        metadata.json       collection + camera metadata for this clip
        README.md           how to read the files

Standard-library Python 3 only -- no pip install, no ffmpeg. Runs the same on
Windows, macOS and Linux.

Typical use on Windows:

    python scripts\\ingest_sd_card.py --drive E: ^
        --collector alice01 --country US ^
        --calibration cal\\unit-aa3d26ba.json ^
        --out D:\\deliveries

The recordings are never altered: they go into the ZIP as byte-for-byte copies
of what is on the card, and the card itself is only ever read from. Nothing is
inspected, judged or filtered -- the script adds metadata and zips, and that is
all (unless you explicitly ask for --repair).
"""

import argparse
import base64
import concurrent.futures
import datetime as _dt
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

# repair_recordings.py lives next to this script and is standard-library only.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from repair_recordings import flatten_file as _flatten_file
except Exception:                                    # pragma: no cover
    _flatten_file = None


# Environment taxonomy from the collection spec. type -> known sub-categories.
# Not enforced: any value is accepted, but one outside this list warns.
ENVIRONMENTS = {
    "residential": [
        "laundry", "kitchen_tidy", "organize_room", "other_household",
    ],
    "commercial": [
        "agriculture_landscaping_grounds", "hospitality_housekeeping",
        "automotive_service_maintenance", "food_service_back_of_house",
        "field_services_light_installation", "commercial_cleaning_janitorial",
        "retail_stocking_back_of_house", "construction_skilled_trades", "other",
    ],
}

METADATA_SCHEMA = "trinet-delivery-metadata/2"

# Sidecar extensions that travel with a clip, in ZIP order.
SIDECARS = ("imu", "vts", "json", "tel")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _i32(b, o):
    return struct.unpack_from("<i", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _i64(b, o):
    return struct.unpack_from("<q", b, o)[0]


def _be32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def _be64(b, o):
    return struct.unpack_from(">Q", b, o)[0]


def _round(x, n=3):
    return None if x is None else round(x, n)


class Log:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.warnings = []

    def info(self, msg):
        if not self.quiet:
            print(msg)

    def warn(self, msg):
        """Something the operator should act on."""
        self.warnings.append(msg)
        print("  [warn] " + msg)


# --------------------------------------------------------------------------- #
# Reading the recording
#
# What the recording contains is measured so it can be described in the
# metadata; the files themselves are always passed through untouched. The full
# byte-level specification of every format is in docs/data_formats.md.
# --------------------------------------------------------------------------- #
IMU_MAGIC = b"TRIMU001"
IMU_HEADER = 64
IMU_SAMPLE_SIZE = {1: 44, 2: 76, 3: 80, 4: 80, 5: 80, 6: 80}
ACCEL_FS_G = {0: 2, 1: 4, 2: 8, 3: 16}
GYRO_FS_DPS = {0: 250, 1: 500, 2: 1000, 3: 2000}

VTS_MAGIC = b"TRIVTS01"
VTS_HEADER = 32
VTS_ENTRY_SIZE = {1: 12, 2: 24, 3: 24, 4: 36}


def read_imu(path):
    """Device id, sensors, ranges, sample rate and span from a .imu sidecar."""
    size = os.path.getsize(path)
    if size < IMU_HEADER:
        return None
    with open(path, "rb") as f:
        h = f.read(IMU_HEADER)
        if h[:8] != IMU_MAGIC:
            return None
        version = _u32(h, 8)
        ssize = IMU_SAMPLE_SIZE.get(version)
        if not ssize:
            return {"version": version, "unsupported": True}
        n = (size - IMU_HEADER) // ssize
        flags = _u32(h, 36) if version >= 3 else 0

        # The device id occupies bytes that were reserved (zero) before v3, so
        # an older recording reads as "unknown" rather than as garbage.
        devid = h[40:56].hex() if version >= 3 else ""
        info = {
            "version": version,
            "samples": n,
            "device_id": "" if devid == "0" * 32 else devid,
            "nominal_rate_hz": _u32(h, 12),
            "accel_range_g": ACCEL_FS_G.get(_u16(h, 16)),
            "gyro_range_dps": GYRO_FS_DPS.get(_u16(h, 18)),
            "frame_sync": bool(flags & 0x1),
            "magnetometer": bool(flags & 0x2),
            "duration_s": 0.0,
            "actual_rate_hz": None,
        }
        if n >= 2:
            first = _u64(f.read(8), 0)
            f.seek(IMU_HEADER + (n - 1) * ssize)
            last = _u64(f.read(8), 0)
            span = (last - first) / 1e9
            info["duration_s"] = span
            if span > 0:
                info["actual_rate_hz"] = (n - 1) / span
    return info


def read_vts(path):
    """Frame count, measured fps and length from the per-frame timestamps."""
    size = os.path.getsize(path)
    if size < VTS_HEADER:
        return None
    with open(path, "rb") as f:
        h = f.read(VTS_HEADER)
        if h[:8] != VTS_MAGIC:
            return None
        version = _u32(h, 8)
        esize = VTS_ENTRY_SIZE.get(version)
        if not esize:
            return {"version": version, "unsupported": True}
        n = (size - VTS_HEADER) // esize
        info = {
            "version": version,
            "frames": n,
            "nominal_fps": _u32(h, 12) / 1000.0,
            "duration_s": 0.0,
            "avg_fps": None,
        }
        if n < 2:
            return info
        body = f.read(n * esize)

    # A zero timestamp means "unavailable" for that frame; skip those.
    stamps = [t for t in (_u64(body, i * esize + 4) for i in range(n)) if t]
    info["timestamped_frames"] = len(stamps)
    if len(stamps) >= 2:
        span = (stamps[-1] - stamps[0]) / 1e9
        info["duration_s"] = span
        if span > 0:
            info["avg_fps"] = (len(stamps) - 1) / span
    return info


# -- MP4 container + H.264 bitstream -------------------------------------------
# The video specs the delivery program cares about -- codec, resolution, GOP,
# whether B-frames are present, colour depth -- are read straight from the MP4
# and the H.264 bitstream, with no external tools. GOP and B-frame presence are
# properties of the encoder, so a bounded scan of the first frames settles them.
CODEC_FOURCC = {b"avc1": "h264", b"avc3": "h264",
                b"hvc1": "h265", b"hev1": "h265"}
_H264_SLICE_NALS = (1, 5)          # coded slice (non-IDR / IDR)
_H264_PROFILE_HAS_BITDEPTH = {100, 110, 122, 244, 44, 83, 86, 118, 128,
                              138, 139, 134, 135}
_SLICE_SCAN_CAP = 4000             # frames; enough to fix GOP/B-frames


def _rbsp(nal):
    """Strip H.264 emulation-prevention 0x03 bytes."""
    out = bytearray()
    i, n = 0, len(nal)
    while i < n:
        if i + 2 < n and nal[i] == 0 and nal[i + 1] == 0 and nal[i + 2] == 3:
            out += nal[i:i + 2]
            i += 3
        else:
            out.append(nal[i])
            i += 1
    return bytes(out)


class _BitReader:
    def __init__(self, data):
        self.d = data
        self.pos = 0

    def bit(self):
        b = (self.d[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v

    def ue(self):
        z = 0
        while self.bit() == 0:
            z += 1
        return (1 << z) - 1 + (self.bits(z) if z else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def _parse_sps(sps):
    """Colour bit depth and profile from an H.264 SPS NAL."""
    r = _BitReader(_rbsp(sps[1:]))             # skip the NAL header byte
    profile = r.bits(8)
    r.bits(8)                                  # constraint flags + reserved
    r.bits(8)                                  # level
    r.ue()                                     # seq_parameter_set_id
    bit_depth = 8
    if profile in _H264_PROFILE_HAS_BITDEPTH:
        chroma = r.ue()
        if chroma == 3:
            r.bit()
        bit_depth = r.ue() + 8                 # bit_depth_luma_minus8
    return {"profile": profile, "color_depth_bits": bit_depth}


def probe_video(paths):
    """Codec, resolution, GOP, B-frame presence and colour depth for a clip.

    `paths` is the clip's MP4 file(s) (more than one only for chunked takes).
    Returns a dict; keys that need the H.264 bitstream are absent for other
    codecs. Never raises -- unreadable files yield an empty-ish result.
    """
    out = {"file_bytes": sum(_safe_size(p) for p in paths)}
    if len(paths) > 1:
        out["parts"] = len(paths)

    nal_len, sps, codec = 4, None, None
    first = paths[0]
    try:
        size = _safe_size(first)
        with open(first, "rb") as f:
            moov = next((r for t, *r in
                         ((t, s, e) for t, s, e in _boxes(f, 0, size))
                         if t == b"moov"), None)
            if moov:
                for st, en in _descend(f, moov[0], moov[1],
                                       (b"trak", b"mdia", b"minf",
                                        b"stbl", b"stsd")):
                    f.seek(st + 8)
                    entry = f.read(min(320, en - st - 8))
                    if len(entry) < 36:
                        continue
                    codec = CODEC_FOURCC.get(entry[4:8])
                    if not codec:
                        continue
                    out["codec"] = codec
                    out["width"] = _u16be(entry, 32)
                    out["height"] = _u16be(entry, 34)
                    idx = entry.find(b"avcC")
                    if idx >= 0:
                        av = entry[idx + 4:]
                        nal_len = (av[4] & 3) + 1
                        if av[5] & 0x1f:
                            L = _u16be(av, 6)
                            sps = av[8:8 + L]
                    break
    except OSError:
        return out

    if sps:
        try:
            out.update(_parse_sps(sps))
        except (IndexError, ValueError):
            pass

    # GOP + B-frames need slice types, so walk the coded-slice NALs. Every mdat
    # (one per fragment in a fragmented file) is a run of length-prefixed NALs.
    if codec == "h264":
        idr, frames, has_b, capped = [], 0, False, False
        try:
            for p in paths:
                if capped:
                    break
                psize = _safe_size(p)
                with open(p, "rb") as f:
                    for t, st, en in _boxes(f, 0, psize):
                        if t != b"mdat":
                            continue
                        o = st
                        while o + nal_len <= en:
                            f.seek(o)
                            lb = f.read(nal_len)
                            if len(lb) < nal_len:
                                break
                            L = int.from_bytes(lb, "big")
                            if L == 0 or o + nal_len + L > en:
                                break
                            f.seek(o + nal_len)
                            sl = f.read(min(16, L))
                            if sl and (sl[0] & 0x1f) in _H264_SLICE_NALS:
                                ntype = sl[0] & 0x1f
                                br = _BitReader(_rbsp(sl[1:]))
                                br.ue()                 # first_mb_in_slice
                                if br.ue() % 5 == 1:    # slice_type B
                                    has_b = True
                                if ntype == 5:
                                    idr.append(frames)
                                frames += 1
                                if frames >= _SLICE_SCAN_CAP:
                                    capped = True
                                    break
                            o += nal_len + L
                        if capped:
                            break
        except OSError:
            pass
        out["b_frames"] = has_b
        out["frames_scanned"] = frames
        if capped:
            out["scan_capped"] = True
        if len(idr) >= 2:
            gaps = [idr[i + 1] - idr[i] for i in range(len(idr) - 1)]
            out["gop_length"] = max(set(gaps), key=gaps.count)
    return out


def _safe_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def container_status(path):
    """Cheap MP4 integrity check, matching the delivery ingest's box audit.

    Returns (status, detail): "ok", "truncated" (a box runs past EOF -- the
    recording was cut mid-write), "no_video" (no moov/moof), "empty", or
    "broken". Follows declared box sizes so an over-long final box is caught.
    """
    size = _safe_size(path)
    if size < 16:
        return "empty", "file is %d bytes" % size
    try:
        with open(path, "rb") as f:
            o = 0
            last_end = 0
            saw_moov = saw_moof = False
            while o + 8 <= size:
                f.seek(o)
                head = f.read(16)
                if len(head) < 8:
                    break
                sz = _be32(head, 0)
                hdr = 8
                if sz == 1:
                    if len(head) < 16:
                        break
                    sz = _be64(head, 8)
                    hdr = 16
                elif sz == 0:
                    sz = size - o
                if sz < hdr:
                    return "broken", "bad box size %d at offset %d" % (sz, o)
                typ = head[4:8]
                if typ == b"moov":
                    saw_moov = True
                elif typ == b"moof":
                    saw_moof = True
                last_end = o + sz
                o += sz
    except OSError as e:
        return "broken", str(e)
    if last_end != size:
        return "truncated", ("last box ends at %d of %d bytes"
                             % (last_end, size))
    if not (saw_moov or saw_moof):
        return "no_video", "no moov/moof box"
    return "ok", None


# -- ffmpeg re-encode ----------------------------------------------------------
def ffmpeg_path():
    return shutil.which("ffmpeg")


# Candidate H.264 encoders, fastest first. Each is verified to actually work on
# this machine before use (compiled-in != usable -- e.g. NVENC with no GPU).
_H264_ENCODERS = [
    ("h264_nvenc", ["-preset", "p1", "-g", "30", "-bf", "0"]),
    ("h264_videotoolbox", ["-g", "30", "-bf", "0"]),   # macOS hardware
    ("libx264", ["-preset", "veryfast", "-g", "30", "-bf", "0", "-threads", "0"]),
]
_ENCODER_CACHE = {}


def _encoder_works(ff, name):
    """One-frame smoke test so a compiled-but-unusable encoder is skipped."""
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", name, "-frames:v", "1", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def _pick_h264_encoder():
    """Pick the fastest H.264 encoder that actually runs here. Cached.
    Returns (encoder_name, extra_args) with GOP 30 / no B-frames."""
    if "enc" in _ENCODER_CACHE:
        return _ENCODER_CACHE["enc"]
    ff = ffmpeg_path()
    chosen = _H264_ENCODERS[-1]                          # libx264 fallback
    if ff:
        for name, extra in _H264_ENCODERS:
            if name == "libx264" or _encoder_works(ff, name):
                chosen = (name, extra)
                break
    _ENCODER_CACHE["enc"] = chosen
    return chosen


def reencode_video(src, dst, target_mbps, log):
    """Transcode src -> dst as spec-compliant H.264 <=8 Mbps. Returns bool.
    Falls back to libx264 if the chosen (hardware) encoder fails at runtime."""
    ff = ffmpeg_path()
    if not ff:
        return False
    target = max(1.0, min(float(target_mbps), 8.0))     # keep inside 6-8 spec
    enc, extra = _pick_h264_encoder()

    def attempt(encoder, encoder_extra):
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
               "-fflags", "+genpts+discardcorrupt", "-i", src,
               "-map", "0:v:0", "-c:v", encoder,
               "-b:v", "%dk" % int(target * 1000),
               "-maxrate", "8000k", "-bufsize", "8000k"]
        cmd += encoder_extra
        cmd += ["-pix_fmt", "yuv420p",                   # 8-bit, no HDR
                "-map", "0:a?", "-c:a", "copy",          # keep audio if any
                "-movflags", "+faststart", dst]
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=3600)
        except Exception as e:                           # pragma: no cover
            return False, str(e)
        if r.returncode != 0 or not os.path.isfile(dst) or _safe_size(dst) == 0:
            tail = (r.stderr.decode("utf-8", "replace").strip().splitlines()
                    or [""])[-1] if r.stderr else ""
            return False, tail
        return True, None

    ok, err = attempt(enc, extra)
    if ok:
        return True
    if enc != "libx264":                                 # hardware failed -> CPU
        log.warn("%s: %s failed (%s); retrying with libx264"
                 % (os.path.basename(src), enc, err))
        _ENCODER_CACHE["enc"] = _H264_ENCODERS[-1]       # stop trying hardware
        lib, lib_extra = _H264_ENCODERS[-1]
        ok, err = attempt(lib, lib_extra)
        if ok:
            return True
    log.warn("ffmpeg could not re-encode %s%s"
             % (os.path.basename(src), ": " + err if err else ""))
    return False


def mp4_duration(path):
    """Container duration, used only when no .vts or .imu is present."""
    size = _safe_size(path)
    try:
        with open(path, "rb") as f:
            moov = None
            for typ, st, en in _boxes(f, 0, size):
                if typ == b"moov":
                    moov = (st, en)
                    break
            if not moov:
                return None
            for st, en in _descend(f, moov[0], moov[1], (b"mvhd",)):
                f.seek(st)
                b = f.read(min(32, en - st))
                if len(b) < 20:
                    return None
                if b[0] == 1 and len(b) >= 32:
                    ts, dur = _be32(b, 20), _be64(b, 24)
                else:
                    ts, dur = _be32(b, 12), _be32(b, 16)
                return (dur / ts) if ts else None
    except OSError:
        return None
    return None


def _u16be(b, o):
    return struct.unpack_from(">H", b, o)[0]


def _boxes(f, start, end):
    """Yield (type, payload_start, payload_end) for boxes in [start, end)."""
    o = start
    while o + 8 <= end:
        f.seek(o)
        head = f.read(16)
        if len(head) < 8:
            return
        sz = _be32(head, 0)
        hdr = 8
        if sz == 1:
            if len(head) < 16:
                return
            sz = _be64(head, 8)
            hdr = 16
        elif sz == 0:
            sz = end - o
        if sz < hdr or o + sz > end:
            return
        yield head[4:8], o + hdr, o + sz
        o += sz


def _descend(f, start, end, path):
    """Walk a box path, e.g. (b'moov', b'mvhd'). Yields leaf ranges."""
    if not path:
        yield start, end
        return
    for typ, st, en in _boxes(f, start, end):
        if typ == path[0]:
            for r in _descend(f, st, en, path[1:]):
                yield r


# --------------------------------------------------------------------------- #
# Per-sample / per-frame iteration (used to build the MCAP)
# --------------------------------------------------------------------------- #
def iter_imu_samples(path):
    """Yield (timestamp_ns, (ax,ay,az), (gx,gy,gz), (mx,my,mz)) per sample."""
    size = os.path.getsize(path)
    if size < IMU_HEADER:
        return
    with open(path, "rb") as f:
        h = f.read(IMU_HEADER)
        if h[:8] != IMU_MAGIC:
            return
        ssize = IMU_SAMPLE_SIZE.get(_u32(h, 8))
        if not ssize or ssize < 44:
            return
        n = (size - IMU_HEADER) // ssize
        f.seek(IMU_HEADER)
        for _ in range(n):
            rec = f.read(ssize)
            if len(rec) < ssize:
                break
            ts = _u64(rec, 0)
            ax, ay, az = struct.unpack_from("<3f", rec, 8)
            gx, gy, gz = struct.unpack_from("<3f", rec, 20)
            mx, my, mz = struct.unpack_from("<3f", rec, 32)
            yield ts, (ax, ay, az), (gx, gy, gz), (mx, my, mz)


def iter_vts_frames(path):
    """Yield sof_timestamp_ns for each frame entry (0 if unavailable)."""
    size = os.path.getsize(path)
    if size < VTS_HEADER:
        return
    with open(path, "rb") as f:
        h = f.read(VTS_HEADER)
        if h[:8] != VTS_MAGIC:
            return
        esize = VTS_ENTRY_SIZE.get(_u32(h, 8))
        if not esize:
            return
        n = (size - VTS_HEADER) // esize
        body = f.read(n * esize)
    for i in range(n):
        yield _u64(body, i * esize + 4)


def read_avcc(path):
    """SPS/PPS NAL lists and NAL length size from the MP4's avcC record."""
    size = _safe_size(path)
    try:
        with open(path, "rb") as f:
            moov = next((r for t, *r in
                         ((t, s, e) for t, s, e in _boxes(f, 0, size))
                         if t == b"moov"), None)
            if not moov:
                return None
            for st, en in _descend(f, moov[0], moov[1],
                                   (b"trak", b"mdia", b"minf",
                                    b"stbl", b"stsd")):
                f.seek(st + 8)
                entry = f.read(min(400, en - st - 8))
                idx = entry.find(b"avcC")
                if idx < 0:
                    continue
                av = entry[idx + 4:]
                nal_len = (av[4] & 3) + 1
                o = 5
                nsps = av[o] & 0x1f
                o += 1
                sps = []
                for _ in range(nsps):
                    L = _u16be(av, o)
                    o += 2
                    sps.append(av[o:o + L])
                    o += L
                npps = av[o]
                o += 1
                pps = []
                for _ in range(npps):
                    L = _u16be(av, o)
                    o += 2
                    pps.append(av[o:o + L])
                    o += L
                return {"nal_len": nal_len, "sps": sps, "pps": pps}
    except (OSError, IndexError, struct.error):
        return None
    return None


_ANNEXB_START = b"\x00\x00\x00\x01"


def iter_h264_access_units(paths, avcc):
    """Yield (annexb_bytes, is_keyframe) per frame, in decode order.

    Trinet frames are single-slice with no B-frames, so decode order equals
    display order and each VCL NAL (type 1/5) delimits one frame. SPS/PPS are
    prepended to every IDR so a decoder can start on any keyframe.
    """
    if not avcc:
        return
    nal_len = avcc["nal_len"]
    param_sets = b"".join(_ANNEXB_START + n for n in avcc["sps"] + avcc["pps"])
    pending = []                       # non-VCL NALs seen since last frame
    for p in paths:
        psize = _safe_size(p)
        try:
            with open(p, "rb") as f:
                for t, st, en in _boxes(f, 0, psize):
                    if t != b"mdat":
                        continue
                    o = st
                    while o + nal_len <= en:
                        f.seek(o)
                        lb = f.read(nal_len)
                        if len(lb) < nal_len:
                            break
                        L = int.from_bytes(lb, "big")
                        if L == 0 or o + nal_len + L > en:
                            break
                        f.seek(o + nal_len)
                        nal = f.read(L)
                        o += nal_len + L
                        if not nal:
                            continue
                        ntype = nal[0] & 0x1f
                        if ntype in (1, 5):            # VCL slice = one frame
                            idr = ntype == 5
                            au = bytearray()
                            if idr:
                                au += param_sets
                            for nn in pending:
                                au += _ANNEXB_START + nn
                            au += _ANNEXB_START + nal
                            pending = []
                            yield bytes(au), idr
                        elif ntype in (7, 8, 9, 6):    # SPS/PPS/AUD/SEI
                            pending.append(nal)
        except OSError:
            pending = []
            continue


# --------------------------------------------------------------------------- #
# MCAP writer (self-contained; no external library)
#
# MCAP is a typed-record container: an 8-byte magic, a Header record, then
# Schema / Channel / Message / Attachment / Metadata records, a DataEnd, and a
# Footer, closed by the magic again. This writes an un-chunked, un-indexed file
# (a valid MCAP per the spec -- the summary section is optional); readers build
# their own index on load. Format reference: https://mcap.dev/spec
# --------------------------------------------------------------------------- #
MCAP_MAGIC = b"\x89MCAP0\r\n"


class McapWriter:
    def __init__(self, fileobj):
        self.f = fileobj
        self.f.write(MCAP_MAGIC)
        self._next_schema = 1
        self._next_channel = 0
        self._msg_count = 0

    # -- record framing --------------------------------------------------- #
    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<I", len(b)) + b

    @staticmethod
    def _map(d):
        body = b"".join(McapWriter._str(k) + McapWriter._str(v)
                        for k, v in d.items())
        return struct.pack("<I", len(body)) + body

    def _record(self, op, body):
        self.f.write(struct.pack("<BQ", op, len(body)))
        self.f.write(body)

    # -- records ---------------------------------------------------------- #
    def header(self, profile, library):
        self._record(0x01, self._str(profile) + self._str(library))

    def add_schema(self, name, encoding, data):
        sid = self._next_schema
        self._next_schema += 1
        body = (struct.pack("<H", sid) + self._str(name) + self._str(encoding)
                + struct.pack("<I", len(data)) + data)
        self._record(0x03, body)
        return sid

    def add_channel(self, schema_id, topic, message_encoding, metadata=None):
        cid = self._next_channel
        self._next_channel += 1
        body = (struct.pack("<HH", cid, schema_id) + self._str(topic)
                + self._str(message_encoding) + self._map(metadata or {}))
        self._record(0x04, body)
        return cid

    def message(self, channel_id, log_time, data, sequence=0, publish_time=None):
        if publish_time is None:
            publish_time = log_time
        head = struct.pack("<HIQQ", channel_id, sequence, log_time, publish_time)
        self._record(0x05, head + data)
        self._msg_count += 1

    def attachment(self, name, media_type, data, log_time=0):
        body = (struct.pack("<QQ", log_time, log_time) + self._str(name)
                + self._str(media_type) + struct.pack("<Q", len(data)) + data
                + struct.pack("<I", 0))       # crc 0 = not computed
        self._record(0x09, body)

    def metadata(self, name, entries):
        self._record(0x0C, self._str(name) + self._map(entries))

    def finish(self):
        self._record(0x0F, struct.pack("<I", 0))            # DataEnd, crc 0
        self._record(0x02, struct.pack("<QQI", 0, 0, 0))    # Footer, no summary
        self.f.write(MCAP_MAGIC)


# JSON Schemas embedded in the MCAP so it is self-describing.
_IMU_SCHEMA = json.dumps({
    "type": "object",
    "title": "trinet.Imu",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "linear_acceleration": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}}, "description": "m/s^2, gravity included"},
        "angular_velocity": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}}, "description": "rad/s"},
    },
}).encode("utf-8")

_VIDEO_SCHEMA = json.dumps({
    "type": "object",
    "title": "foxglove.CompressedVideo",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}).encode("utf-8")


def _ns_time(ns):
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


def write_mcap(path, clip, meta, calibration_path, log):
    """Write one clip as an MCAP: /imu messages, /camera CompressedVideo
    messages, a metadata record, and the metadata / calibration attachments."""
    imu_paths = clip.sidecars.get("imu", [])
    vts_paths = clip.sidecars.get("vts", [])
    avcc = read_avcc(clip.mp4s[0]) if clip.video.get("codec") == "h264" else None

    with open(path, "wb") as fh:
        w = McapWriter(fh)
        w.header("trinet", "trinet-ingest")

        # -- IMU messages -------------------------------------------------- #
        imu_written = 0
        if imu_paths:
            sid = w.add_schema("trinet.Imu", "jsonschema", _IMU_SCHEMA)
            cid = w.add_channel(sid, "/imu", "json",
                                {"units": "accel m/s^2 (incl. gravity), "
                                          "gyro rad/s; timestamps monotonic ns"})
            seq = 0
            for path_i in imu_paths:
                for ts, acc, gyr, _mag in iter_imu_samples(path_i):
                    msg = {
                        "timestamp": _ns_time(ts),
                        "linear_acceleration": {"x": acc[0], "y": acc[1],
                                                "z": acc[2]},
                        "angular_velocity": {"x": gyr[0], "y": gyr[1],
                                             "z": gyr[2]},
                    }
                    w.message(cid, ts,
                              json.dumps(msg, separators=(",", ":")).encode(),
                              sequence=seq)
                    seq += 1
                    imu_written += 1

        # -- Video messages (foxglove.CompressedVideo) --------------------- #
        vid_written = 0
        if avcc:
            sof = [t for p in vts_paths for t in iter_vts_frames(p)]
            sid = w.add_schema("foxglove.CompressedVideo", "jsonschema",
                               _VIDEO_SCHEMA)
            cid = w.add_channel(sid, "/camera", "json")
            nominal = (clip.vts or {}).get("nominal_fps") or 30.0
            period = int(1e9 / nominal) if nominal else 33_333_333
            prev = sof[0] if sof and sof[0] else 0
            for i, (au, _idr) in enumerate(iter_h264_access_units(clip.mp4s,
                                                                  avcc)):
                ts = sof[i] if i < len(sof) and sof[i] else prev + period
                prev = ts
                msg = {
                    "timestamp": _ns_time(ts),
                    "frame_id": "camera",
                    "data": base64.b64encode(au).decode("ascii"),
                    "format": "h264",
                }
                w.message(cid, ts,
                          json.dumps(msg, separators=(",", ":")).encode(),
                          sequence=i)
                vid_written += 1

        # -- Metadata record + attachments --------------------------------- #
        flat = {}
        _flatten_json(meta, "", flat)
        w.metadata("trinet", flat)
        w.attachment("metadata.json", "application/json",
                     (json.dumps(meta, indent=2) + "\n").encode("utf-8"))
        if calibration_path and os.path.isfile(calibration_path):
            try:
                with open(calibration_path, "rb") as cf:
                    w.attachment(os.path.basename(calibration_path),
                                 "application/json", cf.read())
            except OSError:
                pass

        w.finish()

    log.info("    mcap: %d imu + %d video messages" % (imu_written, vid_written))
    if avcc and vid_written and clip.vts and \
            clip.vts.get("frames") and vid_written != clip.vts["frames"]:
        log.warn("%s: mcap video frames (%d) != .vts frames (%d)"
                 % (clip.base, vid_written, clip.vts["frames"]))


def _flatten_json(obj, prefix, out):
    """Flatten nested metadata into dotted string key/values for the MCAP
    Metadata record (which is a flat string->string map)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_json(v, "%s.%s" % (prefix, k) if prefix else k, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, separators=(",", ":"))
    elif obj is None:
        out[prefix] = ""
    else:
        out[prefix] = str(obj)


# --------------------------------------------------------------------------- #
# Calibration -> intrinsics, extrinsics, field of view
# --------------------------------------------------------------------------- #
def _theta_equidistant(r_norm, dist):
    """Invert r = theta*(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8) for theta."""
    k = list(dist[:4]) + [0.0] * (4 - len(dist[:4]))
    th = r_norm
    for _ in range(50):
        t2 = th * th
        poly = 1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))
        dpoly = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
        fx_ = th * poly - r_norm
        if abs(dpoly) < 1e-12:
            break
        step = fx_ / dpoly
        th -= step
        if abs(step) < 1e-12:
            break
    return th


def diagonal_fov_deg(intr):
    """Full diagonal FOV. Returns (degrees, method) or (None, reason)."""
    try:
        w, h = intr["image_size"]
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
    except (KeyError, TypeError, ValueError):
        return None, "incomplete intrinsics"
    if not fx or not fy:
        return None, "zero focal length"
    model = str(intr.get("model", "")).lower()
    dist = [float(d) for d in intr.get("distortion", [])]
    fisheye = model in ("equidistant", "fisheye", "kannala_brandt", "kb4")

    def angle(u, v):
        r = math.hypot((u - cx) / fx, (v - cy) / fy)
        if fisheye:
            return _theta_equidistant(r, dist)
        return math.atan(r)

    # Opposite corners, so an off-centre principal point is handled correctly.
    d1 = angle(0, 0) + angle(w, h)
    d2 = angle(w, 0) + angle(0, h)
    method = "equidistant_newton" if fisheye else "pinhole_linear_approx"
    return math.degrees(max(d1, d2)), method


def load_calibration(path, cam_index, log):
    """Load a calibration.json in either of the two formats the pipeline emits.

    - Flat single-camera: {"intrinsics": {...}, "extrinsics": {"T_cam_imu": ...},
      "imu": {...}}  -- the per-unit / batch Trinet-Calibration output.
    - Multi-camera:   {"cameras": [ {intrinsics, ...}, ... ], "T_cam0_imu": ...}
      -- the stereo layout; `cam_index` selects the camera.
    """
    with open(path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    if cal.get("cameras"):
        cams = cal["cameras"]
        if cam_index >= len(cams):
            raise ValueError("camera index %d out of range (file has %d)"
                             % (cam_index, len(cams)))
        cam = cams[cam_index]
        intr = dict(cam.get("intrinsics") or {})
        t_cam_imu = cal.get("T_cam0_imu") if cam_index == 0 else None
        timeshift = cam.get("timeshift_cam_imu_s")
        rms = cam.get("reprojection_rms_px")
    elif cal.get("intrinsics"):
        intr = dict(cal["intrinsics"])
        ext = cal.get("extrinsics") or {}
        t_cam_imu = ext.get("T_cam_imu")
        imu = cal.get("imu") or {}
        timeshift = (ext.get("timeshift_cam_imu_s")
                     or ext.get("timeshift_cam_imu_sec")
                     or ext.get("time_offset_s")
                     or imu.get("timeshift_cam_imu_s"))
        rms = (intr.get("reprojection_rms_px")
               or (intr.get("qa") or {}).get("reprojection_rms_mean_px"))
    else:
        raise ValueError("calibration has neither 'cameras' nor 'intrinsics'")

    # Prefer the calibration's own diagonal FOV; fall back to computing it.
    fov = _round((intr.get("fov_deg") or {}).get("diagonal"), 1)
    method = "from calibration file"
    if fov is None:
        fov_c, method = diagonal_fov_deg(intr)
        fov = _round(fov_c, 1)
        if fov is None:
            log.warn("could not determine field of view: %s" % method)

    out = {
        "source_file": os.path.basename(path),
        "camera_index": cam_index,
        "intrinsics": intr,
        "diagonal_fov_deg": fov,
        "diagonal_fov_method": method if fov is not None else None,
        "timeshift_cam_imu_s": timeshift,
        "reprojection_rms_px": rms,
    }
    if t_cam_imu:
        out["T_cam_imu"] = t_cam_imu
        out["T_cam_imu_note"] = (
            "4x4 row-major camera-from-inertial transform; translation in metres"
        )
    if cal.get("T_cam1_cam0"):
        out["T_cam1_cam0"] = cal["T_cam1_cam0"]
    return out


def load_head_transform(path):
    """Mount geometry: camera pose in the head frame."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "T_head_cam" in d:
        return {"T_head_cam": d["T_head_cam"], "source": os.path.basename(path)}
    rot = d.get("rotation_deg")
    trs = d.get("translation_m")
    if rot is None and trs is None:
        raise ValueError(
            "head transform needs 'T_head_cam', or 'rotation_deg' + "
            "'translation_m'"
        )
    return {
        "rotation_deg": rot,
        "rotation_order": d.get("rotation_order", "xyz_intrinsic"),
        "translation_m": trs,
        "source": os.path.basename(path),
    }


# --------------------------------------------------------------------------- #
# Clip discovery
# --------------------------------------------------------------------------- #
class Clip:
    """One recording: a flat file set, or a directory of sequential parts."""

    def __init__(self, base, mp4s, sidecar_map, chunked):
        self.base = base
        self.mp4s = mp4s                 # [path, ...] (>1 only when chunked)
        self.sidecars = sidecar_map      # {"imu": [path...], ...}
        self.chunked = chunked
        self.imu = None
        self.vts = None
        self.video = {}
        self.container = ("ok", None)
        self.meta_sidecar = {}
        self.device_id_conflict = None

    # -- parsing ----------------------------------------------------------- #
    def parse(self, log):
        self.video = probe_video(self.mp4s)
        # Container integrity: worst status across the clip's part(s).
        self.container = ("ok", None)
        for p in self.mp4s:
            st, detail = container_status(p)
            if st != "ok":
                self.container = (st, detail)
                break

        for path in self.sidecars.get("imu", []):
            got = read_imu(path)
            if got and not got.get("unsupported"):
                self.imu = self._merge(self.imu, got)
            elif got:
                log.warn("%s: unsupported .imu version %d"
                         % (self.base, got["version"]))

        for path in self.sidecars.get("vts", []):
            got = read_vts(path)
            if got and not got.get("unsupported"):
                self.vts = self._merge(self.vts, got)
            elif got:
                log.warn("%s: unsupported .vts version %d"
                         % (self.base, got["version"]))

        for path in self.sidecars.get("json", []):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.meta_sidecar = json.load(f)
                break
            except (OSError, ValueError):
                log.warn("%s: recording sidecar is unreadable" % self.base)

        # Rates are recomputed from the merged totals, not taken from whichever
        # chunk happened to be read first.
        if self.vts and self.vts.get("duration_s"):
            n = self.vts.get("timestamped_frames") or self.vts.get("frames", 0)
            if n >= 2:
                self.vts["avg_fps"] = (n - 1) / self.vts["duration_s"]
        if self.imu and self.imu.get("duration_s"):
            n = self.imu.get("samples", 0)
            if n >= 2:
                self.imu["actual_rate_hz"] = (n - 1) / self.imu["duration_s"]

        # The camera writes its id into both the .imu header and the sidecar.
        # If they disagree, the card holds files from more than one unit.
        sidecar_id = self.meta_sidecar.get("device_id") or ""
        header_id = (self.imu or {}).get("device_id") or ""
        if sidecar_id and header_id and sidecar_id != header_id:
            self.device_id_conflict = sidecar_id
            log.warn("%s: .imu header (%s) and recording sidecar (%s) disagree "
                     "on device id -- using the .imu header"
                     % (self.base, header_id[:8], sidecar_id[:8]))

    @staticmethod
    def _merge(acc, got):
        """Add a chunked part's sidecar onto the running clip total."""
        if acc is None:
            return got
        for k in ("samples", "frames", "timestamped_frames"):
            if k in acc or k in got:
                acc[k] = (acc.get(k) or 0) + (got.get(k) or 0)
        acc["duration_s"] = (acc.get("duration_s") or 0.0) + \
                            (got.get("duration_s") or 0.0)
        return acc

    # -- derived ----------------------------------------------------------- #
    @property
    def duration_s(self):
        """Per-frame timestamps first, then inertial span, then the container."""
        for src in (self.vts, self.imu):
            if src and src.get("duration_s"):
                return src["duration_s"]
        total = 0.0
        for p in self.mp4s:
            total += mp4_duration(p) or 0.0
        return total or None

    @property
    def bitrate_mbps(self):
        d = self.duration_s
        b = self.video.get("file_bytes")
        if not d or not b:
            return None
        return b * 8 / d / 1e6

    @property
    def device_id(self):
        """The unit that recorded this clip.

        The .imu header is authoritative: it is written by the camera into the
        recording itself. The .json sidecar is a convenience copy and is only
        used when the header predates the field (all-zero) or is missing.
        """
        if self.imu and self.imu.get("device_id"):
            return self.imu["device_id"]
        return self.meta_sidecar.get("device_id", "")

    @property
    def device_tag(self):
        did = self.device_id
        if did:
            return did[:8]
        return self.meta_sidecar.get("device_tag") or "unknown"

    def all_files(self):
        out = list(self.mp4s)
        for ext in SIDECARS:
            out.extend(self.sidecars.get(ext, []))
        return out


def discover(root, log):
    """Find clips under root. A clip exists where an .mp4 exists."""
    clips = []

    def sidecars_for(directory, stem):
        found = {}
        for ext in SIDECARS:
            p = os.path.join(directory, stem + "." + ext)
            if os.path.isfile(p):
                found[ext] = [p]
        return found

    entries = sorted(os.listdir(root))

    # Flat file sets: <base>.mp4 in the recordings folder.
    for name in entries:
        path = os.path.join(root, name)
        if not os.path.isfile(path) or not name.lower().endswith(".mp4"):
            continue
        stem = name[:-4]
        clips.append(Clip(stem, [path], sidecars_for(root, stem), False))

    # Chunked sessions: <base>/part001.mp4, part002.mp4, ...
    for name in entries:
        d = os.path.join(root, name)
        if not _is_chunk_dir(d):
            continue
        parts = sorted(
            x for x in os.listdir(d) if x.lower().endswith(".mp4")
        )
        mp4s = [os.path.join(d, p) for p in parts]
        side = {}
        for p in parts:
            for ext, paths in sidecars_for(d, p[:-4]).items():
                side.setdefault(ext, []).extend(paths)
        clips.append(Clip(name, mp4s, side, True))
        log.info("  %s: chunked session, %d parts" % (name, len(mp4s)))

    return clips


def find_recordings_dir(root, explicit=None):
    """Locate the recordings folder on a card (default name, or auto-detect).

    The folder is either the card root itself, or one folder below it -- the
    camera writes into a named folder whose label can be customised per batch,
    so it is found by content rather than by name.
    """
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(root, explicit)
        return p if os.path.isdir(p) else None
    if _holds_recordings(root):
        return root
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    for name in names:
        d = os.path.join(root, name)
        if os.path.isdir(d) and _holds_recordings(d):
            return d
    return None


def _direct_mp4(d):
    """True when d itself contains .mp4 files."""
    try:
        return any(n.lower().endswith(".mp4") and
                   os.path.isfile(os.path.join(d, n))
                   for n in os.listdir(d))
    except OSError:
        return False


def _is_chunk_dir(d):
    """A chunked session: a folder of part001.mp4, part002.mp4, ..."""
    if not os.path.isdir(d):
        return False
    try:
        mp4s = [n for n in os.listdir(d) if n.lower().endswith(".mp4")]
    except OSError:
        return False
    return bool(mp4s) and all(n.lower().startswith("part") for n in mp4s)


def _holds_recordings(d):
    """True when d is a recordings folder: flat clips and/or chunk folders."""
    if not os.path.isdir(d):
        return False
    if _direct_mp4(d):
        return True
    try:
        return any(_is_chunk_dir(os.path.join(d, n)) for n in os.listdir(d))
    except OSError:
        return False


MOUNT_PREFIXES = ("/media/", "/run/media/", "/mnt/", "/Volumes/")


def _mounted_volumes():
    """Removable volumes, from the mount table where one is available.

    Mount layout varies: /Volumes/<label> on macOS, /media/<label> or
    /media/<user>/<label> or /run/media/<user>/<label> on Linux depending on
    the automounter. Reading the mount table avoids having to guess the depth.
    """
    found = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mp = parts[1].replace("\\040", " ")
                    if mp.startswith(MOUNT_PREFIXES):
                        found.append(mp)
    except OSError:
        pass                                          # macOS: no /proc/mounts
    return found


def _unmounted_removable():
    """Removable partitions that carry a filesystem but are not mounted.

    Restricted to hot-pluggable partitions (lsblk's RM flag) with a recognised
    filesystem, so an internal disk can never be picked up.
    """
    try:
        r = subprocess.run(
            ["lsblk", "-J", "-o", "PATH,FSTYPE,MOUNTPOINT,RM,TYPE,SIZE,LABEL"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
        data = json.loads(r.stdout.decode("utf-8", "replace"))
    except Exception:                                 # no lsblk, or not Linux
        return []

    out = []

    def walk(nodes):
        for n in nodes:
            walk(n.get("children") or [])
            removable = str(n.get("rm")).lower() in ("1", "true")
            if (n.get("type") == "part" and removable
                    and n.get("fstype") and not n.get("mountpoint")):
                out.append({"path": n.get("path"),
                            "label": n.get("label") or "",
                            "size": n.get("size") or ""})
    walk(data.get("blockdevices") or [])
    return out


def mount_readonly(device, log):
    """Mount a device read-only via udisks. Returns the mount point or None."""
    err = ""
    for extra in (["--options", "ro"], []):
        try:
            r = subprocess.run(["udisksctl", "mount", "-b", device] + extra,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=60)
        except Exception:
            return None
        if r.returncode == 0:
            # "Mounted /dev/sda1 at /media/user/LABEL"
            text = r.stdout.decode("utf-8", "replace").strip().rstrip(".")
            if " at " in text:
                return text.rsplit(" at ", 1)[1]
            return None
        err = r.stderr.decode("utf-8", "replace")
        if "already mounted" in err.lower():
            return None
        # A filesystem that refuses a read-only mount gets one plain retry.
    log.warn("could not mount %s (%s)"
             % (device, err.strip().splitlines()[-1] if err.strip() else "?"))
    return None


def automount_cards(log):
    """Mount unmounted removable media that turns out to hold recordings.

    Anything mounted here that has no recordings on it is unmounted again
    immediately, so this never leaves unrelated media attached.
    """
    mounted = []
    for dev in _unmounted_removable():
        label = (" \"%s\"" % dev["label"]) if dev["label"] else ""
        log.info("Found unmounted removable device %s%s (%s) -- mounting"
                 % (dev["path"], label, dev["size"]))
        mp = mount_readonly(dev["path"], log)
        if not mp:
            continue
        if find_recordings_dir(mp):
            log.info("  mounted read-only at %s" % mp)
            mounted.append((dev["path"], mp))
        else:
            log.info("  no recordings on it; unmounting again")
            unmount(dev["path"])
    return mounted


def unmount(device):
    try:
        subprocess.run(["udisksctl", "unmount", "-b", device],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=60)
    except Exception:
        pass


def autodetect_cards():
    """Windows drive letters, or mounted volumes, holding recordings."""
    if os.name == "nt":
        import string
        roots = []
        for letter in string.ascii_uppercase[3:]:     # skip A, B, C
            root = letter + ":\\"
            if os.path.isdir(root) and find_recordings_dir(root):
                roots.append(root)
        return roots

    candidates = _mounted_volumes()

    # Also walk the usual roots one and two levels down, both as a fallback
    # where no mount table exists and to catch cards copied onto disk.
    for base in ("/media", "/run/media", "/mnt", "/Volumes",
                 os.path.expanduser("~/media")):
        if not os.path.isdir(base):
            continue
        try:
            subs = sorted(os.listdir(base))
        except OSError:
            continue
        for sub in subs:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            candidates.append(d)
            try:
                candidates.extend(
                    os.path.join(d, s) for s in sorted(os.listdir(d))
                    if os.path.isdir(os.path.join(d, s))
                )
            except OSError:
                pass

    roots, seen = [], set()
    for d in candidates:
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        if find_recordings_dir(d):
            roots.append(d)
    return roots


# --------------------------------------------------------------------------- #
# Metadata assembly
# --------------------------------------------------------------------------- #
def build_metadata(clip, args, calib, head):
    """Per-clip delivery metadata, mapping the collection specification.

    Covers, per the customer's specification sheet: geographic location,
    collector and session ids, camera intrinsics (focal length, distortion)
    and extrinsics (position and orientation relative to the head frame), the
    video technical properties (codec, resolution, aspect, frame rate, bitrate,
    GOP, B-frames, colour depth, clip length) and the IMU metadata
    (accelerometer/gyroscope, sample rate and video synchronisation). Fields
    that need a calibration file are present only when one is supplied.
    """
    ms = clip.meta_sidecar
    session_no = ms.get("session") or 0
    session_id = args.session_id or "%s-%s-%s-s%s" % (
        args.collector, args.capture_date.replace("-", ""),
        clip.device_tag, session_no,
    )

    meta = {
        "schema": METADATA_SCHEMA,
        "clip_id": clip.base,
        "collector_id": args.collector,          # spec: User ID
        "session_id": session_id,                # spec: Session ID
        # spec: Environment Type. Flat environment_type/subcategory are the
        # fields the delivery ingest checks for; the nested object mirrors them.
        "environment_type": args.env_type,
        "environment_subcategory": args.env_subcategory,
        "environment": {
            "type": args.env_type,
            "subcategory": args.env_subcategory,
        },
        "location": {"country": args.country},   # spec: Geographic Location
        "capture": {"date": args.capture_date},
        "camera": {
            "make": "Panoculon Labs",
            "model": "Trinet",
            "device_id": clip.device_id or None,
        },
        "video": _video_metadata(clip),          # spec: Camera Specifications
        "imu": _imu_metadata(clip),              # spec: IMU / IMU Metadata
        "duration_s": _round(clip.duration_s, 3),  # spec: Clip Length
    }

    if args.env_note:
        meta["environment"]["note"] = args.env_note
    if args.region:
        meta["location"]["region"] = args.region
    if args.participant_id:
        meta["participant_id"] = args.participant_id
    if args.task:
        meta["task"] = {"description": args.task}
        if args.task_labels:
            meta["task"]["labels"] = [
                s.strip() for s in args.task_labels.split(",") if s.strip()
            ]

    # Camera geometry, from --calibration / --head-transform.
    if calib:
        _apply_calibration(meta["camera"], calib, head, args)
    else:
        meta["camera"]["intrinsics"] = None
        meta["camera"]["extrinsics"] = None
        meta["camera"]["calibration_note"] = (
            "No calibration supplied at ingest, so camera intrinsics and "
            "extrinsics are absent. Provide the unit's calibration.json to "
            "populate them."
        )

    # Diagonal field of view (spec: Diagonal Field of View). An explicit
    # --fov-deg wins; otherwise it comes from the calibration if present.
    if args.fov_deg is not None:
        meta["camera"]["diagonal_fov_deg"] = round(args.fov_deg, 1)
        meta["camera"]["diagonal_fov_source"] = "stated (--fov-deg)"
    elif meta["camera"].get("diagonal_fov_deg") is not None:
        meta["camera"]["diagonal_fov_source"] = (
            calib.get("diagonal_fov_method") if calib else None
        ) or "computed from calibration"
    else:
        meta["camera"]["diagonal_fov_deg"] = None
        meta["camera"]["diagonal_fov_note"] = (
            "Not available: pass --fov-deg to state the lens field of view, "
            "or --calibration to compute it from the intrinsics."
        )

    return meta


def _video_metadata(clip):
    """Video technical block: codec, resolution, aspect, fps, bitrate, GOP,
    B-frames, colour depth. Maps the Camera Specifications rows of the spec."""
    v = clip.video or {}
    vts = clip.vts or {}
    w, h = v.get("width"), v.get("height")
    out = {
        "container": "mp4",
        "codec": v.get("codec"),                     # Video Format
        "width": w,
        "height": h,
        "duration_s": _round(clip.duration_s, 3),    # Clip Length
        "bitrate_mbps": _round(clip.bitrate_mbps, 2),  # Bitrate
        "file_bytes": v.get("file_bytes"),
    }
    if w and h:
        out["resolution_mp"] = round(w * h / 1e6, 2)  # Resolution
        out["aspect_ratio"] = "landscape" if w > h else "portrait"  # Aspect
    if vts.get("frames"):
        out["frame_count"] = vts["frames"]
    if vts.get("nominal_fps"):
        out["nominal_fps"] = vts["nominal_fps"]      # Frame Rate
    if vts.get("avg_fps") is not None:
        out["average_fps"] = _round(vts["avg_fps"], 2)  # Avg FPS after drops
    # These come from the H.264 bitstream, so they exist only for H.264.
    for key in ("gop_length", "color_depth_bits"):   # GOP, Colour Depth
        if key in v:
            out[key] = v[key]
    if "b_frames" in v:                              # B-Frames
        out["b_frames"] = v["b_frames"]
    if v.get("chunked") or clip.chunked:
        out["chunked"] = True
        out["parts"] = v.get("parts")
    return out


def _imu_metadata(clip):
    """IMU metadata block: sensors, sample rate, ranges, video sync. Maps the
    IMU Data and IMU Metadata rows of the spec."""
    imu = clip.imu
    if not imu:
        return {
            "present": False,
            "note": "No inertial sidecar found for this clip.",
        }
    sensors = ["accelerometer", "gyroscope"]         # spec: sensors required
    if imu.get("magnetometer"):
        sensors.append("magnetometer")
    imu_files = [os.path.basename(p) for p in clip.sidecars.get("imu", [])]
    sync_method = ("hardware frame-sync pulse" if imu.get("frame_sync")
                   else "shared monotonic clock")
    return {
        "present": True,
        "data_files": imu_files,
        "sensors": sensors,                          # Accelerometer + Gyroscope
        "sample_rate_hz": _round(imu.get("actual_rate_hz"), 2),  # Sample Rate
        "nominal_rate_hz": imu.get("nominal_rate_hz"),
        "sample_count": imu.get("samples"),
        "duration_s": _round(imu.get("duration_s"), 3),
        "accelerometer": {                           # Accelerometer Data
            "range_g": imu.get("accel_range_g"),
            "units": "m/s^2 (gravity included)",
        },
        "gyroscope": {                               # Gyroscope Data
            "range_dps": imu.get("gyro_range_dps"),
            "units": "rad/s",
        },
        "video_sync": {                              # Synchronization Info
            "tolerance_ms": 1,
            "method": sync_method,
            "align_using": "sof_timestamp_ns in the .vts sidecar",
            "note": (
                "Inertial samples and per-frame video timestamps share one "
                "monotonic clock, so they need no cross-correlation; alignment "
                "error is bounded by the IMU sample period (well under 1 ms)."
            ),
        },
    }


def _apply_calibration(cam, calib, head, args):
    """Fill camera.intrinsics and camera.extrinsics from a calibration."""
    intr = calib.get("intrinsics") or {}
    named = {
        "image_size": intr.get("image_size"),
        "projection_model": intr.get("model"),
    }
    if intr.get("fx") is not None:
        named["focal_length_px"] = {"fx": intr.get("fx"),  # spec: Focal Length
                                    "fy": intr.get("fy")}
    if intr.get("cx") is not None:
        named["principal_point_px"] = {"cx": intr.get("cx"),
                                       "cy": intr.get("cy")}
    named["distortion_model"] = intr.get("model")
    named["distortion_coefficients"] = intr.get("distortion")  # spec: Distortion
    cam["intrinsics"] = named
    cam["diagonal_fov_deg"] = calib["diagonal_fov_deg"]        # spec: FOV
    # Full H/V/D breakdown when the calibration carries it (the diagonal is the
    # spec field; horizontal is the more stable metric for wide fisheye lenses).
    fov = intr.get("fov_deg")
    if isinstance(fov, dict):
        cam["fov_deg"] = {k: _round(fov.get(k), 1) for k in
                          ("horizontal", "vertical", "diagonal")
                          if fov.get(k) is not None}
    cam["calibration_source"] = calib["source_file"]

    # Whether this calibration is for the exact unit or a batch representative.
    scope = args.calibration_scope or "unspecified"
    cam["calibration_scope"] = scope
    if args.calibration_id:
        cam["calibration_id"] = args.calibration_id
    cam["calibration_scope_note"] = {
        "device": ("Calibrated for this specific camera unit; the intrinsics "
                   "and extrinsics apply directly to this recording."),
        "batch": ("Batch calibration -- measured on a representative unit and "
                  "applied to every camera in the production batch, NOT this "
                  "exact unit. Per-unit optical variation is not captured."),
        "unspecified": ("Calibration scope not stated at ingest; treat as "
                        "approximate until confirmed as per-device or batch."),
    }[scope]

    # Extrinsics: position + orientation relative to the head frame (spec R28/29).
    ext = {}
    if head:
        hf = {"source": head.get("source")}
        if head.get("T_head_cam") is not None:
            hf["T_head_cam"] = head["T_head_cam"]
            hf["note"] = ("4x4 head-from-camera transform; the translation is "
                          "the camera position in the head frame, the rotation "
                          "its orientation.")
        else:
            hf["position_m"] = head.get("translation_m")        # R28
            hf["orientation_deg"] = head.get("rotation_deg")    # R29
            hf["orientation_convention"] = head.get("rotation_order")
        ext["head_frame"] = hf
    else:
        ext["head_frame"] = {
            "measured": False,
            "note": (
                "No mount pose supplied (--head-transform), so the camera's "
                "position and orientation relative to the head frame are not "
                "given. The camera<-IMU extrinsic below is still provided."
            ),
        }
    if calib.get("T_cam_imu"):
        ext["camera_to_imu"] = {
            "T_cam_imu": calib["T_cam_imu"],
            "note": calib["T_cam_imu_note"],
        }
        if calib.get("timeshift_cam_imu_s") is not None:
            ext["camera_to_imu"]["timeshift_cam_imu_s"] = \
                calib["timeshift_cam_imu_s"]
    cam["extrinsics"] = ext


# --------------------------------------------------------------------------- #
# README shipped inside every ZIP
# --------------------------------------------------------------------------- #
def _provenance(rebuilt):
    if rebuilt:
        return ("copies of what the camera recorded, with each MP4's index "
                "rebuilt so strict players and uploaders read them correctly. "
                "That rebuild is lossless -- the video itself is untouched")
    return "byte-for-byte copies of what the camera recorded"


README = """# Trinet recording -- {clip_id}

This archive holds one continuous egocentric video clip and the inertial data
recorded alongside it, plus the metadata describing how and where it was
collected.

## Contents

| File | What it is |
|---|---|
| `{clip_id}.mp4` | H.264 video. Plays in any standard player. |
| `{clip_id}.imu` | Inertial samples (accelerometer + gyroscope). Binary. |
| `{clip_id}.vts` | One capture timestamp per video frame. Binary. |
| `{clip_id}.json` | The camera's own recording sidecar, when present. |
| `metadata.json` | Collection metadata and camera calibration. |
| `{clip_id}.mcap` | Present only if packaged with `--mcap`: video + IMU + metadata in one Foxglove-ready [MCAP](https://mcap.dev) container. |

The recordings are {provenance}.

If you only need the video, open the `.mp4` -- nothing else is required.

## Reading the inertial data

The `.imu` and `.vts` files are small binary formats. The open-source reader
is at **https://github.com/Panoculon-Labs/Trinet-tools**:

```bash
git clone https://github.com/Panoculon-Labs/Trinet-tools.git
cd Trinet-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```python
from trinet_tools.reader import read_imu, read_vts, interpolate_imu_to_frames

imu = read_imu("{clip_id}.imu")
vts = read_vts("{clip_id}.vts")

print(imu.num_samples, "samples at", imu.actual_rate_hz, "Hz")

# Inertial samples aligned to video frames, one entry per frame:
per_frame = interpolate_imu_to_frames(imu, vts)
for e in per_frame[:5]:
    print(e["frame_number"], e["accel"], e["gyro"])
```

`imu.accel` is m/s^2 with gravity included, `imu.gyro` is rad/s, both as
arrays of `[x, y, z]`.

The complete byte-level specification -- so you can write a reader in any
language -- is in [`docs/data_formats.md`](https://github.com/Panoculon-Labs/Trinet-tools/blob/main/docs/data_formats.md).

## How the video and inertial data line up

**Both use the same monotonic nanosecond clock**, so aligning them needs no
cross-correlation and no drift correction:

- Every entry in `.vts` gives one video frame's capture time
  (`sof_timestamp_ns`).
- Every sample in `.imu` carries its own `timestamp_ns` on that same clock.

To find the inertial state at frame *N*, take that frame's
`sof_timestamp_ns` and interpolate the inertial samples around it --
`interpolate_imu_to_frames()` does exactly this.

Two things to know:

1. **These timestamps are not wall-clock.** They count from the moment the
   camera powered on and reset every power cycle. The real-world capture date
   is in `metadata.json` under `capture.date`, recorded by the operator.
2. **Use `sof_timestamp_ns`, not the video PTS.** The presentation timestamps
   in the MP4 describe playback timing and are delayed relative to the actual
   moment of capture.

## Visualizing it

To render the video with synchronized inertial plots as a sanity check:

```bash
python scripts/visualize.py {clip_id}.mp4
```

## Camera geometry

`metadata.json -> camera.intrinsics` carries the focal lengths, principal
point, projection model and distortion coefficients for this unit, with the
computed diagonal field of view. `camera.extrinsics` carries the transform
between the camera and the inertial sensor, and the mount pose in the head
frame.

The projection model is noted in `intrinsics.model`; `equidistant` is the
fisheye model that OpenCV implements as `cv2.fisheye`.

"""


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
def stage_and_repair(clip, staging, log):
    """Copy the clip's files to staging and rebuild the MP4 indexes there.

    Only used when --repair is asked for. Without it the recordings go into
    the ZIP straight from the card, byte for byte.
    """
    staged = []
    for src in clip.all_files():
        dst = os.path.join(staging, os.path.basename(src))
        if os.path.exists(dst):                       # part names collide
            root, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(
                staging, "%s_%d%s" % (root, len(staged), ext)
            )
        shutil.copy2(src, dst)
        staged.append((src, dst))


    if not _flatten_file:
        log.warn("--repair requested but repair_recordings.py is unavailable")
        return staged, 0

    rebuilt = 0
    for src, dst in staged:
        if not dst.lower().endswith(".mp4"):
            continue
        try:
            result, _detail = _flatten_file(dst, backup=False,
                                            log=lambda *a, **k: None)
        except Exception as e:                        # pragma: no cover
            log.warn("%s: index repair failed (%s)" % (os.path.basename(src), e))
            continue
        # RepairResult.OK == "repaired"; ALREADY_FLAT/SKIP/ERROR mean no rewrite.
        if result == "repaired":
            rebuilt += 1
            log.info("    rebuilt MP4 index: %s" % os.path.basename(src))
    return staged, rebuilt


def stage_and_reencode(clip, staging, args, log):
    """Copy sidecars to staging and re-encode each MP4 to spec there.

    The re-encoded MP4 is already a clean, faststart, spec-bitrate H.264, so
    no separate index repair is needed. The .imu/.vts sidecars are copied
    unchanged. The card is only read.
    """
    staged = []
    for src in clip.all_files():
        base = os.path.basename(src)
        dst = os.path.join(staging, base)
        if os.path.exists(dst):                           # part names collide
            root, ext = os.path.splitext(base)
            dst = os.path.join(staging, "%s_%d%s" % (root, len(staged), ext))
        if src.lower().endswith(".mp4"):
            if reencode_video(src, dst, args.reencode_mbps, log):
                log.info("    re-encoded %s" % base)
            else:
                # Fall back to a verbatim copy so the clip still ships.
                log.warn("%s: re-encode failed; copying original" % base)
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        staged.append((src, dst))
    return staged


def gate_reasons(clip, args):
    """Reasons this clip should be skipped under the active gates, or []."""
    reasons = []
    min_dur = 60.0 if args.gate else args.min_duration
    if min_dur:
        d = clip.duration_s
        if d is None:
            reasons.append("duration unknown")
        elif d < min_dur:
            reasons.append("too short (%.1f s < %g s)" % (d, min_dur))
    if args.gate or args.require_imu:
        if not clip.imu or (clip.imu.get("samples") or 0) < 2:
            reasons.append("no usable IMU")
    if args.gate or args.require_valid_video:
        st, detail = clip.container
        if st != "ok":
            reasons.append("%s video (%s)" % (st, detail))
    return reasons


def package(clip, args, calib, head, log):
    """Build one ZIP. Returns (zip_path, metadata, reject_reason)."""
    log.info("  %s" % clip.base)
    clip.parse(log)

    d = clip.duration_s
    cst = clip.container[0]
    log.info("    %s%s"
             % ("%.1f s" % d if d else "duration unknown",
                "" if cst == "ok" else "  [%s]" % cst))

    # Quality gates: skip clips that would be refused downstream.
    rejects = gate_reasons(clip, args)
    if rejects:
        reason = "; ".join(rejects)
        log.warn("%s: skipped -- %s" % (clip.base, reason))
        return None, None, reason

    # Group-take base names already carry the device tag; don't repeat it.
    stem = [args.collector, args.capture_date.replace("-", "")]
    if clip.device_tag not in clip.base:
        stem.append(clip.device_tag)
    stem.append(clip.base)
    zip_name = "_".join(stem) + ".zip"
    zip_path = os.path.join(args.out, zip_name)
    if os.path.exists(zip_path) and not args.overwrite:
        log.warn("%s already exists; skipping (--overwrite to replace)"
                 % zip_name)
        return None, None, None

    if args.dry_run:
        log.info("    would write %s%s" % (zip_name,
                 " (re-encoded)" if args.reencode else ""))
        return None, None, None

    # Staging is needed to rebuild indexes (--repair) or to re-encode; either
    # way the card is only ever read. Without them, files go in verbatim.
    need_staging = args.repair or args.reencode
    staging = tempfile.mkdtemp(prefix="trinet_ingest_") if need_staging else None
    mcap_path = None
    try:
        rebuilt = 0
        if args.reencode:
            sources = stage_and_reencode(clip, staging, args, log)
            # Re-probe the re-encoded video so metadata reflects it.
            new_mp4s = [dst for src, dst in sources
                        if dst.lower().endswith(".mp4")]
            if new_mp4s:
                clip.video = probe_video(new_mp4s)
        elif staging:
            sources, rebuilt = stage_and_repair(clip, staging, log)
        else:
            sources = [(p, p) for p in clip.all_files()]

        meta = build_metadata(clip, args, calib, head)

        # Optional MCAP: a single time-indexed container (IMU + video +
        # metadata) that Foxglove and other robotics tools open directly.
        mcap_path = None
        if args.mcap:
            mcap_path = os.path.join(staging or tempfile.mkdtemp(
                prefix="trinet_mcap_"), clip.base + ".mcap")
            try:
                write_mcap(mcap_path, clip, meta, args.calibration, log)
            except Exception as e:                    # never fail the ZIP
                log.warn("%s: could not build MCAP (%s); skipping it"
                         % (clip.base, e))
                mcap_path = None

        tmp_zip = zip_path + ".part"
        with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as z:
            for _src, path in sources:
                # Video is already compressed; only deflate the sidecars.
                ext = os.path.splitext(path)[1].lower()
                ctype = (zipfile.ZIP_STORED if ext == ".mp4"
                         else zipfile.ZIP_DEFLATED)
                z.write(path, os.path.basename(path), compress_type=ctype)
            z.writestr("metadata.json",
                       json.dumps(meta, indent=2) + "\n",
                       zipfile.ZIP_DEFLATED)
            z.writestr("README.md",
                       README.format(clip_id=clip.base,
                                     provenance=_provenance(rebuilt)),
                       zipfile.ZIP_DEFLATED)
            if mcap_path and os.path.isfile(mcap_path):
                z.write(mcap_path, clip.base + ".mcap",
                        compress_type=zipfile.ZIP_STORED)
        os.replace(tmp_zip, zip_path)
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
        elif mcap_path:
            shutil.rmtree(os.path.dirname(mcap_path), ignore_errors=True)

    log.info("    -> %s (%.1f MB)"
             % (zip_name, os.path.getsize(zip_path) / 1e6))
    return zip_path, meta, None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_environment(value):
    """Split TYPE/SUBCATEGORY. Any values are accepted; the known collection
    taxonomy is only used to warn (in main), never to reject."""
    if "/" not in value:
        raise argparse.ArgumentTypeError(
            "use TYPE/SUBCATEGORY, e.g. residential/laundry")
    etype, sub = value.split("/", 1)
    etype = etype.strip().lower()
    sub = sub.strip().lower().replace(" ", "_").replace("-", "_")
    if not etype or not sub:
        raise argparse.ArgumentTypeError(
            "both TYPE and SUBCATEGORY are required, e.g. residential/laundry")
    return etype, sub


def parse_date(value):
    try:
        return _dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD, e.g. 2026-07-24")


def build_parser():
    p = argparse.ArgumentParser(
        prog="ingest_sd_card.py",
        description="Package Trinet SD-card recordings into per-clip "
                    "delivery ZIPs with collection metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Windows, card in E:, one ZIP per clip into D:\\deliveries
  python scripts\\ingest_sd_card.py --drive E: --collector alice01 \\
      --country US --environment residential/laundry \\
      --calibration cal\\unit-aa3d26ba.json --out D:\\deliveries

  # Re-encode to spec bitrate and only keep deliverable clips
  python scripts/ingest_sd_card.py --collector bob02 --country GB \\
      --environment commercial/hospitality_housekeeping \\
      --reencode --gate --out ./deliveries

  # See what would happen without writing anything
  python scripts/ingest_sd_card.py --drive E: --collector alice01 \\
      --country US --environment residential/laundry --dry-run

known environment values (others are accepted with a warning):
""" + "\n".join("  %-12s %s" % (t, ", ".join(s))
                for t, s in ENVIRONMENTS.items()),
    )

    src = p.add_argument_group("source")
    src.add_argument("--drive", "--card", dest="drive", metavar="PATH",
                     help="SD card root or recordings folder "
                          "(e.g. E: on Windows). Auto-detected if omitted.")
    src.add_argument("--folder", metavar="NAME",
                     help="Recordings folder name on the card, if it is not "
                          "auto-detected.")

    req = p.add_argument_group("required collection metadata")
    req.add_argument("--collector", required=True, metavar="ID",
                     help="Unique identifier for the person collecting.")
    req.add_argument("--country", required=True, metavar="CC",
                     help="Geographic location, country (e.g. US, GB, IN).")
    req.add_argument("--environment", required=True, type=parse_environment,
                     metavar="TYPE/SUB",
                     help="Environment type and sub-category, e.g. "
                          "residential/laundry. Required by the delivery spec. "
                          "Any value is accepted; one outside the known list "
                          "below warns but still packages.")

    opt = p.add_argument_group("optional metadata")
    opt.add_argument("--region", metavar="NAME",
                     help="Finer-grained location (state / city).")
    opt.add_argument("--task", metavar="TEXT",
                     help='High-level task description, e.g. "dishes cleanup".')
    opt.add_argument("--task-labels", metavar="A,B,C",
                     help="Comma-separated low-level action labels.")
    opt.add_argument("--env-note", metavar="TEXT",
                     help="Free-text note about the environment (recommended "
                          "when the sub-category is 'other').")
    opt.add_argument("--participant-id", metavar="ID",
                     help="Participant identifier, if distinct from the "
                          "collector.")
    opt.add_argument("--session-id", metavar="ID",
                     help="Override the generated per-session identifier.")
    opt.add_argument("--capture-date", type=parse_date, metavar="YYYY-MM-DD",
                     help="Date the footage was captured. Defaults to today; "
                          "the camera has no real-time clock, so this cannot "
                          "be read from the card.")

    cal = p.add_argument_group("camera geometry")
    cal.add_argument("--calibration", metavar="FILE",
                     help="calibration.json for this unit or batch. Supplies "
                          "the intrinsics, distortion, field of view and "
                          "extrinsics.")
    cal.add_argument("--calibration-scope", choices=("device", "batch"),
                     metavar="{device,batch}",
                     help="Whether the calibration is for this exact unit "
                          "('device') or a representative one for the "
                          "production batch ('batch'). Recorded in "
                          "metadata.json. If omitted with --calibration, the "
                          "scope is marked 'unspecified' and a warning is "
                          "printed.")
    cal.add_argument("--calibration-id", metavar="ID",
                     help="Optional identifier for the calibration / batch, "
                          "recorded alongside the scope.")
    cal.add_argument("--fov-deg", type=float, metavar="DEG",
                     help="Diagonal field of view in degrees, stated "
                          "explicitly. Overrides the value computed from the "
                          "calibration; use it to record the nominal lens FOV "
                          "when no calibration is supplied.")
    cal.add_argument("--camera-index", type=int, default=0, metavar="N",
                     help="Which camera in the calibration file (default 0).")
    cal.add_argument("--head-transform", metavar="FILE",
                     help="JSON mount pose: either T_head_cam (4x4), or "
                          "rotation_deg + translation_m.")

    out = p.add_argument_group("output")
    out.add_argument("--out", "-o", default="deliveries", metavar="DIR",
                     help="Where to write the ZIPs (default: ./deliveries).")
    out.add_argument("--overwrite", action="store_true",
                     help="Replace ZIPs that already exist.")
    out.add_argument("--mcap", action="store_true",
                     help="Also write a <clip>.mcap into each ZIP: one "
                          "time-indexed container with the IMU on an /imu "
                          "topic, the video as foxglove.CompressedVideo on a "
                          "/camera topic, and the metadata + calibration "
                          "embedded. Opens directly in Foxglove. Adds the "
                          "video a second time, so the ZIP roughly doubles.")
    out.add_argument("--no-repair", dest="repair", action="store_false",
                     help="Do NOT rebuild each MP4's index. By default the "
                          "index is rebuilt on a staged copy (lossless, no "
                          "re-encode) so strict players and uploaders accept "
                          "the file; the card is never modified. Use this to "
                          "copy the recordings verbatim instead.")

    enc = p.add_argument_group("re-encode (needs ffmpeg)")
    enc.add_argument("--reencode", action="store_true",
                     help="Transcode each video to spec (H.264, GOP 30, no "
                          "B-frames, 8-bit, <=8 Mbps) as fast as the machine "
                          "allows (hardware NVENC if available, else libx264 "
                          "veryfast). The .imu/.vts sidecars are untouched. "
                          "Requires ffmpeg on PATH.")
    enc.add_argument("--reencode-mbps", type=float, default=7.0, metavar="N",
                     help="Target video bitrate for --reencode (default 7; "
                          "hard-capped at 8 to stay inside the 6-8 spec).")
    enc.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                     help="Package this many clips in parallel. Useful with "
                          "--reencode (default 1).")

    gate = p.add_argument_group("quality gating (skip bad clips)")
    gate.add_argument("--gate", action="store_true",
                      help="Skip clips that would be refused: shorter than "
                           "60 s, missing IMU, or an unreadable/truncated "
                           "video. Shortcut for --min-duration 60 "
                           "--require-imu --require-valid-video.")
    gate.add_argument("--min-duration", type=float, default=None, metavar="SECS",
                      help="Skip clips shorter than SECS (the spec floor is "
                           "120).")
    gate.add_argument("--require-imu", action="store_true",
                      help="Skip clips with no usable IMU (accel+gyro, "
                           ">=2 samples).")
    gate.add_argument("--require-valid-video", action="store_true",
                      help="Skip clips whose MP4 is truncated or has no "
                           "readable video track.")

    out.add_argument("--no-automount", dest="automount",
                     action="store_false",
                     help="Do not attach an unmounted card. By default, if no "
                          "mounted card is found, a removable volume holding "
                          "recordings is mounted read-only and detached again "
                          "when the run finishes. (Not needed on Windows, "
                          "where cards attach themselves.)")
    out.add_argument("--dry-run", action="store_true",
                     help="Report what would be packaged; write nothing.")
    out.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = Log(args.quiet)
    we_mounted = []                # volumes this run attached, to detach after

    args.env_type, args.env_subcategory = args.environment
    if args.env_type not in ENVIRONMENTS:
        log.warn("environment type '%s' is outside the spec (expected one of: "
                 "%s). Packaging it anyway."
                 % (args.env_type, ", ".join(sorted(ENVIRONMENTS))))
    elif args.env_subcategory not in ENVIRONMENTS[args.env_type]:
        log.warn("'%s' is not a listed %s sub-category. Packaging it anyway."
                 % (args.env_subcategory, args.env_type))

    if not args.capture_date:
        args.capture_date = _dt.date.today().isoformat()
        log.warn("no --capture-date given; using today (%s). The camera has "
                 "no real-time clock, so card timestamps cannot supply it."
                 % args.capture_date)

    if args.reencode:
        if not ffmpeg_path():
            print("error: --reencode needs ffmpeg on PATH (not found). "
                  "Install ffmpeg, or drop --reencode.")
            return 2
        enc, _ = _pick_h264_encoder()
        log.info("Re-encoding to <=%g Mbps H.264 via %s"
                 % (min(args.reencode_mbps, 8.0), enc))

    # -- locate the card --------------------------------------------------- #
    if args.drive:
        root = args.drive
        if os.name == "nt" and len(root) == 2 and root[1] == ":":
            root += "\\"
        if not os.path.isdir(root):
            print("error: %s is not accessible" % root)
            return 2
        rec_dir = find_recordings_dir(root, args.folder)
    else:
        cards = autodetect_cards()
        if not cards and args.automount and os.name != "nt":
            # Nothing mounted -- the card may be plugged in but not attached
            # by the desktop. Mount it ourselves, read-only.
            we_mounted = automount_cards(log)
            cards = autodetect_cards()
        if not cards:
            print("error: no SD card with recordings found. "
                  "Pass --drive E: (or the folder path).")
            return 2
        if len(cards) > 1:
            print("error: found several cards: %s\n"
                  "       pass --drive to choose one." % ", ".join(cards))
            return 2
        root = cards[0]
        log.info("Auto-detected card: %s" % root)
        rec_dir = find_recordings_dir(root, args.folder)

    if not rec_dir:
        print("error: no recordings folder found under %s" % root)
        return 2
    log.info("Reading %s" % rec_dir)

    # -- calibration ------------------------------------------------------- #
    calib = None
    if args.calibration:
        try:
            calib = load_calibration(args.calibration, args.camera_index, log)
            log.info("Calibration: %s (%s scope, diagonal FOV %s deg)"
                     % (os.path.basename(args.calibration),
                        args.calibration_scope or "unspecified",
                        calib["diagonal_fov_deg"]))
        except (OSError, ValueError, KeyError) as e:
            print("error: could not read calibration: %s" % e)
            return 2
        if not args.calibration_scope:
            log.warn("no --calibration-scope given; metadata will mark the "
                     "calibration 'unspecified'. Pass --calibration-scope "
                     "device or batch to say which it is.")
    else:
        log.warn("no --calibration supplied: the intrinsics and extrinsics "
                 "keys will be absent from every metadata.json.")
        if args.calibration_scope or args.calibration_id:
            log.warn("--calibration-scope/--calibration-id ignored without "
                     "--calibration.")

    head = None
    if args.head_transform:
        try:
            head = load_head_transform(args.head_transform)
        except (OSError, ValueError) as e:
            print("error: could not read head transform: %s" % e)
            return 2

    # -- discover ---------------------------------------------------------- #
    clips = discover(rec_dir, log)
    if not clips:
        print("error: no recordings found in %s" % rec_dir)
        return 2
    log.info("Found %d recording(s)\n" % len(clips))

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    written = skipped = 0
    rejected = []                        # (clip_base, reason)

    def run(clip):
        return clip, package(clip, args, calib, head, log)

    jobs = max(1, args.jobs)
    if jobs > 1 and len(clips) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(run, clips))
    else:
        results = [run(c) for c in clips]

    for clip, (zpath, _meta, reject) in results:
        if zpath:
            written += 1
        else:
            skipped += 1
            if reject:
                rejected.append((clip.base, reject))

    if rejected and not args.dry_run:
        rej_path = os.path.join(
            args.out, "rejected_%s_%s.json"
            % (args.collector, args.capture_date.replace("-", "")))
        with open(rej_path, "w", encoding="utf-8") as f:
            json.dump({"schema": "trinet-ingest-rejects/1",
                       "collector_id": args.collector,
                       "count": len(rejected),
                       "rejected": [{"clip": c, "reason": r}
                                    for c, r in rejected]}, f, indent=2)
            f.write("\n")
        log.info("Rejected %d clip(s); see %s"
                 % (len(rejected), os.path.basename(rej_path)))

    for dev, mp in we_mounted:
        unmount(dev)
        log.info("Unmounted %s (%s)" % (dev, mp))

    print("\n%d packaged, %d skipped (%d gated out), %d warning(s)"
          % (written, skipped, len(rejected), len(log.warnings)))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

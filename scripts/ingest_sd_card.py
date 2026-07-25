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
        --environment residential/laundry ^
        --calibration cal\\unit-aa3d26ba.json ^
        --out D:\\deliveries

The recordings are never altered: they go into the ZIP as byte-for-byte copies
of what is on the card, and the card itself is only ever read from. Nothing is
inspected, judged or filtered -- the script adds metadata and zips, and that is
all (unless you explicitly ask for --repair).
"""

import argparse
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


# Environment taxonomy. type -> allowed sub-categories.
ENVIRONMENTS = {
    "residential": [
        "laundry",
        "kitchen_tidy",
        "organize_room",
        "other_household",
    ],
    "commercial": [
        "agriculture_landscaping_grounds",
        "hospitality_housekeeping",
        "automotive_service_maintenance",
        "food_service_back_of_house",
        "field_services_light_installation",
        "commercial_cleaning_janitorial",
        "retail_stocking_back_of_house",
        "construction_skilled_trades",
        "other",
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
    with open(path, "r", encoding="utf-8") as f:
        cal = json.load(f)
    cams = cal.get("cameras") or []
    if not cams:
        raise ValueError("calibration has no 'cameras' array")
    if cam_index >= len(cams):
        raise ValueError(
            "camera index %d out of range (file has %d)" % (cam_index, len(cams))
        )
    cam = cams[cam_index]
    intr = dict(cam.get("intrinsics") or {})
    fov, method = diagonal_fov_deg(intr)
    if fov is None:
        log.warn("could not compute field of view: %s" % method)

    out = {
        "source_file": os.path.basename(path),
        "camera_index": cam_index,
        "intrinsics": intr,
        "diagonal_fov_deg": _round(fov, 1),
        "diagonal_fov_method": method if fov is not None else None,
        "timeshift_cam_imu_s": cam.get("timeshift_cam_imu_s"),
        "reprojection_rms_px": cam.get("reprojection_rms_px"),
    }
    t = cal.get("T_cam0_imu")
    if t:
        out["T_cam_imu"] = t
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
        self.meta_sidecar = {}
        self.device_id_conflict = None

    # -- parsing ----------------------------------------------------------- #
    def parse(self, log):
        self.video = probe_video(self.mp4s)

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

    Covers, per the customer's specification sheet: environment type and
    sub-category, geographic location, collector and session ids, camera
    intrinsics (focal length, distortion) and extrinsics (position and
    orientation relative to the head frame), the video technical properties
    (codec, resolution, aspect, frame rate, bitrate, GOP, B-frames, colour
    depth, clip length) and the IMU metadata (accelerometer/gyroscope, sample
    rate and video synchronisation). Fields that need a calibration file are
    present only when one is supplied.
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
        "environment": {                         # spec: Environment Type
            "type": args.env_type,
            "subcategory": args.env_subcategory,
        },
        "location": {"country": args.country},   # spec: Geographic Location
        "capture": {"date": args.capture_date},
        "camera": {
            "make": "Panoculon Labs",
            "model": "Trinet",
            "device_id": clip.device_id or None,
            "placement": {                       # spec: Camera Placement
                "mount": args.mount,
                "orientation": args.orientation,
            },
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
        _apply_calibration(meta["camera"], calib, head)
    else:
        meta["camera"]["intrinsics"] = None
        meta["camera"]["extrinsics"] = None
        meta["camera"]["calibration_note"] = (
            "No calibration supplied at ingest, so camera intrinsics and "
            "extrinsics are absent. Provide the unit's calibration.json to "
            "populate them."
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


def _apply_calibration(cam, calib, head):
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
    cam["calibration_source"] = calib["source_file"]

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
        if getattr(result, "name", str(result)) == "OK":
            rebuilt += 1
            log.info("    rebuilt MP4 index: %s" % os.path.basename(src))
    return staged, rebuilt


def package(clip, args, calib, head, log):
    """Build one ZIP. Returns (zip_path, metadata)."""
    log.info("  %s" % clip.base)
    clip.parse(log)

    d = clip.duration_s
    log.info("    %s" % ("%.1f s" % d if d else "duration unknown"))

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
        return None, None

    if args.dry_run:
        log.info("    would write %s" % zip_name)
        return None, None

    # Default path: no staging at all -- the recordings are written into the
    # ZIP straight from the card, byte for byte. Staging happens only when
    # --repair asks for the MP4 indexes to be rebuilt.
    staging = tempfile.mkdtemp(prefix="trinet_ingest_") if args.repair else None
    try:
        rebuilt = 0
        if staging:
            sources, rebuilt = stage_and_repair(clip, staging, log)
        else:
            sources = [(p, p) for p in clip.all_files()]

        meta = build_metadata(clip, args, calib, head)

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
        os.replace(tmp_zip, zip_path)
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

    log.info("    -> %s (%.1f MB)"
             % (zip_name, os.path.getsize(zip_path) / 1e6))
    return zip_path, meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_environment(value):
    """Split TYPE/SUBCATEGORY. Any values are accepted; the known collection
    taxonomy is only used to warn (in main), never to reject -- custom
    environments are a legitimate case the spec allows via "other"."""
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

  # Auto-detect the card, add a task description
  python scripts/ingest_sd_card.py --collector bob02 --country GB \\
      --environment commercial/hospitality_housekeeping \\
      --task "strip and remake guest room" --out ./deliveries

  # See what would happen without writing anything
  python scripts/ingest_sd_card.py --drive E: --collector alice01 \\
      --country US --environment residential/laundry --dry-run

known environment values (from the collection spec; other values are
accepted with a warning):
""" + "\n".join(
            "  %-12s %s" % (t, ", ".join(subs))
            for t, subs in ENVIRONMENTS.items()
        ),
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
                     help="Environment type and sub-category, "
                          "e.g. residential/laundry. Any value is accepted; "
                          "values outside the spec list below warn but still "
                          "package.")

    opt = p.add_argument_group("optional metadata")
    opt.add_argument("--region", metavar="NAME",
                     help="Finer-grained location (state / city).")
    opt.add_argument("--task", metavar="TEXT",
                     help='High-level task description, e.g. "dishes cleanup".')
    opt.add_argument("--task-labels", metavar="A,B,C",
                     help="Comma-separated low-level action labels.")
    opt.add_argument("--env-note", metavar="TEXT",
                     help="Free-text note, required when sub-category "
                          "is 'other'.")
    opt.add_argument("--participant-id", metavar="ID",
                     help="Participant identifier, if distinct from the "
                          "collector.")
    opt.add_argument("--session-id", metavar="ID",
                     help="Override the generated per-session identifier.")
    opt.add_argument("--capture-date", type=parse_date, metavar="YYYY-MM-DD",
                     help="Date the footage was captured. Defaults to today; "
                          "the camera has no real-time clock, so this cannot "
                          "be read from the card.")
    opt.add_argument("--mount", default="head_forehead",
                     metavar="DESC",
                     help="Mounting position on the body "
                          "(default: head_forehead).")
    opt.add_argument("--orientation", default="downward",
                     metavar="DESC",
                     help="Camera orientation (default: downward, for hand "
                          "visibility).")

    cal = p.add_argument_group("camera geometry")
    cal.add_argument("--calibration", metavar="FILE",
                     help="calibration.json for this unit. Supplies the "
                          "intrinsics and extrinsics the delivery spec "
                          "requires.")
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
    out.add_argument("--repair", action="store_true",
                     help="Rebuild each MP4's index on a staged copy before "
                          "zipping, for uploaders that mis-read the recorded "
                          "layout as ~1 second. Lossless -- no re-encode -- "
                          "but the bytes then differ from the card. Off by "
                          "default: recordings are copied verbatim.")
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
    # The collection spec's known taxonomy is advisory: warn on anything
    # outside it (likely a typo, or a custom environment worth a note) but
    # never block -- the operator may legitimately be collecting elsewhere.
    if args.env_type not in ENVIRONMENTS:
        log.warn("environment type '%s' is outside the collection spec "
                 "(expected one of: %s). Packaging it anyway."
                 % (args.env_type, ", ".join(sorted(ENVIRONMENTS))))
    elif args.env_subcategory not in ENVIRONMENTS[args.env_type]:
        log.warn("'%s' is not a listed %s sub-category (expected one of: %s). "
                 "Packaging it anyway."
                 % (args.env_subcategory, args.env_type,
                    ", ".join(ENVIRONMENTS[args.env_type])))
    if args.env_subcategory in ("other", "other_household") and not args.env_note:
        log.warn("sub-category is '%s' but no --env-note was given"
                 % args.env_subcategory)
    if not args.capture_date:
        args.capture_date = _dt.date.today().isoformat()
        log.warn("no --capture-date given; using today (%s). The camera has "
                 "no real-time clock, so card timestamps cannot supply it."
                 % args.capture_date)

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
            log.info("Calibration: %s (diagonal FOV %s deg)"
                     % (os.path.basename(args.calibration),
                        calib["diagonal_fov_deg"]))
        except (OSError, ValueError, KeyError) as e:
            print("error: could not read calibration: %s" % e)
            return 2
    else:
        log.warn("no --calibration supplied: the intrinsics and extrinsics "
                 "keys will be absent from every metadata.json.")

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

    written, skipped = 0, 0
    for clip in clips:
        zpath, _meta = package(clip, args, calib, head, log)
        if zpath:
            written += 1
        else:
            skipped += 1

    for dev, mp in we_mounted:
        unmount(dev)
        log.info("Unmounted %s (%s)" % (dev, mp))

    print("\n%d packaged, %d skipped, %d warning(s)"
          % (written, skipped, len(log.warnings)))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

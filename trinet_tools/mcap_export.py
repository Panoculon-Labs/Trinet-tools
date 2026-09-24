"""Convert a Trinet recording (mono or stereo, global or rolling shutter) to MCAP.

The output is a single ``.mcap`` file with ROS 2 messages (CDR encoding), so it
opens directly in Foxglove and can be replayed with ROS 2 tooling
(``ros2 bag play`` with the MCAP storage plugin).

Video is copied, not re-encoded: every frame of the camera's H.264 or H.265
stream becomes one ``foxglove_msgs/msg/CompressedVideo`` message. Frame times
come from the ``.vts`` sidecar (exposure-centre, device clock), never from the
MP4's presentation timestamps.

Topics::

    /cam0/video          foxglove_msgs/msg/CompressedVideo   format "h264" | "h265"
    /cam0/camera_info    sensor_msgs/msg/CameraInfo          every keyframe interval
    /cam0/frame_info     trinet_msgs/msg/FrameInfo           per frame: exposure,
                                                             readout, frame number
    /cam1/...            (stereo only; cam0 = _L = scene-left, cam1 = _R)
    /imu                 sensor_msgs/msg/Imu                 accel m/s^2, gyro rad/s
    /imu/mag             sensor_msgs/msg/MagneticField       tesla (units with a
                                                             live magnetometer)
    /tf_static           tf2_msgs/msg/TFMessage              cam0->imu, cam0->cam1

Recording metadata (device, firmware, shutter type, calibration JSON) is
written as MCAP metadata records named ``trinet`` and ``trinet_calibration``.

Inputs are found by base name. For a stereo take ``take0004``::

    take0004_L.mp4 take0004_R.mp4 take0004_L.vts take0004_R.vts take0004.imu

For a mono take ``rec_1``: ``rec_1.mp4 rec_1.vts rec_1.imu``. Any sidecar that is
missing is recovered from the metadata embedded in the MP4, as is the
calibration.
"""

from __future__ import annotations

import heapq
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from . import calib_blob
from .reader import (TIMING_FRAME_CENTERED, TIMING_PHASE_UNLOCKED, ImuData, VtsData, read_imu,
                     read_vts)
from .tmf import read_tmf

# --------------------------------------------------------------------------
# ROS 2 message definitions (concatenated msgdef format)
# --------------------------------------------------------------------------

_SEP = "=" * 80 + "\n"
_TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
_HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
_VEC3 = "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"
_QUAT = "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"


def _msgdef(body: str, *deps: str) -> str:
    return body + "".join(_SEP + d for d in deps)


MSGDEFS = {
    "foxglove_msgs/msg/CompressedVideo": _msgdef(
        "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n",
        _TIME),
    "sensor_msgs/msg/CameraInfo": _msgdef(
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
        "float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n"
        "uint32 binning_y\nsensor_msgs/RegionOfInterest roi\n",
        _HEADER, _TIME,
        "MSG: sensor_msgs/RegionOfInterest\nuint32 x_offset\nuint32 y_offset\n"
        "uint32 height\nuint32 width\nbool do_rectify\n"),
    "sensor_msgs/msg/Imu": _msgdef(
        "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n"
        "float64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n"
        "float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\n"
        "float64[9] linear_acceleration_covariance\n",
        _HEADER, _TIME, _QUAT, _VEC3),
    "sensor_msgs/msg/MagneticField": _msgdef(
        "std_msgs/Header header\ngeometry_msgs/Vector3 magnetic_field\n"
        "float64[9] magnetic_field_covariance\n",
        _HEADER, _TIME, _VEC3),
    "tf2_msgs/msg/TFMessage": _msgdef(
        "geometry_msgs/TransformStamped[] transforms\n",
        "MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\n"
        "string child_frame_id\ngeometry_msgs/Transform transform\n",
        _HEADER, _TIME,
        "MSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\n"
        "geometry_msgs/Quaternion rotation\n",
        _VEC3, _QUAT),
    "trinet_msgs/msg/FrameInfo": _msgdef(
        "# Per-frame timing for one camera. header.stamp is the frame time used on\n"
        "# /camN/video. For a rolling shutter, row r was exposed at\n"
        "#   header.stamp + (r - r_ref) * readout_time_us / height\n"
        "# with r_ref = height/2 when frame_centered, else 0 (top row).\n"
        "std_msgs/Header header\nuint32 frame_number\nuint32 exposure_us\n"
        "uint32 readout_time_us\nbool rolling_shutter\nbool frame_centered\n"
        "bool phase_locked\nuint64 device_sof_ns\n",
        _HEADER, _TIME),
}


def _stamp(ns: int) -> dict:
    ns = int(ns)
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


# --------------------------------------------------------------------------
# Input discovery
# --------------------------------------------------------------------------

@dataclass
class CameraInput:
    name: str               # "cam0" / "cam1"
    mp4: Path
    vts: VtsData
    codec: str = "h264"     # "h264" | "h265"
    width: int = 0
    height: int = 0


@dataclass
class Recording:
    base: str
    cameras: List[CameraInput]
    imu: Optional[ImuData]
    calibration: Optional[dict]          # normalised, see _normalise_calibration
    meta: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _probe(mp4: Path) -> Tuple[str, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height", "-of", "json", str(mp4)],
        check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    s = json.loads(out)["streams"][0]
    codec = {"h264": "h264", "hevc": "h265"}.get(s["codec_name"])
    if codec is None:
        raise ValueError(f"{mp4.name}: unsupported video codec {s['codec_name']!r}")
    return codec, int(s["width"]), int(s["height"])


def _sidecar_or_embedded(path: Path, mp4: Path, kind: str, tmpdir: str, notes: list):
    """Read a .vts/.imu sidecar, or rebuild it from the MP4's embedded track."""
    if path.exists():
        return read_vts(str(path)) if kind == "vts" else read_imu(str(path))
    rec = read_tmf(mp4)
    data = rec.vts_bytes() if kind == "vts" else rec.imu_bytes()
    if not data:
        return None
    tmp = Path(tmpdir) / f"{mp4.stem}.{kind}"
    tmp.write_bytes(data)
    notes.append(f"{path.name} not found; used the copy embedded in {mp4.name}")
    return read_vts(str(tmp)) if kind == "vts" else read_imu(str(tmp))


def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return float(x), float(y), float(z), float(w)


def _normalise_calibration(raw: dict) -> dict:
    """Accept an unpacked TBLC blob (v1 mono / v2 stereo) or a calibration.json
    in the same shape, and return {cameras: [...], T_cam0_imu, T_cam1_cam0, imu}."""
    if "cameras" in raw:                          # v2 blob / stereo JSON
        cams = raw["cameras"]
        out = {"cameras": cams, "T_cam0_imu": raw.get("T_cam0_imu"),
               "T_cam1_cam0": raw.get("T_cam1_cam0"), "imu": raw.get("imu")}
    else:                                          # v1 mono blob
        ex = raw.get("extrinsics", {})
        cams = [{"intrinsics": raw["intrinsics"],
                 "timeshift_cam_imu_s": ex.get("timeshift_cam_imu_s", 0.0),
                 "reprojection_rms_px": raw.get("quality", {}).get("reprojection_rms_px")}]
        out = {"cameras": cams,
               "T_cam0_imu": ex.get("T_cam_imu") if ex.get("valid", True) else None,
               "T_cam1_cam0": None, "imu": raw.get("imu")}
    return out


def discover(path: str, calibration_json: Optional[str] = None) -> Recording:
    """Find a recording's files from any of its paths (an MP4, a sidecar, or
    the directory + base name, e.g. ``/data/take0004``)."""
    p = Path(path)
    stem = p.stem if p.suffix else p.name
    folder = p.parent if p.suffix or not p.is_dir() else p
    if p.is_dir():
        raise ValueError("pass a recording (e.g. dir/take0004 or dir/take0004_L.mp4), not a directory")
    base = stem[:-2] if stem.endswith(("_L", "_R")) else stem
    notes: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="trinet_mcap_")

    have = [suf for suf in ("_L", "_R") if (folder / f"{base}{suf}.mp4").exists()]
    if len(have) == 2:
        eyes = [("cam0", "_L"), ("cam1", "_R")]
    elif have:                                     # one eye of a stereo take
        eyes = [("cam0", have[0])]
        notes.append(f"only {base}{have[0]}.mp4 found; writing a single camera")
    else:
        eyes = [("cam0", "")]
    cams: List[CameraInput] = []
    for name, suf in eyes:
        mp4 = folder / f"{base}{suf}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(mp4)
        vts = _sidecar_or_embedded(folder / f"{base}{suf}.vts", mp4, "vts", tmpdir, notes)
        if vts is None:
            raise ValueError(f"{mp4.name}: no .vts sidecar and no embedded frame timestamps")
        codec, w, h = _probe(mp4)
        cams.append(CameraInput(name=name, mp4=mp4, vts=vts, codec=codec, width=w, height=h))

    imu = _sidecar_or_embedded(folder / f"{base}.imu", cams[0].mp4, "imu", tmpdir, notes)
    if imu is None:
        notes.append("no IMU data found; /imu is omitted")

    rec_meta = {}
    calib = None
    try:
        tmf = read_tmf(cams[0].mp4)
        rec_meta = tmf.meta or {}
        if tmf.calib_blob:
            calib = _normalise_calibration(calib_blob.unpack(tmf.calib_blob))
    except Exception as e:                        # older files carry no metadata track
        notes.append(f"no embedded metadata in {cams[0].mp4.name} ({e})")
    if calibration_json:
        calib = _normalise_calibration(json.loads(Path(calibration_json).read_text()))
        notes.append(f"calibration from {calibration_json}")
    elif calib is None:
        notes.append("no calibration found; /camN/camera_info and /tf_static are omitted")
    return Recording(base=base, cameras=cams, imu=imu, calibration=calib,
                     meta=rec_meta, notes=notes)


# --------------------------------------------------------------------------
# Video: split the elementary stream into one access unit per frame
# --------------------------------------------------------------------------

_AUD = {"h264": b"\x00\x00\x01\x09", "h265": b"\x00\x00\x01\x46\x01"}


def iter_access_units(mp4: Path, codec: str, chunk: int = 1 << 20) -> Iterator[bytes]:
    """Yield the video's frames in decode order, each an Annex B access unit
    (keyframes carry their parameter sets). Stream copy, no re-encode."""
    bsf = {"h264": "h264_mp4toannexb,h264_metadata=aud=insert",
           "h265": "hevc_mp4toannexb,hevc_metadata=aud=insert"}[codec]
    fmt = {"h264": "h264", "h265": "hevc"}[codec]
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp4), "-map", "0:v:0", "-c:v", "copy",
         "-bsf:v", bsf, "-f", fmt, "-"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    marker = _AUD[codec]
    buf = bytearray()
    cur = None              # start of the access unit being collected
    try:
        while True:
            data = proc.stdout.read(chunk)
            if data:
                buf += data
            while True:
                # Skip past the current unit's own delimiter before searching.
                i = buf.find(marker, 0 if cur is None else cur + 5)
                if i < 0:
                    break
                cut = i - 1 if i > 0 and buf[i - 1] == 0 else i   # 4-byte start code
                if cur is not None:
                    yield bytes(buf[cur:cut])
                del buf[:cut]
                cur = 0
            if not data:
                break
        if cur is not None and buf:
            yield bytes(buf)
    finally:
        proc.stdout.close()
        proc.wait()


# --------------------------------------------------------------------------
# Time mapping
# --------------------------------------------------------------------------

def frame_times_ns(vts: VtsData, use_global: bool) -> np.ndarray:
    """Per-frame output timestamps: exposure-centre on the device clock, or on
    the multi-camera master clock when the take was synced."""
    return (vts.global_sof_ns() if use_global else vts.best_timestamps_ns.astype(np.int64))


def device_to_output_ns(vts: VtsData, t_dev: np.ndarray, use_global: bool) -> np.ndarray:
    """Map other device-clock times (IMU) onto the same timeline as the frames."""
    t_dev = t_dev.astype(np.int64)
    if not use_global:
        return t_dev
    sof = vts.best_timestamps_ns.astype(np.int64)
    glob = vts.global_sof_ns()
    if len(sof) < 2:
        return t_dev
    off = np.interp(t_dev.astype(np.float64), sof.astype(np.float64),
                    (glob - sof).astype(np.float64))
    return t_dev + off.astype(np.int64)


# --------------------------------------------------------------------------
# Message builders
# --------------------------------------------------------------------------

def _camera_info(cam: dict, frame_id: str, stamp_ns: int) -> dict:
    intr = cam["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    d = list(map(float, intr.get("distortion", [])))
    if intr.get("model") == "equidistant":
        model = "equidistant"                 # Kannala-Brandt k1..k4 (ROS fisheye)
    else:
        model = "plumb_bob"                   # radtan k1 k2 p1 p2 [k3]
        d = (d + [0.0] * 5)[:5]
    w, h = intr["image_size"]
    return {
        "header": {"stamp": _stamp(stamp_ns), "frame_id": frame_id},
        "height": int(h), "width": int(w), "distortion_model": model, "d": d,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0, "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


def _transform(parent: str, child: str, T: np.ndarray, stamp_ns: int) -> dict:
    """ROS TransformStamped: the pose of ``child`` expressed in ``parent``."""
    T = np.asarray(T, dtype=float)
    x, y, z, w = _rot_to_quat(T[:3, :3])
    return {
        "header": {"stamp": _stamp(stamp_ns), "frame_id": parent},
        "child_frame_id": child,
        "transform": {"translation": {"x": T[0, 3], "y": T[1, 3], "z": T[2, 3]},
                      "rotation": {"x": x, "y": y, "z": z, "w": w}},
    }


def _imu_covariances(calib: Optional[dict], rate_hz: float):
    """Per-sample white-noise covariance from the calibrated noise densities
    (sigma^2 = density^2 * rate); unknown (all zero) when there is no model."""
    nm = ((calib or {}).get("imu") or {}).get("noise_model") or {}
    g, a = nm.get("gyro_noise_density"), nm.get("accel_noise_density")
    def diag(v):
        return [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]
    gc = diag(float(g) ** 2 * rate_hz) if g and rate_hz > 0 else [0.0] * 9
    ac = diag(float(a) ** 2 * rate_hz) if a and rate_hz > 0 else [0.0] * 9
    return gc, ac


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

@dataclass
class ExportStats:
    frames: dict = field(default_factory=dict)
    imu_samples: int = 0
    mag_samples: int = 0
    notes: List[str] = field(default_factory=list)


def shutter_info(vts: VtsData) -> Tuple[str, int, bool]:
    """(shutter, readout_us, frame_centered) from the .vts timing flags."""
    flags = int(vts.timing_flags[0]) if vts.timing_flags is not None and len(vts.timing_flags) else 0
    ro = vts.readout_time_us
    readout = int(np.median(ro[ro > 0])) if ro is not None and np.any(ro > 0) else 0
    if vts.is_rolling_shutter:
        return "rolling", readout, bool(flags & TIMING_FRAME_CENTERED)
    if vts.is_global_shutter:
        return "global", 0, False
    return "unknown", 0, False


def export(rec: Recording, out_path: str, *, use_global_clock: bool = True,
           apply_timeshift: bool = False, compression: str = "zstd",
           camera_info_every: int = 30, progress=None) -> ExportStats:
    from mcap.writer import CompressionType
    from mcap_ros2.writer import Writer

    stats = ExportStats(notes=list(rec.notes))
    comp = {"zstd": CompressionType.ZSTD, "lz4": CompressionType.LZ4,
            "none": CompressionType.NONE}[compression]
    calib = rec.calibration
    cams = rec.cameras
    ref_vts = cams[0].vts

    frame_t = [frame_times_ns(c.vts, use_global_clock) for c in cams]
    # Kalibr convention t_imu = t_cam + timeshift: shift camera times onto the IMU clock.
    if apply_timeshift and calib:
        for i, cam in enumerate(calib["cameras"][:len(cams)]):
            ts = float(cam.get("timeshift_cam_imu_s") or 0.0)
            frame_t[i] = frame_t[i] + np.int64(round(ts * 1e9))

    t0 = int(min(ft[0] for ft in frame_t))

    with open(out_path, "wb") as f:
        writer = Writer(f, chunk_size=4 << 20, compression=comp)
        schemas = {k: writer.register_msgdef(k, v) for k, v in MSGDEFS.items()}

        # ---- metadata ---------------------------------------------------
        shutters = [shutter_info(c.vts) for c in cams]
        meta = {
            "base_name": rec.base,
            "cameras": str(len(cams)),
            "codec": ",".join(c.codec for c in cams),
            "shutter": shutters[0][0],
            "readout_time_us": str(shutters[0][1]),
            "frame_centered": str(shutters[0][2]).lower(),
            "clock": "multi-camera master clock" if use_global_clock and ref_vts.synced
                     else "device monotonic clock",
            "camera_time_reference": "exposure centre" + (
                " of the middle row" if shutters[0][2] else
                " of the top row" if shutters[0][0] == "rolling" else ""),
            "timeshift_applied": str(bool(apply_timeshift and calib)).lower(),
            "cam0": "left eye (_L)" if len(cams) == 2 else "camera",
        }
        if len(cams) == 2:
            meta["cam1"] = "right eye (_R)"
        for k in ("device_id", "fw_version", "hw_generation", "source"):
            if rec.meta.get(k) is not None:
                meta[k] = str(rec.meta[k])
        if (rec.meta.get("drops") or {}).get("dropped") is not None:
            meta["frames_dropped"] = str(rec.meta["drops"]["dropped"])
        writer._writer.add_metadata("trinet", meta)
        if calib:
            writer._writer.add_metadata("trinet_calibration", {"json": json.dumps(calib)})

        # ---- static transforms -----------------------------------------
        if calib:
            tfs = []
            if calib.get("T_cam0_imu") is not None:
                tfs.append(_transform("cam0", "imu", calib["T_cam0_imu"], t0))
            if calib.get("T_cam1_cam0") is not None and len(cams) == 2:
                T_cam0_cam1 = np.linalg.inv(np.asarray(calib["T_cam1_cam0"], dtype=float))
                tfs.append(_transform("cam0", "cam1", T_cam0_cam1, t0))
            if tfs:
                writer.write_message("/tf_static", schemas["tf2_msgs/msg/TFMessage"],
                                     {"transforms": tfs}, log_time=t0, publish_time=t0)

        # ---- time-ordered streams --------------------------------------
        def camera_stream(ci: int):
            cam = cams[ci]
            ts = frame_t[ci]
            vts = cam.vts
            shutter, readout, centered = shutters[ci]
            flags = vts.timing_flags if vts.timing_flags is not None else np.zeros(len(ts), np.uint32)
            expo = vts.exposure_us if vts.exposure_us is not None else np.zeros(len(ts), np.uint32)
            ro = vts.readout_time_us if vts.readout_time_us is not None else np.zeros(len(ts), np.uint32)
            dev = vts.best_timestamps_ns
            n = 0
            for n, au in enumerate(iter_access_units(cam.mp4, cam.codec)):
                if n >= len(ts):
                    stats.notes.append(f"{cam.mp4.name}: video has more frames than the .vts "
                                       f"({len(ts)}); extra frames dropped")
                    break
                t = int(ts[n])
                yield (t, 0, ci, "video", n, au)
                yield (t, 1, ci, "info", n, (int(vts.frame_numbers[n]), int(expo[n]), int(ro[n]),
                                              shutter == "rolling", centered,
                                              not bool(int(flags[n]) & TIMING_PHASE_UNLOCKED),
                                              int(dev[n])))
                if calib and ci < len(calib["cameras"]) and n % camera_info_every == 0:
                    yield (t, 2, ci, "caminfo", n, None)
            else:
                n += 1
                if n < len(ts):
                    stats.notes.append(f"{cam.mp4.name}: video has {n} frames, .vts has {len(ts)}")

        def imu_stream():
            imu = rec.imu
            t = device_to_output_ns(ref_vts, imu.timestamps_ns, use_global_clock)
            for i in range(imu.num_samples):
                yield (int(t[i]), 3, -1, "imu", i, None)

        def mag_stream():
            imu = rec.imu
            if imu.mag_age_us is None or imu.header.version < 5 or not imu.header.mag_present:
                return
            # Each magnetometer reading is repeated on the IMU samples that follow
            # it; a new reading is a change in value. Its own time is the sample
            # time minus mag_age_us.
            have = imu.mag_age_us > 0
            m = imu.mag
            new = have.copy()
            new[1:] &= np.any(m[1:] != m[:-1], axis=1) | ~have[:-1]
            idx = np.nonzero(new)[0]
            t_mag = imu.timestamps_ns[idx].astype(np.int64) - imu.mag_age_us[idx].astype(np.int64) * 1000
            t_out = device_to_output_ns(ref_vts, t_mag, use_global_clock)
            order = np.argsort(t_out, kind="stable")
            for k in order:
                yield (int(t_out[k]), 4, -1, "mag", int(idx[k]), None)

        streams = [camera_stream(i) for i in range(len(cams))]
        if rec.imu is not None:
            streams += [imu_stream(), mag_stream()]
            rate = rec.imu.actual_rate_hz
            gyro_cov, acc_cov = _imu_covariances(calib, rate)

        for t, _, ci, kind, idx, payload in heapq.merge(*streams, key=lambda e: (e[0], e[1], e[2])):
            if kind == "video":
                cam = cams[ci]
                writer.write_message(f"/{cam.name}/video", schemas["foxglove_msgs/msg/CompressedVideo"],
                                     {"timestamp": _stamp(t), "frame_id": cam.name,
                                      "data": payload, "format": cam.codec},
                                     log_time=t, publish_time=t)
                stats.frames[cam.name] = stats.frames.get(cam.name, 0) + 1
                if progress and ci == 0 and idx % 300 == 0:
                    progress(idx, len(frame_t[0]))
            elif kind == "info":
                fn, ex, ro, rolling, centered, locked, dev = payload
                writer.write_message(f"/{cams[ci].name}/frame_info", schemas["trinet_msgs/msg/FrameInfo"],
                                     {"header": {"stamp": _stamp(t), "frame_id": cams[ci].name},
                                      "frame_number": fn, "exposure_us": ex, "readout_time_us": ro,
                                      "rolling_shutter": rolling, "frame_centered": centered,
                                      "phase_locked": locked, "device_sof_ns": dev},
                                     log_time=t, publish_time=t)
            elif kind == "caminfo":
                writer.write_message(f"/{cams[ci].name}/camera_info", schemas["sensor_msgs/msg/CameraInfo"],
                                     _camera_info(calib["cameras"][ci], cams[ci].name, t),
                                     log_time=t, publish_time=t)
            elif kind == "imu":
                imu = rec.imu
                a, g = imu.accel[idx], imu.gyro[idx]
                writer.write_message("/imu", schemas["sensor_msgs/msg/Imu"], {
                    "header": {"stamp": _stamp(t), "frame_id": "imu"},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "orientation_covariance": [-1.0] + [0.0] * 8,   # not provided
                    "angular_velocity": {"x": float(g[0]), "y": float(g[1]), "z": float(g[2])},
                    "angular_velocity_covariance": gyro_cov,
                    "linear_acceleration": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
                    "linear_acceleration_covariance": acc_cov,
                }, log_time=t, publish_time=t)
                stats.imu_samples += 1
            elif kind == "mag":
                m = rec.imu.mag[idx] * 1e-6            # uT -> T
                writer.write_message("/imu/mag", schemas["sensor_msgs/msg/MagneticField"], {
                    "header": {"stamp": _stamp(t), "frame_id": "imu"},
                    "magnetic_field": {"x": float(m[0]), "y": float(m[1]), "z": float(m[2])},
                    "magnetic_field_covariance": [0.0] * 9,
                }, log_time=t, publish_time=t)
                stats.mag_samples += 1

        if calib:
            writer._writer.add_attachment(
                create_time=t0, log_time=t0, name="calibration.json",
                media_type="application/json", data=json.dumps(calib, indent=1).encode())
        writer.finish()
    return stats

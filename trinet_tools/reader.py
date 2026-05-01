"""
Author: Panoculon Labs

Trinet Toolkit — Binary Reader for .imu and .vts files

This module reads the binary sidecar files produced by the Trinet IMU recorder.
It requires only numpy (no OpenCV, no scipy, no torch).

File formats:

  .imu  — Timestamped inertial measurements (accel, gyro, mag, temperature).
           Earlier firmware versions also stored fused orientation quaternion
           and gravity-free linear acceleration; current firmware zeroes these
           fields — use post-processing (e.g. Madgwick / complementary filter)
           to derive orientation from raw accel + gyro.
  .vts  — Per-frame video timestamps mapping each MP4 frame to a clock time.

Both files use little-endian byte order and monotonic nanosecond timestamps.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import numpy as np


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

IMU_HEADER_FMT_V2 = "<8sIIHHQQ28s"
IMU_HEADER_FMT_V3 = "<8sIIHHQQI24s"        # +flags(uint32) replaces 4 bytes of reserved
IMU_HEADER_SIZE = 64
IMU_SAMPLE_FMT_V1 = "<Qfffffffff"          # 44 bytes
IMU_SAMPLE_SIZE_V1 = 44
IMU_SAMPLE_FMT_V2 = "<Qfffffffffffffffff"  # 76 bytes
IMU_SAMPLE_SIZE_V2 = 76
IMU_SAMPLE_FMT_V3 = "<Qffffffffffffffffff" # 80 bytes (+fsync_delay_us)
IMU_SAMPLE_SIZE_V3 = 80

IMU_HDR_FLAG_FSYNC = 0x01

VTS_HEADER_FMT = "<8sII16s"
VTS_HEADER_SIZE = 32
VTS_ENTRY_FMT_V1 = "<IQ"
VTS_ENTRY_SIZE_V1 = 12

# v2: frame_number(uint32), sof_timestamp_ns(uint64), venc_seq(uint32), venc_pts_us(uint64)
VTS_ENTRY_FMT_V2 = "<IQIQ"
VTS_ENTRY_SIZE_V2 = 24

ACCEL_FS_NAMES = {0: "±2 g", 1: "±4 g", 2: "±8 g", 3: "±16 g"}
GYRO_FS_NAMES = {0: "±250 dps", 1: "±500 dps", 2: "±1000 dps", 3: "±2000 dps"}


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------

@dataclass
class ImuHeader:
    magic: str
    version: int
    sample_rate_hz: int
    accel_fs: int        # 0=2g, 1=4g, 2=8g, 3=16g
    gyro_fs: int         # 0=250, 1=500, 2=1000, 3=2000 dps
    start_time_ns: int
    video_start_ns: int
    flags: int = 0       # v3+: bit 0 = FSYNC enabled
    # 16-byte public per-unit identifier, derived one-way from a stable
    # factory-burned hardware identifier on the camera. Lives in the first
    # 16 bytes of the header's reserved region; all-zero means the recording
    # was made before device_id support landed.
    device_id: bytes = b"\x00" * 16

    @property
    def accel_fs_name(self) -> str:
        return ACCEL_FS_NAMES.get(self.accel_fs, f"unknown({self.accel_fs})")

    @property
    def gyro_fs_name(self) -> str:
        return GYRO_FS_NAMES.get(self.gyro_fs, f"unknown({self.gyro_fs})")

    @property
    def fsync_enabled(self) -> bool:
        return bool(self.flags & IMU_HDR_FLAG_FSYNC)

    @property
    def device_id_hex(self) -> str:
        """Lowercase hex of the device ID; empty string if unset (all-zero)."""
        if not any(self.device_id):
            return ""
        return self.device_id.hex()


@dataclass
class VtsHeader:
    magic: str
    version: int
    frame_rate_milli: int   # fps * 1000

    @property
    def fps(self) -> float:
        return self.frame_rate_milli / 1000.0


@dataclass
class ImuData:
    """All IMU samples from a single .imu recording."""

    header: ImuHeader
    timestamps_ns: np.ndarray       # (N,) uint64  — monotonic nanoseconds
    accel: np.ndarray               # (N,3) float32 — m/s² (includes gravity)
    gyro: np.ndarray                # (N,3) float32 — rad/s
    mag: np.ndarray                 # (N,3) float32 — µT
    temp_c: Optional[np.ndarray] = None           # (N,) float32  — °C  (v2+)
    quat_xyzw: Optional[np.ndarray] = None        # (N,4) float32 — unit quaternion XYZW (v2+)
    lin_accel: Optional[np.ndarray] = None        # (N,3) float32 — m/s², gravity removed (v2+)
    fsync_delay_us: Optional[np.ndarray] = None   # (N,) float32  — FSYNC delay µs (v3, 0=no pulse)

    @property
    def timestamps_s(self) -> np.ndarray:
        """Timestamps in seconds, starting from 0."""
        return (self.timestamps_ns - self.timestamps_ns[0]).astype(np.float64) / 1e9

    @property
    def num_samples(self) -> int:
        return len(self.timestamps_ns)

    @property
    def duration_s(self) -> float:
        if self.num_samples < 2:
            return 0.0
        return float(self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9

    @property
    def actual_rate_hz(self) -> float:
        if self.num_samples < 2:
            return 0.0
        return (self.num_samples - 1) / self.duration_s


@dataclass
class VtsData:
    """All per-frame timestamps from a single .vts file."""

    header: VtsHeader
    frame_numbers: np.ndarray       # (M,) uint32
    timestamps_ns: np.ndarray       # (M,) uint64
    sof_timestamps_ns: Optional[np.ndarray] = None  # (M,) uint64 (v2 only; 0 if unknown)
    venc_seq: Optional[np.ndarray] = None           # (M,) uint32 (v2 only)
    venc_pts_us: Optional[np.ndarray] = None        # (M,) uint64 (v2 only)

    @property
    def timestamps_s(self) -> np.ndarray:
        """Timestamps in seconds, starting from 0."""
        return (self.timestamps_ns - self.timestamps_ns[0]).astype(np.float64) / 1e9

    @property
    def best_timestamps_ns(self) -> np.ndarray:
        """
        Best-effort frame-time reference.

        - For v1: `timestamps_ns` is the frame-capture callback timestamp.
        - For v2: `timestamps_ns` is derived from encoder PTS, and
          `sof_timestamps_ns` (if nonzero) is a separate frame-capture
          timestamp.  Prefers frame-capture times when present.
        """
        if self.sof_timestamps_ns is None:
            return self.timestamps_ns
        if np.any(self.sof_timestamps_ns != 0):
            return self.sof_timestamps_ns
        return self.timestamps_ns

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def duration_s(self) -> float:
        if self.num_frames < 2:
            return 0.0
        return float(self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9


# ---------------------------------------------------------------------------
#  Reader functions
# ---------------------------------------------------------------------------

def read_imu(path: str) -> ImuData:
    """
    Read a Trinet .imu binary file.

    Parameters
    ----------
    path : str or Path
        Path to the .imu file.

    Returns
    -------
    ImuData
        Parsed IMU recording with numpy arrays for every sensor channel.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw_header = f.read(IMU_HEADER_SIZE)
        if len(raw_header) < IMU_HEADER_SIZE:
            raise ValueError(f"File too small for header ({len(raw_header)} bytes)")

        # Peek at version to pick the right header struct
        peek_ver = struct.unpack_from("<I", raw_header, 8)[0]
        if peek_ver >= 3:
            fields = struct.unpack(IMU_HEADER_FMT_V3, raw_header)
            flags = fields[7]
            reserved_bytes = fields[8]
        else:
            fields = struct.unpack(IMU_HEADER_FMT_V2, raw_header)
            flags = 0
            reserved_bytes = fields[7]

        magic = fields[0].rstrip(b"\x00").decode("ascii", errors="replace")
        if magic != "TRIMU001":
            raise ValueError(f"Not a Trinet IMU file (magic={magic!r}, expected 'TRIMU001')")

        # First 16 bytes of reserved hold the public device_id. All-zero means
        # the recording predates device_id support; ImuHeader.device_id_hex
        # surfaces this as an empty string.
        device_id = reserved_bytes[:16] if reserved_bytes else b"\x00" * 16

        header = ImuHeader(
            magic=magic, version=fields[1], sample_rate_hz=fields[2],
            accel_fs=fields[3], gyro_fs=fields[4],
            start_time_ns=fields[5], video_start_ns=fields[6],
            flags=flags,
            device_id=device_id,
        )
        data = f.read()

    is_v3 = header.version >= 3
    is_v2 = header.version >= 2
    if is_v3:
        sample_size = IMU_SAMPLE_SIZE_V3
        sample_fmt = IMU_SAMPLE_FMT_V3
    elif is_v2:
        sample_size = IMU_SAMPLE_SIZE_V2
        sample_fmt = IMU_SAMPLE_FMT_V2
    else:
        sample_size = IMU_SAMPLE_SIZE_V1
        sample_fmt = IMU_SAMPLE_FMT_V1
    num_samples = len(data) // sample_size

    timestamps = np.zeros(num_samples, dtype=np.uint64)
    accel = np.zeros((num_samples, 3), dtype=np.float32)
    gyro = np.zeros((num_samples, 3), dtype=np.float32)
    mag = np.zeros((num_samples, 3), dtype=np.float32)
    temp_c = np.zeros(num_samples, dtype=np.float32) if is_v2 else None
    quat_xyzw = np.zeros((num_samples, 4), dtype=np.float32) if is_v2 else None
    lin_accel = np.zeros((num_samples, 3), dtype=np.float32) if is_v2 else None
    fsync_delay_us = np.zeros(num_samples, dtype=np.float32) if is_v3 else None

    for i in range(num_samples):
        offset = i * sample_size
        s = struct.unpack(sample_fmt, data[offset:offset + sample_size])
        timestamps[i] = s[0]
        accel[i] = s[1:4]
        gyro[i] = s[4:7]
        mag[i] = s[7:10]
        if is_v2:
            temp_c[i] = s[10]
            quat_xyzw[i] = s[11:15]
            lin_accel[i] = s[15:18]
        if is_v3:
            fsync_delay_us[i] = s[18]

    return ImuData(
        header=header, timestamps_ns=timestamps,
        accel=accel, gyro=gyro, mag=mag,
        temp_c=temp_c, quat_xyzw=quat_xyzw, lin_accel=lin_accel,
        fsync_delay_us=fsync_delay_us,
    )


def read_vts(path: str) -> VtsData:
    """
    Read a Trinet .vts (video timestamp) binary file.

    Parameters
    ----------
    path : str or Path
        Path to the .vts file.

    Returns
    -------
    VtsData
        Frame numbers and their corresponding monotonic-clock timestamps.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw_header = f.read(VTS_HEADER_SIZE)
        if len(raw_header) < VTS_HEADER_SIZE:
            raise ValueError(f"File too small for header ({len(raw_header)} bytes)")

        fields = struct.unpack(VTS_HEADER_FMT, raw_header)
        magic = fields[0].rstrip(b"\x00").decode("ascii", errors="replace")
        if magic != "TRIVTS01":
            raise ValueError(f"Not a Trinet VTS file (magic={magic!r}, expected 'TRIVTS01')")

        header = VtsHeader(magic=magic, version=fields[1], frame_rate_milli=fields[2])
        data = f.read()

    if header.version >= 2:
        num_entries = len(data) // VTS_ENTRY_SIZE_V2
        frame_numbers = np.zeros(num_entries, dtype=np.uint32)
        sof_ts = np.zeros(num_entries, dtype=np.uint64)
        venc_seq = np.zeros(num_entries, dtype=np.uint32)
        venc_pts_us = np.zeros(num_entries, dtype=np.uint64)

        for i in range(num_entries):
            offset = i * VTS_ENTRY_SIZE_V2
            fn, sof, seq, pts = struct.unpack(VTS_ENTRY_FMT_V2, data[offset:offset + VTS_ENTRY_SIZE_V2])
            frame_numbers[i] = fn
            sof_ts[i] = sof
            venc_seq[i] = seq
            venc_pts_us[i] = pts

        # Use VENC PTS as the primary timeline (monotonic, per encoded frame)
        timestamps = (venc_pts_us.astype(np.uint64) * np.uint64(1000))

        return VtsData(
            header=header,
            frame_numbers=frame_numbers,
            timestamps_ns=timestamps,
            sof_timestamps_ns=sof_ts,
            venc_seq=venc_seq,
            venc_pts_us=venc_pts_us,
        )

    # v1
    num_entries = len(data) // VTS_ENTRY_SIZE_V1
    frame_numbers = np.zeros(num_entries, dtype=np.uint32)
    timestamps = np.zeros(num_entries, dtype=np.uint64)

    for i in range(num_entries):
        offset = i * VTS_ENTRY_SIZE_V1
        fn, ts = struct.unpack(VTS_ENTRY_FMT_V1, data[offset:offset + VTS_ENTRY_SIZE_V1])
        frame_numbers[i] = fn
        timestamps[i] = ts

    return VtsData(header=header, frame_numbers=frame_numbers, timestamps_ns=timestamps)


# ---------------------------------------------------------------------------
#  IMU ↔ Video alignment
# ---------------------------------------------------------------------------

def compute_fsync_frame_timestamps(imu: ImuData, vts: VtsData) -> Optional[np.ndarray]:
    """
    Derive precise frame timestamps from the hardware sync delay field.

    Each camera frame pulse is captured as a delay (microseconds) in the
    motion sensor data.  For each approximate VTS frame timestamp we find
    the nearest motion sample that recorded a pulse and reconstruct the
    exact frame time as:

        frame_time = imu_ts[idx] - sync_delay_us[idx] * 1000

    Returns (M,) uint64 nanosecond timestamps (one per VTS frame), or None.
    """
    if imu.fsync_delay_us is None or not imu.header.fsync_enabled:
        return None
    if imu.num_samples < 2 or vts.num_frames == 0:
        return None

    fsync = imu.fsync_delay_us.astype(np.float64)
    ts = imu.timestamps_ns.astype(np.float64)
    vts_ref = vts.best_timestamps_ns.astype(np.float64)

    nonzero = np.where(fsync > 0)[0]
    if len(nonzero) < 2:
        return None

    out = np.empty(vts.num_frames, dtype=np.uint64)
    search_start = int(nonzero[0])
    max_step = max(8, int(round(imu.actual_rate_hz / max(vts.header.fps, 1.0))) * 2)

    for fi, approx_ts in enumerate(vts_ref):
        idx = int(np.searchsorted(ts, approx_ts, side="left"))
        if idx < search_start:
            idx = search_start

        lo = max(search_start, idx - 4)
        hi = min(len(ts), idx + max_step)

        best_idx = -1
        best_err = None
        for j in range(lo, hi):
            if fsync[j] <= 0:
                continue
            sof_ns = ts[j] - fsync[j] * 1000.0
            err = abs(sof_ns - approx_ts)
            if best_err is None or err < best_err:
                best_err = err
                best_idx = j

        if best_idx < 0:
            return None

        out[fi] = int(ts[best_idx] - fsync[best_idx] * 1000.0)
        search_start = best_idx
    return out


def get_per_frame_fsync_delay_us(imu: ImuData, vts: VtsData) -> Optional[np.ndarray]:
    """
    Extract the per-frame sync delay (µs) — one value per video frame.

    This is the hardware-measured time between a camera frame pulse and
    the motion sensor sample that captured it.

    Returns (M,) float64 array of delays in microseconds, or None.
    """
    if imu.fsync_delay_us is None or not imu.header.fsync_enabled:
        return None
    if imu.num_samples < 2 or vts.num_frames == 0:
        return None

    fsync = imu.fsync_delay_us.astype(np.float64)
    ts = imu.timestamps_ns.astype(np.float64)
    vts_ref = vts.best_timestamps_ns.astype(np.float64)

    nonzero = np.where(fsync > 0)[0]
    if len(nonzero) < 2:
        return None

    out = np.empty(vts.num_frames, dtype=np.float64)
    search_start = int(nonzero[0])
    max_step = max(8, int(round(imu.actual_rate_hz / max(vts.header.fps, 1.0))) * 2)

    for fi, approx_ts in enumerate(vts_ref):
        idx = int(np.searchsorted(ts, approx_ts, side="left"))
        if idx < search_start:
            idx = search_start

        lo = max(search_start, idx - 4)
        hi = min(len(ts), idx + max_step)

        best_idx = -1
        best_err = None
        for j in range(lo, hi):
            if fsync[j] <= 0:
                continue
            sof_ns = ts[j] - fsync[j] * 1000.0
            err = abs(sof_ns - approx_ts)
            if best_err is None or err < best_err:
                best_err = err
                best_idx = j

        if best_idx < 0:
            return None

        out[fi] = fsync[best_idx]
        search_start = best_idx

    return out


def interpolate_imu_to_frames(imu: ImuData, vts: VtsData) -> Dict[str, np.ndarray]:
    """
    Interpolate IMU data to the exact timestamps of each video frame.

    When hardware sync delay data is available, uses precise timestamps for
    frame alignment.  Falls back to VTS best_timestamps_ns otherwise.

    Parameters
    ----------
    imu : ImuData
    vts : VtsData

    Returns
    -------
    dict
        Keys: 'frame_numbers', 'timestamps_ns', 'accel', 'gyro', 'mag',
              optionally 'temp_c', 'quat_xyzw', 'lin_accel' (if v2+),
              and 'fsync_frame_ts_ns' (sync-derived frame times, or None).
    """
    fsync_ts = compute_fsync_frame_timestamps(imu, vts)
    if fsync_ts is not None and len(fsync_ts) == vts.num_frames:
        frame_ts = fsync_ts.astype(np.float64)
    else:
        frame_ts = vts.best_timestamps_ns.astype(np.float64)
        fsync_ts = None

    imu_ts = imu.timestamps_ns.astype(np.float64)

    n = vts.num_frames
    result: Dict[str, np.ndarray] = {
        "frame_numbers": vts.frame_numbers,
        "timestamps_ns": vts.timestamps_ns,
        "fsync_frame_ts_ns": fsync_ts,
        "accel": np.zeros((n, 3), dtype=np.float32),
        "gyro": np.zeros((n, 3), dtype=np.float32),
        "mag": np.zeros((n, 3), dtype=np.float32),
    }
    if imu.temp_c is not None:
        result["temp_c"] = np.zeros(n, dtype=np.float32)
    if imu.quat_xyzw is not None:
        result["quat_xyzw"] = np.zeros((n, 4), dtype=np.float32)
    if imu.lin_accel is not None:
        result["lin_accel"] = np.zeros((n, 3), dtype=np.float32)

    if imu.num_samples < 2 or n == 0:
        return result

    for axis in range(3):
        result["accel"][:, axis] = np.interp(frame_ts, imu_ts, imu.accel[:, axis])
        result["gyro"][:, axis] = np.interp(frame_ts, imu_ts, imu.gyro[:, axis])
        result["mag"][:, axis] = np.interp(frame_ts, imu_ts, imu.mag[:, axis])
        if "lin_accel" in result:
            result["lin_accel"][:, axis] = np.interp(frame_ts, imu_ts, imu.lin_accel[:, axis])

    if "temp_c" in result:
        result["temp_c"][:] = np.interp(frame_ts, imu_ts, imu.temp_c)

    if "quat_xyzw" in result:
        for qi in range(4):
            result["quat_xyzw"][:, qi] = np.interp(frame_ts, imu_ts, imu.quat_xyzw[:, qi])

    return result

"""
Madgwick MARG / IMU fusion for Trinet recordings (post-processing).

Uses the ``ahrs`` library implementation of Madgwick :cite:`madgwick2010`.
Output quaternions are **xyzw** (scalar last), matching ``scipy.spatial.transform.Rotation``.

Units (must match ``trinet_reader.ImuData``):
  * accelerometer: m/s² (including gravity)
  * gyroscope: rad/s
  * magnetometer: µT (the library normalises direction internally; µT is fine)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from ahrs.common.orientation import acc2q, ecompass
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation, Slerp


def _ahrs_wxyz_to_scipy_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert [q_w, q_x, q_y, q_z] (ahrs convention) to [x, y, z, w] (scipy)."""
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(
        (q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0])
    )


def _madgwick_loop(
    acc: np.ndarray,
    gyr: np.ndarray,
    mag: Optional[np.ndarray],
    dt: np.ndarray,
    mean_hz: float,
) -> np.ndarray:
    """Return (N, 4) ahrs-style wxyz quaternions."""
    n = acc.shape[0]
    q_out = np.zeros((n, 4), dtype=np.float64)
    madgwick = Madgwick(frequency=float(max(mean_hz, 1.0)))

    if mag is not None:
        q_out[0] = ecompass(acc[0], mag[0], frame="NED", representation="quaternion")
        for t in range(1, n):
            d = float(dt[t - 1])
            if d <= 0.0 or d > 0.5 or not np.isfinite(d):
                q_out[t] = q_out[t - 1]
                continue
            q_out[t] = madgwick.updateMARG(q_out[t - 1], gyr[t], acc[t], mag[t], dt=d)
    else:
        q_out[0] = acc2q(acc[0])
        for t in range(1, n):
            d = float(dt[t - 1])
            if d <= 0.0 or d > 0.5 or not np.isfinite(d):
                q_out[t] = q_out[t - 1]
                continue
            q_out[t] = madgwick.updateIMU(q_out[t - 1], gyr[t], acc[t], dt=d)
    return q_out


def run_madgwick(
    accel_m_s2: np.ndarray,
    gyro_rad_s: np.ndarray,
    mag_uT: Optional[np.ndarray],
    timestamps_s: np.ndarray,
    *,
    use_mag: bool = True,
    max_samples: int = 50000,
) -> np.ndarray:
    """
    Run Madgwick fusion on full-rate IMU streams.

    Parameters
    ----------
    accel_m_s2, gyro_rad_s
        (N, 3) arrays, float-like.
    mag_uT
        (N, 3) or None. If None or ``use_mag`` is False, uses 6-DOF (IMU-only) Madgwick.
    timestamps_s
        (N,) monotonic timestamps in seconds (any offset; only differences matter).
    use_mag
        If False, ignores magnetometer even if provided.
    max_samples
        If N exceeds this, fusion runs on a strided subset and attitudes are Slerp-filled.

    Returns
    -------
    (N, 4) float64, unit quaternions **xyzw** for ``Rotation.from_quat``.
    """
    acc = np.asarray(accel_m_s2, dtype=np.float64)
    gyr = np.asarray(gyro_rad_s, dtype=np.float64)
    ts = np.asarray(timestamps_s, dtype=np.float64)
    n = acc.shape[0]
    if n == 0:
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    if acc.shape != gyr.shape or acc.shape[1] != 3:
        raise ValueError("accel and gyro must be (N, 3) with matching shapes")

    mag: Optional[np.ndarray]
    if use_mag and mag_uT is not None:
        mag = np.asarray(mag_uT, dtype=np.float64)
        if mag.shape != acc.shape:
            raise ValueError("mag must match accel shape")
    else:
        mag = None

    if n == 1:
        if mag is not None:
            q0 = ecompass(acc[0], mag[0], frame="NED", representation="quaternion")
        else:
            q0 = acc2q(acc[0])
        return _ahrs_wxyz_to_scipy_xyzw(q0.reshape(1, 4))

    if n <= max_samples:
        dt = np.diff(ts)
        mean_hz = float((n - 1) / max(ts[-1] - ts[0], 1e-9))
        q_wxyz = _madgwick_loop(acc, gyr, mag, dt, mean_hz)
        return _ahrs_wxyz_to_scipy_xyzw(q_wxyz)

    step = max(1, n // max_samples)
    idx = np.arange(0, n, step, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    acc_s, gyr_s, ts_s = acc[idx], gyr[idx], ts[idx]
    mag_s = mag[idx] if mag is not None else None
    dt_s = np.diff(ts_s)
    mean_hz_s = float((len(ts_s) - 1) / max(ts_s[-1] - ts_s[0], 1e-9))
    q_sub_wxyz = _madgwick_loop(acc_s, gyr_s, mag_s, dt_s, mean_hz_s)
    q_sub_xyzw = _ahrs_wxyz_to_scipy_xyzw(q_sub_wxyz)
    key_rots = Rotation.from_quat(q_sub_xyzw)
    slerp = Slerp(ts_s, key_rots)
    full = slerp(ts)
    return full.as_quat().astype(np.float64)

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Generate an OpenVINS config set from a Trinet stereo recording.

Writes the three files OpenVINS's `ov_msckf` expects —
`estimator_config.yaml`, `kalibr_imucam_chain.yaml`, `kalibr_imu_chain.yaml`
— from the calibration a Trinet recording carries (the embedded TBLC blob, or
a calibration.json / .bin passed with --calibration).

Conventions handled here so you don't have to:
- Trinet/Kalibr store `T_cam_imu` (IMU->camera); OpenVINS wants `T_imu_cam`
  (camera->IMU) — inverted here.
- Per-camera IMU time offsets ride along as `timeshift_cam_imu` and online
  time-offset calibration stays enabled.
- IMU noise densities from the calibration are inflated (default 5x) — raw
  Kalibr-fit noises are optimistic for a VIO estimator; see the OpenVINS
  docs on noise inflation.

Usage:
    python3 scripts/make_openvins_config.py TAKE_L.mp4 OUT_DIR \
        [--calibration CALIB] [--noise-inflation 5] [--imu-rate 400]
Then run the pipeline with scripts/run_openvins.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for _p in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from trinet_tools import calib_blob                           # noqa: E402
from trinet_tools.tmf import read_tmf                         # noqa: E402

ESTIMATOR_TEMPLATE = """%YAML:1.0 # need to specify the file type at the top!

verbosity: "INFO" # ALL, DEBUG, INFO, WARNING, ERROR, SILENT

use_fej: true # if first-estimate Jacobians should be used (enable for good consistency)
integration: "rk4" # discrete, rk4, analytical (if rk4 or analytical used then analytical covariance propagation is used)
use_stereo: true # if we have more than 1 camera, if we should try to track stereo constraints between pairs
max_cameras: 2 # how many cameras we have 1 = mono, 2 = stereo, >2 = binocular (all mono tracking)

calib_cam_extrinsics: true # if the transform between camera and IMU should be optimized R_ItoC, p_CinI
calib_cam_intrinsics: true # if camera intrinsics should be optimized (focal, center, distortion)
calib_cam_timeoffset: true # if timeoffset between camera and IMU should be optimized
calib_imu_intrinsics: false # if imu intrinsics should be calibrated (rotation and skew-scale matrix)
calib_imu_g_sensitivity: false # if gyroscope gravity sensitivity (Tg) should be calibrated

max_clones: 11 # how many clones in the sliding window
max_slam: 50 # number of features in our state vector
max_slam_in_update: 25 # update can be split into sequential updates of batches, how many in a batch
max_msckf_in_update: 40 # how many MSCKF features to use in the update
dt_slam_delay: 1 # delay before initializing (helps with stability from bad initialization...)

gravity_mag: 9.81 # magnitude of gravity in this location

feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"

# zero velocity update parameters we can use
# we support either IMU-based or disparity detection.
try_zupt: false
zupt_chi2_multipler: 0 # set to 0 for only disp-based
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5 # set to 0 for only imu-based
zupt_only_at_beginning: false

# ==================================================================
# ==================================================================

init_window_time: 2.0 # how many seconds to collect initialization information
init_imu_thresh: 1.0 # threshold for variance of the accelerometer to detect a "jerk" in motion
init_max_disparity: 15.0 # max disparity to consider the platform stationary (dependent on resolution)
init_max_features: 50 # how many features to track during initialization (saves on computation)

init_dyn_use: false # if dynamic initialization should be used
init_dyn_mle_opt_calib: false # if we should optimize calibration during intialization (not recommended)
init_dyn_mle_max_iter: 50 # how many iterations the MLE refinement should use (zero to skip the MLE)
init_dyn_mle_max_time: 0.05 # how many seconds the MLE should be completed in
init_dyn_mle_max_threads: 6 # how many threads the MLE should use
init_dyn_num_pose: 6 # number of poses to use within our window time (evenly spaced)
init_dyn_min_deg: 10.0 # orientation change needed to try to init

init_dyn_inflation_ori: 10 # what to inflate the recovered q_GtoI covariance by
init_dyn_inflation_vel: 100 # what to inflate the recovered v_IinG covariance by
init_dyn_inflation_bg: 10 # what to inflate the recovered bias_g covariance by
init_dyn_inflation_ba: 100 # what to inflate the recovered bias_a covariance by
init_dyn_min_rec_cond: 1e-12 # reciprocal condition number thresh for info inversion

init_dyn_bias_g: [ 0.0, 0.0, 0.0 ] # initial gyroscope bias guess
init_dyn_bias_a: [ 0.0, 0.0, 0.0 ] # initial accelerometer bias guess

# ==================================================================
# ==================================================================

record_timing_information: false # if we want to record timing information of the method
record_timing_filepath: "/tmp/traj_timing.txt" # https://docs.openvins.com/eval-timing.html#eval-ov-timing-flame

# if we want to save the simulation state and its diagional covariance
# use this with rosrun ov_eval error_simulation
save_total_state: false
filepath_est: "/tmp/ov_estimate.txt"
filepath_std: "/tmp/ov_estimate_std.txt"
filepath_gt: "/tmp/ov_groundtruth.txt"

# ==================================================================
# ==================================================================

# our front-end feature tracking parameters
# we have a KLT and descriptor based (KLT is better implemented...)
use_klt: true # if true we will use KLT, otherwise use a ORB descriptor + robust matching
num_pts: 200 # number of points (per camera) we will extract and try to track
fast_threshold: 20 # threshold for fast extraction (warning: lower threshs can be expensive)
grid_x: 5 # extraction sub-grid count for horizontal direction (uniform tracking)
grid_y: 5 # extraction sub-grid count for vertical direction (uniform tracking)
min_px_dist: 10 # distance between features (features near each other provide less information)
knn_ratio: 0.70 # descriptor knn threshold for the top two descriptor matches
track_frequency: 30.0 # frequency we will perform feature tracking at (in frames per second / hertz)
downsample_cameras: true # will downsample image in half if true
num_opencv_threads: 4 # -1: auto, 0-1: serial, >1: number of threads
histogram_method: "HISTOGRAM" # NONE, HISTOGRAM, CLAHE

# aruco tag tracker for the system
# DICT_6X6_1000 from https://chev.me/arucogen/
use_aruco: false
num_aruco: 1024
downsize_aruco: true

# ==================================================================
# ==================================================================

# camera noises and chi-squared threshold multipliers
up_msckf_sigma_px: 1
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1

# masks for our images
use_mask: false

# imu and camera spacial-temporal
# imu config should also have the correct noise values
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"




"""


def _mat_yaml(m, indent="    "):
    return "\n".join(f"{indent}- [{', '.join(f'{v:.12f}' for v in row)}]"
                     for row in m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="a take's _L.mp4 (for the embedded "
                                   "calibration), or ignored with --calibration")
    ap.add_argument("out", type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--noise-inflation", type=float, default=5.0)
    ap.add_argument("--imu-rate", type=float, default=400.0)
    ap.add_argument("--auto-align", action="store_true",
                    help="measure the take's residual vertical stereo offset "
                         "(trinet_tools.stereo_align) and fold it into cam1's "
                         "principal point in the generated camchain — use when "
                         "the mount may have moved since calibration. SOURCE "
                         "must be the take's _L.mp4 with its _R sibling.")
    args = ap.parse_args()

    calib = None
    if args.calibration:
        raw = args.calibration.read_bytes()
        try:
            calib = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            calib = calib_blob.unpack(raw)
    else:
        rec = read_tmf(args.source)
        if rec.calib_blob:
            calib = calib_blob.unpack(rec.calib_blob)
    if not calib or "cameras" not in calib:
        sys.exit("no stereo calibration found (embedded or via --calibration)")

    if args.auto_align:
        from trinet_tools.reader import read_vts
        from trinet_tools.stereo_align import auto_align
        from trinet_tools.tmf import read_tmf as _read_tmf
        import tempfile
        src = Path(args.source)
        mp4_r = Path(str(src).replace("_L.mp4", "_R.mp4"))
        with tempfile.TemporaryDirectory() as td:
            def vts_for(p, eye):
                sc = Path(str(p).replace(".mp4", ".vts"))
                if sc.exists():
                    return read_vts(str(sc))
                q = Path(td) / f"{eye}.vts"
                q.write_bytes(_read_tmf(p).vts_bytes())
                return read_vts(str(q))
            vl, vr = vts_for(src, "L"), vts_for(mp4_r, "R")
        tl = vl.sof_timestamps_ns.astype(np.int64)
        tr = vr.sof_timestamps_ns.astype(np.int64)
        gate = int(np.median(np.diff(tl)) * 0.5)
        pairs, j = [], 0
        for i, t in enumerate(tl):
            while j + 1 < len(tr) and abs(int(tr[j+1])-int(t)) < abs(int(tr[j])-int(t)):
                j += 1
            if abs(int(tr[j]) - int(t)) <= gate:
                pairs.append((i, j, int(t)))
        _, shift = auto_align(src, mp4_r, pairs, calib)
        if abs(shift) > 0.5:
            print(f"[auto-align] folding {shift:+.1f} px vertical offset into "
                  f"cam1 cy (mount moved since calibration)")
            calib["cameras"][1]["intrinsics"]["cy"] += shift

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- camchain: invert T_cam_imu, chain cam1 via T_cam1_cam0 ----
    T_c0_i = np.array(calib["T_cam0_imu"], dtype=np.float64)
    T_c1_c0 = np.array(calib["T_cam1_cam0"], dtype=np.float64)
    T_ci = [T_c0_i, T_c1_c0 @ T_c0_i]
    lines = ["%YAML:1.0", ""]
    for i, cam in enumerate(calib["cameras"][:2]):
        it = cam["intrinsics"]
        T_imu_cam = np.linalg.inv(T_ci[i])
        model = "equidistant" if "equi" in it["model"] else "radtan"
        dist = (list(it["distortion"]) + [0.0] * 4)[:4]
        lines += [
            f"cam{i}:",
            "  T_imu_cam:",
            _mat_yaml(T_imu_cam),
            f"  cam_overlaps: [{1 - i}]",
            "  camera_model: pinhole",
            f"  distortion_coeffs: [{', '.join(f'{d:.9f}' for d in dist)}]",
            f"  distortion_model: {model}",
            f"  intrinsics: [{it['fx']:.6f}, {it['fy']:.6f}, "
            f"{it['cx']:.6f}, {it['cy']:.6f}]",
            f"  resolution: [{it['image_size'][0]}, {it['image_size'][1]}]",
            f"  rostopic: /cam{i}/image_raw",
            f"  timeshift_cam_imu: {cam.get('timeshift_cam_imu_s', 0.0):.9f}",
        ]
    (args.out / "kalibr_imucam_chain.yaml").write_text("\n".join(lines) + "\n")

    # ---- imu chain ----
    imu = calib.get("imu", {})
    nm = imu.get("noise_model", imu)
    k = args.noise_inflation

    def noise(*keys, default):
        for key in keys:
            if key in nm and nm[key]:
                return float(nm[key])
        return default

    a_nd = noise("accel_noise_density", "accelerometer_noise_density",
                 default=2.3e-3) * k
    a_rw = noise("accel_random_walk", "accelerometer_random_walk",
                 default=5.0e-5) * k
    g_nd = noise("gyro_noise_density", "gyroscope_noise_density",
                 default=2.6e-4) * k
    g_rw = noise("gyro_random_walk", "gyroscope_random_walk",
                 default=2.0e-5) * k
    rate = float(imu.get("rate_hz") or imu.get("sample_rate_hz")
                 or args.imu_rate) or args.imu_rate
    ident3 = "\n".join("    - [ 1.0, 0.0, 0.0 ]\n    - [ 0.0, 1.0, 0.0 ]\n"
                       "    - [ 0.0, 0.0, 1.0 ]".split("\n"))
    (args.out / "kalibr_imu_chain.yaml").write_text(f"""%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {a_nd:.6e}
  accelerometer_random_walk: {a_rw:.6e}
  gyroscope_noise_density: {g_nd:.6e}
  gyroscope_random_walk: {g_rw:.6e}
  rostopic: /imu0
  time_offset: 0.0
  update_rate: {rate:.1f}
  model: "kalibr"
  Tw:
{ident3}
  R_IMUtoGYRO:
{ident3}
  Ta:
{ident3}
  R_IMUtoACC:
{ident3}
  Tg:
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
""")

    (args.out / "estimator_config.yaml").write_text(ESTIMATOR_TEMPLATE)
    print(f"wrote {args.out}/estimator_config.yaml, kalibr_imucam_chain.yaml, "
          f"kalibr_imu_chain.yaml")
    print(f"  noise inflation x{k:g}; imu rate {rate:g} Hz; "
          f"baseline {np.linalg.norm(T_c1_c0[:3,3])*1e3:.1f} mm")


if __name__ == "__main__":
    main()

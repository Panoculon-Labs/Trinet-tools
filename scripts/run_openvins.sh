#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
#
# Run OpenVINS stereo-inertial VIO on a Trinet stereo recording, headless.
#
# Usage:
#   ./scripts/run_openvins.sh WORKDIR [BAG] [CONFIG_DIR]
#
#   WORKDIR     mounted into the container at /data
#   BAG         rosbag path relative to WORKDIR (default: full/data.bag —
#               the bag layout produced by the calibration tooling:
#               cam0/ cam1/ image folders + imu0.csv -> kalibr_bagcreater)
#   CONFIG_DIR  config dir relative to WORKDIR (default: ov_config, from
#               scripts/make_openvins_config.py)
#
# Output: WORKDIR/ov_out/traj_est.txt (timestamp(s) x y z qx qy qz qw)
#
# Requires the trinet-openvins:latest image; build once with:
#   docker build -t trinet-openvins:latest scripts/openvins-docker/
set -euo pipefail

WORK="$(cd "${1:?usage: run_openvins.sh WORKDIR [BAG] [CONFIG_DIR]}" && pwd)"
BAG="${2:-full/data.bag}"
CFG="${3:-ov_config}"

[ -s "$WORK/$BAG" ] || { echo "missing bag: $WORK/$BAG" >&2; exit 1; }
[ -s "$WORK/$CFG/estimator_config.yaml" ] || {
  echo "missing $WORK/$CFG/estimator_config.yaml — run make_openvins_config.py" >&2
  exit 1
}
mkdir -p "$WORK/ov_out"

docker run --rm --entrypoint /bin/bash \
  -v "$WORK":/data \
  trinet-openvins:latest -lc "
    source /opt/ros/noetic/setup.bash
    source /catkin_ws/devel/setup.bash
    roscore >/tmp/roscore.log 2>&1 &
    sleep 2
    # serial.launch's path_gt default does \$(find ov_data), which roslaunch
    # resolves at parse time even when the arg is overridden — and ov_data
    # isn't shipped. Neutralize the default in a patched copy.
    sed 's|\$(find ov_data)[^\"]*|/dev/null|' \
      /catkin_ws/src/open_vins/ov_msckf/launch/serial.launch \
      > /tmp/serial_trinet.launch
    roslaunch /tmp/serial_trinet.launch \
      config_path:=/data/$CFG/estimator_config.yaml \
      bag:=/data/$BAG bag_start:=0 \
      max_cameras:=2 use_stereo:=true \
      dosave:=true path_est:=/data/ov_out/traj_est.txt \
      dolivetraj:=false path_gt:=/dev/null \
      dataset:=trinet verbosity:=INFO
  "

echo
echo "trajectory: $WORK/ov_out/traj_est.txt"
wc -l "$WORK/ov_out/traj_est.txt" 2>/dev/null || true

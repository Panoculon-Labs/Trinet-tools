# Exporting recordings to MCAP

`scripts/to_mcap.py` turns a Trinet recording into a single
[MCAP](https://mcap.dev) file with ROS 2 messages. Open it in
[Foxglove](https://foxglove.dev), read it with the `mcap` Python/C++ libraries,
or replay it with ROS 2 (`ros2 bag play` with the MCAP storage plugin).

It handles single-camera and stereo recordings, global-shutter and
rolling-shutter cameras, and H.264 or H.265 video.

## Usage

```bash
pip install -r requirements.txt        # adds mcap + mcap-ros2-support

# Stereo: point at either eye's MP4, any sidecar, or dir/basename
python scripts/to_mcap.py /data/take0004_L.mp4
python scripts/to_mcap.py /data/take0004 -o out/take0004.mcap

# Single camera
python scripts/to_mcap.py /data/recording4_1.mp4
```

Options:

| option | effect |
|---|---|
| `-o FILE` | output path (default `<basename>.mcap` next to the input) |
| `--calibration FILE` | use a calibration JSON instead of the one embedded in the MP4 |
| `--apply-timeshift` | shift camera times by the calibrated camera-IMU timeshift (`t_imu = t_cam + timeshift`), putting video on the IMU's time base |
| `--local-clock` | for a synced multi-camera take, keep this camera's own clock instead of the master clock |
| `--compression {zstd,lz4,none}` | MCAP chunk compression (default `zstd`) |

Only the MP4 is required: if a `.vts` or `.imu` sidecar or the calibration is
missing, the copy embedded in the MP4 is used, and the tool says so.

## What is in the file

| topic | message type | contents |
|---|---|---|
| `/cam0/video` | `foxglove_msgs/msg/CompressedVideo` | one message per frame, `format` `h264` or `h265` |
| `/cam0/camera_info` | `sensor_msgs/msg/CameraInfo` | intrinsics, repeated about once a second |
| `/cam0/frame_info` | `trinet_msgs/msg/FrameInfo` | per frame: frame number, exposure, readout time, shutter type, sync lock |
| `/cam1/...` | | same, second eye (stereo only) |
| `/imu` | `sensor_msgs/msg/Imu` | accelerometer (m/s², includes gravity) and gyroscope (rad/s) |
| `/imu/mag` | `sensor_msgs/msg/MagneticField` | magnetometer (tesla), one message per new reading |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `cam0 → imu` and `cam0 → cam1` from the calibration |

**Stereo naming:** `cam0` is the `_L` file (scene-left) and `cam1` the `_R`
file (scene-right), matching the calibration convention.

**Metadata records:**
- `trinet`: device id, firmware version, codec, shutter type, rolling-shutter
  readout time, and which clock and time reference the stamps use.
- `trinet_calibration`: the full calibration JSON.
- The calibration is also attached as `calibration.json`.

**IMU:**
- The orientation is not provided (`orientation_covariance[0] = -1`).
- When the calibration carries an IMU noise model, the gyro and accelerometer
  covariances are the per-sample white-noise variances (noise density² × rate).
  Otherwise they are zero ("unknown").

**Camera model:**
- Fisheye lenses use `distortion_model: "equidistant"`, the ROS name for the
  four-coefficient Kannala-Brandt fisheye model (`k1..k4`).
- Pinhole calibrations use `plumb_bob`.
- The images are not rectified, so `P` is `K` with no translation.

## Video is copied, not re-encoded

Each `/camN/video` message holds exactly one compressed frame from the MP4,
converted to Annex B byte-stream form (keyframes carry their parameter sets).
The camera's encoder never emits B-frames, so decode order equals display
order. Concatenating a topic's messages and decoding gives frames that are
bit-identical to decoding the MP4.

## Timestamps

- **Frame stamps** come from the `.vts` sidecar: the exposure centre of the
  frame on the camera's monotonic clock. They do not come from the MP4
  presentation timestamps. The IMU uses the same clock, so video and IMU line
  up directly.
- **Log time:** `log_time`, `publish_time` and the message stamp are all the
  same value.
- **Multi-camera takes:** for a take synced to a master camera, stamps are
  mapped onto the master's clock (as `VtsData.global_sof_ns()` does), so files
  from different cameras of the take share one timeline.
- **Pair stereo frames by timestamp, not by index.** The two eyes are
  genlocked, but one file can start a frame earlier than the other.

**Rolling shutter:** each row of a rolling-shutter frame is exposed at a
slightly different time. `/camN/frame_info` carries what you need to
reconstruct it:

```
t_row(r) = header.stamp + (r - r_ref) * readout_time_us / height
r_ref    = height / 2   if frame_centered   else 0   (top row)
```

Global-shutter cameras report `rolling_shutter: false` and
`readout_time_us: 0`: every row shares the frame's stamp.

## Not included

- **Audio:** MCAP has no standard audio message.
- **Thermal telemetry** (`.tel`).
- **Recordings captured over USB with the IMU embedded in the video stream:**
  run `trinet_tools/extract_sei.py` first to recover the `.imu`/`.vts`
  sidecars, then convert.

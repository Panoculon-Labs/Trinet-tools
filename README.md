# Trinet-Tools

Open-source utilities for working with recordings from the **Trinet camera** —
a wearable, synchronized video + inertial-measurement camera designed for
**egocentric (first-person) data collection** in camera-IMU calibration,
visual-inertial SLAM, dead-reckoning research, and similar applications.

This repository gives you everything you need to:

- **Parse** the binary `.imu` and `.vts` sidecar files that accompany every
  Trinet recording.
- **Extract** sidecars from a UVC-captured MP4 (where the inertial data lives
  embedded in the H.264 SEI stream) so they look identical to an on-board SD
  recording.
- **Visualize** a single recording as a synchronized video + inertial-plot
  composite for sanity checking and demos.
- **Combine** several cameras from the same take into one synced, side-by-side
  video — optionally with a per-camera orientation gizmo — so a multi-camera rig
  (e.g. a head plus two wrist cameras) plays back on a single shared timeline.

If you only need to play back the video, `video.mp4` is a standard MP4 — open
it in VLC. If you need the inertial data, read on.

## Install

```bash
git clone https://github.com/Panoculon-Labs/Trinet-tools.git
cd Trinet-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` and `ffprobe` must be on your `PATH` for the SEI extractor and the
visualizer (most package managers install them as one package).

## Recording layouts you may encounter

A Trinet recording is **either**:

1. **A triple of files** sharing a base name — `name.mp4 + name.imu + name.vts`.
   This is what you'll find on the camera's SD card.

2. **A per-session subdirectory** containing 10-minute "parts":

       Trinet/recording1/
         part001.mp4 + part001.imu + part001.vts
         part002.mp4 + part002.imu + part002.vts
         ...

   Each part is independently playable and decodes as a complete recording.
   This shape is enabled by some shipping configurations.

3. **A single MP4** captured over USB (UVC) with the inertial data embedded as
   SEI NAL units inside the H.264 stream. Use the SEI extractor below to turn
   it into shape (1).

All three shapes carry the same underlying data and the same monotonic
nanosecond timeline.

## Quickstart

### Parse a recording in Python

```python
from trinet_tools.reader import read_imu, read_vts, interpolate_imu_to_frames

imu = read_imu("recording1.imu")
vts = read_vts("recording1.vts")

print(f"{imu.num_samples} IMU samples over {imu.duration_s:.2f} s")
print(f"actual sample rate: {imu.actual_rate_hz:.1f} Hz")
print(f"device_id: {imu.header.device_id_hex or '(pre-v4 recording)'}")

# Camera generation. "v3" units (radio-beacon sync adapter) carry a live
# magnetometer; on them the per-sample trailing float is mag_age_us and
# imu.mag[xyz] holds real field data. Legacy units report "legacy".
print(f"generation: {imu.header.generation}  (live mag: {imu.header.has_live_mag})")
if imu.header.is_v3_generation:
    print(f"mag age (mean): {float(imu.mag_age_us.mean()):.0f} us")  # ~10-20 ms

# Per-frame IMU samples aligned to video frames.
per_frame = interpolate_imu_to_frames(imu, vts)
for entry in per_frame[:5]:
    print(entry["frame_number"], entry["accel"], entry["gyro"])
```

### Extract sidecars from a UVC MP4

If you captured the camera over USB, your MP4 has the inertial data tucked
inside the bitstream. To get the same `.imu`/`.vts` files an SD-card recording
would have:

```bash
python -m trinet_tools.extract_sei input.mp4 --out my_recording/
# Produces: my_recording/{video.mp4, video.imu, video.vts}
```

### Visualize a recording

Render a synchronized video + inertial-data composite as MP4:

```bash
python scripts/visualize.py path/to/recording.mp4
# Writes path/to/recording_viz.mp4 next to the input.

python scripts/visualize.py path/to/recording.mp4 \
    --plots orientation,accel,gyro,sync_delay
```

Works on a triple of files (`recording.mp4 + recording.imu + recording.vts`)
or on a single UVC MP4 after running the SEI extractor.

### Visualize multiple cameras together (synced)

If you recorded the same take with several Trinet cameras — for example a head
camera plus two wrist cameras — `sync_view.py` renders them **side by side on a
single shared timeline**. Each camera's frames are placed on the group's master
clock, so the same instant lines up across panels; the header shows the live
cross-camera offset (typically well under a millisecond on a synced take) and
each panel is labelled with its role and device tag. No calibration needed.

```bash
# two or more recordings of the same take
# (each: a .mp4, a base name, or a chunk directory)
python scripts/sync_view.py head.mp4 wristL.mp4 wristR.mp4 -o take_sync.mp4

# ...or point at a folder and auto-group the cameras by session id
python scripts/sync_view.py --auto path/to/recordings -o take_sync.mp4

# ...or preview live in a window instead of writing a file
python scripts/sync_view.py head.mp4 wristL.mp4 --show
```

To also show **each camera's orientation** on that timeline, use
`sync_view_imu.py`. Below every video panel it draws a 3-axis gizmo of the
camera's attitude, fused from its accelerometer and gyroscope (Madgwick) and
expressed in the camera frame using the camera-IMU extrinsic from a
`calibration.json` (produced by the [Trinet-Calibration](#sibling-projects)
pipeline):

```bash
python scripts/sync_view_imu.py head.mp4 wristL.mp4 wristR.mp4 \
    --imu calibration.json -o take_oriented.mp4
```

Both viewers accept the same recording shapes as the other commands (a file
triple, a chunk directory, or a SEI-extracted UVC MP4), and the group's master
camera is auto-detected and shown as the `ref` panel. If a camera is mounted
upside-down — common for wrist units — flip its panel with `--rotate180`, e.g.
`--rotate180 0,2` for the first and third panels.

### Inspect a recording from the shell (no plots, just numbers)

```bash
python examples/inspect_recording.py path/to/recording.imu
```

## IMU ↔ video synchronization

To align the IMU with video to sub-millisecond accuracy from a USB-streamed
recording (no SD card needed) — and why a naive test can appear to show ~40 ms
of "latency" — see [`docs/imu_video_sync.md`](docs/imu_video_sync.md). It
includes a verification script you can run on your own clips.

## File format reference

The full byte-level specification of `.imu` and `.vts` is at
[`docs/data_formats.md`](docs/data_formats.md). Read this if you want to write
your own parser in another language, or just to understand exactly what the
camera is recording.

Highlights:

- **All timestamps are monotonic nanoseconds**, not wall-clock — they reset to
  0 every time the camera powers on. This is what gives you tight, jitter-free
  inertial-to-video alignment.
- **Frame-sync alignment**: when enabled by firmware, every video frame
  triggers a hardware pulse to the inertial sensor. The first IMU sample
  after each pulse carries a sub-microsecond `fsync_delay_us`, letting you
  align inertial data to a specific video frame to within ~1 µs (much tighter
  than the IMU's own sample period).
- **Device ID**: a stable 16-byte public identifier per camera unit, used to
  attribute recordings to a specific physical camera. Carried in the `.imu`
  header (on-board recordings) and as the USB iSerialNumber (UVC mode).

## Compatibility

These tools work with both **current and pre-v4 Trinet recordings**. The v4
firmware added a `device_id` field to the `.imu` header's reserved bytes; the
reader gracefully reports `device_id_hex == ""` for older recordings and is
otherwise format-identical.

If you have a recording with a different magic string or version that this
library doesn't recognize, please file an issue — we'll add support.

## Sibling projects

- **[Trinet-Calibration](https://github.com/Panoculon-Labs/Trinet-Calibration)**
  — Camera-IMU calibration pipeline that consumes these recordings and
  produces `calibration.json` (intrinsics, extrinsics, time offset).
- **[Trinet-SDK](https://github.com/Panoculon-Labs/Trinet-SDK)** — Android SDK
  + sample app for capturing recordings from a USB-connected Trinet camera.

## License

MIT. See [LICENSE](LICENSE).

## Reporting issues

Issues and PRs are welcome at
[github.com/Panoculon-Labs/Trinet-tools](https://github.com/Panoculon-Labs/Trinet-tools).

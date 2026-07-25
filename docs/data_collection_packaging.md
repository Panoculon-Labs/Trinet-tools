# Packaging recordings for delivery

`scripts/ingest_sd_card.py` turns a Trinet camera's SD card into upload-ready
ZIPs — one per clip — each carrying the video, the inertial sidecars, and a
`metadata.json` describing where and how the footage was collected.

It is aimed at data-collection programs that require per-video metadata. The
script does one job: attach the metadata and zip. It does not inspect, grade or
filter the footage.

**Standard-library Python 3 only.** No `pip install`, no `ffmpeg`. Runs the
same on Windows, macOS and Linux.

**The recordings are never altered.** They go into the ZIP as byte-for-byte
copies of what is on the card, and the card itself is only ever read from. The
only flag that changes that is `--repair`, which is off unless you ask for it.

## Quickstart

### Windows

Install Python 3 from [python.org](https://www.python.org/downloads/) (tick
*"Add python.exe to PATH"* in the installer), then, with the card in drive `E:`:

```bat
python scripts\ingest_sd_card.py --drive E: ^
    --collector alice01 ^
    --country US ^
    --capture-date 2026-07-20 ^
    --calibration cal\unit-aa3d26ba.json ^
    --out D:\deliveries
```

### macOS / Linux

```bash
python3 scripts/ingest_sd_card.py --drive /Volumes/TRINET \
    --collector alice01 \
    --country US \
    --capture-date 2026-07-20 \
    --calibration cal/unit-aa3d26ba.json \
    --out ~/deliveries
```

Omit `--drive` and the script looks for a card by itself. Add `--dry-run` to
see what would be packaged without writing anything — worth doing on the first
card of a batch.

## What you get

One ZIP per clip, named `<collector>_<date>_<device-tag>_<clip>.zip` (the tag
is left out when the clip name already contains it, as synced group takes do):

```
alice01_20260720_aa3d26ba_recording3_1.zip
    recording3_1.mp4      video
    recording3_1.imu      inertial samples
    recording3_1.vts      per-frame video timestamps
    recording3_1.json     the camera's own recording sidecar (when present)
    metadata.json         collection metadata + calibration
    README.md             how to read the files
```

Each ZIP is self-describing and independently uploadable.

This works the same for **solo recordings** (`recording3_1`, a single camera)
and for **synced multi-camera takes** (`grp10580_329b911e_1`). Clips are found
by their `.mp4`, not by name, so neither naming scheme is special-cased.

## What goes in metadata.json

Each clip's `metadata.json` covers the full collection specification: the
details you supplied on the command line, the camera geometry from the
calibration, the video's technical properties (measured from the file), and the
IMU metadata.

```json
{
  "schema": "trinet-delivery-metadata/2",
  "clip_id": "grp10580_329b911e_1",
  "collector_id": "alice01",
  "session_id": "alice01-20260722-329b911e-s10580",
  "location": { "country": "IN", "region": "Bhopal" },
  "capture": { "date": "2026-07-22" },
  "camera": {
    "make": "Panoculon Labs", "model": "Trinet",
    "device_id": "329b911ecd8c67e288d969f92ca8d4d1",
    "intrinsics": {
      "image_size": [1920, 1080],
      "projection_model": "equidistant",
      "focal_length_px": { "fx": 587.8, "fy": 592.2 },
      "principal_point_px": { "cx": 897.1, "cy": 603.5 },
      "distortion_model": "equidistant",
      "distortion_coefficients": [0.139, 0.063, -0.076, 0.016]
    },
    "diagonal_fov_deg": 182.7,
    "extrinsics": {
      "head_frame": { "position_m": [0.0, 0.02, 0.09],
                      "orientation_deg": [-25.0, 0.0, 0.0] },
      "camera_to_imu": { "T_cam_imu": [[...]], "timeshift_cam_imu_s": 0.0099 }
    }
  },
  "video": {
    "container": "mp4", "codec": "h264",
    "width": 1920, "height": 1080, "resolution_mp": 2.07,
    "aspect_ratio": "landscape",
    "duration_s": 11.567, "frame_count": 344,
    "nominal_fps": 30.0, "average_fps": 29.65,
    "bitrate_mbps": 9.78,
    "gop_length": 30, "b_frames": false, "color_depth_bits": 8
  },
  "imu": {
    "present": true, "data_files": ["grp10580_329b911e_1.imu"],
    "sensors": ["accelerometer", "gyroscope", "magnetometer"],
    "sample_rate_hz": 399.9, "nominal_rate_hz": 400, "sample_count": 4622,
    "accelerometer": { "range_g": 8, "units": "m/s^2 (gravity included)" },
    "gyroscope": { "range_dps": 2000, "units": "rad/s" },
    "video_sync": { "tolerance_ms": 1, "method": "shared monotonic clock",
                    "align_using": "sof_timestamp_ns in the .vts sidecar" }
  },
  "duration_s": 11.567,
  "task": { "description": "fold and put away laundry" }
}
```

### How the fields map to the specification

| Spec row | metadata.json |
|---|---|
| Geographic Location | `location.country` (+ optional `region`) |
| User ID / Session ID | `collector_id` / `session_id` |
| Camera Resolution / Frame Rate | `video.resolution_mp` / `video.nominal_fps` |
| Video Format | `video.codec` + `video.container` |
| Diagonal Field of View | `camera.diagonal_fov_deg` |
| Aspect Ratio | `video.aspect_ratio` |
| Average FPS (after drops) | `video.average_fps` |
| Bitrate | `video.bitrate_mbps` |
| Color Depth | `video.color_depth_bits` |
| GOP Length | `video.gop_length` |
| B-Frames | `video.b_frames` |
| Clip Length | `duration_s` / `video.duration_s` |
| Camera Intrinsics (focal length) | `camera.intrinsics.focal_length_px` |
| Camera Intrinsics (distortion) | `camera.intrinsics.distortion_coefficients` |
| Camera Extrinsics (position/orientation vs head) | `camera.extrinsics.head_frame` |
| IMU sensors / accel / gyro | `imu.sensors`, `imu.accelerometer`, `imu.gyroscope` |
| IMU Sample Rate | `imu.sample_rate_hz` |
| IMU-to-Video Sync | `imu.video_sync` |

`video.gop_length`, `video.b_frames` and `video.color_depth_bits` are read
directly from the H.264 bitstream (no external tools), so they are present only
for H.264 recordings. `camera.intrinsics`, `camera.diagonal_fov_deg` and
`camera.extrinsics` need `--calibration`; without it they are `null` with a
note. `camera.extrinsics.head_frame` carries the camera's position and
orientation relative to the head frame when `--head-transform` is supplied.
Optional flags (`--region`, `--task`, `--task-labels`, `--participant-id`)
add their own keys; keys you did not supply are simply absent.

## Required flags

| Flag | Purpose |
|---|---|
| `--collector ID` | Unique identifier for the person collecting. Also seeds the per-session id. |
| `--country CC` | Geographic location (country). |

## Camera geometry

Delivery programs generally require camera **intrinsics** (focal length,
distortion) and **extrinsics** per video. Supply the unit's `calibration.json`
— produced by the
[Trinet-Calibration](https://github.com/Panoculon-Labs/Trinet-Calibration)
pipeline — with `--calibration`:

```bash
--calibration cal/unit-aa3d26ba.json
```

The script inlines the intrinsics into every `metadata.json` and computes the
**diagonal field of view** from them. For fisheye (`equidistant`) lenses it
inverts the full distortion polynomial numerically rather than assuming a
linear mapping, so the figure is the real optical FOV rather than an estimate.

Without `--calibration` the script warns and the intrinsics/extrinsics keys are
absent, which for most programs means the submission is incomplete.

For stereo units, `--camera-index` selects which camera in the file (0 = the
scene-left eye).

### Batch vs per-device calibration

A calibration may be for the **exact unit** that made the recording, or a
**batch** calibration measured on one representative unit and applied to the
whole production batch. Say which with `--calibration-scope`:

```bash
--calibration cal/batch-2026Q3.json --calibration-scope batch \
    --calibration-id trinet-mono-batch-2026Q3
```

It is recorded in `metadata.json → camera.calibration_scope` (`device`,
`batch`, or `unspecified`) with a note spelling out the implication — a batch
calibration does not capture per-unit optical variation. If you pass
`--calibration` without a scope, the field is marked `unspecified` and the
script warns. `--calibration-id` optionally names the batch.

### Field of view

`camera.diagonal_fov_deg` is normally computed from the calibration. To state
it explicitly — the nominal lens FOV, or when no calibration is on hand — pass
`--fov-deg`:

```bash
--fov-deg 150
```

An explicit `--fov-deg` overrides the computed value, and
`camera.diagonal_fov_source` records which was used (`stated (--fov-deg)` or
`computed from calibration`).

### Head-frame extrinsics

`calibration.json` gives the camera-to-inertial transform, not the camera's
pose on the wearer's head. If the program requires extrinsics *relative to the
head frame*, measure the mount once and pass it:

```json
{
  "rotation_deg": [-25.0, 0.0, 0.0],
  "translation_m": [0.0, 0.02, 0.09]
}
```

```bash
--head-transform mount/forehead-rig.json
```

A 4×4 `T_head_cam` matrix is accepted instead. Supply nothing and the key is
simply absent — the metadata never implies a measurement that was not made.

## Finding and mounting the card

With no `--drive`, the script locates the card itself. On Windows that means
scanning drive letters; cards attach themselves there, so nothing else is
needed.

On Linux and macOS it reads the mount table, which covers every layout the
various automounters use (`/Volumes/<label>`, `/media/<label>`,
`/media/<user>/<label>`, `/run/media/<user>/<label>`).

**If the card is plugged in but not attached** — common on machines with no
desktop session — the script mounts it itself, **read-only**, and detaches it
again when the run finishes:

```
Found unmounted removable device /dev/sda1 (58.2G) -- mounting
  mounted read-only at /media/you/015C-13C0
...
Unmounted /dev/sda1 (/media/you/015C-13C0)
```

Only hot-pluggable partitions carrying a filesystem are considered, so an
internal disk is never touched, and anything mounted this way that turns out
not to hold recordings is detached again immediately. Pass `--no-automount` to
disable it. Mounting this way needs `udisks` (standard on desktop Linux); where
it is unavailable, mount the card yourself and pass `--drive`.

If several cards are attached at once the script stops and asks you to pick one
with `--drive`, rather than guessing.

## MCAP output

Programs that prefer a single time-indexed container over separate files can
get an [MCAP](https://mcap.dev) with `--mcap`. Each clip's ZIP then also holds
a `<clip>.mcap` carrying:

- **`/imu`** — one message per inertial sample (`trinet.Imu`: linear
  acceleration in m/s², angular velocity in rad/s), timestamped on the
  recording's monotonic clock.
- **`/camera`** — one `foxglove.CompressedVideo` message per frame (H.264,
  Annex-B, SPS/PPS prepended to each keyframe), timestamped from the `.vts` so
  it lines up with the IMU on a shared clock. Written only for H.264
  recordings.
- the **metadata** as an MCAP metadata record plus the full `metadata.json` as
  an attachment, and the **`calibration.json`** as an attachment when supplied.

It opens directly in [Foxglove](https://foxglove.dev) — video and IMU scrub on
one timeline — and is written with no external tools or libraries (the same
zero-dependency, stdlib-only Python as the rest of the script).

```bash
python3 scripts/ingest_sd_card.py --drive E: --collector alice01 \
    --country US --calibration cal/unit.json --calibration-scope batch \
    --mcap --out ./deliveries
```

Because the video is stored again inside the MCAP, `--mcap` roughly doubles the
size of each ZIP. The standalone `.mp4` remains in the ZIP either way, so the
MCAP is an addition, not a replacement. Timestamps in the MCAP are monotonic
nanoseconds from power-on (the camera has no real-time clock); both topics
share that clock, so they are mutually synchronised even though they are not
wall-clock.

## Capture dates

**The camera has no real-time clock.** Timestamps inside a recording are
monotonic nanoseconds counted from power-on, and file modification times on the
card are not wall-clock dates. That is deliberate — it is what makes the
inertial-to-video alignment drift-free — but it means the calendar date has to
come from the operator:

```bash
--capture-date 2026-07-20
```

Omit it and the script uses today's date and warns. If a card is ingested days
after a shoot, pass the real date.

## Device identity

Every clip is attributed to the camera unit that recorded it, under
`metadata.json → camera`:

```json
"camera": { "device_id": "329b911ecd8c67e288d969f92ca8d4d1" }
```

The id is read from the **`.imu` header**, where the camera writes it into the
recording itself, so it survives renaming and reorganising. The `.json` sidecar
carries a copy, used only as a fallback — for recordings from firmware
predating the field, or when the sidecar is all that is left.

Its first 8 characters are the tag that appears in the ZIP filename and the
generated session id.

If the two sources **disagree**, the card holds files from more than one unit.
The script warns on the console and keeps the `.imu` header's value, rather
than silently picking one.

The id is one-way — nothing about the camera's hardware can be recovered from
it — so it is safe to log, store and share.

## Recording layouts it handles

- **Flat file sets** — `recording3_1.mp4` plus its sidecars. One ZIP.
- **Chunked sessions** — a folder of `part001.*`, `part002.*`, … All parts go
  into one ZIP and the sidecar totals are merged, with a warning that
  fixed-length slicing cuts across semantic task boundaries.
- **Multi-camera takes** — each camera's card is ingested separately. The
  shared session id, role and clock offset are carried through to
  `metadata.json → multi_camera` so the recipient can regroup them.

Anything else on the card that is not part of a recording is ignored: a clip is
recognised by its `.mp4`, and only sidecars sharing that exact base name travel
with it.

## Clips that look ~1 second long

Some recordings read as only ~1 second in strict players and uploaders even
though the footage is complete, because of the layout the camera writes them
in. If the recipient's pipeline trips over this, `--repair` rebuilds each MP4's
index on a staged copy before zipping:

```bash
--repair
```

It is lossless — the video and audio are untouched and nothing is re-encoded,
only the index is rebuilt — but **the bytes in the ZIP then differ from the
bytes on the card**, so it is off by default and `metadata.json → tooling`
records which of the two you used. The card is not modified either way.

To repair files in place separately, or to recover a clip truncated by power
loss mid-recording, use
[`scripts/repair_recordings.py`](../scripts/repair_recordings.py) directly.

## Full option list

```bash
python3 scripts/ingest_sd_card.py --help
```

## Troubleshooting

**"no SD card with recordings found"** — pass the path explicitly with
`--drive E:` (Windows) or `--drive /Volumes/NAME`. If the camera writes into a
custom folder name, add `--folder NAME`.

**"no recordings folder found under …"** — you may have pointed at the wrong
drive. The script looks for a folder containing `.mp4` files, either at the
card root or one level below it.

**Codec, bitrate or field of view not what the program expects** — all three
are camera configuration or hardware, not packaging. This script will not
re-encode or alter footage to compensate; fix them at the camera before
collecting a batch.

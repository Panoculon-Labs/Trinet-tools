# Packaging recordings for delivery

`scripts/ingest_sd_card.py` turns a Trinet camera's SD card into upload-ready
ZIPs — one per clip — each carrying the video, the inertial sidecars, and a
`metadata.json` describing where and how the footage was collected.

It is aimed at data-collection programs that require per-video metadata, with
optional re-encoding to a target bitrate and quality gating for clips that would
be refused downstream.

**Standard-library Python 3 only** for the core path — no `pip install`. Runs
the same on Windows, macOS and Linux. The one exception is `--reencode`, which
shells out to `ffmpeg` (the script checks for it and errors clearly if absent).

**The card is only ever read from.** By default the packager rebuilds each
MP4's index on a staged copy (`--repair`, on by default) so strict players and
uploaders accept the file — lossless, no re-encode; pass `--no-repair` to copy
the video byte-for-byte instead. `--reencode` transcodes the video (that is its
purpose). Either way the `.imu`/`.vts` sidecars are copied unchanged and the SD
card is never modified.

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
  "environment_type": "residential",
  "environment_subcategory": "laundry",
  "environment": { "type": "residential", "subcategory": "laundry" },
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
| Environment Type | `environment_type` + `environment_subcategory` (+ nested `environment`) |
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
| `--environment TYPE/SUB` | Environment type and sub-category (below). Required by the delivery spec. |

### Environment values

`--environment` takes `type/sub-category`. The categories below come from the
collection spec; **they are not enforced** — any value packages, but one outside
the list warns so a typo is caught. Spaces/dashes in the sub-category normalise
to underscores.

| Type | Sub-categories |
|---|---|
| `residential` | `laundry`, `kitchen_tidy`, `organize_room`, `other_household` |
| `commercial` | `agriculture_landscaping_grounds`, `hospitality_housekeeping`, `automotive_service_maintenance`, `food_service_back_of_house`, `field_services_light_installation`, `commercial_cleaning_janitorial`, `retail_stocking_back_of_house`, `construction_skilled_trades`, `other` |

## Camera geometry

Delivery programs generally require camera **intrinsics** (focal length,
distortion) and **extrinsics** per video. Supply the unit's `calibration.json`
— produced by the
[Trinet-Calibration](https://github.com/Panoculon-Labs/Trinet-Calibration)
pipeline — with `--calibration`:

```bash
--calibration cal/unit-aa3d26ba.json
```

Both `calibration.json` layouts the pipeline emits are accepted: the flat
single-camera form (`{"intrinsics": …, "extrinsics": {"T_cam_imu": …}, …}`) and
the multi-camera form (`{"cameras": [ … ], "T_cam0_imu": …}`, `--camera-index`
picks the camera; 0 = the scene-left eye).

The script inlines the intrinsics into every `metadata.json` and records the
**diagonal field of view**. It uses the calibration's own `fov_deg` when present
and otherwise computes it — for fisheye lenses by inverting the full distortion
polynomial numerically, not a linear approximation. When the calibration carries
a horizontal/vertical/diagonal breakdown it is copied to `camera.fov_deg`
(useful because, for wide fisheye lenses, the horizontal FOV is far more stable
across units than the diagonal).

Without `--calibration` the script warns and the intrinsics/extrinsics keys are
absent, which for most programs means the submission is incomplete.

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

## Index repair (default on)

Some recordings read as only ~1 second in strict players and uploaders even
though the footage is complete, because of the fragmented layout the camera
writes them in. To prevent that, the packager **rebuilds each MP4's index on a
staged copy by default** — lossless, no re-encode, only the index changes. The
SD card is never modified. Pass `--no-repair` to copy the video byte-for-byte
instead (only do this if the recipient handles fragmented MP4).

`metadata.json → tooling` records which path was taken. To repair files in
place outside the packager, use
[`scripts/repair_recordings.py`](../scripts/repair_recordings.py) directly.

## Re-encoding to a target bitrate (default on)

Trinet units record ~10–11 Mbps against a 6–8 Mbps spec, so the packager
**re-encodes each clip to a compliant H.264 by default** — GOP 30, no B-frames,
8-bit, capped at 8 Mbps (target 7). Pass `--no-reencode` to keep the recorded
bitrate (the index is still rebuilt); `--reencode-mbps N` changes the target.

It runs as fast as the machine allows: a hardware encoder (NVENC on NVIDIA,
VideoToolbox on macOS) when one actually works, otherwise `libx264 -preset
veryfast`. Each candidate is smoke-tested first, so a compiled-but-unusable
encoder (e.g. NVENC with no GPU) falls back to CPU cleanly. `-c:a copy` keeps
any audio; `+faststart` makes the result openable everywhere. The `.imu`/`.vts`
sidecars keep their frame count, order and timing, so alignment holds, and
`metadata.json → video` reflects the re-encoded file.

**Needs `ffmpeg` on PATH** — if it is missing, the re-encode is skipped with a
note (the index is still rebuilt). Run several clips at once with `--jobs N`.

## Correcting an off-scale accelerometer (default on)

If a unit recorded its accelerometer at the wrong full-scale range — its at-rest
gravity reads ~2× or ~0.5× of 9.81 m/s² — the packager rescales the `.imu` accel
samples to the true range so gravity reads ~9.81 (the gyroscope is a separate
range and is left alone). Only gross faults are corrected; a borderline value is
left as recorded. `metadata.json → imu.accelerometer.scale_corrected` records
the factor applied. Pass `--no-fix-imu` to disable.

## Quality gating

By default every clip is packaged. To hold back clips that would be refused
downstream, gate on them — they are skipped and listed in
`rejected_<collector>_<date>.json` instead of being packaged:

```bash
--gate                     # all four checks below
--min-duration 120         # skip clips shorter than N seconds (spec floor 120)
--require-imu              # skip clips with no usable accel+gyro
--require-valid-video      # skip truncated / unreadable MP4s
--require-imu-gravity      # skip clips whose at-rest gravity is off-scale
```

`--require-valid-video` uses the same box audit the delivery ingest does (the
last box must end at EOF), so a clip cut mid-write is caught before it ships.

`--require-imu-gravity` measures the accelerometer magnitude over
near-stationary windows: at rest it should read gravity (~9.81 m/s²), and a
value outside 9.0–10.6 means the accelerometer scale is wrong — usually a
full-scale-range misconfiguration on that unit, which shows as a clean ~2× or
~0.5× error. The measured value is always recorded in
`metadata.json → imu.accelerometer.gravity_still_ms2` (with `gravity_in_spec`),
whether or not you gate on it.

## Backfilling a batch that already shipped

If a batch went out missing a required field (e.g. `environment_type` was not
recorded), `scripts/backfill_metadata.py` rewrites `metadata.json` inside the
delivered ZIPs in place — no re-collection, no re-upload of the video:

```bash
# one environment for the whole batch
python3 scripts/backfill_metadata.py DELIVERIES/ --environment residential/laundry

# per-file, from a CSV of  zip_name-or-clip,environment
python3 scripts/backfill_metadata.py DELIVERIES/ --map env_by_clip.csv

# preview first
python3 scripts/backfill_metadata.py DELIVERIES/ --environment residential/laundry --dry-run
```

It copies every other ZIP entry through untouched, is idempotent, and can also
set `--country` / `--session-id`. Stdlib only.

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

**Bitrate over spec** — best fixed at the camera, but `--reencode` transcodes
each clip down to a compliant ≤8 Mbps H.264 at ingest (see above) if you can't
re-record. **Codec or field of view wrong** — those are camera/lens, not
packaging; fix them at the camera before collecting a batch.

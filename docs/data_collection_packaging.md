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
    --environment residential/laundry ^
    --capture-date 2026-07-20 ^
    --calibration cal\unit-aa3d26ba.json ^
    --out D:\deliveries
```

### macOS / Linux

```bash
python3 scripts/ingest_sd_card.py --drive /Volumes/TRINET \
    --collector alice01 \
    --country US \
    --environment residential/laundry \
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

Deliberately small: **what you supplied on the command line**, plus which unit
recorded the clip and how long it runs. Nothing is inferred or editorialised.

```json
{
  "schema": "trinet-delivery-metadata/1",
  "clip_id": "grp10580_329b911e_1",
  "collector_id": "alice01",
  "session_id": "alice01-20260722-329b911e-s10580",
  "environment": { "type": "residential", "subcategory": "laundry" },
  "location": { "country": "US" },
  "capture": { "date": "2026-07-22" },
  "camera": {
    "device_id": "329b911ecd8c67e288d969f92ca8d4d1",
    "placement": { "mount": "head_forehead" }
  },
  "duration_s": 11.567,
  "task": { "description": "fold and put away laundry" }
}
```

`--calibration` adds `camera.intrinsics`, `camera.diagonal_fov_deg` and
`camera.extrinsics`; `--region`, `--task`, `--task-labels`, `--participant-id`
and `--env-note` add their own keys. Keys you did not supply are simply absent.

Everything else — frame rate, sample rate, resolution, codec, per-frame timing —
already lives in the recording itself, so it is not duplicated here. See
[`data_formats.md`](data_formats.md) for how to read it.

## Required flags

| Flag | Purpose |
|---|---|
| `--collector ID` | Unique identifier for the person collecting. Also seeds the per-session id. |
| `--country CC` | Geographic location (country). |
| `--environment TYPE/SUB` | Environment type and sub-category (below). |

### Environment values

`--environment` takes `type/sub-category`, validated against this list:

| Type | Sub-categories |
|---|---|
| `residential` | `laundry`, `kitchen_tidy`, `organize_room`, `other_household` |
| `commercial` | `agriculture_landscaping_grounds`, `hospitality_housekeeping`, `automotive_service_maintenance`, `food_service_back_of_house`, `field_services_light_installation`, `commercial_cleaning_janitorial`, `retail_stocking_back_of_house`, `construction_skilled_trades`, `other` |

Use `--env-note` to describe anything filed under an `other` sub-category.

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

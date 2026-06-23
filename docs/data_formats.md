# Trinet Recording Data Formats

A Trinet camera recording consists of synchronized video and inertial-measurement
data. This document specifies the on-disk file formats so anyone can write their
own reader, validator, or exporter without depending on the Python tools.

All multi-byte integers and floats are **little-endian**. All timestamps are
**monotonic nanoseconds** (i.e. they always increase, but they are not wall-clock
times — they reset to 0 when the camera powers up).

> The format `version` (in each file's header, right after the magic) tells you
> which camera generation produced the recording and which timestamp to align to.
> For the version → generation map **and how to time-sync each one**, see
> [`imu_video_sync.md`](imu_video_sync.md).

## Recording shapes

Every recording is a small set of files that share one **base name**:

    Trinet/<base>.mp4     # H.264 video
    Trinet/<base>.imu     # inertial samples (accel/gyro/mag)
    Trinet/<base>.vts     # per-frame video timestamps (+ cross-camera sync offset)
    Trinet/<base>.json    # recording-meta sidecar (session / role / clock sync — see below)

The base name tells you how the recording was made:

- `grp<session>_<device-tag>_<segment>` — a **synced multi-camera take**, e.g.
  `grp72593_aa3d26ba_1`. Every camera in the same take writes the SAME
  `<session>` with its own `<device-tag>` (see shape 3 below).
- `<name>_<segment>` — a **solo recording**, e.g. `recording4_1`
  (its `session`/`group` are `0` and `role` is `unpaired`).

`<segment>` starts at `1` and increments if a long take rolls into more than one
file. The four files are stored in one of these shapes:

### 1. On-board SD recording (camera writes to its SD card)

The file set lands flat under `Trinet/` on the camera's SD card, e.g.

    Trinet/grp72593_aa3d26ba_1.mp4
    Trinet/grp72593_aa3d26ba_1.imu
    Trinet/grp72593_aa3d26ba_1.vts
    Trinet/grp72593_aa3d26ba_1.json

If the camera firmware is configured for chunked recording, you'll get a
per-session subdirectory instead, with the recording sliced into fixed-length
parts:

    Trinet/recording1/part001.mp4
    Trinet/recording1/part001.imu
    Trinet/recording1/part001.vts
    Trinet/recording1/part002.mp4
    Trinet/recording1/part002.imu
    Trinet/recording1/part002.vts
    ...

Each part is independently playable. Sidecar timestamps within a session are
monotonically increasing across parts (they share the same monotonic clock).

### 2. Host-side UVC recording (Trinet camera streamed over USB)

A single .mp4 file. The inertial samples are embedded inside the H.264 bitstream
as SEI (Supplemental Enhancement Information) NAL units carrying a Trinet
"TRIMU" UUID. Use `trinet_tools.extract_sei` to split such an MP4 back into the
same .mp4 / .imu / .vts triple as an on-board recording — at which point the
rest of the toolkit treats it identically.

### 3. Multi-camera (wrist-kit) session

When several cameras record the same take at once (a "wrist kit"), each camera
writes its OWN file set to its OWN SD card, and **every camera in one take shares
the same `grp<session>`** — only the `<device-tag>` differs. A three-camera
session looks like this (one folder per camera, however you copy them off the
cards):

    LEFT/   grp72593_aa3d26ba_1.{mp4,imu,vts,json}    # role: master
    CENTRE/ grp72593_64ea7f5d_1.{mp4,imu,vts,json}    # role: slave
    RIGHT/  grp72593_e96d9ce4_1.{mp4,imu,vts,json}    # role: slave

The folder names (`LEFT`/`CENTRE`/`RIGHT`) are just how you organise the cards —
the cameras are tied together by the shared `session`/`group` ids in their
filenames and `.json` sidecars, **not** by folder. Exactly one camera is the
`master` (the clock reference, `master_clock_offset_ns = 0`); the rest are
`slave`, each carrying the offset that maps its frames onto the master's clock.
That offset (in the `.json` and the `.vts` header) is what lets the toolkit place
all cameras on a single timeline — see `scripts/sync_view.py` in the README.

## `.imu` file format (TRIMU001)

A 64-byte header followed by fixed-size sample records.

### Version history

The header `version` field identifies the layout. Every version from 3 onward
shares the **same 64-byte header and 80-byte sample** layout — newer revisions
only repurpose previously-reserved/zero bytes or change the *meaning* of an
existing field, so an older reader sees zeros (never garbage) and a v5-aware
reader parses every version. v1/v2 use smaller sample records, detected from the
header `version`.

| version | sample size | what it adds                                                                                       |
| ------- | ----------- | -------------------------------------------------------------------------------------------------- |
| 1       | 44 bytes    | base: `timestamp_ns` + `accel[xyz]` + `gyro[xyz]` + `mag[xyz]`                                       |
| 2       | 76 bytes    | + `temp_c`, on-device orientation `quat_xyzw`, `lin_accel[xyz]`                                     |
| 3       | 80 bytes    | + trailing `fsync_delay_us`; header gains `flags` (bit 0 = frame-sync) and the 16-byte `device_id`  |
| 4       | 80 bytes    | sample layout unchanged; header reserved bytes 56–63 now carry `ios_host_offset_ns` (host-clock offset) |
| 5       | 80 bytes    | **magnetometer generation**: `mag[xyz]` carries live data, `flags` bit 1 (MAG) is set, and the trailing float is reinterpreted as `mag_age_us`. These units have no frame-sync hardware, so `flags` bit 0 is clear and there is no `fsync_delay_us`. |
| 6       | 80 bytes    | **mid-exposure generation**: sample layout unchanged from v5; the v6 change lives in the **video SEI / `.vts` v4** — the per-frame timestamp becomes exposure-centred and gains `exposure_us` + `readout_time_us`. The `.imu` itself is byte-identical to v5. |

The byte tables below describe the current 80-byte layout (versions 3–6). v1/v2
recordings have shorter samples (no `temp_c`/`quat`/`lin_accel`/trailing float);
read their sample size from the version.

### Header (64 bytes)

| offset | size | type     | field            | meaning                                          |
| ------ | ---- | -------- | ---------------- | ------------------------------------------------ |
| 0      | 8    | char[8]  | magic            | ASCII `"TRIMU001"`                               |
| 8      | 4    | uint32   | version          | 1–6 (see version history above; 6 current)       |
| 12     | 4    | uint32   | sample_rate_hz   | nominal IMU output data rate (e.g. 400 or 562, by unit) |
| 16     | 2    | uint16   | accel_fs         | 0=±2 g, 1=±4 g, 2=±8 g, 3=±16 g                  |
| 18     | 2    | uint16   | gyro_fs          | 0=±250, 1=±500, 2=±1000, 3=±2000 dps             |
| 20     | 8    | uint64   | start_time_ns    | monotonic ns of first sample in this file        |
| 28     | 8    | uint64   | video_start_ns   | monotonic ns of first video frame (0 if unknown) |
| 36     | 4    | uint32   | flags            | v3+: bit 0 = frame-sync alignment captured; bit 1 = magnetometer present (v5) |
| 40     | 16   | bytes    | device_id        | v3+: public per-unit ID (all-zero if unknown); see [Device ID](#device-id) |
| 56     | 8    | int64    | ios_host_offset_ns | v4+: ns offset aligning these timestamps to an iOS host clock (0 = none) |

The `flags`, `device_id`, and `ios_host_offset_ns` fields all occupy bytes that
were reserved (zero) in earlier versions, so a reader for an older version sees
them as zero — never as garbage — and the rest of the file stays fully
compatible. An **all-zero `device_id`** means "unknown / pre-v4 recording"; v1/v2
headers have the entire region after `video_start_ns` zeroed.

### Sample (80 bytes, repeated until EOF)

| offset | size | type     | field            | meaning                                                          |
| ------ | ---- | -------- | ---------------- | ---------------------------------------------------------------- |
| 0      | 8    | uint64   | timestamp_ns     | monotonic ns when the sample was acquired                        |
| 8      | 12   | float[3] | accel[xyz]       | m/s² (gravity included)                                          |
| 20     | 12   | float[3] | gyro[xyz]        | rad/s                                                            |
| 32     | 12   | float[3] | mag[xyz]         | µT (live on v5; zero if magnetometer unavailable)                |
| 44     | 4    | float    | temp_c           | °C (sensor die temperature)                                      |
| 48     | 16   | float[4] | quat_xyzw        | reserved for on-device fusion; current firmware writes zeros     |
| 64     | 12   | float[3] | lin_accel[xyz]   | reserved for on-device fusion; current firmware writes zeros     |
| 76     | 4    | float    | fsync_delay_us / mag_age_us | v3/v4: µs offset between the nearest frame-sync pulse and this sample (0 if none). v5: µs from this sample's timestamp back to the magnetometer reading in `mag[xyz]` |

The trailing float at offset 76 is the same 4 bytes in every version; the header
`version` (and `flags` bit 1) says whether it is `fsync_delay_us` (v3/v4) or
`mag_age_us` (v5). Number of samples = `(file_size - 64) / sample_size`, where
`sample_size` is **80** for v3–v5 (76 for v2, 44 for v1). There is no count field
and no trailer — readers compute the count from the file size and version.

### Frame-sync alignment (v3/v4) vs magnetometer age (v5)

The trailing float and `flags` distinguish two camera generations:

- **v3/v4 — `flags` bit 0 set, bit 1 clear:** the inertial sensor receives a
  hardware sync pulse from the imager at the start of every video frame, and the
  `fsync_delay_us` field on the *next* IMU sample after each pulse holds the
  sub-microsecond offset between the pulse and that sample — letting
  post-processing align inertial samples to video frames precisely. Samples with
  no pulse in their window store 0 here.
- **v5 — `flags` bit 1 set, bit 0 clear:** these units carry a live magnetometer
  and have no frame-sync pulse. The same trailing slot is `mag_age_us` —
  microseconds from the IMU sample's `timestamp_ns` back to the magnetometer
  reading in `mag[xyz]`, so the magnetometer's absolute time is
  `timestamp_ns − mag_age_us × 1000`. **It is not a video-sync value** (do not
  read it as a frame offset). For video alignment on v5, use the per-frame
  `sof_timestamp_ns` in the `.vts` sidecar (below) — it is captured on the same
  monotonic clock as the IMU samples.

## `.vts` file format (TRIVTS01)

Per-frame video timestamp sidecar. Maps each encoded video frame to the
monotonic clock that timestamps the IMU samples.

### Version history

| version | entry size | what it adds                                                                                       |
| ------- | ---------- | -------------------------------------------------------------------------------------------------- |
| 1       | 12 bytes   | base: `frame_number` + `sof_timestamp_ns`                                                          |
| 2       | 24 bytes   | + `venc_seq` + `venc_pts_us` (ties each frame to its encoded packet)                               |
| 3       | 24 bytes   | per-frame entry unchanged from v2; the header's 16 reserved bytes now carry a multi-camera clock-sync block (below). v2 readers ignore those bytes and parse v3 entries unchanged. |
| 4       | 36 bytes   | + `exposure_us` + `entry_flags` + `readout_time_us`. `sof_timestamp_ns` becomes the **mid-exposure** frame time when `entry_flags` bit 0 (`MID_EXPOSURE`) is set (else start-of-frame). `readout_time_us` is the rolling-shutter readout span; per-row delay = `readout_time_us / image_height` = the Kalibr `line_delay`. Written by mid-exposure (SEI v6) cameras. |

### Header (32 bytes)

| offset | size | type     | field             | meaning                              |
| ------ | ---- | -------- | ----------------- | ------------------------------------ |
| 0      | 8    | char[8]  | magic             | ASCII `"TRIVTS01"`                   |
| 8      | 4    | uint32   | version           | 1–4 (4 = mid-exposure cameras)       |
| 12     | 4    | uint32   | frame_rate_milli  | configured fps × 1000 (e.g. 30000)   |
| 16     | 16   | bytes    | reserved / sync   | zero in v1/v2; multi-camera clock-sync block in v3 (below) |

**v3 multi-camera clock-sync block** — the 16 header bytes at offset 16, captured
at recording start when the unit is part of a wireless-synced multi-camera group:

| offset | size | type   | field                  | meaning                                                          |
| ------ | ---- | ------ | ---------------------- | ---------------------------------------------------------------- |
| 16     | 8    | int64  | master_clock_offset_ns | add to this file's `sof_timestamp_ns` to get the shared group timeline |
| 24     | 4    | int32  | clock_skew_ppb         | estimated local-vs-master clock skew (parts per billion)         |
| 28     | 2    | uint16 | sync_quality_us        | estimated 1-sigma sync error (µs)                                |
| 30     | 2    | uint16 | sync_flags             | bit 0 = synced (offset valid); bit 1 = this unit is the group master; bit 2 = sync link was down when stamped |

For a solo (un-synced) recording these bytes are zero and `sof_timestamp_ns` is
already the correct per-camera clock.

### Entry (12 bytes in v1; 24 bytes in v2/v3; 36 bytes in v4, one per encoded video frame)

| offset | size | type     | field             | meaning                                                |
| ------ | ---- | -------- | ----------------- | ------------------------------------------------------ |
| 0      | 4    | uint32   | frame_number      | 0-based index into this MP4 file                       |
| 4      | 8    | uint64   | sof_timestamp_ns  | per-frame capture time, monotonic ns (best clock for sync); 0 if unavailable. **Mid-exposure** in v4 when `entry_flags` bit 0 is set; otherwise start-of-frame. |
| 12     | 4    | uint32   | venc_seq          | encoder-internal sequence number (v2+)                 |
| 16     | 8    | uint64   | venc_pts_us       | encoder presentation timestamp in microseconds (v2+). **Delivery timeline — do not use for IMU sync.** |
| 24     | 4    | uint32   | exposure_us       | v4: applied integration time (valid when `entry_flags` bit 1) |
| 28     | 4    | uint32   | entry_flags       | v4: bit 0 = `MID_EXPOSURE` (sof is exposure-centred), bit 1 = `EXPOSURE_VALID`, bit 2 = `READOUT_VALID` |
| 32     | 4    | uint32   | readout_time_us   | v4: rolling-shutter readout span µs (valid when `entry_flags` bit 2); per-row delay = `readout_time_us / image_height` (Kalibr `line_delay`) |

v1 entries stop after `sof_timestamp_ns` (12 bytes); `venc_seq`/`venc_pts_us`
were added in v2.

For inertial-to-video alignment, **prefer `sof_timestamp_ns`** (when nonzero):
it is captured at the camera the moment the imager started reading out a frame,
and matches the same clock as the IMU `timestamp_ns`. `venc_pts_us` is useful
for matching the entry to a packet you read from the MP4, but is delayed by
encoder pipeline latency relative to the actual capture instant.

In chunked recordings, `frame_number` resets to 0 at the start of each part —
so each part's `.vts` reads independently.

## `.json` recording-meta sidecar

A small, human-readable `<base>.json` sits next to every recording. It identifies
the recording and — for a synced multi-camera take — describes how this camera
relates to the others. Example (a wrist-kit **slave** camera):

```json
{
  "format": "trinet-recording-meta/1",
  "session": 72593,
  "group": 48477,
  "role": "slave",
  "device_id": "64ea7f5da17f71e411e8f1a8103a1558",
  "device_tag": "64ea7f5d",
  "segment": 1,
  "synced": true,
  "master_clock_offset_ns": -4642873007,
  "clock_skew_ppb": 28617,
  "sync_quality_us": 610,
  "sync_flags": 1,
  "mode": "single",
  "basename": "grp72593_64ea7f5d_1",
  "created_unix": 79
}
```

| field | meaning |
|---|---|
| `format` | schema id, currently `trinet-recording-meta/1` |
| `session` | recording-session id; **every camera in one synced take shares it** (`0` = solo) |
| `group` | the pairing-group (kit) id the camera belongs to (`0` = unpaired) |
| `role` | `master`, `slave`, or `unpaired`. The **master** is the clock reference |
| `device_id` | 32-hex public camera id — the same value as the `.imu` header device_id and the USB serial in UVC mode |
| `device_tag` | first 8 hex of `device_id`; this is the `<device-tag>` in the `grp<session>_<device-tag>_<segment>` filename |
| `segment` | part index when a long take rolls into multiple files (starts at `1`) |
| `synced` | `true` when `master_clock_offset_ns` is valid (this camera is clock-synced to the master) |
| `master_clock_offset_ns` | add to this camera's `.vts` start-of-frame timestamps to map them onto the **master's** clock (`0` on the master itself) |
| `clock_skew_ppb` | residual clock-rate mismatch vs the master, parts-per-billion, for drift correction over long takes |
| `sync_quality_us` | estimated one-way cross-camera sync uncertainty, microseconds (`0` on the master) |
| `sync_flags` | bitfield: `0x1` = synced (offset valid), `0x2` = this camera is the master. So master = `3`, synced slave = `1`, unpaired = `0` |
| `mode` | recording mode (`single`) |
| `basename` | the shared base name of the file set |
| `created_unix` | the camera's clock at creation — **seconds since the camera last powered on**, not wall-clock (the camera keeps no real-time clock; see the monotonic-timeline note above) |

The sync fields (`synced`, `master_clock_offset_ns`, `clock_skew_ppb`,
`sync_quality_us`, and the master flag) are a convenience copy of what the `.vts`
v3 header already carries — that header is what the Python reader uses, so
`read_vts(path).global_sof_ns()` returns each camera's frames already mapped onto
the master clock. The `.json` is the easy way to see, at a glance, which take a
file belongs to and which camera is the master.

## `video.mp4`

Standard MP4 container with one H.264 (or H.265, if the firmware was so
configured) video track. No audio. Fragmented MP4 internally so that even a
power-loss-truncated file is recoverable up to the last keyframe interval.

### When SEI inertial data is embedded

Host-side UVC recordings additionally carry inertial samples inside SEI
user_data_unregistered NALs whose payload starts with this 16-byte UUID:

    54 52 49 4E 45 54 49 4D 55 53 45 49 00 01 00 00
    T  R  I  N  E  T  I  M  U  S  E  I

After the UUID the payload continues with a small header and then the samples:

| offset | size | type   | field       | meaning                                                            |
| ------ | ---- | ------ | ----------- | ------------------------------------------------------------------ |
| 0      | 16   | bytes  | uuid        | the TRINETIMUSEI UUID above                                        |
| 16     | 1    | uint8  | version     | sample-format version (matches the `.imu` `version`: 5 on current units) |
| 17     | 2    | uint16 | num_samples | number of IMU sample records that follow                           |
| 19     | 2    | uint16 | accel_fs    | accelerometer full-scale code (same codes as the `.imu` header)    |
| 21     | 2    | uint16 | gyro_fs     | gyroscope full-scale code                                          |
| 23     | …    | —      | samples     | `num_samples` × 80-byte records, identical layout to the `.imu` body |

Each frame carries the samples acquired since the previous frame's NAL. The
payload `version` tells you how to read the trailing float — `fsync_delay_us` for
version < 5, `mag_age_us` for version ≥ 5 — exactly as in the `.imu` file. (Older
UVC firmware stamped this byte as `1`; treat any version < 5 as the frame-sync
interpretation.) `trinet_tools.extract_sei` rebuilds full `.imu` and `.vts`
sidecars from such a stream, writing the matching `.imu` header version.

On-board SD recordings do **not** embed SEI samples — they have a separate
`.imu` file already.

## Device ID

Each Trinet camera unit has a stable 16-byte public **device ID**, derived
from the camera's unique factory identifier via a one-way hash. It can be used
to attribute a recording to a specific unit (e.g. for fleet tracking, multi-camera
data sets, or audit logs).

You can obtain the device ID in three places:

  1. **`.imu` header reserved bytes** (offsets 40–55) for on-board SD recordings.
  2. **USB iSerialNumber** when the camera is connected over UVC (e.g. on
     Android, `UsbDevice.getSerialNumber()`; on Linux, `lsusb -v`).
  3. **Camera-side host metadata** when an external recorder application
     (e.g. the Trinet Android app) saves it alongside its recordings.

The ID is one-way — there is no useful information you can recover from it
about the camera's hardware. It is safe to log, store, and share.

Backwards compatibility: recordings made with firmware predating the
device-id field have all-zero `reserved` bytes. The reader treats this as
"device_id unknown" and surfaces an empty string for `header.device_id_hex`.

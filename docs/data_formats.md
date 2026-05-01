# Trinet Recording Data Formats

A Trinet camera recording consists of synchronized video and inertial-measurement
data. This document specifies the on-disk file formats so anyone can write their
own reader, validator, or exporter without depending on the Python tools.

All multi-byte integers and floats are **little-endian**. All timestamps are
**monotonic nanoseconds** (i.e. they always increase, but they are not wall-clock
times — they reset to 0 when the camera powers up).

## Recording shapes

A recording exists in one of two shapes:

### 1. On-board SD recording (camera writes to its SD card)

A triple of files sharing a base name, e.g.

    Trinet/recording1.mp4    # H.264 video
    Trinet/recording1.imu    # inertial samples
    Trinet/recording1.vts    # per-frame video timestamps

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

## `.imu` file format (TRIMU001)

A 64-byte header followed by fixed-size sample records.

### Header (64 bytes)

| offset | size | type     | field            | meaning                                          |
| ------ | ---- | -------- | ---------------- | ------------------------------------------------ |
| 0      | 8    | char[8]  | magic            | ASCII `"TRIMU001"`                               |
| 8      | 4    | uint32   | version          | 3 (current)                                      |
| 12     | 4    | uint32   | sample_rate_hz   | nominal IMU output data rate (e.g. 562)          |
| 16     | 2    | uint16   | accel_fs         | 0=±2 g, 1=±4 g, 2=±8 g, 3=±16 g                  |
| 18     | 2    | uint16   | gyro_fs          | 0=±250, 1=±500, 2=±1000, 3=±2000 dps             |
| 20     | 8    | uint64   | start_time_ns    | monotonic ns of first sample in this file        |
| 28     | 8    | uint64   | video_start_ns   | monotonic ns of first video frame (0 if unknown) |
| 36     | 4    | uint32   | flags            | bit 0 = frame-sync alignment captured            |
| 40     | 24   | bytes    | reserved         | see device_id below                              |

The first 16 bytes of `reserved` carry the **public device ID** — a stable
per-unit identifier (see [Device ID](#device-id) below). The remaining 8
reserved bytes are zero and reserved for future use.

Recordings produced by older firmware (before the device-id field) have the
reserved region all zero. Treat all-zero device_id as "unknown / pre-v4
recording" — the rest of the file is fully compatible.

### Sample (80 bytes, repeated until EOF)

| offset | size | type     | field            | meaning                                                          |
| ------ | ---- | -------- | ---------------- | ---------------------------------------------------------------- |
| 0      | 8    | uint64   | timestamp_ns     | monotonic ns when the sample was acquired                        |
| 8      | 12   | float[3] | accel[xyz]       | m/s² (gravity included)                                          |
| 20     | 12   | float[3] | gyro[xyz]        | rad/s                                                            |
| 32     | 12   | float[3] | mag[xyz]         | µT (zero if magnetometer unavailable)                            |
| 44     | 4    | float    | temp_c           | °C (sensor die temperature)                                      |
| 48     | 16   | float[4] | quat_xyzw        | reserved for on-device fusion; current firmware writes zeros     |
| 64     | 12   | float[3] | lin_accel[xyz]   | reserved for on-device fusion; current firmware writes zeros     |
| 76     | 4    | float    | fsync_delay_us   | µs offset between the nearest frame-sync pulse and this sample;  |
|        |      |          |                  | 0 if no pulse landed in this sample's window                     |

Number of samples = `(file_size - 64) / 80`. There is no count field and no
trailer — readers compute the count from the file size.

### Frame-sync alignment

When `flags` bit 0 is set, the sensor was wired to receive a hardware sync
pulse from the imager at the start of every video frame. The
`fsync_delay_us` field on the *next* IMU sample after each pulse contains the
sub-microsecond offset between the pulse and the sample, allowing
post-processing to align inertial samples to video frames precisely. Samples
without a pulse in their window store 0 here.

## `.vts` file format (TRIVTS01 v2)

Per-frame video timestamp sidecar. Maps each encoded video frame to the
monotonic clock that timestamps the IMU samples.

### Header (32 bytes)

| offset | size | type     | field             | meaning                              |
| ------ | ---- | -------- | ----------------- | ------------------------------------ |
| 0      | 8    | char[8]  | magic             | ASCII `"TRIVTS01"`                   |
| 8      | 4    | uint32   | version           | 2 (current)                          |
| 12     | 4    | uint32   | frame_rate_milli  | configured fps × 1000 (e.g. 30000)   |
| 16     | 16   | bytes    | reserved          | zero                                 |

### Entry (24 bytes, repeated, one per encoded video frame)

| offset | size | type     | field             | meaning                                                |
| ------ | ---- | -------- | ----------------- | ------------------------------------------------------ |
| 0      | 4    | uint32   | frame_number      | 0-based index into this MP4 file                       |
| 4      | 8    | uint64   | sof_timestamp_ns  | start-of-frame monotonic ns (best clock for sync); 0 if unavailable |
| 12     | 4    | uint32   | venc_seq          | encoder-internal sequence number                       |
| 20     | 8    | uint64   | venc_pts_us       | encoder presentation timestamp in microseconds         |

For inertial-to-video alignment, **prefer `sof_timestamp_ns`** (when nonzero):
it is captured at the camera the moment the imager started reading out a frame,
and matches the same clock as the IMU `timestamp_ns`. `venc_pts_us` is useful
for matching the entry to a packet you read from the MP4, but is delayed by
encoder pipeline latency relative to the actual capture instant.

In chunked recordings, `frame_number` resets to 0 at the start of each part —
so each part's `.vts` reads independently.

## `video.mp4`

Standard MP4 container with one H.264 (or H.265, if the firmware was so
configured) video track. No audio. Fragmented MP4 internally so that even a
power-loss-truncated file is recoverable up to the last keyframe interval.

### When SEI inertial data is embedded

Host-side UVC recordings additionally carry inertial samples inside SEI
user_data_unregistered NALs whose payload starts with this 16-byte UUID:

    54 52 49 4E 45 54 49 4D 55 53 45 49 00 01 00 00
    T  R  I  N  E  T  I  M  U  S  E  I

After the UUID, the SEI payload is a tightly packed array of v3 IMU samples
(80 bytes each, exact same record layout as the `.imu` file body). Each frame
carries the samples that were acquired between the previous frame's NAL and
this one. `trinet_tools.extract_sei` rebuilds full `.imu` and `.vts` sidecars
from such a stream.

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

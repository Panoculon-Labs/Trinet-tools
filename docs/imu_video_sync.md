# IMU ↔ Video Time Synchronization (UVC / NCM stream)

This note explains how to align a Trinet camera's IMU with its video frames, in
two layers:

1. **Per-frame timing** — put each frame on the IMU's clock using the hardware
   capture timestamp embedded in the stream (removes video delivery latency →
   sub-frame precision).
2. **Post-calibration offset** — apply the Kalibr-calibrated `timeshift_cam_imu`
   to remove the residual constant offset (rolling-shutter readout centre + IMU
   group delay + pipeline). This is the layer that gets you to a VIO-grade,
   physically-correct alignment.

## TL;DR

- The **hardware frame-capture timestamp is embedded in the video stream**, in
  the IMU SEI of every frame (and recovered into `.vts` as `sof_timestamp_ns`).
- A measured IMU↔video offset of ~30–40 ms is **video delivery latency** (encode +
  USB transport + host decode/jitter buffer), *not* a sync error. It appears only
  if you compare against the **decoded frame's arrival / presentation time (PTS)**.
- Align the IMU against each frame's **`sof_timestamp_ns`** instead — both are on
  the camera's monotonic clock and are latched in hardware, so delivery latency
  cancels out and the nearest IMU sample lands **sub-millisecond** from the frame.
- For the final, physically-correct alignment, **add the Kalibr `timeshift_cam_imu`**
  (≈ −15 ms here). `sof_timestamp_ns` gives you sub-ms *precision*; the timeshift
  removes the systematic *offset* between the frame's reported time and its true
  effective capture instant. See [Post-calibration refinement](#post-calibration-refinement-kalibr-timeshift).

## What's actually in the stream

Each encoded video frame is preceded by an SEI NAL carrying the IMU samples for
that frame plus a per-frame timing block. `extract_sei` recovers this into the
`.imu` + `.vts` sidecars, where `read_vts` exposes:

- **`sof_timestamp_ns`** — the frame's hardware capture time on the camera's
  monotonic clock. On current recordings this is referenced to the **centre of the
  exposure window** (mid-exposure); the `TIMING_MID_EXPOSURE` flag says which.
- **`exposure_us`** — the applied integration time (when `TIMING_EXPOSURE_VALID`).
- **`readout_time_us`** — the rolling-shutter readout span, first row → last row
  (when `TIMING_READOUT_VALID`). The per-row delay (Kalibr's `line_delay`) is
  `readout_time_us / image_height`.

(Older recordings instead carry a per-sample `fsync_delay_us` in the `.imu`, the
offset from each sample to the hardware frame-sync pulse; `extract_sei` subtracts it
to produce the same `sof_timestamp_ns`.)

Because the IMU and the frame timestamp are carried **together** inside the same
SEI, their relationship is fixed no matter how long the encoded frame takes to be
delivered and decoded. **Never align the IMU against the video PTS / frame arrival
time** — that timeline includes the delivery latency.

## Recover it with `extract_sei`

```bash
git clone https://github.com/Panoculon-Labs/Trinet-tools.git
cd Trinet-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ffmpeg/ffprobe must be on PATH

python3 -m trinet_tools.extract_sei your_recording.mp4 --out out/
# -> out/imu.bin     (TRIMU: all IMU samples)
# -> out/frames.bin  (TRIVTS: per-frame sof_timestamp_ns + venc_pts)
# -> out/video.mp4   (clean, decodable copy)
```

`frames.bin` carries the hardware **`sof_timestamp_ns`** for every frame. That is
the timeline to align IMU to.

## Verify the per-frame timing

```python
from trinet_tools.reader import read_imu, read_vts
import numpy as np

imu = read_imu("out/imu.bin")
vts = read_vts("out/frames.bin")

sof = vts.sof_timestamps_ns.astype(np.int64)     # per-frame hardware capture time (ns)
imu_ts = imu.timestamps_ns.astype(np.int64)      # IMU sample times (same monotonic clock)

# 1) Frame cadence from the hardware timestamp (should be a clean 30 fps)
d = np.diff(sof) / 1e6                            # ms
print(f"cadence: {d.mean():.3f} ms ({1000/d.mean():.2f} fps), "
      f"std {d.std():.3f} ms, monotonic={bool(np.all(d > 0))}")

# 2) Distance from each frame timestamp to the nearest IMU sample
idx = np.clip(np.searchsorted(imu_ts, sof), 1, len(imu_ts) - 1)
nearest_us = np.minimum(np.abs(imu_ts[idx] - sof), np.abs(imu_ts[idx-1] - sof)) / 1e3
print(f"frame-to-nearest-IMU: median {np.median(nearest_us):.3f} us, "
      f"95th pct {np.percentile(nearest_us, 95):.3f} us")
```

The IMU sample nearest each frame's timestamp is within **~0.35 ms (median)** and
**< 1 ms (95th percentile)** — sub-millisecond *precision* on the IMU timeline. To
place a frame on that timeline, use its `sof_timestamp_ns` (or interpolate the IMU
at that time); never use the frame's PTS / arrival time.

> **Precision vs. accuracy.** The number above is how close the nearest IMU sample
> is to the frame *timestamp* — it does **not** mean the frame *timestamp* equals
> the frame's true capture instant. There is a constant systematic offset between
> the two; the next section calibrates and removes it.

## Post-calibration refinement (Kalibr `timeshift_cam_imu`)

Aligning to `sof_timestamp_ns` removes the (large, variable) delivery latency, but a
**constant residual offset** remains between the frame's reported timestamp and the
instant the scene was actually integrated onto the sensor. For a rolling-shutter
camera with a wide fisheye + IMU, that residual is:

- **+ half the rolling-shutter readout** — `sof_timestamp_ns` is referenced to the
  top row, but the bulk of the image (the centre row) is read out ~`readout/2`
  later. With a ~26.5 ms readout that's **~13 ms** — the dominant term. (Because
  `sof` is already mid-*exposure*, the exposure time itself cancels here and the
  offset does **not** drift with auto-exposure.)
- **+ IMU group delay + pipeline latency** — the inertial sensor's internal
  filtering delays its samples by ~1–2 ms, plus small constant ISP/transport
  offsets.

These sum to a stable per-design constant (**≈ 15 ms** for the current cameras).
Kalibr's camera–IMU calibration estimates exactly this as **`timeshift_cam_imu`**,
written into `calibration.json`:

```json
"extrinsics": {
  "timeshift_cam_imu_sec": -0.0156,
  "timeshift_sign_convention": "t_imu = t_cam + timeshift_cam_imu_sec"
}
```

### Applying it

To convert a frame's timestamp into the IMU clock, add the timeshift:

```python
import json
cal = json.load(open("calibration.json"))
dt = cal["extrinsics"]["timeshift_cam_imu_sec"]          # e.g. -0.0156 s

# IMU state that corresponds to frame i:
t_imu_ns = sof[i] + int(dt * 1e9)                        # t_imu = t_cam + timeshift
# then interpolate the IMU (accel/gyro) at t_imu_ns.
```

Use the calibrated `timeshift_cam_imu` rather than hand-deriving it — it folds the
readout centre, the IMU group delay, and the pipeline latency into one measured
number, and it is consistent across same-design units (typ. −15 to −16.5 ms, spread
< 1 ms). It is the same value your VIO / fusion stack should consume as the
camera–IMU time offset.

### Per-row (rolling-shutter) precision

For pixel-accurate work, account for the rolling shutter per row. Row `r` is
captured at:

```
t_row(r) = sof_timestamp_ns + timeshift + (r - r_ref) * line_delay
line_delay = readout_time_us * 1000 / image_height        # ns per row
```

where `r_ref` is the row the timestamp references (image centre when
`TIMING_MID_EXPOSURE` is set). `readout_time_us` is a fixed property of the sensor
mode (≈ 26.5 ms over 1080 rows ≈ 24.5 µs/row here) and is available per-frame in
v4+ `.vts`. Most consumers can ignore this and treat the frame as captured at
`sof + timeshift`; VIO front-ends that model rolling shutter should use `line_delay`.

## Summary

| Layer | Source | Removes | Result |
|-------|--------|---------|--------|
| Align to `sof_timestamp_ns` (not PTS) | SEI / `.vts` | delivery latency (~30–40 ms) | sub-ms precision on the IMU clock |
| Add `timeshift_cam_imu` | `calibration.json` (Kalibr) | readout-centre + IMU/pipeline offset (~15 ms) | physically-correct cam↔IMU alignment |
| Apply `line_delay` per row | `readout_time_us` / height | intra-frame rolling-shutter skew | pixel-accurate timing |

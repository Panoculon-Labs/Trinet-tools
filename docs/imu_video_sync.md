# IMU ↔ Video Time Synchronization (UVC / NCM stream)

This note explains how to obtain **sub-millisecond** IMU-to-video frame
synchronization from a Trinet camera

## TL;DR

- The **hardware frame-capture timestamp is embedded in the video stream**, in
  the IMU SEI of every frame. 
- A measured IMU↔video offset of ~30–40 ms is the **video delivery latency**
  (H.264 encode + USB transport + host decode/jitter buffer), *not* a sync
  error. It appears only if you compare the IMU against the **decoded frame's
  arrival / presentation time (PTS)**.
- Align the IMU against each frame's **hardware Start-of-Frame (SoF)** timestamp
  instead — both are on the camera's monotonic clock and are latched in
  hardware, so the result is **sub-millisecond** and the delivery latency
  cancels out.

## What's actually in the stream

Each encoded video frame is preceded by an SEI NAL containing the IMU samples
captured for that frame. Every IMU sample carries:

- `timestamp_ns` — the sample's capture time on the camera's monotonic clock.
- `fsync_delay_us` — the offset between that sample and the **hardware FSYNC
  pulse** that marks the frame's Start-of-Frame. This is the same hardware
  latch the SD-card path uses; over USB it travels per-sample in the SEI.

The per-frame Start-of-Frame time is therefore:

```
sof_ns = imu_timestamp_ns − fsync_delay_us × 1000
```

Because the IMU and the SoF are carried **together** inside the same SEI, the
relationship between them is fixed no matter how long the encoded frame takes to
be delivered and decoded. **Do not align the IMU against the video PTS / frame
arrival time** — that timeline includes the delivery latency.

## Recover it with `extract_sei`

This repo ships a tool that pulls the IMU and the per-frame SoF out of a
UVC/NCM-recorded MP4 and writes the same `.imu` + `.vts` sidecars you would get
from an on-board SD recording:

```bash
git clone https://github.com/Panoculon-Labs/Trinet-tools.git
cd Trinet-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ffmpeg/ffprobe must be on PATH

python3 -m trinet_tools.extract_sei your_recording.mp4 --out out/
# -> out/imu.bin     (TRIMU001: all IMU samples + fsync_delay)
# -> out/frames.bin  (TRIVTS01: per-frame sof_timestamp_ns + venc_pts)
# -> out/video.mp4   (clean, decodable copy)
```

`frames.bin` carries the hardware **`sof_timestamp_ns`** for every frame
(derived from the SEI's `fsync_delay_us`). That is the timeline to align IMU to.

## Verify it yourself

```python
from trinet_tools.reader import read_imu, read_vts
import numpy as np

imu = read_imu("out/imu.bin")
vts = read_vts("out/frames.bin")

sof = vts.best_timestamps_ns.astype(np.int64)   # per-frame hardware Start-of-Frame (ns)
imu_ts = imu.timestamps_ns.astype(np.int64)      # IMU sample times (same monotonic clock)

# 1) Frame cadence from the hardware SoF (should be a clean 30 fps)
d = np.diff(sof) / 1e6                            # ms
print(f"SoF cadence: {d.mean():.3f} ms ({1000/d.mean():.2f} fps), "
      f"std {d.std():.3f} ms, monotonic={bool(np.all(d > 0))}")

# 2) IMU<->frame sync: distance from each frame's SoF to the nearest IMU sample
idx = np.clip(np.searchsorted(imu_ts, sof), 1, len(imu_ts) - 1)
nearest_us = np.minimum(np.abs(imu_ts[idx] - sof), np.abs(imu_ts[idx-1] - sof)) / 1e3
print(f"IMU-to-SoF: median {np.median(nearest_us):.3f} us, "
      f"95th pct {np.percentile(nearest_us, 95):.3f} us")
```

### Expected output (measured on a real 20 s recording)

```
SoF cadence: 33.338 ms (30.00 fps), std 1.017 ms, monotonic=True
IMU-to-SoF: median 352 us, 95th pct 841 us
```

The IMU sample nearest each frame's hardware SoF is within **~0.35 ms (median)**
and **< 1 ms (95th percentile)** — i.e. **sub-millisecond** sync, well inside a
2 ms tolerance. To place a frame on the IMU timeline, use its `sof_timestamp_ns`
(or interpolate the IMU at that time); never use the frame's PTS/arrival time.

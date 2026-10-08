# Changelog

## Unreleased — re-sync long multi-camera takes onto the master clock (`scripts/resync_take.py`)

A synced camera's `.vts` (v3/v4) records the clock offset to the group master
once, at the start of the take. Camera clocks run a few parts per million apart,
so on long takes the recorded timelines slowly separate even though the frames
themselves stay locked together — typically 5–15 ms per hour, and growing.

- `resync_take.py` measures the real clock relationship from the recording:
  it pairs every frame with the master frame it was captured with, fits the
  offset across the take (following slow rate changes as the units warm up),
  and rewrites each camera's frame **and** IMU timestamps onto the master's
  clock. Output files keep the input format with the header offset zeroed, so
  every existing tool reads them unchanged; originals are never modified.
- Also writes a frame-matching CSV (each master frame → the matching frame of
  every camera), a JSON report, and with `--plot` a before/after chart.
- On a 47-minute three-camera take the old timeline had drifted 6.8 / 12.7 ms;
  after re-sync the cameras agree to 0.06 ms (median). On a simulated 2-hour
  take with 9 ppm drift, thermal wander and dropped frames, the offset is
  recovered to under 0.2 ms (`tests/test_resync_take.py`).
- `--check-video` scans each `.mp4` for damaged frames (no decoding, ~2 s per
  file). Decoders drop or stall on them, which shifts every later frame against
  its timestamp in tools that count decoded frames, such as `sync_view.py`.

## Unreleased — stereo calibration field check (`scripts/check_calibration.py`)

A target-free check that a stereo camera's calibration still fits, run on an
ordinary recording. It measures the constant vertical offset between the
rectified eyes and prints ok (≤ 1.5 px), check (1.5–3 px) or recalibrate
(> 3 px), with exit status 0 / 1 / 2 (3 = too few features to measure).

- `stereo_align`: the offset is now measured with sub-pixel optical flow
  (forward-backward checked) over 15 frame pairs instead of whole-pixel ORB
  matches over 5, and reported with its spread across the take
  (`measure_y_offset`, `calibration_verdict`). On the published Stereo GS
  sample it reads +0.69 px (independent measurement: +0.56 px); a 4 px shift of
  one eye's calibration reads +5.0 px → recalibrate.
- `stereo_depth_video.py` always prints the measured offset and its verdict;
  the "consider recalibrating" message now appears only above 1.5 px (it fired
  at 0.5 px before, which healthy units exceed).

## Unreleased — put card recordings on UTC (`scripts/wireless_utc.py`)

Trinet cameras that record to their own memory card can broadcast their status
over Bluetooth LE, and the Trinet Android app (or your own app on the Trinet
SDK) logs that broadcast and exports it. The new `wireless_utc.py` reads the
export together with the card's recordings and gives each take — and optionally
every frame — a UTC time. A camera has no real-time clock, so this is the way to
line card recordings up with anything else you log in wall-clock time.

**How it works.** Every advert carries the camera's clock; the phone notes when
each one arrived. The tool fits camera time to phone time over the whole session
(it keeps the least-delayed advert of every five seconds, rejects late
outliers and follows slow clock drift), then maps phone time to UTC using the
phone's clock references — preferring internet (SNTP) time, then network time,
then the phone's own clock. On a field check with a three-camera kit the phone's
clock was 0.7 s slow; the tool corrected it, and all three cameras agreed on each
take's start within 0.32 ms.

**Which take belongs to which clock.** Recordings from current firmware carry the
camera's boot id; the tool matches on it first. Older recordings are matched by
their take number and the exact time of their first frame, then by time range
(reported as low confidence, or ambiguous with the candidates listed — resolve
with `--pick`). A camera the phone never heard is placed through the other
cameras of its kit. Every result carries a confidence level and two error
estimates: `rel_sigma_ms` (between cameras) and `abs_sigma_ms` (including the
phone's own clock).

**Outputs.** `wireless_utc.json` / `.csv` with one row per file (first and last
frame UTC, duration, match method, confidence, error terms), `--per-frame`
CSVs, `--write-sidecars` `<take>.utc.json` files next to the recordings (never
overwritten without `--force`), a kit-consistency check per session, and
`--inspect` to see what an export contains. Stereo, kit and single-camera card
layouts are all recognised, including chunked recordings.

**Which camera is which.** Exports from the app now carry each camera's model
(Pro Mono / Pro Stereo / Pro Stereo GS), hardware generation, firmware version
and build (`board`, `hw_generation`, `fw_version`, `build` on each `unit`
record; absent for cameras on firmware before 0.5.9). `--inspect` lists them.

**Also new**
- `trinet_tools.tmf.read_tmf_meta()` reads a recording's embedded metadata
  without reading the whole file.
- `docs/wireless_log_format.md`: the export format, and `docs/wireless_utc.md`:
  the field workflow, options and what the error estimates mean.
- A test suite: `python3 -m pytest -q` (see `requirements-dev.txt`).

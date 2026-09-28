# Changelog

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

**Also new**
- `trinet_tools.tmf.read_tmf_meta()` reads a recording's embedded metadata
  without reading the whole file.
- `docs/wireless_log_format.md`: the export format, and `docs/wireless_utc.md`:
  the field workflow, options and what the error estimates mean.
- A test suite: `python3 -m pytest -q` (see `requirements-dev.txt`).

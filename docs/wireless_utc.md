# Putting card recordings on UTC (wireless status log)

A Trinet camera has no wall clock. Its frame timestamps count nanoseconds from
power-on, so a recording on the memory card says *how long after boot* each frame
was captured, not *when*. `scripts/wireless_utc.py` supplies the "when": it uses
the wireless status log exported by the Trinet Android app (or any app built on
the Trinet SDK) to give every take the UTC time of its first and last frame, or of
every frame, with an uncertainty and a confidence.

```bash
python scripts/wireless_utc.py phone_export.jsonl.gz --recordings /media/CARD1 /media/CARD2
```

The log format is specified in [wireless_log_format.md](wireless_log_format.md).

## Recommended field workflow

1. **Before recording**, open the phone app and start the wireless monitor. Keep the
   phone within Bluetooth range of the cameras (a pocket or a bag is fine).
2. **Keep it running for the whole session**, including a few minutes before the
   first take and after the last one. The more of a camera's running time the
   phone hears, the better the clock fit; a take that lies outside what the phone
   heard is extrapolated and its uncertainty grows (see below).
3. If you can, give the phone a network connection. Network time (or the
   app's optional SNTP check) makes the phone's own clock much more trustworthy
   than its wall clock alone.
4. **Export** the log from the app (the `.jsonl.gz` file). Exports from several
   phones, or several exports from one phone, can all be passed together;
   duplicates are removed.
5. Copy or mount the cards and run the tool with every export and every card.
6. Check the summary: every take should be `resolved`, ideally at `high`
   confidence. Investigate anything `ambiguous` or `unmatched` (see
   [Troubleshooting](#troubleshooting)).
7. Optional: `--write-sidecars` stores the result next to each recording as
   `<take>.utc.json`, so the UTC travels with the files.

You do not need to keep the phone connected to anything, and the cameras do not
need to be paired with the phone.

## Running it

```text
python scripts/wireless_utc.py LOG [LOG ...] --recordings DIR [DIR ...]
    [-o OUTDIR] [--json wireless_utc.json] [--csv wireless_utc.csv] [--per-frame]
    [--write-sidecars [--force]] [--unit a1b2c3d4]
    [--fit local|global|live] [--window-s 600] [--max-extrapolation-s 1800]
    [--phone-utc best|system|network|sntp|corrected] [--latency-correction-ms 0]
    [--pick TAKEPATH=STORE:SEGMENT] [--min-confidence low|medium|high] [--strict]

python scripts/wireless_utc.py LOG --inspect
```

| option | meaning |
|---|---|
| `--recordings` | folders to search recursively (card mounts, copies, archives) |
| `-o`, `--json`, `--csv` | where to write the reports (default: current folder) |
| `--per-frame` | also write `<take>_<eye>.utc.csv` with the UTC of every frame |
| `--write-sidecars` | write `<take>.utc.json` next to each resolved take; existing files are kept unless `--force` |
| `--unit` | unit id to assume for recordings that carry none (very old firmware) |
| `--fit` | `local` (default) refit, one `global` line, or the phone's `live` fit (for comparison) |
| `--window-s` | half-width of the local fit window (default 600 s) |
| `--max-extrapolation-s` | how far outside the heard period a take may lie (default 1800 s) |
| `--phone-utc` | which phone time to trust; `best` = SNTP, else network time, else the phone's clock |
| `--latency-correction-ms` | subtract an assumed radio latency from every time (default 0) |
| `--pick` | resolve an ambiguous take by naming its log segment (the candidates are listed in its notes) |
| `--min-confidence` | results below this level are reported as `below_min_confidence` |
| `--strict` | exit with status 2 if any take is not resolved |
| `--inspect` | describe the log (cameras, heard periods, fit quality, phone clock) and exit |

Exit status: `0` success (with `--strict`: every take resolved), `2` some take
unresolved under `--strict`, `1` error (unreadable log, no recordings found, bad
arguments).

The same functions are available from Python:

```python
from trinet_tools.wireless_log import load_logs
from trinet_tools import wireless_utc as wu

log = load_logs(["phone_export.jsonl.gz"])
takes = [wu.load_take(f) for f in wu.discover_takes(["/media/CARD1"])]
for m in wu.match_takes(takes, wu.build_runs(log), log):
    for row in wu.take_to_utc(m).rows:
        print(row["take_path"], row["eye"], row["utc_first_frame_iso"], row["abs_sigma_ms"])
```

## How it works

### 1. What the phone records

While a camera records, it broadcasts a small status message several times a
second: its unit id, a one-byte boot nonce (it changes every time the camera
powers on), the current take number, recording start/stop edges, and **its own
clock** (milliseconds since power-on). The phone stamps each message with its own
arrival time.

Arrival is always *after* the camera stamped the message: radio and operating
system latency can only delay it. So for every 5 s of camera time the phone keeps
only the message that arrived **soonest** (the bucket minimum). Plotted against
camera time, these minima trace the lower edge of the latency cloud, which is the
camera-to-phone clock relation plus a small, nearly constant residual latency.

The phone also records pairs of its uptime clock and its wall clock (every
minute, and whenever the phone's time is changed), plus network time and SNTP
measurements when available.

### 2. Refitting the camera clock

The tool gathers each camera boot as heard by each phone boot into a *run*
(several log segments are merged when their offsets agree within 20 ms, which also
bridges an app restart). For each run it:

- rejects buckets whose minimum was still late (one-sided: a bucket can only be
  too late, never too early), using a robust scatter estimate;
- fits a **local line** around the time of interest: a weighted linear fit over
  the buckets within ±10 minutes (widened until at least 8 buckets are in), so the
  slow drift of the camera's crystal with temperature is followed rather than
  averaged away. `--fit global` uses one straight line over the whole run instead
  (fine for short sessions at stable temperature); `--fit live` uses only the
  phone's own quick fit of the last two minutes (what the app shows live, and all
  a legacy v1 log contains).

Outside the heard period the edge of the fit is extrapolated, and its uncertainty
grows by 2 ppm of the distance (7.2 ms per hour).

The camera's clock counts milliseconds in 32 bits and wraps after about 49.7
days of continuous uptime; the tool reconciles the phone's unwrapping with the
full-width timestamps in the recording.

### 3. From phone uptime to UTC

Per phone boot the tool builds a map from the phone's uptime clock to UTC:

| source | used when | 1σ |
|---|---|---|
| SNTP | the log has SNTP measurements (`--phone-utc best` prefers it) | half the measurement round trip (a few ms) |
| network time | the phone reported network-provided time | ~25 ms |
| phone clock | nothing better | ~100 ms (nominal) |

With the phone clock alone, a change of the phone's time (manual, time zone
servers, or an automatic correction) splits the timeline at that point; a change
within 30 minutes of a take is listed in `phone_clock_steps` and its size is added
to the absolute uncertainty. `--phone-utc corrected` trusts the clock as it was
last set and applies later corrections backwards. Outside the span of the
recorded clock pairs the map is held constant with 50 ppm of drift allowed.

### 4. Which run does a take belong to?

A camera reboots, gets carried between phones, and reuses take numbers across
cards, so the tool first decides which run each take belongs to. It reads from the
card: the unit id (from the MP4's embedded metadata, else the `.imu` header, else
the `.json` sidecar, else the kit file-name prefix, else `--unit`), the camera's
**boot id** (current firmware writes it into the MP4's embedded metadata; its
first byte is the advertised nonce), the take number from the file name, and the
frame timestamps from the `.vts` (or, when that is missing, from the MP4 itself).

| tier | evidence | `match_method` |
|---|---|---|
| 1 | boot id: runs of the same unit and boot nonce that cover the take | `boot_id` |
| 2 | no boot id: a logged start/stop edge with the same take number whose camera time equals the file's first/last frame (±2 ms) | `edge` |
| 3 | neither: runs of the unit whose heard period covers the take | `range` |
| 4 | a camera of a kit that the phone never heard, placed through its kit-mates' matched takes | `kit_mate` |
| — | chosen with `--pick` | `picked` |

Details:

- **Boot id, not seen**: if the unit was heard but never in this boot, the take
  is `unmatched` (`boot_not_observed`) and no weaker tier is tried; any time
  would be a guess.
- **Nonce collisions**: the nonce is one byte, so two boots of one camera can
  share it. Such candidates disagree by at least the time between the boots and
  are told apart by the recorded start/stop edge, or by other takes carrying the
  same full boot id.
- **Older firmware** (recordings without a boot id) advertised the raw
  start-of-frame instant, which differs from the `.vts` time by the
  mid-exposure shift (half the exposure, less half the readout). The edge test
  reconstructs that shift from the `.vts` exposure fields, or widens its tolerance
  when it cannot.
- **Kit cameras**: a paired camera broadcasts the kit master's clock, and its
  `.vts` carries the offset to that clock. Such runs are applied to the
  master-clock timeline of the take (`timeline = kit_master`), but only if the
  take was actually synced; a take recorded unsynced cannot be placed through a
  master-clock run (`timebase_inconsistent`).

### 5. Uncertainty

Every result carries 1-sigma uncertainties in milliseconds:

- **`rel_sigma_ms`** — how well the camera clock is placed on the phone's
  uptime clock: the statistical error of the fit at that time, plus
  extrapolation (2 ppm x distance outside the heard period), plus, for kit
  cameras, the kit offset quality and a 2 ms penalty when the file flags frames
  whose kit offset was carried forward while the kit link was down. This is the
  figure to compare takes and cameras **heard by the same phone**: their relative
  timing is known to about this accuracy.
- **`abs_sigma_ms`** — the absolute UTC uncertainty: `rel_sigma_ms` combined
  with the phone's own UTC uncertainty (source table above, drift beyond the
  clock pairs, and nearby clock steps). This is usually dominated by the phone.
- **`latency_bias_ms`** — the unobservable part: the fit sits on the fastest
  messages, which were still late by the minimum radio latency and a fraction of
  the camera's whole-millisecond stamping. All reported times are therefore late
  by a small constant, typically 0.3 to 1.5 ms, which is not removed and not
  included in the sigmas; `--latency-correction-ms` subtracts a value of your
  choice (the reported range shifts with it).

`fit_residual_ms` is the scatter of the kept bucket minima about the fit, and
`fit_buckets_used` how many were used; `extrapolation_s` is how far the take lies
outside the heard period; `cross_run_delta_ms` is the disagreement when the same
take was placed through several runs (for example two phones), the smallest-sigma
one being reported.

### 6. Confidence

| confidence | meaning |
|---|---|
| `high` | identity by boot id or edge, the take within 60 s of what the phone heard, `rel_sigma_ms` < 5 |
| `medium` | identity by boot id or edge but extrapolated more than 60 s, or `rel_sigma_ms` 5 to 50; or placed through kit-mates |
| `low` | identity by time range only, chosen with `--pick`, or `rel_sigma_ms` > 50 |
| `ambiguous` | several camera boots fit and nothing decides between them; candidates listed in `notes` |
| `unmatched` | no answer; the reason is the first note |

Unmatched reasons: `no_runs_for_unit` (the phone never heard this camera),
`boot_not_observed` (heard, but not in this boot), `no_covering_run` (heard in this
boot, but the take lies more than `--max-extrapolation-s` outside), 
`timebase_inconsistent` (see kit cameras), `no_vts` (no frame timestamps),
`no_unit_id`.

## Outputs

`wireless_utc.json` and `wireless_utc.csv` hold one row per take and camera eye
(`eye` is `L`/`R` for stereo cameras, empty for single-lens cameras):

| column | |
|---|---|
| `take_path`, `unit_id`, `take_number`, `kit_session`, `eye`, `frames` | the recording |
| `boot_id`, `boot_id_source` | from the MP4 metadata (`live`, or `marker` for a take recovered after a power loss) |
| `status`, `match_method`, `confidence` | see above |
| `log_file`, `store_id`, `run_segments` | which log, phone history and segments placed it |
| `timeline` | `local` (the camera's own clock) or `kit_master` |
| `utc_first_frame_ns` / `_iso`, `utc_last_frame_ns` / `_iso`, `duration_s` | the result; nanoseconds since the Unix epoch, and ISO 8601 UTC |
| `rel_sigma_ms`, `abs_sigma_ms`, `latency_bias_ms` | uncertainty (the bias as `[min, max]`, `min..max` in the CSV) |
| `extrapolation_s`, `fit_residual_ms`, `fit_buckets_used` | fit diagnostics |
| `phone_utc_source`, `phone_clock_steps`, `cross_run_delta_ms` | phone clock diagnostics |
| `notes` | reasons, warnings, ambiguous candidates |

The JSON also carries the tool version, the SHA-256 of every log file, the
parameters, and a **kit consistency** block: for each kit session, the spread of
the UTC the different cameras give to one common master-clock instant. It should
be within the combined `rel_sigma_ms`; a larger spread points at a problem with
one camera's match.

The frame times refer to the `.vts` frame timestamps (the exposure midpoint on
current firmware). `--per-frame` writes `<take>_<eye>.utc.csv` with
`frame_number, sof_ns, timeline_ns, phone_elapsed_ns, utc_ns, rel_sigma_ms`.

`--write-sidecars` writes `<take>.utc.json` next to the recording:

```json
{
  "schema": "trinet-take-utc/1",
  "take": "take0007", "unit_id": "a1b2c3d4", "take_number": 7, "kit_session": null,
  "boot_id": "3c…",
  "eyes": {"L": {"utc_first_frame_iso": "2026-09-21T14:38:20.008755Z", "abs_sigma_ms": 25.0, "...": "..."},
           "R": {"...": "..."}},
  "provenance": {"tool": "trinet-wireless-utc", "tool_version": "1.0.0",
                 "log_files": [{"path": "...", "sha256": "..."}], "params": {"...": "..."}}
}
```

## Troubleshooting

- **`no_runs_for_unit`** — the phone never heard that camera: it was out of range,
  its broadcast was off, or the log was exported with a unit filter. Kit cameras
  may still be placed through their kit-mates.
- **`boot_not_observed`** — the camera was restarted (or its battery swapped)
  and the phone was not listening during that power-on. Nothing can be recovered
  for that boot.
- **`ambiguous`** — older recordings without a boot id where more than one
  heard period fits. Read the candidates in `notes`, then rerun with
  `--pick path/to/take0003=STORE:SEGMENT`.
- **Large `abs_sigma_ms`** — the phone had no network time. Enable network time
  (or SNTP in the app) for the next session; `rel_sigma_ms` is still valid for
  comparing takes.
- **Large `extrapolation_s`** — keep the phone listening for the whole session.
- `--inspect` shows what the log contains: cameras, heard periods, bucket rates,
  fit residuals, clock pairs and clock steps.

## Testing

```bash
pip install -r requirements-dev.txt
python3 -m pytest -q
```

The tests build synthetic cameras, phones, logs and recordings on the fly
(`tests/synth.py`); no recordings are committed.

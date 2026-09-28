# Wireless status log format (v2)

A Trinet camera that records to its own memory card can broadcast its status over
Bluetooth LE. The Trinet Android SDK listens, keeps a history, and exports it in the
format described here. `scripts/wireless_utc.py` reads it to put recordings on UTC
(see [wireless_utc.md](wireless_utc.md)).

This page is the canonical specification. The SDK writer and the Python reader
(`trinet_tools/wireless_log.py`) both follow it.

## Concepts

- **Unit id**: 8 lowercase hex characters, the first 8 of the camera's device id.
- **Camera time (`device_ms`)**: milliseconds since the camera booted, unwrapped
  (the broadcast carries 32 bits; the phone unwraps). This is the same clock as the
  `.vts` frame timestamps (`sof_ts_ns / 1e6`). **Exception**: when
  `timebase_is_master` is true, camera time is the *kit master's* clock, which a
  paired camera's `.vts` also carries via its sync offset (`global_sof_ns()`).
- **Boot nonce**: one byte that changes every camera boot (first byte of the
  camera's boot id). A recording's metadata carries the full boot id; its first two
  hex digits equal the nonce.
- **Phone boot**: the phone's `elapsedRealtime` clock restarts when the phone reboots.
  Every phone-elapsed value is scoped to a `phone_boot` id.
- **Segment**: one continuous run of one camera's clock as heard by one phone boot
  (same unit, boot nonce and timebase). A new segment starts when the nonce or
  timebase changes, when the camera clock jumps, or when the logger restarts.
- **Bucket**: for each segment the phone keeps, per 5 s of camera time, the advert
  that arrived with the **least delay**: `min_offset_ns = rx_elapsed_ns - device_ms*1e6`.
  Radio latency is never negative, so these minima trace the true clock offset. The
  desktop tool refits a line through them.

## File

JSON Lines, UTF-8, one JSON object per line, optionally gzip-compressed (`.jsonl.gz`;
detect by the gzip magic `1f 8b`). Every object has a `"type"`. Records appear in the
order below; readers must ignore unknown types and unknown keys.

A file that is a single JSON object with `"version": 1` is the legacy v1 export (see
the end of this page).

### `header` (first line)
```json
{"type":"header","format":"trinet-wireless-log","version":2,
 "store_id":"<uuid>","sdk_version":"0.5.3","device_model":"<phone model>",
 "exported_utc_ms":1790000000000,"exported_elapsed_ns":123456789000,
 "bucket_ms":5000,
 "scope":{"units":null,"groups":null,"from_utc_ms":null,"to_utc_ms":null}}
```
`store_id` identifies the phone's history store; merging several exports from the
same phone dedups segments by `(store_id, segment id)`.

### `phone_boot`
```json
{"type":"phone_boot","id":3,"boot_count":412,"first_elapsed_ns":1000,"first_utc_ms":1790000000000}
```

### `monitor_run`
```json
{"type":"monitor_run","id":9,"phone_boot":3,"start_elapsed_ns":0,"start_utc_ms":0,
 "stop_elapsed_ns":null,"stop_utc_ms":null}
```

### `clock_ref`
The phone's clock pairing, recorded every 60 s, at start/stop, when the phone's time
is changed, and at export.
```json
{"type":"clock_ref","phone_boot":3,"elapsed_ns":123,"read_span_ns":4000,
 "utc_ms":1790000000123,"network_utc_ms":null,"sntp_utc_ms":null,"sntp_rtt_ms":null,
 "reason":"periodic"}
```
- `utc_ms`: the phone's wall clock (`System.currentTimeMillis()`).
- `network_utc_ms`: the platform's network time at that instant, when available.
- `sntp_utc_ms` / `sntp_rtt_ms`: an optional SNTP measurement.
- `reason`: `periodic | run_start | run_stop | time_set | export | sntp`.
`utc_ms - elapsed_ns/1e6` is the phone's clock offset; a jump of more than 20 ms
between consecutive references (or a `time_set` record) is a clock step.

### `unit`
```json
{"type":"unit","unit_id":"a1b2c3d4","label":"Left wrist","group_id":33012,
 "last_role":"slave","last_address":"D2:80:A1:B2:C3:D4"}
```
`group_id` is the 16-bit kit id (0 = not in a kit). `last_role`: `unpaired | master | slave`.

### `segment`
```json
{"type":"segment","id":17,"unit_id":"a1b2c3d4","phone_boot":3,"boot_nonce":60,
 "timebase_is_master":true,"group_id":33012,"role":"slave",
 "reset_reason":"timebase",
 "first_device_ms":1000,"last_device_ms":900000,
 "first_rx_elapsed_ns":5,"last_rx_elapsed_ns":6,"samples":1800,
 "take_min":5,"take_max":9,"rec_count_min":12,"rec_count_max":16,
 "live_fit":{"ref_device_ns":0,"offset_at_ref_ns":0,"skew_ppm":-29.1,
             "residual_ms":0.3,"buckets":24},
 "buckets":{"device_ms":[1000,6000],"min_offset_ns":[123,456],"n":[9,10]}}
```
- `reset_reason`: `first | nonce | timebase | jump | resume`.
- `live_fit`: the phone's own fit at export time, for reference; the model is
  `phone_elapsed_ns = offset_at_ref_ns + device_ns + skew_ppm*1e-6*(device_ns - ref_device_ns)`.
- `buckets`: columnar arrays of equal length, sorted by `device_ms`; `device_ms` is
  the camera time of the least-delayed advert in that bucket, `n` the adverts seen.
Segments are exported whole, never clipped to the export's time range.

### `event`
```json
{"type":"event","id":501,"segment":17,"unit_id":"a1b2c3d4","kind":"started",
 "take_number":7,"boot_nonce":60,"event_seq":12,"device_ms":123456,"missed_edges":0,
 "phone_boot":3,"detected_elapsed_ns":999,"utc_ms_live":1790000000000}
```
- `kind`: `started | stopped | abnormal_stop`.
- `device_ms` for `started`/`stopped` is the camera time of the **first/last frame
  saved in the file**, identical to that file's first/last `.vts` frame timestamp
  (to the millisecond) on current firmware. For `abnormal_stop` it is the time the
  camera *noticed* the recording had died.
- `utc_ms_live`: the UTC the phone showed live; the desktop tool recomputes it.

### `take`
Pairs of start/stop events built by the phone.
```json
{"type":"take","unit_id":"a1b2c3d4","segment":17,"take_number":7,"boot_nonce":60,
 "pairing":"exact",
 "start":{"event_id":501,"device_ms":123456,"utc_ms_live":1790000000000},
 "stop":{"event_id":502,"device_ms":223456,"kind":"stopped","utc_ms_live":1790000100000}}
```
- `pairing`: `exact | missed_edges | start_only | stop_only`.
- `start` or `stop` may be `null`. `stop.kind`: `stopped | abnormal_stop | reboot`.

### `sighting`
60 s windows of reception, for signal history.
```json
{"type":"sighting","unit_id":"a1b2c3d4","phone_boot":3,"window_start_elapsed_ns":0,
 "adverts":110,"rssi_min":-80,"rssi_max":-61,"rssi_mean":-70.2,"gap_max_ms":2300}
```

### `end` (last line)
```json
{"type":"end","records":12345}
```
`records` is the number of lines before `end`, **including the header**. A
missing `end`, or a count that doesn't match, means the file was truncated;
readers warn and use what they have.

## Scope rules
- `scope.units` / `scope.groups` select segments (by the segment's `group_id`, so a
  kit member whose link dropped is still included).
- `from_utc_ms` / `to_utc_ms` select segments that overlap the range, and events,
  takes and sightings detected inside it.
- All `clock_ref`s of the included phone boots are exported.

## Legacy v1
One JSON object: `format` `"trinet-wireless-status-log"`, `version` 1,
`phone_reference{elapsed_realtime_ns, utc_ms}`, `units[].fits[]`
(`a_ns, b, ref_device_ns, offset_at_ref_ns, skew_ppm, residual_ms, samples, buckets,
first_device_ms, last_device_ms, boot_nonce, timebase_is_master`) and `events[]`.
It has no bucket minima, so the desktop tool can only use the phone's live fit.

#!/usr/bin/env python3
"""Repair delivered Trinet ZIPs to match the delivery spec -- in place.

Python 3 only, plus ffmpeg for the video fixes. Ships as two files kept in the
same folder: this one and repair_recordings.py (used to rebuild a fragmented
MP4's index so ffmpeg can read it). Send both to whoever holds the ZIPs; they
run this over their delivery folder.

For each ZIP it will, as needed:
  * add the required environment_type / sub-category to metadata.json
  * rebuild a fragmented / broken MP4 index and remux it into a clean file
  * split a clip longer than 1 hour into ~30-min chunks, each its own ZIP with
    a time-synced IMU/VTS slice (so each stays within the 120-3600 s spec)
  * re-encode video to the spec bitrate (<=8 Mbps H.264, GOP 30, no B-frames,
    8-bit) by default -- fast, uses the GPU when available (--no-reencode skips)
  * correct an accelerometer that was recorded at the wrong scale (its at-rest
    gravity reads ~2x or ~0.5x of 9.81 m/s^2) by rescaling to the true range
  * refresh the video / IMU numbers in metadata.json to match the repaired files

It rewrites each ZIP via a temp file + atomic replace (or writes copies with
--out), and prints, per file, what it fixed and what still cannot pass
(clip too short, no usable video) so those can be re-collected.

    # the full fix -- metadata + IMU scale + video (remux + re-encode to the
    # spec bitrate). Environment defaults to residential/other_household:
    python3 repair_delivery.py  DELIVERIES/

    # a specific environment, or per-file from a CSV:
    python3 repair_delivery.py  DELIVERIES/  --environment commercial/retail_stocking_back_of_house
    python3 repair_delivery.py  DELIVERIES/  --map env_by_clip.csv

    # skip the (slower) bitrate re-encode -- just fix metadata/IMU/truncation:
    python3 repair_delivery.py  DELIVERIES/  --no-reencode

Re-encode (the bitrate fix) is ON by default and needs ffmpeg; if ffmpeg is
absent it is skipped with a note. It uses every CPU core by default (one worker
per core). Add --dry-run first to see the plan, or --out DIR to write copies
instead of editing in place.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile

# flatten_file rebuilds a fragmented MP4's moov so ffmpeg (and strict players)
# can read it. Shipped alongside as repair_recordings.py; ffmpeg cannot repair
# these itself because it cannot even open them ("trun track id unknown").
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from repair_recordings import flatten_file as _flatten_file
except Exception:                                        # pragma: no cover
    _flatten_file = None


def _say(msg):
    """Print one line, flushed, so live progress reaches the terminal even
    from a worker process."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

SPEC = {"clip_seconds": (120.0, 3600.0), "bitrate_mbps": (6.0, 8.0),
        "gravity_ms2": (9.0, 10.6)}
IMU_HEADER = 64
IMU_SAMPLE = {1: 44, 2: 76, 3: 80, 4: 80, 5: 80, 6: 80}


# --------------------------------------------------------------------------- #
# environment parsing
# --------------------------------------------------------------------------- #
def parse_environment(value):
    if "/" not in value:
        raise argparse.ArgumentTypeError("use TYPE/SUBCATEGORY, e.g. "
                                         "residential/laundry")
    t, s = value.split("/", 1)
    t = t.strip().lower()
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    if not t or not s:
        raise argparse.ArgumentTypeError("both TYPE and SUBCATEGORY required")
    return t, s


def load_env_map(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            k = row[0].strip()
            if k.lower() in ("zip", "zip_name", "episode", "clip", "file"):
                continue
            out[k] = parse_environment(row[1].strip())
    return out


# --------------------------------------------------------------------------- #
# MP4 container audit (stdlib) -- matches the delivery ingest's box check
# --------------------------------------------------------------------------- #
def _be32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def _be64(b, o):
    return struct.unpack_from(">Q", b, o)[0]


def _walk_boxes(f, size):
    """Walk the MP4 box chain of a seekable file/stream. Returns (status,
    detail). Works on a plain file or a stored zip member (both seekable)."""
    o = last_end = 0
    saw = set()
    while o + 8 <= size:
        f.seek(o)
        head = f.read(16)
        if len(head) < 8:
            break
        sz = _be32(head, 0)
        hdr = 8
        if sz == 1:
            if len(head) < 16:
                break
            sz = _be64(head, 8)
            hdr = 16
        elif sz == 0:
            sz = size - o
        if sz < hdr:
            return "broken", "bad box @ %d" % o
        saw.add(head[4:8])
        last_end = o + sz
        o += sz
    if last_end != size:
        return "truncated", "last box ends at %d of %d" % (last_end, size), saw
    if not (b"moov" in saw or b"moof" in saw):
        return "no_video", "no moov/moof", saw
    return "ok", None, saw


def container_status_zip(z, name, size):
    """Audit a stored MP4 inside a ZIP without extracting it.
    Returns (status, detail, fragmented) -- fragmented means moof boxes are
    present, so the moov may lack the index ffmpeg/players need."""
    if size < 16:
        return "empty", "%d bytes" % size, False
    try:
        with z.open(name) as f:
            if not f.seekable():
                return "unknown", "member not seekable", False
            st, detail, saw = _walk_boxes(f, size)
            return st, detail, (b"moof" in saw)
    except Exception as e:
        return "broken", str(e), False


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
def have(tool):
    return shutil.which(tool)


_ENC = {}
_ENCODERS = [
    ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-g", "30", "-bf", "0"]),
    ("h264_videotoolbox", ["-g", "30", "-bf", "0"]),
    ("libx264", ["-preset", "veryfast", "-g", "30", "-bf", "0", "-threads", "0"]),
]


def _encoder_ok(name):
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", name, "-frames:v", "1", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=40).returncode == 0
    except Exception:
        return False


def pick_encoder():
    if "e" in _ENC:
        return _ENC["e"]
    chosen = _ENCODERS[-1]
    for name, extra in _ENCODERS:
        if name == "libx264" or _encoder_ok(name):
            chosen = (name, extra)
            break
    _ENC["e"] = chosen
    return chosen


def prep_video(src, tag=""):
    """Flatten a fragmented MP4 in place so ffmpeg can open it. No-op if the
    file is already a standard MP4. Returns True if it rewrote the index."""
    if not _flatten_file:
        return False
    try:
        r, _d = _flatten_file(src, backup=False, log=lambda *a, **k: None)
    except Exception:
        return False
    if r == "repaired":
        _say("     %-30s rebuilt video index (was fragmented)" % tag)
        return True
    return False


def ff_remux(src, dst):
    """Lossless remux: recover a truncated/broken MP4 by copying streams."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-fflags", "+genpts+discardcorrupt", "-err_detect", "ignore_err",
         "-i", src, "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
         "-movflags", "+faststart", dst],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
    return r.returncode == 0 and os.path.getsize(dst) > 0


def ff_reencode(src, dst, target_mbps, threads=0, duration=None, tag="",
                start_s=None, seg_s=None):
    t = max(1.0, min(float(target_mbps), 8.0))
    enc, extra = pick_encoder()
    seek = (["-ss", "%.3f" % start_s] if start_s is not None else [])
    seg = (["-t", "%.3f" % seg_s] if seg_s is not None else [])
    if seg_s is not None:
        duration = seg_s

    def run(name, ex):
        if name == "libx264":                             # cap CPU threads/job
            ex = [str(threads) if a == "0" else a for a in ex]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
               "-y", "-fflags", "+genpts+discardcorrupt"] + seek + \
              ["-i", src] + seg + \
              ["-map", "0:v:0", "-c:v", name,
               "-b:v", "%dk" % int(t * 1000),
               "-maxrate", "8000k", "-bufsize", "8000k"] + ex + \
              ["-pix_fmt", "yuv420p", "-map", "0:a?", "-c:a", "copy",
               "-movflags", "+faststart", "-progress", "pipe:1", dst]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True,
                                bufsize=1)
        last = -100
        for line in proc.stdout:                          # ffmpeg -progress
            if line.startswith("out_time_us=") and duration:
                us = line.strip().split("=", 1)[1]
                if us.isdigit():
                    pct = min(100.0, 100.0 * (int(us) / 1e6) / duration)
                    if pct - last >= 20:                  # throttle
                        last = pct
                        _say("     %-30s re-encoding %3d%% (%s)"
                             % (tag, int(pct), name.replace("h264_", "")))
        proc.wait()
        return proc.returncode
    if run(enc, extra) == 0 and os.path.getsize(dst) > 0:
        return True
    if enc != "libx264":                                  # hw failed -> cpu
        _ENC["e"] = _ENCODERS[-1]
        lib, lex = _ENCODERS[-1]
        _say("     %-30s hardware encoder failed; retrying on CPU" % tag)
        return run(lib, lex) == 0 and os.path.getsize(dst) > 0
    return False


# --------------------------------------------------------------------------- #
# IMU accelerometer scale fix
# --------------------------------------------------------------------------- #
def _imu_meta(data):
    """(sample_size, sample_count, nominal_rate) or (None, 0, 400)."""
    if len(data) < IMU_HEADER or data[:8] != b"TRIMU001":
        return None, 0, 400
    ver = struct.unpack_from("<I", data, 8)[0]
    ss = IMU_SAMPLE.get(ver)
    rate = struct.unpack_from("<I", data, 12)[0] or 400
    if not ss:
        return None, 0, rate
    return ss, (len(data) - IMU_HEADER) // ss, rate


def imu_gravity_and_rate(data):
    """(still-window median |accel|, n_windows, nominal_rate, sample_size)."""
    ss, n, rate = _imu_meta(data)
    if not ss or n < 10:
        return None, 0, rate, ss
    body = memoryview(data)[IMU_HEADER:IMU_HEADER + n * ss]
    # iter_unpack is C-level; one 80-byte record -> accel[8:20], gyro[20:32].
    reader = struct.Struct("<8x3f3f%dx" % (ss - 32))
    recs = list(reader.iter_unpack(body))
    amag = [math.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) for r in recs]
    gmag = [math.sqrt(r[3]*r[3] + r[4]*r[4] + r[5]*r[5]) for r in recs]
    win = max(4, int(rate * 0.25))
    still = []
    for i in range(0, n - win, win):
        if sum(gmag[i:i + win]) / win < 0.08:
            m = sum(amag[i:i + win]) / win
            if sum((x - m) ** 2 for x in amag[i:i + win]) / win < 0.5:
                still.append(m)
    if not still:
        return None, 0, rate, ss
    still.sort()
    return still[len(still) // 2], len(still), rate, ss




def rescale_accel(data, factor):
    """Divide every accel sample by `factor` (a full-scale-range correction)."""
    ss, n, _rate = _imu_meta(data)
    if not ss:
        return data
    buf = bytearray(data)
    inv = 1.0 / factor
    body = memoryview(buf)[IMU_HEADER:IMU_HEADER + n * ss]
    acc = struct.Struct("<3f")                            # accel at record[8:20]
    for i in range(n):
        p = i * ss + 8
        ax, ay, az = acc.unpack_from(body, p)
        acc.pack_into(body, p, ax * inv, ay * inv, az * inv)
    return bytes(buf)


# --------------------------------------------------------------------------- #
# splitting an over-length clip into spec-length chunks (time-synced)
# --------------------------------------------------------------------------- #
VTS_HEADER = 32
VTS_ENTRY = {1: 12, 2: 24, 3: 24, 4: 36}
TEL_HEADER = 32
TEL_RECORD = 24


def _vts_frames(data):
    """(entry_size, [sof_ns per frame]) from a .vts, or (None, [])."""
    if len(data) < VTS_HEADER or data[:8] != b"TRIVTS01":
        return None, []
    es = VTS_ENTRY.get(struct.unpack_from("<I", data, 8)[0])
    if not es:
        return None, []
    n = (len(data) - VTS_HEADER) // es
    sof = [struct.unpack_from("<Q", data, VTS_HEADER + i * es + 4)[0]
           for i in range(n)]
    return es, sof


def _slice_vts(data, f0, f1):
    """.vts covering frames [f0,f1) with frame_number rebased to 0-based."""
    es = VTS_ENTRY[struct.unpack_from("<I", data, 8)[0]]
    out = bytearray(data[:VTS_HEADER])
    for j in range(f0, f1):
        e = bytearray(data[VTS_HEADER + j * es: VTS_HEADER + (j + 1) * es])
        struct.pack_into("<I", e, 0, j - f0)          # rebase frame_number
        out += e
    return bytes(out)


def _slice_imu(data, ts_lo, ts_hi):
    """.imu samples with timestamp in [ts_lo, ts_hi); header video_start updated."""
    ss, n, _rate = _imu_meta(data)
    if not ss:
        return data
    out = bytearray(data[:IMU_HEADER])
    first_ts = None
    for i in range(n):
        p = IMU_HEADER + i * ss
        ts = struct.unpack_from("<Q", data, p)[0]
        if ts_lo <= ts < ts_hi:
            if first_ts is None:
                first_ts = ts
            out += data[p:p + ss]
    if first_ts is not None:
        struct.pack_into("<Q", out, 20, first_ts)     # start_time_ns
        struct.pack_into("<Q", out, 28, ts_lo)        # video_start_ns
    return bytes(out)


def _slice_tel(data, ts_lo, ts_hi):
    """.tel records within the time window (best-effort; passthrough on error)."""
    if len(data) < TEL_HEADER or not data[:7] == b"TRTEL01"[:7]:
        return data
    n = (len(data) - TEL_HEADER) // TEL_RECORD
    out = bytearray(data[:TEL_HEADER])
    kept = 0
    for i in range(n):
        p = TEL_HEADER + i * TEL_RECORD
        ts = struct.unpack_from("<Q", data, p)[0]
        if ts_lo <= ts < ts_hi:
            out += data[p:p + TEL_RECORD]
            kept += 1
    struct.pack_into("<I", out, 16, kept)             # record_count
    return bytes(out)


def split_and_repair(z, names, info, meta, e, country, session, args, tag,
                     out_path, tmpdir):
    """Split an over-length clip into ~30-min chunks, each a standalone ZIP
    with time-synced video + IMU + VTS. Returns a list of chunk basenames."""
    mp4 = next((n for n in names if n.lower().endswith(".mp4")), None)
    imu = next((n for n in names if n.lower().endswith(".imu")), None)
    vts = next((n for n in names if n.lower().endswith(".vts")), None)
    tel = next((n for n in names if n.lower().endswith(".tel")), None)
    jsn = next((n for n in names if n.lower().endswith(".json")
                and n != "metadata.json"), None)
    readme = "README.md" if "README.md" in names else None

    idata = z.read(imu) if imu else b""
    vdata = z.read(vts) if vts else b""
    tdata = z.read(tel) if tel else b""
    jdata = z.read(jsn) if jsn else None
    rdata = z.read(readme) if readme else None
    es, sof = _vts_frames(vdata)
    total_frames = len(sof)
    dur = meta.get("duration_s") or 0
    fps = (total_frames / dur) if dur else 30.0

    target = max(120.0, args.split_minutes * 60.0)
    n = max(2, math.ceil(dur / target))               # equal chunks, each <=target
    _say("     %-30s splitting %.0fs into %d chunks (~%.0f min each)"
         % (tag, dur, n, dur / n / 60.0))

    # Extract the full video once; each chunk is re-encoded from it.
    src = os.path.join(tmpdir, "full.mp4")
    _say("     %-30s extracting video (%.1f GB)..."
         % (tag, info[mp4].file_size / 1e9))
    with z.open(mp4) as s, open(src, "wb") as o:
        shutil.copyfileobj(s, o, 1 << 20)
    prep_video(src, tag)                              # flatten so ffmpeg can read

    base_zip = out_path[:-4] if out_path.lower().endswith(".zip") else out_path
    orig_base = meta.get("clip_id") or os.path.basename(base_zip)
    chunk_names = []
    for k in range(n):
        f0 = round(k * total_frames / n)
        f1 = round((k + 1) * total_frames / n) if k < n - 1 else total_frames
        if f1 <= f0:
            continue
        t0 = f0 / fps
        seg = (f1 - f0) / fps
        ts_lo = sof[f0] if sof and f0 < len(sof) else 0
        ts_hi = (sof[f1] if sof and f1 < len(sof)
                 else (sof[-1] + int(1e9 / fps) if sof else (1 << 63)))
        cbase = "%s_chunk%02d" % (orig_base, k + 1)
        _say("     %-30s chunk %d/%d  frames %d-%d  %.0f-%.0fs"
             % (tag, k + 1, n, f0, f1, t0, t0 + seg))
        cdst = os.path.join(tmpdir, "chunk%02d.mp4" % (k + 1))
        ok = ff_reencode(src, cdst, args.reencode_mbps,
                         getattr(args, "enc_threads", 0), tag="%s#%d" % (tag, k + 1),
                         start_s=t0, seg_s=seg)
        if not ok:
            _say("     %-30s chunk %d re-encode FAILED" % (tag, k + 1))
            continue
        # sliced, time-synced sidecars
        cimu = _slice_imu(idata, ts_lo, ts_hi) if idata else None
        cvts = _slice_vts(vdata, f0, f1) if vdata else None
        ctel = _slice_tel(tdata, ts_lo, ts_hi) if tdata else None
        cmeta = json.loads(json.dumps(meta))         # deep copy
        cmeta["clip_id"] = cbase
        cmeta["duration_s"] = round(seg, 3)
        cmeta["chunk"] = {"index": k + 1, "of": n, "source_clip": orig_base,
                          "note": "one part of a longer recording, split to the "
                                  "120-3600 s clip-length spec; IMU/VTS are the "
                                  "matching time slice."}
        _patch_metadata(cmeta, e, country, session, cdst, None, [])
        czip = base_zip.replace(orig_base, cbase) if orig_base in base_zip \
            else base_zip + ("_chunk%02d" % (k + 1))
        czip = czip + ".zip"
        tmp = czip + ".part"
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zo:
            zo.write(cdst, cbase + ".mp4", compress_type=zipfile.ZIP_STORED)
            if cimu is not None:
                zo.writestr(cbase + ".imu", cimu, zipfile.ZIP_STORED)
            if cvts is not None:
                zo.writestr(cbase + ".vts", cvts, zipfile.ZIP_STORED)
            if ctel is not None:
                zo.writestr(cbase + ".tel", ctel, zipfile.ZIP_DEFLATED)
            if jdata is not None:
                zo.writestr(cbase + ".json", jdata, zipfile.ZIP_DEFLATED)
            zo.writestr("metadata.json", json.dumps(cmeta, indent=2) + "\n",
                        zipfile.ZIP_DEFLATED)
            if rdata is not None:
                zo.writestr("README.md", rdata, zipfile.ZIP_DEFLATED)
        os.replace(tmp, czip)
        os.remove(cdst)
        chunk_names.append(os.path.basename(czip))
    os.remove(src)
    return chunk_names


# --------------------------------------------------------------------------- #
# per-ZIP repair
# --------------------------------------------------------------------------- #
def repair_zip(zp, out_path, env, env_map, country, session, args):
    name = os.path.basename(zp)
    actions, remaining = [], []
    t_start = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix="trinet_repair_")
    try:
        z = zipfile.ZipFile(zp)
        try:
            names = z.namelist()
            info = {i.filename: i for i in z.infolist()}
            mp4 = next((n for n in names if n.lower().endswith(".mp4")), None)
            imu = next((n for n in names if n.lower().endswith(".imu")), None)
            meta = {}
            if "metadata.json" in names:
                meta = json.loads(z.read("metadata.json").decode("utf-8"))
            clip_id = meta.get("clip_id", name)
            tag = (clip_id or name)
            tag = tag[:-4] if tag.lower().endswith(".zip") else tag
            tag = tag[:30]

            e = env
            if env_map is not None:
                base = name[:-4]
                e = next((env_map[k] for k in (name, base, clip_id)
                          if k in env_map), None)
                if e is None:
                    remaining.append("no environment mapping")

            # ---- analyse (cheap): container, bitrate, gravity ----
            size = info[mp4].file_size if mp4 else 0
            dur = meta.get("duration_s")
            status = "no_mp4"
            br = None
            fragmented = False
            if mp4:
                status, _d, fragmented = container_status_zip(z, mp4, size)
                br = (size * 8 / dur / 1e6) if dur else None
            idata = z.read(imu) if (imu and not args.no_fix_imu) else None
            g = imu_gravity_and_rate(idata)[0] if idata else None
            if not args.dry_run:
                _say("* %-30s %-8s | %-9s | %-11s | gravity %s"
                     % (tag, ("%.0fs" % dur) if dur else "?s", status,
                        ("%.1f Mbps" % br) if br else "bitrate ?",
                        ("%.2f m/s^2" % g) if g is not None else "n/a"))

            # ---- over-length clip: split into spec-length chunks ----
            oversized = (mp4 and dur and status != "no_video"
                         and dur > SPEC["clip_seconds"][1] and args.split)
            if oversized and not have("ffmpeg"):
                remaining.append("clip %.0fs > 1h -- needs ffmpeg to split"
                                 % dur)
            elif oversized and args.dry_run:
                nch = max(2, math.ceil(dur / (args.split_minutes * 60.0)))
                actions.append("split into %d chunks of ~%.0f min"
                               % (nch, dur / nch / 60.0))
                return _result(name, actions, remaining, dry=True)
            elif oversized:
                chunks = split_and_repair(z, names, info, meta, e, country,
                                          session, args, tag, out_path, tmpdir)
                z.close()
                if out_path == zp and chunks:         # in place: replace orig
                    os.remove(zp)
                _say("OK  %-30s split into %d chunks: %s (%.0fs)"
                     % (tag, len(chunks), ", ".join(chunks),
                        time.monotonic() - t_start))
                return _result(name, ["split into %d chunks: %s"
                                      % (len(chunks), ", ".join(chunks))], [])

            # ---- video: extract + flatten only when we must touch it ----
            new_mp4 = None
            if mp4:
                ffm = bool(have("ffmpeg"))
                # A fragmented file needs its index rebuilt even without a
                # re-encode -- a bad index is the "trun track id unknown" reject.
                need_fix = (args.reencode or status not in ("ok", "no_video")
                            or fragmented)
                will_reencode = args.reencode and status != "no_video" and ffm
                if status == "no_video":
                    remaining.append("no usable video (cannot repair)")
                elif need_fix and not ffm:
                    remaining.append("%s video (ffmpeg not installed)" % status)
                elif need_fix and args.dry_run:
                    actions.append("re-encode video to <=8 Mbps"
                                   if args.reencode else "rebuild/remux video")
                elif need_fix:
                    src = os.path.join(tmpdir, "in.mp4")
                    _say("     %-30s extracting video (%.1f GB)..."
                         % (tag, size / 1e9))
                    with z.open(mp4) as s, open(src, "wb") as o:
                        shutil.copyfileobj(s, o, 1 << 20)
                    prep_video(src, tag)             # flatten so ffmpeg can read
                    dst = os.path.join(tmpdir, "out.mp4")
                    if args.reencode and ff_reencode(
                            src, dst, args.reencode_mbps,
                            getattr(args, "enc_threads", 0),
                            duration=dur, tag=tag):
                        new_mp4 = dst
                        actions.append("re-encoded video to <=8 Mbps")
                    elif not args.reencode:
                        _say("     %-30s remuxing video..." % tag)
                        if ff_remux(src, dst):
                            new_mp4 = dst
                            actions.append("rebuilt/remuxed video (lossless)")
                        else:
                            remaining.append("video fix failed (%s)" % status)
                    else:
                        remaining.append("video fix failed (%s)" % status)
                    os.remove(src)
                if not will_reencode and br and br > SPEC["bitrate_mbps"][1] + 0.05:
                    remaining.append("bitrate %.1f Mbps > 8 (use re-encode)" % br)
                if dur is not None and not (SPEC["clip_seconds"][0] <= dur
                                            <= SPEC["clip_seconds"][1]):
                    remaining.append("clip length %.0f s out of 120-3600" % dur)

            # ---- imu accel scale (reuse idata/g from analysis) ----
            new_imu = None
            imu_fix = None
            if idata is not None and g is not None:
                ratio = g / 9.81
                gross = ratio >= 1.5 or ratio <= 0.67       # clear FSR fault
                if gross and args.dry_run:
                    actions.append("correct accel scale (gravity %.1f->9.8)" % g)
                elif gross:
                    # Rescale by the measured factor so at-rest |accel| reads
                    # gravity. (The fault is ~2x/0.5x but not exactly a power of
                    # two, so use the measured factor.) Gyro is a separate range
                    # and unaffected.
                    fac = ratio
                    _say("     %-30s correcting accel scale (gravity %.1f)"
                         % (tag, g))
                    new_imu = rescale_accel(idata, fac)
                    g2 = imu_gravity_and_rate(new_imu)[0]
                    imu_fix = {"before": round(g, 3), "factor": fac,
                               "after": round(g2, 3) if g2 else None}
                    actions.append(
                        "corrected accel scale x%.3f (gravity %.1f->%.1f)"
                        % (1.0 / fac, g, g2 or 0))
                elif not (SPEC["gravity_ms2"][0] <= g <= SPEC["gravity_ms2"][1]):
                    remaining.append(
                        "gravity %.2f mildly off-spec -- review, not "
                        "auto-corrected (could be motion, not scale)" % g)

            # ---- stale MCAP: it embeds its own copy of video+imu+metadata ----
            mcap = next((n for n in names if n.lower().endswith(".mcap")), None)
            drop_mcap = bool(mcap and (new_mp4 or new_imu))
            if drop_mcap:
                actions.append("removed stale .mcap (regenerate from sidecars "
                               "if needed)")

            meta_changed = _patch_metadata(meta, e, country, session,
                                           new_mp4, imu_fix, actions)

            if args.dry_run:
                return _result(name, actions, remaining, dry=True)
            need_write = bool(new_mp4 or new_imu or meta_changed or drop_mcap)
            if not need_write:
                _say("= %-30s already compliant (%.0fs)"
                     % (tag, time.monotonic() - t_start))
                return _result(name, actions or ["already compliant"],
                               remaining)

            # ---- rewrite: stream unchanged members, swap fixed ones ----
            _say("     %-30s writing repaired ZIP..." % tag)
            tmp_zip = out_path + ".part"
            with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zo:
                for n in names:
                    if n == mcap and drop_mcap:
                        continue                         # stale; leave it out
                    if n == mp4 and new_mp4:
                        zo.write(new_mp4, mp4, compress_type=zipfile.ZIP_STORED)
                    elif n == mp4:                       # unchanged: stream copy
                        zi = zipfile.ZipInfo(mp4)
                        zi.compress_type = zipfile.ZIP_STORED
                        with z.open(mp4) as sfp, \
                                zo.open(zi, "w", force_zip64=True) as dfp:
                            shutil.copyfileobj(sfp, dfp, 1 << 20)
                    elif n == imu and new_imu is not None:
                        zo.writestr(info[imu], new_imu,
                                    compress_type=zipfile.ZIP_STORED)
                    elif n == imu:
                        with z.open(imu) as sfp, \
                                zo.open(info[imu], "w", force_zip64=True) as dfp:
                            shutil.copyfileobj(sfp, dfp, 1 << 20)
                    elif n == "metadata.json":
                        zo.writestr("metadata.json",
                                    json.dumps(meta, indent=2) + "\n",
                                    zipfile.ZIP_DEFLATED)
                    else:
                        zo.writestr(info[n], z.read(n),
                                    compress_type=info[n].compress_type)
        finally:
            z.close()
        os.replace(tmp_zip, out_path)
        mark = "OK " if not remaining else "!! "
        _say("%s %-30s %s%s (%.0fs)"
             % (mark, tag, "; ".join(actions) or "no change",
                ("  [needs: " + "; ".join(remaining) + "]") if remaining else "",
                time.monotonic() - t_start))
        return _result(name, actions, remaining)
    except Exception as ex:                               # pragma: no cover
        _say("!! %-30s ERROR: %s" % (tag if 'tag' in dir() else name, ex))
        return _result(name, actions, remaining + ["ERROR: %s" % ex],
                       error=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _patch_metadata(meta, env, country, session, mp4_path, imu_fix, actions):
    changed = False
    if env:
        t, s = env
        if meta.get("environment_type") != t:
            meta["environment_type"] = t
            actions.append("set environment_type=%s/%s" % (t, s))
            changed = True
        meta["environment_subcategory"] = s
        ev = meta.get("environment")
        if not isinstance(ev, dict):
            ev = {}
        ev["type"], ev["subcategory"] = t, s
        meta["environment"] = ev
    if country:
        loc = meta.get("location") if isinstance(meta.get("location"), dict) else {}
        loc["country"] = country
        meta["location"] = loc
        changed = True
    if session:
        meta["session_id"] = session
        changed = True
    # refresh video numbers from the repaired mp4
    if mp4_path and os.path.isfile(mp4_path):
        br = _probe_bitrate(mp4_path)
        v = meta.get("video") if isinstance(meta.get("video"), dict) else {}
        if br:
            v["bitrate_mbps"] = round(br, 2)
        v["file_bytes"] = os.path.getsize(mp4_path)
        meta["video"] = v
        changed = True
    if imu_fix:
        i = meta.get("imu") if isinstance(meta.get("imu"), dict) else {}
        acc = i.get("accelerometer") if isinstance(i.get("accelerometer"),
                                                   dict) else {}
        acc["gravity_still_ms2"] = imu_fix["after"]
        acc["gravity_in_spec"] = bool(imu_fix["after"] and
                                      9.0 <= imu_fix["after"] <= 10.6)
        acc["scale_corrected"] = {
            "applied_factor": 1.0 / imu_fix["factor"],
            "gravity_before_ms2": imu_fix["before"],
            "note": ("accelerometer was recorded at the wrong full-scale "
                     "range; samples rescaled so at-rest gravity ~= 9.81. "
                     "gyroscope unaffected."),
        }
        i["accelerometer"] = acc
        meta["imu"] = i
        changed = True
    if actions:
        meta.setdefault("repair", {})["actions"] = list(actions)
        changed = True
    return changed


def _probe_bitrate(path):
    if not have("ffprobe"):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=bit_rate", "-of", "default=nk=1:nw=1",
             path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=60)
        val = r.stdout.decode().strip()
        return int(val) / 1e6 if val.isdigit() else None
    except Exception:
        return None


def _result(name, actions, remaining, dry=False, error=False):
    return {"zip": name, "actions": actions, "remaining": remaining,
            "dry": dry, "error": error}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="repair_delivery.py",
        description="Repair delivered Trinet ZIPs in place to match the "
                    "delivery spec (metadata, video, IMU).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="A ZIP, or a folder of ZIPs.")
    p.add_argument("--environment", type=parse_environment, metavar="TYPE/SUB",
                   default=("residential", "other_household"),
                   help="Environment for every file, e.g. residential/laundry "
                        "(default: residential/other_household).")
    p.add_argument("--map", metavar="CSV",
                   help="CSV of zip-name-or-clip,environment for per-file "
                        "environments (overrides --environment).")
    p.add_argument("--country", metavar="CC", help="Set location.country.")
    p.add_argument("--session-id", metavar="ID", help="Set session_id.")
    enc = p.add_mutually_exclusive_group()
    enc.add_argument("--reencode", dest="reencode", action="store_true",
                     default=True,
                     help="(default) Transcode video to <=8 Mbps H.264 to fix "
                          "the bitrate. Needs ffmpeg.")
    enc.add_argument("--no-reencode", dest="reencode", action="store_false",
                     help="Skip transcoding; leave the video as recorded "
                          "(over-bitrate files are then flagged, not fixed).")
    p.add_argument("--reencode-mbps", type=float, default=7.0, metavar="N",
                   help="Target bitrate for the re-encode (default 7, cap 8).")
    p.add_argument("--no-fix-imu", action="store_true",
                   help="Do not correct off-scale accelerometer data.")
    p.add_argument("--split-minutes", type=float, default=30.0, metavar="M",
                   help="Target chunk length when splitting a clip longer than "
                        "1 hour (default 30). Each chunk becomes its own ZIP "
                        "with a time-synced IMU/VTS slice.")
    p.add_argument("--no-split", dest="split", action="store_false",
                   default=True,
                   help="Do not split over-length (>1 h) clips into chunks; "
                        "flag them instead.")
    p.add_argument("--out", metavar="DIR",
                   help="Write repaired copies here instead of in place.")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--jobs", "-j", type=int, default=0, metavar="N",
                   help="Repair this many ZIPs in parallel. Default 0 = one "
                        "per CPU core (uses the whole machine).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan; change nothing.")
    return p


def find_zips(root, recursive):
    if os.path.isfile(root):
        return [root] if root.lower().endswith(".zip") else []
    hits = []
    for dp, _d, files in os.walk(root):
        hits += [os.path.join(dp, n) for n in sorted(files)
                 if n.lower().endswith(".zip")]
        if not recursive:
            break
    return hits


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _worker(zp):
    cfg = _CFG
    out = (os.path.join(cfg["out"], os.path.basename(zp))
           if cfg["out"] else zp)
    return repair_zip(zp, out, cfg["env"], cfg["env_map"],
                      cfg["country"], cfg["session"], cfg["args"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.map:
        args.environment = None                # a CSV overrides the default
    if args.reencode and not have("ffmpeg"):
        # Re-encode is on by default; if ffmpeg is missing, don't fail the whole
        # run -- do the metadata/IMU/remux fixes and flag bitrate instead.
        print("note: ffmpeg not found -- skipping the bitrate re-encode; "
              "over-bitrate files will be flagged. Install ffmpeg (or pass "
              "--no-reencode) to silence this.\n")
        args.reencode = False
    env_map = load_env_map(args.map) if args.map else None

    zips = find_zips(args.path, args.recursive)
    if not zips:
        print("error: no .zip files found at %s" % args.path)
        return 2
    if args.out and not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    # Use every core: default one worker process per CPU. When re-encoding with
    # libx264 (CPU), divide the cores across the parallel jobs so ffmpeg fills
    # them without oversubscribing; a hardware encoder ignores this.
    cores = os.cpu_count() or 4
    jobs = args.jobs if args.jobs and args.jobs > 0 else cores
    jobs = max(1, min(jobs, len(zips)))
    args.enc_threads = max(1, cores // jobs)

    print("Repairing %d ZIP(s)%s%s using %d worker(s) on %d core(s)\n"
          % (len(zips), " [dry-run]" if args.dry_run else "",
             " with re-encode" if args.reencode else "", jobs, cores))

    cfg = {"env": args.environment, "env_map": env_map,
           "country": args.country, "session": args.session_id,
           "args": args, "out": args.out if not args.dry_run else None}

    if jobs > 1 and len(zips) > 1:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, initializer=_init_worker,
                initargs=(cfg,)) as ex:
            results = list(ex.map(_worker, zips))
    else:
        _init_worker(cfg)
        results = [_worker(z) for z in zips]

    fixed = clean = flagged = errs = 0
    for r in results:
        if r["error"]:
            errs += 1
        elif r["actions"] and r["actions"] != ["already compliant"]:
            fixed += 1
        else:
            clean += 1
        if r["remaining"]:
            flagged += 1
        # Real runs already streamed each file live; only the dry-run plan and
        # the leftover "needs" list are printed here.
        if args.dry_run:
            mark = "flag" if r["remaining"] else "ok  "
            print("[%s] %s" % (mark, r["zip"]))
            for a in r["actions"]:
                print("        + %s" % a)
            for rem in r["remaining"]:
                print("        ! %s" % rem)

    if not args.dry_run and flagged:
        print("\nStill needs attention (could not be auto-fixed):")
        for r in results:
            for rem in r["remaining"]:
                print("  %-40s %s" % (r["zip"][:40], rem))

    print("\n%d repaired, %d already-compliant, %d still-flagged, %d error(s)"
          % (fixed, clean, flagged, errs))
    if not args.dry_run:
        log_path = os.path.join(os.path.dirname(zips[0]) or ".",
                                "repair_log.json")
        try:
            json.dump(results, open(log_path, "w"), indent=1)
            print("log: %s" % log_path)
        except OSError:
            pass
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

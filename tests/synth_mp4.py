# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Synthetic Trinet camera recordings (MP4 + .vts/.imu/.tel/.json) for the cut tests.

The MP4s are laid out like the camera's flattened files (ftyp, ~1 s mdat pieces of
interleaved video + audio, the TMF payloads, moov at the end). Video and audio samples
are random bytes: nothing here decodes them, only their bytes and timing matter.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trinet_tools import mp4 as M
from trinet_tools import tmf_write as TW

NS = 1_000_000_000


@dataclass
class Cam:
    folder: Path
    base: str
    stereo: bool
    t_start_ns: int        # local clock of frame 0
    offset_ns: int         # local -> master
    n_frames: int


def vts_bytes(sof: np.ndarray, version: int = 5, offset_ns: int = 0, skew_ppb: int = 0, flags: int = 1,
              frame_offsets: np.ndarray | None = None, frame_numbers: np.ndarray | None = None) -> bytes:
    n = len(sof)
    hdr = struct.pack("<8sII", b"TRIVTS01", version, 30000)
    hdr += struct.pack("<qiHH", offset_ns, skew_ppb, 7, flags) if version >= 3 else bytes(16)
    es = TW.vts_entry_size(version)
    out = bytearray(hdr)
    fn = np.arange(n) if frame_numbers is None else frame_numbers
    for i in range(n):
        e = struct.pack("<IQ", int(fn[i]), int(sof[i]))
        if es >= 24:
            e += struct.pack("<IQ", i + 5, int(sof[i]) // 1000 + 777)
        if es >= 36:
            e += struct.pack("<III", 8000, 0x0F, 31800)
        if es >= 44:
            e += struct.pack("<q", int(frame_offsets[i]) if frame_offsets is not None else offset_ns)
        out += e
    return bytes(out)


def imu_bytes(t0: int, t1: int, version: int = 5, rate: int = 400, device_id: bytes = bytes(range(1, 17)),
              rng=None) -> bytes:
    rng = rng or np.random.default_rng(1)
    ts = np.arange(t0, t1, NS // rate, dtype=np.int64)
    hdr = struct.pack("<8sIIHHQQI", b"TRIMU001", version, rate, 2, 2, t0 - 3_000_000, 0, 6)
    hdr += device_id + bytes(64 - len(hdr) - 16)
    rows = np.zeros((len(ts), 80), np.uint8)
    rows[:, :8] = ts.astype("<u8").view(np.uint8).reshape(-1, 8)
    rows[:, 8:80] = rng.standard_normal((len(ts), 18)).astype("<f4").view(np.uint8).reshape(-1, 72)
    return hdr + rows.tobytes()


def tel_bytes(t0: int, t1: int, device_id: bytes = bytes(range(1, 9))) -> bytes:
    ts = np.arange(t0, t1, NS, dtype=np.int64)
    out = bytearray(struct.pack("<8sIII", b"TRTEL01\x00", 1, 32, len(ts)) + bytes(4) + device_id)
    for i, t in enumerate(ts):
        out += struct.pack("<QiIIHBB", int(t), 45000 + 10 * i, 1200000, 8000, 30, 0, 0)
    return bytes(out)


def _trak(tid, handler, timescale, stsd, durations, sizes, offsets, sync, name=b""):
    tkhd = M.fullbox(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, tid, 0, 0), bytes(16),
                     struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000),
                     struct.pack(">II", 1920 << 16, 1080 << 16) if handler == b"vide" else bytes(8))
    tkhd = M.patch_duration(tkhd, int(np.sum(durations)) * 1000 // timescale, 20, 28)
    mdhd = M.fullbox(b"mdhd", 0, 0, struct.pack(">IIIII", 0, 0, timescale, int(np.sum(durations)), 0x55C40000))
    hdlr = M.fullbox(b"hdlr", 0, 0, bytes(4), handler, bytes(12), name + b"\x00")
    mhd = {b"vide": M.fullbox(b"vmhd", 0, 1, bytes(8)), b"soun": M.fullbox(b"smhd", 0, 0, bytes(4))}.get(
        handler, M.fullbox(b"nmhd", 0, 0))
    dinf = M.box(b"dinf", M.fullbox(b"dref", 0, 0, struct.pack(">I", 1), M.fullbox(b"url ", 0, 1)))
    return M.box(b"trak", tkhd, M.box(b"mdia", mdhd, hdlr, M.box(b"minf", mhd, dinf, M.stbl(
        stsd, np.asarray(durations), np.asarray(sizes), np.asarray(offsets), sync, False))))


def avc1_stsd() -> bytes:
    entry = M.box(b"avc1", bytes(6), struct.pack(">H", 1), bytes(16), struct.pack(">HH", 1920, 1080),
                  struct.pack(">II", 0x480000, 0x480000), bytes(4), struct.pack(">H", 1), bytes(32),
                  struct.pack(">Hh", 24, -1))
    return M.fullbox(b"stsd", 0, 0, struct.pack(">I", 1), entry)


def mp4a_stsd(rate: int) -> bytes:
    entry = M.box(b"mp4a", bytes(6), struct.pack(">H", 1), bytes(8), struct.pack(">HHHH", 2, 16, 0, 0),
                  struct.pack(">I", rate << 16))
    return M.fullbox(b"stsd", 0, 0, struct.pack(">I", 1), entry)


def write_mp4(path: Path, n_frames: int, *, gop: int = 30, audio_rate: int = 48000, tmf=None, tmfm: bytes | None = None,
              ctts: bool = False, two_descriptions: bool = False, fragmented: bool = False, seed: int = 0,
              video_ts: int = 15360, frame_dur: int = 512) -> dict:
    """A camera-like MP4. Returns the sample tables written (for checks)."""
    rng = np.random.default_rng(seed)
    v_sizes = rng.integers(800, 3000, n_frames)
    v_sizes[::gop] += 20000
    v_sync = np.zeros(n_frames, bool)
    v_sync[::gop] = True
    v_dur = np.full(n_frames, frame_dur, np.int64)
    v_dur[7] = 2 * frame_dur                       # an encoder timestamp jump, as on the cameras
    v_secs = v_dur.sum() / video_ts
    n_audio = int(v_secs * audio_rate / 1024) + 3
    a_sizes = rng.integers(200, 400, n_audio)
    a_dur = np.full(n_audio, 1024, np.int64)
    a_dur[2] = 491                                 # an odd-length frame, as on the cameras
    v_t = np.concatenate(([0], np.cumsum(v_dur)[:-1])) / video_ts
    a_t = np.concatenate(([0], np.cumsum(a_dur)[:-1])) / audio_rate

    body = bytearray()
    ftyp = M.box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isomiso2avc1mp41")
    pos = len(ftyp)
    v_off = np.zeros(n_frames, np.int64)
    a_off = np.zeros(n_audio, np.int64)
    vi = ai = 0
    sec = 0
    while vi < n_frames or ai < n_audio:          # ~1 s pieces: [free][mdat: video..., audio...]
        sec += 1
        piece = bytearray()
        start = pos + 8 + 8                       # after the free box and the mdat header
        while vi < n_frames and v_t[vi] < sec:
            v_off[vi] = start + len(piece)
            piece += rng.integers(0, 255, int(v_sizes[vi]), dtype=np.uint8).tobytes()
            vi += 1
        while ai < n_audio and a_t[ai] < sec:
            a_off[ai] = start + len(piece)
            piece += rng.integers(0, 255, int(a_sizes[ai]), dtype=np.uint8).tobytes()
            ai += 1
        chunk = (M.box(b"moof", bytes(16)) if fragmented else M.box(b"free", bytes(0))) + M.box(b"mdat", bytes(piece))
        body += chunk
        pos += len(chunk)
    traks = []
    stsd_v = avc1_stsd()
    tv = _trak(1, b"vide", video_ts, stsd_v, v_dur, v_sizes, v_off, v_sync, b"VideoHandler")
    if ctts:
        ct = M.fullbox(b"ctts", 0, 0, struct.pack(">I", 2), struct.pack(">IIII", 1, 0, n_frames - 1, 512))
        i = tv.index(b"stsc") - 4
        tv = bytearray(tv[:i] + ct + tv[i:])
        _fix_sizes(tv, len(ct))
        tv = bytes(tv)
    if two_descriptions:
        tv = bytearray(tv)
        i = tv.index(b"stsc") - 4
        new = M.fullbox(b"stsc", 0, 0, struct.pack(">I", 2), struct.pack(">IIIIII", 1, 1, 1, 10, 1, 2))
        (old_len,) = struct.unpack_from(">I", tv, i)
        tv[i:i + old_len] = new
        _fix_sizes(tv, len(new) - old_len)
        tv = bytes(tv)
    traks.append(tv)
    traks.append(_trak(2, b"soun", audio_rate, mp4a_stsd(audio_rate), a_dur, a_sizes, a_off, None, b"SoundHandler"))
    tmf_off = []
    if tmf:
        tmf_data = b"".join(c.data for c in tmf)
        mdat = M.box(b"mdat", tmf_data)
        o = pos + 8
        for c in tmf:
            tmf_off.append(o)
            o += len(c.data)
        body += mdat
        pos += len(mdat)
        stsd = M.fullbox(b"stsd", 0, 0, struct.pack(">I", 1), M.box(b"tmfd", bytes(6), struct.pack(">H", 1)))
        traks.append(_trak(3, b"meta", 1000, stsd, [c.dur_ms for c in tmf], [len(c.data) for c in tmf], tmf_off,
                           None, b"Trinet TMF"))
    mvhd = M.fullbox(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, int(v_secs * 1000)),
                     struct.pack(">IH", 0x10000, 0x100), bytes(10),
                     struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000), bytes(24),
                     struct.pack(">I", 4))
    udta = M.box(b"udta", M.box(b"tmfm", tmfm)) if tmfm else b""
    moov = M.box(b"moov", mvhd, *traks, udta)
    path.write_bytes(ftyp + bytes(body) + moov)
    return dict(v_sizes=v_sizes, v_dur=v_dur, v_sync=v_sync, a_dur=a_dur, a_sizes=a_sizes, video_ts=video_ts,
                audio_rate=audio_rate)


def _fix_sizes(buf: bytearray, delta: int) -> None:
    """After growing a box inside a trak by ``delta``, grow trak/mdia/minf/stbl."""
    for kind in (b"trak", b"mdia", b"minf", b"stbl"):
        i = buf.index(kind) - 4
        (sz,) = struct.unpack_from(">I", buf, i)
        struct.pack_into(">I", buf, i, sz + delta)


def make_kit(root: Path, *, n_frames: int = 400, with_tmf: bool = True, seed: int = 0) -> dict:
    """A wrist kit: a stereo head (master, v5, stale first entries) + two mono wrists (v5 slaves
    with per-frame offsets). Returns {camera: Cam}."""
    rng = np.random.default_rng(seed)
    period = 33_660_000
    cams = {}
    # head: master, shared .imu/.tel, L/R vts (R has one extra frame at the start)
    head = root / "Stereo-Head"
    head.mkdir(parents=True)
    base = "grp123456_cae8dabe_take0001"
    t_head = 500 * NS
    sof = t_head + np.arange(n_frames) * period + rng.integers(-20_000, 20_000, n_frames)
    sof[:4] = sof[4]                                    # stale first entries, as on the cameras
    imu = imu_bytes(t_head - NS, t_head + n_frames * period + NS, rng=rng)
    tel = tel_bytes(t_head - 2 * NS, t_head + n_frames * period + NS)
    (head / f"{base}.imu").write_bytes(imu)
    (head / f"{base}.tel").write_bytes(tel)
    (head / f"calibration_cae8dabe_rig_20260921.json").write_text(json.dumps({"format": "test"}))
    for eye, s in (("L", sof), ("R", np.concatenate(([sof[0] - period], sof[:-1])))):
        vts = vts_bytes(s, 5, 0, 0, 3, frame_offsets=np.zeros(len(s), np.int64), frame_numbers=np.zeros(len(s)))
        (head / f"{base}_{eye}.vts").write_bytes(vts)
        meta = TW.build_meta_json(TW.MetaStrings(device_id="cae8dabe" + "0" * 24, eye=eye,
                                                 pair_file=f"{base}_{'R' if eye == 'L' else 'L'}.mp4", codec="h264"),
                                  imu, vts, tel)
        write_mp4(head / f"{base}_{eye}.mp4", len(s), audio_rate=44100, tmf=TW.encode(imu, vts, tel) if with_tmf else None,
                  tmfm=meta if with_tmf else None, seed=seed + (eye == "R"))
    cams["head"] = Cam(head, base, True, int(sof[0]), 0, n_frames)
    # wrists: slaves on their own clocks, keyframes NOT aligned with the head's
    for side, tag, local0, off, shift in (("Left", "23890f5f", 900 * NS, -400 * NS - 261_000_000, 11),
                                          ("Right", "990dd5dd", 300 * NS, 200 * NS - 209_000_000, 23)):
        folder = root / f"Mono-{side}"
        folder.mkdir()
        b = f"grp123456_{tag}_1"
        master0 = t_head + shift * period
        s = (master0 - off) + np.arange(n_frames) * period + rng.integers(-10_000, 10_000, n_frames)
        foff = off + np.cumsum(rng.integers(-300, 300, n_frames))     # the live offset drifts
        vts = vts_bytes(s, 5, off, 0, 1, frame_offsets=foff)
        im = imu_bytes(int(s[0]) - NS // 2, int(s[-1]) + NS // 2, device_id=bytes.fromhex(tag) + bytes(12), rng=rng)
        (folder / f"{b}.vts").write_bytes(vts)
        (folder / f"{b}.imu").write_bytes(im)
        (folder / f"{b}.json").write_text(json.dumps({"format": "trinet-recording-meta/1", "session": 123456,
                                                      "device_id": tag + "0" * 24, "device_tag": tag, "segment": 1,
                                                      "role": "slave", "master_clock_offset_ns": off}))
        meta = TW.build_meta_json(TW.MetaStrings(fw_version="0.5.4", hw_generation="v4", codec="h264"), im, vts)
        write_mp4(folder / f"{b}.mp4", n_frames, tmf=TW.encode(im, vts) if with_tmf else None,
                  tmfm=meta if with_tmf else None, seed=seed + shift)
        cams[side.lower()] = Cam(folder, b, False, int(s[0]), off, n_frames)
    return cams

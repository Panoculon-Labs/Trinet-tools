# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""The TMF writer is the camera's encoder (tmf_write.c) in Python.

Oracle: folding a real camera's WHOLE sidecars reproduces the ``tmfd`` payloads, their
durations and the ``tmfm`` JSON that the camera itself embedded, byte for byte (needs the
sample recordings: ``$TRINET_SAMPLES`` or ~/PanoculonLabs/Data/wrist_kit; skipped without).
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from tests import synth_mp4 as S
from trinet_tools import tmf_write as TW
from trinet_tools.mp4 import read_movie, read_samples
from trinet_tools.tmf import read_tmf

SAMPLES = Path(os.environ.get("TRINET_SAMPLES", Path.home() / "PanoculonLabs" / "Data" / "wrist_kit"))
CAMERA_MP4S = sorted(p for p in SAMPLES.glob("sample*/*/*.mp4")) if SAMPLES.exists() else []


def _sidecars(mp4: Path):
    stem = mp4.stem
    base = stem[:-2] if stem.endswith(("_L", "_R")) else stem
    tel = mp4.with_name(base + ".tel")
    return (mp4.with_name(base + ".imu").read_bytes(), mp4.with_name(stem + ".vts").read_bytes(),
            tel.read_bytes() if tel.exists() else None)


@pytest.mark.skipif(not CAMERA_MP4S, reason="sample recordings not available")
@pytest.mark.parametrize("mp4", CAMERA_MP4S, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_folding_a_cameras_sidecars_reproduces_its_own_tmf_byte_for_byte(mp4):
    movie = read_movie(mp4)
    track = movie.tmf_track()
    if track is None:
        pytest.skip(f"{mp4.name} carries no TMF (re-muxed file)")
    imu, vts, tel = _sidecars(mp4)
    chunks = TW.encode(imu, vts, tel)
    original = read_samples(mp4, track)
    assert len(chunks) == len(original)
    assert [c.data for c in chunks] == original                       # every DEVC payload
    assert [c.dur_ms for c in chunks] == [int(d) for d in track.durations]
    raw = movie.udta(b"tmfm")
    assert TW.build_meta_json(TW.meta_strings_from(json.loads(raw)), imu, vts, tel) == raw


def test_written_tmf_reads_back_as_the_same_sidecars(tmp_path):
    rng = np.random.default_rng(3)
    imu = S.imu_bytes(10 * S.NS, 14 * S.NS, rng=rng)
    sof = 10 * S.NS + 500_000_000 + np.arange(100) * 33_366_666
    vts = S.vts_bytes(sof, 5, -1234, 0, 1, frame_offsets=np.full(100, -1234))
    tel = S.tel_bytes(9 * S.NS, 15 * S.NS)
    chunks = TW.encode(imu, vts, tel)
    meta = TW.build_meta_json(TW.MetaStrings(device_id="ab" * 16, codec="h264"), imu, vts, tel)
    S.write_mp4(tmp_path / "x.mp4", 100, tmf=chunks, tmfm=meta)
    t = read_tmf(tmp_path / "x.mp4")
    assert t.imu_bytes() == imu and t.vts_bytes() == vts and t.tel_bytes() == tel
    assert t.chunk_count == len(chunks) == 6                      # tel starts 1 s early: t0 = 9 s .. 15 s
    assert t.meta["sync"] == {"offset_ns": -1234, "skew_ppb": 0, "quality_us": 7, "flags": 1}
    assert t.meta["drops"]["recorded"] == 100 and t.meta["drops"]["dropped"] == 0


def _cursor_reference(ts, walls):
    """The camera's loop: ``while (i < n && ts[i] < wall) i++`` per chunk."""
    out, i = [], 0
    for w in walls:
        while i < len(ts) and ts[i] < w:
            i += 1
        out.append(i)
    return out


def test_chunking_follows_the_cameras_cursor_even_for_out_of_order_times():
    rng = np.random.default_rng(5)
    for _ in range(50):
        ts = np.sort(rng.integers(0, 10_000, 200)).astype(np.uint64)
        k = rng.integers(0, 200, 10)
        ts[k] = rng.integers(0, 12_000, 10)                       # stale / jumping entries
        ts[:3] = 0
        walls = np.arange(1000, 12_001, 1000, dtype=np.uint64)
        assert list(TW._cursor_ends(ts, walls)) == _cursor_reference(list(ts), list(walls))


def test_drop_summary_counts_timing_gaps_and_sequence_jumps():
    sof = 1_000_000_000 + np.arange(60) * 33_000_000
    sof[30:] += 100_000_000                                      # ~3 frames missing after frame 29
    vts = bytearray(S.vts_bytes(sof, 4))
    struct.pack_into("<I", vts, 32 + 50 * 36 + 12, 50 + 5 + 1)      # encoder sequence skips 1 at frame 50
    for i in range(51, 60):
        struct.pack_into("<I", vts, 32 + i * 36 + 12, i + 5 + 1)
    d = TW.drops_summary(bytes(vts))
    assert d["gap_count"] == 2 and d["dropped"] == 3 + 1
    assert [g["after_frame"] for g in d["gaps"]] == [29, 49]

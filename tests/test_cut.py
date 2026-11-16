# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Cutting a recording to a segment that is itself a valid Trinet recording.

Synthetic kits cover the rules (always run); the sample recordings
(``$TRINET_SAMPLES`` or ~/PanoculonLabs/Data/wrist_kit) cover real camera files: the
first frame and the sound decode (PyAV), and the sound stays in step with the picture.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests import synth_mp4 as S
from trinet_tools import cut as C
from trinet_tools.mp4 import read_movie
from trinet_tools.reader import read_imu, read_vts
from trinet_tools.tmf import read_tmf

SAMPLES = Path(os.environ.get("TRINET_SAMPLES", Path.home() / "PanoculonLabs" / "Data" / "wrist_kit"))
HAVE_SAMPLES = (SAMPLES / "sample1").is_dir()
NS = S.NS


def _vts_of(mp4: Path) -> Path:
    return mp4.with_suffix(".vts")


def _master_times(mp4: Path) -> np.ndarray:
    return read_vts(str(_vts_of(mp4))).global_sof_ns().astype(np.int64)


def _sidecar(mp4: Path, suffix: str) -> Path:
    stem = mp4.stem
    base = stem[:-2] if stem.endswith(("_L", "_R")) else stem
    return mp4.with_name(base + suffix)


def check_cut_kit(src: Path, out: Path, t0: int, t1: int, report: dict, gop_s: float = 1.05) -> None:
    """Every rule a cut must keep, for every video of a cut recording."""
    seen = 0
    for cam in report["cameras"]:
        for v in cam["videos"]:
            if not v.get("frames"):
                continue
            seen += 1
            mp4 = out / cam["folder"] / v["file"]
            orig = src / cam["folder"] / v["file"]
            k0, k1 = v["first_frame"], v["last_frame"] + 1
            movie = read_movie(mp4)
            video = movie.track("vide")
            # frames == .vts entries; the first frame is a keyframe
            vts = read_vts(str(_vts_of(mp4)))
            assert video.n == vts.num_frames == k1 - k0
            assert video.sync is None or video.sync[0]
            # master-clock times unchanged
            g_cut, g_orig = _master_times(mp4), _master_times(orig)
            assert np.array_equal(g_cut, g_orig[k0:k1])
            # starts at this camera's last keyframe at or before t0, ends at its last frame before t1
            run = np.maximum.accumulate(g_orig)
            okeys = read_movie(orig).track("vide").keyframes()
            if t0 >= run[0]:
                assert run[k0] <= t0 and (t0 - g_orig[k0]) / NS <= gop_s
                later = okeys[okeys > k0]
                assert not len(later) or run[later[0]] > t0
            assert run[k1 - 1] < t1 and (k1 == len(run) or run[k1] >= t1)
            # the video bytes are the original's, frame for frame
            with open(orig, "rb") as fo, open(mp4, "rb") as fc:
                ov = read_movie(orig).track("vide")
                for i in (0, 1, (k1 - k0) // 2, k1 - k0 - 1):
                    fo.seek(int(ov.offsets[k0 + i]))
                    fc.seek(int(video.offsets[i]))
                    assert fo.read(int(ov.sizes[k0 + i])) == fc.read(int(video.sizes[i]))
            # the embedded metadata reads back as the cut sidecars, and describes the cut
            t = read_tmf(mp4)
            assert t.imu_bytes() == _sidecar(mp4, ".imu").read_bytes()
            assert t.vts_bytes() == _vts_of(mp4).read_bytes()
            tel = _sidecar(mp4, ".tel")
            assert t.tel_bytes() == (tel.read_bytes() if tel.exists() else None)
            assert t.meta["cut"] == {"source": v["file"], "first_frame": k0, "last_frame": k1 - 1,
                                     "t0_ns": t0, "t1_ns": t1}
            assert t.meta["drops"]["recorded"] == k1 - k0
            orig_meta = read_tmf(orig).meta or {}
            for key in ("device_id", "fw_version", "eye", "pair_file", "codec"):
                if key in orig_meta:
                    assert t.meta[key] == orig_meta[key]
            # .imu / .tel: 1 s around the kept frames, on the camera's own clock
            local = read_vts(str(_vts_of(mp4))).best_timestamps_ns.astype(np.int64)
            imu = read_imu(str(_sidecar(mp4, ".imu")))
            imu_o = read_imu(str(_sidecar(orig, ".imu")))
            ts, tso = imu.timestamps_ns.astype(np.int64), imu_o.timestamps_ns.astype(np.int64)
            lo, hi = local[local > 0].min() - NS, local.max() + NS
            assert ts[0] >= lo - NS and ts[-1] <= hi + NS
            assert imu.header.start_time_ns == ts[0]
            # every original sample inside the frames' own span (+-1 s) is kept, in order
            inside = tso[(tso >= local[local > 0].min() - NS) & (tso <= local.max() + NS)]
            assert set(inside.tolist()) <= set(ts.tolist())
            assert np.all(np.diff(ts) > 0)
    assert seen


# ----------------------------------------------------------------- synthetic
def test_cut_kit_keeps_every_rule(tmp_path):
    S.make_kit(tmp_path / "src")
    src = tmp_path / "src"
    head = next((src / "Stereo-Head").glob("*_L.mp4"))
    g = _master_times(head)
    t0, t1 = int(g[100]) + 10_000_000, int(g[250]) + 5_000_000
    report = C.cut_recording(src, t0, t1, tmp_path / "out")
    check_cut_kit(src, tmp_path / "out", t0, t1, report)
    names = sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file())
    # keyframes of different cameras are not aligned: each camera starts at its own
    starts = {v["file"]: v["first_frame"] for c in report["cameras"] for v in c["videos"]}
    assert len(set(starts.values())) > 1
    # .json sidecars and calibration files are copied unchanged
    for p in src.rglob("*.json"):
        assert (tmp_path / "out" / p.relative_to(src)).read_bytes() == p.read_bytes()
    # the shared head .imu covers both eyes; .tel keeps its record count right
    tel = (tmp_path / "out" / "Stereo-Head" / "grp123456_cae8dabe_take0001.tel").read_bytes()
    assert struct.unpack_from("<I", tel, 16)[0] == (len(tel) - 32) // 24 > 0


def test_cut_starts_and_ends_inside_the_recording(tmp_path):
    S.make_kit(tmp_path / "src", n_frames=200)
    src = tmp_path / "src"
    head = next((src / "Stereo-Head").glob("*_L.mp4"))
    g = _master_times(head)
    # from before the first frame to after the last: the whole recording
    report = C.cut_recording(src, int(g[0]) - 5 * NS, int(g[-1]) + 5 * NS, tmp_path / "all")
    for cam in report["cameras"]:
        for v in cam["videos"]:
            assert v["first_frame"] == 0 and v["frames"] == v["of_frames"]
    check_cut_kit(src, tmp_path / "all", int(g[0]) - 5 * NS, int(g[-1]) + 5 * NS, report)
    # a window where a camera recorded nothing: it is left out
    left = next((src / "Mono-Left").glob("*.mp4"))
    gl = _master_times(left)
    report = C.cut_recording(src, int(gl[-1]) + NS, int(gl[-1]) + 2 * NS, tmp_path / "late")
    left_rep = [c for c in report["cameras"] if c["folder"] == "Mono-Left"][0]
    assert left_rep["frames"] == 0 and not (tmp_path / "late" / "Mono-Left").exists()


def test_sound_is_kept_and_aligned_by_the_edit_list(tmp_path):
    S.make_kit(tmp_path / "src")
    src = tmp_path / "src"
    left = next((src / "Mono-Left").glob("*.mp4"))
    g = _master_times(left)
    report = C.cut_recording(src, int(g[77]), int(g[300]), tmp_path / "out")
    for cam in report["cameras"]:
        for v in cam["videos"]:
            orig, cut = read_movie(src / cam["folder"] / v["file"]), read_movie(tmp_path / "out" / cam["folder"] / v["file"])
            ov, oa, ca = orig.track("vide"), orig.track("soun"), cut.track("soun")
            k0 = v["first_frame"]
            pv0 = Fraction(int(ov.dts[k0]), ov.timescale)             # original time of the cut's first frame
            j0 = v["audio"]["first_sample"]
            assert ca.n == v["audio"]["samples"] and ca.sync is None
            assert len(ca.edits) == 1 and ca.edits[0][1] == v["audio"]["media_time"]
            # the audio presented at the cut's time 0 is the audio the original presented with frame k0
            at0 = Fraction(int(oa.dts[j0]) + ca.edits[0][1], oa.timescale)
            assert abs(at0 - pv0) <= Fraction(1, oa.timescale)
            assert oa.dts[j0] <= pv0 * oa.timescale                 # starts at or before (pre-roll)
            # and lasts as long as the video
            cv = cut.track("vide")
            a_end = Fraction(int(ca.durations.sum()) - ca.edits[0][1], oa.timescale)
            assert abs(a_end - Fraction(int(cv.durations.sum()), cv.timescale)) < Fraction(1, 20)


def test_a_source_without_tmf_gets_one_built_from_its_sidecars(tmp_path):
    S.make_kit(tmp_path / "src", with_tmf=False)
    src = tmp_path / "src"
    head = next((src / "Stereo-Head").glob("*_L.mp4"))
    g = _master_times(head)
    report = C.cut_recording(src, int(g[50]), int(g[150]), tmp_path / "out", hw_generation={"23890f5f": "v4"})
    check_cut_kit(src, tmp_path / "out", int(g[50]), int(g[150]), report)
    t = read_tmf(tmp_path / "out" / "Mono-Left" / "grp123456_23890f5f_1.mp4")
    assert t.meta["hw_generation"] == "v4" and t.meta["codec"] == "h264"
    assert t.meta["device_id"] == "23890f5f" + "0" * 24             # from the .imu header
    h = read_tmf(tmp_path / "out" / "Stereo-Head" / "grp123456_cae8dabe_take0001_R.mp4")
    assert h.meta["eye"] == "R" and h.meta["pair_file"] == "grp123456_cae8dabe_take0001_L.mp4"
    assert "hw_generation" not in h.meta


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("skew", [0, 37_123, -5_501])
def test_vts_cut_keeps_master_clock_times_for_every_version(tmp_path, version, skew):
    rng = np.random.default_rng(version * 7 + abs(skew))
    n = 500
    sof = 10**12 + np.arange(n) * 33_366_667 + rng.integers(-50_000, 50_000, n)
    foff = -261_000_000 + np.cumsum(rng.integers(-200, 200, n))
    raw = S.vts_bytes(sof, version, -261_000_000, skew if version in (3, 4) else 0, 1, frame_offsets=foff)
    (tmp_path / "o.vts").write_bytes(raw)
    g = read_vts(str(tmp_path / "o.vts")).global_sof_ns().astype(np.int64)
    for k0, k1 in ((0, 100), (90, 333), (300, 500), (499, 500)):
        (tmp_path / "c.vts").write_bytes(C.cut_vts(raw, k0, k1))
        c = read_vts(str(tmp_path / "c.vts"))
        gc = c.global_sof_ns().astype(np.int64)
        if skew == 0 or version not in (3, 4):
            assert np.array_equal(gc, g[k0:k1])                     # exact
        else:
            assert np.abs(gc - g[k0:k1]).max() <= 1                 # skew re-referenced: within 1 ns
        assert c.frame_numbers[0] == 0                              # frame numbers rebased (they counted frames)


def test_refuses_files_it_cannot_cut_with_a_plain_message(tmp_path):
    imu = S.imu_bytes(10 * NS, 25 * NS)
    sof = 10 * NS + np.arange(300) * 33_366_667
    for kind, kw, words in (("frag", dict(fragmented=True), "repair_recordings"),
                            ("bframes", dict(ctts=True), "B-frames"),
                            ("descs", dict(two_descriptions=True), "sample description")):
        d = tmp_path / kind
        d.mkdir()
        (d / "grp1_aabbccdd_1.vts").write_bytes(S.vts_bytes(sof, 5))
        (d / "grp1_aabbccdd_1.imu").write_bytes(imu)
        S.write_mp4(d / "grp1_aabbccdd_1.mp4", 300, **kw)
        with pytest.raises(C.CutError, match=words):
            C.cut_recording(d, int(sof[40]), int(sof[100]), tmp_path / f"out_{kind}")
    with pytest.raises(C.CutError, match="ends before it starts"):
        C.cut_recording(tmp_path / "frag", 5, 5, tmp_path / "x")


def test_large_files_switch_to_64_bit_offsets(tmp_path, monkeypatch):
    S.make_kit(tmp_path / "src", n_frames=120)
    src = tmp_path / "src"
    monkeypatch.setattr(C, "OFFSET_LIMIT", 10_000)             # as if the cut were past 4 GiB
    head = next((src / "Stereo-Head").glob("*_L.mp4"))
    g = _master_times(head)
    report = C.cut_recording(src, int(g[10]), int(g[100]), tmp_path / "out")
    assert all(v["co64"] for c in report["cameras"] for v in c["videos"])
    cut = tmp_path / "out" / "Stereo-Head" / head.name
    with open(cut, "rb") as f:
        assert f.read(64).count(b"moov") == 1                    # index first (fast start)
    data = cut.read_bytes()
    i = data.index(b"mdat")
    assert struct.unpack_from(">I", data, i - 4)[0] == 1          # 64-bit mdat size
    check_cut_kit(src, tmp_path / "out", int(g[10]), int(g[100]), report)


def test_cli_cuts_a_folder(tmp_path):
    S.make_kit(tmp_path / "src")
    script = Path(__file__).resolve().parent.parent / "scripts" / "cut_recording.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "src"), "--from", "2", "--to", "6",
                        "-o", str(tmp_path / "out"), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert {c["folder"] for c in rep["cameras"]} == {"Stereo-Head", "Mono-Left", "Mono-Right"}
    assert all(v["frames"] > 100 for c in rep["cameras"] for v in c["videos"])


# ----------------------------------------------------------------- real camera files
@pytest.fixture(scope="module")
def sample_cut(tmp_path_factory):
    if not HAVE_SAMPLES:
        pytest.skip("sample recordings not available")
    src = SAMPLES / "sample1"
    head = next((src / "Stereo-Head").glob("*_L.mp4"))
    g = _master_times(head)
    t0, t1 = int(g[10]) + int(12.4 * NS), int(g[10]) + 40 * NS
    out = tmp_path_factory.mktemp("sample_cut")
    report = C.cut_recording(src, t0, t1, out)
    return src, out, t0, t1, report


def test_sample_kit_cut_keeps_every_rule(sample_cut):
    src, out, t0, t1, report = sample_cut
    check_cut_kit(src, out, t0, t1, report)
    folders = {c["folder"] for c in report["cameras"]}
    assert folders == {"Stereo-Head", "Mono-Left", "Mono-Right"}
    # the re-muxed left wrist (no TMF in the source) now carries one
    assert read_tmf(out / "Mono-Left" / "grp280625_23890f5f_1.mp4").imu_sample_count > 0


def test_sample_cut_first_frame_and_sound_decode_like_the_original(sample_cut):
    av = pytest.importorskip("av")
    src, out, t0, t1, report = sample_cut
    for cam in report["cameras"]:
        for v in cam["videos"]:
            cut, orig = out / cam["folder"] / v["file"], src / cam["folder"] / v["file"]
            with av.open(str(cut)) as c:
                first = next(c.decode(c.streams.video[0]))
            with av.open(str(orig)) as c:
                for i, f in enumerate(c.decode(c.streams.video[0])):
                    if i == v["first_frame"]:
                        ref = f
                        break
            assert first.key_frame
            assert np.array_equal(first.to_ndarray(format="gray"), ref.to_ndarray(format="gray"))
            pcm_cut, rate, start_cut = _pcm(av, cut)
            pcm_orig, _, start_orig = _pcm(av, orig)
            assert abs(start_cut) < 1e-3                          # sound starts with the picture
            assert len(pcm_cut) / rate == pytest.approx(v["video_seconds"], abs=0.05)
            with av.open(str(orig)) as c:
                pv0 = next(float(p.pts * p.time_base) for i, p in enumerate(c.demux(c.streams.video[0]))
                           if i == v["first_frame"])
            lag_ms = _lag_ms(pcm_cut, pcm_orig, rate, at_cut=1.0, at_orig=pv0 + 1.0 - start_orig)
            assert abs(lag_ms) <= 1.0                             # in step to the millisecond


def _pcm(av, path):
    frames = {}
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        rate = s.rate
        for f in c.decode(s):
            a = f.to_ndarray().astype(np.float64)
            frames[float(f.pts * f.time_base)] = a.mean(0) if a.ndim == 2 and a.shape[0] < 8 else a.reshape(-1)
    lo = min(frames)
    end = max(t + len(a) / rate for t, a in frames.items())
    out = np.zeros(int(round((end - lo) * rate)) + 1)
    for t, a in frames.items():
        i = int(round((t - lo) * rate))
        out[i:i + len(a)] = a
    return out, rate, lo


def _lag_ms(cut, orig, rate, at_cut, at_orig):
    w = int(2 * rate)
    seg = cut[int(at_cut * rate):int(at_cut * rate) + w]
    oi = int(round(at_orig * rate))
    lags = range(-int(0.03 * rate), int(0.03 * rate))
    score = [float(np.dot(seg, orig[oi + k:oi + k + w])) /
             (np.linalg.norm(orig[oi + k:oi + k + w]) * np.linalg.norm(seg) + 1e-12) for k in lags]
    return list(lags)[int(np.argmax(score))] / rate * 1000


def test_toolkit_reader_reads_every_cut_file(sample_cut):
    _src, out, *_ = sample_cut
    for p in out.rglob("*"):
        if p.suffix == ".imu":
            assert read_imu(str(p)).num_samples > 0
        elif p.suffix == ".vts":
            assert read_vts(str(p)).num_frames > 0
        elif p.suffix == ".mp4":
            assert read_tmf(p).chunk_count > 0


# ----------------------------------------------------------------- other tools accept a cut
STUDIO = Path(os.environ.get("STUDIO_DIR", Path.home() / "PanoculonLabs" / "Annotation" / "Studio"))
HANDTRACKING = Path(os.environ.get("HANDTRACKING_DIR", Path.home() / "PanoculonLabs" / "Annotation" / "HandTracking"))
HANDSLAM_PY = Path(os.environ.get("HANDSLAM_PYTHON", Path.home() / ".venvs" / "handslam" / "bin" / "python"))
TOOLKIT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (STUDIO / "panostudio" / "runners" / "inspect_take.py").exists() or not HANDSLAM_PY.exists(),
                    reason="Trinet Studio checkout / its Python not available")
def test_studio_inspection_accepts_a_cut_kit_as_one_synced_calibrated_take(sample_cut, tmp_path):
    _src, out, *_ = sample_cut
    env = {**os.environ, "TRINET_TOOLS": str(TOOLKIT)}
    r = subprocess.run([str(HANDSLAM_PY), str(STUDIO / "panostudio" / "runners" / "inspect_take.py"), str(out),
                        "--qc-dir", str(tmp_path)], capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    takes = [json.loads(line) for line in r.stdout.splitlines() if line.strip()]
    assert len(takes) == 1
    t = takes[0]
    assert t["kit"]["session"] == "280625" and not t["missing"] and not t["kit"]["left_out"]
    assert len(t["cameras"]) == 3
    assert t["calibration"]["status"] == "ok"
    for cam in t["cameras"]:
        assert cam["sync_to"]
        for other, s in cam["sync_to"].items():
            assert s["p95_ms"] < 2 and s["matched_share"] > 0.95
    assert all(t["possible"].values())


@pytest.mark.skipif(not (HANDTRACKING / "handtrack" / "group.py").exists() or not HANDSLAM_PY.exists(),
                    reason="HandTracking checkout / its Python not available")
def test_handtracking_loads_a_cut_kit(sample_cut):
    _src, out, *_ = sample_cut
    code = ("import sys, json; sys.path.insert(0, sys.argv[2]); from handtrack.group import discover_group; "
            "g = discover_group(sys.argv[1]); print(json.dumps({'head': len(g.head_clock.t_ns), "
            "'wrists': {s: len(w.clock.t_ns) for s, w in g.wrists.items()}, 'session': g.session}))")
    env = {**os.environ, "TRINET_TOOLS": str(TOOLKIT)}
    r = subprocess.run([str(HANDSLAM_PY), "-c", code, str(out), str(HANDTRACKING)], capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["session"] == "280625" and d["head"] > 800 and set(d["wrists"]) == {"left", "right"}

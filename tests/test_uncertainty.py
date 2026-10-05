# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
import numpy as np

from tests import synth
from trinet_tools.wireless_log import load_logs
from trinet_tools.wireless_utc import (MatchConfig, build_runs, discover_takes, load_take,
                                       match_takes, take_to_utc)

MS = 1_000_000


def _rows(sc, **cfg):
    log = load_logs([sc["log"]])
    takes = [load_take(f) for f in discover_takes([sc["card"]])]
    m = match_takes(takes, build_runs(log), log, MatchConfig(**cfg))[0]
    return m, take_to_utc(m).rows[0]


def test_extrapolation_lowers_confidence(tmp_path):
    inside = _rows(synth.basic_scenario(tmp_path / "a", take_at_s=1800.0))[1]
    outside_m, outside = _rows(synth.basic_scenario(tmp_path / "b", take_at_s=3600.0 + 600.0))
    assert inside["confidence"] == "high" and inside["extrapolation_s"] == 0
    assert outside["confidence"] == "medium" and 600 <= outside["extrapolation_s"] <= 630   # gap + take length
    assert outside["rel_sigma_ms"] > inside["rel_sigma_ms"]
    assert outside["rel_sigma_ms"] >= 2e-6 * 600_000 * 0.95         # 2 ppm x 10 min
    m, _ = _rows(synth.basic_scenario(tmp_path / "c", take_at_s=3600.0 + 2400.0))
    assert (m.status, m.reason) == ("unmatched", "no_covering_run")  # beyond --max-extrapolation-s


def test_abs_sigma_follows_phone_source(tmp_path):
    _, sys_row = _rows(synth.basic_scenario(tmp_path / "a"))
    assert sys_row["phone_utc_source"] == "system" and sys_row["abs_sigma_ms"] >= 100.0
    sc = synth.basic_scenario(tmp_path / "b", phone=synth.SimPhone(seed=7, sys_err_ms=700.0,
                                                                  sntp_rtt_ms=20.0))
    _, row = _rows(sc)
    assert row["phone_utc_source"] == "sntp" and row["abs_sigma_ms"] < 25.0
    err = (row["utc_first_frame_ns"] - int(sc["take"].true_utc_ns[0])) / MS
    assert abs(err) < 5.0                            # the 700 ms system error is not inherited


def test_step_near_take_is_listed_and_penalised(tmp_path):
    sc = synth.basic_scenario(tmp_path, phone=synth.SimPhone(seed=7, steps=[(1250.0, 800.0)]))
    _, row = _rows(sc)
    assert len(row["phone_clock_steps"]) == 1 and abs(row["phone_clock_steps"][0] - 800) < 2
    assert row["abs_sigma_ms"] >= 800.0
    assert any("stepped" in n for n in row["notes"])


def test_latency_correction(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    _, a = _rows(sc)
    _, b = _rows(sc, latency_correction_ms=0.7)
    assert a["utc_first_frame_ns"] - b["utc_first_frame_ns"] == 700_000
    assert a["latency_bias_ms"] == [0.0, 1.5] and b["latency_bias_ms"] == [-0.7, 0.8]
    err = (b["utc_first_frame_ns"] - int(sc["take"].true_utc_ns[0])) / MS
    assert abs(err) < 0.5


def test_min_confidence(tmp_path):
    m, row = _rows(synth.basic_scenario(tmp_path, take_at_s=4200.0), min_confidence="high")
    assert m.confidence == "medium" and row["status"] == "below_min_confidence"
    assert row["utc_first_frame_ns"] is not None


def test_kit_offset_quality_and_stale_frames(tmp_path):
    from tests.test_matching import _kit, _run_match
    k = _kit(tmp_path)
    m = _run_match([k["log"]], [k["card"]])[k["ps"]]
    base = take_to_utc(m).rows[0]["rel_sigma_ms"]
    assert base >= 0.15                               # includes the 150 us kit-offset quality
    # Mark frames as carrying a stale (carried-forward) kit offset.
    import struct
    from trinet_tools.reader import VTS_ENTRY_SIZE_V5
    for p in (tmp_path / "cards" / "S").glob("*.vts"):
        b = bytearray(p.read_bytes())
        for i in range((len(b) - 32) // VTS_ENTRY_SIZE_V5):
            o = 32 + i * VTS_ENTRY_SIZE_V5 + 28
            (fl,) = struct.unpack_from("<I", b, o)
            struct.pack_into("<I", b, o, fl | 0x20)
        p.write_bytes(bytes(b))
    m = _run_match([k["log"]], [k["card"]])[k["ps"]]
    row = take_to_utc(m).rows[0]
    assert row["rel_sigma_ms"] >= 2.0 and any("carried-forward" in n for n in row["notes"])


def test_per_frame_matches_endpoints(tmp_path):
    sc = synth.basic_scenario(tmp_path, take_dur_s=90.0)
    log = load_logs([sc["log"]])
    takes = [load_take(f) for f in discover_takes([sc["card"]])]
    m = match_takes(takes, build_runs(log), log)[0]
    tu = take_to_utc(m, per_frame=True)
    fr = tu.frames["L"]
    row = tu.rows[0]
    assert abs(int(fr["utc_ns"][0]) - row["utc_first_frame_ns"]) < 5_000
    assert abs(int(fr["utc_ns"][-1]) - row["utc_last_frame_ns"]) < 5_000
    err = (fr["utc_ns"] - sc["take"].true_utc_ns) / MS
    assert np.all((err > 0) & (err < 1.5))
    assert np.all(np.diff(fr["utc_ns"]) > 0)

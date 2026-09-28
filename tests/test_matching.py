# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Matching tiers: boot id, advertised edge, time range, kit-mate."""


from tests import synth
from trinet_tools.wireless_log import load_logs
from trinet_tools.wireless_utc import (MatchConfig, build_runs, discover_takes, kit_consistency,
                                       load_take, match_takes, take_to_utc)

MS = 1_000_000


def _run_match(logs, dirs, **cfg):
    log = load_logs(logs)
    takes = [load_take(f, cfg.pop("unit", None)) for f in discover_takes(dirs)]
    res = match_takes(takes, build_runs(log), log, MatchConfig(**cfg))
    return {m.take.files.take_path: m for m in res}


def _first_utc_err_ms(m, take):
    row = take_to_utc(m).rows[0]
    return (row["utc_first_frame_ns"] - int(take.true_utc_ns[0])) / MS


def _two_boots(tmp_path, same_nonce=True, boot_ids=True, events=(True, True), take_ranges=True):
    """One unit, two camera boots heard by one phone; take 1 recorded in each."""
    phone = synth.SimPhone(seed=11)
    a = synth.SimCamera(boot_t=-600.0, boot_nonce=0x3C)
    b = synth.SimCamera(boot_t=6600.0, boot_nonce=0x3C if same_nonce else 0x51)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 10800)
    sa = lb.listen(a, 0, 3600, take_min=1 if take_ranges else None, take_max=1 if take_ranges else None)
    sb = lb.listen(b, 7200, 10800, take_min=1 if take_ranges else None, take_max=2 if take_ranges else None,
                   reset_reason="nonce")
    ta = synth.make_take(phone, a, 1, 1200.0, 10.0)
    tb = synth.make_take(phone, b, 1, 8437.0, 10.0)
    tb2 = synth.make_take(phone, b, 2, 9000.0, 10.0)
    if events[0]:
        lb.take_events(a, ta, sa, old_firmware=not boot_ids)
    if events[1]:
        lb.take_events(b, tb, sb, old_firmware=not boot_ids)
    pa = synth.write_stereo_take(tmp_path / "cardA", ta, a, boot_id="auto" if boot_ids else None)
    pb = synth.write_stereo_take(tmp_path / "cardB", tb, b, boot_id="auto" if boot_ids else None)
    pb2 = synth.write_stereo_take(tmp_path / "cardB", tb2, b, boot_id="auto" if boot_ids else None)
    log = lb.write_jsonl(tmp_path / "log.jsonl")
    return log, (ta, tb, tb2), (str(pa), str(pb), str(pb2))


def test_tier1_boot_id_exact(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    res = _run_match([sc["log"]], [sc["card"]])
    m = res[str(sc["take_path"])]
    assert (m.status, m.method, m.confidence) == ("resolved", "boot_id", "high")
    err = _first_utc_err_ms(m, sc["take"])
    assert 0.0 < err < 1.5                           # late by the radio latency only
    rows = take_to_utc(m).rows
    assert [r["eye"] for r in rows] == ["L", "R"] and rows[0]["timeline"] == "local"


def test_nonce_collision_resolved_by_edge_then_consensus(tmp_path):
    log, (ta, tb, tb2), (pa, pb, pb2) = _two_boots(tmp_path, events=(True, True))
    res = _run_match([log], [tmp_path / "cardA", tmp_path / "cardB"])
    for p, t in ((pa, ta), (pb, tb), (pb2, tb2)):
        m = res[p]
        assert m.status == "resolved" and m.method == "boot_id", (p, m.reason)
        assert abs(_first_utc_err_ms(m, t)) < 1.5
    assert any("edge" in n for n in res[pb].notes)
    assert any("other takes of the same boot" in n for n in res[pb2].notes)   # no edges of its own


def test_boot_not_observed_has_no_fallback(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    other = synth.SimCamera(boot_nonce=0x99)
    synth.write_stereo_take(tmp_path / "card2", sc["take"], other)   # same times, unseen boot
    res = _run_match([sc["log"]], [tmp_path / "card2"])
    m = next(iter(res.values()))
    assert m.status == "unmatched" and m.reason == "boot_not_observed" and m.chosen is None


def test_no_runs_for_unit(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    stranger = synth.SimCamera(unit_id="99887766")
    synth.write_stereo_take(tmp_path / "card2", sc["take"], stranger)
    m = next(iter(_run_match([sc["log"]], [tmp_path / "card2"]).values()))
    assert (m.status, m.reason) == ("unmatched", "no_runs_for_unit")


def test_tier2_take_number_reused_across_boots(tmp_path):
    log, (ta, tb, _), (pa, pb, _) = _two_boots(tmp_path, same_nonce=False, boot_ids=False)
    res = _run_match([log], [tmp_path / "cardA", tmp_path / "cardB"])
    for p, t in ((pa, ta), (pb, tb)):
        m = res[p]
        assert (m.status, m.method) == ("resolved", "edge"), (p, m.reason)
        assert abs(_first_utc_err_ms(m, t)) < 1.5


def test_tier2_old_firmware_raw_edge(tmp_path):
    # Older firmware advertised the raw start-of-frame, 6 ms off the .vts time
    # (exposure 8 ms, readout 20 ms, frame-centred): matched through the rebuilt shift.
    sc = synth.basic_scenario(tmp_path, boot_id=False, old_firmware=True)
    shift_ms = (sc["take"].raw_ns[0] - sc["take"].sof_ns[0]) / MS
    assert abs(shift_ms) > 5
    m = next(iter(_run_match([sc["log"]], [sc["card"]]).values()))
    assert (m.status, m.method) == ("resolved", "edge")


def test_tier3_range_unique_is_low(tmp_path):
    sc = synth.basic_scenario(tmp_path, boot_id=False, events=False)
    m = next(iter(_run_match([sc["log"]], [sc["card"]]).values()))
    assert (m.status, m.method, m.confidence) == ("resolved", "range", "low")
    assert abs(_first_utc_err_ms(m, sc["take"])) < 1.5


def test_tier3_ambiguous_then_pick(tmp_path):
    log, (ta, tb, _), (pa, pb, _) = _two_boots(tmp_path, same_nonce=False, boot_ids=False,
                                               events=(False, False), take_ranges=False)
    res = _run_match([log], [tmp_path / "cardB"])
    m = res[pb]
    assert m.status == "ambiguous" and len(m.candidates) == 2
    row = take_to_utc(m).rows[0]
    assert row["utc_first_frame_ns"] is None and any("candidates" in n for n in row["notes"])
    res = _run_match([log], [tmp_path / "cardB"], picks={pb: "store-a:2"})
    m = res[pb]
    assert (m.status, m.method, m.confidence) == ("resolved", "picked", "low")
    assert abs(_first_utc_err_ms(m, tb)) < 1.5


def _kit(tmp_path, slave_synced=True, hear_slave=True, third=False):
    phone = synth.SimPhone(seed=21)
    master = synth.SimCamera(unit_id="aaaa0001", boot_t=-900.0, boot_nonce=0x10, skew_ppm=12.0)
    slave = synth.SimCamera(unit_id="bbbb0002", boot_t=-300.0, boot_nonce=0x20, skew_ppm=-25.0)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 3600)
    sm = lb.listen(master, 0, 3600, group_id=0x8123, role="master")
    if hear_slave:
        ss = lb.listen(slave, 0, 3600, clock=master, timebase_is_master=True, group_id=0x8123,
                       role="slave")
    tm = synth.make_take(phone, master, 1, 1500.0, 10.0)
    ts = synth.make_take(phone, slave, 1, 1500.0, 10.0, master=master if slave_synced else None)
    lb.take_events(master, tm, sm)
    if hear_slave:
        lb.take_events(slave, ts, ss)
    card = tmp_path / "cards"
    pm = synth.write_stereo_take(card / "M", tm, master, prefix="grp4242_aaaa0001_", master=True)
    ps = synth.write_stereo_take(card / "S", ts, slave, prefix="grp4242_bbbb0002_")
    out = {"log": lb.write_jsonl(tmp_path / "kit.jsonl"), "card": card, "pm": str(pm), "ps": str(ps),
           "tm": tm, "ts": ts}
    if third:
        c = synth.SimCamera(unit_id="cccc0003", boot_t=-100.0, boot_nonce=0x30, skew_ppm=40.0)
        tc = synth.make_take(phone, c, 1, 1500.0, 10.0, master=master)
        out["pc"] = str(synth.write_stereo_take(card / "C", tc, c, prefix="grp4242_cccc0003_"))
        out["tc"] = tc
    return out


def test_kit_slave_uses_master_timeline(tmp_path):
    k = _kit(tmp_path)
    res = _run_match([k["log"]], [k["card"]])
    ms, mm = res[k["ps"]], res[k["pm"]]
    assert mm.status == ms.status == "resolved"
    assert ms.chosen.timeline == "kit_master" and mm.chosen.timeline == "local"
    assert abs(_first_utc_err_ms(ms, k["ts"])) < 1.5
    assert abs(_first_utc_err_ms(mm, k["tm"])) < 1.5
    kits = kit_consistency([take_to_utc(mm), take_to_utc(ms)])
    assert len(kits) == 1 and kits[0]["basis"] == "master_clock" and kits[0]["spread_ms"] < 1.0


def test_unsynced_slave_rejects_master_timebase_run(tmp_path):
    k = _kit(tmp_path, slave_synced=False)
    m = _run_match([k["log"]], [k["card"]])[k["ps"]]
    assert (m.status, m.reason) == ("unmatched", "timebase_inconsistent")


def test_tier4_kit_mate(tmp_path):
    k = _kit(tmp_path, third=True)
    res = _run_match([k["log"]], [k["card"]])
    m = res[k["pc"]]
    assert (m.status, m.method, m.confidence) == ("resolved", "kit_mate", "medium")
    assert abs(_first_utc_err_ms(m, k["tc"])) < 1.5


def test_mono_take_matches(tmp_path):
    phone = synth.SimPhone(seed=31)
    cam = synth.SimCamera(unit_id="d00d0001")
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 3600)
    seg = lb.listen(cam, 0, 3600)
    t = synth.make_take(phone, cam, 7, 1000.0, 6.0)
    lb.take_events(cam, t, seg)
    synth.write_mono_take(tmp_path / "Trinet", t, cam, "recording7_1", parts=2)
    m = next(iter(_run_match([lb.write_jsonl(tmp_path / "l.jsonl")], [tmp_path / "Trinet"]).values()))
    assert (m.status, m.method, m.confidence) == ("resolved", "boot_id", "high")
    assert abs(_first_utc_err_ms(m, t)) < 1.5


def test_camera_uptime_past_u32_wrap(tmp_path):
    # Camera up ~55 days: its .vts times exceed 2^32 ms; the phone logs the raw
    # (wrapped) camera time.
    phone = synth.SimPhone(seed=41)
    cam = synth.SimCamera(uptime0_ms=(1 << 32) + 5_000_000, boot_t=-100.0)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 3600)
    seg = lb.listen(cam, 0, 3600, log_wrap_k=1)
    t = synth.make_take(phone, cam, 1, 1800.0, 5.0)
    lb.take_events(cam, t, seg, log_wrap_k=1)
    synth.write_stereo_take(tmp_path / "c", t, cam)
    m = next(iter(_run_match([lb.write_jsonl(tmp_path / "l.jsonl")], [tmp_path / "c"]).values()))
    assert m.status == "resolved" and m.chosen.k == 1
    assert abs(_first_utc_err_ms(m, t)) < 1.5


def test_legacy_v1_log_uses_live_fit(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    v1 = sc["lb"].write_v1(tmp_path / "v1.json")
    m = next(iter(_run_match([v1], [sc["card"]]).values()))
    assert m.status == "resolved" and m.method == "boot_id"
    row = take_to_utc(m).rows[0]
    assert any("live fit" in n for n in row["notes"])
    assert row["confidence"] in ("medium", "low")        # far from the 2-minute live window
    err = (row["utc_first_frame_ns"] - int(sc["take"].true_utc_ns[0])) / MS
    assert abs(err) < 3 * row["rel_sigma_ms"] + 1.5

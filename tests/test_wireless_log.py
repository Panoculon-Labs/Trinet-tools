# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
import json

from tests import synth
from trinet_tools.wireless_log import load_log, load_logs


def _builder(seed=1):
    phone = synth.SimPhone(seed=seed)
    cam = synth.SimCamera()
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 600)
    seg = lb.listen(cam, 0, 600, take_min=1, take_max=2)
    lb.event(cam, "started", 1, 700_000, seg, 100)
    return lb


def test_v2_plain_and_gzip_equal(tmp_path):
    lb = _builder()
    a = load_logs([lb.write_jsonl(tmp_path / "a.jsonl")])
    b = load_logs([lb.write_jsonl(tmp_path / "a.jsonl.gz", gz=True)])
    for log in (a, b):
        assert not log.warnings
        assert len(log.segments) == 1 and len(log.events) == 1
        s = log.segments[0]
        assert s.unit_id == "a1b2c3d4" and s.boot_nonce == 0x3C
        assert len(s.device_ms) == 120 and s.device_ms.dtype.kind == "i"
        assert s.live_fit is not None and s.live_fit.buckets == 24
        assert len(log.clock_refs) == 11
    assert (a.segments[0].min_offset_ns == b.segments[0].min_offset_ns).all()


def test_truncated_file_warns_and_keeps_data(tmp_path):
    lb = _builder()
    log = load_logs([lb.write_jsonl(tmp_path / "t.jsonl", truncate=True)])
    assert any("truncated" in w for w in log.warnings)
    assert not log.files[0].complete
    assert len(log.segments) == 1


def test_unknown_types_and_keys_ignored(tmp_path):
    lb = _builder()
    p = lb.write_jsonl(tmp_path / "u.jsonl")
    lines = p.read_text().splitlines()
    lines.insert(2, json.dumps({"type": "future_thing", "x": 1}))
    seg = json.loads(next(l for l in lines if '"segment"' in l and '"buckets"' in l))
    seg["new_key"] = {"a": 1}
    lines = [l if not ('"buckets"' in l) else json.dumps(seg) for l in lines]
    p.write_text("\n".join(lines) + "\n")
    log = load_logs([p])
    assert len(log.segments) == 1
    assert any("end record declares" in w for w in log.warnings)   # count now differs by one


def test_dedup_across_exports_of_one_store(tmp_path):
    lb = _builder()
    early = lb.write_jsonl(tmp_path / "early.jsonl")
    # A later export of the same store: the segment has grown.
    cam = synth.SimCamera()
    lb2 = synth.LogBuilder(lb.phone, store_id=lb.store_id)
    lb2.clock_refs(0, 1200)
    lb2.listen(cam, 0, 1200)
    lb2.records += [r for r in lb.records if r["type"] == "event"]
    late = lb2.write_jsonl(tmp_path / "late.jsonl")
    log = load_logs([early, late])
    assert len(log.segments) == 1
    assert len(log.segments[0].device_ms) == 240        # the richer copy wins
    assert len(log.events) == 1
    assert len(log.clock_refs) == 21                    # union, duplicates dropped


def test_two_phones_kept_apart(tmp_path):
    a = _builder(1)
    b = synth.LogBuilder(synth.SimPhone(seed=2), store_id="store-b")
    b.clock_refs(0, 600)
    b.listen(synth.SimCamera(), 0, 600)
    log = load_logs([a.write_jsonl(tmp_path / "a.jsonl"), b.write_jsonl(tmp_path / "b.jsonl")])
    assert {s.store_id for s in log.segments} == {"store-a", "store-b"}
    assert set(log.phone_boots) == {("store-a", 1), ("store-b", 1)}


def test_legacy_v1(tmp_path):
    lb = _builder()
    log = load_log(lb.write_v1(tmp_path / "v1.json"))
    assert log.files[0].version == 1
    assert any("legacy v1" in w for w in log.warnings)
    s = log.segments[0]
    assert len(s.device_ms) == 0 and s.live_fit is not None
    assert log.events[0].kind == "started" and log.events[0].segment is None
    assert len(log.clock_refs) == 1


def test_unit_identity_keys(tmp_path):
    """Firmware 0.5.9+ exports the camera's board / generation / firmware per unit."""
    import json
    from trinet_tools.wireless_log import load_logs
    p = tmp_path / "id.jsonl"
    lines = [
        {"type": "header", "format": "trinet-wireless-log", "version": 2, "store_id": "s1",
         "bucket_ms": 5000},
        {"type": "unit", "unit_id": "a1b2c3d4", "group_id": 0, "last_role": "unpaired",
         "board": "pro_stereo_gs", "hw_generation": "v6", "fw_version": "0.5.9", "build": "shipping"},
        {"type": "unit", "unit_id": "0000beef", "group_id": 0, "last_role": "unpaired"},
        {"type": "end", "records": 3},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    log = load_logs([str(p)])
    u = log.units["a1b2c3d4"]
    assert (u.board, u.hw_generation, u.fw_version, u.build) == ("pro_stereo_gs", "v6", "0.5.9", "shipping")
    assert log.units["0000beef"].board is None
    assert not log.warnings

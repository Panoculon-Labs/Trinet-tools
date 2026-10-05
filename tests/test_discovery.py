# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.

import numpy as np

from tests import synth
from trinet_tools.tmf import read_tmf, read_tmf_meta
from trinet_tools.wireless_utc import discover_takes, load_take


def _take(n=1, t0=100.0, dur=2.0):
    return synth.make_take(synth.SimPhone(), synth.SimCamera(), n, t0, dur)


def test_stereo_and_kit_names(tmp_path):
    cam = synth.SimCamera()
    d = tmp_path / "CARD" / "Trinet" / "recording"
    synth.write_stereo_take(d, _take(3), cam)
    synth.write_stereo_take(d, _take(4), cam, prefix="grp72593_a1b2c3d4_")
    (d / "._take0003_L.mp4").write_bytes(b"junk")            # macOS resource fork: ignored
    takes = {t.name: t for t in discover_takes([tmp_path])}
    assert set(takes) == {"take0003", "grp72593_a1b2c3d4_take0004"}
    t3 = takes["take0003"]
    assert t3.layout == "stereo" and t3.take_number == 3 and t3.kit_session is None
    assert t3.eyes == ["L", "R"] and t3.imu.name == "take0003.imu"
    k = takes["grp72593_a1b2c3d4_take0004"]
    assert k.kit_session == 72593 and k.kit_dev8 == "a1b2c3d4" and k.take_number == 4


def test_mono_names(tmp_path):
    cam = synth.SimCamera()
    d = tmp_path / "Trinet"
    synth.write_mono_take(d, _take(), cam, "recording7_1")
    synth.write_mono_take(d, _take(), cam, "grp555_a1b2c3d4_3")
    synth.write_mono_take(d, _take(dur=3.0), cam, "recording8_2", parts=3)
    takes = {t.name: t for t in discover_takes([tmp_path])}
    assert set(takes) == {"recording7_1", "grp555_a1b2c3d4_3", "recording8_2"}
    assert takes["recording7_1"].take_number == 7 and takes["recording7_1"].segment == 1
    assert takes["grp555_a1b2c3d4_3"].take_number == 3          # kit mono advertises G
    assert takes["grp555_a1b2c3d4_3"].kit_session == 555
    parts = takes["recording8_2"]
    assert parts.layout == "mono_parts" and parts.take_number == 8 and len(parts.vts[""]) == 3
    info = load_take(parts)
    assert info.primary.frames == 90                             # parts concatenated
    assert np.all(np.diff(info.primary.sof_ns) > 0)


def test_unit_id_priority_and_conflict(tmp_path):
    cam = synth.SimCamera(unit_id="a1b2c3d4")
    d = tmp_path / "r"
    synth.write_stereo_take(d, _take(1), cam, prefix="grp1_0badf00d_")
    info = load_take(discover_takes([tmp_path])[0])
    assert info.unit_id == "a1b2c3d4" and info.unit_id_source == "metadata"
    assert any("conflict" in n and "0badf00d" in n for n in info.notes)
    # No metadata device id: the .imu header is next.
    synth.write_tmfm_mp4(d / "grp1_0badf00d_take0001_L.mp4", {"tmf_schema": 1})
    info = load_take(discover_takes([tmp_path])[0])
    assert (info.unit_id, info.unit_id_source) == ("a1b2c3d4", "imu")
    # Then the kit prefix, then --unit.
    for p in d.glob("*.imu"):
        p.unlink()
    info = load_take(discover_takes([tmp_path])[0])
    assert (info.unit_id, info.unit_id_source) == ("0badf00d", "kit_prefix")
    for p in d.glob("*"):
        p.rename(p.with_name(p.name.replace("grp1_0badf00d_", "")))
    info = load_take(discover_takes([tmp_path])[0], default_unit="feedf00d")
    assert (info.unit_id, info.unit_id_source) == ("feedf00d", "--unit")


def test_boot_id_and_embedded_vts_fallback(tmp_path):
    cam = synth.SimCamera()
    take = _take(2)
    d = tmp_path / "r"
    synth.write_stereo_take(d, take, cam, eyes=("L",), vts_sidecar=False)
    info = load_take(discover_takes([tmp_path])[0])
    assert info.error == "no_vts"                            # no sidecar, no embedded track
    assert info.boot_id == cam.boot_id and info.nonce == 0x3C and info.boot_id_source == "live"


def test_read_tmf_meta_matches_read_tmf_and_takes_last_moov(tmp_path):
    p = synth.write_tmfm_mp4(tmp_path / "x.mp4", {"boot_id": "ab" * 16, "device_id": "cd" * 16},
                             stale_meta={"boot_id": "00" * 16}, mdat_bytes=100_000)
    assert read_tmf_meta(p) == read_tmf(p).meta
    assert read_tmf_meta(p)["boot_id"] == "ab" * 16
    q = synth.write_tmfm_mp4(tmp_path / "none.mp4", None)
    assert read_tmf_meta(q) is None and read_tmf(q).meta is None


def test_read_tmf_meta_does_not_read_whole_file(tmp_path, monkeypatch):
    p = synth.write_tmfm_mp4(tmp_path / "big.mp4", {"boot_id": "ab" * 16}, mdat_bytes=5_000_000)
    import builtins
    real_open = builtins.open
    read_total = []

    class Spy:
        def __init__(self, f):
            self.f = f

        def read(self, n=-1):
            b = self.f.read(n)
            read_total.append(len(b))
            return b

        def __getattr__(self, k):
            return getattr(self.f, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    monkeypatch.setattr(builtins, "open", lambda *a, **k: Spy(real_open(*a, **k)))
    assert read_tmf_meta(p)["boot_id"] == "ab" * 16
    assert sum(read_total) < 10_000

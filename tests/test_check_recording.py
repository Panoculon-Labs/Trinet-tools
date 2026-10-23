# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""check_recording.py on synthetic takes: a solo recording with known drops,
and a three-camera kit take (stereo head + two wrists, one folder per camera)
with a known frame misalignment on one wrist."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.synth import write_vts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_recording.py"
KIT_PERIOD = 33_660_000        # ns, the kit rate (~29.7 fps)


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_solo_recording_counts_drops(tmp_path):
    n = 900
    sof = 10**9 + np.arange(n, dtype=np.int64) * 33_333_333
    keep = np.ones(n, bool)
    keep[[100, 500, 501, 502]] = False           # 4 frames lost in two gaps
    sof = sof[keep]
    sof[2] -= 2_000_000                          # jitter alone is not a drop
    write_vts(tmp_path / "recording1_1.vts", sof)
    report = tmp_path / "r.json"

    p = _run(tmp_path / "recording1_1.vts", "--json", report)
    assert p.returncode == 1, p.stdout + p.stderr
    take = json.loads(report.read_text())[0]
    s = take["cameras"][0]["streams"][0]
    assert not take["kit"]
    assert s["dropped"] == 4
    assert abs(s["fps"] - 895 / (899 / 30)) < 0.002   # 896 frames over 899 periods
    assert any("4 dropped" in m for _, m in take["checks"])


def test_kit_take_across_folders(tmp_path):
    n = 3000
    grid = 5 * 10**9 + np.arange(n, dtype=np.int64) * KIT_PERIOD   # master clock
    rng = np.random.default_rng(1)
    head, left, right = (tmp_path / d for d in ("head", "left", "right"))
    for d in (head, left, right):
        d.mkdir()

    for eye in "LR":
        write_vts(head / f"grp42_aaaaaaaa_take0001_{eye}.vts", grid, version=5,
                  fps_milli=30000, sync=(0, 0, 0, 3), frame_offset_ns=np.zeros(n))
    # left wrist: well aligned; right wrist: frames captured ~3 ms off the master's
    for d, tag, err_ms in ((left, "bbbbbbbb", 0.2), (right, "cccccccc", 3.0)):
        off = 250_000_000
        local = grid - off + (rng.normal(0, err_ms * 0.4, n) * 1e6).astype(np.int64)
        write_vts(d / f"grp42_{tag}_1.vts", local, version=5, sync=(off, 0, 20, 1),
                  frame_offset_ns=np.full(n, off))
        (d / f"grp42_{tag}_1.json").write_text(json.dumps(
            {"session": 42, "group": 777, "role": "slave", "device_tag": tag}))
    # a solo take on the head card must not join the kit take
    write_vts(head / "take0002_L.vts", grid[:300])
    report = tmp_path / "kit.json"

    p = _run(head, left, right, "--take", "grp42", "--json", report)
    out = p.stdout
    takes = json.loads(report.read_text())
    assert len(takes) == 1, out
    t = takes[0]
    assert t["kit"] and len(t["cameras"]) == 3
    cams = {c["name"].split("/")[0]: c for c in t["cameras"]}
    assert cams["head"]["role"] == "master" and len(cams["head"]["streams"]) == 2
    assert t["reference"].startswith("head/")
    assert cams["left"]["kit_group"] == 777
    assert abs(cams["left"]["streams"][0]["fps"] - 29.709) < 0.01
    assert cams["left"]["sync"]["p95_abs_ms"] < 1.0
    assert cams["right"]["sync"]["p95_abs_ms"] > 1.0
    assert cams["head"]["eyes"]["p95_abs_ms"] == 0
    # the kit rate is expected, not a warning; the misaligned wrist is
    assert not any("fps" in m for _, m in t["checks"]), out
    aligned = [m for _, m in t["checks"] if "alignment" in m]
    assert len(aligned) == 1 and aligned[0].startswith("right/"), out
    assert not any("dropped" in m for _, m in t["checks"]), out
    assert "29.7 fps by design" in out
    assert p.returncode == 1

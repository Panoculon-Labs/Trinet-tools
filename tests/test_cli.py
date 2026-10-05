# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tests import synth
from trinet_tools.wireless_utc import ROW_FIELDS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wireless_utc.py"


def _main():
    spec = importlib.util.spec_from_file_location("wireless_utc_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def test_end_to_end_subprocess(tmp_path):
    sc = synth.basic_scenario(tmp_path)
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(SCRIPT), str(sc["log"]), "--recordings", str(sc["card"]),
                        "-o", str(out), "--per-frame"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "resolved=2" in p.stdout and "high=2" in p.stdout
    doc = json.loads((out / "wireless_utc.json").read_text())
    assert doc["schema"] == "trinet-wireless-utc/1"
    assert doc["log_files"][0]["sha256"] and len(doc["takes"]) == 2
    assert set(doc["takes"][0]) == set(ROW_FIELDS)
    with open(out / "wireless_utc.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ROW_FIELDS and len(rows) == 2
    assert rows[0]["latency_bias_ms"] == "0.0..1.5"
    with open(out / "take0001_L.utc.csv") as f:
        frames = list(csv.DictReader(f))
    assert len(frames) == len(sc["take"].sof_ns)
    assert int(frames[0]["utc_ns"]) == doc["takes"][0]["utc_first_frame_ns"]


def test_sidecars_never_overwritten_without_force(tmp_path):
    main = _main()
    sc = synth.basic_scenario(tmp_path)
    args = [str(sc["log"]), "--recordings", str(sc["card"]), "-o", str(tmp_path / "o"), "--write-sidecars"]
    assert main(args) == 0
    side = Path(str(sc["take_path"]) + ".utc.json")
    doc = json.loads(side.read_text())
    assert doc["schema"] == "trinet-take-utc/1" and set(doc["eyes"]) == {"L", "R"}
    assert doc["provenance"]["log_files"][0]["sha256"]
    side.write_text("{}")
    assert main(args) == 0
    assert side.read_text() == "{}"
    assert main(args + ["--force"]) == 0
    assert json.loads(side.read_text())["schema"] == "trinet-take-utc/1"


def test_exit_codes(tmp_path, capsys):
    main = _main()
    sc = synth.basic_scenario(tmp_path)
    other = tmp_path / "card2"
    synth.write_stereo_take(other, sc["take"], synth.SimCamera(boot_nonce=0x77))   # unseen boot
    o = str(tmp_path / "o")
    assert main([str(sc["log"]), "--recordings", str(other), "-o", o]) == 0
    assert main([str(sc["log"]), "--recordings", str(other), "-o", o, "--strict"]) == 2
    assert "UNMATCHED: boot_not_observed" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.jsonl"), "--recordings", str(other)]) == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type":"segment"}\n')
    assert main([str(bad), "--recordings", str(other)]) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(sc["log"]), "--recordings", str(empty), "-o", o]) == 1


def test_inspect(tmp_path, capsys):
    sc = synth.basic_scenario(tmp_path)
    assert _main()([str(sc["log"]), "--inspect"]) == 0
    out = capsys.readouterr().out
    assert "unit a1b2c3d4" in out and "buckets" in out and "clock refs" in out

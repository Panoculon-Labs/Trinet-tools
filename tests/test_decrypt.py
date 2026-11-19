# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""decrypt_recordings.py on a real encrypted take.

tests/fixtures/encrypted_take holds a 1 s 320x240 take encrypted by the camera
firmware's own encryption code (muxed as the camera writes it: fragmented MP4),
its .tkey, the first 8 KB of its encrypted .imu, and the expected plain output.
test_key.pem is a throw-away key made for this fixture only -- never use it for
real recordings."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
from trinet_tools import decrypt as td  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decrypt_recordings.py"
FX = ROOT / "tests" / "fixtures" / "encrypted_take"


def _cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def _frames(mp4):
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(mp4), "-map", "0:v", "-fps_mode",
                          "passthrough", "-f", "framemd5", "-"], capture_output=True, text=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if not ln.startswith("#")]


def _take(tmp_path):
    d = tmp_path / "card"
    d.mkdir()
    for f in ("recording1_1.mp4", "recording1_1.tkey", "recording1_1.imu"):
        shutil.copy(FX / f, d / f)
    return d


def test_decrypts_take_to_original(tmp_path):
    card = _take(tmp_path)
    out = tmp_path / "out"
    r = _cli("decrypt", "--key", FX / "test_key.pem", "--out", out, card)
    assert r.returncode == 0, r.stdout + r.stderr
    # sidecars come back byte-identical
    assert (out / "recording1_1.imu").read_bytes() == (FX / "expected" / "recording1_1.imu").read_bytes()
    # the MP4 keeps its layout (same size) and decodes to the original frames
    assert (out / "recording1_1.mp4").stat().st_size == (FX / "recording1_1.mp4").stat().st_size
    if shutil.which("ffmpeg"):
        want = (FX / "expected" / "frames.md5").read_text().split()
        assert _frames(out / "recording1_1.mp4") == want
        assert not set(_frames(card / "recording1_1.mp4")) & set(want), "encrypted file shows frames"
    # originals untouched
    assert (card / "recording1_1.mp4").read_bytes() == (FX / "recording1_1.mp4").read_bytes()


def test_key_from_the_video_when_tkey_missing(tmp_path):
    card = _take(tmp_path)
    (card / "recording1_1.tkey").unlink()
    (card / "recording1_1.imu").unlink()        # sidecars need the .tkey; the video carries its key
    out = tmp_path / "out"
    r = _cli("decrypt", "--key", FX / "test_key.pem", "--out", out, card / "recording1_1.mp4")
    assert r.returncode == 0, r.stdout + r.stderr
    if shutil.which("ffmpeg"):
        assert _frames(out / "recording1_1.mp4") == (FX / "expected" / "frames.md5").read_text().split()


def test_wrong_private_key_is_refused(tmp_path):
    card = _take(tmp_path)
    other = tmp_path / "other.pem"
    r = _cli("keygen", other, tmp_path / "other.tvk")
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    r = _cli("decrypt", "--key", other, "--out", out, card)
    assert r.returncode != 0
    assert "not this private key" in r.stdout + r.stderr
    assert not (out / "recording1_1.mp4").exists(), "a refused take must leave no output"
    assert not (out / "recording1_1.imu").exists()


def test_keygen_and_export_match(tmp_path):
    pem, tvk, tvk2 = tmp_path / "k.pem", tmp_path / "k.tvk", tmp_path / "k2.tvk"
    assert _cli("keygen", pem, tvk).returncode == 0
    assert _cli("export-pubkey", pem, tvk2).returncode == 0
    assert tvk.read_bytes() == tvk2.read_bytes()
    assert tvk.read_bytes()[:4] == b"TVK1" and len(tvk.read_bytes()) == 264


def test_fixture_key_matches_tkey():
    from cryptography.hazmat.primitives import serialization
    priv = serialization.load_pem_private_key((FX / "test_key.pem").read_bytes(), password=None)
    key_id, _, _ = td.parse_tkey((FX / "recording1_1.tkey").read_bytes())
    assert key_id == td.key_id_of(priv.public_key())
    assert (FX / "test_key.tvk").read_bytes() == td.pubkey_to_tvk(priv.public_key())

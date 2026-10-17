# SPDX-License-Identifier: MIT
"""Reading a camera's calibration from an MP4, over USB (UVC) and over NCM."""

import ctypes
import http.server
import json
import threading
from pathlib import Path

import pytest

from trinet_tools import device_calibration as dc

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "data" / "take0002_L.mp4"


def _frames(blob: bytes):
    """Camera-side chunking: [seq, total, len, rsv, data[56]] padded to 60 bytes."""
    total = (len(blob) + dc.CHUNK - 1) // dc.CHUNK if blob else 0
    def read_chunk(i):
        part = blob[i * dc.CHUNK:(i + 1) * dc.CHUNK] if total else b""
        return bytes([i, total, len(part), 0]) + part.ljust(dc.CHUNK, b"\0")
    return read_chunk


@pytest.mark.parametrize("size", [200, 300, 56, 57])
def test_reassemble_chunks_roundtrip(size):
    blob = bytes(range(256)) * 2
    blob = blob[:size]
    assert dc.reassemble_chunks(_frames(blob)) == blob


def test_reassemble_chunks_none_stored():
    assert dc.reassemble_chunks(_frames(b"")) is None


def test_mono_blob_to_json():
    from trinet_tools import calib_blob
    blob = calib_blob.pack({"intrinsics": {"fx": 720.0, "fy": 720.0, "cx": 960.0, "cy": 540.0,
                                           "model": "equidistant",
                                           "distortion": [0.1, -0.02, 0.003, -0.001]}})
    assert len(blob) == 200
    calib = dc.to_calibration_json(dc.reassemble_chunks(_frames(blob)))
    assert calib["intrinsics"]["cx"] == 960.0 and "chip_id" not in calib


def test_reassemble_chunks_rejects_malformed():
    bad = lambda i: bytes([0, 1, 99, 0]) + b"\0" * 56   # len 99 > 56
    with pytest.raises(dc.CalibrationError):
        dc.reassemble_chunks(bad)


def test_uvc_ioctl_matches_kernel_definition():
    # struct uvc_xu_control_query is 16 bytes on 64-bit Linux -> 0xc0107521
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(dc._XuQuery) == 16
        assert dc.UVCIOC_CTRL_QUERY == 0xC0107521


def test_ncm_camera_ips_from_ip_and_ifconfig():
    ip_out = ("2: usb0    inet 172.32.18.71/24 brd 172.32.18.255 scope global usb0\n"
              "3: wlan0    inet 192.168.1.5/24 scope global wlan0\n"
              "4: usb1    inet 172.32.203.71/24 scope global usb1\n")
    assert dc.ncm_camera_ips(ip_out) == ["172.32.18.70", "172.32.203.70"]
    ifconfig_out = "en7: flags=8863<UP>\n\tinet 172.32.44.71 netmask 0xffffff00 broadcast 172.32.44.255\n"
    assert dc.ncm_camera_ips(ifconfig_out) == ["172.32.44.70"]
    assert dc.ncm_camera_ips("inet 172.32.9.70/24") == []    # that's the camera side, not ours


class _Handler(http.server.BaseHTTPRequestHandler):
    blob = b""

    def do_GET(self):
        if self.path != "/api/calibration" or not self.blob:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(self.blob)))
        self.end_headers()
        self.wfile.write(self.blob)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_camera():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def test_read_ncm(fake_camera):
    _Handler.blob = b"TBLC" + b"\1" * 296
    assert dc.read_ncm("127.0.0.1", port=fake_camera.server_port) == _Handler.blob
    _Handler.blob = b""
    assert dc.read_ncm("127.0.0.1", port=fake_camera.server_port) is None   # 404 = none stored


def test_read_ncm_unreachable():
    with pytest.raises(dc.CalibrationError):
        dc.read_ncm("127.0.0.1", port=1, timeout=1)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example recording not present")
def test_from_mp4_and_json():
    blob = dc.from_mp4(EXAMPLE)
    assert blob and blob[:4] == b"TBLC"
    calib = dc.to_calibration_json(blob)
    assert "chip_id" not in calib and calib.get("cameras")


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example recording not present")
def test_cli_mp4(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "get_calibration", Path(__file__).resolve().parent.parent / "scripts" / "get_calibration.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out, binp = tmp_path / "c.json", tmp_path / "c.bin"
    assert mod.main(["--mp4", str(EXAMPLE), "-o", str(out), "--bin", str(binp)]) == 0
    assert json.loads(out.read_text())["cameras"]
    assert binp.read_bytes()[:4] == b"TBLC"

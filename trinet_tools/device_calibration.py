# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Read a Trinet camera's stored calibration — from a recording, over USB
(UVC webcam mode) or over USB networking (NCM / iPhone mode).

Every camera stores its calibration as a binary blob (TBLC, see
docs/data_formats.md "Calibration blob"): 200 bytes for a single camera, 300
bytes for a stereo camera. The same blob is

  * embedded in every recording the camera makes (`moov/udta/tmfc`),
  * served over USB in webcam mode, via the camera's UVC extension unit, and
  * served over HTTP in network (NCM) mode: GET /api/calibration on port 8081.

`calib_blob.unpack()` turns the blob into the dictionary the rest of the
toolkit accepts wherever a `calibration.json` is expected.

    from trinet_tools.device_calibration import (from_mp4, read_uvc, read_ncm,
                                                 to_calibration_json)
    blob = from_mp4("take0001_L.mp4")   # or read_uvc() / read_ncm()
    calib = to_calibration_json(blob)
"""

from __future__ import annotations

import ctypes
import glob
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from trinet_tools import calib_blob
from trinet_tools.tmf import read_tmf

# --- USB (UVC extension unit) -------------------------------------------------
USB_VID, USB_PID = 0x2207, 0x0016
XU_UNIT_ID = 6            # Trinet UVC extension unit
XU_SEL_GET_CALIB = 0x0F   # SET_CUR latches a chunk index, GET_CUR returns that chunk
XU_LEN = 60               # control length; chunk frame = [seq, total, len, rsv, data[56]]
CHUNK = 56
UVC_SET_CUR, UVC_GET_CUR = 0x01, 0x81

# --- NCM (USB networking) -------------------------------------------------------
NCM_PORT = 8081
# In network mode the camera hands the computer 172.32.<X>.71 and answers on .70.
_NCM_HOST_RE = re.compile(r"\b172\.32\.(\d{1,3})\.71\b")

# Field of calib_blob.unpack() output that binds a blob to one device; not
# needed to use the calibration, so it is left out of calibration.json.
DEVICE_BINDING_KEY = "chip_id"


class CalibrationError(RuntimeError):
    pass


# -------------------------------------------------------------------------------
#  From a recording
# -------------------------------------------------------------------------------
def from_mp4(path: str | Path) -> bytes | None:
    """The calibration blob embedded in a Trinet MP4, or None if the camera had
    none stored when it recorded (or the file was re-encoded / re-muxed)."""
    return read_tmf(path).calib_blob or None


# -------------------------------------------------------------------------------
#  Over USB (webcam mode, Linux)
# -------------------------------------------------------------------------------
def reassemble_chunks(read_chunk: Callable[[int], bytes]) -> bytes | None:
    """Reassemble the blob from the camera's chunked read protocol.

    `read_chunk(i)` returns the 60-byte frame [seq, total, len, rsv, data[56]]
    for chunk i. total == 0 means no calibration is stored (-> None)."""
    first = read_chunk(0)
    if len(first) < 4:
        raise CalibrationError("short reply from the camera")
    total = first[1]
    if total == 0:
        return None
    out = bytearray()
    for i in range(total):
        frame = first if i == 0 else read_chunk(i)
        n = frame[2]
        if n > CHUNK or len(frame) < 4 + n:
            raise CalibrationError(f"malformed chunk {i}")
        out += frame[4:4 + n]
    return bytes(out)


class _XuQuery(ctypes.Structure):
    # struct uvc_xu_control_query (linux/uvcvideo.h)
    _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8),
                ("query", ctypes.c_uint8), ("size", ctypes.c_uint16),
                ("data", ctypes.POINTER(ctypes.c_uint8))]


def _iowr(type_char: str, nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


UVCIOC_CTRL_QUERY = _iowr("u", 0x21, ctypes.sizeof(_XuQuery))


def _usb_ids(node: str):
    path = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(node)}/device")
    for _ in range(6):
        v, p = os.path.join(path, "idVendor"), os.path.join(path, "idProduct")
        if os.path.exists(v) and os.path.exists(p):
            try:
                return int(Path(v).read_text(), 16), int(Path(p).read_text(), 16)
            except (OSError, ValueError):
                return None
        path = os.path.dirname(path)
    return None


def find_uvc_nodes() -> list[str]:
    """Video nodes belonging to Trinet cameras in webcam mode (Linux), lowest first."""
    nodes = sorted(glob.glob("/dev/video*"), key=lambda n: int(re.sub(r"\D", "", n) or 0))
    return [n for n in nodes if _usb_ids(n) == (USB_VID, USB_PID)]


def read_uvc(node: str | None = None) -> bytes | None:
    """Read the stored calibration over USB from a camera in webcam mode (Linux,
    uvcvideo driver). `node` defaults to the first Trinet video node found.
    Works while another application is streaming from the camera."""
    import fcntl  # Linux/macOS only; keep the module importable on Windows
    if node is None:
        nodes = find_uvc_nodes()
        if not nodes:
            raise CalibrationError("no Trinet camera found in USB webcam mode "
                                   "(connect it without a memory card)")
        node = nodes[0]
    fd = os.open(node, os.O_RDWR)
    try:
        def query(q: int, payload: bytes = b"") -> bytes:
            buf = (ctypes.c_uint8 * XU_LEN)(*payload.ljust(XU_LEN, b"\0"))
            req = _XuQuery(XU_UNIT_ID, XU_SEL_GET_CALIB, q, XU_LEN, buf)
            fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, req)
            return bytes(buf)

        def read_chunk(i: int) -> bytes:
            query(UVC_SET_CUR, bytes([i]))
            return query(UVC_GET_CUR)

        try:
            return reassemble_chunks(read_chunk)
        except OSError as e:
            raise CalibrationError(f"{node}: the camera did not answer the calibration "
                                   f"request ({e.strerror}); update its firmware") from e
    finally:
        os.close(fd)


# -------------------------------------------------------------------------------
#  Over USB networking (NCM / iPhone mode)
# -------------------------------------------------------------------------------
def ncm_camera_ips(ip_addr_text: str | None = None) -> list[str]:
    """Camera addresses for every attached camera in network mode, from the
    computer's own interface addresses (172.32.<X>.71 -> camera 172.32.<X>.70)."""
    if ip_addr_text is None:
        ip_addr_text = ""
        for cmd in (["ip", "-4", "-o", "addr"], ["ifconfig"], ["ipconfig"]):
            try:
                ip_addr_text = subprocess.run(cmd, capture_output=True, text=True,
                                              timeout=5).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            if ip_addr_text:
                break
    seen = []
    for x in _NCM_HOST_RE.findall(ip_addr_text):
        ip = f"172.32.{x}.70"
        if ip not in seen:
            seen.append(ip)
    return seen


def read_ncm(ip: str | None = None, timeout: float = 5.0,
             port: int = NCM_PORT) -> bytes | None:
    """Read the stored calibration over HTTP from a camera in network mode.
    `ip` defaults to the first camera found on a USB-network interface."""
    if ip is None:
        ips = ncm_camera_ips()
        if not ips:
            raise CalibrationError("no Trinet camera found in network (NCM) mode")
        ip = ips[0]
    url = f"http://{ip}:{port}/api/calibration"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read() or None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise CalibrationError(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, OSError) as e:
        raise CalibrationError(f"{url}: {getattr(e, 'reason', e)}") from e


# -------------------------------------------------------------------------------
#  Output
# -------------------------------------------------------------------------------
def to_calibration_json(blob: bytes) -> dict:
    """The blob as a calibration dictionary (accepted wherever the toolkit takes
    --calibration). The device-binding field is left out."""
    calib = calib_blob.unpack(blob)
    calib.pop(DEVICE_BINDING_KEY, None)
    return calib

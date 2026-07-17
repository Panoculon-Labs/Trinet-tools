#!/usr/bin/env python3
"""
Author: Panoculon Labs

Trinet calibration blob (TBLC) codec: convert a `calibration.json` to/from the
packed binary blob the Trinet camera stores and serves (and embeds in every
recording's `moov/udta/tmfc` box — see `docs/data_formats.md`).

Two layouts share the `TBLC` magic and trailing CRC-32:

- **v1 (200 B)** — one camera: intrinsics + camera<-IMU extrinsics + IMU noise
  model + biases + calibration-quality residuals.
- **v2 (300 B)** — stereo: two camera blocks (cam0 = scene-LEFT `_L` eye),
  `T_cam0_imu`, the stereo baseline `T_cam1_cam0`, shared IMU block.

The same layout is implemented in the Trinet Android and iOS SDKs; `unpack`
below is the reference decoder.

Usage:
    python3 -m trinet_tools.calib_blob pack   calibration.json calib.bin
    python3 -m trinet_tools.calib_blob unpack calib.bin        # prints JSON
"""

import argparse
import json
import struct
import sys
import zlib

MAGIC = 0x434C4254          # "TBLC" little-endian
VERSION = 1
BLOB_SIZE = 200
VERSION_V2 = 2              # stereo: 2 cams + baseline (Trinet-Pro-Stereo)
BLOB_SIZE_V2 = 300
MAX_DISTORTION = 5

FLAG_CHIP_ID_VALID = 0x0001
FLAG_EXTRINSICS_VALID = 0x0002
FLAG_TIMESHIFT_VALID = 0x0004
FLAG_BIAS_VALID = 0x0008
FLAG_QUALITY_VALID = 0x0010
FLAG_TIMESHIFT_SIGN_TIMU = 0x0020   # convention: t_imu = t_cam + timeshift

MODEL_PINHOLE_RADTAN = 0
MODEL_PINHOLE_EQUI = 1

# Everything except the trailing u32 crc32 (computed over the first 160 bytes).
_STRUCT = struct.Struct(
    "<"      # little-endian
    "I"      # magic
    "H"      # version
    "H"      # flags
    "16s"    # chip_id
    "H"      # res_w
    "H"      # res_h
    "B"      # camera_model
    "B"      # num_distortion
    "2s"     # _rsv0
    "4f"     # fx, fy, cx, cy
    "5f"     # distortion[5]
    "9f"     # R_cam_imu[9]
    "3f"     # t_cam_imu[3]
    "f"      # timeshift_cam_imu_s
    "f"      # accel_noise_density
    "f"      # gyro_noise_density
    "f"      # accel_random_walk
    "f"      # gyro_random_walk
    "3f"     # accel_bias[3]   (m/s^2, VIO init)
    "3f"     # gyro_bias[3]    (rad/s, VIO init)
    "f"      # reprojection_rms_px (QA)
    "f"      # gyro_residual       (QA, Kalibr gyro error mean)
    "f"      # accel_residual      (QA, Kalibr accel error mean)
    "f"      # imu_rate_hz
    "20s"    # _rsv2
)
assert _STRUCT.size == BLOB_SIZE - 4, _STRUCT.size


def _model_code(name: str) -> int:
    n = (name or "").lower()
    if any(k in n for k in ("equi", "fisheye", "kannala", "kb")):
        return MODEL_PINHOLE_EQUI
    return MODEL_PINHOLE_RADTAN  # radtan / plumb_bob / pinhole-radtan / unknown


def _load_calibration(data: dict) -> dict:
    """Normalize either calibration.json schema into a flat dict of fields."""
    out = {
        "res_w": 1920, "res_h": 1080,
        "model": "radtan", "distortion": [],
        "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "t": [0, 0, 0],
        "has_extrinsics": False,
        "timeshift": 0.0, "has_timeshift": False, "timeshift_sign_timu": False,
        "accel_nd": 0.0, "gyro_nd": 0.0, "accel_rw": 0.0, "gyro_rw": 0.0,
        "accel_bias": [0.0, 0.0, 0.0], "gyro_bias": [0.0, 0.0, 0.0],
        "has_bias": False,
        "reproj_rms": 0.0, "gyro_resid": 0.0, "accel_resid": 0.0,
        "has_quality": False, "imu_rate": 0.0,
    }

    # --- intrinsics: nested `intrinsics{}` (calibrate_kalibr) or flat `camera{}` ---
    if "intrinsics" in data and isinstance(data["intrinsics"], dict):
        intr = data["intrinsics"]
        out["fx"], out["fy"] = intr.get("fx", 0.0), intr.get("fy", 0.0)
        out["cx"], out["cy"] = intr.get("cx", 0.0), intr.get("cy", 0.0)
        out["model"] = intr.get("model", "radtan")
        out["distortion"] = list(intr.get("distortion", []))
        size = intr.get("image_size") or intr.get("resolution") or [1920, 1080]
        out["res_w"], out["res_h"] = int(size[0]), int(size[1])
        if "reprojection_rms_px" in intr:
            out["reproj_rms"] = float(intr["reprojection_rms_px"])
            out["has_quality"] = True
    elif "camera" in data and isinstance(data["camera"], dict):
        cam = data["camera"]
        ci = cam.get("intrinsics", {})
        out["fx"], out["fy"] = ci.get("fx", 0.0), ci.get("fy", 0.0)
        out["cx"], out["cy"] = ci.get("cx", 0.0), ci.get("cy", 0.0)
        dist = cam.get("distortion", {})
        out["model"] = dist.get("model", "radtan")
        out["distortion"] = list(dist.get("coeffs", []))
        size = cam.get("resolution", [1920, 1080])
        out["res_w"], out["res_h"] = int(size[0]), int(size[1])
    else:
        raise ValueError("calibration JSON has neither 'intrinsics' nor 'camera'")

    # --- extrinsics: nested `extrinsics{}` or top-level T_cam_imu ---
    T = None
    ext = data.get("extrinsics")
    if isinstance(ext, dict):
        if "R_cam_imu" in ext and "t_cam_imu_m" in ext:
            R = ext["R_cam_imu"]
            out["R"] = [float(R[i][j]) for i in range(3) for j in range(3)]
            out["t"] = [float(x) for x in ext["t_cam_imu_m"]]
            out["has_extrinsics"] = True
        elif "T_cam_imu" in ext:
            T = ext["T_cam_imu"]
        # timeshift: Kalibr/Trinet-Calibration has shipped both "..._s" and
        # "..._sec" across script versions — accept either.
        for _tk in ("timeshift_cam_imu_s", "timeshift_cam_imu_sec"):
            if _tk in ext:
                out["timeshift"] = float(ext[_tk])
                out["has_timeshift"] = True
                break
        if "t_imu = t_cam +" in str(ext.get("timeshift_sign_convention", "")):
            out["timeshift_sign_timu"] = True
        # Kalibr calibration-quality residuals (QA, not used at VIO runtime).
        if "kalibr_gyro_error_mean" in ext:
            out["gyro_resid"] = float(ext["kalibr_gyro_error_mean"]); out["has_quality"] = True
        if "kalibr_accel_error_mean" in ext:
            out["accel_resid"] = float(ext["kalibr_accel_error_mean"]); out["has_quality"] = True
        if "kalibr_reprojection_error_mean_px" in ext and out["reproj_rms"] == 0.0:
            out["reproj_rms"] = float(ext["kalibr_reprojection_error_mean_px"]); out["has_quality"] = True
    if T is None and "T_cam_imu" in data:
        T = data["T_cam_imu"]
    if T is not None:
        out["R"] = [float(T[i][j]) for i in range(3) for j in range(3)]
        out["t"] = [float(T[i][3]) for i in range(3)]
        out["has_extrinsics"] = True
    if not out["has_timeshift"]:
        for _tk in ("timeshift_cam_imu_s", "timeshift_cam_imu_sec"):
            if _tk in data:
                out["timeshift"] = float(data[_tk])
                out["has_timeshift"] = True
                break

    # --- IMU noise: directly under imu{} OR nested under imu.noise_model{},
    # with either short (accel_*) or long (accelerometer_*) names. ---
    imu = data.get("imu", {})
    if isinstance(imu, dict):
        nm = imu.get("noise_model")
        nm = nm if isinstance(nm, dict) else {}

        def _imuget(*names):
            for src in (nm, imu):
                for nme in names:
                    if nme in src:
                        return float(src[nme])
            return 0.0

        out["accel_nd"] = _imuget("accel_noise_density", "accelerometer_noise_density")
        out["gyro_nd"] = _imuget("gyro_noise_density", "gyroscope_noise_density")
        out["accel_rw"] = _imuget("accel_random_walk", "accelerometer_random_walk")
        out["gyro_rw"] = _imuget("gyro_random_walk", "gyroscope_random_walk")
        # IMU biases (VIO init) + sample rate.
        ab = imu.get("accel_bias_m_s2") or imu.get("accel_bias")
        gb = imu.get("gyro_bias_rad_s") or imu.get("gyro_bias")
        if isinstance(ab, list) and len(ab) >= 3:
            out["accel_bias"] = [float(x) for x in ab[:3]]; out["has_bias"] = True
        if isinstance(gb, list) and len(gb) >= 3:
            out["gyro_bias"] = [float(x) for x in gb[:3]]; out["has_bias"] = True
        for _rate_key in ("sample_rate_hz", "rate_hz"):
            if _rate_key in imu:
                out["imu_rate"] = float(imu[_rate_key])
                break
    return out


def pack(data: dict, chip_id: bytes = b"") -> bytes:
    f = _load_calibration(data)
    dist = (f["distortion"] + [0.0] * MAX_DISTORTION)[:MAX_DISTORTION]
    num_dist = min(len(f["distortion"]), MAX_DISTORTION)

    flags = 0
    chip = (chip_id or b"")[:16].ljust(16, b"\x00")
    if chip_id:
        flags |= FLAG_CHIP_ID_VALID
    if f["has_extrinsics"]:
        flags |= FLAG_EXTRINSICS_VALID
    if f["has_timeshift"]:
        flags |= FLAG_TIMESHIFT_VALID
    if f["has_bias"]:
        flags |= FLAG_BIAS_VALID
    if f["has_quality"]:
        flags |= FLAG_QUALITY_VALID
    if f["timeshift_sign_timu"]:
        flags |= FLAG_TIMESHIFT_SIGN_TIMU

    head = _STRUCT.pack(
        MAGIC, VERSION, flags, chip,
        f["res_w"], f["res_h"],
        _model_code(f["model"]), num_dist, b"\x00\x00",
        f["fx"], f["fy"], f["cx"], f["cy"],
        *dist,
        *f["R"],
        *f["t"],
        f["timeshift"],
        f["accel_nd"], f["gyro_nd"], f["accel_rw"], f["gyro_rw"],
        *f["accel_bias"],
        *f["gyro_bias"],
        f["reproj_rms"], f["gyro_resid"], f["accel_resid"], f["imu_rate"],
        b"\x00" * 20,
    )
    crc = zlib.crc32(head) & 0xFFFFFFFF
    return head + struct.pack("<I", crc)


# --------------------------------------------------------------------------
# Version 2 — stereo (two cameras + Kalibr camchain baseline). Layout mirrors
# trinet_calib_blob_v2_t in trinet_calib_blob.h (300 bytes).
# --------------------------------------------------------------------------
_CAM_V2 = struct.Struct("<HHBB2s4f5fff")            # 52 bytes per camera
_HEAD_V2 = struct.Struct("<IHH16sB3s")              # 28 bytes
_TAIL_V2 = struct.Struct("<9f3f9f3f4f3f3f3f16s")    # 164 bytes
assert _HEAD_V2.size + 2 * _CAM_V2.size + _TAIL_V2.size == BLOB_SIZE_V2 - 4


def _pack_cam_v2(c: dict) -> bytes:
    dist = (list(c.get("distortion", [])) + [0.0] * MAX_DISTORTION)[:MAX_DISTORTION]
    return _CAM_V2.pack(
        int(c.get("res_w", 1920)), int(c.get("res_h", 1080)),
        _model_code(c.get("model", "radtan")),
        min(len(c.get("distortion", [])), MAX_DISTORTION), b"\x00\x00",
        c.get("fx", 0.0), c.get("fy", 0.0), c.get("cx", 0.0), c.get("cy", 0.0),
        *dist,
        c.get("timeshift", 0.0),
        c.get("reproj_rms", 0.0),
    )


def _cam_fields(cam_json: dict) -> dict:
    """Normalize one camera's entry of a stereo calibration.json."""
    intr = cam_json.get("intrinsics", cam_json)
    size = intr.get("image_size") or intr.get("resolution") or [1920, 1080]
    return {
        "res_w": int(size[0]), "res_h": int(size[1]),
        "model": intr.get("model", "radtan"),
        "distortion": list(intr.get("distortion", [])),
        "fx": float(intr.get("fx", 0.0)), "fy": float(intr.get("fy", 0.0)),
        "cx": float(intr.get("cx", 0.0)), "cy": float(intr.get("cy", 0.0)),
        "timeshift": float(cam_json.get("timeshift_cam_imu_s", 0.0)),
        "reproj_rms": float(cam_json.get("reprojection_rms_px",
                                         intr.get("reprojection_rms_px", 0.0))),
    }


def _rt_from_T(T) -> tuple:
    R = [float(T[i][j]) for i in range(3) for j in range(3)]
    t = [float(T[i][3]) for i in range(3)]
    return R, t


def pack_v2(data: dict, chip_id: bytes = b"") -> bytes:
    """Stereo calibration.json -> 300-byte v2 blob.

    Expects: cameras[2] (cam0 = physical LEFT), T_cam0_imu (4x4),
    T_cam1_cam0 (4x4, Kalibr camchain T_cn_cnm1), imu{...} as v1.
    """
    cams = data.get("cameras")
    if not isinstance(cams, list) or len(cams) < 1:
        raise ValueError("stereo calibration JSON needs 'cameras': [cam0, cam1]")
    num_cams = min(len(cams), 2)
    cam_blocks = [_pack_cam_v2(_cam_fields(cams[i])) if i < num_cams
                  else _CAM_V2.pack(0, 0, 0, 0, b"\x00\x00", *([0.0] * 11))
                  for i in range(2)]

    R0, t0 = _rt_from_T(data["T_cam0_imu"]) if "T_cam0_imu" in data \
        else ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0])
    R10, t10 = _rt_from_T(data["T_cam1_cam0"]) if "T_cam1_cam0" in data \
        else ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0])

    imu = _load_calibration({"intrinsics": {"fx": 0}, "imu": data.get("imu", {}),
                             })  # reuse the v1 IMU-noise normalizer
    flags = 0
    chip = (chip_id or b"")[:16].ljust(16, b"\x00")
    if chip_id:
        flags |= FLAG_CHIP_ID_VALID
    if "T_cam0_imu" in data:
        flags |= FLAG_EXTRINSICS_VALID
    if any(c.get("timeshift_cam_imu_s") for c in cams if isinstance(c, dict)):
        flags |= FLAG_TIMESHIFT_VALID
    if imu["has_bias"]:
        flags |= FLAG_BIAS_VALID
    if "t_imu = t_cam +" in str(data.get("timeshift_sign_convention", "")):
        flags |= FLAG_TIMESHIFT_SIGN_TIMU

    head = _HEAD_V2.pack(MAGIC, VERSION_V2, flags, chip, num_cams, b"\x00" * 3)
    tail = _TAIL_V2.pack(
        *R0, *t0, *R10, *t10,
        imu["accel_nd"], imu["gyro_nd"], imu["accel_rw"], imu["gyro_rw"],
        *imu["accel_bias"], *imu["gyro_bias"],
        float(data.get("kalibr_gyro_error_mean", 0.0)),
        float(data.get("kalibr_accel_error_mean", 0.0)),
        imu["imu_rate"],
        b"\x00" * 16,
    )
    body = head + b"".join(cam_blocks) + tail
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def _unpack_v2(blob: bytes) -> dict:
    body = blob[:-4]
    magic, version, flags, chip, num_cams, _ = _HEAD_V2.unpack_from(body, 0)
    cams = []
    for i in range(num_cams):
        v = _CAM_V2.unpack_from(body, _HEAD_V2.size + i * _CAM_V2.size)
        (res_w, res_h, model, ndist, _r, fx, fy, cx, cy,
         d0, d1, d2, d3, d4, ts, reproj) = v
        cams.append({
            "intrinsics": {
                "image_size": [res_w, res_h],
                "model": "equidistant" if model == MODEL_PINHOLE_EQUI else "radtan",
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "distortion": [d0, d1, d2, d3, d4][:ndist],
            },
            "timeshift_cam_imu_s": ts,
            "reprojection_rms_px": reproj,
        })
    tv = _TAIL_V2.unpack_from(body, _HEAD_V2.size + 2 * _CAM_V2.size)
    R0, t0 = tv[0:9], tv[9:12]
    R10, t10 = tv[12:21], tv[21:24]
    (accel_nd, gyro_nd, accel_rw, gyro_rw) = tv[24:28]
    ab, gb = tv[28:31], tv[31:34]
    gyro_resid, accel_resid, imu_rate = tv[34:37]

    def _T(R, t):
        return [[R[0], R[1], R[2], t[0]], [R[3], R[4], R[5], t[1]],
                [R[6], R[7], R[8], t[2]], [0.0, 0.0, 0.0, 1.0]]

    return {
        "version": version,
        "flags": flags,
        "chip_id": chip.hex(),
        "num_cams": num_cams,
        "cameras": cams,
        "T_cam0_imu": _T(R0, t0),
        "T_cam1_cam0": _T(R10, t10),
        "imu": {
            "accel_bias_m_s2": list(ab),
            "gyro_bias_rad_s": list(gb),
            "bias_valid": bool(flags & FLAG_BIAS_VALID),
            "sample_rate_hz": imu_rate,
            "noise_model": {
                "accel_noise_density": accel_nd,
                "gyro_noise_density": gyro_nd,
                "accel_random_walk": accel_rw,
                "gyro_random_walk": gyro_rw,
            },
        },
        "quality": {
            "kalibr_gyro_error_mean": gyro_resid,
            "kalibr_accel_error_mean": accel_resid,
        },
    }


def unpack(blob: bytes) -> dict:
    if len(blob) not in (BLOB_SIZE, BLOB_SIZE_V2):
        raise ValueError(f"blob is {len(blob)} bytes, expected "
                         f"{BLOB_SIZE} (v1) or {BLOB_SIZE_V2} (v2)")
    head, (crc,) = blob[:-4], struct.unpack("<I", blob[-4:])
    want = zlib.crc32(head) & 0xFFFFFFFF
    if crc != want:
        raise ValueError(f"CRC mismatch: have {crc:#010x} want {want:#010x}")
    if len(blob) == BLOB_SIZE_V2:
        return _unpack_v2(blob)
    vals = _STRUCT.unpack(head)
    (magic, version, flags, chip, res_w, res_h, model, num_dist, _r0,
     fx, fy, cx, cy,
     d0, d1, d2, d3, d4,
     r0, r1, r2, r3, r4, r5, r6, r7, r8,
     t0, t1, t2,
     timeshift, accel_nd, gyro_nd, accel_rw, gyro_rw,
     ab0, ab1, ab2, gb0, gb1, gb2,
     reproj, gyro_resid, accel_resid, imu_rate, _r1) = vals
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#010x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    dist = [d0, d1, d2, d3, d4][:num_dist]
    return {
        "version": version,
        "flags": flags,
        "chip_id": chip.hex(),
        "intrinsics": {
            "image_size": [res_w, res_h],
            "model": "equidistant" if model == MODEL_PINHOLE_EQUI else "radtan",
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "distortion": dist,
        },
        "extrinsics": {
            "R_cam_imu": [[r0, r1, r2], [r3, r4, r5], [r6, r7, r8]],
            "t_cam_imu_m": [t0, t1, t2],
            "T_cam_imu": [[r0, r1, r2, t0], [r3, r4, r5, t1],
                          [r6, r7, r8, t2], [0.0, 0.0, 0.0, 1.0]],
            "timeshift_cam_imu_s": timeshift,
            "timeshift_sign_convention": ("t_imu = t_cam + timeshift_cam_imu_s"
                                          if (flags & FLAG_TIMESHIFT_SIGN_TIMU) else None),
            "valid": bool(flags & FLAG_EXTRINSICS_VALID),
        },
        "imu": {
            "accel_bias_m_s2": [ab0, ab1, ab2],
            "gyro_bias_rad_s": [gb0, gb1, gb2],
            "bias_valid": bool(flags & FLAG_BIAS_VALID),
            "sample_rate_hz": imu_rate,
            "noise_model": {
                "accel_noise_density": accel_nd,
                "gyro_noise_density": gyro_nd,
                "accel_random_walk": accel_rw,
                "gyro_random_walk": gyro_rw,
            },
        },
        "quality": {
            "reprojection_rms_px": reproj,
            "kalibr_gyro_error_mean": gyro_resid,
            "kalibr_accel_error_mean": accel_resid,
            "valid": bool(flags & FLAG_QUALITY_VALID),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trinet calibration blob codec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="calibration.json -> 164-byte blob")
    p.add_argument("json")
    p.add_argument("out")
    p.add_argument("--chip-id", default="", help="32 hex chars to bind the blob")
    u = sub.add_parser("unpack", help="blob -> JSON (stdout)")
    u.add_argument("blob")
    args = ap.parse_args(argv)

    if args.cmd == "pack":
        with open(args.json) as fh:
            data = json.load(fh)
        chip = bytes.fromhex(args.chip_id) if args.chip_id else b""
        # Stereo JSON ('cameras' list) packs the v2 layout; mono packs v1.
        blob = pack_v2(data, chip) if "cameras" in data else pack(data, chip)
        with open(args.out, "wb") as fh:
            fh.write(blob)
        print(f"wrote {len(blob)} bytes (v{2 if 'cameras' in data else 1}) to {args.out}")
    else:
        with open(args.blob, "rb") as fh:
            blob = fh.read()
        print(json.dumps(unpack(blob), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

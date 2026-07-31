#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Gravity-align a trained splat.ply (gs_train.py layout) so +z is up.

Up = the dominant plane normal of the training camera path, SIGNED by the
consensus of the cameras' own up directions (image-up = -y_cam in world) —
the unsigned eigenvector can point down, flipping the scene 180 degrees in
any viewer. Also recenters xy and puts the floor near z=0.

Usage: gs_align.py SPLAT_PLY DATA_DIR OUT_PLY
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gs_train import load_colmap_text  # noqa: E402


def quat_from_R(M):
    w = np.sqrt(max(0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    return np.array([w, (M[2, 1] - M[1, 2]) / (4 * w),
                     (M[0, 2] - M[2, 0]) / (4 * w),
                     (M[1, 0] - M[0, 1]) / (4 * w)])


def main():
    ply, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    _, _, _, ims, _, _ = load_colmap_text(data / "sparse" / "0")
    Ts = [T for _, T in ims]
    centers = np.stack([np.linalg.inv(T)[:3, 3] for T in Ts])
    votes = np.stack([-np.linalg.inv(T)[:3, 1] for T in Ts]).mean(0)
    cc = centers - centers.mean(0)
    _, V = np.linalg.eigh(cc.T @ cc)
    n = V[:, 0]
    if n @ votes < 0:
        n = -n
    z = np.array([0., 0., 1.])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + Vx + Vx @ Vx * ((1 - (n @ z)) / (s * s)) if s > 1e-8 \
        else np.eye(3)

    raw = ply.read_bytes()
    he = raw.index(b"end_header\n") + len(b"end_header\n")
    nv = int([l for l in raw[:he].split(b"\n")
              if b"element vertex" in l][0].split()[-1])
    D = np.frombuffer(raw[he:he + nv * 17 * 4],
                      dtype=np.float32).reshape(nv, 17).copy()
    qR = quat_from_R(R)
    w1, x1, y1, z1 = qR
    w2, x2, y2, z2 = D[:, 13], D[:, 14], D[:, 15], D[:, 16]
    D[:, 0:3] = (D[:, 0:3] - centers.mean(0)) @ R.T
    D[:, 13] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    D[:, 14] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    D[:, 15] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    D[:, 16] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    D[:, 2] -= np.percentile(D[:, 2], 3)
    out.write_bytes(raw[:he] + D.astype(np.float32).tobytes())
    print(f"aligned {nv} gaussians -> {out} (up was {np.round(n, 3)})")


if __name__ == "__main__":
    main()

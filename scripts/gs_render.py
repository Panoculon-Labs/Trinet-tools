#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Render a trained splat (gs_train.py output): a stabilized walkthrough along
the capture path plus GT|render comparison stills for held-out views.

Inside-out captures cannot be orbited from outside — the walkthrough re-flies
a smoothed version of the original camera path instead.

Usage:
    python3 scripts/gs_render.py DATA_DIR SPLAT_PLY OUT_DIR \
        [--smooth 15] [--fps 30] [--test-every 10]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gs_train import load_colmap_text  # noqa: E402

from gsplat import rasterization  # noqa: E402


def load_splat_ply(path, dev):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        n = 0
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line == b"end_header":
                break
        data = np.frombuffer(f.read(n * 17 * 4), dtype=np.float32)
    data = data.reshape(n, 17)
    return dict(
        means=torch.tensor(data[:, 0:3], device=dev),
        sh0=torch.tensor(data[:, 6:9], device=dev),
        opac=torch.sigmoid(torch.tensor(data[:, 9], device=dev)),
        scales=torch.exp(torch.tensor(data[:, 10:13], device=dev)),
        quats=torch.tensor(data[:, 13:17], device=dev),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("splat", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smooth", type=int, default=15,
                    help="moving-average window (frames) for the path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--compare-views", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    K_np, W, H, ims, _, _ = load_colmap_text(args.data / "sparse" / "0")
    K = torch.from_numpy(K_np).to(dev)[None]
    g = load_splat_ply(args.splat, dev)
    print(f"{len(g['means'])} gaussians, {len(ims)} source views")

    def render(T_cam_world):
        img, _, _ = rasterization(
            g["means"], F.normalize(g["quats"], dim=-1), g["scales"],
            g["opac"], g["sh0"][:, None, :],
            torch.tensor(T_cam_world, dtype=torch.float32, device=dev)[None],
            K, W, H, sh_degree=0)
        return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # --- held-out comparisons ---
    test_ids = list(range(0, len(ims), args.test_every))
    pick = test_ids[:: max(1, len(test_ids) // args.compare_views)][
        : args.compare_views]
    for j, i in enumerate(pick):
        name, T = ims[i]
        gt = cv2.imread(str(args.data / "images" / name))
        ren = cv2.cvtColor(render(T), cv2.COLOR_RGB2BGR)
        both = np.hstack([gt, ren])
        cv2.putText(both, "capture", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(both, "splat render (held-out view)", (W + 12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"compare_{j}.png"), both)
    print(f"wrote {len(pick)} GT|render comparisons")

    # --- stabilized walkthrough: smooth centers + look-at targets ---
    Ts = [T for _, T in ims]
    centers = np.stack([np.linalg.inv(T)[:3, 3] for T in Ts])
    fwd = np.stack([np.linalg.inv(T)[:3, 2] for T in Ts])  # +z = view dir
    k = args.smooth
    pad = k // 2

    def smooth(x):
        xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        ker = np.ones(k) / k
        return np.stack([np.convolve(xp[:, c], ker, mode="valid")
                         for c in range(x.shape[1])], axis=1)

    sc = smooth(centers)
    sf = smooth(fwd)
    # world up = dominant plane normal of the path, SIGNED by the consensus
    # of the cameras' own up directions (image-up = -y_cam in world) — the
    # unsigned eigenvector can point down, which flips the render 180°.
    cc = centers - centers.mean(0)
    w_, V = np.linalg.eigh(cc.T @ cc)
    up = V[:, 0]
    votes = np.stack([np.linalg.inv(T)[:3, 1] for T in Ts]).mean(0)
    if up @ (-votes) < 0:
        up = -up

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         str(args.out / "walkthrough.mp4")], stdin=subprocess.PIPE)
    for i in range(len(sc)):
        z = sf[i] / np.linalg.norm(sf[i])
        x = np.cross(z, up)
        x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ sc[i]
        ff.stdin.write(render(T).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out/'walkthrough.mp4'} ({len(sc)} frames)")


if __name__ == "__main__":
    main()

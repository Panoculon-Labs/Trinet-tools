#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Minimal 3D Gaussian Splatting trainer for gs_prep.py output.

Trains gsplat Gaussians on the rectified keyframes + metric poses prepared by
gs_prep.py (COLMAP text model, seed points from the fused TSDF map). Loss is
L1; densification/pruning via gsplat's DefaultStrategy. Holds out every 10th
view for PSNR. Outputs: <out>/splat.ply (viewer-standard 3DGS layout),
<out>/turntable.mp4, <out>/metrics.json.

Usage:
    python3 scripts/gs_train.py DATA_DIR OUT_DIR [--iters 15000] [--test-every 10]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

C0 = 0.28209479177387814


def load_colmap_text(sparse: Path):
    cam = sparse / "cameras.txt"
    _, model, w, h, fx, fy, cx, cy = cam.read_text().split()[:8]
    assert model == "PINHOLE"
    K = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)],
                  [0, 0, 1]], dtype=np.float32)
    ims = []
    for line in (sparse / "images.txt").read_text().strip().split("\n"):
        f = line.split()
        qw, qx, qy, qz = map(float, f[1:5])
        t = np.array(list(map(float, f[5:8])), dtype=np.float64)
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        ims.append((f[9], T))
    pts, cols = [], []
    for line in (sparse / "points3D.txt").read_text().strip().split("\n"):
        f = line.split()
        pts.append([float(f[1]), float(f[2]), float(f[3])])
        cols.append([int(f[4]), int(f[5]), int(f[6])])
    return K, int(w), int(h), ims, np.array(pts), np.array(cols) / 255.0


def knn_mean_dist(P, k=3):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=k + 1)
    return d[:, 1:].mean(axis=1)


def save_splat_ply(path, means, quats, scales, opac, sh0):
    n = len(means)
    props = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    data = np.concatenate([
        means, np.zeros_like(means), sh0, opac[:, None], scales, quats,
    ], axis=1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--turntable-frames", type=int, default=240)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    K_np, W, H, ims, P, C = load_colmap_text(args.data / "sparse" / "0")
    print(f"{len(ims)} views {W}x{H}, {len(P)} seed points")

    imgs, viewmats = [], []
    for name, T in ims:
        img = cv2.imread(str(args.data / "images" / name))
        imgs.append(torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(dev) / 255.0)
        viewmats.append(torch.from_numpy(T).float().to(dev))
    K = torch.from_numpy(K_np).to(dev)[None]

    test_ids = set(range(0, len(ims), args.test_every))
    train_ids = [i for i in range(len(ims)) if i not in test_ids]

    scene_scale = float(np.linalg.norm(P.std(axis=0)))
    means = torch.tensor(P, dtype=torch.float32, device=dev)
    sh0 = torch.tensor((C - 0.5) / C0, dtype=torch.float32, device=dev)
    d = knn_mean_dist(P)
    scales = torch.tensor(np.log(np.tile(d[:, None], 3) + 1e-7),
                          dtype=torch.float32, device=dev)
    quats = torch.zeros(len(P), 4, device=dev)
    quats[:, 0] = 1.0
    opac = torch.logit(torch.full((len(P),), 0.2, device=dev))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(quats),
        "scales": torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(sh0),
    }).to(dev)
    lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "sh0": 2.5e-3}
    opts = {k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k]}],
                                eps=1e-15) for k in params}
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(viewmat, sh):
        return rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            params["sh0"][:, None, :] if sh else None,
            viewmat[None], K, W, H,
            sh_degree=0 if sh else None, packed=False,
            absgrad=isinstance(strategy, DefaultStrategy) and strategy.absgrad)

    t0 = time.time()
    rng = np.random.default_rng(0)
    for it in range(args.iters):
        i = int(rng.choice(train_ids))
        img, alpha, info = render(viewmats[i], True)
        strategy.step_pre_backward(params, opts, state, it, info)
        loss = (img[0] - imgs[i]).abs().mean()
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info,
                                    packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        # exp decay for position lr
        opts["means"].param_groups[0]["lr"] = \
            lrs["means"] * 0.01 ** (it / args.iters)
        if (it + 1) % 1000 == 0:
            print(f"  iter {it+1}: loss {float(loss):.4f}, "
                  f"{len(params['means'])} gaussians, "
                  f"{time.time()-t0:.0f}s")

    # held-out PSNR
    psnrs = []
    with torch.no_grad():
        for i in sorted(test_ids):
            img, _, _ = render(viewmats[i], True)
            mse = float(((img[0] - imgs[i]) ** 2).mean())
            psnrs.append(-10 * math.log10(mse))
    print(f"held-out PSNR: {np.mean(psnrs):.2f} dB over {len(psnrs)} views")

    with torch.no_grad():
        save_splat_ply(args.out / "splat.ply",
                       params["means"].cpu().numpy(),
                       F.normalize(params["quats"], dim=-1).cpu().numpy(),
                       params["scales"].cpu().numpy(),
                       params["opacities"].cpu().numpy(),
                       params["sh0"].cpu().numpy())
    print(f"wrote {args.out/'splat.ply'} ({len(params['means'])} gaussians)")

    # turntable: orbit around the view-cluster centroid
    centers = np.stack([np.linalg.inv(T.cpu().numpy())[:3, 3]
                        for T in viewmats])
    c = torch.tensor(P.mean(axis=0), dtype=torch.float32, device=dev)
    up = torch.tensor([0., 0., 1.], device=dev)
    r = float(np.linalg.norm(centers - P.mean(axis=0), axis=1).mean()) * 1.15
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", "-pix_fmt", "yuv420p", str(args.out / "turntable.mp4")],
        stdin=subprocess.PIPE)
    with torch.no_grad():
        for k in range(args.turntable_frames):
            a = 2 * math.pi * k / args.turntable_frames
            eye = c + torch.tensor(
                [r * math.cos(a), r * math.sin(a), 0.35 * r], device=dev)
            z = F.normalize(c - eye, dim=0)
            x = F.normalize(torch.linalg.cross(z, up), dim=0)
            y = torch.linalg.cross(z, x)
            Rm = torch.stack([x, y, z])
            T = torch.eye(4, device=dev)
            T[:3, :3] = Rm
            T[:3, 3] = -Rm @ eye
            img, _, _ = render(T, True)
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out/'turntable.mp4'}")
    (args.out / "metrics.json").write_text(json.dumps({
        "psnr_mean": float(np.mean(psnrs)), "views": len(ims),
        "gaussians": int(len(params["means"])),
        "iters": args.iters, "train_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()

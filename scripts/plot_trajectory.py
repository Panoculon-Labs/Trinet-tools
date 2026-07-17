#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Plot a VIO/SLAM trajectory (OpenVINS `path_est` format) and report drift.

Input: whitespace-separated `timestamp(s) x y z qx qy qz qw` (lines starting
with '#' ignored) — what scripts/run_openvins.sh leaves in ov_out/traj_est.txt.

Writes `<out_prefix>_xy.png` (top-down), `<out_prefix>_3d.png`, and
`<out_prefix>_axes.png` (per-axis vs time), and prints path length, total
displacement, and end-point drift — when the recording starts and ends at the
same physical spot, end-drift as % of path length is the headline VIO metric.

Usage:
    python3 scripts/plot_trajectory.py ov_out/traj_est.txt [--out-prefix P]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    rows = []
    for line in Path(args.traj).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = line.replace(",", " ").split()
        if len(v) >= 8:
            rows.append([float(x) for x in v[:8]])
    if len(rows) < 10:
        raise SystemExit(f"{args.traj}: only {len(rows)} poses")
    a = np.array(rows)
    t = a[:, 0] - a[0, 0]
    p = a[:, 1:4]

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path_len = float(seg.sum())
    disp = float(np.linalg.norm(p[-1] - p[0]))
    print(f"{len(p)} poses over {t[-1]:.1f}s")
    print(f"path length: {path_len:.2f} m")
    print(f"start->end displacement: {disp:.3f} m "
          f"({100*disp/max(path_len,1e-9):.2f}% of path — end-point drift if "
          f"the take returned to its start)")
    ext = p.max(axis=0) - p.min(axis=0)
    print(f"workspace extent: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")

    prefix = args.out_prefix or str(Path(args.traj).with_suffix(""))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(p[:, 0], p[:, 1], lw=1)
    ax.scatter([p[0, 0]], [p[0, 1]], c="g", label="start", zorder=3)
    ax.scatter([p[-1, 0]], [p[-1, 1]], c="r", label="end", zorder=3)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"top-down  |  path {path_len:.1f} m, end-drift {disp:.2f} m")
    fig.savefig(f"{prefix}_xy.png", dpi=130, bbox_inches="tight")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1)
    ax.scatter([p[0, 0]], [p[0, 1]], [p[0, 2]], c="g")
    ax.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], c="r")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    fig.savefig(f"{prefix}_3d.png", dpi=130, bbox_inches="tight")

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(t, p[:, i], lw=0.9)
        axs[i].set_ylabel(f"{lbl} [m]"); axs[i].grid(alpha=0.3)
    axs[2].set_xlabel("time [s]")
    fig.savefig(f"{prefix}_axes.png", dpi=130, bbox_inches="tight")
    print(f"wrote {prefix}_xy.png / _3d.png / _axes.png")


if __name__ == "__main__":
    main()

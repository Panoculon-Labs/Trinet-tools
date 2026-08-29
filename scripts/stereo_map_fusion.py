#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Dense 3D map from a Trinet stereo take + a VIO/SLAM trajectory.

Fuses per-frame metric stereo depth along the estimated trajectory into a
TSDF volume (Open3D) and extracts a colored mesh / point cloud — the "full
3D map" a sparse VIO run doesn't give you by itself. Depth is SGBM by
default; --neural switches to HITNet (GPU, scripts/stereo_depth_neural.py)
with a left-right consistency gate, which fills the textureless surfaces
SGBM leaves as holes while still keeping guessed pixels out of the volume.

Inputs: the take's two MP4s (calibration + timing embedded; sidecars used
when present) and the trajectory from scripts/run_openvins.sh
(`ov_out/traj_est.txt`: timestamp x y z qx qy qz qw, IMU pose in the
gravity-aligned world frame). The camera-from-IMU transform comes from the
same embedded calibration, so everything stays consistent.

Outputs (next to --out-prefix): `<prefix>_map.ply` (mesh) and
`<prefix>_map_views.png` (top-down + two elevations of the fused cloud).

Usage:
    python3 scripts/stereo_map_fusion.py TAKE_PREFIX TRAJ.txt \
        [--out-prefix P] [--every 5] [--voxel 0.03] [--max-depth 5.0] \
        [--calibration CALIB] [--no-auto-align]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

for _p in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from trinet_tools import calib_blob                              # noqa: E402
from trinet_tools.reader import read_vts                         # noqa: E402
from trinet_tools.stereo_align import Rectification, auto_align  # noqa: E402
from trinet_tools.tmf import read_tmf                            # noqa: E402


def quat_to_R(q):
    x, y, z, w = q
    n = np.linalg.norm(q)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("traj")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--every", type=int, default=5,
                    help="integrate every Nth synchronized pair (default 5)")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--no-auto-align", action="store_true")
    ap.add_argument("--max-gyro-dps", type=float, default=25.0,
                    help="skip keyframes rotating faster than this — "
                         "motion-blurred depth integrates as spray (0 = off)")
    ap.add_argument("--neural", action="store_true",
                    help="neural disparity (GPU) instead of SGBM, with a "
                         "left-right consistency check so only cross-"
                         "validated pixels enter the volume (needs --model)")
    ap.add_argument("--model", type=Path, default=None,
                    help="stereo-disparity ONNX model for --neural "
                         "(input [1,6,H,W] RGB-left+RGB-right in 0..1)")
    ap.add_argument("--lr-thresh", type=float, default=1.5,
                    help="max |disp_L - disp_R| (px, model scale) for the "
                         "--neural consistency check")
    ap.add_argument("--edge-grad", type=float, default=4.0,
                    help="drop pixels whose local disparity gradient exceeds "
                         "this (px/px) — kills 'flying pixel' spray at depth "
                         "discontinuities (0 = off)")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the post-extraction outlier/small-cluster "
                         "cleanup")
    ap.add_argument("--wls", action="store_true",
                    help="SGBM + WLS-filtered disparity, gated by the WLS "
                         "confidence map (>= 0.5) so interpolated regions "
                         "don't integrate as fake surfaces. Needs "
                         "cv2.ximgproc. Mutually exclusive with --neural.")
    args = ap.parse_args()
    if args.wls and args.neural:
        ap.error("--wls and --neural are mutually exclusive")

    import open3d as o3d

    prefix = args.out_prefix or f"{args.take}"
    mp4_l, mp4_r = Path(f"{args.take}_L.mp4"), Path(f"{args.take}_R.mp4")

    # ---- calibration + timing (embedded, sidecar fallback) ----
    rec = read_tmf(mp4_l)
    if args.calibration:
        raw = args.calibration.read_bytes()
        try:
            calib = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            calib = calib_blob.unpack(raw)
    else:
        calib = calib_blob.unpack(rec.calib_blob)

    with tempfile.TemporaryDirectory() as td:
        def vts_for(eye):
            sc = Path(f"{args.take}_{eye}.vts")
            if sc.exists():
                return read_vts(str(sc))
            q = Path(td) / f"{eye}.vts"
            q.write_bytes((rec if eye == "L" else read_tmf(mp4_r)).vts_bytes())
            return read_vts(str(q))
        vts_l, vts_r = vts_for("L"), vts_for("R")

    tl = vts_l.sof_timestamps_ns.astype(np.int64)
    tr = vts_r.sof_timestamps_ns.astype(np.int64)
    gate = int(np.median(np.diff(tl)) * 0.5)
    pairs, j = [], 0
    for i, t in enumerate(tl):
        while j + 1 < len(tr) and abs(int(tr[j + 1]) - int(t)) < abs(int(tr[j]) - int(t)):
            j += 1
        if abs(int(tr[j]) - int(t)) <= gate:
            pairs.append((i, j, int(t)))

    if args.no_auto_align:
        rect = Rectification(calib)
    else:
        rect, shift = auto_align(mp4_l, mp4_r, pairs, calib)
        if abs(shift) > 0.5:
            print(f"[auto-align] {shift:+.1f} px vertical offset absorbed")

    if args.max_gyro_dps > 0:
        data = rec.imu_bytes()
        if data:
            import tempfile as _tf
            from trinet_tools.reader import read_imu
            with _tf.NamedTemporaryFile(suffix=".imu") as f:
                f.write(data)
                f.flush()
                imu = read_imu(f.name)
            its = np.asarray(imu.timestamps_ns, dtype=np.int64)
            w = np.degrees(np.linalg.norm(imu.gyro, axis=1))
            def calm(ts):
                m = (its >= ts) & (its <= ts + int(33e6))
                return bool(m.any()) and float(w[m].mean()) <= args.max_gyro_dps
            n0 = len(pairs)
            pairs = [pr for pr in pairs if calm(pr[2])]
            print(f"[motion-gate] kept {len(pairs)}/{n0} calm pairs "
                  f"(<= {args.max_gyro_dps} deg/s)")

    # ---- trajectory: T_G_I(t) ----
    rows = [[float(x) for x in ln.replace(",", " ").split()[:8]]
            for ln in Path(args.traj).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    traj = np.array([r for r in rows if len(r) == 8])
    traj_t = traj[:, 0]
    print(f"{len(pairs)} pairs, {len(traj)} poses")

    # camera-from-IMU: rectified-left frame = R1 * cam0 frame
    T_c0_i = np.array(calib["T_cam0_imu"], dtype=np.float64)
    R1h = np.eye(4)
    R1h[:3, :3] = rect.R1
    T_rect_i = R1h @ T_c0_i

    fx, fy = rect.P1[0, 0], rect.P1[1, 1]
    cx, cy = rect.P1[0, 2], rect.P1[1, 2]

    net = None
    if args.neural:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stereo_depth_neural import HitnetMatcher
        net = HitnetMatcher(args.model)
        print(f"[neural] {args.model.name} on {net.provider} "
              f"({net.w}x{net.h}, LR gate {args.lr_thresh} px)")
        # Depth images are produced at model resolution — scale the TSDF
        # intrinsic to match.
        sx, sy = net.w / 1920.0, net.h / 1080.0
        intr = o3d.camera.PinholeCameraIntrinsic(
            net.w, net.h, fx * sx, fy * sy, cx * sx, cy * sy)
        fx_depth = fx * sx
    else:
        intr = o3d.camera.PinholeCameraIntrinsic(1920, 1080, fx, fy, cx, cy)
        fx_depth = rect.fx
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # RAW SGBM with strict validity, deliberately WITHOUT WLS: the WLS
    # filter interpolates unmatched regions, which looks great in a video
    # but integrates as fake surfaces sprayed along view rays. For mapping,
    # only confidently matched pixels may enter the volume — the TSDF
    # averages away the residual noise.
    nd = 128
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=7,
        P1=8 * 3 * 49, P2=32 * 3 * 49, disp12MaxDiff=1, uniquenessRatio=15,
        speckleWindowSize=200, speckleRange=1, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    wls = matcher_r = None
    if args.wls:
        if not hasattr(cv2, "ximgproc"):
            sys.exit("--wls needs opencv-contrib-python (cv2.ximgproc)")
        matcher_r = cv2.ximgproc.createRightMatcher(sgbm)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.2)
        print("[wls] SGBM + WLS, confidence-gated (>= 0.5)")

    caps = (cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r)))
    cur = [-1, -1]
    frame = [None, None]
    used = 0
    for il, ir, ts in pairs[::args.every]:
        # pose lookup (nearest, 50 ms gate) — the trajectory only covers the
        # initialized span of the take
        t_s = ts / 1e9
        k = int(np.searchsorted(traj_t, t_s))
        if k <= 0 or k >= len(traj) or traj_t[k] - traj_t[k - 1] > 0.25:
            continue
        if not (traj_t[k - 1] - 0.02 <= t_s <= traj_t[k] + 0.02):
            continue
        # linear position interp between the bracketing poses; nearest
        # orientation (pose rate ~15 Hz, calm frames only — rotation between
        # samples is small once the motion gate has run)
        a = (t_s - traj_t[k - 1]) / max(traj_t[k] - traj_t[k - 1], 1e-9)
        pos = (1 - a) * traj[k - 1, 1:4] + a * traj[k, 1:4]
        kq = k if a > 0.5 else k - 1
        for c, want in ((0, il), (1, ir)):
            while cur[c] < want:
                ok, frame[c] = caps[c].read()
                if not ok:
                    break
                cur[c] += 1
        if cur[0] != il or cur[1] != ir:
            break

        left = rect.remap(frame[0], "L")
        right = rect.remap(frame[1], "R")
        if net is not None:
            left = cv2.resize(left, (net.w, net.h))
            right = cv2.resize(right, (net.w, net.h))
            disp = net(left, right)
            # Left-right consistency: HITNet is dense even where it's
            # guessing (occlusions, off-image matches). Compute the
            # right-reference disparity by mirroring both eyes, warp it to
            # the left view, and keep only pixels where the two agree —
            # the neural analogue of SGBM's uniqueness/disp12 checks.
            disp_r = net(cv2.flip(right, 1), cv2.flip(left, 1))[:, ::-1]
            xs = np.arange(net.w)[None, :].repeat(net.h, 0)
            xr = np.clip(np.rint(xs - disp).astype(np.int32), 0, net.w - 1)
            dr_at = np.take_along_axis(disp_r, xr, axis=1)
            disp[np.abs(disp - dr_at) > args.lr_thresh] = 0
            if args.edge_grad > 0:
                # Flying-pixel gate: at depth discontinuities the matcher
                # blends foreground and background disparities into points
                # that hover in free space. High local disparity gradient
                # marks exactly those pixels — drop them (plus a 1-px
                # dilation so the blended border goes too).
                gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
                edge = (np.abs(gx) + np.abs(gy)) > args.edge_grad * 8.0
                edge = cv2.dilate(edge.astype(np.uint8),
                                  np.ones((3, 3), np.uint8)) > 0
                disp[edge] = 0
        else:
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            d16 = sgbm.compute(gl, gr)
            if wls is not None:
                d16r = matcher_r.compute(gr, gl)
                d16f = wls.filter(d16, gl, disparity_map_right=d16r)
                conf = wls.getConfidenceMap()
                disp = d16f.astype(np.float32) / 16.0
                # WLS interpolates unmatched regions along view rays; the
                # confidence map marks where the fill is guesswork — those
                # pixels stay out of the volume.
                disp[conf < 0.5 * 255.0] = 0
            else:
                disp = cv2.medianBlur(d16.astype(np.float32) / 16.0, 5)
        depth = np.zeros_like(disp)
        good = disp > 0.5
        depth[good] = fx_depth * rect.baseline_m / disp[good]
        depth[(depth < args.min_depth) | (depth > args.max_depth)] = 0

        T_G_I = np.eye(4)
        # Hamilton x,y,z,w giving R_ItoG — validated against gravity and a
        # cross-frame floor-consistency test (the JPL/transposed reading puts
        # the floor at inconsistent heights).
        T_G_I[:3, :3] = quat_to_R(traj[kq, 4:8])
        T_G_I[:3, 3] = pos
        T_G_rect = T_G_I @ np.linalg.inv(T_rect_i)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image((depth * 1000).astype(np.uint16)),
            depth_scale=1000.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(T_G_rect))
        used += 1
    for c in caps:
        c.release()
    print(f"integrated {used} keyframes")

    mesh = vol.extract_triangle_mesh()
    if not args.no_clean:
        # Floaters are small disconnected islands: TSDF voxels seen by only
        # a frame or two of leaked depth. Real surfaces (walls, desks) form
        # large connected components — keep only components above ~2000
        # triangles (at 2-3 cm voxels that is roughly a 20x20 cm patch).
        tri_clusters, tri_count, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        tri_count = np.asarray(tri_count)
        small = tri_count[tri_clusters] < 2000
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
        print(f"  [clean] dropped {int(small.sum())} triangles in "
              f"{int((tri_count < 2000).sum())} small clusters "
              f"({len(tri_count)} total)")
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"{prefix}_map.ply", mesh)
    print(f"wrote {prefix}_map.ply "
          f"({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")

    # ---- static views (matplotlib; no GPU/EGL needed) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pcd = vol.extract_point_cloud().voxel_down_sample(args.voxel * 1.5)
    if not args.no_clean and len(pcd.points) > 1000:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                std_ratio=1.8)
    P = np.asarray(pcd.points)
    C = np.clip(np.asarray(pcd.colors), 0.0, 1.0)
    if len(P) > 150000:
        sel = np.random.default_rng(0).choice(len(P), 150000, replace=False)
        P, C = P[sel], C[sel]
    # robust axis limits: a few spray points must not zoom the room out
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    keep = np.all((P >= lo - 0.5) & (P <= hi + 0.5), axis=1)
    P, C = P[keep], C[keep]
    tp = traj[:, 1:4]
    fig = plt.figure(figsize=(16, 6))
    for i, (elev, azim, title) in enumerate(
            [(88, -90, "top-down"), (25, -60, "view 1"), (25, 30, "view 2")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=0.3, linewidths=0)
        ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], "r-", lw=1.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((np.ptp(P[:, 0]), np.ptp(P[:, 1]), max(np.ptp(P[:, 2]), 0.5)))
        ax.axis("off")
    fig.suptitle(f"fused stereo map — {used} keyframes, "
                 f"{len(mesh.vertices)} vertices; red = trajectory")
    fig.savefig(f"{prefix}_map_views.png", dpi=110, bbox_inches="tight")
    print(f"wrote {prefix}_map_views.png")


if __name__ == "__main__":
    main()

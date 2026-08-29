#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of Trinet-Tools.
"""Neural stereo depth video for Trinet stereo recordings (HITNet, ONNX).

Same layout and recording IO as stereo_depth_video.py — rectified [L|R] on
top, metric depth below, optional IMU strip — but disparity comes from a
neural matcher (HITNet, CVPR 2021) instead of SGBM. HITNet propagates slanted
plane hypotheses, so it fills textureless regions SGBM leaves as holes and
keeps crisp edges without a WLS pass.

The bundled model runs at a fixed 1280x720; frames are rectified at full
resolution and downscaled, and the reported disparity stays in model pixels
(fx is scaled to match), so the metric depth is unchanged. On an RTX-class
GPU expect ~0.3 s/frame (CUDA); CPU works but is ~12x slower.

Model: bring your own stereo-disparity ONNX (input [1,6,H,W] = RGB left +
RGB right in 0..1, disparity out) and pass it with --model.

Usage:
    .venv/bin/python scripts/stereo_depth_neural.py TAKE_PREFIX OUT.mp4 \
        [--calibration CALIB] [--compare-sgbm] [--imu] [--ema 0.5] \
        [--min-depth 0.25] [--max-depth 6.0] [--start-s S] [--end-s S] \
        [--model PATH] [--cpu]

CUDA needs the pip CUDA-13 runtime wheels (nvidia-cuda-runtime, nvidia-cublas,
nvidia-curand, nvidia-cufft, nvidia-cudnn-cu13); the script preloads them from
the active environment automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in [_HERE.parent, Path.cwd()]:
    if (_p / "trinet_tools" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(_HERE))

from trinet_tools.stereo_align import Rectification, auto_align   # noqa: E402
from stereo_depth_video import (                                  # noqa: E402
    DepthColorizer,
    ImuStrip,
    compose_canvas,
    load_take,
    make_legend,
    pair_frames,
)


def _preload_cuda_libs() -> None:
    """dlopen the pip-installed CUDA/cuDNN libs so ORT's CUDA EP can load.

    LD_LIBRARY_PATH is read once at process start, so mutating os.environ
    here would be a no-op — preloading with absolute paths is what works.
    ORT >= 1.21 ships preload_dlls() for exactly this; fall back to a manual
    ctypes scan of site-packages/nvidia for older versions.
    """
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
            return
        except Exception:
            pass
    import ctypes
    import site
    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    for root in roots:
        if not root.is_dir():
            continue
        for so in sorted(root.glob("*/lib/lib*.so*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class HitnetMatcher:
    """HITNet ONNX stereo matcher. Returns disparity in MODEL pixels."""

    def __init__(self, model_path: Path, use_cpu: bool = False):
        _preload_cuda_libs()
        import onnxruntime as ort
        providers = (["CPUExecutionProvider"] if use_cpu
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.provider = self.sess.get_providers()[0]
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        _, c, self.h, self.w = inp.shape
        if c != 6:
            raise ValueError(f"expected a 6-channel HITNet model, got {c}ch")

    def __call__(self, bgr_l: np.ndarray, bgr_r: np.ndarray) -> np.ndarray:
        """bgr_l/bgr_r: rectified uint8 BGR at any size -> disparity (h, w)."""
        def prep(img):
            img = cv2.resize(img, (self.w, self.h))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return rgb.transpose(2, 0, 1)
        x = np.concatenate([prep(bgr_l), prep(bgr_r)])[None]
        out = self.sess.run(None, {self.name: x})[0]
        return out.reshape(self.h, self.w)


def main():
    ap = argparse.ArgumentParser(
        description="Neural (HITNet) stereo depth video for Trinet takes")
    ap.add_argument("take", help="take prefix (expects <prefix>_L.mp4/_R.mp4)")
    ap.add_argument("out")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration.json or TBLC .bin (default: the blob "
                         "embedded in the recording)")
    ap.add_argument("--model", type=Path, required=True,
                    help="stereo-disparity ONNX model "
                         "(input [1,6,H,W] RGB-left+RGB-right in 0..1)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPUExecutionProvider")
    ap.add_argument("--compare-sgbm", action="store_true",
                    help="render an SGBM panel next to the HITNet one")
    ap.add_argument("--compare-wls", action="store_true",
                    help="the comparison panel is WLS-filtered SGBM "
                         "(needs cv2.ximgproc) instead of raw SGBM")
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--imu", action="store_true",
                    help="scrolling gyro|accel strip under the depth panels")
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--num-disp", type=int, default=128,
                    help="SGBM search range for --compare-sgbm")
    ap.add_argument("--no-auto-align", action="store_true")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth pair")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="trinet_ndepth_") as td:
        workdir = Path(td)
        mp4_l, mp4_r, vts_l, vts_r, imu, calib = load_take(
            Path(args.take), workdir, args.calibration)

    pairs = pair_frames(vts_l, vts_r)
    if args.start_s > 0 or args.end_s > 0:
        t0 = pairs[0][2]
        lo = t0 + int(args.start_s * 1e9)
        hi = t0 + int(args.end_s * 1e9) if args.end_s > 0 else pairs[-1][2] + 1
        pairs = [p for p in pairs if lo <= p[2] <= hi]
    pairs = pairs[::max(1, args.stride)]
    print(f"{len(pairs)} synchronized pairs")

    if args.no_auto_align:
        rect = Rectification(calib)
    else:
        rect, shift = auto_align(mp4_l, mp4_r,
                                 pairs[::max(1, len(pairs) // 30)], calib)
        if abs(shift) > 0.5:
            print(f"[auto-align] residual vertical offset {shift:+.1f} px "
                  f"absorbed into rectification")

    net = HitnetMatcher(args.model, use_cpu=args.cpu)
    print(f"[stereo-net] {args.model.name} on {net.provider} "
          f"({net.w}x{net.h})")

    # Depth math runs at model resolution: scale rectified fx to model width.
    fx_model = rect.fx * (net.w / rect.size[0])
    baseline = rect.baseline_m
    print(f"baseline {baseline*1000:.1f} mm, fx(model) {fx_model:.1f} px")

    sgbm = wls = matcher_r = None
    if args.compare_sgbm or args.compare_wls:
        nd = (args.num_disp + 15) // 16 * 16
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=nd, blockSize=5,
            P1=8 * 3 * 5 * 5, P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=120, speckleRange=2,
            preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    if args.compare_wls:
        if not hasattr(cv2, "ximgproc"):
            sys.exit("--compare-wls needs opencv-contrib-python (cv2.ximgproc)")
        matcher_r = cv2.ximgproc.createRightMatcher(sgbm)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.2)

    pw, ph = net.w // 2 * 2, net.h // 2 * 2  # render at model res
    pw, ph = pw // 2, ph // 2                # half-size tiles, 2x2 canvas
    strip = ImuStrip(imu, pw * 2) if (args.imu and imu) else None
    legend_h = 44
    W, H = pw * 2, ph * 2 + legend_h + (strip.h if strip else 0)
    legend = make_legend(W, args.min_depth, args.max_depth, legend_h)
    color_net = DepthColorizer(fx_model, baseline,
                               args.min_depth, args.max_depth, args.ema)
    color_sgbm = DepthColorizer(fx_model, baseline,
                                args.min_depth, args.max_depth, args.ema)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", str(args.out)],
        stdin=subprocess.PIPE)

    caps = (cv2.VideoCapture(str(mp4_l)), cv2.VideoCapture(str(mp4_r)))
    cur, frame = [-1, -1], [None, None]
    times, n = [], 0
    for il, ir, ts in pairs:
        for k, want in ((0, il), (1, ir)):
            while cur[k] < want:
                ok, frame[k] = caps[k].read()
                if not ok:
                    break
                cur[k] += 1
        if cur[0] != il or cur[1] != ir:
            break
        rl = rect.remap(frame[0], "l")
        rr = rect.remap(frame[1], "r")
        rl = cv2.resize(rl, (net.w, net.h))
        rr = cv2.resize(rr, (net.w, net.h))

        t0 = time.time()
        disp = net(rl, rr)
        times.append(time.time() - t0)
        panels = [(color_net(disp), "HITNet depth")]
        if sgbm is not None:
            gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
            d16 = sgbm.compute(gl, gr)
            if wls is not None:
                d16r = matcher_r.compute(gr, gl)
                d16 = wls.filter(d16, gl, disparity_map_right=d16r)
                sdisp = d16.astype(np.float32) / 16.0
                panels.append((color_sgbm(sdisp), "SGBM+WLS depth"))
            else:
                sdisp = d16.astype(np.float32) / 16.0
                panels.append((color_sgbm(sdisp), "SGBM depth"))

        canvas = compose_canvas(rl, rr, panels, legend, pw, ph,
                                strip.render(ts) if strip else None)
        ff.stdin.write(canvas.tobytes())
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(pairs)} frames "
                  f"(net {np.median(times)*1000:.0f} ms/frame)")

    for c in caps:
        c.release()
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}: {n} frames, HITNet "
          f"{np.median(times)*1000:.0f} ms/frame median on {net.provider}")


if __name__ == "__main__":
    main()

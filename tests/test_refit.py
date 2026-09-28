# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
import numpy as np

from tests import synth
from trinet_tools.wireless_log import load_logs
from trinet_tools.wireless_utc import TWO32_MS, ClockModel, build_runs

MS = 1_000_000


def _run(tmp_path, cam, hours=2.0, seed=5, **listen):
    phone = synth.SimPhone(seed=seed)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, hours * 3600)
    lb.listen(cam, 0, hours * 3600, **listen)
    runs = build_runs(load_logs([lb.write_jsonl(tmp_path / "r.jsonl")]))
    assert len(runs) == 1
    return phone, runs[0]


def _true_offset_ns(phone, cam, t):
    return phone.elapsed_ns(t) - cam.device_ns(t)


# The fitted offset sits on the lower envelope of the adverts, which is late by
# the minimum radio latency (0.4 ms here) plus a residue of the camera's
# whole-millisecond stamping and of the latency spread (~0.3 ms at 4 adverts/s).
# That bias is reported as latency_bias_ms, not removed.
ENVELOPE_BIAS_MS = 0.3


def test_offset_and_skew_over_two_hours(tmp_path):
    cam = synth.SimCamera(skew_ppm=-30.0)
    phone, run = _run(tmp_path, cam)
    for mode in ("local", "global"):
        m = ClockModel(run, mode)
        t = np.array([600.0, 3600.0, 6600.0])
        dev_ms = cam.device_ns(t) / MS
        off, sig, ext, _ = m.eval(dev_ms)
        err_ms = (off - _true_offset_ns(phone, cam, t)) / MS - phone.latency_min_ms - ENVELOPE_BIAS_MS
        assert np.all(np.abs(err_ms) < 0.3), (mode, err_ms)
        assert np.all(ext == 0)
    xm, ym, b, *_ = ClockModel(run, "global").global_line
    # offset slope = d(elapsed - device)/d(device) = 1/(1+skew) - 1 ~ -skew
    assert abs(b * 1e6 - 30.0) < 0.1


def test_outliers_rejected(tmp_path):
    cam = synth.SimCamera()
    phone, run = _run(tmp_path, cam, hours=1.0)
    bad = np.arange(10, len(run.y_ns), 37)
    run.y_ns = run.y_ns.copy()
    run.y_ns[bad] += 40 * MS                         # whole buckets of late adverts
    m = ClockModel(run, "local")
    assert m.rejected >= len(bad)
    t = np.array([1800.0])
    err = (m.eval(cam.device_ns(t) / MS)[0] - _true_offset_ns(phone, cam, t)) / MS - 0.4 - ENVELOPE_BIAS_MS
    assert abs(err[0]) < 0.3


def test_local_beats_global_on_curved_drift(tmp_path):
    cam = synth.SimCamera(skew_ppm=10.0, curvature_ppm_per_h=3.0)
    phone, run = _run(tmp_path, cam, hours=3.0)
    t = np.linspace(300, 3 * 3600 - 300, 25)
    errs = {}
    for mode in ("local", "global"):
        off = ClockModel(run, mode).eval(cam.device_ns(t) / MS)[0]
        errs[mode] = np.max(np.abs((off - _true_offset_ns(phone, cam, t)) / MS - 0.4 - ENVELOPE_BIAS_MS))
    assert errs["local"] < 0.3
    assert errs["global"] > 2 * errs["local"]


def test_sigma_grows_with_extrapolation(tmp_path):
    cam = synth.SimCamera()
    phone, run = _run(tmp_path, cam, hours=1.0)
    m = ClockModel(run, "local")
    lo, hi = m.observed_ms
    _, sig, ext, _ = m.eval(np.array([hi - 60_000, hi + 600_000, hi + 3_600_000]))
    assert ext[0] == 0 and ext[1] == 600_000 and ext[2] == 3_600_000
    assert sig[0] < sig[1] < sig[2]
    assert sig[2] >= 2e-6 * 3_600_000                  # 2 ppm x 1 h = 7.2 ms


def test_u32_wrap_across_logger_restart(tmp_path):
    # Camera clock crosses 2^32 ms; the phone logs the first part unwrapped and,
    # after a logger restart, the second part raw (k=1 lower).
    cam = synth.SimCamera(uptime0_ms=TWO32_MS - 1_800_000, boot_t=0.0)
    phone = synth.SimPhone(seed=9)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 3600)
    lb.listen(cam, 0, 1700)
    lb.listen(cam, 1900, 3600, log_wrap_k=1, reset_reason="resume")
    runs = build_runs(load_logs([lb.write_jsonl(tmp_path / "w.jsonl")]))
    assert len(runs) == 1 and len(runs[0].segments) == 2
    run = runs[0]
    assert run.seg_shift[("store-a", 2)] == 1
    assert run.x_ms[-1] > TWO32_MS                    # second part moved onto the unwrapped axis
    t = np.array([2500.0])
    err = (ClockModel(run).eval(cam.device_ns(t) / MS)[0] - _true_offset_ns(phone, cam, t)) / MS - 0.4 - ENVELOPE_BIAS_MS
    assert abs(err[0]) < 0.3


def test_repeated_nonce_after_reboot_not_merged(tmp_path):
    a = synth.SimCamera(boot_t=-600.0)
    b = synth.SimCamera(boot_t=2400.0)               # same unit, same nonce, rebooted
    phone = synth.SimPhone(seed=4)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 4800)
    lb.listen(a, 0, 2000)
    lb.listen(b, 2500, 4800, reset_reason="jump")
    runs = build_runs(load_logs([lb.write_jsonl(tmp_path / "n.jsonl")]))
    assert len(runs) == 2


def test_live_mode_uses_phone_fit(tmp_path):
    cam = synth.SimCamera()
    phone, run = _run(tmp_path, cam, hours=0.5)
    m = ClockModel(run, "live")
    lo, hi = m.observed_ms
    assert hi - lo <= 24 * 5000 + 1
    t = np.array([1700.0])
    err = (m.eval(cam.device_ns(t) / MS)[0] - _true_offset_ns(phone, cam, t)) / MS - 0.4 - ENVELOPE_BIAS_MS
    assert abs(err[0]) < 1.0

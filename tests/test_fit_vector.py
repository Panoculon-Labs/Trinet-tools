# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Parity of the Python live fit with the Android SDK's DeviceClockFit.

The fixture ``tests/fixtures/fit_vector.json`` is produced by the SDK's JVM tests
(``trinet-sdk/src/test/resources/fit_vector.json``) and copied here. The test is
skipped when the file is absent. Expected shape (keys are read defensively;
extra keys are ignored)::

    {
      "description": "...",
      "config":   {"bucket_ms": 5000, "max_buckets": 24,          # optional
                   "max_skew_ppm": 200.0, "jump_reset_ms": 10000},
      "samples":  [{"device_ms": 123456, "rx_elapsed_ns": 987654321,
                    "boot_nonce": 60, "timebase_is_master": false}, ...],
                    # boot_nonce / timebase_is_master optional
      "expected": {"offset_at_ref_ns": ..., "ref_device_ns": ...,
                   "skew_ppm": ..., "residual_ms": ...,
                   "buckets": 24 | [{"device_ms":..,"min_offset_ns":..}, ...],
                   "samples": ...}                                 # samples optional
    }

``samples`` may also be given columnar: ``{"device_ms": [...], "rx_elapsed_ns": [...]}``.
"""
import json
from pathlib import Path

import pytest

from trinet_tools.wireless_utc import DeviceClockFit

FIXTURE = Path(__file__).parent / "fixtures" / "fit_vector.json"


def _samples(doc):
    s = doc.get("samples") or doc.get("input") or []
    if isinstance(s, dict):
        n = len(s.get("device_ms", []))
        return [{k: (v[i] if isinstance(v, list) else v) for k, v in s.items()} for i in range(n)]
    # The SDK writes each sample as a [device_ms_raw, rx_elapsed_ns] pair, with
    # boot_nonce / timebase_is_master given once per case.
    out = []
    for x in s:
        if isinstance(x, (list, tuple)):
            x = {"device_ms": x[0], "rx_elapsed_ns": x[1]}
        x = dict(x)
        x.setdefault("boot_nonce", doc.get("boot_nonce", 0))
        x.setdefault("timebase_is_master", doc.get("timebase_is_master", False))
        out.append(x)
    return out


def _vectors(doc):
    if isinstance(doc, list):
        return doc
    if "vectors" in doc or "cases" in doc:
        return doc.get("vectors") or doc.get("cases")
    return [doc]


@pytest.mark.skipif(not FIXTURE.exists(), reason="tests/fixtures/fit_vector.json not present "
                                                 "(copy it from the SDK's test resources)")
def test_live_fit_matches_sdk():
    doc = json.loads(FIXTURE.read_text())
    vectors = _vectors(doc)
    assert vectors, "fixture holds no vectors"
    for v in vectors:
        cfg = v.get("config") or v.get("params") or (doc.get("config", {}) if isinstance(doc, dict) else {})
        fit = DeviceClockFit(bucket_ms=cfg.get("bucket_ms", 5000), max_buckets=cfg.get("max_buckets", 24),
                             max_skew_ppm=cfg.get("max_skew_ppm", 200.0),
                             jump_reset_ms=cfg.get("jump_reset_ms", 10_000))
        for s in _samples(v):
            fit.add_sample(int(s["device_ms"]), int(s["rx_elapsed_ns"]), int(s.get("boot_nonce", 0) or 0),
                           bool(s.get("timebase_is_master", False)))
        exp = v.get("expected") or {}
        got = fit.params
        assert got is not None
        name = v.get("name") or v.get("description", "vector")
        if "ref_device_ns" in exp:
            assert got["ref_device_ns"] == int(exp["ref_device_ns"]), name
        if "offset_at_ref_ns" in exp:
            assert abs(got["offset_at_ref_ns"] - int(exp["offset_at_ref_ns"])) <= 1, name
        if "skew_ppm" in exp:
            assert got["skew_ppm"] == pytest.approx(float(exp["skew_ppm"]), rel=1e-9, abs=1e-9), name
        if "residual_ms" in exp:
            assert got["residual_ms"] == pytest.approx(float(exp["residual_ms"]), rel=1e-9, abs=1e-9), name
        b = exp.get("buckets")
        if isinstance(b, int):
            assert got["buckets"] == b, name
        elif isinstance(b, list):
            mine = [fit.buckets[k] for k in sorted(fit.buckets)]
            assert len(mine) == len(b), name
            for (x, y), e in zip(mine, b):
                assert x == int(e.get("device_ms", e.get("x_ms", x)))
                assert y == int(e.get("min_offset_ns", e.get("min_offset", y)))
        if "samples" in exp:
            assert got["samples"] == int(exp["samples"]), name
        for key in ("first_device_ms", "last_device_ms"):
            if key in exp and key in got:
                assert got[key] == int(exp[key]), (name, key)
        # Exact device-ns -> phone-elapsed-ns conversions through the fit.
        for dev_ns, want in v.get("conversions", []):
            assert fit.to_elapsed_ns(int(dev_ns)) == int(want), (name, dev_ns)


def test_live_fit_port_self_consistent():
    """Without the fixture: the port recovers a known line and handles the u32 wrap."""
    fit = DeviceClockFit()
    skew = 25e-6
    base = (1 << 32) - 60_000                       # crosses the wrap after one minute
    for i in range(0, 240_000, 250):
        dev = base + i
        rx = 5_000_000_000 + int(i * (1 + skew) * 1_000_000) + 400_000
        fit.add_sample(dev & 0xFFFF_FFFF, rx, 7)
    p = fit.params
    assert p["buckets"] == 24 and p["last_device_ms"] > (1 << 32)
    assert p["skew_ppm"] == pytest.approx(25.0, abs=0.05)
    el = fit.to_elapsed_ns((base + 200_000) * 1_000_000)
    assert abs(el - (5_000_000_000 + int(200_000 * (1 + skew) * 1_000_000) + 400_000)) < 50_000

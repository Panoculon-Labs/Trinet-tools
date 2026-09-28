# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
import numpy as np

from tests import synth
from trinet_tools.wireless_log import load_logs
from trinet_tools.wireless_utc import PhoneClock

MS = 1_000_000


def _clock(tmp_path, policy="best", **kw):
    phone = synth.SimPhone(seed=3, **kw)
    lb = synth.LogBuilder(phone)
    lb.clock_refs(0, 3600)
    log = load_logs([lb.write_jsonl(tmp_path / "p.jsonl")])
    return phone, PhoneClock(log.refs_for_boot(("store-a", 1)), policy)


def test_system_interpolation(tmp_path):
    phone, c = _clock(tmp_path, sys_err_ms=250.0)
    assert c.source == "system" and not c.steps
    t = np.array([30.5, 1234.567, 3599.0])
    err = (c.utc_ns(phone.elapsed_ns(t)) - phone.true_utc_ns(t)) / MS
    assert np.allclose(err, 250.0, atol=1.0)
    assert c.sigma_ms(int(phone.elapsed_ns(100.0))) == 100.0


def test_hold_beyond_ends_grows_sigma(tmp_path):
    phone, c = _clock(tmp_path)
    inside = c.sigma_ms(int(phone.elapsed_ns(1800.0)))
    after = c.sigma_ms(int(phone.elapsed_ns(3600.0 + 3600.0)))
    assert abs((after - inside) - 180.0) < 1e-6                     # 50 ppm x 1 h = 180 ms


def test_step_split_and_corrected(tmp_path):
    phone, c = _clock(tmp_path, steps=[(1790.0, 800.0)])
    assert len(c.steps) == 1 and abs(c.steps[0].delta_ms - 800.0) < 2.0
    before, after = phone.elapsed_ns(np.array([900.0, 2700.0]))
    errs = (c.utc_ns([before, after]) - phone.true_utc_ns([900.0, 2700.0])) / MS
    assert abs(errs[0]) < 1.0 and abs(errs[1] - 800.0) < 1.0     # system follows its steps
    assert c.step_penalty_ms(int(before), int(before)) > 790.0    # within 30 min of the step
    _, cc = _clock(tmp_path, "corrected", steps=[(1790.0, 800.0)])
    errs = (cc.utc_ns([before, after]) - phone.true_utc_ns([900.0, 2700.0])) / MS
    assert abs(errs[0] - 800.0) < 2.0 and abs(errs[1] - 800.0) < 1.0   # later step applied backwards


def test_network_and_sntp_preferred(tmp_path):
    phone, c = _clock(tmp_path, sys_err_ms=900.0, network=True, network_err_ms=12.0)
    assert c.source == "network"
    t = np.array([1000.0])
    assert abs((c.utc_ns(phone.elapsed_ns(t)) - phone.true_utc_ns(t))[0] / MS - 12.0) < 1.5
    phone, c = _clock(tmp_path, sys_err_ms=900.0, network=True, network_err_ms=12.0, sntp_rtt_ms=20.0)
    assert c.source == "sntp"
    assert abs((c.utc_ns(phone.elapsed_ns(t)) - phone.true_utc_ns(t))[0] / MS) < 3.0
    assert c.sigma_ms(int(phone.elapsed_ns(1000.0))) < 25.0
    _, cs = _clock(tmp_path, "system", sys_err_ms=900.0, sntp_rtt_ms=20.0)
    assert cs.source == "system"


def test_requested_source_missing_falls_back(tmp_path):
    _, c = _clock(tmp_path, "sntp")
    assert c.source == "system" and c.notes

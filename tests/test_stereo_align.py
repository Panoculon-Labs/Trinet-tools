# SPDX-License-Identifier: MIT
"""Stereo vertical-offset measurement and the field-check verdict."""

import cv2
import numpy as np
import pytest

from trinet_tools.stereo_align import (CHECK_MAX_PX, OK_MAX_PX, calibration_verdict,
                                       pair_y_offsets)


def _textured(seed=0, size=(1080, 1920)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=size, dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def _shifted(img, dx, dy):
    """Content of `img` moved by (dx, dy) pixels, sub-pixel interpolated."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("dy", [0.0, 0.4, -0.7, 1.3, 2.6, -4.2])
def test_pair_y_offsets_recovers_known_subpixel_shift(dy):
    left = _textured()
    right = _shifted(left, -40.0, dy)          # disparity 40 px, right content moved by dy
    d = pair_y_offsets(left, right)
    assert len(d) > 500
    # yL - yR = -dy
    assert np.median(d) == pytest.approx(-dy, abs=0.1)


def test_pair_y_offsets_rejects_wrong_direction_disparity():
    left = _textured(1)
    right = _shifted(left, +40.0, 0.5)         # negative disparity: not a valid stereo match
    assert len(pair_y_offsets(left, right)) == 0


def test_pair_y_offsets_featureless_image_gives_nothing():
    flat = np.full((1080, 1920), 128, np.uint8)
    assert len(pair_y_offsets(flat, flat)) == 0


@pytest.mark.parametrize("offset,verdict", [
    (None, "inconclusive"), (0.0, "ok"), (-1.0, "ok"), (OK_MAX_PX, "ok"),
    (OK_MAX_PX + 0.01, "check"), (-2.5, "check"), (CHECK_MAX_PX, "check"),
    (CHECK_MAX_PX + 0.01, "recalibrate"), (-7.0, "recalibrate"),
])
def test_calibration_verdict(offset, verdict):
    assert calibration_verdict(offset) == verdict

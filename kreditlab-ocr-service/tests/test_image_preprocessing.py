# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing helpers for the `dots-ocr` pipeline. Pure math
plus a small PIL-only branch — no fitz or networked calls."""

import pytest
from PIL import Image

from tensorlake_docai.ocr.image_preprocessing_utils import (
    ceil_by_factor,
    floor_by_factor,
    round_by_factor,
    smart_resize,
    to_rgb,
)

# --- factor rounders ------------------------------------------------------


@pytest.mark.parametrize(
    "n,factor,expected",
    [
        (0, 28, 0),
        (1, 28, 0),  # rounds down
        (14, 28, 0),  # banker-ish rounding: 14/28 = 0.5 -> Python banker = 0
        (15, 28, 28),  # rounds up
        (28, 28, 28),
        (50, 28, 56),
        (100, 28, 112),
    ],
)
def test_round_by_factor(n, factor, expected):
    assert round_by_factor(n, factor) == expected


def test_ceil_by_factor_always_ge_input():
    for n in (1, 27, 28, 29, 100):
        out = ceil_by_factor(n, 28)
        assert out >= n
        assert out % 28 == 0


def test_floor_by_factor_always_le_input():
    for n in (28, 29, 55, 56, 100):
        out = floor_by_factor(n, 28)
        assert out <= n
        assert out % 28 == 0


# --- smart_resize ---------------------------------------------------------


def test_smart_resize_dims_divisible_by_factor():
    h, w = smart_resize(1080, 1920, factor=28)
    assert h % 28 == 0 and w % 28 == 0


def test_smart_resize_respects_max_pixels():
    max_px = 1_000_000
    h, w = smart_resize(4000, 4000, factor=28, max_pixels=max_px)
    assert h * w <= max_px


def test_smart_resize_respects_min_pixels():
    min_px = 50_000
    h, w = smart_resize(50, 50, factor=28, min_pixels=min_px, max_pixels=10_000_000)
    assert h * w >= min_px


def test_smart_resize_preserves_aspect_ratio_approximately():
    h, w = smart_resize(2000, 1000, factor=28)
    # Original ratio 2:1; resized should be close (tolerance for factor rounding)
    assert abs((h / w) - 2.0) < 0.15


def test_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError):
        smart_resize(1, 1000, factor=28)


# --- to_rgb ---------------------------------------------------------------


def test_to_rgb_converts_rgba_with_white_background():
    # Build a 2x1 RGBA image: fully transparent pixel + opaque red pixel
    img = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    img.putpixel((1, 0), (255, 0, 0, 255))

    out = to_rgb(img)
    assert out.mode == "RGB"
    # Transparent pixel should composite to white
    assert out.getpixel((0, 0)) == (255, 255, 255)
    # Opaque red stays red
    assert out.getpixel((1, 0)) == (255, 0, 0)


def test_to_rgb_passthrough_for_non_rgba():
    img = Image.new("L", (4, 4), 128)  # grayscale
    out = to_rgb(img)
    assert out.mode == "RGB"
    assert out.size == (4, 4)


def test_to_rgb_passthrough_for_already_rgb():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    out = to_rgb(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (10, 20, 30)

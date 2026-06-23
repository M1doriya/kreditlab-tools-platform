# SPDX-License-Identifier: Apache-2.0
"""Tests for draw_bboxes pure helpers — page-image selection, citation
detection, and the bbox drawing routine. The PDF/image loading and PIL save
paths are covered by integration; this file focuses on the deterministic
logic that wraps them.
"""

import pytest
from PIL import Image

from tensorlake_docai.postprocess.draw_bboxes import (
    _has_citations,
    _select_page_image,
    draw_bboxes_on_image,
)


# Build a tiny fake page-like object with just `page_number` (matches the
# attribute usage inside _select_page_image).
class _FakePage:
    def __init__(self, page_number):
        self.page_number = page_number


def _img(color="white"):
    return Image.new("RGB", (10, 10), color=color)


# --- _select_page_image ----------------------------------------------------


def test_select_page_image_picks_matching_index():
    images = [_img("red"), _img("green"), _img("blue")]
    page = _FakePage(page_number=2)
    out = _select_page_image(images, page)
    assert out is images[1]


def test_select_page_image_clamps_to_available_range():
    """Page number larger than image count clamps to the last image."""
    images = [_img("red"), _img("green")]
    page = _FakePage(page_number=10)
    out = _select_page_image(images, page)
    assert out is images[-1]


def test_select_page_image_clamps_to_zero():
    images = [_img()]
    page = _FakePage(page_number=0)
    out = _select_page_image(images, page)
    assert out is images[0]


def test_select_page_image_copy_returns_independent_image():
    images = [_img("red")]
    page = _FakePage(page_number=1)
    out = _select_page_image(images, page, copy_image=True)
    assert out is not images[0]
    assert out.size == images[0].size


# --- _has_citations --------------------------------------------------------


def test_has_citations_detects_top_level_citation_key():
    assert _has_citations({"name_citation": {"x1": 1}}) is True


def test_has_citations_detects_nested_citation_key():
    data = {"outer": {"inner": {"field_citation": {"x1": 1}}}}
    assert _has_citations(data) is True


def test_has_citations_detects_in_list_items():
    data = {"items": [{"a": 1}, {"b_citation": "ref"}]}
    assert _has_citations(data) is True


def test_has_citations_returns_false_without_citation_suffix():
    assert _has_citations({"name": "x", "age": 5}) is False


def test_has_citations_non_dict_input():
    assert _has_citations("not a dict") is False
    assert _has_citations(None) is False
    assert _has_citations([1, 2, 3]) is False


# --- draw_bboxes_on_image -------------------------------------------------


def test_draw_bboxes_returns_unchanged_when_no_fragments():
    img = _img()
    out = draw_bboxes_on_image(img, page_fragments=[])
    assert out is img  # short-circuit returns the input unchanged


def test_draw_bboxes_skips_fragments_without_bbox():
    """Fragments missing 'bbox' must not break the drawing loop."""
    img = _img()
    out = draw_bboxes_on_image(
        img,
        page_fragments=[
            {"fragment_type": "text"},  # no bbox
            {"bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 5}, "fragment_type": "text"},
        ],
    )
    # Output is a *copy* — draw_bboxes_on_image always copies when it draws.
    assert out is not img
    assert out.size == img.size
    assert out.getpixel((1, 1)) == (0, 255, 0)


@pytest.mark.parametrize(
    "ftype,expected_rgb",
    [
        ("title", (255, 0, 0)),
        ("text", (0, 255, 0)),
        ("table", (0, 0, 255)),
        ("figure", (255, 0, 255)),
        ("section_header", (255, 255, 0)),
        ("form", (0, 255, 255)),
        ("signature", (255, 165, 0)),
        ("unknown_type", (204, 204, 204)),
    ],
)
def test_draw_bboxes_handles_each_color_key_and_unknown(ftype, expected_rgb):
    img = _img()
    out = draw_bboxes_on_image(
        img,
        page_fragments=[{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "fragment_type": ftype}],
    )
    assert out is not img
    assert out.size == img.size
    assert out.getpixel((0, 0)) == expected_rgb

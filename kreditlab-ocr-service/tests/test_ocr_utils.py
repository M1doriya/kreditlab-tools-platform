# SPDX-License-Identifier: Apache-2.0
"""Tests for pure functions in ocr/utils.py."""

import json

from tensorlake_docai.ocr.utils import (
    create_page_elements_from_dotsocr_output,
    detect_consecutive_repetition,
    map_dotsocr_category_to_fragment_type,
    parse_dotsocr_full_page_output,
    scale_bbox_to_original_coordinates,
)
from tensorlake_docai.pipeline.api import PageFragmentType


# ---------------------------------------------------------------------------
# map_dotsocr_category_to_fragment_type
# ---------------------------------------------------------------------------


def test_map_known_categories():
    assert map_dotsocr_category_to_fragment_type("Table") == PageFragmentType.TABLE
    assert map_dotsocr_category_to_fragment_type("Picture") == PageFragmentType.FIGURE
    assert map_dotsocr_category_to_fragment_type("Text") == PageFragmentType.TEXT
    assert map_dotsocr_category_to_fragment_type("Title") == PageFragmentType.TITLE
    assert (
        map_dotsocr_category_to_fragment_type("Section-header") == PageFragmentType.SECTION_HEADER
    )
    assert map_dotsocr_category_to_fragment_type("Page-header") == PageFragmentType.PAGE_HEADER
    assert map_dotsocr_category_to_fragment_type("Page-footer") == PageFragmentType.PAGE_FOOTER
    assert map_dotsocr_category_to_fragment_type("Footnote") == PageFragmentType.PAGE_FOOTER
    assert map_dotsocr_category_to_fragment_type("Formula") == PageFragmentType.FORMULA
    assert map_dotsocr_category_to_fragment_type("List-item") == PageFragmentType.TEXT
    assert map_dotsocr_category_to_fragment_type("Caption") == PageFragmentType.TEXT


def test_map_unknown_category_returns_text():
    assert map_dotsocr_category_to_fragment_type("Unknown") == PageFragmentType.TEXT
    assert map_dotsocr_category_to_fragment_type("") == PageFragmentType.TEXT


# ---------------------------------------------------------------------------
# parse_dotsocr_full_page_output
# ---------------------------------------------------------------------------


def _make_element(bbox, category="Text", text="hello"):
    return {"bbox": bbox, "category": category, "text": text}


def test_parse_valid_json_list():
    data = [
        _make_element([0, 0, 100, 50]),
        _make_element([0, 60, 100, 120], "Table", "<table></table>"),
    ]
    result = parse_dotsocr_full_page_output(json.dumps(data), scale_factor=1.0)
    assert len(result) == 2
    assert result[0]["category"] == "Text"
    assert result[1]["category"] == "Table"


def test_parse_strips_json_code_fence():
    data = [_make_element([0, 0, 10, 10])]
    fenced = f"```json\n{json.dumps(data)}\n```"
    result = parse_dotsocr_full_page_output(fenced, scale_factor=1.0)
    assert len(result) == 1


def test_parse_strips_known_prefix():
    data = [_make_element([0, 0, 10, 10])]
    prefixed = "```json\n" + json.dumps(data) + "\n```"
    result = parse_dotsocr_full_page_output(prefixed, scale_factor=1.0)
    assert len(result) == 1


def test_parse_empty_string_returns_empty():
    assert parse_dotsocr_full_page_output("", scale_factor=1.0) == []


def test_parse_invalid_json_returns_empty_or_fallback():
    result = parse_dotsocr_full_page_output("not json at all!!!", scale_factor=1.0)
    assert isinstance(result, list)


def test_parse_skips_element_with_bad_bbox():
    bad = [{"bbox": [100, 0, 50, 10], "category": "Text", "text": "x"}]  # x2 < x1
    result = parse_dotsocr_full_page_output(json.dumps(bad), scale_factor=1.0)
    assert result == []


def test_parse_skips_element_missing_bbox():
    data = [{"category": "Text", "text": "no bbox"}]
    result = parse_dotsocr_full_page_output(json.dumps(data), scale_factor=1.0)
    assert result == []


def test_parse_returns_reading_order():
    data = [_make_element([0, 0, 10, 10]), _make_element([0, 20, 10, 30])]
    result = parse_dotsocr_full_page_output(json.dumps(data), scale_factor=1.0)
    assert result[0]["reading_order"] == 0
    assert result[1]["reading_order"] == 1


# ---------------------------------------------------------------------------
# scale_bbox_to_original_coordinates
# ---------------------------------------------------------------------------


def test_scale_bbox_identity():
    elements = [{"bbox": [10, 20, 100, 200]}]
    scale_bbox_to_original_coordinates(elements, image_size=(500, 1000), pdf_size=(500, 1000))
    assert elements[0]["bbox"] == [10, 20, 100, 200]


def test_scale_bbox_halve():
    elements = [{"bbox": [100, 100, 200, 200]}]
    scale_bbox_to_original_coordinates(elements, image_size=(1000, 1000), pdf_size=(500, 500))
    assert elements[0]["bbox"] == [50, 50, 100, 100]


def test_scale_bbox_double():
    elements = [{"bbox": [10, 10, 20, 20]}]
    scale_bbox_to_original_coordinates(elements, image_size=(100, 100), pdf_size=(200, 200))
    assert elements[0]["bbox"] == [20, 20, 40, 40]


def test_scale_bbox_zero_image_size_no_crash():
    elements = [{"bbox": [10, 10, 20, 20]}]
    scale_bbox_to_original_coordinates(elements, image_size=(0, 0), pdf_size=(100, 100))
    # No exception; bbox unchanged (scale factor defaults to 1.0)
    assert isinstance(elements[0]["bbox"], list)


def test_scale_bbox_skips_element_without_bbox():
    elements = [{"category": "Text"}]
    scale_bbox_to_original_coordinates(elements, image_size=(100, 100), pdf_size=(200, 200))
    assert "bbox" not in elements[0]


def test_scale_bbox_skips_bad_bbox():
    elements = [{"bbox": "not-a-list"}]
    scale_bbox_to_original_coordinates(elements, image_size=(100, 100), pdf_size=(200, 200))
    assert elements[0]["bbox"] == "not-a-list"


# ---------------------------------------------------------------------------
# detect_consecutive_repetition
# ---------------------------------------------------------------------------


def test_detect_no_repetition_short_text():
    assert not detect_consecutive_repetition("hello world")


def test_detect_no_repetition_varied_text():
    # Sequential integers produce long text with no repeating substring pattern.
    text = " ".join(str(i) for i in range(600))
    assert not detect_consecutive_repetition(text)


def test_detect_repetition_in_long_text():
    # Repeat a 6-char pattern 400 times → well over 2000 chars, high repetition
    text = "ABABAB" * 400
    assert detect_consecutive_repetition(text)


def test_detect_text_shorter_than_window_returns_false():
    assert not detect_consecutive_repetition("x" * 100, window_size=2000)


# ---------------------------------------------------------------------------
# create_page_elements_from_dotsocr_output
# ---------------------------------------------------------------------------


def _elem(category, text, bbox=None):
    return {
        "bbox": bbox or [0.0, 0.0, 100.0, 50.0],
        "category": category,
        "text": text,
        "reading_order": 0,
    }


def test_create_page_elements_basic():
    raw = [_elem("Text", "Hello world")]
    elements = create_page_elements_from_dotsocr_output(raw)
    assert len(elements) == 1
    assert elements[0].fragment_type == PageFragmentType.TEXT
    assert elements[0].ocr_text == "Hello world"


def test_create_page_elements_table_gets_html():
    raw = [_elem("Table", "<table><tr><td>A</td></tr></table>")]
    elements = create_page_elements_from_dotsocr_output(raw)
    assert len(elements) == 1
    assert elements[0].html is not None


def test_create_page_elements_picture_no_html():
    raw = [_elem("Picture", "")]
    elements = create_page_elements_from_dotsocr_output(raw)
    assert elements[0].html is None


def test_create_page_elements_section_header_hierarchy():
    raw = [_elem("Section-header", "## Introduction")]
    elements = create_page_elements_from_dotsocr_output(raw)
    assert elements[0].hierarchy_level is not None


def test_create_page_elements_empty_input():
    assert create_page_elements_from_dotsocr_output([]) == []


def test_create_page_elements_sets_ref_id():
    raw = [_elem("Text", "hi")]
    elements = create_page_elements_from_dotsocr_output(raw, page_number=3)
    assert elements[0].ref_id == "3.0"

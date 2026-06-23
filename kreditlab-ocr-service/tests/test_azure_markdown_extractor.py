# SPDX-License-Identifier: Apache-2.0
"""Tests for AzureMarkdownExtractor — pure helpers.

We bypass `__init__` (which requires Azure credentials and instantiates a
real DocumentIntelligenceClient) using `object.__new__`. The helpers under
test don't touch `self.client`.
"""

import pytest

from tensorlake_docai.ocr.azure_markdown_extractor import AzureMarkdownExtractor


@pytest.fixture
def extractor():
    # Skip the real __init__; we only test stateless helpers below.
    return object.__new__(AzureMarkdownExtractor)


# --- convert_inches_to_pixels ---------------------------------------------


def test_convert_inches_to_pixels_default_dpi(extractor):
    assert extractor.convert_inches_to_pixels(1.0) == 72
    assert extractor.convert_inches_to_pixels(8.5) == 612


def test_convert_inches_to_pixels_custom_dpi(extractor):
    assert extractor.convert_inches_to_pixels(1.0, dpi=300) == 300


# --- convert_inches_bbox_to_pixels & polygon_to_bbox ---------------------


def test_convert_inches_bbox_to_pixels(extractor):
    polygon = [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]
    out = extractor.convert_inches_bbox_to_pixels(polygon, dpi=72)
    assert out == {"x1": 72, "y1": 72, "x2": 144, "y2": 144}


def test_convert_inches_bbox_to_pixels_rejects_short_polygon(extractor):
    assert extractor.convert_inches_bbox_to_pixels(None) is None
    assert extractor.convert_inches_bbox_to_pixels([1, 2, 3]) is None


def test_polygon_to_bbox(extractor):
    polygon = [10, 5, 20, 5, 20, 30, 10, 30]
    assert extractor.polygon_to_bbox(polygon) == {"x1": 10, "y1": 5, "x2": 20, "y2": 30}


def test_polygon_to_bbox_rejects_short_polygon(extractor):
    assert extractor.polygon_to_bbox([]) is None
    assert extractor.polygon_to_bbox([1, 2, 3]) is None


# --- normalize_checkboxes -------------------------------------------------


def test_normalize_checkboxes_symbols(extractor):
    text = "items: ☒ done ☐ todo"
    out = extractor.normalize_checkboxes(text)
    assert "[x]" in out
    assert "[ ]" in out
    assert "☒" not in out and "☐" not in out


def test_normalize_checkboxes_text_markers(extractor):
    text = "field :selected: :unselected:"
    out = extractor.normalize_checkboxes(text)
    assert "[x]" in out
    assert "[ ]" in out


def test_normalize_checkboxes_empty(extractor):
    assert extractor.normalize_checkboxes("") == ""
    assert extractor.normalize_checkboxes(None) is None


# --- map_role_to_fragment_type --------------------------------------------


@pytest.mark.parametrize(
    "role,expected",
    [
        (None, "text"),
        ("", "text"),
        ("title", "section_header"),
        ("sectionHeading", "section_header"),
        ("pageHeader", "page_header"),
        ("pageFooter", "page_footer"),
        ("pageNumber", "page_number"),
        ("footnote", "page_footer"),
        ("unknown_role", "text"),
    ],
)
def test_map_role_to_fragment_type(extractor, role, expected):
    assert extractor.map_role_to_fragment_type(role) == expected


# --- ensure_proper_spacing ------------------------------------------------


def test_ensure_proper_spacing_appends_blank_line(extractor):
    assert extractor.ensure_proper_spacing("hello").endswith("\n\n")


def test_ensure_proper_spacing_strips_then_appends(extractor):
    assert extractor.ensure_proper_spacing("  hello  ") == "hello\n\n"


def test_ensure_proper_spacing_empty(extractor):
    assert extractor.ensure_proper_spacing("") == ""
    assert extractor.ensure_proper_spacing(None) is None


# --- get_page_dimensions --------------------------------------------------


def test_get_page_dimensions_defaults_when_no_pages(extractor):
    class Result:
        pages = None

    assert extractor.get_page_dimensions(Result()) == (100, 100)


def test_get_page_dimensions_reads_first_page(extractor):
    class Page:
        width = 8.5
        height = 11

    class Result:
        pages = [Page()]

    assert extractor.get_page_dimensions(Result()) == (8, 11)


# --- build_section_hierarchy_map -----------------------------------------


def test_build_section_hierarchy_map_no_sections(extractor):
    class Result:
        sections = None

    assert extractor.build_section_hierarchy_map(Result()) == {}


def test_build_section_hierarchy_map_flat_structure(extractor):
    """Section 0 references two paragraphs at the top level."""

    class Result:
        sections = [
            {"elements": ["/paragraphs/0", "/paragraphs/1"]},
        ]

    out = extractor.build_section_hierarchy_map(Result())
    assert out == {"/paragraphs/0": 0, "/paragraphs/1": 0}


def test_build_section_hierarchy_map_nested_levels(extractor):
    """Section 0 -> section 1, which has its own paragraph at level 1."""

    class Result:
        sections = [
            {"elements": ["/paragraphs/0", "/sections/1"]},
            {"elements": ["/paragraphs/1"]},
        ]

    out = extractor.build_section_hierarchy_map(Result())
    assert out["/paragraphs/0"] == 0
    assert out["/paragraphs/1"] == 1


def test_build_section_hierarchy_map_avoids_infinite_recursion(extractor):
    """Cyclic section references (0 -> 1 -> 0) must not crash."""

    class Result:
        sections = [
            {"elements": ["/sections/1", "/paragraphs/0"]},
            {"elements": ["/sections/0", "/paragraphs/1"]},
        ]

    # Should not hang or recurse forever.
    out = extractor.build_section_hierarchy_map(Result())
    assert "/paragraphs/0" in out
    assert "/paragraphs/1" in out


def test_build_section_hierarchy_map_rebases_minimum_to_zero(extractor):
    """If all paragraphs sit at level >= 1, normalize the minimum to 0."""

    class Result:
        sections = [
            {"elements": ["/sections/1"]},  # section 0 is empty of paragraphs
            {"elements": ["/paragraphs/0"]},  # paragraph 0 is at level 1
        ]

    out = extractor.build_section_hierarchy_map(Result())
    assert out["/paragraphs/0"] == 0


# --- is_section_header_paragraph -----------------------------------------


def test_is_section_header_paragraph_via_hierarchy_map(extractor):
    hierarchy = {"/paragraphs/3": 2}

    class P:
        role = None

    is_header, level = extractor.is_section_header_paragraph(P(), hierarchy, paragraph_index=3)
    assert is_header is True
    assert level == 2


def test_is_section_header_paragraph_fallback_to_role(extractor):
    class P:
        role = "sectionHeading"

    is_header, level = extractor.is_section_header_paragraph(P(), {}, paragraph_index=0)
    assert is_header is True
    assert level == 0


def test_is_section_header_paragraph_returns_false_for_regular(extractor):
    class P:
        role = None

    is_header, level = extractor.is_section_header_paragraph(P(), {}, paragraph_index=0)
    assert is_header is False
    assert level == 0

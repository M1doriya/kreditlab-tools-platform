# SPDX-License-Identifier: Apache-2.0
"""Tests for pure utility methods on AzureMarkdownExtractor.

The class requires Azure credentials at __init__ time, so we patch the
DocumentIntelligenceClient constructor to avoid real network calls.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_extractor():
    """Return an AzureMarkdownExtractor with a mocked Azure client."""
    with patch(
        "tensorlake_docai.ocr.azure_markdown_extractor.DocumentIntelligenceClient"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        from tensorlake_docai.ocr.azure_markdown_extractor import AzureMarkdownExtractor

        return AzureMarkdownExtractor(endpoint="https://fake.endpoint", key="fake-key")


# Instantiate once — these tests don't touch the network.
_extractor = _make_extractor()


# ---------------------------------------------------------------------------
# convert_inches_to_pixels
# ---------------------------------------------------------------------------


def test_convert_inches_to_pixels_default_dpi():
    assert _extractor.convert_inches_to_pixels(1.0) == 72


def test_convert_inches_to_pixels_custom_dpi():
    assert _extractor.convert_inches_to_pixels(2.0, dpi=150) == 300


def test_convert_inches_to_pixels_zero():
    assert _extractor.convert_inches_to_pixels(0.0) == 0


# ---------------------------------------------------------------------------
# convert_inches_bbox_to_pixels
# ---------------------------------------------------------------------------


def test_convert_inches_bbox_to_pixels_basic():
    # polygon: x0, y0, x1, y1, x2, y2, x3, y3 (top-left, top-right, bottom-right, bottom-left)
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    result = _extractor.convert_inches_bbox_to_pixels(polygon, dpi=72)
    assert result == {"x1": 0, "y1": 0, "x2": 72, "y2": 72}


def test_convert_inches_bbox_to_pixels_none_for_empty():
    assert _extractor.convert_inches_bbox_to_pixels(None) is None
    assert _extractor.convert_inches_bbox_to_pixels([]) is None


def test_convert_inches_bbox_to_pixels_too_short():
    assert _extractor.convert_inches_bbox_to_pixels([0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# normalize_checkboxes
# ---------------------------------------------------------------------------


def test_normalize_checkboxes_selected():
    assert _extractor.normalize_checkboxes(":selected:") == "[x]"


def test_normalize_checkboxes_unselected():
    assert _extractor.normalize_checkboxes(":unselected:") == "[ ]"


def test_normalize_checkboxes_unicode_checked():
    assert _extractor.normalize_checkboxes("☒ yes") == "[x] yes"


def test_normalize_checkboxes_unicode_unchecked():
    assert _extractor.normalize_checkboxes("☐ no") == "[ ] no"


def test_normalize_checkboxes_empty_returns_empty():
    assert _extractor.normalize_checkboxes("") == ""
    assert _extractor.normalize_checkboxes(None) is None


def test_normalize_checkboxes_no_op_for_plain_text():
    assert _extractor.normalize_checkboxes("Hello world") == "Hello world"


# ---------------------------------------------------------------------------
# ensure_proper_spacing
# ---------------------------------------------------------------------------


def test_ensure_proper_spacing_adds_newlines():
    result = _extractor.ensure_proper_spacing("text")
    assert result.endswith("\n\n")


def test_ensure_proper_spacing_does_not_double_newlines():
    result = _extractor.ensure_proper_spacing("text\n\n")
    assert result == "text\n\n"


def test_ensure_proper_spacing_empty_returns_empty():
    assert _extractor.ensure_proper_spacing("") == ""
    assert _extractor.ensure_proper_spacing(None) is None


# ---------------------------------------------------------------------------
# map_role_to_fragment_type
# ---------------------------------------------------------------------------


def test_map_role_to_fragment_type_known_roles():
    assert _extractor.map_role_to_fragment_type("title") == "section_header"
    assert _extractor.map_role_to_fragment_type("sectionHeading") == "section_header"
    assert _extractor.map_role_to_fragment_type("pageHeader") == "page_header"
    assert _extractor.map_role_to_fragment_type("pageFooter") == "page_footer"
    assert _extractor.map_role_to_fragment_type("pageNumber") == "page_number"
    assert _extractor.map_role_to_fragment_type("footnote") == "page_footer"


def test_map_role_to_fragment_type_unknown_defaults_to_text():
    assert _extractor.map_role_to_fragment_type("unknown_role") == "text"


def test_map_role_to_fragment_type_none_defaults_to_text():
    assert _extractor.map_role_to_fragment_type(None) == "text"


# ---------------------------------------------------------------------------
# polygon_to_bbox
# ---------------------------------------------------------------------------


def test_polygon_to_bbox_basic():
    # polygon_to_bbox returns raw coordinate min/max (no DPI conversion)
    polygon = [0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0]
    result = _extractor.polygon_to_bbox(polygon)
    assert result is not None
    assert result["x1"] == 0
    assert result["x2"] == 2
    assert result["y1"] == 0
    assert result["y2"] == 1


def test_polygon_to_bbox_none_returns_none():
    assert _extractor.polygon_to_bbox(None) is None


def test_polygon_to_bbox_too_short_returns_none():
    assert _extractor.polygon_to_bbox([0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# get_page_dimensions
# ---------------------------------------------------------------------------


def _ns(**kw):
    return SimpleNamespace(**kw)


def test_get_page_dimensions_from_result_pages():
    page = _ns(width=8.5, height=11.0)
    result = _ns(pages=[page])
    w, h = _extractor.get_page_dimensions(result)
    assert w == 8
    assert h == 11


def test_get_page_dimensions_no_pages_returns_default():
    result = _ns(pages=None)
    assert _extractor.get_page_dimensions(result) == (100, 100)


def test_get_page_dimensions_empty_pages_returns_default():
    result = _ns(pages=[])
    assert _extractor.get_page_dimensions(result) == (100, 100)


# ---------------------------------------------------------------------------
# build_section_hierarchy_map
# ---------------------------------------------------------------------------


def test_build_section_hierarchy_map_no_sections():
    result = _ns(sections=None)
    assert _extractor.build_section_hierarchy_map(result) == {}


def test_build_section_hierarchy_map_empty_sections():
    result = _ns(sections=[])
    assert _extractor.build_section_hierarchy_map(result) == {}


def test_build_section_hierarchy_map_with_paragraphs():
    # Section 0 contains two paragraph refs
    section0 = {"elements": ["/paragraphs/0", "/paragraphs/1"]}
    result = _ns(sections=[section0])
    hierarchy_map = _extractor.build_section_hierarchy_map(result)
    assert "/paragraphs/0" in hierarchy_map
    assert "/paragraphs/1" in hierarchy_map
    assert hierarchy_map["/paragraphs/0"] == 0


def test_build_section_hierarchy_map_nested_sections():
    # Section 0 references section 1; section 1 references a paragraph
    section0 = {"elements": ["/paragraphs/0", "/sections/1"]}
    section1 = {"elements": ["/paragraphs/1"]}
    result = _ns(sections=[section0, section1])
    hierarchy_map = _extractor.build_section_hierarchy_map(result)
    assert hierarchy_map["/paragraphs/0"] == 0
    assert hierarchy_map["/paragraphs/1"] == 1


# ---------------------------------------------------------------------------
# is_section_header_paragraph
# ---------------------------------------------------------------------------


def test_is_section_header_paragraph_from_hierarchy_map():
    para = _ns(role=None)
    is_header, level = _extractor.is_section_header_paragraph(para, {"/paragraphs/3": 1}, 3)
    assert is_header is True
    assert level == 1


def test_is_section_header_paragraph_from_role():
    para = _ns(role="sectionHeading")
    is_header, level = _extractor.is_section_header_paragraph(para, {}, 99)
    assert is_header is True
    assert level == 0


def test_is_section_header_paragraph_plain_text():
    para = _ns(role="paragraph")
    is_header, _ = _extractor.is_section_header_paragraph(para, {}, 0)
    assert is_header is False


# ---------------------------------------------------------------------------
# extract_page_layout_from_pdf_result
# ---------------------------------------------------------------------------


def test_extract_page_layout_from_pdf_result_no_pages():
    result = _ns(pages=None)
    layout = _extractor.extract_page_layout_from_pdf_result(result, page_number=1)
    assert layout["page_fragments"] == []
    assert "dimensions" in layout


def test_extract_page_layout_from_pdf_result_page_not_found():
    page = _ns(page_number=2, width=8.5, height=11.0)
    result = _ns(pages=[page], content=None)
    layout = _extractor.extract_page_layout_from_pdf_result(result, page_number=1)
    assert layout["page_fragments"] == []


def test_extract_page_layout_from_pdf_result_no_content():
    page = _ns(page_number=1, width=8.5, height=11.0)
    result = _ns(pages=[page], content=None)
    layout = _extractor.extract_page_layout_from_pdf_result(result, page_number=1)
    assert layout["page_fragments"] == []
    assert layout["dimensions"][0] > 0


# ---------------------------------------------------------------------------
# create_layout_json_in_span_order
# ---------------------------------------------------------------------------


def test_create_layout_json_in_span_order_no_content():
    result = _ns(pages=[_ns(width=8.5, height=11.0)], content=None)
    layout = _extractor.create_layout_json_in_span_order(result)
    assert layout["page_fragments"] == []


def test_create_layout_json_in_span_order_with_paragraphs():
    span = _ns(offset=0, length=5)
    # Provide a real polygon so polygon_to_bbox returns non-None.
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    bounding_region = _ns(page_number=1, polygon=polygon)
    para = _ns(
        content="Hello",
        role=None,
        spans=[span],
        bounding_regions=[bounding_region],
    )
    result = _ns(
        pages=[_ns(width=8.5, height=11.0)],
        content="Hello world",
        sections=None,
        paragraphs=[para],
        tables=None,
        figures=None,
    )
    layout = _extractor.create_layout_json_in_span_order(result)
    assert len(layout["page_fragments"]) >= 1


def test_create_layout_json_in_span_order_skips_paragraphs_in_tables():
    span = _ns(offset=0, length=5)
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    bounding_region = _ns(page_number=1, polygon=polygon)
    para = _ns(content="cell text", role=None, spans=[span], bounding_regions=[bounding_region])

    cell = _ns(elements=["/paragraphs/0"], row_index=0, column_index=0, content="cell text")
    table_span = _ns(offset=0, length=5)
    table = _ns(
        spans=[table_span],
        cells=[cell],
        caption=None,
        html="<table></table>",
        bounding_regions=[bounding_region],
    )
    result = _ns(
        pages=[_ns(width=8.5, height=11.0)],
        content="cell text",
        sections=None,
        paragraphs=[para],
        tables=[table],
        figures=None,
    )
    layout = _extractor.create_layout_json_in_span_order(result)
    # Paragraph should be excluded (it's inside the table); table fragment should appear
    types = [f.get("fragment_type") for f in layout["page_fragments"]]
    assert "table" in types


# ---------------------------------------------------------------------------
# _extract_table_markdown_and_html
# ---------------------------------------------------------------------------


def test_extract_table_markdown_and_html_from_cells():
    cell_00 = _ns(row_index=0, column_index=0, content="Name")
    cell_01 = _ns(row_index=0, column_index=1, content="Age")
    cell_10 = _ns(row_index=1, column_index=0, content="Alice")
    cell_11 = _ns(row_index=1, column_index=1, content="30")
    table = _ns(cells=[cell_00, cell_01, cell_10, cell_11], caption=None, html=None)

    content, markdown, html = _extractor._extract_table_markdown_and_html(table)
    assert "Name" in markdown
    assert "Alice" in markdown
    assert "|" in markdown  # markdown table format


def test_extract_table_markdown_and_html_uses_azure_html_if_available():
    table = _ns(cells=[], caption=None, html="<table><tr><td>x</td></tr></table>")
    content, markdown, html = _extractor._extract_table_markdown_and_html(table)
    assert html == "<table><tr><td>x</td></tr></table>"


def test_extract_table_markdown_and_html_empty_cells():
    table = _ns(cells=[], caption=None, html=None)
    content, markdown, html = _extractor._extract_table_markdown_and_html(table)
    assert content == ""
    assert markdown == ""


# ---------------------------------------------------------------------------
# _consolidate_figure_content
# ---------------------------------------------------------------------------


def test_consolidate_figure_content_with_caption():
    caption = _ns(content="Figure 1 caption")
    figure = _ns(caption=caption, elements=None)
    result_obj = _ns(paragraphs=None)
    text = _extractor._consolidate_figure_content(figure, result_obj)
    assert "Figure 1 caption" in text


def test_consolidate_figure_content_no_caption():
    figure = _ns(caption=None, elements=None)
    result_obj = _ns(paragraphs=None)
    text = _extractor._consolidate_figure_content(figure, result_obj)
    assert text == ""


def test_consolidate_figure_content_with_paragraph_elements():
    para = _ns(content="Related paragraph")
    caption = _ns(content="Caption")
    figure = _ns(caption=caption, elements=["/paragraphs/0"])
    result_obj = _ns(paragraphs=[para])
    text = _extractor._consolidate_figure_content(figure, result_obj)
    assert "Related paragraph" in text


# ---------------------------------------------------------------------------
# extract_layout_representation (routing dispatcher)
# ---------------------------------------------------------------------------


def test_extract_layout_representation_single_page_uses_create_layout_json():
    page = _ns(page_number=1, width=8.5, height=11.0)
    result = _ns(pages=[page], content=None)
    # Single-page result → falls through to create_layout_json path
    layout = _extractor.extract_layout_representation(result, page_width=612, page_height=792)
    assert "page_fragments" in layout


def test_extract_layout_representation_multi_page_uses_pdf_method():
    page1 = _ns(page_number=1, width=8.5, height=11.0)
    page2 = _ns(page_number=2, width=8.5, height=11.0)
    result = _ns(pages=[page1, page2], content=None)
    layout = _extractor.extract_layout_representation(
        result, page_width=612, page_height=792, page_number=1
    )
    assert "page_fragments" in layout

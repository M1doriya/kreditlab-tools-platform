# SPDX-License-Identifier: Apache-2.0
"""Tests for textract_markdown_extractor.

Covers the pure `_create_fragment_from_layout` helper and the
`TextractMarkdownExtractor` class with the AWS Textractor mocked out.
"""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tensorlake_docai.ocr.textract_markdown_extractor import _create_fragment_from_layout


class _Bbox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h


class _Layout:
    """Minimal stand-in for a textractor LayoutLinearItem."""

    def __init__(self, layout_type, bbox, text="hello", markdown=None, html=None):
        self.layout_type = layout_type
        self.bbox = bbox
        self.text = text
        self._markdown = markdown if markdown is not None else text
        self._html = html

    def to_markdown(self):
        return self._markdown

    def to_html(self):
        if self._html is None:
            raise RuntimeError("no html")
        return self._html


# --- fragment_type mapping ------------------------------------------------


@pytest.mark.parametrize(
    "layout_type,expected_fragment",
    [
        ("LAYOUT_TITLE", "title"),
        ("LAYOUT_HEADER", "page_header"),
        ("LAYOUT_FOOTER", "page_footer"),
        ("LAYOUT_SECTION_HEADER", "section_header"),
        ("LAYOUT_PAGE_NUMBER", "page_number"),
        ("LAYOUT_LIST", "text"),
        ("LAYOUT_FIGURE", "figure"),
        ("LAYOUT_TABLE", "table"),
        ("LAYOUT_KEY_VALUE", "key_value_region"),
        ("LAYOUT_TEXT", "text"),
        ("LAYOUT_UNKNOWN", "text"),  # unmapped -> text
    ],
)
def test_create_fragment_maps_layout_types(layout_type, expected_fragment):
    layout = _Layout(layout_type, _Bbox(0.1, 0.2, 0.3, 0.4), text="x")
    frag = _create_fragment_from_layout(
        layout, reading_order=1, image_width=1000, image_height=2000
    )
    assert frag is not None
    assert frag["fragment_type"] == expected_fragment


# --- bbox scaling ---------------------------------------------------------


def test_create_fragment_scales_bbox_to_72_dpi_by_default():
    """Default flow assumes a 200 DPI render and rescales bboxes to 72 DPI for the UI."""
    layout = _Layout(
        "LAYOUT_TEXT", _Bbox(0.0, 0.0, 1.0, 1.0), text="full page"
    )  # bbox covers entire image
    frag = _create_fragment_from_layout(
        layout, reading_order=1, image_width=2000, image_height=4000
    )
    # 72/200 = 0.36; full image of 2000 -> 720
    assert frag["bbox"]["x1"] == 0
    assert frag["bbox"]["x2"] == int(2000 * 72 / 200)
    assert frag["bbox"]["y2"] == int(4000 * 72 / 200)


def test_create_fragment_preserves_original_resolution_when_flag_set():
    layout = _Layout("LAYOUT_TEXT", _Bbox(0.5, 0.5, 0.5, 0.5))
    frag = _create_fragment_from_layout(
        layout,
        reading_order=1,
        image_width=2000,
        image_height=2000,
        preserve_original_resolution=True,
    )
    assert frag["bbox"] == {"x1": 1000, "y1": 1000, "x2": 2000, "y2": 2000}


# --- content extraction --------------------------------------------------


def test_create_fragment_uses_plain_text_for_headers():
    """Headers must NOT come through `.to_markdown()` — double-formatting risk."""
    layout = _Layout(
        "LAYOUT_SECTION_HEADER", _Bbox(0, 0, 1, 1), text="My Section", markdown="## hi"
    )
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["content"] == "My Section"
    # markdown is NOT used for headers.
    assert frag["content"]["content"] != "## hi"


def test_create_fragment_uses_markdown_for_text():
    layout = _Layout("LAYOUT_TEXT", _Bbox(0, 0, 1, 1), text="raw", markdown="**bold**")
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["content"] == "**bold**"


def test_create_fragment_table_includes_html_and_markdown():
    layout = _Layout(
        "LAYOUT_TABLE",
        _Bbox(0, 0, 1, 1),
        text="table",
        markdown="| x |",
        html="<table></table>",
    )
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["html"] == "<table></table>"
    assert frag["content"]["markdown"] == "| x |"


def test_create_fragment_table_swallows_html_failure():
    """If `to_html()` fails, the fragment is still returned without an html key."""
    layout = _Layout("LAYOUT_TABLE", _Bbox(0, 0, 1, 1), text="t", markdown="md", html=None)
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag is not None
    assert "html" not in frag["content"]


# --- header level detection ---------------------------------------------


def test_create_fragment_title_is_level_zero():
    layout = _Layout("LAYOUT_TITLE", _Bbox(0, 0, 1, 1), text="The Title")
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["level"] == 0


def test_create_fragment_section_header_default_level_one():
    layout = _Layout("LAYOUT_SECTION_HEADER", _Bbox(0, 0, 1, 1), text="Plain Header")
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["level"] == 1


@pytest.mark.parametrize(
    "text,expected_level",
    [
        ("1. Introduction", 1),  # 0 dots -> level 1
        ("1.1 Details", 2),  # 1 dot -> level 2
        ("1.1.1 Deeper", 3),  # 2 dots -> level 3
        ("1.1.1.1 Even deeper (capped)", 3),  # capped at 3
        ("A. Appendix", 2),
    ],
)
def test_create_fragment_section_header_uses_numbering_pattern(text, expected_level):
    layout = _Layout("LAYOUT_SECTION_HEADER", _Bbox(0, 0, 1, 1), text=text)
    frag = _create_fragment_from_layout(layout, reading_order=0, image_width=100, image_height=100)
    assert frag["content"]["level"] == expected_level


# --- error handling -------------------------------------------------------


def test_create_fragment_returns_none_on_internal_failure():
    """A layout missing `.layout_type` should return None rather than raise."""

    class Bad:
        bbox = _Bbox(0, 0, 1, 1)

    out = _create_fragment_from_layout(Bad(), reading_order=0, image_width=100, image_height=100)
    assert out is None


# ==========================================================================
# TextractMarkdownExtractor  (mocked Textractor / AWS)
# ==========================================================================


def _make_layout_obj(layout_type="LAYOUT_TEXT", text="hello"):
    bbox = SimpleNamespace(x=0.0, y=0.0, width=0.5, height=0.1)
    obj = SimpleNamespace(layout_type=layout_type, bbox=bbox, text=text)
    obj.to_markdown = lambda: text
    obj.to_html = lambda: f"<p>{text}</p>"
    return obj


def _make_mock_extractor():
    with patch("tensorlake_docai.ocr.textract_markdown_extractor.Textractor") as mock_cls:
        mock_cls.return_value = MagicMock()
        from tensorlake_docai.ocr.textract_markdown_extractor import TextractMarkdownExtractor

        return TextractMarkdownExtractor(region_name="us-east-1")


_tme = _make_mock_extractor()


# --- __init__ ----------------------------------------------------------------


def test_init_success():
    extractor = _make_mock_extractor()
    assert extractor.region_name == "us-east-1"


def test_init_failure_raises():
    with patch("tensorlake_docai.ocr.textract_markdown_extractor.Textractor") as mock_cls:
        mock_cls.side_effect = Exception("connection refused")
        from tensorlake_docai.ocr.textract_markdown_extractor import TextractMarkdownExtractor

        with pytest.raises(Exception, match="Failed to initialize"):
            TextractMarkdownExtractor()


# --- extract_page_layout_from_textract_result --------------------------------


def test_extract_page_layout_passthrough():
    data = {"dimensions": [612, 792], "page_fragments": [{"type": "text"}]}
    result = _tme.extract_page_layout_from_textract_result(data)
    assert result is data


def test_extract_page_layout_missing_key_returns_defaults():
    data = {"other": "stuff"}
    result = _tme.extract_page_layout_from_textract_result(data)
    assert result["dimensions"] == [612, 792]
    assert result["page_fragments"] == []


# --- process_image -----------------------------------------------------------


def _mock_document(layouts=None):
    page = SimpleNamespace(layouts=layouts or [])
    return SimpleNamespace(pages=[page])


def test_process_image_pil():
    with patch(
        "tensorlake_docai.ocr.textract_markdown_extractor.robust_textract_analyze_document",
        return_value=_mock_document([_make_layout_obj("LAYOUT_TEXT", "Hello")]),
    ):
        _tme.extractor = MagicMock()
        result = _tme.process_image(Image.new("RGB", (100, 200)))

    assert result["dimensions"] == [100, 200]
    assert len(result["page_fragments"]) == 1
    assert result["page_fragments"][0]["fragment_type"] == "text"


def test_process_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (50, 80)).save(buf, format="PNG")
    with patch(
        "tensorlake_docai.ocr.textract_markdown_extractor.robust_textract_analyze_document",
        return_value=_mock_document([]),
    ):
        _tme.extractor = MagicMock()
        result = _tme.process_image(buf.getvalue())

    assert result["page_fragments"] == []
    assert result["dimensions"] == [50, 80]


def test_process_image_rgba_converted():
    with patch(
        "tensorlake_docai.ocr.textract_markdown_extractor.robust_textract_analyze_document",
        return_value=_mock_document([]),
    ):
        _tme.extractor = MagicMock()
        result = _tme.process_image(Image.new("RGBA", (10, 10), (0, 0, 0, 128)))

    assert "page_fragments" in result


def test_process_image_empty_pages():
    with patch(
        "tensorlake_docai.ocr.textract_markdown_extractor.robust_textract_analyze_document",
        return_value=SimpleNamespace(pages=[]),
    ):
        _tme.extractor = MagicMock()
        result = _tme.process_image(Image.new("RGB", (10, 10)))

    assert result["page_fragments"] == []


def test_process_image_unsupported_type_raises():
    with pytest.raises(Exception):
        _tme.process_image(42)


def test_process_image_bytes_convenience():
    buf = io.BytesIO()
    Image.new("RGB", (20, 30)).save(buf, format="PNG")
    with patch(
        "tensorlake_docai.ocr.textract_markdown_extractor.robust_textract_analyze_document",
        return_value=_mock_document([]),
    ):
        _tme.extractor = MagicMock()
        result = _tme.process_image_bytes(buf.getvalue())

    assert result["dimensions"] == [20, 30]

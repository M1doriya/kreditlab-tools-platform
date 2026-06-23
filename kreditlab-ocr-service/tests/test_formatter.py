# SPDX-License-Identifier: Apache-2.0
"""Tests for postprocess.formatter — markdown rendering of page fragments."""

import json

import pytest

from tensorlake_docai.pipeline.api import (
    Chart,
    Figure,
    ListItem,
    PageFragment,
    PageFragmentType,
    ParseRequest,
    SectionHeader,
    Table,
    Text,
)
from tensorlake_docai.postprocess.formatter import (
    bbox_to_dict,
    document_to_markdown,
    downsample_bbox_coordinates_for_parsed_output,
    escape_header_content,
    escape_markdown_content,
    page_fragment_to_markdown,
    page_to_markdown,
)


@pytest.fixture
def request_md():
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        table_output_mode="markdown",
    )


@pytest.fixture
def request_html():
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        table_output_mode="html",
    )


# --- escape_markdown_content ---------------------------------------------


def test_escape_markdown_content_passthrough_for_empty():
    assert escape_markdown_content("") == ""
    assert escape_markdown_content(None) is None


def test_escape_markdown_content_escapes_leading_hash():
    out = escape_markdown_content("# heading-like")
    assert out.startswith("\\#")


def test_escape_markdown_content_leaves_plain_text():
    assert escape_markdown_content("nothing special") == "nothing special"


def test_escape_markdown_content_escapes_inline_hash():
    # The regex requires whitespace on BOTH sides of `#+` to escape — that's the
    # header-marker pattern. `word # other` triggers, `word #tag other` does not.
    out = escape_markdown_content("word # other")
    assert "\\#" in out

    untouched = escape_markdown_content("word #tag other")
    assert "\\#" not in untouched


def test_escape_markdown_content_preserves_newlines():
    text = "line one\nline two"
    assert "\n" in escape_markdown_content(text)


# --- escape_header_content ------------------------------------------------


def test_escape_header_content_escapes_all_hashes():
    assert escape_header_content("a # b ## c") == "a \\# b \\#\\# c"


def test_escape_header_content_empty():
    assert escape_header_content("") == ""
    assert escape_header_content(None) is None


# --- page_fragment_to_markdown ------------------------------------------


def _frag(fragment_type, content):
    return PageFragment(fragment_type=fragment_type, content=content)


def test_list_item_rendered_as_bullet(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.LIST_ITEM, ListItem(content="apples")), request_md
    )
    assert out == "* apples\n"


def test_section_header_uses_level_plus_one(request_md):
    # level 0 -> "#", level 1 -> "##", level 2 -> "###"
    for lvl, marker in [(0, "#"), (1, "##"), (2, "###")]:
        out = page_fragment_to_markdown(
            _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="Hello", level=lvl)),
            request_md,
        )
        assert out.strip().startswith(marker + " ")


def test_section_header_preserves_existing_markdown(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="## already-md", level=1)),
        request_md,
    )
    # Already-markdown headers are returned without re-escaping.
    assert "## already-md" in out
    assert "\\#" not in out


def test_section_header_escapes_inline_hash(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="Topic #1", level=1)),
        request_md,
    )
    # Trailing # in content should be escaped so it doesn't fight header markers.
    assert "\\#" in out


def test_text_fragment_appends_double_newline(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.TEXT, Text(content="plain body")), request_md
    )
    assert out.endswith("\n\n")
    assert "plain body" in out


@pytest.mark.parametrize(
    "ftype",
    [
        PageFragmentType.FORMULA,
        PageFragmentType.FORMULA_CAPTION,
        PageFragmentType.TABLE_CAPTION,
        PageFragmentType.FIGURE_CAPTION,
    ],
)
def test_caption_like_fragments_pass_through(ftype, request_md):
    out = page_fragment_to_markdown(_frag(ftype, Text(content="caption-x")), request_md)
    assert "caption-x" in out


def test_figure_includes_image_and_summary(request_md):
    fig = Figure(content="raw figure text", summary="a summary", image_base64="BASE64")
    out = page_fragment_to_markdown(_frag(PageFragmentType.FIGURE, fig), request_md)
    assert "### Figure" in out
    assert "![Figure](BASE64)" in out
    assert "Figure Summary" in out
    assert "a summary" in out


def test_figure_omits_image_when_no_base64(request_md):
    fig = Figure(content="raw figure text", summary="s")
    out = page_fragment_to_markdown(_frag(PageFragmentType.FIGURE, fig), request_md)
    assert "![Figure]" not in out


def test_figure_collapses_when_summary_matches_content(request_md):
    fig = Figure(content="same", summary="same", image_base64="X")
    out = page_fragment_to_markdown(_frag(PageFragmentType.FIGURE, fig), request_md)
    # No duplicated "Figure Summary" block when summary equals content.
    assert "Figure Summary" not in out


def test_chart_renders_json_block(request_md):
    chart = Chart(
        content=json.dumps([{"x": 1, "y": 2}]),
        image_base64="IMG",
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.CHART, chart), request_md)
    assert "### Chart" in out
    assert "![Chart](IMG)" in out
    assert "```json" in out
    # Single-element list is unwrapped per formatter logic.
    assert '"x": 1' in out


def test_chart_with_malformed_json_does_not_raise(request_md):
    chart = Chart(content="not json")
    out = page_fragment_to_markdown(_frag(PageFragmentType.CHART, chart), request_md)
    # Falls through to empty JSON block but doesn't blow up.
    assert "```json" in out


def test_table_uses_markdown_when_mode_is_markdown(request_md):
    table = Table(
        content="table",
        cells=[],
        html="<table><tr><td>x</td></tr></table>",
        markdown="| x |\n|---|",
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_md)
    assert "| x |" in out
    assert "<table>" not in out


def test_table_uses_html_when_mode_is_html(request_html):
    table = Table(
        content="table", cells=[], html="<table><tr><td>x</td></tr></table>", markdown="| x |"
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_html)
    assert "<table>" in out
    assert "| x |" not in out


def test_table_summary_is_appended(request_md):
    table = Table(
        content="t",
        cells=[],
        html="<table></table>",
        markdown="md",
        summary="key insight",
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_md)
    assert "Table Summary" in out
    assert "key insight" in out


# --- bbox helpers ---------------------------------------------------------


def test_bbox_to_dict():
    assert bbox_to_dict((1.0, 2.0, 3.0, 4.0)) == {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}


def test_downsample_bbox_coordinates_for_parsed_output():
    bbox = {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    out = downsample_bbox_coordinates_for_parsed_output(bbox, scale_factor=2)
    assert out == {"x1": 50, "y1": 100, "x2": 150, "y2": 200}


def test_downsample_none_bbox_returns_none():
    assert downsample_bbox_coordinates_for_parsed_output(None, scale_factor=2) is None


def test_downsample_uses_floor_division():
    # Verify int floor — 3 / 2 = 1, not 1.5
    out = downsample_bbox_coordinates_for_parsed_output({"x1": 3}, scale_factor=2)
    assert out == {"x1": 1}


# --- document/page-level integration --------------------------------------


def test_document_to_markdown_concatenates_pages(request_md):
    from tensorlake_docai.pipeline.api import Page

    page1 = Page(
        page_number=1,
        page_fragments=[_frag(PageFragmentType.TEXT, Text(content="page1"))],
    )
    page2 = Page(
        page_number=2,
        page_fragments=[_frag(PageFragmentType.TEXT, Text(content="page2"))],
    )
    out = document_to_markdown([page1, page2], request_md)
    assert out.index("page1") < out.index("page2")


def test_page_to_markdown_joins_fragments(request_md):
    from tensorlake_docai.pipeline.api import Page

    page = Page(
        page_number=1,
        page_fragments=[
            _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H", level=1)),
            _frag(PageFragmentType.TEXT, Text(content="body")),
        ],
    )
    out = page_to_markdown(page, request_md)
    assert "## H" in out
    assert "body" in out

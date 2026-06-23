# SPDX-License-Identifier: Apache-2.0
"""Tests for header_correction — pure helpers around LLM hierarchy fixup.

The OpenAI call itself is exercised in `_get_openai_corrections`; we test
the surrounding pure-Python logic: header extraction, correction
application, and token usage bookkeeping.
"""

import json

import pytest

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import (
    DocumentLayout,
    PageLayout,
    PageLayoutElement,
)
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest, Usage
from tensorlake_docai.postprocess.header_correction import (
    _apply_corrections,
    _extract_headers,
    _update_token_usage,
    correct_document_headers,
)


def _elem(ftype, text, ro, hier=None):
    return PageLayoutElement(
        bbox=(0.0, 0.0, 1.0, 1.0),
        fragment_type=ftype,
        score=0.99,
        reading_order=ro,
        ocr_text=text,
        hierarchy_level=hier,
    )


def _parse_result(pages):
    layout = DocumentLayout(pages=pages, scale_factor=1.0, total_pages=len(pages))
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf")
    return ParseResult(document_layout=layout, request=req)


# --- _extract_headers ----------------------------------------------------


def test_extract_headers_returns_only_titles_and_section_headers():
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.TITLE, "Big Title", ro=0, hier=0),
            _elem(PageFragmentType.SECTION_HEADER, "Section A", ro=1, hier=1),
            _elem(PageFragmentType.TEXT, "body text", ro=2),
            _elem(PageFragmentType.SECTION_HEADER, "Section B", ro=3, hier=1),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    headers_json, header_map = _extract_headers(pr)

    parsed = json.loads(headers_json)
    assert len(parsed["pages"]) == 1
    headers = parsed["pages"][0]["headers"]
    assert [h["text"] for h in headers] == ["Big Title", "Section A", "Section B"]
    assert headers[0]["type"] == "title"
    assert headers[1]["type"] == "section_header"
    assert len(header_map) == 3


def test_extract_headers_skips_empty_text():
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "   ", ro=0),
            _elem(PageFragmentType.SECTION_HEADER, "real", ro=1),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    headers_json, header_map = _extract_headers(_parse_result([page]))
    assert len(header_map) == 1
    assert "real" in headers_json


def test_extract_headers_strips_markdown_prefix():
    """`## Foo` should be normalized to `Foo` and persisted back to the element."""
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "## My Header", ro=0, hier=1),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    _, header_map = _extract_headers(pr)
    # Element ocr_text was mutated in place to drop the `## ` prefix.
    assert page.elements[0].ocr_text == "My Header"
    assert len(header_map) == 1


def test_extract_headers_keeps_non_header_hashes():
    """`#tag` (no space) is not a markdown header and must NOT be stripped."""
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "#tag-not-header", ro=0),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    _extract_headers(pr)
    assert page.elements[0].ocr_text == "#tag-not-header"


def test_extract_headers_includes_content_preview():
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "Section", ro=0),
            _elem(PageFragmentType.TEXT, "this follows the header", ro=1),
            _elem(PageFragmentType.TEXT, "more content", ro=2),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    headers_json, _ = _extract_headers(pr)
    parsed = json.loads(headers_json)
    preview = parsed["pages"][0]["headers"][0]["content_preview"]
    assert "this follows the header" in preview
    assert "more content" in preview


def test_extract_headers_preview_stops_at_next_header():
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "First", ro=0),
            _elem(PageFragmentType.TEXT, "body of first", ro=1),
            _elem(PageFragmentType.SECTION_HEADER, "Second", ro=2),
            _elem(PageFragmentType.TEXT, "body of second", ro=3),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    headers_json, _ = _extract_headers(_parse_result([page]))
    parsed = json.loads(headers_json)
    first_preview = parsed["pages"][0]["headers"][0]["content_preview"]
    assert "body of first" in first_preview
    assert "body of second" not in first_preview


def test_extract_headers_handles_no_layout():
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf")
    pr = ParseResult(document_layout=None, request=req)
    out = correct_document_headers(pr)
    assert out is pr  # short-circuited


# --- _apply_corrections --------------------------------------------------


def test_apply_corrections_updates_hierarchy_level():
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "A", ro=0, hier=1),
            _elem(PageFragmentType.SECTION_HEADER, "B", ro=1, hier=1),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    _, header_map = _extract_headers(pr)
    keys = list(header_map.keys())

    corrections = [
        {"key": keys[0], "corrected_level": 0},
        {"key": keys[1], "corrected_level": 2},
    ]
    applied = _apply_corrections(pr, corrections, header_map)
    assert applied == 2
    assert page.elements[0].hierarchy_level == 0
    assert page.elements[1].hierarchy_level == 2


def test_apply_corrections_ignores_unknown_keys():
    page = PageLayout(
        elements=[_elem(PageFragmentType.SECTION_HEADER, "A", ro=0, hier=1)],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    _, header_map = _extract_headers(pr)
    applied = _apply_corrections(pr, [{"key": "nonexistent", "corrected_level": 0}], header_map)
    assert applied == 0
    assert page.elements[0].hierarchy_level == 1


@pytest.mark.parametrize("bad_level", [-1, 7, "two", None])
def test_apply_corrections_rejects_out_of_range_levels(bad_level):
    page = PageLayout(
        elements=[_elem(PageFragmentType.SECTION_HEADER, "A", ro=0, hier=1)],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])
    _, header_map = _extract_headers(pr)
    key = list(header_map.keys())[0]
    applied = _apply_corrections(pr, [{"key": key, "corrected_level": bad_level}], header_map)
    assert applied == 0


# --- _update_token_usage -------------------------------------------------


def test_update_token_usage_creates_usage_when_missing():
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf")
    pr = ParseResult(document_layout=None, request=req, usage=None)
    _update_token_usage(pr, 100, 200)
    assert pr.usage is not None
    assert pr.usage.header_correction_input_tokens_used == 100
    assert pr.usage.header_correction_output_tokens_used == 200


def test_update_token_usage_updates_existing_usage():
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf")
    pr = ParseResult(document_layout=None, request=req, usage=Usage(pages_parsed=3))
    _update_token_usage(pr, 50, 60)
    assert pr.usage.pages_parsed == 3
    assert pr.usage.header_correction_input_tokens_used == 50
    assert pr.usage.header_correction_output_tokens_used == 60


# --- correct_document_headers wired with mocked LLM ---------------------


def test_correct_document_headers_applies_mocked_corrections(monkeypatch):
    page = PageLayout(
        elements=[
            _elem(PageFragmentType.SECTION_HEADER, "Alpha", ro=0, hier=1),
            _elem(PageFragmentType.SECTION_HEADER, "Beta", ro=1, hier=1),
        ],
        shape=(1000, 1000),
        page_number=1,
    )
    pr = _parse_result([page])

    def fake_corrections(headers_json, api_key=None):
        # Bump everything to level 3
        parsed = json.loads(headers_json)
        corrections = [
            {"key": h["key"], "current_level": h["current_level"], "corrected_level": 3}
            for p in parsed["pages"]
            for h in p["headers"]
        ]
        return corrections, 42, 7

    monkeypatch.setattr(
        "tensorlake_docai.postprocess.header_correction._get_openai_corrections",
        fake_corrections,
    )

    correct_document_headers(pr)

    assert all(e.hierarchy_level == 3 for e in page.elements)
    assert pr.usage.header_correction_input_tokens_used == 42
    assert pr.usage.header_correction_output_tokens_used == 7

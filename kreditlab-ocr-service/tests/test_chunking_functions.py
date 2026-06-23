# SPDX-License-Identifier: Apache-2.0
"""Tests for chunking_functions — page/fragment/section/patterns chunking.

`chunk_document` and `chunk_pages` orbit around ParseResult / DocumentLayout,
so we test the four chunking strategies directly against constructed
Pages, which is what they consume.
"""

import pytest

from tensorlake.applications import RequestError as RequestException

from tensorlake_docai.extraction.chunking_functions import (
    ChunkingStrategy,
    fragment_chunking,
    page_chunking,
    patterns_chunking,
    section_chunking,
)
from tensorlake_docai.pipeline.api import (
    Page,
    PageFragment,
    PageFragmentType,
    ParseRequest,
    SectionHeader,
    Text,
)


@pytest.fixture
def request_md():
    return ParseRequest(file_name="x.pdf", mime_type="application/pdf")


def _frag(ftype, content, ro):
    return PageFragment(fragment_type=ftype, content=content, reading_order=ro)


def _page(num, fragments):
    return Page(page_number=num, page_fragments=fragments)


# --- page_chunking -------------------------------------------------------


def test_page_chunking_produces_one_chunk_per_page(request_md):
    pages = [
        _page(1, [_frag(PageFragmentType.TEXT, Text(content="page1"), ro=0)]),
        _page(2, [_frag(PageFragmentType.TEXT, Text(content="page2"), ro=0)]),
    ]
    chunks = page_chunking(pages, request_md)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "page1" in chunks[0].content
    assert chunks[1].page_number == 2


def test_page_chunking_tracks_element_ids(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.TEXT, Text(content="a"), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="b"), ro=1),
            ],
        )
    ]
    chunks = page_chunking(pages, request_md)
    assert chunks[0].element_ids == ["1.0", "1.1"]


# --- fragment_chunking ---------------------------------------------------


def test_fragment_chunking_one_chunk_per_fragment(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.TEXT, Text(content="a"), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="b"), ro=1),
            ],
        )
    ]
    chunks = fragment_chunking(pages, request_md)
    assert len(chunks) == 2
    assert "a" in chunks[0].content
    assert chunks[0].element_ids == ["1.0"]
    assert chunks[1].element_ids == ["1.1"]


# --- section_chunking ----------------------------------------------------


def test_section_chunking_splits_at_section_headers(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H1", level=1), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="body 1"), ro=1),
                _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H2", level=1), ro=2),
                _frag(PageFragmentType.TEXT, Text(content="body 2"), ro=3),
            ],
        )
    ]
    chunks = section_chunking(pages, request_md)
    assert len(chunks) == 2
    assert "H1" in chunks[0].content and "body 1" in chunks[0].content
    assert "body 2" not in chunks[0].content
    assert "H2" in chunks[1].content and "body 2" in chunks[1].content


def test_section_chunking_handles_consecutive_headers(request_md):
    """Two headers in a row should stay grouped with the body that follows them."""
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H1", level=1), ro=0),
                _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H2", level=2), ro=1),
                _frag(PageFragmentType.TEXT, Text(content="body"), ro=2),
            ],
        )
    ]
    chunks = section_chunking(pages, request_md)
    # Both headers + the body are one section.
    assert len(chunks) == 1
    assert "H1" in chunks[0].content and "H2" in chunks[0].content and "body" in chunks[0].content


def test_section_chunking_tracks_multi_page_span(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H", level=1), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="part1"), ro=1),
            ],
        ),
        _page(
            2,
            [_frag(PageFragmentType.TEXT, Text(content="part2"), ro=0)],
        ),
    ]
    chunks = section_chunking(pages, request_md)
    assert len(chunks) == 1
    assert chunks[0].page_numbers == [1, 2]


def test_section_chunking_returns_empty_when_no_fragments(request_md):
    chunks = section_chunking([_page(1, [])], request_md)
    assert chunks == []


# --- patterns_chunking ---------------------------------------------------


def test_patterns_chunking_requires_at_least_one_pattern(request_md):
    pages = [_page(1, [_frag(PageFragmentType.TEXT, Text(content="x"), ro=0)])]
    with pytest.raises(RequestException):
        patterns_chunking(pages, request_md, start_patterns=None, end_patterns=None)


def test_patterns_chunking_start_only(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.TEXT, Text(content="header alpha"), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="content 1"), ro=1),
                _frag(PageFragmentType.TEXT, Text(content="header beta"), ro=2),
                _frag(PageFragmentType.TEXT, Text(content="content 2"), ro=3),
            ],
        )
    ]
    chunks = patterns_chunking(pages, request_md, start_patterns=[r"^header "], end_patterns=None)
    assert len(chunks) == 2
    assert "alpha" in chunks[0].content
    assert "content 1" in chunks[0].content
    assert "beta" in chunks[1].content


def test_patterns_chunking_is_case_insensitive(request_md):
    pages = [
        _page(
            1,
            [
                _frag(PageFragmentType.TEXT, Text(content="HEADER A"), ro=0),
                _frag(PageFragmentType.TEXT, Text(content="b"), ro=1),
            ],
        )
    ]
    chunks = patterns_chunking(pages, request_md, start_patterns=["^header"], end_patterns=None)
    assert len(chunks) == 1
    assert "HEADER A" in chunks[0].content


# --- ChunkingStrategy enum ------------------------------------------------


def test_chunking_strategy_values():
    assert ChunkingStrategy.PAGE.value == "page"
    assert ChunkingStrategy.SECTION.value == "section"
    assert ChunkingStrategy.FRAGMENT.value == "fragment"
    assert ChunkingStrategy.PATTERNS.value == "patterns"
    assert ChunkingStrategy.NONE.value == "none"

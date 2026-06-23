# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level routing integration test.

Exercises the real routing predicates against constructed `ParseRequest` and
`ParseResult` objects — no network, no provider clients, no LLMs. The intent
is to lock in the public contract: for each supported `ocr_model`, the
file-convertor stage routes to the correct OCR provider task; once OCR
finishes, the post-OCR stage routes to output formatting, structured
extraction, or VLM extraction based on the request flags + actual document
content.

This is the highest-leverage pipeline test we can write without standing up
the full tensorlake workflow runtime.
"""

import pytest

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import (
    DocumentLayout,
    PageLayout,
    PageLayoutElement,
)
from tensorlake_docai.ocr import OCR_BACKENDS, resolve_ocr_backend
from tensorlake_docai.pipeline import routing
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest

# ---- fixtures -------------------------------------------------------------


def _pdf_request(**overrides) -> ParseRequest:
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",  # base64("hello")
        **overrides,
    )


def _text_request(**overrides) -> ParseRequest:
    return ParseRequest(
        file_name="x.txt",
        mime_type="text/plain",
        file_bytes="aGVsbG8=",
        **overrides,
    )


def _element(fragment_type: PageFragmentType, ocr_text: str = "") -> PageLayoutElement:
    return PageLayoutElement(
        bbox=(0.0, 0.0, 100.0, 100.0),
        fragment_type=fragment_type,
        score=0.99,
        ocr_text=ocr_text,
    )


def _parse_result(request: ParseRequest, elements: list[PageLayoutElement]) -> ParseResult:
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    layout = DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)
    return ParseResult(document_layout=layout, request=request)


# ---- ocr_model → backend resolution --------------------------------------


# Verify that each public ocr_model resolves to a distinct backend class, and
# that the catch-all "non-text file needs OCR" predicate fires for every
# supported model. This is the same invariant the old per-backend predicates
# checked, but verified at the registry level.


@pytest.mark.parametrize("ocr_model", sorted(OCR_BACKENDS))
def test_each_ocr_model_resolves_to_a_distinct_backend(ocr_model):
    req = _pdf_request(ocr_model=ocr_model)
    assert routing.file_convertor_should_go_to_ocr(req)
    backend_cls = resolve_ocr_backend(ocr_model)
    # Each model must map to a unique class
    other_classes = {resolve_ocr_backend(m) for m in OCR_BACKENDS if m != ocr_model}
    assert backend_cls not in other_classes


def test_unknown_ocr_model_falls_back_to_default():
    """Unknown values should not crash — they fall back to DEFAULT_OCR_MODEL.
    pydantic's Literal validates real user input at the API boundary; this
    guards against internal callers passing through stale strings."""
    assert resolve_ocr_backend("model99") is resolve_ocr_backend("azure-di")
    assert resolve_ocr_backend(None) is resolve_ocr_backend("azure-di")


def test_text_file_skips_ocr_and_goes_to_output_formatter():
    req = _text_request()
    assert routing.file_convertor_should_go_to_output_formatter(req)
    assert not routing.file_convertor_should_go_to_ocr(req)


def test_text_file_with_structured_extraction_goes_to_structured_extraction():
    from tensorlake_docai.pipeline.api import StructuredExtractionRequest

    req = _text_request(
        structured_extraction_requests=[
            StructuredExtractionRequest(
                json_schema='{"type":"object"}',
                schema_name="x",
            )
        ]
    )
    assert routing.file_convertor_should_go_to_structured_extraction(req)
    assert not routing.file_convertor_should_go_to_output_formatter(req)


# ---- post-OCR routing: bare parse → output formatter ---------------------


def test_post_ocr_bare_request_goes_to_output_formatter():
    req = _pdf_request()
    parse_result = _parse_result(req, [_element(PageFragmentType.TEXT, "hello")])

    assert routing.ocr_should_go_to_output_formatter(req)
    assert not routing.ocr_should_go_to_structured_extraction(req, parse_result)
    assert not routing.ocr_should_go_to_vlm_extraction(req, parse_result)


# ---- post-OCR routing: structured extraction requested -------------------


def test_post_ocr_structured_extraction_routes_to_se():
    from tensorlake_docai.pipeline.api import StructuredExtractionRequest

    req = _pdf_request(
        ocr_model="azure-di",
        structured_extraction_requests=[
            StructuredExtractionRequest(
                json_schema='{"type":"object"}',
                schema_name="x",
            )
        ],
    )
    parse_result = _parse_result(req, [_element(PageFragmentType.TEXT, "hello")])

    assert routing.ocr_should_go_to_structured_extraction(req, parse_result)
    assert not routing.ocr_should_go_to_output_formatter(req)
    assert not routing.ocr_should_go_to_vlm_extraction(req, parse_result)


# ---- post-OCR routing: VLM tasks gated by document content ---------------


def test_table_summarization_skipped_when_no_tables_found():
    req = _pdf_request(ocr_model="azure-di", table_summarization=True)
    parse_result = _parse_result(req, [_element(PageFragmentType.TEXT, "no tables here")])

    # table_summarization=True but document has no tables → don't go to VLM
    assert not routing.ocr_should_go_to_vlm_extraction(req, parse_result)
    assert routing.ocr_should_go_to_output_formatter(req) is False  # gate is set
    # Note: ocr_should_go_to_output_formatter is purely flag-driven, so it
    # returns False whenever any VLM-like flag is set, regardless of content.


def test_table_summarization_routes_to_vlm_when_tables_present():
    req = _pdf_request(ocr_model="azure-di", table_summarization=True)
    parse_result = _parse_result(req, [_element(PageFragmentType.TABLE)])

    assert routing.ocr_should_go_to_vlm_extraction(req, parse_result)


def test_figure_summarization_routes_to_vlm_when_figures_present():
    req = _pdf_request(ocr_model="azure-di", figure_summarization=True)
    parse_result = _parse_result(req, [_element(PageFragmentType.FIGURE)])

    assert routing.ocr_should_go_to_vlm_extraction(req, parse_result)


# ---- should_skip_ocr -----------------------------------------------------


def test_should_skip_ocr_default_false():
    assert not routing.should_skip_ocr(_pdf_request())


def test_should_skip_ocr_when_structured_extraction_says_so():
    from tensorlake_docai.pipeline.api import StructuredExtractionRequest

    req = _pdf_request(
        structured_extraction_requests=[
            StructuredExtractionRequest(
                json_schema='{"type":"object"}',
                schema_name="x",
                skip_ocr=True,
            )
        ]
    )
    assert routing.should_skip_ocr(req)

# SPDX-License-Identifier: Apache-2.0
"""Tests for pure-logic routing predicates and helpers in pipeline/routing.py."""

import json

import pytest

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import DocumentLayout, PageLayout, PageLayoutElement
from tensorlake_docai.pipeline.api import (
    ClassificationRequest,
    PageClassDefinition,
    PageFragmentType,
    ParseRequest,
    StructuredExtractionRequest,
)
from tensorlake_docai.pipeline.routing import (
    _check_has_table_and_figure_and_chart_and_form,
    create_classification_choice_and_prompt,
    dots_ocr_should_go_to_output_formatter,
    dots_ocr_should_go_to_structured_extraction,
    dots_ocr_should_go_to_vlm_extraction,
    file_convertor_should_go_to_ocr,
    file_convertor_should_go_to_output_formatter,
    file_convertor_should_go_to_structured_extraction,
    file_convertor_should_go_to_vlm_extraction,
    handle_processing_error,
    is_markdown_table,
    markdown_to_html_table,
    ocr_should_go_to_output_formatter,
    ocr_should_go_to_structured_extraction,
    ocr_should_go_to_vlm_extraction,
    should_route_to_table_merging,
    should_skip_ocr,
    vlm_extraction_should_go_to_output_formatter,
    vlm_extraction_should_go_to_structured_extraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(**kwargs) -> ParseRequest:
    defaults = dict(file_name="test.pdf", mime_type="application/pdf")
    defaults.update(kwargs)
    return ParseRequest(**defaults)


def _se_req(skip_ocr=False) -> StructuredExtractionRequest:
    return StructuredExtractionRequest(
        json_schema='{"type": "object"}',
        schema_name="test",
        skip_ocr=skip_ocr,
    )


def _layout_with(*fragment_types) -> DocumentLayout:
    elements = [
        PageLayoutElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            fragment_type=ft,
            score=1.0,
        )
        for ft in fragment_types
    ]
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    return DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)


def _parse_result(request: ParseRequest, layout: DocumentLayout = None) -> ParseResult:
    if layout is None:
        layout = DocumentLayout(pages=[], scale_factor=1.0, total_pages=0)
    return ParseResult(request=request, document_layout=layout)


# ---------------------------------------------------------------------------
# is_markdown_table
# ---------------------------------------------------------------------------


def test_is_markdown_table_standard():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert is_markdown_table(md)


def test_is_markdown_table_false_for_plain_text():
    assert not is_markdown_table("Just some text")


def test_is_markdown_table_false_for_empty():
    assert not is_markdown_table("")
    assert not is_markdown_table(None)


def test_is_markdown_table_single_line():
    assert not is_markdown_table("| A |")


# ---------------------------------------------------------------------------
# markdown_to_html_table
# ---------------------------------------------------------------------------


def test_markdown_to_html_table_produces_table_tag():
    md = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
    html = markdown_to_html_table(md)
    assert "<table>" in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert "<th>Name</th>" in html
    assert "<td>Alice</td>" in html


def test_markdown_to_html_table_returns_input_when_no_separator():
    plain = "no separator here"
    assert markdown_to_html_table(plain) == plain


def test_markdown_to_html_table_with_pre_table_text():
    md = "Caption text\n| A | B |\n|---|---|\n| 1 | 2 |"
    html = markdown_to_html_table(md)
    assert "<div>Caption text</div>" in html
    assert "<table>" in html


# ---------------------------------------------------------------------------
# handle_processing_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("rate limit exceeded", "busy"),
        ("quota exceeded", "usage limit"),
        ("timed out processing", "timed out"),
        ("credentials invalid", "authentication"),
        ("corrupted file invalid pdf", "corrupted"),
        ("out of memory", "large"),
        ("unexpected failure", "processing failed"),
    ],
)
def test_handle_processing_error_categorisation(message, expected_fragment):
    result = handle_processing_error(Exception(message), "test context")
    assert expected_fragment.lower() in result.lower()


# ---------------------------------------------------------------------------
# create_classification_choice_and_prompt
# ---------------------------------------------------------------------------


def _cls_def(name, desc="desc"):
    return PageClassDefinition(class_name=name, description=desc)


def test_create_classification_choice_and_prompt_multi_class():
    schema_json, choices, prompt = create_classification_choice_and_prompt(
        [_cls_def("invoice"), _cls_def("receipt")], "multi_class"
    )
    assert "invoice" in choices
    assert "receipt" in choices
    assert "unclassified" in choices
    schema = json.loads(schema_json)
    assert "page_class" in schema["properties"]


def test_create_classification_choice_and_prompt_multi_label():
    schema_json, choices, prompt = create_classification_choice_and_prompt(
        [_cls_def("page_a"), _cls_def("page_b")], "multi_label"
    )
    schema = json.loads(schema_json)
    assert "page_classes" in schema["properties"]
    assert "confidence" in schema["properties"]


def test_create_classification_choice_and_prompt_empty_raises():
    from tensorlake.applications import RequestError as RequestException

    with pytest.raises(RequestException):
        create_classification_choice_and_prompt([], "multi_class")


# ---------------------------------------------------------------------------
# _check_has_table_and_figure_and_chart_and_form
# ---------------------------------------------------------------------------


def test_check_has_table_detects_table():
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(_req(), layout)
    has_table, has_figure, has_chart, has_form = _check_has_table_and_figure_and_chart_and_form(
        result
    )
    assert has_table
    assert not has_figure


def test_check_has_figure_detects_figure():
    layout = _layout_with(PageFragmentType.FIGURE)
    result = _parse_result(_req(), layout)
    _, has_figure, _, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_figure


def test_check_has_chart_detects_chart():
    layout = _layout_with(PageFragmentType.CHART)
    result = _parse_result(_req(), layout)
    _, _, has_chart, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_chart


def test_check_has_form_detects_form():
    layout = _layout_with(PageFragmentType.FORM)
    result = _parse_result(_req(), layout)
    _, _, _, has_form = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_form


def test_check_all_false_for_empty_layout():
    result = _parse_result(_req())
    assert _check_has_table_and_figure_and_chart_and_form(result) == (False, False, False, False)


def test_check_detects_markdown_table_in_text_element():
    md_table = "| A | B |\n|---|---|\n| 1 | 2 |"
    elements = [
        PageLayoutElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            fragment_type=PageFragmentType.TEXT,
            score=1.0,
            ocr_text=md_table,
        )
    ]
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    layout = DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)
    result = _parse_result(_req(), layout)
    has_table, _, _, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_table


# ---------------------------------------------------------------------------
# should_skip_ocr
# ---------------------------------------------------------------------------


def test_should_skip_ocr_all_skip():
    req = _req(structured_extraction_requests=[_se_req(skip_ocr=True), _se_req(skip_ocr=True)])
    assert should_skip_ocr(req)


def test_should_skip_ocr_mixed_returns_false():
    req = _req(structured_extraction_requests=[_se_req(skip_ocr=True), _se_req(skip_ocr=False)])
    assert not should_skip_ocr(req)


def test_should_skip_ocr_no_requests_returns_false():
    assert not should_skip_ocr(_req())


# ---------------------------------------------------------------------------
# file_convertor_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_file_convertor_text_no_extras_goes_to_output_formatter():
    req = _req(mime_type="text/plain")
    assert file_convertor_should_go_to_output_formatter(req)
    assert not file_convertor_should_go_to_ocr(req)


def test_file_convertor_pdf_no_skip_ocr_goes_to_ocr():
    req = _req(mime_type="application/pdf")
    assert file_convertor_should_go_to_ocr(req)
    assert not file_convertor_should_go_to_output_formatter(req)


def test_file_convertor_text_with_se_goes_to_structured_extraction():
    req = _req(mime_type="text/plain", structured_extraction_requests=[_se_req()])
    assert file_convertor_should_go_to_structured_extraction(req)
    assert not file_convertor_should_go_to_output_formatter(req)


def test_file_convertor_text_with_page_classification_goes_to_vlm():
    cls_req = ClassificationRequest(
        class_definitions=[_cls_def("invoice")], classification_type="multi_class"
    )
    req = _req(mime_type="text/plain", page_classification_request=cls_req)
    assert file_convertor_should_go_to_vlm_extraction(req)
    assert not file_convertor_should_go_to_output_formatter(req)


# ---------------------------------------------------------------------------
# ocr_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_ocr_should_go_to_output_formatter_when_no_extras():
    req = _req()
    assert ocr_should_go_to_output_formatter(req)


def test_ocr_should_go_to_output_formatter_false_with_se():
    req = _req(structured_extraction_requests=[_se_req()])
    assert not ocr_should_go_to_output_formatter(req)


def test_ocr_should_go_to_vlm_when_figure_present():
    req = _req(figure_summarization=True)
    layout = _layout_with(PageFragmentType.FIGURE)
    result = _parse_result(req, layout)
    assert ocr_should_go_to_vlm_extraction(req, result)


def test_ocr_should_go_to_vlm_false_when_no_figure():
    req = _req(figure_summarization=True)
    result = _parse_result(req)
    assert not ocr_should_go_to_vlm_extraction(req, result)


def test_ocr_should_go_to_structured_extraction():
    req = _req(structured_extraction_requests=[_se_req()])
    result = _parse_result(req)
    assert ocr_should_go_to_structured_extraction(req, result)


def test_ocr_should_not_go_to_se_when_vlm_needed():
    req = _req(
        structured_extraction_requests=[_se_req()],
        figure_summarization=True,
    )
    layout = _layout_with(PageFragmentType.FIGURE)
    result = _parse_result(req, layout)
    assert not ocr_should_go_to_structured_extraction(req, result)


# ---------------------------------------------------------------------------
# should_route_to_table_merging
# ---------------------------------------------------------------------------


def test_should_route_to_table_merging_true():
    req = _req(table_merging=True)
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(req, layout)
    assert should_route_to_table_merging(req, result)


def test_should_route_to_table_merging_false_no_table():
    req = _req(table_merging=True)
    result = _parse_result(req)
    assert not should_route_to_table_merging(req, result)


def test_should_route_to_table_merging_false_flag_off():
    req = _req(table_merging=False)
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(req, layout)
    assert not should_route_to_table_merging(req, result)


# ---------------------------------------------------------------------------
# vlm_extraction_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_vlm_should_go_to_output_formatter_when_no_se():
    assert vlm_extraction_should_go_to_output_formatter(_req())


def test_vlm_should_go_to_structured_extraction():
    req = _req(structured_extraction_requests=[_se_req(skip_ocr=False)])
    assert vlm_extraction_should_go_to_structured_extraction(req)


def test_vlm_should_not_go_to_se_when_skip_ocr():
    req = _req(structured_extraction_requests=[_se_req(skip_ocr=True)])
    assert not vlm_extraction_should_go_to_structured_extraction(req)


# ---------------------------------------------------------------------------
# dots_ocr_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_dots_ocr_output_formatter_when_nothing_needed():
    req = _req()
    result = _parse_result(req)
    assert dots_ocr_should_go_to_output_formatter(req, result)


def test_dots_ocr_output_formatter_false_with_se():
    req = _req(structured_extraction_requests=[_se_req()])
    result = _parse_result(req)
    assert not dots_ocr_should_go_to_output_formatter(req, result)


def test_dots_ocr_vlm_when_table_summarization_and_table_present():
    req = _req(table_summarization=True)
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(req, layout)
    assert dots_ocr_should_go_to_vlm_extraction(req, result)


def test_dots_ocr_se_when_se_needed_and_no_vlm():
    req = _req(structured_extraction_requests=[_se_req()])
    result = _parse_result(req)
    assert dots_ocr_should_go_to_structured_extraction(req, result)

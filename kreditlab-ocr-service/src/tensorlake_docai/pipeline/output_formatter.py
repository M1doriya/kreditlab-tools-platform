# SPDX-License-Identifier: Apache-2.0
"""
Output Formatter

This module provides a function to format a ParseResult from previous steps into
the final ParsedDocumentRef output format, handling all token aggregation
consistently.
"""

import json
from typing import Optional, List, Dict, Any

from tensorlake_docai.pipeline.api import (
    StructuredData,
    ParsedDocumentRef,
    ParsedDocument,
    Usage,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.extraction.chunking_functions import chunk_document


def format_final_output(
    result: ParseResult,
) -> Optional[dict]:
    """
    Format the final output from any pipeline stage.

    Args:
        result: The ParseResult containing document layout and request info

    Returns:
        Dict representation of ParsedDocumentRef
    """
    print("=== format_final_output ===")
    parsed_document = _create_parsed_document(result)
    _attach_structured_data(parsed_document, result)
    usage = _calculate_usage(result)
    final_output = _create_final_output(parsed_document, usage)
    print("=== end of format_final_output ===")

    return final_output


def _create_parsed_document(result: ParseResult) -> ParsedDocument:
    """Create ParsedDocument from ParseResult using existing chunking logic."""
    parsed_document = chunk_document(result)
    parsed_document.total_pages = (
        result.document_layout.total_pages if result.document_layout else 0
    )
    parsed_document.docx_converted_pdf_base64 = result.docx_converted_pdf_base64

    if result.form_filling_result:
        parsed_document.filled_pdf_base64 = result.form_filling_result.filled_pdf_base64
        parsed_document.filled_docx_base64 = result.form_filling_result.filled_docx_base64
        parsed_document.form_filling_metadata = result.form_filling_result.metadata

    return parsed_document


def _attach_structured_data(parsed_document: ParsedDocument, result: ParseResult) -> None:
    """
    Attach structured data to the parsed document.
    """
    if not result.structured_outputs_by_page:
        return

    structured_data_results = _process_structured_outputs(result.structured_outputs_by_page)

    if structured_data_results:
        parsed_document.structured_data = structured_data_results


def _process_structured_outputs(
    structured_outputs_by_page: Dict[Any, Any],
) -> List[StructuredData]:
    """
    Process structured outputs from `structured_outputs_by_page`.
    """
    structured_data_pages = []

    for page_key, structured_output in structured_outputs_by_page.items():
        try:
            # page_key shape is (page_number_or_tuple, chunk_index); unpack to a flat page list.
            if isinstance(page_key, tuple) and len(page_key) == 2:
                pages = list(page_key[0]) if isinstance(page_key[0], tuple) else [page_key[0]]
            else:
                pages = list(page_key) if isinstance(page_key, tuple) else [page_key]

            for schema_name, result_data in structured_output.items():
                structured_data_pages.append(
                    StructuredData(
                        data=(
                            json.loads(result_data) if isinstance(result_data, str) else result_data
                        ),
                        page_numbers=pages,
                        schema_name=schema_name,
                    )
                )
        except Exception as e:
            print(f"Error processing structured output: {e}\n\n{structured_output}")
            raise Exception(f"Error processing structured output: {e}")

    return structured_data_pages


def _calculate_usage(
    result: ParseResult,
) -> Usage:
    """
    Calculate total usage metrics by aggregating tokens from VLM and LLM tasks.
    """
    u = result.usage

    # Calculate pages parsed
    pages_parsed = 0
    if result.document_layout and result.document_layout.pages:
        pages_parsed = len(result.document_layout.pages)
    elif u and u.pages_parsed:
        pages_parsed = u.pages_parsed

    # Extract request flags
    detect_signature = result.request.detect_signature if result.request else False

    return Usage(
        pages_parsed=pages_parsed,
        signature_detection=detect_signature,
        ocr_input_tokens_used=(u.ocr_input_tokens_used or 0) if u else 0,
        ocr_output_tokens_used=(u.ocr_output_tokens_used or 0) if u else 0,
        extraction_input_tokens_used=(u.extraction_input_tokens_used or 0) if u else 0,
        extraction_output_tokens_used=(u.extraction_output_tokens_used or 0) if u else 0,
        summarization_input_tokens_used=(u.summarization_input_tokens_used or 0) if u else 0,
        summarization_output_tokens_used=(u.summarization_output_tokens_used or 0) if u else 0,
        header_correction_input_tokens_used=(
            (u.header_correction_input_tokens_used or 0) if u else 0
        ),
        header_correction_output_tokens_used=(
            (u.header_correction_output_tokens_used or 0) if u else 0
        ),
    )


def _create_final_output(parsed_document: ParsedDocument, usage: Usage) -> Optional[dict]:
    """Create the final ParsedDocumentRef output."""
    output = ParsedDocumentRef(document=parsed_document.model_dump(), usage=usage).model_dump()
    return output

# SPDX-License-Identifier: Apache-2.0
"""
Module with utility functions to use by workflow steps.
"""

import asyncio
import base64
import json
import os
import re
import time
from typing import List, Optional, Tuple

import boto3
import requests
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest
from tensorlake_docai.models.intermediate_objects import FileData, ParseResult
from tensorlake.applications import RequestError as RequestException

KEY_PATH_PREFIX = "workflow-step-output"

# File type mapping - shared with file_convertor.py
FILE_TYPE_MAPPING = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsm",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.apple.keynote": "key",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/tif": "tiff",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/csv": "csv",
    "text/html": "html",
    "text/xml": "xml",
    "application/xml": "xml",
    "application/msword": "doc",
    "text/rtf": "rtf",
    "application/rtf": "rtf",
    "application/pkcs7-mime": "p7m",
    "application/x-pkcs7-mime": "p7m",
    "application/pkcs7-signature": "p7m",
}


def is_markdown_table(text: str) -> bool:
    if not text:
        return False
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False
    # Check for separator line with dashes and pipes
    for line in lines:
        if re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", line) and "-" in line and "|" in line:
            return True

    # Check for ! separator tables
    # Heuristic: multiple lines with multiple !
    bang_lines_count = 0
    non_empty_lines = [line for line in lines if line.strip()]
    for line in non_empty_lines:
        if line.count("!") >= 2:
            bang_lines_count += 1

    if (
        len(non_empty_lines) > 0
        and bang_lines_count >= 2
        and bang_lines_count >= len(non_empty_lines) * 0.5
    ):
        return True
    return False


def markdown_to_html_table(markdown: str) -> str:
    all_lines = markdown.strip().split("\n")

    # Identify non-empty lines for structure detection
    non_empty_indices = [i for i, line in enumerate(all_lines) if line.strip()]

    if len(non_empty_indices) < 2:
        return markdown

    # Find separator index in non_empty_indices
    separator_map_idx = -1
    separator_char = "|"

    for idx in non_empty_indices:
        line = all_lines[idx]
        if re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", line) and "-" in line and "|" in line:
            separator_map_idx = idx
            separator_char = "|"
            break

    # Check for ! table if no standard separator found
    if separator_map_idx == -1:
        bang_lines_count = 0
        for idx in non_empty_indices:
            if all_lines[idx].count("!") >= 2:
                bang_lines_count += 1

        if bang_lines_count >= 2 and bang_lines_count >= len(non_empty_indices) * 0.5:
            separator_char = "!"
        else:
            return markdown

    html_parts = []

    if separator_map_idx != -1:
        # Find header (previous non-empty line)
        try:
            sep_pos = non_empty_indices.index(separator_map_idx)
        except ValueError:
            return markdown

        if sep_pos == 0:
            return markdown

        header_map_idx = non_empty_indices[sep_pos - 1]

        # Pre-table text
        if header_map_idx > 0:
            pre_text = "\n".join(all_lines[:header_map_idx]).strip()
            if pre_text:
                html_parts.append(f"<div>{pre_text}</div>")

        html_parts.append("<table>")

        # Header
        header_line = all_lines[header_map_idx]
        header_content = header_line.strip()
        if header_content.startswith(separator_char):
            header_content = header_content[1:]
        if header_content.endswith(separator_char):
            header_content = header_content[:-1]

        header_cells = [cell.strip() for cell in header_content.split(separator_char)]
        html_parts.append("<thead><tr>")
        for cell in header_cells:
            html_parts.append(f"<th>{cell}</th>")
        html_parts.append("</tr></thead>")

        # Body
        html_parts.append("<tbody>")
        for idx in non_empty_indices[sep_pos + 1 :]:
            line = all_lines[idx]
            row_content = line.strip()
            if separator_char in row_content:
                if row_content.startswith(separator_char):
                    row_content = row_content[1:]
                if row_content.endswith(separator_char):
                    row_content = row_content[:-1]

                cells = [cell.strip() for cell in row_content.split(separator_char)]
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")
        html_parts.append("</tbody>")
        html_parts.append("</table>")

    else:
        # Table without separator line (likely ! table)
        # Treat all lines as body
        html_parts.append("<table>")
        html_parts.append("<tbody>")

        for idx in non_empty_indices:
            line = all_lines[idx]
            row_content = line.strip()

            if separator_char in row_content:
                if row_content.startswith(separator_char):
                    row_content = row_content[1:]
                if row_content.endswith(separator_char):
                    row_content = row_content[:-1]

                cells = [cell.strip() for cell in row_content.split(separator_char)]
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

    return "".join(html_parts)


def validate_pdf_pages(pdf_file_path: str, requested_pages: Optional[List[int]] = None) -> int:
    """
    Get total page count from PDF. Page validation happens upstream in file_convertor.py,
    so this function just returns the total pages for reference.
    """
    try:
        from pypdf import PdfReader

        pdf_total_pages = len(PdfReader(str(pdf_file_path)).pages)
    except Exception as e:
        print(f"Failed to read PDF for page validation: {e}")
        raise RequestException(
            message="Unable to process PDF file. Please ensure the file is valid and not corrupted."
        )

    return pdf_total_pages


def handle_processing_error(
    exception: Exception, processing_context: str, service_type: str = "document"
) -> str:
    """
    Enhanced error handling with intelligent categorization and user-friendly messaging.

    Args:
        exception: The caught exception
        processing_context: Context string (e.g., "PDF analysis", "Image processing")
        service_type: Type of service for context (e.g., "document", "image")

    Returns:
        str: User-friendly error message
    """
    # Enhanced internal logging with context
    print(f"❌ {processing_context} failed: {exception}")
    print(f"🔍 Error type: {type(exception).__name__}")
    print(f"🔍 Processing context: {processing_context}")
    if hasattr(exception, "__cause__") and exception.__cause__:
        print(f"🔍 Root cause: {exception.__cause__}")

    # Categorize error for better user messaging
    error_str = str(exception).lower()

    # Rate limiting / throttling
    if "rate limit" in error_str or "throttl" in error_str:
        print("📊 Detected: API rate limiting")
        return "Service temporarily busy due to high demand. Please try again in a few minutes."

    # Quota / usage limits
    elif "quota" in error_str or "usage limit" in error_str:
        print("📊 Detected: API quota exceeded")
        return "Service usage limit reached. Please contact Tensorlake support with the trace ID of the job."

    # Timeout issues
    elif "timeout" in error_str or "timed out" in error_str:
        print("📊 Detected: API timeout")
        if service_type == "document":
            return "Document processing timed out. For large documents, please try processing fewer pages at once."
        else:
            return "Processing timed out. Please try again or contact Tensorlake support."

    # Authentication / credential errors
    elif any(
        term in error_str
        for term in ["credentials", "unauthorized", "access denied", "authentication", "403", "401"]
    ):
        print("📊 Detected: Authentication/credential issue")
        return "Service temporarily unavailable due to authentication error. Please contact Tensorlake support with the trace ID of the job."

    # File corruption / format issues
    elif any(
        term in error_str
        for term in [
            "corrupted",
            "invalid pdf",
            "invalid image",
            "cannot read",
            "unsupported format",
        ]
    ):
        print("📊 Detected: File corruption/format issue")
        if service_type == "document":
            return "Document appears to be corrupted or invalid. Please ensure the file is a valid PDF."
        else:
            return (
                "File format not supported or file is corrupted. Please ensure the file is valid."
            )

    # Memory / size issues
    elif any(term in error_str for term in ["memory", "out of memory", "too large", "file size"]):
        print("📊 Detected: Memory/resource issue")
        if service_type == "document":
            return "Document too large to process. Please try processing fewer pages at once or contact Tensorlake support."
        else:
            return "File too large to process. Please try with a smaller file or contact Tensorlake support."

    # Generic processing errors
    else:
        print("📊 Detected: General processing error")
        return f"{service_type.title()} processing failed. Please try again or contact Tensorlake support with the trace ID of the job."


def download_file(request: ParseRequest) -> FileData:
    if request.file_bytes is not None:
        file_data = base64.b64decode(request.file_bytes)
        return FileData(file_bytes=file_data, content_type=request.mime_type)
    if request.file_url is not None:
        print(f"getting file input from {request.file_url}")
        file_data, content_type = _download_file(file_url=request.file_url)
        if len(file_data) == 0:
            raise RequestException(message="file is empty")
        return FileData(file_bytes=file_data, content_type=content_type)

    raise RequestException(message="ParseRequest requires either file_bytes or file_url")


def _download_file(file_url: str) -> Tuple[bytes, str]:
    if file_url.startswith("s3://"):
        s3_client, bucket_name = _create_s3_client()
        key = file_url[len("s3://") :]
        # Allow either "s3://<key>" (uses S3_BUCKET_NAME) or "s3://<bucket>/<key>"
        if "/" in key and bucket_name is None:
            bucket_name, key = key.split("/", 1)
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            print(f"error getting file from S3. {e}")
            raise RequestException(message="error getting file from S3.") from e

        content = response["Body"].read()
        content_type = response["ResponseMetadata"]["HTTPHeaders"]["content-type"]
        return content, content_type

    if file_url.startswith("https://"):
        try:
            response = requests.get(file_url, timeout=(5, 5))
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message=f"HTTP error accessing file. HTTP status code: {response.status_code}, message: {response.text}"
            ) from e
        except requests.exceptions.Timeout as e:
            print(f"Timeout error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message="Timeout error accessing file. We use a timeout of 5 seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            print(f"Error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message=f"Error accessing file: {str(e)}. Please check the file URL."
            ) from e

        return response.content, response.headers.get("content-type", "")

    raise RequestException(message=f"unsupported file URL scheme: {file_url}")


def _create_s3_client():
    s3_client = boto3.client("s3")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    # S3_BUCKET_NAME is optional — only required when the file_url is a bare "s3://<key>"
    # (no bucket in the URL). For "s3://<bucket>/<key>" form it's ignored.
    return s3_client, bucket_name


# Convert class definitions to a JSON schema string for page classification tasks
def create_classification_choice_and_prompt(
    class_definitions, classification_type
) -> Tuple[str, List[str], str]:
    class_choices = [cls.class_name for cls in class_definitions]
    # Check if the input page class is empty, if so, raise an exception
    if len(class_choices) == 0 or sum(len(cls.strip()) for cls in class_choices) == 0:
        raise RequestException(message="No page classes provided")
    print(f"class_choices: {class_choices}")
    descriptions = {cls.class_name: cls.description for cls in class_definitions}

    class_choices.append("unclassified")
    descriptions["unclassified"] = "Page does not belong to any of the above categories."

    confidence_guidance = (
        "Also provide a confidence score between 0.0 and 1.0 reflecting how certain you are: "
        "0.95-1.0 = very clear match with strong signals; "
        "0.7-0.94 = likely match but some ambiguity; "
        "0.4-0.69 = uncertain, weak signals; "
        "below 0.4 = guessing. "
        "Base the score on your reasoning — do not assign it independently."
    )

    if classification_type == "multi_label":
        prompt_lines = [
            "You are document page classifier. You are given a page and a list of categories. ",
            "You need to classify the page into the most relevant categories from the list below. A page may belong to multiple categories.",
            "If the page does not belong to any of the categories, select 'unclassified'.",
            "Do no hellucinate categories that are not present in the provided list.",
            "Then, explain briefly why you chose these categories based on the content of the page.",
            confidence_guidance,
            "Categories:",
        ]
        for cls_name, cls_desc in descriptions.items():
            prompt_lines.append(f"- {cls_name}: {cls_desc}")
        classification_prompt = "\n".join(prompt_lines)
        schema = {
            "type": "object",
            "properties": {
                "page_classes": {
                    "type": "array",
                    "items": {"type": "string", "enum": class_choices},
                    "description": "List of all categories that apply to this page.",
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for why these categories were chosen.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0 indicating certainty of classification.",
                },
            },
            "required": ["page_classes", "reason", "confidence"],
        }
    else:  # multi_class
        # Add a default page class in case there are pages that dont match nay of the page classes

        prompt_lines = [
            "Classify this page into one of the following categories. IMPORTANT: Only select a specific category if the page clearly and definitively belongs to it. If there is any doubt, ambiguity, or if the page doesn't match the descriptions well, select 'unclassified'. It's better to be conservative and use 'unclassified' than to force an incorrect classification.",
            "Then, briefly explain why you chose this category based on the content of the page.",
            confidence_guidance,
            "Categories:",
        ]
        # Put unclassified first to emphasize it as the default choice
        if "unclassified" in descriptions:
            prompt_lines.append(f"- unclassified: {descriptions['unclassified']}")
        for cls_name, cls_desc in descriptions.items():
            if cls_name != "unclassified":
                prompt_lines.append(f"- {cls_name}: {cls_desc}")

        classification_prompt = "\n".join(prompt_lines)
        schema = {
            "type": "object",
            "properties": {
                "page_class": {"type": "string", "enum": class_choices},
                "reason": {
                    "type": "string",
                    "description": "Brief explanation for why this category was chosen.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0 indicating certainty of classification.",
                },
            },
            "required": ["page_class", "reason", "confidence"],
        }

    return json.dumps(schema), class_choices, classification_prompt


# ========================
# NODE-BY-NODE ROUTING CONDITIONS
# ========================


# FILE_CONVERTOR NODE ROUTING
def file_convertor_should_go_to_output_formatter(request) -> bool:
    """Text files with no structured extraction or page classification go directly to OutputFormatter"""
    return (
        request.mime_type.startswith("text/")
        and not request.structured_extraction_requests
        and not request.page_classification_request
    )


def file_convertor_should_go_to_vlm_extraction(request) -> bool:
    """Should go to VLM when skip_ocr is enabled OR when page_classification is requested"""
    # Text files: go to VLM only if page classification is requested
    if request.mime_type.startswith("text/"):
        return request.page_classification_request
    # Non-text files: go to VLM if skip_ocr is enabled OR page_classification is requested
    else:
        return should_skip_ocr(request)


def file_convertor_should_go_to_structured_extraction(request) -> bool:
    """Text files with structured extraction but no VLM tasks"""
    has_structured_extraction = bool(request.structured_extraction_requests)

    # Don't go directly to structured extraction if VLM tasks (like page classification) are needed first
    has_vlm_tasks = file_convertor_should_go_to_vlm_extraction(request)

    return request.mime_type.startswith("text/") and has_structured_extraction and not has_vlm_tasks


def file_convertor_should_go_to_ocr(request) -> bool:
    """Non-text file that needs OCR. The actual backend is selected from the
    :data:`tensorlake_docai.ocr.OCR_BACKENDS` registry using
    ``request.ocr_model``."""
    return not request.mime_type.startswith("text/") and not should_skip_ocr(request)


# OCR NODE ROUTING (used by Azure, Textact and Model03)
def ocr_should_go_to_output_formatter(request) -> bool:
    """Go to OutputFormatter if no further processing needed"""
    return not (
        request.structured_extraction_requests
        or request.figure_summarization
        or request.figure_grounding
        or request.chart_extraction
        or request.table_summarization
        or request.table_cell_grounding
        or request.page_classification_request
        or request.detect_signature
    )


def _check_has_table_and_figure_and_chart_and_form(
    parse_result: ParseResult,
) -> tuple[bool, bool, bool, bool]:
    """Helper function to check if document has tables and/or figures and/or charts"""
    has_table = False
    has_figure = False
    has_chart = False
    has_form = False
    for page_layout in parse_result.document_layout.pages:
        for element in page_layout.elements:
            if element.fragment_type == PageFragmentType.TABLE:
                has_table = True
            if element.fragment_type == PageFragmentType.TEXT and is_markdown_table(
                element.ocr_text
            ):
                has_table = True
            if element.fragment_type == PageFragmentType.FIGURE:
                has_figure = True
            if element.fragment_type == PageFragmentType.CHART:
                has_chart = True
            if (
                element.fragment_type == PageFragmentType.FORM
                or element.fragment_type == PageFragmentType.KEY_VALUE_REGION
            ):
                has_form = True
            if has_table and has_figure and has_chart and has_form:  # Early exit once we found both
                break
        if has_table and has_figure and has_chart and has_form:
            break
    return has_table, has_figure, has_chart, has_form


def should_route_to_table_merging(request, parse_result: ParseResult) -> bool:
    """Check if we should route to TableMerging task."""
    if not request.table_merging:
        return False

    has_table, _, _, _ = _check_has_table_and_figure_and_chart_and_form(parse_result)
    if not has_table:
        print(
            "Skipping TableMerging: No tables found in document despite table_merging=True request."
        )
        return False

    return True


def route_after_ocr(parse_result: ParseResult, *, log_prefix: str, dots_ocr: bool = False):
    """Dispatch an OCR backend's output to the next pipeline step.

    Every OCR backend ends with the same downstream choice: TableMerging if
    tables exist and ``table_merging`` was requested; otherwise VLM,
    StructuredExtraction, or OutputFormatter depending on which post-OCR
    tasks the request asked for.

    Args:
        parse_result: ParseResult produced by the OCR backend.
        log_prefix: Tag for ``🔀`` log lines (e.g. ``"FULL_PAGE_AZURE"``).
        dots_ocr: If True, use the dots-ocr predicates that additionally
            gate VLM/SE on the actual presence of tables/figures/forms in
            the extracted layout.

    Returns:
        Either a ``ParseResult`` (terminal OutputFormatter step) or a
        Tensorlake future from ``<Task>.run.future(parse_result)``.
    """
    # Lazy imports — these modules import predicates from this file, so a
    # top-level import here would create a cycle.
    from tensorlake_docai.pipeline.output_formatter import format_final_output
    from tensorlake_docai.vlm.cloud import VLMExtractionTask
    from tensorlake_docai.extraction.structured_extraction_functions import (
        StructuredExtraction,
    )
    from tensorlake_docai.tables.table_merging import TableMerging

    request = parse_result.request

    if should_route_to_table_merging(request, parse_result):
        print(f"🔀 {log_prefix} → TableMerging")
        return TableMerging().run.future(parse_result)

    if dots_ocr:
        go_output = dots_ocr_should_go_to_output_formatter(request, parse_result)
        go_vlm = dots_ocr_should_go_to_vlm_extraction(request, parse_result)
        go_se = dots_ocr_should_go_to_structured_extraction(request, parse_result)
    else:
        go_output = ocr_should_go_to_output_formatter(request)
        go_vlm = ocr_should_go_to_vlm_extraction(request, parse_result)
        go_se = ocr_should_go_to_structured_extraction(request, parse_result)

    if go_output:
        print(f"🔀 {log_prefix} → OutputFormatter")
        return format_final_output(parse_result)

    if go_vlm:
        print(f"🔀 {log_prefix} → VLMExtractionTask")
        return VLMExtractionTask().run.future(parse_result)

    if go_se:
        print(f"🔀 {log_prefix} → StructuredExtraction")
        return StructuredExtraction().run.future(parse_result)

    print(f"🔀 {log_prefix} → OutputFormatter")
    return format_final_output(parse_result)


def dots_ocr_should_go_to_output_formatter(request, parse_result: ParseResult) -> bool:
    """Go to OutputFormatter if no further processing needed for dots-ocr."""
    has_table, has_figure, has_chart, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return not (
        request.structured_extraction_requests
        or (request.figure_summarization and has_figure)
        or (request.table_summarization and has_table)
        or (request.table_cell_grounding and has_table)
        or (request.figure_grounding and has_figure)
        or (request.chart_extraction and (has_chart or has_figure or has_table))
        or (request.key_value_extraction and (has_form or has_figure or has_table))
        or request.page_classification_request
        or request.detect_signature
    )


def ocr_should_go_to_vlm_extraction(request, parse_result: ParseResult) -> bool:
    """After OCR, go to VLM if VLM tasks are needed"""
    has_table, has_figure, has_chart, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return (
        (request.figure_summarization and has_figure)
        or (request.table_summarization and has_table)
        or (request.table_cell_grounding and has_table)
        or (request.chart_extraction and (has_figure or has_table))
        or (request.key_value_extraction and (has_figure or has_table or has_form))
        or (request.figure_grounding and has_figure)
        or request.page_classification_request
        or request.detect_signature
        or should_skip_ocr(request)
    )


def dots_ocr_should_go_to_vlm_extraction(request, parse_result: ParseResult) -> bool:
    """After dots-ocr, go to VLM if VLM tasks are needed.

    Checks table_summarization / figure_summarization explicitly since
    figure summarization is no longer baked into the dots-ocr path.
    """
    has_table, has_figure, has_chart, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return (
        (request.table_summarization and has_table)
        or (request.table_cell_grounding and has_table)
        or (request.figure_summarization and has_figure)
        or (request.chart_extraction and (has_chart or has_figure or has_table))
        or (request.key_value_extraction and (has_form or has_figure or has_table))
        or (request.figure_grounding and has_figure)
        or request.page_classification_request
        or request.detect_signature
        or should_skip_ocr(request)
    )


def ocr_should_go_to_structured_extraction(request, parse_result: ParseResult) -> bool:
    """Go to StructuredExtraction if SE needed but no VLM tasks"""
    has_structured_extraction = bool(request.structured_extraction_requests)

    has_vlm_tasks = ocr_should_go_to_vlm_extraction(request, parse_result)

    return has_structured_extraction and not has_vlm_tasks


def dots_ocr_should_go_to_structured_extraction(request, parse_result: ParseResult) -> bool:
    """Go to StructuredExtraction if SE needed but no VLM tasks"""
    has_structured_extraction = bool(request.structured_extraction_requests)

    has_vlm_tasks = dots_ocr_should_go_to_vlm_extraction(request, parse_result)

    return has_structured_extraction and not has_vlm_tasks


# VLM_EXTRACTION NODE ROUTING
def vlm_extraction_should_go_to_output_formatter(request) -> bool:
    """Go to OutputFormatter if no structured extraction needed"""
    return not request.structured_extraction_requests


def vlm_extraction_should_go_to_structured_extraction(request) -> bool:
    """Go to StructuredExtraction if SE needed and skip_ocr is false"""
    return bool(request.structured_extraction_requests) and not should_skip_ocr(request)


# HELPER FUNCTIONS
def skip_ocr_requests(request):
    """Return the subset of structured_extraction_requests with skip_ocr=True."""
    return [req for req in (request.structured_extraction_requests or []) if req.skip_ocr]


def should_skip_ocr(request) -> bool:
    """OCR is skipped only when every structured-extraction request opts out of OCR."""
    reqs = request.structured_extraction_requests or []
    return bool(reqs) and all(req.skip_ocr for req in reqs)


def pil_image_to_base64(image) -> str:
    """Convert PIL Image to base64-encoded string (data URI format)"""
    import base64
    import io

    # Handle alpha channels
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGB")

    # Convert image to bytes
    buffer = io.BytesIO()
    # Use JPEG format with compression for preview
    image.save(buffer, format="JPEG", quality=60)
    buffer.seek(0)

    # Encode to base64
    image_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Return as data URI
    return f"data:image/jpeg;base64,{base64_encoded}"


async def stream_with_timeout(stream, timeout_seconds: int = 120):
    """
    Yields chunks from stream with a per-chunk timeout.
    """
    # Convert the stream to an asynchronous iterator
    iterator = aiter(stream)

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(anext(iterator), timeout=timeout_seconds)
                yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"No response chunk received within {timeout_seconds} seconds"
                ) from None
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


def update_progress_if_needed(
    chunk_count: int, accumulated_chars: int, start_time: float, last_update_time: float
) -> float:
    """
    Update progress to keep function-level timeout alive.
    Returns the new last_update_time if update was performed, otherwise returns the input last_update_time.
    """
    from tensorlake.applications import RequestContext

    try:
        ctx = RequestContext.get()
    except Exception:
        return last_update_time

    # Update every 5 chunks or every 30 seconds
    current_time = time.time()
    if chunk_count % 5 == 0 or (current_time - last_update_time) >= 30:
        elapsed = current_time - start_time
        ctx.progress.update(
            current=accumulated_chars,
            total=accumulated_chars + 1000,  # Estimated total
            message=f"Streaming response ({accumulated_chars} chars, {elapsed:.0f}s)",
        )
        return current_time
    return last_update_time

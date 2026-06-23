"""API adapter for the Kredit Lab financial statement analyzer.

The dashboard now runs the PDF/TXT/Claude flow in its own Next.js server route,
but this standalone service still needs to work when the financial tool is used
directly. Keep the checked-in renderer as the source of truth while exposing the
same practical flow:

PDF -> OCR service markdown -> Claude JSON -> renderer HTML
TXT/MD -> Claude JSON -> renderer HTML
JSON -> renderer HTML
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import base64
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


class _StreamlitStub(types.ModuleType):
    """Let the API import renderer functions without booting Streamlit."""

    def __getattr__(self, name: str) -> Any:
        return _StreamlitStub(f"streamlit.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __enter__(self) -> "_StreamlitStub":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


sys.modules.setdefault("streamlit", _StreamlitStub("streamlit"))

from streamlit_financial_report_v7_7 import (  # noqa: E402
    convert_html_to_pdf,
    generate_full_html,
    validate_json_structure,
)
from excel_export import convert_json_to_excel  # noqa: E402

app = FastAPI(title="Kredit Lab Financial Statement Analyzer API")

SUPPORTED_UPLOAD_EXTENSIONS = [".pdf", ".txt", ".md", ".json"]
TEXT_UPLOAD_EXTENSIONS = [".txt", ".md"]
JSON_UPLOAD_EXTENSIONS = [".json"]
PDF_UPLOAD_EXTENSIONS = [".pdf"]

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_OCR_SERVICE_TIMEOUT_SECONDS = 240
DEFAULT_ANTHROPIC_MAX_TOKENS = 64000

CLAUDE_SCHEMA_INSTRUCTIONS_FILE = Path(__file__).with_name(
    "claude_schema_instructions.md"
)
CLAUDE_MODELS = {
    "claude-opus-4-8": 128000,
    "claude-sonnet-4-6": 64000,
    "claude-haiku-4-5-20251001": 64000,
    "claude-opus-4-7": 128000,
    "claude-opus-4-1-20250805": 20000,
    "claude-opus-4-20250514": 20000,
    "claude-sonnet-4-20250514": 20000,
    "claude-3-5-haiku-20241022": 8192,
}
CLAUDE_MODEL_EFFORT = {
    "claude-opus-4-8": "high",
    "claude-sonnet-4-6": "medium",
    "claude-opus-4-7": "high",
}
CLAUDE_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"

CLAUDE_SYSTEM_PROMPT = CLAUDE_SCHEMA_INSTRUCTIONS_FILE.read_text(
    encoding="utf-8"
).strip()

if not CLAUDE_SYSTEM_PROMPT:
    raise RuntimeError(
        f"Claude schema instructions file is empty: {CLAUDE_SCHEMA_INSTRUCTIONS_FILE}"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    default_model = resolve_claude_model(None)

    return {
        "status": "ok",
        "tool": "financial_statement",
        "active_flow": "pdf_txt_md_json_pipeline",
        "supported_upload_extensions": SUPPORTED_UPLOAD_EXTENSIONS,
        "has_anthropic_api_key": bool(get_anthropic_api_key()),
        "has_ocr_service_url": bool(ocr_service_url()),
        "has_ocr_service_api_key": bool(ocr_service_api_key()),
        "default_claude_model": default_model,
        "convert_endpoint": "/convert",
        "analyze_endpoint": "/analyze",
        "render_bridge_endpoint": "/render-bridge",
    }


@app.post("/convert")
async def convert(files: list[UploadFile] = File(..., alias="file")) -> dict[str, Any]:
    if not files:
        raise api_error(400, "missing_input", "At least one PDF file is required.")

    generated_text_files: list[dict[str, Any]] = []

    for index, upload in enumerate(files):
        file_name = upload.filename or "financial-statement.pdf"
        raw_bytes = await upload.read()

        if not is_pdf_upload(file_name, upload.content_type):
            raise api_error(
                400,
                "invalid_file_type",
                "The convert endpoint only accepts PDF files.",
                {"file_name": file_name, "content_type": upload.content_type},
            )

        markdown, metadata = extract_pdf_markdown_with_ocr_service(
            file_name=file_name,
            content_type=upload.content_type,
            raw_bytes=raw_bytes,
        )

        generated_text_files.append(
            {
                "id": f"{int(time.time() * 1000)}-{index}-{slugify_file_name(file_name)}",
                "originalFileName": file_name,
                "generatedFileName": generated_text_file_name(file_name),
                "fileType": "text/plain",
                "text": markdown,
                "textLength": len(markdown),
                "ocrProvider": metadata.get("provider"),
                "ocrPagesParsed": metadata.get("pages_parsed"),
                "servedBy": metadata.get("served_by"),
            }
        )

    return {
        "success": True,
        "tool": "financial_statement",
        "generatedTextFiles": generated_text_files,
    }


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(..., alias="file"),
    model: str | None = Form(None),
) -> dict[str, Any]:
    if not files:
        raise api_error(
            400,
            "missing_input",
            "At least one PDF, TXT, MD, or JSON file is required.",
        )

    text_documents: list[dict[str, Any]] = []
    json_reports: list[dict[str, Any]] = []
    extraction: list[dict[str, Any]] = []

    for upload in files:
        file_name = upload.filename or "financial-statement"
        content_type = upload.content_type
        raw_bytes = await upload.read()

        if is_json_upload(file_name, content_type):
            data = parse_json_upload(file_name, raw_bytes)
            json_reports.append(render_report(file_name, content_type, data))
            extraction.append({"file_name": file_name, "source_kind": "json"})
            continue

        if is_text_upload(file_name, content_type):
            text = decode_text_upload(file_name, raw_bytes).strip()

            if not text:
                raise api_error(
                    400,
                    "text_file_empty",
                    f"{file_name} is empty.",
                )

            text_documents.append(
                {
                    "file_name": file_name,
                    "file_type": content_type or "text/plain",
                    "source_kind": "markdown" if file_name.lower().endswith(".md") else "text",
                    "text": text,
                }
            )
            extraction.append(
                {
                    "file_name": file_name,
                    "source_kind": "markdown" if file_name.lower().endswith(".md") else "text",
                }
            )
            continue

        if is_pdf_upload(file_name, content_type):
            markdown, metadata = extract_pdf_markdown_with_ocr_service(
                file_name=file_name,
                content_type=content_type,
                raw_bytes=raw_bytes,
            )
            text_documents.append(
                {
                    "file_name": generated_text_file_name(file_name),
                    "file_type": "text/plain",
                    "source_kind": "ocr_markdown",
                    "text": markdown,
                }
            )
            extraction.append(
                {
                    "file_name": file_name,
                    "source_kind": "ocr_markdown",
                    "ocr_provider": metadata.get("provider"),
                    "ocr_pages_parsed": metadata.get("pages_parsed"),
                    "served_by": metadata.get("served_by"),
                }
            )
            continue

        raise api_error(
            400,
            "invalid_file_type",
            "Only PDF, TXT, MD, or JSON files are supported.",
            {"file_name": file_name, "content_type": content_type},
        )

    reports: list[dict[str, Any]] = []
    claude: dict[str, Any] | None = None

    if text_documents:
        selected_model = resolve_claude_model(model)
        data, usage = analyze_text_with_claude(text_documents, selected_model)
        primary_text_document = text_documents[0]
        reports.append(
            render_report(
                primary_text_document["file_name"],
                primary_text_document["file_type"],
                data,
            )
        )
        claude = {"model": selected_model, "usage": usage}

    reports.extend(json_reports)

    if not reports:
        raise api_error(
            400,
            "missing_input",
            "No analyzable financial statement files were provided.",
        )

    primary_report = reports[0]

    return {
        "success": True,
        "tool": "financial_statement",
        "html": primary_report["html"],
        "json": primary_report["json"],
        "warnings": primary_report["warnings"],
        "reports": reports,
        "extraction": extraction,
        "claude": claude,
    }


@app.post("/render-bridge")
async def render_bridge(payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("mode", "render")
    data = payload.get("data")

    if mode not in {"validate", "render", "pdf", "excel"}:
        raise api_error(
            400,
            "invalid_renderer_mode",
            "Renderer mode must be validate, render, pdf, or excel.",
            {"mode": mode},
        )

    if not isinstance(data, dict):
        raise api_error(
            400,
            "malformed_intermediate_output",
            "Financial analysis data must be a JSON object.",
        )

    validation = normalize_renderer_validation(validate_json_structure(data))

    if mode == "validate":
        return {"success": True, "validation": validation}

    if not validation["isValid"]:
        return {"success": True, "validation": validation, "html": None}

    html = generate_full_html(data)

    if not isinstance(html, str) or not html.strip():
        raise api_error(
            502,
            "renderer_invalid_output",
            "Financial statement renderer did not return report HTML.",
        )

    if mode == "pdf":
        pdf_bytes = convert_html_to_pdf(html)
        return {
            "success": True,
            "validation": validation,
            "contentBase64": base64.b64encode(pdf_bytes).decode("ascii"),
            "contentType": "application/pdf",
            "fileExtension": "pdf",
        }

    if mode == "excel":
        excel_bytes = convert_json_to_excel(data)
        return {
            "success": True,
            "validation": validation,
            "contentBase64": base64.b64encode(excel_bytes).decode("ascii"),
            "contentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "fileExtension": "xlsx",
        }

    return {"success": True, "validation": validation, "html": html}


def normalize_renderer_validation(result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise api_error(
            502,
            "renderer_invalid_output",
            "Renderer validation returned an unexpected response.",
        )

    is_valid, errors, warnings = result

    if not isinstance(is_valid, bool):
        raise api_error(
            502,
            "renderer_invalid_output",
            "Renderer validation status is malformed.",
        )

    return {
        "isValid": is_valid,
        "errors": [str(error) for error in errors] if isinstance(errors, list) else [],
        "warnings": [str(warning) for warning in warnings]
        if isinstance(warnings, list)
        else [],
    }


def render_report(file_name: str, content_type: str | None, data: dict[str, Any]) -> dict[str, Any]:
    is_valid, errors, warnings = validate_json_structure(data)

    if not is_valid:
        raise api_error(
            400,
            "renderer_validation_failure",
            "JSON structure is not compatible with the financial statement renderer.",
            {"file": file_name, "errors": errors, "warnings": warnings},
        )

    html = generate_full_html(data)

    if not isinstance(html, str) or not html.strip():
        raise api_error(
            502,
            "renderer_invalid_output",
            "Financial statement renderer did not return report HTML.",
            {"file": file_name},
        )

    return {
        "file_name": file_name,
        "source_file_type": content_type,
        "html": html,
        "json": data,
        "errors": errors,
        "warnings": warnings,
    }


def analyze_text_with_claude(
    documents: list[dict[str, Any]],
    model: str,
) -> tuple[dict[str, Any], Any]:
    api_key = get_anthropic_api_key()

    if not api_key:
        raise api_error(
            500,
            "missing_claude_api_key",
            "ANTHROPIC_API_KEY or CLAUDE_API_KEY is missing.",
        )

    user_prompt = build_claude_user_prompt(documents)
    first_response = call_claude_messages(api_key, model, user_prompt)
    try:
        data = parse_claude_json(first_response["text"])
    except HTTPException as exc:
        errors = [http_exception_message(exc)]
    else:
        is_valid, errors, _warnings = validate_json_structure(data)

        if is_valid:
            return data, first_response.get("usage")

    correction_prompt = build_claude_correction_prompt(
        original_prompt=user_prompt,
        previous_response=first_response["text"],
        validation_errors=errors,
    )
    corrected_response = call_claude_messages(api_key, model, correction_prompt)
    corrected_data = parse_claude_json(corrected_response["text"])
    corrected_is_valid, corrected_errors, _corrected_warnings = validate_json_structure(
        corrected_data
    )

    if not corrected_is_valid:
        raise api_error(
            502,
            "claude_analysis_failure",
            "Claude did not return renderer-compatible financial analysis JSON.",
            {"errors": corrected_errors},
        )

    return corrected_data, merge_claude_usage(
        first_response.get("usage"), corrected_response.get("usage")
    )


def call_claude_messages(api_key: str, model: str, user_prompt: str) -> dict[str, Any]:
    request_body = {
        "model": model,
        "max_tokens": get_claude_max_output_tokens(model),
        "system": CLAUDE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    effort = resolve_claude_effort(model)

    if effort:
        request_body["output_config"] = {"effort": effort}

    try:
        response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise api_error(
            502,
            "claude_analysis_failure",
            "Claude analysis request failed.",
            {"detail": str(exc)},
        ) from exc

    response_body = read_response_body(response)

    if not response.ok:
        raise api_error(
            claude_http_status(response.status_code, response_body),
            claude_error_code(response.status_code, response_body),
            claude_error_message(response_body) or "Claude analysis request failed.",
            {"status": response.status_code, "body": response_body},
        )

    text = extract_claude_text(response_body)
    stop_reason = (
        response_body.get("stop_reason") if isinstance(response_body, dict) else ""
    )
    usage = response_body.get("usage") if isinstance(response_body, dict) else None

    if stop_reason == "max_tokens":
        raise api_error(
            502,
            "claude_output_truncated",
            (
                f"Claude hit the {request_body['max_tokens']:,} output-token limit "
                "before completing renderer JSON. No correction request was sent "
                "to avoid additional token spend. Increase ANTHROPIC_MAX_TOKENS "
                "for a model that supports it, choose a higher-output Claude model, "
                "or reduce the selected input files."
            ),
            {
                "model": model,
                "max_tokens": request_body["max_tokens"],
                "usage": usage,
                "response_text_length": len(text),
                "stop_reason": stop_reason,
            },
        )

    if not text.strip():
        raise api_error(
            502,
            "claude_analysis_failure",
            "Claude analysis returned an empty response.",
            response_body,
        )

    return {
        "text": text,
        "usage": usage,
        "stop_reason": stop_reason,
    }


def build_claude_user_prompt(documents: list[dict[str, Any]]) -> str:
    files = "\n\n".join(
        f"""===== SOURCE DOCUMENT {index + 1}: {document["file_name"]} ({document["source_kind"]}) =====
{document["text"]}""".strip()
        for index, document in enumerate(documents)
    )

    return f"""Analyze the financial statements below and return one consolidated Kredit Lab financial analysis JSON object.

{files}""".strip()


def build_claude_correction_prompt(
    original_prompt: str,
    previous_response: str,
    validation_errors: list[str],
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors)

    return f"""The previous response was not compatible with the Kredit Lab renderer.

Validation errors:
{errors}

Original request:
{original_prompt}

Previous response:
{previous_response}

Return a corrected JSON object only.""".strip()


def parse_claude_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise api_error(
                502,
                "claude_analysis_failure",
                "Claude response did not contain a JSON object.",
            )

        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise api_error(
                502,
                "claude_analysis_failure",
                "Claude response was not valid JSON.",
                {"detail": str(exc)},
            ) from exc

    if not isinstance(parsed, dict):
        raise api_error(
            502,
            "claude_analysis_failure",
            "Claude JSON root must be an object.",
        )

    return parsed


def extract_pdf_markdown_with_ocr_service(
    file_name: str,
    content_type: str | None,
    raw_bytes: bytes,
) -> tuple[str, dict[str, Any]]:
    base_url = ocr_service_url()

    if not base_url:
        raise api_error(
            500,
            "missing_ocr_service_url",
            "OCR_SERVICE_URL is missing.",
        )

    headers: dict[str, str] = {}
    api_key = ocr_service_api_key()

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            f"{base_url}/parse",
            headers=headers,
            files={
                "file": (
                    file_name,
                    raw_bytes,
                    get_upload_mime_type(file_name, content_type),
                )
            },
            timeout=ocr_service_timeout_seconds(),
        )
    except requests.RequestException as exc:
        raise api_error(
            502,
            "ocr_extraction_failure",
            "OCR service request failed.",
            {"detail": str(exc)},
        ) from exc

    response_body = read_response_body(response)

    if not response.ok:
        raise api_error(
            502,
            "ocr_extraction_failure",
            "OCR service extraction failed.",
            {"status": response.status_code, "body": response_body},
        )

    markdown = extract_markdown_from_ocr_result(response_body).strip()
    pages_parsed = get_number_from_path(response_body, ["usage", "pages_parsed"])
    if pages_parsed is None:
        pages_parsed = get_number_from_path(response_body, ["parsed_pages_count"])
    served_by = get_string_from_mapping(response_body, "served_by")
    provider = (
        served_by
        or get_string_from_mapping(response_body, "provider")
        or get_string_from_mapping(response_body, "ocr_model")
        or "azure"
    )

    if not markdown:
        raise api_error(
            502,
            "ocr_empty_output",
            "OCR service did not return markdown content.",
            {"file_name": file_name, "provider": provider},
        )

    return markdown, {
        "file_id": get_string_from_mapping(response_body, "file_id"),
        "parse_id": get_string_from_mapping(response_body, "parse_id"),
        "pages_parsed": pages_parsed,
        "provider": provider,
        "served_by": served_by,
    }


def extract_markdown_from_ocr_result(result: Any) -> str:
    if not isinstance(result, dict):
        return ""

    chunks = result.get("chunks")
    if isinstance(chunks, list):
        chunk_markdown = [
            str(chunk.get("content", "")).strip()
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("content", "")).strip()
        ]
        if chunk_markdown:
            return "\n\n".join(chunk_markdown)

    pages = result.get("pages")
    if not isinstance(pages, list):
        return ""

    page_markdown: list[str] = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        page_number = page.get("page_number")
        prefix = f"## Page {page_number}\n" if isinstance(page_number, int) else ""
        fragments = page.get("page_fragments")

        if not isinstance(fragments, list):
            continue

        content = "\n\n".join(
            str(fragment.get("content", "")).strip()
            for fragment in fragments
            if isinstance(fragment, dict)
            and str(fragment.get("content", "")).strip()
        )

        if content:
            page_markdown.append(f"{prefix}{content}".strip())

    return "\n\n".join(page_markdown)


def parse_json_upload(file_name: str, raw_bytes: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_bytes.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise api_error(
            400,
            "malformed_input",
            f"{file_name} is not valid UTF-8 JSON.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise api_error(
            400,
            "malformed_input",
            f"{file_name} is not valid JSON.",
            {"detail": str(exc)},
        ) from exc

    if not isinstance(parsed, dict):
        raise api_error(
            400,
            "malformed_input",
            f"{file_name} must contain a JSON object.",
        )

    return parsed


def decode_text_upload(file_name: str, raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise api_error(
            400,
            "malformed_input",
            f"{file_name} is not valid UTF-8 text.",
        ) from exc


def is_json_upload(file_name: str, content_type: str | None) -> bool:
    return file_name.lower().endswith(".json") or content_type == "application/json"


def is_text_upload(file_name: str, content_type: str | None) -> bool:
    normalized = file_name.lower()
    return (
        normalized.endswith(".txt")
        or normalized.endswith(".md")
        or content_type in {"text/plain", "text/markdown", "text/x-markdown"}
    )


def is_pdf_upload(file_name: str, content_type: str | None) -> bool:
    return file_name.lower().endswith(".pdf") or content_type == "application/pdf"


def get_upload_mime_type(file_name: str, content_type: str | None) -> str:
    if content_type:
        return content_type

    normalized = file_name.lower()

    if normalized.endswith(".pdf"):
        return "application/pdf"
    if normalized.endswith(".md"):
        return "text/markdown"
    if normalized.endswith(".txt"):
        return "text/plain"
    if normalized.endswith(".json"):
        return "application/json"

    return "application/octet-stream"


def resolve_claude_model(requested_model: str | None) -> str:
    requested = (requested_model or "").strip()
    env_default = os.getenv("ANTHROPIC_MODEL", "").strip()

    if requested:
        if requested in CLAUDE_MODELS:
            return requested
        raise api_error(
            400,
            "invalid_claude_model",
            "Selected Claude model is not allowed.",
            {"model": requested_model},
        )

    if env_default in CLAUDE_MODELS:
        return env_default

    return DEFAULT_CLAUDE_MODEL


def get_claude_max_output_tokens(model: str) -> int:
    configured = positive_number_env(
        "ANTHROPIC_MAX_TOKENS",
        DEFAULT_ANTHROPIC_MAX_TOKENS,
    )
    model_max = CLAUDE_MODELS.get(model, DEFAULT_ANTHROPIC_MAX_TOKENS)
    return int(min(configured, model_max))


def resolve_claude_effort(model: str) -> str | None:
    configured = os.getenv("ANTHROPIC_EFFORT", "").strip()

    if configured in CLAUDE_EFFORT_LEVELS:
        return configured

    return CLAUDE_MODEL_EFFORT.get(model)


def get_anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""


def ocr_service_url() -> str:
    return (
        os.getenv("OCR_SERVICE_URL")
        or os.getenv("FINANCIAL_OCR_SERVICE_URL")
        or ""
    ).rstrip("/")


def ocr_service_api_key() -> str:
    return os.getenv("OCR_SERVICE_API_KEY") or os.getenv("SERVICE_API_KEY") or ""


def ocr_service_timeout_seconds() -> int | float:
    timeout_ms = positive_number_env("OCR_SERVICE_TIMEOUT_MS", 0)

    if timeout_ms > 0:
        return timeout_ms / 1000

    return positive_number_env(
        "OCR_SERVICE_TIMEOUT_SECONDS",
        DEFAULT_OCR_SERVICE_TIMEOUT_SECONDS,
    )


def get_string_from_mapping(value: Any, key: str) -> str | None:
    if not isinstance(value, dict):
        return None

    item = value.get(key)
    return item.strip() if isinstance(item, str) and item.strip() else None


def merge_claude_usage(*usages: Any) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}

    for usage in usages:
        if not isinstance(usage, dict):
            continue

        for key, value in usage.items():
            if isinstance(value, (int, float)):
                current = merged.get(key)
                merged[key] = (current if isinstance(current, (int, float)) else 0) + value
            elif key not in merged:
                merged[key] = value

    return merged or None


def http_exception_message(exc: HTTPException) -> str:
    detail = exc.detail

    if isinstance(detail, dict):
        message = detail.get("message")

        if isinstance(message, str) and message:
            return message

    return str(detail)


def extract_claude_text(response_body: Any) -> str:
    if not isinstance(response_body, dict):
        return ""

    content = response_body.get("content")
    if not isinstance(content, list):
        return ""

    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ).strip()


def claude_error_code(status: int, body: Any) -> str:
    error_type = claude_error_type(body)
    message = claude_error_message(body).lower()

    if status in {401, 403} or error_type == "authentication_error":
        return "claude_auth_failure"
    if status == 404 or error_type == "not_found_error" or (
        "model" in message and "not" in message
    ):
        return "claude_model_not_found"
    if status == 429 or error_type == "rate_limit_error":
        return "claude_rate_limit"
    if status == 413 or "context" in message or "too long" in message:
        return "claude_context_too_large"
    if status == 400 or error_type == "invalid_request_error":
        return "invalid_claude_request"

    return "claude_analysis_failure"


def claude_http_status(status: int, body: Any) -> int:
    code = claude_error_code(status, body)

    if code in {"claude_auth_failure", "claude_model_not_found"}:
        return 502
    if code == "claude_rate_limit":
        return 429
    if code == "claude_context_too_large":
        return 413
    if code == "invalid_claude_request":
        return 400

    return status if 400 <= status < 600 else 502


def claude_error_type(body: Any) -> str:
    if not isinstance(body, dict):
        return ""

    if isinstance(body.get("type"), str):
        return body["type"]

    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]

    return ""


def claude_error_message(body: Any) -> str:
    if isinstance(body, str):
        return body

    if not isinstance(body, dict):
        return ""

    if isinstance(body.get("message"), str):
        return body["message"]

    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]

    return ""


def read_response_body(response: requests.Response) -> Any:
    if not response.text:
        return None

    try:
        return response.json()
    except ValueError:
        return response.text


def get_number_from_path(value: Any, path: list[str]) -> int | float | None:
    current = value

    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current if isinstance(current, (int, float)) else None


def positive_number_env(name: str, fallback: int | float) -> int | float:
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return fallback

    return value if value > 0 else fallback


def generated_text_file_name(file_name: str) -> str:
    if "." in file_name:
        base = file_name.rsplit(".", 1)[0]
    else:
        base = file_name

    return f"{base or 'financial-statement'}.txt"


def slugify_file_name(file_name: str) -> str:
    slug_chars: list[str] = []

    for char in file_name.lower():
        if char.isalnum():
            slug_chars.append(char)
        elif not slug_chars or slug_chars[-1] != "-":
            slug_chars.append("-")

    return "".join(slug_chars).strip("-")[:80] or "financial-statement"


def api_error(
    status_code: int,
    code: str,
    message: str,
    detail: Any | None = None,
) -> HTTPException:
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return HTTPException(status_code=status_code, detail=payload)

#!/usr/bin/env python3
"""Direct Azure Document Intelligence OCR fallback for the dashboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

LLMWHISPERER_BASE = "https://llmwhisperer-api.us-central.unstract.com/api/v2"
LLMWHISPERER_DEFAULT_MODE = "form"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract markdown from a PDF with Azure DI")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--out", required=True, help="Path to write JSON result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    raw = pdf_path.read_bytes()

    try:
        output = extract_pdf_bytes(raw, pdf_path.name)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    Path(args.out).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return 0


def extract_pdf_bytes(raw: bytes, file_name: str = "document.pdf") -> dict[str, Any]:
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").strip()
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "").strip()
    primary_error: Exception | None = None

    if endpoint and key:
        try:
            output = extract_with_azure(raw, endpoint, key)
            if has_markdown(output):
                return output
            primary_error = RuntimeError("Azure OCR returned empty markdown")
        except Exception as exc:
            primary_error = exc
    else:
        primary_error = RuntimeError(
            "Missing AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY"
        )

    if os.getenv("LLMWHISPERER_API_KEY", "").strip():
        try:
            return extract_with_llmwhisperer(raw, file_name)
        except Exception as backup_error:
            raise RuntimeError(
                "Azure OCR failed and LLM Whisperer backup also failed: "
                f"azure={primary_error}; llmwhisperer={backup_error}"
            ) from backup_error

    raise RuntimeError(f"Azure OCR failed: {primary_error}") from primary_error


def has_markdown(output: dict[str, Any]) -> bool:
    chunks = output.get("chunks")
    if not isinstance(chunks, list):
        return False

    return any(
        isinstance(chunk, dict) and str(chunk.get("content") or "").strip()
        for chunk in chunks
    )


def extract_with_azure(raw: bytes, endpoint: str, key: str) -> dict[str, Any]:
    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=raw),
        output_content_format="markdown",
    )
    result = poller.result()
    markdown = getattr(result, "content", "") or ""
    pages = getattr(result, "pages", None) or []

    output = {
        "chunks": [{"content": markdown}] if markdown.strip() else [],
        "parsed_pages_count": len(pages),
        "served_by": "azure",
        "provider": "azure",
    }
    return output


def extract_with_llmwhisperer(raw: bytes, file_name: str) -> dict[str, Any]:
    api_key = os.getenv("LLMWHISPERER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LLM Whisperer backup is not configured")

    base_url = os.getenv("LLMWHISPERER_BASE", LLMWHISPERER_BASE).rstrip("/")
    mode = os.getenv("LLMWHISPERER_MODE", LLMWHISPERER_DEFAULT_MODE)
    headers = {
        "unstract-key": api_key,
        "Content-Type": "application/octet-stream",
    }
    params = {
        "mode": mode,
        "output_mode": "layout_preserving",
        "file_name": file_name or "document.pdf",
    }

    response = requests.post(
        f"{base_url}/whisper",
        params=params,
        headers=headers,
        data=raw,
        timeout=120,
    )
    response.raise_for_status()
    whisper_hash = response.json()["whisper_hash"]

    for _ in range(120):
        time.sleep(3)
        status_response = requests.get(
            f"{base_url}/whisper-status",
            params={"whisper_hash": whisper_hash},
            headers={"unstract-key": api_key},
            timeout=30,
        )
        status_response.raise_for_status()
        status = status_response.json().get("status")

        if status == "processed":
            break
        if status in {"error", "unknown"}:
            raise RuntimeError(f"LLM Whisperer status={status}")
    else:
        raise RuntimeError("LLM Whisperer timed out")

    output_response = requests.get(
        f"{base_url}/whisper-retrieve",
        params={"whisper_hash": whisper_hash},
        headers={"unstract-key": api_key},
        timeout=60,
    )
    output_response.raise_for_status()
    output = output_response.json()

    text = output.get("result_text") or output.get("extraction", {}).get(
        "result_text", ""
    )
    pages = [page for page in text.split("\f") if page.strip()] or [text]
    chunks = [{"content": page} for page in pages if page.strip()]
    return {
        "chunks": chunks,
        "parsed_pages_count": len(chunks),
        "served_by": "llmwhisperer",
        "provider": "llmwhisperer",
    }


if __name__ == "__main__":
    raise SystemExit(main())

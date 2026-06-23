"""Kredit Lab OCR service — self-hosted replacement for the Tensorlake hosted
Document AI API (retires 2026-06-30).

Wraps OpenIngest's in-process runner (`run_local_application`) with the azure-di
backend, which was proven to reproduce Tensorlake's output at 100% value parity.

POST /parse  (multipart: file=<pdf>)  ->  Tensorlake-parse-shaped JSON:
    { "chunks": [{"content": "<page markdown>"}...], "parsed_pages_count": N }
so the dashboard's existing extractMarkdownFromTensorlakeResult() works unchanged.

Backup OCR: if the primary (Azure) path fails AND LLMWHISPERER_API_KEY is set,
the request automatically falls back to LLMWhisperer (Unstract). Proven at 100%
value parity vs Tensorlake on the golden samples. Same response shape, so callers
never know which engine served them. Unset the key -> Azure-only (today's behavior).

Auth: Authorization: Bearer $SERVICE_API_KEY  (if SERVICE_API_KEY is set).
Health: GET /health
"""
import base64
import os
import time

import requests
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from tensorlake.applications import run_local_application
from tensorlake_docai.pipeline.api import ParseRequest, ParsedDocument
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload
from tensorlake_docai.postprocess.formatter import page_to_markdown

OCR_MODEL = os.environ.get("OCR_MODEL", "azure-di")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")  # set in Railway; protects the endpoint

# Backup OCR (LLMWhisperer / Unstract). Active only when the key is set.
LLMWHISPERER_API_KEY = os.environ.get("LLMWHISPERER_API_KEY")
LLMWHISPERER_MODE = os.environ.get("LLMWHISPERER_MODE", "form")  # form = HQ+Table, best for financial tables
LLMWHISPERER_BASE = os.environ.get(
    "LLMWHISPERER_BASE", "https://llmwhisperer-api.us-central.unstract.com/api/v2"
)

app = FastAPI(title="Kredit Lab OCR", version="1.1")


def _parse_azure(raw: bytes, filename: str, content_type: str | None) -> dict:
    """Primary path: OpenIngest in-process runner with the azure-di backend."""
    req = ParseRequest(
        file_name=filename or "document.pdf",
        mime_type=content_type or "application/pdf",
        file_bytes=base64.b64encode(raw).decode(),
        ocr_model=OCR_MODEL,
        chunk_strategy="page",          # matches dashboard lean profile
        table_output_mode="markdown",
        xpage_header_detection=False,   # production lean does not use it (needs OpenAI key)
    )
    handle = run_local_application(normalize_file_type_and_upload, req.model_dump())
    result = handle.output()
    if not result or "document" not in result:
        raise RuntimeError("No document returned")
    doc = ParsedDocument.model_validate(result["document"])
    pages = doc.pages or []
    chunks = [{"content": page_to_markdown(p, req)} for p in pages]
    return {"chunks": chunks, "parsed_pages_count": doc.parsed_pages_count or len(pages)}


def _parse_llmwhisperer(raw: bytes, filename: str) -> dict:
    """Backup path: LLMWhisperer hosted API. Reshaped to the Tensorlake parse shape
    (one chunk per page, split on the form-feed page separator)."""
    headers = {"unstract-key": LLMWHISPERER_API_KEY, "Content-Type": "application/octet-stream"}
    params = {"mode": LLMWHISPERER_MODE, "output_mode": "layout_preserving",
              "file_name": filename or "document.pdf"}
    r = requests.post(f"{LLMWHISPERER_BASE}/whisper", params=params, headers=headers,
                      data=raw, timeout=120)
    r.raise_for_status()
    whash = r.json()["whisper_hash"]

    for _ in range(120):  # poll up to ~6 min
        time.sleep(3)
        status = requests.get(
            f"{LLMWHISPERER_BASE}/whisper-status", params={"whisper_hash": whash},
            headers={"unstract-key": LLMWHISPERER_API_KEY}, timeout=30,
        ).json().get("status")
        if status == "processed":
            break
        if status in ("error", "unknown"):
            raise RuntimeError(f"LLMWhisperer status={status}")
    else:
        raise RuntimeError("LLMWhisperer timed out")

    out = requests.get(
        f"{LLMWHISPERER_BASE}/whisper-retrieve", params={"whisper_hash": whash},
        headers={"unstract-key": LLMWHISPERER_API_KEY}, timeout=60,
    ).json()
    text = out.get("result_text") or out.get("extraction", {}).get("result_text", "")
    pages = [p for p in text.split("\f") if p.strip()] or [text]
    return {"chunks": [{"content": p} for p in pages], "parsed_pages_count": len(pages)}


def _check_auth(authorization: str | None) -> None:
    if not SERVICE_API_KEY:
        return  # no key configured -> open (dev only)
    expected = f"Bearer {SERVICE_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_model": OCR_MODEL,
        "backup": "llmwhisperer" if LLMWHISPERER_API_KEY else None,
    }


@app.post("/parse")
def parse(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # Primary: Azure (via OpenIngest). On failure, fall back to LLMWhisperer if configured.
    try:
        return JSONResponse(_parse_azure(raw, file.filename, file.content_type))
    except Exception as primary_err:
        if not LLMWHISPERER_API_KEY:
            raise HTTPException(status_code=502, detail=f"OCR failed: {type(primary_err).__name__}: {primary_err}")
        print(f"[kreditlab-ocr] primary (azure) failed: {primary_err!r} -> trying LLMWhisperer backup", flush=True)
        try:
            result = _parse_llmwhisperer(raw, file.filename)
        except Exception as backup_err:
            raise HTTPException(
                status_code=502,
                detail=f"OCR failed (azure: {primary_err}; backup: {backup_err})",
            )
        return JSONResponse({**result, "served_by": "llmwhisperer"})


if __name__ == "__main__":
    # Read PORT in Python so no shell/${PORT} expansion is needed (Railway-proof).
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"[kreditlab-ocr] starting uvicorn on 0.0.0.0:{port} (ocr_model={OCR_MODEL})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)

# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""
Run structured extraction against a JSON schema with Open Ingest.

Usage:
    python examples/extract_structured.py --file invoice.pdf --schema Invoice --local

Define your schema as a pydantic.BaseModel in this file (or import one from
`tensorlake_docai.extraction.schema_collections`).
"""

import argparse
import base64
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from tensorlake.applications import run_local_application, run_remote_application

from tensorlake_docai.extraction.schema_collections import BankStatement, Receipt
from tensorlake_docai.pipeline.api import (
    ParsedDocument,
    ParseRequest,
    StructuredExtractionRequest,
)
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload

# ---- Example schemas ---------------------------------------------------------


class LineItem(BaseModel):
    description: Optional[str] = Field(None, description="Description of the line item")
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    vendor_name: Optional[str] = None
    customer_name: Optional[str] = None
    line_items: Optional[List[LineItem]] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None


class Customer(BaseModel):
    customer_name: str
    customer_address: str
    account_number: str


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "Invoice": Invoice,
    "Customer": Customer,
    "BankStatement": BankStatement,
    "Receipt": Receipt,
}


# ---- Entry point -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Local path, s3://, or https:// URL")
    parser.add_argument(
        "--schema",
        default="Invoice",
        choices=list(SCHEMA_REGISTRY),
        help="Which BaseModel in this file to extract against",
    )
    parser.add_argument(
        "--ocr-model",
        default="azure-di",
        choices=["dots-ocr", "azure-di", "textract", "gemini"],
    )
    parser.add_argument(
        "--model-provider",
        default="openai",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider for extraction",
    )
    parser.add_argument(
        "--chunk-strategy",
        default="none",
        choices=["none", "page", "section", "fragment"],
    )
    parser.add_argument("--enable-citation", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true", help="VLM-only extraction, no OCR pass")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    schema_cls = SCHEMA_REGISTRY[args.schema]

    se_req = StructuredExtractionRequest(
        json_schema=json.dumps(schema_cls.model_json_schema()),
        model_provider=args.model_provider,
        schema_name=args.schema,
        skip_ocr=args.skip_ocr,
        enable_citation=args.enable_citation,
        chunking_strategy=args.chunk_strategy,
    )

    path = args.file
    if path.startswith("s3://") or path.startswith("http"):
        req = ParseRequest(
            file_url=path,
            file_name=Path(path).name,
            mime_type="application/pdf",
            ocr_model=args.ocr_model,
            structured_extraction_requests=[se_req],
        )
    else:
        local = Path(path)
        req = ParseRequest(
            file_bytes=base64.b64encode(local.read_bytes()).decode(),
            file_name=local.name,
            mime_type="application/pdf",
            ocr_model=args.ocr_model,
            structured_extraction_requests=[se_req],
        )

    runner = run_local_application if args.local else run_remote_application
    handle = runner(normalize_file_type_and_upload, req.model_dump())
    print(f"Request ID: {handle.id}")

    raw = handle.output()
    if not raw or "document" not in raw:
        raise RuntimeError("No document returned")

    parsed = ParsedDocument.model_validate(raw["document"])
    if not parsed.structured_data:
        print("No structured_data returned — check the schema and provider keys")
        return

    print(json.dumps([item.model_dump() for item in parsed.structured_data], indent=2))


if __name__ == "__main__":
    main()

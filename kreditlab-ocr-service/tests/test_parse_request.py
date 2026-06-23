# SPDX-License-Identifier: Apache-2.0
"""ParseRequest schema invariants — these are the public API surface."""

import pytest
from pydantic import ValidationError

from tensorlake_docai.pipeline.api import ParseRequest


def test_minimal_request_with_bytes():
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf", file_bytes="aGVsbG8=")
    assert req.ocr_model == "azure-di"  # default
    assert req.table_parsing_strategy == "vlm"


def test_minimal_request_with_url():
    req = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_url="s3://bucket/key",
    )
    assert req.file_bytes is None


def test_unknown_ocr_model_rejected():
    with pytest.raises(ValidationError):
        ParseRequest(
            file_name="x.pdf",
            mime_type="application/pdf",
            file_bytes="aGVsbG8=",
            ocr_model="marker",  # dropped backend
        )


def test_legacy_model_codes_rejected():
    # Internal `model0X` names from the Inkwell codebase are no longer
    # accepted on the public API.
    for legacy in ("model01", "model02", "model03", "model05", "gemini3"):
        with pytest.raises(ValidationError):
            ParseRequest(
                file_name="x.pdf",
                mime_type="application/pdf",
                file_bytes="aGVsbG8=",
                ocr_model=legacy,
            )


def test_inkwell_only_field_removed():
    # `is_server_presigned_url` was Inkwell-specific and stripped on OSS publish.
    # If it sneaks back in, model_fields will pick it up.
    assert "is_server_presigned_url" not in ParseRequest.model_fields

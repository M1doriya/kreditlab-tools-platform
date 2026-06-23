# SPDX-License-Identifier: Apache-2.0
"""Tests for textract_retry_utils — thin wrapper that adds timeout semantics."""

import time

import pytest

from tensorlake_docai.ocr import textract_retry_utils
from tensorlake_docai.ocr.textract_retry_utils import robust_textract_analyze_document


class _StubExtractor:
    def __init__(self, return_value=None, raise_exc=None):
        self.return_value = return_value
        self.raise_exc = raise_exc
        self.calls = []

    def analyze_document(self, file_source=None, features=None, **kwargs):
        self.calls.append({"file_source": file_source, "features": features, **kwargs})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.return_value


def test_robust_returns_extractor_result():
    stub = _StubExtractor(return_value="doc-result")
    out = robust_textract_analyze_document(stub, file_source="img.png", features=["LAYOUT"])
    assert out == "doc-result"


def test_robust_forwards_features_and_kwargs():
    stub = _StubExtractor(return_value=None)
    robust_textract_analyze_document(
        stub, file_source="x", features=["F1"], save_image=True, extra=1
    )
    call = stub.calls[0]
    assert call["file_source"] == "x"
    assert call["features"] == ["F1"]
    assert call["save_image"] is True
    assert call["extra"] == 1


def test_robust_propagates_fast_exception():
    """Fast failure (before timeout window) re-raises the original exception."""
    stub = _StubExtractor(raise_exc=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        robust_textract_analyze_document(stub, file_source="x", features=[], timeout=300)


def test_robust_converts_slow_failure_to_timeout(monkeypatch):
    """If elapsed >= timeout, the exception is reshaped into a TimeoutError."""
    # First time.time() returns 0 (start), second returns 100 (after the call) so elapsed = 100
    # with timeout=10 -> elapsed >= timeout triggers TimeoutError.
    times = iter([0.0, 100.0])
    monkeypatch.setattr(time, "time", lambda: next(times))

    stub = _StubExtractor(raise_exc=RuntimeError("downstream gave up"))

    with pytest.raises(TimeoutError, match="timed out"):
        robust_textract_analyze_document(stub, file_source="x", features=[], timeout=10)


def test_timeout_constants_present():
    assert textract_retry_utils.TIMEOUT_IMAGE == 600
    assert textract_retry_utils.TIMEOUT_PDF_PAGE == 600
    assert textract_retry_utils.TIMEOUT_LARGE == 600

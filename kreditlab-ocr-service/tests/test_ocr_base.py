# SPDX-License-Identifier: Apache-2.0
"""Tests for the OCRTask protocol defined in ocr/base.py."""

from tensorlake_docai.ocr.base import OCRTask
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.pipeline.api import ParseRequest


def _make_request():
    return ParseRequest(file_name="test.pdf", mime_type="application/pdf")


def test_ocr_task_is_protocol():
    """OCRTask is a runtime-checkable Protocol."""
    assert hasattr(OCRTask, "__protocol_attrs__") or hasattr(OCRTask, "_is_protocol")


def test_conforming_class_is_recognised():
    """A class with a matching run() method satisfies the OCRTask protocol."""

    class FakeOCR:
        def run(self, parse_result: ParseResult) -> ParseResult:
            return parse_result

    assert isinstance(FakeOCR(), OCRTask)


def test_non_conforming_class_is_not_recognised():
    """A class without run() does not satisfy OCRTask."""

    class NotOCR:
        def process(self, x):
            return x

    assert not isinstance(NotOCR(), OCRTask)

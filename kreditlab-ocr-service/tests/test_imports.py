# SPDX-License-Identifier: Apache-2.0
"""Smoke test: every OCR provider module imports and exports its task class.

This is the cheapest test that catches the most common breakage in a
1-hr/week maintenance window: module-level import errors caused by upstream
package changes or refactors that miss a reference.
"""

import importlib

import pytest

OCR_PROVIDERS = [
    ("tensorlake_docai.ocr.azure", "FullPageAzureTask"),
    ("tensorlake_docai.ocr.textract", "FullPageTextractTask"),
    ("tensorlake_docai.ocr.gemini", "FullPageGeminiTask"),
    ("tensorlake_docai.ocr.dots_ocr", "DotsOCRTask"),
    ("tensorlake_docai.ocr.figure_ocr", "OvisFigureOCRTask"),
]


@pytest.mark.parametrize("module_path,cls_name", OCR_PROVIDERS)
def test_ocr_provider_importable(module_path: str, cls_name: str):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name), f"{module_path} missing {cls_name}"


def test_pipeline_modules_importable():
    for mod in (
        "tensorlake_docai.pipeline.api",
        "tensorlake_docai.pipeline.file_converter",
        "tensorlake_docai.pipeline.routing",
        "tensorlake_docai.pipeline.output_formatter",
        "tensorlake_docai.models.intermediate_objects",
        "tensorlake_docai.models.layout_objects",
        "tensorlake_docai.tables.table_merging",
        "tensorlake_docai.vlm.cloud",
        "tensorlake_docai.extraction.structured_extraction_functions",
    ):
        importlib.import_module(mod)

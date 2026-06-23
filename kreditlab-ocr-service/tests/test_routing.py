# SPDX-License-Identifier: Apache-2.0
"""Registry-table invariants — guards against silently dropping a provider
or accidentally re-introducing the dropped `model06` (Marker)."""

from tensorlake_docai.ocr import OCR_BACKENDS, DEFAULT_OCR_MODEL


def test_supported_models():
    assert set(OCR_BACKENDS) == {
        "dots-ocr",
        "azure-di",
        "textract",
        "gemini",
    }


def test_marker_is_dropped():
    assert "model06" not in OCR_BACKENDS
    assert not any("marker" in spec.lower() for spec in OCR_BACKENDS.values())


def test_backend_class_paths_stable():
    # Stringified class paths — the pipeline imports these lazily so that
    # GPU-only modules don't load in non-GPU workers.
    assert OCR_BACKENDS["azure-di"] == "tensorlake_docai.ocr.azure.FullPageAzureTask"
    assert OCR_BACKENDS["textract"] == "tensorlake_docai.ocr.textract.FullPageTextractTask"
    assert OCR_BACKENDS["gemini"] == "tensorlake_docai.ocr.gemini.FullPageGeminiTask"
    assert OCR_BACKENDS["dots-ocr"] == "tensorlake_docai.ocr.dots_ocr.DotsOCRTask"


def test_default_model_is_in_registry():
    assert DEFAULT_OCR_MODEL in OCR_BACKENDS

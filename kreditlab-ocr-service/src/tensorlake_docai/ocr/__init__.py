# SPDX-License-Identifier: Apache-2.0
"""OCR backend registry.

See ``ocr/README.md`` for the step-by-step recipe to add a new backend.
"""

from importlib import import_module
from typing import Optional

# Maps the public ``ParseRequest.ocr_model`` value to the dotted path of the
# Tensorlake ``@cls()`` that implements it. Class paths are stored as strings
# (resolved lazily by :func:`resolve_ocr_backend`) so that GPU-only modules
# such as ``dots_ocr`` — which import heavyweight CUDA deps at
# module load — are only imported when actually dispatched to.
OCR_BACKENDS: dict[str, str] = {
    "azure-di": "tensorlake_docai.ocr.azure.FullPageAzureTask",
    "textract": "tensorlake_docai.ocr.textract.FullPageTextractTask",
    "gemini": "tensorlake_docai.ocr.gemini.FullPageGeminiTask",
    "dots-ocr": "tensorlake_docai.ocr.dots_ocr.DotsOCRTask",
}

DEFAULT_OCR_MODEL = "azure-di"


def resolve_ocr_backend(ocr_model: Optional[str]):
    """Return the Tensorlake task class registered for ``ocr_model``.

    Unknown or ``None`` values fall back to :data:`DEFAULT_OCR_MODEL`. The
    target module is imported on first call.
    """
    spec = OCR_BACKENDS.get(ocr_model or DEFAULT_OCR_MODEL, OCR_BACKENDS[DEFAULT_OCR_MODEL])
    module_path, attr = spec.rsplit(".", 1)
    return getattr(import_module(module_path), attr)

# SPDX-License-Identifier: Apache-2.0
"""
`tl deploy` entrypoint for the Open Ingest pipeline.

This file must sit ONE LEVEL ABOVE the `tensorlake_docai/` package
(i.e. at `src/workflow.py`, not inside `src/tensorlake_docai/`).
`tl deploy` ships the directory containing the entry file as the
zip root, so keeping `tensorlake_docai/` as a sibling preserves the
package name — without that, absolute imports like
`from tensorlake_docai.vlm.cloud import ...` inside the bundled
submodules fail with `ModuleNotFoundError` in the function executor.

The SDK still requires every `@function()`/`@application()` source
file to live under the entry file's directory; `src/` satisfies that
because all functions are defined inside `src/tensorlake_docai/...`.

Usage:
    pip install -e .
    tl deploy src/workflow.py

Invoke with `run_remote_application(normalize_file_type_and_upload, ...)`
from a Python client — see `examples/parse_pdf.py`.
"""

# Application entry — file conversion + OCR routing.
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload  # noqa: F401

# Downstream OCR tasks.
from tensorlake_docai.ocr.azure import FullPageAzureTask  # noqa: F401
from tensorlake_docai.ocr.textract import FullPageTextractTask  # noqa: F401
from tensorlake_docai.ocr.gemini import FullPageGeminiTask  # noqa: F401

# `dots-ocr` GPU path — requires a CUDA-equipped Tensorlake worker.
# Disabled by default so `tl deploy` does NOT build the heavy
# `ocr-gpu-cuda` image (vLLM + CUDA, multi-GB). Re-enable both imports
# once you have a GPU pool provisioned.
# from tensorlake_docai.ocr.dots_ocr import DotsOCRTask  # noqa: F401
# from tensorlake_docai.ocr.figure_ocr import OvisFigureOCRTask  # noqa: F401

# Post-OCR enrichment.
from tensorlake_docai.tables.table_merging import TableMerging  # noqa: F401
from tensorlake_docai.vlm.cloud import VLMExtractionTask  # noqa: F401
from tensorlake_docai.extraction.form_filling import FormFilling  # noqa: F401

# Output formatting (terminal node).
from tensorlake_docai.pipeline.output_formatter import format_final_output  # noqa: F401

# SPDX-License-Identifier: Apache-2.0
"""
Contract every OCR backend in this package follows.

A backend is a Tensorlake ``@cls()`` class with a single ``run`` method that
takes a :class:`ParseResult` and returns a :class:`ParseResult` (or a future
of one, when handing off to a downstream task via ``route_after_ocr``).

To add a new backend, see ``ocr/README.md``. The short version:

1. Create ``ocr/<your_backend>.py`` that defines a class matching the
   :class:`OCRTask` protocol below.
2. End ``run`` with ``return route_after_ocr(parse_result, log_prefix=...)``
   so the downstream TableMerging / StructuredExtraction / VLM / Output
   dispatch stays consistent with the other backends.
3. Add one line to :data:`tensorlake_docai.ocr.OCR_BACKENDS` mapping your
   public model name to the dotted class path, widen the ``ocr_model``
   ``Literal`` in ``pipeline/api.py``, and import the class in
   ``workflow.py`` so ``tl deploy`` picks it up.

See ``ocr/gemini.py`` for the canonical reference implementation.
"""

from typing import Protocol, runtime_checkable

from tensorlake_docai.models.intermediate_objects import ParseResult


@runtime_checkable
class OCRTask(Protocol):
    """Structural type every OCR backend in this package conforms to.

    The ``run`` method is wrapped by Tensorlake's ``@function()`` decorator at
    the class level, so the protocol is documentation rather than something
    the SDK enforces. Backends should still match this signature so the
    pipeline can dispatch to them uniformly.
    """

    def run(self, parse_result: ParseResult) -> ParseResult: ...

# OCR backends

Each file in this directory is a self-contained OCR backend. They all share
the same shape — a Tensorlake `@cls()` whose `run(parse_result)` method
populates `parse_result.document_layout` and hands the result off to the
post-OCR pipeline via `route_after_ocr`.

The dispatcher in `pipeline/file_converter.py` does not know about specific
backends — it looks up `request.ocr_model` in the
[`OCR_BACKENDS`](__init__.py) registry and dispatches by class. Adding a new
backend is therefore three small edits and one new file.

## Add a new backend in 4 steps

Say you want to add `dolphin`, a hypothetical new OCR model.

### 1. Drop in `ocr/dolphin.py`

```python
# SPDX-License-Identifier: Apache-2.0
import os

from tensorlake.applications import cls, function, Retries
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.vlm.workflow_images import simple_page_creator_image


SECRETS = ["DOLPHIN_API_KEY"]


@cls()
class FullPageDolphinTask:
    """Dolphin OCR provider."""

    @function(
        image=simple_page_creator_image,
        timeout=30 * 60,
        cpu=2,
        memory=4,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        if parse_result.request.mime_type.startswith("text/"):
            return parse_result

        # ...call your OCR provider, fill in parse_result.document_layout...

        return route_after_ocr(parse_result, log_prefix="FULL_PAGE_DOLPHIN")
```

The contract is in [`base.py`](base.py): the `run` method must take a
`ParseResult` and return one (or a future of one — `route_after_ocr` handles
both). The early-return for text files matches what every other backend
does; the dispatcher sends text-only requests through the OCR ladder when
no other path applies, so backends must no-op on them.

### 2. Register it in [`ocr/__init__.py`](__init__.py)

```python
OCR_BACKENDS = {
    ...
    "dolphin": "tensorlake_docai.ocr.dolphin.FullPageDolphinTask",
}
```

The dotted-path form keeps the import lazy, which matters for GPU-only
backends (see how `dots-ocr` is wired today — the heavy CUDA imports
only fire when the registry actually resolves that entry).

### 3. Widen the `ocr_model` enum in [`pipeline/api.py`](../pipeline/api.py)

```python
ocr_model: Optional[
    Literal["dots-ocr", "azure-di", "textract", "gemini", "dolphin"]
] = "azure-di"
```

This is pydantic's API-boundary validation — without it, requests with
`ocr_model="dolphin"` are rejected before they reach the registry.

### 4. Import it in [`workflow.py`](../workflow.py) for `tl deploy`

```python
from tensorlake_docai.ocr.dolphin import FullPageDolphinTask  # noqa: F401
```

`tl deploy` walks `workflow.py` to discover every `@function()` / `@cls()`
that should be packed into the deployment bundle. The registry's lazy
import is great for runtime workers but invisible to the deploy CLI —
hence the explicit import here.

That's it. Try a request with `ocr_model="dolphin"` against your local
deployment.

## What you get for free

By calling `route_after_ocr`, your backend automatically participates in:

- **Table merging** when the request asks for it and your output contains tables.
- **Structured extraction** when a schema is provided.
- **VLM enrichment** (figure/table summarization, chart extraction, signature
  detection, key-value extraction, page classification) when those flags are
  set and the relevant fragments exist in your output.
- **Output formatting** as the terminal step.

Pass `dots_ocr=True` if your backend, like `dots_ocr`, gates
VLM/SE on the actual presence of tables/figures/forms in the extracted
layout instead of just the request flags.

## Hosting concerns

The shipped `dots-ocr` backend needs a CUDA worker. Its tasks pin themselves
to `["H100", "A100-80GB"]` via the `gpu=` parameter on their `@function(...)`
decorators and raise a `RequestError` at startup if `torch.cuda.is_available()`
is false. If your new backend has similar host requirements, add an analogous
guard at the top of its `run()` method.

# OCR models

Open Ingest ships four OCR backends. Pick the one that matches your
constraints (cost, latency, features, deployment model) and pass it via
`ocr_model` on the `ParseRequest`.

## Backend comparison

| `ocr_model` | Provider | Native PDF | Cell bboxes | Forms / KV | Barcodes | Custom prompt | Hardware |
|-------------|----------|:---:|:---:|:---:|:---:|:---:|----------|
| `dots-ocr` | DotsOCR + Ovis2.5 | converts | ✅ | ✅ | ✅ | ✅ | CUDA GPU (local or managed) |
| `azure-di` *(default)* | Azure Document Intelligence | ✅ | ✅ | ✅ | — | — | Azure cloud |
| `textract` | AWS Textract | ✅ | ✅ | ✅ | — | — | AWS cloud |
| `gemini` | Google Gemini VLM | ✅ | ✅ | partial | — | — | Google cloud |

`dots-ocr` is the backend with the most serving work in this repo —
open-sourced with the full setup (vLLM, two-stage Ovis figure OCR,
masked-region retries). It needs a CUDA-equipped host — either a
`--local` run on your own GPU, or a managed Tensorlake GPU
deployment. The `@function()` decorators are pre-pinned to
`H100`/`A100-80GB`, but GPU workers aren't part of the open
serverless tier today — reach out to support@tensorlake.ai if you'd
like a managed deployment. Requests that ask for `dots-ocr` without
a GPU fail at task start with a `RequestError` suggesting one of the
cloud backends.

## Picking a model

- **Don't know yet?** Start with `azure-di` — fast cloud OCR with cell-level
  table bboxes, no GPU required.
- **Need signatures or async S3 jobs?** `textract`.
- **VLM-style semantic OCR?** `gemini` — slower but reads context.
- **Complex documents on your own GPU?** `dots-ocr` ships with the full
  serving setup (vLLM, two-stage Ovis figure OCR, masked-region retries).
  Needs a CUDA host (local or managed).

## Required env vars

See [`.env.example`](../.env.example) — each backend lists its keys with the
features it unlocks. Missing keys disable that backend; the rest of the
pipeline still runs.

## Figure OCR (dots-ocr only)

When `ocr_model='dots-ocr'`, DotsOCR outputs are post-processed by
[Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B) running on a
separate GPU container. The Ovis pass classifies each cropped figure
(`BARCODE`, `CHART`, `DIAGRAM`, `FORM`, `TABLE`, `OTHER`) and extracts
content with a type-specific prompt.

# Open Ingest

[![ci](https://github.com/tensorlakeai/openingest/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorlakeai/openingest/actions/workflows/ci.yml)

Open Ingest is a distributed ingestion API that turns unstructured data into LLM-ready outputs: layout-aware markdown, structured data from charts and figures.

It runs on Tensorlake's serverless orchestration engine: Python functions executed inside Firecracker microVMs, with S3 as the durable substrate for RPC, queues, and checkpoints. There's no separate queues or orchestration engine to wire up — the runtime treats S3 as the system of record, so every stage of an ingestion pipeline is durable and resumable by default. 

This codebase has been in production for over a year, processing documents for Tensorlake's customers across finance, healthcare, and legal workloads. We're open-sourcing it so developers can build their own ingestion pipelines on the same runtime — fork the repo, write Python functions for whatever extraction, enrichment, or routing logic your project needs, and Tensorlake orchestrate handles scheduling, scaling, and durability. 

Open Ingest is also the reference implementation for ingestion projects using Tensorlake's Orchestration engine: if you want to see how durable execution, microVM sandboxing, and S3-backed coordination compose into a real production system, the code is here.

**Works out of the box** — point it at a PDF, get back layout-aware markdown, tables, and structured data. Pick an OCR backend, supply a key, you're done.

**And every layer is swappable** if you want to go further:

- **BYO OCR** — 4 backends ship (DotsOCR, Azure DI, Textract, Gemini); register your own in `src/tensorlake_docai/ocr/`.
- **BYO LLM** — provider-agnostic; supply only the keys for what you use (OpenAI, Anthropic, Gemini).
- **BYO schema** — Pydantic `BaseModel` or JSON Schema for structured extraction with citations.
- **Forkable stages** — table merging, VLM enrichment, page classification, structured extraction are all standalone `@function()`s you can swap or extend.
- **`--local` for hacking, `tl deploy` for production** — same `ParseRequest`, two runners.

---

## What it does

Given a PDF, image, DOCX, or Office file, the workflow:

1. **Normalizes** the file. MIME type is detected from content (via `python-magic`) with filename and `Content-Type` as fallbacks, then each format is routed through a dedicated path:

   | Input | What happens | Tool | Downstream MIME |
   |-------|--------------|------|-----------------|
   | PDF | Passed through; page count via `pypdf` | — | `application/pdf` |
   | JPG / PNG / single-page TIFF | Passed through as 1 page | — | `image/*` |
   | Multi-page TIFF | Frame count via `PIL.Image.n_frames` | Pillow | `image/tiff` |
   | DOCX | Parsed in-process to structured pages + bboxes; a preview PDF is rendered and returned as base64 | `docx_parsing` | `text/html` |
   | DOC | Converted to DOCX, then routed to the DOCX path | LibreOffice (`soffice`) | `text/html` |
   | XLSX / XLS / XLSM | Each sheet → HTML table → split on empty rows → Markdown; sheet name kept as `page_class` | `pandas` + `markdownify` | `text/table` |
   | TXT / HTML / CSV / XML / MD | Decoded as UTF-8, with `chardet` fallback for other encodings (CSV preserved for structured extraction) | `chardet` | `text/plain` or `text/csv` |
   | P7M (PKCS#7) | Inner payload extracted, MIME re-detected, then routed by the new type | `openssl cms` / `smime` / `pkcs7` | depends on inner content |
   | Anything else LibreOffice can open (RTF, ODT, PPT, …) | Converted to PDF | LibreOffice (`soffice`) | `application/pdf` |

   Quota and `pages_to_parse` validation runs after normalization, so out-of-range pages fail before any OCR or VLM cost is incurred.
2. **Runs OCR** with the provider of your choice — four backends ship:

   | `ocr_model` | Provider | Best for |
   |-------------|----------|----------|
   | `dots-ocr` (model03) | [DotsOCR](https://github.com/rednote-hilab/dots.mocr) + [Ovis2.5](https://huggingface.co/AIDC-AI/Ovis2.5-9B) on a CUDA GPU | Complex documents — open-sourced with the full serving setup (vLLM, two-stage Ovis figure OCR, masked-region retries). Needs your own GPU host (`--local`) or a managed Tensorlake deployment. |
   | `azure-di` (model01)  | [Azure Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) | Fast cloud OCR with cell-level table bboxes |
   | `textract` (model02) | [AWS Textract](https://aws.amazon.com/textract/) | Native PDF, async S3 jobs |
   | `gemini` | [Google Gemini](https://ai.google.dev/) VLM | VLM-powered semantic OCR |

3. **Enriches** with VLM passes (optional): table summarization, figure
   summarization, table merging for cross-page tables, chart extraction, page classification, signature detection.
4. **Extracts structured data** against a JSON schema with citation tracking.
5. **Returns** a single `ParsedDocument` (pages, fragments, tables, structured
   outputs, usage) — no DB, no webhooks.

See [`docs/pipeline.md`](docs/pipeline.md) for the full DAG and
[`docs/models.md`](docs/models.md) for the OCR backend comparison.

---

## Features

Each stage is independent — toggle only what you need. Every feature below is gated by one field on `ParseRequest` and exposed as a CLI flag on `examples/parse_pdf.py`. Run `python examples/parse_pdf.py --help` for the full list, or [`docs/running.md`](docs/running.md) §4 for the field-by-field map.

### Layout & OCR (always on)

Layout-aware text, tables, figures, charts, formulas, headers, footers, page numbers, and reading order — across four interchangeable OCR backends ([`docs/models.md`](docs/models.md)). Output is a `ParsedDocument` with per-fragment bounding boxes and `ref_id`s, so anything downstream can re-locate the source pixels.

### Table enrichment

| Feature | CLI flag | What it does |
|---|---|---|
| **Cross-page table merging** | `--table-merging` | Stitches tables that wrap across pages, and same-page tables split by intervening text. Uses a fast Gemini "is this a continuation?" prompt per pair and falls back to a visual-alignment pass when column counts disagree. |
| **Table summarization** | `--table-summarization` | One-sentence VLM summary attached to each table; `--table-summarization-prompt` overrides the default. |
| **Table cell grounding** | `--table-cell-grounding` | Per-cell bounding boxes — useful for click-to-source UIs and entity location. |
| **Table output format** | `--table-output-mode {markdown,html,json}` | Markdown by default; HTML preserves merged cells and spans. |

### Figure & chart enrichment

| Feature | CLI flag | What it does |
|---|---|---|
| **Figure summarization** | `--figure-summarization` | VLM caption per figure; `--figure-summarization-prompt` overrides. |
| **Figure grounding** | `--figure-grounding` | Bounding boxes for text regions inside figures. |
| **Chart extraction** | `--chart-extraction` | Extracts the underlying data series as JSON — line, bar, pie, scatter. |
| **Figure OCR prompt** | `--figure-ocr-prompt` | Override the DotsOCR figure-OCR prompt (`dots-ocr` only). |

### Forms

| Feature | CLI flag | What it does |
|---|---|---|
| **Key-value extraction** | `--key-value-extraction` | Pulls key/value pairs out of form-shaped regions without a schema. |
| **Structured extraction** | (use `extract_structured.py`) | Schema-driven extraction with citations — JSON Schema or Pydantic. See [`docs/running.md`](docs/running.md). |
| **Form filling** | `form_filling=FormFillingRequest(...)` | Fills the source PDF/DOCX with extracted values and returns a base64 of the filled doc. Python-only — see [`docs/running.md`](docs/running.md). |

### Detection

| Feature | CLI flag | What it does |
|---|---|---|
| **Signature detection** | `--detect-signature` | Locates signatures via Textract; needs AWS keys. |
| **Barcode detection** | `--detect-barcode` | Decodes 1D/2D barcodes (QR, Code-128, etc.). |

### Page-level routing

| Feature | CLI flag | What it does |
|---|---|---|
| **Page classification** | `--classify NAME:DESCRIPTION` (repeatable) | Multi-label or `--classification-type multi_class` classification using natural-language class definitions. |
| **Cross-page header detection** | `--xpage-header-detection` | Drops repeating page headers/footers from the output. |
| **Page selection** | `--pages 1 2 5` | 1-indexed; saves both money and time on long docs. |
| **Chunking** | `--chunk-strategy {none,page,section,fragment}` | Controls the granularity of `chunks[]` in the output. |
| **Drop fragment types** | `--ignore-sections page_footer figure` | Filter unwanted fragments from the final document. |

---

## Quickstart

Two ways to run — same `ParseRequest` payload, different runner:

- **Local** (`--local`) — no deploy, no Tensorlake account. Every task runs in your Python process. Best for reading, debugging, and iterating on a request.
- **Remote on Tensorlake** — `tl deploy` once, then each task runs in its own container with autoscaling and retries.

### Local run (no deploy)

```bash
git clone https://github.com/tensorlakeai/openingest
cd openingest
pip install -e .

export GEMINI_API_KEY=...  
python examples/parse_pdf.py --file my.pdf --ocr-model gemini --local
```

#### Installing with uv

```bash
git clone https://github.com/tensorlakeai/openingest
cd openingest
uv sync --extra cpu      # CPU machine (installs torch + transformers)
# uv sync --extra gpu    # Linux GPU machine (installs CUDA torch + transformers)
# pip install vllm       # additionally required for dots-ocr on a GPU machine

export GEMINI_API_KEY=...
python examples/parse_pdf.py --file my.pdf --ocr-model gemini --local
```

Results are written to debug/

Add `--draw-bboxes` to also write `debug/bbox_page_N.png` per page, with fragment bounding boxes overlayed on the rendered page image — handy for sanity-checking layout output. Local files only.

`TENSORLAKE_API_KEY` is **not** required for local runs.

`--local` only changes where the workflow itself runs — it doesn't bypass external OCR backends. Each backend needs its own provider keys (`azure-di`, `textract`, `gemini`); `dots-ocr` needs a CUDA-equipped host.

### Remote run on Tensorlake (Production grade)

```bash
cp .env.example .env
$EDITOR .env                   # add TENSORLAKE_API_KEY + provider keys you'll use
set -a; source .env; set +a

bash scripts/sync-secrets.sh   # push provider keys from .env to Tensorlake
tl deploy src/workflow.py
python examples/parse_pdf.py --file my.pdf --ocr-model azure-di
```

`scripts/sync-secrets.sh` reads your `.env` and pushes the provider keys the
workflow declares in `@function(secrets=[...])` to Tensorlake via
`tl secrets set`. Re-run it whenever those values change.

`tl deploy` packages every `@function()` in `src/workflow.py` and ships it to Tensorlake's pool — re-deploy when those source files change.

For structured extraction (works in both modes):

```bash
python examples/extract_structured.py --file invoice.pdf --schema Invoice --local
```

Once deployed, you can view and inspect job runs in the Tensorlake UI:

https://github.com/user-attachments/assets/17c7e834-b0a1-43af-abff-6e8aa64203ae

See [`docs/running.md`](docs/running.md) for the full walkthrough — per-backend key
matrix, VLM/structured-extraction/page-classification examples, and the
`ParseRequest` knobs that toggle each DAG stage.

---

## Bring-your-own-keys

The pipeline is provider-agnostic — supply only the keys for the backends you
plan to use. The `.env.example` file groups them by feature:

- **[Tensorlake](https://tensorlake.ai)** (required if you want a deployed application): `TENSORLAKE_API_KEY`
- **[DotsOCR](https://github.com/rednote-hilab/dots.mocr) on a CUDA GPU** (`dots-ocr`): no provider keys. Run it on any CUDA-equipped host via `--local`, or on a managed Tensorlake GPU deployment — the `@function()` decorators are already pinned to `H100`/`A100-80GB`, but GPU workers aren't part of the open serverless tier today, so reach out to support@tensorlake.ai if you'd like one provisioned. Weights (`rednote-hilab/dots.mocr` and `AIDC-AI/Ovis2.5-9B`) are pulled from Hugging Face Hub on first cold-start.
- **[Azure Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)** (`azure-di`): `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY`
- **[AWS Textract](https://aws.amazon.com/textract/)** (`textract`): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME`
- **[Google Gemini](https://ai.google.dev/) VLM** (`gemini`): `GEMINI_API_KEY`
- **VLM enrichment / structured extraction**: any of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

Missing keys silently disable the dependent feature — the rest of the pipeline
keeps running.

---

## Extending the pipeline

Every stage in the workflow is a Tensorlake `@function()` you can fork, replace,
or add to. Common extension points:

- **Bring your own OCR backend.** Drop a file in `src/tensorlake_docai/ocr/`,
  register it in `ocr/__init__.py`, widen the `ocr_model` enum in
  `pipeline/api.py`, and import it in `workflow.py`. By calling
  `route_after_ocr` at the end of your task, the new backend automatically
  participates in table merging, structured extraction, VLM enrichment, and
  the unified output format. Step-by-step walkthrough in
  [`src/tensorlake_docai/ocr/README.md`](src/tensorlake_docai/ocr/README.md).
- **Add a VLM enrichment pass.** Table/figure summarization, chart extraction,
  and signature detection all live in `src/tensorlake_docai/vlm/cloud.py` as
  batched passes over the document — add another by following the same shape.
- **Drop in your own structured-extraction schema.** Define a Pydantic
  `BaseModel` in your own code (no SDK edits needed) and pass
  `json.dumps(YourModel.model_json_schema())` to `StructuredExtractionRequest`.
  See `examples/extract_structured.py` for a runnable end-to-end wiring and
  [`docs/running.md`](docs/running.md) §"Bringing your own schema" for the
  recommended pattern. The SDK also ships a few sample schemas
  (`Invoice`, `Customer`, `BankStatement`, `Receipt`) in
  `tensorlake_docai.extraction.schema_collections` you can import for quick
  testing.

The `dots-ocr` backend doubles as the reference implementation for serving a
GPU model on this pipeline — vLLM engine setup, model caching across
containers, two-stage classification → extraction, and masked-region retries
all live in `src/tensorlake_docai/ocr/dots_ocr.py` and `figure_ocr.py`.

---

## Project layout

```
openingest/
├── examples/                    # parse_pdf.py, extract_structured.py
├── src/
│   ├── workflow.py              # `tl deploy` entrypoint — must sit one level above the package
│   └── tensorlake_docai/
│       ├── pipeline/            # file_converter, routing, output_formatter, api
│       ├── ocr/                 # azure, textract, gemini, dots_ocr_*, figure_ocr
│       ├── vlm/                 # cloud VLM summarization, grounding, chart extraction
│       ├── extraction/          # structured extraction + chunking + schemas
│       ├── tables/              # cross-page merging, cell grounding, correction
│       ├── postprocess/         # header correction, formatter, output cleaner
│       ├── models/              # ParseResult, PageLayout, etc.
│       ├── providers/           # LLM client wrappers
│       └── prompts/             # prompt templates
├── docs/                        # pipeline.md, models.md, running.md, deployment.md
├── pyproject.toml
└── .env.example
```

---

## Status & support

- **License**: Apache-2.0
- **Maintenance**: bug fixes and security patches, light feature work
- **Issues**: [github.com/tensorlakeai/openingest/issues](https://github.com/tensorlakeai/openingest/issues)
- **Security**: report to support@tensorlake.ai (see [`SECURITY.md`](SECURITY.md))
- **Contributing**: see [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Acknowledgments

This pipeline stands on the shoulders of several open-source models and
libraries. Credit and thanks to:

- **[DotsOCR](https://github.com/rednote-hilab/dots.mocr)** (rednote-hilab) —
  the layout + OCR model behind `dots-ocr`. A small amount of
  preprocessing and output-cleaning code is adapted from the upstream repo
  (see SPDX headers in `src/tensorlake_docai/ocr/` and
  `src/tensorlake_docai/postprocess/output_cleaner.py`).
- **[Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B)** (AIDC-AI) —
  the VLM used for figure OCR classification and extraction when
  `ocr_model='dots-ocr'`.
- **[vLLM](https://github.com/vllm-project/vllm)** — the inference server
  used inside the GPU OCR container.
- **[jdeskew](https://github.com/phamquiluan/jdeskew)** — skew-correction
  used during page preprocessing.

The commercial OCR/VLM providers above (Azure, AWS, Google) are accessed via
their public APIs using your own keys; no provider code is redistributed in
this repo.

# Running the pipeline

Three ways to run, same `ParseRequest` payload:

| Mode | Runner | When to use |
|---|---|---|
| **Local** (`--local`) | `run_local_application` | Reading/debugging code, iterating on a request, no Tensorlake account needed |
| **Remote (Python)** | `run_remote_application` | Real workloads, large/parallel jobs, work that should survive your laptop closing |
| **Remote (HTTP)** | `POST https://api.tensorlake.ai/applications/normalize_file_type_and_upload` | Calling the deployed workflow from a non-Python client (Node, Go, shell, another service) |

You can prototype with `--local`, then drop the flag once you have keys and a deployed workflow. The HTTP path is documented in [`deployment.md`](deployment.md) §4.

---

## 1. Install

```bash
git clone https://github.com/tensorlakeai/OpenIngest
cd OpenIngest
pip install -e .
```

The `-e` (editable) flag means edits under `src/tensorlake_docai/` take effect immediately — no reinstall after each change.

Optional dev tools:

```bash
pip install -e ".[dev]"   # pytest, ruff, black
```

Sanity check:

```bash
python -c "from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload; print('ok')"
pytest tests/ -q
```

---

## 2. Configure keys

Only the providers you actually use need keys. The `.env.example` file groups them by feature.

You can either source a `.env` file or `export` individual variables — both put the values in the environment your Python process inherits:

```bash
# Option A — .env file
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a

# Option B — direct export
export GEMINI_API_KEY=...
```

`export` only lives in the current shell; new terminal = re-export.

### Keys by `ocr_model` 

| `ocr_model` | Required env vars |
|---|---|
| `gemini` | `GEMINI_API_KEY` |
| `azure-di` | `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY` |
| `textract` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME` |
| `dots-ocr` | CUDA GPU — `--local` on your own CUDA host, or a managed Tensorlake GPU deployment (contact support@tensorlake.ai) |

VLM enrichment and structured extraction additionally need one of: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.

`TENSORLAKE_API_KEY` is **only** required for remote runs (`tl deploy` and `run_remote_application`). Pure local runs do not need it.

Missing keys silently disable the dependent feature — the rest of the pipeline keeps running.

---

## 2b. GPU pipeline (`dots-ocr`)

`dots-ocr` runs both DotsOCR (layout + OCR) and Ovis2.5-9B (figure OCR) on a CUDA GPU. Two ways to host it: a `--local` Python process on your own CUDA-equipped host, or a managed Tensorlake GPU deployment. GPU workers aren't part of the open serverless tier today — contact support@tensorlake.ai if you'd like one provisioned.

**Worker setup.** The two GPU tasks (`DotsOCRTask` and `OvisFigureOCRTask`) declare `gpu=["H100", "A100-80GB"]` on their `@function(...)` decorators, so the code is ready for those pools as soon as you have access. If you need a different GPU type, edit the `GPU_MODELS` constant at the top of `src/tensorlake_docai/ocr/dots_ocr.py` and `src/tensorlake_docai/ocr/figure_ocr.py`.

Both tasks raise a `RequestError` at startup if `torch.cuda.is_available()` returns false, so misconfigurations fail fast instead of stalling on a CPU container.

**Model weights.** Both [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) and [AIDC-AI/Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B) are pulled from Hugging Face Hub on first cold-start (~25 GB). The container caches them on its ephemeral disk for its lifetime. Both `@function(...)` decorators already set `min_containers=1, max_containers=1`, so one warm container amortizes the download across requests; raise `max_containers` if you need to absorb more concurrent traffic (each new container pays the cold-start).

**Serving optimizations.** The `dots-ocr` path is where the repo's serving work lives: vLLM engine, two-stage Ovis classification → type-specific extraction (with guided decoding on the classifier), masked-region iterative retry when DotsOCR shows repetition on a page, and separate GPU containers for DotsOCR vs. Ovis figure OCR with batched in-task `vllm.generate()` calls across pages/figures. DotsOCR runs first and then routes to Ovis as a downstream task — they don't run in parallel, but they each get their own GPU container (`max_containers=1` on both), so one isn't starved by the other. See `src/tensorlake_docai/ocr/dots_ocr.py` and `src/tensorlake_docai/ocr/figure_ocr.py`.

Skip this section entirely if you only plan to use `gemini`, `azure-di`, or `textract`.

---

## 3a. Run locally (no deploy)

```bash
export GEMINI_API_KEY=...
python examples/parse_pdf.py --file my.pdf --ocr-model gemini --local
```

What `--local` does:

- Every `@function`/`@application` in `src/tensorlake_docai/` runs **in your current Python process**, sequentially.
- External HTTP calls (Gemini, Azure, Textract, OpenAI, Anthropic) still happen for real — you just skip Tensorlake's cloud orchestrator.
- `print(...)` and `breakpoint()` work normally; you see every routing decision in your terminal.
- Single-process, single-machine — slow for large PDFs (>50 pages).
- `dots-ocr` runs in-process too; you need a CUDA GPU on the host (the task raises a `RequestError` at startup if `torch.cuda.is_available()` is false).

---

## 3b. Run remotely

```bash
export TENSORLAKE_API_KEY=...
# plus whatever provider keys your request will use

tl deploy src/workflow.py
python examples/parse_pdf.py --file my.pdf --ocr-model gemini
```

Remote mode hands the request to the Tensorlake control plane, which runs each task in its own container. Routing, retries, autoscaling (e.g. `@function(max_containers=200, ...)` in `file_converter.py`) only apply here.

Re-deploy any time you change a file under `src/tensorlake_docai/`. Local mode doesn't need that — it re-imports on each invocation.

See [`deployment.md`](deployment.md) for deploy mechanics, scaling knobs, and removing a deployment.

---

## 4. Exercising more of the DAG

Both example scripts construct a `ParseRequest` — that object is the single API surface. Every enrichment / detection stage is gated by a field on `ParseRequest` (defined in [`src/tensorlake_docai/pipeline/api.py`](../src/tensorlake_docai/pipeline/api.py)) and `examples/parse_pdf.py` exposes them as CLI flags. Run `python examples/parse_pdf.py --help` for the full list.

### Just OCR

```bash
python examples/parse_pdf.py --file my.pdf --ocr-model gemini --local
```

### OCR + VLM enrichment

```bash
python examples/parse_pdf.py --file my.pdf --local \
    --ocr-model gemini \
    --table-summarization \
    --figure-summarization \
    --chart-extraction \
    --detect-signature
```

| Flag | Maps to `ParseRequest` field | Notes |
|---|---|---|
| `--table-merging` | `table_merging` | Stitch tables across pages or split by intervening text |
| `--table-summarization` | `table_summarization` | One-sentence VLM summary per table; pair with `--table-summarization-prompt` |
| `--table-cell-grounding` | `table_cell_grounding` | Per-cell bboxes |
| `--figure-summarization` | `figure_summarization` | Pair with `--figure-summarization-prompt` |
| `--figure-grounding` | `figure_grounding` | Bboxes for text regions inside figures |
| `--figure-ocr-prompt` | `figure_ocr_prompt` | DotsOCR figure OCR prompt override (`dots-ocr` only) |
| `--chart-extraction` | `chart_extraction` | Returns chart data as JSON |
| `--key-value-extraction` | `key_value_extraction` | Form-region KV pairs |
| `--detect-signature` | `detect_signature` | Textract-based; needs AWS keys |
| `--detect-barcode` | `detect_barcode` | |
| `--xpage-header-detection` | `xpage_header_detection` | Remove repeating headers/footers |

### OCR + structured extraction

Use `examples/extract_structured.py`:

```bash
python examples/extract_structured.py \
    --file invoice.pdf \
    --schema Invoice \
    --ocr-model gemini \
    --model-provider gemini \
    --chunk-strategy page \
    --enable-citation \
    --local
```

**Bringing your own schema.** You do **not** need to edit anything inside the
`tensorlake_docai` package, and a new schema does **not** require `tl deploy` —
the schema is serialized into the request as a JSON string, so remote workers
see it without ever importing your Pydantic class. Re-deploys are only needed
when you change code under `src/tensorlake_docai/`.

Define a Pydantic `BaseModel` in your own code and pass its JSON schema to
`StructuredExtractionRequest`:

```python
import json
from pydantic import BaseModel, Field
from tensorlake_docai.pipeline.api import ParseRequest, StructuredExtractionRequest

class MyReceipt(BaseModel):
    store: str | None = Field(None, description="Store name")
    receipt_date: str | None = Field(None, description="Date on the receipt")
    total: float | None = Field(None, description="Grand total")

se_req = StructuredExtractionRequest(
    json_schema=json.dumps(MyReceipt.model_json_schema()),
    schema_name="MyReceipt",
    model_provider="openai",          # or "anthropic" | "gemini"
)
req = ParseRequest(file_url="s3://...", structured_extraction_requests=[se_req])
```

The script `examples/extract_structured.py` shows the same wiring end-to-end. It
defines `Invoice` and `Customer` locally, and imports `BankStatement` and
`Receipt` from `tensorlake_docai.extraction.schema_collections` (a sample
collection bundled with the SDK) — both patterns are valid. To try the example
with your own model, add the class to `SCHEMA_REGISTRY` in that file so
`--schema YourName` can find it; for production use, just pass your model into
`StructuredExtractionRequest` directly as shown above.

`--skip-ocr` routes straight from `FILE_CONVERTOR → VLMExtractionTask`, skipping OCR entirely. Useful for screenshots and forms with poor OCR signal.

### Page classification

`--classify NAME:DESCRIPTION` is repeatable; `--classification-type` defaults to `multi_label`.

```bash
python examples/parse_pdf.py --file my.pdf --local \
    --ocr-model gemini \
    --classify invoice:"Has invoice header + line items" \
    --classify contract:"Legal terms, signature block" \
    --classification-type multi_class
```

### Page selection, chunking, table format

```bash
python examples/parse_pdf.py --file my.pdf --local \
    --ocr-model gemini \
    --pages 1 2 5 \
    --chunk-strategy page \
    --table-output-mode html \
    --table-merging
```

| Flag | Maps to `ParseRequest` field |
|---|---|
| `--pages` | `pages_to_parse` (1-indexed) |
| `--chunk-strategy` | `chunk_strategy` — `none` \| `page` \| `section` \| `fragment` |
| `--table-output-mode` | `table_output_mode` — `markdown` \| `html` \| `json` |
| `--ignore-sections` | `ignore_sections` — e.g. `--ignore-sections page_footer figure` |
| `--include-images` | `include_images` |

### Form filling

Form filling needs a `FormFillingRequest` (richer than a single string), so it's not on `parse_pdf.py`'s CLI. Add it inline:

```python
from tensorlake_docai.pipeline.api import FormFillingRequest

ParseRequest(
    ..., form_filling=FormFillingRequest(fill_prompt="Fill applicant fields from the source PDF"),
)
```

Routes through the `FormFilling` task instead of the OCR branch. See `extraction/form_filling.py`.

---

## What you get back

Both examples produce a `ParsedDocument` (`pipeline/api.py`):

| Field | Contents |
|---|---|
| `pages[]` | Every page with `page_fragments[]` (text/table/figure/chart/signature/...), bounding boxes, `ref_id`s |
| `chunks[]` | Flattened content per `chunk_strategy` |
| `structured_data` | Schema-extracted JSON |
| `page_classes[]` | Classification results |
| `merged_tables[]` | Cross-page table stitching |
| `usage` | Token counts per stage (OCR / VLM / extraction / header correction) |

`parse_pdf.py` writes `./debug/document.json` plus one markdown file per chunk. `extract_structured.py` prints `structured_data` to stdout.

---

## Tests as runnable docs

```bash
pytest tests/ -q                          # full suite
pytest tests/test_pipeline_routing.py -v  # one passing test per DAG branch
pytest tests/test_routing.py -v
```

`tests/test_pipeline_routing.py` is the most useful file to skim once — every routing predicate has a test that shows the exact `ParseRequest` fields that send a request down each branch.

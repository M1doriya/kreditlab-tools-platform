# Deployment

The pipeline is shipped as a [Tensorlake](https://tensorlake.ai) workflow.
`tl deploy` registers every `@function()`-decorated task and the
`@application()` entry function with the Tensorlake control plane; from then on
you can invoke the workflow remotely from any client.

## 1. Get the code

Clone (or fork) the repo — `tl deploy` needs `src/workflow.py`,
which isn't shipped via PyPI:

```bash
git clone https://github.com/tensorlakeai/OpenIngest
cd OpenIngest
pip install -e .
```

## 2. Configure keys

Credentials live in **two places**, and you need both for a remote run:

1. **Your local shell** — used by `tl deploy`, `run_remote_application`,
   and any `--local` invocation. This is what `.env` is for.
2. **Tensorlake's secret store** — used by the deployed `@function()`
   containers at runtime. Local env vars do **not** propagate into remote
   workers; you have to register them with Tensorlake explicitly.

### 2a. Local shell

```bash
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
```

For a remote deploy you only strictly need `TENSORLAKE_API_KEY` in the
shell (used by `tl deploy` and `run_remote_application`). For a
`--local` dry-run you need every key the features you'll exercise
require. See [`models.md`](models.md) for the per-backend list.

### 2b. Tensorlake secrets (remote runs)

Each `@function(secrets=[...])` in the codebase declares which env-var
names Tensorlake should inject into its container at runtime. For
example, `pipeline/file_converter.py` requests:

```python
SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
]
```

Register the values you need with `tl secrets set NAME=VALUE`. Names
must match the strings in the `SECRETS = [...]` lists exactly — inside
each function the value is read via `os.environ["NAME"]`.

```bash
# OCR / VLM providers — set only what you'll actually call
tl secrets set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=... \
                AZURE_DOCUMENT_INTELLIGENCE_KEY=...
tl secrets set AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
                AWS_REGION=us-east-1 S3_BUCKET_NAME=...
tl secrets set GEMINI_API_KEY=...

tl secrets ls             # verify (values are hidden)
tl secrets unset NAME       # remove
```

`.env.example` is the canonical list of names you might need —
copy the variable names from there straight into `tl secrets set`.

`scripts/sync-secrets.sh` reads your `.env` and pushes the provider keys
the workflow declares in `@function(secrets=[...])` to Tensorlake via
`tl secrets set`. Re-run it whenever those values change.

```bash
bash scripts/sync-secrets.sh
```

> **Re-deploy after changing secrets.** Per Tensorlake's docs, when you
> add or update a secret used by an already-deployed application, you
> have to re-run `tl deploy` for the new values to take effect.

## 3. Deploy the workflow

```bash
tl deploy src/workflow.py
```

`src/workflow.py` imports the application entrypoint
(`normalize_file_type_and_upload` from `pipeline.file_converter`) and every
downstream task. Tensorlake walks the imported symbols to register the DAG.
The entry file must sit **one level above** the `tensorlake_docai/` package
(i.e. at `src/workflow.py`) so the bundled zip keeps `tensorlake_docai/`
intact — otherwise the absolute imports in the submodules
(`from tensorlake_docai.X import ...`) can't resolve inside the function
executor and you'll see `ModuleNotFoundError: No module named 'tensorlake_docai'`.
The SDK's recursive check (`import_file_path.startswith(code_dir_path)`)
still requires every registered task to live under `src/` — true here, since
they're all defined inside `src/tensorlake_docai/...`.

You should see one line per `@function()` registered, ending with the
deployed application id.

## 4. Invoke it

The deployed app's `@application()` entrypoint is
`normalize_file_type_and_upload` (in `pipeline/file_converter.py`), so its
URL path segment is that function name. You can call it from a Python
client or directly over HTTP.

### Picking a file input

`ParseRequest` accepts the file in **one of two** mutually-exclusive
forms — set exactly one:

| Field | Use it when… | Value |
|---|---|---|
| `file_bytes` | the PDF/image lives on your local disk | base64-encoded raw bytes (string) |
| `file_url` | the file is already in S3 or reachable over HTTP(S) | `"s3://my-bucket/key.pdf"` or `"https://example.com/inv.pdf"` |

`file_name` and `mime_type` are required in both cases — the file
converter dispatches on `mime_type`, and `file_name` shows up in
logs and the output artifact.

**Size caveat for `file_bytes`.** Base64 inflates payload size by
~33 %, so a 10 MB PDF becomes ~13 MB of JSON. The Tensorlake HTTP
ingress will reject very large bodies — a safe rule of thumb is to
keep raw file size under ~25 MB when using `file_bytes`. Anything
bigger should go through `file_url` (upload to S3 first; the Textract
path already needs `S3_BUCKET_NAME` so the creds are usually in
place).

**`s3://` URLs.** Two equivalent forms:

- Bucket-in-URL: `s3://my-bucket/path/to/invoice.pdf` — works without
  `S3_BUCKET_NAME` set.
- Bare key: `s3://path/to/invoice.pdf` — requires `S3_BUCKET_NAME` to
  be registered as a Tensorlake secret; the worker prefixes the bucket
  automatically. The same `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
  `AWS_REGION` secrets used by Textract are reused for the fetch.

**`https://` URLs.** Fetched with a plain `GET` from the worker, no
auth header injected. If the URL needs a signature (e.g. a pre-signed
S3 link), put the signature inside the URL itself.

### From a Python client

Local file via `file_bytes`:

```python
from tensorlake.applications import run_remote_application
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload
from tensorlake_docai.pipeline.api import ParseRequest
import base64, pathlib

req = ParseRequest(
    file_name="invoice.pdf",
    mime_type="application/pdf",
    file_bytes=base64.b64encode(pathlib.Path("invoice.pdf").read_bytes()).decode(),
    ocr_model="azure-di",
)
handle = run_remote_application(normalize_file_type_and_upload, req.model_dump())
print(handle.id)
result = handle.output()
```

Remote file via `file_url` (S3 or HTTPS) — same request, swap the input:

```python
req = ParseRequest(
    file_name="invoice.pdf",
    mime_type="application/pdf",
    file_url="s3://my-bucket/invoices/invoice.pdf",   # or "https://…"
    ocr_model="azure-di",
)
```

### Over HTTP

Same `ParseRequest` payload, just serialized as JSON. The URL segment
matches the `@application()` function name.

All requests must send **three** headers:

- `Authorization: Bearer $TENSORLAKE_API_KEY` — auth.
- `Content-Type: application/json` — describes the body.
- `Accept: application/json` — **required**. Omitting it returns
  `HTTP 400 — accept header must be application/json or text/event-stream`.
  Use `text/event-stream` instead if you want a streaming response.

**Local file (base64 in the body):**

```bash
# Build the request body (ParseRequest as JSON)
python -c "
import base64, json, pathlib
print(json.dumps({
    'file_name': 'invoice.pdf',
    'mime_type': 'application/pdf',
    'file_bytes': base64.b64encode(pathlib.Path('invoice.pdf').read_bytes()).decode(),
    'ocr_model': 'azure-di',
}))" > req.json

# Submit the run
curl https://api.tensorlake.ai/applications/normalize_file_type_and_upload \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  --data @req.json
# → {"request_id":"beae8736ece31ef9"}
```

**Remote file (S3 or HTTPS via `file_url`):**

```bash
cat > req.json <<'JSON'
{
  "file_name": "invoice.pdf",
  "mime_type": "application/pdf",
  "file_url": "s3://my-bucket/invoices/invoice.pdf",
  "ocr_model": "azure-di"
}
JSON

curl https://api.tensorlake.ai/applications/normalize_file_type_and_upload \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  --data @req.json
```

Swap the `file_url` for an `https://...` URL when the file is on a
public web server or behind a pre-signed link.

**Structured extraction over HTTP.** Same endpoint — add a
`structured_extraction_requests` array to the body. Each entry needs a
`schema_name` plus a `json_schema` *string* (the JSON schema, serialized
with `json.dumps(...)`, not a nested JSON object). The schema travels
inside the request, so a new Pydantic model does **not** require
`tl deploy`:

```bash
python - <<'PY' > req.json
import base64, json, pathlib
from tensorlake_docai.extraction.schema_collections import Receipt

print(json.dumps({
    "file_name": "receipt.pdf",
    "mime_type": "application/pdf",
    "file_bytes": base64.b64encode(pathlib.Path("receipt.pdf").read_bytes()).decode(),
    "ocr_model": "gemini",
    "chunk_strategy": "page",
    "structured_extraction_requests": [{
        "schema_name": "Receipt",
        "json_schema": json.dumps(Receipt.model_json_schema()),
        "model_provider": "gemini",   # or "openai" | "anthropic"
        "enable_citation": True,
        # "skip_ocr": True,           # VLM-only path; skips the OCR branch
    }],
}))
PY

curl https://api.tensorlake.ai/applications/normalize_file_type_and_upload \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  --data @req.json
```

The extracted JSON lands in `structured_data` on the returned
`ParsedDocument`. Any locally-defined `BaseModel` works the same way —
swap the `from tensorlake_docai...` import for your own class.

Fetch the result — the GET blocks until the run finishes, so no polling
loop is required:

```bash
curl -X GET \
  https://api.tensorlake.ai/applications/normalize_file_type_and_upload/requests/<request_id>/output \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  -H "Accept: application/json"
```

The response body is the `ParseResult` JSON — the same object
`handle.output()` returns from the Python SDK. See the [Tensorlake
quickstart](https://docs.tensorlake.ai/applications/quickstart) for the
authoritative API reference.

## Local dry-run (no deploy)

For iteration, run the entry function locally with
`run_local_application` — every task executes in-process, no Tensorlake
roundtrip:

```python
from tensorlake.applications import run_local_application
handle = run_local_application(normalize_file_type_and_upload, req.model_dump())
```

The `examples/parse_pdf.py` and `examples/extract_structured.py` scripts both
support `--local` for this mode.

## Scaling knobs

- `TENSORLAKE_MIN_CONTAINERS` — keep N warm containers per `@function`. `0`
  scales to zero (cheap, cold starts). Bump to 1–3 for low-latency prod.
  Applies to the cloud OCR tasks (`azure-di`, `gemini`, `textract`), the
  file converter, and the VLM enrichment task. The two GPU tasks
  (`DotsOCRTask`, `OvisFigureOCRTask`) hardcode `min_containers=1,
  max_containers=1` and ignore this variable.
- `OCR_GPU_MEMORY_IN_GB` — memory ceiling for the `dots-ocr` GPU container.
  Default 32.
- `OVIS_MEMORY_IN_GB` — memory ceiling for the Ovis2.5 figure-OCR GPU
  container. Default 24.
- Per-function `max_containers` — set in code, currently capped at 200
  for the cloud OCR tasks. Raise carefully — each container costs.
  (The GPU tasks are capped at 1 in code; raise both `min_containers` and
  `max_containers` in `dots_ocr.py` / `figure_ocr.py` if you need more.)

The `dots-ocr` GPU containers also need a GPU pool to schedule on, which
isn't part of Tensorlake's open serverless tier today — see [`models.md`](models.md)
for the available hosting paths.

## Removing a deployment

Remove via the Tensorlake dashboard or the cloud SDK — there is no
`tl undeploy` CLI subcommand.

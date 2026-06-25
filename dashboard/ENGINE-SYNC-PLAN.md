# Engine Sync Plan

The dashboard owns its vendored analysis engine under
`financial-statement-analysis-logic/`.

## Current OCR Flow

PDF inputs are converted with Azure Document Intelligence inside the dashboard
Railway service before Claude analysis:

```text
PDF -> Azure Document Intelligence -> Claude -> render
```

LLM Whisperer remains available as a backup credential for OCR work. The
dashboard PDF-to-TXT path uses the Azure and backup OCR variables on the same
Railway service.

## Sync Notes

- The Next.js server runs the local financial statement analyzer.
- `render_bridge.py` is dashboard-only glue for HTML, PDF, and Excel output.
- If the standalone analyzer repo changes, sync only the engine files that are
  still owned by the dashboard integration.
- Keep OCR provider secrets in Railway variables, not in the repository.

## Required Dashboard OCR Config

```bash
SERVICE_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
LLMWHISPERER_API_KEY=...
OCR_MODEL=...
OCR_GPU_MEMORY_IN_GB=...
OVIS_MEMORY_IN_GB=...
TENSORLAKE_MIN_CONTAINERS=...
USE_AZURE_OPENAI=...
AWS_REGION=...
```

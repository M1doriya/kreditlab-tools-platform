# Engine Sync Plan

The dashboard owns its vendored analysis engine under
`financial-statement-analysis-logic/`.

## Current OCR Flow

PDF inputs are converted through the Railway OCR service before Claude analysis:

```text
PDF -> OCR service -> Claude -> render
```

The OCR service uses Azure Document Intelligence first and LLM Whisperer as an
optional backup.

## Sync Notes

- The Next.js server runs the local financial statement analyzer.
- `render_bridge.py` is dashboard-only glue for HTML, PDF, and Excel output.
- If the standalone analyzer repo changes, sync only the engine files that are
  still owned by the dashboard integration.
- Keep OCR provider secrets in the OCR service deployment, not in the dashboard.

## Required Dashboard OCR Config

```bash
SERVICE_API_KEY=...
```

`OCR_SERVICE_URL` is optional. Leave it blank in the current single-service
Railway deployment so the dashboard runs Azure OCR directly from
`AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY`.

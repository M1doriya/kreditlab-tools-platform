# Financial Statement OCR Handover

The financial statement analysis flow now uses the self-hosted Railway OCR
service. The dashboard does not call the legacy hosted OCR API directly.

## Runtime Flow

1. Dashboard receives PDF upload.
2. Dashboard calls the OCR service `POST /parse`.
3. OCR service runs Azure Document Intelligence first.
4. OCR service falls back to LLM Whisperer only when `LLMWHISPERER_API_KEY` is
   configured and Azure fails.
5. Dashboard sends the extracted markdown into the financial analysis flow.

## Dashboard Variables

Required:

```bash
SERVICE_API_KEY=...
```

Optional:

```bash
OCR_SERVICE_URL=
OCR_SERVICE_TIMEOUT_MS=240000
```

Leave `OCR_SERVICE_URL` blank in the current single-service Railway deployment
so the dashboard runs Azure OCR directly. Only set it if the OCR service is
deployed as a separate Railway service or public URL.

## OCR Service Variables

Required:

```bash
SERVICE_API_KEY=...
OCR_MODEL=azure-di
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

Optional:

```bash
LLMWHISPERER_API_KEY=
OCR_GPU_MEMORY_IN_GB=32
OVIS_MEMORY_IN_GB=24
TENSORLAKE_MIN_CONTAINERS=0
USE_AZURE_OPENAI=false
AWS_REGION=us-east-1
```

The `SERVICE_API_KEY` value must match between the dashboard and OCR service.

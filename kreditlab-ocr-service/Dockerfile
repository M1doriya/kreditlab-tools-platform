# Kredit Lab OCR service — OpenIngest (--local) + azure-di, for Railway.
FROM python:3.12-slim

# Native deps OpenIngest needs: poppler (pdf2image), libmagic (python-magic),
# libGL/glib for image libs used in page rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the OpenIngest engine (base deps only — no torch; that's dots-ocr/GPU only)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Service deps
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY service ./service

# Railway injects $PORT at runtime; app.py reads it directly (no shell expansion).
ENV PORT=8000
EXPOSE 8000
CMD ["python", "-m", "service.app"]

# Self-contained image for the Next.js app + the standalone Python
# financial-analysis engine (analyze.py). Built on Debian bookworm, whose
# default python3 is 3.11 — matching the engine's requirements.
#
# Why a Dockerfile (not nixpacks/railpack): Railway's Nixpacks is deprecated and
# was ignoring nixpacks.toml (so Python never got installed), and the
# FINANCIAL_RENDERER_PYTHON_BIN dashboard variable had a hidden trailing-tab in
# its name. Baking Python + deps + the interpreter path into the image removes
# both failure modes for good.
FROM node:20-bookworm-slim

# Python 3.11 + venv/pip, and the system libraries weasyprint needs to render
# PDFs in-process (so the renderer works even without the external service).
# Fonts: Inter for text; Symbola (monochrome outline) as the emoji fallback.
# We deliberately do NOT install fonts-noto-color-emoji: Debian's package is a
# CBDT/CBLC *bitmap* color font and weasyprint renders it as nothing (icons
# vanish). For COLOR icons we vendor a COLR/SVG vector build of Noto Color Emoji
# below — weasyprint draws that via its SVG table.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
      libcairo2 libgdk-pixbuf-2.0-0 libffi8 \
      fontconfig fonts-inter fonts-liberation fonts-dejavu-core \
      fonts-symbola shared-mime-info poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Vendored COLR/SVG color emoji font + a fontconfig rule that makes it win for
# emoji codepoints. fc-cache must run AFTER these are in place so weasyprint
# finds the font on the slim image.
COPY docker/fonts/NotoColorEmoji.ttf /usr/local/share/fonts/NotoColorEmoji.ttf
COPY docker/fonts/99-color-emoji.conf /etc/fonts/conf.d/99-color-emoji.conf
RUN fc-cache -fv

WORKDIR /app

# --- Node dependencies (cached on the lockfile) ---
# --include=dev so `next build` has its build-time devDependencies.
COPY package.json package-lock.json ./
RUN npm ci --include=dev

# --- Python engine dependencies into an isolated venv ---
COPY financial-statement-analysis-logic/requirements.txt financial-statement-analysis-logic/requirements.txt
COPY bank-statement-analysis-logic/requirements.txt bank-statement-analysis-logic/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r financial-statement-analysis-logic/requirements.txt \
 && /opt/venv/bin/pip install --no-cache-dir -r bank-statement-analysis-logic/requirements.txt

# --- App source ---
COPY . .

# NEXT_PUBLIC_* are inlined into the client bundle at build time (see
# lib/supabase.ts), so they must be present when `next build` runs. Railway
# passes service variables to the build; these ARGs capture them.
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY
ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL \
    NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# --- Runtime ---
# Point the server at the venv interpreter that actually has `anthropic` et al.
# Baked here so it can never be broken by a mistyped dashboard variable.
ENV NODE_ENV=production \
    FINANCIAL_RENDERER_PYTHON_BIN=/opt/venv/bin/python \
    BANK_STATEMENT_ANALYZER_PYTHON_BIN=/opt/venv/bin/python

EXPOSE 3000
CMD ["npm", "run", "start"]

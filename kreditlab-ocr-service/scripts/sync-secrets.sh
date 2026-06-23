#!/usr/bin/env bash
# Push secrets from a local .env file to Tensorlake via `tl secrets set`.
#
# Usage:
#   bash scripts/sync-secrets.sh            # uses ./.env
#   bash scripts/sync-secrets.sh path/to/env
#
# After running, redeploy so functions pick up new values:
#   tl deploy src/workflow.py

set -euo pipefail

ENV_FILE="${1:-.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found" >&2
  exit 1
fi

if ! command -v tl >/dev/null 2>&1; then
  echo "error: 'tl' CLI not on PATH. Run 'pip install -e .' first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Keys the deployed workflow declares in @function(secrets=[...]).
# TENSORLAKE_API_KEY / TENSORLAKE_MIN_CONTAINERS are intentionally excluded —
# the first authenticates the CLI/client locally, the second is read at
# deploy time, not by the running function.
SECRET_KEYS=(
  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
  AZURE_DOCUMENT_INTELLIGENCE_KEY
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_REGION
  S3_BUCKET_NAME
  GEMINI_API_KEY
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  QWEN_API_KEY
  USE_AZURE_OPENAI
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_MODEL_DEPLOYMENT_NAME
)

args=()
skipped=()
for k in "${SECRET_KEYS[@]}"; do
  v="${!k:-}"
  if [[ -n "$v" ]]; then
    args+=("$k=$v")
  else
    skipped+=("$k")
  fi
done

if [[ ${#args[@]} -eq 0 ]]; then
  echo "error: no non-empty secret values found in $ENV_FILE" >&2
  exit 1
fi

echo "Pushing ${#args[@]} secret(s) from $ENV_FILE:"
for pair in "${args[@]}"; do
  echo "  - ${pair%%=*}"
done

if [[ ${#skipped[@]} -gt 0 ]]; then
  echo "Skipping (empty in $ENV_FILE): ${skipped[*]}"
fi

tl secrets set "${args[@]}"

echo
echo "Current secrets on Tensorlake:"
tl secrets ls

echo
echo "Done. Redeploy to pick up changes:"
echo "  tl deploy src/workflow.py"

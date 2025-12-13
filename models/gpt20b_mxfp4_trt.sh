#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "openai/gpt-oss-20b" \
  --tp 1 \
  --max-batch 720 \
  --max-tokens 2048 \
  --mem 0.9 \
  --attention-dp false \
  --kv-dtype auto \
  --trust-remote true \
  --cuda-graph-max-batch 720 \
  --stream-interval 20

  # --kv-dtype fp8 \
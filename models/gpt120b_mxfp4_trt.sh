#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "openai/gpt-oss-120b" \
  --tp 8 \
  --mem 0.9 \
  --attention-dp true \
  --kv-dtype auto \
  --trust-remote true \
  --cuda-graph-max-batch 720 \
  --num-postprocess-workers 4 \
  --stream-interval 20

  # --kv-dtype fp8 \
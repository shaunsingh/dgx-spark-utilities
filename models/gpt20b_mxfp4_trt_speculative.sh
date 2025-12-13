#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "openai/gpt-oss-20b" \
  --tp 1 \
  --backend "pytorch" \
  --max-tokens 16384 \
  --mem 0.9 \
  --max-batch 720 \
  --cuda-graph-max-batch 720 \
  --attention-dp false \
  --trust-remote true \
  --enable-speculative \
  --max-draft-len 4 \
  --eagle-model "RedHatAI/gpt-oss-20b-speculator.eagle3" \

  # --kv-dtype fp8 \

  # --attention-dp true \
  # --attention-dp-enable-balance true \
  # --attention-dp-batching-wait-iters 50 \
  # --attention-dp-timeout-iters 1 \
  # --kv-dtype auto \
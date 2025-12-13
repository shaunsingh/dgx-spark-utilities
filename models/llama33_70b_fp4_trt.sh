#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "nvidia/Llama-3.3-70B-Instruct-FP4" \
  --backend "tensorrt" \
  --tp 1 \
  --max-batch 32 \
  --max-tokens 2048 \
  --attention-dp false \
  --kv-dtype auto \
  --trust-remote true \
  --cuda-graph-max-batch 32

  # --kv-dtype fp8 \
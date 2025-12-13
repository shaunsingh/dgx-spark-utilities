#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8" \
  --backend "tensorrt" \
  --tp 4 \
  --ep 4 \
  --max-batch 16 \
  --max-tokens 4096 \
  --mem 0.6 \
  --attention-dp false \
  --trust-remote true \
  --stream-interval 20 \
  --cuda-graph-max-batch 720 \
  --kv-block-reuse false \
  --moe-backend TRTLLM \
  --num-postprocess-workers 4


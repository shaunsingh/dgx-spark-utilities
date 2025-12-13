#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "nvidia/Llama-4-Maverick-17B-128E-Instruct" \
  --backend "pytorch" \
  --tp 8 \
  --ep 8 \
  --mem 0.85 \
  --attention-dp false \
  --kv-dtype auto \
  --trust-remote true \
  --cuda-graph-max-batch 1 \
  --moe-backend TRTLLM \
  --enable-speculative \
  --max-draft-len 3 \
  --eagle-model "nvidia/Llama-4-Maverick-17B-128E-Eagle3" \
  --num-postprocess-workers 4 \


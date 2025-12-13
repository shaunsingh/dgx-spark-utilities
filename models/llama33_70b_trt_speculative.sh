#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "nvidia/Llama-3.3-70B-Instruct-FP4" \
  --tp 1 \
  --max-batch 1024 \
  --max-tokens 2048 \
  --attention-dp false \
  --kv-dtype auto \
  --trust-remote true \
  --cuda-graph-max-batch 1024 \
  --enable-speculative \
  --max-draft-len 3 \
  --eagle-model "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B" \
  --num-postprocess-workers 4 \


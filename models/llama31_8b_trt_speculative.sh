#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "nvidia/Llama-3.1-8B-Instruct-FP4" \
  --backend "pytorch" \
  --tp 1 \
  --max-batch 720 \
  --cuda-graph-max-batch 720 \
  --mem 0.90 \
  --attention-dp false \
  --kv-dtype auto \
  --trust-remote true \
  --enable-speculative \
  --max-draft-len 3 \
  --eagle-model "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
  --num-postprocess-workers 4 \


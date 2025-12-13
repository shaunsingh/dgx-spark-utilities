#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TRTLLM_ENABLE_PDL=1

exec "${ROOT_DIR}/tensorrt.sh" \
  --model "openai/gpt-oss-120b" \
  --tp 1 \
  --backend "pytorch" \
  --max-tokens 16384 \
  --mem 0.9 \
  --kv-block-reuse false \
  --max-batch 1 \
  --cuda-graph-max-batch 1 \
  --trust-remote true \
  --attention-dp false \
  --enable-autotuner false \
  --enable-speculative \
  --max-draft-len 3 \
  --eagle-model "nvidia/gpt-oss-120b-Eagle3-throughput" \
  --moe-backend "CUTLASS" \
  --chunked-prefill true \
  --disable-overlap-scheduler true

#  --attention-dp true \
#  --attention-dp-enable-balance true \
#  --attention-dp-batching-wait-iters 50 \
#  --attention-dp-timeout-iters 1 \
#  --kv-dtype auto \

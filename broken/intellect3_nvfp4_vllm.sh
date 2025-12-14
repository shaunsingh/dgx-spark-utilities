#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_IMAGE="${VLLM_IMAGE:-vllm:intellect3}"

exec "${ROOT_DIR}/vllm.sh" \
  --model "Firworks/INTELLECT-3-nvfp4" \
  --mem 0.90 \
  --max-tokens 23040 \
  --skip-pull \
  --expert-parallel false \
  --trust-remote-code false \
  --extra-args "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser deepseek_r1 --dtype auto"

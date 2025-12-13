#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export VLLM_USE_FLASHINFER_MOE_FP4=1

exec "${ROOT_DIR}/vllm.sh" \
  --model "zai-org/GLM-4.5-Air-FP8" \
  --mem 0.90 \
  --max-tokens 23040 \
  --extra-args "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser deepseek_r1"



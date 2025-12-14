#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ./benchmark.sh --backends vllm --model Firworks/INTELLECT-3-nvfp4 --text-output bench.txt

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  vllm:intellect3 \
  --model Firworks/INTELLECT-3-nvfp4 \
  --dtype auto \
  --max-model-len 23040

  # vllm/vllm-openai:latest \


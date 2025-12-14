#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ./benchmark.sh --backends vllm --model openai/gpt-oss-20b --text-output bench.txt

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm:intellect3 \
  --model openai/gpt-oss-20b \
  --gpu-memory-utilization 0.4 \
  --async-scheduling \
  --tool-call-parser openai \
  --enable-auto-tool-choice

  # vllm/vllm-openai:latest \

#!/usr/bin/env bash
set -euo pipefail

# ./benchmark.sh --backends vllm --model openai/gpt-oss-20b --text-output bench.txt

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  --shm-size=16g \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}" \
  -e FLASHINFER_LOGGING_LEVEL="${FLASHINFER_LOGGING_LEVEL:-error}" \
  vllm:intellect3 \
  --model openai/gpt-oss-20b \
  --async-scheduling \
  --gpu-memory-utilization 0.4 \
  --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --enable-auto-tool-choice \
  --tool-call-parser openai

  # --model 2imi9/gpt-oss-20B-NVFP4A16-BF16 \

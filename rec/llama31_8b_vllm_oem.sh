#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# HuggingFace cache setup
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
HF_HOME_IN_CONTAINER="/root/.cache/huggingface"
HF_CACHE_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-$HOME/.cache/flashinfer}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton}"
TORCH_CACHE="${TORCH_CACHE:-$HOME/.cache/torch}"

mkdir -p "${FLASHINFER_CACHE%/}" "${TRITON_CACHE%/}" "${TORCH_CACHE%/}"

# throughput
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# ./benchmark.sh --backends vllm --model nvidia/Llama-3.1-8B-Instruct-FP4 --text-output bench.txt

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v "${HF_CACHE}:${HF_CACHE_IN_CONTAINER}" \
  -v "${FLASHINFER_CACHE%/}:/root/.cache/flashinfer" \
  -v "${TRITON_CACHE%/}:/root/.cache/triton" \
  -v "${TORCH_CACHE%/}:/root/.cache/torch" \
  -e HF_HOME="${HF_HOME_IN_CONTAINER}" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  -e VLLM_FLOAT32_MATMUL_PRECISION="${VLLM_FLOAT32_MATMUL_PRECISION:-tf32}" \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}" \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}" \
  -e VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-flashinfer-cutlass}" \
  -e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="${VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE:-1}" \
  -e VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING="${VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING:-1}" \
  -e VLLM_COMPILE_CACHE_SAVE_FORMAT="${VLLM_COMPILE_CACHE_SAVE_FORMAT:-binary}" \
  -e FLASHINFER_LOGGING_LEVEL="${FLASHINFER_LOGGING_LEVEL:-error}" \
  vllm:intellect3 \
  --model nvidia/Llama-3.1-8B-Instruct-FP4 \
  --dtype auto \
  --download-dir "${HF_CACHE_IN_CONTAINER}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --disable-log-stats

  # vllm/vllm-openai:latest \
  # --compilation-config '{"cudagraph_mode": "none"}' \
  # --enable-auto-tool-choice \


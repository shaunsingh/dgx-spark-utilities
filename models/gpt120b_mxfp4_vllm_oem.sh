#!/usr/bin/env bash
set -euo pipefail

# ./benchmark.sh --backends vllm --model openai/gpt-oss-120b --text-output bench.txt

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

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

supports_native_fp4() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local cc major
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  [ -n "${cc}" ] || return 1
  major="${cc%%.*}"
  [[ "${major}" =~ ^[0-9]+$ ]] || return 1
  [ "${major}" -ge 10 ]
}

USE_FP4="${USE_FP4:-auto}"          # auto|0|1
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-auto}"  # auto|0|1
DEFAULT_SPECULATIVE_CONFIG='{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'

if [ "${USE_FP4}" = "auto" ]; then
  if supports_native_fp4; then USE_FP4=1; else USE_FP4=0; fi
fi
if [ "${ENABLE_SPECULATIVE}" = "auto" ]; then
  ENABLE_SPECULATIVE="${USE_FP4}"
fi

SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
if [ "${ENABLE_SPECULATIVE}" = "1" ] && [ -z "${SPECULATIVE_CONFIG}" ]; then
  SPECULATIVE_CONFIG="${DEFAULT_SPECULATIVE_CONFIG}"
fi

FP4_ENV=()
if [ "${USE_FP4}" = "1" ]; then
  FP4_ENV+=(-e VLLM_USE_FLASHINFER_MOE_FP4=1)
fi

SPEC_ARGS=()
if [ -n "${SPECULATIVE_CONFIG}" ]; then
  SPEC_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  --shm-size=16g \
  -v "${HF_CACHE}:${HF_CACHE_IN_CONTAINER}" \
  -v "${FLASHINFER_CACHE%/}:/root/.cache/flashinfer" \
  -v "${TRITON_CACHE%/}:/root/.cache/triton" \
  -v "${TORCH_CACHE%/}:/root/.cache/torch" \
  -e HF_HOME="${HF_HOME_IN_CONTAINER}" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  -e VLLM_FLOAT32_MATMUL_PRECISION="${VLLM_FLOAT32_MATMUL_PRECISION:-tf32}" \
  "${FP4_ENV[@]}" \
  -e VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}" \
  -e FLASHINFER_LOGGING_LEVEL="${FLASHINFER_LOGGING_LEVEL:-error}" \
  vllm:intellect3 \
  --model openai/gpt-oss-120b \
  --async-scheduling \
  --download-dir "${HF_CACHE_IN_CONTAINER}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser openai

  # --model 2imi9/gpt-oss-20B-NVFP4A16-BF16 \

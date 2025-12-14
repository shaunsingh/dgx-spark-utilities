#!/usr/bin/env bash
set -euo pipefail

# ./benchmark.sh --backends vllm --model openai/gpt-oss-20b --text-output bench.txt

docker run \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  vllm:intellect3 \
  --model openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --block-size 32 \
  --swap-space 8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --enable-auto-tool-choice \
  --tool-call-parser openai

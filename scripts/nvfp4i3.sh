#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_OPT_TAG="${MODEL_OPT_TAG:-0.35.0}"
MODEL_ID="${MODEL_ID:-PrimeIntellect/INTELLECT-3-FP8}"
QUANT="${QUANT:-nvfp4}"
TP="${TP:-1}"
PP="${PP:-1}"

# Memory knobs
CALIB_SIZE="${CALIB_SIZE:-720}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-23040}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-23040}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-720}"
GPU_MAX_MEM_PCT="${GPU_MAX_MEM_PCT:-0.90}"
KV_CACHE_FREE_FRAC="${KV_CACHE_FREE_FRAC:-0.80}"
USE_SEQ_DEVICE_MAP="${USE_SEQ_DEVICE_MAP:-true}"
LOW_MEMORY_MODE="${LOW_MEMORY_MODE:-false}"

ENABLE_SPEC="${ENABLE_SPEC:-false}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-PrimeIntellect/INTELLECT-3-FP8}"
SPEC_MAX_DRAFT_LEN="${SPEC_MAX_DRAFT_LEN:-4}"

HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
QUANT_OUT_HOST="${QUANT_OUT_HOST:-${ROOT_DIR}/quantized}"

MODEL_STEM="$(basename "${MODEL_ID}" | sed 's/[^0-9A-Za-z-]/_/g')"
PTQ_SAVE_DIR="/workspace/quantized/saved_models_${MODEL_STEM}_${QUANT}_hf"
SPEC_OUT_DIR="/workspace/quantized/speculative_${MODEL_STEM}_${QUANT}"

echo "[nvfp4i3] Quantizing ${MODEL_ID} -> ${QUANT} (TP=${TP}, PP=${PP})"
echo "[nvfp4i3] Calib=${CALIB_SIZE} batch=${CALIB_BATCH_SIZE} max_in=${MAX_INPUT_LEN} max_out=${MAX_OUTPUT_LEN} max_batch=${MAX_BATCH_SIZE}"
echo "[nvfp4i3] GPU mem pct=${GPU_MAX_MEM_PCT} kv_free_frac=${KV_CACHE_FREE_FRAC} seq_device_map=${USE_SEQ_DEVICE_MAP} low_mem=${LOW_MEMORY_MODE}"
[ "${ENABLE_SPEC}" = "true" ] && \
  echo "[nvfp4i3] Speculative training enabled: draft=${SPEC_DRAFT_MODEL}, max_draft_len=${SPEC_MAX_DRAFT_LEN}, algo=${SPEC_ALGO}"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${QUANT_OUT_HOST}:/workspace/quantized" \
  -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -lc "
    set -euo pipefail
    git clone -b ${MODEL_OPT_TAG} --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer
    cd /app/TensorRT-Model-Optimizer
    pip install -e '.[torch]' && pip install compressed-tensors

    export ROOT_SAVE_PATH='/workspace/quantized'
    PTQ_CMD=\"/app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
      --model '${MODEL_ID}' \
      --quant ${QUANT} \
      --tp ${TP} \
      --pp ${PP} \
      --calib ${CALIB_SIZE} \
      --calib_batch_size ${CALIB_BATCH_SIZE} \
      --input ${MAX_INPUT_LEN} \
      --output ${MAX_OUTPUT_LEN} \
      --batch ${MAX_BATCH_SIZE} \
      --tasks build \
      --export_fmt hf \
      --gpu_max_mem_percentage ${GPU_MAX_MEM_PCT} \
      --kv_cache_free_gpu_memory_fraction ${KV_CACHE_FREE_FRAC} \
      --use_seq_device_map ${USE_SEQ_DEVICE_MAP} \
      --low_memory_mode ${LOW_MEMORY_MODE}\"

    echo \"[nvfp4i3] Running PTQ: \${PTQ_CMD}\"
    eval \${PTQ_CMD}

    if [ \"${ENABLE_SPEC}\" = \"true\" ]; then
      SPEC_SCRIPT=\"/app/TensorRT-Model-Optimizer/examples/speculative_decoding/train_speculative_decoder.py\"
      if [ -f \"\${SPEC_SCRIPT}\" ]; then
        echo \"[nvfp4i3] Training speculative decoder...\"
        python \"\${SPEC_SCRIPT}\" \
          --target_model_dir \"${PTQ_SAVE_DIR}\" \
          --draft_model_id \"${SPEC_DRAFT_MODEL}\" \
          --output_dir \"${SPEC_OUT_DIR}\" \
          --max_draft_len ${SPEC_MAX_DRAFT_LEN} \
      else
        echo \"[nvfp4i3] WARN: speculative decoder script not found at \${SPEC_SCRIPT}; skipping.\" >&2
      fi
    fi
  "
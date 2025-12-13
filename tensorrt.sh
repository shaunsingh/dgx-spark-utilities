#!/usr/bin/env bash
set -euo pipefail

# TensorRT-LLM single-node launcher for NVIDIA DGX Spark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime defaults (env overridable)
CONTAINER_NAME="${CONTAINER_NAME:-trtllm-single}"
TRT_IMAGE="${TRT_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5}"
TRT_PORT="${TRT_PORT:-8355}"
TRT_BACKEND="${TRT_BACKEND:-pytorch}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/}"
TIKTOKEN_DIR="${TIKTOKEN_DIR:-${HOME}/tiktoken_encodings}"
SHM_SIZE="${SHM_SIZE:-32g}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
CUDA_GRAPH_PADDING="${CUDA_GRAPH_PADDING:-true}"
CUDA_GRAPH_MAX_BATCH_ENV="${CUDA_GRAPH_MAX_BATCH:-}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KV_CACHE_DTYPE_ENV="${KV_CACHE_DTYPE:-}"
KV_BLOCK_REUSE_ENV="${KV_BLOCK_REUSE:-}"
ATTENTION_DP_ENV="${ATTENTION_DP:-}"
ATTENTION_DP_ENABLE_BALANCE_ENV="${ATTENTION_DP_ENABLE_BALANCE:-}"
ATTENTION_DP_BATCHING_WAIT_ITERS_ENV="${ATTENTION_DP_BATCHING_WAIT_ITERS:-}"
ATTENTION_DP_TIMEOUT_ITERS_ENV="${ATTENTION_DP_TIMEOUT_ITERS:-}"
STREAM_INTERVAL_ENV="${STREAM_INTERVAL:-}"
NUM_POSTPROCESS_ENV="${NUM_POSTPROCESS_WORKERS:-}"
MOE_BACKEND_ENV="${MOE_BACKEND:-}"
EXPERT_PARALLEL_ENV="${EXPERT_PARALLEL_SIZE:-}"
CHUNKED_PREFILL_ENV="${ENABLE_CHUNKED_PREFILL:-}"
TRTLLM_ENABLE_PDL_ENV="${TRTLLM_ENABLE_PDL:-}"
DISABLE_OVERLAP_SCHEDULER_ENV="${DISABLE_OVERLAP_SCHEDULER:-}"
ENABLE_AUTOTUNER_ENV="${ENABLE_AUTOTUNER:-}"
ENABLE_EAGLE_ENV="${ENABLE_EAGLE:-false}"
EAGLE_MODEL_DEFAULT="nvidia/gpt-oss-120b-Eagle3-v2"
EAGLE_MODEL_ENV="${EAGLE_MODEL:-${EAGLE_MODEL_DEFAULT}}"
EAGLE_DRAFT_LEN_ENV="${EAGLE_DRAFT_LEN:-3}"
EAGLE_LAYERS_ENV="${EAGLE_LAYERS:--1}"
EAGLE_SUPPORTED_MODELS=(
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "nvidia/Llama-4-Maverick-17B-128E-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
)

# helpers
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

get_attr() {
  local array_name="$1"
  local idx="$2"
  local ref="${array_name}[@]"
  local arr=("${!ref}")
  echo "${arr[$idx]}"
}

stop_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    log "Stopping existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null || true
  fi
}

CATALOG_FILE="${SCRIPT_DIR}/model_catalog.sh"
[ -f "${CATALOG_FILE}" ] || die "Missing model catalog at ${CATALOG_FILE}"
# shellcheck source=/dev/null
source "${CATALOG_FILE}"
catalog_load "trt"
DEFAULT_MODEL="${DEFAULT_MODEL:-${CATALOG_DEFAULT_MODEL}}"

list_models() {
  echo "Available models (single node):"
  for i in "${!MODELS[@]}"; do
    local marker=""
    [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ] && marker=" [default]"
    [ "$(get_attr MODEL_NEEDS_TOKEN "$i")" = "true" ] && marker="${marker} [HF token]"
    printf "  %2d. %s%s\n" "$((i + 1))" "${MODEL_NAMES[$i]}" "${marker}"
  done
}

usage() {
  cat << EOF
Usage: $0 [options]

Options:
  -m, --model <id|number>   Model id or menu number (default: ${DEFAULT_MODEL})
  -l, --list                List available models and exit
  --skip-pull               Do not pull Docker image
  --port <port>             API port (default: ${TRT_PORT})
  --container-name <name>   Container name (default: ${CONTAINER_NAME})
  --image <name>            TensorRT-LLM image (default: ${TRT_IMAGE})
  --backend <name>          Backend (default: ${TRT_BACKEND})
  --tp <n>                  Tensor parallel (default: ${TENSOR_PARALLEL})
  --max-tokens <n>          Override max tokens
  --max-batch <n>           Override max batch size
  --mem <fraction>          KV cache free memory fraction (0-1, default per model)
  --trust-remote <true|false>  Override trust_remote_code
  --shm-size <size>         Shared memory size (default: ${SHM_SIZE})
  --extra-args "<args>"     Extra args passed to trtllm-serve
  --attention-dp <bool>     Enable attention DP (default: per model)
  --attention-dp-enable-balance <bool>  Balance rebalancing across ranks
  --attention-dp-batching-wait-iters <n>  Iters to wait before batching balance
  --attention-dp-timeout-iters <n>  Timeout iterations for attention DP
  --kv-dtype <auto|fp8|fp16> KV cache dtype (default: auto/per model)
  --stream-interval <n>     Push token stream interval
  --num-postprocess-workers <n>  Postprocess worker threads
  --moe-backend <name>      MoE backend (e.g., TRTLLM)
  --ep, --expert-parallel-size <n>  MoE expert parallel size
  --kv-block-reuse <bool>   Enable/disable KV block reuse
  --enable-autotuner <bool> Enable/disable TRT-LLM autotuner
  --cuda-graph-max-batch <n>  Max batch size for CUDA graph capture
  --chunked-prefill <bool>  Enable chunked prefill
  --disable-overlap-scheduler <bool>  Disable overlap scheduler
  --enable-speculative      Enable Eagle speculative decoding (120b)
  --eagle-model <path|hf>   Draft model dir/id (default: ${EAGLE_MODEL_DEFAULT})
  --max-draft-len <n>       Max draft length for Eagle (default: 3)
  --eagle-layers <list>     Layers to capture, e.g. "-1" or "-1, -2"
  --no-wait                 Do not wait for health
  --no-test                 Skip quick inference test
  -h, --help                Show this help
EOF
}

# arg parsing
MODEL_INPUT=""
LIST_ONLY=false
SKIP_PULL=false
NO_WAIT=false
NO_TEST=false
TP_OVERRIDE=""
MAX_TOKENS_OVERRIDE=""
MAX_BATCH_OVERRIDE=""
MEM_OVERRIDE=""
TRUST_OVERRIDE=""
PORT_OVERRIDE=""
IMAGE_OVERRIDE=""
BACKEND_OVERRIDE=""
SHM_OVERRIDE=""
CONTAINER_OVERRIDE=""
ATT_DP_OVERRIDE=""
KV_DTYPE_OVERRIDE=""
STREAM_INTERVAL_OVERRIDE=""
NUM_POSTPROCESS_OVERRIDE=""
MOE_BACKEND_OVERRIDE=""
EP_OVERRIDE=""
KV_BLOCK_REUSE_OVERRIDE=""
CUDA_GRAPH_MAX_BATCH_OVERRIDE=""
ATT_DP_ENABLE_BALANCE_OVERRIDE=""
ATT_DP_BATCHING_WAIT_ITERS_OVERRIDE=""
ATT_DP_TIMEOUT_ITERS_OVERRIDE=""
CHUNKED_PREFILL_OVERRIDE=""
DISABLE_OVERLAP_SCHEDULER_OVERRIDE=""
ENABLE_AUTOTUNER_OVERRIDE=""
ENABLE_EAGLE_FLAG=""
EAGLE_MODEL_OVERRIDE=""
EAGLE_DRAFT_LEN_OVERRIDE=""
EAGLE_LAYERS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_INPUT="$2"; shift 2 ;;
    -l|--list) LIST_ONLY=true; shift ;;
    --skip-pull) SKIP_PULL=true; shift ;;
    --port) PORT_OVERRIDE="$2"; shift 2 ;;
    --container-name) CONTAINER_OVERRIDE="$2"; shift 2 ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    --backend) BACKEND_OVERRIDE="$2"; shift 2 ;;
    --tp) TP_OVERRIDE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS_OVERRIDE="$2"; shift 2 ;;
    --max-batch) MAX_BATCH_OVERRIDE="$2"; shift 2 ;;
    --mem) MEM_OVERRIDE="$2"; shift 2 ;;
    --trust-remote) TRUST_OVERRIDE="$2"; shift 2 ;;
    --shm-size) SHM_OVERRIDE="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --attention-dp) ATT_DP_OVERRIDE="$2"; shift 2 ;;
    --attention-dp-enable-balance) ATT_DP_ENABLE_BALANCE_OVERRIDE="$2"; shift 2 ;;
    --attention-dp-batching-wait-iters) ATT_DP_BATCHING_WAIT_ITERS_OVERRIDE="$2"; shift 2 ;;
    --attention-dp-timeout-iters) ATT_DP_TIMEOUT_ITERS_OVERRIDE="$2"; shift 2 ;;
    --kv-dtype) KV_DTYPE_OVERRIDE="$2"; shift 2 ;;
    --stream-interval) STREAM_INTERVAL_OVERRIDE="$2"; shift 2 ;;
    --num-postprocess-workers) NUM_POSTPROCESS_OVERRIDE="$2"; shift 2 ;;
    --moe-backend) MOE_BACKEND_OVERRIDE="$2"; shift 2 ;;
    --ep|--expert-parallel-size) EP_OVERRIDE="$2"; shift 2 ;;
    --kv-block-reuse) KV_BLOCK_REUSE_OVERRIDE="$2"; shift 2 ;;
    --enable-autotuner) ENABLE_AUTOTUNER_OVERRIDE="$2"; shift 2 ;;
    --cuda-graph-max-batch) CUDA_GRAPH_MAX_BATCH_OVERRIDE="$2"; shift 2 ;;
    --chunked-prefill|--enable-chunked-prefill) CHUNKED_PREFILL_OVERRIDE="$2"; shift 2 ;;
    --disable-overlap-scheduler) DISABLE_OVERLAP_SCHEDULER_OVERRIDE="$2"; shift 2 ;;
    --enable-speculative|--speculative) ENABLE_SPECULATIVE_FLAG=true; shift ;;
    --eagle-model) EAGLE_MODEL_OVERRIDE="$2"; shift 2 ;;
    --max-draft-len|--draft-len) EAGLE_DRAFT_LEN_OVERRIDE="$2"; shift 2 ;;
    --eagle-layers) EAGLE_LAYERS_OVERRIDE="$2"; shift 2 ;;
    --no-wait) NO_WAIT=true; shift ;;
    --no-test) NO_TEST=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [ "${LIST_ONLY}" = "true" ]; then
  list_models
  exit 0
fi

resolve_model_index() {
  local input="$1"
  if [ -z "${input}" ]; then
    for i in "${!MODELS[@]}"; do
      if [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ]; then
        echo "$i"; return
      fi
    done
  fi

  if [[ "${input}" =~ ^[0-9]+$ ]]; then
    local idx=$((input - 1))
    [ "${idx}" -ge 0 ] && [ "${idx}" -lt "${#MODELS[@]}" ] && { echo "${idx}"; return; }
    die "Model number out of range (1-${#MODELS[@]})"
  fi

  for i in "${!MODELS[@]}"; do
    if [ "${MODELS[$i]}" = "${input}" ]; then
      echo "$i"; return
    fi
  done

  die "Unknown model: ${input}"
}

MODEL_IDX="$(resolve_model_index "${MODEL_INPUT}")"
MODEL_ID="${MODELS[$MODEL_IDX]}"
MODEL_NAME="${MODEL_NAMES[$MODEL_IDX]}"
MODEL_MEM_FRACTION="${MEM_OVERRIDE:-$(get_attr MODEL_MEM "${MODEL_IDX}")}"
MODEL_MAXTOK="${MAX_TOKENS_OVERRIDE:-$(get_attr MODEL_MAX_TOKENS "${MODEL_IDX}")}"
MODEL_BATCH="${MAX_BATCH_OVERRIDE:-$(get_attr MODEL_BATCH_SIZE "${MODEL_IDX}")}"
MODEL_TRUST="${TRUST_OVERRIDE:-$(get_attr MODEL_TRUST_REMOTE "${MODEL_IDX}")}"
MODEL_NEEDS_TOKEN_FLAG="$(get_attr MODEL_NEEDS_TOKEN "${MODEL_IDX}")"
ATTENTION_DP="${ATT_DP_OVERRIDE:-${ATTENTION_DP_ENV:-$(get_attr MODEL_ATT_DP "${MODEL_IDX}")}}"
ATT_DP_ENABLE_BALANCE="${ATT_DP_ENABLE_BALANCE_OVERRIDE:-${ATTENTION_DP_ENABLE_BALANCE_ENV:-}}"
ATT_DP_BATCHING_WAIT_ITERS="${ATT_DP_BATCHING_WAIT_ITERS_OVERRIDE:-${ATTENTION_DP_BATCHING_WAIT_ITERS_ENV:-}}"
ATT_DP_TIMEOUT_ITERS="${ATT_DP_TIMEOUT_ITERS_OVERRIDE:-${ATTENTION_DP_TIMEOUT_ITERS_ENV:-}}"
KV_CACHE_DTYPE="${KV_DTYPE_OVERRIDE:-${KV_CACHE_DTYPE_ENV:-$(get_attr MODEL_KV_DTYPE "${MODEL_IDX}")}}"
STREAM_INTERVAL="${STREAM_INTERVAL_OVERRIDE:-${STREAM_INTERVAL_ENV:-}}"
NUM_POSTPROCESS_WORKERS="${NUM_POSTPROCESS_OVERRIDE:-${NUM_POSTPROCESS_ENV:-}}"
MOE_BACKEND="${MOE_BACKEND_OVERRIDE:-${MOE_BACKEND_ENV:-}}"
EXPERT_PARALLEL_SIZE="${EP_OVERRIDE:-${EXPERT_PARALLEL_ENV:-}}"
KV_BLOCK_REUSE="${KV_BLOCK_REUSE_OVERRIDE:-${KV_BLOCK_REUSE_ENV:-}}"
CUDA_GRAPH_MAX_BATCH="${CUDA_GRAPH_MAX_BATCH_OVERRIDE:-${CUDA_GRAPH_MAX_BATCH_ENV:-}}"
CHUNKED_PREFILL="${CHUNKED_PREFILL_OVERRIDE:-${CHUNKED_PREFILL_ENV:-}}"
DISABLE_OVERLAP_SCHEDULER="${DISABLE_OVERLAP_SCHEDULER_OVERRIDE:-${DISABLE_OVERLAP_SCHEDULER_ENV:-}}"
if [ "${ENABLE_EAGLE_FLAG}" = "true" ]; then
  ENABLE_EAGLE="true"
else
  ENABLE_EAGLE="${ENABLE_EAGLE_ENV}"
fi
EAGLE_MODEL="${EAGLE_MODEL_OVERRIDE:-${EAGLE_MODEL_ENV}}"
EAGLE_DRAFT_LEN="${EAGLE_DRAFT_LEN_OVERRIDE:-${EAGLE_DRAFT_LEN_ENV}}"
EAGLE_LAYERS="${EAGLE_LAYERS_OVERRIDE:-${EAGLE_LAYERS_ENV}}"
KV_BLOCK_REUSE_EFFECTIVE="${KV_BLOCK_REUSE}"
ENABLE_AUTOTUNER_VALUE="${ENABLE_AUTOTUNER_OVERRIDE:-${ENABLE_AUTOTUNER_ENV:-}}"
if [ "${ENABLE_EAGLE}" = "true" ]; then
  KV_BLOCK_REUSE_EFFECTIVE="false"
  [ -z "${DISABLE_OVERLAP_SCHEDULER}" ] && DISABLE_OVERLAP_SCHEDULER="true"
  ENABLE_AUTOTUNER_VALUE="false"
fi

TRT_PORT="${PORT_OVERRIDE:-${TRT_PORT}}"
TRT_IMAGE="${IMAGE_OVERRIDE:-${TRT_IMAGE}}"
TRT_BACKEND="${BACKEND_OVERRIDE:-${TRT_BACKEND}}"
CONTAINER_NAME="${CONTAINER_OVERRIDE:-${CONTAINER_NAME}}"
SHM_SIZE="${SHM_OVERRIDE:-${SHM_SIZE}}"
TENSOR_PARALLEL="${TP_OVERRIDE:-${TENSOR_PARALLEL}}"

require_cmd docker
require_cmd curl

if [ "${MODEL_NEEDS_TOKEN_FLAG}" = "true" ] && [ -z "${HF_TOKEN:-}" ]; then
  die "Model ${MODEL_ID} requires HF_TOKEN. Export HF_TOKEN first."
fi

if [ "${ENABLE_EAGLE}" = "true" ]; then
  if ! printf '%s\n' "${EAGLE_SUPPORTED_MODELS[@]}" | grep -Fxq -- "${MODEL_ID}"; then
    log "Warning: Eagle speculative decoding currently validated for: ${EAGLE_SUPPORTED_MODELS[*]}; proceeding anyway."
  fi
fi

PULL_FLAG=""
if [ "${SKIP_PULL}" != "true" ]; then
  PULL_FLAG="--pull=missing"
  log "Using docker pull policy: ${PULL_FLAG}"
else
  log "Skipping image pull (--skip-pull)"
fi

log "Model: ${MODEL_NAME} (${MODEL_ID})"
log "TP: ${TENSOR_PARALLEL}, mem: ${MODEL_MEM_FRACTION}, max_tokens: ${MODEL_MAXTOK}, max_batch: ${MODEL_BATCH}"
log "Port: ${TRT_PORT}, Image: ${TRT_IMAGE}, Container: ${CONTAINER_NAME}"
log "Attention DP: ${ATTENTION_DP}, KV dtype: ${KV_CACHE_DTYPE}, Stream interval: ${STREAM_INTERVAL:-none}"
[ "${ATTENTION_DP}" = "true" ] && [ -n "${ATT_DP_ENABLE_BALANCE}${ATT_DP_BATCHING_WAIT_ITERS}${ATT_DP_TIMEOUT_ITERS}" ] && \
  log "Attention DP config: enable_balance=${ATT_DP_ENABLE_BALANCE:-default}, batching_wait_iters=${ATT_DP_BATCHING_WAIT_ITERS:-default}, timeout_iters=${ATT_DP_TIMEOUT_ITERS:-default}"
[ -n "${CHUNKED_PREFILL}" ] && log "Chunked prefill: ${CHUNKED_PREFILL}"
[ -n "${KV_BLOCK_REUSE_EFFECTIVE}" ] && log "KV block reuse: ${KV_BLOCK_REUSE_EFFECTIVE}"
[ -n "${ENABLE_AUTOTUNER_VALUE}" ] && log "Autotuner: ${ENABLE_AUTOTUNER_VALUE}"
[ -n "${DISABLE_OVERLAP_SCHEDULER}" ] && log "Disable overlap scheduler: ${DISABLE_OVERLAP_SCHEDULER}"
[ -n "${EXPERT_PARALLEL_SIZE}" ] && log "Expert parallel: ${EXPERT_PARALLEL_SIZE}"
[ -n "${MOE_BACKEND}" ] && log "MoE backend: ${MOE_BACKEND}"
[ -n "${NUM_POSTPROCESS_WORKERS}" ] && log "Postprocess workers: ${NUM_POSTPROCESS_WORKERS}"
[ "${ENABLE_EAGLE}" = "true" ] && log "Speculative (Eagle): model=${EAGLE_MODEL}, max_draft_len=${EAGLE_DRAFT_LEN}, layers=[${EAGLE_LAYERS}]"

mkdir -p "${HF_CACHE}" "${TIKTOKEN_DIR}"

ensure_tokenizer() {
  local file="$1"
  local url="$2"
  if [ ! -f "${file}" ]; then
    log "Downloading $(basename "${file}")..."
    curl -fL --retry 3 --retry-delay 2 -o "${file}" "${url}" || die "Failed to download ${file}"
  fi
}

ensure_tokenizer "${TIKTOKEN_DIR}/o200k_base.tiktoken" \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ensure_tokenizer "${TIKTOKEN_DIR}/cl100k_base.tiktoken" \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

stop_container

TRT_CFG="$(mktemp /tmp/trtllm-api.XXXX.yml)"
KV_CFG_DTYPE="${KV_CACHE_DTYPE}"
if [ "${TRT_BACKEND}" = "tensorrt" ]; then
  KV_CFG_DTYPE="auto"
fi
cat > "${TRT_CFG}" << EOF
enable_attention_dp: ${ATTENTION_DP}
max_batch_size: ${MODEL_BATCH}
max_num_tokens: ${MODEL_MAXTOK}
EOF
if [ -n "${CHUNKED_PREFILL}" ]; then
cat >> "${TRT_CFG}" << EOF
enable_chunked_prefill: ${CHUNKED_PREFILL}
EOF
fi
if [ "${ATTENTION_DP}" = "true" ] && [ -n "${ATT_DP_ENABLE_BALANCE}${ATT_DP_BATCHING_WAIT_ITERS}${ATT_DP_TIMEOUT_ITERS}" ]; then
cat >> "${TRT_CFG}" << EOF
attention_dp_config:
EOF
  if [ -n "${ATT_DP_ENABLE_BALANCE}" ]; then
cat >> "${TRT_CFG}" << EOF
  enable_balance: ${ATT_DP_ENABLE_BALANCE}
EOF
  fi
  if [ -n "${ATT_DP_BATCHING_WAIT_ITERS}" ]; then
cat >> "${TRT_CFG}" << EOF
  batching_wait_iters: ${ATT_DP_BATCHING_WAIT_ITERS}
EOF
  fi
  if [ -n "${ATT_DP_TIMEOUT_ITERS}" ]; then
cat >> "${TRT_CFG}" << EOF
  timeout_iters: ${ATT_DP_TIMEOUT_ITERS}
EOF
  fi
fi
if [ -n "${DISABLE_OVERLAP_SCHEDULER}" ]; then
cat >> "${TRT_CFG}" << EOF
disable_overlap_scheduler: ${DISABLE_OVERLAP_SCHEDULER}
EOF
fi
if [ -n "${ENABLE_AUTOTUNER_VALUE}" ]; then
cat >> "${TRT_CFG}" << EOF
enable_autotuner: ${ENABLE_AUTOTUNER_VALUE}
EOF
fi
cat >> "${TRT_CFG}" << EOF
kv_cache_config:
  dtype: "${KV_CFG_DTYPE}"
  free_gpu_memory_fraction: ${MODEL_MEM_FRACTION}
EOF
if [ -n "${KV_BLOCK_REUSE_EFFECTIVE}" ]; then
cat >> "${TRT_CFG}" << EOF
  enable_block_reuse: ${KV_BLOCK_REUSE_EFFECTIVE}
EOF
fi

if [ "${TRT_BACKEND}" = "pytorch" ]; then
cat >> "${TRT_CFG}" << EOF
print_iter_log: false
cuda_graph_config:
  enable_padding: ${CUDA_GRAPH_PADDING}
EOF
  if [ -n "${CUDA_GRAPH_MAX_BATCH}" ]; then
cat >> "${TRT_CFG}" << EOF
  max_batch_size: ${CUDA_GRAPH_MAX_BATCH}
EOF
  fi
fi

if [ -n "${STREAM_INTERVAL}" ] && [ "${TRT_BACKEND}" = "pytorch" ]; then
cat >> "${TRT_CFG}" << EOF
stream_interval: ${STREAM_INTERVAL}
EOF
fi

if [ -n "${NUM_POSTPROCESS_WORKERS}" ]; then
cat >> "${TRT_CFG}" << EOF
num_postprocess_workers: ${NUM_POSTPROCESS_WORKERS}
EOF
fi

if [ -n "${EXPERT_PARALLEL_SIZE}" ]; then
cat >> "${TRT_CFG}" << EOF
moe_expert_parallel_size: ${EXPERT_PARALLEL_SIZE}
EOF
fi

if [ -n "${MOE_BACKEND}" ]; then
cat >> "${TRT_CFG}" << EOF
moe_config:
  backend: ${MOE_BACKEND}
EOF
fi

if [ "${ENABLE_EAGLE}" = "true" ]; then
cat >> "${TRT_CFG}" << EOF
speculative_config:
  decoding_type: Eagle
  max_draft_len: ${EAGLE_DRAFT_LEN}
  speculative_model_dir: "${EAGLE_MODEL}"
  eagle3_layers_to_capture: [${EAGLE_LAYERS}]
EOF
fi

TRT_ARGS=(
  --tp_size "${TENSOR_PARALLEL}"
  --backend "${TRT_BACKEND}"
  --max_num_tokens "${MODEL_MAXTOK}"
  --max_batch_size "${MODEL_BATCH}"
  --host 0.0.0.0
  --port "${TRT_PORT}"
  --extra_llm_api_options /etc/trtllm-api-config.yml
)
[ "${MODEL_TRUST}" = "true" ] && TRT_ARGS+=(--trust_remote_code)
if [ -n "${EXTRA_ARGS}" ]; then
  read -ra EXTRA_ARR <<< "${EXTRA_ARGS}"
  TRT_ARGS+=("${EXTRA_ARR[@]}")
fi

ENV_ARGS=(
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings"
  -e "NCCL_DEBUG=${NCCL_DEBUG}"
  -e "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  -e "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
  -e "NCCL_TIMEOUT=${NCCL_TIMEOUT}"
)
[ -n "${TRTLLM_ENABLE_PDL_ENV}" ] && ENV_ARGS+=(-e "TRTLLM_ENABLE_PDL=${TRTLLM_ENABLE_PDL_ENV}")

VOLUME_ARGS=(
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${TIKTOKEN_DIR}:/tiktoken_encodings:ro"
  -v "${TRT_CFG}:/etc/trtllm-api-config.yml:ro"
)

DEVICE_ARGS=()
[ -d "/dev/infiniband" ] && DEVICE_ARGS+=(--device=/dev/infiniband)

log "Starting container..."
docker run -d \
  --init \
  --restart no \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${TRT_PORT}:${TRT_PORT}" \
  --shm-size="${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  "${DEVICE_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  ${PULL_FLAG:+${PULL_FLAG}} \
  "${TRT_IMAGE}" \
  trtllm-serve serve "${MODEL_ID}" "${TRT_ARGS[@]}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  die "Container failed to start. Check: docker logs ${CONTAINER_NAME}"
fi

API_URL="http://127.0.0.1:${TRT_PORT}"

stream_logs_until_ready() {
  local start=${SECONDS}
  log "Streaming container logs until health endpoint is ready..."
  docker logs -f --tail 50 "${CONTAINER_NAME}" &
  local log_pid=$!
  cleanup_logs() {
    kill "${log_pid}" >/dev/null 2>&1 || true
    wait "${log_pid}" 2>/dev/null || true
  }
  while true; do
    if curl -sf --max-time 2 "${API_URL}/health" >/dev/null 2>&1; then
      local elapsed=$(( SECONDS - start ))
      cleanup_logs
      log "API ready in ${elapsed}s"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      cleanup_logs
      log "Container ${CONTAINER_NAME} stopped before health ready"
      return 1
    fi
    sleep 2
  done
}

print_logs_tail() {
  docker logs --tail 80 "${CONTAINER_NAME}" 2>&1 || true
}

if [ "${NO_WAIT}" != "true" ]; then
  if ! stream_logs_until_ready; then
    log "API failed health check; recent container logs:"
    print_logs_tail
    exit 1
  fi
else
  log "Skipping wait (--no-wait)"
fi

if [ "${NO_TEST}" != "true" ]; then
  log "Running quick test..."
  TEST_BODY=$(cat << JSON
{"model":"${MODEL_ID}","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}
JSON
)
  if curl -sf "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${TEST_BODY}" >/dev/null 2>&1; then
    log "Inference test: PASSED"
  else
    log "Inference test: FAILED (check docker logs ${CONTAINER_NAME})"
    print_logs_tail
    exit 1
  fi
else
  log "Skipping test (--no-test)"
fi

echo ""
echo " TensorRT-LLM single-node is running"
echo "   Model:   ${MODEL_ID}"
echo "   API:     ${API_URL}"
echo "   Health:  ${API_URL}/health"
echo "   Logs:    docker logs -f ${CONTAINER_NAME}"
echo "   Stop:    docker rm -f ${CONTAINER_NAME}"

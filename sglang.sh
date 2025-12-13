#!/usr/bin/env bash
set -euo pipefail

# SGLang single-node launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime defaults (env overridable)
CONTAINER_NAME="${CONTAINER_NAME:-sglang-single}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:spark}"
SGLANG_PORT="${SGLANG_PORT:-8002}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/}"
TIKTOKEN_DIR="${TIKTOKEN_DIR:-${HOME}/tiktoken_encodings}"
SHM_SIZE="${SHM_SIZE:-32g}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-false}"
ATTENTION_BACKEND_DEFAULT="${ATTENTION_BACKEND_DEFAULT:-}"
# KV_CACHE_DTYPE_DEFAULT="${KV_CACHE_DTYPE_DEFAULT:-fp4_e2m1}"
KV_CACHE_DTYPE_DEFAULT="${KV_CACHE_DTYPE_DEFAULT:-fp8_e4m3}"
SCHEDULE_CONSERVATIVENESS_DEFAULT="${SCHEDULE_CONSERVATIVENESS_DEFAULT:-0.25}"
SPECULATIVE_ALGO_DEFAULT="${SPECULATIVE_ALGO_DEFAULT:-}"
SPECULATIVE_DRAFT_MODEL_DEFAULT="${SPECULATIVE_DRAFT_MODEL_DEFAULT:-}"
SPECULATIVE_MAX_DRAFT_DEFAULT="${SPECULATIVE_MAX_DRAFT_DEFAULT:-}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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

CATALOG_FILE="${SCRIPT_DIR}/model_catalog.sh"
[ -f "${CATALOG_FILE}" ] || die "Missing model catalog at ${CATALOG_FILE}"
# shellcheck source=/dev/null
source "${CATALOG_FILE}"
catalog_load "sgl"
DEFAULT_MODEL="${DEFAULT_MODEL:-${CATALOG_DEFAULT_MODEL}}"

list_models() {
  echo "Available models (single-node):"
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
  -m, --model <id|number>  Model id or menu number (default: ${DEFAULT_MODEL})
  -l, --list               List available models and exit
  --skip-pull              Do not pull Docker image
  --port <port>            API port (default: ${SGLANG_PORT})
  --container-name <name>  Container name (default: ${CONTAINER_NAME})
  --extra-args "<args>"    Extra args passed to sglang.launch_server
  --mem <fraction>         Override mem fraction (0.0-1.0)
  --attention-backend <b>  Attention backend (default: ${ATTENTION_BACKEND_DEFAULT})
  --kv-cache-dtype <t>     KV cache dtype (default: ${KV_CACHE_DTYPE_DEFAULT})
  --schedule-conservativeness <v>  Scheduler aggressiveness (default: ${SCHEDULE_CONSERVATIVENESS_DEFAULT})
  --speculative-algo <a>   Speculative decoding algorithm (e.g., EAGLE)
  --speculative-draft-model <id>  Draft model id/path for speculative decode
  --speculative-max-draft <n>     Max draft tokens (optional)
  --no-wait                Do not wait for health
  --no-test                Skip quick inference test
  -h, --help               Show this help
EOF
}

# arg parsing
MODEL_INPUT=""
LIST_ONLY=false
SKIP_PULL=false
NO_WAIT=false
NO_TEST=false
MEM_OVERRIDE=""
ATTENTION_BACKEND="${ATTENTION_BACKEND_DEFAULT}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE_DEFAULT}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS_DEFAULT}"
SPECULATIVE_ALGO="${SPECULATIVE_ALGO_DEFAULT}"
SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL_DEFAULT}"
SPECULATIVE_MAX_DRAFT="${SPECULATIVE_MAX_DRAFT_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL_INPUT="$2"
      shift 2
      ;;
    -l|--list)
      LIST_ONLY=true
      shift
      ;;
    --skip-pull)
      SKIP_PULL=true
      shift
      ;;
    --port)
      SGLANG_PORT="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    --attention-backend)
      ATTENTION_BACKEND="$2"
      shift 2
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --schedule-conservativeness)
      SCHEDULE_CONSERVATIVENESS="$2"
      shift 2
      ;;
    --speculative-algo)
      SPECULATIVE_ALGO="$2"
      shift 2
      ;;
    --speculative-draft-model)
      SPECULATIVE_DRAFT_MODEL="$2"
      shift 2
      ;;
    --speculative-max-draft)
      SPECULATIVE_MAX_DRAFT="$2"
      shift 2
      ;;
    --mem)
      MEM_OVERRIDE="$2"
      shift 2
      ;;
    --no-wait)
      NO_WAIT=true
      shift
      ;;
    --no-test)
      NO_TEST=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "${LIST_ONLY}" = "true" ]; then
  list_models
  exit 0
fi

# resolve model index
resolve_model_index() {
  local input="$1"
  if [ -z "${input}" ]; then
    for i in "${!MODELS[@]}"; do
      if [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ]; then
        echo "$i"
        return
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
      echo "$i"
      return
    fi
  done

  die "Unknown model: ${input}"
}

MODEL_IDX="$(resolve_model_index "${MODEL_INPUT}")"
MODEL_ID="${MODELS[$MODEL_IDX]}"
MODEL_NAME="${MODEL_NAMES[$MODEL_IDX]}"
MODEL_MEM_FRACTION="${MEM_OVERRIDE:-$(get_attr MODEL_MEM "${MODEL_IDX}")}"
MODEL_REASONING="$(get_attr MODEL_REASONING_PARSER "${MODEL_IDX}")"
MODEL_TOOL="$(get_attr MODEL_TOOL_PARSER "${MODEL_IDX}")"
MODEL_TRUST="$(get_attr MODEL_TRUST_REMOTE "${MODEL_IDX}")"
MODEL_NEEDS_TOKEN_FLAG="$(get_attr MODEL_NEEDS_TOKEN "${MODEL_IDX}")"

TENSOR_PARALLEL=1

[ -z "${MODEL_MEM_FRACTION}" ] && MODEL_MEM_FRACTION="0.90"

# # infer attention if possible
# infer_attention_backend() {
#   if [ -n "${ATTENTION_BACKEND}" ]; then
#     log "Using user-specified attention backend '${ATTENTION_BACKEND}'"
#     return
#   fi
# 
#   # spark doesn't support mha/mla???
#   case "${MODEL_ID}" in
# #     openai/gpt-oss-*)                                ATTENTION_BACKEND="trtllm_mha" ;;
# #     *DeepSeek-V3*|*DeepSeek-V2*|*DeepSeek-R1*)       ATTENTION_BACKEND="trtllm_mla" ;;
# #     meta-llama/*|*Llama*|*llama*)                    ATTENTION_BACKEND="trtllm_mha" ;;
# #     *Qwen*|*qwen*)                                   ATTENTION_BACKEND="trtllm_mha" ;;
# #     *Gemma*|*gemma*)                                 ATTENTION_BACKEND="trtllm_mha" ;;
#     openai/gpt-oss-*)                                ATTENTION_BACKEND="fa3" ;;
#     *DeepSeek-V3*|*DeepSeek-V2*|*DeepSeek-R1*)       ATTENTION_BACKEND="flashmla" ;;
#     meta-llama/*|*Llama*|*llama*)                    ATTENTION_BACKEND="fa3" ;;
#     *Qwen*|*qwen*)                                   ATTENTION_BACKEND="fa3" ;;
#     *Gemma*|*gemma*)                                 ATTENTION_BACKEND="fa3" ;;
#     *)
#                                                      ATTENTION_BACKEND="${ATTENTION_BACKEND_DEFAULT:-fa3}" ;;
#   esac
# 
#   log "Auto-selected attention backend '${ATTENTION_BACKEND}' for ${MODEL_ID}"
# }
# 
# # enforce backend compatibility
# enforce_backend_constraints() {
#   if [[ "${MODEL_ID}" == openai/gpt-oss-* ]]; then
#     local GPT_OSS_ALLOWED_BACKENDS=(triton trtllm_mha fa3 fa4)
#     for b in "${GPT_OSS_ALLOWED_BACKENDS[@]}"; do
#       if [ "${ATTENTION_BACKEND}" = "${b}" ]; then
#         return
#       fi
#     done
#     log "Attention backend '${ATTENTION_BACKEND}' unsupported for GPT-OSS; using 'fa3'"
#     ATTENTION_BACKEND="fa3"
#   fi
# }
# 
# infer_attention_backend
# enforce_backend_constraints

# validate
require_cmd docker
require_cmd curl
require_cmd python3

if [ "${MODEL_NEEDS_TOKEN_FLAG}" = "true" ] && [ -z "${HF_TOKEN:-}" ]; then
  die "Model ${MODEL_ID} requires HF_TOKEN. Export HF_TOKEN first."
fi

log "Model: ${MODEL_NAME} (${MODEL_ID})"
log "PP: ${PIPELINE_PARALLEL}, mem: ${MODEL_MEM_FRACTION}"
log "Port: ${SGLANG_PORT}, Image: ${SGLANG_IMAGE}, Container: ${CONTAINER_NAME}"

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

PULL_FLAG=""
if [ "${SKIP_PULL}" != "true" ]; then
  PULL_FLAG="--pull=missing"
  log "Using docker pull policy: ${PULL_FLAG}"
else
  log "Skipping image pull (--skip-pull)"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  log "Stopping existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null || true
fi

# build sglang args
SGLANG_ARGS=(
  --model-path "${MODEL_ID}"
  --tp "${TENSOR_PARALLEL}"
  --pp-size "${PIPELINE_PARALLEL}"
  --nnodes 1
  --node-rank 0
  --host 0.0.0.0
  --port "${SGLANG_PORT}"
  --mem-fraction-static "${MODEL_MEM_FRACTION}"
)

[ -n "${ATTENTION_BACKEND}" ] && SGLANG_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
[ -n "${KV_CACHE_DTYPE}" ] && SGLANG_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[ -n "${SCHEDULE_CONSERVATIVENESS}" ] && SGLANG_ARGS+=(--schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}")
if [ -n "${MODEL_REASONING}" ]; then
  SGLANG_ARGS+=(--reasoning-parser "${MODEL_REASONING}")
fi
if [ -n "${MODEL_TOOL}" ]; then
  SGLANG_ARGS+=(--tool-call-parser "${MODEL_TOOL}")
fi
[ "${DISABLE_CUDA_GRAPH}" = "true" ] && SGLANG_ARGS+=(--disable-cuda-graph)
if [ "${MODEL_TRUST}" = "true" ]; then
  SGLANG_ARGS+=(--trust-remote-code)
fi
if [ -n "${SPECULATIVE_ALGO}" ]; then
  SGLANG_ARGS+=(--speculative-algorithm "${SPECULATIVE_ALGO}")
  [ -n "${SPECULATIVE_DRAFT_MODEL}" ] && SGLANG_ARGS+=(--speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL}")
  [ -n "${SPECULATIVE_MAX_DRAFT}" ] && SGLANG_ARGS+=(--speculative-max-draft-tokens "${SPECULATIVE_MAX_DRAFT}")
fi

if [ -n "${EXTRA_ARGS}" ]; then
  read -ra EXTRA_ARR <<< "${EXTRA_ARGS}"
  SGLANG_ARGS+=("${EXTRA_ARR[@]}")
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

VOLUME_ARGS=(
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${TIKTOKEN_DIR}:/tiktoken_encodings"
)

DEVICE_ARGS=()
[ -d "/dev/infiniband" ] && DEVICE_ARGS+=(--device=/dev/infiniband)

# launch container
log "Starting container..."
docker run -d \
  --init \
  --restart no \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${SGLANG_PORT}:${SGLANG_PORT}" \
  --shm-size="${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  "${DEVICE_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  ${PULL_FLAG:+${PULL_FLAG}} \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  die "Container failed to start. Check: docker logs ${CONTAINER_NAME}"
fi

# health and smoke test
API_URL="http://127.0.0.1:${SGLANG_PORT}"

stream_logs_until_ready() {
  local start=${SECONDS}
  local cleanup_logs
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

if [ "${NO_WAIT}" != "true" ]; then
  stream_logs_until_ready || log "Warning: API not ready after streaming logs"
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
  fi
else
  log "Skipping test (--no-test)"
fi

echo ""
echo " SGLang single-node is running"
echo "   Model:   ${MODEL_ID}"
echo "   API:     ${API_URL}"
echo "   Health:  ${API_URL}/health"
echo "   Logs:    docker logs -f ${CONTAINER_NAME}"
echo "   Stop:    docker rm -f ${CONTAINER_NAME}"
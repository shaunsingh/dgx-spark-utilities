#!/usr/bin/env bash
set -euo pipefail

# vLLM single-node launcher (no InfiniBand or cluster orchestration)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CATALOG_FILE="${SCRIPT_DIR}/model_catalog.sh"
[ -f "${CATALOG_FILE}" ] || { echo "Missing model catalog at ${CATALOG_FILE}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${CATALOG_FILE}"
catalog_load "vllm"
DEFAULT_MODEL="${DEFAULT_MODEL:-${CATALOG_DEFAULT_MODEL}}"

# Runtime defaults (env overridable)
CONTAINER_NAME="${CONTAINER_NAME:-vllm-single}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.11-py3}"
VLLM_PORT="${VLLM_PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
SHM_SIZE="${SHM_SIZE:-16g}"
SWAP_SPACE="${SWAP_SPACE:-16}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# cache in container
HF_HOME_IN_CONTAINER="${HF_HOME:-/root/.cache/huggingface}"
HF_CACHE_IN_CONTAINER="${HF_HOME_IN_CONTAINER%/}/hub"
HF_CACHE_MOUNT="${HF_CACHE%/}"

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

usage() {
  cat << EOF
Usage: $0 [options]

Options:
  -m, --model <id|number>   Model id or menu number (default: ${DEFAULT_MODEL})
  -l, --list                List available models and exit
  --port <port>             vLLM API port (default: ${VLLM_PORT})
  --image <name>            vLLM image (default: ${VLLM_IMAGE})
  --container-name <name>   Container name (default: ${CONTAINER_NAME})
  --hf-cache <path>         HF cache path (default: ${HF_CACHE})
  --tp <n>                  Tensor parallel size (default: from catalog or 1)
  --mem <fraction>          GPU memory util (default per model)
  --max-tokens <tokens>     Max tokens/context (default per model)
  --trust-remote <bool>     Force trust_remote_code on/off
  --expert-parallel <bool>  Force enable_expert_parallel on/off
  --shm-size <size>         Shared memory size (default: ${SHM_SIZE})
  --swap <gb>               Swap space in GB (default: ${SWAP_SPACE})
  --load-format <fmt>       Load format (default: ${LOAD_FORMAT})
  --extra-args "<args>"     Extra args passed to vllm serve
  --skip-pull               Skip docker pull
  --no-wait                 Do not wait for health endpoint
  --no-test                 Skip quick inference test
  --stop                    Stop existing container and exit
  -h, --help                Show this help
EOF
}

list_models() {
  echo "Available models:"
  for i in "${!MODELS[@]}"; do
    local marker=""
    [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ] && marker=" [default]"
    [ "$(get_attr MODEL_NEEDS_TOKEN "$i")" = "true" ] && marker="${marker} [HF token]"
    printf "  %2d. %s%s\n" "$((i + 1))" "${MODEL_NAMES[$i]}" "${marker}"
  done
}

stop_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    log "Stopping ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null || true
  fi
}

print_logs_tail() {
  docker exec "${CONTAINER_NAME}" bash -lc "tail -n 80 /var/log/vllm.log" 2>/dev/null || true
}

stream_logs_until_ready() {
  local url="$1"
  local start=${SECONDS}
  log "Streaming vLLM logs until health endpoint is ready..."
  docker exec "${CONTAINER_NAME}" bash -lc "tail -n 50 -f /var/log/vllm.log" &
  local log_pid=$!
  cleanup_logs() {
    kill "${log_pid}" >/dev/null 2>&1 || true
    wait "${log_pid}" 2>/dev/null || true
  }
  while true; do
    if curl -sf --max-time 2 "${url}/health" >/dev/null 2>&1; then
      cleanup_logs
      log "vLLM healthy after $((SECONDS - start))s"
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

# Arg parsing
MODEL_INPUT=""
LIST_ONLY=false
STOP_ONLY=false
SKIP_PULL=false
NO_WAIT=false
NO_TEST=false
TP_OVERRIDE=""
MEM_OVERRIDE=""
MAX_TOKENS_OVERRIDE=""
PORT_OVERRIDE=""
IMAGE_OVERRIDE=""
CONTAINER_OVERRIDE=""
HF_CACHE_OVERRIDE=""
SHM_OVERRIDE=""
SWAP_OVERRIDE=""
LOAD_FORMAT_OVERRIDE=""
TRUST_OVERRIDE=""
EXPERT_OVERRIDE=""
EXTRA_ARGS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_INPUT="$2"; shift 2 ;;
    -l|--list) LIST_ONLY=true; shift ;;
    --stop) STOP_ONLY=true; shift ;;
    --tp) TP_OVERRIDE="$2"; shift 2 ;;
    --mem) MEM_OVERRIDE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS_OVERRIDE="$2"; shift 2 ;;
    --port) PORT_OVERRIDE="$2"; shift 2 ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    --container-name) CONTAINER_OVERRIDE="$2"; shift 2 ;;
    --hf-cache) HF_CACHE_OVERRIDE="$2"; shift 2 ;;
    --shm-size) SHM_OVERRIDE="$2"; shift 2 ;;
    --swap) SWAP_OVERRIDE="$2"; shift 2 ;;
    --load-format) LOAD_FORMAT_OVERRIDE="$2"; shift 2 ;;
    --trust-remote) TRUST_OVERRIDE="$2"; shift 2 ;;
    --expert-parallel) EXPERT_OVERRIDE="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS_OVERRIDE="$2"; shift 2 ;;
    --skip-pull) SKIP_PULL=true; shift ;;
    --no-wait) NO_WAIT=true; shift ;;
    --no-test) NO_TEST=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [ "${LIST_ONLY}" = "true" ]; then
  list_models
  exit 0
fi

require_cmd docker
require_cmd curl
require_cmd ip

if [ "${STOP_ONLY}" = "true" ]; then
  stop_container
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
MODEL_MAX_TOKENS="${MAX_TOKENS_OVERRIDE:-$(get_attr MODEL_MAX_TOKENS "${MODEL_IDX}")}"
MODEL_TRUST="${TRUST_OVERRIDE:-$(get_attr MODEL_TRUST_REMOTE "${MODEL_IDX}")}"
[ -z "${MODEL_TRUST}" ] && MODEL_TRUST="false"
MODEL_NEEDS_TOKEN_FLAG="$(get_attr MODEL_NEEDS_TOKEN "${MODEL_IDX}")"
MODEL_EXPERT_PARALLEL="${EXPERT_OVERRIDE:-$(get_attr MODEL_EXPERT_PARALLEL "${MODEL_IDX}")}"

HF_CACHE="${HF_CACHE_OVERRIDE:-${HF_CACHE}}"
VLLM_PORT="${PORT_OVERRIDE:-${VLLM_PORT}}"
VLLM_IMAGE="${IMAGE_OVERRIDE:-${VLLM_IMAGE}}"
CONTAINER_NAME="${CONTAINER_OVERRIDE:-${CONTAINER_NAME}}"
SHM_SIZE="${SHM_OVERRIDE:-${SHM_SIZE}}"
SWAP_SPACE="${SWAP_OVERRIDE:-${SWAP_SPACE}}"
LOAD_FORMAT="${LOAD_FORMAT_OVERRIDE:-${LOAD_FORMAT}}"
EXTRA_ARGS="${EXTRA_ARGS_OVERRIDE:-${EXTRA_ARGS}}"

[ -z "${MODEL_MAX_TOKENS}" ] && MODEL_MAX_TOKENS=8192
TENSOR_PARALLEL="${TP_OVERRIDE:-${TENSOR_PARALLEL:-1}}"

if [ "${MODEL_NEEDS_TOKEN_FLAG}" = "true" ] && [ -z "${HF_TOKEN:-}" ]; then
  die "Model ${MODEL_ID} requires HF_TOKEN. Export HF_TOKEN first."
fi

log "Model: ${MODEL_NAME} (${MODEL_ID})"
log "TP: ${TENSOR_PARALLEL}, mem: ${MODEL_MEM_FRACTION}, max_tokens: ${MODEL_MAX_TOKENS}"
log "Port: ${VLLM_PORT}, Image: ${VLLM_IMAGE}, HF cache: ${HF_CACHE_MOUNT}"

mkdir -p "${HF_CACHE_MOUNT}"

if [ "${SKIP_PULL}" != "true" ]; then
  log "Pulling image ${VLLM_IMAGE}..."
  docker pull "${VLLM_IMAGE}" >/dev/null
else
  log "Skipping image pull (--skip-pull)"
fi

stop_container

DOCKER_ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  DOCKER_ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi

mapfile -t VLLM_ENV_NAMES < <(compgen -v VLLM_)
VLLM_ENV_ARGS=()
for name in "${VLLM_ENV_NAMES[@]}"; do
  VLLM_ENV_ARGS+=("${name}=${!name}")
done

log "Starting container ${CONTAINER_NAME}..."
docker run -d \
  --restart unless-stopped \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --shm-size="${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_NICE \
  -v "${HF_CACHE_MOUNT}:${HF_CACHE_IN_CONTAINER}" \
  "${DOCKER_ENV_ARGS[@]}" \
  "${VLLM_IMAGE}" sleep infinity

docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}" || die "Container failed to start"

docker exec "${CONTAINER_NAME}" env \
  HF_HOME="${HF_HOME_IN_CONTAINER}" \
  ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
  bash -lc "mkdir -p /var/log && touch /var/log/vllm.log" >/dev/null 2>&1 || true

log "Downloading model ${MODEL_ID}..."
docker exec "${CONTAINER_NAME}" env \
  HF_HOME="${HF_HOME_IN_CONTAINER}" \
  ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
  bash -lc "hf download ${MODEL_ID} ${HF_TOKEN:+--token ${HF_TOKEN}} --exclude 'original/*' --exclude 'metal/*' >/tmp/hf.log 2>&1" \
  || die "Model download failed. Check docker exec ${CONTAINER_NAME} cat /tmp/hf.log"

log "Starting vLLM server..."
docker exec "${CONTAINER_NAME}" bash -lc "pkill -f 'vllm serve' 2>/dev/null || true" || true

VLLM_ARGS=(
  --host 0.0.0.0
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL}"
  --max-model-len "${MODEL_MAX_TOKENS}"
  --gpu-memory-utilization "${MODEL_MEM_FRACTION}"
  --swap-space "${SWAP_SPACE}"
  --download-dir "${HF_CACHE_IN_CONTAINER}"
  --load-format "${LOAD_FORMAT}"
)

[ "${MODEL_EXPERT_PARALLEL}" = "true" ] && VLLM_ARGS+=(--enable-expert-parallel)
[ "${MODEL_TRUST}" = "true" ] && VLLM_ARGS+=(--trust-remote-code)

if [ -n "${EXTRA_ARGS}" ]; then
  read -ra EXTRA_ARR <<< "${EXTRA_ARGS}"
  VLLM_ARGS+=("${EXTRA_ARR[@]}")
fi

docker exec "${CONTAINER_NAME}" env \
  HF_HOME="${HF_HOME_IN_CONTAINER}" \
  ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
  "${VLLM_ENV_ARGS[@]}" \
  bash -lc "export PYTHONUNBUFFERED=1; nohup vllm serve ${MODEL_ID} ${VLLM_ARGS[*]} > /var/log/vllm.log 2>&1 &"

API_URL="http://127.0.0.1:${VLLM_PORT}"

if [ "${NO_WAIT}" != "true" ]; then
  if ! stream_logs_until_ready "${API_URL}"; then
    log "Health not ready; recent logs:"
    print_logs_tail
  fi
else
  log "Skipping health wait (--no-wait)"
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
    log "Inference test: FAILED (tailing logs)"
    print_logs_tail
  fi
else
  log "Skipping test (--no-test)"
fi

PUBLIC_IP=$(ip -o addr show | awk '/inet / && $2 !~ /^lo/ && $2 !~ /^docker/ {print $4}' | cut -d/ -f1 | head -1)

echo ""
echo " vLLM is running"
echo "   Model:   ${MODEL_ID}"
echo "   API:     http://${PUBLIC_IP}:${VLLM_PORT}"
echo "   Health:  http://${PUBLIC_IP}:${VLLM_PORT}/health"
echo "   Logs:    docker exec ${CONTAINER_NAME} tail -f /var/log/vllm.log"
echo "   Stop:    ${SCRIPT_DIR}/vllm.sh --stop"
 
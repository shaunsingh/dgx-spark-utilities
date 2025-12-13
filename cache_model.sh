#!/usr/bin/env bash
set -euo pipefail

# ./cache_model.sh --list
# ./cache_model.sh --model 3
# ./cache_model.sh --model openai/gpt-oss-120b --hf-cache /data/hf --max-workers 16 --no-exclude

# additional, openai-community/gpt2, deepseek-ai/DeepSeek-OCR, openai/whisper-large-v3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_FILE="${SCRIPT_DIR}/model_catalog.sh"

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

ensure_writable_dir() {
  local dir="$1"
  mkdir -p "${dir}" || die "Cannot create cache dir: ${dir}"
  if [ ! -w "${dir}" ]; then
    local user="${USER:-$(id -un)}"
    die "Cache dir not writable: ${dir} (user=${user}). Fix perms (e.g., sudo chown -R ${user}:${user} \"${dir}\") or set HF_CACHE to a writable path."
  fi
}

[ -f "${CATALOG_FILE}" ] || die "Missing model catalog at ${CATALOG_FILE}"
# shellcheck source=/dev/null
source "${CATALOG_FILE}"

# Defaults (env overridable)
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MODEL_INPUT=""
LIST_ONLY=false
NO_EXCLUDES=false
TOKEN_OVERRIDE=""

# By default skip obviously unnecessary artifacts (can be disabled).
EXCLUDES=("original/*" "metal/*")

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -m, --model <id|number>   Model id or menu number (default: ${DEFAULT_MODEL})
  -l, --list                List catalog models and exit
  --hf-cache <path>         HF cache path (default: ${HF_CACHE})
  --max-workers <n>         Downloader concurrency (default: ${MAX_WORKERS})
  --no-exclude              Do not skip optional artifacts (download everything)
  --token <value>           HF token (overrides HF_TOKEN env)
  -h, --help                Show this help

Examples:
  $0 --list
  $0 --model 3
  HF_TOKEN=... $0 --model openai/gpt-oss-120b --hf-cache /data/hf
EOF
}

MODELS=()
MODEL_NAMES=()
MODEL_NEEDS_TOKEN=()

load_catalog_all() {
  local entry
  for entry in "${MODEL_CATALOG[@]}"; do
    MODELS+=("$(_catalog_field "$entry" "id")")
    MODEL_NAMES+=("$(_catalog_field "$entry" "name")")
    MODEL_NEEDS_TOKEN+=("$(_catalog_field "$entry" "needs_token")")
  done
}

list_models() {
  echo "Available models:"
  local i marker
  for i in "${!MODELS[@]}"; do
    marker=""
    [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ] && marker=" [default]"
    [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ] && marker="${marker} [HF token]"
    printf "  %2d. %s%s\n" "$((i + 1))" "${MODEL_NAMES[$i]}" "${marker}"
  done
}

resolve_model_index() {
  local input="$1"
  if [ -z "${input}" ]; then
    local i
    for i in "${!MODELS[@]}"; do
      if [ "${MODELS[$i]}" = "${DEFAULT_MODEL}" ]; then
        echo "$i"
        return
      fi
    done
    echo "-1"
    return
  fi

  if [[ "${input}" =~ ^[0-9]+$ ]]; then
    local idx=$((input - 1))
    if [ "${idx}" -ge 0 ] && [ "${idx}" -lt "${#MODELS[@]}" ]; then
      echo "${idx}"
      return
    fi
    die "Model number out of range (1-${#MODELS[@]})"
  fi

  local i
  for i in "${!MODELS[@]}"; do
    if [ "${MODELS[$i]}" = "${input}" ]; then
      echo "${i}"
      return
    fi
  done

  echo "-1"
}

select_downloader() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli"
    return
  fi
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return
  fi
  die "Install the Hugging Face CLI: pip install 'huggingface_hub[cli]'"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_INPUT="$2"; shift 2 ;;
    -l|--list) LIST_ONLY=true; shift ;;
    --hf-cache) HF_CACHE="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --no-exclude) NO_EXCLUDES=true; shift ;;
    --token) TOKEN_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

load_catalog_all

if [ "${LIST_ONLY}" = "true" ]; then
  list_models
  exit 0
fi

[ "${NO_EXCLUDES}" = "true" ] && EXCLUDES=()

MODEL_IDX="$(resolve_model_index "${MODEL_INPUT}")"
CUSTOM_MODEL=false
if [ "${MODEL_IDX}" -ge 0 ]; then
  MODEL_ID="${MODELS[$MODEL_IDX]}"
  MODEL_NAME="${MODEL_NAMES[$MODEL_IDX]}"
  MODEL_NEEDS_TOKEN_FLAG="${MODEL_NEEDS_TOKEN[$MODEL_IDX]}"
else
  if [ -z "${MODEL_INPUT}" ]; then
    die "Default model '${DEFAULT_MODEL}' not found; specify --model explicitly."
  fi
  CUSTOM_MODEL=true
  MODEL_ID="${MODEL_INPUT}"
  MODEL_NAME="${MODEL_INPUT}"
  MODEL_NEEDS_TOKEN_FLAG="false"
fi

HF_TOKEN_VALUE="${TOKEN_OVERRIDE:-${HF_TOKEN:-}}"

if [ "${MODEL_NEEDS_TOKEN_FLAG}" = "true" ] && [ -z "${HF_TOKEN_VALUE}" ]; then
  die "Model ${MODEL_ID} requires HF_TOKEN. Export HF_TOKEN or pass --token."
fi

DOWNLOADER="$(select_downloader)"
require_cmd "${DOWNLOADER}"

CACHE_ROOT="${HF_CACHE%/}"
CACHE_DIR="${CACHE_ROOT}/hub"

log "Caching ${MODEL_NAME} (${MODEL_ID})"
log "Cache: ${CACHE_DIR}"
log "Downloader: ${DOWNLOADER}, workers: ${MAX_WORKERS}"
[ "${#EXCLUDES[@]}" -gt 0 ] && log "Excluding: ${EXCLUDES[*]}"
[ "${CUSTOM_MODEL}" = "true" ] && log "Note: ${MODEL_ID} is not in catalog; caching as custom entry."
if [ -n "${HF_TOKEN_VALUE}" ]; then
  if [ -n "${TOKEN_OVERRIDE}" ]; then
    log "HF token: provided via --token"
  else
    log "HF token: found in HF_TOKEN env"
  fi
fi

ensure_writable_dir "${CACHE_DIR}"
export HF_HOME="${CACHE_ROOT}"
[ -n "${HF_TOKEN_VALUE}" ] && export HF_TOKEN="${HF_TOKEN_VALUE}"

CMD=(
  "${DOWNLOADER}" download "${MODEL_ID}"
  --repo-type model
  --cache-dir "${CACHE_DIR}"
  --max-workers "${MAX_WORKERS}"
)
if [ "${#EXCLUDES[@]}" -gt 0 ]; then
  for pattern in "${EXCLUDES[@]}"; do
    CMD+=(--exclude "${pattern}")
  done
fi
if [ -n "${HF_TOKEN_VALUE}" ]; then
  CMD+=(--token "${HF_TOKEN_VALUE}")
fi

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" "${CMD[@]}"

log "Completed. Models are cached under ${CACHE_DIR} (HF_HOME=${CACHE_ROOT})."


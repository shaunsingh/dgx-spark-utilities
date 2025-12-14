#!/usr/bin/env bash
set -euo pipefail

# Unified benchmark runner for TensorRT-LLM, SGLang, and vLLM against the
# default model from model_catalog.sh.

# ./benchmark.sh --all --text-output ./bench.txt
# ./benchmark.sh --model-scripts --text-output bench.txt
# ./benchmark.sh --model-script ./models/llama33_70b_fp4_trt.sh --text-output bench.txt
# ./benchmark.sh --model ibm-granite/granite-4.0-h-small --all --quick --text-output ./bench.txt
# ./benchmark.sh --backends vllm --model openai/gpt-oss-20b --text-output bench.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_FILE="${SCRIPT_DIR}/model_catalog.sh"
[ -f "${CATALOG_FILE}" ] || { echo "Missing model catalog at ${CATALOG_FILE}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${CATALOG_FILE}"
MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL}}"

# Defaults (env overridable)
NUM_PROMPTS="${NUM_PROMPTS:-100}"
CONCURRENCY="${CONCURRENCY:-32}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_FILE=""
TEXT_OUTPUT_FILE="${TEXT_OUTPUT_FILE:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BACKEND_KEYS="${BACKEND_KEYS:-trt,sgl,vllm}"
RUN_ALL=false
MODEL_SCRIPTS_DIR="${MODEL_SCRIPTS_DIR:-}"
MODEL_SCRIPT_PATH="${MODEL_SCRIPT_PATH:-}"

TRT_PORT="${TRT_PORT:-8355}"
SGLANG_PORT="${SGLANG_PORT:-8002}"
VLLM_PORT="${VLLM_PORT:-8000}"

TRT_CONTAINER_NAME="${TRT_CONTAINER_NAME:-trtllm-single}"
SGLANG_CONTAINER_NAME="${SGLANG_CONTAINER_NAME:-sglang-single}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-single}"

DEFAULT_DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

TRT_URL="${TRT_URL:-http://127.0.0.1:${TRT_PORT}}"
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:${SGLANG_PORT}}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:${VLLM_PORT}}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -m, --model ID        Model ID to benchmark (default: from catalog)
  -n, --num-prompts N   Number of prompts (default: ${NUM_PROMPTS})
  -c, --concurrency N   Max concurrent requests (default: ${CONCURRENCY})
  -d, --dataset PATH    ShareGPT dataset path (default: auto-download)
  -o, --output FILE     Write JSON results to FILE
  -t, --text-output FILE
                        Write text report to FILE (nvidia-smi style)
  -b, --backends LIST   Comma-separated backends (trt,sgl,vllm; default: ${BACKEND_KEYS})
  --all                 Start/bench/stop each selected backend sequentially
  --model-scripts       Benchmark each executable .sh script in models/ (TensorRT)
  --model-script FILE   Benchmark a single TensorRT model script
  --models-dir DIR      Directory for --model-scripts (default: ${SCRIPT_DIR}/models)
  -q, --quick           Quick mode (20 prompts, concurrency 8)
  -s, --single          Single request latency test
  -h, --help            Show this help

Environment overrides:
  MODEL_ID, MAX_OUTPUT_TOKENS, TEMPERATURE, BACKEND_KEYS
  TRT_URL, SGLANG_URL, VLLM_URL (default to localhost ports)
  TRT_PORT, SGLANG_PORT, VLLM_PORT (used when launching with --all)
  TRT_CONTAINER_NAME, SGLANG_CONTAINER_NAME, VLLM_CONTAINER_NAME (for --all)
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

QUICK=false
SINGLE=false
RUN_MODEL_SCRIPTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_ID="$2"; shift 2 ;;
    -n|--num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
    -d|--dataset) DATASET_PATH="$2"; shift 2 ;;
    -o|--output) OUTPUT_FILE="$2"; shift 2 ;;
    -t|--text-output) TEXT_OUTPUT_FILE="$2"; shift 2 ;;
    -b|--backends) BACKEND_KEYS="$2"; shift 2 ;;
    --all) RUN_ALL=true; shift ;;
    --model-scripts) RUN_MODEL_SCRIPTS=true; MODEL_SCRIPTS_DIR="${MODEL_SCRIPTS_DIR:-${SCRIPT_DIR}/models}"; shift ;;
    --model-script) RUN_MODEL_SCRIPTS=true; MODEL_SCRIPT_PATH="$2"; shift 2 ;;
    --models-dir) RUN_MODEL_SCRIPTS=true; MODEL_SCRIPTS_DIR="$2"; shift 2 ;;
    -q|--quick) QUICK=true; shift ;;
    -s|--single) SINGLE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ${QUICK}; then
  NUM_PROMPTS=20
  CONCURRENCY=8
fi

if ${SINGLE}; then
  NUM_PROMPTS=1
  CONCURRENCY=1
fi

require_cmd python3
require_cmd curl

if ${RUN_MODEL_SCRIPTS} && [ -z "${MODEL_SCRIPT_PATH}" ] && [ -z "${MODEL_SCRIPTS_DIR}" ]; then
  MODEL_SCRIPTS_DIR="${SCRIPT_DIR}/models"
fi

ensure_dataset() {
  if [ -n "${DATASET_PATH}" ]; then
    [ -f "${DATASET_PATH}" ] || die "Dataset not found at ${DATASET_PATH}"
    echo "${DATASET_PATH}"
    return
  fi

  local target="/tmp/${DEFAULT_DATASET}"
  if [ ! -f "${target}" ]; then
    log "Downloading ShareGPT dataset to ${target}..." >&2
    curl -sSfL --retry 3 --retry-delay 2 -o "${target}" "${DATASET_URL}" || die "Failed to download dataset"
  fi
  echo "${target}"
}

BACKEND_LIST=()
AGG_OUTPUTS=()
MODEL_SCRIPT_FILES=()

select_backends() {
  local keys_csv="${1// /}"
  local IFS=',' keys entries descs
  IFS=',' read -r -a keys <<< "${keys_csv}"
  [ "${#keys[@]}" -gt 0 ] || die "No backends selected"

  entries=()
  descs=()
  for key in "${keys[@]}"; do
    case "${key}" in
      trt)
        entries+=("{\"name\":\"TensorRT-LLM\",\"key\":\"trt\",\"url\":\"${TRT_URL}\"}")
        descs+=("TensorRT-LLM=${TRT_URL}")
        ;;
      sgl)
        entries+=("{\"name\":\"SGLang\",\"key\":\"sgl\",\"url\":\"${SGLANG_URL}\"}")
        descs+=("SGLang=${SGLANG_URL}")
        ;;
      vllm)
        entries+=("{\"name\":\"vLLM\",\"key\":\"vllm\",\"url\":\"${VLLM_URL}\"}")
        descs+=("vLLM=${VLLM_URL}")
        ;;
      *)
        die "Unknown backend key '${key}'. Supported: trt,sgl,vllm"
        ;;
    esac
  done

  BACKENDS_JSON="$(IFS=','; echo "[${entries[*]}]")"
  BACKENDS_DESC="$(IFS=', '; echo "${descs[*]}")"
  BACKEND_LIST=("${keys[@]}")
}

model_supported_for_backend() {
  local backend="$1" model="$2" entry id backends
  for entry in "${MODEL_CATALOG[@]}"; do
    id="$(_catalog_field "${entry}" "id")"
    backends="$(_catalog_field "${entry}" "backends")"
    if [ "${id}" = "${model}" ] && [[ ",${backends}," == *",${backend},"* ]]; then
      return 0
    fi
  done
  return 1
}

filter_backends_for_model() {
  local model="$1" key filtered=() new_descs=() new_entries=()
  for key in "${BACKEND_LIST[@]}"; do
    if model_supported_for_backend "${key}" "${model}"; then
      filtered+=("${key}")
      case "${key}" in
        trt)
          new_entries+=("{\"name\":\"TensorRT-LLM\",\"key\":\"trt\",\"url\":\"${TRT_URL}\"}")
          new_descs+=("TensorRT-LLM=${TRT_URL}")
          ;;
        sgl)
          new_entries+=("{\"name\":\"SGLang\",\"key\":\"sgl\",\"url\":\"${SGLANG_URL}\"}")
          new_descs+=("SGLang=${SGLANG_URL}")
          ;;
        vllm)
          new_entries+=("{\"name\":\"vLLM\",\"key\":\"vllm\",\"url\":\"${VLLM_URL}\"}")
          new_descs+=("vLLM=${VLLM_URL}")
          ;;
      esac
    else
      log "Skipping backend '${key}' for model ${model}: not in catalog for this backend"
    fi
  done

  if [ "${#filtered[@]}" -eq 0 ]; then
    die "Model ${model} is not available for selected backends (${BACKEND_KEYS})"
  fi

  BACKEND_LIST=("${filtered[@]}")
  BACKENDS_DESC="$(IFS=', '; echo "${new_descs[*]}")"
  BACKENDS_JSON="$(IFS=','; echo "[${new_entries[*]}]")"
}

DATASET_PATH="$(ensure_dataset)"

if ${RUN_MODEL_SCRIPTS}; then
  BACKENDS_DESC="per-script (auto-detect)"
  BACKENDS_JSON="[]"
  BACKEND_LIST=()
else
  select_backends "${BACKEND_KEYS}"
  filter_backends_for_model "${MODEL_ID}"
fi

cleanup_containers() {
  local name
  for name in "${STARTED_CONTAINERS[@]:-}"; do
    docker rm -f "${name}" >/dev/null 2>&1 || true
  done
}

STARTED_CONTAINERS=()
trap cleanup_containers EXIT

collect_model_scripts() {
  if [ -n "${MODEL_SCRIPT_PATH}" ]; then
    [ -f "${MODEL_SCRIPT_PATH}" ] || die "Model script not found: ${MODEL_SCRIPT_PATH}"
    [ -x "${MODEL_SCRIPT_PATH}" ] || die "Model script not executable: ${MODEL_SCRIPT_PATH}"
    MODEL_SCRIPT_FILES=("${MODEL_SCRIPT_PATH}")
    return
  fi

  local dir="$1"
  [ -d "${dir}" ] || die "Model scripts directory not found: ${dir}"

  MODEL_SCRIPT_FILES=()
  while IFS= read -r -d '' file; do
    MODEL_SCRIPT_FILES+=("${file}")
  done < <(find "${dir}" -maxdepth 1 -type f -name "*.sh" -perm -111 -print0)

  if [ "${#MODEL_SCRIPT_FILES[@]}" -eq 0 ]; then
    die "No executable .sh model scripts found in ${dir}"
  fi

  IFS=$'\n' MODEL_SCRIPT_FILES=($(printf '%s\n' "${MODEL_SCRIPT_FILES[@]}" | sort))
  unset IFS
}

detect_model_script_backend() {
  local script="$1"
  if grep -Eq 'vllm\.sh' "${script}"; then
    echo "vllm"
    return 0
  fi
  if grep -Eq 'sglang\.sh' "${script}"; then
    echo "sgl"
    return 0
  fi
  if grep -Eq 'tensorrt\.sh' "${script}"; then
    echo "trt"
    return 0
  fi
  echo ""
}

extract_model_id() {
  local file="$1" model=""
  model="$(sed -nE 's/.*--model[[:space:]]+"([^"]+)".*/\1/p' "${file}" | head -n1)"
  if [ -z "${model}" ]; then
    model="$(sed -nE "s/.*--model[[:space:]]+'([^']+)'.*/\1/p" "${file}" | head -n1)"
  fi
  if [ -z "${model}" ]; then
    model="$(sed -nE 's/.*--model[[:space:]]+([^[:space:]]+).*/\1/p' "${file}" | head -n1)"
  fi
  echo "${model}"
}

start_backend() {
  local key="$1"
  case "${key}" in
    trt)
      STARTED_CONTAINERS+=("${TRT_CONTAINER_NAME}")
      log "Starting TensorRT-LLM (${MODEL_ID})..."
      CONTAINER_NAME="${TRT_CONTAINER_NAME}" "${SCRIPT_DIR}/tensorrt.sh" --model "${MODEL_ID}" --port "${TRT_PORT}"
      ;;
    sgl)
      STARTED_CONTAINERS+=("${SGLANG_CONTAINER_NAME}")
      log "Starting SGLang (${MODEL_ID})..."
      CONTAINER_NAME="${SGLANG_CONTAINER_NAME}" "${SCRIPT_DIR}/sglang.sh" --model "${MODEL_ID}" --port "${SGLANG_PORT}"
      ;;
    vllm)
      STARTED_CONTAINERS+=("${VLLM_CONTAINER_NAME}")
      log "Starting vLLM (${MODEL_ID})..."
      CONTAINER_NAME="${VLLM_CONTAINER_NAME}" "${SCRIPT_DIR}/vllm.sh" --model "${MODEL_ID}" --port "${VLLM_PORT}"
      ;;
    *)
      die "Unsupported backend '${key}' for start"
      ;;
  esac
}

stop_backend() {
  local key="$1"
  case "${key}" in
    trt) docker rm -f "${TRT_CONTAINER_NAME}" >/dev/null 2>&1 || true ;;
    sgl) docker rm -f "${SGLANG_CONTAINER_NAME}" >/dev/null 2>&1 || true ;;
    vllm) "${SCRIPT_DIR}/vllm.sh" --container-name "${VLLM_CONTAINER_NAME}" --stop >/dev/null 2>&1 || true ;;
    *) return ;;
  esac
}

start_model_script() {
  local script="$1" container="$2" backend="$3"
  local envs=() name=""

  case "${backend}" in
    trt)
      name="TensorRT-LLM"
      envs=(CONTAINER_NAME="${container}" TRT_CONTAINER_NAME="${container}" TRT_PORT="${TRT_PORT}")
      ;;
    sgl)
      name="SGLang"
      envs=(CONTAINER_NAME="${container}" SGLANG_CONTAINER_NAME="${container}" SGLANG_PORT="${SGLANG_PORT}")
      ;;
    vllm)
      name="vLLM"
      envs=(CONTAINER_NAME="${container}" VLLM_CONTAINER_NAME="${container}" VLLM_PORT="${VLLM_PORT}")
      ;;
    *)
      die "Unsupported backend '${backend}' for model script ${script}"
      ;;
  esac

  STARTED_CONTAINERS+=("${container}")
  log "Starting ${name} via ${script}..."
  env "${envs[@]}" "${script}"
}

benchmark_backend() {
  local key="$1" name url output_tmp="" status=0 orig_output="${OUTPUT_FILE}" orig_text="${TEXT_OUTPUT_FILE}"
  case "${key}" in
    trt) name="TensorRT-LLM"; url="${TRT_URL}" ;;
    sgl) name="SGLang"; url="${SGLANG_URL}" ;;
    vllm) name="vLLM"; url="${VLLM_URL}" ;;
    *) die "Unsupported backend '${key}' for benchmark" ;;
  esac

  BACKENDS_JSON="[{\"name\":\"${name}\",\"key\":\"${key}\",\"url\":\"${url}\"}]"
  export BACKENDS_JSON MODEL_ID NUM_PROMPTS CONCURRENCY DATASET_PATH MAX_OUTPUT_TOKENS TEMPERATURE

  if [ -n "${orig_output}" ] || { ${RUN_ALL} && [ -n "${orig_text}" ]; }; then
    output_tmp="$(mktemp "/tmp/benchmark-${key}.XXXX.json")"
    export OUTPUT_FILE="${output_tmp}"
  fi

  # Avoid writing partial text reports in --all mode; aggregate later instead.
  if ${RUN_ALL} && [ -n "${orig_text}" ]; then
    export TEXT_OUTPUT_FILE=""
  fi

  log "Benchmarking ${name} at ${url}..."
  if ! python3 "${SCRIPT_DIR}/benchmark_runner.py"; then
    status=$?
  fi

  if [ -n "${output_tmp}" ]; then
    AGG_OUTPUTS+=("${output_tmp}")
    export OUTPUT_FILE="${orig_output}"
  fi

  export TEXT_OUTPUT_FILE="${orig_text}"

  return "${status}"
}

aggregate_outputs() {
  local json_dest="$1" text_dest="$2"
  if [ -z "${json_dest}" ] && [ -z "${text_dest}" ]; then
    return 0
  fi
  [ "${#AGG_OUTPUTS[@]}" -gt 0 ] || return 0

  local dest="${json_dest}"
  local cleanup=false
  if [ -z "${dest}" ]; then
    dest="$(mktemp "/tmp/benchmark-aggregate.XXXX.json")"
    cleanup=true
  fi

  local args=(--aggregate "${dest}" "${AGG_OUTPUTS[@]}")
  if [ -n "${text_dest}" ]; then
    args+=(--text-report "${text_dest}")
  fi

  python3 "${SCRIPT_DIR}/benchmark_runner.py" "${args[@]}"

  if ${cleanup}; then
    rm -f "${dest}"
  fi
}

run_all_backends() {
  local key
  log "Mode: sequential (--all). Backends: ${BACKENDS_DESC}"
  for key in "${BACKEND_LIST[@]}"; do
    start_backend "${key}"
    if ! benchmark_backend "${key}"; then
      stop_backend "${key}"
      return 1
    fi
    stop_backend "${key}"
  done
  aggregate_outputs "${OUTPUT_FILE}" "${TEXT_OUTPUT_FILE}"
}

run_model_scripts() {
  local orig_run_all="${RUN_ALL}"
  local orig_output="${OUTPUT_FILE}"
  local orig_text="${TEXT_OUTPUT_FILE}"
  local orig_model_id="${MODEL_ID}"
  local orig_trt_container="${TRT_CONTAINER_NAME}"
  local orig_sgl_container="${SGLANG_CONTAINER_NAME}"
  local orig_vllm_container="${VLLM_CONTAINER_NAME}"
  local script model container backend

  RUN_ALL=true
  if [ -n "${MODEL_SCRIPT_PATH}" ]; then
    log "Mode: model script (${MODEL_SCRIPT_PATH})"
  else
    log "Mode: model scripts (${MODEL_SCRIPTS_DIR})"
  fi

  for script in "${MODEL_SCRIPT_FILES[@]}"; do
    container="trtllm-$(basename "${script%.*}")"
    backend="$(detect_model_script_backend "${script}")"
    if [ -z "${backend}" ]; then
      die "Could not detect backend from ${script}; expected vllm.sh, sglang.sh, or tensorrt.sh"
    fi
    model="$(extract_model_id "${script}")"
    if [ -z "${model}" ]; then
      model="${DEFAULT_MODEL}"
      log "Model id not found in ${script}; defaulting to ${model}"
    fi
    if ! model_supported_for_backend "${backend}" "${model}"; then
      log "Model ${model} not in catalog for backend ${backend}; continuing anyway"
    fi
    MODEL_ID="${model}"
    case "${backend}" in
      trt) TRT_CONTAINER_NAME="${container}" ;;
      sgl) SGLANG_CONTAINER_NAME="${container}" ;;
      vllm) VLLM_CONTAINER_NAME="${container}" ;;
    esac

    if ! start_model_script "${script}" "${container}" "${backend}"; then
      RUN_ALL="${orig_run_all}"
      MODEL_ID="${orig_model_id}"
      TRT_CONTAINER_NAME="${orig_trt_container}"
      SGLANG_CONTAINER_NAME="${orig_sgl_container}"
      VLLM_CONTAINER_NAME="${orig_vllm_container}"
      return 1
    fi

    if ! benchmark_backend "${backend}"; then
      stop_backend "${backend}"
      RUN_ALL="${orig_run_all}"
      MODEL_ID="${orig_model_id}"
      TRT_CONTAINER_NAME="${orig_trt_container}"
      SGLANG_CONTAINER_NAME="${orig_sgl_container}"
      VLLM_CONTAINER_NAME="${orig_vllm_container}"
      return 1
    fi

    stop_backend "${backend}"
  done

  aggregate_outputs "${orig_output}" "${orig_text}"

  RUN_ALL="${orig_run_all}"
  MODEL_ID="${orig_model_id}"
  TRT_CONTAINER_NAME="${orig_trt_container}"
  SGLANG_CONTAINER_NAME="${orig_sgl_container}"
  VLLM_CONTAINER_NAME="${orig_vllm_container}"
}

export BACKENDS_JSON MODEL_ID NUM_PROMPTS CONCURRENCY DATASET_PATH OUTPUT_FILE TEXT_OUTPUT_FILE
export MAX_OUTPUT_TOKENS TEMPERATURE DATASET_URL DEFAULT_DATASET

if ${RUN_MODEL_SCRIPTS}; then
  collect_model_scripts "${MODEL_SCRIPTS_DIR}"
  if [ -n "${MODEL_SCRIPT_PATH}" ]; then
    log "Benchmarking model script: ${MODEL_SCRIPT_PATH}"
  else
    log "Benchmarking model scripts in ${MODEL_SCRIPTS_DIR}"
  fi
else
  log "Benchmarking model: ${MODEL_ID}"
fi
log "Prompts: ${NUM_PROMPTS}, Concurrency: ${CONCURRENCY}, Max output tokens: ${MAX_OUTPUT_TOKENS}"
log "Backends: ${BACKENDS_DESC}"

if ${RUN_MODEL_SCRIPTS}; then
  run_model_scripts
elif ${RUN_ALL}; then
  run_all_backends
else
  python3 "${SCRIPT_DIR}/benchmark_runner.py"
fi



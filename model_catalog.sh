# Shared model catalog for TensorRT-LLM, SGLang, and vLLM launchers.
# Source this file and call `catalog_load <backend>` where backend is one of:
#   trt  - TensorRT-LLM
#   sgl  - SGLang
#   vllm - vLLM (Ray)
# After loading, these arrays are available (aligned by index):
#   MODELS, MODEL_NAMES, MODEL_MEM, MODEL_MAX_TOKENS, MODEL_BATCH_SIZE,
#   MODEL_TRUST_REMOTE, MODEL_NEEDS_TOKEN, MODEL_ATT_DP, MODEL_KV_DTYPE,
#   MODEL_REASONING_PARSER, MODEL_TOOL_PARSER, MODEL_EXPERT_PARALLEL.

# Default model
DEFAULT_MODEL="nvidia/Llama-3.1-8B-Instruct-FP8"

# Catalog entries use pipe-delimited key=value pairs.
# Supported keys: id, name, backends, mem, max_tokens, batch, needs_token,
# trust_remote, expert_parallel, att_dp, kv_dtype, reasoning_parser,
# tool_parser.
MODEL_CATALOG=(
  # Nvidia quantized
  "id=nvidia/Llama-3.1-8B-Instruct-FP8|name=Llama-3.1-8B-FP8 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=16|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Llama-3.1-8B-Instruct-FP4|name=Llama-3.1-8B-FP4 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=16|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Llama-3.3-70B-Instruct-FP4|name=Llama-3.3-70B-FP4 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=4|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5|name=Llama-3.3 Nemotron Super 49B v1.5 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=2048|batch=1024|needs_token=true|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=fp8"
  "id=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4|name=Llama-4 Scout 17B 16E FP4 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=8|needs_token=true|trust_remote=false|expert_parallel=true|att_dp=false|kv_dtype=auto"
  "id=nvidia/Llama-4-Maverick-17B-128E-Instruct|name=Llama-4 Maverick 17B 128E (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=8|needs_token=true|trust_remote=false|expert_parallel=true|att_dp=false|kv_dtype=auto"

  "id=nvidia/Qwen3-8B-FP8|name=Qwen3-8B-FP8 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=32768|batch=16|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Qwen3-14B-FP8|name=Qwen3-14B-FP8 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=32768|batch=16|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Qwen3-32B-FP4|name=Qwen3-32B-FP4 (NVIDIA)|backends=trt,sgl,vllm|mem=0.85|max_tokens=32768|batch=8|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=nvidia/Qwen3-30B-A3B-FP4|name=Qwen3-30B-A3B-FP4 (NVIDIA MoE)|backends=trt,sgl,vllm|mem=0.90|max_tokens=32768|batch=4|needs_token=false|trust_remote=false|expert_parallel=true|att_dp=false|kv_dtype=auto"
  "id=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8|name=Qwen3 Next 80B A3B|backends=trt,sgl,vllm|mem=0.60|max_tokens=4096|batch=16|needs_token=false|trust_remote=true|expert_parallel=true|att_dp=false|kv_dtype=auto"

  "id=nvidia/Phi-4-reasoning-plus-FP4|name=Phi-4 reasoning+ FP4 (NVIDIA)|backends=trt,sgl,vllm|mem=0.90|max_tokens=16384|batch=16|needs_token=false|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"

  # Zhipu GLM
  "id=zai-org/GLM-4.5-Air-FP8|name=GLM-4.5-Air-FP8|backends=trt,sgl,vllm|mem=0.70|max_tokens=32768|batch=4|needs_token=true|trust_remote=true|expert_parallel=true|att_dp=false|kv_dtype=auto|reasoning_parser=glm45|tool_parser=glm45"
  "id=PrimeIntellect/INTELLECT-3-FP8|name=INTELLECT-3-FP8|backends=vllm|mem=0.70|max_tokens=32768|batch=4|needs_token=false|trust_remote=true|expert_parallel=true|att_dp=false|kv_dtype=auto|reasoning_parser=deepseek-r1|tool_parser=qwen3_coder"
  "id=PrimeIntellect/INTELLECT-3|name=INTELLECT-3|backends=trt,sgl,vllm|mem=0.70|max_tokens=32768|batch=4|needs_token=false|trust_remote=true|expert_parallel=true|att_dp=false|kv_dtype=auto|reasoning_parser=deepseek-r1|tool_parser=qwen3_coder"
  "id=Firworks/INTELLECT-3-nvfp4|name=INTELLECT-3-NVFP4|backends=sgl,vllm|mem=0.65|max_tokens=32768|batch=4|needs_token=false|trust_remote=true|expert_parallel=true|att_dp=false|kv_dtype=auto|reasoning_parser=deepseek-r1|tool_parser=qwen3_coder"

  # GPT OSS
  "id=openai/gpt-oss-120b|name=GPT-OSS-120B|backends=trt,sgl,vllm|mem=0.90|max_tokens=8192|batch=4|needs_token=false|trust_remote=false|expert_parallel=true|att_dp=true|kv_dtype=auto|reasoning_parser=gpt-oss|tool_parser=gpt-oss"
  "id=openai/gpt-oss-20b|name=GPT-OSS-20B|backends=trt,sgl,vllm|mem=0.90|max_tokens=8192|batch=8|needs_token=false|trust_remote=false|expert_parallel=true|att_dp=false|kv_dtype=auto|reasoning_parser=gpt-oss|tool_parser=gpt-oss"

  # IBM Granite4
  "id=ibm-granite/granite-4.0-micro|name=Granite-4.0 Micro 3B|backends=vllm|mem=0.90|max_tokens=32768|batch=16|needs_token=true|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=ibm-granite/granite-4.0-h-micro|name=Granite-4.0 H Micro 3B|backends=vllm|mem=0.90|max_tokens=32768|batch=16|needs_token=true|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=ibm-granite/granite-4.0-h-tiny|name=Granite-4.0 H Tiny 7B|backends=vllm|mem=0.90|max_tokens=32768|batch=16|needs_token=true|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=ibm-granite/granite-4.0-h-small|name=Granite-4.0 H Small 32B|backends=vllm|mem=0.85|max_tokens=32768|batch=8|needs_token=true|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"

  # OEM
  "id=Qwen/Qwen2.5-7B-Instruct|name=Qwen2.5-7B|backends=sgl|mem=0.90|max_tokens=32768|batch=16|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=Qwen/Qwen2.5-14B-Instruct|name=Qwen2.5-14B|backends=sgl|mem=0.90|max_tokens=32768|batch=16|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=Qwen/Qwen2.5-32B-Instruct|name=Qwen2.5-32B|backends=sgl|mem=0.85|max_tokens=32768|batch=8|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=Qwen/Qwen2.5-72B-Instruct|name=Qwen2.5-72B|backends=sgl|mem=0.90|max_tokens=32768|batch=4|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"

  "id=mistralai/Mistral-7B-Instruct-v0.3|name=Mistral-7B v0.3|backends=trt,sgl|mem=0.90|max_tokens=32768|batch=16|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=mistralai/Mistral-Nemo-Instruct-2407|name=Mistral-Nemo-12B (128k)|backends=trt,sgl,vllm|mem=0.85|max_tokens=131072|batch=8|needs_token=false|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=mistralai/Mixtral-8x7B-Instruct-v0.1|name=Mixtral-8x7B (MoE)|backends=trt,sgl,vllm|mem=0.85|max_tokens=32768|batch=8|needs_token=false|trust_remote=false|expert_parallel=true|att_dp=false|kv_dtype=auto"

  "id=meta-llama/Llama-3.1-8B-Instruct|name=Llama-3.1-8B|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=16|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto|tool_parser=llama3"
  "id=meta-llama/Llama-3.3-8B-Instruct|name=Llama-3.3-8B|backends=trt,sgl,vllm|mem=0.90|max_tokens=131072|batch=16|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto|tool_parser=llama3"
  "id=meta-llama/Llama-3.3-70B-Instruct|name=Llama-3.3-70B|backends=sgl,vllm|mem=0.90|max_tokens=131072|batch=4|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto|tool_parser=llama3"

  "id=microsoft/phi-4|name=Phi-4|backends=sgl,vllm|mem=0.90|max_tokens=16384|batch=16|needs_token=false|trust_remote=true|expert_parallel=false|att_dp=false|kv_dtype=auto"
  "id=google/gemma-2-27b-it|name=Gemma2-27B|backends=trt,sgl,vllm|mem=0.90|max_tokens=8192|batch=8|needs_token=true|trust_remote=false|expert_parallel=false|att_dp=false|kv_dtype=auto"
)

_catalog_field() {
  local entry="$1" key="$2"
  local IFS='|'
  for part in $entry; do
    if [[ "$part" == "${key}="* ]]; then
      echo "${part#*=}"
      return
    fi
  done
  echo ""
}

_catalog_default() {
  echo "${DEFAULT_MODEL}"
}

catalog_load() {
  local backend="$1"
  if [ -z "$backend" ]; then
    echo "catalog_load: missing backend (trt|sgl|vllm)" >&2
    return 1
  fi

  MODELS=()
  MODEL_NAMES=()
  MODEL_MEM=()
  MODEL_MAX_TOKENS=()
  MODEL_BATCH_SIZE=()
  MODEL_TRUST_REMOTE=()
  MODEL_NEEDS_TOKEN=()
  MODEL_ATT_DP=()
  MODEL_KV_DTYPE=()
  MODEL_REASONING_PARSER=()
  MODEL_TOOL_PARSER=()
  MODEL_EXPERT_PARALLEL=()

  local entry
  for entry in "${MODEL_CATALOG[@]}"; do
    local backends
    backends="$(_catalog_field "$entry" "backends")"
    if [[ ",${backends}," != *",${backend},"* ]]; then
      continue
    fi
    MODELS+=("$(_catalog_field "$entry" "id")")
    MODEL_NAMES+=("$(_catalog_field "$entry" "name")")
    MODEL_MEM+=("$(_catalog_field "$entry" "mem")")
    MODEL_MAX_TOKENS+=("$(_catalog_field "$entry" "max_tokens")")
    MODEL_BATCH_SIZE+=("$(_catalog_field "$entry" "batch")")
    MODEL_TRUST_REMOTE+=("$(_catalog_field "$entry" "trust_remote")")
    MODEL_NEEDS_TOKEN+=("$(_catalog_field "$entry" "needs_token")")
    MODEL_ATT_DP+=("$(_catalog_field "$entry" "att_dp")")
    MODEL_KV_DTYPE+=("$(_catalog_field "$entry" "kv_dtype")")
    MODEL_REASONING_PARSER+=("$(_catalog_field "$entry" "reasoning_parser")")
    MODEL_TOOL_PARSER+=("$(_catalog_field "$entry" "tool_parser")")
    MODEL_EXPERT_PARALLEL+=("$(_catalog_field "$entry" "expert_parallel")")
  done

  if [ ${#MODELS[@]} -eq 0 ]; then
    echo "catalog_load: no models for backend ${backend}" >&2
    return 1
  fi

  CATALOG_DEFAULT_MODEL="$(_catalog_default "$backend")"
  if ! printf '%s\0' "${MODELS[@]}" | grep -Fqx -- "${CATALOG_DEFAULT_MODEL}"; then
    CATALOG_DEFAULT_MODEL="${MODELS[0]}"
  fi
}


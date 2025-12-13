set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

mkdir -p tiktoken_encodings && \
  wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
  wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

export TIKTOKEN_ENCODINGS_BASE=$PWD/tiktoken_encodings

# clone vllm & init venv
git clone https://github.com/vllm-project/vllm.git
cd "${ROOT_DIR}/vllm"
uv venv --python 3.12 --seed
source .venv/bin/activate

# install pre-release deps
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install xgrammar triton --prerelease=allow
uv pip install flashinfer-python --prerelease=allow --index-url https://flashinfer.ai/whl/nightly/ --no-deps
uv pip install flashinfer-cubin --prerelease=allow --index-url https://flashinfer.ai/whl/nightly/
uv pip install flashinfer-jit-cache --prerelease=allow --index-url https://flashinfer.ai/whl/nightly/cu130

# use existing torch
python use_existing_torch.py
uv pip install -r requirements/build.txt
sed -i “/flashinfer/d” requirements/cuda.txt

uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e . -v --pre
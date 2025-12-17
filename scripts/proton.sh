#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
git clone --recurse-submodules https://github.com/ValveSoftware/Proton.git proton
cd proton

git checkout bleeding-edge
git submodule update --init --recursive

cd docker && make BUILD_ARCH=aarch64 proton-llvm && cd ..
mkdir -p build && cd build
../configure.sh --enable-ccache --target-arch=arm64 --build-name=fex1 --proton-sdk-image=registry.gitlab.steamos.cloud/proton/sniper/sdk/arm64/llvm:latest --container-engine=docker
make redist
cd ..

echo "Proton built successfully"
echo "Pass: ${ROOT_DIR}/proton/build/redist/proton"
echo "e.x. WINEPREFIX=~/.wine PROTONPATH=${ROOT_DIR}/proton/build/redist umu-run 'test.exe'"

exit 0
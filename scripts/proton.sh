#!/usr/bin/env bash

PROTON_REPO="https://github.com/ValveSoftware/Proton.git"
PROTON_BRANCH="bleeding-edge"
BUILD_ARCH="aarch64"
BUILD_NAME="fex1"
SDK_IMAGE="registry.gitlab.steamos.cloud/proton/sniper/sdk/${BUILD_ARCH}/llvm:latest"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
PROTON_PATH="${ROOT_DIR}/proton"
REDIST_PATH="${PROTON_PATH}/build/redist"

cleanup() {
    echo ""
    echo "Cleaning up"
    cd "${ROOT_DIR}"
    echo "Current directory: $(pwd)"
}

trap cleanup EXIT ERR

build_proton() {
    echo "Proton Build Target: ${BUILD_ARCH})"

    # 1. Clone or Update Proton
    if [[ -d "${PROTON_PATH}" ]]; then
        echo "Directory 'proton' exists. Updating existing repository"
        cd "${PROTON_PATH}"
        git fetch origin
    else
        echo "Cloning Proton repository: ${PROTON_REPO}"
        # We perform the clone outside the Proton directory to keep the root clean
        cd "${ROOT_DIR}"
        git clone "${PROTON_REPO}" proton
        cd "${PROTON_PATH}"
    fi

    # Checkout desired branch and update submodules
    echo "Checking out branch: ${PROTON_BRANCH} and updating submodules"
    git checkout "${PROTON_BRANCH}"
    git submodule update --init --recursive
    
    # 2. Build LLVM/Clang in Docker
    echo "Building LLVM/Clang using Docker for ${BUILD_ARCH}"
    cd "${PROTON_PATH}/docker"
    # The 'make' target will implicitly build the required image if it doesn't exist
    make BUILD_ARCH="${BUILD_ARCH}" proton-llvm
    cd "${PROTON_PATH}"
    
    # 3. Configure and Build Redistributable
    echo "Configuring Proton"
    mkdir -p build && cd build
    
    # Run the configuration script
    ../configure.sh \
        --enable-ccache \
        --target-arch="${BUILD_ARCH}" \
        --build-name="${BUILD_NAME}" \
        --proton-sdk-image="${SDK_IMAGE}" \
        --container-engine=docker
        
    # Build the final redistributable package
    echo "'redist'"
    make redist
    
    cd "${PROTON_PATH}"
}

patch_manifest() {
    echo "Patching toolmanifest.vdf for umu-run compatibility"
    
    local MANIFEST_FILE="${REDIST_PATH}/toolmanifest.vdf"
    
    if [[ -f "${MANIFEST_FILE}" ]]; then
        echo "Patching ${MANIFEST_FILE}"
        sed -i 's/"3810310"/"1628350"/' "${MANIFEST_FILE}"
        echo "Patched."
    else
        echo "ERROR: toolmanifest.vdf not found at ${MANIFEST_FILE}. Patching failed." >&2
        return 1
    fi
}

build_proton
patch_manifest

echo "Built successfully"
echo "Example usage with umu-run"
echo "WINEPREFIX=~/.wine PROTONPATH=${REDIST_PATH} umu-run 'path/to/your/app.exe'"
echo ""
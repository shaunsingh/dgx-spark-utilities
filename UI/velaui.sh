#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-vela-chat}"
# Default behavior: build VelaChat locally from the fork repo, then run the built image.
# You can still override IMAGE to point at any prebuilt image, e.g. IMAGE=ghcr.io/<org>/<image>:<tag>.
IMAGE="${IMAGE:-vela-chat:local}"
REPO_URL="${REPO_URL:-https://github.com/lumitry/vela-chat.git}"
REPO_REF="${REPO_REF:-main}"
HOST_ALIAS="${HOST_ALIAS:-host.docker.internal}"
HOST_PORT="${HOST_PORT:-8080}"
DATA_VOL="${DATA_VOL:-vela-chat}"
OLLAMA_VOL="${OLLAMA_VOL:-vela-chat-ollama}"
PROBE_IMAGE="${PROBE_IMAGE:-alpine:3.20}"
AUTO_BUILD="${AUTO_BUILD:-1}"
UPDATE_REPO="${UPDATE_REPO:-0}"

MY_AGENT_PORT="${MY_AGENT_PORT:-8002}"
MY_OPENAI_API_KEY="${MY_OPENAI_API_KEY:-sk-local}"
ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API:-false}"
WEBUI_AUTH="${WEBUI_AUTH:-false}"
OPENWEBUI_ENABLE_USER_MEMORY="${OPENWEBUI_ENABLE_USER_MEMORY:-true}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCAL_REPO_DIR="${ROOT_DIR}/../vela-chat"
if [ -n "${REPO_DIR:-}" ]; then
  : # user override
elif [ -f "${LOCAL_REPO_DIR}/Dockerfile" ]; then
  REPO_DIR="${LOCAL_REPO_DIR}"
else
  REPO_DIR="${ROOT_DIR}/.cache/vela-chat"
fi

started_container=0

log() { printf '[vela-chat] %s\n' "$*"; }
fatal() { log "Error: $*" >&2; exit 1; }

cleanup() {
  if [ "${started_container}" -eq 1 ]; then
    log "Stopping ${NAME}..."
    docker stop "${NAME}" >/dev/null 2>&1 || true
  fi
  exit 0
}
trap cleanup INT TERM HUP QUIT EXIT

ensure_docker() {
  docker info >/dev/null 2>&1 || fatal "Docker daemon not reachable."
}

ensure_repo() {
  if ! command -v git >/dev/null 2>&1; then
    fatal "git is required to build VelaChat locally. Install git, or set IMAGE to a prebuilt image."
  fi

  if [ ! -d "${REPO_DIR}/.git" ]; then
    log "Cloning VelaChat repo into ${REPO_DIR}..."
    mkdir -p "$(dirname -- "${REPO_DIR}")"
    git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${REPO_DIR}" >/dev/null
    return
  fi

  if [ "${UPDATE_REPO}" = "1" ]; then
    log "Updating VelaChat repo in ${REPO_DIR}..."
    git -C "${REPO_DIR}" fetch --depth 1 origin "${REPO_REF}" >/dev/null
    git -C "${REPO_DIR}" checkout -q "${REPO_REF}" || true
    git -C "${REPO_DIR}" reset -q --hard "origin/${REPO_REF}" >/dev/null
  fi
}

ensure_image() {
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    return
  fi

  if [ "${AUTO_BUILD}" != "1" ]; then
    fatal "Docker image ${IMAGE} not found. Set AUTO_BUILD=1 to build, or set IMAGE to an existing image."
  fi

  ensure_repo
  log "Building ${IMAGE} from ${REPO_URL}@${REPO_REF}..."
  docker build -t "${IMAGE}" "${REPO_DIR}" >/dev/null
}

supports_host_network() {
  # Docker Desktop on macOS/Windows does not support --network=host.
  case "$(uname -s)" in
    Linux*) docker run --rm --network host --entrypoint /bin/true "${PROBE_IMAGE}" >/dev/null 2>&1 ;;
    *) false ;;
  esac
}

choose_network() {
  if supports_host_network; then
    NETWORK_MODE="host"
    PORT_ARGS=()
    HOST_ARGS=()
  else
    NETWORK_MODE="bridge"
    PORT_ARGS=( -p "${HOST_PORT}:8080" )
    HOST_ARGS=(
      --add-host "${HOST_ALIAS}:host-gateway"
    )
  fi
}

compute_openai_base() {
  # In host networking, localhost inside container == host. In bridge, use host alias.
  if [ "${NETWORK_MODE}" = "host" ]; then
    OPENAI_BASE="http://127.0.0.1:${MY_AGENT_PORT}/v1"
  else
    OPENAI_BASE="http://${HOST_ALIAS}:${MY_AGENT_PORT}/v1"
  fi
}

recreate_if_network_changed() {
  local existing_id current_net
  existing_id="$(docker ps -aq --filter "name=^${NAME}$" || true)"
  if [ -z "${existing_id}" ]; then
    return
  fi
  current_net="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "${NAME}" 2>/dev/null || echo "")"
  if [ "${current_net}" != "${NETWORK_MODE}" ]; then
    log "Recreating ${NAME} with ${NETWORK_MODE} networking..."
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
}

start_container() {
  if [ -n "$(docker ps -q --filter "name=^${NAME}$" --filter "status=running")" ]; then
    log "Container ${NAME} is already running."
    return
  fi

  ensure_image
  recreate_if_network_changed

  if [ -n "$(docker ps -aq --filter "name=^${NAME}$")" ]; then
    log "Starting existing container ${NAME}..."
    docker start "${NAME}" >/dev/null
  else
    log "Creating and starting ${NAME} with ${NETWORK_MODE} networking..."
    docker run -d --gpus=all \
      "${HOST_ARGS[@]}" \
      "${PORT_ARGS[@]}" \
      --network "${NETWORK_MODE}" \
      -v "${DATA_VOL}":/app/backend/data \
      -v "${OLLAMA_VOL}":/root/.ollama \
      -e OPENAI_API_BASE="${OPENAI_BASE}" \
      -e OPENAI_API_BASE_URLS="${OPENAI_BASE}" \
      -e OPENAI_BASE_URL="${OPENAI_BASE}" \
      -e OPENAI_API_KEY="${MY_OPENAI_API_KEY}" \
      -e ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API}" \
      -e WEBUI_AUTH="${WEBUI_AUTH}" \
      -e OPENWEBUI_ENABLE_USER_MEMORY="${OPENWEBUI_ENABLE_USER_MEMORY}" \
      --name "${NAME}" "${IMAGE}" >/dev/null
  fi

  started_container=1
}

main() {
  ensure_docker
  choose_network
  compute_openai_base
  start_container

  log "Running. Press Ctrl+C to stop ${NAME} (if this script started it)."
  if [ "${NETWORK_MODE}" = "host" ]; then
    log "VelaChat is reachable at http://127.0.0.1:8080"
    log "Inside VelaChat, use http://127.0.0.1:<port>/ for host services."
  else
    log "VelaChat is reachable at http://127.0.0.1:${HOST_PORT}"
    log "Inside VelaChat, use http://${HOST_ALIAS}:<port>/ for host services."
    log "If host services listen only on 127.0.0.1, expose them on 0.0.0.0 or a routable IP."
  fi
  log "Configured OpenAI base: ${OPENAI_BASE}"
  while :; do sleep 86400; done
}

main "$@"

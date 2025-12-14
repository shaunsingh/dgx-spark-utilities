#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-open-webui}"
IMAGE="${IMAGE:-ghcr.io/open-webui/open-webui:ollama}"
HOST_ALIAS="${HOST_ALIAS:-host.docker.internal}"
HOST_PORT="${HOST_PORT:-3000}"          # Only used when host networking is unavailable
DATA_VOL="${DATA_VOL:-open-webui}"
OLLAMA_VOL="${OLLAMA_VOL:-open-webui-ollama}"
PROBE_IMAGE="${PROBE_IMAGE:-alpine:3.20}"

started_container=0

log() { printf '[open-webui] %s\n' "$*"; }
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
    PORT_ARGS=( -p "${HOST_PORT}:3000" )
    HOST_ARGS=(
      --add-host "${HOST_ALIAS}:host-gateway"
    )
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
      --name "${NAME}" "${IMAGE}" >/dev/null
  fi

  started_container=1
}

main() {
  ensure_docker
  choose_network
  start_container

  log "Running. Press Ctrl+C to stop ${NAME} (if this script started it)."
  if [ "${NETWORK_MODE}" = "host" ]; then
    log "OpenWebUI is reachable at http://127.0.0.1:3000"
    log "Inside OpenWebUI, use http://127.0.0.1:<port>/ for host services."
  else
    log "OpenWebUI is reachable at http://127.0.0.1:${HOST_PORT}"
    log "Inside OpenWebUI, use http://${HOST_ALIAS}:<port>/ for host services."
    log "If host services listen only on 127.0.0.1, expose them on 0.0.0.0 or a routable IP."
  fi
  while :; do sleep 86400; done
}

main "$@"

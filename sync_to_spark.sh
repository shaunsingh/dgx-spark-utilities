#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [push|pull]"
  echo "  push (default): local -> remote"
  echo "  pull: remote -> local"
  exit 1
}

DIRECTION="${1:-${DIRECTION:-push}}"

LOCAL_PATH="${LOCAL_PATH:-$HOME/Projects/server/}"
REMOTE_USER="${REMOTE_USER:-shaurizard}"
REMOTE_HOST="${REMOTE_HOST:-spark-3653}"
REMOTE_PATH="${REMOTE_PATH:-~/Projects/server/}"
REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# SSH key (can override with KEY env)
KEY="${KEY:-$HOME/.ssh/tailscale_spark}"

RSYNC_OPTS=(
  -avh --delete
  -e "ssh -i ${KEY} -o StrictHostKeyChecking=accept-new"
)

case "${DIRECTION}" in
  push)
    SRC="${LOCAL_PATH}"
    DEST="${REMOTE_SPEC}"
    ;;
  pull)
    SRC="${REMOTE_SPEC}"
    DEST="${LOCAL_PATH}"
    ;;
  *)
    usage
    ;;
esac

echo "Syncing ${SRC} -> ${DEST}"
rsync "${RSYNC_OPTS[@]}" "${SRC}" "${DEST}"


#!/usr/bin/env bash
set -euo pipefail

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
HUB_DIR="${HF_CACHE%/}/hub"

TARGETS=(
  '*models--Firworks--INTELLECT-3-nvfp4/snapshots/*/config.json'
  '*models--PrimeIntellect--INTELLECT-3/snapshots/*/config.json'
)

for pat in "${TARGETS[@]}"; do
  find "${HUB_DIR}" \
    -path "${pat}" \
    -print \
    -exec python3 - <<'PY' {} \;
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
cfg = json.loads(p.read_text())
if cfg.pop("auto_map", None) is not None:
    p.write_text(json.dumps(cfg, indent=2))
    print(f"patched {p}")
else:
    print(f"no change {p}")
PY
done


HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
find "${HF_CACHE%/}/hub" \
  -path '*models--nvidia--Llama-3_3-Nemotron-Super-49B-v1_5/snapshots/*/config.json' \
  -print -exec python3 - <<'PY' {} \;
import json, sys, pathlib
p=pathlib.Path(sys.argv[1])
cfg=json.loads(p.read_text())
changed=False
# Force a scalar KV head count (all non-pruned blocks use 8 groups: 64 // 8 = 8)
if cfg.get("num_key_value_heads") != 8:
    cfg["num_key_value_heads"] = 8
    changed=True
# Fill missing n_heads_in_group for no-op blocks to avoid list construction
for blk in cfg.get("block_configs", []):
    att=blk.get("attention", {})
    if att.get("n_heads_in_group") is None:
        att["n_heads_in_group"] = 8
        changed=True
if changed:
    p.write_text(json.dumps(cfg, indent=2))
    print(f"patched {p}")
else:
    print(f"no change {p}")
PY
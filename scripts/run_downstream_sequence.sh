#!/bin/bash
set -euo pipefail

INIT_FROM=""
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --config) CONFIGS+=("$2"); shift 2 ;;
    -h|--help)
      echo "Usage: $0 --init-from PRETRAIN_RUN_OR_CKPT --config cfg1.yaml [--config cfg2.yaml ...]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "ERROR: at least one --config is required" >&2
  exit 1
fi

i=0
for CFG in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo "============================================================"
  echo "[sequence] START ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] common init-from: ${INIT_FROM:-<none>}"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
  if [[ -n "$INIT_FROM" ]]; then
    python3 scripts/downstream_finetune_jax.py --config "$CFG" --init-from "$INIT_FROM"
  else
    python3 scripts/downstream_finetune_jax.py --config "$CFG"
  fi
  echo "============================================================"
  echo "[sequence] DONE ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
done
echo "[sequence] ALL DONE (${#CONFIGS[@]} configs)"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/eval_paper_performance.py \
  --manifest configs/paper_eval_manifest.yaml \
  --max_batches -1 \
  --mesh_auto \
  --dtype bf16 \
  --resume_existing \
  --progress_every 5 \
  "$@"

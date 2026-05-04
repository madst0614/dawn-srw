#!/bin/bash
set -euo pipefail

BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIGS="${CONFIGS:?ERROR: CONFIGS env var not set}"
INIT_FROM="${INIT_FROM:-}"
WORK_DIR="$HOME/dawn-spatial"

IFS='|' read -r -a CONFIG_ARRAY <<< "$CONFIGS"

echo "============================================"
echo "Host $(hostname) — Setting up downstream TPU training"
echo "  Branch:    $BRANCH"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Configs:   ${CONFIG_ARRAY[*]}"
echo "============================================"

cd "$WORK_DIR"

echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs datasets transformers sentencepiece -q

echo "[2/4] Verifying downstream files..."
test -f scripts/downstream_finetune_jax.py || { echo "missing scripts/downstream_finetune_jax.py" >&2; exit 2; }
test -f scripts/run_downstream_sequence.sh || { echo "missing scripts/run_downstream_sequence.sh" >&2; exit 2; }
for c in "${CONFIG_ARRAY[@]}"; do
  test -f "$c" || { echo "missing config: $c" >&2; exit 2; }
done

echo "[3/4] Skipping standalone JAX TPU preflight."

echo "[4/4] Starting downstream sequence in tmux session 'train'..."
tmux kill-session -t train 2>/dev/null || true
pkill -f downstream_finetune_jax.py 2>/dev/null || true

XLA_DUMP_DIR="${XLA_DUMP_DIR:-/tmp/xla_dump_downstream}"
mkdir -p "$XLA_DUMP_DIR"
export XLA_DUMP_DIR
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
if [ -z "${XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
else
  export XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
fi

CMD="bash scripts/run_downstream_sequence.sh"
if [ -n "$INIT_FROM" ]; then
  CMD="$CMD --init-from '$INIT_FROM'"
fi
for c in "${CONFIG_ARRAY[@]}"; do
  CMD="$CMD --config '$c'"
done

echo "  Command: $CMD"
echo "  Log: ~/train.log"

tmux new-session -d -s train \
  "export XLA_DUMP_DIR='$XLA_DUMP_DIR'; \
   export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; \
   export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; \
   export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; \
   export XLA_FLAGS='$XLA_FLAGS'; \
   cd '$WORK_DIR'; \
   $CMD 2>&1 | tee ~/train.log; \
   echo 'Downstream sequence finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"

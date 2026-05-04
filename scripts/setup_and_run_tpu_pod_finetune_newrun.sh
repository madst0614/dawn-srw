#!/bin/bash
set -euo pipefail

BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
INIT_FROM="${INIT_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_jax_finetune_newrun.py}"
WORK_DIR="$HOME/dawn-spatial"

printf '%s\n' "============================================"
printf '%s\n' "Host $(hostname) — Setting up TPU Pod finetune-newrun training"
printf '%s\n' "  Branch:       $BRANCH"
printf '%s\n' "  Config:       $CONFIG"
printf '%s\n' "  Init from:    ${INIT_FROM:-<none>}"
printf '%s\n' "  Resume from:  ${RESUME_FROM:-<none>}"
printf '%s\n' "  Train script: $TRAIN_SCRIPT"
printf '%s\n' "  Train args:   $TRAIN_ARGS"
printf '%s\n' "============================================"

cd "$WORK_DIR"

echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

echo "[2/4] Verifying files..."
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "ERROR: missing train script: $TRAIN_SCRIPT" >&2
  ls -l scripts >&2 || true
  exit 2
fi

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: missing config: $CONFIG" >&2
  ls -l configs >&2 || true
  exit 2
fi

echo "[3/4] Skipping standalone JAX TPU preflight."

echo "[4/4] Starting training in tmux session 'train'..."
tmux kill-session -t train 2>/dev/null || true
pkill -f 'train_jax_.*newrun.py' 2>/dev/null || true

XLA_DUMP_DIR="${XLA_DUMP_DIR:-/tmp/xla_dump_train}"
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

CMD="python3 '$TRAIN_SCRIPT' --config '$CONFIG'"

if [ -n "$INIT_FROM" ]; then
  CMD="$CMD --init-from '$INIT_FROM'"
fi

if [ -n "$RESUME_FROM" ]; then
  CMD="$CMD --resume-from '$RESUME_FROM'"
fi

if [ -n "$TRAIN_ARGS" ]; then
  CMD="$CMD $TRAIN_ARGS"
fi

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
   echo 'Training finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"

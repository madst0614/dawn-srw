#!/bin/bash
# =============================================================================
# TPU Pod Setup + Init-NewRun Training Script (runs on each worker)
# =============================================================================
# Reads an existing checkpoint via --init-from, then writes checkpoints/logs to
# a fresh run folder from the config's checkpoint_dir. Source checkpoint is read-only.
#
# Required env:
#   BRANCH, CONFIG, INIT_FROM
# Optional env:
#   TRAIN_SCRIPT=scripts/train_jax_init_newrun.py
#   TRAIN_ARGS="--batch-size 128 --lr 2e-5"
#   GH_TOKEN=...
# =============================================================================

set -euo pipefail

GH_TOKEN="${GH_TOKEN:-}"
if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi
BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
INIT_FROM="${INIT_FROM:?ERROR: INIT_FROM env var not set}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_jax_init_newrun.py}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
WORK_DIR="$HOME/dawn-spatial"

cat <<EOF
============================================
Host $(hostname) — Setting up TPU Pod init-newrun training
  Branch:       $BRANCH
  Config:       $CONFIG
  Init from:    $INIT_FROM
  Train script: $TRAIN_SCRIPT
  Train args:   $TRAIN_ARGS
============================================
EOF

# 1. Install dependencies (all workers)
echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

# 2. Deploy code via git (clone or update)
echo "[2/4] Syncing repo (branch: $BRANCH)..."
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    git fetch origin "$BRANCH" --depth 1
    git checkout -B "$BRANCH" FETCH_HEAD
else
    cd "$HOME"
    rm -rf dawn-spatial
    git clone -b "$BRANCH" --single-branch --depth 1 "$REPO_URL" dawn-spatial
    cd dawn-spatial
fi

# 3. Skip standalone JAX preflight.
echo "[3/4] Skipping standalone JAX TPU preflight; train script will verify devices."

# 4. Launch training in tmux (survives SSH disconnect)
echo "[4/4] Starting training in tmux session 'train'..."
cd "$WORK_DIR"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: TRAIN_SCRIPT not found: $TRAIN_SCRIPT" >&2
    echo "Copy train_jax_init_newrun.py into scripts/ or pass TRAIN_SCRIPT correctly." >&2
    exit 1
fi
if [ ! -f "$CONFIG" ] && [[ "$CONFIG" != gs://* ]]; then
    echo "ERROR: CONFIG not found: $CONFIG" >&2
    exit 1
fi

tmux kill-session -t train 2>/dev/null || true

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

tmux new-session -d -s train \
    "export XLA_DUMP_DIR='$XLA_DUMP_DIR'; export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; export XLA_FLAGS='$XLA_FLAGS'; python3 '$TRAIN_SCRIPT' --config '$CONFIG' --init-from '$INIT_FROM' $TRAIN_ARGS 2>&1 | tee ~/train.log; echo 'Training finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"

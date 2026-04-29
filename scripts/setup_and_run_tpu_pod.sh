#!/bin/bash
# =============================================================================
# TPU Pod Setup + Training Script (runs on each worker)
# =============================================================================
# Expects BRANCH and CONFIG passed as environment variables from the launcher.
#
# Usage (direct):
#   BRANCH=main CONFIG=configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml \
#     bash scripts/setup_and_run_tpu_pod.sh
#
# Usually invoked via launch_tpu_pod.sh which sets env vars automatically.
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
WORK_DIR="$HOME/dawn-spatial"

echo "============================================"
echo "Host $(hostname) — Setting up TPU Pod training"
echo "  Branch: $BRANCH"
echo "  Config: $CONFIG"
echo "============================================"

# 1. Install dependencies (all workers)
echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

# 2. Deploy code via git (clone or update)
echo "[2/4] Syncing repo (branch: $BRANCH)..."
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
        echo "  Already on branch $BRANCH, pulling latest..."
        git pull origin "$BRANCH" --ff-only || true
    else
        echo "  Switching to branch $BRANCH..."
        git fetch origin "$BRANCH" --depth 1
        git checkout -B "$BRANCH" FETCH_HEAD
    fi
else
    echo "  Fresh clone (branch: $BRANCH)..."
    cd "$HOME"
    rm -rf dawn-spatial
    git clone -b "$BRANCH" --single-branch --depth 1 "$REPO_URL" dawn-spatial
    cd dawn-spatial
fi

# 3. Skip standalone JAX preflight.
#
# On multi-host TPU pods a short-lived standalone JAX process can initialize
# PJRT, print device info, and then abort during teardown with:
#   GetSliceInfo can only be invoked after a slice is built...
# The real training process below performs the same backend/device checks and
# keeps the slice alive, so avoid opening a throwaway slice here.
echo "[3/4] Skipping standalone JAX TPU preflight; train_jax.py will verify devices."

# 4. Launch training in tmux (survives SSH disconnect)
echo "[4/4] Starting training in tmux session 'train'..."
echo "  Config: $CONFIG"
echo "  Host: $(hostname)"
echo "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Log: ~/train.log"

cd "$WORK_DIR"

# Kill existing train session if any
tmux kill-session -t train 2>/dev/null || true

# Enable XLA dumps by default so OOM-check failures can point at the HLO
# memory report. Keep console logging quiet; train_jax.py prints a compact
# excerpt from the dump only when the OOM check fails.
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

# Start new tmux session running training, tee to ~/train.log
TRAIN_ARGS="${TRAIN_ARGS:-}"
tmux new-session -d -s train \
    "export XLA_DUMP_DIR='$XLA_DUMP_DIR'; export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; export XLA_FLAGS='$XLA_FLAGS'; python3 scripts/train_jax.py --config '$CONFIG' $TRAIN_ARGS 2>&1 | tee ~/train.log; echo 'Training finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"

#!/bin/bash
# =============================================================================
# TPU launcher for DAWN-SRW paper data extraction
# =============================================================================
# Runs scripts/analysis/run_paper_data.sh on a TPU VM or pod.  By default it
# launches on all workers so the validation eval can use a multi-host JAX mesh.
# Host-0-only mechanism scripts are skipped automatically on non-primary hosts by
# the runner.  Use --single-device to force worker 0 / one-device execution.
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
GH_TOKEN=""

TF_CONFIG="configs/train_config_baseline_tpu_1B_c4_40B_v4_32.yaml"
TF_CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/baseline_1B_c4_40B"
DAWN_CONFIG="configs/train_config_dawn_srw_1B_c4_40B_v4_32.yaml"
DAWN_CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_1B_c4_40B"
VAL_DATA="gs://dawn-tpu-data-c4/c4_val.bin"
OUT_ROOT="gs://dawn-tpu-data-c4/paper_results/dawn_srw_compare"

TOKENIZER="bert-base-uncased"
MAX_VAL_TOKENS="10000000"
VAL_BATCH_SIZE="16"
EVAL_SEQ_LEN="0"
MAX_BATCHES="0"
ACTIVE_MAX_VAL_TOKENS="65536"
ACTIVE_BATCH_SIZE="1"
ACTIVE_SEQ_LEN="128"
ACTIVE_MAX_BATCHES="0"
EVAL_MESH_DATA=""
EVAL_MESH_MODEL=""
PAPER_SINGLE_DEVICE="0"

WORKERS="auto"
DETACH="0"
INSTALL_DEPS="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --tf-config) TF_CONFIG="$2"; shift 2 ;;
        --tf-checkpoint) TF_CHECKPOINT="$2"; shift 2 ;;
        --dawn-config) DAWN_CONFIG="$2"; shift 2 ;;
        --dawn-checkpoint) DAWN_CHECKPOINT="$2"; shift 2 ;;
        --val-data) VAL_DATA="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --max-val-tokens) MAX_VAL_TOKENS="$2"; shift 2 ;;
        --batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) EVAL_SEQ_LEN="$2"; shift 2 ;;
        --max-batches) MAX_BATCHES="$2"; shift 2 ;;
        --active-max-val-tokens) ACTIVE_MAX_VAL_TOKENS="$2"; shift 2 ;;
        --active-batch-size) ACTIVE_BATCH_SIZE="$2"; shift 2 ;;
        --active-seq-len) ACTIVE_SEQ_LEN="$2"; shift 2 ;;
        --active-max-batches) ACTIVE_MAX_BATCHES="$2"; shift 2 ;;
        --mesh-data) EVAL_MESH_DATA="$2"; shift 2 ;;
        --mesh-model) EVAL_MESH_MODEL="$2"; shift 2 ;;
        --single-device) PAPER_SINGLE_DEVICE="1"; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --detach) DETACH="1"; shift ;;
        --no-install) INSTALL_DEPS="0"; shift ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [options]"
            echo ""
            echo "Core:"
            echo "  --tf-config PATH"
            echo "  --tf-checkpoint PATH_OR_GS"
            echo "  --dawn-config PATH"
            echo "  --dawn-checkpoint PATH_OR_GS"
            echo "  --val-data PATH_OR_GS"
            echo "  --out-root PATH_OR_GS"
            echo ""
            echo "Execution:"
            echo "  --workers all|0|N       Default: all, or 0 with --single-device"
            echo "  --single-device         Force worker 0 / one-device execution"
            echo "  --mesh-data N           Override validation mesh data axis"
            echo "  --mesh-model N          Override validation mesh model axis"
            echo "  --detach                Run in tmux on remote workers"
            echo "  --no-install            Skip pip dependency install"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu required" >&2
    exit 1
fi

if [[ "$WORKERS" = "auto" ]]; then
    if [[ "$PAPER_SINGLE_DEVICE" = "1" ]]; then
        WORKERS="0"
    else
        WORKERS="all"
    fi
fi

if [[ -n "$GH_TOKEN" ]]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

echo "============================================"
echo "Launching DAWN-SRW paper data extraction"
echo "  TPU:          $TPU_NAME"
echo "  Zone:         $ZONE"
echo "  Project:      $PROJECT"
echo "  Branch:       $BRANCH"
echo "  Workers:      $WORKERS"
echo "  Detached:     $DETACH"
echo "  SingleDevice: $PAPER_SINGLE_DEVICE"
echo "  Output:       $OUT_ROOT"
echo "============================================"

echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
TF_CONFIG='${TF_CONFIG}'
TF_CHECKPOINT='${TF_CHECKPOINT}'
DAWN_CONFIG='${DAWN_CONFIG}'
DAWN_CHECKPOINT='${DAWN_CHECKPOINT}'
VAL_DATA='${VAL_DATA}'
OUT_ROOT='${OUT_ROOT}'
TOKENIZER='${TOKENIZER}'
MAX_VAL_TOKENS='${MAX_VAL_TOKENS}'
VAL_BATCH_SIZE='${VAL_BATCH_SIZE}'
EVAL_SEQ_LEN='${EVAL_SEQ_LEN}'
MAX_BATCHES='${MAX_BATCHES}'
ACTIVE_MAX_VAL_TOKENS='${ACTIVE_MAX_VAL_TOKENS}'
ACTIVE_BATCH_SIZE='${ACTIVE_BATCH_SIZE}'
ACTIVE_SEQ_LEN='${ACTIVE_SEQ_LEN}'
ACTIVE_MAX_BATCHES='${ACTIVE_MAX_BATCHES}'
EVAL_MESH_DATA='${EVAL_MESH_DATA}'
EVAL_MESH_MODEL='${EVAL_MESH_MODEL}'
PAPER_SINGLE_DEVICE='${PAPER_SINGLE_DEVICE}'
DETACH='${DETACH}'
INSTALL_DEPS='${INSTALL_DEPS}'
WORK_DIR="\$HOME/dawn-spatial"

echo "[setup] host=\$(hostname) branch=\$BRANCH"
if [ -d "\$WORK_DIR/.git" ]; then
    cd "\$WORK_DIR"
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
else
    rm -rf "\$WORK_DIR"
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
    cd "\$WORK_DIR"
fi

if [ "\$INSTALL_DEPS" = "1" ]; then
    echo "[setup] installing dependencies"
    python3 -m pip install --upgrade pip -q
    python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
    python3 -m pip install flax optax numpy pyyaml gcsfs transformers matplotlib -q
fi

export PYTHON_BIN=python3
export TOKENIZER="\$TOKENIZER"
export MAX_VAL_TOKENS="\$MAX_VAL_TOKENS"
export VAL_BATCH_SIZE="\$VAL_BATCH_SIZE"
export EVAL_SEQ_LEN="\$EVAL_SEQ_LEN"
export MAX_BATCHES="\$MAX_BATCHES"
export ACTIVE_MAX_VAL_TOKENS="\$ACTIVE_MAX_VAL_TOKENS"
export ACTIVE_BATCH_SIZE="\$ACTIVE_BATCH_SIZE"
export ACTIVE_SEQ_LEN="\$ACTIVE_SEQ_LEN"
export ACTIVE_MAX_BATCHES="\$ACTIVE_MAX_BATCHES"
export EVAL_MESH_DATA="\$EVAL_MESH_DATA"
export EVAL_MESH_MODEL="\$EVAL_MESH_MODEL"
export PAPER_SINGLE_DEVICE="\$PAPER_SINGLE_DEVICE"
export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"

RUN_CMD=(
    bash scripts/analysis/run_paper_data.sh
    "\$TF_CONFIG"
    "\$TF_CHECKPOINT"
    "\$DAWN_CONFIG"
    "\$DAWN_CHECKPOINT"
    "\$VAL_DATA"
    "\$OUT_ROOT"
)
RUN_CMD_STR=\$(printf "%q " "\${RUN_CMD[@]}")

if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session paper_data_analysis"
    tmux kill-session -t paper_data_analysis 2>/dev/null || true
    tmux new-session -d -s paper_data_analysis \
        "cd '\$WORK_DIR'; \$RUN_CMD_STR > >(tee ~/paper_data_analysis.log) 2> >(tee ~/paper_data_analysis_progress.log >&2); echo 'Analysis finished. Press enter to close.'; read"
    echo "[run] detached. data=~/paper_data_analysis.log progress=~/paper_data_analysis_progress.log"
else
    echo "[run] foreground analysis; progress and data stream below"
    cd "\$WORK_DIR"
    "\${RUN_CMD[@]}" > >(tee ~/paper_data_analysis.log) 2> >(tee ~/paper_data_analysis_progress.log >&2)
fi
EOFCMD

echo "Sending command to worker(s): $WORKERS"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker="$WORKERS" \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_paper_data_analysis_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Done launching/running."
echo "  Data worker 0:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='cat ~/paper_data_analysis.log'"
echo "  Progress worker 0: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/paper_data_analysis_progress.log'"

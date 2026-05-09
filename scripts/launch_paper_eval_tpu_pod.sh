#!/bin/bash
# =============================================================================
# TPU launcher for paper performance evaluation
# =============================================================================
# Launches scripts/eval_paper_performance.py on a TPU VM/pod. The default path
# is tuned for the 40M v4-32 paper sweep manifest:
#   - full validation, max_batches=-1
#   - all checkpoint paths listed in configs/paper_eval_manifest.yaml
#   - all workers, detached tmux session
#   - results written to GCS for easy host-side inspection
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
GH_TOKEN=""

MANIFEST="configs/paper_eval_manifest.yaml"
OUT_DIR="gs://dawn-tpu-data-c4/paper_results/paper_eval_v4_32_20260509"
MAX_BATCHES="-1"
BATCH_SIZE=""
SEQ_LEN=""
DTYPE="bf16"
PROGRESS_EVERY="5"
EXTRA_ARGS=""

WORKERS="auto"
DETACH="1"
INSTALL_DEPS="1"
SINGLE_DEVICE="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --output-dir) OUT_DIR="$2"; shift 2 ;;
        --max-batches) MAX_BATCHES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --single-device) SINGLE_DEVICE="1"; shift ;;
        --foreground) DETACH="0"; shift ;;
        --detach) DETACH="1"; shift ;;
        --no-install) INSTALL_DEPS="0"; shift ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [options]"
            echo ""
            echo "Core:"
            echo "  --manifest PATH       Default: $MANIFEST"
            echo "  --output-dir PATH     Default: $OUT_DIR"
            echo "  --max-batches N       Default: -1 (full validation)"
            echo "  --batch-size N        Optional override"
            echo "  --seq-len N           Optional override"
            echo "  --dtype bf16|fp32     Default: $DTYPE"
            echo ""
            echo "Execution:"
            echo "  --workers all|0|N     Default: all, or 0 with --single-device"
            echo "  --single-device       Worker 0 only"
            echo "  --foreground          Stream run instead of tmux detach"
            echo "  --detach              Run in tmux (default)"
            echo "  --no-install          Skip pip dependency install"
            echo "  --extra-args '...'    Extra eval_paper_performance.py args"
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
    if [[ "$SINGLE_DEVICE" = "1" ]]; then
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
echo "Launching DAWN-SRW paper performance eval"
echo "  TPU:          $TPU_NAME"
echo "  Zone:         $ZONE"
echo "  Project:      $PROJECT"
echo "  Branch:       $BRANCH"
echo "  Workers:      $WORKERS"
echo "  Detached:     $DETACH"
echo "  Manifest:     $MANIFEST"
echo "  Output:       $OUT_DIR"
echo "  Max batches:  $MAX_BATCHES"
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
MANIFEST='${MANIFEST}'
OUT_DIR='${OUT_DIR}'
MAX_BATCHES='${MAX_BATCHES}'
BATCH_SIZE='${BATCH_SIZE}'
SEQ_LEN='${SEQ_LEN}'
DTYPE='${DTYPE}'
PROGRESS_EVERY='${PROGRESS_EVERY}'
EXTRA_ARGS='${EXTRA_ARGS}'
DETACH='${DETACH}'
INSTALL_DEPS='${INSTALL_DEPS}'
SINGLE_DEVICE='${SINGLE_DEVICE}'
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
    python3 -m pip install flax optax numpy pyyaml gcsfs -q
fi

export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"

RUN_CMD=(
    python3 scripts/eval_paper_performance.py
    --manifest "\$MANIFEST"
    --output_dir "\$OUT_DIR"
    --max_batches "\$MAX_BATCHES"
    --mesh_auto
    --dtype "\$DTYPE"
    --resume_existing
    --progress_every "\$PROGRESS_EVERY"
)
if [ -n "\$BATCH_SIZE" ]; then RUN_CMD+=(--batch_size "\$BATCH_SIZE"); fi
if [ -n "\$SEQ_LEN" ]; then RUN_CMD+=(--seq_len "\$SEQ_LEN"); fi
if [ "\$SINGLE_DEVICE" = "1" ]; then RUN_CMD+=(--single_device); fi
if [ -n "\$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARRAY=(\$EXTRA_ARGS)
    RUN_CMD+=("\${EXTRA_ARRAY[@]}")
fi
RUN_CMD_STR=\$(printf "%q " "\${RUN_CMD[@]}")

if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session paper_eval_performance"
    tmux kill-session -t paper_eval_performance 2>/dev/null || true
    tmux new-session -d -s paper_eval_performance \
        "cd '\$WORK_DIR'; \$RUN_CMD_STR > >(tee ~/paper_eval_performance.log) 2> >(tee ~/paper_eval_performance_progress.log >&2); echo 'Paper eval finished. Press enter to close.'; read"
    echo "[run] detached. data=~/paper_eval_performance.log progress=~/paper_eval_performance_progress.log"
else
    echo "[run] foreground eval; progress and data stream below"
    cd "\$WORK_DIR"
    "\${RUN_CMD[@]}" > >(tee ~/paper_eval_performance.log) 2> >(tee ~/paper_eval_performance_progress.log >&2)
fi
EOFCMD

echo "Sending command to worker(s): $WORKERS"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker="$WORKERS" \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_paper_eval_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launched."
echo "  Data log:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='cat ~/paper_eval_performance.log'"
echo "  Progress:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/paper_eval_performance_progress.log'"
echo "  Attach:       gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t paper_eval_performance'"
echo "  Kill:         gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t paper_eval_performance'"
echo "  List GCS:     gcloud storage ls $OUT_DIR"
echo "  Main table:   gcloud storage cat $OUT_DIR/table_main_performance.csv"
echo "  FLOPs table:  gcloud storage cat $OUT_DIR/table_theoretical_flops_main.csv"

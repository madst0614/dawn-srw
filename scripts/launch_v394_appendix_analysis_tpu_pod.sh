#!/bin/bash
# =============================================================================
# TPU Pod Launcher - v3.9.4 appendix analysis
# =============================================================================
# Run from a local machine or Cloud Shell. This mirrors launch_tpu_pod.sh:
# it updates/clones the repo on every TPU worker, installs TPU dependencies,
# and starts the one-shot appendix analysis in a tmux session.
#
# Usage:
#   bash scripts/launch_v394_appendix_analysis_tpu_pod.sh --tpu dawn-400m-v4-32 --branch main
#
# The Python analyzer prints copyable paper output only on JAX host 0.
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
GH_TOKEN=""

RESULTS="gs://dawn-tpu-data-c4/downstream_runs/result"
TF_CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/baseline_400M_c4_40B_v4_32/run_vbaseline_20260322_185640_3201"
DAWN394_CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_spatial_r1v3.9.4_400M_c4_40B/run_vspatial-r1-v3.9.4_20260409_042116_3201"
VAL_DATA="gs://dawn-tpu-data-c4/c4_val.bin"
BATCH_SIZE="16"
SEQ_LEN="512"
RUN_FORWARD="1"
SHOW_PATHS="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --tf-checkpoint) TF_CHECKPOINT="$2"; shift 2 ;;
        --dawn394-checkpoint) DAWN394_CHECKPOINT="$2"; shift 2 ;;
        --val-data) VAL_DATA="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --no-forward) RUN_FORWARD="0"; shift ;;
        --show-paths) SHOW_PATHS="1"; shift ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [--zone ZONE] [--project PROJECT] [--branch BRANCH] [--token GH_TOKEN]"
            echo ""
            echo "Defaults:"
            echo "  --results             $RESULTS"
            echo "  --tf-checkpoint       $TF_CHECKPOINT"
            echo "  --dawn394-checkpoint  $DAWN394_CHECKPOINT"
            echo "  --val-data            $VAL_DATA"
            echo "  --batch-size          $BATCH_SIZE"
            echo "  --seq-len             $SEQ_LEN"
            echo ""
            echo "Flags:"
            echo "  --no-forward          Parse downstream/checkpoints only; skip validation/utilization forward pass."
            echo "  --show-paths          Include internal paths in paper output."
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

if [[ -n "$GH_TOKEN" ]]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

echo "============================================"
echo "Launching v3.9.4 appendix analysis"
echo "  TPU:       $TPU_NAME"
echo "  Zone:      $ZONE"
echo "  Project:   $PROJECT"
echo "  Branch:    $BRANCH"
echo "  Forward:   $RUN_FORWARD"
echo "  Batch:     $BATCH_SIZE"
echo "  Seq len:   $SEQ_LEN"
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
RESULTS='${RESULTS}'
TF_CHECKPOINT='${TF_CHECKPOINT}'
DAWN394_CHECKPOINT='${DAWN394_CHECKPOINT}'
VAL_DATA='${VAL_DATA}'
BATCH_SIZE='${BATCH_SIZE}'
SEQ_LEN='${SEQ_LEN}'
RUN_FORWARD='${RUN_FORWARD}'
SHOW_PATHS='${SHOW_PATHS}'
WORK_DIR="\$HOME/dawn-spatial"

echo "============================================"
echo "Host \$(hostname) - Setting up v3.9.4 appendix analysis"
echo "  Branch: \$BRANCH"
echo "============================================"

echo "[1/4] Syncing repo..."
if [ -d "\$WORK_DIR/.git" ]; then
    cd "\$WORK_DIR"
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
    echo "Repo updated to \$BRANCH"
else
    rm -rf "\$WORK_DIR"
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
    cd "\$WORK_DIR"
    echo "Repo cloned at \$BRANCH"
fi

echo "[2/4] Installing dependencies..."
python3 -m pip install --upgrade pip -q
python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
python3 -m pip install flax optax numpy pyyaml gcsfs -q

echo "[3/4] Preparing analysis command..."
cd "\$WORK_DIR"
ANALYSIS_CMD=(
    python3 scripts/analyze_v394_appendix.py
    --results "\$RESULTS"
    --tf-checkpoint "\$TF_CHECKPOINT"
    --dawn394-checkpoint "\$DAWN394_CHECKPOINT"
    --batch-size "\$BATCH_SIZE"
    --seq-len "\$SEQ_LEN"
)
if [ "\$RUN_FORWARD" = "1" ]; then
    ANALYSIS_CMD+=(--run-forward --val-data "\$VAL_DATA")
fi
if [ "\$SHOW_PATHS" = "1" ]; then
    ANALYSIS_CMD+=(--show-paths)
fi
ANALYSIS_CMD_STR=\$(printf "%q " "\${ANALYSIS_CMD[@]}")

echo "[4/4] Starting tmux session 'v394_analysis'..."
tmux kill-session -t v394_analysis 2>/dev/null || true
export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"
tmux new-session -d -s v394_analysis \
    "export JAX_TRACEBACK_FILTERING='\$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='\$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='\$TF_CPP_MIN_LOG_LEVEL'; \$ANALYSIS_CMD_STR 2>&1 | tee ~/v394_appendix_analysis.log; echo 'Analysis finished. Press enter to close.'; read"

echo "  tmux session 'v394_analysis' started."
echo "  Monitor: tail -f ~/v394_appendix_analysis.log"
EOFCMD

echo "Sending analysis command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_v394_analysis_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete. Analysis is running in tmux session 'v394_analysis' on all workers."
echo "  Log:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/v394_appendix_analysis.log'"
echo "  Attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t v394_analysis'"
echo "  Kill:   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t v394_analysis'"

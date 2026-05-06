#!/bin/bash
# =============================================================================
# TPU launcher for v3.9.4 data extraction
# =============================================================================
# This is only a launcher, matching the training launch style. The analysis
# itself lives in scripts/analyze_v394_appendix.py.
#
# Default mode runs the validation/utilization forward pass, so it uses all TPU
# workers. Use --no-forward to parse downstream/checkpoint metadata on worker 0.
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
WORKERS="auto"
DETACH="0"

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
        --workers) WORKERS="$2"; shift 2 ;;
        --detach) DETACH="1"; shift ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME --branch BRANCH [options]"
            echo ""
            echo "Defaults:"
            echo "  --results             $RESULTS"
            echo "  --tf-checkpoint       $TF_CHECKPOINT"
            echo "  --dawn394-checkpoint  $DAWN394_CHECKPOINT"
            echo "  --val-data            $VAL_DATA"
            echo "  --batch-size          $BATCH_SIZE"
            echo "  --seq-len             $SEQ_LEN"
            echo ""
            echo "Options:"
            echo "  --no-forward          Skip validation/utilization; runs on worker 0 by default."
            echo "  --workers all|0|N     Override worker selection. Default: all with forward, 0 without forward."
            echo "  --detach              Run inside tmux on workers and return immediately."
            echo "  --show-paths          Include checkpoint/log paths in final output."
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
    if [[ "$RUN_FORWARD" = "1" ]]; then
        WORKERS="all"
    else
        WORKERS="0"
    fi
fi

if [[ -n "$GH_TOKEN" ]]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

echo "============================================"
echo "Launching v3.9.4 data extraction"
echo "  TPU:       $TPU_NAME"
echo "  Zone:      $ZONE"
echo "  Project:   $PROJECT"
echo "  Branch:    $BRANCH"
echo "  Workers:   $WORKERS"
echo "  Forward:   $RUN_FORWARD"
echo "  Detached:  $DETACH"
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
DETACH='${DETACH}'
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

echo "[setup] installing dependencies"
python3 -m pip install --upgrade pip -q
python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
python3 -m pip install flax optax numpy pyyaml gcsfs -q

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

export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"

if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session v394_analysis"
    tmux kill-session -t v394_analysis 2>/dev/null || true
    tmux new-session -d -s v394_analysis \
        "cd '\$WORK_DIR'; export JAX_TRACEBACK_FILTERING='\$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='\$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='\$TF_CPP_MIN_LOG_LEVEL'; \$ANALYSIS_CMD_STR > >(tee ~/v394_appendix_data.log) 2> >(tee ~/v394_appendix_progress.log >&2); echo 'Analysis finished. Press enter to close.'; read"
    echo "[run] detached. data=~/v394_appendix_data.log progress=~/v394_appendix_progress.log"
else
    echo "[run] foreground analysis; progress and data stream below"
    cd "\$WORK_DIR"
    "\${ANALYSIS_CMD[@]}" > >(tee ~/v394_appendix_data.log) 2> >(tee ~/v394_appendix_progress.log >&2)
fi
EOFCMD

echo "Sending command to worker(s): $WORKERS"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker="$WORKERS" \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_v394_analysis_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Done launching/running."
echo "  Data worker 0:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='cat ~/v394_appendix_data.log'"
echo "  Progress worker 0: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/v394_appendix_progress.log'"

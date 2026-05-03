#!/bin/bash
# =============================================================================
# TPU Pod Launcher — init from checkpoint, write a NEW run folder
# =============================================================================
# Run from local machine / Cloud Shell. Sends setup script to all workers.
# Source checkpoint folder/file is only read. New checkpoints are written to
# config.checkpoint_dir/run_v...
# =============================================================================

set -euo pipefail

TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG="configs/train_config_baseline_tpu_400M_c4_40B_v4_32_stable.yaml"
INIT_FROM=""
TRAIN_SCRIPT="scripts/train_jax_init_newrun.py"
GH_TOKEN=""
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu)          TPU_NAME="$2"; shift 2 ;;
        --zone)         ZONE="$2"; shift 2 ;;
        --project)      PROJECT="$2"; shift 2 ;;
        --branch)       BRANCH="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --init-from)    INIT_FROM="$2"; shift 2 ;;
        --train-script) TRAIN_SCRIPT="$2"; shift 2 ;;
        --token)        GH_TOKEN="$2"; shift 2 ;;
        --load-opt-state) TRAIN_ARGS="$TRAIN_ARGS --load-opt-state"; shift ;;
        --preserve-step)  TRAIN_ARGS="$TRAIN_ARGS --preserve-step"; shift ;;
        --batch-size)   TRAIN_ARGS="$TRAIN_ARGS --batch-size $2"; shift 2 ;;
        --epochs)       TRAIN_ARGS="$TRAIN_ARGS --epochs $2"; shift 2 ;;
        --lr)           TRAIN_ARGS="$TRAIN_ARGS --lr $2"; shift 2 ;;
        --no-inherit-model-config) TRAIN_ARGS="$TRAIN_ARGS --no-inherit-model-config"; shift ;;
        --inherit-training-config) TRAIN_ARGS="$TRAIN_ARGS --inherit-training-config"; shift ;;
        --debug)        TRAIN_ARGS="$TRAIN_ARGS --debug"; shift ;;
        -h|--help)
            cat <<EOF
Usage:
  $0 --init-from CHECKPOINT_OR_RUN_FOLDER --config CONFIG [options]

Required:
  --init-from PATH       .flax file or run folder to READ from

Common:
  --config PATH          config YAML for the NEW run
  --branch NAME          git branch
  --tpu NAME --zone Z --project P
  --batch-size N --epochs N --lr LR
  --load-opt-state       load optimizer state too; default loads params only
  --preserve-step        keep checkpoint step/epoch; default resets to 0
  --token TOKEN          GitHub token for private repo

Examples:
  # Transformer baseline
  $0 --config configs/train_config_baseline_tpu_400M_c4_40B_v4_32_stable.yaml \\
     --init-from gs://BUCKET/tf_400m/run_xxx

  # DAWN v3.9.4 legacy
  $0 --config configs/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32_stable.yaml \\
     --init-from gs://BUCKET/dawn394_400m/run_xxx
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1 (use --help)" >&2; exit 1 ;;
    esac
done

if [ -z "$INIT_FROM" ]; then
    echo "ERROR: --init-from is required" >&2
    exit 1
fi

cat <<EOF
============================================
Launching TPU Pod init-newrun training
  TPU:          $TPU_NAME
  Zone:         $ZONE
  Project:      $PROJECT
  Branch:       $BRANCH
  Config:       $CONFIG
  Init from:    $INIT_FROM
  Train script: $TRAIN_SCRIPT
  Args:         $TRAIN_ARGS
============================================
EOF

echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)"

if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -e
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
INIT_FROM='${INIT_FROM}'
GH_TOKEN='${GH_TOKEN}'
TRAIN_SCRIPT='${TRAIN_SCRIPT}'
TRAIN_ARGS='${TRAIN_ARGS}'
export BRANCH CONFIG INIT_FROM GH_TOKEN TRAIN_SCRIPT TRAIN_ARGS

if [ -d ~/dawn-spatial/.git ]; then
    cd ~/dawn-spatial
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
    echo "Repo updated to \$BRANCH"
else
    rm -rf ~/dawn-spatial
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" ~/dawn-spatial
    echo "Repo cloned (branch: \$BRANCH)"
fi

cd ~/dawn-spatial
bash scripts/setup_and_run_tpu_pod_init_newrun.sh
EOFCMD

echo "Sending bootstrap+training command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_${TPU_NAME}_init_newrun_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete. Training is running in tmux session 'train' on all workers."
echo "  Log:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"

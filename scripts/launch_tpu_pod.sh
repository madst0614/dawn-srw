#!/bin/bash
# =============================================================================
# TPU Pod Launcher — run from local machine or Cloud Shell
# =============================================================================
# Sends setup_and_run_tpu_pod.sh to all workers with the specified branch/config.
#
# Usage:
#   bash scripts/launch_tpu_pod.sh --tpu dawn-400m-v4-64 --branch main --config configs/v4_64.yaml
#   bash scripts/launch_tpu_pod.sh  # uses defaults (v4-64 settings)
#
# Prerequisites:
#   1. TPU VM created:
#      gcloud compute tpus tpu-vm create dawn-400m-v4-64 \
#        --zone=us-central2-b --accelerator-type=v4-64 \
#        --version=tpu-vm-v4-base --spot
# =============================================================================

set -euo pipefail

# Defaults
TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG="configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml"
GH_TOKEN=""
TRAIN_ARGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu)      TPU_NAME="$2"; shift 2 ;;
        --zone)     ZONE="$2";     shift 2 ;;
        --project)  PROJECT="$2";  shift 2 ;;
        --branch)   BRANCH="$2";   shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        --token)    GH_TOKEN="$2"; shift 2 ;;
        --from-scratch) TRAIN_ARGS="$TRAIN_ARGS --from-scratch"; shift ;;
        -h|--help)
            echo "Usage: $0 [--tpu NAME] [--zone ZONE] [--project PROJECT] [--branch BRANCH] [--config CONFIG] [--token GH_TOKEN] [--from-scratch]"
            echo ""
            echo "  --tpu      TPU VM name         (default: $TPU_NAME)"
            echo "  --zone     GCP zone            (default: $ZONE)"
            echo "  --project  GCP project          (default: $PROJECT)"
            echo "  --branch   Git branch to clone  (default: $BRANCH)"
            echo "  --config   Training config YAML (default: $CONFIG)"
            echo "  --from-scratch  Start training from scratch (ignore checkpoints)"
            echo "  --token    GitHub access token   (for private repos)"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Launching TPU Pod training"
echo "  TPU:     $TPU_NAME"
echo "  Zone:    $ZONE"
echo "  Project: $PROJECT"
echo "  Branch:  $BRANCH"
echo "  Config:  $CONFIG"
if [ -n "$TRAIN_ARGS" ]; then
echo "  Args:    $TRAIN_ARGS"
fi
echo "============================================"

# Check TPU status
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

# Build inline bootstrap: clone/update repo first, then run setup script
read -r -d '' REMOTE_CMD <<EOFCMD || true
set -e
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
GH_TOKEN='${GH_TOKEN}'
TRAIN_ARGS='${TRAIN_ARGS}'
export BRANCH CONFIG GH_TOKEN TRAIN_ARGS

# Bootstrap: ensure ~/dawn-spatial exists with the right branch
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

# Run the setup+training script (nohup inside will detach training)
cd ~/dawn-spatial
bash scripts/setup_and_run_tpu_pod.sh
EOFCMD

# Send command to all workers
echo "Sending bootstrap+training command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete. Training is running in tmux session 'train' on all workers."
echo "  Log:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tail -f ~/train.log'"
echo "  Attach:  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tmux attach -t train'"
echo "  Kill:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command='tmux kill-session -t train'"

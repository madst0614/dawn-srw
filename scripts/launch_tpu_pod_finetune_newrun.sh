#!/bin/bash
set -euo pipefail

TPU_NAME="baseline-400M"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG=""
INIT_FROM=""
RESUME_FROM=""
GH_TOKEN=""
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --token) GH_TOKEN="$2"; shift 2 ;;
    --from-scratch) TRAIN_ARGS="$TRAIN_ARGS --from-scratch"; shift ;;
    --load-opt-state) TRAIN_ARGS="$TRAIN_ARGS --load-opt-state"; shift ;;
    --preserve-step) TRAIN_ARGS="$TRAIN_ARGS --preserve-step"; shift ;;
    -h|--help)
      echo "Usage: $0 --tpu NAME --zone ZONE --project PROJECT --branch BRANCH --config CONFIG [--init-from PATH|--resume-from PATH]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: --config required" >&2
  exit 1
fi

if [[ -n "$INIT_FROM" && -n "$RESUME_FROM" ]]; then
  echo "ERROR: use only one of --init-from or --resume-from" >&2
  exit 1
fi

if [[ -n "$GH_TOKEN" ]]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
  REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

TRAIN_SCRIPT="scripts/train_jax_finetune_newrun.py"

printf '%s\n' "============================================"
printf '%s\n' "Launching TPU Pod finetune-newrun training"
printf '%s\n' "  TPU:          $TPU_NAME"
printf '%s\n' "  Zone:         $ZONE"
printf '%s\n' "  Project:      $PROJECT"
printf '%s\n' "  Branch:       $BRANCH"
printf '%s\n' "  Config:       $CONFIG"
printf '%s\n' "  Init from:    ${INIT_FROM:-<none>}"
printf '%s\n' "  Resume from:  ${RESUME_FROM:-<none>}"
printf '%s\n' "  Train script: $TRAIN_SCRIPT"
printf '%s\n' "  Args:         $TRAIN_ARGS"
printf '%s\n' "============================================"

echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail

REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
INIT_FROM='${INIT_FROM}'
RESUME_FROM='${RESUME_FROM}'
GH_TOKEN='${GH_TOKEN}'
TRAIN_ARGS='${TRAIN_ARGS}'
TRAIN_SCRIPT='${TRAIN_SCRIPT}'

export BRANCH CONFIG INIT_FROM RESUME_FROM GH_TOKEN TRAIN_ARGS TRAIN_SCRIPT

if [ -d "\$HOME/dawn-spatial/.git" ]; then
  cd "\$HOME/dawn-spatial"
  git fetch origin "\$BRANCH" --depth 1
  git checkout -B "\$BRANCH" FETCH_HEAD
  git reset --hard FETCH_HEAD
  git clean -fd
else
  rm -rf "\$HOME/dawn-spatial"
  git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$HOME/dawn-spatial"
  cd "\$HOME/dawn-spatial"
fi

if [ ! -f "\$TRAIN_SCRIPT" ]; then
  echo "ERROR: missing train script: \$TRAIN_SCRIPT" >&2
  ls -l scripts >&2 || true
  exit 2
fi

if [ ! -f "scripts/setup_and_run_tpu_pod_finetune_newrun.sh" ]; then
  echo "ERROR: missing setup script" >&2
  ls -l scripts >&2 || true
  exit 2
fi

bash scripts/setup_and_run_tpu_pod_finetune_newrun.sh
EOFCMD

echo "Sending bootstrap+training command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --worker=all \
  --command="$REMOTE_CMD" \
  2>&1 | tee "launch_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete."
echo "  Log:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"

#!/bin/bash
set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
INIT_FROM=""
GH_TOKEN=""
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --config) CONFIGS+=("$2"); shift 2 ;;
    --token) GH_TOKEN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tpu NAME --branch BRANCH --init-from PRETRAIN_RUN_OR_CKPT --config cfg1.yaml [--config cfg2.yaml ...]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TPU_NAME" ]]; then echo "ERROR: --tpu required" >&2; exit 1; fi
if [[ ${#CONFIGS[@]} -eq 0 ]]; then echo "ERROR: at least one --config required" >&2; exit 1; fi

if [[ -n "$GH_TOKEN" ]]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
  REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi
CONFIGS_JOINED=$(IFS='|'; echo "${CONFIGS[*]}")

echo "============================================"
echo "Launching TPU Pod downstream sequence"
echo "  TPU:       $TPU_NAME"
echo "  Zone:      $ZONE"
echo "  Project:   $PROJECT"
echo "  Branch:    $BRANCH"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Configs:   ${CONFIGS[*]}"
echo "============================================"

echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
INIT_FROM='${INIT_FROM}'
CONFIGS='${CONFIGS_JOINED}'
export BRANCH INIT_FROM CONFIGS

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

bash scripts/setup_and_run_downstream_tpu_pod.sh
EOFCMD

echo "Sending downstream command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --worker=all \
  --command="$REMOTE_CMD" \
  2>&1 | tee "launch_downstream_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete."
echo "  Log:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"

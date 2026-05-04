#!/bin/bash
set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
GH_TOKEN_ARGS=()
TASKS=(sst2 rte wic boolq mnli)
MODELS=(baseline dawn_v394)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --token) GH_TOKEN_ARGS=(--token "$2"); shift 2 ;;
    --tasks)
      IFS=',' read -r -a TASKS <<< "$2"
      shift 2 ;;
    --models)
      IFS=',' read -r -a MODELS <<< "$2"
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tpu TPU_NAME --branch BRANCH [--tasks sst2,rte,wic,boolq,mnli] [--models baseline,dawn_v394]"
      echo "Runs random-init downstream controls. No --init-from is passed."
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TPU_NAME" ]]; then
  echo "ERROR: --tpu required" >&2
  exit 1
fi

CONFIG_ARGS=()
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    CFG="configs/downstream/random_init/${MODEL}/${TASK}.yaml"
    CONFIG_ARGS+=(--config "$CFG")
  done
done

echo "============================================================"
echo "Launching RANDOM-INIT downstream controls"
echo "  TPU:     $TPU_NAME"
echo "  Branch:  $BRANCH"
echo "  Models:  ${MODELS[*]}"
echo "  Tasks:   ${TASKS[*]}"
echo "  Init:    <none>  (random init)"
echo "  Root:    gs://dawn-tpu-data-c4/downstream_runs/random_init/"
echo "============================================================"

bash scripts/launch_downstream_tpu_pod.sh \
  --tpu "$TPU_NAME" \
  --zone "$ZONE" \
  --project "$PROJECT" \
  --branch "$BRANCH" \
  "${GH_TOKEN_ARGS[@]}" \
  "${CONFIG_ARGS[@]}"

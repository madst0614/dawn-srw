#!/usr/bin/env bash
set -euo pipefail

CURRENT_STAGE="setup"
trap 'code=$?; echo; echo "ERROR: stage failed: ${CURRENT_STAGE} (exit ${code})" >&2' ERR

usage() {
  cat <<'EOF'
Usage:
  bash scripts/analysis/run_paper_data.sh \
    TF_CONFIG TF_CHECKPOINT DAWN_CONFIG DAWN_CHECKPOINT VAL_DATA OUT_ROOT

Environment knobs:
  PYTHON_BIN=python3
  TOKENIZER=bert-base-uncased
  MAX_VAL_TOKENS=10000000
  VAL_BATCH_SIZE=16
  EVAL_SEQ_LEN=0                 # 0 = config max_seq_len
  MAX_BATCHES=0                  # 0 = all batches under MAX_VAL_TOKENS
  EVAL_MESH_DATA=                # optional override
  EVAL_MESH_MODEL=               # optional override
  PAPER_SINGLE_DEVICE=0          # 1 = force single-device path

  ACTIVE_MAX_VAL_TOKENS=65536
  ACTIVE_BATCH_SIZE=1
  ACTIVE_SEQ_LEN=128
  ACTIVE_MAX_BATCHES=0
  DENSE_D_FF=                    # optional dense FFN width for active-ratio denominator
  BASIC_MAX_BATCHES=100          # dawn_srw_basic_analysis does not treat 0 as all

  RUN_TF_EVAL=1
  RUN_DAWN_EVAL=1
  RUN_DAWN_ACTIVE=1
  RUN_DAWN_BASIC=1
  RUN_DECISION_DIAGNOSTICS=1
  RUN_AMBIGUITY=1
  RUN_INTERVENTION=1
  RUN_AGGREGATE=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 6 ]]; then
  usage >&2
  exit 1
fi

TF_CONFIG="$1"
TF_CKPT="$2"
DAWN_CONFIG="$3"
DAWN_CKPT="$4"
VAL_DATA="$5"
OUT_ROOT="$6"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOKENIZER="${TOKENIZER:-bert-base-uncased}"
MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-10000000}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
EVAL_MESH_DATA="${EVAL_MESH_DATA:-}"
EVAL_MESH_MODEL="${EVAL_MESH_MODEL:-}"
PAPER_SINGLE_DEVICE="${PAPER_SINGLE_DEVICE:-0}"

ACTIVE_MAX_VAL_TOKENS="${ACTIVE_MAX_VAL_TOKENS:-65536}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-1}"
ACTIVE_SEQ_LEN="${ACTIVE_SEQ_LEN:-128}"
ACTIVE_MAX_BATCHES="${ACTIVE_MAX_BATCHES:-0}"
ACTIVE_EPS="${ACTIVE_EPS:-1e-8}"
DENSE_D_FF="${DENSE_D_FF:-}"

RUN_TF_EVAL="${RUN_TF_EVAL:-1}"
RUN_DAWN_EVAL="${RUN_DAWN_EVAL:-1}"
RUN_DAWN_ACTIVE="${RUN_DAWN_ACTIVE:-1}"
RUN_DAWN_BASIC="${RUN_DAWN_BASIC:-1}"
RUN_DECISION_DIAGNOSTICS="${RUN_DECISION_DIAGNOSTICS:-1}"
RUN_AMBIGUITY="${RUN_AMBIGUITY:-1}"
RUN_INTERVENTION="${RUN_INTERVENTION:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"

DAWN_OUT="${OUT_ROOT%/}/dawn_srw"
TF_OUT="${OUT_ROOT%/}/tf_baseline"
MECH_OUT="${OUT_ROOT%/}/mechanism"
TABLE_OUT="${OUT_ROOT%/}/paper_tables"

is_gcs() {
  [[ "$1" == gs://* ]]
}

mkdir_if_local() {
  if ! is_gcs "$1"; then
    mkdir -p "$1"
  fi
}

stage() {
  CURRENT_STAGE="$1"
  echo
  echo "=== ${CURRENT_STAGE} ==="
}

detect_primary_host() {
  "$PYTHON_BIN" - <<'PY'
try:
    import jax
    print("1" if int(jax.process_index()) == 0 else "0")
except Exception:
    print("1")
PY
}

PRIMARY_HOST="$(detect_primary_host)"

DEVICE_ARGS=()
if [[ "$PAPER_SINGLE_DEVICE" == "1" ]]; then
  DEVICE_ARGS+=(--single-device)
fi
if [[ -n "$EVAL_MESH_DATA" ]]; then
  DEVICE_ARGS+=(--mesh-data "$EVAL_MESH_DATA")
fi
if [[ -n "$EVAL_MESH_MODEL" ]]; then
  DEVICE_ARGS+=(--mesh-model "$EVAL_MESH_MODEL")
fi

ACTIVE_DEVICE_ARGS=()
if [[ "$PAPER_SINGLE_DEVICE" == "1" ]]; then
  ACTIVE_DEVICE_ARGS+=(--single-device)
fi

BASIC_BATCH_ARGS=()
if [[ "${BASIC_MAX_BATCHES:-100}" != "0" ]]; then
  BASIC_BATCH_ARGS+=(--max-batches "${BASIC_MAX_BATCHES:-100}")
fi

mkdir_if_local "$TF_OUT"
mkdir_if_local "$DAWN_OUT"
mkdir_if_local "$MECH_OUT"
mkdir_if_local "$TABLE_OUT"
mkdir -p /tmp/dawn_paper_data

stage "Paper data generation manifest"
echo "primary_host: ${PRIMARY_HOST}"
echo "tf_config:    ${TF_CONFIG}"
echo "tf_ckpt:      ${TF_CKPT}"
echo "dawn_config:  ${DAWN_CONFIG}"
echo "dawn_ckpt:    ${DAWN_CKPT}"
echo "val_data:     ${VAL_DATA}"
echo "out_root:     ${OUT_ROOT}"
echo "batch_size:   ${VAL_BATCH_SIZE}"
echo "max_tokens:   ${MAX_VAL_TOKENS}"
echo "single_device:${PAPER_SINGLE_DEVICE}"

if [[ "$RUN_TF_EVAL" == "1" ]]; then
  stage "1. TF baseline validation on fixed valset"
  "$PYTHON_BIN" scripts/analysis/paper_eval_jax.py \
    --config "$TF_CONFIG" \
    --checkpoint "$TF_CKPT" \
    --val-data "$VAL_DATA" \
    --output "${TF_OUT}/eval" \
    --model-name "tf_baseline" \
    --max-val-tokens "$MAX_VAL_TOKENS" \
    --batch-size "$VAL_BATCH_SIZE" \
    --seq-len "$EVAL_SEQ_LEN" \
    --max-batches "$MAX_BATCHES" \
    "${DEVICE_ARGS[@]}"
else
  stage "1. TF baseline validation skipped"
fi

if [[ "$RUN_DAWN_EVAL" == "1" ]]; then
  stage "2. DAWN-SRW validation on same fixed valset"
  "$PYTHON_BIN" scripts/analysis/paper_eval_jax.py \
    --config "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --val-data "$VAL_DATA" \
    --output "${DAWN_OUT}/eval" \
    --model-name "dawn_srw" \
    --max-val-tokens "$MAX_VAL_TOKENS" \
    --batch-size "$VAL_BATCH_SIZE" \
    --seq-len "$EVAL_SEQ_LEN" \
    --max-batches "$MAX_BATCHES" \
    "${DEVICE_ARGS[@]}"
else
  stage "2. DAWN-SRW validation skipped"
fi

if [[ "$RUN_DAWN_ACTIVE" == "1" ]]; then
  stage "3. DAWN-SRW validation-set active/compute analysis"
  ACTIVE_ARGS=(
    --config "$DAWN_CONFIG"
    --checkpoint "$DAWN_CKPT"
    --val-data "$VAL_DATA"
    --output "${DAWN_OUT}/val_active"
    --max-val-tokens "$ACTIVE_MAX_VAL_TOKENS"
    --batch-size "$ACTIVE_BATCH_SIZE"
    --seq-len "$ACTIVE_SEQ_LEN"
    --max-batches "$ACTIVE_MAX_BATCHES"
    --active-eps "$ACTIVE_EPS"
  )
  if [[ -n "$DENSE_D_FF" ]]; then
    ACTIVE_ARGS+=(--dense-d-ff "$DENSE_D_FF")
  fi
  "$PYTHON_BIN" scripts/analysis/dawn_srw/dawn_srw_val_active_analysis.py \
    "${ACTIVE_ARGS[@]}" \
    "${ACTIVE_DEVICE_ARGS[@]}"
else
  stage "3. DAWN-SRW active/compute analysis skipped"
fi

if [[ "$PRIMARY_HOST" != "1" ]]; then
  stage "Host-local mechanism stages skipped on non-primary host"
  echo "Non-primary host completed multi-device stages."
  exit 0
fi

if [[ "$RUN_DAWN_BASIC" == "1" ]]; then
  stage "4. DAWN-SRW basic validation / health / weights / samples"
  "$PYTHON_BIN" scripts/analysis/dawn_srw/dawn_srw_basic_analysis.py \
    --config "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --val-data "$VAL_DATA" \
    --output "${DAWN_OUT}/basic" \
    --only all \
    --max-val-tokens "$MAX_VAL_TOKENS" \
    --batch-size "$VAL_BATCH_SIZE" \
    --tokenizer "$TOKENIZER" \
    "${BASIC_BATCH_ARGS[@]}"
else
  stage "4. DAWN-SRW basic analysis skipped"
fi

if [[ "$RUN_DECISION_DIAGNOSTICS" == "1" ]]; then
  stage "5. DAWN-SRW prompt decision diagnostics"
  "$PYTHON_BIN" scripts/analysis/dawn_srw/dawn_srw_decision_diagnostics.py \
    --config "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --output "${DAWN_OUT}/decision_diagnostics" \
    --tokenizer "$TOKENIZER" \
    --max-length 128
else
  stage "5. DAWN-SRW prompt decision diagnostics skipped"
fi

if [[ "$RUN_AMBIGUITY" == "1" ]]; then
  stage "6. DAWN-SRW ambiguity / context-dependent operator selection"
  "$PYTHON_BIN" scripts/analysis/dawn_srw/dawn_srw_ambiguity_experiment.py \
    --config "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --output "${MECH_OUT}/ambiguity" \
    --tokenizer "$TOKENIZER" \
    --layers all \
    --top-k 50 \
    --sort-by abs_projection \
    --mass-metric abs_projection \
    --mass-threshold 0.8
else
  stage "6. DAWN-SRW ambiguity experiment skipped"
fi

if [[ "$RUN_INTERVENTION" == "1" ]]; then
  stage "7. DAWN-SRW causal RST intervention"
  "$PYTHON_BIN" scripts/analysis/dawn_srw/dawn_srw_rst_intervention_experiment.py \
    --config "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --output "${MECH_OUT}/rst_intervention" \
    --tokenizer "$TOKENIZER" \
    --layers all \
    --top-k 20 \
    --sort-by abs_projection \
    --random-repeats 5 \
    --ablation-mode both
else
  stage "7. DAWN-SRW causal RST intervention skipped"
fi

if [[ "$RUN_AGGREGATE" == "1" ]]; then
  stage "8. Aggregate copyable paper tables"
  "$PYTHON_BIN" scripts/analysis/aggregate_paper_data.py \
    --tf-dir "${TF_OUT}/eval" \
    --dawn-dir "${DAWN_OUT}/eval" \
    --active-json "${DAWN_OUT}/val_active/dawn_active_compute.json" \
    --output "$TABLE_OUT"
else
  stage "8. Aggregation skipped"
fi

stage "Done"
echo "model table:              ${TABLE_OUT}/model_table.csv"
echo "validation table:         ${TABLE_OUT}/validation_table.csv"
echo "loss/active compute table:${TABLE_OUT}/loss_vs_active_compute_table.csv"
echo "DAWN active table:        ${DAWN_OUT}/val_active/dawn_active_compute_table.csv"
echo "ambiguity output:         ${MECH_OUT}/ambiguity"
echo "intervention output:      ${MECH_OUT}/rst_intervention"

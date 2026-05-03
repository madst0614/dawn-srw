#!/usr/bin/env bash
set -euo pipefail

# Run prompt-style downstream transfer for DAWN and Transformer.
# Edit these paths before running.
TOKENIZER=${TOKENIZER:-/path/to/tokenizer}
DAWN_MODULE=${DAWN_MODULE:-/path/to/dawn_spatial_v394_exp.py}
TRANSFORMER_MODULE=${TRANSFORMER_MODULE:-/path/to/baseline_transformer_jax.py}
DAWN_CONFIG=${DAWN_CONFIG:-/path/to/dawn_model_config.json}
TRANSFORMER_CONFIG=${TRANSFORMER_CONFIG:-/path/to/transformer_model_config.json}
DAWN_CKPT=${DAWN_CKPT:-/path/to/dawn_pretrain_ckpt}
TRANSFORMER_CKPT=${TRANSFORMER_CKPT:-/path/to/transformer_pretrain_ckpt}
OUT_ROOT=${OUT_ROOT:-runs/downstream}

TASKS=(sst2 rte wic boolq qqp mnli)

for TASK in "${TASKS[@]}"; do
  python downstream_lm_transfer_jax.py \
    --model_type dawn \
    --model_module "$DAWN_MODULE" \
    --config_json "$DAWN_CONFIG" \
    --checkpoint "$DAWN_CKPT" \
    --tokenizer "$TOKENIZER" \
    --task "$TASK" \
    --output_dir "$OUT_ROOT/dawn/$TASK" \
    --num_epochs 3 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --eval_every_steps 500

  python downstream_lm_transfer_jax.py \
    --model_type transformer \
    --model_module "$TRANSFORMER_MODULE" \
    --config_json "$TRANSFORMER_CONFIG" \
    --checkpoint "$TRANSFORMER_CKPT" \
    --tokenizer "$TOKENIZER" \
    --task "$TASK" \
    --output_dir "$OUT_ROOT/transformer/$TASK" \
    --num_epochs 3 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --eval_every_steps 500

done

# DAWN / Transformer downstream transfer

This package evaluates downstream transfer with causal-LM prompt fine-tuning.
It does not add a classification head. Each task is converted into a text prompt plus a short answer candidate such as ` yes` / ` no`, and accuracy is computed by candidate answer log-likelihood.

## Required comparisons

For a clean paper table, run at least:

1. DAWN pretrained -> fine-tune
2. DAWN random-init -> fine-tune, by omitting `--checkpoint`
3. Transformer pretrained -> fine-tune
4. Transformer random-init -> fine-tune, by omitting `--checkpoint`

The most important transfer signal is:

```text
pretrained DAWN fine-tune  >>  random-init DAWN fine-tune
```

The architecture comparison is:

```text
pretrained DAWN fine-tune  vs  pretrained Transformer fine-tune
```

## Recommended tasks

Main paper subset:

```text
sst2, rte, wic, boolq, qqp, mnli
```

For quick debugging, run:

```bash
python downstream_lm_transfer_jax.py \
  --model_type transformer \
  --model_module ./baseline_transformer_jax.py \
  --config_json ./transformer_model_config.json \
  --tokenizer ./tokenizer \
  --task sst2 \
  --output_dir runs/debug_transformer_sst2 \
  --max_train_samples 512 \
  --max_eval_samples 256 \
  --num_epochs 1 \
  --batch_size 4
```

For DAWN:

```bash
python downstream_lm_transfer_jax.py \
  --model_type dawn \
  --model_module ./dawn_spatial_v394_exp.py \
  --config_json ./dawn_model_config.json \
  --checkpoint /path/to/dawn_pretrain_ckpt \
  --tokenizer ./tokenizer \
  --task rte \
  --output_dir runs/dawn_rte \
  --num_epochs 3 \
  --batch_size 8
```

## Notes

- `--aux_weight` defaults to `0.0`. For downstream transfer, this is usually better because you are measuring task adaptation, not continuing architectural regularization.
- Evaluation writes `eval_step_*.json`, `final_eval.json`, and `best_eval.json`.
- If your checkpoint loader fails, convert your pretraining checkpoint into a Flax `{ "params": params }` checkpoint or pickle with a top-level `params` key.

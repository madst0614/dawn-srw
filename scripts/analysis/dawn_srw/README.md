# DAWN-SRW analysis and mechanism experiments

These scripts target the official module:

```python
models.dawn_srw
```

They use the new terminology:

- `attn_qk`, `attn_v`, `rst`
- `SRW neuron`
- `RW operator`
- `model decision`
- `Attention Layer`
- `RST Layer`
- `raw_scan_offset` / `scan_offset`

## Files

```text
dawn_srw_common.py
  Shared config/checkpoint/model/tokenizer utilities.

 dawn_srw_basic_analysis.py
  Basic pre-experiment sanity: params, validation PPL/loss/acc, neuron health,
  signature embedding analysis, generation samples.

 dawn_srw_decision_diagnostics.py
  Layer-wise aggregate model-decision diagnostics: active count, entropy,
  effective selected count, top-1 gate fraction, Attention/RST output norms.

 dawn_srw_decision_probe.py
  Token/layer-level RST model-decision trace: top selected SRW neurons and
  RW-operator contribution statistics.

 dawn_srw_ambiguity_experiment.py
  Compares the same ambiguous token across contexts, e.g. bank/charge/light.

 dawn_srw_intervention.py
  Suppresses top selected RST neurons and measures causal effect on loss/logits.
```

## Suggested order

### 1. Basic sanity

```bash
python scripts/analysis/dawn_srw/dawn_srw_basic_analysis.py \
  --config configs/train_config_dawn_srw.yaml \
  --checkpoint checkpoints/run_xxx \
  --val-data data/val.bin \
  --output results/dawn_srw_basic \
  --only all
```

Use `--only quick` to skip validation and weight analysis.

### 2. Aggregate decision diagnostics

```bash
python scripts/analysis/dawn_srw/dawn_srw_decision_diagnostics.py \
  --config configs/train_config_dawn_srw.yaml \
  --checkpoint checkpoints/run_xxx \
  --output results/dawn_srw_decision_diagnostics
```

### 3. Per-token RST model-decision trace

```bash
python scripts/analysis/dawn_srw/dawn_srw_decision_probe.py \
  --config configs/train_config_dawn_srw.yaml \
  --checkpoint checkpoints/run_xxx \
  --prompt "I deposited money in the bank" \
  --target-token bank \
  --layer 5 \
  --top-k 20 \
  --output results/dawn_srw_decision_probe
```

Use `--layer all` to scan all layers.

### 4. Ambiguous token comparison

```bash
python scripts/analysis/dawn_srw/dawn_srw_ambiguity_experiment.py \
  --config configs/train_config_dawn_srw.yaml \
  --checkpoint checkpoints/run_xxx \
  --top-k 20 \
  --layers all \
  --output results/dawn_srw_ambiguity
```

Custom case format:

```bash
--case "bank|||I deposited money in the bank|||He sat by the river bank"
```

### 5. Intervention

```bash
python scripts/analysis/dawn_srw/dawn_srw_intervention.py \
  --config configs/train_config_dawn_srw.yaml \
  --checkpoint checkpoints/run_xxx \
  --prompt "I deposited money in the bank and withdrew cash" \
  --target-token bank \
  --layer 5 \
  --top-k 20 \
  --random-baseline \
  --output results/dawn_srw_intervention
```

## Install location

Recommended repo path:

```text
scripts/analysis/dawn_srw/
```

Copy all `.py` files into that folder. Run from the repository root.

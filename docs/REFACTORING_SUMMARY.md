# DAWN Analysis Refactoring Summary

Branch: `claude/unify-weight-threshold-TQtVt`

## Overview

This refactoring unifies the neuron analysis system to use **physical neuron pools** with **consistent naming conventions** across all analysis outputs.

---

## 1. Physical Neuron Pool System

### Before (Logical Pools - 1104 neurons)
```
fqk_q (64) + fqk_k (64) = 128  ← Q/K counted separately
fv (264)
rqk_q (64) + rqk_k (64) = 128
rv (264)
fknow (160)
rknow (160)
Total: 1104
```

### After (Physical Pools - 976 neurons)
```
fqk (64)   ← Q/K shared, counted once
fv (264)
rqk (64)   ← Q/K shared, counted once
rv (264)
fknow (160)
rknow (160)
Total: 976
```

### Q/K Shared Pool Handling
```python
# Mask: OR combine (active if Q OR K is active)
mask = mask_q | mask_k

# Weight: MAX combine (use larger weight)
weight = torch.max(weight_q, weight_k)
```

---

## 2. Unified Naming Convention

All neuron references now use `{pool}_{local_idx}` format:

| Pool | Range | Examples |
|------|-------|----------|
| fqk | 0-63 | `fqk_0`, `fqk_63` |
| fv | 0-263 | `fv_0`, `fv_263` |
| rqk | 0-63 | `rqk_0`, `rqk_63` |
| rv | 0-263 | `rv_0`, `rv_263` |
| fknow | 0-159 | `fknow_0`, `fknow_159` |
| rknow | 0-159 | `rknow_0`, `rknow_159` |

### Helper Methods Added
```python
def _get_neuron_name(self, global_idx: int) -> str:
    """Convert global index to unified name: fv_45, fknow_12, etc."""

def _get_pool_for_neuron(self, global_idx: int) -> str:
    """Get pool name for a global neuron index."""

def _get_pool_ranges(self) -> Dict[str, Tuple[int, int]]:
    """Get (start, end) index ranges for each pool."""
```

---

## 3. Removed `--pool_type` Parameter

### Before
```bash
python -m scripts.analysis.analyze_all --checkpoint model.pt --pool_type fv
```

### After
```bash
python -m scripts.analysis.analyze_all --checkpoint model.pt
# Automatically analyzes all pools
```

### Affected Functions
| Function | Before | After |
|----------|--------|-------|
| `analyze_pos()` | Single pool via `pool_type` | Legacy (fv only), use `analyze_neuron_features()` |
| `analyze_factual()` | `pool_type='all'` default | Always all pools: fv, rv, fknow, rknow |
| `analyze_neuron_features()` | All pools | All pools (unchanged) |
| `generate_figure5()` | Single pool via config | All pools: fv, rv, fknow, rknow |

---

## 4. Forward-Pass Based Utilization (Table 2)

Replaced EMA-based utilization with forward-pass counting.

### Before (EMA-based)
```python
# Used exponential moving average from training
ema_distribution = health.get('ema_distribution', {})
```

### After (Forward-pass based)
```python
# Count actual activations during inference
def _compute_utilization_stats(self, activation_counts, total_tokens):
    active = (activation_counts > 0).sum()
    dead = (activation_counts == 0).sum()
    active_ratio = active / total
    gini = self._compute_gini(activation_counts)
```

---

## 5. Unified `paper_data.json` Structure

Consolidated `paper_results.json` and `training_comparison.json` into unified format.

### New Structure
```json
{
  "metadata": {
    "generated": "2024-01-22T...",
    "model_name": "dawn_24m",
    "checkpoint_path": "..."
  },
  "models": {
    "dawn": {
      "parameters_M": 24.5,
      "d_model": 512,
      "neuron_pools": {"fqk": 64, "fv": 264, ...}
    },
    "vanilla": {...}
  },
  "training": {
    "dawn": {"dataset": "C4", "batch_size": 64, ...},
    "vanilla": {...},
    "validation": {"dataset": "C4", "n_batches": 200}
  },
  "figures": {
    "fig3_fqk_specialization": {...},
    "fig4_attention_knowledge_balance": {...},
    "fig5_rqk_specialization": {...},
    "fig6_convergence_comparison": {...},
    "fig7_pos_selectivity_heatmap": {...},
    "fig8_knowledge_neurons": {...}
  },
  "tables": {
    "table1_model_stats": {...},
    "table2_neuron_util": {...}
  },
  "appendix": {
    "diversity": {...},
    "probing": {...}
  }
}
```

---

## 6. Improved Config Parsing

Added robust fallback chains for checkpoint config extraction:

| Field | Fallback Keys |
|-------|---------------|
| `vocab_size` | vocab_size → n_vocab → num_embeddings → infer from embedding |
| `d_model` | d_model → hidden_size → n_embd → dim → infer from embedding |
| `d_ff` | d_ff → intermediate_size → ffn_dim → mlp_dim → infer from MLP |
| `n_layers` | n_layers → num_hidden_layers → num_layers → infer from state dict |
| `total_steps` | total_steps → max_steps → num_training_steps → num_steps |
| `batch_size` | batch_size → train_batch_size → per_device_train_batch_size |
| `learning_rate` | learning_rate → lr → peak_lr |
| `warmup_steps` | warmup_steps → num_warmup_steps |

---

## 7. Fig 4: Selectivity Heatmap

### Before (80% Threshold)
```
Neuron 45: NOUN (82.3%)  ← Binary: specialized or not
```

### After (Selectivity Score)
```
fv_45: NOUN selectivity=2.34  ← Continuous: P(active|POS) / P(active)
```

### Per-Pool Analysis
```python
selectivity = {
    'top_selective_per_pos': {
        'NOUN': [
            {'neuron': 'fv_45', 'selectivity': 2.34, 'pool': 'fv'},
            {'neuron': 'fknow_12', 'selectivity': 2.12, 'pool': 'fknow'},
        ],
        'VERB': [...],
    },
    'mean_selectivity_by_pos': {'NOUN': 1.45, 'VERB': 1.32, ...},
}
```

---

## 8. Fig 5: Multi-Pool Factual Analysis (4x Faster)

### Before (Inefficient - 4 separate loops)
```python
# Single pool per call - runs 4 times for 4 pools
for pool in ['fv', 'rv', 'fknow', 'rknow']:
    pool_results = analyzer.analyze_factual_neurons(prompts, targets, pool_type=pool)
```

### After (Efficient - Single pass)
```python
# All pools in ONE generation loop
factual_data = analyzer.analyze_factual_neurons(
    prompts, targets,
    pools=['fv', 'rv', 'fknow', 'rknow']  # Extracts all simultaneously
)
```

### Key Optimization
```python
# In each forward pass, extract neurons from ALL pools at once
step_neurons_per_pool = {pool: set() for pool in pools}
if routing:
    for layer in routing:
        for pool in pools:
            m = layer.get_mask(pool)
            active = m.nonzero().cpu().tolist()
            step_neurons_per_pool[pool].update(active)
```

### Output Format
```json
{
  "pools_analyzed": ["fv", "rv", "fknow", "rknow"],
  "per_pool": {
    "fv": {"n_common_100": 5, "top_neurons": ["fv_45", "fv_123"]},
    "fknow": {"n_common_100": 8, "top_neurons": ["fknow_12", "fknow_89"]}
  },
  "per_target": {
    "Paris": {
      "fv": {"common_100": ["fv_45"], "match_rate": 0.85},
      "fknow": {"common_100": ["fknow_12", "fknow_89"], "match_rate": 0.92}
    }
  }
}
```

---

## 9. Routing Analysis Single-Pass (10-15x Faster)

### Before (Inefficient - 13+ separate loops)
```python
# Each analysis method independently iterates through dataloader
results = {
    'entropy': self.analyze_entropy(dataloader, n_batches),           # Loop 1
    'selection_frequency': self.analyze_selection_frequency(...),      # Loop 2
    'selection_diversity': self.analyze_selection_diversity(...),      # Loop 3
    'qk_overlap': self.analyze_qk_overlap(...),                        # Loop 4
    'qk_usage': self.analyze_qk_usage(...),                            # Loop 5
    'qk_entropy': self.analyze_qk_entropy(...),                        # Loop 6
    'qk_union_coverage': self.analyze_qk_union_coverage(...),          # Loop 7
    'activation_sparsity': self.analyze_activation_sparsity(...),      # Loop 8
    'token_coselection': self.analyze_token_coselection(...),          # Loop 9
    'weight_concentration': self.analyze_weight_concentration(...),    # Loop 10
    'path_usage': self.analyze_path_usage(...),                        # Loop 11
    'coverage_progression': self.analyze_coverage_progression(...),    # Loop 12 (calls diversity again!)
    'layer_contribution': self.analyze_layer_contribution(...),        # Loop 13
}
# Total: ~1500 forward passes (with 2x multipliers)
```

### After (Efficient - Single pass)
```python
# ALL analyses computed in ONE dataloader iteration
def run_all(self, dataloader, output_dir, n_batches):
    # Initialize all accumulators
    entropy_data = {...}
    selection_tensors = {...}
    qk_layer_data = {...}
    # ... etc

    # Single loop through dataloader
    for batch in dataloader:
        routing = self.extractor.extract(model(batch))

        # Update ALL accumulators from same routing data
        for layer in routing:
            # Entropy
            for key in ROUTING_KEYS:
                weights = layer.get_weight(key)
                entropy_data[key].append(calc_entropy_ratio(weights))

            # Selection frequency, sparsity, concentration...
            # Q/K usage, entropy, overlap, co-selection...
            # Layer contribution...

    # Finalize all results
    return {
        'entropy': self._finalize_entropy(entropy_data),
        'selection_frequency': self._finalize_selection_frequency(...),
        # ... etc
    }
```

### Key Optimization
- ~100 forward passes instead of ~1500 (10-15x speedup)
- All routing data extracted once per batch
- Tensor accumulators for GPU-efficient computation
- Results identical to separate analysis calls

---

## 10. TokenCombinationAnalyzer Batch Processing (5-10x Faster)

### Before (Sentence-by-sentence, batch_size=1)
```python
for i in tqdm(range(n_sentences), desc="Processing"):
    example = dataset[i]
    self.analyze_sentence(example['tokens'], example['upos'], ...)
    # Each analyze_sentence() creates batch_size=1 tensor:
    # input_ids = torch.tensor([token_ids], device=self.device)
```

### After (Batched processing)
```python
def analyze_dataset(self, dataset, max_sentences, batch_size=16):
    # Pre-process all sentences
    preprocessed = [{'token_ids': ..., 'pos_tags': ...} for ...]

    # Process in batches
    for batch_idx in tqdm(range(n_batches)):
        batch = preprocessed[start:end]
        token_masks = self._extract_batch_token_masks(batch, ...)
```

### Key Optimization
```python
def _extract_batch_token_masks(self, batch, store_layer_masks):
    # Pad sequences to same length
    max_len = max(len(item['token_ids']) for item in batch)
    padded_input = torch.full((batch_size, max_len), pad_id, device=self.device)

    # Single forward pass for entire batch
    outputs = self.model(padded_input, return_routing_info=True)

    # Extract masks/weights for all batch items simultaneously
    batch_masks = self._extract_batch_layer_mask(layer, batch_size, max_len)
```

### Benefits
- 16 sentences processed per forward pass instead of 1
- GPU utilization improved significantly
- 5-10x speedup for POS neuron analysis

---

## Commit History

| Commit | Description |
|--------|-------------|
| `TBD` | **perf: Batched TokenCombinationAnalyzer (5-10x faster)** |
| `TBD` | **perf: Single-pass routing analysis (10-15x faster)** |
| `3c9e447` | **perf: Analyze all pools in single forward pass (4x faster)** |
| `c2cb87f` | docs: Add refactoring summary |
| `233c689` | Remove pool_type from paper_figures.py |
| `157e4c6` | Remove --pool_type parameter from CLI |
| `000cdde` | Unified paper_data.json and improved config parsing |
| `52b0529` | Extract validation dataset name from path |
| `45f6373` | Unified pool naming and forward-pass analysis |
| `356b251` | Use physical neuron count for Q/K shared pools |
| `11c6d0a` | Add pool_order parameter to NeuronFeatureAnalyzer |
| `b2080bc` | Add selectivity heatmap for Fig 4 |
| `749fb72` | Add selective figure/table execution (--only flag) |
| `b4849e2` | Fix Fig 6 model mismatch |
| `b84138e` | Add sensitivity analysis for Fig 3 & 4 |
| `75da375` | Unify weight threshold to > 0 |

---

## Files Modified

- `scripts/analysis/analyze_all.py` - Main analysis orchestration
- `scripts/analysis/pos_neuron.py` - POS neuron analysis with unified naming
- `scripts/analysis/paper_figures.py` - Paper figure generation
- `scripts/analysis/routing.py` - Single-pass routing analysis optimization
- `scripts/analysis/behavioral.py` - Multi-pool factual analysis optimization

---

## Usage Examples

### Run Full Analysis (All Pools)
```bash
python -m scripts.analysis.analyze_all \
    --checkpoint checkpoints/dawn_24m/best.pt \
    --val_data data/val/c4_val.pt \
    --output analysis_results/ \
    --paper-only
```

### Run Specific Figures
```bash
python -m scripts.analysis.analyze_all \
    --checkpoint checkpoints/dawn_24m/best.pt \
    --val_data data/val/c4_val.pt \
    --output analysis_results/ \
    --only fig4,fig5,table2
```

### Generate Paper Figures Only
```bash
python -m scripts.analysis.paper_figures \
    --checkpoint checkpoints/dawn_24m/best.pt \
    --output paper_figures/ \
    --figures 3,4,5,6,7
```

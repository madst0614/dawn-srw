# Legacy DAWN Models

Model files here are **not** registered in `scripts/train_jax.py`'s
`MODEL_REGISTRY`. They are kept for:

- resuming older checkpoints
- reproducing results from the DAWN-SRW paper / rebuttal
- bisecting regressions against pre-refactor behavior
- comparison experiments

Never delete from this folder. If a file is confirmed obsolete the
right move is a PR that explains why and removes it with full history
context, not an ad-hoc `rm`.

## Restore procedure

1. Move the needed file back to `models/`:
   ```bash
   git mv models/legacy/dawn_spatial_vXXX_exp.py models/
   ```

2. Re-import and register in `scripts/train_jax.py` next to the
   other `MODEL_REGISTRY` entries:
   ```python
   from models.dawn_spatial_vXXX_exp import DAWN as DAWN_VXXX
   # ...
   'spatial-r1-vX.X.X': ModelSpec(
       name='spatial-r1-vX.X.X',
       module_path='models.dawn_spatial_vXXX_exp',
       cls=DAWN_VXXX,
       build_kwargs=_dawn_shared_kwargs,  # or a version-specific builder
       supports_sharded=True,             # if the module defines make_sharded_srw
       force_sharded=False,               # True if no non-sharded fallback
       sharded_kwargs=None,               # closure constants for make_sharded_srw
   ),
   ```

3. If the model needs kwargs beyond `_dawn_shared_kwargs`, write a
   dedicated builder (see `_baseline_kwargs`, `_v41_sharded_kwargs`
   for templates).

4. Run the short-train parity check described in
   `scripts/legacy/train_jax_pre_refactor_20260422.py`'s companion
   notes before committing to a long run.

## What's in here

File names encode the DAWN version. Brief context:

| File                                   | Version     | Notes |
|----------------------------------------|-------------|-------|
| `dawn_spatial.py`                      | v1 spatial  | Earliest DAWN-spatial prototype |
| `dawn_spatial_v2.py`                   | v2          | 2D positional grid routing |
| `dawn_spatial_v3.py`                   | v3.x        | Default v3 fallback (was `.get(default=...)` target) |
| `dawn_spatial_v3_baseline.py`          | v3.9.1      | |
| `dawn_spatial_v3_exp.py`               | v3.9.3      | |
| `dawn_spatial_v3{95,96,97,971,98,981,99}_exp.py` | v3.9.5 – v3.9.9 | |
| `dawn_spatial_v40{0,1}_exp.py`         | v4.0.0, v4.0.1 | DAWN-SRW paper baseline (v4.0.1) |
| `dawn_spatial_v402_exp.py`             | rw-v4.0.2   | Separate Q/K pools; dead-neuron diagnostics |
| `dawn_spatial_v40{3,4,5,6}_exp.py`     | v4.0.3 – v4.0.6 | Progression toward v4.1 two-stage gate |
| `dawn_spatial_v3_*backup*.py`          | snapshots   | Archived checkpoints (39M / 400M) |
| `model_v17_1.py` / `_tpu*.py`          | v17.1 non-jax | Pre-JAX originals; v17.1-JAX (`models/model_v17_1_jax.py`) kept at repo root for analysis tooling |
| `model_v17_2.py`, `model_v18*.py`      | v17.2 / v18.x | Exploration branches superseded by v4.x |

Fill in individual "Notes" when restoring a file for a specific purpose so
the table stays useful.

## Why dead-neuron diagnostics were removed

The `diagnose_dead_neurons` helper (and its `if model_version.startswith('rw-v'):`
call site in `train_jax.py`) was v4.0.2-specific:

- indexed separate `n_q`, `n_k` pools (v4.1 uses a fused `n_qk`)
- referenced `pool['q_read'] / ['k_read']` and `router['tau_q'] / ['tau_k']` (no v4.1 equivalents)
- built its own forward using a GELU-z gate; v4.1 uses an activation×intensity two-stage gate

Porting it requires rewriting against v4.1's gate math, which was out of
scope for the registry refactor. The original implementation lives in
`dawn_spatial_v402_exp.py` — reference it when writing a v4.1 version.

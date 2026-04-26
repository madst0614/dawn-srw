# Legacy configs

Configs in this folder are archived experiments that are not registered in
`scripts/train_jax.py` by default.

To rerun one, restore the matching model file from `models/legacy/`, then add
the corresponding `ModelSpec` entry back to `MODEL_REGISTRY`.

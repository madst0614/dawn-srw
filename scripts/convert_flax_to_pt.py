"""
Flax → PyTorch checkpoint converter for DAWN v17.1 and Baseline models.

Converts .flax checkpoints (from train_jax.py) to .pt format
compatible with the existing PyTorch analysis pipeline (analyze_all.py).

Usage:
    python scripts/convert_flax_to_pt.py checkpoint.flax -o checkpoint.pt
    python scripts/convert_flax_to_pt.py gs://bucket/checkpoint.flax -o local.pt
    python scripts/convert_flax_to_pt.py checkpoint.flax  # auto: checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch is required for conversion output. pip install torch")
    sys.exit(1)

try:
    import jax
    import flax.serialization as serialization
except ImportError:
    print("ERROR: jax + flax required to read .flax checkpoints. pip install jax flax")
    sys.exit(1)


# ============================================================
# Utilities
# ============================================================

def _is_gcs(path):
    return str(path).startswith('gs://')


def load_flax_checkpoint(path):
    """Load a .flax checkpoint (local or GCS)."""
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(path_str, 'rb') as f:
                data = f.read()
        except ImportError:
            import tensorflow as tf
            data = tf.io.gfile.GFile(path_str, 'rb').read()
    else:
        with open(path_str, 'rb') as f:
            data = f.read()

    # Deserialize without target (raw dict/array structure)
    ckpt = serialization.from_bytes(None, data)
    return ckpt


def flatten_params(d, prefix=''):
    """Flatten nested Flax params dict to dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        else:
            flat[key] = np.array(v)
    return flat


# ============================================================
# DAWN v17.1 mapping
# ============================================================

def convert_dawn_params(flax_params):
    """Convert DAWN v17.1 Flax params to PyTorch state_dict."""
    flat = flatten_params(flax_params)
    state_dict = {}

    for flax_key, value in flat.items():
        pt_key = _dawn_key_map(flax_key)
        if pt_key is None:
            continue

        tensor = torch.from_numpy(np.array(value))

        # Transpose Dense/Linear weights: Flax [in, out] → PyTorch [out, in]
        if pt_key.endswith('.weight') and not any(
            s in pt_key for s in ['_emb.weight', 'norm.weight', '.neuron_emb',
                                   '.f_neurons', '.r_neurons',
                                   '.feature_know', '.restore_know']
        ):
            if tensor.ndim == 2:
                tensor = tensor.T

        state_dict[pt_key] = tensor

    # Weight tying: lm_head.weight = token_emb.weight
    if 'token_emb.weight' in state_dict and 'lm_head.weight' not in state_dict:
        state_dict['lm_head.weight'] = state_dict['token_emb.weight']

    return state_dict


def _dawn_key_map(flax_key):
    """Map a single Flax key to PyTorch key for DAWN v17.1."""
    k = flax_key

    # --- Embeddings ---
    k = k.replace('token_emb.embedding', 'token_emb.weight')
    k = k.replace('pos_emb.embedding', 'pos_emb.weight')

    # --- LayerNorm: scale → weight ---
    k = k.replace('.scale', '.weight')

    # --- Dense: kernel → weight ---
    k = k.replace('.kernel', '.weight')

    # --- Layer blocks: block_N → layers.N ---
    import re
    k = re.sub(r'block_(\d+)\.', r'layers.\1.', k)

    # --- Router path ---
    k = k.replace('router.neuron_router.', 'router.neuron_router.')
    # shared_neurons params are direct (f_neurons, r_neurons, etc.)

    # --- GlobalSSM: JAX doesn't have it, skip any leftover ---
    if 'global_ssm' in k or 'ssm' in k.lower():
        return None

    return k


# ============================================================
# Baseline mapping
# ============================================================

def convert_baseline_params(flax_params):
    """Convert Baseline VanillaTransformer Flax params to PyTorch state_dict."""
    flat = flatten_params(flax_params)
    state_dict = {}

    for flax_key, value in flat.items():
        pt_key = _baseline_key_map(flax_key)
        if pt_key is None:
            continue

        tensor = torch.from_numpy(np.array(value))

        # Transpose Dense/Linear weights: Flax [in, out] → PyTorch [out, in]
        if pt_key.endswith('.weight') and not any(
            s in pt_key for s in ['_emb.weight', 'norm.weight', 'head.weight']
        ):
            if tensor.ndim == 2:
                tensor = tensor.T

        state_dict[pt_key] = tensor

    # Weight tying: head.weight = token_emb.weight
    if 'token_emb.weight' in state_dict:
        state_dict['head.weight'] = state_dict['token_emb.weight']

    return state_dict


def _baseline_key_map(flax_key):
    """Map a single Flax key to PyTorch key for VanillaTransformer."""
    k = flax_key

    # --- Embeddings ---
    k = k.replace('token_emb.embedding', 'token_emb.weight')
    k = k.replace('pos_emb.embedding', 'pos_emb.weight')

    # --- LayerNorm: scale → weight ---
    k = k.replace('.scale', '.weight')

    # --- Dense: kernel → weight ---
    k = k.replace('.kernel', '.weight')

    # --- Layer naming: layer_N → layers.N ---
    import re
    k = re.sub(r'layer_(\d+)\.', r'layers.\1.', k)

    # --- FFN: Flax nn.compact auto-names Dense_0/Dense_1 → PyTorch w_up/w_down ---
    k = k.replace('.ffn.Dense_0.', '.ffn.w_up.')
    k = k.replace('.ffn.Dense_1.', '.ffn.w_down.')

    return k


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Flax checkpoint to PyTorch')
    parser.add_argument('input', type=str, help='Path to .flax checkpoint (local or gs://)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .pt path (default: same name with .pt extension)')
    parser.add_argument('--model', type=str, default=None, choices=['dawn', 'baseline'],
                        help='Model type (auto-detected from checkpoint config if omitted)')
    args = parser.parse_args()

    # Output path
    output = args.output
    if output is None:
        name = Path(args.input).stem if not _is_gcs(args.input) else args.input.rsplit('/', 1)[-1].replace('.flax', '')
        output = f"{name}.pt"

    print(f"Loading Flax checkpoint: {args.input}")
    ckpt = load_flax_checkpoint(args.input)

    # Extract pieces
    params = ckpt['params']
    config = ckpt.get('config', {})
    epoch = ckpt.get('epoch', 0)
    step = ckpt.get('step', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))

    # Auto-detect model type
    model_type = args.model
    if model_type is None:
        version = config.get('model_version', '')
        if version == 'baseline' or version == 'baseline-JAX':
            model_type = 'baseline'
        else:
            model_type = 'dawn'
    print(f"Model type: {model_type}")

    # Convert
    if model_type == 'dawn':
        state_dict = convert_dawn_params(params)
        # Normalize version for PyTorch side
        config['model_version'] = '17.1'
    else:
        state_dict = convert_baseline_params(params)
        config['model_version'] = 'baseline'

    # Build PyTorch checkpoint dict (same format as train.py saves)
    pt_ckpt = {
        'model_state_dict': state_dict,
        'model_config': config,
        'model_version': config.get('model_version'),
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
    }

    # Save
    torch.save(pt_ckpt, output)
    print(f"Saved PyTorch checkpoint: {output}")

    # Summary
    print(f"\n  Parameters converted: {len(state_dict)}")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"  Total param count: {total_params:,}")
    print(f"  Config: {config}")
    print(f"  Epoch: {epoch}, Step: {step}, Best val loss: {best_val_loss}")

    # Verify by loading into PyTorch model
    print(f"\n  Verifying by loading into PyTorch model...")
    try:
        from models import create_model_by_version, normalize_version
        version = normalize_version(config.get('model_version', 'baseline'))
        model = create_model_by_version(version, config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            # Filter out expected missing keys
            real_missing = [k for k in missing if not any(
                s in k for s in ['global_ssm', 'ssm_norm', 'importance_proj',
                                  'context_proj', 'context_scale',
                                  'A_log', 'W_delta', 'W_B', 'W_C']
            )]
            if real_missing:
                print(f"  WARNING - Missing keys: {real_missing}")
            else:
                print(f"  OK - Only GlobalSSM keys missing (expected, JAX has no SSM)")
        else:
            print(f"  OK - All keys loaded successfully")

        if unexpected:
            print(f"  WARNING - Unexpected keys: {unexpected}")

        if not missing and not unexpected:
            print(f"  PERFECT - Exact match!")

    except Exception as e:
        print(f"  Verification skipped: {e}")


if __name__ == '__main__':
    main()

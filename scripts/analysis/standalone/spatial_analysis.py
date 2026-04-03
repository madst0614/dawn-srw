#!/usr/bin/env python3
"""
DAWN-Spatial v3 Analysis Script
================================
Comprehensive analysis for DAWN-Spatial SRW checkpoints (JAX/Flax).
Matches DAWN v17.1 analysis suite adapted for rank-1 SRW architecture.

Analyses:
  D1: Model Info — params, architecture, FLOPs estimation
  D2: Validation — loss, perplexity, accuracy (vectorized)
  D3: Neuron Health — norms, dead neurons, activation stats
  D4: Generation — KV-cache autoregressive generation
  D5: Weight Analysis — cosine similarity, effective rank, SVD
  D6: Routing — gate entropy, neuron usage frequency, load balance
  D7: Generation Samples — multi-prompt with timing, categories

Usage:
    python scripts/analysis/standalone/spatial_analysis.py \\
        --checkpoint gs://.../.../run_XXX \\
        --config configs/train_config_spatial_r1_v3_40M_c4_5B.yaml \\
        --output results/spatial_analysis

    # Quick generation only:
    python scripts/analysis/standalone/spatial_analysis.py \\
        --checkpoint gs://...  --config configs/... \\
        --only generate --prompt "The meaning of life is"
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
import math
import numpy as np

import jax
import jax.numpy as jnp
import flax.serialization as serialization
import yaml


# ============================================================
# GCS helpers
# ============================================================

def _is_gcs(path):
    return str(path).startswith("gs://")

def _open_file(path, mode="rb"):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().open(path_str, mode)
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
    else:
        p = Path(path_str)
        if "w" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)

def _list_dir(path):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            return [f"gs://{f}" for f in gcsfs.GCSFileSystem().ls(path_str)]
        except ImportError:
            import tensorflow as tf
            return [os.path.join(path_str, f) for f in tf.io.gfile.listdir(path_str)]
    return [str(p) for p in Path(path_str).iterdir()]


# ============================================================
# Model + checkpoint loading
# ============================================================

def build_model(cfg):
    from models.dawn_spatial_v3 import DAWN as DAWN_SpatialV3
    mcfg = cfg['model']
    return DAWN_SpatialV3(
        vocab_size=mcfg.get('vocab_size', 30522),
        d_model=mcfg.get('d_model', 384),
        n_layers=mcfg.get('n_layers', 12),
        n_heads=mcfg.get('n_heads', 6),
        max_seq_len=mcfg.get('max_seq_len', 512),
        d_bottleneck=mcfg.get('d_bottleneck', 128),
        n_qk=mcfg.get('n_qk', 1580),
        n_v=mcfg.get('n_v', 2600),
        n_know=mcfg.get('n_know', 25200),
        max_k_qk=mcfg.get('max_k_qk', 158),
        max_k_v=mcfg.get('max_k_v', 260),
        max_k_know=mcfg.get('max_k_know', 1810),
        dropout_rate=mcfg.get('dropout', 0.1),
        router_dropout=mcfg.get('router_dropout', 0.1),
        gradient_checkpointing=False,
        n_chunks_know=cfg.get('training', {}).get('n_chunks_know', 1),
        n_chunks_qk=cfg.get('training', {}).get('n_chunks_qk', 1),
        n_chunks_v=cfg.get('training', {}).get('n_chunks_v', 1),
    )


def load_checkpoint_params(ckpt_path, model, cfg):
    # Find .flax file
    if not ckpt_path.endswith('.flax'):
        files = _list_dir(ckpt_path)
        flax_files = sorted([f for f in files if f.endswith('.flax')])
        best = [f for f in flax_files if 'best_model' in f]
        if best:
            ckpt_path = best[0]
        elif flax_files:
            ckpt_path = flax_files[-1]
        else:
            raise FileNotFoundError(
                f"No .flax checkpoints in {ckpt_path}\n"
                f"  Files: {[os.path.basename(f) for f in files]}")
        print(f"  Selected: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # Init model for param structure
    rng = jax.random.PRNGKey(0)
    max_seq = cfg['model'].get('max_seq_len', 512)
    dummy = jnp.ones((1, max_seq), dtype=jnp.int32)
    variables = model.init({'params': rng, 'dropout': rng}, dummy, deterministic=True)
    target_params = variables['params']

    # Load checkpoint — params only
    with _open_file(ckpt_path, 'rb') as f:
        bytes_data = f.read()
    raw = serialization.msgpack_restore(bytes_data)
    params = serialization.from_state_dict(target_params, raw['params'])

    step = int(raw.get('step', 0))
    epoch = int(raw.get('epoch', 0))
    print(f"  Step: {step}, Epoch: {epoch}")
    return params, {'step': step, 'epoch': epoch}


def get_model_cfg(cfg):
    """Extract model config dict for inference functions."""
    mcfg = cfg['model']
    return {
        'd_model': mcfg.get('d_model', 384),
        'n_layers': mcfg.get('n_layers', 12),
        'n_heads': mcfg.get('n_heads', 6),
        'max_seq_len': mcfg.get('max_seq_len', 512),
        'n_qk': mcfg.get('n_qk', 1580),
        'n_v': mcfg.get('n_v', 2600),
        'n_know': mcfg.get('n_know', 25200),
        'n_chunks_qk': cfg.get('training', {}).get('n_chunks_qk', 1),
        'n_chunks_v': cfg.get('training', {}).get('n_chunks_v', 1),
        'n_chunks_know': cfg.get('training', {}).get('n_chunks_know', 1),
    }


def load_val_tokens(val_data_path, max_tokens=None):
    """Load .bin uint16 validation data. Returns numpy array."""
    print(f"  Loading: {val_data_path}")
    if _is_gcs(val_data_path):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(val_data_path, 'rb') as f:
            raw = f.read()
        tokens = np.frombuffer(raw, dtype=np.uint16).copy()
    else:
        tokens = np.memmap(val_data_path, dtype=np.uint16, mode='r')
        tokens = np.array(tokens)  # copy to allow reshape
    if max_tokens:
        tokens = tokens[:max_tokens]
    print(f"  Loaded {len(tokens):,} tokens")
    return tokens


# ============================================================
# D1: Model Info
# ============================================================

def analyze_model_info(params, cfg, output_dir):
    print("\n" + "="*60)
    print("D1: Model Info")
    print("="*60)

    mcfg = cfg['model']
    n_params = sum(x.size for x in jax.tree.leaves(params) if hasattr(x, 'size'))

    # Param breakdown
    pool_params = sum(x.size for x in jax.tree.leaves(params.get('neuron_pool', {})) if hasattr(x, 'size'))
    router_params_n = sum(x.size for x in jax.tree.leaves(params.get('router', {})) if hasattr(x, 'size'))
    emb_params = sum(x.size for x in jax.tree.leaves(params.get('token_emb', {})) if hasattr(x, 'size'))
    emb_params += sum(x.size for x in jax.tree.leaves(params.get('pos_emb', {})) if hasattr(x, 'size'))
    block_params = n_params - pool_params - router_params_n - emb_params
    norm_params_n = sum(x.size for x in jax.tree.leaves(params.get('norm', {})) if hasattr(x, 'size'))

    # FLOPs estimation (per forward, single sequence)
    d = mcfg.get('d_model', 384)
    L = mcfg.get('n_layers', 12)
    S = mcfg.get('max_seq_len', 512)
    H = mcfg.get('n_heads', 6)
    db = mcfg.get('d_bottleneck', 128)
    n_qk = mcfg.get('n_qk', 1580)
    n_v = mcfg.get('n_v', 2600)
    n_know = mcfg.get('n_know', 25200)

    # Per layer: router proj + scores + gate + SRW (Q,K,V,Know) + attention + O proj
    router_flops = 2 * S * d * (db * 3 + db + 3 + 1)  # proj_attn + proj_know + tau
    qk_flops = 2 * (2 * S * db * n_qk + 2 * S * d * n_qk + S * n_qk * d)  # Q+K: scores + read + write
    v_flops = 2 * S * db * n_v + 2 * S * d * n_v + S * n_v * d
    know_flops = 2 * S * db * n_know + 2 * S * d * n_know + S * n_know * d
    attn_flops = 2 * 2 * S * S * d  # QK^T + scores@V
    o_flops = 2 * S * d * d  # O projection
    layer_flops = router_flops + qk_flops + v_flops + know_flops + attn_flops + o_flops
    total_flops = L * layer_flops
    gflops = total_flops / 1e9

    info = {
        'version': mcfg.get('model_version', 'unknown'),
        'n_params': n_params,
        'n_params_M': n_params / 1e6,
        'd_model': d, 'n_layers': L, 'n_heads': H,
        'd_bottleneck': db,
        'n_qk': n_qk, 'n_v': n_v, 'n_know': n_know,
        'max_seq_len': S,
        'param_breakdown': {
            'neuron_pool': pool_params,
            'router': router_params_n,
            'embeddings': emb_params,
            'blocks': block_params,
            'final_norm': norm_params_n,
        },
        'flops_G': gflops,
    }

    print(f"  Version:      {info['version']}")
    print(f"  Parameters:   {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  FLOPs/fwd:    {gflops:.1f}G")
    print(f"  d_model={d}, n_layers={L}, n_heads={H}, d_bn={db}")
    print(f"  QK={n_qk}, V={n_v}, Know={n_know}")
    print(f"\n  Param breakdown:")
    print(f"    NeuronPool:  {pool_params:>12,} ({pool_params/n_params*100:.1f}%)")
    print(f"    Router:      {router_params_n:>12,} ({router_params_n/n_params*100:.1f}%)")
    print(f"    Embeddings:  {emb_params:>12,} ({emb_params/n_params*100:.1f}%)")
    print(f"    Blocks:      {block_params:>12,} ({block_params/n_params*100:.1f}%)")

    _save_json(info, output_dir, 'model_info', 'info.json')
    return info


# ============================================================
# D2: Validation
# ============================================================

def analyze_validation(params, cfg, val_tokens, output_dir, batch_size=32, max_batches=200):
    print("\n" + "="*60)
    print("D2: Validation Performance")
    print("="*60)

    from models.dawn_spatial_v3 import vectorized_eval

    max_seq = cfg['model'].get('max_seq_len', 512)
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)

    # Limit batches
    max_seqs = min(max_batches * batch_size, n_seqs)
    tokens = tokens[:max_seqs]
    n_batches = max_seqs // batch_size

    print(f"  Sequences: {max_seqs:,}, Batches: {n_batches}, bs={batch_size}")

    # Transfer to device once
    tokens_dev = jnp.array(tokens, dtype=jnp.int32)
    model_cfg = get_model_cfg(cfg)

    # JIT compile + run
    eval_fn = jax.jit(lambda p, t: vectorized_eval(p, model_cfg, t, batch_size))

    print(f"  JIT compiling...")
    t0 = time.time()
    avg_loss, ppl, acc, total_valid = eval_fn(params, tokens_dev)
    jax.block_until_ready(avg_loss)
    elapsed = time.time() - t0

    # Single sync
    avg_loss = float(avg_loss)
    ppl = float(ppl)
    acc = float(acc)
    total_valid = int(total_valid)

    result = {
        'loss': avg_loss,
        'perplexity': ppl,
        'accuracy': acc,
        'total_tokens': total_valid,
        'n_batches': n_batches,
        'time_seconds': elapsed,
        'tokens_per_sec': total_valid / elapsed if elapsed > 0 else 0,
    }

    print(f"\n  Results:")
    print(f"    Loss:       {avg_loss:.4f}")
    print(f"    Perplexity: {ppl:.2f}")
    print(f"    Accuracy:   {acc:.2f}%")
    print(f"    Tokens:     {total_valid:,}")
    print(f"    Time:       {elapsed:.1f}s ({total_valid/elapsed:.0f} tok/s)")

    _save_json(result, output_dir, 'performance', 'validation.json')
    return result


# ============================================================
# Helpers
# ============================================================

def _save_json(data, output_dir, subdir, filename):
    path = os.path.join(output_dir, subdir)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    def convert(obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, jnp.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, jnp.ndarray):
            return np.array(obj).tolist()
        return obj

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"  Saved: {filepath}")


# ============================================================
# Main (D1 + D2 for now, more analyses added below)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DAWN-Spatial v3 Analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output", default="results/spatial_analysis")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only", default=None, help="Comma-separated: info,val,health,generate,weights,routing,samples")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    params, ckpt_info = load_checkpoint_params(args.checkpoint, model, cfg)

    only = set(args.only.split(',')) if args.only else None
    os.makedirs(args.output, exist_ok=True)

    # Save checkpoint info
    _save_json(ckpt_info, args.output, '.', 'checkpoint_info.json')

    if only is None or 'info' in only:
        analyze_model_info(params, cfg, args.output)

    if only is None or 'val' in only:
        val_path = args.val_data or cfg.get('data', {}).get('bin_val')
        if val_path:
            val_tokens = load_val_tokens(val_path)
            analyze_validation(params, cfg, val_tokens, args.output,
                             args.batch_size, args.max_batches)
        else:
            print("\n  Skipping validation (no --val_data)")

    # D3-D7 will be added in subsequent commits

    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    main()

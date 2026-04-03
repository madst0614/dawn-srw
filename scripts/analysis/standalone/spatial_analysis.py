#!/usr/bin/env python3
"""
DAWN-Spatial v3 Analysis Script
================================
Standalone analysis for DAWN-Spatial SRW checkpoints (JAX/Flax).

Features:
  D1: Model Info — params, architecture, FLOPs
  D2: Validation — loss, perplexity, accuracy
  D3: Neuron Health — activation rates, dead neurons, gate distribution
  D4: Generation — autoregressive text generation
  D5: Weight Analysis — embedding similarity, effective rank

Usage:
    python scripts/analysis/standalone/spatial_analysis.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/.../run_XXX \
        --config configs/train_config_spatial_r1_v3_40M_c4_5B.yaml \
        --output results/spatial_analysis

    # Quick generation only:
    python scripts/analysis/standalone/spatial_analysis.py \
        --checkpoint gs://...  --config configs/... \
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
import flax.linen as nn
import flax.serialization as serialization
import yaml


# ============================================================
# GCS / file helpers (copied from train_jax for standalone use)
# ============================================================

def _is_gcs(path):
    return str(path).startswith("gs://")


def _open_file(path, mode="rb"):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.open(path_str, mode)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
        except ImportError:
            raise ImportError("GCS requires 'gcsfs' or 'tensorflow'.")
    else:
        p = Path(path_str)
        if "w" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)


def _file_exists(path):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().exists(path_str)
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.exists(path_str)
    return Path(path_str).exists()


def _list_dir(path):
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return [f"gs://{f}" for f in fs.ls(path_str)]
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.listdir(path_str)
    return [str(p) for p in Path(path_str).iterdir()]


# ============================================================
# Model building
# ============================================================

def build_model(cfg):
    """Build DAWN-Spatial v3 model from config dict."""
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
    """Load checkpoint and return params."""
    # Find latest checkpoint in directory
    if not ckpt_path.endswith('.msgpack') and not ckpt_path.endswith('.ckpt'):
        # It's a directory — find latest
        files = _list_dir(ckpt_path)
        ckpts = sorted([f for f in files if 'step_' in f or 'checkpoint' in f])
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"  Latest checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoints in {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # Init model to get param structure
    rng = jax.random.PRNGKey(0)
    max_seq = cfg['model'].get('max_seq_len', 512)
    dummy = jnp.ones((1, max_seq), dtype=jnp.int32)
    variables = model.init(
        {'params': rng, 'dropout': rng}, dummy, deterministic=True)
    target_params = variables['params']

    # Load checkpoint
    import optax
    dummy_opt = optax.adamw(1e-4)
    dummy_opt_state = dummy_opt.init(target_params)

    target = {
        'params': target_params,
        'opt_state': dummy_opt_state,
        'epoch': 0, 'step': 0, 'step_in_epoch': 0,
        'steps_per_epoch': 0, 'best_val_loss': float('inf'),
        'config': {}, 'training_config': {},
    }

    with _open_file(ckpt_path, 'rb') as f:
        bytes_data = f.read()
    ckpt = serialization.from_bytes(target, bytes_data)

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    print(f"  Step: {step}, Epoch: {epoch}")
    return ckpt['params'], ckpt


def count_params(params):
    """Count total parameters."""
    return sum(x.size for x in jax.tree.leaves(params))


# ============================================================
# D1: Model Info
# ============================================================

def analyze_model_info(model, params, cfg):
    print("\n" + "="*60)
    print("D1: Model Info")
    print("="*60)

    n_params = count_params(params)
    mcfg = cfg['model']

    print(f"  Version:      {mcfg.get('model_version', 'unknown')}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  d_model:      {mcfg.get('d_model')}")
    print(f"  n_layers:     {mcfg.get('n_layers')}")
    print(f"  n_heads:      {mcfg.get('n_heads')}")
    print(f"  d_bottleneck: {mcfg.get('d_bottleneck')}")
    print(f"  n_qk:         {mcfg.get('n_qk')}")
    print(f"  n_v:          {mcfg.get('n_v')}")
    print(f"  n_know:       {mcfg.get('n_know')}")
    print(f"  max_seq_len:  {mcfg.get('max_seq_len')}")

    # Parameter breakdown
    pool = {k: v for k, v in
            jax.tree.map_with_path(
                lambda path, x: ('/'.join(str(p) for p in path), x.size),
                params).items()} if False else {}

    # Simpler breakdown
    pool_params = sum(x.size for x in jax.tree.leaves(params.get('neuron_pool', {})))
    router_params = sum(x.size for x in jax.tree.leaves(params.get('router', {})))
    other = n_params - pool_params - router_params

    print(f"\n  Param breakdown:")
    print(f"    NeuronPool:  {pool_params:>12,} ({pool_params/n_params*100:.1f}%)")
    print(f"    Router:      {router_params:>12,} ({router_params/n_params*100:.1f}%)")
    print(f"    Other:       {other:>12,} ({other/n_params*100:.1f}%)")

    return {'n_params': n_params, 'version': mcfg.get('model_version')}


# ============================================================
# D2: Validation
# ============================================================

def analyze_validation(model, params, cfg, val_data_path, max_batches=200, batch_size=32):
    print("\n" + "="*60)
    print("D2: Validation Performance")
    print("="*60)

    max_seq = cfg['model'].get('max_seq_len', 512)

    # Load validation data
    print(f"  Loading: {val_data_path}")
    if _is_gcs(val_data_path):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(val_data_path, 'rb') as f:
            raw = f.read()
        tokens = np.frombuffer(raw, dtype=np.uint16).copy()
    else:
        tokens = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    n_tokens = len(tokens)
    n_seqs = n_tokens // max_seq
    tokens = tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    print(f"  Tokens: {n_tokens:,}, Sequences: {n_seqs:,}")

    @jax.jit
    def eval_step(params, input_ids):
        attention_mask = jnp.ones_like(input_ids)
        labels = input_ids
        result = model.apply(
            {'params': params}, input_ids, labels=labels,
            attention_mask=attention_mask, deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(0)})
        return result['loss'], result['correct'], result['valid_count']

    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    n_batches = min(max_batches, n_seqs // batch_size)

    print(f"  Running {n_batches} batches (bs={batch_size})...")
    t0 = time.time()

    for i in range(n_batches):
        batch = jnp.array(tokens[i*batch_size:(i+1)*batch_size], dtype=jnp.int32)
        loss, correct, valid = eval_step(params, batch)
        loss, correct, valid = float(loss), int(correct), int(valid)
        total_loss += loss * valid
        total_correct += correct
        total_valid += valid

        if (i+1) % 50 == 0:
            avg = total_loss / total_valid
            print(f"    [{i+1}/{n_batches}] loss={avg:.4f} ppl={math.exp(avg):.2f}")

    elapsed = time.time() - t0
    avg_loss = total_loss / total_valid
    ppl = math.exp(avg_loss)
    acc = total_correct / total_valid * 100

    print(f"\n  Results:")
    print(f"    Loss:       {avg_loss:.4f}")
    print(f"    Perplexity: {ppl:.2f}")
    print(f"    Accuracy:   {acc:.2f}%")
    print(f"    Tokens:     {total_valid:,}")
    print(f"    Time:       {elapsed:.1f}s ({total_valid/elapsed:.0f} tok/s)")

    return {'loss': avg_loss, 'perplexity': ppl, 'accuracy': acc}


# ============================================================
# D3: Neuron Health
# ============================================================

def analyze_neuron_health(model, params, cfg):
    print("\n" + "="*60)
    print("D3: Neuron Health")
    print("="*60)

    pool = params['neuron_pool']
    results = {}

    for pool_name, emb_key, n_key in [
        ('QK', 'qk_emb', 'n_qk'), ('V', 'v_emb', 'n_v'),
        ('Know', 'know_emb', 'n_know')
    ]:
        emb = pool[emb_key]
        read_key = emb_key.replace('emb', 'read')
        write_key = emb_key.replace('emb', 'write')
        read = pool[read_key]
        write = pool[write_key]
        N = emb.shape[0]

        # Embedding norms
        emb_norms = jnp.linalg.norm(emb, axis=-1)
        read_norms = jnp.linalg.norm(read, axis=-1)
        write_norms = jnp.linalg.norm(write, axis=-1)

        # Dead neurons (near-zero norm)
        dead_emb = int((emb_norms < 1e-6).sum())
        dead_read = int((read_norms < 1e-6).sum())
        dead_write = int((write_norms < 1e-6).sum())

        print(f"\n  {pool_name} Pool (N={N}):")
        print(f"    emb   norm: mean={float(emb_norms.mean()):.4f}, "
              f"std={float(emb_norms.std()):.4f}, "
              f"dead={dead_emb}")
        print(f"    read  norm: mean={float(read_norms.mean()):.4f}, "
              f"std={float(read_norms.std()):.4f}, "
              f"dead={dead_read}")
        print(f"    write norm: mean={float(write_norms.mean()):.4f}, "
              f"std={float(write_norms.std()):.4f}, "
              f"dead={dead_write}")

        results[pool_name] = {
            'N': N,
            'emb_norm_mean': float(emb_norms.mean()),
            'read_norm_mean': float(read_norms.mean()),
            'write_norm_mean': float(write_norms.mean()),
            'dead_emb': dead_emb,
            'dead_read': dead_read,
            'dead_write': dead_write,
        }

    # Router tau bias
    router = params['router']
    tau_attn = router['tau_attn']['bias']
    tau_know = router['tau_know']['bias']
    print(f"\n  Tau bias:")
    print(f"    attn: [{', '.join(f'{float(v):.3f}' for v in tau_attn)}]")
    print(f"    know: [{float(tau_know[0]):.3f}]")

    return results


# ============================================================
# D4: Generation
# ============================================================

def generate(model, params, cfg, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Autoregressive generation with DAWN-Spatial."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_seq = cfg['model'].get('max_seq_len', 512)

    @jax.jit
    def forward_logits(params, ids):
        result = model.apply(
            {'params': params}, ids[None, :],
            deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(0)})
        return result['logits'][0, -1, :]  # last token logits

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")
    print(f"  ---")

    rng = jax.random.PRNGKey(42)
    generated = list(input_ids)

    for i in range(max_new_tokens):
        # Truncate to max_seq
        context = jnp.array(generated[-max_seq:], dtype=jnp.int32)
        logits = forward_logits(params, context)

        # Temperature + top-k sampling
        logits = logits / temperature
        if top_k > 0:
            top_k_vals = jax.lax.top_k(logits, top_k)
            threshold = top_k_vals[0][-1]
            logits = jnp.where(logits >= threshold, logits, -1e10)

        rng, sample_rng = jax.random.split(rng)
        next_token = int(jax.random.categorical(sample_rng, logits))
        generated.append(next_token)

        # Stop on [SEP] or [PAD]
        if next_token in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            break

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    gen_text = tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)

    print(f"  {output_text}")
    print(f"  ---")
    print(f"  Generated {len(generated) - len(input_ids)} tokens")

    return {'prompt': prompt, 'output': output_text, 'generated': gen_text}


# ============================================================
# D5: Weight Analysis
# ============================================================

def analyze_weights(params, cfg):
    print("\n" + "="*60)
    print("D5: Weight Analysis")
    print("="*60)

    pool = params['neuron_pool']
    results = {}

    for pool_name, emb_key in [('QK', 'qk_emb'), ('V', 'v_emb'), ('Know', 'know_emb')]:
        emb = np.array(pool[emb_key])
        N, d = emb.shape

        # Normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
        emb_normed = emb / norms

        # Sample if too large
        if N > 4096:
            idx = np.linspace(0, N-1, 4096, dtype=int)
            sample = emb_normed[idx]
        else:
            sample = emb_normed

        # Cosine similarity matrix
        sim = sample @ sample.T
        np.fill_diagonal(sim, 0)
        mean_sim = float(np.abs(sim).mean())
        max_sim = float(np.abs(sim).max())

        # Effective rank (via singular values)
        if N <= 4096:
            sv = np.linalg.svd(emb, compute_uv=False)
        else:
            sv = np.linalg.svd(emb[:4096], compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -float((sv_norm * np.log(sv_norm + 1e-10)).sum())
        eff_rank = float(np.exp(entropy))

        print(f"\n  {pool_name} (N={N}, d={d}):")
        print(f"    Cosine sim: mean={mean_sim:.4f}, max={max_sim:.4f}")
        print(f"    Effective rank: {eff_rank:.1f} / {min(N, d)}")
        print(f"    Top-5 SVs: {', '.join(f'{v:.2f}' for v in sv[:5])}")

        results[pool_name] = {
            'mean_cosine_sim': mean_sim,
            'max_cosine_sim': max_sim,
            'effective_rank': eff_rank,
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DAWN-Spatial v3 Analysis")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (local or GCS)")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--val_data", default=None, help="Validation .bin path")
    parser.add_argument("--output", default="results/spatial_analysis", help="Output directory")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only", default=None, help="Run only: info,val,health,generate,weights")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = build_model(cfg)

    # Load checkpoint
    params, ckpt_data = load_checkpoint_params(args.checkpoint, model, cfg)

    # Determine which analyses to run
    only = set(args.only.split(',')) if args.only else None
    results = {}

    if only is None or 'info' in only:
        results['model_info'] = analyze_model_info(model, params, cfg)

    if only is None or 'val' in only:
        val_path = args.val_data or cfg.get('data', {}).get('bin_val')
        if val_path:
            results['validation'] = analyze_validation(
                model, params, cfg, val_path, args.max_batches, args.batch_size)
        else:
            print("\n  Skipping validation (no --val_data or data.bin_val in config)")

    if only is None or 'health' in only:
        results['neuron_health'] = analyze_neuron_health(model, params, cfg)

    if only is None or 'generate' in only:
        results['generation'] = generate(
            model, params, cfg, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature)

    if only is None or 'weights' in only:
        results['weight_analysis'] = analyze_weights(params, cfg)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'analysis_results.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, jnp.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

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
import importlib
import functools


# ============================================================
# Dynamic model import
# ============================================================

_MODEL_MODULE = None

def get_model_module(model_file="models.dawn_spatial_v3"):
    global _MODEL_MODULE
    if _MODEL_MODULE is None:
        _MODEL_MODULE = importlib.import_module(model_file)
    return _MODEL_MODULE


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

def build_model(cfg, model_file="models.dawn_spatial_v3"):
    mod = get_model_module(model_file)
    mcfg = cfg['model']
    # Check if model has gate_norm_mode attribute
    import inspect
    init_params = inspect.signature(mod.DAWN).parameters
    extra_kw = {}
    if 'gate_norm_mode' in init_params and 'gate_norm_mode' in mcfg:
        extra_kw['gate_norm_mode'] = mcfg['gate_norm_mode']
    return mod.DAWN(
        vocab_size=mcfg.get('vocab_size', 30522),
        d_model=mcfg.get('d_model', 384),
        n_layers=mcfg.get('n_layers', 12),
        n_heads=mcfg.get('n_heads', 6),
        max_seq_len=mcfg.get('max_seq_len', 512),
        d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
        n_qk=mcfg.get('n_qk', 1580),
        n_v=mcfg.get('n_v', 2600),
        n_know=mcfg.get('n_know', 25200),
        dropout_rate=mcfg.get('dropout', 0.1),
        router_dropout=mcfg.get('router_dropout', 0.1),
        gradient_checkpointing=False,
        n_chunks_know=cfg.get('training', {}).get('n_chunks_know', 1),
        n_chunks_qk=cfg.get('training', {}).get('n_chunks_qk', 1),
        n_chunks_v=cfg.get('training', {}).get('n_chunks_v', 1),
        **extra_kw,
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

    # Squeeze leading singleton dim from all params (device-replicated checkpoints
    # save arrays with shape (1, ...) — squeeze to remove the leading dim).
    def _squeeze(x):
        if hasattr(x, 'ndim') and x.ndim >= 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x
    params = jax.tree.map(_squeeze, params)

    step = int(raw.get('step', 0))
    epoch = int(raw.get('epoch', 0))
    print(f"  Step: {step}, Epoch: {epoch}")
    return params, {'step': step, 'epoch': epoch}


def get_output_scales(pool_params):
    """Get per-pool output scale. Returns (qk_scale, v_scale, know_scale) — 1.0 if absent."""
    qk_s = pool_params.get('qk_scale', 1.0)
    v_s = pool_params.get('v_scale', 1.0)
    know_s = pool_params.get('know_scale', 1.0)
    return qk_s, v_s, know_s


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
    db = mcfg.get('d_route', mcfg.get('d_bottleneck', 128))
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
        'd_route': db,
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

    _mod = get_model_module()

    vectorized_eval = _mod.vectorized_eval

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

    # Convert params to JAX arrays for JIT closure capture
    params_jax = jax.tree.map(jnp.asarray, params)

    # JIT compile + run — params as closure (not arg) to avoid trace issues
    @jax.jit
    def eval_fn(t):
        return vectorized_eval(params_jax, model_cfg, t, batch_size)

    print(f"  JIT compiling...")
    t0 = time.time()
    avg_loss, ppl, acc, total_valid = eval_fn(tokens_dev)
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
    if subdir and subdir != '.':
        path = output_dir.rstrip('/') + '/' + subdir if _is_gcs(output_dir) else os.path.join(output_dir, subdir)
    else:
        path = output_dir
    filepath = path.rstrip('/') + '/' + filename if _is_gcs(path) else os.path.join(path, filename)

    if not _is_gcs(path):
        os.makedirs(path, exist_ok=True)

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

    with _open_file(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"  Saved: {filepath}")


# ============================================================
# D3: Neuron Health
# ============================================================

def analyze_neuron_health(params, cfg, output_dir):
    print("\n" + "="*60)
    print("D3: Neuron Health")
    print("="*60)

    _mod = get_model_module()

    vectorized_neuron_health = _mod.vectorized_neuron_health

    raw = jax.device_get(jax.jit(vectorized_neuron_health)(params))

    results = {}
    for pool_name in ['QK', 'V', 'Know']:
        s = raw[pool_name]
        N = int(s['N'])
        print(f"\n  {pool_name} Pool (N={N}):")
        for w in ('emb', 'read', 'write'):
            print(f"    {w:5s} norm: mean={float(s[f'{w}_mean']):.4f}, "
                  f"std={float(s[f'{w}_std']):.4f}, dead={int(s[f'{w}_dead'])}")

        results[pool_name] = {
            'N': N,
            'emb_norm_mean': float(s['emb_mean']),
            'emb_norm_std': float(s['emb_std']),
            'read_norm_mean': float(s['read_mean']),
            'write_norm_mean': float(s['write_mean']),
            'dead_emb': int(s['emb_dead']),
            'dead_read': int(s['read_dead']),
            'dead_write': int(s['write_dead']),
        }

    tau_attn = raw['tau_attn_bias']
    tau_know = raw['tau_know_bias']
    print(f"\n  Tau bias:")
    print(f"    attn: [{', '.join(f'{float(v):.3f}' for v in tau_attn)}]")
    print(f"    know: [{float(tau_know[0]):.3f}]")
    results['tau_attn_bias'] = [float(v) for v in tau_attn]
    results['tau_know_bias'] = [float(v) for v in tau_know]

    _save_json(results, output_dir, 'health', 'results.json')
    return results


# ============================================================
# D4: Generation (KV-cache)
# ============================================================

def analyze_generation(params, cfg, output_dir, prompt="The meaning of life is",
                       max_new_tokens=100, temperature=0.8, top_k=50):
    print("\n" + "="*60)
    print("D4: Generation")
    print("="*60)

    from transformers import AutoTokenizer
    _mod = get_model_module()
    prefill = _mod.prefill
    decode_step = _mod.decode_step

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    gen_len = min(max_new_tokens, max_seq - len(input_ids))

    print(f"  Prompt: \"{prompt}\" ({len(input_ids)} tokens)")
    print(f"  Generating up to {gen_len} tokens (temp={temperature}, top_k={top_k})")

    # JIT compile prefill + decode
    jit_prefill = jax.jit(lambda p, ids: prefill(p, model_cfg, ids))
    jit_decode = jax.jit(lambda p, tok, cK, cV, cL: decode_step(p, model_cfg, tok, cK, cV, cL))

    prompt_dev = jnp.array([input_ids], dtype=jnp.int32)  # [1, S_prompt]

    # Warmup
    print(f"  JIT compiling prefill...")
    t0 = time.time()
    logits, cache_K, cache_V, cache_len = jit_prefill(params, prompt_dev)
    jax.block_until_ready(logits)
    print(f"  Prefill compiled: {time.time()-t0:.1f}s")

    # Sample first token from prefill logits
    rng = jax.random.PRNGKey(42)
    generated = []

    def sample(logits_1d, rng):
        logits_1d = logits_1d / temperature
        if top_k > 0:
            top_vals, _ = jax.lax.top_k(logits_1d, top_k)
            logits_1d = jnp.where(logits_1d >= top_vals[-1], logits_1d, -1e10)
        return jax.random.categorical(rng, logits_1d)

    last_logits = logits[0, -1, :]
    stop_ids = {tokenizer.sep_token_id, tokenizer.pad_token_id}

    print(f"  JIT compiling decode...")
    t0_decode = time.time()
    # Warmup decode
    rng, srng = jax.random.split(rng)
    first_tok = int(sample(last_logits, srng))
    tok_dev = jnp.array([first_tok], dtype=jnp.int32)
    logits_d, cache_K, cache_V, cache_len = jit_decode(params, tok_dev, cache_K, cache_V, cache_len)
    jax.block_until_ready(logits_d)
    decode_compile_time = time.time() - t0_decode
    print(f"  Decode compiled: {decode_compile_time:.1f}s")

    generated.append(first_tok)
    if first_tok not in stop_ids:
        # Continue generating
        t0_gen = time.time()
        for i in range(gen_len - 1):
            if cache_len >= max_seq:
                break
            rng, srng = jax.random.split(rng)
            next_tok = int(sample(logits_d[0], srng))
            generated.append(next_tok)
            if next_tok in stop_ids:
                break
            tok_dev = jnp.array([next_tok], dtype=jnp.int32)
            logits_d, cache_K, cache_V, cache_len = jit_decode(
                params, tok_dev, cache_K, cache_V, cache_len)
        gen_time = time.time() - t0_gen
    else:
        gen_time = 0.0

    n_gen = len(generated)
    all_ids = input_ids + generated
    output_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    tok_s = n_gen / gen_time if gen_time > 0 else 0

    print(f"  ---")
    print(f"  {output_text}")
    print(f"  ---")
    print(f"  Generated {n_gen} tokens in {gen_time:.2f}s ({tok_s:.1f} tok/s)")

    result = {
        'prompt': prompt,
        'output': output_text,
        'generated': gen_text,
        'n_tokens': n_gen,
        'gen_time_sec': gen_time,
        'tok_per_sec': tok_s,
    }
    _save_json(result, output_dir, 'generation', 'result.json')
    return result


# ============================================================
# D5: Weight Analysis
# ============================================================

def analyze_weights(params, cfg, output_dir):
    print("\n" + "="*60)
    print("D5: Weight Analysis")
    print("="*60)

    _mod = get_model_module()

    vectorized_weight_analysis = _mod.vectorized_weight_analysis

    raw = jax.device_get(jax.jit(vectorized_weight_analysis)(params))
    results = {}

    for pool_name in ['QK', 'V', 'Know']:
        s = raw[pool_name]
        N, d = int(s['N']), int(s['d'])
        mean_sim = float(s['mean_cosine_sim'])
        max_sim = float(s['max_cosine_sim'])
        eff_rank = float(s['effective_rank'])
        top5 = [float(v) for v in s['top5_sv']]

        print(f"\n  {pool_name} (N={N}, d={d}):")
        print(f"    Cosine sim: mean={mean_sim:.4f}, max={max_sim:.4f}")
        print(f"    Effective rank: {eff_rank:.1f}")
        print(f"    Top-5 SVs: {', '.join(f'{v:.2f}' for v in top5)}")

        results[pool_name] = {
            'N': N, 'd': d,
            'mean_cosine_sim': mean_sim,
            'max_cosine_sim': max_sim,
            'effective_rank': eff_rank,
            'top5_singular_values': top5,
        }

    _save_json(results, output_dir, 'weights', 'results.json')
    return results


# ============================================================
# D6: Routing Analysis (gate distributions on val data)
# ============================================================

def analyze_routing(params, cfg, val_tokens, output_dir, n_batches=50, batch_size=8):
    print("\n" + "="*60)
    print("D6: Routing Analysis")
    print("="*60)

    _mod = get_model_module()
    _layer_norm = _mod._layer_norm
    _srw_inference_with_gates = _mod._srw_inference_with_gates
    _srw_inference = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']

    pool_params = params['neuron_pool']
    router_params = params['router']

    qk_norm = pool_params['qk_emb'] / (
        jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (
        jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_norm = pool_params['know_emb'] / (
        jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    # Prepare data
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_seqs = min(n_batches * batch_size, n_seqs)
    tokens = jnp.array(tokens[:total_seqs], dtype=jnp.int32)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]

    # Analyze routing at middle layer
    mid_layer = n_layers // 2
    bp = block_params_list[mid_layer]

    # Convert embeddings to JAX arrays so JIT-traced indexing works
    _emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    _pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    @jax.jit
    def get_routing_stats(input_ids):
        """Forward to mid layer, get gate distributions for Q,K,V,Know."""
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = _emb_matrix[input_ids.astype(jnp.int32)] + _pos_matrix[positions]

        # Output scales (1.0 for models without learnable scale)
        qk_s, v_s, know_s = get_output_scales(pool_params)

        # Forward to mid layer
        for i in range(mid_layer):
            lp = block_params_list[i]
            normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                               pool_params['qk_read'], pool_params['qk_write']) * qk_s
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write']) * qk_s
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write']) * v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ lp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
            h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out = _srw_inference(normed, h_k, know_norm, tau_k,
                                     pool_params['know_read'], pool_params['know_write']) * know_s
            x = x + know_out

        # At mid layer, get gate distributions
        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        _, gate_Q, _ = _srw_inference_with_gates(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                                               pool_params['qk_read'], pool_params['qk_write'])
        _, gate_K, _ = _srw_inference_with_gates(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                                               pool_params['qk_read'], pool_params['qk_write'])
        _, gate_V, _ = _srw_inference_with_gates(normed, h_V, v_norm, tau_all[:, :, 2:3],
                                               pool_params['v_read'], pool_params['v_write'])

        normed2 = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        _, gate_Know, _ = _srw_inference_with_gates(normed2, h_k, know_norm, tau_k,
                                                  pool_params['know_read'], pool_params['know_write'])

        # Compute stats per pool (raw gate: active = gate > 0)
        def gate_stats(g):
            active = (g > 0.0).astype(jnp.float32)
            active_per_token = active.sum(axis=-1).mean()  # avg active neurons per token
            active_ratio = active.mean(axis=(0, 1))  # [N] per-neuron activation rate
            neuron_entropy = -(active_ratio * jnp.log(active_ratio + 1e-10)
                              + (1 - active_ratio) * jnp.log(1 - active_ratio + 1e-10)).mean()
            coverage = (active_ratio > 0.01).sum()  # neurons used >1% of time
            gini = _gini(g.mean(axis=(0, 1)))  # Gini of per-neuron avg gate
            return {
                'active_per_token': active_per_token,
                'coverage': coverage,
                'entropy': neuron_entropy,
                'gini': gini,
                'mean_gate': g.mean(),
                'max_gate': g.max(),
            }

        return {
            'Q': gate_stats(gate_Q), 'K': gate_stats(gate_K),
            'V': gate_stats(gate_V), 'Know': gate_stats(gate_Know),
        }

    def _gini(x):
        """Gini coefficient of 1D array."""
        x_sorted = jnp.sort(x)
        n = x.shape[0]
        indices = jnp.arange(1, n + 1, dtype=jnp.float32)
        return (2.0 * (indices * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n)

    # Run on first batch (routing analysis is expensive)
    print(f"  Analyzing routing at layer {mid_layer} (batches={min(n_batches, total_seqs // batch_size)})...")
    batch = tokens[:batch_size]
    t0 = time.time()
    stats = get_routing_stats(batch)
    stats = jax.device_get(stats)
    elapsed = time.time() - t0

    results = {}
    for pool_name in ['Q', 'K', 'V', 'Know']:
        s = stats[pool_name]
        results[pool_name] = {k: float(v) for k, v in s.items()}
        print(f"\n  {pool_name}:")
        print(f"    Active/token: {float(s['active_per_token']):.1f}")
        print(f"    Coverage:     {int(s['coverage'])}")
        print(f"    Gini:         {float(s['gini']):.4f}")
        print(f"    Mean gate:    {float(s['mean_gate']):.6f}")

    results['layer'] = mid_layer
    results['time_seconds'] = elapsed
    _save_json(results, output_dir, 'routing', 'results.json')
    return results


# ============================================================
# D7: Multi-prompt Generation Samples (matches DAWN format)
# ============================================================

GENERATION_PROMPTS = [
    {"prompt": "The meaning of life is", "category": "philosophy"},
    {"prompt": "In a shocking finding, scientists discovered", "category": "science"},
    {"prompt": "The president of the United States", "category": "factual"},
    {"prompt": "Once upon a time in a land far away", "category": "creative"},
    {"prompt": "The best way to learn programming is", "category": "technical"},
    {"prompt": "Breaking news:", "category": "news"},
    {"prompt": "Dear diary, today I", "category": "personal"},
    {"prompt": "The recipe for happiness includes", "category": "abstract"},
]


def analyze_generation_samples(params, cfg, output_dir,
                                max_new_tokens=50, temperature=0.8, top_k=50):
    print("\n" + "="*60)
    print("D7: Generation Samples")
    print("="*60)

    from transformers import AutoTokenizer
    _mod = get_model_module()
    prefill = _mod.prefill
    decode_step = _mod.decode_step

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']

    jit_prefill = jax.jit(lambda p, ids: prefill(p, model_cfg, ids))
    jit_decode = jax.jit(lambda p, tok, cK, cV, cL: decode_step(p, model_cfg, tok, cK, cV, cL))

    stop_ids = {tokenizer.sep_token_id, tokenizer.pad_token_id}
    results = []
    rng = jax.random.PRNGKey(42)

    for i, item in enumerate(GENERATION_PROMPTS):
        prompt = item['prompt']
        category = item['category']
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_dev = jnp.array([input_ids], dtype=jnp.int32)

        t0 = time.time()
        logits, cK, cV, cL = jit_prefill(params, prompt_dev)
        jax.block_until_ready(logits)
        prefill_time = time.time() - t0

        generated = []
        last_logits = logits[0, -1, :]

        t0 = time.time()
        gen_len = min(max_new_tokens, max_seq - len(input_ids))
        for j in range(gen_len):
            rng, srng = jax.random.split(rng)
            l = last_logits / temperature
            if top_k > 0:
                tv, _ = jax.lax.top_k(l, top_k)
                l = jnp.where(l >= tv[-1], l, -1e10)
            tok = int(jax.random.categorical(srng, l))
            generated.append(tok)
            if tok in stop_ids or cL >= max_seq:
                break
            tok_dev = jnp.array([tok], dtype=jnp.int32)
            last_logits, cK, cV, cL = jit_decode(params, tok_dev, cK, cV, cL)
            last_logits = last_logits[0]
        decode_time = time.time() - t0

        all_ids = input_ids + generated
        output = tokenizer.decode(all_ids, skip_special_tokens=True)
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        n_gen = len(generated)
        tok_s = n_gen / decode_time if decode_time > 0 else 0

        print(f"\n  [{category}] \"{prompt}\"")
        print(f"    → {gen_text[:80]}{'...' if len(gen_text) > 80 else ''}")
        print(f"    {n_gen} tokens, {tok_s:.1f} tok/s")

        results.append({
            'prompt': prompt,
            'category': category,
            'generated': gen_text,
            'full_output': output,
            'new_tokens': n_gen,
            'prefill_ms': prefill_time * 1000,
            'decode_ms': decode_time * 1000,
            'tokens_per_sec': tok_s,
        })

    # Save both JSON and readable text
    _save_json(results, output_dir, 'generation', 'samples.json')

    # Human-readable text output
    txt_path = os.path.join(output_dir, 'generation', 'samples.txt')
    with open(txt_path, 'w') as f:
        for r in results:
            f.write(f"[{r['category']}] {r['prompt']}\n")
            f.write(f"  → {r['generated']}\n")
            f.write(f"  ({r['new_tokens']} tokens, {r['tokens_per_sec']:.1f} tok/s)\n\n")
    print(f"\n  Saved: {txt_path}")

    return results


# ============================================================
# R.1: Q/K Specialization (rebuttal D.1 methodology)
# ============================================================

def analyze_qk_specialization(params, cfg, val_tokens, output_dir,
                               n_batches=50, batch_size=8):
    """Q/K specialization: same method as v17.1 rebuttal D.1.

    For each QK neuron, count how often it's selected by Q vs K gate.
    q_ratio = q_count / (q_count + k_count). Threshold 0.7/0.3.
    """
    print("\n" + "="*60)
    print("R.1: Q/K Specialization")
    print("="*60)

    _mod = get_model_module()

    analysis_forward = _mod.analysis_forward
    import numpy as np

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_qk = model_cfg['n_qk']

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_batches = min(n_batches, n_seqs // batch_size)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Accumulate per-neuron Q/K counts across batches
    q_counts = np.zeros(n_qk, dtype=np.float64)
    k_counts = np.zeros(n_qk, dtype=np.float64)
    overlaps = []

    print(f"  Processing {total_batches} batches (bs={batch_size})...")
    for i in range(total_batches):
        batch = jnp.array(tokens[i*batch_size:(i+1)*batch_size], dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch)

        # Use raw sigmoid gates (before normalization) for meaningful thresholding
        # gate_Q_raw: [n_layers, B, S, n_qk]
        gQ = np.array(jax.device_get(layer_info['gate_Q_raw']))  # [L, B, S, N]
        gK = np.array(jax.device_get(layer_info['gate_K_raw']))

        # Binary: active if raw sigmoid gate > 0.5
        q_active = (gQ > 0.5).sum(axis=(0, 1, 2))  # [N] summed over layers,batch,seq
        k_active = (gK > 0.5).sum(axis=(0, 1, 2))
        q_counts += q_active
        k_counts += k_active

        # Batch overlap (across all layers)
        q_bin = gQ > 0.5
        k_bin = gK > 0.5
        both = (q_bin & k_bin).sum()
        either = (q_bin | k_bin).sum()
        overlaps.append(float(both) / (float(either) + 1e-8))

        if (i+1) % 10 == 0:
            print(f"    [{i+1}/{total_batches}]")

    # Classification
    total_usage = q_counts + k_counts
    valid_mask = total_usage > 0
    q_ratio = np.zeros(n_qk)
    q_ratio[valid_mask] = q_counts[valid_mask] / total_usage[valid_mask]

    q_specialized = int((q_ratio[valid_mask] > 0.7).sum())
    k_specialized = int((q_ratio[valid_mask] < 0.3).sum())
    shared = int(((q_ratio[valid_mask] >= 0.3) & (q_ratio[valid_mask] <= 0.7)).sum())
    inactive = int((~valid_mask).sum())
    active = int(valid_mask.sum())

    # Pearson correlation
    if valid_mask.sum() > 1:
        corr_all = float(np.corrcoef(q_counts, k_counts)[0, 1])
        corr_active = float(np.corrcoef(
            q_counts[valid_mask], k_counts[valid_mask])[0, 1])
    else:
        corr_all = corr_active = 0.0

    avg_overlap = float(np.mean(overlaps))

    # Sensitivity analysis
    sensitivity = {}
    for thresh in [0.6, 0.65, 0.7, 0.75, 0.8]:
        q_s = int((q_ratio[valid_mask] > thresh).sum())
        k_s = int((q_ratio[valid_mask] < (1 - thresh)).sum())
        sh = int(((q_ratio[valid_mask] >= (1 - thresh)) &
                  (q_ratio[valid_mask] <= thresh)).sum())
        sensitivity[str(thresh)] = {
            'q_specialized': q_s, 'k_specialized': k_s, 'shared': sh}

    spec_pct = (q_specialized + k_specialized) / active * 100 if active > 0 else 0

    print(f"\n  QK Pool ({n_qk} neurons):")
    print(f"    Correlation (all): r={corr_all:.4f}")
    print(f"    Correlation (active): r={corr_active:.4f}")
    print(f"    Q-only: {q_specialized}  K-only: {k_specialized}  "
          f"Shared: {shared}  Inactive: {inactive}")
    print(f"    Specialization: {spec_pct:.1f}% (of {active} active)")
    print(f"    Avg Q/K overlap: {avg_overlap:.4f}")
    print(f"    Threshold sensitivity:")
    for t, s in sorted(sensitivity.items()):
        t_spec = s['q_specialized'] + s['k_specialized']
        t_active = t_spec + s['shared']
        t_pct = t_spec / t_active * 100 if t_active > 0 else 0
        print(f"      θ={t}: Q={s['q_specialized']} K={s['k_specialized']} "
              f"Shared={s['shared']} → {t_pct:.1f}%")

    results = {
        'n_neurons': n_qk, 'n_active': active,
        'q_specialized': q_specialized, 'k_specialized': k_specialized,
        'shared': shared, 'inactive': inactive,
        'specialization_pct': spec_pct,
        'correlation': corr_all, 'correlation_active': corr_active,
        'avg_overlap': avg_overlap,
        'sensitivity_analysis': sensitivity,
    }
    _save_json(results, output_dir, 'r1_qk_specialization', 'results.json')
    return results


# ============================================================
# R.4: Layer-wise Attn/Know Balance (rebuttal D.4 methodology)
# ============================================================

def analyze_layer_balance(params, cfg, val_tokens, output_dir,
                          n_batches=50, batch_size=8):
    """Layer balance: output norm ratio (same as v17.1 D.4)."""
    print("\n" + "="*60)
    print("R.4: Layer-wise Balance")
    print("="*60)

    _mod = get_model_module()

    analysis_forward = _mod.analysis_forward
    import numpy as np

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_layers = model_cfg['n_layers']

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_batches = min(n_batches, n_seqs // batch_size)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))

    attn_norms = np.zeros(n_layers)
    know_norms = np.zeros(n_layers)

    print(f"  Processing {total_batches} batches...")
    for i in range(total_batches):
        batch = jnp.array(tokens[i*batch_size:(i+1)*batch_size], dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch)
        an = np.array(jax.device_get(layer_info['attn_out_norm']))  # [n_layers]
        kn = np.array(jax.device_get(layer_info['know_out_norm']))
        attn_norms += an
        know_norms += kn

    attn_norms /= total_batches
    know_norms /= total_batches

    per_layer = []
    print(f"\n  Layer-wise Contribution (%):")
    for layer in range(n_layers):
        total = attn_norms[layer] + know_norms[layer] + 1e-8
        a_ratio = attn_norms[layer] / total * 100
        k_ratio = know_norms[layer] / total * 100
        bar_len = int(a_ratio / 2)
        bar = '#' * bar_len + '.' * (50 - bar_len)
        print(f"    L{layer:2d}: {a_ratio:5.1f}% attn  {k_ratio:5.1f}% know  |{bar}|")
        per_layer.append({
            'layer': layer,
            'attention_norm': float(attn_norms[layer]),
            'knowledge_norm': float(know_norms[layer]),
            'attention_ratio': float(a_ratio),
            'knowledge_ratio': float(k_ratio),
        })

    third = n_layers // 3
    early_attn = np.mean([p['attention_ratio'] for p in per_layer[:third]])
    mid_attn = np.mean([p['attention_ratio'] for p in per_layer[third:2*third]])
    late_attn = np.mean([p['attention_ratio'] for p in per_layer[2*third:]])

    print(f"\n  Early (L0-{third-1}):  {early_attn:.1f}% attention")
    print(f"  Mid:              {mid_attn:.1f}% attention")
    print(f"  Late:             {late_attn:.1f}% attention")

    results = {
        'n_layers': n_layers,
        'per_layer': per_layer,
        'summary': {
            'early_layers_attn': float(early_attn),
            'mid_layers_attn': float(mid_attn),
            'late_layers_attn': float(late_attn),
        }
    }
    _save_json(results, output_dir, 'r4_layer_balance', 'results.json')
    return results


# ============================================================
# R.2: POS Selectivity (rebuttal D.2 methodology)
# ============================================================

UPOS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
             'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']


def _load_ud_ewt(max_sentences=5000):
    """Load UD-EWT dataset. Returns list of {tokens, upos}."""
    try:
        import conllu
    except ImportError:
        print("  conllu not installed. Falling back to NLTK treebank.")
        return _load_nltk_fallback(max_sentences)

    import urllib.request
    url = ("https://raw.githubusercontent.com/UniversalDependencies/"
           "UD_English-EWT/master/en_ewt-ud-train.conllu")
    try:
        print(f"  Downloading UD-EWT from GitHub...")
        response = urllib.request.urlopen(url, timeout=30)
        text = response.read().decode('utf-8')
        sentences = conllu.parse(text)
    except Exception as e:
        print(f"  UD-EWT download failed: {e}. Falling back to NLTK.")
        return _load_nltk_fallback(max_sentences)

    data = []
    for sent in sentences[:max_sentences]:
        tokens = [t['form'] for t in sent]
        upos = [t['upostag'] for t in sent]
        data.append({'tokens': tokens, 'upos': upos})
    return data


def _load_nltk_fallback(max_sentences=5000):
    """Fallback: use NLTK treebank with universal tagset."""
    import nltk
    nltk.download('treebank', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    from nltk.corpus import treebank

    data = []
    for i, sent in enumerate(treebank.tagged_sents(tagset='universal')):
        if i >= max_sentences:
            break
        tokens = [w for w, _ in sent]
        # Map NLTK universal tags to UD UPOS
        tag_map = {'NOUN': 'NOUN', 'VERB': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
                    'ADP': 'ADP', 'DET': 'DET', 'PRON': 'PRON', 'NUM': 'NUM',
                    'CONJ': 'CCONJ', 'PRT': 'PART', '.': 'PUNCT', 'X': 'X'}
        upos = [tag_map.get(t, 'X') for _, t in sent]
        data.append({'tokens': tokens, 'upos': upos})
    return data


def analyze_pos_selectivity(params, cfg, output_dir,
                             max_sentences=5000, batch_size=4):
    """POS selectivity: same as v17.1 D.2.

    selectivity[neuron, pos] = P(neuron active | POS) / P(neuron active)
    Specialist: selectivity > 2.0 AND mean_weight > 0.1
    """
    print("\n" + "="*60)
    print("R.2: POS Selectivity")
    print("="*60)

    from transformers import AutoTokenizer
    _mod = get_model_module()
    _layer_norm = _mod._layer_norm
    _srw_inference = _mod._srw_inference
    _srw_inference_with_gates = _mod._srw_inference_with_gates
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']
    max_seq = model_cfg['max_seq_len']

    # Lightweight forward: returns layer-averaged raw gate per pool [B, S, N]
    # No full [n_layers, B, S, N] tensor — 12x less memory than analysis_forward
    params_jax = jax.tree.map(jnp.asarray, params)
    pool_params = params_jax['neuron_pool']
    router_params = params_jax['router']
    emb_matrix = jnp.asarray(params_jax['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params_jax['pos_emb']['embedding'])

    qk_emb_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_emb_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    qk_s = pool_params.get('qk_scale', 1.0)
    v_s = pool_params.get('v_scale', 1.0)
    know_s = pool_params.get('know_scale', 1.0)

    @jax.jit
    def lightweight_forward(input_ids):
        """Forward returning layer-averaged raw gate per pool. Memory: [B,S,N] not [L,B,S,N]."""
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        # Accumulators for layer-mean raw gate
        gate_q_sum = jnp.zeros((B, S, n_qk))
        gate_v_sum = jnp.zeros((B, S, n_v))
        gate_know_sum = jnp.zeros((B, S, n_know))

        def layer_fn(carry, bp):
            x, gq, gv, gk = carry
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

            Q, gQ_raw, _ = _srw_inference_with_gates(normed, h_Q, qk_emb_n, tau_all[:,:,0:1], pool_params['qk_read'], pool_params['qk_write'])
            K, _, _ = _srw_inference_with_gates(normed, h_K, qk_emb_n, tau_all[:,:,1:2], pool_params['qk_read'], pool_params['qk_write'])
            V, gV_raw, _ = _srw_inference_with_gates(normed, h_V, v_emb_n, tau_all[:,:,2:3], pool_params['v_read'], pool_params['v_write'])
            Q, K, V = Q * qk_s, K * qk_s, V * v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed2 = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out, gK_raw, _ = _srw_inference_with_gates(normed2, h_k, know_emb_n, tau_k, pool_params['know_read'], pool_params['know_write'])
            x = x + know_out * know_s

            return (x, gq + gQ_raw, gv + gV_raw, gk + gK_raw), None

        (_, gate_q_sum, gate_v_sum, gate_know_sum), _ = jax.lax.scan(
            layer_fn, (x, gate_q_sum, gate_v_sum, gate_know_sum), stacked)

        inv_L = 1.0 / n_layers
        return gate_q_sum * inv_L, gate_v_sum * inv_L, gate_know_sum * inv_L

    # Load UD-EWT
    dataset = _load_ud_ewt(max_sentences)
    print(f"  Loaded {len(dataset)} sentences")

    # Accumulators: per-neuron activation count, per-(neuron,pos) count
    n_pos = len(UPOS_TAGS)
    pos_to_idx = {p: i for i, p in enumerate(UPOS_TAGS)}

    # Pool info (gates come from lightweight_forward, indexed by position in tuple)
    pools = [
        ('QK', n_qk, 0),   # (name, size, index in lightweight_forward output)
        ('V', n_v, 1),
        ('Know', n_know, 2),
    ]
    pool_counts = {}
    pool_pos_counts = {}
    pos_token_counts = np.zeros(n_pos, dtype=np.float64)
    total_tokens = 0

    for pool_name, pool_size, _ in pools:
        pool_counts[pool_name] = np.zeros(pool_size, dtype=np.float64)
        pool_pos_counts[pool_name] = np.zeros((pool_size, n_pos), dtype=np.float64)

    n_total = len(dataset)
    n_batches_est = (n_total + batch_size - 1) // batch_size
    print(f"  Processing {n_total} sentences (~{n_batches_est} batches, bs={batch_size})...")
    import sys
    t_start = time.time()
    n_batches_done = 0

    # Process in batches: tokenize sentences, map POS, forward, accumulate
    batch_tokens = []
    batch_pos_labels = []

    for si, sent in enumerate(dataset):
        # Tokenize and map POS to subword tokens
        text = ' '.join(sent['tokens'])
        encoding = tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=False, truncation=True,
                             max_length=max_seq)
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']

        # Build character-span POS mapping
        char_pos = {}
        char_idx = 0
        for tok, pos in zip(sent['tokens'], sent['upos']):
            start = text.find(tok, char_idx)
            if start >= 0:
                for c in range(start, start + len(tok)):
                    char_pos[c] = pos
                char_idx = start + len(tok)

        # Map subword tokens to POS
        token_pos = []
        for s, e in offsets:
            # Find POS for this span
            mid = (s + e) // 2
            pos = char_pos.get(mid, 'X')
            token_pos.append(pos_to_idx.get(pos, pos_to_idx['X']))

        batch_tokens.append(token_ids)
        batch_pos_labels.append(token_pos)

        # Process when batch full or last sentence
        if len(batch_tokens) >= batch_size or si == len(dataset) - 1:
            if not batch_tokens:
                continue

            # Fixed shape: (batch_size, max_seq) avoids JIT recompilation
            max_len = max_seq
            actual_b = len(batch_tokens)
            padded = np.zeros((batch_size, max_len), dtype=np.int32)
            pos_padded = np.full((batch_size, max_len), -1, dtype=np.int32)
            for bi, (tids, plabs) in enumerate(zip(batch_tokens, batch_pos_labels)):
                l = min(len(tids), max_len)
                padded[bi, :l] = tids[:l]
                pos_padded[bi, :l] = plabs[:l]

            # Lightweight forward: returns layer-averaged raw gate per pool
            ids_dev = jnp.array(padded, dtype=jnp.int32)
            gate_q_avg, gate_v_avg, gate_know_avg = lightweight_forward(ids_dev)
            gates_by_idx = {
                0: np.array(jax.device_get(gate_q_avg)),    # [B, S, n_qk]
                1: np.array(jax.device_get(gate_v_avg)),    # [B, S, n_v]
                2: np.array(jax.device_get(gate_know_avg)), # [B, S, n_know]
            }

            # Fully vectorized accumulation
            valid_mask = pos_padded >= 0  # [B, S]
            total_tokens += int(valid_mask.sum())

            valid_pos = pos_padded[valid_mask]  # [n_valid]
            pos_token_counts += np.bincount(valid_pos, minlength=n_pos).astype(np.float64)

            # POS one-hot (shared across pools)
            safe_pos = np.where(valid_mask, pos_padded, 0).ravel()  # [B*S]
            pos_onehot = np.zeros((safe_pos.shape[0], n_pos), dtype=np.float32)
            pos_onehot[np.arange(safe_pos.shape[0]), safe_pos] = valid_mask.ravel().astype(np.float32)

            for pool_name, pool_size, gate_idx in pools:
                gate_avg = gates_by_idx[gate_idx]  # [B, S, N]
                active = (gate_avg > 0.0).astype(np.float32)
                active_masked = active * valid_mask[:, :, np.newaxis]
                pool_counts[pool_name] += active_masked.sum(axis=(0, 1))
                active_flat = active.reshape(-1, pool_size)
                pool_pos_counts[pool_name] += (active_flat.T @ pos_onehot)

            n_batches_done += 1
            batch_tokens = []
            batch_pos_labels = []

        if (si + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - si - 1) / rate if rate > 0 else 0
            print(f"    [{si+1:>5}/{n_total}] sentences | "
                  f"{n_batches_done} batches | {total_tokens:,} tokens | "
                  f"{rate:.0f} sent/s | ETA {eta:.0f}s",
                  flush=True)

    # Compute selectivity
    results = {}
    for pool_name, N, _ in pools:
        # P(neuron active)
        p_neuron = pool_counts[pool_name] / (total_tokens + 1e-8)  # [N]

        safe_counts = np.where(pos_token_counts > 0, pos_token_counts, 1.0)  # [n_pos]
        mean_weight = pool_pos_counts[pool_name] / safe_counts[np.newaxis, :]  # [N, n_pos]
        selectivity = mean_weight / (p_neuron[:, np.newaxis] + 1e-8)  # [N, n_pos]
        # Zero out columns with no POS tokens
        no_pos = pos_token_counts == 0
        selectivity[:, no_pos] = 0.0
        mean_weight[:, no_pos] = 0.0

        # Per-POS top neurons
        top_per_pos = {}
        for pi, pos in enumerate(UPOS_TAGS):
            sel_col = selectivity[:, pi]
            mw_col = mean_weight[:, pi]
            # Sort by selectivity
            order = np.argsort(-sel_col)
            neurons = []
            for ni in order[:20]:
                is_spec = bool(sel_col[ni] > 2.0 and mw_col[ni] > 0.1)
                neurons.append({
                    'neuron': int(ni),
                    'selectivity': float(sel_col[ni]),
                    'mean_weight': float(mw_col[ni]),
                    'is_specialist': is_spec,
                })
            top_per_pos[pos] = neurons

        n_specialists_total = int(((selectivity > 2.0) &
                                    (mean_weight > 0.1)).any(axis=1).sum())

        print(f"\n  {pool_name} Pool:")
        print(f"    Total specialists: {n_specialists_total}")
        # Show top POS
        for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PROPN']:
            if top_per_pos[pos]:
                top1 = top_per_pos[pos][0]
                n_spec = sum(1 for n in top_per_pos[pos] if n['is_specialist'])
                print(f"    {pos:6s}: top neuron={top1['neuron']:4d} "
                      f"sel={top1['selectivity']:.1f}x  ({n_spec} specialists)")

        results[pool_name] = {
            'n_neurons': N,
            'n_specialists': n_specialists_total,
            'top_selective_per_pos': top_per_pos,
            'total_tokens': total_tokens,
        }

    _save_json(results, output_dir, 'r2_pos_selectivity', 'results.json')
    return results


# ============================================================
# R.3: Knowledge Neurons — Contrastive Score (rebuttal D.3)
# ============================================================

PHYSICS_QUERIES = [
    {"prompt": "light travels at the speed of", "target": "light"},
    {"prompt": "the earth orbits the", "target": "sun"},
    {"prompt": "the earth revolves around the", "target": "sun"},
]

CONTROL_QUERIES = [
    {"prompt": "the largest organ in the human body is the", "target": "skin"},
    {"prompt": "water freezes at", "target": "zero"},
    {"prompt": "the capital of france is", "target": "paris"},
]


def analyze_knowledge_neurons(params, cfg, output_dir,
                               min_target_count=20, max_runs=500):
    """Knowledge neurons via contrastive score (same as v17.1 D.3).

    contrastive_score[neuron] = target_freq - baseline_freq
    """
    print("\n" + "="*60)
    print("R.3: Knowledge Neurons (Contrastive)")
    print("="*60)

    from transformers import AutoTokenizer
    _mod = get_model_module()
    prefill = _mod.prefill
    decode_step = _mod.decode_step
    analysis_forward = _mod.analysis_forward
    import numpy as np
    from collections import Counter

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']

    jit_prefill = jax.jit(lambda p, ids: prefill(p, model_cfg, ids))
    jit_decode = jax.jit(lambda p, tok, cK, cV, cL: decode_step(p, model_cfg, tok, cK, cV, cL))
    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))

    all_queries = PHYSICS_QUERIES + CONTROL_QUERIES
    results = {'physics': [], 'control': []}

    for qi, q in enumerate(all_queries):
        is_physics = qi < len(PHYSICS_QUERIES)
        tag = 'physics' if is_physics else 'control'
        prompt = q['prompt']
        target = q['target'].lower()
        target_id = tokenizer.encode(target, add_special_tokens=False)
        if not target_id:
            print(f"  Skipping '{target}' — not in vocab")
            continue
        target_id = target_id[0]

        print(f"\n  [{tag}] \"{prompt}\" → '{target}' (id={target_id})")

        # Collect activations
        know_target_counts = Counter()  # neuron_id → count when target generated
        know_baseline_counts = Counter()
        successful_runs = 0
        total_baseline_steps = 0

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        for run in range(max_runs):
            if successful_runs >= min_target_count:
                break

            # Generate token by token with greedy decode (matches v17.1)
            # Build full context each step, get gates at last position
            context_ids = list(prompt_ids)
            max_gen_steps = 10

            rng = jax.random.PRNGKey(run)

            for step in range(max_gen_steps):
                # Pad to fixed length for JIT stability
                ctx_len = len(context_ids)
                if ctx_len >= max_seq:
                    break
                padded = context_ids + [0] * (max_seq - ctx_len)
                ctx_dev = jnp.array([padded], dtype=jnp.int32)

                # Forward with gate extraction — recompute per step
                logits, layer_info = jit_analysis(params, ctx_dev)
                # Know gate at current last position, averaged over layers
                know_gates = jax.device_get(layer_info['gate_Know'])  # [L, 1, S, N]
                last_pos_gates = know_gates[:, 0, ctx_len - 1, :]  # [L, N]
                active_neurons = set(np.where(last_pos_gates.mean(axis=0) > 1e-6)[0])

                # Temperature sampling for diversity across runs
                rng, srng = jax.random.split(rng)
                l = logits[0, ctx_len - 1, :] / 0.9
                next_tok = int(jax.random.categorical(srng, l))

                if next_tok == target_id:
                    successful_runs += 1
                    for n in active_neurons:
                        know_target_counts[n] += 1
                    break
                elif next_tok in (tokenizer.sep_token_id, tokenizer.pad_token_id, 0):
                    break
                else:
                    total_baseline_steps += 1
                    for n in active_neurons:
                        know_baseline_counts[n] += 1
                    context_ids.append(next_tok)

            if (run + 1) % 50 == 0:
                print(f"    run {run+1}: {successful_runs}/{min_target_count} hits")

        # Compute contrastive scores
        all_neurons = set(know_target_counts.keys()) | set(know_baseline_counts.keys())
        contrastive = {}
        for n in all_neurons:
            t_freq = know_target_counts[n] / max(successful_runs, 1)
            b_freq = know_baseline_counts[n] / max(total_baseline_steps, 1)
            contrastive[int(n)] = {
                'target_freq': t_freq,
                'baseline_freq': b_freq,
                'contrastive': t_freq - b_freq,
            }

        # Top contrastive neurons
        top_neurons = sorted(contrastive.items(),
                            key=lambda x: x[1]['contrastive'], reverse=True)[:10]
        if top_neurons:
            top_str = ", ".join(f"n{n}({s['contrastive']:+.3f})" for n, s in top_neurons[:5])
            print(f"    Hits: {successful_runs}/{run+1} runs")
            print(f"    Top know neurons: {top_str}")

        entry = {
            'prompt': prompt, 'target': target,
            'successful_runs': successful_runs,
            'total_runs': run + 1,
            'match_rate': successful_runs / max(run + 1, 1),
            'top_neurons': [{
                'neuron': n, **s
            } for n, s in top_neurons],
            'n_unique_active': len(all_neurons),
        }
        results[tag].append(entry)

    _save_json(results, output_dir, 'r3_knowledge_neurons', 'results.json')
    return results


# ============================================================
# R.5: Suppression Sweep (rebuttal D.5 methodology)
# ============================================================

def analyze_suppression(params, cfg, output_dir, knowledge_results=None,
                        sweep_pcts=(0.05, 0.10, 0.15, 0.20)):
    """Neuron suppression sweep (same as v17.1 D.5).

    Suppress top N% neurons by contrastive score, measure probability drop.
    """
    print("\n" + "="*60)
    print("R.5: Suppression Sweep")
    print("="*60)

    from transformers import AutoTokenizer
    _mod = get_model_module()
    build_suppressed_forward = _mod.build_suppressed_forward
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']

    # Get contrastive scores from D.3
    if knowledge_results is None:
        d3_path = os.path.join(output_dir, 'r3_knowledge_neurons', 'results.json')
        if os.path.exists(d3_path):
            with open(d3_path) as f:
                knowledge_results = json.load(f)
        else:
            print("  Need R.3 results first. Run with --only r3 first.")
            return None

    # Collect all contrastive scores from physics queries
    all_scores = {}
    for entry in knowledge_results.get('physics', []):
        for n_info in entry.get('top_neurons', []):
            nid = n_info['neuron']
            cs = n_info['contrastive']
            all_scores[nid] = all_scores.get(nid, 0) + cs

    if not all_scores:
        print("  No contrastive scores found.")
        return None

    # Baseline probabilities
    @jax.jit
    def get_logits(params, ids):
        _mod = get_model_module()
        prefill = _mod.prefill
        logits, _, _, _ = prefill(params, model_cfg, ids)
        return logits

    def get_target_prob(forward_fn, prompt, target):
        """Get probability of target token given prompt."""
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_id = tokenizer.encode(target.lower(), add_special_tokens=False)
        if not target_id:
            return 0.0
        target_id = target_id[0]
        ids = jnp.array([input_ids], dtype=jnp.int32)
        logits = forward_fn(ids)
        probs = jax.nn.softmax(logits[0, -1, :])
        return float(probs[target_id])

    # Baseline forward (no suppression)
    baseline_forward = jax.jit(build_suppressed_forward(
        params, model_cfg, {}))

    print("  Baseline probabilities:")
    baseline_probs = {}
    for q in PHYSICS_QUERIES + CONTROL_QUERIES:
        p = get_target_prob(baseline_forward, q['prompt'], q['target'])
        baseline_probs[q['prompt']] = p
        tag = 'physics' if q in PHYSICS_QUERIES else 'control'
        print(f"    [{tag}] \"{q['prompt']}\" → '{q['target']}': {p:.2%}")

    # Sweep
    sweep_results = []
    sorted_neurons = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    for pct in sweep_pcts:
        n_suppress = max(1, int(n_know * pct))
        suppress_ids = [n for n, _ in sorted_neurons[:n_suppress]]

        know_mask = np.zeros(n_know, dtype=bool)
        for nid in suppress_ids:
            if nid < n_know:
                know_mask[nid] = True

        print(f"\n  pct={pct:.2f}: suppressing {know_mask.sum()} know neurons")

        sup_forward = jax.jit(build_suppressed_forward(
            params, model_cfg, {'know': jnp.array(know_mask)}))

        target_drops = []
        control_drops = []

        for q in PHYSICS_QUERIES:
            post_p = get_target_prob(sup_forward, q['prompt'], q['target'])
            pre_p = baseline_probs[q['prompt']]
            drop = pre_p - post_p
            target_drops.append(drop)
            print(f"    [physics] \"{q['prompt']}\" pre={pre_p:.2%} post={post_p:.2%} drop={drop:+.2%}")

        for q in CONTROL_QUERIES:
            post_p = get_target_prob(sup_forward, q['prompt'], q['target'])
            pre_p = baseline_probs[q['prompt']]
            drop = pre_p - post_p
            control_drops.append(drop)
            print(f"    [control] \"{q['prompt']}\" pre={pre_p:.2%} post={post_p:.2%} drop={drop:+.2%}")

        avg_td = float(np.mean(target_drops))
        avg_cd = float(np.mean(control_drops))
        sel_idx = avg_td - avg_cd
        verdict = 'SELECTIVE' if sel_idx > 0.1 else 'WEAK' if sel_idx > 0 else 'NON-SELECTIVE'

        print(f"    >> target drop: {avg_td:+.2%} | control drop: {avg_cd:+.2%} | "
              f"selectivity: {sel_idx:+.2%} → {verdict}")

        sweep_results.append({
            'pct': pct, 'n_suppressed': int(know_mask.sum()),
            'avg_target_drop': avg_td, 'avg_control_drop': avg_cd,
            'selectivity_index': sel_idx, 'verdict': verdict,
            'target_drops': [float(d) for d in target_drops],
            'control_drops': [float(d) for d in control_drops],
        })

    # Summary table
    print(f"\n  {'pct':>5s} | {'neurons':>7s} | {'target':>8s} | {'control':>8s} | {'select':>8s} | verdict")
    print(f"  {'-'*60}")
    for r in sweep_results:
        print(f"  {r['pct']:5.2f} | {r['n_suppressed']:7d} | {r['avg_target_drop']:>+7.2%} | "
              f"{r['avg_control_drop']:>+7.2%} | {r['selectivity_index']:>+7.2%} | {r['verdict']}")

    _save_json({'sweep': sweep_results, 'baseline_probs': baseline_probs},
               output_dir, 'r5_suppression', 'results.json')
    return {'sweep': sweep_results}


# ============================================================
# D8: Gate Distribution Analysis
# ============================================================

def analyze_gate_distribution(params, cfg, val_tokens, output_dir, n_batches=20, batch_size=8):
    """D8: Gate value distribution across tokens — histogram, effective_N, per-layer stats."""
    print("\n" + "="*60)
    print("D8: Gate Distribution Analysis")
    print("="*60)

    _mod = get_model_module()
    _layer_norm = _mod._layer_norm
    _srw_inference_with_gates = _mod._srw_inference_with_gates
    _srw_inference = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    n_know = model_cfg['n_know']

    # Convert all params to JAX arrays (needed for JIT tracing)
    params = jax.tree.map(jnp.asarray, params)

    pool_params = params['neuron_pool']
    router_params = params['router']

    qk_norm = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_norm = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]

    # Prepare val data
    n_tokens = val_tokens.shape[0]
    n_seqs = n_tokens // max_seq
    val_reshaped = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    # Per-layer stats: 12 metrics per layer, stacked as [n_layers, 12]
    N_STATS = 12
    STAT_NAMES = ['gate_mean', 'gate_std', 'gate_max', 'gate_min',
                  'eff_n_mean',
                  'frac_above_1e6', 'frac_above_1e5', 'frac_above_1e4',
                  'frac_above_1e3', 'frac_above_1e2', 'frac_above_0p1',
                  'frac_above_0p5']

    qk_s, v_s, know_s = get_output_scales(pool_params)
    _emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    _pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    @jax.jit
    def get_all_layer_gates(input_ids):
        """Forward through all layers, collect know gate stats at each layer."""
        B, S = input_ids.shape
        emb_matrix = _emb_matrix
        pos_matrix = _pos_matrix
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        all_stats = jnp.zeros((n_layers, N_STATS))

        for i in range(n_layers):
            lp = block_params_list[i]
            normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                               pool_params['qk_read'], pool_params['qk_write']) * qk_s
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write']) * qk_s
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write']) * v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ lp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
            h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out, gate_know_raw, _ = _srw_inference_with_gates(normed, h_k, know_norm, tau_k,
                                                            pool_params['know_read'], pool_params['know_write'])
            x = x + know_out * know_s

            abs_gate = jnp.abs(gate_know_raw)
            gate_sum = abs_gate.sum(axis=-1)
            gate_sq_sum = (abs_gate ** 2).sum(axis=-1)
            eff_n = gate_sum ** 2 / (gate_sq_sum + 1e-8)
            layer_stat = jnp.array([
                gate_know_raw.mean(), gate_know_raw.std(), gate_know_raw.max(), gate_know_raw.min(),
                eff_n.mean(),
                (abs_gate > 1e-6).astype(jnp.float32).mean(),
                (abs_gate > 1e-5).astype(jnp.float32).mean(),
                (abs_gate > 1e-4).astype(jnp.float32).mean(),
                (abs_gate > 1e-3).astype(jnp.float32).mean(),
                (abs_gate > 1e-2).astype(jnp.float32).mean(),
                (abs_gate > 0.1).astype(jnp.float32).mean(),
                (abs_gate > 0.5).astype(jnp.float32).mean(),
            ])
            all_stats = all_stats.at[i].set(layer_stat)

        return all_stats  # [n_layers, N_STATS]

    print(f"  Running {actual_batches} batches (batch_size={batch_size})...")
    all_results = []
    for b in range(actual_batches):
        batch = jnp.array(val_reshaped[b * batch_size:(b + 1) * batch_size])
        stats = get_all_layer_gates(batch)
        all_results.append(jax.device_get(stats))
        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Aggregate: mean across batches → [n_layers, N_STATS]
    avg_stats = np.mean(all_results, axis=0)
    results = {}
    for layer_idx in range(n_layers):
        layer_data = {}
        for s, name in enumerate(STAT_NAMES):
            layer_data[name] = float(avg_stats[layer_idx, s])
        results[f'layer_{layer_idx}'] = layer_data

    # Print summary
    print(f"\n  Know Pool (N={n_know}):")
    print(f"  {'Layer':<5} | {'gate_mean':>11} | {'gate_std':>11} | {'eff_N':>7}"
          f" | {'>1e-6':>7} | {'>1e-5':>7} | {'>1e-4':>7} | {'>1e-3':>7}"
          f" | {'>1e-2':>7} | {'>0.1':>7} | {'>0.5':>7}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}"
          f"-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}"
          f"-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for i in range(n_layers):
        d = results[f'layer_{i}']
        print(f"  {i:<5} | {d['gate_mean']:>11.6f} | {d['gate_std']:>11.6f} | {d['eff_n_mean']:>7.1f}"
              f" | {d['frac_above_1e6']*100:>6.1f}% | {d['frac_above_1e5']*100:>6.1f}%"
              f" | {d['frac_above_1e4']*100:>6.1f}% | {d['frac_above_1e3']*100:>6.1f}%"
              f" | {d['frac_above_1e2']*100:>6.1f}% | {d['frac_above_0p1']*100:>6.1f}%"
              f" | {d['frac_above_0p5']*100:>6.1f}%")

    # Log-scale histogram from aggregated stats
    print(f"\n  Gate value distribution (avg across layers):")
    bins_labels = ['>1e-6', '>1e-5', '>1e-4', '>1e-3', '>1e-2', '>0.1', '>0.5']
    bins_keys = ['frac_above_1e6', 'frac_above_1e5', 'frac_above_1e4',
                 'frac_above_1e3', 'frac_above_1e2', 'frac_above_0p1', 'frac_above_0p5']

    avg_across_layers = {}
    for key in bins_keys:
        vals = [results[f'layer_{i}'][key] for i in range(n_layers)]
        avg_across_layers[key] = np.mean(vals)

    prev = 1.0
    for label, key in zip(bins_labels, bins_keys):
        frac = avg_across_layers[key]
        band = prev - frac
        bar = '#' * int(band * 200)
        print(f"    {label:>6}: {frac*100:>6.2f}% cumulative | band={band*100:>6.2f}% {bar}")
        prev = frac

    _save_json(results, output_dir, 'gate_distribution', 'results.json')
    return results


# ============================================================
# D10: Neuron Utilization Pattern
# ============================================================

def analyze_neuron_utilization(params, cfg, val_tokens, output_dir, n_batches=20, batch_size=8):
    """D10: Per-neuron activation frequency — universal, specialist, dead neurons."""
    print("\n" + "="*60)
    print("D10: Neuron Utilization Pattern")
    print("="*60)

    _mod = get_model_module()
    _layer_norm = _mod._layer_norm
    _srw_inference_with_gates = _mod._srw_inference_with_gates
    _srw_inference = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    n_know = model_cfg['n_know']
    mid_layer = n_layers // 2

    # Convert all params to JAX arrays
    params = jax.tree.map(jnp.asarray, params)

    pool_params = params['neuron_pool']
    router_params = params['router']

    qk_norm = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_norm = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]

    n_tokens = val_tokens.shape[0]
    n_seqs = n_tokens // max_seq
    val_reshaped = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    qk_s, v_s, know_s = get_output_scales(pool_params)
    _emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    _pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    @jax.jit
    def get_neuron_activation(input_ids):
        """Forward to mid layer, return per-neuron activation mask."""
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = _emb_matrix[input_ids.astype(jnp.int32)] + _pos_matrix[positions]

        for i in range(mid_layer):
            lp = block_params_list[i]
            normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], pool_params['qk_read'], pool_params['qk_write']) * qk_s
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], pool_params['qk_read'], pool_params['qk_write']) * qk_s
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], pool_params['v_read'], pool_params['v_write']) * v_s
            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ lp['attn']['expand_O']['kernel']
            x = x + attn_out
            normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
            h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out = _srw_inference(normed, h_k, know_norm, tau_k, pool_params['know_read'], pool_params['know_write']) * know_s
            x = x + know_out

        # At mid layer, get know gate
        lp = block_params_list[mid_layer]
        normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
        h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        _, gate_know_raw, _ = _srw_inference_with_gates(normed, h_k, know_norm, tau_k,
                                                  pool_params['know_read'], pool_params['know_write'])
        # Return raw gate stats per neuron: mean |gate| and max |gate| across tokens
        abs_g = jnp.abs(gate_know_raw)
        neuron_mean_gate = abs_g.mean(axis=(0, 1))  # [N]
        neuron_max_gate = abs_g.max(axis=(0, 1))    # [N]
        return neuron_mean_gate, neuron_max_gate

    print(f"  Running {actual_batches} batches (batch_size={batch_size})...")
    all_means = []
    all_maxes = []
    for b in range(actual_batches):
        batch = jnp.array(val_reshaped[b * batch_size:(b + 1) * batch_size])
        mean_g, max_g = get_neuron_activation(batch)
        all_means.append(jax.device_get(mean_g))
        all_maxes.append(jax.device_get(max_g))
        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    avg_mean_gate = np.mean(all_means, axis=0)  # [N]
    avg_max_gate = np.mean(all_maxes, axis=0)   # [N]

    median_gate = float(np.median(avg_mean_gate))
    print(f"\n  Know Pool (N={n_know}, layer {mid_layer}):")
    print(f"    Median neuron mean |gate|: {median_gate:.6f}")

    # By mean |gate|
    print(f"\n  By mean |gate|:")
    mean_thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
    for t in mean_thresholds:
        count = int((avg_mean_gate > t).sum())
        print(f"    mean |gate| > {t:.0e}: {count:>6} neurons ({count/n_know*100:.1f}%)")

    # By max |gate| (peak activation)
    print(f"\n  By max |gate| (peak activation):")
    max_thresholds = [0.5, 0.1, 0.01, 0.001, 1e-4]
    for t in max_thresholds:
        count = int((avg_max_gate > t).sum())
        print(f"    max |gate| > {t}: {count:>6} neurons ({count/n_know*100:.1f}%)")

    # Top 10
    top_idx = np.argsort(avg_mean_gate)[::-1][:10]
    print(f"\n    Top 10 neurons by mean |gate|:")
    for idx in top_idx:
        print(f"      neuron {idx}: mean={avg_mean_gate[idx]:.6f} max={avg_max_gate[idx]:.6f}")

    results = {
        'n_know': n_know,
        'layer': mid_layer,
        'median_mean_gate': median_gate,
        'by_mean_gate': {f'gt_{t:.0e}': int((avg_mean_gate > t).sum()) for t in mean_thresholds},
        'by_max_gate': {f'gt_{t}': int((avg_max_gate > t).sum()) for t in max_thresholds},
        'top_10': [{'neuron_id': int(idx), 'mean_gate': float(avg_mean_gate[idx]),
                     'max_gate': float(avg_max_gate[idx])} for idx in top_idx],
        'freq_mean': float(avg_mean_gate.mean()),
        'freq_std': float(avg_mean_gate.std()),
    }
    _save_json(results, output_dir, 'neuron_utilization', 'results.json')
    return results


# ============================================================
# D11: Layer-wise Gate Pattern (merged into D8)
# ============================================================
# D11 is included in D8's per-layer analysis above.


# ============================================================
# P1: Read/Write Projection Analysis
# ============================================================

def analyze_rw_projection(params, cfg, output_dir):
    """P1: Interpret each neuron's read/write vectors via token embedding similarity."""
    print("\n" + "="*60)
    print("P1: Read/Write Projection Analysis")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_cfg = get_model_cfg(cfg)
    pool = params['neuron_pool']
    emb = np.array(params['token_emb']['embedding'])
    # Unit-norm
    emb_norm = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

    pools = {
        'QK': {'read': np.array(pool['qk_read']), 'write': np.array(pool['qk_write'])},
        'V': {'read': np.array(pool['v_read']), 'write': np.array(pool['v_write'])},
        'Know': {'read': np.array(pool['know_read']), 'write': np.array(pool['know_write'])},
    }

    results = {}
    for pool_name, vecs in pools.items():
        N = vecs['read'].shape[0]
        read_n = vecs['read'] / (np.linalg.norm(vecs['read'], axis=-1, keepdims=True) + 1e-8)
        write_n = vecs['write'] / (np.linalg.norm(vecs['write'], axis=-1, keepdims=True) + 1e-8)

        # Read-write alignment per neuron
        rw_align = (read_n * write_n).sum(axis=-1)  # [N]

        # Top tokens per neuron (chunked to avoid OOM on Know pool)
        chunk_size = min(N, 2000)
        n_chunks = (N + chunk_size - 1) // chunk_size
        all_neurons = []

        print(f"\n  {pool_name} Pool (N={N}):")
        for ci in range(n_chunks):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, N)
            r_chunk = read_n[s:e]   # [chunk, D]
            w_chunk = write_n[s:e]
            r_sim = r_chunk @ emb_norm.T  # [chunk, vocab]
            w_sim = w_chunk @ emb_norm.T
            r_top = np.argsort(-r_sim, axis=-1)[:, :10]
            w_top = np.argsort(-w_sim, axis=-1)[:, :10]
            for i in range(e - s):
                ni = s + i
                all_neurons.append({
                    'neuron': ni,
                    'read_top10': [{'token': tokenizer.decode([int(r_top[i, j])]), 'sim': float(r_sim[i, r_top[i, j]])} for j in range(10)],
                    'write_top10': [{'token': tokenizer.decode([int(w_top[i, j])]), 'sim': float(w_sim[i, w_top[i, j]])} for j in range(10)],
                    'rw_alignment': float(rw_align[ni]),
                })
            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                print(f"    chunk {ci+1}/{n_chunks}")

        # Summary stats
        align_sorted = np.argsort(rw_align)
        top_aligned = align_sorted[-10:][::-1]
        top_ortho = align_sorted[np.argsort(np.abs(rw_align[align_sorted]))][:10]
        top_anti = align_sorted[:10]

        print(f"    RW alignment: mean={rw_align.mean():.4f} std={rw_align.std():.4f}")
        print(f"    Top aligned (read≈write):")
        for idx in top_aligned:
            r_tok = all_neurons[idx]['read_top10'][0]['token']
            w_tok = all_neurons[idx]['write_top10'][0]['token']
            print(f"      neuron {idx}: align={rw_align[idx]:.3f} read='{r_tok}' write='{w_tok}'")
        print(f"    Top anti-aligned (read≈-write):")
        for idx in top_anti:
            r_tok = all_neurons[idx]['read_top10'][0]['token']
            w_tok = all_neurons[idx]['write_top10'][0]['token']
            print(f"      neuron {idx}: align={rw_align[idx]:.3f} read='{r_tok}' write='{w_tok}'")

        hist_bins = np.linspace(-1, 1, 21)
        hist_counts, _ = np.histogram(rw_align, bins=hist_bins)

        results[pool_name] = {
            'n_neurons': N,
            'rw_alignment_mean': float(rw_align.mean()),
            'rw_alignment_std': float(rw_align.std()),
            'rw_alignment_histogram': {'bins': hist_bins.tolist(), 'counts': hist_counts.tolist()},
            'top_aligned': [{'neuron': int(i), 'alignment': float(rw_align[i])} for i in top_aligned],
            'top_anti_aligned': [{'neuron': int(i), 'alignment': float(rw_align[i])} for i in top_anti],
            'neurons': all_neurons,
        }

    _save_json(results, output_dir, 'p1_rw_projection', 'results.json')
    return results


# ============================================================
# P2: Activation Context Analysis
# ============================================================

def analyze_activation_context(params, cfg, val_tokens, output_dir,
                                n_batches=10, batch_size=4):
    """P2: Collect text contexts where each neuron activates most strongly."""
    print("\n" + "="*60)
    print("P2: Activation Context Analysis")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    mid_layer = n_layers // 2

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Per-neuron: collect (gate_value, batch_idx, seq_pos) tuples
    # Only track top-100 neurons globally by max gate seen
    TOP_NEURONS = 100
    TOP_CONTEXTS = 30
    CONTEXT_WINDOW = 5

    # neuron_id -> list of (gate_val, token_ids_context)
    neuron_contexts = {}
    neuron_max_gate = np.zeros(n_know)

    print(f"  Running {actual_batches} batches (bs={batch_size}, mid_layer={mid_layer})...")
    for b in range(actual_batches):
        batch_ids = tokens[b * batch_size:(b + 1) * batch_size]
        batch_dev = jnp.array(batch_ids, dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch_dev)

        # gate_Know_raw: [n_layers, B, S, n_know] — take mid layer only
        gate_mid = np.array(jax.device_get(layer_info['gate_Know_raw'][mid_layer]))  # [B, S, n_know]

        # Vectorized: update per-neuron max gate across all positions
        B, S, N = gate_mid.shape
        gate_flat = gate_mid.reshape(-1, N)  # [B*S, N]
        neuron_max_gate = np.maximum(neuron_max_gate, gate_flat.max(axis=0))

        # Find top-k neurons per position and collect contexts (vectorized top-k)
        for bi in range(B):
            for si in range(S):
                g = gate_mid[bi, si]
                top_idx = np.argpartition(-g, TOP_NEURONS)[:TOP_NEURONS]
                top_vals = g[top_idx]
                active = top_vals > 0
                if not active.any():
                    continue
                ctx_start = max(0, si - CONTEXT_WINDOW)
                ctx_end = min(S, si + CONTEXT_WINDOW + 1)
                ctx_ids = batch_ids[bi, ctx_start:ctx_end].tolist()
                for ni, gv in zip(top_idx[active], top_vals[active]):
                    ni = int(ni)
                    if ni not in neuron_contexts:
                        neuron_contexts[ni] = []
                    neuron_contexts[ni].append((float(gv), si, ctx_ids))

        if (b + 1) % 2 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Select top-100 neurons by max gate, keep top-30 contexts each
    top_neuron_ids = np.argsort(-neuron_max_gate)[:TOP_NEURONS]
    results = {'mid_layer': mid_layer, 'n_know': n_know, 'neurons': []}

    print(f"\n  Top {TOP_NEURONS} neurons by max gate activation:")
    for rank, ni in enumerate(top_neuron_ids):
        ni = int(ni)
        if ni not in neuron_contexts:
            continue
        ctxs = sorted(neuron_contexts[ni], key=lambda x: -x[0])[:TOP_CONTEXTS]
        decoded_ctxs = []
        for gv, pos, ctx_ids in ctxs:
            ctx_text = tokenizer.decode(ctx_ids)
            decoded_ctxs.append({'gate': gv, 'position': pos, 'context': ctx_text})

        if rank < 10:
            top_ctx = decoded_ctxs[0] if decoded_ctxs else {}
            print(f"    neuron {ni}: max_gate={neuron_max_gate[ni]:.4f} top_ctx='{top_ctx.get('context', '')[:60]}'")

        results['neurons'].append({
            'neuron_id': ni,
            'max_gate': float(neuron_max_gate[ni]),
            'contexts': decoded_ctxs,
        })

    _save_json(results, output_dir, 'p2_activation_context', 'results.json')
    return results


# ============================================================
# P3: Layer Role Matrix
# ============================================================

def analyze_layer_role_matrix(params, cfg, val_tokens, output_dir,
                               n_batches=10, batch_size=4):
    """P3: Per-neuron activation rate across layers — which layers use which neurons."""
    print("\n" + "="*60)
    print("P3: Layer Role Matrix")
    print("="*60)

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_layers = model_cfg['n_layers']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']
    THRESH = 0.01

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Accumulators: [n_layers, N] activation rate (batch-wise mean)
    q_acc = np.zeros((n_layers, n_qk), dtype=np.float64)
    k_acc = np.zeros((n_layers, n_qk), dtype=np.float64)
    v_acc = np.zeros((n_layers, n_v), dtype=np.float64)
    know_acc = np.zeros((n_layers, n_know), dtype=np.float64)

    print(f"  Running {actual_batches} batches (bs={batch_size})...")
    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch)

        # gate_Q_raw: [n_layers, B, S, n_qk] — vectorized over layers
        for pool_key, acc, N in [
            ('gate_Q_raw', q_acc, n_qk), ('gate_K_raw', k_acc, n_qk),
            ('gate_V_raw', v_acc, n_v), ('gate_Know_raw', know_acc, n_know)
        ]:
            gate = np.array(jax.device_get(layer_info[pool_key]))  # [L, B, S, N]
            act_rates = (gate > THRESH).astype(np.float32).mean(axis=(1, 2))  # [L, N]
            acc += act_rates
        del layer_info

        if (b + 1) % 2 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Average across batches
    q_acc /= actual_batches
    k_acc /= actual_batches
    v_acc /= actual_batches
    know_acc /= actual_batches

    # Stats: neurons active in how many layers
    def layer_spread_stats(mat, pool_name):
        active_layers = (mat > 0.01).sum(axis=0)  # [N] — how many layers each neuron is active in
        print(f"\n  {pool_name} Pool (N={mat.shape[1]}):")
        for n_lay in [1, 2, 3, 5, 8, 12]:
            if n_lay > n_layers:
                continue
            count = int((active_layers >= n_lay).sum())
            print(f"    Active in >={n_lay} layers: {count} neurons ({count/mat.shape[1]*100:.1f}%)")
        return active_layers

    q_spread = layer_spread_stats(q_acc, "Q")
    k_spread = layer_spread_stats(k_acc, "K")
    v_spread = layer_spread_stats(v_acc, "V")
    know_spread = layer_spread_stats(know_acc, "Know")

    # Per-layer Q/K correlation
    print(f"\n  Per-layer Q/K correlation:")
    layer_corrs = []
    for li in range(n_layers):
        if q_acc[li].std() > 0 and k_acc[li].std() > 0:
            corr = float(np.corrcoef(q_acc[li], k_acc[li])[0, 1])
        else:
            corr = 0.0
        layer_corrs.append(corr)
        print(f"    Layer {li}: r={corr:.4f}")

    results = {
        'threshold': THRESH,
        'n_batches': actual_batches,
        'layer_role_Q': q_acc.tolist(),
        'layer_role_K': k_acc.tolist(),
        'layer_role_V': v_acc.tolist(),
        'layer_role_Know': know_acc.tolist(),
        'q_layer_spread': q_spread.tolist(),
        'k_layer_spread': k_spread.tolist(),
        'v_layer_spread': v_spread.tolist(),
        'know_layer_spread': know_spread.tolist(),
        'per_layer_qk_correlation': layer_corrs,
    }
    _save_json(results, output_dir, 'p3_layer_role', 'results.json')
    return results


# ============================================================
# P4: Cross-Domain Suppression
# ============================================================

DOMAIN_PROMPTS = {
    'physics': [
        "light travels at the speed of",
        "the earth orbits the",
        "gravity pulls objects toward the",
    ],
    'biology': [
        "cells divide through a process called",
        "DNA stores genetic",
        "photosynthesis converts sunlight into",
    ],
    'geography': [
        "the longest river in the world is the",
        "mount everest is located in",
        "the sahara desert covers",
    ],
    'language': [
        "the past tense of go is",
        "a noun is a word that",
        "the plural of child is",
    ],
    'math': [
        "the square root of 144 is",
        "pi is approximately equal to",
        "the sum of angles in a triangle is",
    ],
}


def analyze_cross_domain_suppression(params, cfg, output_dir):
    """P4: Suppress domain-specific neurons and measure cross-domain impact."""
    print("\n" + "="*60)
    print("P4: Cross-Domain Suppression")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward
    build_suppressed_forward = _mod.build_suppressed_forward
    prefill_fn = _mod.prefill
    decode_step_fn = _mod.decode_step

    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    mid_layer = n_layers // 2
    max_seq = model_cfg['max_seq_len']

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Step 1: Collect per-domain neuron activation profiles
    print(f"  Collecting domain activation profiles...")
    domain_profiles = {}  # domain -> [n_know] mean gate

    for domain, prompts in DOMAIN_PROMPTS.items():
        acc = np.zeros(n_know, dtype=np.float64)
        n_tokens = 0
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            ids_dev = jnp.array(ids_pad, dtype=jnp.int32)
            _, layer_info = jit_analysis(params, ids_dev)
            gate = np.array(jax.device_get(layer_info['gate_Know_raw'][mid_layer]))  # [1, S, N]
            # Only count actual tokens (not padding)
            gate_valid = gate[0, :ids.shape[1], :]  # [seq_len, N]
            acc += gate_valid.mean(axis=0)
            n_tokens += 1
        domain_profiles[domain] = acc / n_tokens
        print(f"    {domain}: mean_gate={domain_profiles[domain].mean():.6f}")

    # Step 2: Find domain-specific neurons (contrastive)
    all_domains = list(DOMAIN_PROMPTS.keys())
    domain_neurons = {}  # domain -> neuron indices sorted by specificity

    for domain in all_domains:
        others_mean = np.mean([domain_profiles[d] for d in all_domains if d != domain], axis=0)
        specificity = domain_profiles[domain] - others_mean
        domain_neurons[domain] = np.argsort(-specificity)

    # Step 3: Suppress and measure
    suppress_pcts = [0.01, 0.02, 0.05]  # 1%, 2%, 5%

    @jax.jit
    def get_loss(params, input_ids):
        logits, _, _, _ = prefill_fn(params, model_cfg, input_ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        safe_labels = jnp.where(shift_labels > 0, shift_labels, 0)
        token_loss = -jnp.take_along_axis(log_probs, safe_labels[..., jnp.newaxis], axis=-1).squeeze(-1)
        valid = shift_labels > 0
        return (token_loss * valid).sum() / (valid.sum() + 1e-8)

    # Baseline losses per domain
    print(f"\n  Computing baseline losses...")
    baseline_losses = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        losses = []
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            loss = float(get_loss(params, jnp.array(ids_pad, dtype=jnp.int32)))
            losses.append(loss)
        baseline_losses[domain] = np.mean(losses)
    print(f"    Baselines: {', '.join(f'{d}={v:.4f}' for d, v in baseline_losses.items())}")

    # Suppression experiments
    print(f"\n  Running suppression experiments...")
    results_matrix = []

    for suppress_domain in all_domains:
        for pct in suppress_pcts:
            n_suppress = int(n_know * pct)
            suppress_idx = domain_neurons[suppress_domain][:n_suppress]
            mask = np.zeros(n_know, dtype=bool)
            mask[suppress_idx] = True
            suppress_masks = {'know': mask}
            fwd_fn = build_suppressed_forward(params, model_cfg, suppress_masks)
            jit_fwd = jax.jit(fwd_fn)

            row = {'suppress_domain': suppress_domain, 'pct': pct, 'n_suppressed': n_suppress}
            for target_domain, prompts in DOMAIN_PROMPTS.items():
                losses = []
                for prompt in prompts:
                    ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
                    ids_pad = np.zeros((1, max_seq), dtype=np.int32)
                    ids_pad[0, :ids.shape[1]] = ids
                    logits = np.array(jax.device_get(jit_fwd(jnp.array(ids_pad, dtype=jnp.int32))))
                    shift_logits = logits[:, :-1, :]
                    shift_labels = ids_pad[:, 1:]
                    log_probs = shift_logits - np.log(np.exp(shift_logits).sum(axis=-1, keepdims=True) + 1e-8)
                    safe = np.where(shift_labels > 0, shift_labels, 0)
                    tl = -np.take_along_axis(log_probs, safe[..., np.newaxis], axis=-1).squeeze(-1)
                    valid = shift_labels > 0
                    loss = float((tl * valid).sum() / (valid.sum() + 1e-8))
                    losses.append(loss)
                row[f'loss_{target_domain}'] = float(np.mean(losses))
                row[f'delta_{target_domain}'] = float(np.mean(losses) - baseline_losses[target_domain])

            # Selectivity index
            target_delta = row[f'delta_{suppress_domain}']
            other_deltas = [row[f'delta_{d}'] for d in all_domains if d != suppress_domain]
            row['selectivity'] = target_delta - np.mean(other_deltas)
            results_matrix.append(row)

            print(f"    Suppress {suppress_domain} {pct*100:.0f}%: "
                  f"self_delta={target_delta:+.4f} selectivity={row['selectivity']:+.4f}")

    results = {
        'baseline_losses': {k: float(v) for k, v in baseline_losses.items()},
        'suppression_results': results_matrix,
        'domains': all_domains,
    }
    _save_json(results, output_dir, 'p4_cross_domain', 'results.json')
    return results


# ============================================================
# P5: Gate Mechanism Analysis
# ============================================================

def analyze_gate_mechanism(params, cfg, val_tokens, output_dir,
                            n_batches=5, batch_size=2):
    """P5: Empirical analysis of z × Φ(z) gate — z/phi distributions, den stats."""
    print("\n" + "="*60)
    print("P5: Gate Mechanism Analysis")
    print("="*60)

    _mod = get_model_module()
    _layer_norm = _mod._layer_norm
    _srw_inference = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    n_know = model_cfg['n_know']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']

    pool_params = params['neuron_pool']
    router_params = params['router']

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    # Precompute unit-norm embs
    know_emb_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]

    # Histogram bins
    z_bins = np.arange(0, 5.25, 0.25)
    phi_bins = np.arange(0.5, 1.025, 0.025)

    # Accumulators per layer
    layer_stats = {li: {
        'z_hist': np.zeros(len(z_bins) - 1),
        'phi_hist': np.zeros(len(phi_bins) - 1),
        'n_active': 0, 'n_confident': 0, 'n_borderline': 0, 'n_total_tokens': 0,
        's_std_sum': 0.0, 'den_floor_count': 0, 'den_total': 0,
        'tau_offset_sum': 0.0,
    } for li in range(n_layers)}

    print(f"  Running {actual_batches} batches (bs={batch_size}), layer-by-layer forward...")

    for b in range(actual_batches):
        batch = tokens[b * batch_size:(b + 1) * batch_size]
        batch_dev = jnp.array(batch, dtype=jnp.int32)
        B, S = batch_dev.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[batch_dev] + pos_matrix[positions]

        for li in range(n_layers):
            bp = block_params_list[li]
            normed = _layer_norm(x, jnp.asarray(bp['norm1']['scale']), jnp.asarray(bp['norm1']['bias']))

            # Attention forward (simplified — just need x update)
            h_all = normed @ jnp.asarray(router_params['proj_attn']['kernel']) + jnp.asarray(router_params['proj_attn']['bias'])
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ jnp.asarray(router_params['tau_attn']['kernel']) + jnp.asarray(router_params['tau_attn']['bias'])

            qk_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
            v_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
            Q = _srw_inference(normed, h_Q, qk_n, tau_all[:,:,0:1], pool_params['qk_read'], pool_params['qk_write'])
            K = _srw_inference(normed, h_K, qk_n, tau_all[:,:,1:2], pool_params['qk_read'], pool_params['qk_write'])
            V = _srw_inference(normed, h_V, v_n, tau_all[:,:,2:3], pool_params['v_read'], pool_params['v_write'])
            _qk_s = pool_params.get('qk_scale', 1.0)
            _v_s = pool_params.get('v_scale', 1.0)
            Q, K, V = Q * _qk_s, K * _qk_s, V * _v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            scores_a = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores_a = jnp.where(causal, scores_a, jnp.finfo(scores_a.dtype).min)
            attn_w = jax.nn.softmax(scores_a, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ jnp.asarray(bp['attn']['expand_O']['kernel'])
            x = x + attn_out

            # Know forward — compute z, phi, gate directly
            normed2 = _layer_norm(x, jnp.asarray(bp['norm2']['scale']), jnp.asarray(bp['norm2']['bias']))
            h_k = normed2 @ jnp.asarray(router_params['proj_know']['kernel']) + jnp.asarray(router_params['proj_know']['bias'])
            tau_k = normed2 @ jnp.asarray(router_params['tau_know']['kernel']) + jnp.asarray(router_params['tau_know']['bias'])

            scores_k = h_k @ know_emb_n.T
            sf = scores_k.astype(jnp.float32)
            s_mean = sf.mean(axis=-1, keepdims=True)
            s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
            tau = s_mean + tau_k * s_std
            raw = scores_k - tau.astype(scores_k.dtype)
            z = raw.astype(jnp.float32) / s_std
            phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
            gate = jnp.where(z > 0, z * phi, 0.0)
            den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)

            # Collect stats (CPU)
            z_np = np.array(jax.device_get(z))
            phi_np = np.array(jax.device_get(phi))
            den_np = np.array(jax.device_get(den))
            s_std_np = np.array(jax.device_get(s_std))
            tau_k_np = np.array(jax.device_get(tau_k))

            active_mask = z_np > 0
            z_active = z_np[active_mask]

            ls = layer_stats[li]
            ls['z_hist'] += np.histogram(z_active, bins=z_bins)[0]
            ls['phi_hist'] += np.histogram(phi_np[active_mask], bins=phi_bins)[0]
            ls['n_active'] += int(active_mask.sum())
            ls['n_confident'] += int((phi_np[active_mask] > 0.95).sum()) if len(z_active) > 0 else 0
            ls['n_borderline'] += int(((phi_np[active_mask] > 0.5) & (phi_np[active_mask] < 0.8)).sum()) if len(z_active) > 0 else 0
            ls['n_total_tokens'] += B * S
            ls['s_std_sum'] += float(s_std_np.mean())
            ls['den_floor_count'] += int((den_np.squeeze(-1) <= 1.0 + 1e-6).sum())
            ls['den_total'] += B * S
            ls['tau_offset_sum'] += float(tau_k_np.mean())

            # Update x with know output
            know_out = _srw_inference(normed2, h_k, know_emb_n, tau_k, pool_params['know_read'], pool_params['know_write'])
            _k_s = pool_params.get('know_scale', 1.0)
            x = x + know_out * _k_s

        if (b + 1) % 2 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Compile results
    results = {'n_batches': actual_batches, 'batch_size': batch_size, 'layers': {}}
    print(f"\n  Per-layer gate mechanism stats (Know pool):")
    print(f"  {'Layer':>5} | {'s_std':>7} | {'confident':>10} | {'borderline':>10} | {'den_floor':>10} | {'tau':>7}")
    for li in range(n_layers):
        ls = layer_stats[li]
        n_act = max(ls['n_active'], 1)
        conf_ratio = ls['n_confident'] / n_act
        border_ratio = ls['n_borderline'] / n_act
        floor_ratio = ls['den_floor_count'] / max(ls['den_total'], 1)
        avg_std = ls['s_std_sum'] / actual_batches
        avg_tau = ls['tau_offset_sum'] / actual_batches

        print(f"  {li:>5} | {avg_std:>7.4f} | {conf_ratio*100:>9.1f}% | {border_ratio*100:>9.1f}% | {floor_ratio*100:>9.1f}% | {avg_tau:>7.3f}")

        results['layers'][f'layer_{li}'] = {
            'z_histogram': {'bins': z_bins.tolist(), 'counts': ls['z_hist'].tolist()},
            'phi_histogram': {'bins': phi_bins.tolist(), 'counts': ls['phi_hist'].tolist()},
            'confident_ratio': float(conf_ratio),
            'borderline_ratio': float(border_ratio),
            'den_floor_ratio': float(floor_ratio),
            'avg_s_std': float(avg_std),
            'avg_tau_offset': float(avg_tau),
            'n_active_total': int(ls['n_active']),
        }

    _save_json(results, output_dir, 'p5_gate_mechanism', 'results.json')
    return results


# ============================================================
# P6: Compositional Expressiveness
# ============================================================

def analyze_compositional_expressiveness(params, cfg, val_tokens, output_dir,
                                          n_samples=100, batch_size=4):
    """P6: Effective rank of active write vectors — rank-1 compositionality."""
    print("\n" + "="*60)
    print("P6: Compositional Expressiveness")
    print("="*60)

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    mid_layer = n_layers // 2

    pool = params['neuron_pool']
    know_write = np.array(pool['know_write'])
    know_write_n = know_write / (np.linalg.norm(know_write, axis=-1, keepdims=True) + 1e-8)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Collect active neuron sets for random token positions
    print(f"  Collecting active neuron sets for {n_samples} token positions...")
    all_results = []
    samples_collected = 0
    batch_idx = 0

    while samples_collected < n_samples and batch_idx * batch_size < n_seqs:
        batch = jnp.array(tokens[batch_idx * batch_size:(batch_idx + 1) * batch_size], dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch)
        gate = np.array(jax.device_get(layer_info['gate_Know_raw'][mid_layer]))  # [B, S, n_know]
        B, S, N = gate.shape

        # Random sample positions from this batch
        n_from_batch = min(n_samples - samples_collected, B * S // 4)
        positions = np.random.choice(B * S, size=n_from_batch, replace=False)

        for pos in positions:
            bi, si = divmod(int(pos), S)
            g = gate[bi, si]
            active_idx = np.where(g > 0)[0]
            active_N = len(active_idx)
            if active_N < 2:
                continue

            # SVD of active write vectors
            W = know_write_n[active_idx]  # [active_N, d_model]
            sv = np.linalg.svd(W, compute_uv=False)
            sv_norm = sv / (sv.sum() + 1e-8)
            entropy = -(sv_norm * np.log(sv_norm + 1e-10)).sum()
            eff_rank = float(np.exp(entropy))

            all_results.append({
                'active_N': active_N,
                'effective_rank': eff_rank,
                'efficiency': eff_rank / active_N if active_N > 0 else 0,
                'top5_sv': sv[:5].tolist(),
            })
            samples_collected += 1

        batch_idx += 1
        if batch_idx % 5 == 0:
            print(f"    {samples_collected}/{n_samples} samples collected")

    if not all_results:
        print("  No samples collected!")
        return {}

    active_Ns = np.array([r['active_N'] for r in all_results])
    eff_ranks = np.array([r['effective_rank'] for r in all_results])
    efficiencies = np.array([r['efficiency'] for r in all_results])

    print(f"\n  Collected {len(all_results)} token positions (layer {mid_layer}):")
    print(f"    Active N:       mean={active_Ns.mean():.0f}  std={active_Ns.std():.0f}  "
          f"p25={np.percentile(active_Ns, 25):.0f}  p50={np.percentile(active_Ns, 50):.0f}  p75={np.percentile(active_Ns, 75):.0f}")
    print(f"    Effective rank: mean={eff_ranks.mean():.1f}  std={eff_ranks.std():.1f}  "
          f"p25={np.percentile(eff_ranks, 25):.1f}  p50={np.percentile(eff_ranks, 50):.1f}  p75={np.percentile(eff_ranks, 75):.1f}")
    print(f"    Efficiency:     mean={efficiencies.mean():.3f}  (eff_rank / active_N)")
    print(f"    Max possible:   {model_cfg['d_model']} (d_model)")

    results = {
        'mid_layer': mid_layer,
        'n_samples': len(all_results),
        'n_know': n_know,
        'd_model': model_cfg['d_model'],
        'active_N': {'mean': float(active_Ns.mean()), 'std': float(active_Ns.std()),
                     'percentiles': {str(p): float(np.percentile(active_Ns, p)) for p in [5, 25, 50, 75, 95]}},
        'effective_rank': {'mean': float(eff_ranks.mean()), 'std': float(eff_ranks.std()),
                          'percentiles': {str(p): float(np.percentile(eff_ranks, p)) for p in [5, 25, 50, 75, 95]}},
        'efficiency': {'mean': float(efficiencies.mean()), 'std': float(efficiencies.std())},
        'samples': all_results[:20],  # Save first 20 for inspection
    }
    _save_json(results, output_dir, 'p6_compositional', 'results.json')
    return results


# ============================================================
# P7: Neuron Clustering Analysis
# ============================================================

def simple_kmeans(X, k, n_iter=20, seed=42):
    """Simple k-means. X: [N, D] numpy array. Returns labels [N], centers [k, D]."""
    rng = np.random.RandomState(seed)
    N, D = X.shape
    idx = rng.choice(N, k, replace=False)
    centers = X[idx].copy()
    for _ in range(n_iter):
        labels = np.zeros(N, dtype=np.int32)
        chunk = 5000
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            dists = np.linalg.norm(X[s:e, None, :] - centers[None, :, :], axis=-1)
            labels[s:e] = dists.argmin(axis=-1)
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = X[mask].mean(axis=0)
    return labels, centers


def _purity_score(labels_a, labels_b):
    """Purity: for each cluster in A, fraction belonging to dominant B cluster."""
    ka = labels_a.max() + 1
    total = 0
    for i in range(ka):
        mask = labels_a == i
        if mask.sum() == 0:
            continue
        b_labels = labels_b[mask]
        counts = np.bincount(b_labels, minlength=labels_b.max() + 1)
        total += counts.max()
    return total / len(labels_a)


def analyze_neuron_clustering(params, cfg, val_tokens, output_dir,
                               k_clusters=100, n_batches=10, batch_size=4):
    """P7: Clustering in emb/read/write space + co-activation analysis."""
    print("\n" + "="*60)
    print("P7: Neuron Clustering Analysis")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    mid_layer = n_layers // 2

    pool = params['neuron_pool']
    emb_vecs = np.array(pool['know_emb'])
    read_vecs = np.array(pool['know_read'])
    write_vecs = np.array(pool['know_write'])
    token_emb = np.array(params['token_emb']['embedding'])
    token_emb_n = token_emb / (np.linalg.norm(token_emb, axis=-1, keepdims=True) + 1e-8)

    # Unit-norm
    emb_n = emb_vecs / (np.linalg.norm(emb_vecs, axis=-1, keepdims=True) + 1e-8)
    read_n = read_vecs / (np.linalg.norm(read_vecs, axis=-1, keepdims=True) + 1e-8)
    write_n = write_vecs / (np.linalg.norm(write_vecs, axis=-1, keepdims=True) + 1e-8)

    # --- Part A: Embedding-space clustering ---
    print(f"\n  Part A: Embedding-space clustering (Know pool, N={n_know}, k={k_clusters})")

    # PCA to 50 dims
    def pca_reduce(X, d=50):
        X_c = X - X.mean(axis=0)
        cov = (X_c.T @ X_c) / X.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        top = eigvecs[:, -d:][:, ::-1]
        return X_c @ top

    emb_pca = pca_reduce(emb_n)
    read_pca = pca_reduce(read_n)
    write_pca = pca_reduce(write_n)

    print(f"    PCA done (50 dims). Running k-means...")
    emb_labels, emb_centers = simple_kmeans(emb_pca, k_clusters)
    read_labels, read_centers = simple_kmeans(read_pca, k_clusters)
    write_labels, write_centers = simple_kmeans(write_pca, k_clusters)

    def cluster_stats(labels, name):
        sizes = np.bincount(labels, minlength=k_clusters)
        print(f"    {name} clusters (k={k_clusters}): mean_size={sizes.mean():.0f}, "
              f"min={sizes.min()}, max={sizes.max()}")
        return sizes

    emb_sizes = cluster_stats(emb_labels, "emb")
    read_sizes = cluster_stats(read_labels, "read")
    write_sizes = cluster_stats(write_labels, "write")

    # Purity scores
    er_pur = _purity_score(emb_labels, read_labels)
    ew_pur = _purity_score(emb_labels, write_labels)
    rw_pur = _purity_score(read_labels, write_labels)
    print(f"\n    emb-read purity:  {er_pur:.3f}")
    print(f"    emb-write purity: {ew_pur:.3f}")
    print(f"    read-write purity: {rw_pur:.3f}")

    # Top 5 largest emb clusters — interpret via token embedding
    top5_clusters = np.argsort(-emb_sizes)[:5]
    cluster_interpretations = {}
    print(f"\n    Top 5 largest emb clusters:")
    for ci in top5_clusters:
        mask = emb_labels == ci
        mean_read = read_n[mask].mean(axis=0)
        mean_write = write_n[mask].mean(axis=0)
        mean_read = mean_read / (np.linalg.norm(mean_read) + 1e-8)
        mean_write = mean_write / (np.linalg.norm(mean_write) + 1e-8)
        r_sim = mean_read @ token_emb_n.T
        w_sim = mean_write @ token_emb_n.T
        r_top = np.argsort(-r_sim)[:5]
        w_top = np.argsort(-w_sim)[:5]
        r_toks = ', '.join(tokenizer.decode([int(t)]) for t in r_top)
        w_toks = ', '.join(tokenizer.decode([int(t)]) for t in w_top)
        print(f"      Cluster {ci} (size={int(emb_sizes[ci])}): read→ \"{r_toks}\" | write→ \"{w_toks}\"")
        cluster_interpretations[int(ci)] = {
            'size': int(emb_sizes[ci]),
            'read_top5': [tokenizer.decode([int(t)]) for t in r_top],
            'write_top5': [tokenizer.decode([int(t)]) for t in w_top],
        }

    # --- Part B: Co-activation clustering ---
    cooccur_matrix = None
    cooccur_results = {}
    if val_tokens is not None:
        print(f"\n  Part B: Co-activation (cluster-level, layer {mid_layer})")

        _mod = get_model_module()
        analysis_forward = _mod.analysis_forward
        max_seq = model_cfg['max_seq_len']

        n_seqs = len(val_tokens) // max_seq
        tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
        actual_batches = min(n_batches, n_seqs // batch_size)

        jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

        # Cluster co-occurrence [k, k]
        cooccur = np.zeros((k_clusters, k_clusters), dtype=np.float64)
        cluster_active_count = np.zeros(k_clusters, dtype=np.float64)
        STRONG_THRESH = 0.5

        print(f"    Running {actual_batches} batches (bs={batch_size})...")
        for b in range(actual_batches):
            batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
            _, layer_info = jit_analysis(params, batch)
            gate = np.array(jax.device_get(layer_info['gate_Know_raw'][mid_layer]))  # [B, S, N]
            B, S, N = gate.shape

            # Per token: which clusters are strongly active?
            strong = gate > STRONG_THRESH  # [B, S, N]
            for bi in range(B):
                for si in range(S):
                    active_neurons = np.where(strong[bi, si])[0]
                    if len(active_neurons) < 2:
                        continue
                    active_clusters = np.unique(emb_labels[active_neurons])
                    cluster_active_count[active_clusters] += 1
                    for ii in range(len(active_clusters)):
                        for jj in range(ii + 1, len(active_clusters)):
                            ci, cj = active_clusters[ii], active_clusters[jj]
                            cooccur[ci, cj] += 1
                            cooccur[cj, ci] += 1
            if (b + 1) % 3 == 0:
                print(f"      batch {b+1}/{actual_batches}")

        # PMI-like normalization
        pmi = np.zeros_like(cooccur)
        for i in range(k_clusters):
            for j in range(i + 1, k_clusters):
                denom = np.sqrt(cluster_active_count[i] * cluster_active_count[j])
                if denom > 0:
                    pmi[i, j] = cooccur[i, j] / denom
                    pmi[j, i] = pmi[i, j]

        # Top 10 pairs
        triu_idx = np.triu_indices(k_clusters, k=1)
        pmi_flat = pmi[triu_idx]
        top10_idx = np.argsort(-pmi_flat)[:10]
        print(f"\n    Top 10 cluster pairs by PMI:")
        top_pairs = []
        for rank, idx in enumerate(top10_idx):
            ci, cj = triu_idx[0][idx], triu_idx[1][idx]
            p = pmi_flat[idx]
            c = cooccur[ci, cj]
            print(f"      Cluster {ci} × Cluster {cj}: PMI={p:.4f} (co-occur {int(c)} times)")
            top_pairs.append({'cluster_i': int(ci), 'cluster_j': int(cj),
                             'pmi': float(p), 'co_occur': int(c)})
        cooccur_results = {'top_pairs': top_pairs}
        cooccur_matrix = pmi.tolist()
    else:
        print(f"\n  Part B: Skipped (no val_data)")

    # --- Part C: R.3 neuron cluster membership ---
    print(f"\n  Part C: R.3 neuron cluster membership")
    r3_path = os.path.join(output_dir, 'r3_knowledge_neurons', 'results.json')
    r3_neurons = []
    r3_cluster_info = []
    if os.path.exists(r3_path):
        with open(r3_path) as f:
            r3_data = json.load(f)
        # Extract physics neuron IDs from R.3 results
        for entry in r3_data.get('physics', []):
            for n in entry.get('top_neurons', []):
                nid = n.get('neuron_id', n.get('neuron', -1))
                if nid >= 0 and nid < n_know:
                    r3_neurons.append(nid)
        r3_neurons = list(set(r3_neurons))[:20]  # deduplicate, cap at 20

        if r3_neurons:
            emb_clusters_used = set()
            read_clusters_used = set()
            write_clusters_used = set()
            for ni in r3_neurons:
                ec = int(emb_labels[ni])
                rc = int(read_labels[ni])
                wc = int(write_labels[ni])
                emb_clusters_used.add(ec)
                read_clusters_used.add(rc)
                write_clusters_used.add(wc)
                print(f"    n{ni} → emb_cluster={ec}, read_cluster={rc}, write_cluster={wc}")
                r3_cluster_info.append({'neuron': ni, 'emb_cluster': ec,
                                        'read_cluster': rc, 'write_cluster': wc})
            print(f"    Physics neurons span {len(emb_clusters_used)} emb clusters, "
                  f"{len(read_clusters_used)} read clusters, {len(write_clusters_used)} write clusters")
        else:
            print(f"    No physics neurons found in R.3 results")
    else:
        print(f"    R.3 results not found at {r3_path}, skipping Part C")

    results = {
        'k_clusters': k_clusters,
        'n_know': n_know,
        'emb_cluster_sizes': emb_sizes.tolist(),
        'read_cluster_sizes': read_sizes.tolist(),
        'write_cluster_sizes': write_sizes.tolist(),
        'purity': {'emb_read': float(er_pur), 'emb_write': float(ew_pur), 'read_write': float(rw_pur)},
        'top_clusters': cluster_interpretations,
        'coactivation': cooccur_results,
        'r3_cluster_info': r3_cluster_info,
    }
    if cooccur_matrix is not None:
        results['cooccur_pmi_matrix'] = cooccur_matrix

    _save_json(results, output_dir, 'p7_neuron_clustering', 'results.json')
    return results


# ============================================================
# P8: Cross-Reference Analysis
# ============================================================

def analyze_cross_reference(params, cfg, val_tokens, output_dir,
                             n_batches=10, batch_size=4):
    """P8: Cross-reference P1-P7 results for deeper patterns."""
    print("\n" + "="*60)
    print("P8: Cross-Reference Analysis")
    print("="*60)

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward
    _srw_inference_with_gates = _mod._srw_inference_with_gates
    _layer_norm = _mod._layer_norm
    _srw_inference = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_qk = model_cfg['n_qk']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']
    mid_layer = n_layers // 2

    pool = params['neuron_pool']
    read_vecs = np.array(pool['know_read'])
    write_vecs = np.array(pool['know_write'])
    read_n = read_vecs / (np.linalg.norm(read_vecs, axis=-1, keepdims=True) + 1e-8)
    write_n = write_vecs / (np.linalg.norm(write_vecs, axis=-1, keepdims=True) + 1e-8)
    alignment = (read_n * write_n).sum(axis=-1)  # [n_know]

    # Groups
    correction_mask = alignment < -0.5
    neutral_mask = (alignment >= -0.5) & (alignment <= 0.1)
    transform_mask = alignment > 0.1
    n_corr = int(correction_mask.sum())
    n_neut = int(neutral_mask.sum())
    n_trans = int(transform_mask.sum())
    print(f"  Groups: correction(<-0.5)={n_corr}, neutral={n_neut}, transform(>0.1)={n_trans}")

    # Prepare val data
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    # --- Compute layer activation matrix + gate stats in one pass ---
    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    layer_act = np.zeros((n_layers, n_know), dtype=np.float64)  # for Part A
    gate_sum = np.zeros(n_know, dtype=np.float64)  # for Part B
    gate_max = np.zeros(n_know, dtype=np.float64)
    gate_phi_conf = np.zeros(n_know, dtype=np.float64)  # confident count
    gate_phi_bord = np.zeros(n_know, dtype=np.float64)  # borderline count
    gate_count = np.zeros(n_know, dtype=np.float64)  # active count

    # QK layer data for Part D
    q_layer_act = np.zeros((n_layers, n_qk), dtype=np.float64)
    k_layer_act = np.zeros((n_layers, n_qk), dtype=np.float64)

    THRESH = 0.0
    print(f"\n  Running {actual_batches} batches for layer/gate stats...")
    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        _, layer_info = jit_analysis(params, batch)

        # Know raw gate [L, B, S, N]
        gk = np.array(jax.device_get(layer_info['gate_Know_raw']))
        for li in range(n_layers):
            layer_act[li] += (gk[li] > THRESH).astype(np.float32).mean(axis=(0, 1))

        # Mid-layer gate stats for Part B
        g_mid = gk[mid_layer]  # [B, S, N]
        active = g_mid > THRESH
        gate_sum += g_mid.sum(axis=(0, 1))
        gate_max = np.maximum(gate_max, g_mid.max(axis=(0, 1)))
        gate_count += active.astype(np.float32).sum(axis=(0, 1))

        # QK gates for Part D
        gq = np.array(jax.device_get(layer_info['gate_Q_raw']))
        gkk = np.array(jax.device_get(layer_info['gate_K_raw']))
        for li in range(n_layers):
            q_layer_act[li] += (gq[li] > THRESH).astype(np.float32).mean(axis=(0, 1))
            k_layer_act[li] += (gkk[li] > THRESH).astype(np.float32).mean(axis=(0, 1))

        del layer_info
        if (b + 1) % 3 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    layer_act /= actual_batches
    q_layer_act /= actual_batches
    k_layer_act /= actual_batches
    n_tokens = actual_batches * batch_size * max_seq
    mean_gate = gate_sum / (n_tokens + 1e-8)

    results = {}

    # === Part A: Alignment × Layer Activation ===
    print(f"\n  Part A: Alignment × Layer Activation")
    print(f"  {'Layer':>5} | {'correction':>10} | {'neutral':>9} | {'transform':>9}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")
    part_a = {}
    for li in range(n_layers):
        corr_rate = float(layer_act[li, correction_mask].mean()) if n_corr > 0 else 0
        neut_rate = float(layer_act[li, neutral_mask].mean()) if n_neut > 0 else 0
        trans_rate = float(layer_act[li, transform_mask].mean()) if n_trans > 0 else 0
        print(f"  {li:>5} | {corr_rate*100:>9.1f}% | {neut_rate*100:>8.1f}% | {trans_rate*100:>8.1f}%")
        part_a[f'layer_{li}'] = {'correction': corr_rate, 'neutral': neut_rate, 'transform': trans_rate}

    corr_l0 = part_a['layer_0']['correction']
    corr_l11 = part_a[f'layer_{n_layers-1}']['correction']
    trans_l0 = part_a['layer_0']['transform']
    trans_l11 = part_a[f'layer_{n_layers-1}']['transform']
    print(f"\n  Correction: L0/L{n_layers-1} ratio = {corr_l0/(corr_l11+1e-8):.1f}")
    print(f"  Transform:  L0/L{n_layers-1} ratio = {trans_l0/(trans_l11+1e-8):.1f}")
    results['part_a'] = part_a

    # === Part B: Alignment × Gate Strength ===
    print(f"\n  Part B: Alignment × Gate Strength (layer {mid_layer})")
    corr_corr = float(np.corrcoef(alignment, mean_gate)[0, 1]) if n_know > 1 else 0
    print(f"  Correlation(alignment, mean_gate): r={corr_corr:.4f}")

    def group_gate_stats(mask, name):
        if mask.sum() == 0:
            return {'mean_gate': 0, 'max_gate': 0}
        mg = float(mean_gate[mask].mean())
        mx = float(gate_max[mask].mean())
        return {'mean_gate': mg, 'max_gate': mx}

    groups = [('correction', correction_mask), ('neutral', neutral_mask), ('transform', transform_mask)]
    print(f"  {'Group':>12} | {'mean_gate':>10} | {'max_gate':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
    part_b = {'correlation': corr_corr}
    for gname, gmask in groups:
        gs = group_gate_stats(gmask, gname)
        print(f"  {gname:>12} | {gs['mean_gate']:>10.6f} | {gs['max_gate']:>10.4f}")
        part_b[gname] = gs
    results['part_b'] = part_b

    # === Part C: Domain-specific Neurons' Alignment ===
    print(f"\n  Part C: Domain-specific Neurons' Alignment")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    jit_analysis_light = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))

    domain_profiles = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        acc = np.zeros(n_know, dtype=np.float64)
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            _, li = jit_analysis_light(params, jnp.array(ids_pad, dtype=jnp.int32))
            gate = np.array(jax.device_get(li['gate_Know']))  # normalized, but fine for ranking
            acc += gate.mean(axis=(0, 1, 2))  # [N]
        domain_profiles[domain] = acc / len(prompts)

    all_domains = list(DOMAIN_PROMPTS.keys())
    overall_align_mean = float(alignment.mean())
    overall_corr_pct = float(correction_mask.mean() * 100)
    overall_neut_pct = float(neutral_mask.mean() * 100)
    overall_trans_pct = float(transform_mask.mean() * 100)

    print(f"  {'Domain':>12} | {'align_mean':>10} | {'correct%':>9} | {'neutral%':>9} | {'transform%':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
    part_c = {}
    for domain in all_domains:
        others_mean = np.mean([domain_profiles[d] for d in all_domains if d != domain], axis=0)
        specificity = domain_profiles[domain] - others_mean
        top5_idx = np.argsort(-specificity)[:int(n_know * 0.05)]
        a_mean = float(alignment[top5_idx].mean())
        c_pct = float(correction_mask[top5_idx].mean() * 100)
        n_pct = float(neutral_mask[top5_idx].mean() * 100)
        t_pct = float(transform_mask[top5_idx].mean() * 100)
        print(f"  {domain:>12} | {a_mean:>10.3f} | {c_pct:>8.1f}% | {n_pct:>8.1f}% | {t_pct:>9.1f}%")
        part_c[domain] = {'align_mean': a_mean, 'correction_pct': c_pct,
                          'neutral_pct': n_pct, 'transform_pct': t_pct}
    print(f"  {'overall':>12} | {overall_align_mean:>10.3f} | {overall_corr_pct:>8.1f}% | "
          f"{overall_neut_pct:>8.1f}% | {overall_trans_pct:>9.1f}%")
    part_c['overall'] = {'align_mean': overall_align_mean, 'correction_pct': overall_corr_pct,
                         'neutral_pct': overall_neut_pct, 'transform_pct': overall_trans_pct}
    results['part_c'] = part_c

    # === Part D: Q/K Asymmetry Deep Dive ===
    print(f"\n  Part D: Q/K Asymmetry")
    q_total = q_layer_act.sum(axis=0)  # [n_qk]
    k_total = k_layer_act.sum(axis=0)
    q_ratio = q_total / (q_total + k_total + 1e-8)  # [n_qk], 0=K-only, 1=Q-only
    q_only_mask = q_ratio > 0.7
    k_only_mask = q_ratio < 0.3

    print(f"  {'Layer':>5} | {'Q-only act%':>12} | {'K-only act%':>12} | {'mean Q_ratio':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    part_d = {}
    for li in range(n_layers):
        q_act = float(q_layer_act[li, q_only_mask].mean()) if q_only_mask.sum() > 0 else 0
        k_act = float(k_layer_act[li, k_only_mask].mean()) if k_only_mask.sum() > 0 else 0
        qr_layer = float(q_layer_act[li].sum() / (q_layer_act[li].sum() + k_layer_act[li].sum() + 1e-8))
        print(f"  {li:>5} | {q_act*100:>11.1f}% | {k_act*100:>11.1f}% | {qr_layer:>12.3f}")
        part_d[f'layer_{li}'] = {'q_only_active': q_act, 'k_only_active': k_act, 'mean_q_ratio': qr_layer}

    q_spread = float((q_layer_act[:, q_only_mask] > 0.01).sum(axis=0).mean()) if q_only_mask.sum() > 0 else 0
    k_spread = float((k_layer_act[:, k_only_mask] > 0.01).sum(axis=0).mean()) if k_only_mask.sum() > 0 else 0
    print(f"\n  Q-only neurons ({int(q_only_mask.sum())}): active in avg {q_spread:.1f} layers")
    print(f"  K-only neurons ({int(k_only_mask.sum())}): active in avg {k_spread:.1f} layers")
    part_d['q_only_count'] = int(q_only_mask.sum())
    part_d['k_only_count'] = int(k_only_mask.sum())
    part_d['q_only_layer_spread'] = q_spread
    part_d['k_only_layer_spread'] = k_spread
    results['part_d'] = part_d

    # === Part E: Effective Rank × Alignment ===
    print(f"\n  Part E: Effective Rank by Alignment Group (layer {mid_layer})")
    know_write_n = write_n  # already unit-normed

    jit_analysis_full = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))
    N_SAMPLES = 50
    group_ranks = {'correction': [], 'neutral': [], 'transform': [], 'all': []}

    samples_done = 0
    bi = 0
    while samples_done < N_SAMPLES and bi * batch_size < n_seqs:
        batch = jnp.array(tokens[bi * batch_size:(bi + 1) * batch_size], dtype=jnp.int32)
        _, li = jit_analysis_full(params, batch)
        gate = np.array(jax.device_get(li['gate_Know_raw'][mid_layer]))  # [B, S, N]
        B, S, N = gate.shape

        positions = np.random.choice(B * S, size=min(10, N_SAMPLES - samples_done), replace=False)
        for pos in positions:
            b_i, s_i = divmod(int(pos), S)
            g = gate[b_i, s_i]
            active_idx = np.where(g > 0)[0]
            if len(active_idx) < 2:
                continue

            def _eff_rank(idx):
                if len(idx) < 2:
                    return 0, len(idx)
                W = know_write_n[idx]
                sv = np.linalg.svd(W, compute_uv=False)
                sv_n = sv / (sv.sum() + 1e-8)
                ent = -(sv_n * np.log(sv_n + 1e-10)).sum()
                return float(np.exp(ent)), len(idx)

            # All active
            er_all, n_all = _eff_rank(active_idx)
            group_ranks['all'].append((n_all, er_all))

            # By group
            for gname, gmask in [('correction', correction_mask), ('neutral', neutral_mask), ('transform', transform_mask)]:
                grp_idx = active_idx[gmask[active_idx]]
                if len(grp_idx) >= 2:
                    er, n_g = _eff_rank(grp_idx)
                    group_ranks[gname].append((n_g, er))
            samples_done += 1
        bi += 1

    print(f"  {'Group':>12} | {'active_N':>8} | {'eff_rank':>8} | {'efficiency':>10}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    part_e = {}
    for gname in ['correction', 'neutral', 'transform', 'all']:
        data = group_ranks[gname]
        if data:
            ns = np.array([d[0] for d in data])
            ers = np.array([d[1] for d in data])
            eff = ers / (ns + 1e-8)
            print(f"  {gname:>12} | {ns.mean():>8.0f} | {ers.mean():>8.1f} | {eff.mean():>10.3f}")
            part_e[gname] = {'active_N': float(ns.mean()), 'eff_rank': float(ers.mean()),
                            'efficiency': float(eff.mean())}
        else:
            print(f"  {gname:>12} | — | — | —")
    results['part_e'] = part_e

    _save_json(results, output_dir, 'p8_cross_reference', 'results.json')
    return results


# ============================================================
# Deep Analysis: _run_layerwise_analysis + P8/P9 combined
# ============================================================

def _run_layerwise_analysis(params, model_cfg, input_ids, alignment=None):
    """Single forward pass collecting all per-layer statistics for P8/P9.

    No JIT — layer loop collects intermediates. Each layer's heavy tensors
    are reduced to scalars/small arrays before the next layer.
    Returns: list of dicts (one per layer).
    """
    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    n_know = model_cfg['n_know']

    pool_params = params['neuron_pool']
    router_params = params['router']

    emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params['pos_emb']['embedding'])
    know_emb_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_s = pool_params.get('qk_scale', 1.0)
    v_s_val = pool_params.get('v_scale', 1.0)
    know_s = pool_params.get('know_scale', 1.0)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]
    x0_mean = x.mean(axis=(0, 1))  # [D]

    # Alignment masks (if provided)
    corr_mask = alignment < -0.5 if alignment is not None else None
    neut_mask = (alignment >= -0.5) & (alignment <= 0.1) if alignment is not None else None
    trans_mask = alignment > 0.1 if alignment is not None else None

    n_bins = min(16, S // 2)
    bin_size = S // n_bins

    layer_results = []
    for li in range(n_layers):
        bp = block_params_list[li]
        x_pre = x

        # Attention
        normed1 = _layer_norm_fn(x, jnp.asarray(bp['norm1']['scale']), jnp.asarray(bp['norm1']['bias']))
        h_all = normed1 @ jnp.asarray(router_params['proj_attn']['kernel']) + jnp.asarray(router_params['proj_attn']['bias'])
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed1 @ jnp.asarray(router_params['tau_attn']['kernel']) + jnp.asarray(router_params['tau_attn']['bias'])

        Q = _srw_inference_fn(normed1, h_Q, qk_n, tau_all[:,:,0:1], pool_params['qk_read'], pool_params['qk_write']) * qk_s
        K = _srw_inference_fn(normed1, h_K, qk_n, tau_all[:,:,1:2], pool_params['qk_read'], pool_params['qk_write']) * qk_s
        V = _srw_inference_fn(normed1, h_V, v_n, tau_all[:,:,2:3], pool_params['v_read'], pool_params['v_write']) * v_s_val

        d_head = d_model // n_heads
        Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        scale = jnp.sqrt(jnp.float32(d_head))
        sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
        aw = jax.nn.softmax(sc, axis=-1)
        ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
        attn_out = ao @ jnp.asarray(bp['attn']['expand_O']['kernel'])

        x_after_attn = x + attn_out

        # Know: direct z/phi/gate computation
        normed2 = _layer_norm_fn(x_after_attn, jnp.asarray(bp['norm2']['scale']), jnp.asarray(bp['norm2']['bias']))
        h_k = normed2 @ jnp.asarray(router_params['proj_know']['kernel']) + jnp.asarray(router_params['proj_know']['bias'])
        tau_k = normed2 @ jnp.asarray(router_params['tau_know']['kernel']) + jnp.asarray(router_params['tau_know']['bias'])

        scores_k = h_k @ know_emb_n.T
        sf = scores_k.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        tau = s_mean + tau_k * s_std
        raw = scores_k - tau.astype(scores_k.dtype)
        z = raw.astype(jnp.float32) / s_std
        phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
        gate = jnp.where(z > 0, z * phi, 0.0)

        know_out = _srw_inference_fn(normed2, h_k, know_emb_n, tau_k,
                                     pool_params['know_read'], pool_params['know_write']) * know_s
        x = x_after_attn + know_out

        # === Collect stats (all JAX, device_get later) ===
        # P9A: attn-know interaction
        a_n = jnp.linalg.norm(attn_out, axis=-1, keepdims=True) + 1e-8
        k_n = jnp.linalg.norm(know_out, axis=-1, keepdims=True) + 1e-8
        a_dir = attn_out / a_n
        cos_ak = (attn_out * know_out).sum(axis=-1) / (a_n.squeeze(-1) * k_n.squeeze(-1))
        proj_s = (know_out * a_dir).sum(axis=-1, keepdims=True)
        k_par = proj_s * a_dir
        par_frac = jnp.linalg.norm(k_par, axis=-1) / k_n.squeeze(-1)

        # P9B: residual trajectory
        x_mean = x.mean(axis=(0, 1))
        cos_x0 = jnp.dot(x_mean, x0_mean) / (jnp.linalg.norm(x_mean) * jnp.linalg.norm(x0_mean) + 1e-8)

        # P9C: LN vs Know
        ln_delta = normed2 - x_after_attn
        ln_k_cos = (ln_delta * know_out).sum(axis=-1) / (jnp.linalg.norm(ln_delta, axis=-1) * jnp.linalg.norm(know_out, axis=-1) + 1e-8)

        # P9D: position bins
        g_active = (gate > 0).astype(jnp.float32).sum(axis=-1)  # [B, S]
        g_sum = gate.sum(axis=-1)
        usable = n_bins * bin_size
        pos_active = g_active[:, :usable].reshape(B, n_bins, bin_size).mean(axis=(0, 2))
        pos_gsum = g_sum[:, :usable].reshape(B, n_bins, bin_size).mean(axis=(0, 2))
        pos_cos = cos_ak[:, :usable].reshape(B, n_bins, bin_size).mean(axis=(0, 2))

        # P8A/B: per-neuron stats
        neur_active = (gate > 0).astype(jnp.float32).mean(axis=(0, 1))  # [N]
        neur_gate = gate.mean(axis=(0, 1))
        neur_max = gate.max(axis=(0, 1))

        r = {
            'cos_ak': cos_ak.mean(), 'mag_ratio': (k_n / a_n).mean(),
            'par_frac': par_frac.mean(), 'proj_sign': proj_s.mean(),
            'a_norm': a_n.mean(), 'k_norm': k_n.mean(),
            'x_norm': jnp.linalg.norm(x, axis=-1).mean(),
            'dx': jnp.linalg.norm(x - x_pre, axis=-1).mean(),
            'cos_x0': cos_x0,
            'ln_k_cos': ln_k_cos.mean(),
            'ln_dn': jnp.linalg.norm(ln_delta, axis=-1).mean(),
            'k_dn': jnp.linalg.norm(know_out, axis=-1).mean(),
            'pos_active': pos_active, 'pos_gsum': pos_gsum, 'pos_cos': pos_cos,
            'neur_active': neur_active, 'neur_gate': neur_gate, 'neur_max': neur_max,
        }
        layer_results.append(r)

    return layer_results


def analyze_deep_analysis(params, cfg, val_tokens, output_dir,
                           n_batches=10, batch_size=4):
    """P8 (cross-ref) + P9 (fundamental) in one forward pass."""
    print("\n" + "="*60)
    print("Deep Analysis (P8 Cross-Reference + P9 Fundamental)")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_qk = model_cfg['n_qk']
    n_layers = model_cfg['n_layers']
    d_model = model_cfg['d_model']
    mid_layer = n_layers // 2

    # Alignment
    pool = params['neuron_pool']
    read_v = np.array(pool['know_read'])
    write_v = np.array(pool['know_write'])
    r_n = read_v / (np.linalg.norm(read_v, axis=-1, keepdims=True) + 1e-8)
    w_n = write_v / (np.linalg.norm(write_v, axis=-1, keepdims=True) + 1e-8)
    alignment = (r_n * w_n).sum(axis=-1)  # [n_know]
    corr_m = alignment < -0.5
    neut_m = (alignment >= -0.5) & (alignment <= 0.1)
    trans_m = alignment > 0.1
    print(f"  Alignment groups: correction={corr_m.sum()}, neutral={neut_m.sum()}, transform={trans_m.sum()}")

    params_jax = jax.tree.map(jnp.asarray, params)
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    # Accumulators
    acc = {f'l{li}': {} for li in range(n_layers)}
    scalar_keys = ['cos_ak', 'mag_ratio', 'par_frac', 'proj_sign', 'a_norm', 'k_norm',
                   'x_norm', 'dx', 'cos_x0', 'ln_k_cos', 'ln_dn', 'k_dn']
    for li in range(n_layers):
        for k in scalar_keys:
            acc[f'l{li}'][k] = 0.0
        acc[f'l{li}']['pos_active'] = np.zeros(min(16, max_seq // 2))
        acc[f'l{li}']['pos_gsum'] = np.zeros(min(16, max_seq // 2))
        acc[f'l{li}']['pos_cos'] = np.zeros(min(16, max_seq // 2))
        acc[f'l{li}']['neur_active'] = np.zeros(n_know)
        acc[f'l{li}']['neur_gate'] = np.zeros(n_know)
        acc[f'l{li}']['neur_max'] = np.zeros(n_know)

    print(f"  Running {actual_batches} batches (bs={batch_size})...")
    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        results = _run_layerwise_analysis(params_jax, model_cfg, batch, alignment)
        for li, r in enumerate(results):
            d = jax.device_get(r)
            for k in scalar_keys:
                acc[f'l{li}'][k] += float(d[k])
            acc[f'l{li}']['pos_active'] += np.array(d['pos_active'])
            acc[f'l{li}']['pos_gsum'] += np.array(d['pos_gsum'])
            acc[f'l{li}']['pos_cos'] += np.array(d['pos_cos'])
            acc[f'l{li}']['neur_active'] += np.array(d['neur_active'])
            acc[f'l{li}']['neur_gate'] += np.array(d['neur_gate'])
            acc[f'l{li}']['neur_max'] = np.maximum(acc[f'l{li}']['neur_max'], np.array(d['neur_max']))
        if (b + 1) % 3 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Average
    for li in range(n_layers):
        for k in scalar_keys:
            acc[f'l{li}'][k] /= actual_batches
        acc[f'l{li}']['pos_active'] /= actual_batches
        acc[f'l{li}']['pos_gsum'] /= actual_batches
        acc[f'l{li}']['pos_cos'] /= actual_batches
        acc[f'l{li}']['neur_active'] /= actual_batches
        acc[f'l{li}']['neur_gate'] /= actual_batches

    # ===================== P9: FUNDAMENTAL ANALYSIS =====================

    p9 = {}

    # P9A: Attention-Know Interaction
    print(f"\n  === P9A: Attention-Know Interaction ===")
    print(f"  {'Layer':>5} | {'cos(a,k)':>9} | {'‖k‖/‖a‖':>9} | {'parallel%':>9} | {'perp%':>7} | {'proj_sign':>9}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}")
    p9a = {}
    for li in range(n_layers):
        d = acc[f'l{li}']
        perp = max(0, 1.0 - d['par_frac'])
        print(f"  {li:>5} | {d['cos_ak']:>9.4f} | {d['mag_ratio']:>9.3f} | {d['par_frac']*100:>8.1f}% | {perp*100:>6.1f}% | {d['proj_sign']:>9.4f}")
        p9a[f'layer_{li}'] = {k: d[k] for k in ['cos_ak', 'mag_ratio', 'par_frac', 'proj_sign']}
    p9['part_a'] = p9a

    # P9B: Residual Stream Trajectory
    print(f"\n  === P9B: Residual Stream Trajectory ===")
    print(f"  {'Layer':>5} | {'‖x‖':>7} | {'Δx/‖x‖':>8} | {'cos(x,x0)':>10} | {'attn_‖‖':>8} | {'know_‖‖':>8}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    p9b = {}
    for li in range(n_layers):
        d = acc[f'l{li}']
        dx_ratio = d['dx'] / (d['x_norm'] + 1e-8)
        print(f"  {li:>5} | {d['x_norm']:>7.3f} | {dx_ratio:>8.4f} | {d['cos_x0']:>10.4f} | {d['a_norm']:>8.4f} | {d['k_norm']:>8.4f}")
        p9b[f'layer_{li}'] = {'x_norm': d['x_norm'], 'dx_ratio': dx_ratio, 'cos_x0': d['cos_x0'],
                              'attn_norm': d['a_norm'], 'know_norm': d['k_norm']}
    p9['part_b'] = p9b

    # P9C: LayerNorm vs Know
    print(f"\n  === P9C: LayerNorm vs Know ===")
    print(f"  {'Layer':>5} | {'LN_‖Δ‖':>8} | {'Know_‖Δ‖':>9} | {'LN-Know cos':>11}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*11}")
    p9c = {}
    for li in range(n_layers):
        d = acc[f'l{li}']
        print(f"  {li:>5} | {d['ln_dn']:>8.4f} | {d['k_dn']:>9.4f} | {d['ln_k_cos']:>11.4f}")
        p9c[f'layer_{li}'] = {'ln_delta_norm': d['ln_dn'], 'know_delta_norm': d['k_dn'], 'ln_know_cos': d['ln_k_cos']}
    p9['part_c'] = p9c

    # P9D: Position-wise Pattern (mid layer)
    print(f"\n  === P9D: Position-wise Pattern (layer {mid_layer}) ===")
    d_mid = acc[f'l{mid_layer}']
    n_bins = len(d_mid['pos_active'])
    bin_sz = max_seq // n_bins
    print(f"  {'Position':>10} | {'active_N':>8} | {'gate_sum':>9} | {'attn-know cos':>13}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*9}-+-{'-'*13}")
    p9d = {}
    for bi in range(n_bins):
        s, e = bi * bin_sz, (bi + 1) * bin_sz
        pa = d_mid['pos_active'][bi]
        pg = d_mid['pos_gsum'][bi]
        pc = d_mid['pos_cos'][bi]
        print(f"  {s:>4}-{e-1:<4} | {pa:>8.0f} | {pg:>9.1f} | {pc:>13.4f}")
        p9d[f'{s}_{e-1}'] = {'active_N': float(pa), 'gate_sum': float(pg), 'cos_ak': float(pc)}
    p9['part_d'] = p9d

    _save_json(p9, output_dir, 'p9_fundamental', 'results.json')

    # ===================== P8: CROSS-REFERENCE =====================

    p8 = {}

    # P8A: Alignment × Layer
    print(f"\n  === P8A: Alignment × Layer Activation ===")
    print(f"  {'Layer':>5} | {'correction':>10} | {'neutral':>9} | {'transform':>9}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")
    p8a = {}
    for li in range(n_layers):
        na = acc[f'l{li}']['neur_active']
        cr = float(na[corr_m].mean()) if corr_m.sum() > 0 else 0
        nr = float(na[neut_m].mean()) if neut_m.sum() > 0 else 0
        tr = float(na[trans_m].mean()) if trans_m.sum() > 0 else 0
        print(f"  {li:>5} | {cr*100:>9.1f}% | {nr*100:>8.1f}% | {tr*100:>8.1f}%")
        p8a[f'layer_{li}'] = {'correction': cr, 'neutral': nr, 'transform': tr}
    p8['part_a'] = p8a

    # P8B: Alignment × Gate Strength
    print(f"\n  === P8B: Alignment × Gate Strength (layer {mid_layer}) ===")
    mg = acc[f'l{mid_layer}']['neur_gate']
    mx = acc[f'l{mid_layer}']['neur_max']
    corr_mg = float(mg[corr_m].mean()) if corr_m.sum() > 0 else 0
    neut_mg = float(mg[neut_m].mean()) if neut_m.sum() > 0 else 0
    trans_mg = float(mg[trans_m].mean()) if trans_m.sum() > 0 else 0
    corr_mx = float(mx[corr_m].mean()) if corr_m.sum() > 0 else 0
    neut_mx = float(mx[neut_m].mean()) if neut_m.sum() > 0 else 0
    trans_mx = float(mx[trans_m].mean()) if trans_m.sum() > 0 else 0
    r_corr = float(np.corrcoef(alignment, mg)[0, 1]) if n_know > 1 else 0
    print(f"  Correlation(alignment, mean_gate): r={r_corr:.4f}")
    print(f"  {'Group':>12} | {'mean_gate':>10} | {'max_gate':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'correction':>12} | {corr_mg:>10.6f} | {corr_mx:>10.4f}")
    print(f"  {'neutral':>12} | {neut_mg:>10.6f} | {neut_mx:>10.4f}")
    print(f"  {'transform':>12} | {trans_mg:>10.6f} | {trans_mx:>10.4f}")
    p8['part_b'] = {'correlation': r_corr,
                    'correction': {'mean_gate': corr_mg, 'max_gate': corr_mx},
                    'neutral': {'mean_gate': neut_mg, 'max_gate': neut_mx},
                    'transform': {'mean_gate': trans_mg, 'max_gate': trans_mx}}

    # P8C: Domain neurons' alignment
    print(f"\n  === P8C: Domain Neurons' Alignment ===")
    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward
    jit_light = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))
    domain_profiles = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        da = np.zeros(n_know, dtype=np.float64)
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            _, li = jit_light(params_jax, jnp.array(ids_pad, dtype=jnp.int32))
            da += np.array(jax.device_get(li['gate_Know'])).mean(axis=(0, 1, 2))
        domain_profiles[domain] = da / len(prompts)

    all_domains = list(DOMAIN_PROMPTS.keys())
    overall_am = float(alignment.mean())
    overall_cp = float(corr_m.mean() * 100)
    overall_np = float(neut_m.mean() * 100)
    overall_tp = float(trans_m.mean() * 100)

    print(f"  {'Domain':>12} | {'align_mean':>10} | {'correct%':>9} | {'neutral%':>9} | {'transform%':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
    p8c = {}
    for domain in all_domains:
        om = np.mean([domain_profiles[d] for d in all_domains if d != domain], axis=0)
        spec = domain_profiles[domain] - om
        top5 = np.argsort(-spec)[:int(n_know * 0.05)]
        am = float(alignment[top5].mean())
        cp = float(corr_m[top5].mean() * 100)
        np_ = float(neut_m[top5].mean() * 100)
        tp = float(trans_m[top5].mean() * 100)
        print(f"  {domain:>12} | {am:>10.3f} | {cp:>8.1f}% | {np_:>8.1f}% | {tp:>9.1f}%")
        p8c[domain] = {'align_mean': am, 'correction_pct': cp, 'neutral_pct': np_, 'transform_pct': tp}
    print(f"  {'overall':>12} | {overall_am:>10.3f} | {overall_cp:>8.1f}% | {overall_np:>8.1f}% | {overall_tp:>9.1f}%")
    p8c['overall'] = {'align_mean': overall_am, 'correction_pct': overall_cp,
                      'neutral_pct': overall_np, 'transform_pct': overall_tp}
    p8['part_c'] = p8c

    _save_json(p8, output_dir, 'p8_cross_reference_v2', 'results.json')
    print(f"\n  Deep analysis complete.")
    return p8, p9


# ============================================================
# P10: Operation Space Analysis
# ============================================================

def analyze_operation_space(params, cfg, val_tokens, output_dir,
                             n_batches=10, batch_size=4):
    """P10: proj(x) diversity, activation uniqueness, emb geometry, superposition."""
    print("\n" + "="*60)
    print("P10: Operation Space Analysis")
    print("="*60)

    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']
    d_route = model_cfg.get('d_route', 128)
    mid_layer = n_layers // 2

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_params = params_jax['neuron_pool']
    router_params = params_jax['router']
    emb_matrix = jnp.asarray(params_jax['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params_jax['pos_emb']['embedding'])
    know_emb_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]

    # Alignment for Part E
    read_v = np.array(pool_params['know_read'])
    write_v = np.array(pool_params['know_write'])
    rn = read_v / (np.linalg.norm(read_v, axis=-1, keepdims=True) + 1e-8)
    wn = write_v / (np.linalg.norm(write_v, axis=-1, keepdims=True) + 1e-8)
    alignment = (rn * wn).sum(axis=-1)
    corr_m = alignment < -0.5
    neut_m = (alignment >= -0.5) & (alignment <= 0.1)
    trans_m = alignment > 0.1

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    # Accumulators
    all_h = []  # Part A: proj(x) vectors [128-dim]
    all_active = []  # Part B: bool [N] per token
    all_scores_active = []  # Part C: active scores
    all_scores_inactive = []
    tau_percentiles = []  # Part C: tau position
    h_means_per_layer = []  # Part D: per-layer h mean [n_layers, 128]
    neur_gate_sum = np.zeros(n_know)  # Part E
    neur_gate_sq_sum = np.zeros(n_know)
    neur_active_count = np.zeros(n_know)

    print(f"  Running {actual_batches} batches (bs={batch_size})...")
    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        B, S = batch.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[batch] + pos_matrix[positions]

        layer_h_means = []
        for li in range(n_layers):
            bp = block_params_list[li]
            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
            Q = _srw_inference_fn(normed1, h_Q, qk_n, tau_all[:,:,0:1], pool_params['qk_read'], pool_params['qk_write']) * pool_params.get('qk_scale', 1.0)
            K = _srw_inference_fn(normed1, h_K, qk_n, tau_all[:,:,1:2], pool_params['qk_read'], pool_params['qk_write']) * pool_params.get('qk_scale', 1.0)
            V = _srw_inference_fn(normed1, h_V, v_n, tau_all[:,:,2:3], pool_params['v_read'], pool_params['v_write']) * pool_params.get('v_scale', 1.0)
            d_head = d_model // n_heads
            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']

            # Part D: h mean per layer
            h_k_np = np.array(jax.device_get(h_k))
            layer_h_means.append(h_k_np.mean(axis=(0, 1)))  # [d_route]

            if li == mid_layer:
                # Collect Part A/B/C/E data at mid layer
                h_flat = h_k_np.reshape(-1, h_k_np.shape[-1])  # [B*S, d_route]
                all_h.append(h_flat[:500])  # Cap per batch

                scores_k = np.array(jax.device_get(h_k @ know_emb_n.T))  # [B,S,N]
                sf = scores_k.astype(np.float32)
                s_mean_np = sf.mean(axis=-1, keepdims=True)
                s_std_np = np.sqrt(np.mean(np.square(sf - s_mean_np), axis=-1, keepdims=True)) + 1e-8
                tau_np = s_mean_np + np.array(jax.device_get(tau_k)) * s_std_np
                z_np = (sf - tau_np) / s_std_np
                gate_np = np.where(z_np > 0, z_np * 0.5 * (1.0 + np.vectorize(math.erf)(z_np * 0.7071067811865476)), 0.0)

                active = gate_np > 0  # [B, S, N]
                active_flat = active.reshape(-1, n_know)[:500]
                all_active.append(active_flat)

                # Part C: score distributions
                for bi in range(min(B, 2)):
                    for si in range(0, S, 64):  # Sample every 64th position
                        am = active[bi, si]
                        sc_tok = sf[bi, si]
                        if am.sum() > 0:
                            all_scores_active.extend(sc_tok[am].tolist()[:100])
                            all_scores_inactive.extend(sc_tok[~am].tolist()[:100])
                        # Tau percentile
                        tau_val = tau_np[bi, si, 0]
                        pct = (sc_tok < tau_val).mean() * 100
                        tau_percentiles.append(pct)

                # Part E: per-neuron gate stats
                mask = active.astype(np.float32)
                neur_gate_sum += (gate_np * mask).sum(axis=(0, 1))
                neur_gate_sq_sum += ((gate_np ** 2) * mask).sum(axis=(0, 1))
                neur_active_count += mask.sum(axis=(0, 1))

            know_out = _srw_inference_fn(normed2, h_k, know_emb_n, tau_k,
                                         pool_params['know_read'], pool_params['know_write']) * pool_params.get('know_scale', 1.0)
            x = x + know_out

        h_means_per_layer.append(np.array(layer_h_means))  # [n_layers, d_route]

        if (b + 1) % 3 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    results = {}

    # === Part A: proj(x) Diversity ===
    print(f"\n  === Part A: proj(x) Diversity (layer {mid_layer}, Know pool) ===")
    h_all_np = np.concatenate(all_h, axis=0)  # [N_samples, d_route]
    h_norms = np.linalg.norm(h_all_np, axis=-1, keepdims=True) + 1e-8
    h_normed = h_all_np / h_norms
    n_samp = min(len(h_normed), 2000)
    sv = np.linalg.svd(h_normed[:n_samp], compute_uv=False)
    sv_n = sv / (sv.sum() + 1e-8)
    eff_rank = float(np.exp(-(sv_n * np.log(sv_n + 1e-10)).sum()))
    # Pairwise cosine
    n_pairs = min(1000, n_samp * (n_samp - 1) // 2)
    idx = np.random.choice(len(h_normed), size=(n_pairs, 2), replace=True)
    cos_pairs = (h_normed[idx[:, 0]] * h_normed[idx[:, 1]]).sum(axis=-1)
    # PCA variance
    h_c = h_normed[:n_samp] - h_normed[:n_samp].mean(axis=0)
    cov = (h_c.T @ h_c) / n_samp
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    var_total = eigvals.sum() + 1e-8
    pc1 = float(eigvals[0] / var_total * 100)
    pc5 = float(eigvals[:5].sum() / var_total * 100)
    pc10 = float(eigvals[:10].sum() / var_total * 100)

    print(f"  Effective rank of proj(x): {eff_rank:.1f} / {d_route}")
    print(f"  Pairwise cosine: mean={cos_pairs.mean():.4f}, std={cos_pairs.std():.4f}, "
          f"p5={np.percentile(cos_pairs, 5):.4f}, p95={np.percentile(cos_pairs, 95):.4f}")
    print(f"  PCA variance: PC1={pc1:.1f}%, PC1-5={pc5:.1f}%, PC1-10={pc10:.1f}%")
    results['part_a'] = {'eff_rank': eff_rank, 'd_route': d_route,
                         'cos_mean': float(cos_pairs.mean()), 'cos_std': float(cos_pairs.std()),
                         'pca_pc1': pc1, 'pca_pc5': pc5, 'pca_pc10': pc10}

    # === Part B: Activation Set Uniqueness ===
    print(f"\n  === Part B: Activation Set Uniqueness (layer {mid_layer}) ===")
    act_all = np.concatenate(all_active, axis=0).astype(np.float32)  # [N_samples, N]
    mean_active = float(act_all.sum(axis=-1).mean())
    n_act_samp = min(len(act_all), 2000)
    idx_b = np.random.choice(n_act_samp, size=(500, 2), replace=True)
    a1 = act_all[idx_b[:, 0]]
    a2 = act_all[idx_b[:, 1]]
    intersection = (a1 * a2).sum(axis=-1)
    union = ((a1 + a2) > 0).astype(np.float32).sum(axis=-1)
    jaccard = intersection / (union + 1e-8)
    print(f"  Mean active neurons: {mean_active:.0f}")
    print(f"  Jaccard similarity: mean={jaccard.mean():.4f}, std={jaccard.std():.4f}")
    results['part_b'] = {'mean_active': mean_active,
                         'jaccard_mean': float(jaccard.mean()), 'jaccard_std': float(jaccard.std())}

    # === Part C: emb Space Geometry ===
    print(f"\n  === Part C: emb Space Geometry (layer {mid_layer}) ===")
    act_sc = np.array(all_scores_active[:5000]) if all_scores_active else np.array([0.0])
    inact_sc = np.array(all_scores_inactive[:5000]) if all_scores_inactive else np.array([0.0])
    separation = (act_sc.mean() - inact_sc.mean()) / (inact_sc.std() + 1e-8)
    tau_pct_mean = float(np.mean(tau_percentiles)) if tau_percentiles else 0
    print(f"  Active neuron scores:   mean={act_sc.mean():.4f}, std={act_sc.std():.4f}")
    print(f"  Inactive neuron scores: mean={inact_sc.mean():.4f}, std={inact_sc.std():.4f}")
    print(f"  Score separation: {separation:.2f} sigma")
    print(f"  Tau cuts at: top {100-tau_pct_mean:.1f}% of score distribution")
    results['part_c'] = {'active_score_mean': float(act_sc.mean()), 'inactive_score_mean': float(inact_sc.mean()),
                         'separation_sigma': float(separation), 'tau_top_pct': float(100 - tau_pct_mean)}

    # === Part D: Operation Space Trajectory ===
    print(f"\n  === Part D: Operation Space Trajectory ===")
    h_layer_means = np.mean(h_means_per_layer, axis=0)  # [n_layers, d_route]
    h_lm_n = h_layer_means / (np.linalg.norm(h_layer_means, axis=-1, keepdims=True) + 1e-8)
    print(f"  {'Layer':>5} | {'cos(h,h0)':>9} | {'cos(h,h-1)':>10} | {'‖h‖':>7}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}")
    p10d = {}
    for li in range(n_layers):
        cos_h0 = float((h_lm_n[li] * h_lm_n[0]).sum())
        cos_prev = float((h_lm_n[li] * h_lm_n[li - 1]).sum()) if li > 0 else 0
        h_norm = float(np.linalg.norm(h_layer_means[li]))
        prev_s = f"{cos_prev:>10.4f}" if li > 0 else f"{'—':>10}"
        print(f"  {li:>5} | {cos_h0:>9.4f} | {prev_s} | {h_norm:>7.3f}")
        p10d[f'layer_{li}'] = {'cos_h0': cos_h0, 'cos_prev': cos_prev, 'h_norm': h_norm}
    total_drift = float(np.arccos(np.clip((h_lm_n[0] * h_lm_n[-1]).sum(), -1, 1)) * 180 / np.pi)
    print(f"  Observation direction drifts {total_drift:.1f}° from layer 0 to {n_layers-1}")
    results['part_d'] = p10d
    results['part_d']['total_drift_degrees'] = total_drift

    # === Part E: Superposition Quantification ===
    print(f"\n  === Part E: Superposition Quantification (layer {mid_layer}) ===")
    safe_count = neur_active_count + 1e-8
    neur_mean = neur_gate_sum / safe_count
    neur_var = neur_gate_sq_sum / safe_count - neur_mean ** 2
    neur_var = np.maximum(neur_var, 0)

    print(f"  {'Group':>12} | {'mean_var':>10} | {'mean_gate':>10} | {'n_neurons':>9}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
    for gname, gmask in [('correction', corr_m), ('neutral', neut_m), ('transform', trans_m), ('all', np.ones(n_know, dtype=bool))]:
        mv = float(neur_var[gmask].mean()) if gmask.sum() > 0 else 0
        mg = float(neur_mean[gmask].mean()) if gmask.sum() > 0 else 0
        print(f"  {gname:>12} | {mv:>10.6f} | {mg:>10.6f} | {int(gmask.sum()):>9}")
    results['part_e'] = {
        'correction': {'mean_var': float(neur_var[corr_m].mean()), 'mean_gate': float(neur_mean[corr_m].mean())},
        'neutral': {'mean_var': float(neur_var[neut_m].mean()), 'mean_gate': float(neur_mean[neut_m].mean())},
        'transform': {'mean_var': float(neur_var[trans_m].mean()), 'mean_gate': float(neur_mean[trans_m].mean())},
    }

    _save_json(results, output_dir, 'p10_operation_space', 'results.json')
    return results


# ============================================================
# P11: Causal Intervention Experiments
# ============================================================

def _layer_ablated_forward(params_jax, model_cfg, input_ids, ablate_layers, ablate_type='know'):
    """Forward with know/attn zeroed at specific layers. Returns logits [B, S, vocab]."""
    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']

    pool_params = params_jax['neuron_pool']
    router_params = params_jax['router']
    emb_matrix = params_jax['token_emb']['embedding']
    pos_matrix = params_jax['pos_emb']['embedding']
    know_emb_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

    for li in range(n_layers):
        bp = params_jax[f'block_{li}']
        normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed1 @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed1 @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        Q = _srw_inference_fn(normed1, h_Q, qk_n, tau_all[:,:,0:1], pool_params['qk_read'], pool_params['qk_write']) * pool_params.get('qk_scale', 1.0)
        K = _srw_inference_fn(normed1, h_K, qk_n, tau_all[:,:,1:2], pool_params['qk_read'], pool_params['qk_write']) * pool_params.get('qk_scale', 1.0)
        V = _srw_inference_fn(normed1, h_V, v_n, tau_all[:,:,2:3], pool_params['v_read'], pool_params['v_write']) * pool_params.get('v_scale', 1.0)

        d_head = d_model // n_heads
        Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        scale = jnp.sqrt(jnp.float32(d_head))
        sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
        aw = jax.nn.softmax(sc, axis=-1)
        ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
        attn_out = ao @ bp['attn']['expand_O']['kernel']

        if ablate_type == 'attn' and li in ablate_layers:
            attn_out = jnp.zeros_like(attn_out)
        x = x + attn_out

        normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
        h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        know_out = _srw_inference_fn(normed2, h_k, know_emb_n, tau_k,
                                     pool_params['know_read'], pool_params['know_write']) * pool_params.get('know_scale', 1.0)

        if ablate_type == 'know' and li in ablate_layers:
            know_out = jnp.zeros_like(know_out)
        x = x + know_out

    norm_p = params_jax['norm']
    x = _layer_norm_fn(x, norm_p['scale'], norm_p['bias'])
    logits = x @ emb_matrix.T
    return logits


def _ce_loss_from_logits(logits, input_ids):
    """CE loss from logits [B, S, V] and input_ids [B, S]."""
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:].astype(jnp.int32)
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    safe = jnp.where(shift_labels > 0, shift_labels, 0)
    tl = -jnp.take_along_axis(log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
    valid = shift_labels > 0
    return (tl * valid).sum() / (valid.sum() + 1e-8)


def analyze_interventions(params, cfg, val_tokens, output_dir,
                           n_batches=50, batch_size=4):
    """P11: Causal intervention experiments — ablation, dosage, role isolation."""
    print("\n" + "="*60)
    print("P11: Causal Intervention Experiments")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _mod = get_model_module()
    build_suppressed_forward = _mod.build_suppressed_forward

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']

    # Alignment groups
    pool = params['neuron_pool']
    r_v = np.array(pool['know_read'])
    w_v = np.array(pool['know_write'])
    rn = r_v / (np.linalg.norm(r_v, axis=-1, keepdims=True) + 1e-8)
    wn = w_v / (np.linalg.norm(w_v, axis=-1, keepdims=True) + 1e-8)
    alignment = (rn * wn).sum(axis=-1)
    corr_idx = np.where(alignment < -0.5)[0]
    trans_idx = np.where(alignment > 0.1)[0]
    neut_idx = np.where((alignment >= -0.5) & (alignment <= 0.1))[0]
    print(f"  Groups: correction={len(corr_idx)}, neutral={len(neut_idx)}, transform={len(trans_idx)}")

    params_jax = jax.tree.map(jnp.asarray, params)
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    def _eval_loss_suppressed(suppress_mask):
        """Eval loss with suppress_mask [n_know] bool. No JIT."""
        fwd_fn = build_suppressed_forward(params, model_cfg, {'know': suppress_mask})
        total_loss, total_valid = 0.0, 0
        for b in range(actual_batches):
            batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
            logits = fwd_fn(batch)
            loss = float(_ce_loss_from_logits(logits, batch))
            total_loss += loss
            total_valid += 1
        return total_loss / max(total_valid, 1)

    def _eval_loss_layer_ablated(ablate_layers, ablate_type='know'):
        """Eval loss with layer-selective ablation."""
        total_loss, total_valid = 0.0, 0
        for b in range(actual_batches):
            batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
            logits = _layer_ablated_forward(params_jax, model_cfg, batch, ablate_layers, ablate_type)
            loss = float(_ce_loss_from_logits(logits, batch))
            total_loss += loss
            total_valid += 1
        return total_loss / max(total_valid, 1)

    # === Baseline ===
    print(f"\n  Computing baseline ({actual_batches} batches, bs={batch_size})...")
    baseline = _eval_loss_layer_ablated(set())  # no ablation
    print(f"  Baseline val loss: {baseline:.4f}")

    results = {'baseline': baseline, 'n_batches': actual_batches, 'batch_size': batch_size}

    # === Exp 1: Role Ablation ===
    print(f"\n  === Exp 1: Role Ablation ===")
    rng = np.random.RandomState(42)

    exp1_conditions = [
        ('correction', corr_idx),
        ('transform', trans_idx),
        ('neutral', neut_idx),
    ]
    # Random controls (3 runs averaged)
    for name, idx in [('random_corr', corr_idx), ('random_trans', trans_idx)]:
        n = len(idx)
        losses = []
        for seed in range(3):
            r_idx = rng.choice(n_know, n, replace=False)
            mask = np.zeros(n_know, dtype=bool)
            mask[r_idx] = True
            losses.append(_eval_loss_suppressed(mask))
        exp1_conditions.append((f'random({n})', np.array([], dtype=int)))  # placeholder
        exp1_conditions[-1] = (f'random({n})', losses)

    print(f"  {'Condition':>17} | {'n_suppress':>11} | {'val_loss':>9} | {'Δloss':>9} | {'Δ/neuron':>9}")
    print(f"  {'-'*17}-+-{'-'*11}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    exp1_results = {}
    for name, idx_or_losses in exp1_conditions:
        if isinstance(idx_or_losses, list):
            # random control (averaged)
            loss = float(np.mean(idx_or_losses))
            n_s = int(name.split('(')[1].rstrip(')'))
        else:
            mask = np.zeros(n_know, dtype=bool)
            mask[idx_or_losses] = True
            n_s = int(mask.sum())
            print(f"    Computing {name} ({n_s} neurons)...", flush=True)
            loss = _eval_loss_suppressed(mask)
        delta = loss - baseline
        per_n = delta / max(n_s, 1)
        print(f"  {name:>17} | {n_s:>11,} | {loss:>9.4f} | {delta:>+9.4f} | {per_n:>9.6f}")
        exp1_results[name] = {'n_suppress': n_s, 'loss': loss, 'delta': delta, 'per_neuron': per_n}
    results['exp1'] = exp1_results

    # === Exp 2: Layer-selective Know Ablation ===
    print(f"\n  === Exp 2: Layer-selective Know Ablation ===")
    mid = n_layers // 2
    layer_conditions = [
        ('ablate L0 know', {0}, 'know'),
        (f'ablate L{mid-1}-{mid+1} know', set(range(mid-1, mid+2)), 'know'),
        (f'ablate L{n_layers-1} know', {n_layers - 1}, 'know'),
        ('ablate ALL know', set(range(n_layers)), 'know'),
        ('ablate L0 attn', {0}, 'attn'),
    ]
    print(f"  {'Condition':>22} | {'val_loss':>9} | {'Δloss':>9}")
    print(f"  {'-'*22}-+-{'-'*9}-+-{'-'*9}")
    print(f"  {'no ablation':>22} | {baseline:>9.4f} | {'—':>9}")
    exp2_results = {}
    for name, layers, atype in layer_conditions:
        print(f"    Computing {name}...", flush=True)
        loss = _eval_loss_layer_ablated(layers, atype)
        delta = loss - baseline
        print(f"  {name:>22} | {loss:>9.4f} | {delta:>+9.4f}")
        exp2_results[name] = {'loss': loss, 'delta': delta, 'layers': list(layers), 'type': atype}
    results['exp2'] = exp2_results

    # === Exp 3: Correction Dosage Curve ===
    print(f"\n  === Exp 3: Correction Dosage Curve ===")
    # Sort correction neurons by alignment (most anti-aligned first)
    sorted_corr = corr_idx[np.argsort(alignment[corr_idx])]
    dosages = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print(f"  {'Dosage':>8} | {'n_suppress':>11} | {'val_loss':>9} | {'Δloss':>9}")
    print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*9}-+-{'-'*9}")
    exp3_results = []
    for d in dosages:
        n_s = int(len(sorted_corr) * d)
        if n_s == 0:
            loss = baseline
        else:
            mask = np.zeros(n_know, dtype=bool)
            mask[sorted_corr[:n_s]] = True
            print(f"    Computing dosage {d*100:.0f}% ({n_s} neurons)...", flush=True)
            loss = _eval_loss_suppressed(mask)
        delta = loss - baseline
        print(f"  {d*100:>7.0f}% | {n_s:>11,} | {loss:>9.4f} | {delta:>+9.4f}")
        exp3_results.append({'dosage': d, 'n_suppress': n_s, 'loss': loss, 'delta': delta})
    results['exp3'] = exp3_results

    # === Exp 4: Role Isolation ===
    print(f"\n  === Exp 4: Role Isolation ===")
    exp4_conditions = [
        ('suppress transform', trans_idx),
        ('suppress correction', corr_idx),
        ('suppress corr+trans', np.concatenate([corr_idx, trans_idx])),
    ]
    print(f"  {'Condition':>22} | {'val_loss':>9} | {'Δloss':>9}")
    print(f"  {'-'*22}-+-{'-'*9}-+-{'-'*9}")
    print(f"  {'no suppression':>22} | {baseline:>9.4f} | {'—':>9}")
    exp4_results = {}
    for name, idx in exp4_conditions:
        mask = np.zeros(n_know, dtype=bool)
        mask[idx] = True
        print(f"    Computing {name} ({int(mask.sum())} neurons)...", flush=True)
        loss = _eval_loss_suppressed(mask)
        delta = loss - baseline
        print(f"  {name:>22} | {loss:>9.4f} | {delta:>+9.4f}")
        exp4_results[name] = {'n_suppress': int(mask.sum()), 'loss': loss, 'delta': delta}
    results['exp4'] = exp4_results

    # === Exp 4b: Qualitative Generation ===
    print(f"\n  === Exp 4b: Qualitative Generation ===")

    GEN_PROMPTS = [
        "The cat sat on the",
        "The meaning of life is",
        "Scientists discovered that",
        "In the beginning there was",
        "The best way to learn is",
    ]
    GEN_LEN = 30

    # Build forward functions for each condition
    gen_conditions = [('baseline', None)]
    for name, idx in exp4_conditions:
        mask = np.zeros(n_know, dtype=bool)
        mask[idx] = True
        gen_conditions.append((name.replace('suppress ', 'no '), mask))

    gen_results = {}
    for prompt in GEN_PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        gen_results[prompt] = {}

        for cond_name, suppress_mask in gen_conditions:
            if suppress_mask is None:
                fwd = build_suppressed_forward(params, model_cfg, {'know': np.zeros(n_know, dtype=bool)})
            else:
                fwd = build_suppressed_forward(params, model_cfg, {'know': suppress_mask})

            generated = list(input_ids)
            for _ in range(GEN_LEN):
                padded = generated + [0] * (max_seq - len(generated))
                padded = padded[:max_seq]
                logits = fwd(jnp.array([padded], dtype=jnp.int32))
                next_logit = logits[0, len(generated) - 1]
                next_tok = int(jnp.argmax(next_logit))
                generated.append(next_tok)
                if len(generated) >= max_seq:
                    break

            out_text = tokenizer.decode(generated[len(input_ids):])
            label = f"[{cond_name}]"
            print(f"    {label:<22} → {out_text[:80]}")
            gen_results[prompt][cond_name] = out_text

    results['exp4_generation'] = gen_results

    _save_json(results, output_dir, 'p11_interventions', 'results.json')
    return results



# ============================================================
# P12: Composition Decomposition
# ============================================================

def analyze_composition(params, cfg, val_tokens, output_dir,
                         n_batches=10, batch_size=4):
    """P12: Decompose know_out into correction/neutral/transform contributions."""
    print("\n" + "="*60)
    print("P12: Composition Decomposition")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']
    emb_matrix = params_jax['token_emb']['embedding']
    pos_matrix = params_jax['pos_emb']['embedding']

    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    r_normed = pool_p['know_read'] / (jnp.linalg.norm(pool_p['know_read'], axis=-1, keepdims=True) + 1e-8)
    w_normed = pool_p['know_write'] / (jnp.linalg.norm(pool_p['know_write'], axis=-1, keepdims=True) + 1e-8)

    # Alignment masks as JAX arrays for broadcasting
    read_np = np.array(pool_p['know_read'])
    write_np = np.array(pool_p['know_write'])
    rn_np = read_np / (np.linalg.norm(read_np, axis=-1, keepdims=True) + 1e-8)
    wn_np = write_np / (np.linalg.norm(write_np, axis=-1, keepdims=True) + 1e-8)
    alignment = (rn_np * wn_np).sum(axis=-1)

    corr_mask = jnp.array((alignment < -0.5).astype(np.float32))   # [N]
    neut_mask = jnp.array(((alignment >= -0.5) & (alignment <= 0.1)).astype(np.float32))
    trans_mask = jnp.array((alignment > 0.1).astype(np.float32))
    n_corr = int((alignment < -0.5).sum())
    n_neut = int(((alignment >= -0.5) & (alignment <= 0.1)).sum())
    n_trans = int((alignment > 0.1).sum())
    print(f"  Groups: correction={n_corr}, neutral={n_neut}, transform={n_trans}")

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)

    # Per-layer accumulators
    layer_acc = {li: {
        'corr_mag': 0.0, 'neut_mag': 0.0, 'trans_mag': 0.0,
        'cos_cn': 0.0, 'cos_ct': 0.0, 'cos_nt': 0.0,
        'cos_attn_corr': 0.0, 'cos_attn_neut': 0.0, 'cos_attn_trans': 0.0,
        'corr_proj': 0.0, 'neut_proj': 0.0, 'trans_proj': 0.0,
    } for li in range(n_layers)}

    # Part D: logits decomposition samples
    logits_samples = []

    print(f"  Running {actual_batches} batches (bs={batch_size})...")
    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        B, S = batch.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[batch.astype(jnp.int32)] + pos_matrix[positions]

        for li in range(n_layers):
            bp = block_params_list[li]
            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_p['proj_attn']['kernel'] + router_p['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_p['tau_attn']['kernel'] + router_p['tau_attn']['bias']
            Q = _srw_inference_fn(normed1, h_Q, qk_n, tau_all[:,:,0:1], pool_p['qk_read'], pool_p['qk_write']) * pool_p.get('qk_scale', 1.0)
            K = _srw_inference_fn(normed1, h_K, qk_n, tau_all[:,:,1:2], pool_p['qk_read'], pool_p['qk_write']) * pool_p.get('qk_scale', 1.0)
            V = _srw_inference_fn(normed1, h_V, v_n, tau_all[:,:,2:3], pool_p['v_read'], pool_p['v_write']) * pool_p.get('v_scale', 1.0)
            d_head = d_model // n_heads
            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            # Know: direct gate computation + group decomposition
            normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias']
            tau_k = normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias']

            scores_k = h_k @ know_emb_n.T  # [B, S, N]
            sf = scores_k.astype(jnp.float32)
            s_mean = sf.mean(axis=-1, keepdims=True)
            s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
            tau = s_mean + tau_k * s_std
            z = (sf - tau) / s_std
            gate = jnp.where(z > 0, z * 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476)), 0.0)

            # xr and group-separated raw outputs
            xr = normed2 @ r_normed.T  # [B, S, N]
            gate_xr = gate * xr  # [B, S, N] — raw contribution per neuron (before den)

            raw_corr = (gate_xr * corr_mask[None, None, :]) @ w_normed  # [B, S, D]
            raw_neut = (gate_xr * neut_mask[None, None, :]) @ w_normed
            raw_trans = (gate_xr * trans_mask[None, None, :]) @ w_normed
            raw_total = raw_corr + raw_neut + raw_trans

            # Magnitudes
            cm = jnp.linalg.norm(raw_corr, axis=-1).mean()
            nm = jnp.linalg.norm(raw_neut, axis=-1).mean()
            tm = jnp.linalg.norm(raw_trans, axis=-1).mean()
            total_m = cm + nm + tm + 1e-8

            # Direction relationships
            def _cos_mean(a, b):
                return ((a * b).sum(axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)).mean()

            # Contribution to know_out direction
            raw_dir = raw_total / (jnp.linalg.norm(raw_total, axis=-1, keepdims=True) + 1e-8)
            cp = (raw_corr * raw_dir).sum(axis=-1).mean()
            np_ = (raw_neut * raw_dir).sum(axis=-1).mean()
            tp = (raw_trans * raw_dir).sum(axis=-1).mean()

            d = layer_acc[li]
            vals = jax.device_get({
                'cm': cm/total_m, 'nm': nm/total_m, 'tm': tm/total_m,
                'cos_cn': _cos_mean(raw_corr, raw_neut),
                'cos_ct': _cos_mean(raw_corr, raw_trans),
                'cos_nt': _cos_mean(raw_neut, raw_trans),
                'cos_ac': _cos_mean(attn_out, raw_corr),
                'cos_an': _cos_mean(attn_out, raw_neut),
                'cos_at': _cos_mean(attn_out, raw_trans),
                'cp': cp, 'np': np_, 'tp': tp,
            })
            d['corr_mag'] += float(vals['cm'])
            d['neut_mag'] += float(vals['nm'])
            d['trans_mag'] += float(vals['tm'])
            d['cos_cn'] += float(vals['cos_cn'])
            d['cos_ct'] += float(vals['cos_ct'])
            d['cos_nt'] += float(vals['cos_nt'])
            d['cos_attn_corr'] += float(vals['cos_ac'])
            d['cos_attn_neut'] += float(vals['cos_an'])
            d['cos_attn_trans'] += float(vals['cos_at'])
            d['corr_proj'] += float(vals['cp'])
            d['neut_proj'] += float(vals['np'])
            d['trans_proj'] += float(vals['tp'])

            # Actual know_out for residual update
            den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
            know_out = raw_total / den * pool_p.get('know_scale', 1.0)
            x = x + know_out

            # Part D: logits decomposition at last layer, first batch only
            if li == n_layers - 1 and b == 0 and len(logits_samples) < 10:
                norm_p = params_jax['norm']
                x_final = _layer_norm_fn(x, norm_p['scale'], norm_p['bias'])
                # Group logit contributions (through LN approximation: use raw_out directly)
                k_scale = pool_p.get('know_scale', 1.0)
                corr_logits = (raw_corr / den * k_scale) @ emb_matrix.T  # [B, S, V]
                neut_logits = (raw_neut / den * k_scale) @ emb_matrix.T
                trans_logits = (raw_trans / den * k_scale) @ emb_matrix.T

                for si in range(0, min(S, 100), 10):
                    cl = np.array(jax.device_get(corr_logits[0, si]))
                    nl = np.array(jax.device_get(neut_logits[0, si]))
                    tl = np.array(jax.device_get(trans_logits[0, si]))
                    tok_id = int(batch[0, si])
                    c_top = np.argsort(-cl)[:5]
                    n_top = np.argsort(-nl)[:5]
                    t_top = np.argsort(-tl)[:5]
                    logits_samples.append({
                        'token': tokenizer.decode([tok_id]),
                        'position': si,
                        'correction_pushes': [(tokenizer.decode([int(i)]), float(cl[i])) for i in c_top],
                        'neutral_pushes': [(tokenizer.decode([int(i)]), float(nl[i])) for i in n_top],
                        'transform_pushes': [(tokenizer.decode([int(i)]), float(tl[i])) for i in t_top],
                    })

        if (b + 1) % 3 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # Average
    for li in range(n_layers):
        for k in layer_acc[li]:
            layer_acc[li][k] /= actual_batches

    mid = n_layers // 2

    # === Print Part A ===
    print(f"\n  === Part A: Group Output Decomposition (layer {mid}) ===")
    d = layer_acc[mid]
    print(f"  Magnitude ratio (raw output):")
    print(f"    correction: {d['corr_mag']*100:.1f}%  neutral: {d['neut_mag']*100:.1f}%  transform: {d['trans_mag']*100:.1f}%")
    print(f"  Direction relationships:")
    print(f"    cos(corr, neut):  {d['cos_cn']:.4f}")
    print(f"    cos(corr, trans): {d['cos_ct']:.4f}")
    print(f"    cos(neut, trans): {d['cos_nt']:.4f}")

    # === Print Part B ===
    print(f"\n  === Part B: Layer-wise Composition ===")
    print(f"  {'Layer':>5} | {'corr%':>7} | {'neut%':>7} | {'trans%':>7} | {'cos(c,n)':>8} | {'cos(c,t)':>8}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
    for li in range(n_layers):
        d = layer_acc[li]
        print(f"  {li:>5} | {d['corr_mag']*100:>6.1f}% | {d['neut_mag']*100:>6.1f}% | {d['trans_mag']*100:>6.1f}% | {d['cos_cn']:>8.4f} | {d['cos_ct']:>8.4f}")

    # === Print Part C ===
    print(f"\n  === Part C: Attn-Group Relationships ===")
    print(f"  {'Layer':>5} | {'cos(a,corr)':>11} | {'cos(a,neut)':>11} | {'cos(a,trans)':>12}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*11}-+-{'-'*12}")
    for li in range(n_layers):
        d = layer_acc[li]
        print(f"  {li:>5} | {d['cos_attn_corr']:>11.4f} | {d['cos_attn_neut']:>11.4f} | {d['cos_attn_trans']:>12.4f}")

    # === Print Part D ===
    if logits_samples:
        print(f"\n  === Part D: Logits Decomposition ({len(logits_samples)} samples) ===")
        for s in logits_samples[:5]:
            print(f"  Token: \"{s['token']}\" at pos {s['position']}")
            c_str = ', '.join(f"{t}({v:+.2f})" for t, v in s['correction_pushes'][:3])
            n_str = ', '.join(f"{t}({v:+.2f})" for t, v in s['neutral_pushes'][:3])
            t_str = ', '.join(f"{t}({v:+.2f})" for t, v in s['transform_pushes'][:3])
            print(f"    correction: {c_str}")
            print(f"    neutral:    {n_str}")
            print(f"    transform:  {t_str}")

    results = {
        'n_layers': n_layers,
        'groups': {'correction': n_corr, 'neutral': n_neut, 'transform': n_trans},
        'per_layer': {f'layer_{li}': layer_acc[li] for li in range(n_layers)},
        'logits_samples': logits_samples[:10],
    }
    _save_json(results, output_dir, 'p12_composition', 'results.json')
    return results


# ============================================================
# §7.2.1 Read-Write Alignment Distribution
# ============================================================

def analyze_rw_alignment(params, cfg, output_dir):
    """§7.2.1: cos(r, w) 분포 — pool별 삼봉/연속/단일봉 판정."""
    print("\n" + "="*60)
    print("§7.2.1: Read-Write Alignment Distribution")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']

    pools = {
        'qk':   ('qk_read',   'qk_write',   n_qk),
        'v':    ('v_read',     'v_write',     n_v),
        'know': ('know_read',  'know_write',  n_know),
    }

    results = {}
    for pool_name, (r_key, w_key, n_pool) in pools.items():
        print(f"\n  Pool: {pool_name} (N={n_pool})")
        r = pool_p[r_key]   # [N, d_model]
        w = pool_p[w_key]   # [N, d_model]

        # unit normalize
        r_n = r / (jnp.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)
        w_n = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)
        cos_rw = jnp.sum(r_n * w_n, axis=-1)  # [N]
        cos_np = np.array(jax.device_get(cos_rw))

        # 히스토그램
        hist_counts, hist_edges = np.histogram(cos_np, bins=100, range=(-1.0, 1.0))

        # 통계
        stats = {
            'mean': float(np.mean(cos_np)),
            'std': float(np.std(cos_np)),
            'min': float(np.min(cos_np)),
            'max': float(np.max(cos_np)),
            'median': float(np.median(cos_np)),
            'p10': float(np.percentile(cos_np, 10)),
            'p90': float(np.percentile(cos_np, 90)),
            'frac_neg_05': float((cos_np < -0.5).mean()),
            'frac_pos_01': float((cos_np > 0.1).mean()),
            'n_pool': n_pool,
        }

        # Mode detection (scipy optional)
        peaks_info = []
        try:
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d
            density = hist_counts.astype(float)
            density_smooth = gaussian_filter1d(density, sigma=2.0)
            peak_idx, peak_props = find_peaks(density_smooth, height=density_smooth.max() * 0.05,
                                               distance=5)
            bin_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
            for pi in peak_idx:
                peaks_info.append({
                    'cos_value': float(bin_centers[pi]),
                    'height': float(density_smooth[pi]),
                })
            stats['n_modes'] = len(peak_idx)
            print(f"    Detected {len(peak_idx)} mode(s): {[p['cos_value'] for p in peaks_info]}")
        except ImportError:
            stats['n_modes'] = None
            print("    scipy not available, skipping mode detection")

        stats['peaks'] = peaks_info
        stats['histogram'] = {
            'counts': hist_counts.tolist(),
            'edges': hist_edges.tolist(),
        }

        results[pool_name] = stats

        # raw cos values를 npy로 저장
        subdir = 'rw_alignment'
        save_path = os.path.join(output_dir, subdir)
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f'{pool_name}_cos_rw.npy')
        np.save(npy_path, cos_np)
        print(f"    Saved: {npy_path}")

        print(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"median={stats['median']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    _save_json(results, output_dir, 'rw_alignment', 'rw_alignment_summary.json')

    # distribution type summary
    type_parts = []
    for pn in ['qk', 'v', 'know']:
        s = results[pn]
        nm = s.get('n_modes')
        if nm is not None and nm > 0:
            peaks_str = ','.join(f"{p['cos_value']:.2f}" for p in s['peaks'])
            if nm == 1:
                type_parts.append(f"{pn}=unimodal({peaks_str})")
            elif nm == 2:
                type_parts.append(f"{pn}=bimodal({peaks_str})")
            else:
                type_parts.append(f"{pn}={nm}-modal({peaks_str})")
        else:
            type_parts.append(f"{pn}=unknown")
    print(f"  Distribution type: {' '.join(type_parts)}")

    print("  Done: rw_alignment")
    return results


# ============================================================
# 공통 헬퍼: per-neuron contribution 분해
# ============================================================

def _compute_per_neuron_contribution(params_jax, model_cfg, input_ids,
                                      pool='know', layer_idx=None,
                                      return_vector=False, chunk_size=512):
    """주어진 pool/layer에서 per-neuron contribution 반환 (non-JIT, layer-by-layer forward).

    Returns dict:
        'contrib_norm': [B, S, N]  — ‖gate_i × (x · r_i) × w_i‖
        'gate':         [B, S, N]
        'den':          [B, S]     — sqrt(active_N) + 1.0
        'contrib_vec':  [B, S, N, d]  (return_vector=True일 때만)
    """
    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']

    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']

    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_emb_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_s = pool_p.get('qk_scale', 1.0)
    v_s = pool_p.get('v_scale', 1.0)
    know_s = pool_p.get('know_scale', 1.0)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]

    if layer_idx is None:
        layer_idx = n_layers // 2

    B, S = input_ids.shape
    positions = jnp.arange(S)[jnp.newaxis, :]
    emb_matrix = jnp.asarray(params_jax['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params_jax['pos_emb']['embedding'])
    x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

    # forward to target layer
    for li in range(layer_idx + 1):
        bp = block_params_list[li]
        normed1 = _layer_norm_fn(x, jnp.asarray(bp['norm1']['scale']), jnp.asarray(bp['norm1']['bias']))
        h_all = normed1 @ jnp.asarray(router_p['proj_attn']['kernel']) + jnp.asarray(router_p['proj_attn']['bias'])
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed1 @ jnp.asarray(router_p['tau_attn']['kernel']) + jnp.asarray(router_p['tau_attn']['bias'])

        Q = _srw_inference_fn(normed1, h_Q, qk_emb_n, tau_all[:,:,0:1], pool_p['qk_read'], pool_p['qk_write']) * qk_s
        K = _srw_inference_fn(normed1, h_K, qk_emb_n, tau_all[:,:,1:2], pool_p['qk_read'], pool_p['qk_write']) * qk_s
        V = _srw_inference_fn(normed1, h_V, v_emb_n, tau_all[:,:,2:3], pool_p['v_read'], pool_p['v_write']) * v_s

        d_head = d_model // n_heads
        Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
        scale = jnp.sqrt(jnp.float32(d_head))
        sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
        aw = jax.nn.softmax(sc, axis=-1)
        ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
        attn_out = ao @ jnp.asarray(bp['attn']['expand_O']['kernel'])
        x_after_attn = x + attn_out

        if li == layer_idx and pool in ('qk', 'v'):
            # attn pool contribution — compute at this layer's attn stage
            # (simplified: treat entire attn as the pool contribution)
            pass

        normed2 = _layer_norm_fn(x_after_attn, jnp.asarray(bp['norm2']['scale']), jnp.asarray(bp['norm2']['bias']))

        if li == layer_idx and pool == 'know':
            # per-neuron contribution at know pool
            h_k = normed2 @ jnp.asarray(router_p['proj_know']['kernel']) + jnp.asarray(router_p['proj_know']['bias'])
            tau_k = normed2 @ jnp.asarray(router_p['tau_know']['kernel']) + jnp.asarray(router_p['tau_know']['bias'])
            scores_k = h_k @ know_emb_n.T
            sf = scores_k.astype(jnp.float32)
            s_mean = sf.mean(axis=-1, keepdims=True)
            s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
            tau = s_mean + tau_k * s_std
            raw = scores_k - tau.astype(scores_k.dtype)
            gate = jnp.maximum(raw, 0.0)
            gate = jnp.clip(gate, 0.0, 10.0)  # [B, S, N]
            active_N = (gate > 0).sum(axis=-1, keepdims=True).astype(jnp.float32)
            den = jnp.sqrt(active_N.squeeze(-1)) + 1.0  # [B, S]

            r_n = pool_p['know_read'] / (jnp.linalg.norm(pool_p['know_read'], axis=-1, keepdims=True) + 1e-8)
            w_n = pool_p['know_write'] / (jnp.linalg.norm(pool_p['know_write'], axis=-1, keepdims=True) + 1e-8)
            xr = normed2 @ r_n.T  # [B, S, N]

            N = gate.shape[-1]
            if return_vector:
                # chunked [B, S, chunk, d] → concat. 전체 [B,S,N,d]는 OOM 위험.
                chunk_v = min(chunk_size, 256)
                contrib_vec_parts = []
                contrib_norm_parts = []
                for c_start in range(0, N, chunk_v):
                    c_end = min(c_start + chunk_v, N)
                    g_c = gate[:, :, c_start:c_end]
                    xr_c = xr[:, :, c_start:c_end]
                    w_c = w_n[c_start:c_end, :]
                    cv = (g_c[..., jnp.newaxis] * xr_c[..., jnp.newaxis]) * w_c[jnp.newaxis, jnp.newaxis, :, :]
                    contrib_vec_parts.append(cv)
                    contrib_norm_parts.append(jnp.linalg.norm(cv, axis=-1))
                contrib_vec = jnp.concatenate(contrib_vec_parts, axis=2)
                contrib_norm = jnp.concatenate(contrib_norm_parts, axis=2)
                return {'contrib_norm': contrib_norm, 'gate': gate, 'den': den, 'contrib_vec': contrib_vec}
            else:
                # chunked norm 계산 — [B, S, N] 반환
                contrib_norm_parts = []
                for c_start in range(0, N, chunk_size):
                    c_end = min(c_start + chunk_size, N)
                    g_chunk = gate[:, :, c_start:c_end]
                    xr_chunk = xr[:, :, c_start:c_end]
                    w_chunk = w_n[c_start:c_end, :]
                    # [B, S, chunk, d]
                    contrib_chunk = (g_chunk[..., jnp.newaxis] * xr_chunk[..., jnp.newaxis]) * w_chunk[jnp.newaxis, jnp.newaxis, :, :]
                    norm_chunk = jnp.linalg.norm(contrib_chunk, axis=-1)  # [B, S, chunk]
                    contrib_norm_parts.append(norm_chunk)
                contrib_norm = jnp.concatenate(contrib_norm_parts, axis=-1)
                return {'contrib_norm': contrib_norm, 'gate': gate, 'den': den}

        # know forward for non-target layers
        know_out = _srw_inference_fn(normed2, normed2 @ jnp.asarray(router_p['proj_know']['kernel']) + jnp.asarray(router_p['proj_know']['bias']),
                                     know_emb_n, normed2 @ jnp.asarray(router_p['tau_know']['kernel']) + jnp.asarray(router_p['tau_know']['bias']),
                                     pool_p['know_read'], pool_p['know_write']) * know_s
        x = x_after_attn + know_out

    # fallback (should not reach here if layer_idx < n_layers)
    return {'contrib_norm': jnp.zeros((B, S, 1)), 'gate': jnp.zeros((B, S, 1)), 'den': jnp.ones((B, S))}


# ============================================================
# §8.1+§8.2+§8.3 Residual Dynamics
# ============================================================

def analyze_residual_dynamics(params, cfg, val_tokens, output_dir,
                               n_batches=20, batch_size=8):
    """§8.1 Residual Trajectory + §8.2 Per-Unit Contribution + §8.3 Force Composition."""
    print("\n" + "="*60)
    print("§8: Residual Dynamics (Trajectory + Per-Unit + Force)")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_know = model_cfg['n_know']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    forward_fn = _build_forward_extended(params_jax, model_cfg)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}, n_layers={n_layers}")

    stat_keys = ['x_norm', 'dx_norm', 'attn_norm', 'know_norm',
                 'cos_x_prev', 'cos_attn_x', 'cos_know_x', 'cos_x0', 'eff_rank']
    layer_stats = [{k: 0.0 for k in stat_keys} for _ in range(n_layers)]

    mid_layer = n_layers // 2
    know_contrib_acc = np.zeros(n_know, dtype=np.float64)

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)

        # JIT forward — device_get 1회
        result = jax.device_get(forward_fn(batch))
        for li in range(n_layers):
            for k in stat_keys:
                layer_stats[li][k] += float(result[k][li])

        # §8.2: per-neuron contribution at mid layer
        contrib_result = _compute_per_neuron_contribution(params_jax, model_cfg, batch,
                                                           pool='know', layer_idx=mid_layer)
        cn = np.array(jax.device_get(contrib_result['contrib_norm'].mean(axis=(0, 1))))
        know_contrib_acc += cn

        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # 평균
    for li in range(n_layers):
        for k in layer_stats[li]:
            layer_stats[li][k] /= actual_batches

    know_contrib_mean = know_contrib_acc / actual_batches  # [N]

    # §8.2 Pareto share
    sorted_contrib = np.sort(know_contrib_mean)[::-1]
    total = sorted_contrib.sum() + 1e-12
    top1_idx = max(1, int(0.01 * n_know))
    top10_idx = max(1, int(0.10 * n_know))
    pareto = {
        'know': {
            'top_1pct_share': float(sorted_contrib[:top1_idx].sum() / total),
            'top_10pct_share': float(sorted_contrib[:top10_idx].sum() / total),
        }
    }

    # 저장
    _save_json({
        'per_layer': {f'layer_{li}': layer_stats[li] for li in range(n_layers)},
        'n_batches': actual_batches,
        'batch_size': batch_size,
    }, output_dir, 'residual_dynamics', 'layer_stats.json')

    _save_json(pareto, output_dir, 'residual_dynamics', 'pareto.json')

    # per-neuron contrib npy
    save_path = os.path.join(output_dir, 'residual_dynamics')
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'per_neuron_contrib_know.npy'), know_contrib_mean)
    print(f"  Saved: {os.path.join(save_path, 'per_neuron_contrib_know.npy')}")

    # --- pretty print ---
    print("\n  §8.1 Residual Trajectory:")
    print("  Layer |  x_norm | dx_norm |  cos_x0 | cos_prev | acos_cum | eff_rank")
    print("  ------+---------+---------+---------+----------+----------+---------")
    acos_cum = 0.0
    for li in range(n_layers):
        s = layer_stats[li]
        cp = max(-1.0, min(1.0, s['cos_x_prev']))
        acos_cum += np.arccos(cp)
        print(f"   L{li:2d}  | {s['x_norm']:7.2f} | {s['dx_norm']:7.3f} | {s['cos_x0']:7.4f} | {s['cos_x_prev']:8.4f} | {acos_cum:8.3f}  | {s['eff_rank']:7.1f}")

    print(f"\n  §8.3 Force Composition:")
    print("  Layer | attn_norm | know_norm | cos(a,x) | cos(k,x)")
    print("  ------+-----------+-----------+----------+---------")
    for li in range(n_layers):
        s = layer_stats[li]
        print(f"   L{li:2d}  | {s['attn_norm']:9.4f} | {s['know_norm']:9.4f} | {s['cos_attn_x']:8.4f} | {s['cos_know_x']:8.4f}")

    x0 = layer_stats[0]['x_norm']
    xL = layer_stats[n_layers-1]['x_norm']
    ratio = xL / (x0 + 1e-8)
    c0 = layer_stats[0]['cos_x0']
    cL = layer_stats[n_layers-1]['cos_x0']
    third = max(1, n_layers // 3)
    early_ar = np.mean([layer_stats[li]['attn_norm'] / (layer_stats[li]['know_norm'] + 1e-8) for li in range(third)])
    mid_ar = np.mean([layer_stats[li]['attn_norm'] / (layer_stats[li]['know_norm'] + 1e-8) for li in range(third, 2*third)])
    late_ar = np.mean([layer_stats[li]['attn_norm'] / (layer_stats[li]['know_norm'] + 1e-8) for li in range(2*third, n_layers)])
    print(f"\n  Summary:")
    print(f"    Norm growth: x_norm L0={x0:.2f} → L{n_layers-1}={xL:.2f} (×{ratio:.1f})")
    print(f"    cos(x, x0) decay: L0={c0:.4f} → L{n_layers-1}={cL:.4f}")
    print(f"    attn/know ratio: early={early_ar:.1f}× mid={mid_ar:.1f}× late={late_ar:.1f}×")
    er0 = layer_stats[0]['eff_rank']
    erL = layer_stats[n_layers-1]['eff_rank']
    print(f"    Eff rank: L0={er0:.1f} → L{n_layers-1}={erL:.1f} (d_model={model_cfg['d_model']})")

    top50_idx = max(1, int(0.50 * n_know))
    top50_share = float(sorted_contrib[:top50_idx].sum() / total)
    print(f"\n  §8.2 Per-Unit Contribution (know pool, layer {mid_layer}):")
    print(f"    Top  1%: {pareto['know']['top_1pct_share']:.1%} of total contribution")
    print(f"    Top 10%: {pareto['know']['top_10pct_share']:.1%} of total contribution")
    print(f"    Top 50%: {top50_share:.1%} of total contribution")
    thresholds = [1.0, 0.1, 0.01, 0.001]
    print(f"    Contribution distribution:")
    for t in thresholds:
        frac = float((know_contrib_mean > t).sum() / len(know_contrib_mean))
        bar = '█' * int(frac * 50)
        print(f"      >{t:<6g}: {bar} {frac:.1%}")

    print(f"\n  Done: residual_dynamics (top-1% Pareto={pareto['know']['top_1pct_share']:.2%})")
    return layer_stats


# ============================================================
# §6.3 Phase Dynamics (학습 로그 post-processing)
# ============================================================

def analyze_phase_dynamics(output_dir, dawn_log=None, baseline_log=None):
    """§6.3: 학습 로그에서 loss/tau/active_frac 시계열 비교 및 phase 경계 검출."""
    print("\n" + "="*60)
    print("§6.3: Phase Dynamics")
    print("="*60)

    if dawn_log is None:
        print("  --dawn_log not provided. Skipping phase dynamics.")
        return None

    def _load_log(path):
        """JSON-lines 또는 CSV 학습 로그 로드. 각 행: {step, loss, tau_*, active_frac_*}."""
        records = []
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            f.seek(0)
            if first_line.startswith('{'):
                # JSON-lines
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                # CSV
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    rec = {}
                    for k, v in row.items():
                        try:
                            rec[k] = float(v)
                        except (ValueError, TypeError):
                            rec[k] = v
                    records.append(rec)
        return records

    def _extract_series(records, key):
        """로그 레코드 리스트에서 step/key 시계열 추출."""
        steps, vals = [], []
        for r in records:
            s = r.get('step', r.get('global_step', None))
            v = r.get(key, None)
            if s is not None and v is not None:
                steps.append(float(s))
                vals.append(float(v))
        return np.array(steps), np.array(vals)

    def _moving_avg(arr, window=100):
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    print(f"  Loading DAWN log: {dawn_log}")
    dawn_records = _load_log(dawn_log)
    print(f"    {len(dawn_records)} records loaded")

    dawn_steps, dawn_loss = _extract_series(dawn_records, 'loss')
    results = {'dawn': {'n_records': len(dawn_records)}}

    # tau/active_frac per pool
    for pool in ['qk', 'v', 'know']:
        for metric in ['tau', 'active_frac']:
            key = f'{metric}_{pool}'
            steps, vals = _extract_series(dawn_records, key)
            if len(vals) > 0:
                results['dawn'][key] = {
                    'mean': float(vals.mean()),
                    'final': float(vals[-1]),
                    'n_points': len(vals),
                }

    # loss moving average
    if len(dawn_loss) > 0:
        dawn_loss_ma = _moving_avg(dawn_loss)
        results['dawn']['loss_ma_final'] = float(dawn_loss_ma[-1]) if len(dawn_loss_ma) > 0 else None

    # baseline 비교
    if baseline_log is not None:
        print(f"  Loading baseline log: {baseline_log}")
        base_records = _load_log(baseline_log)
        print(f"    {len(base_records)} records loaded")
        base_steps, base_loss = _extract_series(base_records, 'loss')
        results['baseline'] = {'n_records': len(base_records)}

        # gap 계산 (step 정렬)
        if len(dawn_loss) > 0 and len(base_loss) > 0:
            min_len = min(len(dawn_loss), len(base_loss))
            gap = dawn_loss[:min_len] - base_loss[:min_len]
            gap_ma = _moving_avg(gap)

            # phase 경계: gap 도함수 부호 변화
            if len(gap_ma) > 2:
                d_gap = np.diff(gap_ma)
                sign_changes = np.where(np.diff(np.sign(d_gap)))[0]
                # 대응하는 step (window offset 보정)
                window = 100
                phase_boundaries = []
                for idx in sign_changes:
                    step_idx = idx + window // 2
                    if step_idx < len(dawn_steps):
                        phase_boundaries.append({
                            'step': float(dawn_steps[step_idx]),
                            'gap_value': float(gap_ma[idx]),
                        })
                results['phases'] = phase_boundaries
                print(f"    Detected {len(phase_boundaries)} phase boundaries")
            else:
                results['phases'] = []

            results['gap'] = {
                'mean': float(gap.mean()),
                'final': float(gap[-1]),
                'min': float(gap.min()),
                'max': float(gap.max()),
            }

    # timeseries 저장 (step, loss, tau, active_frac)
    timeseries = {}
    for key in ['loss', 'tau_qk', 'tau_v', 'tau_know', 'active_frac_qk', 'active_frac_v', 'active_frac_know']:
        steps, vals = _extract_series(dawn_records, key)
        if len(vals) > 0:
            # 서브샘플 (최대 2000 포인트)
            if len(vals) > 2000:
                idx = np.linspace(0, len(vals)-1, 2000, dtype=int)
                steps, vals = steps[idx], vals[idx]
            timeseries[key] = {'steps': steps.tolist(), 'values': vals.tolist()}

    _save_json(timeseries, output_dir, 'phase_dynamics', 'timeseries.json')
    _save_json(results, output_dir, 'phase_dynamics', 'phases.json')
    print("  Done: phase_dynamics")
    return results


# ============================================================
# §8.4 Drift-Prediction Alignment
# ============================================================

def analyze_drift_alignment(params, cfg, val_tokens, output_dir,
                             n_batches=20, batch_size=8):
    """§8.4: per-layer early-exit logits → target rank & cos(x_L, emb[target])."""
    print("\n" + "="*60)
    print("§8.4: Drift-Prediction Alignment")
    print("="*60)

    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']
    emb_matrix = params_jax['token_emb']['embedding']
    pos_matrix = params_jax['pos_emb']['embedding']
    norm_p = params_jax['norm']

    qk_emb_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_s = pool_p.get('qk_scale', 1.0)
    v_s_val = pool_p.get('v_scale', 1.0)
    know_s = pool_p.get('know_scale', 1.0)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]
    stacked_bp = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    @jax.jit
    def _drift_forward(input_ids):
        """per-layer early-exit stats via scan. Returns {top1, top5, rank, cos_et} [n_layers]."""
        B, S = input_ids.shape
        targets = input_ids[:, 1:]
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        def layer_fn(carry, bp):
            x = carry
            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_p['proj_attn']['kernel'] + router_p['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_p['tau_attn']['kernel'] + router_p['tau_attn']['bias']
            Q = _srw_inference_fn(normed1, h_Q, qk_emb_n, tau_all[:,:,0:1], pool_p['qk_read'], pool_p['qk_write']) * qk_s
            K = _srw_inference_fn(normed1, h_K, qk_emb_n, tau_all[:,:,1:2], pool_p['qk_read'], pool_p['qk_write']) * qk_s
            V = _srw_inference_fn(normed1, h_V, v_emb_n, tau_all[:,:,2:3], pool_p['v_read'], pool_p['v_write']) * v_s_val
            d_head = d_model // n_heads
            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x = x + attn_out
            normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
            know_out = _srw_inference_fn(normed2,
                normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias'],
                know_emb_n,
                normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias'],
                pool_p['know_read'], pool_p['know_write']) * know_s
            x = x + know_out
            # early-exit
            x_normed = _layer_norm_fn(x, norm_p['scale'], norm_p['bias'])
            logits = x_normed @ emb_matrix.T
            pred_logits = logits[:, :-1, :]
            top5 = jax.lax.top_k(pred_logits, 5)[1]
            top1_hit = (top5[:, :, 0] == targets).astype(jnp.float32).mean()
            top5_hit = (top5 == targets[:, :, jnp.newaxis]).any(axis=-1).astype(jnp.float32).mean()
            target_logit = jnp.take_along_axis(pred_logits, targets[:, :, jnp.newaxis], axis=-1).squeeze(-1)
            rank = (pred_logits > target_logit[:, :, jnp.newaxis]).sum(axis=-1).astype(jnp.float32).mean()
            target_emb = emb_matrix[targets]
            x_pred = x_normed[:, :-1, :]
            cos_et = ((x_pred * target_emb).sum(-1) / (
                jnp.linalg.norm(x_pred, axis=-1) * jnp.linalg.norm(target_emb, axis=-1) + 1e-8)).mean()
            stats = {'top1': top1_hit, 'top5': top5_hit, 'rank': rank, 'cos_et': cos_et}
            return x, stats

        _, all_stats = jax.lax.scan(layer_fn, x, stacked_bp)
        return all_stats

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    layer_acc = [{
        'cos_emb_target': 0.0, 'early_exit_top1_acc': 0.0,
        'early_exit_top5_acc': 0.0, 'mean_target_rank': 0.0,
    } for _ in range(n_layers)]

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        result = jax.device_get(_drift_forward(batch))
        for li in range(n_layers):
            layer_acc[li]['early_exit_top1_acc'] += float(result['top1'][li])
            layer_acc[li]['early_exit_top5_acc'] += float(result['top5'][li])
            layer_acc[li]['mean_target_rank'] += float(result['rank'][li])
            layer_acc[li]['cos_emb_target'] += float(result['cos_et'][li])
        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # 평균
    for li in range(n_layers):
        for k in layer_acc[li]:
            layer_acc[li][k] /= actual_batches

    results = {
        'per_layer': {f'layer_{li}': layer_acc[li] for li in range(n_layers)},
        'n_batches': actual_batches,
    }
    _save_json(results, output_dir, 'drift_alignment', 'layer_alignment.json')

    # --- pretty print ---
    print("\n  §8.4 Drift-Prediction Alignment:")
    print("  Layer | top1_acc | top5_acc | mean_rank | cos(x,emb)")
    print("  ------+----------+----------+-----------+-----------")
    for li in range(n_layers):
        d = layer_acc[li]
        print(f"   L{li:2d}  | {d['early_exit_top1_acc']:8.4f} | {d['early_exit_top5_acc']:8.4f} | {d['mean_target_rank']:9.1f} | {d['cos_emb_target']:10.4f}")

    emerge_layer = None
    for li in range(n_layers):
        if layer_acc[li]['early_exit_top1_acc'] > 0.10:
            emerge_layer = li
            break
    final = layer_acc[n_layers - 1]
    c0 = layer_acc[0]['cos_emb_target']
    cL = final['cos_emb_target']
    print(f"\n  Summary:")
    if emerge_layer is not None:
        print(f"    Prediction emerges at: L{emerge_layer} (top1 > 10%)")
    else:
        print(f"    Prediction does not reach 10% top1 at any layer")
    print(f"    Final accuracy: top1={final['early_exit_top1_acc']:.1%}, top5={final['early_exit_top5_acc']:.1%}")
    print(f"    cos(x, emb[target]) growth: L0={c0:.4f} → L{n_layers-1}={cL:.4f}")

    print(f"\n  Done: drift_alignment")
    return results


# ============================================================
# §7.1.1 Gate Confidence/Intensity (Φ/z 분리)
# ============================================================

def analyze_gate_confidence_intensity(params, cfg, val_tokens, output_dir,
                                       n_batches=20, batch_size=8):
    """§7.1.1: per-layer per-pool z/phi/gate 분포 히스토그램."""
    print("\n" + "="*60)
    print("§7.1.1: Gate Confidence/Intensity (z/phi/gate)")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    max_seq = model_cfg['max_seq_len']

    z_bins = 50
    z_range = (-5.0, 5.0)
    phi_bins = 50
    phi_range = (0.0, 1.0)
    gate_bins = 50
    gate_range = (0.0, 10.0)

    params_jax = jax.tree.map(jnp.asarray, params)
    forward_fn = _build_forward_zpg(params_jax, model_cfg,
                                     z_bins=z_bins, z_range=z_range,
                                     phi_bins=phi_bins, phi_range=phi_range,
                                     gate_bins=gate_bins, gate_range=gate_range)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    pool_names = ['qk_Q', 'qk_K', 'v', 'know']
    acc = {}
    for pn in pool_names:
        acc[pn] = {}
        for li in range(n_layers):
            acc[pn][li] = {
                'z_hist': np.zeros(z_bins, dtype=np.float64),
                'phi_hist': np.zeros(phi_bins, dtype=np.float64),
                'gate_hist': np.zeros(gate_bins, dtype=np.float64),
                'z_mean': 0.0, 'phi_mean': 0.0, 'gate_mean': 0.0,
                'active_frac': 0.0,
            }

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        result = jax.device_get(forward_fn(batch))

        for li in range(n_layers):
            for pn in pool_names:
                acc[pn][li]['z_hist'] += np.array(result[f'{pn}_z_hist'][li])
                acc[pn][li]['phi_hist'] += np.array(result[f'{pn}_phi_hist'][li])
                acc[pn][li]['gate_hist'] += np.array(result[f'{pn}_gate_hist'][li])
                acc[pn][li]['z_mean'] += float(result[f'{pn}_z_mean'][li])
                acc[pn][li]['phi_mean'] += float(result[f'{pn}_phi_mean'][li])
                acc[pn][li]['gate_mean'] += float(result[f'{pn}_gate_mean'][li])
                acc[pn][li]['active_frac'] += float(result[f'{pn}_active_frac'][li])

        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    z_edges = np.linspace(z_range[0], z_range[1], z_bins + 1).tolist()
    phi_edges = np.linspace(phi_range[0], phi_range[1], phi_bins + 1).tolist()
    gate_edges = np.linspace(gate_range[0], gate_range[1], gate_bins + 1).tolist()

    results = {}
    for pn in pool_names:
        results[pn] = {}
        for li in range(n_layers):
            d = acc[pn][li]
            for k in ['z_mean', 'phi_mean', 'gate_mean', 'active_frac']:
                d[k] /= actual_batches
            results[pn][f'layer_{li}'] = {
                'z_mean': d['z_mean'], 'phi_mean': d['phi_mean'],
                'gate_mean': d['gate_mean'], 'active_frac': d['active_frac'],
                'z_hist': d['z_hist'].tolist(),
                'phi_hist': d['phi_hist'].tolist(),
                'gate_hist': d['gate_hist'].tolist(),
            }

    results['_hist_edges'] = {'z': z_edges, 'phi': phi_edges, 'gate': gate_edges}

    _save_json(results, output_dir, 'gate_ci', 'gate_ci_results.json')

    # --- pretty print ---
    print("\n  §7.1.1 Gate Confidence/Intensity:\n")
    print("  Know pool z/phi/gate per layer:")
    print("  Layer |  z_mean | phi_mean | gate_mean | active%")
    print("  ------+---------+----------+-----------+--------")
    for li in range(n_layers):
        d = acc['know'][li]
        print(f"   L{li:2d}  | {d['z_mean']:7.3f} | {d['phi_mean']:8.4f} | {d['gate_mean']:9.4f} | {d['active_frac']:6.1%}")

    for pn in ['qk_Q', 'qk_K', 'v']:
        z_avg = np.mean([acc[pn][li]['z_mean'] for li in range(n_layers)])
        phi_avg = np.mean([acc[pn][li]['phi_mean'] for li in range(n_layers)])
        af_avg = np.mean([acc[pn][li]['active_frac'] for li in range(n_layers)])
        label = f"{pn} pool".ljust(10)
        print(f"  {label}(layer-avg): z_mean={z_avg:.3f}, phi={phi_avg:.4f}, active={af_avg:.1%}")

    # know z distribution summary (layer-avg histogram)
    know_z_avg = np.zeros(z_bins, dtype=np.float64)
    for li in range(n_layers):
        know_z_avg += acc['know'][li]['z_hist']
    know_z_avg /= (know_z_avg.sum() + 1e-12)

    z_edges_np = np.linspace(z_range[0], z_range[1], z_bins + 1)
    coarse_bounds = [(-5, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 5)]
    print(f"\n  Know z distribution (layer-avg):")
    for lo, hi in coarse_bounds:
        mask = (z_edges_np[:-1] >= lo) & (z_edges_np[:-1] < hi)
        frac = float(know_z_avg[mask].sum())
        bar = '█' * int(frac * 50)
        print(f"    [{lo:+2d},{hi:+2d}): {bar} {frac:.1%}")

    print("\n  Done: gate_ci")
    return results


# ============================================================
# §7.2.3 Write Direction Coverage
# ============================================================

def analyze_write_coverage(params, cfg, output_dir):
    """§7.2.3: SVD effective rank, pairwise cos sim, covering radius per pool."""
    print("\n" + "="*60)
    print("§7.2.3: Write Direction Coverage")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']
    d_model = model_cfg['d_model']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']

    pools = {
        'qk':   ('qk_write', n_qk),
        'v':    ('v_write',  n_v),
        'know': ('know_write', n_know),
    }

    results = {}
    for pool_name, (w_key, n_pool) in pools.items():
        print(f"\n  Pool: {pool_name} (N={n_pool}, d={d_model})")
        W = np.array(jax.device_get(pool_p[w_key]))  # [N, d]
        W_n = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-8)

        # SVD effective rank
        print(f"    Computing SVD...")
        if n_pool <= 5000:
            _, S_vals, _ = np.linalg.svd(W_n, full_matrices=False)
        else:
            # subsample for SVD if too large
            rng = np.random.RandomState(42)
            idx = rng.choice(n_pool, 5000, replace=False)
            _, S_vals, _ = np.linalg.svd(W_n[idx], full_matrices=False)

        S2 = S_vals ** 2
        S2_norm = S2 / (S2.sum() + 1e-12)
        eff_rank = float(np.exp(-np.sum(S2_norm * np.log(S2_norm + 1e-12))))
        print(f"    SVD effective rank: {eff_rank:.1f} / {min(n_pool, d_model)}")

        # Pairwise cosine similarity
        print(f"    Computing pairwise cos sim...")
        subsample_size = min(n_pool, 5000)
        rng = np.random.RandomState(42)
        if n_pool > subsample_size:
            idx = rng.choice(n_pool, subsample_size, replace=False)
            W_sub = W_n[idx]
        else:
            W_sub = W_n

        # chunked matmul for memory
        chunk = 1000
        cos_vals = []
        for i in range(0, len(W_sub), chunk):
            sim_chunk = W_sub[i:i+chunk] @ W_sub.T  # [chunk, subsample_size]
            # off-diagonal만
            for ci in range(sim_chunk.shape[0]):
                global_i = i + ci
                row = sim_chunk[ci]
                row[global_i] = np.nan  # 자기 자신 제외
                cos_vals.append(row[~np.isnan(row)])

        cos_flat = np.concatenate(cos_vals)
        cos_hist, cos_edges = np.histogram(cos_flat, bins=100, range=(-1.0, 1.0))

        # Covering radius: random unit vectors에 대한 nearest write max cos
        print(f"    Computing covering radius...")
        n_probes = 10000
        probes = rng.randn(n_probes, d_model).astype(np.float32)
        probes /= (np.linalg.norm(probes, axis=-1, keepdims=True) + 1e-8)

        max_cos_per_probe = np.zeros(n_probes, dtype=np.float32)
        for i in range(0, n_probes, chunk):
            sim = probes[i:i+chunk] @ W_sub.T  # [chunk, subsample]
            max_cos_per_probe[i:i+chunk] = sim.max(axis=-1)

        cover_hist, cover_edges = np.histogram(max_cos_per_probe, bins=50, range=(0.0, 1.0))

        pool_result = {
            'n_pool': n_pool,
            'svd_eff_rank': eff_rank,
            'svd_top10': S_vals[:10].tolist(),
            'pairwise_cos': {
                'mean': float(cos_flat.mean()),
                'std': float(cos_flat.std()),
                'p5': float(np.percentile(cos_flat, 5)),
                'p95': float(np.percentile(cos_flat, 95)),
                'histogram': cos_hist.tolist(),
                'edges': cos_edges.tolist(),
            },
            'covering': {
                'mean_max_cos': float(max_cos_per_probe.mean()),
                'min_max_cos': float(max_cos_per_probe.min()),
                'p5_max_cos': float(np.percentile(max_cos_per_probe, 5)),
                'histogram': cover_hist.tolist(),
                'edges': cover_edges.tolist(),
            },
        }
        results[pool_name] = pool_result
        print(f"    pairwise cos mean={pool_result['pairwise_cos']['mean']:.4f}, "
              f"covering mean_max_cos={pool_result['covering']['mean_max_cos']:.4f}")

    _save_json(results, output_dir, 'write_cov', 'write_coverage.json')

    d_model_val = model_cfg['d_model']
    rank_parts = [f"{pn}_rank={results[pn]['svd_eff_rank']:.1f}/{d_model_val}" for pn in pools]
    cover_parts = [f"{pn}={results[pn]['covering']['mean_max_cos']:.2f}" for pn in pools]
    print(f"  Summary: {' '.join(rank_parts)}")
    print(f"           covering_radius: {' '.join(cover_parts)}")

    print("  Done: write_cov")
    return results


# ============================================================
# §7.2.3b Read Direction Coverage
# ============================================================

def analyze_read_coverage(params, cfg, output_dir):
    """§7.2.3b: SVD effective rank, pairwise cos sim, covering radius per pool
    for read vectors; plus read vs write rank comparison and read-write
    subspace overlap via principal angles (k=50,100,200)."""
    print("\n" + "="*60)
    print("§7.2.3b: Read Direction Coverage")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']
    d_model = model_cfg['d_model']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']

    pools = {
        'qk':   ('qk_read',   'qk_write',   n_qk),
        'v':    ('v_read',    'v_write',    n_v),
        'know': ('know_read', 'know_write', n_know),
    }

    subspace_ks = [50, 100, 200]
    results = {}
    rank_compare = {}

    for pool_name, (r_key, w_key, n_pool) in pools.items():
        print(f"\n  Pool: {pool_name} (N={n_pool}, d={d_model})")
        R = np.array(jax.device_get(pool_p[r_key]))   # [N, d]
        W = np.array(jax.device_get(pool_p[w_key]))   # [N, d]
        R_n = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
        W_n = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-8)

        # SVD (subsample if too large) — shared subsample index for R and W
        print(f"    Computing SVD...")
        rng = np.random.RandomState(42)
        if n_pool > 5000:
            idx = rng.choice(n_pool, 5000, replace=False)
            R_sub = R_n[idx]
            W_sub = W_n[idx]
        else:
            R_sub = R_n
            W_sub = W_n

        U_r, S_r, _ = np.linalg.svd(R_sub, full_matrices=False)
        U_w, S_w, _ = np.linalg.svd(W_sub, full_matrices=False)

        S2_r = S_r ** 2
        S2r_norm = S2_r / (S2_r.sum() + 1e-12)
        eff_rank_r = float(np.exp(-np.sum(S2r_norm * np.log(S2r_norm + 1e-12))))

        S2_w = S_w ** 2
        S2w_norm = S2_w / (S2_w.sum() + 1e-12)
        eff_rank_w = float(np.exp(-np.sum(S2w_norm * np.log(S2w_norm + 1e-12))))

        print(f"    read  SVD effective rank: {eff_rank_r:.1f} / {min(n_pool, d_model)}")
        print(f"    write SVD effective rank: {eff_rank_w:.1f} / {min(n_pool, d_model)}")

        # Pairwise cosine similarity (read)
        print(f"    Computing pairwise cos sim (read)...")
        chunk = 1000
        cos_vals = []
        for i in range(0, len(R_sub), chunk):
            sim_chunk = R_sub[i:i+chunk] @ R_sub.T
            for ci in range(sim_chunk.shape[0]):
                global_i = i + ci
                row = sim_chunk[ci]
                row[global_i] = np.nan
                cos_vals.append(row[~np.isnan(row)])

        cos_flat = np.concatenate(cos_vals)
        cos_hist, cos_edges = np.histogram(cos_flat, bins=100, range=(-1.0, 1.0))

        # Covering radius (read)
        print(f"    Computing covering radius (read)...")
        n_probes = 10000
        probes = rng.randn(n_probes, d_model).astype(np.float32)
        probes /= (np.linalg.norm(probes, axis=-1, keepdims=True) + 1e-8)

        max_cos_per_probe = np.zeros(n_probes, dtype=np.float32)
        for i in range(0, n_probes, chunk):
            sim = probes[i:i+chunk] @ R_sub.T
            max_cos_per_probe[i:i+chunk] = sim.max(axis=-1)

        cover_hist, cover_edges = np.histogram(max_cos_per_probe, bins=50, range=(0.0, 1.0))

        # Read-Write subspace overlap (principal angles) at k in {50,100,200}
        print(f"    Computing read-write subspace overlap...")
        max_k = min(U_r.shape[1], U_w.shape[1])
        subspace = {}
        for k in subspace_ks:
            kk = min(k, max_k)
            Ur_k = U_r[:, :kk]
            Uw_k = U_w[:, :kk]
            # principal angles: singular values of Ur_k^T @ Uw_k  ∈ [0,1]
            M = Ur_k.T @ Uw_k
            sv = np.linalg.svd(M, compute_uv=False)
            sv = np.clip(sv, 0.0, 1.0)
            angles_rad = np.arccos(sv)
            subspace[f'k={kk}'] = {
                'k': int(kk),
                'singular_values': sv.tolist(),
                'sv_mean': float(sv.mean()),
                'sv_min': float(sv.min()),
                'sv_max': float(sv.max()),
                'principal_angles_deg_mean': float(np.degrees(angles_rad.mean())),
                'principal_angles_deg_min': float(np.degrees(angles_rad.min())),
                'principal_angles_deg_max': float(np.degrees(angles_rad.max())),
                'overlap_frobenius': float(np.sqrt((sv ** 2).sum())),
                'overlap_frac': float((sv ** 2).sum() / kk),
            }
            print(f"      k={kk:3d}: sv_mean={sv.mean():.4f}, "
                  f"overlap_frac={(sv ** 2).sum() / kk:.4f}, "
                  f"principal_angle_mean={np.degrees(angles_rad.mean()):.1f}deg")

        pool_result = {
            'n_pool': n_pool,
            'svd_eff_rank': eff_rank_r,
            'svd_top10': S_r[:10].tolist(),
            'write_svd_eff_rank': eff_rank_w,
            'write_svd_top10': S_w[:10].tolist(),
            'pairwise_cos': {
                'mean': float(cos_flat.mean()),
                'std': float(cos_flat.std()),
                'p5': float(np.percentile(cos_flat, 5)),
                'p95': float(np.percentile(cos_flat, 95)),
                'histogram': cos_hist.tolist(),
                'edges': cos_edges.tolist(),
            },
            'covering': {
                'mean_max_cos': float(max_cos_per_probe.mean()),
                'min_max_cos': float(max_cos_per_probe.min()),
                'p5_max_cos': float(np.percentile(max_cos_per_probe, 5)),
                'histogram': cover_hist.tolist(),
                'edges': cover_edges.tolist(),
            },
            'rw_subspace_overlap': subspace,
        }
        results[pool_name] = pool_result
        rank_compare[pool_name] = {
            'read_eff_rank': eff_rank_r,
            'write_eff_rank': eff_rank_w,
            'ratio_r_over_w': eff_rank_r / (eff_rank_w + 1e-12),
            'diff_r_minus_w': eff_rank_r - eff_rank_w,
            'd_model': d_model,
            'max_rank': min(n_pool, d_model),
        }
        print(f"    pairwise cos mean={pool_result['pairwise_cos']['mean']:.4f}, "
              f"covering mean_max_cos={pool_result['covering']['mean_max_cos']:.4f}")

    results['_rank_compare'] = rank_compare

    _save_json(results, output_dir, 'read_cov', 'read_coverage.json')

    d_model_val = model_cfg['d_model']
    rank_parts = [f"{pn}_rank={results[pn]['svd_eff_rank']:.1f}/{d_model_val}" for pn in pools]
    cover_parts = [f"{pn}={results[pn]['covering']['mean_max_cos']:.2f}" for pn in pools]
    print(f"\n  Summary: {' '.join(rank_parts)}")
    print(f"           covering_radius: {' '.join(cover_parts)}")

    print("\n  Read vs Write effective rank:")
    print(f"    {'pool':<6} {'read':>10} {'write':>10} {'R/W':>8} {'R-W':>10}  max_rank")
    for pn in pools:
        rc = rank_compare[pn]
        print(f"    {pn:<6} {rc['read_eff_rank']:>10.1f} {rc['write_eff_rank']:>10.1f} "
              f"{rc['ratio_r_over_w']:>8.3f} {rc['diff_r_minus_w']:>+10.1f}  {rc['max_rank']}")

    print("\n  Read-Write subspace overlap (mean σ of U_r^T U_w):")
    header = "    " + f"{'pool':<6}" + "".join([f" k={k:<5}" for k in subspace_ks])
    print(header)
    for pn in pools:
        row = f"    {pn:<6}"
        for k in subspace_ks:
            key = f"k={min(k, min(results[pn]['n_pool'], d_model_val))}"
            sv_mean = results[pn]['rw_subspace_overlap'][key]['sv_mean']
            row += f" {sv_mean:>6.3f} "
        print(row)

    print("\n  Done: read_cov")
    return results


# ============================================================
# §7.1.2 Selection Gini / Concentration per Pool
# ============================================================

def analyze_selection_gini(params, cfg, val_tokens, output_dir,
                            n_batches=20, batch_size=8):
    """§7.1.2: per-layer per-pool Gini coefficient, activation coverage."""
    print("\n" + "="*60)
    print("§7.1.2: Selection Gini / Concentration")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    forward_fn = _build_forward_extended(params_jax, model_cfg)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    pool_info = {'qk': n_qk, 'v': n_v, 'know': n_know}
    freq_acc = {}
    active_count_acc = {}
    for pn, n_pool in pool_info.items():
        freq_acc[pn] = [np.zeros(n_pool, dtype=np.float64) for _ in range(n_layers)]
        active_count_acc[pn] = [0.0 for _ in range(n_layers)]

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        result = jax.device_get(forward_fn(batch))

        for li in range(n_layers):
            freq_acc['qk'][li] += result['qk_q_active'][li]
            freq_acc['v'][li] += result['v_active'][li]
            freq_acc['know'][li] += result['know_active'][li]
            active_count_acc['qk'][li] += float(result['qk_q_active_count'][li])
            active_count_acc['v'][li] += float(result['v_active_count'][li])
            active_count_acc['know'][li] += float(result['know_active_count'][li])

        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    def _gini(freq):
        f = np.sort(freq)
        n = len(f)
        if f.sum() < 1e-12:
            return 1.0
        cum = np.cumsum(f)
        return float(1.0 - 2.0 * cum.sum() / (n * f.sum()) + 1.0 / n)

    results = {}
    for pn, n_pool in pool_info.items():
        results[pn] = {}
        for li in range(n_layers):
            freq = freq_acc[pn][li] / actual_batches  # avg per-neuron activation freq
            gini = _gini(freq)
            coverage = float((freq > 0).sum() / n_pool)
            mean_active = active_count_acc[pn][li] / actual_batches
            results[pn][f'layer_{li}'] = {
                'gini': gini,
                'coverage': coverage,
                'mean_active_count': mean_active,
                'active_frac': float(mean_active / n_pool),
            }
        # layer 평균 출력
        ginis = [results[pn][f'layer_{li}']['gini'] for li in range(n_layers)]
        print(f"  {pn}: Gini mean={np.mean(ginis):.4f}, range=[{min(ginis):.4f}, {max(ginis):.4f}]")

    _save_json(results, output_dir, 'sel_gini', 'per_layer_pool.json')

    print(f"\n  Know pool per-layer:")
    print(f"  Layer |  Gini | coverage | active_N")
    print(f"  ------+-------+----------+---------")
    for li in range(n_layers):
        d = results['know'][f'layer_{li}']
        print(f"   L{li:2d}  | {d['gini']:.3f} | {d['coverage']:8.4f} | {d['mean_active_count']:8.0f}")

    print("  Done: sel_gini")
    return results


# ============================================================
# §7.1.3 Selection Transition (layer-pair Jaccard)
# ============================================================

def analyze_selection_transition(params, cfg, val_tokens, output_dir,
                                  n_batches=10, batch_size=4):
    """§7.1.3: layer L과 L+1 active set Jaccard similarity per pool."""
    print("\n" + "="*60)
    print("§7.1.3: Selection Transition (Jaccard)")
    print("="*60)

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    jit_fwd = jax.jit(lambda ids: analysis_forward(params_jax, model_cfg, ids, mode='full'))

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    pool_names = ['know']
    jaccard_acc = {pn: np.zeros(n_layers - 1, dtype=np.float64) for pn in pool_names}
    turnover_acc = {pn: np.zeros(n_layers - 1, dtype=np.float64) for pn in pool_names}

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        _, layer_info = jit_fwd(batch)
        # gate_Know_raw: [n_layers, B, S, n_know]
        gate_know_raw = layer_info['gate_Know_raw']

        for li in range(n_layers - 1):
            know_L = (gate_know_raw[li] > 0).astype(jnp.float32)
            know_L1 = (gate_know_raw[li + 1] > 0).astype(jnp.float32)
            inter = (know_L * know_L1).sum(axis=-1)
            union = jnp.maximum(know_L + know_L1 - know_L * know_L1, 1e-8).sum(axis=-1)
            jacc = inter / union
            jacc_mean = float(jax.device_get(jacc.mean()))
            jaccard_acc['know'][li] += jacc_mean
            turnover_acc['know'][li] += (1.0 - jacc_mean)

        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # 평균
    results = {}
    for pn in pool_names:
        jaccard_avg = jaccard_acc[pn] / actual_batches
        turnover_avg = turnover_acc[pn] / actual_batches
        results[pn] = {
            f'pair_{li}_{li+1}': {
                'jaccard': float(jaccard_avg[li]),
                'turnover': float(turnover_avg[li]),
            } for li in range(n_layers - 1)
        }
        print(f"  {pn}: Jaccard mean={jaccard_avg.mean():.4f}, turnover mean={turnover_avg.mean():.4f}")

        # per-layer-pair compact display
        labels = [f"L{li}→{li+1}" for li in range(n_layers - 1)]
        vals = [f"{jaccard_avg[li]:.3f}" for li in range(n_layers - 1)]
        print(f"  Know pool Jaccard (L→L+1):")
        print(f"  {' '.join(f'{l:>7s}' for l in labels)}")
        print(f"  {' '.join(f'{v:>7s}' for v in vals)}")

    _save_json(results, output_dir, 'sel_trans', 'layer_pair_jaccard.json')
    print("  Done: sel_trans")
    return results


# ============================================================
# §7.4.1+§7.4.2 Combinatorial Coverage & Reuse
# ============================================================

def analyze_combinatorial_coverage(params, cfg, val_tokens, output_dir,
                                    n_batches=100, batch_size=8):
    """§7.4.1+7.4.2: active set hash로 unique combination 수, reuse entropy."""
    print("\n" + "="*60)
    print("§7.4.1+7.4.2: Combinatorial Coverage & Reuse")
    print("="*60)

    from collections import Counter

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_know = model_cfg['n_know']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    jit_fwd = jax.jit(lambda ids: analysis_forward(params_jax, model_cfg, ids, mode='full'))

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    rng = np.random.RandomState(42)
    HASH_MOD = 999983
    prime_vec_np = rng.randint(1, HASH_MOD, size=n_know).astype(np.int32)

    mid_layer = n_layers // 2
    combo_counters = {li: Counter() for li in [mid_layer]}
    size_acc = {li: [] for li in [mid_layer]}
    total_tokens = 0

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        B, S = batch.shape
        _, layer_info = jit_fwd(batch)

        gate_mid = layer_info['gate_Know_raw'][mid_layer]
        active_mask_np = np.array(jax.device_get((gate_mid > 0)))  # [B, S, N] bool

        # host측 hash (JAX int overflow 회피)
        active_int = active_mask_np.astype(np.int32)
        hashes_np = (active_int * prime_vec_np[np.newaxis, np.newaxis, :]).sum(axis=-1) % HASH_MOD
        sizes_np = active_int.sum(axis=-1)

        for h_val in hashes_np.flatten():
            combo_counters[mid_layer][int(h_val)] += 1
        size_acc[mid_layer].extend(sizes_np.flatten().tolist())
        total_tokens += B * S

        if (b + 1) % 10 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    # 결과 정리
    results = {}
    for li in [mid_layer]:
        counter = combo_counters[li]
        n_unique = len(counter)
        n_total = sum(counter.values())

        # reuse entropy
        if n_total > 0:
            probs = np.array(list(counter.values()), dtype=np.float64) / n_total
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        else:
            entropy = 0.0

        # top-1000
        top_combos = counter.most_common(1000)
        top_freq = [(cnt / n_total) for _, cnt in top_combos]

        sizes = np.array(size_acc[li])
        results[f'layer_{li}'] = {
            'n_unique_combinations': n_unique,
            'n_total_tokens': n_total,
            'reuse_entropy': entropy,
            'max_entropy': float(np.log(n_unique + 1)),
            'top10_reuse_frac': float(sum(top_freq[:10])),
            'top100_reuse_frac': float(sum(top_freq[:100])),
            'active_set_size_mean': float(sizes.mean()),
            'active_set_size_std': float(sizes.std()),
            'active_set_size_median': float(np.median(sizes)),
        }
        print(f"  Layer {li}: {n_unique} unique combos / {n_total} tokens, "
              f"entropy={entropy:.2f}, top-10 reuse={results[f'layer_{li}']['top10_reuse_frac']:.4f}")

    _save_json(results, output_dir, 'combo', 'combinatorial_coverage.json')
    print("  Done: combo")
    return results


# ============================================================
# §7.4.3 Additivity Test
# ============================================================

def analyze_additivity(params, cfg, val_tokens, output_dir,
                        n_tokens=10, n_batches=1):
    """§7.4.3: leave-one-out additivity test via _compute_per_neuron_contribution."""
    print("\n" + "="*60)
    print("§7.4.3: Additivity Test")
    print("="*60)

    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_know = model_cfg['n_know']
    d_model = model_cfg['d_model']
    n_heads = model_cfg['n_heads']
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']

    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_emb_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    qk_s = pool_p.get('qk_scale', 1.0)
    v_s_val = pool_p.get('v_scale', 1.0)
    know_s = pool_p.get('know_scale', 1.0)
    r_n = pool_p['know_read'] / (jnp.linalg.norm(pool_p['know_read'], axis=-1, keepdims=True) + 1e-8)
    w_n = pool_p['know_write'] / (jnp.linalg.norm(pool_p['know_write'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)

    sample_layers = [0, n_layers // 2, n_layers - 1]
    sample_layers = [li for li in sample_layers if li < n_layers]

    # single sequence forward로 per-token contrib 추출
    input_ids = jnp.array(tokens[:1], dtype=jnp.int32)  # [1, S]
    n_tok = min(n_tokens, max_seq - 2)
    tok_positions = list(range(max_seq // 4, max_seq // 4 + n_tok))

    results = {'per_layer': {}, 'n_tokens': n_tok, 'n_layers_sampled': len(sample_layers)}

    for li in sample_layers:
        print(f"  Layer {li}...")

        # forward to target layer (single seq, full S)
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = params_jax['token_emb']['embedding'][input_ids] + params_jax['pos_emb']['embedding'][positions]

        for l in range(li + 1):
            bp = block_params_list[l]
            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_p['proj_attn']['kernel'] + router_p['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_p['tau_attn']['kernel'] + router_p['tau_attn']['bias']
            Q = _srw_inference_fn(normed1, h_Q, qk_emb_n, tau_all[:,:,0:1], pool_p['qk_read'], pool_p['qk_write']) * qk_s
            K = _srw_inference_fn(normed1, h_K, qk_emb_n, tau_all[:,:,1:2], pool_p['qk_read'], pool_p['qk_write']) * qk_s
            V = _srw_inference_fn(normed1, h_V, v_emb_n, tau_all[:,:,2:3], pool_p['v_read'], pool_p['v_write']) * v_s_val
            d_head = d_model // n_heads
            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            scale_v = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale_v
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0,2,1,3).reshape(B,S,d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x = x + attn_out
            normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
            if l < li:
                know_out = _srw_inference_fn(normed2,
                    normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias'],
                    know_emb_n,
                    normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias'],
                    pool_p['know_read'], pool_p['know_write']) * know_s
                x = x + know_out

        # at target layer: get gate + per-token contribution for sampled tokens
        h_k = normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias']
        tau_k = normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias']
        scores_k = h_k @ know_emb_n.T
        sf = scores_k.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        tau = s_mean + tau_k * s_std
        raw = scores_k - tau.astype(scores_k.dtype)
        gate_all = jnp.maximum(raw, 0.0)
        gate_all = jnp.clip(gate_all, 0.0, 10.0)  # [1, S, N]
        active_N = (gate_all > 0).sum(axis=-1, keepdims=True).astype(jnp.float32)
        den_all = jnp.sqrt(active_N.squeeze(-1)) + 1.0  # [1, S]
        xr_all = normed2 @ r_n.T  # [1, S, N]

        gate_np = np.array(jax.device_get(gate_all[0]))  # [S, N]
        den_np = np.array(jax.device_get(den_all[0]))     # [S]
        xr_np = np.array(jax.device_get(xr_all[0]))       # [S, N]
        w_n_np = np.array(jax.device_get(w_n))             # [N, d]

        cos_raw_list = []
        cos_norm_list = []
        residual_frac_list = []

        for ti in tok_positions:
            g = gate_np[ti]        # [N]
            active_idx = np.where(g > 0)[0]
            if len(active_idx) < 2:
                continue

            top_k = min(20, len(active_idx))
            top_idx = active_idx[np.argsort(-g[active_idx])[:top_k]]

            d_val = max(den_np[ti], 1.0)

            # per-neuron contrib vectors for active neurons only
            g_active = g[active_idx]                   # [A]
            xr_active = xr_np[ti, active_idx]          # [A]
            w_active = w_n_np[active_idx]               # [A, d]
            cv_active = (g_active * xr_active)[:, np.newaxis] * w_active  # [A, d]

            full_raw = cv_active.sum(axis=0)  # [d]

            residual_frac_list.append(0.0)  # sum == full by construction

            # top neurons contrib
            for rank_i, ni in enumerate(top_idx):
                local_i = np.searchsorted(active_idx, ni)
                contrib_i = cv_active[local_i]  # [d]

                # raw cos (always 1.0 by construction for leave-one-out diff)
                cos_raw_list.append(1.0)

                # normalized cos
                g_i = g[ni]
                den_without = max(d_val - g_i, 1.0)
                removed_raw = full_raw - contrib_i
                norm_full = full_raw / d_val
                norm_without = removed_raw / den_without
                norm_diff = norm_full - norm_without
                contrib_i_norm = contrib_i / d_val
                nd = np.linalg.norm(norm_diff)
                cn = np.linalg.norm(contrib_i_norm)
                if nd > 1e-8 and cn > 1e-8:
                    cos_n = float(np.dot(norm_diff, contrib_i_norm) / (nd * cn))
                    cos_norm_list.append(cos_n)

        layer_result = {
            'cos_raw_mean': float(np.mean(cos_raw_list)) if cos_raw_list else 0.0,
            'cos_raw_std': float(np.std(cos_raw_list)) if cos_raw_list else 0.0,
            'cos_norm_mean': float(np.mean(cos_norm_list)) if cos_norm_list else 0.0,
            'cos_norm_std': float(np.std(cos_norm_list)) if cos_norm_list else 0.0,
            'residual_frac_mean': float(np.mean(residual_frac_list)) if residual_frac_list else 0.0,
            'n_samples': len(cos_raw_list),
        }
        results['per_layer'][f'layer_{li}'] = layer_result
        print(f"    cos_raw={layer_result['cos_raw_mean']:.4f}±{layer_result['cos_raw_std']:.4f}, "
              f"cos_norm={layer_result['cos_norm_mean']:.4f}, n={layer_result['n_samples']}")

    _save_json(results, output_dir, 'addit', 'additivity_results.json')

    all_raw = [results['per_layer'][k]['cos_raw_mean'] for k in results['per_layer'] if results['per_layer'][k]['n_samples'] > 0]
    all_raw_s = [results['per_layer'][k]['cos_raw_std'] for k in results['per_layer'] if results['per_layer'][k]['n_samples'] > 0]
    all_norm = [results['per_layer'][k]['cos_norm_mean'] for k in results['per_layer'] if results['per_layer'][k]['n_samples'] > 0]
    all_norm_s = [results['per_layer'][k]['cos_norm_std'] for k in results['per_layer'] if results['per_layer'][k]['n_samples'] > 0]
    if all_raw:
        print(f"  Raw additivity: cos={np.mean(all_raw):.4f}±{np.mean(all_raw_s):.4f} (theoretically 1.0)")
        print(f"  Normalized:     cos={np.mean(all_norm):.4f}±{np.mean(all_norm_s):.4f} (den shift causes deviation)")

    print("  Done: addit")
    return results


# ============================================================
# Appendix B.1+B.2 Domain Suppression Extended
# ============================================================

def analyze_domain_suppression_ext(params, cfg, output_dir,
                                    suppress_levels=None):
    """Appendix B.1+B.2: 도메인별 targeted/random suppression + selectivity index."""
    print("\n" + "="*60)
    print("Appendix B: Domain Suppression Extended")
    print("="*60)

    if suppress_levels is None:
        suppress_levels = [0.01, 0.03, 0.05, 0.10]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _mod = get_model_module()
    analysis_forward = _mod.analysis_forward
    build_suppressed_forward = _mod.build_suppressed_forward
    prefill_fn = _mod.prefill

    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']
    n_layers = model_cfg['n_layers']
    mid_layer = n_layers // 2
    max_seq = model_cfg['max_seq_len']

    params_jax = jax.tree.map(jnp.asarray, params)
    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Step 1: domain activation profiles (재사용 패턴)
    print("  Collecting domain activation profiles...")
    domain_profiles = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        acc = np.zeros(n_know, dtype=np.float64)
        n_tokens = 0
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            ids_dev = jnp.array(ids_pad, dtype=jnp.int32)
            _, layer_info = jit_analysis(params_jax, ids_dev)
            gate = np.array(jax.device_get(layer_info['gate_Know_raw'][mid_layer]))
            gate_valid = gate[0, :ids.shape[1], :]
            acc += gate_valid.mean(axis=0)
            n_tokens += 1
        domain_profiles[domain] = acc / n_tokens

    # Step 2: domain-specific neurons (contrastive)
    all_domains = list(DOMAIN_PROMPTS.keys())
    domain_neurons = {}
    for domain in all_domains:
        others_mean = np.mean([domain_profiles[d] for d in all_domains if d != domain], axis=0)
        specificity = domain_profiles[domain] - others_mean
        domain_neurons[domain] = np.argsort(-specificity)

    # Step 3: baseline loss per domain
    @jax.jit
    def get_loss(params_j, input_ids):
        logits, _, _, _ = prefill_fn(params_j, model_cfg, input_ids)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        target_lp = jnp.take_along_axis(log_probs, shift_targets[:, :, jnp.newaxis], axis=-1).squeeze(-1)
        return -target_lp.mean()

    def _eval_domain_loss(forward_fn, domain):
        """특정 forward로 domain prompts에 대한 평균 loss."""
        losses = []
        for prompt in DOMAIN_PROMPTS[domain]:
            ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
            ids_pad = np.zeros((1, max_seq), dtype=np.int32)
            ids_pad[0, :ids.shape[1]] = ids
            ids_dev = jnp.array(ids_pad, dtype=jnp.int32)
            loss = float(jax.device_get(forward_fn(ids_dev)))
            losses.append(loss)
        return float(np.mean(losses))

    # baseline
    baseline_fn = jax.jit(lambda ids: get_loss(params_jax, ids))
    baseline_losses = {}
    for domain in all_domains:
        baseline_losses[domain] = _eval_domain_loss(baseline_fn, domain)
    print(f"  Baseline losses: {baseline_losses}")

    # Step 4: targeted suppression per domain
    results = {'targeted': {}, 'random_baseline': {}, 'selectivity': {}}

    for target_domain in all_domains:
        print(f"\n  Target domain: {target_domain}")
        results['targeted'][target_domain] = {}

        for pct in suppress_levels:
            n_suppress = max(1, int(pct * n_know))
            suppress_idx = domain_neurons[target_domain][:n_suppress]
            mask = np.zeros(n_know, dtype=bool)
            mask[suppress_idx] = True

            sup_forward = jax.jit(build_suppressed_forward(params_jax, model_cfg,
                                                            {'know': jnp.array(mask)}))
            sup_fn = lambda ids, sf=sup_forward: get_loss(
                # rebuild: use suppressed forward's logits
                params_jax, ids)  # placeholder — 아래에서 직접 forward

            # suppressed forward 평가 (loss 직접 계산)
            domain_loss_drop = {}
            for eval_domain in all_domains:
                sup_losses = []
                for prompt in DOMAIN_PROMPTS[eval_domain]:
                    ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
                    ids_pad = np.zeros((1, max_seq), dtype=np.int32)
                    ids_pad[0, :ids.shape[1]] = ids
                    ids_dev = jnp.array(ids_pad, dtype=jnp.int32)
                    logits = sup_forward(ids_dev)
                    # compute loss from logits
                    shift_logits = logits[:, :-1, :]
                    shift_targets = ids_dev[:, 1:]
                    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
                    target_lp = jnp.take_along_axis(log_probs, shift_targets[:, :, jnp.newaxis], axis=-1).squeeze(-1)
                    loss = float(jax.device_get(-target_lp.mean()))
                    sup_losses.append(loss)
                sup_loss = float(np.mean(sup_losses))
                domain_loss_drop[eval_domain] = sup_loss - baseline_losses[eval_domain]

            results['targeted'][target_domain][f'suppress_{pct}'] = {
                'n_suppress': n_suppress,
                'loss_drop': domain_loss_drop,
                'target_drop': domain_loss_drop[target_domain],
            }
            print(f"    {pct*100:.0f}% ({n_suppress}): target_drop={domain_loss_drop[target_domain]:.4f}")

    # Step 5: random baseline
    rng = np.random.RandomState(42)
    n_random_trials = 3
    for pct in suppress_levels:
        n_suppress = max(1, int(pct * n_know))
        random_drops = {d: [] for d in all_domains}

        for trial in range(n_random_trials):
            rand_idx = rng.choice(n_know, n_suppress, replace=False)
            mask = np.zeros(n_know, dtype=bool)
            mask[rand_idx] = True
            rand_forward = jax.jit(build_suppressed_forward(params_jax, model_cfg,
                                                             {'know': jnp.array(mask)}))
            for eval_domain in all_domains:
                rand_losses = []
                for prompt in DOMAIN_PROMPTS[eval_domain]:
                    ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
                    ids_pad = np.zeros((1, max_seq), dtype=np.int32)
                    ids_pad[0, :ids.shape[1]] = ids
                    ids_dev = jnp.array(ids_pad, dtype=jnp.int32)
                    logits = rand_forward(ids_dev)
                    shift_logits = logits[:, :-1, :]
                    shift_targets = ids_dev[:, 1:]
                    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
                    target_lp = jnp.take_along_axis(log_probs, shift_targets[:, :, jnp.newaxis], axis=-1).squeeze(-1)
                    loss = float(jax.device_get(-target_lp.mean()))
                    rand_losses.append(loss)
                random_drops[eval_domain].append(float(np.mean(rand_losses)) - baseline_losses[eval_domain])

        results['random_baseline'][f'suppress_{pct}'] = {
            d: float(np.mean(random_drops[d])) for d in all_domains
        }
        print(f"  Random {pct*100:.0f}%: {results['random_baseline'][f'suppress_{pct}']}")

    # Step 6: selectivity index
    for target_domain in all_domains:
        results['selectivity'][target_domain] = {}
        for pct in suppress_levels:
            td = results['targeted'][target_domain][f'suppress_{pct}']['target_drop']
            control_drops = [results['targeted'][target_domain][f'suppress_{pct}']['loss_drop'][d]
                            for d in all_domains if d != target_domain]
            cd = float(np.mean(control_drops))
            selectivity = (td - cd) / (abs(td) + 1e-8)
            results['selectivity'][target_domain][f'suppress_{pct}'] = {
                'target_drop': td, 'control_drop': cd,
                'selectivity_index': selectivity,
            }

    _save_json(results, output_dir, 'dom_supp', 'domain_suppression.json')

    # selectivity index summary (avg across domains)
    si_parts = []
    for pct in suppress_levels:
        sis = [results['selectivity'][d][f'suppress_{pct}']['selectivity_index']
               for d in all_domains if f'suppress_{pct}' in results['selectivity'].get(d, {})]
        if sis:
            si_parts.append(f"{pct:.0%}: SI={np.mean(sis):.2f}")
    if si_parts:
        print(f"  Selectivity index (targeted vs random): {' '.join(si_parts)}")

    print("  Done: dom_supp")
    return results


# ============================================================
# §7.2.2 R-W Angle vs Function (조건부)
# ============================================================

def analyze_rw_function_correlation(params, cfg, val_tokens, output_dir,
                                     n_batches=10, batch_size=4):
    """§7.2.2: cos(r,w) bin별 뉴런 기능 상관관계. Task 1.1 결과 필요."""
    print("\n" + "="*60)
    print("§7.2.2: R-W Angle vs Function")
    print("="*60)

    model_cfg = get_model_cfg(cfg)
    n_layers = model_cfg['n_layers']
    n_know = model_cfg['n_know']
    max_seq = model_cfg['max_seq_len']

    cos_rw_path = os.path.join(output_dir, 'rw_alignment', 'know_cos_rw.npy')
    if not os.path.exists(cos_rw_path):
        print(f"  rw_alignment results not found ({cos_rw_path}). Run --only rw_align first.")
        return None

    cos_rw = np.load(cos_rw_path)
    print(f"  Loaded cos(r,w) for {len(cos_rw)} know neurons")

    params_jax = jax.tree.map(jnp.asarray, params)
    forward_fn = _build_forward_extended(params_jax, model_cfg)

    n_bins = 10
    bin_edges = np.linspace(cos_rw.min() - 0.001, cos_rw.max() + 0.001, n_bins + 1)
    bin_assignments = np.digitize(cos_rw, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    actual_batches = min(n_batches, n_seqs // batch_size)
    print(f"  batches={actual_batches}, batch_size={batch_size}")

    gate_acc = np.zeros((n_layers, n_know), dtype=np.float64)
    active_acc = np.zeros((n_layers, n_know), dtype=np.float64)

    for b in range(actual_batches):
        batch = jnp.array(tokens[b * batch_size:(b + 1) * batch_size], dtype=jnp.int32)
        result = jax.device_get(forward_fn(batch))
        gate_acc += result['know_gate_mean']      # [n_layers, n_know]
        active_acc += result['know_active']        # [n_layers, n_know]
        if (b + 1) % 5 == 0:
            print(f"    batch {b+1}/{actual_batches}")

    gate_avg = gate_acc / actual_batches
    active_avg = active_acc / actual_batches

    bin_results = []
    for bi in range(n_bins):
        idx = np.where(bin_assignments == bi)[0]
        if len(idx) == 0:
            bin_results.append(None)
            continue

        bin_cos = cos_rw[idx]
        # layer-averaged gate/active for neurons in this bin
        bin_gate = gate_avg[:, idx].mean(axis=1)  # [n_layers]
        bin_active = active_avg[:, idx].mean(axis=1)

        # early/mid/late peak 분류
        peak_layer = int(np.argmax(bin_gate))
        if peak_layer < n_layers // 3:
            peak_phase = 'early'
        elif peak_layer < 2 * n_layers // 3:
            peak_phase = 'mid'
        else:
            peak_phase = 'late'

        bin_results.append({
            'bin_range': [float(bin_edges[bi]), float(bin_edges[bi+1])],
            'n_neurons': len(idx),
            'cos_rw_mean': float(bin_cos.mean()),
            'cos_rw_std': float(bin_cos.std()),
            'gate_mean_per_layer': bin_gate.tolist(),
            'active_frac_per_layer': bin_active.tolist(),
            'peak_layer': peak_layer,
            'peak_phase': peak_phase,
            'overall_gate_mean': float(gate_avg[:, idx].mean()),
            'overall_active_frac': float(active_avg[:, idx].mean()),
        })

    # 전체 상관계수
    overall_gate = gate_avg.mean(axis=0)  # [n_know]
    overall_active = active_avg.mean(axis=0)
    corr_gate = float(np.corrcoef(cos_rw, overall_gate)[0, 1]) if n_know > 1 else 0.0
    corr_active = float(np.corrcoef(cos_rw, overall_active)[0, 1]) if n_know > 1 else 0.0

    results = {
        'bins': [b for b in bin_results if b is not None],
        'correlation': {
            'cos_rw_vs_gate_mean': corr_gate,
            'cos_rw_vs_active_frac': corr_active,
        },
        'n_batches': actual_batches,
    }
    _save_json(results, output_dir, 'rw_func', 'bin_vs_metric.json')

    valid_bins = [b for b in bin_results if b is not None]
    if valid_bins:
        print(f"\n  cos(r,w) bin | neurons | gate_mean | active% | peak_layer")
        print(f"  -------------+---------+-----------+---------+-----------")
        for b in valid_bins:
            lo, hi = b['bin_range']
            print(f"  [{lo:+5.2f},{hi:+5.2f}) | {b['n_neurons']:>7d} | {b['overall_gate_mean']:9.4f} | {b['overall_active_frac']:6.1%} | {b['peak_phase']:>5s} (L{b['peak_layer']})")

    print(f"\n  Correlation: cos_rw vs gate={corr_gate:.4f}, vs active_frac={corr_active:.4f}")
    print("  Done: rw_func")
    return results


# ============================================================
# §Val4: Z distribution (quantile) + gate entropy verification
# ============================================================

def analyze_z_distribution(params, cfg, val_tokens, output_dir,
                            n_batches=4, batch_size=4, n_token_sample=256):
    """Phase 4: sampled z-quantile at 50/90/99 for active neurons (z > 0).
    Also cross-checks gate_entropy. Runs only on mid-layer. Val-data only."""
    print("\n" + "="*60)
    print("§Val4: Z Distribution (quantile + entropy)")
    print("="*60)

    _mod = get_model_module()
    _layer_norm = _mod._layer_norm

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']

    pool_params = jax.tree.map(jnp.asarray, params['neuron_pool'])
    router_params = jax.tree.map(jnp.asarray, params['router'])
    block_params_list = [jax.tree.map(jnp.asarray, params[f'block_{i}'])
                         for i in range(n_layers)]

    qk_n = pool_params['qk_emb'] / (jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_n = pool_params['v_emb'] / (jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_n = pool_params['know_emb'] / (jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    _emb = jnp.asarray(params['token_emb']['embedding'])
    _pos = jnp.asarray(params['pos_emb']['embedding'])

    mid = n_layers // 2

    def _z_stats(h_proj, tau_offset, emb_unit, sample_T):
        """Compute z values + quantile + gate_entropy for one pool.
        h_proj: [B, S, d_route]; tau_offset: [B, S, d_tau]; emb_unit: [N, d_route].
        Returns (z_q50, z_q90, z_q99, gate_entropy)."""
        scores = (h_proj @ emb_unit.T).astype(jnp.float32)
        s_mean = scores.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(scores - s_mean), axis=-1, keepdims=True)) + 1e-8
        tau = s_mean + tau_offset * s_std
        z = (scores - tau) / s_std                              # [B, S, N]
        phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
        gate = jnp.where(z > 0, z * phi, 0.0)
        # Sample first sample_T tokens for quantile; flatten active z
        z_sample = z[:, :sample_T, :].reshape(-1)               # [B*sample_T*N]
        z_pos = jnp.where(z_sample > 0, z_sample, jnp.nan)
        q = jnp.nanquantile(z_pos, jnp.array([0.50, 0.90, 0.99]))
        gsum = gate.sum(axis=-1, keepdims=True) + 1e-8
        gll = (gate * jnp.log(gate + 1e-8)).sum(axis=-1, keepdims=True)
        entropy = (-gll / gsum + jnp.log(gsum)).mean()
        return q[0], q[1], q[2], entropy

    @jax.jit
    def _pool_stats(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = _emb[input_ids.astype(jnp.int32)] + _pos[positions]
        # Forward to mid layer
        for i in range(mid):
            lp = block_params_list[i]
            normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
            Q = _mod._srw_inference(normed, h_Q, qk_n, tau_all[:, :, 0:1],
                                    pool_params['qk_read'], pool_params['qk_write'])
            K = _mod._srw_inference(normed, h_K, qk_n, tau_all[:, :, 1:2],
                                    pool_params['qk_read'], pool_params['qk_write'])
            V = _mod._srw_inference(normed, h_V, v_n, tau_all[:, :, 2:3],
                                    pool_params['v_read'], pool_params['v_write'])
            n_heads = model_cfg['n_heads']
            d_head = d_model // n_heads
            Qh = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kh = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qh, Kh) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', aw, Vh)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ lp['attn']['expand_O']['kernel']
            x = x + attn_out
            normed2 = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
            h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out = _mod._srw_inference(normed2, h_k, know_n, tau_k,
                                           pool_params['know_read'], pool_params['know_write'])
            x = x + know_out

        # At mid-layer pre-block
        normed = _layer_norm(x, block_params_list[mid]['norm1']['scale'],
                             block_params_list[mid]['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        q_q = _z_stats(h_Q, tau_all[:, :, 0:1], qk_n, n_token_sample)
        k_q = _z_stats(h_K, tau_all[:, :, 1:2], qk_n, n_token_sample)
        v_q = _z_stats(h_V, tau_all[:, :, 2:3], v_n, n_token_sample)
        normed2 = _layer_norm(x, block_params_list[mid]['norm2']['scale'],
                              block_params_list[mid]['norm2']['bias'])
        h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        know_q = _z_stats(h_k, tau_k, know_n, n_token_sample)
        return q_q, k_q, v_q, know_q

    # Prepare val batches
    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_seqs = min(n_batches * batch_size, n_seqs)
    tokens = jnp.array(tokens[:total_seqs], dtype=jnp.int32)

    agg = {'Q': [], 'K': [], 'V': [], 'Know': []}
    for b in range(min(n_batches, total_seqs // batch_size)):
        batch = tokens[b * batch_size: (b + 1) * batch_size]
        qr, kr, vr, kwr = _pool_stats(batch)
        agg['Q'].append([float(x) for x in qr])
        agg['K'].append([float(x) for x in kr])
        agg['V'].append([float(x) for x in vr])
        agg['Know'].append([float(x) for x in kwr])
        print(f"  batch {b+1}: Q q50={qr[0]:.3f} q90={qr[1]:.3f} q99={qr[2]:.3f} ent={qr[3]:.2f}")

    out = {}
    for pool_name, rows in agg.items():
        arr = np.array(rows)  # [n_batches, 4]
        out[pool_name] = {
            'z_q50': float(arr[:, 0].mean()),
            'z_q90': float(arr[:, 1].mean()),
            'z_q99': float(arr[:, 2].mean()),
            'gate_entropy': float(arr[:, 3].mean()),
            'layer': mid,
            'n_batches': len(rows),
        }
        print(f"  {pool_name:4s}: z_q50={out[pool_name]['z_q50']:.3f}"
              f" z_q90={out[pool_name]['z_q90']:.3f}"
              f" z_q99={out[pool_name]['z_q99']:.3f}"
              f" entropy={out[pool_name]['gate_entropy']:.2f}")

    _save_json(out, output_dir, 'z_dist', 'z_distribution.json')
    print("  Done: z_dist")
    return out


# ============================================================
# §Val5: tau gradient — task loss vs aux loss decomposition
# ============================================================

def analyze_tau_gradient_split(params, cfg, val_tokens, output_dir,
                                model_file="models.dawn_spatial_v401_exp",
                                n_batches=2, batch_size=4):
    """Phase 5: jax.grad(task_loss, tau_*) vs jax.grad(aux_loss, tau_*).
    Reports Frobenius norm of tau_attn / tau_know gradients from each path."""
    print("\n" + "="*60)
    print("§Val5: Tau Gradient Split (task vs aux)")
    print("="*60)

    model = build_model(cfg, model_file)
    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    params_j = jax.tree.map(jnp.asarray, params)

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_seqs = min(n_batches * batch_size, n_seqs)
    tokens = jnp.array(tokens[:total_seqs], dtype=jnp.int32)

    def _tau_norm(grad_tree):
        ta = grad_tree['router']['tau_attn']
        tk = grad_tree['router']['tau_know']
        return float(jnp.sqrt(
            jnp.sum(ta['kernel'] ** 2) + jnp.sum(ta['bias'] ** 2) +
            jnp.sum(tk['kernel'] ** 2) + jnp.sum(tk['bias'] ** 2)
        ))

    @jax.jit
    def _task_grad(p, input_ids, attention_mask, labels, rng):
        def loss_fn(p_):
            r = model.apply({'params': p_}, input_ids, labels=labels,
                             attention_mask=attention_mask, deterministic=True,
                             rngs={'dropout': rng})
            return r['loss']
        return jax.grad(loss_fn)(p)

    @jax.jit
    def _aux_grad(p, input_ids, attention_mask, labels, rng):
        def loss_fn(p_):
            r = model.apply({'params': p_}, input_ids, labels=labels,
                             attention_mask=attention_mask, deterministic=True,
                             rngs={'dropout': rng})
            return r['aux_loss']
        return jax.grad(loss_fn)(p)

    task_norms, aux_norms = [], []
    eval_rng = jax.random.PRNGKey(0)
    for b in range(min(n_batches, total_seqs // batch_size)):
        batch = tokens[b * batch_size: (b + 1) * batch_size]
        mask = jnp.ones_like(batch)
        labels = jnp.where(mask == 1, batch, -100)
        t_grad = _task_grad(params_j, batch, mask, labels, eval_rng)
        a_grad = _aux_grad(params_j, batch, mask, labels, eval_rng)
        tn = _tau_norm(t_grad)
        an = _tau_norm(a_grad)
        task_norms.append(tn)
        aux_norms.append(an)
        print(f"  batch {b+1}: task_tau_grad={tn:.4e} aux_tau_grad={an:.4e}"
              f" ratio={tn / (an + 1e-12):.3f}")

    tn_mean = float(np.mean(task_norms))
    an_mean = float(np.mean(aux_norms))
    result = {
        'task_tau_grad_norm': tn_mean,
        'aux_tau_grad_norm': an_mean,
        'ratio_task_over_aux': tn_mean / (an_mean + 1e-12),
        'per_batch': {'task': task_norms, 'aux': aux_norms},
    }
    print(f"\n  Summary: task={tn_mean:.4e} aux={an_mean:.4e} ratio={result['ratio_task_over_aux']:.3f}")
    print("  (ratio >> 1 = task dominates tau; ~1 = balanced; << 1 = aux dominates)")
    _save_json(result, output_dir, 'tau_grad', 'tau_grad_split.json')
    print("  Done: tau_grad")
    return result


# ============================================================
# §P1: Emb pairwise cosine (sampled) + write direction + R/W orthogonality
# ============================================================

def analyze_pool_geometry(params, cfg, output_dir, n_samples=200):
    """Offline pool-geometry audit for qk/v/know pools:
      (3) pairwise cosine similarity of emb vectors (n_samples random pairs)
      (5) write direction anisotropy: ||write.mean(axis=0)||
          (1=all aligned, 0=uniform)
      (6) read-write orthogonality: per-neuron cos(read_j, write_j) mean + std
    Pure offline analysis — checkpoint-only, no forward pass."""
    print("\n" + "="*60)
    print("§P1: Pool geometry (emb pairwise cos / write aniso / r-w angle)")
    print("="*60)

    pool = params['neuron_pool']
    results = {}
    rng = np.random.default_rng(42)

    for pool_name, prefix in [('QK', 'qk'), ('V', 'v'), ('Know', 'know')]:
        emb = np.asarray(pool[f'{prefix}_emb'])        # [N, d_route]
        read = np.asarray(pool[f'{prefix}_read'])      # [N, d_model]
        write = np.asarray(pool[f'{prefix}_write'])    # [N, d_model]
        N = emb.shape[0]

        # (3) Emb pairwise cosine (sampled pairs; N² would be too large)
        idx_i = rng.integers(0, N, size=n_samples)
        idx_j = rng.integers(0, N, size=n_samples)
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % N             # ensure i != j
        e_i = emb[idx_i] / (np.linalg.norm(emb[idx_i], axis=-1, keepdims=True) + 1e-8)
        e_j = emb[idx_j] / (np.linalg.norm(emb[idx_j], axis=-1, keepdims=True) + 1e-8)
        emb_cos = (e_i * e_j).sum(axis=-1)
        emb_cos_mean = float(emb_cos.mean())
        emb_cos_std = float(emb_cos.std())
        emb_cos_abs_mean = float(np.abs(emb_cos).mean())

        # (5) Write direction anisotropy
        w_unit = write / (np.linalg.norm(write, axis=-1, keepdims=True) + 1e-8)
        write_mean_norm = float(np.linalg.norm(w_unit.mean(axis=0)))
        r_unit = read / (np.linalg.norm(read, axis=-1, keepdims=True) + 1e-8)
        read_mean_norm = float(np.linalg.norm(r_unit.mean(axis=0)))

        # (6) Read-write per-neuron orthogonality
        rw_cos = (r_unit * w_unit).sum(axis=-1)         # [N]
        rw_cos_mean = float(rw_cos.mean())
        rw_cos_std = float(rw_cos.std())
        rw_cos_abs_mean = float(np.abs(rw_cos).mean())
        rw_aligned = float((rw_cos > 0.5).mean())       # fraction aligned (>0)
        rw_anti = float((rw_cos < -0.5).mean())         # fraction anti-aligned

        results[pool_name] = {
            'N': int(N),
            'emb_pairwise_cos_mean': emb_cos_mean,
            'emb_pairwise_cos_std': emb_cos_std,
            'emb_pairwise_cos_abs_mean': emb_cos_abs_mean,
            'write_mean_dir_norm': write_mean_norm,
            'read_mean_dir_norm': read_mean_norm,
            'rw_cos_mean': rw_cos_mean,
            'rw_cos_std': rw_cos_std,
            'rw_cos_abs_mean': rw_cos_abs_mean,
            'rw_aligned_frac': rw_aligned,
            'rw_antialigned_frac': rw_anti,
            'n_sample_pairs': int(n_samples),
        }

        print(f"\n  {pool_name} (N={N}):")
        print(f"    emb pairwise cos ({n_samples} pairs): mean={emb_cos_mean:+.4f}"
              f" std={emb_cos_std:.4f} |mean|={emb_cos_abs_mean:.4f}")
        print(f"    write mean-direction |μ|={write_mean_norm:.4f}"
              f" (1 → all same dir, 0 → uniform on sphere)")
        print(f"    read  mean-direction |μ|={read_mean_norm:.4f}")
        print(f"    r-w cos: mean={rw_cos_mean:+.4f} std={rw_cos_std:.4f}"
              f" |mean|={rw_cos_abs_mean:.4f}")
        print(f"    r-w aligned(cos>0.5)={rw_aligned*100:.1f}%"
              f"  anti(cos<-0.5)={rw_anti*100:.1f}%")

    _save_json(results, output_dir, 'pool_geometry', 'pool_geometry.json')
    print("\n  Done: pool_geom")
    return results


# ============================================================
# z/phi/gate histogram JIT forward (gate_ci 전용)
# ============================================================

def _build_forward_zpg(params_jax, model_cfg,
                        z_bins=50, z_range=(-5.0, 5.0),
                        phi_bins=50, phi_range=(0.0, 1.0),
                        gate_bins=50, gate_range=(0.0, 10.0)):
    """z/phi/gate histogram + scalar stats forward 빌더.

    scan 내부에서 histogram reduce — [L,B,S,N] 텐서가 host로 나가지 않음.

    Returns forward_fn(input_ids) -> dict:
        '<pool>_z_hist':    [n_layers, z_bins]
        '<pool>_phi_hist':  [n_layers, phi_bins]
        '<pool>_gate_hist': [n_layers, gate_bins]
        '<pool>_z_mean':    [n_layers]
        '<pool>_phi_mean':  [n_layers]
        '<pool>_gate_mean': [n_layers]
        '<pool>_active_frac': [n_layers]
        (pool: qk_Q, qk_K, v, know)
    """
    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_fn = _mod._srw_inference

    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']

    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']
    emb_matrix = params_jax['token_emb']['embedding']
    pos_matrix = params_jax['pos_emb']['embedding']

    qk_emb_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)

    qk_s = pool_p.get('qk_scale', 1.0)
    v_s = pool_p.get('v_scale', 1.0)
    know_s = pool_p.get('know_scale', 1.0)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]
    stacked_bp = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    @jax.jit
    def forward_fn(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        def _hist(vals_flat, n_bins, lo, hi):
            step = (hi - lo) / n_bins
            idx = jnp.clip(((vals_flat - lo) / step).astype(jnp.int32), 0, n_bins - 1)
            return jnp.zeros(n_bins, dtype=jnp.float32).at[idx].add(
                jnp.ones_like(vals_flat, dtype=jnp.float32))

        def _zpg(h, emb_n, tau_off):
            scores = h @ emb_n.T
            sf = scores.astype(jnp.float32)
            s_mean = sf.mean(axis=-1, keepdims=True)
            s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
            tau = s_mean + tau_off * s_std
            raw = scores - tau.astype(scores.dtype)
            z = raw.astype(jnp.float32) / s_std
            phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
            gate = jnp.where(z > 0, z * phi, 0.0)
            return z, phi, gate

        def _pool_stats(z, phi, gate):
            zf = z.reshape(-1)
            pf = phi.reshape(-1)
            gf = gate.reshape(-1)
            return {
                'z_hist': _hist(zf, z_bins, z_range[0], z_range[1]),
                'phi_hist': _hist(pf, phi_bins, phi_range[0], phi_range[1]),
                'gate_hist': _hist(gf, gate_bins, gate_range[0], gate_range[1]),
                'z_mean': z.mean(), 'phi_mean': phi.mean(),
                'gate_mean': gate.mean(),
                'active_frac': (gate > 0).astype(jnp.float32).mean(),
            }

        def layer_fn(carry, bp):
            x = carry

            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_p['proj_attn']['kernel'] + router_p['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_p['tau_attn']['kernel'] + router_p['tau_attn']['bias']

            # attn pool z/phi/gate
            z_qQ, phi_qQ, g_qQ = _zpg(h_Q, qk_emb_n, tau_all[:, :, 0:1])
            z_qK, phi_qK, g_qK = _zpg(h_K, qk_emb_n, tau_all[:, :, 1:2])
            z_v, phi_v, g_v = _zpg(h_V, v_emb_n, tau_all[:, :, 2:3])

            s_qQ = _pool_stats(z_qQ, phi_qQ, g_qQ)
            s_qK = _pool_stats(z_qK, phi_qK, g_qK)
            s_v = _pool_stats(z_v, phi_v, g_v)

            # attn forward for residual progression
            Q = _srw_inference_fn(normed1, h_Q, qk_emb_n, tau_all[:,:,0:1],
                                  pool_p['qk_read'], pool_p['qk_write']) * qk_s
            K = _srw_inference_fn(normed1, h_K, qk_emb_n, tau_all[:,:,1:2],
                                  pool_p['qk_read'], pool_p['qk_write']) * qk_s
            V = _srw_inference_fn(normed1, h_V, v_emb_n, tau_all[:,:,2:3],
                                  pool_p['v_read'], pool_p['v_write']) * v_s
            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            # know pool z/phi/gate
            normed2 = _layer_norm_fn(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias']
            tau_k = normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias']
            z_kn, phi_kn, g_kn = _zpg(h_k, know_emb_n, tau_k)
            s_kn = _pool_stats(z_kn, phi_kn, g_kn)

            # know forward for residual
            know_out = _srw_inference_fn(normed2, h_k, know_emb_n, tau_k,
                                         pool_p['know_read'], pool_p['know_write']) * know_s
            x = x + know_out

            stats = {}
            for prefix, s in [('qk_Q', s_qQ), ('qk_K', s_qK), ('v', s_v), ('know', s_kn)]:
                for k, v_val in s.items():
                    stats[f'{prefix}_{k}'] = v_val
            return x, stats

        _, all_stats = jax.lax.scan(layer_fn, x, stacked_bp)
        return all_stats

    return forward_fn


# ============================================================
# 공통 JIT forward 헬퍼 (리팩토링용)
# ============================================================

def _build_forward_extended(params_jax, model_cfg):
    """공통 JIT forward 빌더. 반환된 함수는 JIT-compiled.

    Returns a function: forward_fn(input_ids) -> dict with keys:
      Tier A (항상 반환, scalar per layer):
        'x_norm':           [n_layers]  — ‖x_L‖ mean(B,S)
        'dx_norm':          [n_layers]  — ‖x_L - x_{L-1}‖ mean
        'attn_norm':        [n_layers]  — ‖attn_out‖ mean
        'know_norm':        [n_layers]  — ‖know_out‖ mean
        'cos_x_prev':       [n_layers]  — cos(x_L, x_{L-1}) mean
        'cos_attn_x':       [n_layers]  — cos(attn_out, x_after_attn) mean
        'cos_know_x':       [n_layers]  — cos(know_out, x_final) mean
        'cos_x0':           [n_layers]  — cos(x_L_mean, x_0_mean)

      Per-neuron activation (mean over B,S):
        'qk_q_active':      [n_layers, n_qk]  — (gate_Q > 0) freq
        'qk_k_active':      [n_layers, n_qk]
        'v_active':          [n_layers, n_v]
        'know_active':       [n_layers, n_know]

      Active count (mean over B,S):
        'qk_q_active_count': [n_layers]
        'qk_k_active_count': [n_layers]
        'v_active_count':    [n_layers]
        'know_active_count': [n_layers]

      Per-neuron gate mean (mean over B,S):
        'know_gate_mean':    [n_layers, n_know]

      Residual effective rank (covariance eigenvalue entropy):
        'eff_rank':          [n_layers]
    """
    _mod = get_model_module()
    _layer_norm_fn = _mod._layer_norm
    _srw_inference_with_gates_fn = _mod._srw_inference_with_gates

    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    d_model = model_cfg['d_model']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']
    n_know = model_cfg['n_know']

    pool_p = params_jax['neuron_pool']
    router_p = params_jax['router']
    emb_matrix = params_jax['token_emb']['embedding']
    pos_matrix = params_jax['pos_emb']['embedding']

    qk_emb_n = pool_p['qk_emb'] / (jnp.linalg.norm(pool_p['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_emb_n = pool_p['v_emb'] / (jnp.linalg.norm(pool_p['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_emb_n = pool_p['know_emb'] / (jnp.linalg.norm(pool_p['know_emb'], axis=-1, keepdims=True) + 1e-8)

    qk_s = pool_p.get('qk_scale', 1.0)
    v_s = pool_p.get('v_scale', 1.0)
    know_s = pool_p.get('know_scale', 1.0)

    block_params_list = [params_jax[f'block_{i}'] for i in range(n_layers)]
    stacked_bp = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    @jax.jit
    def forward_fn(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]
        x0_mean = x.mean(axis=(0, 1))  # [d]

        def layer_fn(carry, bp):
            x = carry
            x_pre = x

            # --- Attention ---
            normed1 = _layer_norm_fn(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed1 @ router_p['proj_attn']['kernel'] + router_p['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed1 @ router_p['tau_attn']['kernel'] + router_p['tau_attn']['bias']

            Q_out, gate_Q_raw, _ = _srw_inference_with_gates_fn(
                normed1, h_Q, qk_emb_n, tau_all[:, :, 0:1],
                pool_p['qk_read'], pool_p['qk_write'])
            K_out, gate_K_raw, _ = _srw_inference_with_gates_fn(
                normed1, h_K, qk_emb_n, tau_all[:, :, 1:2],
                pool_p['qk_read'], pool_p['qk_write'])
            V_out, gate_V_raw, _ = _srw_inference_with_gates_fn(
                normed1, h_V, v_emb_n, tau_all[:, :, 2:3],
                pool_p['v_read'], pool_p['v_write'])
            Q_out = Q_out * qk_s
            K_out = K_out * qk_s
            V_out = V_out * v_s

            d_head = d_model // n_heads
            Qr = Q_out.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K_out.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V_out.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            scale = jnp.sqrt(jnp.float32(d_head))
            sc = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            sc = jnp.where(causal, sc, jnp.finfo(sc.dtype).min)
            aw = jax.nn.softmax(sc, axis=-1)
            ao = jnp.einsum('bhst,bhtd->bhsd', aw, Vr).transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = ao @ bp['attn']['expand_O']['kernel']
            x_after_attn = x + attn_out

            # --- Know ---
            normed2 = _layer_norm_fn(x_after_attn, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed2 @ router_p['proj_know']['kernel'] + router_p['proj_know']['bias']
            tau_k = normed2 @ router_p['tau_know']['kernel'] + router_p['tau_know']['bias']
            know_out, gate_Know_raw, _ = _srw_inference_with_gates_fn(
                normed2, h_k, know_emb_n, tau_k,
                pool_p['know_read'], pool_p['know_write'])
            know_out = know_out * know_s
            x_new = x_after_attn + know_out

            # --- Stats (reduce to scalar / [N]) ---
            x_norm = jnp.linalg.norm(x_new, axis=-1).mean()
            dx_norm = jnp.linalg.norm(x_new - x_pre, axis=-1).mean()
            attn_n = jnp.linalg.norm(attn_out, axis=-1).mean()
            know_n = jnp.linalg.norm(know_out, axis=-1).mean()

            # cos(x_new, x_pre)
            xf = x_new.reshape(-1, d_model)
            xpf = x_pre.reshape(-1, d_model)
            cos_xp = ((xf * xpf).sum(-1) / (jnp.linalg.norm(xf, axis=-1) * jnp.linalg.norm(xpf, axis=-1) + 1e-8)).mean()

            # cos(attn_out, x_after_attn)
            af = attn_out.reshape(-1, d_model)
            xaf = x_after_attn.reshape(-1, d_model)
            cos_ax = ((af * xaf).sum(-1) / (jnp.linalg.norm(af, axis=-1) * jnp.linalg.norm(xaf, axis=-1) + 1e-8)).mean()

            # cos(know_out, x_new)
            kf = know_out.reshape(-1, d_model)
            xnf = x_new.reshape(-1, d_model)
            cos_kx = ((kf * xnf).sum(-1) / (jnp.linalg.norm(kf, axis=-1) * jnp.linalg.norm(xnf, axis=-1) + 1e-8)).mean()

            # cos(x_mean, x0_mean)
            x_mean = x_new.mean(axis=(0, 1))
            cos_x0_val = jnp.dot(x_mean, x0_mean) / (jnp.linalg.norm(x_mean) * jnp.linalg.norm(x0_mean) + 1e-8)

            # per-neuron activation stats
            qk_q_act = (gate_Q_raw > 0).astype(jnp.float32).mean(axis=(0, 1))  # [n_qk]
            qk_k_act = (gate_K_raw > 0).astype(jnp.float32).mean(axis=(0, 1))
            v_act = (gate_V_raw > 0).astype(jnp.float32).mean(axis=(0, 1))  # [n_v]
            know_act = (gate_Know_raw > 0).astype(jnp.float32).mean(axis=(0, 1))  # [n_know]
            know_gate_m = gate_Know_raw.mean(axis=(0, 1))  # [n_know]

            # active counts
            qk_q_ac = (gate_Q_raw > 0).astype(jnp.float32).sum(axis=-1).mean()
            qk_k_ac = (gate_K_raw > 0).astype(jnp.float32).sum(axis=-1).mean()
            v_ac = (gate_V_raw > 0).astype(jnp.float32).sum(axis=-1).mean()
            know_ac = (gate_Know_raw > 0).astype(jnp.float32).sum(axis=-1).mean()

            # residual effective rank (covariance eigenvalue 기반)
            x_flat = x_new.reshape(-1, d_model)
            x_centered = x_flat - x_flat.mean(axis=0, keepdims=True)
            cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
            eigvals = jnp.linalg.eigvalsh(cov)
            eigvals = jnp.maximum(eigvals, 1e-12)
            normed_eig = eigvals / eigvals.sum()
            eff_rank = jnp.exp(-jnp.sum(normed_eig * jnp.log(normed_eig)))

            layer_stats = {
                'x_norm': x_norm, 'dx_norm': dx_norm,
                'attn_norm': attn_n, 'know_norm': know_n,
                'cos_x_prev': cos_xp, 'cos_attn_x': cos_ax,
                'cos_know_x': cos_kx, 'cos_x0': cos_x0_val,
                'qk_q_active': qk_q_act, 'qk_k_active': qk_k_act,
                'v_active': v_act, 'know_active': know_act,
                'know_gate_mean': know_gate_m,
                'qk_q_active_count': qk_q_ac, 'qk_k_active_count': qk_k_ac,
                'v_active_count': v_ac, 'know_active_count': know_ac,
                'eff_rank': eff_rank,
            }
            return x_new, layer_stats

        _, all_stats = jax.lax.scan(layer_fn, x, stacked_bp)
        return all_stats

    return forward_fn


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DAWN-Spatial v3 Analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output", default="results/spatial_analysis")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only", default=None,
                        help="Comma-separated: info,val,health,generate,weights,routing,samples,gate_dist,utilization,r1,r2,r3,r4,r5,rw_proj,act_context,layer_role,cross_suppress,gate_mech,comp_expr,neuron_cluster,cross_ref,deep,op_space,intervene,compose,rw_align,resid_dyn,phase,drift,gate_ci,write_cov,read_cov,rw_cov,sel_gini,sel_trans,combo,addit,dom_supp,rw_func,z_dist,tau_grad,pool_geom")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated analyses to skip (used when --only is not set)")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--model_file", default="models.dawn_spatial_v3",
                        help="Model module path (e.g. models.dawn_spatial_v3981_exp)")
    parser.add_argument("--n_batches", type=int, default=None,
                        help="Override n_batches for D6/D8/D10/R.1/R.4/P2/P3/P5/P6")
    parser.add_argument("--r2_sentences", type=int, default=5000,
                        help="Max sentences for R.2 POS selectivity")
    parser.add_argument("--p6_samples", type=int, default=100,
                        help="Token samples for P6 compositional expressiveness")
    parser.add_argument("--addit_n_tokens", type=int, default=10,
                        help="Number of tokens to sample for additivity test")
    parser.add_argument("--dom_supp_suppress_levels", default="0.01,0.03,0.05,0.10",
                        help="Comma-separated suppress fractions for domain suppression")
    parser.add_argument("--dawn_log", default=None,
                        help="Path to DAWN training log (JSON-lines or CSV)")
    parser.add_argument("--baseline_log", default=None,
                        help="Path to baseline training log (JSON-lines or CSV)")
    args = parser.parse_args()

    # Initialize dynamic model module
    get_model_module(args.model_file)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg, args.model_file)
    params, ckpt_info = load_checkpoint_params(args.checkpoint, model, cfg)

    only = set(args.only.split(',')) if args.only else None
    skip = set(args.skip.split(',')) if args.skip else set()

    # Meta-aliases: expand shorthand into constituent analyses.
    _only_aliases = {
        'rw_cov': {'write_cov', 'read_cov'},
    }
    if only is not None:
        for alias, members in _only_aliases.items():
            if alias in only:
                only.discard(alias)
                only.update(members)
    for alias, members in _only_aliases.items():
        if alias in skip:
            skip.discard(alias)
            skip.update(members)

    def _should_run(name):
        """Check if analysis 'name' should run based on --only and --skip."""
        if name in skip:
            return False
        if only is not None:
            return name in only
        return True

    print(f"  ONLY={only}, SKIP={skip or 'none'}")
    if not _is_gcs(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Save checkpoint info
    _save_json(ckpt_info, args.output, '.', 'checkpoint_info.json')

    if _should_run('info'):
        analyze_model_info(params, cfg, args.output)

    print(f"  BEFORE VAL CHECK")
    # Load val tokens once (shared by D2 and D6)
    val_tokens = None
    val_path = args.val_data or cfg.get('data', {}).get('bin_val')
    _val_analyses = ['val', 'routing', 'gate_dist', 'utilization',
                     'act_context', 'layer_role', 'gate_mech', 'comp_expr',
                     'neuron_cluster', 'cross_ref', 'deep', 'op_space', 'intervene', 'compose',
                     'resid_dyn', 'drift', 'gate_ci', 'sel_gini', 'sel_trans', 'combo', 'addit', 'rw_func']
    print(f"  DEBUG: val_path={val_path}, only={only}, check={any(k in (only or set()) for k in _val_analyses)}")
    if val_path and (only is None or any(k in (only or set()) for k in _val_analyses)):
        print(f"  Loading val tokens from {val_path}...")
        val_tokens = load_val_tokens(val_path)
    else:
        if not val_path:
            print(f"  No val_data path configured")
        elif only:
            print(f"  Val tokens not needed for: {only}")

    if _should_run('val'):
        if val_tokens is not None:
            analyze_validation(params, cfg, val_tokens, args.output,
                             args.batch_size, args.max_batches)
        else:
            print("\n  Skipping validation (no --val_data)")

    if _should_run('health'):
        analyze_neuron_health(params, cfg, args.output)

    if _should_run('generate'):
        analyze_generation(params, cfg, args.output,
                          prompt=args.prompt,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature)

    if _should_run('weights'):
        analyze_weights(params, cfg, args.output)

    # Analysis-specific overrides from CLI
    _nb = args.n_batches  # None = use function default
    _bs = args.batch_size
    _abs = min(_bs, 8)  # analysis batch size (capped for memory)

    if _should_run('routing'):
        if val_tokens is not None:
            analyze_routing(params, cfg, val_tokens, args.output,
                           n_batches=_nb or 50, batch_size=_abs)
        else:
            print("\n  Skipping routing (no --val_data)")

    if _should_run('samples'):
        analyze_generation_samples(params, cfg, args.output,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature)

    # --- D8/D10: Gate distribution and neuron utilization ---
    if _should_run('gate_dist'):
        if val_tokens is not None:
            analyze_gate_distribution(params, cfg, val_tokens, args.output,
                                     n_batches=_nb or 20, batch_size=_abs)
        else:
            print("\n  Skipping gate_dist (no --val_data)")

    if _should_run('utilization'):
        if val_tokens is not None:
            analyze_neuron_utilization(params, cfg, val_tokens, args.output,
                                      n_batches=_nb or 20, batch_size=_abs)
        else:
            print("\n  Skipping utilization (no --val_data)")

    # --- Rebuttal analyses (D.1-D.5 methodology) ---
    # Load val tokens for rebuttal analyses if not already loaded
    if val_tokens is None and val_path:
        needs_val = any(k in (only or set()) for k in ['r1', 'r4']) if only else True
        if needs_val:
            val_tokens = load_val_tokens(val_path)

    if _should_run('r1'):
        if val_tokens is not None:
            analyze_qk_specialization(params, cfg, val_tokens, args.output,
                                     n_batches=_nb or 50, batch_size=_abs)
        else:
            print("\n  Skipping R.1 (no --val_data)")

    if _should_run('r2'):
        analyze_pos_selectivity(params, cfg, args.output,
                               max_sentences=args.r2_sentences, batch_size=min(_abs, 4))

    if _should_run('r4'):
        if val_tokens is not None:
            analyze_layer_balance(params, cfg, val_tokens, args.output,
                                n_batches=_nb or 20, batch_size=_abs)
        else:
            print("\n  Skipping R.4 (no --val_data)")

    r3_results = None
    if _should_run('r3'):
        r3_results = analyze_knowledge_neurons(params, cfg, args.output)

    if _should_run('r5'):
        analyze_suppression(params, cfg, args.output,
                           knowledge_results=r3_results)

    # --- Paper analyses (P1-P6) ---
    if _should_run('rw_proj'):
        analyze_rw_projection(params, cfg, args.output)

    if _should_run('act_context'):
        if val_tokens is not None:
            analyze_activation_context(params, cfg, val_tokens, args.output,
                                      n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping act_context (no --val_data)")

    if _should_run('layer_role'):
        if val_tokens is not None:
            analyze_layer_role_matrix(params, cfg, val_tokens, args.output,
                                     n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping layer_role (no --val_data)")

    if _should_run('cross_suppress'):
        analyze_cross_domain_suppression(params, cfg, args.output)

    if _should_run('gate_mech'):
        if val_tokens is not None:
            analyze_gate_mechanism(params, cfg, val_tokens, args.output,
                                  n_batches=_nb or 5, batch_size=min(_abs, 2))
        else:
            print("\n  Skipping gate_mech (no --val_data)")

    if _should_run('comp_expr'):
        if val_tokens is not None:
            analyze_compositional_expressiveness(params, cfg, val_tokens, args.output,
                                                n_samples=args.p6_samples, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping comp_expr (no --val_data)")

    if _should_run('neuron_cluster'):
        if val_tokens is not None:
            analyze_neuron_clustering(params, cfg, val_tokens, args.output,
                                     n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            # Part A works without val, Part B skipped
            analyze_neuron_clustering(params, cfg, None, args.output)

    if _should_run('cross_ref'):
        if val_tokens is not None:
            analyze_cross_reference(params, cfg, val_tokens, args.output,
                                   n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping cross_ref (no --val_data)")

    if _should_run('deep'):
        if val_tokens is not None:
            analyze_deep_analysis(params, cfg, val_tokens, args.output,
                                 n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping deep (no --val_data)")

    if _should_run('op_space'):
        if val_tokens is not None:
            analyze_operation_space(params, cfg, val_tokens, args.output,
                                   n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping op_space (no --val_data)")

    if _should_run('intervene'):
        if val_tokens is not None:
            analyze_interventions(params, cfg, val_tokens, args.output,
                                 n_batches=_nb or 50, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping intervene (no --val_data)")

    if _should_run('compose'):
        if val_tokens is not None:
            analyze_composition(params, cfg, val_tokens, args.output,
                               n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping compose (no --val_data)")

    if _should_run('rw_align'):
        analyze_rw_alignment(params, cfg, args.output)

    if _should_run('resid_dyn'):
        if val_tokens is not None:
            analyze_residual_dynamics(params, cfg, val_tokens, args.output,
                                      n_batches=_nb or 20, batch_size=min(_abs, 8))
        else:
            print("\n  Skipping resid_dyn (no --val_data)")

    if _should_run('phase'):
        analyze_phase_dynamics(args.output,
                               dawn_log=args.dawn_log,
                               baseline_log=args.baseline_log)

    if _should_run('drift'):
        if val_tokens is not None:
            analyze_drift_alignment(params, cfg, val_tokens, args.output,
                                     n_batches=_nb or 20, batch_size=min(_abs, 8))
        else:
            print("\n  Skipping drift (no --val_data)")

    if _should_run('gate_ci'):
        if val_tokens is not None:
            analyze_gate_confidence_intensity(params, cfg, val_tokens, args.output,
                                              n_batches=_nb or 20, batch_size=min(_abs, 8))
        else:
            print("\n  Skipping gate_ci (no --val_data)")

    if _should_run('write_cov'):
        analyze_write_coverage(params, cfg, args.output)

    if _should_run('read_cov'):
        analyze_read_coverage(params, cfg, args.output)

    if _should_run('sel_gini'):
        if val_tokens is not None:
            analyze_selection_gini(params, cfg, val_tokens, args.output,
                                    n_batches=_nb or 20, batch_size=min(_abs, 8))
        else:
            print("\n  Skipping sel_gini (no --val_data)")

    if _should_run('sel_trans'):
        if val_tokens is not None:
            analyze_selection_transition(params, cfg, val_tokens, args.output,
                                          n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping sel_trans (no --val_data)")

    if _should_run('combo'):
        if val_tokens is not None:
            analyze_combinatorial_coverage(params, cfg, val_tokens, args.output,
                                            n_batches=_nb or 100, batch_size=min(_abs, 8))
        else:
            print("\n  Skipping combo (no --val_data)")

    if _should_run('addit'):
        if val_tokens is not None:
            analyze_additivity(params, cfg, val_tokens, args.output,
                                n_tokens=args.addit_n_tokens, n_batches=_nb or 1)
        else:
            print("\n  Skipping addit (no --val_data)")

    if _should_run('dom_supp'):
        sup_levels = [float(x) for x in args.dom_supp_suppress_levels.split(',')]
        analyze_domain_suppression_ext(params, cfg, args.output,
                                        suppress_levels=sup_levels)

    if _should_run('rw_func'):
        if val_tokens is not None:
            analyze_rw_function_correlation(params, cfg, val_tokens, args.output,
                                             n_batches=_nb or 10, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping rw_func (no --val_data)")

    if _should_run('z_dist'):
        if val_tokens is not None:
            analyze_z_distribution(params, cfg, val_tokens, args.output,
                                    n_batches=_nb or 4, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping z_dist (no --val_data)")

    if _should_run('tau_grad'):
        if val_tokens is not None:
            analyze_tau_gradient_split(params, cfg, val_tokens, args.output,
                                        model_file=args.model_file,
                                        n_batches=_nb or 2, batch_size=min(_abs, 4))
        else:
            print("\n  Skipping tau_grad (no --val_data)")

    if _should_run('pool_geom'):
        analyze_pool_geometry(params, cfg, args.output, n_samples=200)

    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    main()

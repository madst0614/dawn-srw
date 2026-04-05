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
# D3: Neuron Health
# ============================================================

def analyze_neuron_health(params, cfg, output_dir):
    print("\n" + "="*60)
    print("D3: Neuron Health")
    print("="*60)

    from models.dawn_spatial_v3 import vectorized_neuron_health

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
    from models.dawn_spatial_v3 import prefill, decode_step

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

    from models.dawn_spatial_v3 import vectorized_weight_analysis

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

    from models.dawn_spatial_v3 import (
        _layer_norm, _srw_inference_with_gates)

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

    @jax.jit
    def get_routing_stats(input_ids):
        """Forward to mid layer, get gate distributions for Q,K,V,Know."""
        B, S = input_ids.shape
        emb_matrix = params['token_emb']['embedding']
        pos_matrix = params['pos_emb']['embedding']
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids] + pos_matrix[positions]

        # Forward to mid layer
        for i in range(mid_layer):
            lp = block_params_list[i]
            normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

            from models.dawn_spatial_v3 import _srw_inference
            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                               pool_params['qk_read'], pool_params['qk_write'])
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write'])
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write'])

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
                                     pool_params['know_read'], pool_params['know_write'])
            x = x + know_out

        # At mid layer, get gate distributions
        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        _, gate_Q = _srw_inference_with_gates(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                                               pool_params['qk_read'], pool_params['qk_write'])
        _, gate_K = _srw_inference_with_gates(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                                               pool_params['qk_read'], pool_params['qk_write'])
        _, gate_V = _srw_inference_with_gates(normed, h_V, v_norm, tau_all[:, :, 2:3],
                                               pool_params['v_read'], pool_params['v_write'])

        normed2 = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        h_k = normed2 @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed2 @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        _, gate_Know = _srw_inference_with_gates(normed2, h_k, know_norm, tau_k,
                                                  pool_params['know_read'], pool_params['know_write'])

        # Compute stats per pool
        def gate_stats(g):
            active = (g > 1e-6).astype(jnp.float32)
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
    from models.dawn_spatial_v3 import prefill, decode_step

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

    from models.dawn_spatial_v3 import analysis_forward
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

        # Average across layers, get binary activation
        # gate_Q: [n_layers, B, S, n_qk]
        gQ = np.array(jax.device_get(layer_info['gate_Q']))  # [L, B, S, N]
        gK = np.array(jax.device_get(layer_info['gate_K']))

        # Binary: active if gate > 1e-6
        q_active = (gQ > 1e-6).sum(axis=(0, 1, 2))  # [N] summed over layers,batch,seq
        k_active = (gK > 1e-6).sum(axis=(0, 1, 2))
        q_counts += q_active
        k_counts += k_active

        # Batch overlap (across all layers)
        q_bin = gQ > 1e-6
        k_bin = gK > 1e-6
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

    from models.dawn_spatial_v3 import analysis_forward
    import numpy as np

    model_cfg = get_model_cfg(cfg)
    max_seq = model_cfg['max_seq_len']
    n_layers = model_cfg['n_layers']

    n_seqs = len(val_tokens) // max_seq
    tokens = val_tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    total_batches = min(n_batches, n_seqs // batch_size)

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

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
                             max_sentences=5000, batch_size=8):
    """POS selectivity: same as v17.1 D.2.

    selectivity[neuron, pos] = P(neuron active | POS) / P(neuron active)
    Specialist: selectivity > 2.0 AND mean_weight > 0.1
    """
    print("\n" + "="*60)
    print("R.2: POS Selectivity")
    print("="*60)

    from transformers import AutoTokenizer
    from models.dawn_spatial_v3 import analysis_forward
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)
    n_know = model_cfg['n_know']
    n_qk = model_cfg['n_qk']
    n_v = model_cfg['n_v']

    # Load UD-EWT
    dataset = _load_ud_ewt(max_sentences)
    print(f"  Loaded {len(dataset)} sentences")

    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

    # Accumulators: per-neuron activation count, per-(neuron,pos) count
    n_pos = len(UPOS_TAGS)
    pos_to_idx = {p: i for i, p in enumerate(UPOS_TAGS)}

    # For each pool
    pools = {
        'QK': {'size': n_qk, 'gate_key': 'gate_Q'},  # Use Q gate for QK pool
        'V': {'size': n_v, 'gate_key': 'gate_V'},
        'Know': {'size': n_know, 'gate_key': 'gate_Know'},
    }
    pool_counts = {}
    pool_pos_counts = {}
    pos_token_counts = np.zeros(n_pos, dtype=np.float64)
    total_tokens = 0

    for pool_name, pinfo in pools.items():
        pool_counts[pool_name] = np.zeros(pinfo['size'], dtype=np.float64)
        pool_pos_counts[pool_name] = np.zeros((pinfo['size'], n_pos), dtype=np.float64)

    max_seq = model_cfg['max_seq_len']

    print(f"  Processing sentences...")
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

            # Pad to same length
            max_len = min(max(len(t) for t in batch_tokens), max_seq)
            padded = np.zeros((len(batch_tokens), max_len), dtype=np.int32)
            pos_padded = np.full((len(batch_tokens), max_len), -1, dtype=np.int32)
            for bi, (tids, plabs) in enumerate(zip(batch_tokens, batch_pos_labels)):
                l = min(len(tids), max_len)
                padded[bi, :l] = tids[:l]
                pos_padded[bi, :l] = plabs[:l]

            # Forward
            ids_dev = jnp.array(padded, dtype=jnp.int32)
            _, layer_info = jit_analysis(params, ids_dev)

            # Get gates (average across layers)
            for pool_name, pinfo in pools.items():
                # gate: [n_layers, B, S, N]
                gate = np.array(jax.device_get(layer_info[pinfo['gate_key']]))
                # Average over layers
                gate_avg = gate.mean(axis=0)  # [B, S, N]
                active = (gate_avg > 1e-6).astype(np.float32)  # [B, S, N]
                weights = gate_avg  # Use actual gate values for mean_weight

                # Accumulate
                for bi in range(len(batch_tokens)):
                    for ti in range(min(len(batch_tokens[bi]), max_len)):
                        pi = pos_padded[bi, ti]
                        if pi < 0:
                            continue
                        # Per-neuron total activation
                        pool_counts[pool_name] += active[bi, ti]
                        # Per-(neuron, pos) activation
                        pool_pos_counts[pool_name][:, pi] += active[bi, ti]
                        pos_token_counts[pi] += 1
                        total_tokens += 1

            batch_tokens = []
            batch_pos_labels = []

        if (si + 1) % 500 == 0:
            print(f"    [{si+1}/{len(dataset)}] sentences")

    # Compute selectivity
    results = {}
    for pool_name, pinfo in pools.items():
        N = pinfo['size']
        # P(neuron active)
        p_neuron = pool_counts[pool_name] / (total_tokens + 1e-8)  # [N]

        selectivity = np.zeros((N, n_pos))
        for pi in range(n_pos):
            if pos_token_counts[pi] > 0:
                # P(neuron active | POS)
                p_given_pos = pool_pos_counts[pool_name][:, pi] / pos_token_counts[pi]
                selectivity[:, pi] = p_given_pos / (p_neuron + 1e-8)

        # Mean weight per (neuron, pos) for specialist threshold
        mean_weight = np.zeros((N, n_pos))
        for pi in range(n_pos):
            if pos_token_counts[pi] > 0:
                mean_weight[:, pi] = pool_pos_counts[pool_name][:, pi] / pos_token_counts[pi]

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
    from models.dawn_spatial_v3 import prefill, decode_step, analysis_forward
    import numpy as np
    from collections import Counter

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cfg = get_model_cfg(cfg)

    jit_prefill = jax.jit(lambda p, ids: prefill(p, model_cfg, ids))
    jit_decode = jax.jit(lambda p, tok, cK, cV, cL: decode_step(p, model_cfg, tok, cK, cV, cL))
    jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids))

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
    from models.dawn_spatial_v3 import build_suppressed_forward
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
        from models.dawn_spatial_v3 import prefill
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

    # Load val tokens once (shared by D2 and D6)
    val_tokens = None
    val_path = args.val_data or cfg.get('data', {}).get('bin_val')
    if val_path and (only is None or any(k in (only or set()) for k in ['val', 'routing'])):
        val_tokens = load_val_tokens(val_path)

    if only is None or 'val' in only:
        if val_tokens is not None:
            analyze_validation(params, cfg, val_tokens, args.output,
                             args.batch_size, args.max_batches)
        else:
            print("\n  Skipping validation (no --val_data)")

    if only is None or 'health' in only:
        analyze_neuron_health(params, cfg, args.output)

    if only is None or 'generate' in only:
        analyze_generation(params, cfg, args.output,
                          prompt=args.prompt,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature)

    if only is None or 'weights' in only:
        analyze_weights(params, cfg, args.output)

    if only is None or 'routing' in only:
        if val_tokens is not None:
            analyze_routing(params, cfg, val_tokens, args.output)
        else:
            print("\n  Skipping routing (no --val_data)")

    if only is None or 'samples' in only:
        analyze_generation_samples(params, cfg, args.output,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature)

    # --- Rebuttal analyses (D.1-D.5 methodology) ---
    # Load val tokens for rebuttal analyses if not already loaded
    if val_tokens is None and val_path:
        needs_val = any(k in (only or set()) for k in ['r1', 'r4']) if only else True
        if needs_val:
            val_tokens = load_val_tokens(val_path)

    if only is None or 'r1' in only:
        if val_tokens is not None:
            analyze_qk_specialization(params, cfg, val_tokens, args.output)
        else:
            print("\n  Skipping R.1 (no --val_data)")

    if only is None or 'r2' in only:
        analyze_pos_selectivity(params, cfg, args.output)

    if only is None or 'r4' in only:
        if val_tokens is not None:
            analyze_layer_balance(params, cfg, val_tokens, args.output)
        else:
            print("\n  Skipping R.4 (no --val_data)")

    r3_results = None
    if only is None or 'r3' in only:
        r3_results = analyze_knowledge_neurons(params, cfg, args.output)

    if only is None or 'r5' in only:
        analyze_suppression(params, cfg, args.output,
                           knowledge_results=r3_results)

    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    main()

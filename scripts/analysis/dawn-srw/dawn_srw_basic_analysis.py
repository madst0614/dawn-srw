#!/usr/bin/env python3
"""Basic DAWN-SRW checkpoint analysis.

Runs the pre-experiment sanity checks:
  A0. checkpoint load + legacy migration sanity
  A1. parameter/model summary
  A2. validation loss / perplexity / accuracy
  A3. SRW neuron health
  A4. signature embedding weight analysis
  A5. generation samples

Example:
  python scripts/analysis/dawn_srw/dawn_srw_basic_analysis.py \
    --config configs/train_config_dawn_srw.yaml \
    --checkpoint checkpoints/run_xxx \
    --val-data data/val.bin \
    --output results/dawn_srw_basic --only all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model, count_params, encode_prompt, ensure_dir, import_dawn_srw,
    load_checkpoint_params, load_config, load_tokenizer, load_val_tokens,
    model_cfg_from_config, save_json,
)


def summarize_model(params, cfg, meta, output_dir):
    m = cfg["model"]
    n_total = count_params(params)
    pool_n = count_params(params.get("neuron_pool", {}))
    router_n = count_params(params.get("router", {}))
    emb_n = count_params(params.get("token_emb", {})) + count_params(params.get("pos_emb", {}))
    blocks_n = n_total - pool_n - router_n - emb_n - count_params(params.get("norm", {}))

    info = {
        "checkpoint": meta,
        "model_version": m.get("model_version", "dawn_srw"),
        "n_params": n_total,
        "n_params_M": n_total / 1e6,
        "d_model": m.get("d_model", 384),
        "n_layers": m.get("n_layers", 12),
        "n_heads": m.get("n_heads", 6),
        "d_route": m.get("d_route", m.get("d_bottleneck", 128)),
        "n_qk": m.get("n_qk", 1580),
        "n_v": m.get("n_v", 2600),
        "n_rst": m.get("n_rst", m.get("n_know", 25200)),
        "param_breakdown": {
            "neuron_pool": pool_n,
            "router": router_n,
            "embeddings": emb_n,
            "blocks": blocks_n,
        },
    }
    print("\nA1. Model / parameter summary")
    print(f"  params: {n_total:,} ({n_total/1e6:.2f}M)")
    print(f"  pools: attn_qk={info['n_qk']}, attn_v={info['n_v']}, rst={info['n_rst']}")
    print(f"  d_model={info['d_model']} layers={info['n_layers']} heads={info['n_heads']} d_route={info['d_route']}")
    path = save_json(info, output_dir, "model_info.json")
    print(f"  saved: {path}")
    return info


def analyze_validation(params, cfg, val_data, output_dir, max_tokens, batch_size, max_batches):
    print("\nA2. Validation loss / perplexity / accuracy")
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    max_seq = model_cfg["max_seq_len"]
    tokens = load_val_tokens(val_data, max_tokens=max_tokens)
    n_seqs = len(tokens) // max_seq
    if n_seqs == 0:
        raise ValueError(f"Not enough tokens for max_seq_len={max_seq}")
    if max_batches is not None:
        n_seqs = min(n_seqs, max_batches * batch_size)
    n_seqs = (n_seqs // batch_size) * batch_size
    arr = tokens[:n_seqs * max_seq].reshape(n_seqs, max_seq)
    print(f"  sequences={n_seqs:,}, batch_size={batch_size}, max_seq={max_seq}")
    params_jax = jax.tree.map(jnp.asarray, params)
    tokens_dev = jnp.asarray(arr, dtype=jnp.int32)

    @jax.jit
    def eval_fn(t):
        return mod.vectorized_eval(params_jax, model_cfg, t, batch_size=batch_size)

    t0 = time.time()
    avg_loss, ppl, acc, total_valid = eval_fn(tokens_dev)
    jax.block_until_ready(avg_loss)
    elapsed = time.time() - t0
    result = {
        "loss": float(avg_loss),
        "perplexity": float(ppl),
        "accuracy_pct": float(acc),
        "total_valid_tokens": int(total_valid),
        "time_sec": elapsed,
        "tokens_per_sec": int(total_valid) / elapsed if elapsed > 0 else 0.0,
    }
    print(f"  loss={result['loss']:.4f} ppl={result['perplexity']:.2f} acc={result['accuracy_pct']:.2f}%")
    path = save_json(result, output_dir, "validation.json")
    print(f"  saved: {path}")
    return result


def analyze_health(params, output_dir):
    print("\nA3. SRW neuron health")
    mod = import_dawn_srw()
    raw = jax.device_get(jax.jit(mod.vectorized_neuron_health)(params))
    result = {}
    for pool_name in ["Attention-QK", "Attention-V", "RST"]:
        s = raw[pool_name]
        result[pool_name] = {k: float(v) if hasattr(v, "shape") and np.asarray(v).shape == () else np.asarray(v).tolist() for k, v in s.items()}
        print(f"  {pool_name:12s} N={int(s['N'])} emb={float(s['emb_mean']):.4f} read={float(s['read_mean']):.4f} write={float(s['write_mean']):.4f}")
    result["tau_attn_bias"] = np.asarray(raw["tau_attn_bias"]).tolist()
    result["tau_rst_bias"] = np.asarray(raw["tau_rst_bias"]).tolist()
    path = save_json(result, output_dir, "neuron_health.json")
    print(f"  saved: {path}")
    return result


def analyze_weights(params, output_dir):
    print("\nA4. Signature embedding weight analysis")
    mod = import_dawn_srw()
    raw = jax.device_get(jax.jit(mod.vectorized_weight_analysis)(params))
    result = {}
    for pool_name in ["Attention-QK", "Attention-V", "RST"]:
        s = raw[pool_name]
        result[pool_name] = {k: float(v) if np.asarray(v).shape == () else np.asarray(v).tolist() for k, v in s.items()}
        print(f"  {pool_name:12s} N={int(s['N'])} mean_cos={float(s['mean_cosine_sim']):.4f} max_cos={float(s['max_cosine_sim']):.4f} eff_rank={float(s['effective_rank']):.1f}")
    path = save_json(result, output_dir, "weight_analysis.json")
    print(f"  saved: {path}")
    return result


def sample_from_logits(logits, rng, temperature=0.8, top_k=50):
    l = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        vals, _ = jax.lax.top_k(l, min(top_k, l.shape[-1]))
        l = jnp.where(l >= vals[-1], l, -1e10)
    return int(jax.random.categorical(rng, l))


def generate_samples(params, cfg, prompts, output_dir, max_new_tokens, temperature, top_k, tokenizer_name):
    print("\nA5. Generation samples")
    mod = import_dawn_srw()
    tok = load_tokenizer(tokenizer_name)
    model_cfg = model_cfg_from_config(cfg)
    max_seq = model_cfg["max_seq_len"]

    jit_prefill = jax.jit(lambda p, ids: mod.prefill(p, model_cfg, ids))
    jit_decode = jax.jit(lambda p, token_id, cK, cV, cL: mod.decode_step(p, model_cfg, token_id, cK, cV, cL))

    stop_ids = {x for x in [tok.sep_token_id, tok.pad_token_id, tok.eos_token_id] if x is not None}
    rng = jax.random.PRNGKey(123)
    results = []
    for prompt in prompts:
        ids = encode_prompt(tok, prompt, max_seq)
        if not ids:
            continue
        logits, cK, cV, cL = jit_prefill(params, jnp.asarray([ids], dtype=jnp.int32))
        jax.block_until_ready(logits)
        last_logits = logits[0, -1, :]
        generated = []
        gen_len = min(max_new_tokens, max_seq - len(ids))
        for _ in range(gen_len):
            rng, srng = jax.random.split(rng)
            next_id = sample_from_logits(last_logits, srng, temperature, top_k)
            generated.append(next_id)
            if next_id in stop_ids or int(cL) >= max_seq:
                break
            logits_d, cK, cV, cL = jit_decode(params, jnp.asarray([next_id], dtype=jnp.int32), cK, cV, cL)
            last_logits = logits_d[0]
        text = tok.decode(ids + generated, skip_special_tokens=True)
        gen_text = tok.decode(generated, skip_special_tokens=True)
        print(f"\n  prompt: {prompt!r}")
        print(f"  -> {gen_text[:180]}")
        results.append({"prompt": prompt, "generated": gen_text, "full_text": text, "new_tokens": len(generated)})
    path = save_json(results, output_dir, "generation_samples.json")
    print(f"  saved: {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="results/dawn_srw_basic")
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--max-val-tokens", type=int, default=262144)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--only", choices=["quick", "all", "validate", "generate", "health", "weights"], default="quick")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    output_dir = ensure_dir(args.output)
    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)

    summarize_model(params, cfg, meta, output_dir)

    if args.only in ("quick", "all", "health"):
        analyze_health(params, output_dir)
    if args.only in ("all", "weights"):
        analyze_weights(params, output_dir)
    if args.only in ("all", "validate"):
        if args.val_data is None:
            raise ValueError("--val-data is required for validation")
        analyze_validation(params, cfg, args.val_data, output_dir, args.max_val_tokens, args.batch_size, args.max_batches)
    if args.only in ("quick", "all", "generate"):
        prompts = args.prompt or [
            "The meaning of life is",
            "In a shocking finding, scientists discovered",
            "The capital of France is",
            "Once upon a time",
            "The best way to learn programming is",
        ]
        generate_samples(params, cfg, prompts, output_dir, args.max_new_tokens, args.temperature, args.top_k, args.tokenizer)


if __name__ == "__main__":
    main()

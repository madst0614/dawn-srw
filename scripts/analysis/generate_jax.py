#!/usr/bin/env python3
"""
TPU Text Generation Script  (KV-cached, jit-compiled)
======================================================
Load a DAWN checkpoint (local or GCS) and generate text via autoregressive
sampling.  Uses KV cache so each decode step processes only the new token
instead of the full sequence.

Usage:
    python scripts/analysis/generate_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/.../best_model.flax \
        --prompt "The capital of France is" \
        --max_tokens 50 --temperature 0.8 --top_k 50

    # Run predefined prompt suite:
    python scripts/analysis/generate_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/.../best_model.flax \
        --suite

    # Save results to JSON:
    python scripts/analysis/generate_jax.py \
        --checkpoint .../best_model.flax --suite --output gen_results.json
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from scripts.analysis.utils_jax import load_model_jax
from models.model_v17_1_jax import (
    dawn_init_kv_cache,
    dawn_cached_forward,
)


# ============================================================
# Sampling helpers  (pure numpy — runs on CPU, negligible cost)
# ============================================================

def _sample_token(logits_np, temperature, top_k, top_p, rng_key):
    """Sample next token from logits.  Returns (token_id, new_rng_key)."""
    if temperature <= 0:
        return int(np.argmax(logits_np)), rng_key

    logits_np = logits_np / temperature

    if top_k > 0:
        top_idx = np.argpartition(logits_np, -top_k)[-top_k:]
        mask = np.full_like(logits_np, -np.inf)
        mask[top_idx] = logits_np[top_idx]
        logits_np = mask

    if top_p < 1.0:
        sorted_idx = np.argsort(logits_np)[::-1]
        sorted_logits = logits_np[sorted_idx]
        probs_sorted = np.exp(sorted_logits - sorted_logits[0])
        probs_sorted /= probs_sorted.sum()
        cum = np.cumsum(probs_sorted)
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep = sorted_idx[:cutoff]
        mask = np.full_like(logits_np, -np.inf)
        mask[keep] = logits_np[keep]
        logits_np = mask

    probs = np.exp(logits_np - np.max(logits_np))
    probs = probs / (probs.sum() + 1e-8)
    rng_key, subkey = jax.random.split(rng_key)
    token = int(jax.random.choice(subkey, len(probs), p=probs))
    return token, rng_key


# ============================================================
# KV-cached generation
# ============================================================

def generate(params, config, input_ids, decode_step_fn,
             max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
    """Autoregressive generation with KV cache.

    Args:
        params:          model parameters  (FrozenDict)
        config:          model config dict
        input_ids:       1-D numpy array of prompt token IDs
        decode_step_fn:  jit-compiled dawn_cached_forward  (for S=1)
        max_new_tokens:  max tokens to generate
        temperature:     sampling temperature (0 = greedy)
        top_k:           top-k filtering (0 = disabled)
        top_p:           nucleus sampling threshold (1.0 = disabled)

    Returns:
        (generated_ids, elapsed_ms)
    """
    max_seq_len = config.get('max_seq_len', 512)
    eos_id = 102  # BERT [SEP]
    prompt_len = len(input_ids)

    # Truncate prompt if too long
    if prompt_len > max_seq_len - 1:
        input_ids = input_ids[-(max_seq_len - 1):]
        prompt_len = len(input_ids)

    # --- Prefill: process entire prompt at once ---
    kv_k, kv_v = dawn_init_kv_cache(config, batch_size=1)
    prompt_2d = jnp.array(input_ids[np.newaxis, :])  # [1, prompt_len]

    logits, kv_k, kv_v = dawn_cached_forward(
        params, config, prompt_2d, kv_k, kv_v, 0)
    # Wait for prefill to finish (XLA is async)
    logits.block_until_ready()

    generated = list(input_ids)
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
    cache_pos = prompt_len

    # Sample first token from prefill logits
    first_logits = np.array(logits[0, -1, :])
    next_token, rng_key = _sample_token(
        first_logits, temperature, top_k, top_p, rng_key)
    generated.append(next_token)

    if next_token == eos_id or cache_pos >= max_seq_len:
        return np.array(generated), 0.0

    # --- Decode loop: one token at a time with KV cache ---
    start = time.time()

    for _ in range(max_new_tokens - 1):
        token_2d = jnp.array([[next_token]])  # [1, 1]

        logits, kv_k, kv_v = decode_step_fn(
            params, config, token_2d, kv_k, kv_v, cache_pos)

        next_logits = np.array(logits[0, 0, :])
        next_token, rng_key = _sample_token(
            next_logits, temperature, top_k, top_p, rng_key)

        generated.append(next_token)
        cache_pos += 1

        if next_token == eos_id or cache_pos >= max_seq_len:
            break

    elapsed_ms = (time.time() - start) * 1000
    return np.array(generated), elapsed_ms


# ============================================================
# Prompt suites
# ============================================================

PROMPT_SUITE = {
    'Factual': [
        "The capital of France is",
        "The largest ocean on Earth is",
        "Einstein developed the theory of",
        "Water boils at",
    ],
    'Common Sense': [
        "If you drop a glass, it will",
        "Fire is hot, ice is",
        "At night, the sky is",
    ],
    'Narrative': [
        "Once upon a time, there was a",
        "She walked into the room and",
        "After years of hard work, he finally",
    ],
    'Technical': [
        "In machine learning, gradient descent is",
        "The mitochondria is the",
        "A neural network consists of",
    ],
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN TPU Text Generation')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint path (local or gs://)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate from')
    parser.add_argument('--suite', action='store_true',
                        help='Run predefined prompt suite')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true',
                        help='Greedy decoding (temperature=0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    if args.prompt is None and not args.suite:
        args.suite = True

    temperature = 0.0 if args.greedy else args.temperature

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    max_seq_len = config.get('max_seq_len', 512)
    print(f"  Model: v{config.get('model_version', '?')}, "
          f"d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")
    print(f"  JAX devices: {jax.devices()}")

    # --- Load tokenizer ---
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # --- JIT compile the decode step (S=1 fixed) ---
    # config is a Python dict with static values, so we pass it as a
    # closure via partial.  Only params / kv caches / cache_index are
    # dynamic JAX arrays.
    print("\nJIT compiling decode step...", end=" ", flush=True)
    compile_start = time.time()

    @jax.jit
    def decode_step(params, config_unused, token_ids, kv_k, kv_v, cache_pos):
        """Thin jit wrapper.  config captured from outer scope."""
        return dawn_cached_forward(
            params, config, token_ids, kv_k, kv_v, cache_pos)

    # Trigger compilation with dummy inputs
    dummy_kv_k, dummy_kv_v = dawn_init_kv_cache(config, batch_size=1)
    dummy_tok = jnp.array([[0]])
    _out = decode_step(params, config, dummy_tok, dummy_kv_k, dummy_kv_v, 0)
    _out[0].block_until_ready()  # wait for compile
    compile_time = time.time() - compile_start
    print(f"done ({compile_time:.1f}s)")

    results = []

    def run_prompt(prompt, category="custom"):
        input_ids = np.array(tokenizer.encode(prompt, add_special_tokens=False))
        gen_ids, elapsed = generate(
            params, config, input_ids, decode_step,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Decode ALL token IDs at once (never decode per-token and concat)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Split prompt vs continuation by decoding prompt IDs separately
        prompt_decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        if full_text.startswith(prompt_decoded):
            continuation = full_text[len(prompt_decoded):]
        else:
            continuation = full_text

        new_tokens = len(gen_ids) - len(input_ids)
        tok_per_sec = new_tokens / (elapsed / 1000) if elapsed > 0 else 0

        print(f"  [{category}] {prompt_decoded}\033[1m{continuation}\033[0m")
        print(f"           ({new_tokens} tokens, {elapsed:.0f}ms, "
              f"{tok_per_sec:.1f} tok/s)")

        results.append({
            'category': category,
            'prompt': prompt,
            'generated': full_text,
            'continuation': continuation.strip(),
            'new_tokens': new_tokens,
            'time_ms': round(elapsed, 1),
            'tokens_per_sec': round(tok_per_sec, 1),
        })

    # --- Generate ---
    print("\n" + "=" * 60)
    print("Text Generation  (KV-cached)")
    print("=" * 60)

    if args.prompt:
        run_prompt(args.prompt)

    if args.suite:
        for category, prompts in PROMPT_SUITE.items():
            print(f"\n--- {category} ---")
            for p in prompts:
                run_prompt(p, category)

    # --- Summary ---
    if results:
        avg_tps = np.mean([r['tokens_per_sec'] for r in results])
        print(f"\n{'=' * 60}")
        print(f"Average: {avg_tps:.1f} tok/s across {len(results)} prompts")

    # --- Save ---
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'config': {k: v for k, v in config.items()
                           if isinstance(v, (int, float, str, bool))},
                'generation_params': {
                    'temperature': temperature,
                    'top_k': args.top_k,
                    'top_p': args.top_p,
                    'max_tokens': args.max_tokens,
                },
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

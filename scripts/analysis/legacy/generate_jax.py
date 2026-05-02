#!/usr/bin/env python3
"""
TPU Text Generation Script  (KV-cached, jit-compiled)
======================================================
Load DAWN and/or Baseline checkpoints and generate text via autoregressive
sampling.  Uses KV cache so each decode step processes only the new token.

Usage:
    # Single model:
    python scripts/analysis/generate_jax.py \
        --checkpoint gs://.../dawn/best_model.flax \
        --suite --greedy

    # DAWN vs Baseline comparison:
    python scripts/analysis/generate_jax.py \
        --checkpoint gs://.../dawn/best_model.flax \
        --checkpoint2 gs://.../baseline/best_model.flax \
        --suite --greedy --output gen_comparison.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from scripts.analysis.utils_jax import load_model_jax
from models.model_v17_1_jax import dawn_init_kv_cache, dawn_cached_forward
from models.baseline_transformer_jax import vanilla_init_kv_cache, vanilla_cached_forward


# ============================================================
# Sampling
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
# Model loader — returns (params, config, decode_step, init_cache, fwd)
# ============================================================

def _is_baseline(config):
    v = config.get('model_version', '')
    return v in ('baseline', 'baseline-JAX')


def load_and_compile(checkpoint_path):
    """Load checkpoint, detect model type, JIT compile decode step.

    Returns dict with keys:
        params, config, name, decode_step, init_cache_fn, cached_fwd_fn
    """
    print(f"Loading: {checkpoint_path}")
    model_cls, params, config = load_model_jax(checkpoint_path)
    is_base = _is_baseline(config)
    version = config.get('model_version', '?')
    d_model = config.get('d_model', '?')
    n_layers = config.get('n_layers', '?')
    name = "Baseline" if is_base else "DAWN"
    print(f"  {name}: v{version}, d_model={d_model}, n_layers={n_layers}")

    if is_base:
        init_fn = vanilla_init_kv_cache
        fwd_fn = vanilla_cached_forward
    else:
        init_fn = dawn_init_kv_cache
        fwd_fn = dawn_cached_forward

    @jax.jit
    def decode_step(params, token_ids, kv_k, kv_v, cache_pos):
        return fwd_fn(params, config, token_ids, kv_k, kv_v, cache_pos)

    # Warm-up compile
    print(f"  JIT compiling {name}...", end=" ", flush=True)
    t0 = time.time()
    dummy_kv_k, dummy_kv_v = init_fn(config, batch_size=1)
    _out = decode_step(params, jnp.array([[0]]), dummy_kv_k, dummy_kv_v, 0)
    _out[0].block_until_ready()
    print(f"done ({time.time() - t0:.1f}s)")

    return {
        'params': params,
        'config': config,
        'name': name,
        'decode_step': decode_step,
        'init_cache_fn': init_fn,
        'cached_fwd_fn': fwd_fn,
    }


# ============================================================
# KV-cached generation
# ============================================================

def generate(model, input_ids,
             max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
    """Autoregressive generation with KV cache.

    Args:
        model:           dict from load_and_compile()
        input_ids:       1-D numpy array of prompt token IDs

    Returns:
        (generated_ids, elapsed_ms)
    """
    params = model['params']
    config = model['config']
    decode_step = model['decode_step']
    init_cache = model['init_cache_fn']
    fwd_fn = model['cached_fwd_fn']
    max_seq_len = config.get('max_seq_len', 512)
    eos_id = 102

    prompt_len = len(input_ids)
    if prompt_len > max_seq_len - 1:
        input_ids = input_ids[-(max_seq_len - 1):]
        prompt_len = len(input_ids)

    # Prefill
    kv_k, kv_v = init_cache(config, batch_size=1)
    prompt_2d = jnp.array(input_ids[np.newaxis, :])
    logits, kv_k, kv_v = fwd_fn(params, config, prompt_2d, kv_k, kv_v, 0)
    logits.block_until_ready()

    generated = list(input_ids)
    rng_key = jax.random.PRNGKey(42)
    cache_pos = prompt_len

    first_logits = np.array(logits[0, -1, :])
    next_token, rng_key = _sample_token(
        first_logits, temperature, top_k, top_p, rng_key)
    generated.append(next_token)

    if next_token == eos_id or cache_pos >= max_seq_len:
        return np.array(generated), 0.0

    # Decode loop
    start = time.time()
    for _ in range(max_new_tokens - 1):
        token_2d = jnp.array([[next_token]])
        logits, kv_k, kv_v = decode_step(params, token_2d, kv_k, kv_v, cache_pos)

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
# Prompt suite  (matches GPU performance.py PAPER_COMPARISON_PROMPTS)
# ============================================================

PROMPT_SUITE = {
    'Factual Knowledge': [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water boils at",
        "Albert Einstein was born in",
    ],
    'Common Sense': [
        "If you drop a glass on the floor, it will",
        "Fire is hot, but ice is",
        "Birds fly in the sky, fish swim in",
        "At night, the sun is",
    ],
    'Narrative': [
        "Once upon a time, in a small village,",
        "The detective examined the evidence and",
        "She opened the door and saw",
        "After years of hard work, he finally",
    ],
    'Technical': [
        "In machine learning, gradient descent is",
        "The function of the mitochondria is",
        "A neural network consists of",
    ],
}


# ============================================================
# Output formatting
# ============================================================

def _decode_and_split(tokenizer, prompt, gen_ids, input_ids):
    """Decode token IDs and split into prompt + continuation."""
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    if full_text.startswith(prompt_decoded):
        continuation = full_text[len(prompt_decoded):]
    else:
        continuation = full_text
    return full_text, prompt_decoded, continuation


def run_single(model, tokenizer, prompts, gen_kwargs):
    """Run generation on a single model.  Returns list of result dicts."""
    results = []
    for category, cat_prompts in prompts.items():
        print(f"\n--- {category} ---")
        for prompt in cat_prompts:
            input_ids = np.array(
                tokenizer.encode(prompt, add_special_tokens=False))
            gen_ids, elapsed = generate(model, input_ids, **gen_kwargs)

            full_text, prompt_dec, cont = _decode_and_split(
                tokenizer, prompt, gen_ids, input_ids)
            new_tokens = len(gen_ids) - len(input_ids)
            tps = new_tokens / (elapsed / 1000) if elapsed > 0 else 0

            print(f"  [{model['name']}] {prompt_dec}\033[1m{cont}\033[0m")
            print(f"           ({new_tokens} tok, {elapsed:.0f}ms, {tps:.1f} tok/s)")

            results.append({
                'category': category,
                'prompt': prompt,
                'generated': full_text,
                'continuation': cont.strip(),
                'new_tokens': new_tokens,
                'time_ms': round(elapsed, 1),
                'tokens_per_sec': round(tps, 1),
            })
    return results


def run_comparison(model_a, model_b, tokenizer, prompts, gen_kwargs):
    """Side-by-side comparison (GPU performance.py format).

    Returns (results_a, results_b, comparison_lines).
    """
    results_a, results_b = [], []
    lines = []

    hdr = "=" * 100
    lines.append(hdr)
    lines.append(f"GENERATION COMPARISON: {model_a['name']} vs {model_b['name']}  (JAX / KV-cached)")
    lines.append(hdr)
    cfg_a, cfg_b = model_a['config'], model_b['config']
    lines.append(f"{model_a['name']:>10}: v{cfg_a.get('model_version','?')}, "
                 f"d_model={cfg_a.get('d_model')}, n_layers={cfg_a.get('n_layers')}")
    lines.append(f"{model_b['name']:>10}: v{cfg_b.get('model_version','?')}, "
                 f"d_model={cfg_b.get('d_model')}, n_layers={cfg_b.get('n_layers')}")
    lines.append(f"{'Params':>10}: temperature={gen_kwargs.get('temperature',0.8)}, "
                 f"top_k={gen_kwargs.get('top_k',50)}, max_tokens={gen_kwargs.get('max_new_tokens',50)}")
    lines.append(hdr)
    print("\n" + "\n".join(lines))

    for category, cat_prompts in prompts.items():
        cat_hdr = f"\n{'=' * 100}\n[{category.upper()}]\n{'=' * 100}"
        lines.append(cat_hdr)
        print(cat_hdr)

        for prompt in cat_prompts:
            input_ids = np.array(
                tokenizer.encode(prompt, add_special_tokens=False))

            # Generate from both models
            gen_a, elapsed_a = generate(model_a, input_ids, **gen_kwargs)
            gen_b, elapsed_b = generate(model_b, input_ids, **gen_kwargs)

            full_a, _, cont_a = _decode_and_split(
                tokenizer, prompt, gen_a, input_ids)
            full_b, prompt_dec, cont_b = _decode_and_split(
                tokenizer, prompt, gen_b, input_ids)

            new_a = len(gen_a) - len(input_ids)
            new_b = len(gen_b) - len(input_ids)
            tps_a = new_a / (elapsed_a / 1000) if elapsed_a > 0 else 0
            tps_b = new_b / (elapsed_b / 1000) if elapsed_b > 0 else 0

            prompt_line = f"\n{'─' * 100}\nPrompt: \"{prompt_dec}\"\n{'─' * 100}"
            a_line = f"  {model_a['name']:>10}:  {cont_a.strip()}"
            b_line = f"  {model_b['name']:>10}:  {cont_b.strip()}"
            stats_a = f"               ({new_a} tok, {elapsed_a:.0f}ms, {tps_a:.1f} tok/s)"
            stats_b = f"               ({new_b} tok, {elapsed_b:.0f}ms, {tps_b:.1f} tok/s)"

            lines.extend([prompt_line, a_line, stats_a, b_line, stats_b])
            print(prompt_line)
            print(a_line)
            print(stats_a)
            print(b_line)
            print(stats_b)

            for r_list, full, cont, new, elapsed, tps in [
                (results_a, full_a, cont_a, new_a, elapsed_a, tps_a),
                (results_b, full_b, cont_b, new_b, elapsed_b, tps_b),
            ]:
                r_list.append({
                    'category': category,
                    'prompt': prompt,
                    'generated': full,
                    'continuation': cont.strip(),
                    'new_tokens': new,
                    'time_ms': round(elapsed, 1),
                    'tokens_per_sec': round(tps, 1),
                })

    return results_a, results_b, lines


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAWN TPU Text Generation (KV-cached)')
    parser.add_argument('--checkpoint', required=True,
                        help='Primary checkpoint (local or gs://)')
    parser.add_argument('--checkpoint2', default=None,
                        help='Second checkpoint for comparison')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate from')
    parser.add_argument('--suite', action='store_true',
                        help='Run predefined prompt suite')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true',
                        help='Greedy decoding (temperature=0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON (or dir for comparison)')
    args = parser.parse_args()

    if args.prompt is None and not args.suite:
        args.suite = True

    temperature = 0.0 if args.greedy else args.temperature
    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        temperature=temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(f"JAX devices: {jax.devices()}\n")

    # --- Load models ---
    model_a = load_and_compile(args.checkpoint)

    model_b = None
    if args.checkpoint2:
        model_b = load_and_compile(args.checkpoint2)

    # --- Tokenizer ---
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # --- Prompts ---
    prompts = PROMPT_SUITE
    if args.prompt:
        prompts = {'Custom': [args.prompt]}

    # ---- Run ----
    if model_b is not None:
        # === Comparison mode ===
        results_a, results_b, comp_lines = run_comparison(
            model_a, model_b, tokenizer, prompts, gen_kwargs)

        # Summary
        avg_a = np.mean([r['tokens_per_sec'] for r in results_a]) if results_a else 0
        avg_b = np.mean([r['tokens_per_sec'] for r in results_b]) if results_b else 0
        print(f"\n{'=' * 100}")
        print(f"  {model_a['name']}: avg {avg_a:.1f} tok/s   |   "
              f"{model_b['name']}: avg {avg_b:.1f} tok/s")
        print(f"{'=' * 100}")

        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)

            # JSON
            with open(out_dir / 'generation_comparison.json', 'w') as f:
                json.dump({
                    model_a['name']: {
                        'checkpoint': args.checkpoint,
                        'config': {k: v for k, v in model_a['config'].items()
                                   if isinstance(v, (int, float, str, bool))},
                        'results': results_a,
                    },
                    model_b['name']: {
                        'checkpoint': args.checkpoint2,
                        'config': {k: v for k, v in model_b['config'].items()
                                   if isinstance(v, (int, float, str, bool))},
                        'results': results_b,
                    },
                    'generation_params': gen_kwargs,
                }, f, indent=2)

            # TXT
            with open(out_dir / 'generation_comparison.txt', 'w') as f:
                f.write("\n".join(comp_lines))

            print(f"\nSaved to {out_dir}/")

    else:
        # === Single model mode ===
        print(f"\n{'=' * 60}")
        print(f"Text Generation: {model_a['name']}  (KV-cached)")
        print(f"{'=' * 60}")

        results = run_single(model_a, tokenizer, prompts, gen_kwargs)

        if results:
            avg = np.mean([r['tokens_per_sec'] for r in results])
            print(f"\n{'=' * 60}")
            print(f"Average: {avg:.1f} tok/s across {len(results)} prompts")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump({
                    'checkpoint': args.checkpoint,
                    'config': {k: v for k, v in model_a['config'].items()
                               if isinstance(v, (int, float, str, bool))},
                    'generation_params': gen_kwargs,
                    'results': results,
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

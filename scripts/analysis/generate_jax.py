#!/usr/bin/env python3
"""
TPU Text Generation Script
===========================
Load a DAWN checkpoint (local or GCS) and generate text via autoregressive sampling.
Useful as a quick sanity check that the model + checkpoint are functional.

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
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from scripts.analysis.utils_jax import load_model_jax, create_model_from_config


# ============================================================
# Generation
# ============================================================

def generate(model_instance, params, input_ids, max_new_tokens=50,
             temperature=0.8, top_k=50, top_p=0.9, max_seq_len=512):
    """Autoregressive text generation with top-k/top-p sampling.

    Args:
        model_instance: Instantiated Flax model
        params: FrozenDict parameters
        input_ids: 1-D numpy array of prompt token IDs
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        max_seq_len: Model's maximum sequence length

    Returns:
        (generated_ids, elapsed_ms)
    """
    generated = list(input_ids.flatten())
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
    eos_id = 102  # BERT [SEP]

    start = time.time()

    for _ in range(max_new_tokens):
        seq = np.array([generated[-max_seq_len:]])

        result = model_instance.apply(
            params,
            jnp.array(seq),
            deterministic=True,
            rngs={'dropout': rng_key},
        )
        logits = np.array(result['logits'][0, -1])

        if temperature > 0:
            logits = logits / temperature

            if top_k > 0:
                top_indices = np.argsort(logits)[-top_k:]
                mask = np.full_like(logits, -np.inf)
                mask[top_indices] = logits[top_indices]
                logits = mask

            if top_p < 1.0:
                sorted_idx = np.argsort(logits)[::-1]
                sorted_logits = logits[sorted_idx]
                probs_sorted = np.exp(sorted_logits - np.max(sorted_logits))
                probs_sorted /= probs_sorted.sum()
                cum = np.cumsum(probs_sorted)
                cutoff = np.searchsorted(cum, top_p) + 1
                keep = sorted_idx[:cutoff]
                mask = np.full_like(logits, -np.inf)
                mask[keep] = logits[keep]
                logits = mask

            probs = np.exp(logits - np.max(logits))
            probs = probs / (probs.sum() + 1e-8)
            rng_key, subkey = jax.random.split(rng_key)
            next_token = int(jax.random.choice(subkey, len(probs), p=probs))
        else:
            next_token = int(np.argmax(logits))

        generated.append(next_token)
        if next_token == eos_id:
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
        args.suite = True  # Default to suite if no prompt given

    temperature = 0.0 if args.greedy else args.temperature

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    model_instance = create_model_from_config(config)
    max_seq_len = config.get('max_seq_len', 512)
    print(f"  Model: v{config.get('model_version', '?')}, "
          f"d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")
    print(f"  JAX devices: {jax.devices()}")

    # --- Load tokenizer ---
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # --- Warm-up (first call compiles XLA) ---
    print("\nWarm-up (XLA compile)...", end=" ", flush=True)
    dummy = np.array(tokenizer.encode("hello", add_special_tokens=False))
    _ = generate(model_instance, params, dummy, max_new_tokens=2,
                 temperature=temperature, max_seq_len=max_seq_len)
    print("done.")

    results = []

    def run_prompt(prompt, category="custom"):
        input_ids = np.array(tokenizer.encode(prompt, add_special_tokens=False))
        gen_ids, elapsed = generate(
            model_instance, params, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_seq_len=max_seq_len,
        )
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        new_tokens = len(gen_ids) - len(input_ids)
        tok_per_sec = new_tokens / (elapsed / 1000) if elapsed > 0 else 0

        # Print: prompt in bold, continuation normal
        continuation = text[len(prompt):] if text.startswith(prompt) else text
        print(f"  [{category}] {prompt}\033[1m{continuation}\033[0m")
        print(f"           ({new_tokens} tokens, {elapsed:.0f}ms, {tok_per_sec:.1f} tok/s)")

        results.append({
            'category': category,
            'prompt': prompt,
            'generated': text,
            'new_tokens': new_tokens,
            'time_ms': round(elapsed, 1),
            'tokens_per_sec': round(tok_per_sec, 1),
        })

    # --- Generate ---
    print("\n" + "=" * 60)
    print("Text Generation")
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

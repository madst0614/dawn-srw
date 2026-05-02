#!/usr/bin/env python3
"""
TPU Validation Loss / Perplexity Script
=========================================
Evaluate a DAWN checkpoint on C4 validation data.
Reports loss, perplexity, and accuracy.

Usage:
    # With GCS val data:
    python scripts/analysis/eval_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/.../best_model.flax \
        --val_data gs://dawn-tpu-data-c4/data/c4_val.bin

    # With custom batch size / token limit:
    python scripts/analysis/eval_jax.py \
        --checkpoint .../best_model.flax \
        --val_data .../c4_val.bin \
        --batch_size 16 --seq_len 512 --max_tokens 5000000

    # Save results + loss curve PNG:
    python scripts/analysis/eval_jax.py \
        --checkpoint .../best_model.flax \
        --val_data .../c4_val.bin \
        --output eval_results/
"""

import matplotlib
matplotlib.use('Agg')

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

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config,
    load_bin_data, create_batches,
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model_instance, params, val_tokens, batch_size=32, seq_len=512,
             verbose=True):
    """Evaluate model on validation tokens.

    Args:
        model_instance: Instantiated Flax model
        params: FrozenDict parameters
        val_tokens: Flat numpy array of token IDs
        batch_size: Batch size
        seq_len: Sequence length
        verbose: Print per-batch progress

    Returns:
        Dict with loss, perplexity, accuracy, per-batch losses
    """
    batches = create_batches(val_tokens, batch_size, seq_len)
    n_batches = len(batches)

    if n_batches == 0:
        raise ValueError(
            f"Not enough tokens for even 1 batch "
            f"(have {len(val_tokens)}, need {batch_size * seq_len})")

    rng_key = jax.random.PRNGKey(42)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batch_losses = []

    start = time.time()

    for i, batch in enumerate(batches):
        batch_jax = jnp.array(batch)

        result = model_instance.apply(
            params,
            batch_jax,
            labels=batch_jax,
            deterministic=True,
            rngs={'dropout': rng_key},
        )

        if 'loss' in result:
            loss_val = float(result['loss'])
            n_valid = int(result.get('valid_count', batch_size * (seq_len - 1)))

            total_loss += loss_val * n_valid
            total_correct += int(result.get('correct', 0))
            total_tokens += n_valid
            batch_losses.append(loss_val)

            if verbose and (i + 1) % max(1, n_batches // 20) == 0:
                elapsed = time.time() - start
                avg = total_loss / total_tokens
                ppl = np.exp(avg) if avg < 100 else float('inf')
                print(f"  [{i+1}/{n_batches}] loss={avg:.4f} ppl={ppl:.2f} "
                      f"acc={total_correct/total_tokens*100:.1f}% "
                      f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'n_batches': n_batches,
        'elapsed_sec': round(elapsed, 1),
        'batch_losses': batch_losses,
    }


# ============================================================
# Plotting
# ============================================================

def plot_eval_results(results, output_path):
    """Save per-batch loss curve as PNG."""
    import matplotlib.pyplot as plt

    batch_losses = results.get('batch_losses', [])
    if len(batch_losses) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax = axes[0]
    ax.plot(batch_losses, linewidth=0.8, alpha=0.7, color='steelblue')
    # Smoothed
    if len(batch_losses) > 10:
        window = max(5, len(batch_losses) // 20)
        smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, window-1+len(smoothed)), smoothed,
                linewidth=2, color='darkblue', label=f'MA-{window}')
        ax.legend()
    ax.axhline(results['loss'], color='red', linestyle='--', linewidth=1,
               label=f"avg={results['loss']:.4f}")
    ax.legend()
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Per-Batch Validation Loss')
    ax.grid(True, alpha=0.3)

    # Loss distribution
    ax = axes[1]
    ax.hist(batch_losses, bins=min(50, len(batch_losses)), color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(results['loss'], color='red', linestyle='--', linewidth=2,
               label=f"mean={results['loss']:.4f}")
    ax.axvline(np.median(batch_losses), color='orange', linestyle='--',
               linewidth=2, label=f"median={np.median(batch_losses):.4f}")
    ax.legend()
    ax.set_xlabel('Loss')
    ax.set_ylabel('Count')
    ax.set_title('Loss Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN TPU Validation Evaluation')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint path (local or gs://)')
    parser.add_argument('--val_data', required=True,
                        help='Validation data path (.bin, local or gs://)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--max_tokens', type=int, default=10_000_000,
                        help='Max tokens to load from val data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results JSON + plot')
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    model_instance = create_model_from_config(config)
    print(f"  Model: v{config.get('model_version', '?')}, "
          f"d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")
    print(f"  JAX devices: {jax.devices()}")

    # --- Load val data ---
    print(f"\nLoading val data: {args.val_data}")
    val_tokens = load_bin_data(args.val_data, max_tokens=args.max_tokens)
    print(f"  Loaded {len(val_tokens):,} tokens")

    # --- Evaluate ---
    print(f"\nEvaluating (batch_size={args.batch_size}, seq_len={args.seq_len})...")
    results = evaluate(
        model_instance, params, val_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # --- Print results ---
    print(f"\n{'=' * 50}")
    print(f"  Loss:       {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Accuracy:   {results['accuracy']:.2f}%")
    print(f"  Tokens:     {results['total_tokens']:,}")
    print(f"  Batches:    {results['n_batches']}")
    print(f"  Time:       {results['elapsed_sec']}s")
    print(f"{'=' * 50}")

    # --- Save ---
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON (without batch_losses for compactness)
        save_data = {k: v for k, v in results.items() if k != 'batch_losses'}
        save_data['checkpoint'] = args.checkpoint
        save_data['val_data'] = args.val_data
        save_data['config'] = {k: v for k, v in config.items()
                               if isinstance(v, (int, float, str, bool))}

        json_path = out_dir / 'eval_results.json'
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved: {json_path}")

        # Plot
        plot_eval_results(results, out_dir / 'eval_loss_curve.png')


if __name__ == '__main__':
    main()

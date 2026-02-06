"""
Performance Analysis (JAX Version)
==================================
Performance analysis tools for JAX/Flax DAWN models.

Includes:
- Text generation
- Speed benchmarking
- Model comparison
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from .utils_jax import (
    evaluate_jax, forward_jax, create_model_from_config,
    load_val_data_jax, create_batches,
    estimate_flops_jax, count_params_jax,
)


class TextGeneratorJAX:
    """Text generation utility for JAX models."""

    def __init__(self, tokenizer=None, vocab_size: int = 30522):
        """
        Args:
            tokenizer: Optional tokenizer for encoding/decoding
            vocab_size: Vocabulary size if no tokenizer provided
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def generate(
        self,
        model,
        params,
        config: Dict,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[np.ndarray, float]:
        """Generate text continuation.

        Args:
            model: JAX model class
            params: FrozenDict of parameters
            config: Model configuration
            prompt_ids: Input token IDs [seq_len] or [1, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            Tuple of (generated_ids, time_ms)
        """
        if not HAS_JAX:
            return prompt_ids, 0.0

        # Ensure 2D input
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[np.newaxis, :]

        model_instance = create_model_from_config(config)
        max_seq_len = config.get('max_seq_len', 512)

        generated = list(prompt_ids.flatten())
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)

        start_time = time.time()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_ids = np.array([generated[-max_seq_len:]])

            # Forward pass
            result = model_instance.apply(
                params,
                jnp.array(input_ids),
                deterministic=True,
                rngs={'dropout': rng_key}
            )

            logits = np.array(result['logits'][0, -1])

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_indices = np.argsort(logits)[-top_k:]
                    mask = np.ones_like(logits) * float('-inf')
                    mask[top_indices] = logits[top_indices]
                    logits = mask

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_indices = np.argsort(logits)[::-1]
                    sorted_logits = logits[sorted_indices]
                    probs = np.exp(sorted_logits - np.max(sorted_logits))
                    probs = probs / probs.sum()
                    cumsum = np.cumsum(probs)
                    cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                    mask = np.ones_like(logits) * float('-inf')
                    mask[sorted_indices[:cutoff_idx]] = logits[sorted_indices[:cutoff_idx]]
                    logits = mask

                # Sample
                probs = np.exp(logits - np.max(logits))
                probs = probs / (probs.sum() + 1e-8)
                rng_key, subkey = jax.random.split(rng_key)
                next_token = int(jax.random.choice(subkey, len(probs), p=probs))
            else:
                # Greedy
                next_token = int(np.argmax(logits))

            generated.append(next_token)

            # Stop on EOS
            if next_token in [102, 0]:  # BERT [SEP] or padding
                break

        elapsed_ms = (time.time() - start_time) * 1000

        return np.array(generated), elapsed_ms

    def generate_from_text(
        self,
        model,
        params,
        config: Dict,
        prompt: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text from string prompt.

        Args:
            model: JAX model class
            params: FrozenDict of parameters
            config: Model configuration
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_text, metadata)
        """
        if self.tokenizer is None:
            return "", {'error': 'No tokenizer provided'}

        # Encode prompt
        encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_ids = np.array(encoded)

        # Generate
        generated_ids, time_ms = self.generate(
            model, params, config, prompt_ids,
            max_new_tokens=max_new_tokens, **kwargs
        )

        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        metadata = {
            'prompt': prompt,
            'prompt_tokens': len(prompt_ids),
            'generated_tokens': len(generated_ids),
            'new_tokens': len(generated_ids) - len(prompt_ids),
            'time_ms': time_ms,
            'tokens_per_sec': (len(generated_ids) - len(prompt_ids)) / (time_ms / 1000) if time_ms > 0 else 0,
        }

        return generated_text, metadata


class SpeedBenchmarkJAX:
    """Speed benchmarking for JAX models."""

    def __init__(self):
        pass

    def benchmark_forward(
        self,
        model,
        params,
        config: Dict,
        batch_size: int = 16,
        seq_len: int = 512,
        n_warmup: int = 5,
        n_runs: int = 20,
    ) -> Dict:
        """Benchmark forward pass speed.

        Args:
            model: JAX model class
            params: FrozenDict of parameters
            config: Model configuration
            batch_size: Batch size
            seq_len: Sequence length
            n_warmup: Number of warmup runs
            n_runs: Number of timed runs

        Returns:
            Dictionary with timing statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        model_instance = create_model_from_config(config)
        vocab_size = config.get('vocab_size', 30522)

        # Create random input
        rng_key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)

        # Warmup
        for _ in range(n_warmup):
            _ = model_instance.apply(
                params,
                input_ids,
                deterministic=True,
                rngs={'dropout': rng_key}
            )

        # Block until computation is complete
        jax.block_until_ready(input_ids)

        # Timed runs
        times = []
        for _ in range(n_runs):
            start = time.time()
            result = model_instance.apply(
                params,
                input_ids,
                deterministic=True,
                rngs={'dropout': rng_key}
            )
            # Block until complete
            if 'logits' in result:
                jax.block_until_ready(result['logits'])
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms

        times = np.array(times)
        total_tokens = batch_size * seq_len

        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'n_runs': n_runs,
            'avg_time_ms': float(times.mean()),
            'std_time_ms': float(times.std()),
            'min_time_ms': float(times.min()),
            'max_time_ms': float(times.max()),
            'tokens_per_sec': float(total_tokens / (times.mean() / 1000)),
            'batches_per_sec': float(1000 / times.mean()),
        }

    def benchmark_generation(
        self,
        model,
        params,
        config: Dict,
        prompt_len: int = 50,
        gen_len: int = 50,
        n_runs: int = 10,
    ) -> Dict:
        """Benchmark text generation speed.

        Args:
            model: JAX model class
            params: FrozenDict of parameters
            config: Model configuration
            prompt_len: Prompt length
            gen_len: Generation length
            n_runs: Number of runs

        Returns:
            Dictionary with timing statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        generator = TextGeneratorJAX(vocab_size=config.get('vocab_size', 30522))

        # Create random prompt
        vocab_size = config.get('vocab_size', 30522)
        prompt_ids = np.random.randint(1, vocab_size, size=prompt_len)

        times = []
        tokens_generated = []

        for _ in range(n_runs):
            generated, time_ms = generator.generate(
                model, params, config, prompt_ids,
                max_new_tokens=gen_len, temperature=0.8
            )
            times.append(time_ms)
            tokens_generated.append(len(generated) - prompt_len)

        times = np.array(times)
        tokens = np.array(tokens_generated)

        return {
            'prompt_len': prompt_len,
            'target_gen_len': gen_len,
            'n_runs': n_runs,
            'avg_time_ms': float(times.mean()),
            'std_time_ms': float(times.std()),
            'avg_tokens_generated': float(tokens.mean()),
            'tokens_per_sec': float(tokens.mean() / (times.mean() / 1000)) if times.mean() > 0 else 0,
        }


class ModelComparatorJAX:
    """Compare multiple JAX models."""

    def __init__(self, val_data_path: str, batch_size: int = 32, seq_len: int = 512):
        """
        Args:
            val_data_path: Path to validation data
            batch_size: Batch size for evaluation
            seq_len: Sequence length
        """
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._val_tokens = None

    def _load_val_tokens(self, max_batches: int = 200):
        """Load validation tokens (cached)."""
        if self._val_tokens is None:
            max_tokens = max_batches * self.batch_size * self.seq_len
            self._val_tokens = load_val_data_jax(self.val_data_path, max_tokens)
        return self._val_tokens

    def compare_performance(
        self,
        models: List[Tuple[str, any, any, Dict]],
        n_batches: int = 100
    ) -> Dict:
        """Compare model performance.

        Args:
            models: List of (name, model, params, config) tuples
            n_batches: Number of batches for evaluation

        Returns:
            Dictionary with comparison results
        """
        val_tokens = self._load_val_tokens(n_batches)
        results = {}

        for name, model, params, config in models:
            print(f"  Evaluating {name}...")

            # Count params and FLOPs
            total_params = count_params_jax(params)
            flops = estimate_flops_jax(config, self.seq_len)

            # Evaluate
            model_instance = create_model_from_config(config)
            eval_results = evaluate_jax(
                model_instance, params, config,
                val_tokens,
                batch_size=self.batch_size,
                seq_len=self.seq_len
            )

            results[name] = {
                'params_M': total_params / 1e6,
                'flops_G': flops / 1e9,
                'loss': eval_results.get('loss', 0),
                'perplexity': eval_results.get('perplexity', 0),
                'accuracy': eval_results.get('accuracy', 0),
            }

        return results

    def compare_speed(
        self,
        models: List[Tuple[str, any, any, Dict]],
        batch_size: int = 16,
        seq_len: int = 512,
    ) -> Dict:
        """Compare model speed.

        Args:
            models: List of (name, model, params, config) tuples
            batch_size: Batch size for benchmarking
            seq_len: Sequence length

        Returns:
            Dictionary with speed comparison
        """
        benchmark = SpeedBenchmarkJAX()
        results = {}

        for name, model, params, config in models:
            print(f"  Benchmarking {name}...")
            results[name] = benchmark.benchmark_forward(
                model, params, config,
                batch_size=batch_size,
                seq_len=seq_len
            )

        return results


def save_generation_results(results: List[Dict], output_path: str):
    """Save generation results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def print_generation_summary(results: List[Dict], model_name: str = "Model"):
    """Print generation summary."""
    if not results:
        print(f"  {model_name}: No generation results")
        return

    total_tokens = sum(r.get('new_tokens', 0) for r in results)
    total_time = sum(r.get('time_ms', 0) for r in results)
    avg_speed = total_tokens / (total_time / 1000) if total_time > 0 else 0

    print(f"  {model_name}: {len(results)} prompts, {total_tokens} tokens, avg {avg_speed:.1f} tok/s")

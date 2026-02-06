"""
Performance Analysis Module - JAX Version
==========================================
Text generation, speed benchmarking, and model comparison utilities.

JAX/Flax compatible version for TPU analysis.

Provides:
- TextGeneratorJAX: Text generation with JAX models
- SpeedBenchmarkJAX: Inference speed measurement
- ModelComparatorJAX: Side-by-side model comparison

NOTE: This is a JAX port of performance.py (493 lines).
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from .utils_jax import create_model_from_config, evaluate_jax, load_val_data_jax


# Standard prompt categories for generation
GENERATION_PROMPTS = {
    'factual': [
        "The capital of France is",
        "The largest ocean on Earth is",
        "Einstein developed the theory of",
    ],
    'common_sense': [
        "If you drop a glass, it will",
        "Fire is hot, ice is",
        "At night, the sky is",
    ],
    'narrative': [
        "Once upon a time, there was a",
        "She walked into the room and",
    ],
    'technical': [
        "def fibonacci(n):",
        "The mitochondria is the",
    ],
    'conversational': [
        "I think the best way to",
        "In my opinion,",
    ],
}

# Paper-quality prompts for comparison
PAPER_COMPARISON_PROMPTS = {
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


class TextGeneratorJAX:
    """Text generation utilities for JAX models."""

    def __init__(self, tokenizer=None, vocab_size: int = 30522):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_token_id = 102  # BERT [SEP]

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
            model: JAX/Flax model class (unused, uses config to create instance)
            params: Model parameters
            config: Model configuration
            prompt_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            Tuple of (generated_ids, elapsed_ms)
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
            if next_token == self.eos_token_id:
                break

        elapsed_ms = (time.time() - start_time) * 1000

        return np.array(generated), elapsed_ms

    def generate_simple(
        self,
        model,
        params,
        config: Dict,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """Generate text from a prompt (simple, no streaming).

        Args:
            model: JAX/Flax model class
            params: Model parameters
            config: Model configuration
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter

        Returns:
            Full generated text including prompt
        """
        if self.tokenizer is None:
            return prompt

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = np.array(input_ids)

        generated_ids, _ = self.generate(
            model, params, config, prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_from_text(
        self,
        model,
        params,
        config: Dict,
        prompt: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text from string prompt with metadata.

        Args:
            model: JAX/Flax model class
            params: Model parameters
            config: Model configuration
            prompt: Input text prompt
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

    def generate_samples(
        self,
        model,
        params,
        config: Dict,
        model_name: str = "model",
        prompts: Dict[str, List[str]] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        streaming: bool = True
    ) -> List[Dict]:
        """Generate text samples with top-k sampling.

        Args:
            model: JAX/Flax model class
            params: Model parameters
            config: Model configuration
            model_name: Name for display
            prompts: Dict of category -> list of prompts
            max_new_tokens: Maximum tokens per sample
            temperature: Sampling temperature
            top_k: Top-k filtering
            streaming: Whether to print tokens as generated

        Returns:
            List of generation results with stats
        """
        if prompts is None:
            prompts = GENERATION_PROMPTS

        if self.tokenizer is None:
            return [{'error': 'No tokenizer'}]

        results = []

        for category, category_prompts in prompts.items():
            for prompt in category_prompts:
                start_time = time.perf_counter()

                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = np.array(input_ids)
                prompt_len = len(prompt_ids)

                if streaming:
                    print(f"  [{model_name}|{category}] {prompt}", end="", flush=True)

                generated_ids, gen_time_ms = self.generate(
                    model, params, config, prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                new_tokens = len(generated_ids) - prompt_len
                elapsed = time.perf_counter() - start_time

                if streaming:
                    continuation = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
                    print(f"{continuation}  ({new_tokens} tok, {elapsed*1000:.0f}ms)", flush=True)

                results.append({
                    'category': category,
                    'prompt': prompt,
                    'generated': generated_text,
                    'new_tokens': new_tokens,
                    'time_ms': elapsed * 1000,
                    'tokens_per_sec': new_tokens / elapsed if elapsed > 0 else 0,
                })

        return results


class SpeedBenchmarkJAX:
    """Speed benchmarking for JAX models."""

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
            model: JAX/Flax model class (unused, uses config)
            params: Model parameters
            config: Model configuration
            batch_size: Batch size
            seq_len: Sequence length
            n_warmup: Number of warmup iterations
            n_runs: Number of benchmark iterations

        Returns:
            Dict with timing statistics
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

    def quick_benchmark(
        self,
        model,
        params,
        config: Dict,
        iterations: int = 20
    ) -> Dict:
        """Quick speed test for comparison purposes."""
        return self.benchmark_forward(model, params, config, batch_size=1, seq_len=512, n_runs=iterations)


class ModelComparatorJAX:
    """Compare multiple JAX models."""

    def __init__(self, val_data_path: str, tokenizer=None, batch_size: int = 32, seq_len: int = 512):
        self.val_data_path = val_data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.generator = TextGeneratorJAX(tokenizer) if tokenizer else None
        self.benchmark = SpeedBenchmarkJAX()

    def compare_performance(
        self,
        models: List[Tuple[str, any, any, Dict]],
        n_batches: int = 100
    ) -> Dict:
        """Compare model performance.

        Args:
            models: List of (name, model_class, params, config) tuples
            n_batches: Number of batches to evaluate

        Returns:
            Dict with performance comparison
        """
        from .utils_jax import count_params_jax, estimate_flops_jax

        val_tokens = load_val_data_jax(self.val_data_path, n_batches * self.batch_size * self.seq_len)
        results = {}

        for name, model, params, config in models:
            print(f"  Evaluating {name}...")

            total_params = count_params_jax(params)
            flops = estimate_flops_jax(config, self.seq_len)

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

    def analyze_comparison_model(
        self,
        model,
        params,
        config: Dict,
        model_name: str,
        n_batches: int = 50
    ) -> Tuple[Dict, Dict, Dict]:
        """Run quick analysis on a model for table generation.

        Args:
            model: Model class (unused, uses config)
            params: Model parameters
            config: Model configuration
            model_name: Name for display
            n_batches: Number of batches for validation

        Returns:
            Tuple of (model_info, validation_results, speed_results)
        """
        from .utils_jax import count_params_jax, estimate_flops_jax

        print(f"    Analyzing: {model_name}")

        # Model info
        total_params = count_params_jax(params)
        flops = estimate_flops_jax(config, seq_len=512)
        model_info = {
            'total': total_params,
            'total_M': total_params / 1e6,
            'flops': flops,
            'flops_G': flops / 1e9,
        }
        print(f"    Parameters: {model_info['total_M']:.2f}M, FLOPs: {model_info['flops_G']:.2f}G")

        # Performance (quick eval)
        print(f"    Running validation...")
        val_tokens = load_val_data_jax(self.val_data_path, max_tokens=n_batches * self.batch_size * self.seq_len)
        model_instance = create_model_from_config(config)
        val_results = evaluate_jax(
            model_instance, params, config,
            val_tokens, batch_size=self.batch_size, seq_len=self.seq_len
        )
        print(f"    PPL: {val_results.get('perplexity', 0):.2f}, Acc: {val_results.get('accuracy', 0):.1f}%")

        # Speed benchmark
        speed_results = self.benchmark.quick_benchmark(model, params, config)
        print(f"    Speed: {speed_results.get('tokens_per_sec', 0)/1000:.1f}K tok/s")

        return model_info, val_results, speed_results

    def generate_comparison_samples(
        self,
        dawn_model,
        dawn_params,
        dawn_config: Dict,
        baseline_model,
        baseline_params,
        baseline_config: Dict,
        dawn_name: str,
        baseline_name: str,
        output_path: Path,
        prompts: Dict[str, List[str]] = None
    ) -> str:
        """Generate text samples comparing DAWN vs Baseline.

        Args:
            dawn_model: DAWN model class
            dawn_params: DAWN model parameters
            dawn_config: DAWN model configuration
            baseline_model: Baseline model class
            baseline_params: Baseline model parameters
            baseline_config: Baseline model configuration
            dawn_name: DAWN model name for display
            baseline_name: Baseline model name for display
            output_path: Path to save comparison file
            prompts: Optional custom prompts

        Returns:
            Path to saved comparison file
        """
        if self.generator is None:
            return str(output_path)

        if prompts is None:
            prompts = PAPER_COMPARISON_PROMPTS

        lines = []
        lines.append("=" * 100)
        lines.append("GENERATION COMPARISON: DAWN vs Baseline (JAX)")
        lines.append("=" * 100)
        lines.append(f"DAWN Model:     {dawn_name}")
        lines.append(f"Baseline Model: {baseline_name}")
        lines.append("=" * 100)

        print("\n" + "\n".join(lines[-4:]))

        for category, category_prompts in prompts.items():
            cat_header = f"\n{'='*100}\n[{category.upper()}]\n{'='*100}"
            lines.append(cat_header)
            print(cat_header)

            for prompt in category_prompts:
                prompt_line = f"\n{'─'*100}\nPrompt: \"{prompt}\"\n{'─'*100}"
                lines.append(prompt_line)
                print(prompt_line)

                dawn_output = self.generator.generate_simple(
                    dawn_model, dawn_params, dawn_config, prompt
                )
                baseline_output = self.generator.generate_simple(
                    baseline_model, baseline_params, baseline_config, prompt
                )

                dawn_gen = dawn_output[len(prompt):].strip() if dawn_output.startswith(prompt) else dawn_output
                baseline_gen = baseline_output[len(prompt):].strip() if baseline_output.startswith(prompt) else baseline_output

                dawn_line = f"  DAWN:     {dawn_gen}"
                baseline_line = f"  Baseline: {baseline_gen}"

                lines.append(dawn_line)
                lines.append(baseline_line)
                print(dawn_line)
                print(baseline_line)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"\n    Saved to: {output_path}")
        return str(output_path)


def save_generation_results(results: List[Dict], output_path: str):
    """Save generation results to file.

    Args:
        results: List of generation result dicts
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def print_generation_summary(results: List[Dict], model_name: str = "Model"):
    """Print generation summary.

    Args:
        results: List of generation result dicts
        model_name: Model name for display
    """
    if not results:
        print(f"  {model_name}: No generation results")
        return

    total_tokens = sum(r.get('new_tokens', 0) for r in results)
    total_time = sum(r.get('time_ms', 0) for r in results)
    avg_speed = total_tokens / (total_time / 1000) if total_time > 0 else 0

    print(f"  {model_name}: {len(results)} prompts, {total_tokens} tokens, avg {avg_speed:.1f} tok/s")

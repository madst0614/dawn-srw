"""
Performance Analysis Module
===========================
Text generation, speed benchmarking, and model comparison utilities.

Provides:
- TextGenerator: Text generation with streaming output
- SpeedBenchmark: Inference speed measurement
- ModelComparator: Side-by-side model comparison
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F


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


class TextGenerator:
    """Text generation utilities with streaming support."""

    def __init__(self, tokenizer, device: str = 'cuda'):
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = tokenizer.sep_token_id

    def generate_simple(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """Generate text from a prompt (simple, no streaming).

        Args:
            model: The language model
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter

        Returns:
            Full generated text including prompt
        """
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors='pt'
        ).to(self.device)
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = model(generated, attention_mask=None)
                logits = output[0] if isinstance(output, tuple) else output
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == self.eos_token_id:
                    break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def generate_samples(
        self,
        model,
        model_name: str = "model",
        prompts: Dict[str, List[str]] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        streaming: bool = True
    ) -> List[Dict]:
        """Generate text samples with top-k sampling.

        Args:
            model: The language model
            model_name: Name for display
            prompts: Dict of category -> list of prompts (defaults to GENERATION_PROMPTS)
            max_new_tokens: Maximum tokens per sample
            temperature: Sampling temperature
            top_k: Top-k filtering
            streaming: Whether to print tokens as generated

        Returns:
            List of generation results with stats
        """
        if prompts is None:
            prompts = GENERATION_PROMPTS

        results = []
        model.eval()

        for category, category_prompts in prompts.items():
            for prompt in category_prompts:
                start_time = time.perf_counter()

                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors='pt'
                ).to(self.device)
                generated = input_ids.clone()
                prompt_len = input_ids.shape[1]

                if streaming:
                    print(f"  [{model_name}|{category}] {prompt}", end="", flush=True)

                prev_full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        output = model(generated, attention_mask=None)
                        logits = output[0] if isinstance(output, tuple) else output
                        next_token_logits = logits[:, -1, :] / temperature

                        if top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(
                                next_token_logits, top_k
                            )[0][..., -1, None]
                            next_token_logits[indices_to_remove] = float('-inf')

                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token], dim=1)

                        if streaming:
                            curr_full_text = self.tokenizer.decode(
                                generated[0], skip_special_tokens=True
                            )
                            new_part = curr_full_text[len(prev_full_text):]
                            if new_part:
                                print(new_part, end="", flush=True)
                            prev_full_text = curr_full_text

                        if next_token.item() == self.eos_token_id:
                            break

                elapsed = time.perf_counter() - start_time
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                new_tokens = generated.shape[1] - prompt_len

                if streaming:
                    print(f"  ({new_tokens} tok, {elapsed*1000:.0f}ms)", flush=True)

                results.append({
                    'category': category,
                    'prompt': prompt,
                    'generated': generated_text,
                    'new_tokens': new_tokens,
                    'time_ms': elapsed * 1000,
                    'tokens_per_sec': new_tokens / elapsed if elapsed > 0 else 0,
                })

        return results


class SpeedBenchmark:
    """Inference speed benchmarking utilities."""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def benchmark(
        self,
        model,
        seq_len: int = 512,
        batch_size: int = 1,
        warmup: int = 10,
        iterations: int = 50
    ) -> Dict:
        """Benchmark inference speed.

        Args:
            model: The model to benchmark
            seq_len: Sequence length for input
            batch_size: Batch size
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Dict with timing statistics
        """
        model.eval()
        dummy_input = torch.randint(
            0, 1000, (batch_size, seq_len), device=self.device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        tokens_per_sec = (batch_size * seq_len) / avg_time

        return {
            'avg_time_ms': avg_time * 1000,
            'tokens_per_sec': tokens_per_sec,
            'iterations': iterations,
            'batch_size': batch_size,
            'seq_len': seq_len,
        }

    def quick_benchmark(self, model, iterations: int = 20) -> Dict:
        """Quick speed test for comparison purposes."""
        model.eval()
        dummy = torch.randint(0, 1000, (1, 512)).to(self.device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(dummy)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                model(dummy)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        tokens_per_sec = (iterations * 512) / elapsed

        return {'tokens_per_sec': tokens_per_sec}


class ModelComparator:
    """Side-by-side model comparison utilities."""

    def __init__(
        self,
        tokenizer,
        device: str = 'cuda',
        gen_tokens: int = 50
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.gen_tokens = gen_tokens
        self.generator = TextGenerator(tokenizer, device)
        self.benchmark = SpeedBenchmark(device)

    def generate_comparison_samples(
        self,
        dawn_model,
        baseline_model,
        dawn_name: str,
        baseline_name: str,
        output_path: Path,
        prompts: Dict[str, List[str]] = None
    ) -> str:
        """Generate text samples comparing DAWN vs Baseline for paper.

        Args:
            dawn_model: DAWN model
            baseline_model: Baseline/vanilla model
            dawn_name: DAWN model name for display
            baseline_name: Baseline model name for display
            output_path: Path to save comparison file
            prompts: Optional custom prompts (defaults to PAPER_COMPARISON_PROMPTS)

        Returns:
            Path to saved comparison file
        """
        if prompts is None:
            prompts = PAPER_COMPARISON_PROMPTS

        lines = []
        lines.append("=" * 100)
        lines.append("GENERATION COMPARISON: DAWN vs Baseline")
        lines.append("=" * 100)
        lines.append(f"DAWN Model:     {dawn_name}")
        lines.append(f"Baseline Model: {baseline_name}")
        lines.append(f"Max tokens: {self.gen_tokens}")
        lines.append("=" * 100)

        print("\n" + "\n".join(lines[-5:]))

        for category, category_prompts in prompts.items():
            cat_header = f"\n{'='*100}\n[{category.upper()}]\n{'='*100}"
            lines.append(cat_header)
            print(cat_header)

            for prompt in category_prompts:
                prompt_line = f"\n{'─'*100}\nPrompt: \"{prompt}\"\n{'─'*100}"
                lines.append(prompt_line)
                print(prompt_line)

                dawn_output = self.generator.generate_simple(
                    dawn_model, prompt, max_new_tokens=self.gen_tokens
                )
                baseline_output = self.generator.generate_simple(
                    baseline_model, prompt, max_new_tokens=self.gen_tokens
                )

                dawn_gen = dawn_output[len(prompt):].strip() if dawn_output.startswith(prompt) else dawn_output
                baseline_gen = baseline_output[len(prompt):].strip() if baseline_output.startswith(prompt) else baseline_output

                dawn_line = f"  DAWN:     {dawn_gen}"
                baseline_line = f"  Baseline: {baseline_gen}"

                lines.append(dawn_line)
                lines.append(baseline_line)
                print(dawn_line)
                print(baseline_line)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"\n    Saved to: {output_path}")
        return str(output_path)

    def analyze_comparison_model(
        self,
        model,
        model_name: str,
        val_data_path: str,
        n_batches: int = 50
    ) -> Tuple[Dict, Dict, Dict]:
        """Run quick analysis on a model for table generation.

        Args:
            model: Model to analyze
            model_name: Name for display
            val_data_path: Path to validation data
            n_batches: Number of batches for validation

        Returns:
            Tuple of (model_info, validation_results, speed_results)
        """
        from scripts.evaluation.evaluate import evaluate_model, load_val_data, estimate_flops

        print(f"    Analyzing: {model_name}")

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        model_cfg = getattr(model, 'config', None)
        flops = estimate_flops(model, config=model_cfg, seq_len=512)
        model_info = {
            'total': total_params,
            'total_M': total_params / 1e6,
            'flops': flops,
            'flops_G': flops / 1e9,
        }
        print(f"    Parameters: {model_info['total_M']:.2f}M, FLOPs: {model_info['flops_G']:.2f}G")

        # Performance (quick eval)
        print(f"    Running validation...")
        val_tokens = load_val_data(val_data_path, max_tokens=n_batches * 32 * 512)
        val_results = evaluate_model(
            model, val_tokens, batch_size=32, seq_len=512, device=self.device
        )
        print(f"    PPL: {val_results.get('perplexity', 0):.2f}, Acc: {val_results.get('accuracy', 0):.1f}%")

        # Speed benchmark
        speed_results = self.benchmark.quick_benchmark(model)
        print(f"    Speed: {speed_results['tokens_per_sec']/1000:.1f}K tok/s")

        return model_info, val_results, speed_results


def save_generation_results(
    results: Dict[str, List[Dict]],
    output_dir: Path,
    json_filename: str = 'generation_samples.json',
    txt_filename: str = 'generation_samples.txt'
):
    """Save generation results to JSON and human-readable text files.

    Args:
        results: Dict mapping model_name -> list of generation results
        output_dir: Output directory
        json_filename: JSON output filename
        txt_filename: Text output filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save human-readable text
    with open(output_dir / txt_filename, 'w') as f:
        for model_name, samples in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write(f"{'='*60}\n")

            categories = {}
            for s in samples:
                cat = s['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(s)

            for category, cat_samples in categories.items():
                f.write(f"\n[{category.upper()}]\n")
                f.write("-" * 60 + "\n")
                for s in cat_samples:
                    f.write(f"Prompt: {s['prompt']}\n")
                    f.write(f"Generated: {s['generated']}\n")
                    f.write(f"Stats: {s['new_tokens']} tokens, {s['time_ms']:.0f}ms, {s['tokens_per_sec']:.1f} tok/s\n")
                    f.write("-" * 40 + "\n")


def print_generation_summary(samples: List[Dict], model_name: str = "Model"):
    """Print summary statistics for generation samples."""
    total_tokens = sum(s['new_tokens'] for s in samples)
    total_time = sum(s['time_ms'] for s in samples)
    avg_speed = total_tokens / (total_time / 1000) if total_time > 0 else 0

    print(f"  │ {model_name} Summary: {len(samples)} prompts, {total_tokens} tokens, avg {avg_speed:.1f} tok/s")

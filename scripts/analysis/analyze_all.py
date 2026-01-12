#!/usr/bin/env python3
"""
DAWN Complete Analysis Tool
============================
One-touch analysis for DAWN models.

Supports:
- Single checkpoint analysis with full report
- Multi-checkpoint comparison
- Paper-ready figures and tables
- Selective analysis modes
- Configurable batch sizes and analysis parameters

Usage:
    # Single analysis
    python scripts/analysis/analyze_all.py \
        --checkpoint dawn_v18.pt \
        --val_data val.pt \
        --output results/

    # Multi comparison
    python scripts/analysis/analyze_all.py \
        --checkpoints dawn_v18.pt dawn_v17.pt vanilla.pt \
        --val_data val.pt \
        --output results/

    # Paper-only (faster)
    python scripts/analysis/analyze_all.py \
        --checkpoint dawn.pt \
        --val_data val.pt \
        --output results/ \
        --paper-only

    # Specific analyses
    python scripts/analysis/analyze_all.py \
        --checkpoint dawn.pt \
        --val_data val.pt \
        --output results/ \
        --only health,routing,performance

    # Custom batch settings (faster)
    python scripts/analysis/analyze_all.py \
        --checkpoint dawn.pt \
        --val_data val.pt \
        --output results/ \
        --n_batches 50 --val_batches 100

    # Large-scale analysis
    python scripts/analysis/analyze_all.py \
        --checkpoint dawn.pt \
        --val_data val.pt \
        --output results/ \
        --n_batches 200 --max_sentences 5000 --batch_size 32

CLI Arguments:
    Input/Output:
        --checkpoint      Single checkpoint path
        --checkpoints     Multiple checkpoint paths
        --val_data        Validation data path (required)
        --output          Output directory (default: analysis_results)
        --device          Device: cuda/cpu (default: cuda, auto-fallback to cpu)

    Analysis Mode:
        --paper-only      Generate paper outputs only (faster)
        --only            Run only specific analyses (comma-separated)
                          Options: health,routing,embedding,semantic,pos,
                                   factual,behavioral,coselection,weight,v18

    Analysis Parameters:
        --n_batches       Batches for routing/semantic/behavioral/coselection (default: 100)
        --val_batches     Batches for validation performance (default: 200)
        --max_sentences   Max sentences for POS analysis (default: 2000)
        --n_runs          Runs for factual analysis (default: 10)
        --batch_size      Dataloader batch size (default: 16)
        --max_samples     Max samples for dataloader (default: 5000)
        --n_clusters      Clusters for embedding analysis (default: 5)
        --pool_type       Neuron pool type: fv, fqk, rv, rqk (default: fv)
        --target_layer    Target layer for POS analysis (default: all layers)
        --gen_tokens      Max tokens to generate per sample (default: 50)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class ModelAnalyzer:
    """Single model analyzer."""

    def __init__(
        self,
        checkpoint_path: str,
        val_data_path: str,
        output_dir: str,
        device: str = 'cuda',
        # Analysis parameters
        n_batches: int = 100,
        val_batches: int = 200,
        max_sentences: int = 2000,
        n_runs: int = 10,
        batch_size: int = 16,
        max_samples: int = 5000,
        n_clusters: int = 5,
        pool_type: str = 'fv',
        gen_tokens: int = 50,
        target_layer: int = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.val_data_path = val_data_path
        self.output_dir = Path(output_dir)
        self.device = device

        # Analysis parameters
        self.n_batches = n_batches
        self.val_batches = val_batches
        self.max_sentences = max_sentences
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.n_clusters = n_clusters
        self.pool_type = pool_type
        self.gen_tokens = gen_tokens
        self.target_layer = target_layer

        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_type = None  # 'dawn' or 'vanilla'
        self.version = None
        self.name = None

        self.results = {}
        self._dataloader = None

    def load_model(self):
        """Load model with auto version detection."""
        from scripts.analysis.utils import load_model

        self.model, self.tokenizer, self.config = load_model(self.checkpoint_path, self.device)

        # Detect model type
        if hasattr(self.model, 'router') or hasattr(self.model, 'shared_neurons'):
            self.model_type = 'dawn'
            self.version = self.config.get('model_version', 'unknown')
        else:
            self.model_type = 'vanilla'
            self.version = 'vanilla'

        # Extract name from path
        path = Path(self.checkpoint_path)
        if path.is_dir():
            self.name = path.name
        else:
            self.name = path.parent.name if path.parent.name not in ['checkpoints', '.'] else path.stem

        print(f"Loaded: {self.name} ({self.model_type}, v{self.version})")

    def _get_dataloader(self):
        """Get or create dataloader."""
        if self._dataloader is None:
            from scripts.analysis.utils import create_dataloader
            self._dataloader = create_dataloader(
                self.val_data_path, self.tokenizer,
                batch_size=self.batch_size, max_samples=self.max_samples
            )
        return self._dataloader

    def analyze_model_info(self) -> Dict:
        """Analyze model parameters, architecture."""
        output_dir = self.output_dir / 'model_info'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate FLOPs
        from scripts.evaluation.evaluate import estimate_flops
        flops = estimate_flops(self.model, seq_len=512)

        params_info = {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1e6,
            'flops': flops,
            'flops_G': flops / 1e9,
        }

        with open(output_dir / 'parameters.json', 'w') as f:
            json.dump(params_info, f, indent=2)

        # Architecture summary
        arch_lines = [
            f"Model: {self.model_type} v{self.version}",
            f"Name: {self.name}",
            f"",
            f"Architecture:",
            f"  d_model: {self.config.get('d_model', 'N/A')}",
            f"  n_layers: {self.config.get('n_layers', 'N/A')}",
            f"  n_heads: {self.config.get('n_heads', 'N/A')}",
            f"  vocab_size: {self.config.get('vocab_size', 'N/A')}",
            f"  max_seq_len: {self.config.get('max_seq_len', 'N/A')}",
            f"",
            f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)",
            f"FLOPs: {flops:,} ({flops/1e9:.2f}G)",
        ]

        if self.model_type == 'dawn':
            arch_lines.extend([
                f"",
                f"DAWN Configuration:",
                f"  rank: {self.config.get('rank', 'N/A')}",
                f"  knowledge_rank: {self.config.get('knowledge_rank', 'N/A')}",
                f"  d_space: {self.config.get('d_space', 'N/A')}",
                f"",
                f"Neuron Counts:",
                f"  n_feature_qk: {self.config.get('n_feature_qk', 'N/A')}",
                f"  n_feature_v: {self.config.get('n_feature_v', 'N/A')}",
                f"  n_restore_qk: {self.config.get('n_restore_qk', 'N/A')}",
                f"  n_restore_v: {self.config.get('n_restore_v', 'N/A')}",
                f"  n_feature_know: {self.config.get('n_feature_know', 'N/A')}",
                f"  n_restore_know: {self.config.get('n_restore_know', 'N/A')}",
            ])

        with open(output_dir / 'architecture.txt', 'w') as f:
            f.write('\n'.join(arch_lines))

        self.results['model_info'] = params_info
        return params_info

    def analyze_performance(self, n_batches: int = 200) -> Dict:
        """Analyze validation performance."""
        from scripts.evaluation.evaluate import evaluate_model, load_val_data

        output_dir = self.output_dir / 'performance'
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Load validation tokens
        print("  [1/3] Loading validation data...")
        val_tokens = load_val_data(self.val_data_path, max_tokens=n_batches * 32 * 512)

        # Validation metrics
        print("  [2/3] Running validation...")
        val_results = evaluate_model(
            self.model, val_tokens,
            batch_size=32, seq_len=512, device=self.device
        )
        results['validation'] = val_results

        # Print validation results
        print(f"\n  ┌─ Validation Results ─────────────────")
        print(f"  │ Loss:       {val_results.get('loss', 0):.4f}")
        print(f"  │ Perplexity: {val_results.get('perplexity', 0):.2f}")
        print(f"  │ Accuracy:   {val_results.get('accuracy', 0):.2f}%")
        print(f"  └───────────────────────────────────────\n")

        with open(output_dir / 'validation.json', 'w') as f:
            json.dump(val_results, f, indent=2)

        # Speed benchmark
        print("  [3/3] Running speed benchmark...")
        speed_results = self._benchmark_speed()
        results['speed'] = speed_results

        print(f"  ┌─ Speed Results ─────────────────────")
        print(f"  │ Avg Time:   {speed_results.get('avg_time_ms', 0):.2f} ms/batch")
        print(f"  │ Throughput: {speed_results.get('tokens_per_sec', 0)/1000:.1f}K tokens/sec")
        print(f"  └───────────────────────────────────────\n")

        with open(output_dir / 'speed_benchmark.json', 'w') as f:
            json.dump(speed_results, f, indent=2)

        # Generation samples (streaming output)
        print(f"\n  ┌─ Generation Samples (max {self.gen_tokens} tokens) ────────────────────────")
        samples = self._generate_samples(max_new_tokens=self.gen_tokens)
        results['generation'] = samples

        # Group by category for file output
        categories = {}
        for s in samples:
            cat = s['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(s)

        # Summary stats
        total_tokens = sum(s['new_tokens'] for s in samples)
        total_time = sum(s['time_ms'] for s in samples)
        avg_speed = total_tokens / (total_time / 1000) if total_time > 0 else 0
        print(f"  ├─────────────────────────────────────────────────────────────────────────────")
        print(f"  │ Summary: {len(samples)} prompts, {total_tokens} tokens, avg {avg_speed:.1f} tok/s")
        print(f"  └─────────────────────────────────────────────────────────────────────────────\n")

        with open(output_dir / 'generation_samples.json', 'w') as f:
            json.dump(samples, f, indent=2, default=str)

        with open(output_dir / 'generation_samples.txt', 'w') as f:
            for category, cat_samples in categories.items():
                f.write(f"\n[{category.upper()}]\n")
                f.write("=" * 60 + "\n")
                for s in cat_samples:
                    f.write(f"Prompt: {s['prompt']}\n")
                    f.write(f"Generated: {s['generated']}\n")
                    f.write(f"Stats: {s['new_tokens']} tokens, {s['time_ms']:.0f}ms, {s['tokens_per_sec']:.1f} tok/s\n")
                    f.write("-" * 60 + "\n")

        self.results['performance'] = results
        return results

    def _benchmark_speed(self, warmup: int = 10, iterations: int = 50) -> Dict:
        """Benchmark inference speed."""
        import time

        self.model.eval()
        seq_len = 512
        batch_size = 1

        # Create dummy input
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)

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

    def _generate_samples(self, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 50) -> List[Dict]:
        """Generate text samples with top-k sampling and streaming output."""
        import time
        import torch.nn.functional as F

        # Comprehensive prompt categories
        prompt_categories = {
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

        results = []
        self.model.eval()
        eos_token_id = self.tokenizer.sep_token_id

        for category, prompts in prompt_categories.items():
            for prompt in prompts:
                start_time = time.perf_counter()

                # Encode without special tokens for cleaner generation
                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors='pt'
                ).to(self.device)
                generated = input_ids.clone()
                prompt_len = input_ids.shape[1]

                # Print prompt with streaming indicator
                print(f"  [{category}] {prompt}", end="", flush=True)

                # Track full decoded text for proper spacing (BERT tokenizer needs full context)
                prev_full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        output = self.model(generated, attention_mask=None)
                        logits = output[0] if isinstance(output, tuple) else output
                        next_token_logits = logits[:, -1, :] / temperature

                        # Top-k filtering
                        if top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = float('-inf')

                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token], dim=1)

                        # Decode FULL sequence to preserve spacing (tokenizer needs full context)
                        curr_full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        new_part = curr_full_text[len(prev_full_text):]
                        if new_part:
                            print(new_part, end="", flush=True)
                        prev_full_text = curr_full_text

                        # Stop if EOS token generated
                        if next_token.item() == eos_token_id:
                            break

                elapsed = time.perf_counter() - start_time
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                new_tokens = generated.shape[1] - prompt_len

                # End line with stats
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

    def analyze_health(self) -> Dict:
        """Analyze neuron health (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.neuron_health import NeuronHealthAnalyzer

        output_dir = self.output_dir / 'health'
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Analyzing neuron health...")
        analyzer = NeuronHealthAnalyzer(self.model, device=self.device)
        results = analyzer.run_all(str(output_dir))

        # Print detailed summary
        ema = results.get('ema_distribution', {})
        diversity = results.get('diversity', {})
        qk_overlap = results.get('qk_ema_overlap', {})

        if ema:
            print(f"\n  ┌─ Neuron Health Summary ───────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Active':>8} {'Dead':>8} {'Total':>8} {'Ratio':>8} {'Gini':>8} {'EMA Mean':>10} {'EMA Std':>10}")
            print(f"  │ {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

            total_active = 0
            total_neurons = 0
            for name, data in ema.items():
                if isinstance(data, dict) and 'total' in data:
                    total_active += data.get('active', 0)
                    total_neurons += data.get('total', 0)
                    stats = data.get('stats', {})
                    print(f"  │ {data.get('display', name):<12} {data['active']:>8d} {data['dead']:>8d} "
                          f"{data['total']:>8d} {data['active_ratio']*100:>7.1f}% {data.get('gini', 0):>8.3f} "
                          f"{stats.get('mean', 0):>10.4f} {stats.get('std', 0):>10.4f}")

            if total_neurons > 0:
                print(f"  │ {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
                print(f"  │ {'TOTAL':<12} {total_active:>8d} {total_neurons - total_active:>8d} "
                      f"{total_neurons:>8d} {total_active/total_neurons*100:>7.1f}%")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Diversity metrics
        if diversity:
            print(f"\n  ┌─ Diversity Metrics ────────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Entropy':>10} {'Norm. Ent':>12} {'Eff. Count':>14} {'Coverage':>12}")
            print(f"  │ {'─'*12} {'─'*10} {'─'*12} {'─'*14} {'─'*12}")
            for name, data in diversity.items():
                if isinstance(data, dict) and 'entropy' in data:
                    print(f"  │ {data.get('display', name):<12} {data['entropy']:>10.3f} {data.get('normalized_entropy', 0):>12.3f} "
                          f"{data.get('effective_count', 0):>14.1f} {data.get('coverage', 0)*100:>11.1f}%")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Q/K overlap for v18.x
        if qk_overlap and 'error' not in qk_overlap:
            print(f"\n  ┌─ Q/K EMA Overlap (v18.x) ──────────────────────────────────────────────")
            for pool_name, overlap_data in qk_overlap.items():
                if isinstance(overlap_data, dict) and 'q_only' in overlap_data:
                    print(f"  │ {pool_name}:")
                    print(f"  │   Shared: {overlap_data.get('shared', 0):>6d}  |  Q-only: {overlap_data.get('q_only', 0):>6d}  |  K-only: {overlap_data.get('k_only', 0):>6d}  |  Dead: {overlap_data.get('dead', 0):>6d}")
                    print(f"  │   Corr: {overlap_data.get('correlation', 0):.3f}  |  Jaccard: {overlap_data.get('jaccard', 0):.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['health'] = results
        return results

    def analyze_routing(self, n_batches: int = 100) -> Dict:
        """Analyze routing patterns (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.routing import RoutingAnalyzer

        output_dir = self.output_dir / 'routing'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing routing patterns ({n_batches} batches)...")
        dataloader = self._get_dataloader()
        analyzer = RoutingAnalyzer(self.model, device=self.device)
        results = analyzer.run_all(dataloader, str(output_dir), n_batches)

        # Print detailed summary
        entropy = results.get('entropy', {})
        selection_freq = results.get('selection_frequency', {})
        selection_div = results.get('selection_diversity', {})
        qk_overlap = results.get('qk_overlap', {})
        qk_entropy = results.get('qk_entropy', {})

        if entropy:
            print(f"\n  ┌─ Routing Entropy ─────────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print(f"  │ {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
            for pool, data in entropy.items():
                if isinstance(data, dict) and 'mean_entropy' in data:
                    print(f"  │ {data.get('display', pool):<12} {data['mean_entropy']:>10.3f} {data.get('std_entropy', 0):>10.3f} "
                          f"{data.get('min_entropy', 0):>10.3f} {data.get('max_entropy', 0):>10.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if selection_freq:
            print(f"\n  ┌─ Selection Frequency ─────────────────────────────────────────────────")
            for pool, data in selection_freq.items():
                if isinstance(data, dict) and 'total_selections' in data:
                    print(f"  │ {data.get('display', pool)}:")
                    print(f"  │   Total selections: {data['total_selections']:,}")
                    print(f"  │   Unique neurons: {data.get('unique_selected', 0):,}")
                    print(f"  │   Coverage: {data.get('coverage', 0)*100:.1f}%")
                    if data.get('top10'):
                        top_5 = data['top10'][:5]
                        print(f"  │   Top 5: {', '.join(f'N{n[0]}({n[1]})' for n in top_5)}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if selection_div:
            print(f"\n  ┌─ Selection Diversity ─────────────────────────────────────────────────")
            print(f"  │ {'Key':<20} {'Union':>8} {'Total':>8} {'Coverage':>10} {'Ratio':>8}")
            print(f"  │ {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
            for key, data in selection_div.items():
                if isinstance(data, dict) and 'union_count' in data:
                    print(f"  │ {data.get('display', key):<20} {data.get('union_count', 0):>8d} "
                          f"{data.get('n_total', 0):>8d} {data.get('union_coverage', 0)*100:>9.1f}% "
                          f"{data.get('diversity_ratio', 0):>8.2f}")
            # Summary
            summary = selection_div.get('summary', {})
            if summary:
                print(f"  │")
                print(f"  │ Processed: {summary.get('n_batches_processed', 0)} batches, {summary.get('n_layers', 0)} layers")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if qk_overlap:
            print(f"\n  ┌─ Q/K Selection Overlap ────────────────────────────────────────────────")
            for pool, data in qk_overlap.items():
                if isinstance(data, dict) and 'overlap_ratio' in data:
                    print(f"  │ {pool}: overlap={data['overlap_ratio']*100:.1f}%, jaccard={data.get('jaccard', 0):.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if qk_entropy:
            print(f"\n  ┌─ Q/K Entropy Comparison ───────────────────────────────────────────────")
            for pool, data in qk_entropy.items():
                if isinstance(data, dict) and 'q_entropy_mean' in data:
                    q_ent = data['q_entropy_mean']
                    k_ent = data['k_entropy_mean']
                    # Skip pools with no data (both 0)
                    if q_ent == 0 and k_ent == 0:
                        print(f"  │ {pool}: N/A (v18.5 context-based routing)")
                    else:
                        print(f"  │ {pool}: Q={q_ent:.1f}%, K={k_ent:.1f}%, diff={data.get('entropy_diff', 0):.1f}%")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['routing'] = results
        return results

    def analyze_embedding(self, n_clusters: int = 5) -> Dict:
        """Analyze embeddings (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.embedding import EmbeddingAnalyzer

        output_dir = self.output_dir / 'embedding'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing neuron embeddings ({n_clusters} clusters)...")
        analyzer = EmbeddingAnalyzer(self.model, device=self.device)
        results = analyzer.run_all(str(output_dir), n_clusters=n_clusters)

        # Print detailed summary
        sim = results.get('similarity', {})
        cross_sim = results.get('cross_type_similarity', {})
        clustering = results.get('clustering', {})

        if sim:
            print(f"\n  ┌─ Embedding Similarity ─────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print(f"  │ {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
            for pool, data in sim.items():
                if isinstance(data, dict) and 'avg_similarity' in data:
                    print(f"  │ {data.get('display', pool):<12} {data['avg_similarity']:>10.4f} {data.get('std_similarity', 0):>10.4f} "
                          f"{data.get('min_similarity', 0):>10.4f} {data.get('max_similarity', 0):>10.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if cross_sim:
            print(f"\n  ┌─ Cross-Type Similarity ────────────────────────────────────────────────")
            for pair, value in cross_sim.items():
                if isinstance(value, (int, float)):
                    print(f"  │ {pair}: {value:.4f}")
                elif isinstance(value, dict) and 'mean' in value:
                    print(f"  │ {pair}: {value['mean']:.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if clustering:
            print(f"\n  ┌─ Clustering Results ───────────────────────────────────────────────────")
            for pool, data in clustering.items():
                if isinstance(data, dict) and 'n_clusters' in data:
                    print(f"  │ {data.get('display', pool)}:")
                    print(f"  │   Clusters: {data['n_clusters']}")
                    if data.get('clusters'):
                        sizes = [c.get('size', 0) for c in data['clusters']]
                        print(f"  │   Cluster sizes: {sizes}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['embedding'] = results
        return results

    def analyze_semantic(self, n_batches: int = 50) -> Dict:
        """Analyze semantic properties (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.semantic import SemanticAnalyzer

        output_dir = self.output_dir / 'semantic'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing semantic patterns ({n_batches} batches)...")
        dataloader = self._get_dataloader()
        analyzer = SemanticAnalyzer(self.model, tokenizer=self.tokenizer, device=self.device)
        results = analyzer.run_all(dataloader, str(output_dir), max_batches=n_batches)

        # Print detailed summary
        path_sim = results.get('path_similarity', {})
        context_routing = results.get('context_routing', {})
        pos_routing = results.get('pos_routing', {})

        if path_sim:
            print(f"\n  ┌─ Semantic Path Similarity ─────────────────────────────────────────────")
            # Similar/different pairs summary
            sim_pairs = path_sim.get('similar_pairs', {})
            diff_pairs = path_sim.get('different_pairs', {})
            if sim_pairs or diff_pairs:
                print(f"  │ Similar pairs: cosine={sim_pairs.get('cosine_mean', 0):.4f}, n={sim_pairs.get('count', 0)}")
                print(f"  │ Different pairs: cosine={diff_pairs.get('cosine_mean', 0):.4f}, n={diff_pairs.get('count', 0)}")
                gap = sim_pairs.get('cosine_mean', 0) - diff_pairs.get('cosine_mean', 0)
                print(f"  │ Similarity gap: {gap:.4f}")
            # Interpretation
            interp = path_sim.get('interpretation', {})
            if interp:
                print(f"  │ Verdict: {interp.get('verdict', 'N/A')}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if context_routing:
            print(f"\n  ┌─ Context-Dependent Routing ────────────────────────────────────────────")
            for word, data in context_routing.items():
                if word == 'summary':
                    continue
                if isinstance(data, dict) and 'n_contexts' in data:
                    n_contexts = data['n_contexts']
                    avg_var = data.get('avg_variance', 0)
                    print(f"  │ '{word}': {n_contexts} contexts, variance={avg_var:.4f}")
            # Summary
            summary = context_routing.get('summary', {})
            if summary:
                print(f"  │ Overall context variance: {summary.get('overall_context_variance', 0):.4f}")
                print(f"  │ More context-sensitive: {summary.get('more_context_sensitive', 'N/A')}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if pos_routing:
            print(f"\n  ┌─ POS Routing Patterns ─────────────────────────────────────────────────")
            routing_by_pos = pos_routing.get('routing_by_pos', {})
            # Sort by count and show top 10
            sorted_pos = sorted(routing_by_pos.items(), key=lambda x: x[1].get('count', 0), reverse=True)[:10]
            for pos, data in sorted_pos:
                if isinstance(data, dict) and 'mean_activation' in data:
                    print(f"  │ {pos}: mean_act={data['mean_activation']:.4f}, count={data.get('count', 0)}")
            if len(routing_by_pos) > 10:
                print(f"  │ ... and {len(routing_by_pos) - 10} more POS tags")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['semantic'] = results
        return results

    def analyze_pos(self, max_sentences: int = 2000, pool_type: str = 'fv', target_layer: int = None) -> Dict:
        """Analyze POS neuron specialization (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import POSNeuronAnalyzer

        output_dir = self.output_dir / 'pos'
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_str = f"layer={target_layer}" if target_layer is not None else "all layers"
        print(f"  Analyzing POS neuron specialization ({max_sentences} sentences, pool={pool_type}, {layer_str})...")
        analyzer = POSNeuronAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device,
            target_layer=target_layer
        )
        results = analyzer.run_all(str(output_dir), pool_type=pool_type, max_sentences=max_sentences)

        # Print summary
        specificity = results.get('neuron_specificity', {})
        if specificity:
            top_specific = list(specificity.items())[:5]
            print(f"\n  ┌─ Top POS-Specific Neurons ───────────")
            for neuron_id, data in top_specific:
                print(f"  │ N{neuron_id}: {data['top_pos']} (specificity={data['specificity']:.3f})")
            print(f"  └───────────────────────────────────────\n")

        self.results['pos'] = results
        return results

    def analyze_factual(self, n_runs: int = 10, pool_type: str = 'fv') -> Dict:
        """Analyze factual knowledge neurons (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.behavioral import BehavioralAnalyzer

        output_dir = self.output_dir / 'factual'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing factual neurons ({n_runs} runs per prompt, pool={pool_type})...")
        analyzer = BehavioralAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device
        )

        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Japan is",
            "The color of the sky is",
        ]
        targets = ["Paris", "Berlin", "Tokyo", "blue"]

        results = analyzer.analyze_factual_neurons(prompts, targets, n_runs=n_runs, pool_type=pool_type)

        # Print detailed summary
        per_target = results.get('per_target', {})
        if per_target:
            print(f"\n  ┌─ Factual Analysis Results ─────────────────────────────────────────────")
            print(f"  │ {'Target':<10} {'Predicted':<12} {'Rank':>6} {'Match%':>8} {'Common':>8}")
            print(f"  │ {'─'*10} {'─'*12} {'─'*6} {'─'*8} {'─'*8}")
            for target, data in per_target.items():
                if isinstance(data, dict):
                    match_rate = data.get('match_rate', 0) * 100
                    predicted = data.get('predicted_token', 'N/A')[:10]
                    rank = data.get('target_rank', -1)
                    rank_str = str(rank) if rank > 0 else '>20'
                    n_common = len(data.get('common_neurons_80', []))
                    print(f"  │ {target:<10} {predicted:<12} {rank_str:>6} {match_rate:>7.0f}% {n_common:>8d}")

            # Interpretation
            any_match = any(d.get('match_rate', 0) > 0 for d in per_target.values())
            if not any_match:
                print(f"  │")
                print(f"  │ Note: Model not generating target tokens (likely undertrained)")
                print(f"  │       Check 'Rank' column - lower is better (1=top prediction)")

            # Show top common neurons if any matches
            all_common = results.get('all_common_neurons', [])
            if all_common:
                print(f"  │")
                print(f"  │ Cross-target common neurons: {len(all_common)}")
                print(f"  │ Top neurons: {', '.join(f'N{n}' for n in all_common[:10])}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        with open(output_dir / 'factual_neurons.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate visualizations
        try:
            from scripts.analysis.visualizers import plot_factual_heatmap, plot_factual_comparison
            plot_factual_heatmap(results, str(output_dir / 'factual_heatmap.png'))
            plot_factual_comparison(results, str(output_dir / 'factual_comparison.png'))
        except Exception as e:
            print(f"    Warning: Could not generate factual plots: {e}")

        self.results['factual'] = results
        return results

    def analyze_behavioral(self, n_batches: int = 50) -> Dict:
        """Analyze behavioral patterns (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.behavioral import BehavioralAnalyzer

        output_dir = self.output_dir / 'behavioral'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing behavioral patterns ({n_batches} batches)...")
        dataloader = self._get_dataloader()
        analyzer = BehavioralAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device
        )
        results = analyzer.run_all(dataloader, str(output_dir), n_batches)

        # Print detailed summary
        trajectory = results.get('trajectory', {})
        probing = results.get('probing', {})

        if trajectory and 'error' not in trajectory:
            print(f"\n  ┌─ Token Trajectory Analysis ────────────────────────────────────────────")
            for pool, data in trajectory.items():
                if isinstance(data, dict) and 'mean_change' in data:
                    print(f"  │ {pool}:")
                    print(f"  │   Mean change: {data['mean_change']:.4f}")
                    print(f"  │   Consistency: {data.get('consistency', 0):.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if probing and 'error' not in probing:
            print(f"\n  ┌─ Probing Analysis ──────────────────────────────────────────────────────")
            for task, data in probing.items():
                if isinstance(data, dict) and 'accuracy' in data:
                    print(f"  │ {task}: accuracy={data['accuracy']*100:.1f}%")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['behavioral'] = results
        return results

    def analyze_coselection(self, n_batches: int = 50) -> Dict:
        """Analyze co-selection patterns (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.coselection import CoselectionAnalyzer

        output_dir = self.output_dir / 'coselection'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing co-selection patterns ({n_batches} batches)...")
        dataloader = self._get_dataloader()
        analyzer = CoselectionAnalyzer(self.model, device=self.device)
        results = analyzer.run_all(dataloader, str(output_dir), 'all', n_batches)

        # Print detailed summary
        print(f"\n  ┌─ Co-Selection Analysis ───────────────────────────────────────────────────")
        for pair_name, data in results.items():
            if isinstance(data, dict) and 'coselection_rate' in data:
                print(f"  │ {pair_name}:")
                print(f"  │   Co-selection rate: {data['coselection_rate']*100:.2f}%")
                print(f"  │   Correlation: {data.get('correlation', 0):.4f}")
                if data.get('top_pairs'):
                    top_3 = data['top_pairs'][:3]
                    print(f"  │   Top pairs: {', '.join(f'({p[0]},{p[1]})' for p in top_3)}")
        print(f"  └─────────────────────────────────────────────────────────────────────────────")

        self.results['coselection'] = results
        return results

    def analyze_weight(self) -> Dict:
        """Analyze weight matrices (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.weight import WeightAnalyzer

        output_dir = self.output_dir / 'weight'
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Analyzing weight matrices...")
        analyzer = WeightAnalyzer(model=self.model, device=self.device)
        results = analyzer.run_all(str(output_dir))

        # Print detailed summary
        svd = results.get('svd', {})
        norms = results.get('norms', {})
        similarity = results.get('similarity', {})

        if norms:
            print(f"\n  ┌─ Weight Norms ─────────────────────────────────────────────────────────")
            for name, data in norms.items():
                if isinstance(data, dict) and 'frobenius' in data:
                    print(f"  │ {name}:")
                    print(f"  │   Frobenius: {data['frobenius']:.4f}, Spectral: {data.get('spectral', 0):.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if svd:
            print(f"\n  ┌─ SVD Analysis ─────────────────────────────────────────────────────────")
            for name, data in svd.items():
                if isinstance(data, dict) and 'rank' in data:
                    print(f"  │ {name}: rank={data['rank']}, condition={data.get('condition_number', 0):.2f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if similarity:
            print(f"\n  ┌─ Weight Similarity ────────────────────────────────────────────────────")
            for pair, data in similarity.items():
                if isinstance(data, (int, float)):
                    print(f"  │ {pair}: {data:.4f}")
                elif isinstance(data, dict) and 'cosine' in data:
                    print(f"  │ {pair}: cosine={data['cosine']:.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['weight'] = results
        return results

    def analyze_v18(self, n_batches: int = 50) -> Dict:
        """Analyze v18.x specific features (DAWN v18.x only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.v18 import V18Analyzer

        # Check if model is v18.x
        analyzer = V18Analyzer(self.model, device=self.device)
        if not analyzer.is_v18:
            print("  Skipping (not v18.x model)")
            return {}

        output_dir = self.output_dir / 'v18'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing v18.x specific features ({n_batches} batches)...")
        dataloader = self._get_dataloader()
        results = analyzer.run_all(dataloader, str(output_dir), n_batches)

        # Print detailed summary
        tau_params = results.get('tau_parameters', {})
        per_layer = results.get('per_layer', {})

        if tau_params:
            print(f"\n  ┌─ Tau Parameters ({tau_params.get('structure', 'v18')}) ─────────────────────────────────────")

            # Bias values
            tau_bias = tau_params.get('tau_bias', {})
            print(f"  │ {'Pool':<16} {'Bias':>10} {'Weight Norm':>14} {'Weight Std':>12}")
            print(f"  │ {'─'*16} {'─'*10} {'─'*14} {'─'*12}")

            for pool in tau_bias.keys():
                bias = tau_bias.get(pool, 0)
                norm = tau_params.get('tau_weight_norm', {}).get(pool, 0)
                std = tau_params.get('tau_weight_std', {}).get(pool, 0)
                print(f"  │ {pool:<16} {bias:>10.4f} {norm:>14.4f} {std:>12.4f}")

            # Q/K differentiation
            qk_diff = tau_params.get('qk_differentiation', {})
            if qk_diff:
                print(f"  │")
                print(f"  │ Q/K Differentiation:")
                for key, val in qk_diff.items():
                    print(f"  │   {key}: {val:.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Per-layer gate summary
        if per_layer:
            gate_data = per_layer.get('gate', {})
            if gate_data:
                print(f"\n  ┌─ Per-Layer Gate Strength ─────────────────────────────────────────────")
                print(f"  │ {'Layer':<8} {'FQ':>8} {'FK':>8} {'FV':>8} {'RQ':>8} {'RK':>8} {'RV':>8}")
                print(f"  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

                for layer, pools in sorted(gate_data.items(), key=lambda x: int(x[0][1:])):
                    row = f"  │ {layer:<8}"
                    for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                        val = pools.get(pool, 0)
                        row += f" {val:>8.3f}" if val else " {:>8}".format('-')
                    print(row)
                print(f"  └─────────────────────────────────────────────────────────────────────────")

            # Q/K overlap
            qk_data = per_layer.get('qk_overlap', {})
            if qk_data:
                print(f"\n  ┌─ Per-Layer Q/K Overlap ───────────────────────────────────────────────")
                print(f"  │ {'Layer':<8} {'FQK':>10} {'RQK':>10}")
                print(f"  │ {'─'*8} {'─'*10} {'─'*10}")

                for layer, data in sorted(qk_data.items(), key=lambda x: int(x[0][1:])):
                    fqk = data.get('fqk', 0)
                    rqk = data.get('rqk', 0)
                    print(f"  │ {layer:<8} {fqk:>10.4f} {rqk:>10.4f}")
                print(f"  └─────────────────────────────────────────────────────────────────────────")

            # Confidence (v18.3+)
            conf_data = per_layer.get('confidence', {})
            has_conf = any(pools for pools in conf_data.values())
            if has_conf:
                print(f"\n  ┌─ Per-Layer Confidence (v18.3+) ───────────────────────────────────────")
                print(f"  │ {'Layer':<8} {'FQ':>8} {'FK':>8} {'FV':>8} {'RQ':>8} {'RK':>8} {'RV':>8}")
                print(f"  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

                for layer, pools in sorted(conf_data.items(), key=lambda x: int(x[0][1:])):
                    if pools:
                        row = f"  │ {layer:<8}"
                        for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                            val = pools.get(pool, 0)
                            row += f" {val:>8.3f}" if val else " {:>8}".format('-')
                        print(row)
                print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['v18'] = results
        return results

    def generate_paper_outputs(self):
        """Generate paper-ready figures and tables."""
        paper_dir = self.output_dir / 'paper'
        figures_dir = paper_dir / 'figures'
        tables_dir = paper_dir / 'tables'

        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type != 'dawn':
            print("  Skipping paper outputs (not a DAWN model)")
            return

        # Generate figures using PaperFigureGenerator
        print("  Generating paper figures...")
        try:
            from scripts.analysis.paper_figures import PaperFigureGenerator

            dataloader = self._get_dataloader()
            gen = PaperFigureGenerator(
                self.checkpoint_path,
                dataloader,
                device=self.device
            )
            gen.generate('3,4,6,7', str(figures_dir), n_batches=50)
        except Exception as e:
            print(f"    Warning: Could not generate paper figures: {e}")

        # Generate tables
        print("  Generating paper tables...")
        self._generate_tables(tables_dir)

        # Summary
        self._generate_paper_summary(paper_dir)

    def _generate_tables(self, tables_dir: Path):
        """Generate LaTeX and CSV tables."""
        model_info = self.results.get('model_info', {})
        perf = self.results.get('performance', {})
        val = perf.get('validation', {})
        speed = perf.get('speed', {})

        # Model stats CSV
        with open(tables_dir / 'model_stats.csv', 'w') as f:
            f.write("metric,value\n")
            f.write(f"parameters,{model_info.get('total', 0)}\n")
            f.write(f"parameters_M,{model_info.get('total_M', 0):.2f}\n")
            f.write(f"flops,{model_info.get('flops', 0)}\n")
            f.write(f"flops_G,{model_info.get('flops_G', 0):.2f}\n")
            f.write(f"ppl,{val.get('perplexity', 0):.2f}\n")
            f.write(f"accuracy,{val.get('accuracy', 0):.2f}\n")
            f.write(f"tokens_per_sec,{speed.get('tokens_per_sec', 0):.0f}\n")

        # Model stats LaTeX
        with open(tables_dir / 'model_stats.tex', 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Statistics}\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\midrule\n")
            f.write(f"Parameters & {model_info.get('total_M', 0):.2f}M \\\\\n")
            f.write(f"FLOPs & {model_info.get('flops_G', 0):.2f}G \\\\\n")
            f.write(f"Perplexity & {val.get('perplexity', 0):.2f} \\\\\n")
            f.write(f"Accuracy & {val.get('accuracy', 0):.1f}\\% \\\\\n")
            f.write(f"Speed & {speed.get('tokens_per_sec', 0)/1000:.1f}K tok/s \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Neuron utilization table
        if 'health' in self.results:
            health = self.results['health']
            ema = health.get('ema_distribution', {})

            with open(tables_dir / 'neuron_utilization.csv', 'w') as f:
                f.write("pool,total,active,dead,active_ratio,gini\n")
                for name, data in ema.items():
                    if isinstance(data, dict) and 'total' in data:
                        f.write(f"{data.get('display', name)},{data['total']},{data['active']},"
                               f"{data['dead']},{data['active_ratio']:.3f},{data.get('gini', 0):.3f}\n")

            with open(tables_dir / 'neuron_utilization.tex', 'w') as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Neuron Utilization}\n")
                f.write("\\begin{tabular}{lrrrr}\n")
                f.write("\\toprule\n")
                f.write("Pool & Total & Active & Dead & Gini \\\\\n")
                f.write("\\midrule\n")
                for name, data in ema.items():
                    if isinstance(data, dict) and 'total' in data:
                        f.write(f"{data.get('display', name)} & {data['total']} & {data['active']} & "
                               f"{data['dead']} & {data.get('gini', 0):.3f} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")

    def _generate_paper_summary(self, paper_dir: Path):
        """Generate paper summary markdown."""
        model_info = self.results.get('model_info', {})
        perf = self.results.get('performance', {})
        val = perf.get('validation', {})
        health = self.results.get('health', {})

        lines = [
            "# Paper Summary",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.name} ({self.model_type} v{self.version})",
            "",
            "## Key Numbers",
            "",
            "### Model",
            f"- Parameters: {model_info.get('total_M', 0):.2f}M",
            f"- FLOPs: {model_info.get('flops_G', 0):.2f}G",
            "",
            "### Performance",
            f"- Validation PPL: {val.get('perplexity', 0):.2f}",
            f"- Validation Accuracy: {val.get('accuracy', 0):.1f}%",
            f"- Validation Loss: {val.get('loss', 0):.4f}",
            "",
        ]

        if health:
            ema = health.get('ema_distribution', {})
            total_active = sum(d.get('active', 0) for d in ema.values() if isinstance(d, dict))
            total_neurons = sum(d.get('total', 0) for d in ema.values() if isinstance(d, dict))

            if total_neurons > 0:
                lines.extend([
                    "### Neuron Health",
                    f"- Active neurons: {total_active}/{total_neurons} ({total_active/total_neurons*100:.1f}%)",
                    f"- Dead neurons: {total_neurons - total_active}",
                    "",
                ])

        lines.extend([
            "## Files",
            "",
            "### Figures",
            "- `figures/fig3_qk_specialization.png`",
            "- `figures/fig4_pos_neurons.png`",
            "- `figures/fig6a_neuron_util.png`",
            "- `figures/fig6b_layer_contrib.png`",
            "- `figures/fig7_factual_heatmap.png`",
            "",
            "### Tables",
            "- `tables/model_stats.csv` / `.tex`",
            "- `tables/neuron_utilization.csv` / `.tex`",
        ])

        with open(paper_dir / 'summary.md', 'w') as f:
            f.write('\n'.join(lines))

    def generate_report(self):
        """Generate unified report for single model."""
        self._generate_markdown_report()

    def _generate_markdown_report(self):
        """Generate markdown report."""
        model_info = self.results.get('model_info', {})
        perf = self.results.get('performance', {})
        val = perf.get('validation', {})
        speed = perf.get('speed', {})
        health = self.results.get('health', {})

        lines = [
            "# DAWN Analysis Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Information",
            "",
            f"- **Name**: {self.name}",
            f"- **Type**: {self.model_type}",
            f"- **Version**: {self.version}",
            f"- **Parameters**: {model_info.get('total_M', 0):.2f}M",
            f"- **FLOPs**: {model_info.get('flops_G', 0):.2f}G",
            "",
            "## Performance",
            "",
            f"- **Validation Loss**: {val.get('loss', 'N/A')}",
            f"- **Perplexity**: {val.get('perplexity', 'N/A')}",
            f"- **Accuracy**: {val.get('accuracy', 'N/A')}%",
            f"- **Speed**: {speed.get('tokens_per_sec', 0)/1000:.1f}K tokens/sec",
            "",
        ]

        if health:
            ema = health.get('ema_distribution', {})
            lines.extend([
                "## Neuron Health",
                "",
                "| Pool | Total | Active | Dead | Active % | Gini |",
                "|------|-------|--------|------|----------|------|",
            ])
            for name, data in ema.items():
                if isinstance(data, dict) and 'total' in data:
                    lines.append(
                        f"| {data.get('display', name)} | {data['total']} | {data['active']} | "
                        f"{data['dead']} | {data['active_ratio']*100:.1f}% | {data.get('gini', 0):.3f} |"
                    )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "```",
            str(self.output_dir),
            "├── model_info/",
            "├── performance/",
        ])

        if self.model_type == 'dawn':
            lines.extend([
                "├── health/",
                "├── routing/",
                "├── embedding/",
                "├── semantic/",
                "├── pos/",
                "├── factual/",
                "├── behavioral/",
                "├── coselection/",
                "├── weight/",
                "├── v18/",
            ])

        lines.extend([
            "├── paper/",
            "└── report.md",
            "```",
        ])

        with open(self.output_dir / 'report.md', 'w') as f:
            f.write('\n'.join(lines))

    def run_all(self, paper_only: bool = False, only: List[str] = None):
        """Run all analyses."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()

        # Define all analyses using instance parameters
        analyses = [
            ('model_info', self.analyze_model_info, {}),
            ('performance', self.analyze_performance, {'n_batches': self.val_batches}),
            ('health', self.analyze_health, {}),
            ('routing', self.analyze_routing, {'n_batches': self.n_batches}),
            ('embedding', self.analyze_embedding, {'n_clusters': self.n_clusters}),
            ('semantic', self.analyze_semantic, {'n_batches': self.n_batches // 2}),
            ('pos', self.analyze_pos, {'max_sentences': self.max_sentences, 'pool_type': self.pool_type, 'target_layer': self.target_layer}),
            ('factual', self.analyze_factual, {'n_runs': self.n_runs, 'pool_type': self.pool_type}),
            ('behavioral', self.analyze_behavioral, {'n_batches': self.n_batches // 2}),
            ('coselection', self.analyze_coselection, {'n_batches': self.n_batches // 2}),
            ('weight', self.analyze_weight, {}),
            ('v18', self.analyze_v18, {'n_batches': self.n_batches // 2}),
        ]

        if paper_only:
            # Reduced params for faster paper-only mode
            analyses = [
                ('model_info', self.analyze_model_info, {}),
                ('performance', self.analyze_performance, {'n_batches': self.val_batches // 2}),
                ('health', self.analyze_health, {}),
                ('routing', self.analyze_routing, {'n_batches': self.n_batches // 2}),
                ('factual', self.analyze_factual, {'n_runs': max(5, self.n_runs // 2), 'pool_type': self.pool_type}),
            ]

        if only:
            analyses = [(n, f, a) for n, f, a in analyses if n in only]

        total_analyses = len(analyses)
        for i, (name, func, kwargs) in enumerate(analyses, 1):
            print(f"\n[{i}/{total_analyses}] {name.upper()}")
            try:
                func(**kwargs)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                self.results[name] = {'error': str(e)}

        # Paper outputs
        print(f"\n[{total_analyses+1}/{total_analyses+2}] PAPER")
        self.generate_paper_outputs()

        # Report
        print(f"\n[{total_analyses+2}/{total_analyses+2}] REPORT")
        self.generate_report()

        # Final summary
        self._print_final_summary()

        return self.results

    def _print_final_summary(self):
        """Print final analysis summary."""
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        # Model info
        model_info = self.results.get('model_info', {})
        print(f"\nModel: {self.name} ({self.model_type} v{self.version})")
        print(f"Parameters: {model_info.get('total_M', 0):.2f}M")

        # Performance
        perf = self.results.get('performance', {})
        val = perf.get('validation', {})
        speed = perf.get('speed', {})
        print(f"\nPerformance:")
        print(f"  Loss: {val.get('loss', 0):.4f}")
        print(f"  PPL:  {val.get('perplexity', 0):.2f}")
        print(f"  Acc:  {val.get('accuracy', 0):.2f}%")
        print(f"  Speed: {speed.get('tokens_per_sec', 0)/1000:.1f}K tok/s")

        # Health summary (DAWN only)
        health = self.results.get('health', {})
        if health:
            ema = health.get('ema_distribution', {})
            total_active = sum(d.get('active', 0) for d in ema.values() if isinstance(d, dict))
            total_neurons = sum(d.get('total', 0) for d in ema.values() if isinstance(d, dict))
            if total_neurons > 0:
                print(f"\nNeuron Health:")
                print(f"  Active: {total_active}/{total_neurons} ({total_active/total_neurons*100:.1f}%)")

        # Errors
        errors = [k for k, v in self.results.items() if isinstance(v, dict) and 'error' in v]
        if errors:
            print(f"\nErrors in: {', '.join(errors)}")

        print(f"\nOutput: {self.output_dir}")
        print("=" * 60)


class MultiModelAnalyzer:
    """Multi-model comparison analyzer."""

    def __init__(
        self,
        checkpoint_paths: List[str],
        val_data_path: str,
        output_dir: str,
        device: str = 'cuda',
        # Analysis parameters
        n_batches: int = 100,
        val_batches: int = 200,
        max_sentences: int = 2000,
        n_runs: int = 10,
        batch_size: int = 16,
        max_samples: int = 5000,
        n_clusters: int = 5,
        pool_type: str = 'fv',
        gen_tokens: int = 50,
        target_layer: int = None,
    ):
        self.checkpoint_paths = checkpoint_paths
        self.val_data_path = val_data_path
        self.output_dir = Path(output_dir)
        self.device = device

        # Analysis parameters
        self.n_batches = n_batches
        self.val_batches = val_batches
        self.max_sentences = max_sentences
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.n_clusters = n_clusters
        self.pool_type = pool_type
        self.gen_tokens = gen_tokens
        self.target_layer = target_layer

        self.analyzers = []
        self.results = {}

    def run_all(self, paper_only: bool = False, only: List[str] = None):
        """Run analysis on all models."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze each model
        for ckpt_path in self.checkpoint_paths:
            path = Path(ckpt_path)
            if path.is_dir():
                name = path.name
            else:
                name = path.parent.name if path.parent.name not in ['checkpoints', '.'] else path.stem

            model_dir = self.output_dir / name

            print(f"\n{'='*60}")
            print(f"Analyzing: {name}")
            print(f"{'='*60}")

            analyzer = ModelAnalyzer(
                ckpt_path, self.val_data_path,
                str(model_dir), self.device,
                n_batches=self.n_batches,
                val_batches=self.val_batches,
                max_sentences=self.max_sentences,
                n_runs=self.n_runs,
                batch_size=self.batch_size,
                max_samples=self.max_samples,
                n_clusters=self.n_clusters,
                pool_type=self.pool_type,
                gen_tokens=self.gen_tokens,
                target_layer=self.target_layer,
            )
            analyzer.run_all(paper_only=paper_only, only=only)

            self.analyzers.append(analyzer)
            self.results[name] = analyzer.results

            # Clear memory
            del analyzer.model
            torch.cuda.empty_cache()

        # Generate comparison
        print(f"\n{'='*60}")
        print("Generating Comparison")
        print(f"{'='*60}")
        self.generate_comparison()

        # Generate reports
        self.generate_report()

    def generate_comparison(self):
        """Generate comparison analysis."""
        comp_dir = self.output_dir / 'comparison'
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary = {}
        for name, results in self.results.items():
            model_info = results.get('model_info', {})
            perf = results.get('performance', {})
            val = perf.get('validation', {})
            speed = perf.get('speed', {})

            summary[name] = {
                'params': model_info.get('total', 0),
                'params_M': model_info.get('total_M', 0),
                'flops': model_info.get('flops', 0),
                'flops_G': model_info.get('flops_G', 0),
                'ppl': val.get('perplexity', 0),
                'accuracy': val.get('accuracy', 0),
                'loss': val.get('loss', 0),
                'tokens_per_sec': speed.get('tokens_per_sec', 0),
            }

        with open(comp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # CSV table
        with open(comp_dir / 'performance_table.csv', 'w') as f:
            f.write("model,params_M,flops_G,loss,ppl,accuracy,tokens_per_sec\n")
            for name, data in summary.items():
                f.write(f"{name},{data['params_M']:.2f},{data['flops_G']:.2f},"
                       f"{data['loss']:.4f},{data['ppl']:.2f},{data['accuracy']:.1f},"
                       f"{data['tokens_per_sec']:.0f}\n")

        # Comparison plots
        self._plot_comparison(summary, comp_dir)

    def _plot_comparison(self, summary: Dict, output_dir: Path):
        """Generate comparison plots."""
        if not HAS_MATPLOTLIB:
            print("  Matplotlib not available, skipping plots")
            return

        import numpy as np

        names = list(summary.keys())
        n_models = len(names)

        # Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # PPL
        ppls = [summary[n]['ppl'] for n in names]
        colors = ['steelblue' if summary[n].get('params_M', 0) > 0 else 'gray' for n in names]
        axes[0].bar(range(n_models), ppls, color=colors)
        axes[0].set_xticks(range(n_models))
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].set_ylabel('Perplexity')
        axes[0].set_title('Validation Perplexity')

        # Params
        params = [summary[n]['params_M'] for n in names]
        axes[1].bar(range(n_models), params, color='coral')
        axes[1].set_xticks(range(n_models))
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_ylabel('Parameters (M)')
        axes[1].set_title('Model Size')

        # Speed
        speeds = [summary[n]['tokens_per_sec']/1000 for n in names]
        axes[2].bar(range(n_models), speeds, color='green')
        axes[2].set_xticks(range(n_models))
        axes[2].set_xticklabels(names, rotation=45, ha='right')
        axes[2].set_ylabel('Tokens/sec (K)')
        axes[2].set_title('Inference Speed')

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Scatter: params vs ppl
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(names):
            ax.scatter(
                summary[name]['params_M'],
                summary[name]['ppl'],
                s=100, label=name
            )
            ax.annotate(name, (summary[name]['params_M'], summary[name]['ppl']),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Perplexity')
        ax.set_title('Parameters vs Perplexity')
        ax.legend(loc='best')
        plt.savefig(output_dir / 'params_ppl_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Generate unified report."""
        self._generate_markdown_report()
        self._generate_paper_comparison()

    def _generate_markdown_report(self):
        """Generate markdown report."""
        lines = [
            "# DAWN Multi-Model Analysis Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Comparison",
            "",
            "| Model | Type | Params | FLOPs | Loss | PPL | Accuracy | Speed |",
            "|-------|------|--------|-------|------|-----|----------|-------|",
        ]

        for analyzer in self.analyzers:
            model_info = analyzer.results.get('model_info', {})
            perf = analyzer.results.get('performance', {})
            val = perf.get('validation', {})
            speed = perf.get('speed', {})

            lines.append(
                f"| {analyzer.name} | {analyzer.model_type} | {model_info.get('total_M', 0):.1f}M | "
                f"{model_info.get('flops_G', 0):.1f}G | {val.get('loss', 0):.4f} | "
                f"{val.get('perplexity', 0):.1f} | {val.get('accuracy', 0):.1f}% | "
                f"{speed.get('tokens_per_sec', 0)/1000:.1f}K |"
            )

        lines.extend([
            "",
            "## Individual Analysis",
            "",
        ])

        for analyzer in self.analyzers:
            lines.extend([
                f"### {analyzer.name}",
                "",
                f"- Type: {analyzer.model_type}",
                f"- Version: {analyzer.version}",
                f"- Output: `{analyzer.output_dir}/`",
                "",
            ])

            if analyzer.model_type == 'dawn':
                health = analyzer.results.get('health', {})
                ema = health.get('ema_distribution', {})

                if ema:
                    lines.append("#### Neuron Health")
                    lines.append("")
                    lines.append("| Pool | Total | Active | Dead | Gini |")
                    lines.append("|------|-------|--------|------|------|")

                    for pool_name, data in ema.items():
                        if isinstance(data, dict) and 'total' in data:
                            lines.append(
                                f"| {data.get('display', pool_name)} | {data['total']} | "
                                f"{data['active']} | {data['dead']} | {data.get('gini', 0):.3f} |"
                            )
                    lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "```",
            str(self.output_dir),
            "├── report.md",
            "├── comparison/",
        ])

        for analyzer in self.analyzers:
            lines.append(f"├── {analyzer.name}/")

        lines.extend([
            "└── paper/",
            "```",
        ])

        with open(self.output_dir / 'report.md', 'w') as f:
            f.write('\n'.join(lines))

    def _generate_paper_comparison(self):
        """Generate paper comparison outputs."""
        paper_dir = self.output_dir / 'paper'
        tables_dir = paper_dir / 'tables'
        figures_dir = paper_dir / 'figures'
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Copy comparison figure
        comp_fig = self.output_dir / 'comparison' / 'performance_comparison.png'
        if comp_fig.exists():
            import shutil
            shutil.copy(comp_fig, figures_dir / 'fig_model_comparison.png')

        # LaTeX comparison table
        with open(tables_dir / 'model_comparison.tex', 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Comparison}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Model & Params & FLOPs & PPL & Acc & Speed \\\\\n")
            f.write("\\midrule\n")

            for analyzer in self.analyzers:
                model_info = analyzer.results.get('model_info', {})
                perf = analyzer.results.get('performance', {})
                val = perf.get('validation', {})
                speed = perf.get('speed', {})

                f.write(
                    f"{analyzer.name} & {model_info.get('total_M', 0):.1f}M & "
                    f"{model_info.get('flops_G', 0):.1f}G & "
                    f"{val.get('perplexity', 0):.1f} & {val.get('accuracy', 0):.1f}\\% & "
                    f"{speed.get('tokens_per_sec', 0)/1000:.1f}K \\\\\n"
                )

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # CSV
        with open(tables_dir / 'model_comparison.csv', 'w') as f:
            f.write("model,type,params_M,flops_G,loss,ppl,accuracy,tokens_per_sec\n")
            for analyzer in self.analyzers:
                model_info = analyzer.results.get('model_info', {})
                perf = analyzer.results.get('performance', {})
                val = perf.get('validation', {})
                speed = perf.get('speed', {})

                f.write(f"{analyzer.name},{analyzer.model_type},"
                       f"{model_info.get('total_M', 0):.2f},{model_info.get('flops_G', 0):.2f},"
                       f"{val.get('loss', 0):.4f},{val.get('perplexity', 0):.2f},"
                       f"{val.get('accuracy', 0):.1f},{speed.get('tokens_per_sec', 0):.0f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='DAWN Complete Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint
  python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/

  # Multiple checkpoints
  python scripts/analysis/analyze_all.py --checkpoints dawn.pt vanilla.pt --val_data val.pt --output results/

  # Paper outputs only (faster)
  python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/ --paper-only

  # Specific analyses
  python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/ --only health,routing

  # Custom batch settings (faster)
  python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/ --n_batches 50 --val_batches 100

  # Large-scale analysis
  python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/ --n_batches 200 --max_sentences 5000 --batch_size 32
        """
    )
    # Input/output
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='Multiple checkpoint paths')
    parser.add_argument('--val_data', type=str, required=True, help='Validation data path')
    parser.add_argument('--output', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # Analysis mode
    parser.add_argument('--paper-only', action='store_true', help='Generate paper outputs only (faster)')
    parser.add_argument('--only', type=str, help='Run only specific analyses (comma-separated: health,routing,embedding,semantic,pos,factual,behavioral,coselection,weight,v18)')

    # Analysis parameters
    parser.add_argument('--n_batches', type=int, default=100, help='Number of batches for routing/semantic/behavioral/coselection (default: 100)')
    parser.add_argument('--val_batches', type=int, default=200, help='Number of batches for validation performance (default: 200)')
    parser.add_argument('--max_sentences', type=int, default=2000, help='Max sentences for POS analysis (default: 2000)')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs for factual analysis (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size (default: 16)')
    parser.add_argument('--max_samples', type=int, default=5000, help='Max samples for dataloader (default: 5000)')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for embedding analysis (default: 5)')
    parser.add_argument('--pool_type', type=str, default='fv', help='Neuron pool type: fv, fqk, rv, rqk (default: fv)')
    parser.add_argument('--target_layer', type=int, default=None, help='Target layer for POS/routing analysis (default: all layers)')
    parser.add_argument('--gen_tokens', type=int, default=50, help='Max tokens to generate per sample (default: 50)')

    args = parser.parse_args()

    # Print settings
    print("\n" + "=" * 60)
    print("DAWN Analysis Tool")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"n_batches: {args.n_batches}")
    print(f"val_batches: {args.val_batches}")
    print(f"max_sentences: {args.max_sentences}")
    print(f"n_runs: {args.n_runs}")
    print(f"max_samples: {args.max_samples}")
    print(f"n_clusters: {args.n_clusters}")
    print(f"pool_type: {args.pool_type}")
    print("=" * 60 + "\n")

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Determine checkpoints
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    elif args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        parser.error('Either --checkpoint or --checkpoints required')

    only = args.only.split(',') if args.only else None

    # Run analysis
    if len(checkpoint_paths) == 1:
        analyzer = ModelAnalyzer(
            checkpoint_paths[0], args.val_data,
            args.output, args.device,
            n_batches=args.n_batches,
            val_batches=args.val_batches,
            max_sentences=args.max_sentences,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            n_clusters=args.n_clusters,
            pool_type=args.pool_type,
            gen_tokens=args.gen_tokens,
            target_layer=args.target_layer,
        )
        analyzer.run_all(paper_only=args.paper_only, only=only)
    else:
        analyzer = MultiModelAnalyzer(
            checkpoint_paths, args.val_data,
            args.output, args.device,
            n_batches=args.n_batches,
            val_batches=args.val_batches,
            max_sentences=args.max_sentences,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            n_clusters=args.n_clusters,
            pool_type=args.pool_type,
            gen_tokens=args.gen_tokens,
            target_layer=args.target_layer,
        )
        analyzer.run_all(paper_only=args.paper_only, only=only)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

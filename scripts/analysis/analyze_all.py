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
                          Options: model_info,performance,health,routing,embedding,
                                   neuron_embedding,semantic,pos,token_combination,
                                   neuron_features,layerwise_semantic,factual,behavioral,
                                   coselection,weight,v18,paper,report

    Analysis Parameters:
        --n_batches       Batches for routing/semantic/behavioral/coselection (default: 100)
        --val_batches     Batches for validation performance (default: 200)
        --max_sentences   Max sentences for POS analysis (default: 2000)
        --min_targets     Min target occurrences for factual analysis (default: 100)
        --max_runs        Max runs for factual analysis safety limit (default: 500)
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
        min_targets: int = 100,
        max_runs: int = 500,
        batch_size: int = 16,
        max_samples: int = 5000,
        n_clusters: int = 5,
        pool_type: str = 'fv',
        gen_tokens: int = 50,
        target_layer: int = None,
        compare_checkpoint: str = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.val_data_path = val_data_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.compare_checkpoint = compare_checkpoint

        # Analysis parameters
        self.n_batches = n_batches
        self.val_batches = val_batches
        self.max_sentences = max_sentences
        self.min_targets = min_targets
        self.max_runs = max_runs
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
        """Analyze validation performance.

        Computes loss, perplexity, and accuracy on validation data.

        Args:
            n_batches: Number of batches to evaluate (default: 200)
                - 50: Quick test (~30 sec)
                - 200: Standard evaluation (~2 min)
                - 500+: Full validation pass

        Returns:
            Dict with loss, perplexity, accuracy, speed metrics
        """
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

        self.results['health'] = results

        # Save results.json for later --only paper usage
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def analyze_routing(self, n_batches: int = 100) -> Dict:
        """Analyze routing patterns (DAWN only).

        Analyzes how the router distributes attention across neuron pools.

        Args:
            n_batches: Number of batches to analyze (default: 100)
                - 20: Quick test (~1 min)
                - 100: Standard analysis (~5 min)
                - 200+: Comprehensive analysis

        Returns:
            Dict with entropy, selection frequency, diversity metrics per pool
        """
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
        qk_usage = results.get('qk_usage', {})

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
                    # Debug info
                    debug = data.get('debug', {})
                    if debug:
                        print(f"  │   Retrieved: {debug.get('retrieved_count', 0)} layers")
                        samples = debug.get('samples', [])
                        if samples:
                            s = samples[0]
                            print(f"  │   Sample: Q_active={s.get('q_active', 0)}, K_active={s.get('k_active', 0)}, "
                                  f"Q_max={s.get('q_max', 0):.4f}, K_max={s.get('k_max', 0):.4f}")
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

        # Q/K Specialization (forward-based, Fig 3)
        if qk_usage:
            print(f"\n  ┌─ Q/K Specialization (Forward-based) ─────────────────────────────────────")
            print(f"  │ {'Pool':<8} {'Total':>8} {'Q-spec':>8} {'K-spec':>8} {'Shared':>8} {'Inactive':>8} {'Corr':>8}")
            print(f"  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
            for pool, data in qk_usage.items():
                if isinstance(data, dict) and 'q_specialized' in data:
                    print(f"  │ {pool:<8} {data.get('n_neurons', 0):>8d} {data['q_specialized']:>8d} "
                          f"{data['k_specialized']:>8d} {data['shared']:>8d} {data['inactive']:>8d} "
                          f"{data.get('correlation', 0):>8.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Q/K Union Coverage (true dead neurons)
        qk_union = results.get('qk_union_coverage', {})
        if qk_union and 'n_batches' in qk_union:
            print(f"\n  ┌─ Q/K Union Coverage (True Dead) ──────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Total':>8} {'Q-only':>8} {'K-only':>8} {'Shared':>8} {'Dead':>8} {'Union%':>8}")
            print(f"  │ {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
            for pool, data in qk_union.items():
                if isinstance(data, dict) and 'n_total' in data:
                    print(f"  │ {pool:<12} {data['n_total']:>8d} {data['q_only']:>8d} {data['k_only']:>8d} "
                          f"{data['shared']:>8d} {data['dead']:>8d} {data['union_coverage']*100:>7.1f}%")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Activation Sparsity
        sparsity = results.get('activation_sparsity', {})
        if sparsity and 'n_batches' in sparsity:
            print(f"\n  ┌─ Activation Sparsity ─────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Avg Active':>12} {'Total':>8} {'Sparsity':>10} {'Wgt Sum':>10}")
            print(f"  │ {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*10}")
            for key, data in sparsity.items():
                if isinstance(data, dict) and 'avg_active_per_token' in data:
                    print(f"  │ {data.get('display', key):<12} {data['avg_active_per_token']:>12.1f} {data['n_total']:>8d} "
                          f"{data['sparsity_ratio']*100:>9.1f}% {data['avg_weight_sum']:>10.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Token Co-selection
        coselect = results.get('token_coselection', {})
        if coselect and 'n_batches' in coselect:
            print(f"\n  ┌─ Per-Token Q/K Co-selection ──────────────────────────────────────────")
            for pool, data in coselect.items():
                if isinstance(data, dict) and 'mean_coselect_per_token' in data:
                    print(f"  │ {pool}:")
                    print(f"  │   Coselect: {data['mean_coselect_per_token']:.1f}/token  |  "
                          f"Rate: {data['coselect_rate']*100:.1f}%  |  "
                          f"Q: {data['mean_q_active']:.1f}  K: {data['mean_k_active']:.1f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Weight Concentration
        concentration = results.get('weight_concentration', {})
        if concentration and 'n_batches' in concentration:
            print(f"\n  ┌─ Weight Concentration ─────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Top1%':>10} {'Top5%':>10} {'AvgActive':>12}")
            print(f"  │ {'─'*12} {'─'*10} {'─'*10} {'─'*12}")
            for key, data in concentration.items():
                if isinstance(data, dict) and 'top1_weight_ratio' in data:
                    print(f"  │ {data.get('display', key):<12} {data['top1_weight_ratio']*100:>9.1f}% "
                          f"{data['top5_weight_ratio']*100:>9.1f}% "
                          f"{data['avg_active_neurons']:>12.1f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Path Usage (v18.5)
        path_usage = results.get('path_usage', {})
        if path_usage and 'per_path' in path_usage and path_usage['per_path']:
            print(f"\n  ┌─ Path Usage (v18.5) ──────────────────────────────────────────────────")
            print(f"  │ Max paths: {path_usage.get('max_paths', 'N/A')}")
            for key, data in path_usage.get('per_path', {}).items():
                if isinstance(data, dict) and 'activation_rate' in data:
                    print(f"  │ {key}: activation={data['activation_rate']*100:.1f}%")
            for pool, data in path_usage.get('per_pool', {}).items():
                if isinstance(data, dict) and 'avg_active_paths' in data:
                    print(f"  │ {pool}: avg_active_paths={data['avg_active_paths']:.2f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Coverage Progression
        progression = results.get('coverage_progression', {})
        if progression and 'n_batches' in progression:
            print(f"\n  ┌─ Coverage Progression ─────────────────────────────────────────────────")
            for pool, data in progression.items():
                if isinstance(data, dict) and 'trend' in data:
                    print(f"  │ {pool}: early={data['early_coverage']*100:.1f}% → "
                          f"mid={data['mid_coverage']*100:.1f}% → late={data['late_coverage']*100:.1f}% "
                          f"({data['trend']}, {data['growth_ratio']:.2f}x)")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['routing'] = results

        # Save results.json for later --only paper usage
        output_dir = self.output_dir / 'routing'
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

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

    def analyze_neuron_embedding(self, n_batches: int = 50, k_range: Tuple[int, int] = (5, 20)) -> Dict:
        """Analyze neuron embeddings with clustering and token projections (DAWN only).

        Clusters neurons based on their embedding vectors and analyzes
        how different pools align with clusters.

        Args:
            n_batches: Number of batches for token projection analysis (default: 50)
                - 10: Quick test
                - 50: Standard analysis
                - 100+: More accurate projections
            k_range: Range of cluster counts to try for K-means (default: (5, 20))
                - (3, 10): Coarse clustering
                - (5, 20): Standard range (recommended)
                - (10, 30): Fine-grained clustering

        Returns:
            Dict with pool distribution, optimal clustering, silhouette scores
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.neuron_embedding import NeuronEmbeddingAnalyzer

        output_dir = self.output_dir / 'neuron_embedding'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing neuron embeddings (k_range={k_range}, {n_batches} batches)...")
        dataloader = self._get_dataloader()
        analyzer = NeuronEmbeddingAnalyzer(self.model, device=self.device)
        analyzer.tokenizer = self.tokenizer

        results = analyzer.run_full_analysis(
            dataloader,
            output_dir=str(output_dir),
            n_batches=n_batches,
            k_range=k_range,
            tokenizer=self.tokenizer
        )

        # Print summary
        pool_dist = results.get('pool_distribution', {})
        clustering = results.get('clustering', {})
        pos_analysis = results.get('pos_projection_analysis', {})
        neuron_pos = results.get('neuron_pos_similarity', {})

        if pool_dist.get('pools'):
            print(f"\n  ┌─ Pool Distribution ────────────────────────────────────────────────────")
            print(f"  │ {'Pool':<12} {'N':>8} {'Norm μ':>10} {'CosSim μ':>10}")
            print(f"  │ {'─'*12} {'─'*8} {'─'*10} {'─'*10}")
            for pool, data in pool_dist['pools'].items():
                print(f"  │ {data.get('display', pool):<12} {data['n_neurons']:>8} "
                      f"{data['norm_mean']:>10.4f} {data['mean_cosine_sim']:>10.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if clustering:
            print(f"\n  ┌─ Clustering ────────────────────────────────────────────────────────────")
            print(f"  │ Optimal k: {clustering.get('optimal_k', 'N/A')}")
            print(f"  │ Best silhouette: {clustering.get('best_silhouette', 0):.4f}")
            print(f"  │ Total embeddings: {clustering.get('n_embeddings', 0)}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Pool alignment analysis (k=6 fixed)
        pool_alignment = results.get('pool_alignment', {})
        if pool_alignment and 'pool_coverage' in pool_alignment:
            print(f"\n  ┌─ Pool-Cluster Alignment (k=6) ─────────────────────────────────────────")
            print(f"  │ Overall Purity: {pool_alignment.get('overall_purity', 0)*100:.1f}%")
            print(f"  │ Mean Pool Coverage: {pool_alignment.get('mean_pool_coverage', 0)*100:.1f}%")
            print(f"  │ Silhouette (true): {pool_alignment.get('silhouette_true_labels', 0):.4f}")
            print(f"  │ Silhouette (pred): {pool_alignment.get('silhouette_pred_labels', 0):.4f}")
            print(f"  │")
            print(f"  │ Pool → Best Cluster (coverage):")
            for pool, data in pool_alignment.get('best_cluster_for_pool', {}).items():
                print(f"  │   {pool:<12} → cluster {data['cluster']} ({data['coverage']*100:.1f}%)")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Per-pool clustering
        per_pool = results.get('per_pool_clustering', {})
        if per_pool:
            print(f"\n  ┌─ Per-Pool Clustering (internal structure) ───────────────────────────")
            print(f"  │ {'Pool':<12} {'N':>6} {'Opt k':>6} {'Silhouette':>12}")
            print(f"  │ {'─'*12} {'─'*6} {'─'*6} {'─'*12}")
            for pool_name, data in per_pool.items():
                n = data.get('n_neurons', 0)
                k = data.get('optimal_k', 'N/A')
                sil = data.get('best_silhouette', 0)
                print(f"  │ {pool_name:<12} {n:>6} {k:>6} {sil:>12.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if pos_analysis and 'pos_stats' in pos_analysis:
            print(f"\n  ┌─ POS Projection Analysis ──────────────────────────────────────────────")
            print(f"  │ POS categories: {pos_analysis.get('n_pos_categories', 0)}")
            print(f"  │ Silhouette score: {pos_analysis.get('silhouette_score', 'N/A')}")
            top_pos = sorted(pos_analysis['pos_stats'].items(),
                           key=lambda x: x[1].get('n_tokens', 0), reverse=True)[:5]
            for pos, data in top_pos:
                print(f"  │   {pos}: n={data['n_tokens']}, dist={data['mean_dist_to_centroid']:.3f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if neuron_pos and 'pos_pool_affinity' in neuron_pos:
            print(f"\n  ┌─ Neuron-POS Affinity ──────────────────────────────────────────────────")
            for pos, affinity in list(neuron_pos['pos_pool_affinity'].items())[:5]:
                pools = ', '.join(f"{p}:{d['count']}" for p, d in affinity.items())
                print(f"  │ {pos}: {pools}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['neuron_embedding'] = results
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

    def analyze_pos(self, max_sentences: int = 2000, pool_type: str = 'fv', target_layer: int = None,
                     compute_coactivation: bool = False) -> Dict:
        """Analyze POS neuron specialization (DAWN only).

        Analyzes how neurons specialize for different Part-of-Speech tags.
        Uses Universal Dependencies English Web Treebank dataset.

        Args:
            max_sentences: Number of sentences to analyze (default: 2000)
                - 500: Quick test (~1 min)
                - 2000: Standard analysis (~5 min)
                - 5000+: Comprehensive analysis
            pool_type: Which neuron pool to analyze
                - 'fv': Feature V pool (default, recommended)
                - 'rv': Restore V pool
                - 'fqk', 'fqk_q', 'fqk_k': Feature QK pools
                - 'rqk', 'rqk_q', 'rqk_k': Restore QK pools
                - 'fknow', 'rknow': Knowledge pools
            target_layer: Specific layer to analyze (default: None = all layers)
                - 0-11: Specific layer index
                - None: Average across all layers
            compute_coactivation: Compute neuron co-activation correlation matrix
                - False: Skip (faster, less memory)
                - True: Compute correlation between neurons (memory intensive)

        Returns:
            Dict with selectivity matrix, top neurons per POS, clustering results
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import POSNeuronAnalyzer

        output_dir = self.output_dir / 'pos'
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_str = f"layer={target_layer}" if target_layer is not None else "all layers"
        coact_str = ", coactivation=ON" if compute_coactivation else ""
        print(f"  Analyzing POS neuron specialization ({max_sentences} sentences, pool={pool_type}, {layer_str}{coact_str})...")
        analyzer = POSNeuronAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device,
            target_layer=target_layer
        )
        results = analyzer.run_all(
            str(output_dir), pool_type=pool_type, max_sentences=max_sentences,
            compute_coactivation=compute_coactivation
        )

        # Print detailed summary
        pos_counts = results.get('pos_token_counts', {})
        top_neurons = results.get('top_neurons_per_pos', {})
        specificity = results.get('neuron_specificity', {})

        if pos_counts:
            print(f"\n  ┌─ POS Token Distribution ───────────────────────────────────────────────")
            total_tokens = sum(pos_counts.values())
            print(f"  │ Total tokens analyzed: {total_tokens:,}")
            print(f"  │")
            # Sort by count and show top POS
            sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"  │ {'POS':<10} {'Count':>10} {'Ratio':>10}")
            print(f"  │ {'─'*30}")
            for pos, count in sorted_pos[:10]:
                ratio = count / total_tokens * 100 if total_tokens > 0 else 0
                print(f"  │ {pos:<10} {count:>10,} {ratio:>9.1f}%")
            if len(sorted_pos) > 10:
                print(f"  │ ... and {len(sorted_pos) - 10} more POS tags")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if top_neurons:
            print(f"\n  ┌─ Top Neurons per POS ──────────────────────────────────────────────────")
            print(f"  │ {'POS':<10} {'Top Neurons (id:sel)':<50}")
            print(f"  │ {'─'*60}")
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PUNCT']:
                if pos in top_neurons and top_neurons[pos]:
                    neurons = top_neurons[pos][:5]
                    # Handle both old format (tuples) and new format (dicts)
                    if neurons and isinstance(neurons[0], dict):
                        neuron_str = ', '.join(f"N{n['neuron']}:{n['selectivity']:.2f}" for n in neurons)
                    else:
                        neuron_str = ', '.join(f'{n}:{f:.2f}' for n, f in neurons)
                    print(f"  │ {pos:<10} {neuron_str}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if specificity:
            print(f"\n  ┌─ POS-Specific Neurons (High Selectivity) ──────────────────────────────")
            top_specific = list(specificity.items())[:10]
            print(f"  │ {'Neuron':<10} {'Top POS':<10} {'Selectivity':>12} {'Mean Weight':>12}")
            print(f"  │ {'─'*44}")
            for neuron_id, data in top_specific:
                # Handle both old format and new format
                selectivity = data.get('selectivity', data.get('specificity', 0))
                mean_weight = data.get('mean_weight', data.get('top_score', 0))
                print(f"  │ N{neuron_id:<9} {data['top_pos']:<10} {selectivity:>12.2f}x {mean_weight:>12.4f}")
            print(f"  │")
            print(f"  │ Total specific neurons: {len(specificity)}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['pos'] = results
        return results

    def analyze_token_combination(self, max_sentences: int = 2000, target_layer: int = None,
                                   activation_threshold: float = 1e-6) -> Dict:
        """Analyze token-based neuron combinations (DAWN only).

        Token-centric analysis: which neuron combinations each token activates.
        Uses Jaccard similarity and silhouette score to measure POS clustering quality.

        Args:
            max_sentences: Number of sentences to analyze (default: 2000)
                - 500: Quick test (~2 min)
                - 2000: Standard analysis (~8 min)
                - 5000+: Comprehensive analysis
            target_layer: Specific layer to analyze (default: None = all layers majority vote)
                - 0-11: Specific layer index
                - None: Majority vote across all layers
            activation_threshold: Threshold for binary activation (default: 1e-6)
                - 1e-6: Include all non-zero weights
                - 0.01: Only strong activations
                - 0.1: Only very strong activations

        Returns:
            Dict with silhouette score, Jaccard similarities, content/function analysis
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import TokenCombinationAnalyzer

        output_dir = self.output_dir / 'token_combination'
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_str = f"layer={target_layer}" if target_layer is not None else "all layers"
        print(f"  Analyzing token neuron combinations ({max_sentences} sentences, {layer_str})...")

        analyzer = TokenCombinationAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device,
            target_layer=target_layer, activation_threshold=activation_threshold
        )
        results = analyzer.run_all(str(output_dir), max_sentences=max_sentences)

        # Print key metrics
        sil = results.get('silhouette_score', {})
        pos_sim = results.get('pos_similarity', {})

        if sil.get('score') is not None:
            print(f"\n  ┌─ Token Combination Results ────────────────────────────────────────────")
            print(f"  │ Silhouette Score: {sil['score']:.4f}  (target: > 0.3)")
            print(f"  │ Samples: {sil.get('n_samples', 0)}, POS categories: {sil.get('n_pos_categories', 0)}")
            if pos_sim:
                print(f"  │")
                print(f"  │ Jaccard Similarity:")
                print(f"  │   Within-POS mean:  {pos_sim.get('mean_within', 0):.4f}")
                print(f"  │   Between-POS mean: {pos_sim.get('mean_between', 0):.4f}")
                print(f"  │   Separation:       {pos_sim.get('separation', 0):.4f}")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['token_combination'] = results
        return results

    def analyze_neuron_features(self, max_sentences: int = 2000, target_layer: int = None) -> Dict:
        """Analyze neuron-centric features (DAWN only).

        Inverts the token->neuron perspective to analyze what features each neuron responds to.

        Analyzes per neuron:
        - POS distribution of activating tokens
        - Sentence position distribution
        - Token frequency (high/med/low)
        - Subword position (word-initial vs continuation)
        - Next token POS patterns

        Args:
            max_sentences: Number of sentences to analyze (default: 2000)
            target_layer: Specific layer to analyze (default: None = all layers)

        Returns:
            Dict with neuron profiles, specialized neurons, clusters
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import TokenCombinationAnalyzer, NeuronFeatureAnalyzer

        output_dir = self.output_dir / 'neuron_features'
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_str = f"layer={target_layer}" if target_layer is not None else "all layers"
        print(f"  Analyzing neuron features ({max_sentences} sentences, {layer_str})...")

        # First collect token data
        tca = TokenCombinationAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device,
            target_layer=target_layer
        )
        dataset = tca.load_ud_dataset('train', max_sentences)
        tca.analyze_dataset(dataset, max_sentences=max_sentences, analyze_layer_divergence=False)

        # Then run neuron feature analysis
        nfa = NeuronFeatureAnalyzer.from_token_combination_analyzer(tca)
        results = nfa.run_full_analysis(output_dir=str(output_dir))

        # Print key metrics
        specialized = results.get('specialized_neurons', {})
        clusters = results.get('clusters', {})

        print(f"\n  ┌─ Neuron Feature Analysis Results ─────────────────────────────────────")
        print(f"  │ Neurons profiled: {results.get('n_neurons_profiled', 0)}")
        print(f"  │")
        print(f"  │ Specialized neurons (80%+ concentration):")
        for feature, neurons in specialized.items():
            if neurons:
                print(f"  │   {feature:12s}: {len(neurons)} neurons")
        print(f"  │")
        if clusters.get('silhouette_score'):
            print(f"  │ Cluster silhouette: {clusters['silhouette_score']:.4f}")
        print(f"  └─────────────────────────────────────────────────────────────────────────")

        self.results['neuron_features'] = results

        # Save results.json for later --only paper usage
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def analyze_layerwise_semantic(self, max_sentences: int = 500) -> Dict:
        """Analyze layer-wise semantic emergence (DAWN only).

        Runs per-layer analysis to measure:
        1. Semantic correlation: GloVe similarity vs neuron weight similarity
        2. POS silhouette: How well neurons cluster by part-of-speech

        Hypothesis:
        - Early layers: Strong POS clustering (syntax)
        - Later layers: Strong semantic correlation (semantics)
        - Crossover point = syntax→semantics transition

        Args:
            max_sentences: Number of sentences per layer (default: 500)
                - 200: Quick test (~5 min)
                - 500: Standard analysis (~15 min)
                - 1000: Comprehensive analysis (~30 min)

        Returns:
            Dict with per-layer semantic correlation and silhouette scores
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import TokenCombinationAnalyzer

        output_dir = self.output_dir / 'layerwise_semantic'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get number of layers from model
        n_layers = getattr(self.model, 'n_layers', 8)

        print(f"  Analyzing layer-wise semantic emergence ({n_layers} layers, {max_sentences} sentences/layer)...")

        results = TokenCombinationAnalyzer.run_layerwise_analysis(
            model=self.model,
            tokenizer=self.tokenizer,
            n_layers=n_layers,
            output_dir=str(output_dir),
            max_sentences=max_sentences,
            device=self.device,
        )

        self.results['layerwise_semantic'] = results
        return results

    def analyze_factual(self, min_target_count: int = 100, max_runs: int = 500, pool_type: str = 'fv') -> Dict:
        """Analyze factual knowledge neurons (DAWN only).

        Finds neurons that consistently activate for factual knowledge
        (e.g., "The capital of France is" -> Paris).

        Independent runs approach: each run generates until target appears,
        records neurons, then starts fresh. Repeats until min_target_count.

        Args:
            min_target_count: Minimum target occurrences to collect (default: 100)
            max_runs: Maximum runs as safety limit (default: 500)
            pool_type: Which neuron pool to analyze ('fv', 'rv', 'fqk', etc.)

        Returns:
            Dict with per-target neuron activations, common neurons, match rates
        """
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.behavioral import BehavioralAnalyzer

        output_dir = self.output_dir / 'factual'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing factual neurons (min_targets={min_target_count}, max_runs={max_runs}, pool={pool_type})...")
        analyzer = BehavioralAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device
        )

        prompts = [
            "The capital of France is",
            "The capital of England is",
            "The capital of Japan is",
            "The color of the sky is",
        ]
        targets = ["Paris", "London", "Tokyo", "blue"]

        results = analyzer.analyze_factual_neurons(
            prompts, targets,
            pool_type=pool_type,
            min_target_count=min_target_count,
            max_runs=max_runs,
            temperature=1.0,
            top_k=50,
        )

        # Print detailed summary
        per_target = results.get('per_target', {})
        if per_target:
            for target, data in per_target.items():
                if not isinstance(data, dict):
                    continue

                successful = data.get('successful_runs', 0)
                total_runs = data.get('total_runs', 0)
                match_rate = data.get('match_rate', 0) * 100

                print(f"\n  {'='*70}")
                print(f"  RESULTS: '{target}' found {successful}/{total_runs} runs ({match_rate:.1f}%)")
                print(f"  {'='*70}")

                if successful > 0:
                    # Common neurons at different thresholds
                    n100 = data.get('common_neurons_100', [])
                    n80 = data.get('common_neurons_80', [])
                    n50 = data.get('common_neurons_50', [])

                    print(f"\n  100% common neurons ({len(n100)}): {n100[:20]}{'...' if len(n100) > 20 else ''}")
                    print(f"   80%+ common neurons ({len(n80)}): {n80[:20]}{'...' if len(n80) > 20 else ''}")
                    print(f"   50%+ common neurons ({len(n50)}): {n50[:20]}{'...' if len(n50) > 20 else ''}")

                    # Top neurons by frequency (bar chart style)
                    neuron_freqs = data.get('neuron_frequencies', [])[:15]
                    if neuron_freqs:
                        print(f"\n  Top 15 neurons by frequency:")
                        for nf in neuron_freqs:
                            bar = '█' * int(nf['percentage'] / 5)
                            print(f"    Neuron {nf['neuron']:4d}: {nf['count']:3d}/{successful} ({nf['percentage']:5.1f}%) {bar}")

                    # Sample generations
                    samples = data.get('sample_generations', [])[:3]
                    if samples:
                        print(f"\n  Sample generations with '{target}':")
                        for i, sample in enumerate(samples):
                            text = sample[:80] if isinstance(sample, str) else str(sample)[:80]
                            print(f"    Run {i}: {text}...")

                    # Top contrastive neurons
                    contrastive = data.get('contrastive_top50', [])[:5]
                    if contrastive:
                        print(f"\n  Top target-specific neurons (contrastive score):")
                        for c in contrastive:
                            print(f"    Neuron {c['neuron']:4d}: target={c['target_freq']:5.1f}% baseline={c['baseline_freq']:5.1f}% score={c['score']:.3f}")
                else:
                    first_gen = data.get('first_generation', '')[:80]
                    print(f"\n  Note: Target '{target}' not found in {total} generations")
                    print(f"  First generation: \"{first_gen}...\"")

            print(f"\n  {'='*70}")

        with open(output_dir / 'factual_neurons.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Also save as results.json for --only paper consistency
        with open(output_dir / 'results.json', 'w') as f:
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
            print(f"\n  ┌─ Token Trajectory (Entropy by Position) ─────────────────────────────────")
            print(f"  │ {'Pool':<12} {'Early (<10)':<12} {'Late (≥10)':<12} {'Δ Change':<12}")
            print(f"  │ {'─'*48}")
            for pool, data in trajectory.items():
                if isinstance(data, dict) and 'early_avg' in data:
                    display = data.get('display', pool)
                    early = data['early_avg']
                    late = data['late_avg']
                    delta = late - early
                    sign = '+' if delta >= 0 else ''
                    print(f"  │ {display:<12} {early:>10.1f}% {late:>10.1f}% {sign}{delta:>10.1f}%")
            n_layers = trajectory.get('n_layers', 0)
            if n_layers:
                print(f"  │ {'─'*48}")
                print(f"  │ Analyzed {n_layers} layers")
            print(f"  └─────────────────────────────────────────────────────────────────────────")

        if probing and 'error' not in probing:
            print(f"\n  ┌─ Probing Classifier (POS Prediction from Routing Weights) ───────────────")
            overall = probing.get('overall', {})
            if overall:
                print(f"  │ Overall: {overall.get('n_classifiers', 0)} classifiers trained")
                print(f"  │   Mean accuracy: {overall.get('mean_accuracy', 0)*100:.1f}% (±{overall.get('std_accuracy', 0)*100:.1f}%)")
                print(f"  │   Range: {overall.get('min_accuracy', 0)*100:.1f}% ~ {overall.get('max_accuracy', 0)*100:.1f}%")

            # Per-routing type summary
            summary = probing.get('summary', {})
            if summary:
                print(f"  │")
                print(f"  │ {'Routing':<12} {'Mean Acc':<12} {'Max Acc':<12}")
                print(f"  │ {'─'*36}")
                for rkey, data in sorted(summary.items()):
                    if isinstance(data, dict) and 'mean_accuracy' in data:
                        print(f"  │ {rkey:<12} {data['mean_accuracy']*100:>10.1f}% {data['max_accuracy']*100:>10.1f}%")

            # Best/worst layers
            per_layer = probing.get('per_layer', {})
            if per_layer:
                layer_accs = []
                for layer_key, layer_data in per_layer.items():
                    accs = [d['accuracy'] for d in layer_data.values() if isinstance(d, dict) and 'accuracy' in d]
                    if accs:
                        layer_accs.append((layer_key, sum(accs)/len(accs)))
                if layer_accs:
                    layer_accs.sort(key=lambda x: x[1], reverse=True)
                    print(f"  │")
                    print(f"  │ Best layers:  {layer_accs[0][0]} ({layer_accs[0][1]*100:.1f}%)", end='')
                    if len(layer_accs) > 1:
                        print(f", {layer_accs[1][0]} ({layer_accs[1][1]*100:.1f}%)", end='')
                    print()
                    print(f"  │ Worst layers: {layer_accs[-1][0]} ({layer_accs[-1][1]*100:.1f}%)", end='')
                    if len(layer_accs) > 1:
                        print(f", {layer_accs[-2][0]} ({layer_accs[-2][1]*100:.1f}%)", end='')
                    print()

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
            if isinstance(data, dict) and 'pair_name' in data:
                print(f"  │ {data['pair_name']}:")
                # Show concentration metrics
                conc = data.get('concentration', {})
                if conc:
                    print(f"  │   Top10: {conc.get('top10_pct', 0):.1f}%, Top50: {conc.get('top50_pct', 0):.1f}%")
                    print(f"  │   Entropy: {conc.get('normalized_entropy', 0):.3f} (normalized)")
                # Show top pairs
                top_pairs = data.get('top_pairs', [])
                if top_pairs:
                    top_3 = [(p['a_idx'], p['b_idx'], p['pct']) for p in top_pairs[:3]]
                    print(f"  │   Top pairs: {', '.join(f'({a},{b}):{pct:.1f}%' for a,b,pct in top_3)}")
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

        # Check if shared_neurons is available
        if analyzer.neurons is None:
            print("  Skipping (no shared_neurons found - may be v18.5+ model)")
            return {}

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

            gen = PaperFigureGenerator(
                self.checkpoint_path,
                self.val_data_path,  # Pass path, not dataloader
                device=self.device
            )

            # Build precomputed results from already-run analyses
            precomputed = {}
            if 'routing' in self.results:
                precomputed['routing'] = self.results['routing']
            if 'health' in self.results:
                precomputed['health'] = self.results['health']
            if 'factual' in self.results:
                precomputed['factual'] = self.results['factual']
            if 'neuron_features' in self.results:
                precomputed['neuron_features'] = self.results['neuron_features']

            # Build config from instance parameters
            config = {
                'pool_type': self.pool_type,
                'gen_tokens': self.gen_tokens,
                'max_sentences': self.max_sentences,
                'target_layer': self.target_layer,
            }

            # Add checkpoint paths for figure 6 (training dynamics comparison)
            checkpoint_paths = [self.checkpoint_path]
            if self.compare_checkpoint:
                checkpoint_paths.append(self.compare_checkpoint)
            config['checkpoint_paths'] = checkpoint_paths

            # Generate figures 3,4,5,6,7 (6 = training dynamics)
            gen.generate('3,4,5,6,7', str(figures_dir), n_batches=self.min_targets // 10,
                        precomputed=precomputed, config=config)
        except Exception as e:
            print(f"    Warning: Could not generate paper figures: {e}")
            import traceback
            traceback.print_exc()

        # Generate tables
        print("  Generating paper tables...")
        self._generate_tables(tables_dir)

        # Generate comparison samples if baseline available
        if self.compare_checkpoint:
            print("  Generating comparison samples...")
            self._generate_comparison_samples(paper_dir)

        # Generate paper_results.json with all numeric data
        print("  Generating paper_results.json...")
        self._generate_paper_results_json(paper_dir)

        # Generate training config comparison
        if self.compare_checkpoint:
            print("  Generating training_comparison...")
            self._generate_training_comparison(paper_dir)

        # Summary
        self._generate_paper_summary(paper_dir)

    def _generate_tables(self, tables_dir: Path):
        """Generate LaTeX and CSV tables with optional vanilla comparison."""
        # Try to get model_info from results, fallback to saved JSON
        model_info = self.results.get('model_info', {})
        if not model_info:
            params_file = self.output_dir / 'model_info' / 'parameters.json'
            if params_file.exists():
                with open(params_file) as f:
                    model_info = json.load(f)

        perf = self.results.get('performance', {})
        if not perf:
            # Load from individual files
            val_file = self.output_dir / 'performance' / 'validation.json'
            speed_file = self.output_dir / 'performance' / 'speed_benchmark.json'
            if val_file.exists():
                with open(val_file) as f:
                    perf['validation'] = json.load(f)
            if speed_file.exists():
                with open(speed_file) as f:
                    perf['speed'] = json.load(f)

        val = perf.get('validation', {})
        speed = perf.get('speed', {})

        # Load comparison model data if available
        vanilla_info = {}
        vanilla_val = {}
        vanilla_speed = {}
        if self.compare_checkpoint:
            comp_path = Path(self.compare_checkpoint)
            # Try to find analysis results for comparison checkpoint
            comp_dirs = [
                comp_path.parent / 'analysis',
                comp_path / 'analysis',
                self.output_dir.parent / comp_path.stem / 'analysis' if comp_path.is_file() else None,
                self.output_dir.parent / comp_path.name if comp_path.is_dir() else None,
            ]
            found_results = False
            for comp_dir in comp_dirs:
                if comp_dir and comp_dir.exists():
                    comp_params = comp_dir / 'model_info' / 'parameters.json'
                    comp_val_file = comp_dir / 'performance' / 'validation.json'
                    comp_speed_file = comp_dir / 'performance' / 'speed_benchmark.json'
                    if comp_params.exists():
                        with open(comp_params) as f:
                            vanilla_info = json.load(f)
                        found_results = True
                    if comp_val_file.exists():
                        with open(comp_val_file) as f:
                            vanilla_val = json.load(f)
                    if comp_speed_file.exists():
                        with open(comp_speed_file) as f:
                            vanilla_speed = json.load(f)
                    if vanilla_info or vanilla_val:
                        found_results = True
                        break

            # If no pre-existing results, run quick analysis on vanilla model
            if not found_results:
                print("    Running quick analysis on comparison model...")
                try:
                    vanilla_info, vanilla_val, vanilla_speed = self._analyze_comparison_model()
                except Exception as e:
                    print(f"    Warning: Could not analyze comparison model: {e}")

        has_comparison = bool(vanilla_info or vanilla_val)

        # Model stats CSV
        with open(tables_dir / 'model_stats.csv', 'w') as f:
            if has_comparison:
                f.write("metric,dawn,vanilla\n")
                f.write(f"parameters_M,{model_info.get('total_M', 0):.2f},{vanilla_info.get('total_M', 0):.2f}\n")
                f.write(f"flops_G,{model_info.get('flops_G', 0):.2f},{vanilla_info.get('flops_G', 0):.2f}\n")
                f.write(f"ppl,{val.get('perplexity', 0):.2f},{vanilla_val.get('perplexity', 0):.2f}\n")
                f.write(f"accuracy,{val.get('accuracy', 0):.2f},{vanilla_val.get('accuracy', 0):.2f}\n")
                f.write(f"tokens_per_sec,{speed.get('tokens_per_sec', 0):.0f},{vanilla_speed.get('tokens_per_sec', 0):.0f}\n")
            else:
                f.write("metric,value\n")
                f.write(f"parameters,{model_info.get('total', 0)}\n")
                f.write(f"parameters_M,{model_info.get('total_M', 0):.2f}\n")
                f.write(f"flops,{model_info.get('flops', 0)}\n")
                f.write(f"flops_G,{model_info.get('flops_G', 0):.2f}\n")
                f.write(f"ppl,{val.get('perplexity', 0):.2f}\n")
                f.write(f"accuracy,{val.get('accuracy', 0):.2f}\n")
                f.write(f"tokens_per_sec,{speed.get('tokens_per_sec', 0):.0f}\n")

        # Model stats LaTeX - comparison table if vanilla available
        with open(tables_dir / 'model_stats.tex', 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            if has_comparison:
                f.write("\\caption{Model Comparison: DAWN vs Vanilla}\n")
                f.write("\\begin{tabular}{lrr}\n")
                f.write("\\toprule\n")
                f.write("Metric & DAWN & Vanilla \\\\\n")
                f.write("\\midrule\n")
                f.write(f"Parameters & {model_info.get('total_M', 0):.2f}M & {vanilla_info.get('total_M', 0):.2f}M \\\\\n")
                f.write(f"FLOPs & {model_info.get('flops_G', 0):.2f}G & {vanilla_info.get('flops_G', 0):.2f}G \\\\\n")
                f.write(f"Perplexity & {val.get('perplexity', 0):.2f} & {vanilla_val.get('perplexity', 0):.2f} \\\\\n")
                f.write(f"Accuracy & {val.get('accuracy', 0):.1f}\\% & {vanilla_val.get('accuracy', 0):.1f}\\% \\\\\n")
                f.write(f"Speed & {speed.get('tokens_per_sec', 0)/1000:.1f}K & {vanilla_speed.get('tokens_per_sec', 0)/1000:.1f}K tok/s \\\\\n")
            else:
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

    def _generate_comparison_samples(self, paper_dir: Path):
        """Generate text samples comparing DAWN vs Baseline for paper."""
        import torch.nn.functional as F

        # Paper-quality prompts
        PROMPTS = {
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

        def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
            """Generate text with top-k sampling."""
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            generated = input_ids.clone()
            eos_token_id = self.tokenizer.sep_token_id

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    output = model(generated, attention_mask=None)
                    logits = output[0] if isinstance(output, tuple) else output
                    next_token_logits = logits[:, -1, :] / temperature

                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)

                    if next_token.item() == eos_token_id:
                        break

            return self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Load baseline model
        print("    Loading baseline model...")
        try:
            from scripts.analysis.utils import load_model
            baseline_model, _, baseline_config = load_model(self.compare_checkpoint, self.device)
            baseline_model.eval()

            # Get baseline name
            baseline_path = Path(self.compare_checkpoint)
            baseline_name = baseline_path.name if baseline_path.is_dir() else baseline_path.parent.name
        except Exception as e:
            print(f"    Error loading baseline: {e}")
            return

        # Generate samples
        lines = []
        lines.append("=" * 100)
        lines.append("GENERATION COMPARISON: DAWN vs Baseline")
        lines.append("=" * 100)
        lines.append(f"DAWN Model:     {self.name} (v{self.version})")
        lines.append(f"Baseline Model: {baseline_name}")
        lines.append(f"Max tokens: {self.gen_tokens}")
        lines.append("=" * 100)

        # Print header to console
        print("\n" + "\n".join(lines[-5:]))

        for category, prompts in PROMPTS.items():
            cat_header = f"\n{'='*100}\n[{category.upper()}]\n{'='*100}"
            lines.append(cat_header)
            print(cat_header)

            for prompt in prompts:
                prompt_line = f"\n{'─'*100}\nPrompt: \"{prompt}\"\n{'─'*100}"
                lines.append(prompt_line)
                print(prompt_line)

                # Generate from both models
                dawn_output = generate_text(self.model, prompt, max_new_tokens=self.gen_tokens)
                baseline_output = generate_text(baseline_model, prompt, max_new_tokens=self.gen_tokens)

                # Extract generated part (remove prompt)
                dawn_gen = dawn_output[len(prompt):].strip() if dawn_output.startswith(prompt) else dawn_output
                baseline_gen = baseline_output[len(prompt):].strip() if baseline_output.startswith(prompt) else baseline_output

                dawn_line = f"  DAWN:     {dawn_gen}"
                baseline_line = f"  Baseline: {baseline_gen}"

                lines.append(dawn_line)
                lines.append(baseline_line)
                print(dawn_line)
                print(baseline_line)

        # Save to file
        output_path = paper_dir / 'generation_comparison.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"\n    Saved to: {output_path}")

        # Cleanup
        del baseline_model
        torch.cuda.empty_cache()

    def _analyze_comparison_model(self):
        """Run quick analysis on comparison model for table generation."""
        from scripts.analysis.utils import load_model
        from scripts.evaluation.evaluate import evaluate_model, load_val_data, estimate_flops

        print(f"    Loading: {self.compare_checkpoint}")
        comp_model, _, comp_config = load_model(self.compare_checkpoint, self.device)
        comp_model.eval()

        # Model info
        total_params = sum(p.numel() for p in comp_model.parameters())
        flops = estimate_flops(comp_model, seq_len=512)
        vanilla_info = {
            'total': total_params,
            'total_M': total_params / 1e6,
            'flops': flops,
            'flops_G': flops / 1e9,
        }
        print(f"    Parameters: {vanilla_info['total_M']:.2f}M, FLOPs: {vanilla_info['flops_G']:.2f}G")

        # Performance (quick eval)
        print(f"    Running validation...")
        val_tokens = load_val_data(self.val_data_path, max_tokens=50 * 32 * 512)
        val_results = evaluate_model(comp_model, val_tokens, batch_size=32, seq_len=512, device=self.device)
        vanilla_val = val_results
        print(f"    PPL: {vanilla_val.get('perplexity', 0):.2f}, Acc: {vanilla_val.get('accuracy', 0):.1f}%")

        # Speed benchmark
        import time
        comp_model.eval()
        dummy = torch.randint(0, 1000, (1, 512)).to(self.device)
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                comp_model(dummy)
        # Benchmark
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start = time.time()
        for _ in range(20):
            with torch.no_grad():
                comp_model(dummy)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        elapsed = time.time() - start
        tokens_per_sec = (20 * 512) / elapsed
        vanilla_speed = {'tokens_per_sec': tokens_per_sec}
        print(f"    Speed: {tokens_per_sec/1000:.1f}K tok/s")

        # Cleanup
        del comp_model
        torch.cuda.empty_cache()

        return vanilla_info, vanilla_val, vanilla_speed

    def _extract_checkpoint_config(self, checkpoint_path: str) -> Dict:
        """Extract training config from a checkpoint file or directory."""
        config_data = {
            'model': {},
            'training': {},
            'optimizer': {},
        }

        try:
            # Handle directory paths (find actual checkpoint file)
            path = Path(checkpoint_path)
            checkpoint_dir = None

            if path.is_dir():
                checkpoint_dir = path
                pt_files = list(path.glob('*.pt'))
                found = False
                for f in pt_files:
                    if 'best' in f.name.lower() or 'final' in f.name.lower():
                        checkpoint_path = str(f)
                        found = True
                        break
                if not found and pt_files:
                    # Use most recent .pt file
                    checkpoint_path = str(sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1])
                elif not found:
                    raise FileNotFoundError(f"No .pt files found in {path}")
            else:
                checkpoint_dir = path.parent

            # Try to load config.json first (saved separately by train.py)
            config_json_path = checkpoint_dir / 'config.json'
            json_config = {}
            if config_json_path.exists():
                with open(config_json_path, 'r') as f:
                    json_config = json.load(f)
                    print(f"      Loaded config.json from {checkpoint_dir.name}")

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Model config: prefer config.json > checkpoint['config'] > checkpoint['model_config']
            model_config = json_config.get('model', {})
            if not model_config:
                model_config = checkpoint.get('config', checkpoint.get('model_config', {}))
            config_data['model'] = {
                'd_model': model_config.get('d_model', model_config.get('hidden_size', 0)),
                'n_layers': model_config.get('n_layers', model_config.get('num_hidden_layers', 0)),
                'n_heads': model_config.get('n_heads', model_config.get('num_attention_heads', 0)),
                'vocab_size': model_config.get('vocab_size', 0),
                'd_ff': model_config.get('d_ff', model_config.get('intermediate_size', 0)),
                'max_seq_len': model_config.get('max_seq_len', model_config.get('max_position_embeddings', 0)),
            }

            # Count parameters from state dict (exclude EMA/optimizer buffers)
            state_dict = checkpoint.get('model_state_dict', {})
            if not state_dict:
                # Fallback: try 'state_dict' but filter out optimizer/EMA keys
                raw_state = checkpoint.get('state_dict', {})
                state_dict = {
                    k: v for k, v in raw_state.items()
                    if not k.startswith('optimizer') and not k.startswith('ema_') and
                       not k.startswith('_') and hasattr(v, 'numel')
                }
            if state_dict:
                # Only count actual parameters (weight, bias, embeddings)
                param_keys = [k for k in state_dict.keys()
                             if any(x in k for x in ['.weight', '.bias', 'embedding', '_emb'])]
                if param_keys:
                    total_params = sum(state_dict[k].numel() for k in param_keys if hasattr(state_dict[k], 'numel'))
                else:
                    # Fallback: count all tensors
                    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
                config_data['model']['total_params'] = total_params
                config_data['model']['total_params_M'] = round(total_params / 1e6, 2)

            # Training config: prefer config.json > checkpoint
            training_config = json_config.get('training', {})
            if not training_config:
                training_config = checkpoint.get('training_config', checkpoint.get('train_config', {}))
            if not training_config:
                # Try to extract from top-level keys
                training_config = {
                    'batch_size': checkpoint.get('batch_size'),
                    'learning_rate': checkpoint.get('learning_rate', checkpoint.get('lr')),
                    'total_steps': checkpoint.get('total_steps', checkpoint.get('max_steps')),
                    'warmup_steps': checkpoint.get('warmup_steps'),
                    'dataset': checkpoint.get('dataset', checkpoint.get('data_path')),
                }

            config_data['training'] = {
                'dataset': training_config.get('dataset', training_config.get('data_path', 'unknown')),
                'batch_size': training_config.get('batch_size', training_config.get('train_batch_size', 0)),
                'learning_rate': training_config.get('learning_rate', training_config.get('lr', 0)),
                'total_steps': training_config.get('total_steps', training_config.get('max_steps', 0)),
                'warmup_steps': training_config.get('warmup_steps', 0),
                'epochs': training_config.get('epochs', training_config.get('num_epochs', 0)),
            }

            # Current step/epoch from checkpoint
            config_data['training']['current_step'] = checkpoint.get('step', checkpoint.get('global_step', 0))
            config_data['training']['current_epoch'] = checkpoint.get('epoch', 0)

            # Optimizer config
            optimizer_config = checkpoint.get('optimizer_config', {})
            if not optimizer_config and 'optimizer_state_dict' in checkpoint:
                # Try to infer from optimizer state
                opt_state = checkpoint['optimizer_state_dict']
                if 'param_groups' in opt_state and opt_state['param_groups']:
                    pg = opt_state['param_groups'][0]
                    optimizer_config = {
                        'lr': pg.get('lr'),
                        'weight_decay': pg.get('weight_decay'),
                        'betas': pg.get('betas'),
                    }

            config_data['optimizer'] = {
                'optimizer': training_config.get('optimizer', 'AdamW'),
                'weight_decay': optimizer_config.get('weight_decay', training_config.get('weight_decay', 0)),
                'grad_clip': training_config.get('grad_clip', training_config.get('max_grad_norm', 0)),
                'scheduler': training_config.get('scheduler', training_config.get('lr_scheduler', 'unknown')),
                'betas': optimizer_config.get('betas', [0.9, 0.999]),
            }

            # DAWN-specific config
            if 'n_feature_qk' in model_config or 'n_neurons' in model_config:
                config_data['dawn_specific'] = {
                    'n_feature_qk': model_config.get('n_feature_qk', 0),
                    'n_restore_qk': model_config.get('n_restore_qk', 0),
                    'n_feature_v': model_config.get('n_feature_v', 0),
                    'n_restore_v': model_config.get('n_restore_v', 0),
                    'n_knowledge': model_config.get('n_knowledge', 0),
                    'top_k': model_config.get('top_k', 0),
                }

            del checkpoint  # Free memory

        except Exception as e:
            config_data['error'] = str(e)

        return config_data

    def _extract_training_configs(self) -> Dict:
        """Extract and compare training configs from DAWN and Vanilla checkpoints."""
        result = {
            'dawn': {},
            'vanilla': {},
            'comparison': {},
        }

        # Extract DAWN config
        if self.checkpoint_path:
            print(f"    Extracting config from DAWN: {self.checkpoint_path}")
            result['dawn'] = self._extract_checkpoint_config(str(self.checkpoint_path))

        # Extract Vanilla config
        if self.compare_checkpoint:
            print(f"    Extracting config from Vanilla: {self.compare_checkpoint}")
            result['vanilla'] = self._extract_checkpoint_config(str(self.compare_checkpoint))

        # Build comparison table
        if result['dawn'] and result['vanilla']:
            dawn_m = result['dawn'].get('model', {})
            dawn_t = result['dawn'].get('training', {})
            dawn_o = result['dawn'].get('optimizer', {})
            van_m = result['vanilla'].get('model', {})
            van_t = result['vanilla'].get('training', {})
            van_o = result['vanilla'].get('optimizer', {})

            result['comparison'] = {
                # Model
                'd_model': {'dawn': dawn_m.get('d_model'), 'vanilla': van_m.get('d_model')},
                'n_layers': {'dawn': dawn_m.get('n_layers'), 'vanilla': van_m.get('n_layers')},
                'n_heads': {'dawn': dawn_m.get('n_heads'), 'vanilla': van_m.get('n_heads')},
                'params_M': {'dawn': dawn_m.get('total_params_M'), 'vanilla': van_m.get('total_params_M')},
                # Training
                'dataset': {'dawn': dawn_t.get('dataset'), 'vanilla': van_t.get('dataset')},
                'batch_size': {'dawn': dawn_t.get('batch_size'), 'vanilla': van_t.get('batch_size')},
                'learning_rate': {'dawn': dawn_t.get('learning_rate'), 'vanilla': van_t.get('learning_rate')},
                'total_steps': {'dawn': dawn_t.get('total_steps'), 'vanilla': van_t.get('total_steps')},
                'warmup_steps': {'dawn': dawn_t.get('warmup_steps'), 'vanilla': van_t.get('warmup_steps')},
                # Optimizer
                'optimizer': {'dawn': dawn_o.get('optimizer'), 'vanilla': van_o.get('optimizer')},
                'weight_decay': {'dawn': dawn_o.get('weight_decay'), 'vanilla': van_o.get('weight_decay')},
                'grad_clip': {'dawn': dawn_o.get('grad_clip'), 'vanilla': van_o.get('grad_clip')},
            }

            # Check if configs match
            matches = []
            mismatches = []
            for key, vals in result['comparison'].items():
                if vals['dawn'] == vals['vanilla']:
                    matches.append(key)
                elif vals['dawn'] is not None and vals['vanilla'] is not None:
                    mismatches.append(key)

            result['validation'] = {
                'matching_configs': matches,
                'mismatching_configs': mismatches,
                'is_comparable': len(mismatches) == 0 or all(
                    k in ['params_M', 'd_model', 'n_layers']  # These can differ
                    for k in mismatches
                ),
            }

        return result

    def _generate_training_comparison(self, paper_dir: Path):
        """Generate training_comparison.json and training_comparison.md."""
        config_data = self._extract_training_configs()

        # Save JSON
        json_path = paper_dir / 'training_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print(f"    Saved: {json_path}")

        # Generate Markdown table
        md_lines = [
            "# Training Configuration Comparison",
            "",
            "## Model Architecture",
            "",
            "| Config | DAWN | Vanilla |",
            "|--------|------|---------|",
        ]

        comparison = config_data.get('comparison', {})

        # Model section
        for key in ['d_model', 'n_layers', 'n_heads', 'params_M']:
            vals = comparison.get(key, {})
            dawn_val = vals.get('dawn', '-')
            van_val = vals.get('vanilla', '-')
            if key == 'params_M' and dawn_val and van_val:
                md_lines.append(f"| {key} | {dawn_val}M | {van_val}M |")
            else:
                md_lines.append(f"| {key} | {dawn_val} | {van_val} |")

        md_lines.extend([
            "",
            "## Training Settings",
            "",
            "| Config | DAWN | Vanilla |",
            "|--------|------|---------|",
        ])

        # Training section
        for key in ['dataset', 'batch_size', 'learning_rate', 'total_steps', 'warmup_steps']:
            vals = comparison.get(key, {})
            dawn_val = vals.get('dawn', '-')
            van_val = vals.get('vanilla', '-')
            # Format learning rate
            if key == 'learning_rate' and dawn_val and isinstance(dawn_val, float):
                dawn_val = f"{dawn_val:.0e}"
            if key == 'learning_rate' and van_val and isinstance(van_val, float):
                van_val = f"{van_val:.0e}"
            # Format steps with K
            if key in ['total_steps', 'warmup_steps'] and dawn_val and isinstance(dawn_val, (int, float)) and dawn_val >= 1000:
                dawn_val = f"{dawn_val/1000:.0f}K"
            if key in ['total_steps', 'warmup_steps'] and van_val and isinstance(van_val, (int, float)) and van_val >= 1000:
                van_val = f"{van_val/1000:.0f}K"
            md_lines.append(f"| {key} | {dawn_val} | {van_val} |")

        md_lines.extend([
            "",
            "## Optimizer Settings",
            "",
            "| Config | DAWN | Vanilla |",
            "|--------|------|---------|",
        ])

        # Optimizer section
        for key in ['optimizer', 'weight_decay', 'grad_clip']:
            vals = comparison.get(key, {})
            dawn_val = vals.get('dawn', '-')
            van_val = vals.get('vanilla', '-')
            md_lines.append(f"| {key} | {dawn_val} | {van_val} |")

        # Validation summary
        validation = config_data.get('validation', {})
        if validation:
            md_lines.extend([
                "",
                "## Validation",
                "",
                f"- **Comparable**: {'✅ Yes' if validation.get('is_comparable') else '⚠️ No'}",
                f"- **Matching configs**: {', '.join(validation.get('matching_configs', []))}",
            ])
            if validation.get('mismatching_configs'):
                md_lines.append(f"- **Mismatching configs**: {', '.join(validation['mismatching_configs'])}")

        # DAWN-specific
        dawn_specific = config_data.get('dawn', {}).get('dawn_specific', {})
        if dawn_specific:
            md_lines.extend([
                "",
                "## DAWN-Specific Config",
                "",
                "| Pool | Size |",
                "|------|------|",
            ])
            for key, val in dawn_specific.items():
                if val:
                    md_lines.append(f"| {key} | {val} |")

        md_lines.append("")
        md_lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Save Markdown
        md_path = paper_dir / 'training_comparison.md'
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"    Saved: {md_path}")

    def _generate_paper_results_json(self, paper_dir: Path):
        """Generate paper_results.json with all numeric data for paper figures."""
        from scripts.analysis.utils import convert_to_serializable

        paper_results = {}

        # Table 1: Model Statistics (DAWN + Vanilla comparison)
        model_info = self.results.get('model_info', {})
        perf = self.results.get('performance', {})
        val = perf.get('validation', {})
        speed = perf.get('speed', {})

        # DAWN stats
        dawn_stats = {
            'parameters_M': round(model_info.get('total_M', 0), 2),
            'flops_G': round(model_info.get('flops_G', 0), 2),
            'perplexity': round(val.get('perplexity', 0), 2),
            'accuracy': round(val.get('accuracy', 0), 2),
            'tokens_per_sec': round(speed.get('tokens_per_sec', 0), 0),
        }

        # Load Vanilla stats if compare_checkpoint exists
        vanilla_stats = {}
        if self.compare_checkpoint:
            vanilla_info, vanilla_val, vanilla_speed = {}, {}, {}
            comp_path = Path(self.compare_checkpoint)
            found_results = False

            # Try to load from pre-existing analysis results
            comp_dirs = [
                comp_path.parent / 'analysis',
                comp_path / 'analysis',
                comp_path.parent,
            ]
            for comp_dir in comp_dirs:
                if comp_dir.exists():
                    comp_params = comp_dir / 'model_info' / 'parameters.json'
                    comp_val_file = comp_dir / 'performance' / 'validation.json'
                    comp_speed_file = comp_dir / 'performance' / 'speed_benchmark.json'
                    if comp_params.exists():
                        with open(comp_params) as f:
                            vanilla_info = json.load(f)
                    if comp_val_file.exists():
                        with open(comp_val_file) as f:
                            vanilla_val = json.load(f)
                    if comp_speed_file.exists():
                        with open(comp_speed_file) as f:
                            vanilla_speed = json.load(f)
                    if vanilla_info or vanilla_val:
                        found_results = True
                        break

            # Fallback: run quick analysis on vanilla model
            if not found_results:
                print("    Running quick analysis on comparison model...")
                try:
                    vanilla_info, vanilla_val, vanilla_speed = self._analyze_comparison_model()
                except Exception as e:
                    print(f"    Warning: Could not analyze comparison model: {e}")

            vanilla_stats = {
                'parameters_M': round(vanilla_info.get('total_M', 0), 2),
                'flops_G': round(vanilla_info.get('flops_G', 0), 2),
                'perplexity': round(vanilla_val.get('perplexity', 0), 2),
                'accuracy': round(vanilla_val.get('accuracy', 0), 2),
                'tokens_per_sec': round(vanilla_speed.get('tokens_per_sec', 0), 0),
            }

        paper_results['table1_model_stats'] = {
            'dawn': dawn_stats,
            'vanilla': vanilla_stats if vanilla_stats else None,
        }

        # Table 2: Neuron Utilization
        health = self.results.get('health', {})
        ema = health.get('ema_distribution', {})
        paper_results['table2_neuron_util'] = {}
        for name, data in ema.items():
            if isinstance(data, dict) and 'total' in data:
                paper_results['table2_neuron_util'][data.get('display', name)] = {
                    'total': data['total'],
                    'active': data['active'],
                    'dead': data['dead'],
                    'active_ratio': round(data['active_ratio'], 3),
                    'gini': round(data.get('gini', 0), 3),
                }

        # Fig 3: Q/K Specialization (summary only, no raw arrays)
        routing = self.results.get('routing', {})
        qk_usage = routing.get('qk_usage', {})
        paper_results['fig3_qk_specialization'] = {}
        for pool_name, data in qk_usage.items():
            if isinstance(data, dict) and 'q_specialized' in data:
                paper_results['fig3_qk_specialization'][pool_name] = {
                    'correlation': round(data.get('correlation', 0), 3),
                    'q_specialized': data.get('q_specialized', 0),
                    'k_specialized': data.get('k_specialized', 0),
                    'shared': data.get('shared', 0),
                    'inactive': data.get('inactive', 0),
                    'n_neurons': data.get('n_neurons', 0),
                }

        # Fig 4: POS Specialization (summary only for paper)
        neuron_features = self.results.get('neuron_features', {})
        if neuron_features:
            specialized = neuron_features.get('specialized_neurons', {})
            summary = neuron_features.get('summary', {})
            n_specialized = summary.get('n_specialized', {})

            # Get POS-specialized neurons (specialized['pos'] is a list)
            pos_neurons = specialized.get('pos', [])
            total_neurons = summary.get('n_neurons', neuron_features.get('n_neurons_profiled', 0))
            n_specialized_total = n_specialized.get('pos', 0) if isinstance(n_specialized, dict) else len(pos_neurons)

            # Count neurons per actual POS tag
            per_pos = {}
            top_list = []
            for neuron_info in pos_neurons:
                if isinstance(neuron_info, dict):
                    pos_tag = neuron_info.get('specialized_for', 'unknown')
                    per_pos[pos_tag] = per_pos.get(pos_tag, 0) + 1
                    top_list.append({
                        'neuron_id': neuron_info.get('neuron', 0),
                        'pos': pos_tag,
                        'concentration': round(neuron_info.get('pct', 0), 1),
                    })

            # Sort by concentration and keep top 10
            top_list.sort(key=lambda x: x['concentration'], reverse=True)

            paper_results['fig4_pos_specialization'] = {
                'total_neurons_analyzed': total_neurons,
                'specialized_neurons': n_specialized_total,
                'specialized_ratio': round(n_specialized_total / total_neurons, 3) if total_neurons > 0 else 0,
                'per_pos': per_pos,
                'top_10': top_list[:10],
            }

        # Fig 5: Factual Heatmap (summary only, no neuron lists)
        factual = self.results.get('factual', {})
        if factual:
            # Extract per-target summary (counts only, no lists)
            per_target_summary = {}
            all_common_100 = set()
            all_common_80 = set()

            for target, data in factual.get('per_target', {}).items():
                if isinstance(data, dict):
                    common_100 = data.get('common_neurons_100', [])
                    common_80 = data.get('common_neurons_80', [])
                    all_common_100.update(common_100)
                    all_common_80.update(common_80)

                    per_target_summary[target] = {
                        'match_rate': round(data.get('match_rate', 0), 3),
                        'successful_runs': data.get('successful_runs', 0),
                        'total_runs': data.get('total_runs', 0),
                        'n_common_100': len(common_100),
                        'n_common_80': len(common_80),
                        'n_unique_neurons': data.get('total_unique_neurons', 0),
                    }

            paper_results['fig5_factual'] = {
                'pool_type': factual.get('pool_type', 'fv'),
                'n_all_common_100': len(all_common_100),
                'n_all_common_80': len(all_common_80),
                'per_target': per_target_summary,
            }

        # Fig 6: Training Dynamics - Extract config from checkpoints (summary only)
        fig6_config = self._extract_training_configs()
        # Keep only essential summary, remove detailed comparison
        paper_results['fig6_training_dynamics'] = {
            'dawn': {
                'params_M': fig6_config.get('dawn', {}).get('model', {}).get('total_params_M'),
                'd_model': fig6_config.get('dawn', {}).get('model', {}).get('d_model'),
                'n_layers': fig6_config.get('dawn', {}).get('model', {}).get('n_layers'),
                'batch_size': fig6_config.get('dawn', {}).get('training', {}).get('batch_size'),
                'learning_rate': fig6_config.get('dawn', {}).get('training', {}).get('learning_rate'),
            },
            'vanilla': {
                'params_M': fig6_config.get('vanilla', {}).get('model', {}).get('total_params_M'),
                'd_model': fig6_config.get('vanilla', {}).get('model', {}).get('d_model'),
                'n_layers': fig6_config.get('vanilla', {}).get('model', {}).get('n_layers'),
                'batch_size': fig6_config.get('vanilla', {}).get('training', {}).get('batch_size'),
                'learning_rate': fig6_config.get('vanilla', {}).get('training', {}).get('learning_rate'),
            } if fig6_config.get('vanilla') else None,
        }

        # Fig 7: Layer Contribution (with computed summary)
        layer_contrib = routing.get('layer_contribution', {})
        if layer_contrib:
            per_layer = layer_contrib.get('per_layer', {})

            # Build per_layer dict with rounded values
            per_layer_data = {}
            attention_ratios = []
            knowledge_ratios = []

            for layer, data in per_layer.items():
                att_ratio = data.get('attention_ratio', 50)
                know_ratio = data.get('knowledge_ratio', 50)
                per_layer_data[layer] = {
                    'attention_ratio': round(att_ratio, 1),
                    'knowledge_ratio': round(know_ratio, 1),
                }
                attention_ratios.append(att_ratio)
                knowledge_ratios.append(know_ratio)

            # Compute summary statistics
            import numpy as np
            att_arr = np.array(attention_ratios) if attention_ratios else np.array([50])
            know_arr = np.array(knowledge_ratios) if knowledge_ratios else np.array([50])

            paper_results['fig7_layer_contribution'] = {
                'per_layer': per_layer_data,
                'summary': {
                    'attention_mean': round(float(att_arr.mean()), 1),
                    'attention_std': round(float(att_arr.std()), 1),
                    'knowledge_mean': round(float(know_arr.mean()), 1),
                    'knowledge_std': round(float(know_arr.std()), 1),
                },
            }

        # === Appendix Data (Important) ===

        # Diversity Metrics - Dead neuron 문제 없음 증명
        diversity = health.get('diversity', {})
        if diversity:
            paper_results['appendix_diversity'] = {}
            for name, data in diversity.items():
                if isinstance(data, dict) and 'normalized_entropy' in data:
                    paper_results['appendix_diversity'][data.get('display', name)] = {
                        'normalized_entropy': round(data.get('normalized_entropy', 0), 3),
                        'effective_count': round(data.get('effective_count', 0), 1),
                        'coverage': round(data.get('coverage', 0), 3),
                    }
            overall = diversity.get('overall', {})
            if overall:
                paper_results['appendix_diversity']['overall'] = {
                    'diversity_score': round(overall.get('diversity_score', 0), 3),
                    'health': overall.get('health', 'unknown'),
                }

        # Probing Accuracy - Interpretability 강화
        behavioral = self.results.get('behavioral', {})
        probing = behavioral.get('probing', {})
        if probing:
            paper_results['appendix_probing'] = {}
            for pool, data in probing.items():
                if isinstance(data, dict) and 'accuracy' in data:
                    paper_results['appendix_probing'][pool] = {
                        'accuracy': round(data.get('accuracy', 0), 3),
                        'f1_macro': round(data.get('f1_macro', 0), 3),
                    }

        # Save JSON
        results_path = paper_dir / 'paper_results.json'
        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable(paper_results), f, indent=2)
        print(f"    Saved: {results_path}")

    def _generate_paper_summary(self, paper_dir: Path):
        """Generate paper summary markdown."""
        # Try to load from saved files if results are empty
        model_info = self.results.get('model_info', {})
        if not model_info:
            params_file = self.output_dir / 'model_info' / 'parameters.json'
            if params_file.exists():
                with open(params_file) as f:
                    model_info = json.load(f)

        perf = self.results.get('performance', {})
        if not perf:
            val_file = self.output_dir / 'performance' / 'validation.json'
            speed_file = self.output_dir / 'performance' / 'speed_benchmark.json'
            if val_file.exists():
                with open(val_file) as f:
                    perf['validation'] = json.load(f)
            if speed_file.exists():
                with open(speed_file) as f:
                    perf['speed'] = json.load(f)

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

    def _get_paper_analyses(self) -> List[tuple]:
        """Paper generation에 필요한 분석 목록 (Single Source of Truth)"""
        return [
            ('model_info', self.analyze_model_info, {}),
            ('performance', self.analyze_performance, {'n_batches': self.val_batches // 2}),
            ('health', self.analyze_health, {}),
            ('routing', self.analyze_routing, {'n_batches': self.n_batches // 2}),
            ('factual', self.analyze_factual, {
                'min_target_count': self.min_targets,
                'max_runs': self.max_runs,
                'pool_type': self.pool_type
            }),
        ]

    def _get_full_analyses(self) -> List[tuple]:
        """전체 분석 목록"""
        return [
            ('model_info', self.analyze_model_info, {}),
            ('performance', self.analyze_performance, {'n_batches': self.val_batches}),
            ('health', self.analyze_health, {}),
            ('routing', self.analyze_routing, {'n_batches': self.n_batches}),
            ('embedding', self.analyze_embedding, {'n_clusters': self.n_clusters}),
            ('neuron_embedding', self.analyze_neuron_embedding, {'n_batches': self.n_batches // 2}),
            ('semantic', self.analyze_semantic, {'n_batches': self.n_batches // 2}),
            ('pos', self.analyze_pos, {'max_sentences': self.max_sentences, 'pool_type': self.pool_type, 'target_layer': self.target_layer}),
            ('token_combination', self.analyze_token_combination, {'max_sentences': self.max_sentences, 'target_layer': self.target_layer}),
            ('neuron_features', self.analyze_neuron_features, {'max_sentences': self.max_sentences, 'target_layer': self.target_layer}),
            ('layerwise_semantic', self.analyze_layerwise_semantic, {'max_sentences': self.max_sentences // 4}),
            ('factual', self.analyze_factual, {
                'min_target_count': self.min_targets,
                'max_runs': self.max_runs,
                'pool_type': self.pool_type
            }),
            ('behavioral', self.analyze_behavioral, {'n_batches': self.n_batches // 2}),
            ('coselection', self.analyze_coselection, {'n_batches': self.n_batches // 2}),
            ('weight', self.analyze_weight, {}),
            ('v18', self.analyze_v18, {'n_batches': self.n_batches // 2}),
        ]

    def run_all(self, paper_only: bool = False, only: List[str] = None):
        """Run all analyses."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()

        # Get analysis list based on mode
        if paper_only:
            analyses = self._get_paper_analyses()
        else:
            analyses = self._get_full_analyses()

        # Filter by --only option
        if only:
            analyses = [(n, f, a) for n, f, a in analyses if n in only]

        # If --only paper, check existing results and run missing analyses
        if only and 'paper' in only and not analyses:
            print("\n[Checking existing analysis results...]")
            self._load_existing_results()

            # Find missing analyses from paper requirements
            missing = []
            for name, func, kwargs in self._get_paper_analyses():
                if name not in self.results or not self.results[name]:
                    missing.append((name, func, kwargs))

            if missing:
                print(f"\n[Running {len(missing)} missing analyses for paper...]")
                analyses = missing
            else:
                print(f"  All required results found.")

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

        # Paper outputs (only if not filtering or 'paper' in filter)
        if not only or 'paper' in only:
            print(f"\n[PAPER] Generating paper outputs...")
            self.generate_paper_outputs()

        # Report (only if not filtering or 'report' in filter)
        if not only or 'report' in only:
            print(f"\n[{total_analyses+2}/{total_analyses+2}] REPORT")
            self.generate_report()

        # Final summary
        self._print_final_summary()

        return self.results

    def _load_existing_results(self):
        """Load previously saved analysis results from files."""
        print(f"  Loading from: {self.output_dir}")

        # Model info
        params_file = self.output_dir / 'model_info' / 'parameters.json'
        if params_file.exists():
            with open(params_file) as f:
                self.results['model_info'] = json.load(f)
                print(f"    ✓ model_info")

        # Performance
        val_file = self.output_dir / 'performance' / 'validation.json'
        speed_file = self.output_dir / 'performance' / 'speed_benchmark.json'
        if val_file.exists() or speed_file.exists():
            self.results['performance'] = {}
            if val_file.exists():
                with open(val_file) as f:
                    self.results['performance']['validation'] = json.load(f)
            if speed_file.exists():
                with open(speed_file) as f:
                    self.results['performance']['speed'] = json.load(f)
            print(f"    ✓ performance")

        # Health
        health_file = self.output_dir / 'health' / 'results.json'
        if health_file.exists():
            with open(health_file) as f:
                self.results['health'] = json.load(f)
                print(f"    ✓ health")

        # Routing
        routing_file = self.output_dir / 'routing' / 'results.json'
        if routing_file.exists():
            with open(routing_file) as f:
                self.results['routing'] = json.load(f)
                print(f"    ✓ routing")

        # Factual
        factual_file = self.output_dir / 'factual' / 'results.json'
        if factual_file.exists():
            with open(factual_file) as f:
                self.results['factual'] = json.load(f)
                print(f"    ✓ factual")

        # Neuron features
        nf_file = self.output_dir / 'neuron_features' / 'results.json'
        if nf_file.exists():
            with open(nf_file) as f:
                self.results['neuron_features'] = json.load(f)
                print(f"    ✓ neuron_features")

        if not self.results:
            print(f"    (No existing results found)")

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
        min_targets: int = 100,
        max_runs: int = 500,
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
        self.min_targets = min_targets
        self.max_runs = max_runs
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
                min_targets=self.min_targets,
                max_runs=self.max_runs,
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
    parser.add_argument('--compare_checkpoint', type=str, help='Comparison checkpoint for training dynamics (e.g., vanilla baseline)')
    parser.add_argument('--val_data', type=str, required=True, help='Validation data path')
    parser.add_argument('--output', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # Analysis mode
    parser.add_argument('--paper-only', action='store_true', help='Generate paper outputs only (faster)')
    parser.add_argument('--only', type=str, help='Run only specific analyses (comma-separated: model_info,performance,health,routing,embedding,semantic,pos,token_combination,neuron_features,layerwise_semantic,factual,behavioral,coselection,weight,v18,paper,report)')

    # Analysis parameters
    parser.add_argument('--n_batches', type=int, default=100, help='Number of batches for routing/semantic/behavioral/coselection (default: 100)')
    parser.add_argument('--val_batches', type=int, default=200, help='Number of batches for validation performance (default: 200)')
    parser.add_argument('--max_sentences', type=int, default=2000, help='Max sentences for POS analysis (default: 2000)')
    parser.add_argument('--min_targets', type=int, default=100, help='Min target occurrences for factual analysis (default: 100)')
    parser.add_argument('--max_runs', type=int, default=500, help='Max runs for factual analysis safety limit (default: 500)')
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size (default: 16)')
    parser.add_argument('--max_samples', type=int, default=5000, help='Max samples for dataloader (default: 5000)')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for embedding analysis (default: 5)')
    parser.add_argument('--pool_type', type=str, default='fv', help='Neuron pool type: fv, fqk, fqk_q, fqk_k, rv, rqk, rqk_q, rqk_k, fknow, rknow (default: fv)')
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
    print(f"min_targets: {args.min_targets}")
    print(f"max_runs: {args.max_runs}")
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
            min_targets=args.min_targets,
            max_runs=args.max_runs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            n_clusters=args.n_clusters,
            pool_type=args.pool_type,
            gen_tokens=args.gen_tokens,
            target_layer=args.target_layer,
            compare_checkpoint=args.compare_checkpoint,
        )
        analyzer.run_all(paper_only=args.paper_only, only=only)
    else:
        analyzer = MultiModelAnalyzer(
            checkpoint_paths, args.val_data,
            args.output, args.device,
            n_batches=args.n_batches,
            val_batches=args.val_batches,
            max_sentences=args.max_sentences,
            min_targets=args.min_targets,
            max_runs=args.max_runs,
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

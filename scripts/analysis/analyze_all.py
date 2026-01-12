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
                                   factual,behavioral,coselection,weight

    Analysis Parameters:
        --n_batches       Batches for routing/semantic/behavioral/coselection (default: 100)
        --val_batches     Batches for validation performance (default: 200)
        --max_sentences   Max sentences for POS analysis (default: 2000)
        --n_runs          Runs for factual analysis (default: 10)
        --batch_size      Dataloader batch size (default: 16)
        --max_samples     Max samples for dataloader (default: 5000)
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

        # Generation samples
        print("  Generating samples...")
        samples = self._generate_samples()

        print(f"  ┌─ Generation Samples ────────────────")
        for prompt, generated in samples:
            print(f"  │ Prompt: {prompt}")
            print(f"  │ → {generated}")
            print(f"  │")
        print(f"  └───────────────────────────────────────\n")

        with open(output_dir / 'generation_samples.txt', 'w') as f:
            for prompt, generated in samples:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {generated}\n")
                f.write("-" * 50 + "\n")

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

    def _generate_samples(self, max_new_tokens: int = 30, temperature: float = 0.8, top_k: int = 50) -> List[Tuple[str, str]]:
        """Generate text samples with top-k sampling."""
        import torch.nn.functional as F

        prompts = [
            "The capital of France is",
            "In the beginning",
            "Machine learning is",
        ]

        results = []
        self.model.eval()
        eos_token_id = self.tokenizer.sep_token_id

        for prompt in prompts:
            # Encode without special tokens for cleaner generation
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors='pt'
            ).to(self.device)
            generated = input_ids.clone()

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

                    # Stop if EOS token generated
                    if next_token.item() == eos_token_id:
                        break

            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            results.append((prompt, generated_text))

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

        # Print summary
        ema = results.get('ema_distribution', {})
        if ema:
            print(f"\n  ┌─ Neuron Health Summary ─────────────")
            total_active = 0
            total_neurons = 0
            for name, data in ema.items():
                if isinstance(data, dict) and 'total' in data:
                    total_active += data.get('active', 0)
                    total_neurons += data.get('total', 0)
                    print(f"  │ {data.get('display', name):12s}: {data['active']:4d}/{data['total']:4d} active ({data['active_ratio']*100:.1f}%)")
            if total_neurons > 0:
                print(f"  │ {'TOTAL':12s}: {total_active:4d}/{total_neurons:4d} active ({total_active/total_neurons*100:.1f}%)")
            print(f"  └───────────────────────────────────────\n")

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

        # Print summary
        entropy = results.get('entropy', {})
        if entropy:
            print(f"\n  ┌─ Routing Entropy Summary ────────────")
            for pool, data in entropy.items():
                if isinstance(data, dict) and 'mean' in data:
                    print(f"  │ {pool:12s}: mean={data['mean']:.3f}, std={data.get('std', 0):.3f}")
            print(f"  └───────────────────────────────────────\n")

        self.results['routing'] = results
        return results

    def analyze_embedding(self) -> Dict:
        """Analyze embeddings (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.embedding import EmbeddingAnalyzer

        output_dir = self.output_dir / 'embedding'
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Analyzing neuron embeddings...")
        analyzer = EmbeddingAnalyzer(self.model, device=self.device)
        results = analyzer.run_all(str(output_dir))

        # Print summary
        sim = results.get('similarity', {})
        if sim:
            print(f"\n  ┌─ Embedding Similarity Summary ───────")
            for pool, data in sim.items():
                if isinstance(data, dict) and 'mean' in data:
                    print(f"  │ {pool:12s}: mean={data['mean']:.3f}")
            print(f"  └───────────────────────────────────────\n")

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

        print("  Done.")
        self.results['semantic'] = results
        return results

    def analyze_pos(self, max_sentences: int = 2000) -> Dict:
        """Analyze POS neuron specialization (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.pos_neuron import POSNeuronAnalyzer

        output_dir = self.output_dir / 'pos'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing POS neuron specialization ({max_sentences} sentences)...")
        analyzer = POSNeuronAnalyzer(
            self.model, tokenizer=self.tokenizer, device=self.device
        )
        results = analyzer.run_all(str(output_dir), max_sentences=max_sentences)

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

    def analyze_factual(self, n_runs: int = 10) -> Dict:
        """Analyze factual knowledge neurons (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        from scripts.analysis.behavioral import BehavioralAnalyzer

        output_dir = self.output_dir / 'factual'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing factual neurons ({n_runs} runs per prompt)...")
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

        results = analyzer.analyze_factual_neurons(prompts, targets, n_runs=n_runs)

        # Print summary
        per_target = results.get('per_target', {})
        if per_target:
            print(f"\n  ┌─ Factual Analysis Results ────────────")
            for target, data in per_target.items():
                if isinstance(data, dict):
                    match_rate = data.get('match_rate', 0) * 100
                    n_common = len(data.get('common_neurons_80', []))
                    print(f"  │ {target:10s}: match={match_rate:.0f}%, common_neurons={n_common}")
            print(f"  └───────────────────────────────────────\n")

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

        print("  Done.")
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

        print("  Done.")
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

        print("  Done.")
        self.results['weight'] = results
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
            ('embedding', self.analyze_embedding, {}),
            ('semantic', self.analyze_semantic, {'n_batches': self.n_batches // 2}),
            ('pos', self.analyze_pos, {'max_sentences': self.max_sentences}),
            ('factual', self.analyze_factual, {'n_runs': self.n_runs}),
            ('behavioral', self.analyze_behavioral, {'n_batches': self.n_batches // 2}),
            ('coselection', self.analyze_coselection, {'n_batches': self.n_batches // 2}),
            ('weight', self.analyze_weight, {}),
        ]

        if paper_only:
            # Reduced params for faster paper-only mode
            analyses = [
                ('model_info', self.analyze_model_info, {}),
                ('performance', self.analyze_performance, {'n_batches': self.val_batches // 2}),
                ('health', self.analyze_health, {}),
                ('routing', self.analyze_routing, {'n_batches': self.n_batches // 2}),
                ('factual', self.analyze_factual, {'n_runs': max(5, self.n_runs // 2)}),
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
    parser.add_argument('--only', type=str, help='Run only specific analyses (comma-separated: health,routing,embedding,semantic,pos,factual,behavioral,coselection,weight)')

    # Analysis parameters
    parser.add_argument('--n_batches', type=int, default=100, help='Number of batches for routing/semantic/behavioral/coselection (default: 100)')
    parser.add_argument('--val_batches', type=int, default=200, help='Number of batches for validation performance (default: 200)')
    parser.add_argument('--max_sentences', type=int, default=2000, help='Max sentences for POS analysis (default: 2000)')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs for factual analysis (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size (default: 16)')
    parser.add_argument('--max_samples', type=int, default=5000, help='Max samples for dataloader (default: 5000)')

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
        )
        analyzer.run_all(paper_only=args.paper_only, only=only)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

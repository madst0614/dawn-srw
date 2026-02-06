#!/usr/bin/env python3
"""
DAWN Complete Analysis Tool (JAX/TPU Version)
==============================================
One-touch analysis for DAWN models trained with JAX/Flax on TPU.

Parallel to analyze_all.py but works with native .flax checkpoints
without PyTorch conversion.

Supports:
- Single checkpoint analysis with full report
- Multi-checkpoint comparison
- Paper-ready figures and tables
- Selective analysis modes

Usage:
    # Single analysis on TPU
    python scripts/analysis/analyze_all_jax.py \
        --checkpoint gs://bucket/checkpoints/run_xxx/best.flax \
        --val_data gs://bucket/c4_val.bin \
        --output results/

    # Local analysis
    python scripts/analysis/analyze_all_jax.py \
        --checkpoint checkpoints/dawn.flax \
        --val_data data/val.bin \
        --output results/

    # Paper-only (faster)
    python scripts/analysis/analyze_all_jax.py \
        --checkpoint checkpoints/dawn.flax \
        --val_data data/val.bin \
        --output results/ \
        --paper-only

    # Specific analyses
    python scripts/analysis/analyze_all_jax.py \
        --checkpoint checkpoints/dawn.flax \
        --val_data data/val.bin \
        --output results/ \
        --only health,routing,performance
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

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("WARNING: JAX not available. Install with: pip install jax[tpu]")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config,
    load_val_data_jax, create_batches,
    evaluate_jax, forward_jax,
    JAXRoutingDataExtractor, JAXRoutingData,
    get_shared_neurons_jax, get_neuron_embeddings_jax,
    gini_coefficient, calc_entropy, calc_entropy_ratio,
    convert_to_serializable, save_results,
    NEURON_TYPES, POOL_N_ATTR, POOL_DISPLAY_NAMES,
    _is_gcs_path,
)


class JAXModelAnalyzer:
    """Single model analyzer for JAX/Flax models."""

    def __init__(
        self,
        checkpoint_path: str,
        val_data_path: str,
        output_dir: str,
        # Analysis parameters
        n_batches: int = 100,
        val_batches: int = 200,
        batch_size: int = 32,
        seq_len: int = 512,
    ):
        self.checkpoint_path = checkpoint_path
        self.val_data_path = val_data_path
        self.output_dir = Path(output_dir)
        self.n_batches = n_batches
        self.val_batches = val_batches
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.model = None
        self.params = None
        self.config = None
        self.model_type = None  # 'dawn' or 'baseline'
        self.version = None
        self.name = None

        self.results = {}
        self._val_tokens = None
        self._extractor = None

    def load_model(self):
        """Load JAX model from .flax checkpoint."""
        self.model, self.params, self.config = load_model_jax(self.checkpoint_path)

        # Detect model type
        version = self.config.get('model_version', 'unknown')
        if version in ('baseline', 'baseline-JAX'):
            self.model_type = 'baseline'
            self.version = 'baseline'
        else:
            self.model_type = 'dawn'
            self.version = version

        # Extract name from path
        path_str = str(self.checkpoint_path)
        if _is_gcs_path(path_str):
            parts = path_str.rstrip('/').split('/')
            # Find run folder name
            for i, p in enumerate(parts):
                if p.startswith('run_'):
                    self.name = p
                    break
            if self.name is None:
                self.name = parts[-1].replace('.flax', '')
        else:
            path = Path(self.checkpoint_path)
            if path.is_dir():
                self.name = path.name
            else:
                self.name = path.parent.name if path.parent.name not in ['checkpoints', '.'] else path.stem

        print(f"Loaded: {self.name} ({self.model_type}, v{self.version})")

        # Create routing extractor for DAWN models
        if self.model_type == 'dawn':
            self._extractor = JAXRoutingDataExtractor(self.model, self.params, self.config)

    def _get_val_tokens(self):
        """Get or load validation tokens."""
        if self._val_tokens is None:
            max_tokens = self.val_batches * self.batch_size * self.seq_len
            self._val_tokens = load_val_data_jax(self.val_data_path, max_tokens)
        return self._val_tokens

    def analyze_model_info(self) -> Dict:
        """Analyze model parameters, architecture."""
        output_dir = self.output_dir / 'model_info'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        # Count parameters
        def count_params(params):
            return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))

        total_params = count_params(self.params)

        # Estimate FLOPs (simplified)
        d_model = self.config.get('d_model', 768)
        n_layers = self.config.get('n_layers', 16)
        seq_len = self.config.get('max_seq_len', 512)
        vocab_size = self.config.get('vocab_size', 30522)

        # Rough FLOPs estimate: 6 * params * seq_len (forward + backward)
        flops = 6 * total_params * seq_len

        params_info = {
            'total': int(total_params),
            'total_M': total_params / 1e6,
            'flops': int(flops),
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

        print(f"    Parameters: {total_params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")

        self.results['model_info'] = params_info
        return params_info

    def analyze_performance(self) -> Dict:
        """Analyze validation performance."""
        output_dir = self.output_dir / 'performance'
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Load validation tokens
        print("  [1/2] Loading validation data...")
        val_tokens = self._get_val_tokens()

        # Run evaluation
        print("  [2/2] Running validation...")
        model_instance = create_model_from_config(self.config)

        val_results = evaluate_jax(
            model_instance, self.params, self.config,
            val_tokens,
            batch_size=self.batch_size,
            seq_len=self.seq_len
        )
        results['validation'] = val_results

        # Print results
        print(f"\n  ┌─ Validation Results ─────────────────")
        print(f"  │ Loss:       {val_results.get('loss', 0):.4f}")
        print(f"  │ Perplexity: {val_results.get('perplexity', 0):.2f}")
        print(f"  │ Accuracy:   {val_results.get('accuracy', 0):.2f}%")
        print(f"  └───────────────────────────────────────\n")

        with open(output_dir / 'validation.json', 'w') as f:
            json.dump(val_results, f, indent=2)

        self.results['performance'] = results
        return results

    def analyze_health(self) -> Dict:
        """Analyze neuron health (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        output_dir = self.output_dir / 'health'
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Analyzing neuron health...")

        # Get shared neurons
        neurons = get_shared_neurons_jax(self.params)

        results = {
            'neuron_stats': {},
            'activation_distribution': {},
        }

        # Analyze each neuron pool
        pool_configs = [
            ('feature_qk', 'f_neurons', 0, self.config.get('n_feature_qk', 0)),
            ('feature_v', 'f_neurons', self.config.get('n_feature_qk', 0), None),
            ('restore_qk', 'r_neurons', 0, self.config.get('n_restore_qk', 0)),
            ('restore_v', 'r_neurons', self.config.get('n_restore_qk', 0), None),
            ('feature_know', 'feature_know', None, None),
            ('restore_know', 'restore_know', None, None),
        ]

        for pool_name, neuron_key, start_idx, end_idx in pool_configs:
            pool_neurons = neurons.get(neuron_key)
            if pool_neurons is None or len(pool_neurons) == 0:
                continue

            if start_idx is not None and end_idx is not None:
                pool_neurons = pool_neurons[start_idx:end_idx]
            elif start_idx is not None:
                pool_neurons = pool_neurons[start_idx:]

            # Compute neuron norms
            norms = np.linalg.norm(pool_neurons.reshape(len(pool_neurons), -1), axis=-1)

            results['neuron_stats'][pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name.replace('_', ''), pool_name),
                'count': len(pool_neurons),
                'mean_norm': float(norms.mean()),
                'std_norm': float(norms.std()),
                'min_norm': float(norms.min()),
                'max_norm': float(norms.max()),
            }

        # Analyze routing-based activation
        # Sample some batches to see which neurons are actually used
        val_tokens = self._get_val_tokens()
        batches = create_batches(val_tokens, self.batch_size, self.seq_len)[:min(self.n_batches, len(batches))]

        pool_activation_counts = {
            'fqk_q': np.zeros(self.config.get('n_feature_qk', 0)),
            'fqk_k': np.zeros(self.config.get('n_feature_qk', 0)),
            'fv': np.zeros(self.config.get('n_feature_v', 0)),
            'rqk_q': np.zeros(self.config.get('n_restore_qk', 0)),
            'rqk_k': np.zeros(self.config.get('n_restore_qk', 0)),
            'rv': np.zeros(self.config.get('n_restore_v', 0)),
            'fknow': np.zeros(self.config.get('n_feature_know', 0)),
            'rknow': np.zeros(self.config.get('n_restore_know', 0)),
        }

        print(f"  Processing {len(batches)} batches for activation analysis...")

        for batch in batches:
            routing = self._extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in pool_activation_counts.keys():
                weights = routing_data.get_weight(key)
                if weights is not None:
                    # Count non-zero activations
                    active_mask = weights > 1e-6
                    pool_activation_counts[key] += active_mask.sum(axis=(0, 1))

        # Compute activation statistics
        total_tokens = len(batches) * self.batch_size * self.seq_len

        for key, counts in pool_activation_counts.items():
            n_total = len(counts)
            if n_total == 0:
                continue

            active = (counts > 0).sum()
            dead = n_total - active
            gini = gini_coefficient(counts)

            results['activation_distribution'][key] = {
                'display': POOL_DISPLAY_NAMES.get(key, key),
                'total': int(n_total),
                'active': int(active),
                'dead': int(dead),
                'active_ratio': float(active / n_total) if n_total > 0 else 0,
                'gini': float(gini),
                'mean_activation': float(counts.mean()),
                'max_activation': float(counts.max()),
            }

        # Print summary
        print(f"\n  ┌─ Neuron Health Summary ──────────────────────────────")
        print(f"  │ {'Pool':<12} {'Active':>8} {'Dead':>8} {'Total':>8} {'Ratio':>8} {'Gini':>8}")
        print(f"  │ {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        for key, data in results['activation_distribution'].items():
            print(f"  │ {data['display']:<12} {data['active']:>8d} {data['dead']:>8d} "
                  f"{data['total']:>8d} {data['active_ratio']*100:>7.1f}% {data['gini']:>8.3f}")

        print(f"  └─────────────────────────────────────────────────────────")

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        self.results['health'] = results
        return results

    def analyze_routing(self) -> Dict:
        """Analyze routing patterns (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        output_dir = self.output_dir / 'routing'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Analyzing routing patterns ({self.n_batches} batches)...")

        val_tokens = self._get_val_tokens()
        batches = create_batches(val_tokens, self.batch_size, self.seq_len)[:min(self.n_batches, len(batches))]

        results = {
            'entropy': {},
            'selection_frequency': {},
            'selection_diversity': {},
        }

        # Pool configurations
        pools = ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow']

        # Initialize accumulators
        all_prefs = {key: [] for key in pools}
        selection_counts = {key: None for key in pools}

        print(f"  Processing batches...")
        for i, batch in enumerate(batches):
            if (i + 1) % 20 == 0:
                print(f"    Batch {i+1}/{len(batches)}")

            routing = self._extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in pools:
                pref = routing_data.get_pref(key)
                if pref is not None:
                    all_prefs[key].append(pref)

                    # Track selection counts
                    weights = routing_data.get_weight(key)
                    if weights is not None:
                        active_mask = weights > 1e-6
                        batch_counts = active_mask.sum(axis=(0, 1))
                        if selection_counts[key] is None:
                            selection_counts[key] = batch_counts
                        else:
                            selection_counts[key] += batch_counts

        # Compute entropy statistics
        for key in pools:
            prefs = all_prefs[key]
            if not prefs:
                continue

            prefs = np.concatenate(prefs, axis=0)  # [total_batch*seq, N]
            prefs = prefs.reshape(-1, prefs.shape[-1])

            # Token-level entropy
            entropies = calc_entropy(prefs, axis=-1)

            results['entropy'][key] = {
                'display': POOL_DISPLAY_NAMES.get(key, key),
                'mean_entropy': float(entropies.mean()),
                'std_entropy': float(entropies.std()),
                'min_entropy': float(entropies.min()),
                'max_entropy': float(entropies.max()),
                'entropy_ratio': calc_entropy_ratio(prefs),
            }

        # Selection frequency
        for key, counts in selection_counts.items():
            if counts is None:
                continue

            n_total = len(counts)
            total_selections = counts.sum()
            unique_selected = (counts > 0).sum()

            # Top-10 most selected
            top_indices = np.argsort(counts)[-10:][::-1]
            top10 = [(int(idx), int(counts[idx])) for idx in top_indices]

            results['selection_frequency'][key] = {
                'display': POOL_DISPLAY_NAMES.get(key, key),
                'total_selections': int(total_selections),
                'unique_selected': int(unique_selected),
                'coverage': float(unique_selected / n_total) if n_total > 0 else 0,
                'top10': top10,
            }

        # Selection diversity
        for key, counts in selection_counts.items():
            if counts is None:
                continue

            n_total = len(counts)
            union_count = (counts > 0).sum()

            results['selection_diversity'][key] = {
                'display': POOL_DISPLAY_NAMES.get(key, key),
                'union_count': int(union_count),
                'n_total': int(n_total),
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
            }

        # Print summary
        print(f"\n  ┌─ Routing Entropy ─────────────────────────────────────────")
        print(f"  │ {'Pool':<12} {'Mean':>10} {'Std':>10} {'Ratio':>10}")
        print(f"  │ {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
        for key, data in results['entropy'].items():
            print(f"  │ {data['display']:<12} {data['mean_entropy']:>10.3f} {data['std_entropy']:>10.3f} "
                  f"{data['entropy_ratio']:>9.1f}%")
        print(f"  └─────────────────────────────────────────────────────────────")

        print(f"\n  ┌─ Selection Coverage ─────────────────────────────────────")
        for key, data in results['selection_frequency'].items():
            print(f"  │ {data['display']}: {data['unique_selected']}/{len(selection_counts[key])} "
                  f"({data['coverage']*100:.1f}%)")
        print(f"  └─────────────────────────────────────────────────────────────")

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        self.results['routing'] = results
        return results

    def analyze_embedding(self) -> Dict:
        """Analyze neuron embeddings (DAWN only)."""
        if self.model_type != 'dawn':
            print("  Skipping (not DAWN model)")
            return {}

        output_dir = self.output_dir / 'embedding'
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Analyzing neuron embeddings...")

        # Get neuron embeddings
        emb = get_neuron_embeddings_jax(self.params)
        if emb is None:
            print("  No neuron embeddings found")
            return {}

        results = {
            'similarity': {},
            'pca_variance': {},
        }

        # Pool boundaries
        n_fqk = self.config.get('n_feature_qk', 0)
        n_fv = self.config.get('n_feature_v', 0)
        n_rqk = self.config.get('n_restore_qk', 0)
        n_rv = self.config.get('n_restore_v', 0)
        n_fk = self.config.get('n_feature_know', 0)
        n_rk = self.config.get('n_restore_know', 0)

        pools = {
            'feature_qk': emb[:n_fqk],
            'feature_v': emb[n_fqk:n_fqk+n_fv],
            'restore_qk': emb[n_fqk+n_fv:n_fqk+n_fv+n_rqk],
            'restore_v': emb[n_fqk+n_fv+n_rqk:n_fqk+n_fv+n_rqk+n_rv],
            'feature_know': emb[n_fqk+n_fv+n_rqk+n_rv:n_fqk+n_fv+n_rqk+n_rv+n_fk],
            'restore_know': emb[n_fqk+n_fv+n_rqk+n_rv+n_fk:],
        }

        # Compute within-pool similarity
        for pool_name, pool_emb in pools.items():
            if len(pool_emb) < 2:
                continue

            # Cosine similarity matrix
            sim_matrix = pool_emb @ pool_emb.T

            # Get upper triangle (excluding diagonal)
            n = len(pool_emb)
            upper_tri = sim_matrix[np.triu_indices(n, k=1)]

            results['similarity'][pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name.replace('_', ''), pool_name),
                'avg_similarity': float(upper_tri.mean()),
                'std_similarity': float(upper_tri.std()),
                'min_similarity': float(upper_tri.min()),
                'max_similarity': float(upper_tri.max()),
                'n_neurons': int(n),
            }

        # Print summary
        print(f"\n  ┌─ Embedding Similarity ─────────────────────────────────")
        print(f"  │ {'Pool':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  │ {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for key, data in results['similarity'].items():
            print(f"  │ {data['display']:<12} {data['avg_similarity']:>10.4f} {data['std_similarity']:>10.4f} "
                  f"{data['min_similarity']:>10.4f} {data['max_similarity']:>10.4f}")
        print(f"  └─────────────────────────────────────────────────────────────")

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        self.results['embedding'] = results
        return results

    def generate_paper_figures(self) -> Dict:
        """Generate paper-ready figures."""
        if not HAS_MATPLOTLIB:
            print("  Skipping figures (matplotlib not available)")
            return {}

        output_dir = self.output_dir / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        print("  Generating paper figures...")

        # Figure: Routing entropy distribution
        if 'routing' in self.results and 'entropy' in self.results['routing']:
            entropy_data = self.results['routing']['entropy']

            fig, ax = plt.subplots(figsize=(10, 6))

            pools = list(entropy_data.keys())
            means = [entropy_data[p]['mean_entropy'] for p in pools]
            stds = [entropy_data[p]['std_entropy'] for p in pools]
            labels = [entropy_data[p]['display'] for p in pools]

            x = np.arange(len(pools))
            ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Entropy (nats)')
            ax.set_title('Routing Entropy by Pool')
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'routing_entropy.png', dpi=300)
            plt.savefig(output_dir / 'routing_entropy.pdf')
            plt.close()

            results['routing_entropy'] = str(output_dir / 'routing_entropy.png')

        # Figure: Neuron activation distribution
        if 'health' in self.results and 'activation_distribution' in self.results['health']:
            act_data = self.results['health']['activation_distribution']

            fig, ax = plt.subplots(figsize=(10, 6))

            pools = list(act_data.keys())
            active_ratios = [act_data[p]['active_ratio'] * 100 for p in pools]
            labels = [act_data[p]['display'] for p in pools]

            x = np.arange(len(pools))
            colors = ['green' if r > 90 else 'orange' if r > 50 else 'red' for r in active_ratios]
            ax.bar(x, active_ratios, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Active Neurons (%)')
            ax.set_title('Neuron Utilization by Pool')
            ax.set_ylim(0, 105)
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'neuron_utilization.png', dpi=300)
            plt.savefig(output_dir / 'neuron_utilization.pdf')
            plt.close()

            results['neuron_utilization'] = str(output_dir / 'neuron_utilization.png')

        print(f"    Saved {len(results)} figures to {output_dir}")

        self.results['figures'] = results
        return results

    def generate_report(self) -> str:
        """Generate analysis report."""
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# DAWN Analysis Report",
            f"",
            f"**Model:** {self.name}",
            f"**Type:** {self.model_type} v{self.version}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
        ]

        # Model Info
        if 'model_info' in self.results:
            info = self.results['model_info']
            lines.extend([
                f"## Model Information",
                f"",
                f"- **Parameters:** {info['total_M']:.2f}M",
                f"- **FLOPs:** {info['flops_G']:.2f}G",
                f"",
            ])

        # Performance
        if 'performance' in self.results:
            perf = self.results['performance'].get('validation', {})
            lines.extend([
                f"## Performance",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Loss | {perf.get('loss', 0):.4f} |",
                f"| Perplexity | {perf.get('perplexity', 0):.2f} |",
                f"| Accuracy | {perf.get('accuracy', 0):.2f}% |",
                f"",
            ])

        # Health
        if 'health' in self.results and 'activation_distribution' in self.results['health']:
            lines.extend([
                f"## Neuron Health",
                f"",
                f"| Pool | Active | Dead | Total | Ratio | Gini |",
                f"|------|--------|------|-------|-------|------|",
            ])
            for key, data in self.results['health']['activation_distribution'].items():
                lines.append(
                    f"| {data['display']} | {data['active']} | {data['dead']} | "
                    f"{data['total']} | {data['active_ratio']*100:.1f}% | {data['gini']:.3f} |"
                )
            lines.append("")

        # Routing
        if 'routing' in self.results and 'entropy' in self.results['routing']:
            lines.extend([
                f"## Routing Analysis",
                f"",
                f"| Pool | Mean Entropy | Std | Ratio |",
                f"|------|--------------|-----|-------|",
            ])
            for key, data in self.results['routing']['entropy'].items():
                lines.append(
                    f"| {data['display']} | {data['mean_entropy']:.3f} | "
                    f"{data['std_entropy']:.3f} | {data['entropy_ratio']:.1f}% |"
                )
            lines.append("")

        report = '\n'.join(lines)

        with open(output_dir / 'report.md', 'w') as f:
            f.write(report)

        print(f"  Report saved to {output_dir / 'report.md'}")

        return report

    def run_all(self, analyses: List[str] = None, paper_only: bool = False) -> Dict:
        """Run all or selected analyses.

        Args:
            analyses: List of analyses to run (None = all)
            paper_only: Only run analyses needed for paper

        Returns:
            Dict with all results
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default analyses
        if analyses is None:
            if paper_only:
                analyses = ['model_info', 'performance', 'health', 'routing', 'paper']
            else:
                analyses = ['model_info', 'performance', 'health', 'routing', 'embedding', 'paper', 'report']

        analysis_methods = {
            'model_info': ('Model Info', self.analyze_model_info),
            'performance': ('Performance', self.analyze_performance),
            'health': ('Neuron Health', self.analyze_health),
            'routing': ('Routing', self.analyze_routing),
            'embedding': ('Embedding', self.analyze_embedding),
            'paper': ('Paper Figures', self.generate_paper_figures),
            'report': ('Report', self.generate_report),
        }

        print(f"\n{'='*60}")
        print(f"DAWN Analysis: {self.name}")
        print(f"{'='*60}\n")

        for analysis in analyses:
            if analysis not in analysis_methods:
                print(f"  Unknown analysis: {analysis}")
                continue

            name, method = analysis_methods[analysis]
            print(f"[{name}]")
            try:
                method()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            print()

        # Save all results
        with open(self.output_dir / 'all_results.json', 'w') as f:
            json.dump(convert_to_serializable(self.results), f, indent=2)

        print(f"{'='*60}")
        print(f"Analysis complete. Results saved to {self.output_dir}")
        print(f"{'='*60}")

        return self.results


def main():
    parser = argparse.ArgumentParser(description='DAWN Analysis Tool (JAX/TPU)')

    # Input/Output
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to .flax checkpoint (local or gs://)')
    parser.add_argument('--val_data', '-v', type=str, required=True,
                        help='Path to validation data (.bin or .pt)')
    parser.add_argument('--output', '-o', type=str, default='analysis_results',
                        help='Output directory')

    # Analysis Mode
    parser.add_argument('--paper-only', action='store_true',
                        help='Generate paper outputs only (faster)')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only specific analyses (comma-separated)')

    # Analysis Parameters
    parser.add_argument('--n_batches', type=int, default=100,
                        help='Batches for routing/health analysis')
    parser.add_argument('--val_batches', type=int, default=200,
                        help='Batches for validation performance')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')

    args = parser.parse_args()

    if not HAS_JAX:
        print("ERROR: JAX not available. Install with: pip install jax[tpu]")
        sys.exit(1)

    # Parse analyses
    analyses = None
    if args.only:
        analyses = [a.strip() for a in args.only.split(',')]

    # Create analyzer
    analyzer = JAXModelAnalyzer(
        checkpoint_path=args.checkpoint,
        val_data_path=args.val_data,
        output_dir=args.output,
        n_batches=args.n_batches,
        val_batches=args.val_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Load model
    analyzer.load_model()

    # Run analysis
    analyzer.run_all(analyses=analyses, paper_only=args.paper_only)


if __name__ == '__main__':
    main()

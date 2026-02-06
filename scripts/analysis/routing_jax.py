"""
Routing Pattern Analysis (JAX Version)
======================================
Analyze routing patterns in DAWN v17.1 models using JAX/Flax.

Includes:
- Routing entropy analysis
- Selection frequency analysis
- Selection diversity analysis
- Q/K overlap analysis
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from collections import Counter, defaultdict

from .utils_jax import (
    JAXRoutingDataExtractor, JAXRoutingData,
    calc_entropy, calc_entropy_ratio, gini_coefficient,
    POOL_N_ATTR, POOL_DISPLAY_NAMES,
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
)

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


# All routing keys for analysis
ALL_ROUTING_KEYS = {
    'fqk_q': ('Feature QK (Q)', 'fqk', 'q', 'fqk'),
    'fqk_k': ('Feature QK (K)', 'fqk', 'k', 'fqk'),
    'fv': ('Feature V', 'fv', None, 'fv'),
    'rqk_q': ('Restore QK (Q)', 'rqk', 'q', 'rqk'),
    'rqk_k': ('Restore QK (K)', 'rqk', 'k', 'rqk'),
    'rv': ('Restore V', 'rv', None, 'rv'),
    'fknow': ('Feature Know', 'fknow', None, 'fknow'),
    'rknow': ('Restore Know', 'rknow', None, 'rknow'),
}


class RoutingAnalyzerJAX:
    """Routing pattern analyzer for JAX DAWN models."""

    def __init__(self, model, params, config: Dict, extractor: JAXRoutingDataExtractor = None):
        """
        Initialize analyzer.

        Args:
            model: JAX DAWN model class
            params: FrozenDict of model parameters
            config: Model configuration dict
            extractor: JAXRoutingDataExtractor instance (created if None)
        """
        self.model = model
        self.params = params
        self.config = config
        self.extractor = extractor or JAXRoutingDataExtractor(model, params, config)

        # Pool configuration
        self.pools = list(ALL_ROUTING_KEYS.keys())

    def analyze_entropy(self, batches: List[np.ndarray], n_batches: int = 50) -> Dict:
        """
        Analyze routing entropy across batches.

        Args:
            batches: List of input batches [batch_size, seq_len]
            n_batches: Number of batches to process

        Returns:
            Dictionary with entropy statistics per routing key
        """
        entropy_data = {name: [] for name in self.pools}

        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Entropy', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in self.pools:
                pref = routing_data.get_pref(key)
                if pref is not None:
                    ent = calc_entropy_ratio(pref)
                    entropy_data[key].append(ent)

        results = {}
        for key, (display, _, _, pool) in ALL_ROUTING_KEYS.items():
            if entropy_data[key]:
                results[key] = {
                    'display': display,
                    'pool': pool,
                    'mean_entropy': float(np.mean(entropy_data[key])),
                    'std_entropy': float(np.std(entropy_data[key])),
                    'min_entropy': float(np.min(entropy_data[key])),
                    'max_entropy': float(np.max(entropy_data[key])),
                }

        return results

    def analyze_selection_frequency(self, batches: List[np.ndarray], n_batches: int = 50) -> Dict:
        """
        Analyze neuron selection frequency.

        Args:
            batches: List of input batches
            n_batches: Number of batches to process

        Returns:
            Dictionary with selection frequency statistics
        """
        # Accumulators for selection counts
        selection_counts = {}

        # Initialize counters
        for key in self.pools:
            n_neurons = self.config.get(POOL_N_ATTR.get(key, ''), 0)
            if n_neurons > 0:
                selection_counts[key] = np.zeros(n_neurons)

        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Selection', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in selection_counts.keys():
                weights = routing_data.get_weight(key)
                if weights is None:
                    continue

                # Count active neurons (weight > 0)
                active_mask = weights > 0
                counts = active_mask.sum(axis=(0, 1))
                selection_counts[key] += counts

        results = {}
        for key, (display, _, _, pool) in ALL_ROUTING_KEYS.items():
            if key not in selection_counts:
                continue

            counts = selection_counts[key]
            total = counts.sum()
            unique = int((counts > 0).sum())

            # Top-10 most selected
            top_indices = np.argsort(counts)[-10:][::-1]
            top10 = [(int(idx), int(counts[idx]), float(counts[idx]/total) if total > 0 else 0)
                     for idx in top_indices if counts[idx] > 0]

            n_total = len(counts)

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': int(total),
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': top10,
                'concentration': sum(c for _, c, _ in top10[:10]) / total if total > 0 else 0,
            }

        return results

    def analyze_selection_diversity(
        self,
        batches: List[np.ndarray],
        n_batches: int = 100,
        threshold: float = 0.01
    ) -> Dict:
        """
        Analyze selection diversity across batches.

        Measures how many unique neurons are selected across the entire dataset
        vs per-batch selection.

        Args:
            batches: List of input batches
            n_batches: Number of batches to process
            threshold: Weight threshold for "selected"

        Returns:
            Dictionary with diversity metrics
        """
        # Track ever-selected neurons
        union_selected = {}
        per_batch_counts = defaultdict(list)

        # Initialize
        for key in self.pools:
            n_neurons = self.config.get(POOL_N_ATTR.get(key, ''), 0)
            if n_neurons > 0:
                union_selected[key] = np.zeros(n_neurons, dtype=bool)

        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Diversity', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in union_selected.keys():
                weights = routing_data.get_weight(key)
                if weights is None:
                    continue

                # Track unique selections in this batch
                batch_selected = (weights > threshold).any(axis=(0, 1))
                batch_count = int(batch_selected.sum())
                per_batch_counts[key].append(batch_count)

                # Update union
                union_selected[key] |= batch_selected

        results = {}
        for key, (display, _, _, pool) in ALL_ROUTING_KEYS.items():
            if key not in union_selected:
                continue

            union = union_selected[key]
            n_total = len(union)
            union_count = int(union.sum())
            batch_counts = per_batch_counts[key]

            results[key] = {
                'display': display,
                'pool': pool,
                'union_count': union_count,
                'union_coverage': union_count / n_total if n_total > 0 else 0,
                'n_total': n_total,
                'avg_batch_count': float(np.mean(batch_counts)) if batch_counts else 0,
                'std_batch_count': float(np.std(batch_counts)) if batch_counts else 0,
            }

        return results

    def analyze_qk_overlap(self, batches: List[np.ndarray], n_batches: int = 50) -> Dict:
        """
        Analyze Q/K selection overlap in QK pools.

        Args:
            batches: List of input batches
            n_batches: Number of batches to process

        Returns:
            Dictionary with Q/K overlap statistics
        """
        qk_pairs = [
            ('fqk_q', 'fqk_k', 'Feature QK'),
            ('rqk_q', 'rqk_k', 'Restore QK'),
        ]

        overlap_data = {name: [] for _, _, name in qk_pairs}
        jaccard_data = {name: [] for _, _, name in qk_pairs}

        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Q/K Overlap', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for q_key, k_key, name in qk_pairs:
                q_weights = routing_data.get_weight(q_key)
                k_weights = routing_data.get_weight(k_key)

                if q_weights is None or k_weights is None:
                    continue

                # Calculate per-token overlap
                q_active = q_weights > 0  # [B, S, N]
                k_active = k_weights > 0

                # Intersection and union per token
                intersection = (q_active & k_active).sum(axis=-1)  # [B, S]
                union = (q_active | k_active).sum(axis=-1)

                # Jaccard similarity
                jaccard = intersection / (union + 1e-8)
                jaccard_data[name].append(float(jaccard.mean()))

                # Simple overlap ratio
                q_count = q_active.sum(axis=-1)
                overlap_ratio = intersection / (q_count + 1e-8)
                overlap_data[name].append(float(overlap_ratio.mean()))

        results = {}
        for _, _, name in qk_pairs:
            if overlap_data[name]:
                results[name] = {
                    'mean_overlap': float(np.mean(overlap_data[name])),
                    'std_overlap': float(np.std(overlap_data[name])),
                    'mean_jaccard': float(np.mean(jaccard_data[name])),
                    'std_jaccard': float(np.std(jaccard_data[name])),
                }

        return results

    def visualize_entropy(self, entropy_results: Dict, output_dir: str):
        """Generate entropy visualization."""
        if not HAS_MATPLOTLIB:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pools = list(entropy_results.keys())
        means = [entropy_results[p]['mean_entropy'] for p in pools]
        stds = [entropy_results[p]['std_entropy'] for p in pools]
        labels = [entropy_results[p]['display'] for p in pools]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(pools))
        ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Entropy Ratio (%)')
        ax.set_title('Routing Entropy by Pool')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'entropy_by_pool.png', dpi=300)
        plt.savefig(output_path / 'entropy_by_pool.pdf')
        plt.close()

    def run_all(
        self,
        batches: List[np.ndarray],
        output_dir: str,
        n_batches: int = 50
    ) -> Dict:
        """Run all routing analyses.

        Args:
            batches: List of input batches
            output_dir: Directory to save results
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Entropy analysis
        print("  [1/4] Analyzing routing entropy...")
        results['entropy'] = self.analyze_entropy(batches, n_batches)

        # Selection frequency
        print("  [2/4] Analyzing selection frequency...")
        results['selection_frequency'] = self.analyze_selection_frequency(batches, n_batches)

        # Selection diversity
        print("  [3/4] Analyzing selection diversity...")
        results['selection_diversity'] = self.analyze_selection_diversity(batches, n_batches)

        # Q/K overlap
        print("  [4/4] Analyzing Q/K overlap...")
        results['qk_overlap'] = self.analyze_qk_overlap(batches, n_batches)

        # Visualizations
        if results.get('entropy'):
            self.visualize_entropy(results['entropy'], str(output_path))

        # Save results
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

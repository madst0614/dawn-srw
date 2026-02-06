"""
Neuron Health Analysis (JAX Version)
====================================
Analyze neuron health status using JAX/Flax models.

All metrics computed from actual routing weights during inference,
using JAXRoutingDataExtractor.
"""

import numpy as np
from typing import Dict, Optional, List, Any
from pathlib import Path
import json

from .utils_jax import (
    JAXRoutingDataExtractor, JAXRoutingData,
    get_shared_neurons_jax, get_neuron_embeddings_jax,
    gini_coefficient, calc_entropy,
    create_batches, load_val_data_jax,
    POOL_N_ATTR, POOL_DISPLAY_NAMES,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x


class NeuronHealthAnalyzerJAX:
    """Forward-based neuron health analyzer for JAX models."""

    def __init__(self, model, params, config: Dict, extractor: JAXRoutingDataExtractor = None):
        """
        Args:
            model: JAX model class
            params: FrozenDict of model parameters
            config: Model configuration dict
            extractor: Optional pre-created JAXRoutingDataExtractor
        """
        self.model = model
        self.params = params
        self.config = config
        self.extractor = extractor or JAXRoutingDataExtractor(model, params, config)

        # Pool keys to analyze
        self.pools = ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow']

    def analyze_activation_distribution(
        self,
        batches: List[np.ndarray],
        n_batches: int = 50,
        threshold: float = 0.01
    ) -> Dict:
        """
        Analyze neuron activation distribution from forward passes.

        Args:
            batches: List of input batches [batch_size, seq_len]
            n_batches: Number of batches to process
            threshold: Weight threshold for "active" classification

        Returns:
            Dictionary with per-pool activation statistics
        """
        # Accumulators: pool -> array of activation counts
        activation_counts = {}
        total_tokens = 0

        # Initialize counters
        for key in self.pools:
            n_neurons = self.config.get(POOL_N_ATTR.get(key, ''), 0)
            if n_neurons > 0:
                activation_counts[key] = np.zeros(n_neurons)

        # Process batches
        batches_to_process = batches[:min(n_batches, len(batches))]
        for i, batch in enumerate(tqdm(batches_to_process, desc='Health Analysis', disable=not HAS_TQDM)):
            batch_tokens = batch.size
            total_tokens += batch_tokens

            # Extract routing weights
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in activation_counts.keys():
                weights = routing_data.get_weight(key)
                if weights is None:
                    continue

                # Count activations (weight > threshold)
                active = (weights > threshold).sum(axis=(0, 1))
                activation_counts[key] += active

        # Compute statistics
        results = {}
        for pool, counts in activation_counts.items():
            n_total = len(counts)
            n_active = int((counts > 0).sum())
            n_dead = n_total - n_active

            # Normalize to get activation frequency
            freq = counts / (total_tokens + 1e-8)

            results[pool] = {
                'display': POOL_DISPLAY_NAMES.get(pool, pool),
                'total': n_total,
                'active': n_active,
                'dead': n_dead,
                'active_ratio': n_active / n_total if n_total > 0 else 0,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'gini': float(gini_coefficient(freq)),
                'stats': {
                    'min_freq': float(freq.min()),
                    'max_freq': float(freq.max()),
                    'mean_freq': float(freq.mean()),
                    'std_freq': float(freq.std()),
                    'median_freq': float(np.median(freq)),
                },
                'total_tokens': total_tokens,
            }

        return results

    def analyze_dead_neurons(
        self,
        batches: List[np.ndarray],
        n_batches: int = 50,
        threshold: float = 0.01
    ) -> Dict:
        """
        Identify dead neurons from forward passes.

        Args:
            batches: List of input batches
            n_batches: Number of batches to process
            threshold: Weight threshold for activation

        Returns:
            Dictionary with dead neuron analysis
        """
        # Track which neurons were ever activated
        ever_activated = {}

        # Initialize
        for key in self.pools:
            n_neurons = self.config.get(POOL_N_ATTR.get(key, ''), 0)
            if n_neurons > 0:
                ever_activated[key] = np.zeros(n_neurons, dtype=bool)

        # Process batches
        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Dead Neuron Analysis', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in ever_activated.keys():
                weights = routing_data.get_weight(key)
                if weights is None:
                    continue

                # Mark neurons that were activated
                active = (weights > threshold).any(axis=(0, 1))
                ever_activated[key] |= active

        # Compile results
        results = {}
        total_dead = 0
        total_neurons = 0

        for pool, activated in ever_activated.items():
            n_total = len(activated)
            n_active = int(activated.sum())
            n_dead = n_total - n_active
            dead_ids = np.where(~activated)[0].tolist()

            total_dead += n_dead
            total_neurons += n_total

            results[pool] = {
                'display': POOL_DISPLAY_NAMES.get(pool, pool),
                'n_total': n_total,
                'n_active': n_active,
                'n_dead': n_dead,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'dead_neuron_ids': dead_ids,
            }

        results['summary'] = {
            'total_dead': total_dead,
            'total_neurons': total_neurons,
            'dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
        }

        return results

    def analyze_neuron_norms(self) -> Dict:
        """Analyze neuron weight norms from model parameters."""
        neurons = get_shared_neurons_jax(self.params)

        results = {}

        pool_configs = [
            ('fqk', 'f_neurons', 0, self.config.get('n_feature_qk', 0)),
            ('fv', 'f_neurons', self.config.get('n_feature_qk', 0), None),
            ('rqk', 'r_neurons', 0, self.config.get('n_restore_qk', 0)),
            ('rv', 'r_neurons', self.config.get('n_restore_qk', 0), None),
            ('fknow', 'feature_know', None, None),
            ('rknow', 'restore_know', None, None),
        ]

        for pool_name, neuron_key, start_idx, end_idx in pool_configs:
            pool_neurons = neurons.get(neuron_key)
            if pool_neurons is None or len(pool_neurons) == 0:
                continue

            if start_idx is not None and end_idx is not None:
                pool_neurons = pool_neurons[start_idx:end_idx]
            elif start_idx is not None:
                pool_neurons = pool_neurons[start_idx:]

            if len(pool_neurons) == 0:
                continue

            # Compute norms
            norms = np.linalg.norm(pool_neurons.reshape(len(pool_neurons), -1), axis=-1)

            results[pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name, pool_name),
                'count': len(pool_neurons),
                'mean_norm': float(norms.mean()),
                'std_norm': float(norms.std()),
                'min_norm': float(norms.min()),
                'max_norm': float(norms.max()),
                'median_norm': float(np.median(norms)),
            }

        return results

    def analyze_diversity(
        self,
        batches: List[np.ndarray],
        n_batches: int = 50
    ) -> Dict:
        """Analyze routing diversity (entropy-based metrics)."""
        # Accumulate routing preferences
        pool_prefs = {}

        for key in self.pools:
            n_neurons = self.config.get(POOL_N_ATTR.get(key, ''), 0)
            if n_neurons > 0:
                pool_prefs[key] = np.zeros(n_neurons)

        # Process batches
        batches_to_process = batches[:min(n_batches, len(batches))]
        for batch in tqdm(batches_to_process, desc='Diversity Analysis', disable=not HAS_TQDM):
            routing = self.extractor.extract_routing(batch)
            routing_data = JAXRoutingData(routing)

            for key in pool_prefs.keys():
                pref = routing_data.get_pref(key)
                if pref is not None:
                    # Sum preferences across batch and sequence
                    pool_prefs[key] += pref.sum(axis=(0, 1))

        # Compute diversity metrics
        results = {}
        for key, prefs in pool_prefs.items():
            n_total = len(prefs)
            if n_total == 0:
                continue

            # Normalize to probability distribution
            probs = prefs / (prefs.sum() + 1e-8)

            # Entropy
            ent = float(calc_entropy(probs))
            max_ent = np.log(n_total) if n_total > 1 else 1.0

            # Effective count (exponential of entropy)
            eff_count = float(np.exp(ent))

            # Coverage (non-zero neurons)
            coverage = (probs > 1e-8).sum() / n_total

            results[key] = {
                'display': POOL_DISPLAY_NAMES.get(key, key),
                'entropy': ent,
                'max_entropy': float(max_ent),
                'normalized_entropy': ent / max_ent if max_ent > 0 else 0,
                'effective_count': eff_count,
                'coverage': float(coverage),
                'n_total': n_total,
            }

        return results

    def run_all(
        self,
        batches: List[np.ndarray],
        output_dir: str,
        n_batches: int = 50
    ) -> Dict:
        """Run all health analyses.

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

        # Activation distribution
        print("  [1/4] Analyzing activation distribution...")
        results['activation_distribution'] = self.analyze_activation_distribution(
            batches, n_batches=n_batches
        )

        # Dead neurons
        print("  [2/4] Analyzing dead neurons...")
        results['dead_neurons'] = self.analyze_dead_neurons(
            batches, n_batches=n_batches
        )

        # Neuron norms
        print("  [3/4] Analyzing neuron norms...")
        results['neuron_norms'] = self.analyze_neuron_norms()

        # Diversity
        print("  [4/4] Analyzing routing diversity...")
        results['diversity'] = self.analyze_diversity(
            batches, n_batches=n_batches
        )

        # Save results
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

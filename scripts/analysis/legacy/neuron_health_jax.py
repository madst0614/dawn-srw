"""
Neuron Health Analysis (Forward-based) - JAX Version
=====================================================
Analyze neuron health status using forward pass data.

JAX/Flax compatible version for TPU analysis.

All metrics computed from actual routing weights during inference,
not from EMA statistics.

NOTE: This is a JAX port of neuron_health.py (354 lines).
"""

import os
import numpy as np
from typing import Dict, Optional
from collections import defaultdict

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

from .base_jax import BaseAnalyzerJAX
from .utils_jax import (
    ALL_ROUTING_KEYS, POOL_N_ATTR,
    gini_coefficient,
    create_batches,
    JAXRoutingData,
)


class NeuronHealthAnalyzerJAX(BaseAnalyzerJAX):
    """Forward-based neuron health analyzer (JAX version)."""

    def __init__(self, model, params, config: Dict):
        """
        Initialize analyzer.

        Args:
            model: JAX/Flax model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        super().__init__(model, params, config)

    def analyze_activation_distribution(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze neuron activation distribution from forward passes.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for "active" classification
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with per-pool activation statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Initialize accumulators
        activation_counts = {pool: np.zeros(n) for pool, n in pools.items() if n > 0}
        total_tokens = 0

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Health Analysis'):
            input_ids = np.array(batch)
            batch_tokens = input_ids.size
            total_tokens += batch_tokens

            # Extract routing data
            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # Process attention weights
            for key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                # Map key to pool
                if key.startswith('fqk'):
                    pool = 'feature_qk'
                elif key == 'fv':
                    pool = 'feature_v'
                elif key.startswith('rqk'):
                    pool = 'restore_qk'
                elif key == 'rv':
                    pool = 'restore_v'
                else:
                    continue

                if pool not in activation_counts:
                    continue

                # Count activations (weight > threshold)
                if weights.ndim == 3:  # [B, S, N]
                    active = (weights > threshold).astype(np.float32).sum(axis=(0, 1))
                else:  # [B, N]
                    active = (weights > threshold).astype(np.float32).sum(axis=0)

                activation_counts[pool] += active

            # Process knowledge weights
            for key in ['fknow', 'rknow']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                pool = 'feature_know' if key == 'fknow' else 'restore_know'
                if pool not in activation_counts:
                    continue

                if weights.ndim == 3:
                    active = (weights > threshold).astype(np.float32).sum(axis=(0, 1))
                else:
                    active = (weights > threshold).astype(np.float32).sum(axis=0)

                activation_counts[pool] += active

        # Compute statistics
        results = {}
        for pool, counts in activation_counts.items():
            n_total = len(counts)
            n_active = int((counts > 0).sum())
            n_dead = n_total - n_active

            # Normalize to get activation frequency
            freq = counts / (total_tokens + 1e-8)

            results[pool] = {
                'total': n_total,
                'active': n_active,
                'dead': n_dead,
                'active_ratio': n_active / n_total if n_total > 0 else 0,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'gini': gini_coefficient(freq),
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
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Identify dead neurons from forward passes.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for activation
            output_dir: Directory for visualization output
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with dead neuron analysis
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Track which neurons were ever activated
        ever_activated = {pool: np.zeros(n, dtype=bool) for pool, n in pools.items() if n > 0}

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Dead Neuron Analysis'):
            input_ids = np.array(batch)

            # Extract routing data
            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # Process attention weights
            for key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                # Map key to pool
                if key.startswith('fqk'):
                    pool = 'feature_qk'
                elif key == 'fv':
                    pool = 'feature_v'
                elif key.startswith('rqk'):
                    pool = 'restore_qk'
                elif key == 'rv':
                    pool = 'restore_v'
                else:
                    continue

                if pool not in ever_activated:
                    continue

                # Mark neurons that were activated
                if weights.ndim == 3:
                    active = (weights > threshold).any(axis=0).any(axis=0)
                else:
                    active = (weights > threshold).any(axis=0)

                ever_activated[pool] |= active

            # Process knowledge weights
            for key in ['fknow', 'rknow']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                pool = 'feature_know' if key == 'fknow' else 'restore_know'
                if pool not in ever_activated:
                    continue

                if weights.ndim == 3:
                    active = (weights > threshold).any(axis=0).any(axis=0)
                else:
                    active = (weights > threshold).any(axis=0)

                ever_activated[pool] |= active

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

        # Visualization (skip for JAX version - requires matplotlib)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # TODO: Add JAX-compatible visualization

        return results

    def analyze_diversity(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze neuron usage diversity from forward passes.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for activation
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with diversity metrics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Initialize accumulators
        activation_counts = {pool: np.zeros(n) for pool, n in pools.items() if n > 0}

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Diversity Analysis'):
            input_ids = np.array(batch)

            # Extract routing data
            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # Process attention weights
            for key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                # Map key to pool
                if key.startswith('fqk'):
                    pool = 'feature_qk'
                elif key == 'fv':
                    pool = 'feature_v'
                elif key.startswith('rqk'):
                    pool = 'restore_qk'
                elif key == 'rv':
                    pool = 'restore_v'
                else:
                    continue

                if pool not in activation_counts:
                    continue

                if weights.ndim == 3:
                    active = (weights > threshold).astype(np.float32).sum(axis=(0, 1))
                else:
                    active = (weights > threshold).astype(np.float32).sum(axis=0)

                activation_counts[pool] += active

            # Process knowledge weights
            for key in ['fknow', 'rknow']:
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                pool = 'feature_know' if key == 'fknow' else 'restore_know'
                if pool not in activation_counts:
                    continue

                if weights.ndim == 3:
                    active = (weights > threshold).astype(np.float32).sum(axis=(0, 1))
                else:
                    active = (weights > threshold).astype(np.float32).sum(axis=0)

                activation_counts[pool] += active

        # Compute diversity metrics
        results = {}
        entropies = []

        for pool, counts in activation_counts.items():
            n_total = len(counts)
            active_mask = counts > 0
            n_active = int(active_mask.sum())

            if n_active == 0:
                results[pool] = {
                    'n_active': 0,
                    'n_total': n_total,
                    'entropy': 0,
                    'normalized_entropy': 0,
                    'effective_count': 0,
                    'coverage': 0,
                }
                continue

            # Compute entropy from activation distribution
            active_counts = counts[active_mask]
            p = active_counts / active_counts.sum()

            entropy = -np.sum(p * np.log(p + 1e-8))
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            # Top-k concentration
            top5_indices = np.argsort(active_counts)[-min(5, n_active):]
            top5 = active_counts[top5_indices]
            top5_share = float(top5.sum() / active_counts.sum())

            results[pool] = {
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': top5_share,
                'gini': gini_coefficient(counts),
            }
            entropies.append(normalized_entropy)

        # Overall score
        overall = sum(entropies) / len(entropies) if entropies else 0
        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def run_all(
        self,
        val_tokens: np.ndarray,
        output_dir: str = './neuron_health',
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Run all neuron health analyses (forward-based).

        Args:
            val_tokens: Validation token array
            output_dir: Directory for outputs
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'activation_distribution': self.analyze_activation_distribution(
                val_tokens, n_batches, batch_size=batch_size, seq_len=seq_len
            ),
            'diversity': self.analyze_diversity(
                val_tokens, n_batches, batch_size=batch_size, seq_len=seq_len
            ),
            'dead_neurons': self.analyze_dead_neurons(
                val_tokens, n_batches, output_dir=output_dir, batch_size=batch_size, seq_len=seq_len
            ),
        }

        return results

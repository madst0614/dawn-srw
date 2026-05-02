"""
Routing Pattern Analysis - JAX Version
=======================================
Analyze routing patterns in DAWN v17.1 models.

JAX/Flax compatible version for TPU analysis.

Includes:
- Routing entropy analysis
- Selection frequency analysis
- Selection diversity analysis
- Q/K overlap analysis
- Q/K usage pattern analysis

NOTE: This is a JAX port of routing.py (2270 lines).
Core analysis methods are fully ported; visualization requires matplotlib.
"""

import os
import json
import numpy as np
from typing import Dict, Optional
from collections import Counter, defaultdict

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
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS, QK_POOLS,
    POOL_N_ATTR,
    calc_entropy_ratio, gini_coefficient,
    create_batches,
    JAXRoutingData,
)


class RoutingAnalyzerJAX(BaseAnalyzerJAX):
    """Routing pattern analyzer for DAWN v17.1+ (JAX version)."""

    def __init__(self, model, params, config: Dict):
        """
        Initialize analyzer.

        Args:
            model: DAWN JAX model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        super().__init__(model, params, config)

    def analyze_entropy(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze routing entropy across batches.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with entropy statistics per routing key
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        entropy_data = {name: [] for name in ROUTING_KEYS.keys()}

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Entropy Analysis'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for key in ROUTING_KEYS.keys():
                weights = routing.get_weight(key)
                if weights is not None:
                    ent = calc_entropy_ratio(weights)
                    entropy_data[key].append(ent)

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
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

    def analyze_selection_frequency(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze neuron selection frequency.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with selection frequency statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Pool sizes
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
        }

        # Initialize accumulators
        selection_counts = {}
        for key, (_, _, _, pool) in ROUTING_KEYS.items():
            n = pool_sizes.get(pool, 0)
            if n > 0:
                selection_counts[key] = np.zeros(n, dtype=np.int64)

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Selection Frequency'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for key in ROUTING_KEYS.keys():
                weights = routing.get_weight(key)
                if weights is None or key not in selection_counts:
                    continue

                # Count active neurons
                if weights.ndim == 3:  # [B, S, N]
                    counts = (weights > 0).astype(np.int64).sum(axis=(0, 1))
                else:  # [B, N]
                    counts = (weights > 0).astype(np.int64).sum(axis=0)

                selection_counts[key] += counts

        # Build results
        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            if key not in selection_counts:
                continue

            counts = selection_counts[key]
            total = int(counts.sum())
            unique = int((counts > 0).sum())
            n_total = len(counts)

            # Top-10 neurons
            top_indices = np.argsort(counts)[-10:][::-1]
            top10 = [(int(idx), int(counts[idx]), float(counts[idx]/total) if total > 0 else 0)
                     for idx in top_indices if counts[idx] > 0]

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': total,
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': top10,
                'concentration': sum(c for _, c, _ in top10) / total if total > 0 else 0,
            }

        return results

    def analyze_selection_diversity(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 100,
        threshold: float = 0.01,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze selection diversity across batches.

        Measures how many unique neurons are selected across the entire dataset
        vs per-batch selection.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for "selected"
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with diversity metrics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Pool sizes
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Union accumulators (ever-selected)
        union_arrays = {}
        per_batch_counts = defaultdict(list)

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Selection Diversity'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # Attention routing
            for key in ROUTING_KEYS.keys():
                pool = ROUTING_KEYS[key][3]
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                n_neurons = weights.shape[-1]
                if key not in union_arrays:
                    union_arrays[key] = np.zeros(n_neurons, dtype=bool)

                if weights.ndim == 3:
                    selected = (weights > threshold).any(axis=0).any(axis=0)
                else:
                    selected = (weights > threshold).any(axis=0)

                union_arrays[key] |= selected
                per_batch_counts[key].append(int(selected.sum()))

            # Knowledge routing
            for key in KNOWLEDGE_ROUTING_KEYS.keys():
                pool = KNOWLEDGE_ROUTING_KEYS[key][2]
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                n_neurons = weights.shape[-1]
                if key not in union_arrays:
                    union_arrays[key] = np.zeros(n_neurons, dtype=bool)

                if weights.ndim == 3:
                    selected = (weights > threshold).any(axis=0).any(axis=0)
                else:
                    selected = (weights > threshold).any(axis=0)

                union_arrays[key] |= selected
                per_batch_counts[key].append(int(selected.sum()))

        # Build results
        results = {}

        for key, union_arr in union_arrays.items():
            if key in ROUTING_KEYS:
                display, _, _, pool = ROUTING_KEYS[key]
            elif key in KNOWLEDGE_ROUTING_KEYS:
                display, _, pool = KNOWLEDGE_ROUTING_KEYS[key]
            else:
                continue

            n_total = pool_sizes.get(pool, len(union_arr))
            union_count = int(union_arr.sum())
            batch_counts = per_batch_counts[key]

            if len(batch_counts) > 0:
                per_batch_avg = float(np.mean(batch_counts))
                per_batch_std = float(np.std(batch_counts))
            else:
                per_batch_avg = 0
                per_batch_std = 0

            results[key] = {
                'display': display,
                'pool': pool,
                'n_total': n_total,
                'per_batch_avg': per_batch_avg,
                'per_batch_std': per_batch_std,
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        results['summary'] = {
            'n_batches_processed': len(batches),
            'n_layers': self.config.get('n_layers', 12),
            'total_keys_analyzed': len(union_arrays),
            'interpretation': (
                'High diversity_ratio (>2) = many neurons selected differently per batch\n'
                'Low diversity_ratio (~1) = same neurons always selected'
            )
        }

        return results

    def analyze_qk_overlap(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze Q/K routing overlap (Jaccard similarity).

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with Q/K overlap statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        overlap_data = {'fqk': [], 'rqk': []}

        def compute_jaccard(q_weights, k_weights):
            """Compute Jaccard similarity from weight tensors."""
            if q_weights.ndim == 3:
                q_flat = q_weights.reshape(-1, q_weights.shape[-1])
                k_flat = k_weights.reshape(-1, k_weights.shape[-1])
            else:
                q_flat = q_weights
                k_flat = k_weights

            q_active = (q_flat > 0).astype(float)
            k_active = (k_flat > 0).astype(float)

            intersection = (q_active * k_active).sum(axis=-1)
            union = ((q_active + k_active) > 0).astype(float).sum(axis=-1)

            jaccard = intersection / (union + 1e-8)
            return jaccard.tolist()

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Q/K Overlap'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # F-QK Q/K overlap
            fqk_q = routing.get_weight('fqk_q')
            fqk_k = routing.get_weight('fqk_k')
            if fqk_q is not None and fqk_k is not None:
                overlap_data['fqk'].extend(compute_jaccard(fqk_q, fqk_k))

            # R-QK Q/K overlap
            rqk_q = routing.get_weight('rqk_q')
            rqk_k = routing.get_weight('rqk_k')
            if rqk_q is not None and rqk_k is not None:
                overlap_data['rqk'].extend(compute_jaccard(rqk_q, rqk_k))

        results = {}
        for key in ['fqk', 'rqk']:
            if overlap_data[key]:
                mean_overlap = float(np.mean(overlap_data[key]))
                results[key] = {
                    'overlap_ratio': mean_overlap,
                    'jaccard': mean_overlap,
                    'std_overlap': float(np.std(overlap_data[key])),
                    'interpretation': (
                        'Q and K select similar neurons' if mean_overlap > 0.3
                        else 'Q and K select different neurons'
                    ),
                }

        return results

    def analyze_qk_usage(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 100,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze per-neuron Q/K selection counts.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with Q/K usage statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        results = {}

        for pool_name, pool_info in QK_POOLS.items():
            n_attr = pool_info['n_attr']
            n_neurons = self.config.get(n_attr.replace('n_', ''), 0)
            if n_neurons == 0:
                continue

            # Initialize count arrays
            q_counts = np.zeros(n_neurons, dtype=np.float64)
            k_counts = np.zeros(n_neurons, dtype=np.float64)
            batch_overlaps = []

            # Get standardized keys for this pool
            if pool_name == 'feature_qk':
                std_q_key, std_k_key = 'fqk_q', 'fqk_k'
            else:  # restore_qk
                std_q_key, std_k_key = 'rqk_q', 'rqk_k'

            batches = create_batches(val_tokens, batch_size, seq_len)
            if n_batches:
                batches = batches[:n_batches]

            for batch in tqdm(batches, desc=f'{pool_info["display"]} Q/K'):
                input_ids = np.array(batch)

                routing_info = self.extractor.extract_routing(input_ids)
                routing = JAXRoutingData(routing_info)

                w_q = routing.get_weight(std_q_key)
                w_k = routing.get_weight(std_k_key)

                if w_q is None or w_k is None:
                    continue

                # Count selections
                if w_q.ndim == 3:
                    selected_q = (w_q > 0).astype(float).sum(axis=(0, 1))
                    selected_k = (w_k > 0).astype(float).sum(axis=(0, 1))
                else:
                    selected_q = (w_q > 0).astype(float).sum(axis=0)
                    selected_k = (w_k > 0).astype(float).sum(axis=0)

                q_counts += selected_q
                k_counts += selected_k

                # Calculate batch overlap
                if w_q.ndim >= 2:
                    overlap = ((w_q > 0) & (w_k > 0)).astype(float)
                    active_q = (w_q > 0).astype(float).sum(axis=-1)
                    overlap_ratio = (overlap.sum(axis=-1) / (active_q + 1e-8)).mean()
                    batch_overlaps.append(float(overlap_ratio))

            # Compute statistics
            if q_counts.sum() > 0 and k_counts.sum() > 0:
                corr = float(np.corrcoef(q_counts, k_counts)[0, 1])
            else:
                corr = 0.0

            # Ratio-based specialization analysis
            total_usage = q_counts + k_counts
            q_ratio = np.zeros_like(q_counts)
            valid_mask = total_usage > 0
            q_ratio[valid_mask] = q_counts[valid_mask] / total_usage[valid_mask]

            # Classify by ratio
            q_specialized = int((q_ratio > 0.7).sum())
            k_specialized = int((q_ratio < 0.3).sum())
            shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7)).sum())

            # Sensitivity analysis
            sensitivity_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
            sensitivity_analysis = {}
            for t in sensitivity_thresholds:
                q_spec = int((q_ratio > t).sum())
                k_spec = int((q_ratio < (1 - t)).sum())
                shared_t = int(((q_ratio >= (1 - t)) & (q_ratio <= t)).sum())
                sensitivity_analysis[str(t)] = {
                    'q_specialized': q_spec,
                    'k_specialized': k_spec,
                    'shared': shared_t,
                    'total': n_neurons,
                }

            results[pool_name] = {
                'display': pool_info['display'],
                'n_neurons': n_neurons,
                'q_counts': q_counts.tolist(),
                'k_counts': k_counts.tolist(),
                'correlation': corr,
                'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
                'std_overlap': float(np.std(batch_overlaps)) if batch_overlaps else 0,
                'q_specialized': q_specialized,
                'k_specialized': k_specialized,
                'shared': shared,
                'q_total': int(q_counts.sum()),
                'k_total': int(k_counts.sum()),
                'q_ratio': q_ratio.tolist(),
                'specialization_thresholds': {
                    'q_specialized': 0.7,
                    'k_specialized': 0.3,
                },
                'sensitivity_analysis': sensitivity_analysis,
            }

        results['n_batches'] = n_batches
        return results

    def analyze_qk_entropy(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze Q/K routing entropy patterns.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with entropy statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        POOL_STD_KEYS = {
            'feature_qk': ('fqk_q', 'fqk_k'),
            'restore_qk': ('rqk_q', 'rqk_k'),
        }

        entropy_data = {pool: {'q': [], 'k': []} for pool in QK_POOLS.keys()}

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Q/K Entropy'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for pool_name, pool_info in QK_POOLS.items():
                std_q_key, std_k_key = POOL_STD_KEYS.get(pool_name, (None, None))
                if std_q_key is None:
                    continue

                w_q = routing.get_weight(std_q_key)
                w_k = routing.get_weight(std_k_key)

                if w_q is not None:
                    ent = calc_entropy_ratio(w_q)
                    entropy_data[pool_name]['q'].append(ent)

                if w_k is not None:
                    ent = calc_entropy_ratio(w_k)
                    entropy_data[pool_name]['k'].append(ent)

        # Build results
        results = {}
        for pool_name, pool_info in QK_POOLS.items():
            q_entropy = entropy_data[pool_name]['q']
            k_entropy = entropy_data[pool_name]['k']

            if q_entropy and k_entropy:
                results[pool_name] = {
                    'display': pool_info['display'],
                    'q_entropy_mean': float(np.mean(q_entropy)),
                    'q_entropy_std': float(np.std(q_entropy)),
                    'k_entropy_mean': float(np.mean(k_entropy)),
                    'k_entropy_std': float(np.std(k_entropy)),
                    'entropy_diff': float(np.mean(q_entropy) - np.mean(k_entropy)),
                }

        return results

    def analyze_activation_sparsity(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze per-token activation sparsity.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with sparsity statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        sparsity_stats = defaultdict(lambda: {
            'active_sum': 0, 'weight_sum': 0.0, 'count': 0, 'n_total': 0
        })

        # Pool sizes
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
        }

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Activation Sparsity'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for key, (display, _, _, pool) in ROUTING_KEYS.items():
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                n_total = pool_sizes.get(pool, weights.shape[-1])

                if weights.ndim == 3:
                    w_flat = weights.reshape(-1, weights.shape[-1])
                else:
                    w_flat = weights

                active_mask = w_flat > 0
                active_counts = active_mask.sum(axis=-1)
                weight_sums = (w_flat * active_mask.astype(float)).sum(axis=-1)

                sparsity_stats[key]['active_sum'] += int(active_counts.sum())
                sparsity_stats[key]['weight_sum'] += float(weight_sums.sum())
                sparsity_stats[key]['count'] += w_flat.shape[0]
                sparsity_stats[key]['n_total'] = n_total

        # Build results
        results = {}
        for key, stats in sparsity_stats.items():
            if stats['count'] == 0:
                continue

            avg_active = stats['active_sum'] / stats['count']
            n_total = stats['n_total']
            avg_weight_sum = stats['weight_sum'] / stats['count']

            results[key] = {
                'display': ROUTING_KEYS[key][0],
                'avg_active_per_token': avg_active,
                'n_total': n_total,
                'sparsity_ratio': 1 - (avg_active / n_total) if n_total > 0 else 0,
                'avg_weight_sum': avg_weight_sum,
                'active_ratio_pct': (avg_active / n_total * 100) if n_total > 0 else 0,
            }

        results['n_batches'] = n_batches
        return results

    def analyze_weight_concentration(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze weight concentration among selected neurons.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with concentration statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        concentration_stats = defaultdict(lambda: {
            'top1_sum': 0.0, 'top5_sum': 0.0, 'active_sum': 0, 'count': 0
        })

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Weight Concentration'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for key in ROUTING_KEYS.keys():
                weights = routing.get_weight(key)
                if weights is None:
                    continue

                if weights.ndim == 3:
                    w_flat = weights.reshape(-1, weights.shape[-1])
                else:
                    w_flat = weights

                # Sort each token's weights descending
                sorted_w = np.sort(w_flat, axis=-1)[:, ::-1]

                # Total weight per token
                total_w = sorted_w.sum(axis=-1)
                valid_mask = total_w > 0

                if not valid_mask.any():
                    continue

                # Top1 and Top5 ratios
                top1 = sorted_w[:, 0] / (total_w + 1e-8)
                top5 = sorted_w[:, :5].sum(axis=-1) / (total_w + 1e-8)

                # Active count per token
                active_counts = (w_flat > 0).sum(axis=-1)

                concentration_stats[key]['top1_sum'] += float(top1[valid_mask].sum())
                concentration_stats[key]['top5_sum'] += float(top5[valid_mask].sum())
                concentration_stats[key]['active_sum'] += int(active_counts[valid_mask].sum())
                concentration_stats[key]['count'] += int(valid_mask.sum())

        # Build results
        results = {}
        for key, stats in concentration_stats.items():
            if stats['count'] == 0:
                continue

            results[key] = {
                'display': ROUTING_KEYS[key][0],
                'top1_weight_ratio': stats['top1_sum'] / stats['count'],
                'top5_weight_ratio': stats['top5_sum'] / stats['count'],
                'avg_active_neurons': stats['active_sum'] / stats['count'],
            }

        results['n_batches'] = n_batches
        return results

    def analyze_layer_contribution(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze layer-wise contribution to output.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with layer contribution statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        n_layers = self.config.get('n_layers', 12)
        layer_norms = {i: [] for i in range(n_layers)}

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Layer Contribution'):
            input_ids = np.array(batch)

            # Forward pass to get hidden states
            rng_key = jax.random.PRNGKey(0)
            result = self.model_instance.apply(
                self.params,
                jnp.array(input_ids),
                deterministic=True,
                rngs={'dropout': rng_key}
            )

            # Get hidden states if available
            if 'hidden_states' in result:
                for i, h in enumerate(result['hidden_states']):
                    if i < n_layers:
                        norm = float(np.linalg.norm(np.array(h)))
                        layer_norms[i].append(norm)

        # Compute statistics
        results = {}
        for layer_idx, norms in layer_norms.items():
            if norms:
                results[f'L{layer_idx}'] = {
                    'mean_norm': float(np.mean(norms)),
                    'std_norm': float(np.std(norms)),
                    'contribution_pct': 0,  # Computed below
                }

        # Compute relative contribution
        total_mean = sum(r['mean_norm'] for r in results.values())
        if total_mean > 0:
            for layer_key in results:
                results[layer_key]['contribution_pct'] = results[layer_key]['mean_norm'] / total_mean * 100

        results['n_batches'] = n_batches
        return results

    def analyze_qk_union_coverage(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 100,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze Q/K union coverage over batches.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with Q/K union coverage statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'restore_qk': self.n_restore_qk,
        }

        q_unions = {}
        k_unions = {}

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Q/K Union Coverage'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for pool_name in ['feature_qk', 'restore_qk']:
                prefix = 'fqk' if pool_name == 'feature_qk' else 'rqk'
                q_key, k_key = f'{prefix}_q', f'{prefix}_k'

                w_q = routing.get_weight(q_key)
                w_k = routing.get_weight(k_key)

                if w_q is None or w_k is None:
                    continue

                n_neurons = pool_sizes.get(pool_name, 0)
                if n_neurons == 0:
                    continue

                if pool_name not in q_unions:
                    q_unions[pool_name] = np.zeros(n_neurons, dtype=bool)
                    k_unions[pool_name] = np.zeros(n_neurons, dtype=bool)

                if w_q.ndim == 3:
                    q_selected = (w_q > 0).any(axis=0).any(axis=0)
                    k_selected = (w_k > 0).any(axis=0).any(axis=0)
                else:
                    q_selected = (w_q > 0).any(axis=0)
                    k_selected = (w_k > 0).any(axis=0)

                q_unions[pool_name] |= q_selected
                k_unions[pool_name] |= k_selected

        results = {}
        for pool_name in pool_sizes:
            if pool_name not in q_unions:
                continue

            n_total = pool_sizes[pool_name]
            q_count = int(q_unions[pool_name].sum())
            k_count = int(k_unions[pool_name].sum())
            union = q_unions[pool_name] | k_unions[pool_name]
            union_count = int(union.sum())
            intersection = q_unions[pool_name] & k_unions[pool_name]
            intersect_count = int(intersection.sum())

            results[pool_name] = {
                'q_coverage': q_count / n_total if n_total > 0 else 0,
                'k_coverage': k_count / n_total if n_total > 0 else 0,
                'union_coverage': union_count / n_total if n_total > 0 else 0,
                'intersection_coverage': intersect_count / n_total if n_total > 0 else 0,
                'q_unique': q_count,
                'k_unique': k_count,
                'union_count': union_count,
                'intersection_count': intersect_count,
                'n_total': n_total,
            }

        results['n_batches'] = n_batches
        return results

    def analyze_path_usage(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze path usage for multi-path routing (v18+).

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with path usage statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        max_paths = self.config.get('max_paths', 1)
        if max_paths <= 1:
            return {'note': 'Single-path model, path usage analysis not applicable'}

        path_counts = np.zeros(max_paths, dtype=np.int64)
        total_selections = 0

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Path Usage'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)

            # Look for path selection info
            if 'path_indices' in routing_info:
                paths = routing_info['path_indices']
                for p in range(max_paths):
                    path_counts[p] += (paths == p).sum()
                total_selections += paths.size

        if total_selections == 0:
            return {'note': 'No path selection data available'}

        results = {
            'max_paths': max_paths,
            'path_distribution': (path_counts / total_selections).tolist(),
            'path_counts': path_counts.tolist(),
            'total_selections': int(total_selections),
            'entropy': float(-np.sum((path_counts / total_selections) * np.log(path_counts / total_selections + 1e-8))),
        }
        results['n_batches'] = n_batches
        return results

    def analyze_token_coselection(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze token-level co-selection patterns.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with co-selection statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Track co-selection between Feature and Restore pools
        coselect_counts = defaultdict(lambda: {'both': 0, 'feature_only': 0, 'restore_only': 0, 'total': 0})

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Token Coselection'):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            # Compare FQK vs RQK selection
            fqk_w = routing.get_weight('fqk_q')
            rqk_w = routing.get_weight('rqk_q')

            if fqk_w is not None and rqk_w is not None:
                if fqk_w.ndim == 3:
                    fqk_active = (fqk_w > 0).any(axis=-1)  # [B, S]
                    rqk_active = (rqk_w > 0).any(axis=-1)  # [B, S]
                else:
                    fqk_active = (fqk_w > 0).any(axis=-1)
                    rqk_active = (rqk_w > 0).any(axis=-1)

                both = (fqk_active & rqk_active).sum()
                fqk_only = (fqk_active & ~rqk_active).sum()
                rqk_only = (~fqk_active & rqk_active).sum()
                total = fqk_active.size

                coselect_counts['qk']['both'] += int(both)
                coselect_counts['qk']['feature_only'] += int(fqk_only)
                coselect_counts['qk']['restore_only'] += int(rqk_only)
                coselect_counts['qk']['total'] += int(total)

            # Compare FV vs RV selection
            fv_w = routing.get_weight('fv')
            rv_w = routing.get_weight('rv')

            if fv_w is not None and rv_w is not None:
                if fv_w.ndim == 3:
                    fv_active = (fv_w > 0).any(axis=-1)
                    rv_active = (rv_w > 0).any(axis=-1)
                else:
                    fv_active = (fv_w > 0).any(axis=-1)
                    rv_active = (rv_w > 0).any(axis=-1)

                both = (fv_active & rv_active).sum()
                fv_only = (fv_active & ~rv_active).sum()
                rv_only = (~fv_active & rv_active).sum()
                total = fv_active.size

                coselect_counts['v']['both'] += int(both)
                coselect_counts['v']['feature_only'] += int(fv_only)
                coselect_counts['v']['restore_only'] += int(rv_only)
                coselect_counts['v']['total'] += int(total)

        results = {}
        for key, counts in coselect_counts.items():
            total = counts['total']
            if total > 0:
                results[key] = {
                    'both_pct': counts['both'] / total * 100,
                    'feature_only_pct': counts['feature_only'] / total * 100,
                    'restore_only_pct': counts['restore_only'] / total * 100,
                    'counts': dict(counts),
                }

        results['n_batches'] = n_batches
        return results

    def analyze_coverage_progression(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze how neuron coverage increases over batches.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with coverage progression data
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
        }

        # Track cumulative coverage
        cumulative = {pool: np.zeros(n, dtype=bool) for pool, n in pool_sizes.items() if n > 0}
        progression = {pool: [] for pool in cumulative}

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch_idx, batch in enumerate(tqdm(batches, desc='Coverage Progression')):
            input_ids = np.array(batch)

            routing_info = self.extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            for key in ROUTING_KEYS.keys():
                pool = ROUTING_KEYS[key][3]
                weights = routing.get_weight(key)
                if weights is None or pool not in cumulative:
                    continue

                if weights.ndim == 3:
                    selected = (weights > 0).any(axis=0).any(axis=0)
                else:
                    selected = (weights > 0).any(axis=0)

                cumulative[pool] |= selected

            # Record coverage at this batch
            for pool in cumulative:
                coverage = cumulative[pool].sum() / len(cumulative[pool])
                progression[pool].append(float(coverage))

        results = {
            'progression': {pool: prog for pool, prog in progression.items()},
            'final_coverage': {pool: float(cumulative[pool].sum() / len(cumulative[pool]))
                              for pool in cumulative},
            'n_batches': len(batches),
        }

        return results

    def run_all(
        self,
        val_tokens: np.ndarray,
        output_dir: str = './routing_analysis',
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Run all routing analyses.

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

        print("Running routing analysis...")

        results = {
            'entropy': self.analyze_entropy(val_tokens, n_batches, batch_size, seq_len),
            'selection_frequency': self.analyze_selection_frequency(val_tokens, n_batches, batch_size, seq_len),
            'selection_diversity': self.analyze_selection_diversity(val_tokens, n_batches, batch_size=batch_size, seq_len=seq_len),
            'qk_overlap': self.analyze_qk_overlap(val_tokens, n_batches, batch_size, seq_len),
            'qk_usage': self.analyze_qk_usage(val_tokens, n_batches, batch_size, seq_len),
            'qk_entropy': self.analyze_qk_entropy(val_tokens, n_batches, batch_size, seq_len),
            'activation_sparsity': self.analyze_activation_sparsity(val_tokens, n_batches, batch_size, seq_len),
            'weight_concentration': self.analyze_weight_concentration(val_tokens, n_batches, batch_size, seq_len),
            'layer_contribution': self.analyze_layer_contribution(val_tokens, n_batches, batch_size, seq_len),
            'qk_union_coverage': self.analyze_qk_union_coverage(val_tokens, n_batches, batch_size, seq_len),
            'path_usage': self.analyze_path_usage(val_tokens, n_batches, batch_size, seq_len),
            'token_coselection': self.analyze_token_coselection(val_tokens, n_batches, batch_size, seq_len),
            'coverage_progression': self.analyze_coverage_progression(val_tokens, n_batches, batch_size, seq_len),
        }

        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            from .utils_jax import convert_to_serializable
            json.dump(convert_to_serializable(results), f, indent=2)

        return results

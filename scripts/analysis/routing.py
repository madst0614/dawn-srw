"""
Routing Pattern Analysis
=========================
Analyze routing patterns in DAWN v17.1 models.

Includes:
- Routing entropy analysis
- Selection frequency analysis
- Selection diversity analysis
- Q/K overlap analysis
- Q/K usage pattern analysis (from analyze_dawn_qk.py)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import Counter, defaultdict

from .base import BaseAnalyzer
from .utils import (
    NEURON_TYPES, NEURON_TYPES_V18, ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS, QK_POOLS,
    POOL_N_ATTR,  # Direct pool_type to n_attr mapping
    calc_entropy_ratio, gini_coefficient,
    get_batch_input_ids,
    RoutingDataExtractor,  # New extraction layer
    HAS_MATPLOTLIB, HAS_TQDM, tqdm, plt
)


class RoutingAnalyzer(BaseAnalyzer):
    """Routing pattern analyzer for DAWN v17.1+."""

    def __init__(self, model, router=None, device='cuda', extractor=None):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance (auto-detected if None)
            device: Device for computation
            extractor: RoutingDataExtractor instance (created if None)
        """
        super().__init__(model, router=router, device=device)
        self.extractor = extractor or RoutingDataExtractor(model, device=device)

    def analyze_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze routing entropy across batches.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with entropy statistics per routing key
        """
        entropy_data = {name: [] for name in ROUTING_KEYS.keys()}

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Entropy')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    # Use standardized weight access (key matches ROUTING_KEYS)
                    for key in ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
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

    def analyze_selection_frequency(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze neuron selection frequency.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            threshold: Weight threshold for "selected" (for soft gating models)

        Returns:
            Dictionary with selection frequency statistics
        """
        # Use tensor accumulators instead of Counter for speed
        selection_tensors = {}  # key -> tensor of counts

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Selection')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    # Use standardized weight access
                    for key in ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        # Get pool size for initialization
                        pool = ROUTING_KEYS[key][3]
                        n_attr = POOL_N_ATTR.get(pool)
                        n_neurons = getattr(self.router, n_attr, 0) if n_attr else weights.shape[-1]

                        # Initialize tensor if needed
                        if key not in selection_tensors:
                            selection_tensors[key] = torch.zeros(n_neurons, device=self.device, dtype=torch.long)

                        # Vectorized count: sum active neurons across batch/seq dims
                        active_mask = weights > 0
                        if active_mask.dim() == 3:  # [B, S, N]
                            counts = active_mask.sum(dim=[0, 1])
                        else:  # [B, N]
                            counts = active_mask.sum(dim=0)
                        selection_tensors[key] += counts

        # Convert tensors to Counters for result
        selection_counts = {}
        for key, tensor in selection_tensors.items():
            counts_np = tensor.cpu().numpy()
            selection_counts[key] = Counter({i: int(c) for i, c in enumerate(counts_np) if c > 0})

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            counts = selection_counts[key]
            if not counts:
                continue

            total = sum(counts.values())
            top10 = counts.most_common(10)
            unique = len(counts)

            # Use direct POOL_N_ATTR mapping for consistent n_total lookup
            n_attr = POOL_N_ATTR.get(pool)
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': total,
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': [(idx, cnt, cnt/total) for idx, cnt in top10],
                'concentration': sum(cnt for _, cnt in top10) / total if total > 0 else 0,
            }

        return results

    def analyze_selection_diversity(self, dataloader, n_batches: int = 100, threshold: float = 0.01) -> Dict:
        """
        Analyze selection diversity across batches.

        Measures how many unique neurons are selected across the entire dataset
        vs per-batch selection.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            threshold: Weight threshold for "selected" (for soft gating models)

        Returns:
            Dictionary with diversity metrics
        """
        # Use tensor accumulators instead of sets for speed
        union_tensors = {}  # layer_key -> bool tensor tracking ever-selected neurons
        per_batch_counts = defaultdict(list)

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, desc='Selection Diversity', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True)
                    routing = self.extractor.extract(outputs)
                    if not routing:
                        continue
                except Exception:
                    continue

                # Process ALL layers using extractor
                for layer in routing:
                    layer_idx = layer.layer_idx

                    # Attention routing - use standardized keys
                    for key in ROUTING_KEYS.keys():
                        layer_key = f'L{layer_idx}/{key}'
                        weights = layer.get_weight(key)
                        if weights is not None:
                            n_neurons = weights.shape[-1]
                            # Initialize tensor if needed
                            if layer_key not in union_tensors:
                                union_tensors[layer_key] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                            # Use threshold for soft gating (v18.5+)
                            if weights.dim() == 3:
                                selected = (weights > threshold).any(dim=0).any(dim=0)
                            else:  # dim == 2
                                selected = (weights > threshold).any(dim=0)

                            # OR accumulate (vectorized union)
                            union_tensors[layer_key] |= selected
                            per_batch_counts[layer_key].append(selected.sum().item())

                    # Knowledge routing - use standardized keys
                    for key in KNOWLEDGE_ROUTING_KEYS.keys():
                        layer_key = f'L{layer_idx}/{key}'
                        weights = layer.get_weight(key)
                        if weights is not None:
                            n_neurons = weights.shape[-1]
                            # Initialize tensor if needed
                            if layer_key not in union_tensors:
                                union_tensors[layer_key] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                            # Use threshold for soft gating (v18.5+)
                            if weights.dim() == 3:
                                selected = (weights > threshold).any(dim=0).any(dim=0)
                            else:  # dim == 2
                                selected = (weights > threshold).any(dim=0)

                            # OR accumulate (vectorized union)
                            union_tensors[layer_key] |= selected
                            per_batch_counts[layer_key].append(selected.sum().item())

        # Build results for all layer/key combinations
        results = {}

        for layer_key, union_tensor in union_tensors.items():
            # Parse layer_key: L{layer_idx}/{routing_key}
            parts = layer_key.split('/')
            if len(parts) != 2:
                continue

            layer_str, routing_key = parts
            layer_idx = int(layer_str[1:])

            # Get pool info
            if routing_key in ROUTING_KEYS:
                pool = ROUTING_KEYS[routing_key][3]
                display = f'{layer_str}/{ROUTING_KEYS[routing_key][0]}'
            elif routing_key in KNOWLEDGE_ROUTING_KEYS:
                pool = KNOWLEDGE_ROUTING_KEYS[routing_key][2]
                display = f'{layer_str}/{KNOWLEDGE_ROUTING_KEYS[routing_key][0]}'
            else:
                continue

            # Use direct POOL_N_ATTR mapping for consistent n_total lookup
            n_attr = POOL_N_ATTR.get(pool)
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            # Count from tensor (fast)
            union_count = union_tensor.sum().item()
            batch_counts = per_batch_counts[layer_key]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
            else:
                per_batch_avg = 0
                per_batch_std = 0

            results[layer_key] = {
                'display': display,
                'layer': layer_idx,
                'pool': pool,
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'union_count': int(union_count),
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        total_union = sum(t.sum().item() for t in union_tensors.values())
        n_batches_processed = 0
        if per_batch_counts:
            first_key = next(iter(per_batch_counts))
            n_batches_processed = len(per_batch_counts[first_key])

        results['summary'] = {
            'n_batches_processed': min(n_batches, n_batches_processed),
            'n_layers': len(set(k.split('/')[0] for k in union_tensors.keys())),
            'total_keys_analyzed': len(union_tensors),
            'interpretation': (
                'High diversity_ratio (>2) = many neurons selected differently per batch\n'
                'Low diversity_ratio (~1) = same neurons always selected'
            )
        }

        return results

    def analyze_qk_overlap(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze Q/K routing overlap (Jaccard similarity).
        Optimized with vectorized tensor operations.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with Q/K overlap statistics
        """
        overlap_data = {'fqk': [], 'rqk': []}
        debug_info = {'fqk_retrieved': 0, 'rqk_retrieved': 0, 'fqk_samples': [], 'rqk_samples': []}

        def compute_jaccard_from_weights(q_weights, k_weights, debug_key=None):
            """Compute Jaccard similarity from weight tensors (active neurons)."""
            # Flatten if 3D: [B, S, N] -> [B*S, N]
            if q_weights.dim() == 3:
                q_flat = q_weights.view(-1, q_weights.shape[-1])
                k_flat = k_weights.view(-1, k_weights.shape[-1])
            else:
                q_flat = q_weights
                k_flat = k_weights

            # Active neurons: non-zero weights (sparse from top-k selection)
            # v18.5 uses scatter_ so selected neurons have weight > 0
            q_active = (q_flat > 0).float()
            k_active = (k_flat > 0).float()

            # Debug: store sample info (first batch only)
            if debug_key and len(debug_info[f'{debug_key}_samples']) < 5:
                q_count = q_active[0].sum().item() if q_active.shape[0] > 0 else 0
                k_count = k_active[0].sum().item() if k_active.shape[0] > 0 else 0
                debug_info[f'{debug_key}_samples'].append({
                    'q_active': int(q_count),
                    'k_active': int(k_count),
                    'q_max': float(q_flat[0].max().item()) if q_flat.shape[0] > 0 else 0,
                    'k_max': float(k_flat[0].max().item()) if k_flat.shape[0] > 0 else 0,
                })

            # Compute intersection and union
            intersection = (q_active * k_active).sum(dim=-1)  # [B*S]
            union = ((q_active + k_active) > 0).float().sum(dim=-1)  # [B*S]

            # Jaccard = intersection / union
            jaccard = intersection / (union + 1e-8)

            return jaccard.cpu().tolist()

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Q/K Overlap')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    # F-QK Q/K overlap using standardized keys
                    fqk_q = layer.get_weight('fqk_q')
                    fqk_k = layer.get_weight('fqk_k')
                    if fqk_q is not None and fqk_k is not None:
                        debug_info['fqk_retrieved'] += 1
                        overlap_data['fqk'].extend(compute_jaccard_from_weights(fqk_q, fqk_k, 'fqk'))

                    # R-QK Q/K overlap using standardized keys
                    rqk_q = layer.get_weight('rqk_q')
                    rqk_k = layer.get_weight('rqk_k')
                    if rqk_q is not None and rqk_k is not None:
                        debug_info['rqk_retrieved'] += 1
                        overlap_data['rqk'].extend(compute_jaccard_from_weights(rqk_q, rqk_k, 'rqk'))

        results = {}
        for key in ['fqk', 'rqk']:
            if overlap_data[key]:
                mean_overlap = np.mean(overlap_data[key])
                results[key] = {
                    'overlap_ratio': float(mean_overlap),
                    'jaccard': float(mean_overlap),
                    'std_overlap': float(np.std(overlap_data[key])),
                    'interpretation': (
                        'Q and K select similar neurons' if mean_overlap > 0.3
                        else 'Q and K select different neurons'
                    ),
                    'debug': {
                        'retrieved_count': debug_info[f'{key}_retrieved'],
                        'samples': debug_info[f'{key}_samples'],
                    }
                }

        return results

    def analyze_qk_usage(self, dataloader, n_batches: int = 100, layer_idx: int = None) -> Dict:
        """
        Analyze per-neuron Q/K selection counts across ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            layer_idx: Specific layer to analyze (None = aggregate all layers)

        Returns:
            Dictionary with Q/K usage statistics
        """
        # Map pool names to standardized Q/K keys
        POOL_STD_KEYS = {
            'feature_qk': ('fqk_q', 'fqk_k'),
            'restore_qk': ('rqk_q', 'rqk_k'),
        }

        results = {}
        self.model.eval()

        for pool_name, pool_info in QK_POOLS.items():
            n_neurons = getattr(self.router, pool_info['n_attr'], 0)
            if n_neurons == 0:
                continue

            # Get standardized keys for this pool
            std_q_key, std_k_key = POOL_STD_KEYS.get(pool_name, (None, None))
            if std_q_key is None:
                continue

            # Per-layer counts: {lidx: (q_counts, k_counts, overlaps)}
            layer_data = {}

            with self.extractor.analysis_context():
                for i, batch in enumerate(tqdm(dataloader, desc=f'{pool_info["display"]} Q/K', total=n_batches)):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)

                    try:
                        with torch.no_grad():
                            outputs = self.model(input_ids, return_routing_info=True)
                        routing = self.extractor.extract(outputs)
                        if not routing:
                            continue
                    except Exception:
                        continue

                    # Process ALL layers using extractor
                    for layer in routing:
                        lidx = layer.layer_idx
                        if layer_idx is not None and lidx != layer_idx:
                            continue

                        # Get Q/K weights using standardized keys
                        w_q = layer.get_weight(std_q_key)
                        w_k = layer.get_weight(std_k_key)

                        if w_q is None or w_k is None:
                            continue

                        # Initialize layer data if needed
                        if lidx not in layer_data:
                            layer_data[lidx] = (
                                torch.zeros(n_neurons, device=self.device),
                                torch.zeros(n_neurons, device=self.device),
                                []
                            )

                        q_counts, k_counts, batch_overlaps = layer_data[lidx]

                        # Count selections
                        if w_q.dim() == 3:  # [B, S, N]
                            selected_q = (w_q > 0).float().sum(dim=[0, 1])
                            selected_k = (w_k > 0).float().sum(dim=[0, 1])
                        else:  # [B, N]
                            selected_q = (w_q > 0).float().sum(dim=0)
                            selected_k = (w_k > 0).float().sum(dim=0)

                        q_counts += selected_q
                        k_counts += selected_k

                        # Calculate batch overlap
                        if w_q.dim() >= 2:
                            overlap = ((w_q > 0) & (w_k > 0)).float()
                            active_q = (w_q > 0).float().sum(-1)
                            overlap_ratio = (overlap.sum(-1) / (active_q + 1e-8)).mean().item()
                            batch_overlaps.append(overlap_ratio)

                        layer_data[lidx] = (q_counts, k_counts, batch_overlaps)

                # Aggregate across all layers
                total_q = torch.zeros(n_neurons, device=self.device)
                total_k = torch.zeros(n_neurons, device=self.device)
                all_overlaps = []

                per_layer_results = {}
                for lidx, (q_counts, k_counts, batch_overlaps) in layer_data.items():
                    total_q += q_counts
                    total_k += k_counts
                    all_overlaps.extend(batch_overlaps)

                    # Per-layer stats
                    q_np = q_counts.cpu().numpy()
                    k_np = k_counts.cpu().numpy()
                    corr = float(np.corrcoef(q_np, k_np)[0, 1]) if q_np.sum() > 0 and k_np.sum() > 0 else 0.0

                    per_layer_results[f'L{lidx}'] = {
                        'correlation': corr,
                        'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
                        'q_total': int(q_np.sum()),
                        'k_total': int(k_np.sum()),
                    }

                # Aggregated statistics
                q_np = total_q.cpu().numpy()
                k_np = total_k.cpu().numpy()

                # Correlation
                if q_np.sum() > 0 and k_np.sum() > 0:
                    corr = float(np.corrcoef(q_np, k_np)[0, 1])
                else:
                    corr = 0.0

                # Ratio-based specialization analysis (for paper Fig 3)
                # q_ratio = q_count / (q_count + k_count) per neuron
                total_usage = q_np + k_np
                q_ratio = np.zeros_like(q_np, dtype=float)
                valid_mask = total_usage > 0
                q_ratio[valid_mask] = q_np[valid_mask] / total_usage[valid_mask]

                # Classify by ratio (all neurons with valid usage)
                # Q-specialized: q_ratio > 0.7 (mostly selected by Q)
                # K-specialized: q_ratio < 0.3 (mostly selected by K)
                # Shared: 0.3 <= q_ratio <= 0.7 (selected by both)
                q_specialized = int((q_ratio > 0.7).sum())
                k_specialized = int((q_ratio < 0.3).sum())
                shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7)).sum())

                # Sensitivity analysis: multiple thresholds for paper
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
                    'n_layers': len(layer_data),
                    'q_counts': q_np.tolist(),
                    'k_counts': k_np.tolist(),
                    'correlation': corr,
                    'avg_overlap': float(np.mean(all_overlaps)) if all_overlaps else 0,
                    'std_overlap': float(np.std(all_overlaps)) if all_overlaps else 0,
                    'q_specialized': q_specialized,
                    'k_specialized': k_specialized,
                    'shared': shared,
                    'q_total': int(q_np.sum()),
                    'k_total': int(k_np.sum()),
                    'per_layer': per_layer_results,
                    # Paper Fig 3 specific data
                    'q_ratio': q_ratio.tolist(),  # Per-neuron Q ratio for scatter/histogram
                    'specialization_thresholds': {
                        'q_specialized': 0.7,
                        'k_specialized': 0.3,
                    },
                    # Sensitivity analysis for different thresholds
                    'sensitivity_analysis': sensitivity_analysis,
                }

            results['n_batches'] = n_batches

        return results

    def analyze_qk_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze Q/K routing entropy patterns across ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with entropy statistics per layer
        """
        # Map pool names to standardized Q/K keys
        POOL_STD_KEYS = {
            'feature_qk': ('fqk_q', 'fqk_k'),
            'restore_qk': ('rqk_q', 'rqk_k'),
        }

        # {pool_name: {layer_idx: {'q': [], 'k': []}}}
        layer_entropy = {pool: {} for pool in QK_POOLS.keys()}

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, desc='Q/K Entropy', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True)
                    routing = self.extractor.extract(outputs)
                    if not routing:
                        continue
                except Exception:
                    continue

                # Process ALL layers using extractor
                for layer in routing:
                    layer_idx = layer.layer_idx

                    for pool_name, pool_info in QK_POOLS.items():
                        if layer_idx not in layer_entropy[pool_name]:
                            layer_entropy[pool_name][layer_idx] = {'q': [], 'k': []}

                        # Get standardized keys for this pool
                        std_q_key, std_k_key = POOL_STD_KEYS.get(pool_name, (None, None))
                        if std_q_key is None:
                            continue

                        # Use standardized keys
                        w_q = layer.get_weight(std_q_key)
                        w_k = layer.get_weight(std_k_key)

                        if w_q is not None:
                            ent = calc_entropy_ratio(w_q)
                            layer_entropy[pool_name][layer_idx]['q'].append(ent)

                        if w_k is not None:
                            ent = calc_entropy_ratio(w_k)
                            layer_entropy[pool_name][layer_idx]['k'].append(ent)

        # Build results
        results = {}
        for pool_name, pool_info in QK_POOLS.items():
            pool_results = {}

            for layer_idx, ent_data in layer_entropy[pool_name].items():
                q_entropy = ent_data['q']
                k_entropy = ent_data['k']

                if q_entropy and k_entropy:
                    pool_results[f'L{layer_idx}'] = {
                        'q_entropy_mean': float(np.mean(q_entropy)),
                        'q_entropy_std': float(np.std(q_entropy)),
                        'k_entropy_mean': float(np.mean(k_entropy)),
                        'k_entropy_std': float(np.std(k_entropy)),
                        'entropy_diff': float(np.mean(q_entropy) - np.mean(k_entropy)),
                    }

            # Summary across layers
            all_q = [v['q_entropy_mean'] for v in pool_results.values()]
            all_k = [v['k_entropy_mean'] for v in pool_results.values()]

            if all_q and all_k:
                pool_results['summary'] = {
                    'q_entropy_avg': float(np.mean(all_q)),
                    'k_entropy_avg': float(np.mean(all_k)),
                    'entropy_diff_avg': float(np.mean(all_q) - np.mean(all_k)),
                }

            results[pool_name] = {
                'display': pool_info['display'],
                'per_layer': pool_results,
                # Keep backward compatibility
                'q_entropy_mean': float(np.mean(all_q)) if all_q else 0,
                'q_entropy_std': float(np.std(all_q)) if all_q else 0,
                'k_entropy_mean': float(np.mean(all_k)) if all_k else 0,
                'k_entropy_std': float(np.std(all_k)) if all_k else 0,
                'entropy_diff': float(np.mean(all_q) - np.mean(all_k)) if all_q and all_k else 0,
            }

        return results

    def analyze_layer_contribution(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze per-layer contribution from attention vs knowledge circuits.

        Uses OUTPUT NORM based measurement (most accurate):
            attn_out_norm = attn_out.norm(dim=-1).mean()
            know_out_norm = know_out.norm(dim=-1).mean()
            attention_ratio = attn_norm / (attn_norm + know_norm)

        This measures actual output magnitude, not just routing weights.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with layer-wise attention/knowledge ratios based on output norms
        """
        # Track per-layer output norms
        layer_norms = defaultdict(lambda: {'attn': [], 'know': []})

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, desc='Layer Contribution (output norm)', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True)
                    routing = self.extractor.extract(outputs)
                    if not routing:
                        continue
                except Exception:
                    continue

                # Process all layers - use output norms
                for layer in routing:
                    layer_idx = layer.layer_idx
                    norms = layer.get_output_norms()

                    if 'attn_out_norm' in norms and 'know_out_norm' in norms:
                        layer_norms[layer_idx]['attn'].append(norms['attn_out_norm'])
                        layer_norms[layer_idx]['know'].append(norms['know_out_norm'])

        # Aggregate results using output norms
        results = {'per_layer': {}, 'method': 'output_norm'}

        for layer_idx in sorted(layer_norms.keys()):
            attn_norms = layer_norms[layer_idx]['attn']
            know_norms = layer_norms[layer_idx]['know']

            if attn_norms and know_norms:
                mean_attn = float(np.mean(attn_norms))
                mean_know = float(np.mean(know_norms))
                total = mean_attn + mean_know

                results['per_layer'][f'L{layer_idx}'] = {
                    'layer_idx': layer_idx,
                    'attn_out_norm': mean_attn,
                    'know_out_norm': mean_know,
                    'attention_ratio': float(mean_attn / total * 100) if total > 0 else 50.0,
                    'knowledge_ratio': float(mean_know / total * 100) if total > 0 else 50.0,
                }

        # Summary
        if results['per_layer']:
            attn_ratios = [v['attention_ratio'] for v in results['per_layer'].values()]
            know_ratios = [v['knowledge_ratio'] for v in results['per_layer'].values()]
            results['summary'] = {
                'attention_ratio_mean': float(np.mean(attn_ratios)),
                'knowledge_ratio_mean': float(np.mean(know_ratios)),
                'attention_ratio_std': float(np.std(attn_ratios)),
                'knowledge_ratio_std': float(np.std(know_ratios)),
                'n_layers': len(results['per_layer']),
            }

        return results

    def analyze_qk_union_coverage(self, dataloader, n_batches: int = 100) -> Dict:
        """
        Analyze Q/K union coverage to find true dead neurons.

        Computes Q ∪ K to determine which neurons are never selected
        by either Q or K routing.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with union coverage statistics:
            - q_only: neurons selected only by Q
            - k_only: neurons selected only by K
            - shared: neurons selected by both
            - dead: neurons never selected (true dead)
            - union_coverage: (q_only + k_only + shared) / total
        """
        # Use tensor accumulators instead of sets (vectorized)
        q_tensors = {}  # pool -> bool tensor
        k_tensors = {}

        # Pool info mapping
        pool_info = {
            'feature_qk': ('fqk_q', 'fqk_k', 'n_feature_qk'),
            'restore_qk': ('rqk_q', 'rqk_k', 'n_restore_qk'),
        }

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Q/K Union Coverage')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for pool_name, (q_key, k_key, n_attr) in pool_info.items():
                        w_q = layer.get_weight(q_key)
                        w_k = layer.get_weight(k_key)

                        if w_q is not None:
                            n_neurons = w_q.shape[-1]
                            # Initialize tensor if needed
                            if pool_name not in q_tensors:
                                q_tensors[pool_name] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                            # Get selected neurons (weight > 0) - vectorized OR
                            if w_q.dim() == 3:
                                selected = (w_q > 0).any(dim=0).any(dim=0)
                            else:
                                selected = (w_q > 0).any(dim=0)
                            q_tensors[pool_name] |= selected

                        if w_k is not None:
                            n_neurons = w_k.shape[-1]
                            # Initialize tensor if needed
                            if pool_name not in k_tensors:
                                k_tensors[pool_name] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                            if w_k.dim() == 3:
                                selected = (w_k > 0).any(dim=0).any(dim=0)
                            else:
                                selected = (w_k > 0).any(dim=0)
                            k_tensors[pool_name] |= selected

        # Compute statistics using tensor operations
        results = {}
        for pool_name, (_, _, n_attr) in pool_info.items():
            n_total = getattr(self.router, n_attr, 0)
            if n_total == 0:
                continue

            q_mask = q_tensors.get(pool_name)
            k_mask = k_tensors.get(pool_name)

            if q_mask is None or k_mask is None:
                continue

            # Vectorized set operations using tensor logic
            q_only = (q_mask & ~k_mask).sum().item()
            k_only = (k_mask & ~q_mask).sum().item()
            shared = (q_mask & k_mask).sum().item()
            union = (q_mask | k_mask).sum().item()
            dead = n_total - union

            results[pool_name] = {
                'n_total': n_total,
                'q_only': int(q_only),
                'k_only': int(k_only),
                'shared': int(shared),
                'dead': int(dead),
                'union_coverage': union / n_total if n_total > 0 else 0,
                'q_coverage': q_mask.sum().item() / n_total if n_total > 0 else 0,
                'k_coverage': k_mask.sum().item() / n_total if n_total > 0 else 0,
                'separation_ratio': (q_only + k_only) / union if union > 0 else 0,
            }

        results['n_batches'] = n_batches
        return results

    def analyze_path_usage(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze v18.5 multi-path usage patterns.

        Examines how different paths are utilized in the routing.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with path usage statistics:
            - path_activation_rate: percentage of tokens using each path
            - avg_active_paths: average number of active paths per token
            - path_weight_distribution: weight ratio between paths
        """
        # Check if model supports multi-path (v18.5+)
        max_paths = getattr(self.router, 'max_paths', 1)
        if max_paths <= 1:
            return {'note': 'Model does not support multi-path routing'}

        path_stats = defaultdict(lambda: {'active_count': 0, 'weight_sum': 0.0, 'total_tokens': 0})

        self.model.eval()
        # Enable path weight storage
        if hasattr(self.router, 'store_path_weights'):
            self.router.store_path_weights = True

        try:
            with self.extractor.analysis_context():
                for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Path Usage')):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True, return_path_weights=True)

                    routing = self.extractor.extract(outputs)
                    if not routing:
                        continue

                    for layer in routing:
                        # Check for path_weights in layer data
                        path_weights = layer.raw.get('path_weights', {})
                        if not path_weights:
                            continue

                        for pool_name, weights_list in path_weights.items():
                            if not isinstance(weights_list, list):
                                continue

                            for path_idx, path_w in enumerate(weights_list):
                                if path_w is None:
                                    continue

                                key = f'{pool_name}_path{path_idx}'
                                # Count active tokens (any neuron selected)
                                if path_w.dim() == 3:  # [B, S, N]
                                    active = (path_w > 0).any(dim=-1)  # [B, S]
                                    active_count = active.sum().item()
                                    total_tokens = path_w.shape[0] * path_w.shape[1]
                                    weight_sum = path_w.sum().item()
                                else:  # [B, N]
                                    active = (path_w > 0).any(dim=-1)  # [B]
                                    active_count = active.sum().item()
                                    total_tokens = path_w.shape[0]
                                    weight_sum = path_w.sum().item()

                                path_stats[key]['active_count'] += active_count
                                path_stats[key]['total_tokens'] += total_tokens
                                path_stats[key]['weight_sum'] += weight_sum
        finally:
            if hasattr(self.router, 'store_path_weights'):
                self.router.store_path_weights = False

        # Compute results
        results = {'per_path': {}}
        pool_totals = defaultdict(lambda: {'active': 0, 'total': 0, 'weight': 0.0})

        for key, stats in path_stats.items():
            if stats['total_tokens'] > 0:
                activation_rate = stats['active_count'] / stats['total_tokens']
                results['per_path'][key] = {
                    'activation_rate': activation_rate,
                    'weight_sum': stats['weight_sum'],
                    'active_count': stats['active_count'],
                    'total_tokens': stats['total_tokens'],
                }

                # Aggregate by pool
                pool_name = '_'.join(key.split('_')[:-1])
                pool_totals[pool_name]['active'] += stats['active_count']
                pool_totals[pool_name]['total'] += stats['total_tokens']
                pool_totals[pool_name]['weight'] += stats['weight_sum']

        # Per-pool summary
        results['per_pool'] = {}
        for pool_name, totals in pool_totals.items():
            if totals['total'] > 0:
                results['per_pool'][pool_name] = {
                    'avg_active_paths': totals['active'] / (totals['total'] / max_paths),
                    'total_weight': totals['weight'],
                }

        results['max_paths'] = max_paths
        results['n_batches'] = n_batches
        return results

    def analyze_activation_sparsity(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze per-token activation sparsity.

        Measures how many neurons are actually active per token.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with sparsity statistics:
            - avg_active_per_token: average active neurons per token
            - sparsity_ratio: percentage of inactive neurons
            - active_weight_sum: average weight sum of active neurons
        """
        # Accumulate statistics per key
        sparsity_stats = defaultdict(lambda: {'active_sum': 0, 'weight_sum': 0.0, 'count': 0, 'n_total': 0})

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Activation Sparsity')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for key in ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        pool = ROUTING_KEYS[key][3]
                        n_attr = POOL_N_ATTR.get(pool)
                        n_total = getattr(self.router, n_attr, 0) if n_attr else 0

                        if n_total == 0:
                            continue

                        # Vectorized computation
                        if weights.dim() == 3:  # [B, S, N]
                            w_flat = weights.view(-1, weights.shape[-1])
                        else:  # [B, N]
                            w_flat = weights

                        # Per-token active count and weight sum (vectorized)
                        active_mask = w_flat > 0
                        active_counts = active_mask.sum(dim=-1)  # [B*S]
                        weight_sums = (w_flat * active_mask.float()).sum(dim=-1)  # [B*S]

                        sparsity_stats[key]['active_sum'] += active_counts.sum().item()
                        sparsity_stats[key]['weight_sum'] += weight_sums.sum().item()
                        sparsity_stats[key]['count'] += w_flat.shape[0]
                        sparsity_stats[key]['n_total'] = n_total

        # Aggregate results
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

    def analyze_token_coselection(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze per-token Q/K co-selection patterns.

        Measures how many neurons are selected by BOTH Q and K
        for the same token position.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with co-selection statistics:
            - mean_coselect_per_token: average shared neurons per token
            - coselect_rate: ratio of shared to union
            - per_layer: layer-wise patterns
        """
        # Accumulate statistics
        coselect_stats = defaultdict(lambda: {
            'coselect_sum': 0, 'union_sum': 0, 'q_sum': 0, 'k_sum': 0, 'count': 0
        })
        per_layer_stats = defaultdict(lambda: defaultdict(lambda: {'coselect_sum': 0, 'count': 0}))

        pool_info = {
            'feature_qk': ('fqk_q', 'fqk_k'),
            'restore_qk': ('rqk_q', 'rqk_k'),
        }

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Token Co-selection')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    layer_idx = layer.layer_idx

                    for pool_name, (q_key, k_key) in pool_info.items():
                        w_q = layer.get_weight(q_key)
                        w_k = layer.get_weight(k_key)

                        if w_q is None or w_k is None:
                            continue

                        # Compute per-token co-selection (vectorized)
                        if w_q.dim() == 3:  # [B, S, N]
                            q_flat = w_q.view(-1, w_q.shape[-1])
                            k_flat = w_k.view(-1, w_k.shape[-1])
                        else:
                            q_flat = w_q
                            k_flat = w_k

                        q_active = (q_flat > 0)
                        k_active = (k_flat > 0)

                        # Vectorized metrics
                        coselect = (q_active & k_active).sum(dim=-1).float()  # [B*S]
                        union = (q_active | k_active).sum(dim=-1).float()
                        q_count = q_active.sum(dim=-1).float()
                        k_count = k_active.sum(dim=-1).float()

                        n_tokens = coselect.shape[0]
                        coselect_stats[pool_name]['coselect_sum'] += coselect.sum().item()
                        coselect_stats[pool_name]['union_sum'] += union.sum().item()
                        coselect_stats[pool_name]['q_sum'] += q_count.sum().item()
                        coselect_stats[pool_name]['k_sum'] += k_count.sum().item()
                        coselect_stats[pool_name]['count'] += n_tokens

                        per_layer_stats[pool_name][layer_idx]['coselect_sum'] += coselect.sum().item()
                        per_layer_stats[pool_name][layer_idx]['count'] += n_tokens

        # Aggregate results
        results = {}
        for pool_name, stats in coselect_stats.items():
            if stats['count'] == 0:
                continue

            mean_coselect = stats['coselect_sum'] / stats['count']
            mean_union = stats['union_sum'] / stats['count']
            mean_q = stats['q_sum'] / stats['count']
            mean_k = stats['k_sum'] / stats['count']

            # Per-layer breakdown
            per_layer = {}
            for layer_idx, layer_stats in sorted(per_layer_stats[pool_name].items()):
                if layer_stats['count'] > 0:
                    per_layer[f'L{layer_idx}'] = {
                        'mean_coselect': layer_stats['coselect_sum'] / layer_stats['count'],
                    }

            results[pool_name] = {
                'mean_coselect_per_token': mean_coselect,
                'mean_union_per_token': mean_union,
                'coselect_rate': mean_coselect / mean_union if mean_union > 0 else 0,
                'mean_q_active': mean_q,
                'mean_k_active': mean_k,
                'per_layer': per_layer,
            }

        results['n_batches'] = n_batches
        return results

    def analyze_coverage_progression(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze how coverage changes across layers.

        Summarizes selection diversity trends from early to late layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with coverage progression:
            - early_layers: average coverage for first 1/3 layers
            - mid_layers: average coverage for middle 1/3 layers
            - late_layers: average coverage for last 1/3 layers
            - trend: increasing/decreasing/stable
        """
        # Get selection diversity data
        diversity_results = self.analyze_selection_diversity(dataloader, n_batches)

        # Extract per-layer coverages
        layer_coverages = defaultdict(dict)

        for layer_key, data in diversity_results.items():
            if layer_key == 'summary' or not isinstance(data, dict):
                continue

            if 'layer' in data and 'union_coverage' in data:
                layer_idx = data['layer']
                pool = data.get('pool', 'unknown')
                layer_coverages[pool][layer_idx] = data['union_coverage']

        # Analyze progression per pool
        results = {}
        for pool, coverages in layer_coverages.items():
            if not coverages:
                continue

            sorted_layers = sorted(coverages.keys())
            n_layers = len(sorted_layers)

            if n_layers < 3:
                continue

            # Split into thirds
            third = n_layers // 3
            early_layers = sorted_layers[:third]
            mid_layers = sorted_layers[third:2*third]
            late_layers = sorted_layers[2*third:]

            early_cov = np.mean([coverages[l] for l in early_layers])
            mid_cov = np.mean([coverages[l] for l in mid_layers])
            late_cov = np.mean([coverages[l] for l in late_layers])

            # Determine trend
            if late_cov > early_cov * 1.1:
                trend = 'increasing'
            elif late_cov < early_cov * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'

            growth_ratio = late_cov / early_cov if early_cov > 0 else 1.0

            results[pool] = {
                'early_coverage': early_cov,
                'mid_coverage': mid_cov,
                'late_coverage': late_cov,
                'trend': trend,
                'growth_ratio': growth_ratio,
                'early_layers': list(early_layers),
                'late_layers': list(late_layers),
            }

        results['n_batches'] = n_batches
        return results

    def analyze_weight_concentration(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze weight concentration among selected neurons.

        Measures how much weight is concentrated in top neurons.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with concentration statistics:
            - top1_weight_ratio: fraction of weight in top-1 neuron
            - top5_weight_ratio: fraction of weight in top-5 neurons
            - weight_gini: gini coefficient among active neurons
        """
        # Accumulate statistics
        concentration_stats = defaultdict(lambda: {
            'top1_sum': 0.0, 'top5_sum': 0.0, 'active_sum': 0, 'count': 0
        })

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Weight Concentration')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for key in ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        # Flatten to [B*S, N]
                        if weights.dim() == 3:
                            w_flat = weights.view(-1, weights.shape[-1])
                        else:
                            w_flat = weights

                        # Vectorized: sort each token's weights descending
                        sorted_w, _ = w_flat.sort(dim=-1, descending=True)

                        # Total weight per token
                        total_w = sorted_w.sum(dim=-1, keepdim=True)  # [B*S, 1]
                        valid_mask = (total_w.squeeze(-1) > 0)

                        if not valid_mask.any():
                            continue

                        # Top1 and Top5 ratios (vectorized)
                        top1 = sorted_w[:, 0] / (total_w.squeeze(-1) + 1e-8)
                        top5 = sorted_w[:, :5].sum(dim=-1) / (total_w.squeeze(-1) + 1e-8)

                        # Active count per token
                        active_counts = (w_flat > 0).sum(dim=-1).float()

                        # Only count valid tokens
                        concentration_stats[key]['top1_sum'] += top1[valid_mask].sum().item()
                        concentration_stats[key]['top5_sum'] += top5[valid_mask].sum().item()
                        concentration_stats[key]['active_sum'] += active_counts[valid_mask].sum().item()
                        concentration_stats[key]['count'] += valid_mask.sum().item()

        # Aggregate results
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

    def visualize_qk_usage(self, usage_results: Dict, output_dir: str) -> Optional[str]:
        """
        Visualize Q/K usage patterns.

        Delegates to visualizers.qk_specialization.plot_qk_usage().

        Args:
            usage_results: Results from analyze_qk_usage()
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        from .visualizers import plot_qk_usage
        return plot_qk_usage(usage_results, output_dir)

    def run_all(self, dataloader, output_dir: str = './routing_analysis', n_batches: int = 50) -> Dict:
        """
        Run all routing analyses in a single efficient pass.

        Optimized: All analyses computed in ONE dataloader iteration (10-15x faster).
        Previously ran 13+ separate loops through the dataloader.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # ===== Initialize all accumulators =====
        # Entropy accumulator
        entropy_data = {name: [] for name in ROUTING_KEYS.keys()}

        # Selection frequency accumulator (tensor-based)
        selection_tensors = {}

        # Selection diversity accumulators
        diversity_threshold = 0.01
        union_tensors = {}
        per_batch_counts = defaultdict(list)

        # Q/K overlap accumulator
        qk_overlap_data = {'fqk': [], 'rqk': []}

        # Q/K usage accumulators (per pool, per layer)
        POOL_STD_KEYS = {
            'feature_qk': ('fqk_q', 'fqk_k'),
            'restore_qk': ('rqk_q', 'rqk_k'),
        }
        qk_layer_data = {pool: {} for pool in QK_POOLS.keys()}

        # Q/K entropy accumulator
        qk_entropy_data = {pool: {} for pool in QK_POOLS.keys()}

        # Q/K union coverage accumulators
        qk_union_q_tensors = {}
        qk_union_k_tensors = {}
        pool_info_union = {
            'feature_qk': ('fqk_q', 'fqk_k', 'n_feature_qk'),
            'restore_qk': ('rqk_q', 'rqk_k', 'n_restore_qk'),
        }

        # Activation sparsity accumulator
        sparsity_stats = defaultdict(lambda: {'active_sum': 0, 'weight_sum': 0.0, 'count': 0, 'n_total': 0})

        # Token co-selection accumulators
        coselect_stats = defaultdict(lambda: {
            'coselect_sum': 0, 'union_sum': 0, 'q_sum': 0, 'k_sum': 0, 'count': 0
        })
        per_layer_coselect = defaultdict(lambda: defaultdict(lambda: {'coselect_sum': 0, 'count': 0}))

        # Weight concentration accumulator
        concentration_stats = defaultdict(lambda: {
            'top1_sum': 0.0, 'top5_sum': 0.0, 'active_sum': 0, 'count': 0
        })

        # Layer contribution accumulator
        layer_norms = defaultdict(lambda: {'attn': [], 'know': []})

        # Path usage accumulator
        max_paths = getattr(self.router, 'max_paths', 1)
        path_stats = defaultdict(lambda: {'active_count': 0, 'weight_sum': 0.0, 'total_tokens': 0})

        # ===== Single-pass data collection =====
        self.model.eval()

        # Enable path weight storage if supported
        if hasattr(self.router, 'store_path_weights'):
            self.router.store_path_weights = True

        try:
            with self.extractor.analysis_context():
                # Verify store_pref_tensors is enabled
                global_router = self.extractor._get_global_router()
                store_enabled = getattr(global_router, 'store_pref_tensors', False) if global_router else False
                is_wrapped = hasattr(self.model, '_orig_mod')
                print(f"  [Debug] Model wrapped: {is_wrapped}, store_pref_tensors: {store_enabled}")
                if not store_enabled:
                    print(f"  [Warning] store_pref_tensors not enabled - routing weights may be empty")

                processed_batches = 0
                empty_routing_count = 0
                for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Routing Analysis (single-pass)')):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)

                    try:
                        with torch.no_grad():
                            outputs = self.model(input_ids, return_routing_info=True)
                        routing = self.extractor.extract(outputs)
                        if not routing:
                            empty_routing_count += 1
                            continue
                    except Exception as e:
                        if processed_batches == 0:
                            print(f"  [Warning] First batch failed: {e}")
                            import traceback
                            traceback.print_exc()
                        continue

                    processed_batches += 1

                    # Debug: log first batch info
                    if processed_batches == 1:
                        first_layer = next(iter(routing), None)
                        if first_layer:
                            available_keys = []
                            for key in ROUTING_KEYS.keys():
                                w = first_layer.get_weight(key)
                                if w is not None:
                                    available_keys.append(f"{key}:{list(w.shape)}")
                            print(f"  [Debug] First batch: {len(routing)} layers, weights: {available_keys}")
                            # Also check raw keys in attention dict
                            raw_attn_keys = list(first_layer.attention.keys()) if first_layer.attention else []
                            print(f"  [Debug] Raw attention keys: {raw_attn_keys}")

                    # Process all layers from this batch
                    for layer in routing:
                        layer_idx = layer.layer_idx

                        # ----- 1. Entropy + Selection Frequency + Sparsity + Concentration -----
                        for key in ROUTING_KEYS.keys():
                            weights = layer.get_weight(key)
                            if weights is None:
                                continue

                            pool = ROUTING_KEYS[key][3]
                            n_attr = POOL_N_ATTR.get(pool)
                            n_neurons = getattr(self.router, n_attr, 0) if n_attr else weights.shape[-1]

                            # Entropy
                            ent = calc_entropy_ratio(weights)
                            entropy_data[key].append(ent)

                            # Selection frequency (tensor accumulation)
                            if key not in selection_tensors:
                                selection_tensors[key] = torch.zeros(n_neurons, device=self.device, dtype=torch.long)
                            active_mask = weights > 0
                            if active_mask.dim() == 3:
                                counts = active_mask.sum(dim=[0, 1])
                            else:
                                counts = active_mask.sum(dim=0)
                            selection_tensors[key] += counts

                            # Activation sparsity
                            if n_neurons > 0:
                                if weights.dim() == 3:
                                    w_flat = weights.view(-1, weights.shape[-1])
                                else:
                                    w_flat = weights
                                active_mask_flat = w_flat > 0
                                active_counts = active_mask_flat.sum(dim=-1)
                                weight_sums = (w_flat * active_mask_flat.float()).sum(dim=-1)
                                sparsity_stats[key]['active_sum'] += active_counts.sum().item()
                                sparsity_stats[key]['weight_sum'] += weight_sums.sum().item()
                                sparsity_stats[key]['count'] += w_flat.shape[0]
                                sparsity_stats[key]['n_total'] = n_neurons

                            # Weight concentration
                            if weights.dim() == 3:
                                w_flat = weights.view(-1, weights.shape[-1])
                            else:
                                w_flat = weights
                            sorted_w, _ = w_flat.sort(dim=-1, descending=True)
                            total_w = sorted_w.sum(dim=-1, keepdim=True)
                            valid_mask = (total_w.squeeze(-1) > 0)
                            if valid_mask.any():
                                top1 = sorted_w[:, 0] / (total_w.squeeze(-1) + 1e-8)
                                top5 = sorted_w[:, :5].sum(dim=-1) / (total_w.squeeze(-1) + 1e-8)
                                active_per_token = (w_flat > 0).sum(dim=-1).float()
                                concentration_stats[key]['top1_sum'] += top1[valid_mask].sum().item()
                                concentration_stats[key]['top5_sum'] += top5[valid_mask].sum().item()
                                concentration_stats[key]['active_sum'] += active_per_token[valid_mask].sum().item()
                                concentration_stats[key]['count'] += valid_mask.sum().item()

                        # ----- 2. Selection Diversity -----
                        for key in list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys()):
                            layer_key = f'L{layer_idx}/{key}'
                            if key in ROUTING_KEYS:
                                weights = layer.get_weight(key)
                            else:
                                weights = layer.get_weight(key)
                            if weights is None:
                                continue

                            n_neurons = weights.shape[-1]
                            if layer_key not in union_tensors:
                                union_tensors[layer_key] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                            if weights.dim() == 3:
                                selected = (weights > diversity_threshold).any(dim=0).any(dim=0)
                            else:
                                selected = (weights > diversity_threshold).any(dim=0)
                            union_tensors[layer_key] |= selected
                            per_batch_counts[layer_key].append(selected.sum().item())

                        # ----- 3. Q/K Overlap -----
                        fqk_q = layer.get_weight('fqk_q')
                        fqk_k = layer.get_weight('fqk_k')
                        if fqk_q is not None and fqk_k is not None:
                            jaccard = self._compute_qk_jaccard(fqk_q, fqk_k)
                            qk_overlap_data['fqk'].extend(jaccard)

                        rqk_q = layer.get_weight('rqk_q')
                        rqk_k = layer.get_weight('rqk_k')
                        if rqk_q is not None and rqk_k is not None:
                            jaccard = self._compute_qk_jaccard(rqk_q, rqk_k)
                            qk_overlap_data['rqk'].extend(jaccard)

                        # ----- 4. Q/K Usage + Q/K Entropy + Union Coverage + Co-selection -----
                        for pool_name, pool_info in QK_POOLS.items():
                            std_q_key, std_k_key = POOL_STD_KEYS.get(pool_name, (None, None))
                            if std_q_key is None:
                                continue

                            n_neurons = getattr(self.router, pool_info['n_attr'], 0)
                            if n_neurons == 0:
                                continue

                            w_q = layer.get_weight(std_q_key)
                            w_k = layer.get_weight(std_k_key)

                            if w_q is None or w_k is None:
                                continue

                            # Q/K usage accumulation
                            if layer_idx not in qk_layer_data[pool_name]:
                                qk_layer_data[pool_name][layer_idx] = (
                                    torch.zeros(n_neurons, device=self.device),
                                    torch.zeros(n_neurons, device=self.device),
                                    []
                                )
                            q_counts, k_counts, batch_overlaps = qk_layer_data[pool_name][layer_idx]

                            if w_q.dim() == 3:
                                selected_q = (w_q > 0).float().sum(dim=[0, 1])
                                selected_k = (w_k > 0).float().sum(dim=[0, 1])
                            else:
                                selected_q = (w_q > 0).float().sum(dim=0)
                                selected_k = (w_k > 0).float().sum(dim=0)
                            q_counts += selected_q
                            k_counts += selected_k

                            if w_q.dim() >= 2:
                                overlap = ((w_q > 0) & (w_k > 0)).float()
                                active_q = (w_q > 0).float().sum(-1)
                                overlap_ratio = (overlap.sum(-1) / (active_q + 1e-8)).mean().item()
                                batch_overlaps.append(overlap_ratio)
                            qk_layer_data[pool_name][layer_idx] = (q_counts, k_counts, batch_overlaps)

                            # Q/K entropy
                            if layer_idx not in qk_entropy_data[pool_name]:
                                qk_entropy_data[pool_name][layer_idx] = {'q': [], 'k': []}
                            qk_entropy_data[pool_name][layer_idx]['q'].append(calc_entropy_ratio(w_q))
                            qk_entropy_data[pool_name][layer_idx]['k'].append(calc_entropy_ratio(w_k))

                            # Q/K union coverage
                            q_union_key, k_union_key, n_attr = pool_info_union.get(pool_name, (None, None, None))
                            if q_union_key:
                                if pool_name not in qk_union_q_tensors:
                                    qk_union_q_tensors[pool_name] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)
                                    qk_union_k_tensors[pool_name] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                                if w_q.dim() == 3:
                                    q_sel = (w_q > 0).any(dim=0).any(dim=0)
                                    k_sel = (w_k > 0).any(dim=0).any(dim=0)
                                else:
                                    q_sel = (w_q > 0).any(dim=0)
                                    k_sel = (w_k > 0).any(dim=0)
                                qk_union_q_tensors[pool_name] |= q_sel
                                qk_union_k_tensors[pool_name] |= k_sel

                            # Token co-selection
                            if w_q.dim() == 3:
                                q_flat = w_q.view(-1, w_q.shape[-1])
                                k_flat = w_k.view(-1, w_k.shape[-1])
                            else:
                                q_flat = w_q
                                k_flat = w_k
                            q_active = (q_flat > 0)
                            k_active = (k_flat > 0)
                            coselect = (q_active & k_active).sum(dim=-1).float()
                            union_co = (q_active | k_active).sum(dim=-1).float()
                            q_cnt = q_active.sum(dim=-1).float()
                            k_cnt = k_active.sum(dim=-1).float()
                            n_tokens = coselect.shape[0]

                            coselect_stats[pool_name]['coselect_sum'] += coselect.sum().item()
                            coselect_stats[pool_name]['union_sum'] += union_co.sum().item()
                            coselect_stats[pool_name]['q_sum'] += q_cnt.sum().item()
                            coselect_stats[pool_name]['k_sum'] += k_cnt.sum().item()
                            coselect_stats[pool_name]['count'] += n_tokens

                            per_layer_coselect[pool_name][layer_idx]['coselect_sum'] += coselect.sum().item()
                            per_layer_coselect[pool_name][layer_idx]['count'] += n_tokens

                        # ----- 5. Layer Contribution (output norms) -----
                        norms = layer.get_output_norms()
                        if 'attn_out_norm' in norms and 'know_out_norm' in norms:
                            layer_norms[layer_idx]['attn'].append(norms['attn_out_norm'])
                            layer_norms[layer_idx]['know'].append(norms['know_out_norm'])

                        # ----- 6. Path Usage -----
                        if max_paths > 1:
                            path_weights = layer.raw.get('path_weights', {})
                            if path_weights:
                                for pname, weights_list in path_weights.items():
                                    if not isinstance(weights_list, list):
                                        continue
                                    for path_idx, path_w in enumerate(weights_list):
                                        if path_w is None:
                                            continue
                                        pkey = f'{pname}_path{path_idx}'
                                        if path_w.dim() == 3:
                                            active = (path_w > 0).any(dim=-1)
                                            active_count = active.sum().item()
                                            total_tokens = path_w.shape[0] * path_w.shape[1]
                                            weight_sum = path_w.sum().item()
                                        else:
                                            active = (path_w > 0).any(dim=-1)
                                            active_count = active.sum().item()
                                            total_tokens = path_w.shape[0]
                                            weight_sum = path_w.sum().item()
                                        path_stats[pkey]['active_count'] += active_count
                                        path_stats[pkey]['total_tokens'] += total_tokens
                                        path_stats[pkey]['weight_sum'] += weight_sum

        finally:
            if hasattr(self.router, 'store_path_weights'):
                self.router.store_path_weights = False

        # Debug summary
        print(f"  [Summary] Processed: {processed_batches} batches, empty routing: {empty_routing_count}")
        print(f"  [Summary] Entropy data keys: {[k for k in entropy_data if entropy_data[k]]}")
        print(f"  [Summary] Selection tensors: {list(selection_tensors.keys())}")
        print(f"  [Summary] Union tensors: {len(union_tensors)} layer/key combinations")

        # ===== Aggregate all results =====
        results = {
            'entropy': self._finalize_entropy(entropy_data),
            'selection_frequency': self._finalize_selection_frequency(selection_tensors),
            'selection_diversity': self._finalize_selection_diversity(union_tensors, per_batch_counts, n_batches),
            'qk_overlap': self._finalize_qk_overlap(qk_overlap_data),
            'qk_usage': self._finalize_qk_usage(qk_layer_data, n_batches),
            'qk_entropy': self._finalize_qk_entropy(qk_entropy_data),
            'qk_union_coverage': self._finalize_qk_union_coverage(qk_union_q_tensors, qk_union_k_tensors, n_batches),
            'activation_sparsity': self._finalize_activation_sparsity(sparsity_stats, n_batches),
            'token_coselection': self._finalize_token_coselection(coselect_stats, per_layer_coselect, n_batches),
            'weight_concentration': self._finalize_weight_concentration(concentration_stats, n_batches),
            'path_usage': self._finalize_path_usage(path_stats, max_paths, n_batches),
            'layer_contribution': self._finalize_layer_contribution(layer_norms),
        }

        # Coverage progression (derived from selection_diversity)
        results['coverage_progression'] = self._compute_coverage_progression(results['selection_diversity'], n_batches)

        # Visualizations
        viz_path = self.visualize_qk_usage(results['qk_usage'], output_dir)
        if viz_path:
            results['qk_visualization'] = viz_path

        return results

    # ===== Helper methods for single-pass finalization =====

    def _compute_qk_jaccard(self, q_weights, k_weights) -> list:
        """Compute Jaccard similarity from Q/K weight tensors."""
        if q_weights.dim() == 3:
            q_flat = q_weights.view(-1, q_weights.shape[-1])
            k_flat = k_weights.view(-1, k_weights.shape[-1])
        else:
            q_flat = q_weights
            k_flat = k_weights

        q_active = (q_flat > 0).float()
        k_active = (k_flat > 0).float()
        intersection = (q_active * k_active).sum(dim=-1)
        union = ((q_active + k_active) > 0).float().sum(dim=-1)
        jaccard = intersection / (union + 1e-8)
        return jaccard.cpu().tolist()

    def _finalize_entropy(self, entropy_data: dict) -> dict:
        """Finalize entropy results."""
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

    def _finalize_selection_frequency(self, selection_tensors: dict) -> dict:
        """Finalize selection frequency results."""
        selection_counts = {}
        for key, tensor in selection_tensors.items():
            counts_np = tensor.cpu().numpy()
            selection_counts[key] = Counter({i: int(c) for i, c in enumerate(counts_np) if c > 0})

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            if key not in selection_counts:
                continue
            counts = selection_counts[key]
            if not counts:
                continue

            total = sum(counts.values())
            top10 = counts.most_common(10)
            unique = len(counts)

            n_attr = POOL_N_ATTR.get(pool)
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': total,
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': [(idx, cnt, cnt/total) for idx, cnt in top10],
                'concentration': sum(cnt for _, cnt in top10) / total if total > 0 else 0,
            }
        return results

    def _finalize_selection_diversity(self, union_tensors: dict, per_batch_counts: dict, n_batches: int) -> dict:
        """Finalize selection diversity results."""
        results = {}

        for layer_key, union_tensor in union_tensors.items():
            parts = layer_key.split('/')
            if len(parts) != 2:
                continue

            layer_str, routing_key = parts
            layer_idx = int(layer_str[1:])

            if routing_key in ROUTING_KEYS:
                pool = ROUTING_KEYS[routing_key][3]
                display = f'{layer_str}/{ROUTING_KEYS[routing_key][0]}'
            elif routing_key in KNOWLEDGE_ROUTING_KEYS:
                pool = KNOWLEDGE_ROUTING_KEYS[routing_key][2]
                display = f'{layer_str}/{KNOWLEDGE_ROUTING_KEYS[routing_key][0]}'
            else:
                continue

            n_attr = POOL_N_ATTR.get(pool)
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            union_count = union_tensor.sum().item()
            batch_counts = per_batch_counts[layer_key]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
            else:
                per_batch_avg = 0
                per_batch_std = 0

            results[layer_key] = {
                'display': display,
                'layer': layer_idx,
                'pool': pool,
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'union_count': int(union_count),
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        n_batches_processed = 0
        if per_batch_counts:
            first_key = next(iter(per_batch_counts))
            n_batches_processed = len(per_batch_counts[first_key])

        results['summary'] = {
            'n_batches_processed': min(n_batches, n_batches_processed),
            'n_layers': len(set(k.split('/')[0] for k in union_tensors.keys())),
            'total_keys_analyzed': len(union_tensors),
            'interpretation': (
                'High diversity_ratio (>2) = many neurons selected differently per batch\n'
                'Low diversity_ratio (~1) = same neurons always selected'
            )
        }
        return results

    def _finalize_qk_overlap(self, qk_overlap_data: dict) -> dict:
        """Finalize Q/K overlap results."""
        results = {}
        for key in ['fqk', 'rqk']:
            if qk_overlap_data[key]:
                mean_overlap = np.mean(qk_overlap_data[key])
                results[key] = {
                    'overlap_ratio': float(mean_overlap),
                    'jaccard': float(mean_overlap),
                    'std_overlap': float(np.std(qk_overlap_data[key])),
                    'interpretation': (
                        'Q and K select similar neurons' if mean_overlap > 0.3
                        else 'Q and K select different neurons'
                    ),
                }
        return results

    def _finalize_qk_usage(self, qk_layer_data: dict, n_batches: int) -> dict:
        """Finalize Q/K usage results."""
        results = {}

        for pool_name, pool_info in QK_POOLS.items():
            n_neurons = getattr(self.router, pool_info['n_attr'], 0)
            if n_neurons == 0 or not qk_layer_data[pool_name]:
                continue

            total_q = torch.zeros(n_neurons, device=self.device)
            total_k = torch.zeros(n_neurons, device=self.device)
            all_overlaps = []
            per_layer_results = {}

            for lidx, (q_counts, k_counts, batch_overlaps) in qk_layer_data[pool_name].items():
                total_q += q_counts
                total_k += k_counts
                all_overlaps.extend(batch_overlaps)

                q_np = q_counts.cpu().numpy()
                k_np = k_counts.cpu().numpy()
                corr = float(np.corrcoef(q_np, k_np)[0, 1]) if q_np.sum() > 0 and k_np.sum() > 0 else 0.0

                per_layer_results[f'L{lidx}'] = {
                    'correlation': corr,
                    'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
                    'q_total': int(q_np.sum()),
                    'k_total': int(k_np.sum()),
                }

            q_np = total_q.cpu().numpy()
            k_np = total_k.cpu().numpy()

            if q_np.sum() > 0 and k_np.sum() > 0:
                corr = float(np.corrcoef(q_np, k_np)[0, 1])
            else:
                corr = 0.0

            total_usage = q_np + k_np
            q_ratio = np.zeros_like(q_np, dtype=float)
            valid_mask = total_usage > 0
            q_ratio[valid_mask] = q_np[valid_mask] / total_usage[valid_mask]

            # active_mask 제거 - 모든 뉴런을 Q/K/Shared로 분류
            q_specialized = int((q_ratio > 0.7).sum())
            k_specialized = int((q_ratio < 0.3).sum())
            shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7)).sum())

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

            q_ratio_active = q_ratio.tolist()

            results[pool_name] = {
                'display': pool_info['display'],
                'n_neurons': n_neurons,
                'n_layers': len(qk_layer_data[pool_name]),
                'q_counts': q_np.tolist(),
                'k_counts': k_np.tolist(),
                'correlation': corr,
                'avg_overlap': float(np.mean(all_overlaps)) if all_overlaps else 0,
                'std_overlap': float(np.std(all_overlaps)) if all_overlaps else 0,
                'q_specialized': q_specialized,
                'k_specialized': k_specialized,
                'shared': shared,
                'q_total': int(q_np.sum()),
                'k_total': int(k_np.sum()),
                'per_layer': per_layer_results,
                'q_ratio': q_ratio.tolist(),
                'q_ratio_active': q_ratio_active,
                'specialization_thresholds': {
                    'q_specialized': 0.7,
                    'k_specialized': 0.3,
                },
                'sensitivity_analysis': sensitivity_analysis,
            }

        results['n_batches'] = n_batches
        return results

    def _finalize_qk_entropy(self, qk_entropy_data: dict) -> dict:
        """Finalize Q/K entropy results."""
        results = {}

        for pool_name, pool_info in QK_POOLS.items():
            pool_results = {}

            for layer_idx, ent_data in qk_entropy_data[pool_name].items():
                q_entropy = ent_data['q']
                k_entropy = ent_data['k']

                if q_entropy and k_entropy:
                    pool_results[f'L{layer_idx}'] = {
                        'q_entropy_mean': float(np.mean(q_entropy)),
                        'q_entropy_std': float(np.std(q_entropy)),
                        'k_entropy_mean': float(np.mean(k_entropy)),
                        'k_entropy_std': float(np.std(k_entropy)),
                        'entropy_diff': float(np.mean(q_entropy) - np.mean(k_entropy)),
                    }

            all_q = [v['q_entropy_mean'] for v in pool_results.values() if 'q_entropy_mean' in v]
            all_k = [v['k_entropy_mean'] for v in pool_results.values() if 'k_entropy_mean' in v]

            if all_q and all_k:
                pool_results['summary'] = {
                    'q_entropy_avg': float(np.mean(all_q)),
                    'k_entropy_avg': float(np.mean(all_k)),
                    'entropy_diff_avg': float(np.mean(all_q) - np.mean(all_k)),
                }

            results[pool_name] = {
                'display': pool_info['display'],
                'per_layer': pool_results,
                'q_entropy_mean': float(np.mean(all_q)) if all_q else 0,
                'q_entropy_std': float(np.std(all_q)) if all_q else 0,
                'k_entropy_mean': float(np.mean(all_k)) if all_k else 0,
                'k_entropy_std': float(np.std(all_k)) if all_k else 0,
                'entropy_diff': float(np.mean(all_q) - np.mean(all_k)) if all_q and all_k else 0,
            }

        return results

    def _finalize_qk_union_coverage(self, q_tensors: dict, k_tensors: dict, n_batches: int) -> dict:
        """Finalize Q/K union coverage results."""
        pool_info_union = {
            'feature_qk': ('fqk_q', 'fqk_k', 'n_feature_qk'),
            'restore_qk': ('rqk_q', 'rqk_k', 'n_restore_qk'),
        }

        results = {}
        for pool_name, (_, _, n_attr) in pool_info_union.items():
            n_total = getattr(self.router, n_attr, 0)
            if n_total == 0:
                continue

            q_mask = q_tensors.get(pool_name)
            k_mask = k_tensors.get(pool_name)

            if q_mask is None or k_mask is None:
                continue

            q_only = (q_mask & ~k_mask).sum().item()
            k_only = (k_mask & ~q_mask).sum().item()
            shared = (q_mask & k_mask).sum().item()
            union = (q_mask | k_mask).sum().item()
            dead = n_total - union

            results[pool_name] = {
                'n_total': n_total,
                'q_only': int(q_only),
                'k_only': int(k_only),
                'shared': int(shared),
                'dead': int(dead),
                'union_coverage': union / n_total if n_total > 0 else 0,
                'q_coverage': q_mask.sum().item() / n_total if n_total > 0 else 0,
                'k_coverage': k_mask.sum().item() / n_total if n_total > 0 else 0,
                'separation_ratio': (q_only + k_only) / union if union > 0 else 0,
            }

        results['n_batches'] = n_batches
        return results

    def _finalize_activation_sparsity(self, sparsity_stats: dict, n_batches: int) -> dict:
        """Finalize activation sparsity results."""
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

    def _finalize_token_coselection(self, coselect_stats: dict, per_layer_stats: dict, n_batches: int) -> dict:
        """Finalize token co-selection results."""
        results = {}
        for pool_name, stats in coselect_stats.items():
            if stats['count'] == 0:
                continue

            mean_coselect = stats['coselect_sum'] / stats['count']
            mean_union = stats['union_sum'] / stats['count']
            mean_q = stats['q_sum'] / stats['count']
            mean_k = stats['k_sum'] / stats['count']

            per_layer = {}
            for layer_idx, layer_stats in sorted(per_layer_stats[pool_name].items()):
                if layer_stats['count'] > 0:
                    per_layer[f'L{layer_idx}'] = {
                        'mean_coselect': layer_stats['coselect_sum'] / layer_stats['count'],
                    }

            results[pool_name] = {
                'mean_coselect_per_token': mean_coselect,
                'mean_union_per_token': mean_union,
                'coselect_rate': mean_coselect / mean_union if mean_union > 0 else 0,
                'mean_q_active': mean_q,
                'mean_k_active': mean_k,
                'per_layer': per_layer,
            }

        results['n_batches'] = n_batches
        return results

    def _finalize_weight_concentration(self, concentration_stats: dict, n_batches: int) -> dict:
        """Finalize weight concentration results."""
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

    def _finalize_path_usage(self, path_stats: dict, max_paths: int, n_batches: int) -> dict:
        """Finalize path usage results."""
        if max_paths <= 1:
            return {'note': 'Model does not support multi-path routing'}

        results = {'per_path': {}}
        pool_totals = defaultdict(lambda: {'active': 0, 'total': 0, 'weight': 0.0})

        for key, stats in path_stats.items():
            if stats['total_tokens'] > 0:
                activation_rate = stats['active_count'] / stats['total_tokens']
                results['per_path'][key] = {
                    'activation_rate': activation_rate,
                    'weight_sum': stats['weight_sum'],
                    'active_count': stats['active_count'],
                    'total_tokens': stats['total_tokens'],
                }

                pool_name = '_'.join(key.split('_')[:-1])
                pool_totals[pool_name]['active'] += stats['active_count']
                pool_totals[pool_name]['total'] += stats['total_tokens']
                pool_totals[pool_name]['weight'] += stats['weight_sum']

        results['per_pool'] = {}
        for pool_name, totals in pool_totals.items():
            if totals['total'] > 0:
                results['per_pool'][pool_name] = {
                    'avg_active_paths': totals['active'] / (totals['total'] / max_paths),
                    'total_weight': totals['weight'],
                }

        results['max_paths'] = max_paths
        results['n_batches'] = n_batches
        return results

    def _finalize_layer_contribution(self, layer_norms: dict) -> dict:
        """Finalize layer contribution results."""
        results = {'per_layer': {}, 'method': 'output_norm'}

        for layer_idx in sorted(layer_norms.keys()):
            attn_norms = layer_norms[layer_idx]['attn']
            know_norms = layer_norms[layer_idx]['know']

            if attn_norms and know_norms:
                mean_attn = float(np.mean(attn_norms))
                mean_know = float(np.mean(know_norms))
                total = mean_attn + mean_know

                results['per_layer'][f'L{layer_idx}'] = {
                    'layer_idx': layer_idx,
                    'attn_out_norm': mean_attn,
                    'know_out_norm': mean_know,
                    'attention_ratio': float(mean_attn / total * 100) if total > 0 else 50.0,
                    'knowledge_ratio': float(mean_know / total * 100) if total > 0 else 50.0,
                }

        if results['per_layer']:
            attn_ratios = [v['attention_ratio'] for v in results['per_layer'].values()]
            know_ratios = [v['knowledge_ratio'] for v in results['per_layer'].values()]
            results['summary'] = {
                'attention_ratio_mean': float(np.mean(attn_ratios)),
                'knowledge_ratio_mean': float(np.mean(know_ratios)),
                'attention_ratio_std': float(np.std(attn_ratios)),
                'knowledge_ratio_std': float(np.std(know_ratios)),
                'n_layers': len(results['per_layer']),
            }

        return results

    def _compute_coverage_progression(self, diversity_results: dict, n_batches: int) -> dict:
        """Compute coverage progression from selection diversity results."""
        layer_coverages = defaultdict(dict)

        for layer_key, data in diversity_results.items():
            if layer_key == 'summary' or not isinstance(data, dict):
                continue

            if 'layer' in data and 'union_coverage' in data:
                layer_idx = data['layer']
                pool = data.get('pool', 'unknown')
                layer_coverages[pool][layer_idx] = data['union_coverage']

        results = {}
        for pool, coverages in layer_coverages.items():
            if not coverages:
                continue

            sorted_layers = sorted(coverages.keys())
            n_layers = len(sorted_layers)

            if n_layers < 3:
                continue

            third = n_layers // 3
            early_layers = sorted_layers[:third]
            mid_layers = sorted_layers[third:2*third]
            late_layers = sorted_layers[2*third:]

            early_cov = np.mean([coverages[l] for l in early_layers])
            mid_cov = np.mean([coverages[l] for l in mid_layers])
            late_cov = np.mean([coverages[l] for l in late_layers])

            if late_cov > early_cov * 1.1:
                trend = 'increasing'
            elif late_cov < early_cov * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'

            growth_ratio = late_cov / early_cov if early_cov > 0 else 1.0

            results[pool] = {
                'early_coverage': early_cov,
                'mid_coverage': mid_cov,
                'late_coverage': late_cov,
                'trend': trend,
                'growth_ratio': growth_ratio,
                'early_layers': list(early_layers),
                'late_layers': list(late_layers),
            }

        results['n_batches'] = n_batches
        return results

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

        Returns:
            Dictionary with selection frequency statistics
        """
        selection_counts = {name: Counter() for name in ROUTING_KEYS.keys()}

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

                        # Get indices of active neurons (weight > 0)
                        active_mask = weights > 0
                        active_indices = active_mask.nonzero(as_tuple=False)[:, -1].cpu().numpy()

                        for idx in active_indices:
                            selection_counts[key][int(idx)] += 1

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

    def analyze_selection_diversity(self, dataloader, n_batches: int = 100) -> Dict:
        """
        Analyze selection diversity across batches.

        Measures how many unique neurons are selected across the entire dataset
        vs per-batch selection.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with diversity metrics
        """
        # Use defaultdict for dynamic layer keys
        union_selected = defaultdict(set)
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
                            if weights.dim() == 3:
                                selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())
                            elif weights.dim() == 2:
                                selected = (weights > 0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())

                    # Knowledge routing - use standardized keys
                    for key in KNOWLEDGE_ROUTING_KEYS.keys():
                        layer_key = f'L{layer_idx}/{key}'
                        weights = layer.get_weight(key)
                        if weights is not None:
                            if weights.dim() == 3:
                                selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())
                            elif weights.dim() == 2:
                                selected = (weights > 0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())

        # Build results for all layer/key combinations
        results = {}

        for layer_key in union_selected.keys():
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

            union_count = len(union_selected[layer_key])
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
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        total_union = sum(len(union_selected[k]) for k in union_selected)
        n_batches_processed = 0
        if per_batch_counts:
            first_key = next(iter(per_batch_counts))
            n_batches_processed = len(per_batch_counts[first_key])

        results['summary'] = {
            'n_batches_processed': min(n_batches, n_batches_processed),
            'n_layers': len(set(k.split('/')[0] for k in union_selected.keys())),
            'total_keys_analyzed': len(union_selected),
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

        def compute_jaccard_from_weights(q_weights, k_weights):
            """Compute Jaccard similarity from weight tensors (active neurons)."""
            # Flatten if 3D: [B, S, N] -> [B*S, N]
            if q_weights.dim() == 3:
                q_flat = q_weights.view(-1, q_weights.shape[-1])
                k_flat = k_weights.view(-1, k_weights.shape[-1])
            else:
                q_flat = q_weights
                k_flat = k_weights

            # Active neurons (weight > 0)
            q_active = (q_flat > 0).float()
            k_active = (k_flat > 0).float()

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
                        overlap_data['fqk'].extend(compute_jaccard_from_weights(fqk_q, fqk_k))

                    # R-QK Q/K overlap using standardized keys
                    rqk_q = layer.get_weight('rqk_q')
                    rqk_k = layer.get_weight('rqk_k')
                    if rqk_q is not None and rqk_k is not None:
                        overlap_data['rqk'].extend(compute_jaccard_from_weights(rqk_q, rqk_k))

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
                    )
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

                # Specialization analysis
                threshold = (q_np.sum() + k_np.sum()) / (2 * len(q_np)) * 0.1
                q_only = int(((q_np > threshold) & (k_np < threshold)).sum())
                k_only = int(((k_np > threshold) & (q_np < threshold)).sum())
                shared = int(((q_np > threshold) & (k_np > threshold)).sum())
                inactive = int(((q_np < threshold) & (k_np < threshold)).sum())

                results[pool_name] = {
                    'display': pool_info['display'],
                    'n_neurons': n_neurons,
                    'n_layers': len(layer_data),
                    'q_counts': q_np.tolist(),
                    'k_counts': k_np.tolist(),
                    'correlation': corr,
                    'avg_overlap': float(np.mean(all_overlaps)) if all_overlaps else 0,
                    'std_overlap': float(np.std(all_overlaps)) if all_overlaps else 0,
                    'q_specialized': q_only,
                    'k_specialized': k_only,
                    'shared': shared,
                    'inactive': inactive,
                    'q_total': int(q_np.sum()),
                    'k_total': int(k_np.sum()),
                    'per_layer': per_layer_results,
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

        Paper Figure 6b data generation.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with layer-wise attention/knowledge ratios
        """
        # {layer_idx: {'attention': [], 'knowledge': []}}
        layer_contributions = defaultdict(lambda: {'attention': [], 'knowledge': []})

        # Standardized keys for attention and knowledge
        ATTENTION_KEYS = ['fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k']
        KNOWLEDGE_KEYS = ['fknow', 'rknow']

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, desc='Layer Contribution', total=n_batches)):
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

                # Process all layers using extractor
                for layer in routing:
                    layer_idx = layer.layer_idx

                    # Attention contribution: sum of all attention routing weights
                    attn_contrib = 0.0
                    for key in ATTENTION_KEYS:
                        w = layer.get_weight(key)
                        if w is not None:
                            attn_contrib += w.sum().item()

                    # Knowledge contribution: sum of knowledge weights
                    know_contrib = 0.0
                    for key in KNOWLEDGE_KEYS:
                        w = layer.get_weight(key)
                        if w is not None:
                            know_contrib += w.sum().item()

                    layer_contributions[layer_idx]['attention'].append(attn_contrib)
                    layer_contributions[layer_idx]['knowledge'].append(know_contrib)

        # Aggregate results
        results = {'per_layer': {}}
        for layer_idx in sorted(layer_contributions.keys()):
            contribs = layer_contributions[layer_idx]
            attn_mean = np.mean(contribs['attention']) if contribs['attention'] else 0
            know_mean = np.mean(contribs['knowledge']) if contribs['knowledge'] else 0
            total = attn_mean + know_mean

            results['per_layer'][f'L{layer_idx}'] = {
                'layer_idx': layer_idx,
                'attention_sum': float(attn_mean),
                'knowledge_sum': float(know_mean),
                'attention_ratio': float(attn_mean / total) if total > 0 else 0.5,
                'knowledge_ratio': float(know_mean / total) if total > 0 else 0.5,
            }

        # Summary
        if results['per_layer']:
            attn_ratios = [v['attention_ratio'] for v in results['per_layer'].values()]
            know_ratios = [v['knowledge_ratio'] for v in results['per_layer'].values()]
            results['summary'] = {
                'attention_ratio_mean': float(np.mean(attn_ratios)),
                'knowledge_ratio_mean': float(np.mean(know_ratios)),
                'n_layers': len(results['per_layer']),
            }

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
        Run all routing analyses.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'entropy': self.analyze_entropy(dataloader, n_batches),
            'selection_frequency': self.analyze_selection_frequency(dataloader, n_batches),
            'selection_diversity': self.analyze_selection_diversity(dataloader, n_batches * 2),
            'qk_overlap': self.analyze_qk_overlap(dataloader, n_batches),
            'qk_usage': self.analyze_qk_usage(dataloader, n_batches * 2),
            'qk_entropy': self.analyze_qk_entropy(dataloader, n_batches),
        }

        # Visualizations
        viz_path = self.visualize_qk_usage(results['qk_usage'], output_dir)
        if viz_path:
            results['qk_visualization'] = viz_path

        return results

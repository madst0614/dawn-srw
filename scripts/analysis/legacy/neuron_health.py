"""
Neuron Health Analysis (Forward-based)
======================================
Analyze neuron health status using forward pass data.

All metrics computed from actual routing weights during inference,
not from EMA statistics.
"""

import os
import numpy as np
import torch
from typing import Dict, Optional
from collections import defaultdict

from .base import BaseAnalyzer
from .utils import (
    ALL_ROUTING_KEYS, POOL_N_ATTR,
    gini_coefficient, get_batch_input_ids,
    RoutingDataExtractor,
    HAS_TQDM, tqdm
)


class NeuronHealthAnalyzer(BaseAnalyzer):
    """Forward-based neuron health analyzer."""

    def __init__(self, model, router=None, device: str = 'cuda', extractor=None):
        super().__init__(model, router=router, device=device)
        self.extractor = extractor or RoutingDataExtractor(model, device=device)

    def analyze_activation_distribution(
        self,
        dataloader,
        n_batches: int = 50,
        threshold: float = 0.01
    ) -> Dict:
        """
        Analyze neuron activation distribution from forward passes.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            threshold: Weight threshold for "active" classification

        Returns:
            Dictionary with per-pool activation statistics
        """
        # Accumulators: pool -> tensor of activation counts
        activation_counts = {}
        total_tokens = 0

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Health Analysis')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens

                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for key in ALL_ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        pool = ALL_ROUTING_KEYS[key][3]
                        n_attr = POOL_N_ATTR.get(pool)
                        n_neurons = getattr(self.router, n_attr, 0) if n_attr else weights.shape[-1]

                        if pool not in activation_counts:
                            activation_counts[pool] = torch.zeros(n_neurons, device=self.device)

                        # Count activations (weight > threshold)
                        if weights.dim() == 3:
                            active = (weights > threshold).float().sum(dim=[0, 1])
                        else:
                            active = (weights > threshold).float().sum(dim=0)
                        activation_counts[pool] += active

        # Compute statistics
        results = {}
        for pool, counts in activation_counts.items():
            counts_np = counts.cpu().numpy()
            n_total = len(counts_np)
            n_active = int((counts_np > 0).sum())
            n_dead = n_total - n_active

            # Normalize to get activation frequency
            freq = counts_np / (total_tokens + 1e-8)

            results[pool] = {
                'total': n_total,
                'active': n_active,
                'dead': n_dead,
                'active_ratio': n_active / n_total if n_total > 0 else 0,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'gini': gini_coefficient(torch.from_numpy(freq)),
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
        dataloader,
        n_batches: int = 50,
        threshold: float = 0.01,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Identify dead neurons from forward passes.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            threshold: Weight threshold for activation
            output_dir: Directory for visualization output

        Returns:
            Dictionary with dead neuron analysis
        """
        # Track which neurons were ever activated
        ever_activated = {}

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Dead Neuron Analysis')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for key in ALL_ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        pool = ALL_ROUTING_KEYS[key][3]
                        n_attr = POOL_N_ATTR.get(pool)
                        n_neurons = getattr(self.router, n_attr, 0) if n_attr else weights.shape[-1]

                        if pool not in ever_activated:
                            ever_activated[pool] = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

                        # Mark neurons that were activated
                        if weights.dim() == 3:
                            active = (weights > threshold).any(dim=0).any(dim=0)
                        else:
                            active = (weights > threshold).any(dim=0)
                        ever_activated[pool] |= active

        # Compile results
        results = {}
        total_dead = 0
        total_neurons = 0

        for pool, activated in ever_activated.items():
            n_total = len(activated)
            n_active = int(activated.sum().item())
            n_dead = n_total - n_active
            dead_ids = (~activated).nonzero().squeeze(-1).tolist()

            total_dead += n_dead
            total_neurons += n_total

            results[pool] = {
                'n_total': n_total,
                'n_active': n_active,
                'n_dead': n_dead,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'dead_neuron_ids': dead_ids if isinstance(dead_ids, list) else [dead_ids] if dead_ids else [],
            }

        results['summary'] = {
            'total_dead': total_dead,
            'total_neurons': total_neurons,
            'dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
        }

        # Visualization
        if output_dir:
            try:
                from .visualizers import plot_dead_neurons
                os.makedirs(output_dir, exist_ok=True)
                pools = [p for p in results.keys() if p != 'summary']
                path = plot_dead_neurons(results, pools, os.path.join(output_dir, 'dead_neurons.png'))
                if path:
                    results['visualization'] = path
            except ImportError:
                pass

        return results

    def analyze_diversity(
        self,
        dataloader,
        n_batches: int = 50,
        threshold: float = 0.01
    ) -> Dict:
        """
        Analyze neuron usage diversity from forward passes.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            threshold: Weight threshold for activation

        Returns:
            Dictionary with diversity metrics
        """
        # Accumulators
        activation_counts = {}
        total_tokens = 0

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Diversity Analysis')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                total_tokens += input_ids.numel()

                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    for key in ALL_ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        pool = ALL_ROUTING_KEYS[key][3]
                        n_attr = POOL_N_ATTR.get(pool)
                        n_neurons = getattr(self.router, n_attr, 0) if n_attr else weights.shape[-1]

                        if pool not in activation_counts:
                            activation_counts[pool] = torch.zeros(n_neurons, device=self.device)

                        if weights.dim() == 3:
                            active = (weights > threshold).float().sum(dim=[0, 1])
                        else:
                            active = (weights > threshold).float().sum(dim=0)
                        activation_counts[pool] += active

        # Compute diversity metrics
        results = {}
        entropies = []

        for pool, counts in activation_counts.items():
            n_total = len(counts)
            active_mask = counts > 0
            n_active = int(active_mask.sum().item())

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

            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            # Top-k concentration
            top5 = torch.topk(active_counts, min(5, n_active))[0]
            top5_share = (top5.sum() / active_counts.sum()).item()

            results[pool] = {
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': float(top5_share),
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
        dataloader,
        output_dir: str = './neuron_health',
        n_batches: int = 50
    ) -> Dict:
        """
        Run all neuron health analyses (forward-based).

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'activation_distribution': self.analyze_activation_distribution(dataloader, n_batches),
            'diversity': self.analyze_diversity(dataloader, n_batches),
            'dead_neurons': self.analyze_dead_neurons(dataloader, n_batches, output_dir=output_dir),
        }

        return results

"""
Weight Matrix Analysis
======================
Analyze neuron weight matrices in DAWN v17.1 models.

Includes:
- SVD decomposition analysis
- Effective rank calculation
- Condition number analysis
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .base import BaseAnalyzer
from .utils import (
    NEURON_TYPES, NEURON_ATTRS,
    HAS_MATPLOTLIB, plt
)


class WeightAnalyzer(BaseAnalyzer):
    """Neuron weight matrix analyzer."""

    def __init__(self, model=None, neurons=None, device='cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model (optional, for auto-detection)
            neurons: SharedNeurons instance (direct, or auto-detected from model)
            device: Device for computation
        """
        if model is not None:
            super().__init__(model, device=device)
            self.neurons = self.shared_neurons
        else:
            # Backward compatibility: direct neurons argument
            self.model = None
            self.router = None
            self.shared_neurons = neurons
            self.neurons = neurons
            self.device = device

    def analyze_weight_svd(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze neuron weights using SVD decomposition.
        Optimized with GPU tensor operations.

        Computes:
        - Singular value distribution
        - Effective rank (number of significant singular values)
        - Condition number
        - Variance explained by top singular values

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with SVD analysis results
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            # Keep on GPU for SVD
            W = getattr(self.neurons, attr).detach().to(self.device)
            n_neurons = W.shape[0]

            # Flatten to 2D for SVD
            if W.dim() > 2:
                W = W.reshape(n_neurons, -1)

            try:
                # SVD on GPU
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                results[name] = {'error': 'SVD failed'}
                continue

            # Compute metrics on GPU
            S_normalized = S / S.sum()
            cumsum = torch.cumsum(S_normalized, dim=0)
            effective_rank = float((S > S.max() * 0.01).sum().item())
            var_top5 = float(cumsum[min(4, len(cumsum)-1)].item())
            var_top10 = float(cumsum[min(9, len(cumsum)-1)].item())

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'weight_shape': list(W.shape),
                'effective_rank': effective_rank,
                'var_explained_by_top5': var_top5,
                'var_explained_by_top10': var_top10,
                'top_singular_values': S[:10].cpu().tolist(),
                'condition_number': float(S[0] / S[-1]) if S[-1] > 0 else float('inf'),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._visualize_svd(results, output_dir)

        return results

    def _visualize_svd(self, results: Dict, output_dir: str):
        """Generate SVD visualization."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if not valid_results:
            return

        n_plots = len(valid_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, valid_results.items()):
            sv = data['top_singular_values']
            ax.bar(range(len(sv)), sv)
            ax.set_title(f'{data["display"]} Singular Values')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')

        plt.tight_layout()
        path = os.path.join(output_dir, 'weight_svd.png')
        plt.savefig(path, dpi=150)
        plt.close()
        results['visualization'] = path

    def analyze_weight_norms(self) -> Dict:
        """
        Analyze weight matrix norms per neuron.
        Optimized with GPU tensor operations.

        Returns:
            Dictionary with norm statistics
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            # Keep on GPU
            W = getattr(self.neurons, attr).detach().to(self.device)
            n_neurons = W.shape[0]

            # Compute per-neuron norms on GPU
            norms = torch.norm(W.reshape(n_neurons, -1), dim=1)

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'mean_norm': float(norms.mean().item()),
                'std_norm': float(norms.std().item()),
                'min_norm': float(norms.min().item()),
                'max_norm': float(norms.max().item()),
            }

        return results

    def analyze_weight_similarity(self) -> Dict:
        """
        Analyze pairwise similarity between neuron weight matrices.
        Optimized with GPU tensor operations.

        Returns:
            Dictionary with similarity statistics
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            # Keep on GPU
            W = getattr(self.neurons, attr).detach().to(self.device)
            n_neurons = W.shape[0]

            if n_neurons < 2:
                continue

            # Flatten and normalize on GPU
            W_flat = W.reshape(n_neurons, -1)
            W_norm = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-8)

            # Compute similarity matrix on GPU
            sim = torch.mm(W_norm, W_norm.t())

            # Extract off-diagonal
            mask = ~torch.eye(n_neurons, dtype=torch.bool, device=self.device)
            off_diag = sim[mask]

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'mean_similarity': float(off_diag.mean().item()),
                'std_similarity': float(off_diag.std().item()),
                'max_similarity': float(off_diag.max().item()),
                'min_similarity': float(off_diag.min().item()),
            }

        return results

    def run_all(self, output_dir: str = './weight_analysis') -> Dict:
        """
        Run all weight analyses.

        Args:
            output_dir: Directory for outputs

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'svd': self.analyze_weight_svd(output_dir),
            'norms': self.analyze_weight_norms(),
            'similarity': self.analyze_weight_similarity(),
        }

        return results

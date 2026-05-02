"""
Embedding Analysis
==================
Analyze neuron embeddings in DAWN models.

Includes:
- Intra-type similarity analysis
- Cross-type similarity analysis
- Clustering analysis
- t-SNE/PCA visualization
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .base import BaseAnalyzer
from .utils import HAS_SKLEARN

if HAS_SKLEARN:
    from sklearn.cluster import KMeans


class EmbeddingAnalyzer(BaseAnalyzer):
    """Neuron embedding analyzer."""

    def __init__(self, model, router=None, device: str = 'cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model instance
            router: NeuronRouter (auto-detected if None)
            device: Device for computation
        """
        super().__init__(model, router=router, device=device)

    def get_embeddings_by_type(self, as_tensor: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract embeddings grouped by embedding pool (not EMA type).

        Args:
            as_tensor: Return torch tensors on GPU instead of numpy arrays

        Returns:
            Dictionary mapping pool name to embedding array/tensor
        """
        emb = self.router.neuron_emb.detach()
        if not as_tensor:
            emb = emb.cpu().numpy()

        # Use embedding pools (6 unique pools) not neuron types (8 with Q/K separate)
        pools = self.get_embedding_pools()
        result = {}
        offset = 0
        for name, (display, n_attr, _) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                result[name] = emb[offset:offset + n]
                offset += n

        return result

    def analyze_similarity(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze intra-type similarity using cosine similarity.
        Optimized with GPU tensor operations.

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with similarity statistics
        """
        # Get embeddings as GPU tensors
        embeddings_gpu = self.get_embeddings_by_type(as_tensor=True)
        results = {}
        pools = self.get_embedding_pools()

        for name, emb in embeddings_gpu.items():
            if len(emb) < 2:
                continue

            # Ensure on GPU
            emb = emb.to(self.device)

            # Normalize and compute similarity matrix on GPU
            emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = torch.mm(emb_norm, emb_norm.t())

            # Extract off-diagonal elements
            n = sim_matrix.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
            off_diag = sim_matrix[mask]

            display = pools[name][0]
            results[name] = {
                'display': display,
                'n_neurons': n,
                'avg_similarity': float(off_diag.mean().item()),
                'max_similarity': float(off_diag.max().item()),
                'min_similarity': float(off_diag.min().item()),
                'std_similarity': float(off_diag.std().item()),
            }

        # Visualization (needs numpy)
        if output_dir:
            from .visualizers import plot_similarity_heatmap
            os.makedirs(output_dir, exist_ok=True)
            embeddings_np = self.get_embeddings_by_type(as_tensor=False)
            pool_display = {name: pools[name][0] for name in pools}
            path = plot_similarity_heatmap(
                embeddings_np, pool_display,
                os.path.join(output_dir, 'similarity_heatmap.png')
            )
            if path:
                results['visualization'] = path

        return results

    def analyze_cross_type_similarity(self) -> Dict:
        """
        Analyze similarity between neuron types using centroids.
        Optimized with GPU tensor operations.

        Returns:
            Dictionary with pairwise centroid similarities
        """
        embeddings = self.get_embeddings_by_type(as_tensor=True)
        pools = self.get_embedding_pools()

        # Compute centroids on GPU
        centroids = {}
        for name, emb in embeddings.items():
            emb = emb.to(self.device)
            centroids[name] = emb.mean(dim=0)

        # Compute pairwise similarities on GPU
        results = {}
        names = list(centroids.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                c1, c2 = centroids[n1], centroids[n2]
                sim = torch.dot(c1, c2) / (c1.norm() * c2.norm() + 1e-8)
                key = f"{pools[n1][0]}-{pools[n2][0]}"
                results[key] = float(sim.item())

        return results

    def analyze_clustering(self, n_clusters: int = 5, output_dir: Optional[str] = None) -> Dict:
        """
        Perform clustering analysis on neuron embeddings.

        Args:
            n_clusters: Number of clusters
            output_dir: Directory for visualization output

        Returns:
            Dictionary with clustering results
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        results = {}
        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Use embedding pools (6 unique pools) for correct boundaries
        pools = self.get_embedding_pools()

        # Build boundaries for each pool
        boundaries = {}
        offset = 0
        for name, (display, n_attr, _) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                boundaries[name] = (offset, offset + n, display)
                offset += n

        # Cluster each pool
        for name, (start, end, display) in boundaries.items():
            pool_emb = emb[start:end]
            n_neurons = pool_emb.shape[0]

            if n_neurons < n_clusters:
                results[name] = {'error': f'Not enough neurons for {n_clusters} clusters'}
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pool_emb)

            cluster_stats = []
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_size = cluster_mask.sum()
                cluster_stats.append({
                    'cluster_id': c,
                    'size': int(cluster_size),
                })

            results[name] = {
                'display': display,
                'n_clusters': n_clusters,
                'clusters': sorted(cluster_stats, key=lambda x: -x['size']),
                'labels': labels.tolist(),
            }

        # Visualization
        if output_dir:
            from .visualizers import plot_clustering
            os.makedirs(output_dir, exist_ok=True)
            viz_boundaries = {name: (start, end, display) for name, (start, end, _, display) in boundaries.items()}
            path = plot_clustering(emb, viz_boundaries, results, os.path.join(output_dir, 'clustering.png'))
            if path:
                results['visualization'] = path

        return results

    def visualize(self, output_dir: str) -> Optional[str]:
        """
        Generate t-SNE/PCA visualization of all embeddings.

        Args:
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        from .visualizers import plot_embedding_space

        os.makedirs(output_dir, exist_ok=True)

        emb = self.router.neuron_emb.detach().cpu().numpy()
        pools = self.get_embedding_pools()

        # Build labels and color map
        labels = []
        colors_map = {}
        for name, (display, n_attr, color) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                labels.extend([display] * n)
                colors_map[display] = color

        return plot_embedding_space(emb, labels, colors_map, os.path.join(output_dir, 'dawn_embeddings.png'))

    def run_all(self, output_dir: str = './embedding_analysis', n_clusters: int = 5) -> Dict:
        """
        Run all embedding analyses.

        Args:
            output_dir: Directory for outputs
            n_clusters: Number of clusters for clustering analysis

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'similarity': self.analyze_similarity(output_dir),
            'cross_type_similarity': self.analyze_cross_type_similarity(),
            'clustering': self.analyze_clustering(n_clusters, output_dir),
        }

        # Main visualization
        viz_path = self.visualize(output_dir)
        if viz_path:
            results['visualization'] = viz_path

        return results

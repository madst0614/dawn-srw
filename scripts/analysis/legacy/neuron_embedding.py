"""
Neuron Embedding Analysis
=========================
Unsupervised analysis of DAWN neuron embeddings to understand
what features neurons have learned.

Includes:
- Clustering analysis (K-means, optimal k finding)
- Pool-wise embedding distribution
- Cluster interpretation with activation patterns
- Visualization (t-SNE/UMAP)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .base import BaseAnalyzer
from .utils import (
    EMBEDDING_POOLS_V18, POOL_N_ATTR,
    get_batch_input_ids, gini_coefficient,
    HAS_MATPLOTLIB, HAS_SKLEARN, HAS_TQDM, tqdm, plt, sns
)

if HAS_SKLEARN:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class NeuronEmbeddingAnalyzer(BaseAnalyzer):
    """Analyzer for DAWN neuron embeddings."""

    def __init__(self, model, router=None, device='cuda', extractor=None):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance (auto-detected if None)
            device: Device for computation
            extractor: RoutingDataExtractor instance (optional)
        """
        super().__init__(model, router=router, device=device)

        # Import extractor if needed
        if extractor is None:
            from .utils import RoutingDataExtractor
            extractor = RoutingDataExtractor(model, device=device)
        self.extractor = extractor

        # Get neuron embeddings
        self.neuron_emb = self._get_neuron_embeddings()
        self.pool_boundaries = self._get_pool_boundaries()

    def _get_neuron_embeddings(self) -> Optional[torch.Tensor]:
        """Extract neuron embeddings from model (normalized to match routing)."""
        if hasattr(self.router, 'neuron_emb'):
            emb = self.router.neuron_emb.detach()
            # Normalize to match actual routing (cosine similarity based)
            return F.normalize(emb, dim=-1)
        return None

    def _get_pool_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """
        Get pool boundaries from router.

        Returns:
            Dict mapping pool_name -> (start_idx, end_idx)
        """
        boundaries = {}
        current_idx = 0

        # Order of pools in neuron_emb
        pool_order = [
            'feature_qk', 'feature_v', 'restore_qk', 'restore_v',
            'feature_know', 'restore_know'
        ]

        for pool_name in pool_order:
            n_attr = POOL_N_ATTR.get(pool_name)
            if n_attr:
                n_neurons = getattr(self.router, n_attr, 0)
                if n_neurons > 0:
                    boundaries[pool_name] = (current_idx, current_idx + n_neurons)
                    current_idx += n_neurons

        return boundaries

    def get_pool_embeddings(self, pool_name: str) -> Optional[torch.Tensor]:
        """Get embeddings for a specific pool."""
        if self.neuron_emb is None:
            return None

        if pool_name not in self.pool_boundaries:
            return None

        start, end = self.pool_boundaries[pool_name]
        return self.neuron_emb[start:end]

    def analyze_clustering(
        self,
        k_range: Tuple[int, int] = (5, 30),
        method: str = 'kmeans',
        pool_name: Optional[str] = None
    ) -> Dict:
        """
        Perform clustering analysis on neuron embeddings.

        Args:
            k_range: Range of k values to try for optimal k
            method: Clustering method ('kmeans' or 'hierarchical')
            pool_name: Specific pool to analyze (None = all)

        Returns:
            Dictionary with clustering results
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        # Get embeddings
        if pool_name:
            emb = self.get_pool_embeddings(pool_name)
            if emb is None:
                return {'error': f'Pool {pool_name} not found'}
        else:
            emb = self.neuron_emb

        emb_np = emb.cpu().numpy()

        # Find optimal k using silhouette score
        k_min, k_max = k_range
        k_max = min(k_max, len(emb_np) - 1)

        silhouette_scores = []
        calinski_scores = []
        inertias = []

        for k in tqdm(range(k_min, k_max + 1), desc='Finding optimal k'):
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                clusterer = AgglomerativeClustering(n_clusters=k)

            labels = clusterer.fit_predict(emb_np)

            sil = silhouette_score(emb_np, labels)
            cal = calinski_harabasz_score(emb_np, labels)

            silhouette_scores.append(sil)
            calinski_scores.append(cal)

            if hasattr(clusterer, 'inertia_'):
                inertias.append(clusterer.inertia_)

        # Find optimal k
        optimal_k_sil = k_min + np.argmax(silhouette_scores)
        optimal_k_cal = k_min + np.argmax(calinski_scores)

        # Use silhouette-optimal k for final clustering
        optimal_k = optimal_k_sil
        if method == 'kmeans':
            final_clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        else:
            final_clusterer = AgglomerativeClustering(n_clusters=optimal_k)

        final_labels = final_clusterer.fit_predict(emb_np)

        # Cluster statistics
        cluster_stats = {}
        for c in range(optimal_k):
            mask = final_labels == c
            cluster_emb = emb_np[mask]
            cluster_stats[c] = {
                'size': int(mask.sum()),
                'mean_norm': float(np.linalg.norm(cluster_emb, axis=1).mean()),
                'std_norm': float(np.linalg.norm(cluster_emb, axis=1).std()),
            }

            # If analyzing all pools, compute pool composition
            if pool_name is None:
                pool_composition = {}
                cluster_indices = np.where(mask)[0]
                for pname, (start, end) in self.pool_boundaries.items():
                    count = ((cluster_indices >= start) & (cluster_indices < end)).sum()
                    pool_composition[pname] = int(count)
                cluster_stats[c]['pool_composition'] = pool_composition

        # Compute cluster centroids
        if hasattr(final_clusterer, 'cluster_centers_'):
            centroids = final_clusterer.cluster_centers_
        else:
            centroids = np.array([emb_np[final_labels == c].mean(axis=0) for c in range(optimal_k)])

        return {
            'method': method,
            'pool': pool_name or 'all',
            'n_embeddings': len(emb_np),
            'd_space': emb_np.shape[1],
            'k_range': list(range(k_min, k_max + 1)),
            'silhouette_scores': [float(s) for s in silhouette_scores],
            'calinski_scores': [float(s) for s in calinski_scores],
            'inertias': [float(i) for i in inertias] if inertias else None,
            'optimal_k_silhouette': int(optimal_k_sil),
            'optimal_k_calinski': int(optimal_k_cal),
            'optimal_k': int(optimal_k),
            'best_silhouette': float(max(silhouette_scores)),
            'cluster_labels': final_labels.tolist(),
            'cluster_stats': cluster_stats,
            'centroids': centroids.tolist(),
        }

    def analyze_pool_alignment(self, method: str = 'kmeans') -> Dict:
        """
        Analyze how well k=6 clusters align with the 6 neuron pools.

        This tests whether neurons naturally cluster by their pool assignments,
        which would indicate that pools have distinct embedding patterns.

        Args:
            method: Clustering method ('kmeans' or 'hierarchical')

        Returns:
            Dictionary with pool-cluster alignment analysis
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        emb_np = self.neuron_emb.cpu().numpy()
        n_pools = len(self.pool_boundaries)

        if n_pools == 0:
            return {'error': 'No pools found'}

        # Create true pool labels for each neuron
        true_labels = np.zeros(len(emb_np), dtype=int)
        pool_names = list(self.pool_boundaries.keys())
        for pool_idx, (pool_name, (start, end)) in enumerate(self.pool_boundaries.items()):
            true_labels[start:end] = pool_idx

        # Cluster with k = number of pools
        k = n_pools
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            clusterer = AgglomerativeClustering(n_clusters=k)

        pred_labels = clusterer.fit_predict(emb_np)

        # Build confusion matrix: [n_pools, n_clusters]
        confusion = np.zeros((n_pools, k), dtype=int)
        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion[true_label, pred_label] += 1

        # Compute metrics
        # 1. Purity: for each cluster, count majority pool / total
        cluster_purity = []
        for c in range(k):
            cluster_mask = pred_labels == c
            if cluster_mask.sum() == 0:
                cluster_purity.append(0.0)
                continue
            cluster_true = true_labels[cluster_mask]
            majority_count = np.bincount(cluster_true).max()
            purity = majority_count / len(cluster_true)
            cluster_purity.append(float(purity))

        overall_purity = np.mean(cluster_purity)

        # 2. Pool coverage: for each pool, what fraction ends up in one cluster?
        pool_coverage = []
        for p in range(n_pools):
            pool_dist = confusion[p]
            if pool_dist.sum() == 0:
                pool_coverage.append(0.0)
                continue
            coverage = pool_dist.max() / pool_dist.sum()
            pool_coverage.append(float(coverage))

        # 3. Silhouette score with true labels vs predicted labels
        sil_true = silhouette_score(emb_np, true_labels)
        sil_pred = silhouette_score(emb_np, pred_labels)

        # 4. Best cluster assignment for each pool (Hungarian-like matching)
        best_cluster_for_pool = {}
        best_pool_for_cluster = {}
        for p_idx, pool_name in enumerate(pool_names):
            best_cluster = int(np.argmax(confusion[p_idx]))
            best_cluster_for_pool[pool_name] = {
                'cluster': best_cluster,
                'count': int(confusion[p_idx, best_cluster]),
                'coverage': float(confusion[p_idx, best_cluster] / confusion[p_idx].sum())
            }

        for c in range(k):
            best_pool_idx = int(np.argmax(confusion[:, c]))
            best_pool_for_cluster[c] = {
                'pool': pool_names[best_pool_idx],
                'count': int(confusion[best_pool_idx, c]),
                'purity': cluster_purity[c]
            }

        # 5. Detailed confusion matrix with pool names
        confusion_dict = {}
        for p_idx, pool_name in enumerate(pool_names):
            confusion_dict[pool_name] = {
                f'cluster_{c}': int(confusion[p_idx, c]) for c in range(k)
            }

        return {
            'method': method,
            'k': k,
            'n_neurons': len(emb_np),
            'n_pools': n_pools,
            'pool_names': pool_names,
            'confusion_matrix': confusion_dict,
            'cluster_purity': {c: cluster_purity[c] for c in range(k)},
            'overall_purity': float(overall_purity),
            'pool_coverage': {pool_names[p]: pool_coverage[p] for p in range(n_pools)},
            'mean_pool_coverage': float(np.mean(pool_coverage)),
            'silhouette_true_labels': float(sil_true),
            'silhouette_pred_labels': float(sil_pred),
            'best_cluster_for_pool': best_cluster_for_pool,
            'best_pool_for_cluster': best_pool_for_cluster,
            'cluster_labels': pred_labels.tolist(),
        }

    def analyze_pool_distribution(self) -> Dict:
        """
        Analyze embedding distribution across pools.

        Returns:
            Dictionary with pool-wise statistics
        """
        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        results = {'pools': {}}

        for pool_name, (start, end) in self.pool_boundaries.items():
            pool_emb = self.neuron_emb[start:end].cpu().numpy()

            # Basic statistics
            norms = np.linalg.norm(pool_emb, axis=1)
            mean_emb = pool_emb.mean(axis=0)

            # Compute pairwise cosine similarity (sample for large pools)
            n = len(pool_emb)
            if n > 1000:
                sample_idx = np.random.choice(n, 1000, replace=False)
                sample_emb = pool_emb[sample_idx]
            else:
                sample_emb = pool_emb

            # Normalize for cosine similarity
            sample_norm = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-8)
            cos_sim = sample_norm @ sample_norm.T
            np.fill_diagonal(cos_sim, 0)  # Exclude self-similarity

            results['pools'][pool_name] = {
                'display': EMBEDDING_POOLS_V18.get(pool_name, (pool_name,))[0],
                'n_neurons': int(end - start),
                'emb_dim': pool_emb.shape[1],
                'norm_mean': float(norms.mean()),
                'norm_std': float(norms.std()),
                'norm_min': float(norms.min()),
                'norm_max': float(norms.max()),
                'mean_cosine_sim': float(cos_sim.mean()),
                'std_cosine_sim': float(cos_sim.std()),
                'max_cosine_sim': float(cos_sim.max()),
            }

        # Cross-pool analysis
        if HAS_SKLEARN and len(self.pool_boundaries) > 1:
            pool_centroids = {}
            for pool_name, (start, end) in self.pool_boundaries.items():
                pool_emb = self.neuron_emb[start:end].cpu().numpy()
                pool_centroids[pool_name] = pool_emb.mean(axis=0)

            # Pairwise pool distance
            pool_distances = {}
            pool_names = list(pool_centroids.keys())
            for i, p1 in enumerate(pool_names):
                for p2 in pool_names[i+1:]:
                    c1 = pool_centroids[p1]
                    c2 = pool_centroids[p2]
                    # Euclidean distance
                    dist = float(np.linalg.norm(c1 - c2))
                    # Cosine similarity
                    cos = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8))
                    pool_distances[f'{p1}-{p2}'] = {
                        'euclidean': dist,
                        'cosine_similarity': cos,
                    }

            results['cross_pool'] = pool_distances

        return results

    def analyze_cluster_activations(
        self,
        dataloader,
        cluster_labels: List[int],
        n_batches: int = 50,
        top_k: int = 10,
        pool_name: Optional[str] = None
    ) -> Dict:
        """
        Analyze which tokens activate neurons in each cluster.

        Args:
            dataloader: DataLoader for input data
            cluster_labels: Cluster labels from clustering analysis
            n_batches: Number of batches to process
            top_k: Number of top activating tokens per cluster
            pool_name: Pool to analyze (required if labels are pool-specific)

        Returns:
            Dictionary with cluster activation patterns
        """
        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        # Determine neuron offset if pool-specific
        if pool_name:
            start, end = self.pool_boundaries.get(pool_name, (0, 0))
            neuron_offset = start
            n_pool_neurons = end - start
        else:
            neuron_offset = 0
            n_pool_neurons = len(cluster_labels)

        n_clusters = max(cluster_labels) + 1
        cluster_labels_np = np.array(cluster_labels)

        # Pre-compute cluster masks for vectorization
        cluster_labels_t = torch.tensor(cluster_labels_np, device=self.device)
        cluster_masks = [(cluster_labels_t == c) for c in range(n_clusters)]

        # Track activations per cluster: cluster_id -> {token_id: weight}
        cluster_activations = [defaultdict(float) for _ in range(n_clusters)]
        cluster_total_weight = [0.0 for _ in range(n_clusters)]

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Cluster Activations')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                # Aggregate weights across layers
                for layer in routing:
                    # Get weights for the target pool
                    if pool_name:
                        pool_to_keys = {
                            'feature_qk': ['fqk_q', 'fqk_k'],
                            'feature_v': ['fv'],
                            'restore_qk': ['rqk_q', 'rqk_k'],
                            'restore_v': ['rv'],
                            'feature_know': ['fknow'],
                            'restore_know': ['rknow'],
                        }
                        keys = pool_to_keys.get(pool_name, [])
                    else:
                        keys = ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']

                    for key in keys:
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        # weights: [B, S, N] or [B, N]
                        if weights.dim() == 2:
                            weights = weights.unsqueeze(1)  # [B, 1, N]

                        B, S, N = weights.shape

                        # Slice to pool-specific neurons if needed
                        if pool_name and neuron_offset > 0:
                            if N > n_pool_neurons:
                                weights = weights[:, :, neuron_offset:neuron_offset + n_pool_neurons]
                        elif N != n_pool_neurons:
                            continue  # Shape mismatch

                        # Vectorized: compute per-cluster weights using masks
                        input_ids_flat = input_ids.view(-1).cpu().numpy().astype(np.int64)  # [B*S]

                        for c in range(n_clusters):
                            mask = cluster_masks[c]
                            if mask.shape[0] != weights.shape[-1]:
                                continue

                            # Sum weights for neurons in this cluster: [B, S]
                            cluster_w = (weights * mask.float()).sum(dim=-1)  # [B, S]
                            cluster_w_flat = cluster_w.view(-1).cpu().numpy()  # [B*S]

                            # Use np.bincount for fast token-wise accumulation
                            nonzero_mask = cluster_w_flat > 0
                            if not nonzero_mask.any():
                                continue

                            token_ids_active = input_ids_flat[nonzero_mask]
                            weights_active = cluster_w_flat[nonzero_mask]

                            # bincount sums weights per token_id in one shot
                            max_token = int(token_ids_active.max()) + 1
                            token_weights = np.bincount(
                                token_ids_active, weights=weights_active, minlength=max_token
                            )

                            # Merge into cluster_activations[c]
                            active_tokens = np.where(token_weights > 0)[0]
                            for token_id in active_tokens:
                                cluster_activations[c][int(token_id)] += token_weights[token_id]
                            cluster_total_weight[c] += float(weights_active.sum())

        # Get tokenizer for decoding
        tokenizer = getattr(self, 'tokenizer', None)

        # Build results
        results = {'clusters': {}}
        for c in range(n_clusters):
            activations = cluster_activations[c]
            total = cluster_total_weight[c]

            if not activations:
                results['clusters'][c] = {
                    'total_weight': 0,
                    'n_unique_tokens': 0,
                    'top_tokens': [],
                }
                continue

            # Sort by weight
            sorted_tokens = sorted(activations.items(), key=lambda x: -x[1])[:top_k]

            top_tokens = []
            for token_id, weight in sorted_tokens:
                token_info = {
                    'token_id': token_id,
                    'weight': weight,
                    'weight_ratio': weight / total if total > 0 else 0,
                }
                if tokenizer:
                    try:
                        token_info['token'] = tokenizer.decode([token_id])
                    except:
                        pass
                top_tokens.append(token_info)

            results['clusters'][c] = {
                'total_weight': total,
                'n_unique_tokens': len(activations),
                'top_tokens': top_tokens,
            }

        results['n_batches'] = n_batches
        return results

    def visualize_embeddings(
        self,
        output_path: str,
        method: str = 'tsne',
        cluster_labels: Optional[List[int]] = None,
        perplexity: int = 30,
        n_neighbors: int = 15
    ) -> Optional[str]:
        """
        Visualize neuron embeddings using dimensionality reduction.

        Args:
            output_path: Path to save visualization
            method: 'tsne' or 'umap'
            cluster_labels: Optional cluster labels for coloring
            perplexity: t-SNE perplexity parameter
            n_neighbors: UMAP n_neighbors parameter

        Returns:
            Path to saved visualization
        """
        if not HAS_MATPLOTLIB:
            return None

        if self.neuron_emb is None:
            return None

        emb_np = self.neuron_emb.cpu().numpy()

        # Dimensionality reduction
        if method == 'tsne' and HAS_SKLEARN:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            emb_2d = reducer.fit_transform(emb_np)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            emb_2d = reducer.fit_transform(emb_np)
        elif HAS_SKLEARN:
            # Fallback to PCA
            reducer = PCA(n_components=2)
            emb_2d = reducer.fit_transform(emb_np)
            method = 'pca'
        else:
            return None

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Color by pool
        ax = axes[0]
        colors = []
        pool_colors = {
            'feature_qk': 'red',
            'feature_v': 'orange',
            'restore_qk': 'blue',
            'restore_v': 'green',
            'feature_know': 'purple',
            'restore_know': 'cyan',
        }

        for i in range(len(emb_np)):
            color = 'gray'
            for pool_name, (start, end) in self.pool_boundaries.items():
                if start <= i < end:
                    color = pool_colors.get(pool_name, 'gray')
                    break
            colors.append(color)

        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.5, s=10)
        ax.set_title(f'Neuron Embeddings by Pool ({method.upper()})')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=EMBEDDING_POOLS_V18.get(p, (p,))[0])
            for p, c in pool_colors.items()
            if p in self.pool_boundaries
        ]
        ax.legend(handles=legend_elements, loc='best')

        # Right: Color by cluster (if provided)
        ax = axes[1]
        if cluster_labels is not None:
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels,
                               cmap='tab20', alpha=0.5, s=10)
            ax.set_title(f'Neuron Embeddings by Cluster ({method.upper()})')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            # Color by embedding norm
            norms = np.linalg.norm(emb_np, axis=1)
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=norms,
                               cmap='viridis', alpha=0.5, s=10)
            ax.set_title(f'Neuron Embeddings by Norm ({method.upper()})')
            plt.colorbar(scatter, ax=ax, label='L2 Norm')

        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def visualize_pool_separation(self, output_path: str) -> Optional[str]:
        """
        Visualize pool separation in embedding space.

        Args:
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None

        if self.neuron_emb is None:
            return None

        # PCA for 2D projection
        emb_np = self.neuron_emb.cpu().numpy()
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb_np)

        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        pool_colors = {
            'feature_qk': 'red',
            'feature_v': 'orange',
            'restore_qk': 'blue',
            'restore_v': 'green',
            'feature_know': 'purple',
            'restore_know': 'cyan',
        }

        # Plot each pool separately
        for idx, (pool_name, (start, end)) in enumerate(self.pool_boundaries.items()):
            if idx >= 6:
                break

            ax = axes[idx]
            pool_emb = emb_2d[start:end]
            color = pool_colors.get(pool_name, 'gray')

            # Plot other pools in gray
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='lightgray', alpha=0.2, s=5)
            # Highlight this pool
            ax.scatter(pool_emb[:, 0], pool_emb[:, 1], c=color, alpha=0.7, s=15)

            display_name = EMBEDDING_POOLS_V18.get(pool_name, (pool_name,))[0]
            ax.set_title(f'{display_name} (n={end-start})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')

        plt.suptitle('Pool Separation in Embedding Space (PCA)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def visualize_clustering_metrics(
        self,
        clustering_results: Dict,
        output_path: str
    ) -> Optional[str]:
        """
        Visualize clustering metric curves (elbow, silhouette).

        Args:
            clustering_results: Results from analyze_clustering
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        k_range = clustering_results['k_range']

        # Silhouette score
        ax = axes[0]
        ax.plot(k_range, clustering_results['silhouette_scores'], 'b-o')
        optimal_k_sil = clustering_results['optimal_k_silhouette']
        ax.axvline(x=optimal_k_sil, color='r', linestyle='--', label=f'Optimal k={optimal_k_sil}')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score vs k')
        ax.legend()

        # Calinski-Harabasz score
        ax = axes[1]
        ax.plot(k_range, clustering_results['calinski_scores'], 'g-o')
        optimal_k_cal = clustering_results['optimal_k_calinski']
        ax.axvline(x=optimal_k_cal, color='r', linestyle='--', label=f'Optimal k={optimal_k_cal}')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Calinski-Harabasz Score')
        ax.set_title('Calinski-Harabasz Score vs k')
        ax.legend()

        # Inertia (elbow method)
        ax = axes[2]
        if clustering_results.get('inertias'):
            ax.plot(k_range, clustering_results['inertias'], 'm-o')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method (Inertia vs k)')
        else:
            ax.text(0.5, 0.5, 'Inertia not available\n(hierarchical clustering)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Elbow Method')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def run_all(
        self,
        dataloader=None,
        output_dir: str = './neuron_embedding_analysis',
        n_batches: int = 50,
        k_range: Tuple[int, int] = (5, 30)
    ) -> Dict:
        """
        Run complete neuron embedding analysis.

        Args:
            dataloader: DataLoader for activation analysis (optional)
            output_dir: Directory for outputs
            n_batches: Number of batches for activation analysis
            k_range: Range of k values for clustering

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 1. Pool distribution analysis
        print("Analyzing pool distribution...")
        results['pool_distribution'] = self.analyze_pool_distribution()

        # 2. Clustering analysis (all neurons)
        print("Performing clustering analysis...")
        clustering = self.analyze_clustering(k_range=k_range)
        results['clustering'] = clustering

        # 3. Visualizations
        if HAS_MATPLOTLIB and HAS_SKLEARN:
            print("Creating visualizations...")

            # Main embedding visualization
            cluster_labels = clustering.get('cluster_labels')
            viz_path = self.visualize_embeddings(
                os.path.join(output_dir, 'embedding_visualization.png'),
                method='tsne',
                cluster_labels=cluster_labels
            )
            if viz_path:
                results['embedding_visualization'] = viz_path

            # Pool separation visualization
            sep_path = self.visualize_pool_separation(
                os.path.join(output_dir, 'pool_separation.png')
            )
            if sep_path:
                results['pool_separation_visualization'] = sep_path

            # Clustering metrics visualization
            metrics_path = self.visualize_clustering_metrics(
                clustering,
                os.path.join(output_dir, 'clustering_metrics.png')
            )
            if metrics_path:
                results['clustering_metrics_visualization'] = metrics_path

        # 4. Cluster activation analysis (if dataloader provided)
        if dataloader is not None and clustering.get('cluster_labels'):
            print("Analyzing cluster activations...")
            results['cluster_activations'] = self.analyze_cluster_activations(
                dataloader,
                clustering['cluster_labels'],
                n_batches=n_batches
            )

        # 5. Pool alignment analysis (k=6 fixed)
        print("Analyzing pool-cluster alignment (k=6)...")
        pool_alignment = self.analyze_pool_alignment()
        results['pool_alignment'] = pool_alignment

        # 6. Per-pool clustering (internal structure within each pool)
        print("Analyzing per-pool clustering...")
        results['per_pool_clustering'] = {}
        for pool_name in self.pool_boundaries.keys():
            pool_clustering = self.analyze_clustering(
                k_range=(3, min(15, k_range[1])),
                pool_name=pool_name
            )
            # Store detailed results including cluster stats
            results['per_pool_clustering'][pool_name] = {
                'n_neurons': pool_clustering.get('n_embeddings'),
                'optimal_k': pool_clustering.get('optimal_k'),
                'optimal_k_silhouette': pool_clustering.get('optimal_k_silhouette'),
                'optimal_k_calinski': pool_clustering.get('optimal_k_calinski'),
                'best_silhouette': pool_clustering.get('best_silhouette'),
                'cluster_stats': pool_clustering.get('cluster_stats'),
            }

        return results

    # ================================================================
    # Part 2: Token Projection Analysis
    # ================================================================

    def collect_token_projections(
        self,
        dataloader,
        n_batches: int = 100,
        pos_tags: Optional[Dict[int, str]] = None
    ) -> Dict:
        """
        Collect h_proj (hidden state projections) for tokens.

        h_proj is the hidden state projected to d_space for routing.

        Args:
            dataloader: DataLoader with input data
            n_batches: Number of batches to process
            pos_tags: Optional dict mapping token_id -> POS tag

        Returns:
            Dictionary with collected projections
        """
        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        d_space = self.neuron_emb.shape[1]

        # Collect projections per token
        token_projections = defaultdict(list)  # token_id -> list of h_proj vectors
        token_counts = defaultdict(int)

        # Get the projection layer
        proj_layer = None
        if hasattr(self.router, 'h_proj'):
            proj_layer = self.router.h_proj
        elif hasattr(self.router, 'hidden_proj'):
            proj_layer = self.router.hidden_proj

        if proj_layer is None:
            return {'error': 'Could not find projection layer (h_proj)'}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Collecting h_proj')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                B, S = input_ids.shape

                # Get hidden states from model
                outputs = self.model(input_ids, output_hidden_states=True)

                # Extract hidden states (use middle layer for balanced representation)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states
                    mid_layer = len(hidden_states) // 2
                    h = hidden_states[mid_layer]  # [B, S, d_model]
                elif isinstance(outputs, tuple) and len(outputs) > 2:
                    # Try to get from tuple
                    h = outputs[0]  # logits, assume need to use encoder output
                    # This may need adjustment based on model structure
                    continue
                else:
                    continue

                # Project to d_space
                h_proj = proj_layer(h)  # [B, S, d_space]

                # Collect per-token projections (sample to limit memory)
                for b in range(B):
                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        if token_counts[token_id] < 100:  # Limit samples per token
                            token_projections[token_id].append(h_proj[b, s].cpu())
                            token_counts[token_id] += 1

        # Compute mean projection per token
        token_mean_proj = {}
        for token_id, projs in token_projections.items():
            if projs:
                stacked = torch.stack(projs)
                token_mean_proj[token_id] = stacked.mean(dim=0).numpy()

        return {
            'token_mean_proj': token_mean_proj,
            'token_counts': dict(token_counts),
            'd_space': d_space,
            'n_tokens': len(token_mean_proj),
        }

    def analyze_pos_projection_clusters(
        self,
        token_projections: Dict,
        pos_tagger=None,
        tokenizer=None
    ) -> Dict:
        """
        Analyze if token projections cluster by POS tag.

        Args:
            token_projections: Results from collect_token_projections
            pos_tagger: Function to get POS tag for a token string
            tokenizer: Tokenizer for decoding token IDs

        Returns:
            Dictionary with POS clustering analysis
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        token_mean_proj = token_projections.get('token_mean_proj', {})
        if not token_mean_proj:
            return {'error': 'No token projections available'}

        # Use simple POS tagger if not provided
        if pos_tagger is None:
            from .utils import simple_pos_tag
            pos_tagger = simple_pos_tag

        if tokenizer is None:
            tokenizer = self.tokenizer

        # Collect projections by POS
        pos_projections = defaultdict(list)
        pos_token_ids = defaultdict(list)

        for token_id, proj in token_mean_proj.items():
            if tokenizer:
                try:
                    token_str = tokenizer.decode([token_id]).strip()
                    pos = pos_tagger(token_str)
                except:
                    pos = 'UNKNOWN'
            else:
                pos = 'UNKNOWN'

            pos_projections[pos].append(proj)
            pos_token_ids[pos].append(token_id)

        # Compute POS centroids
        pos_centroids = {}
        pos_stats = {}
        for pos, projs in pos_projections.items():
            if len(projs) >= 5:  # Minimum samples
                projs_np = np.array(projs)
                centroid = projs_np.mean(axis=0)
                pos_centroids[pos] = centroid

                # Compute within-POS variance
                dists = np.linalg.norm(projs_np - centroid, axis=1)
                pos_stats[pos] = {
                    'n_tokens': len(projs),
                    'mean_dist_to_centroid': float(dists.mean()),
                    'std_dist_to_centroid': float(dists.std()),
                }

        # Compute between-POS distances
        pos_distances = {}
        pos_names = list(pos_centroids.keys())
        for i, p1 in enumerate(pos_names):
            for p2 in pos_names[i+1:]:
                c1 = pos_centroids[p1]
                c2 = pos_centroids[p2]
                dist = float(np.linalg.norm(c1 - c2))
                cos = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8))
                pos_distances[f'{p1}-{p2}'] = {
                    'euclidean': dist,
                    'cosine_similarity': cos,
                }

        # Compute silhouette score for POS clustering
        all_projs = []
        all_labels = []
        label_map = {pos: i for i, pos in enumerate(pos_names)}

        for pos, projs in pos_projections.items():
            if pos in label_map:
                for proj in projs:
                    all_projs.append(proj)
                    all_labels.append(label_map[pos])

        if len(set(all_labels)) > 1 and len(all_projs) > 10:
            all_projs_np = np.array(all_projs)
            sil_score = silhouette_score(all_projs_np, all_labels)
        else:
            sil_score = None

        return {
            'pos_stats': pos_stats,
            'pos_distances': pos_distances,
            'pos_centroids': {p: c.tolist() for p, c in pos_centroids.items()},
            'silhouette_score': sil_score,
            'n_pos_categories': len(pos_centroids),
        }

    def visualize_pos_projections(
        self,
        token_projections: Dict,
        output_path: str,
        pos_tagger=None,
        tokenizer=None,
        method: str = 'tsne'
    ) -> Optional[str]:
        """
        Visualize token projections colored by POS tag.

        Args:
            token_projections: Results from collect_token_projections
            output_path: Path to save visualization
            pos_tagger: Function to get POS tag for a token string
            tokenizer: Tokenizer for decoding token IDs
            method: 'tsne' or 'umap'

        Returns:
            Path to saved visualization
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None

        token_mean_proj = token_projections.get('token_mean_proj', {})
        if not token_mean_proj:
            return None

        if pos_tagger is None:
            from .utils import simple_pos_tag
            pos_tagger = simple_pos_tag

        if tokenizer is None:
            tokenizer = self.tokenizer

        # Collect data
        projs = []
        pos_labels = []
        token_strs = []

        for token_id, proj in token_mean_proj.items():
            if tokenizer:
                try:
                    token_str = tokenizer.decode([token_id]).strip()
                    pos = pos_tagger(token_str)
                except:
                    token_str = f'[{token_id}]'
                    pos = 'UNKNOWN'
            else:
                token_str = f'[{token_id}]'
                pos = 'UNKNOWN'

            projs.append(proj)
            pos_labels.append(pos)
            token_strs.append(token_str)

        projs_np = np.array(projs)

        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(30, len(projs) - 1), random_state=42)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
            method = 'pca'

        projs_2d = reducer.fit_transform(projs_np)

        # Color mapping for POS
        pos_colors = {
            'DET': 'red',
            'VERB': 'blue',
            'PREP': 'green',
            'CONJ': 'orange',
            'PRON': 'purple',
            'PUNCT': 'gray',
            'NUM': 'brown',
            'OTHER': 'lightblue',
            'SPECIAL': 'pink',
            'UNKNOWN': 'lightgray',
        }

        colors = [pos_colors.get(p, 'lightgray') for p in pos_labels]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(projs_2d[:, 0], projs_2d[:, 1], c=colors, alpha=0.6, s=20)

        ax.set_title(f'Token Projections by POS ({method.upper()})')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

        # Legend
        from matplotlib.patches import Patch
        unique_pos = list(set(pos_labels))
        legend_elements = [
            Patch(facecolor=pos_colors.get(p, 'lightgray'), label=p)
            for p in sorted(unique_pos)
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    # ================================================================
    # Part 3: Neuron-Token Mapping
    # ================================================================

    def analyze_neuron_pos_similarity(
        self,
        token_projections: Dict,
        pos_tagger=None,
        tokenizer=None,
        top_k: int = 10
    ) -> Dict:
        """
        Analyze similarity between neuron embeddings and POS-averaged projections.

        Args:
            token_projections: Results from collect_token_projections
            pos_tagger: Function to get POS tag for a token string
            tokenizer: Tokenizer for decoding token IDs
            top_k: Number of top similar neurons per POS

        Returns:
            Dictionary with neuron-POS similarity analysis
        """
        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        token_mean_proj = token_projections.get('token_mean_proj', {})
        if not token_mean_proj:
            return {'error': 'No token projections available'}

        if pos_tagger is None:
            from .utils import simple_pos_tag
            pos_tagger = simple_pos_tag

        if tokenizer is None:
            tokenizer = self.tokenizer

        # Compute POS centroids
        pos_projections = defaultdict(list)
        for token_id, proj in token_mean_proj.items():
            if tokenizer:
                try:
                    token_str = tokenizer.decode([token_id]).strip()
                    pos = pos_tagger(token_str)
                except:
                    pos = 'UNKNOWN'
            else:
                pos = 'UNKNOWN'
            pos_projections[pos].append(proj)

        pos_centroids = {}
        for pos, projs in pos_projections.items():
            if len(projs) >= 5:
                pos_centroids[pos] = np.array(projs).mean(axis=0)

        # Get neuron embeddings
        neuron_emb_np = self.neuron_emb.cpu().numpy()
        neuron_norms = np.linalg.norm(neuron_emb_np, axis=1, keepdims=True)
        neuron_emb_normalized = neuron_emb_np / (neuron_norms + 1e-8)

        # Compute similarity between each POS centroid and all neurons
        results = {'pos_neuron_mapping': {}}

        for pos, centroid in pos_centroids.items():
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Cosine similarity with all neurons
            similarities = neuron_emb_normalized @ centroid_norm

            # Get top-k most similar neurons
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            top_neurons = []
            for idx in top_indices:
                # Determine pool
                pool_name = 'unknown'
                for pname, (start, end) in self.pool_boundaries.items():
                    if start <= idx < end:
                        pool_name = pname
                        break

                top_neurons.append({
                    'neuron_idx': int(idx),
                    'pool': pool_name,
                    'similarity': float(similarities[idx]),
                })

            results['pos_neuron_mapping'][pos] = {
                'n_tokens': len(pos_projections[pos]),
                'top_neurons': top_neurons,
                'mean_similarity': float(similarities.mean()),
                'max_similarity': float(similarities.max()),
            }

        # Compute pool affinity for each POS
        results['pos_pool_affinity'] = {}
        for pos, data in results['pos_neuron_mapping'].items():
            pool_counts = defaultdict(int)
            pool_sim_sum = defaultdict(float)
            for neuron in data['top_neurons']:
                pool = neuron['pool']
                pool_counts[pool] += 1
                pool_sim_sum[pool] += neuron['similarity']

            results['pos_pool_affinity'][pos] = {
                pool: {
                    'count': count,
                    'avg_similarity': pool_sim_sum[pool] / count if count > 0 else 0
                }
                for pool, count in pool_counts.items()
            }

        return results

    def analyze_neuron_token_alignment(
        self,
        token_projections: Dict,
        tokenizer=None,
        top_k_tokens: int = 5,
        top_k_neurons: int = 100
    ) -> Dict:
        """
        Find which specific tokens each neuron is most aligned with.

        Args:
            token_projections: Results from collect_token_projections
            tokenizer: Tokenizer for decoding token IDs
            top_k_tokens: Number of top tokens per neuron
            top_k_neurons: Number of neurons to analyze (sample for speed)

        Returns:
            Dictionary with neuron-token alignment
        """
        if self.neuron_emb is None:
            return {'error': 'No neuron embeddings found'}

        token_mean_proj = token_projections.get('token_mean_proj', {})
        if not token_mean_proj:
            return {'error': 'No token projections available'}

        if tokenizer is None:
            tokenizer = self.tokenizer

        # Build token projection matrix
        token_ids = list(token_mean_proj.keys())
        token_projs = np.array([token_mean_proj[tid] for tid in token_ids])
        token_norms = np.linalg.norm(token_projs, axis=1, keepdims=True)
        token_projs_normalized = token_projs / (token_norms + 1e-8)

        # Get neuron embeddings
        neuron_emb_np = self.neuron_emb.cpu().numpy()
        n_neurons = len(neuron_emb_np)

        # Sample neurons if too many
        if n_neurons > top_k_neurons:
            sampled_indices = np.random.choice(n_neurons, top_k_neurons, replace=False)
        else:
            sampled_indices = np.arange(n_neurons)

        neuron_emb_sampled = neuron_emb_np[sampled_indices]
        neuron_norms = np.linalg.norm(neuron_emb_sampled, axis=1, keepdims=True)
        neuron_emb_normalized = neuron_emb_sampled / (neuron_norms + 1e-8)

        # Compute similarity matrix: [n_sampled_neurons, n_tokens]
        similarity_matrix = neuron_emb_normalized @ token_projs_normalized.T

        # For each neuron, find top tokens
        results = {'neuron_token_alignment': {}}

        for i, neuron_idx in enumerate(sampled_indices):
            sims = similarity_matrix[i]
            top_token_indices = np.argsort(sims)[-top_k_tokens:][::-1]

            # Determine pool
            pool_name = 'unknown'
            for pname, (start, end) in self.pool_boundaries.items():
                if start <= neuron_idx < end:
                    pool_name = pname
                    break

            top_tokens = []
            for tidx in top_token_indices:
                token_id = token_ids[tidx]
                token_info = {
                    'token_id': token_id,
                    'similarity': float(sims[tidx]),
                }
                if tokenizer:
                    try:
                        token_info['token'] = tokenizer.decode([token_id]).strip()
                    except:
                        pass
                top_tokens.append(token_info)

            results['neuron_token_alignment'][int(neuron_idx)] = {
                'pool': pool_name,
                'top_tokens': top_tokens,
                'mean_similarity': float(sims.mean()),
            }

        return results

    def visualize_neuron_pos_heatmap(
        self,
        neuron_pos_results: Dict,
        output_path: str
    ) -> Optional[str]:
        """
        Visualize neuron-POS affinity as a heatmap.

        Args:
            neuron_pos_results: Results from analyze_neuron_pos_similarity
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        if not HAS_MATPLOTLIB:
            return None

        pos_pool_affinity = neuron_pos_results.get('pos_pool_affinity', {})
        if not pos_pool_affinity:
            return None

        # Build matrix
        pos_list = sorted(pos_pool_affinity.keys())
        pool_list = list(self.pool_boundaries.keys())

        matrix = np.zeros((len(pos_list), len(pool_list)))
        for i, pos in enumerate(pos_list):
            for j, pool in enumerate(pool_list):
                if pool in pos_pool_affinity[pos]:
                    matrix[i, j] = pos_pool_affinity[pos][pool]['count']

        # Normalize by row
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_normalized = matrix / (row_sums + 1e-8)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        pool_display = [EMBEDDING_POOLS_V18.get(p, (p,))[0] for p in pool_list]

        im = ax.imshow(matrix_normalized, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pool_list)))
        ax.set_xticklabels(pool_display, rotation=45, ha='right')
        ax.set_yticks(range(len(pos_list)))
        ax.set_yticklabels(pos_list)

        ax.set_xlabel('Neuron Pool')
        ax.set_ylabel('POS Category')
        ax.set_title('POS-Pool Affinity (Top Neurons per POS)')

        plt.colorbar(im, ax=ax, label='Proportion')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def run_full_analysis(
        self,
        dataloader,
        output_dir: str = './neuron_embedding_analysis',
        n_batches: int = 50,
        k_range: Tuple[int, int] = (5, 30),
        tokenizer=None
    ) -> Dict:
        """
        Run complete analysis including token projections and neuron-token mapping.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process
            k_range: Range of k values for clustering
            tokenizer: Tokenizer for decoding

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        if tokenizer:
            self.tokenizer = tokenizer

        results = {}

        # Part 1: Neuron embedding analysis
        print("=" * 50)
        print("Part 1: Neuron Embedding Analysis")
        print("=" * 50)
        results.update(self.run_all(dataloader, output_dir, n_batches, k_range))

        # Part 2: Token projection analysis
        print("\n" + "=" * 50)
        print("Part 2: Token Projection Analysis")
        print("=" * 50)

        print("Collecting token projections...")
        token_projs = self.collect_token_projections(dataloader, n_batches=n_batches)
        results['token_projections'] = {
            'n_tokens': token_projs.get('n_tokens', 0),
            'd_space': token_projs.get('d_space', 0),
        }

        if 'error' not in token_projs:
            print("Analyzing POS projection clusters...")
            pos_analysis = self.analyze_pos_projection_clusters(token_projs, tokenizer=tokenizer)
            results['pos_projection_analysis'] = pos_analysis

            if HAS_MATPLOTLIB:
                print("Visualizing POS projections...")
                viz_path = self.visualize_pos_projections(
                    token_projs,
                    os.path.join(output_dir, 'pos_projections.png'),
                    tokenizer=tokenizer
                )
                if viz_path:
                    results['pos_projections_visualization'] = viz_path

            # Part 3: Neuron-token mapping
            print("\n" + "=" * 50)
            print("Part 3: Neuron-Token Mapping")
            print("=" * 50)

            print("Analyzing neuron-POS similarity...")
            neuron_pos = self.analyze_neuron_pos_similarity(token_projs, tokenizer=tokenizer)
            results['neuron_pos_similarity'] = neuron_pos

            print("Analyzing neuron-token alignment...")
            neuron_token = self.analyze_neuron_token_alignment(token_projs, tokenizer=tokenizer)
            results['neuron_token_alignment'] = neuron_token

            if HAS_MATPLOTLIB:
                print("Visualizing neuron-POS heatmap...")
                heatmap_path = self.visualize_neuron_pos_heatmap(
                    neuron_pos,
                    os.path.join(output_dir, 'neuron_pos_heatmap.png')
                )
                if heatmap_path:
                    results['neuron_pos_heatmap'] = heatmap_path

        print("\n" + "=" * 50)
        print("Analysis Complete")
        print("=" * 50)

        return results

"""
Embedding Analysis (JAX Version)
================================
Analyze neuron embeddings in DAWN models using JAX/Flax.

Includes:
- Neuron embedding similarity analysis
- Embedding clustering
- PCA variance analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .utils_jax import (
    get_neuron_embeddings_jax,
    POOL_N_ATTR, POOL_DISPLAY_NAMES,
)

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class EmbeddingAnalyzerJAX:
    """Neuron embedding analyzer for JAX DAWN models."""

    def __init__(self, model, params, config: Dict):
        """
        Initialize analyzer.

        Args:
            model: JAX DAWN model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        self.model = model
        self.params = params
        self.config = config

        # Get neuron embeddings
        self.embeddings = get_neuron_embeddings_jax(params)

        # Pool boundaries
        self.n_fqk = config.get('n_feature_qk', 0)
        self.n_fv = config.get('n_feature_v', 0)
        self.n_rqk = config.get('n_restore_qk', 0)
        self.n_rv = config.get('n_restore_v', 0)
        self.n_fk = config.get('n_feature_know', 0)
        self.n_rk = config.get('n_restore_know', 0)

    def _get_pool_embeddings(self) -> Dict[str, np.ndarray]:
        """Split embeddings by pool."""
        if self.embeddings is None:
            return {}

        emb = self.embeddings
        pools = {}

        idx = 0
        if self.n_fqk > 0:
            pools['feature_qk'] = emb[idx:idx + self.n_fqk]
            idx += self.n_fqk
        if self.n_fv > 0:
            pools['feature_v'] = emb[idx:idx + self.n_fv]
            idx += self.n_fv
        if self.n_rqk > 0:
            pools['restore_qk'] = emb[idx:idx + self.n_rqk]
            idx += self.n_rqk
        if self.n_rv > 0:
            pools['restore_v'] = emb[idx:idx + self.n_rv]
            idx += self.n_rv
        if self.n_fk > 0:
            pools['feature_know'] = emb[idx:idx + self.n_fk]
            idx += self.n_fk
        if self.n_rk > 0:
            pools['restore_know'] = emb[idx:idx + self.n_rk]

        return pools

    def analyze_similarity(self) -> Dict:
        """Analyze within-pool and cross-pool similarity.

        Returns:
            Dictionary with similarity statistics
        """
        pools = self._get_pool_embeddings()
        if not pools:
            return {}

        results = {
            'within_pool': {},
            'cross_pool': {},
        }

        # Within-pool similarity
        for pool_name, pool_emb in pools.items():
            if len(pool_emb) < 2:
                continue

            # Cosine similarity matrix (embeddings are already normalized)
            sim_matrix = pool_emb @ pool_emb.T

            # Get upper triangle (excluding diagonal)
            n = len(pool_emb)
            upper_tri = sim_matrix[np.triu_indices(n, k=1)]

            results['within_pool'][pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name.replace('_', ''), pool_name),
                'avg_similarity': float(upper_tri.mean()),
                'std_similarity': float(upper_tri.std()),
                'min_similarity': float(upper_tri.min()),
                'max_similarity': float(upper_tri.max()),
                'n_neurons': n,
            }

        # Cross-pool similarity (between each pair of pools)
        pool_names = list(pools.keys())
        for i, pool1 in enumerate(pool_names):
            for pool2 in pool_names[i+1:]:
                emb1 = pools[pool1]
                emb2 = pools[pool2]

                # Cross-similarity matrix
                cross_sim = emb1 @ emb2.T

                key = f"{pool1}_vs_{pool2}"
                results['cross_pool'][key] = {
                    'pool1': pool1,
                    'pool2': pool2,
                    'avg_similarity': float(cross_sim.mean()),
                    'std_similarity': float(cross_sim.std()),
                    'min_similarity': float(cross_sim.min()),
                    'max_similarity': float(cross_sim.max()),
                }

        return results

    def analyze_clustering(self, n_clusters: int = 5) -> Dict:
        """Analyze embedding clustering.

        Args:
            n_clusters: Number of clusters for K-means

        Returns:
            Dictionary with clustering results
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        pools = self._get_pool_embeddings()
        if not pools:
            return {}

        results = {}

        for pool_name, pool_emb in pools.items():
            if len(pool_emb) < n_clusters:
                continue

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pool_emb)

            # Cluster statistics
            cluster_sizes = np.bincount(labels, minlength=n_clusters)
            inertia = kmeans.inertia_

            # Silhouette score (if we have enough samples)
            if len(pool_emb) > n_clusters:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(pool_emb, labels)
            else:
                silhouette = 0.0

            results[pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name.replace('_', ''), pool_name),
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes.tolist(),
                'inertia': float(inertia),
                'silhouette': float(silhouette),
                'n_neurons': len(pool_emb),
            }

        return results

    def analyze_pca(self, n_components: int = 10) -> Dict:
        """Analyze PCA variance in embeddings.

        Args:
            n_components: Number of PCA components

        Returns:
            Dictionary with PCA variance results
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        pools = self._get_pool_embeddings()
        if not pools:
            return {}

        results = {}

        for pool_name, pool_emb in pools.items():
            if len(pool_emb) < n_components:
                continue

            # PCA
            pca = PCA(n_components=min(n_components, len(pool_emb), pool_emb.shape[1]))
            pca.fit(pool_emb)

            results[pool_name] = {
                'display': POOL_DISPLAY_NAMES.get(pool_name.replace('_', ''), pool_name),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_90': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1),
                'n_components_95': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1),
                'n_neurons': len(pool_emb),
            }

        return results

    def visualize_similarity(self, similarity_results: Dict, output_dir: str):
        """Generate similarity visualization."""
        if not HAS_MATPLOTLIB:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        within_pool = similarity_results.get('within_pool', {})
        if not within_pool:
            return

        pools = list(within_pool.keys())
        means = [within_pool[p]['avg_similarity'] for p in pools]
        stds = [within_pool[p]['std_similarity'] for p in pools]
        labels = [within_pool[p]['display'] for p in pools]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pools))
        ax.bar(x, means, yerr=stds, capsize=5, color='coral', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Average Cosine Similarity')
        ax.set_title('Within-Pool Neuron Embedding Similarity')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'embedding_similarity.png', dpi=300)
        plt.savefig(output_path / 'embedding_similarity.pdf')
        plt.close()

    def visualize_pca(self, pca_results: Dict, output_dir: str):
        """Generate PCA variance visualization."""
        if not HAS_MATPLOTLIB:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for pool_name, data in pca_results.items():
            if 'cumulative_variance' not in data:
                continue
            cumvar = data['cumulative_variance']
            ax.plot(range(1, len(cumvar) + 1), cumvar,
                    marker='o', label=data['display'])

        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA Variance by Pool')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'pca_variance.png', dpi=300)
        plt.savefig(output_path / 'pca_variance.pdf')
        plt.close()

    def run_all(self, output_dir: str, n_clusters: int = 5) -> Dict:
        """Run all embedding analyses.

        Args:
            output_dir: Directory to save results
            n_clusters: Number of clusters for K-means

        Returns:
            Combined results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Similarity analysis
        print("  [1/3] Analyzing embedding similarity...")
        results['similarity'] = self.analyze_similarity()

        # Clustering analysis
        print("  [2/3] Analyzing embedding clustering...")
        results['clustering'] = self.analyze_clustering(n_clusters)

        # PCA analysis
        print("  [3/3] Analyzing PCA variance...")
        results['pca'] = self.analyze_pca()

        # Visualizations
        if results.get('similarity'):
            self.visualize_similarity(results['similarity'], str(output_path))
        if results.get('pca'):
            self.visualize_pca(results['pca'], str(output_path))

        # Save results
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results


class NeuronEmbeddingAnalyzerJAX(EmbeddingAnalyzerJAX):
    """Alias for backward compatibility."""
    pass

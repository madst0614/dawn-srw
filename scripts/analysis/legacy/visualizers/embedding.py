"""
Embedding Visualizations
========================
Visualization functions for neuron embedding analysis.

Paper Figure 5: Embedding Structure
- t-SNE/PCA visualization
- Similarity heatmaps
- Clustering visualization
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def plot_similarity_heatmap(
    embeddings: Dict[str, np.ndarray],
    pool_display_names: Dict[str, str],
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate similarity heatmap visualization.

    Args:
        embeddings: Dictionary mapping pool name to embedding array
        pool_display_names: Mapping from pool name to display name
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    n_types = len(embeddings)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
    if n_types == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings.items()):
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim_matrix = emb_norm @ emb_norm.T
        sns.heatmap(sim_matrix, ax=ax, cmap='coolwarm', vmin=-1, vmax=1)
        display = pool_display_names.get(name, name)
        ax.set_title(f'{display} Similarity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_clustering(
    emb: np.ndarray,
    boundaries: Dict[str, Tuple[int, int, str]],
    cluster_results: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate clustering visualization using PCA.

    Args:
        emb: Full embedding array
        boundaries: Dict mapping pool name to (start, end, display_name)
        cluster_results: Clustering analysis results with labels
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if dependencies unavailable
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        return None

    n_types = len([k for k in cluster_results if isinstance(cluster_results.get(k), dict) and 'labels' in cluster_results[k]])
    if n_types == 0:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    ax_idx = 0
    for name, (start, end, display) in boundaries.items():
        if name not in cluster_results or 'labels' not in cluster_results[name]:
            continue

        pool_emb = emb[start:end]
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(pool_emb)

        labels = cluster_results[name]['labels']
        axes[ax_idx].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        axes[ax_idx].set_title(f'{display} Clusters')
        ax_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_embedding_space(
    emb: np.ndarray,
    labels: List[str],
    colors_map: Dict[str, str],
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate t-SNE/PCA visualization of all embeddings.

    Args:
        emb: Full embedding array (n_neurons x d_space)
        labels: List of type labels for each neuron
        colors_map: Mapping from label to color
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if dependencies unavailable
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
    emb_tsne = tsne.fit_transform(emb)

    # PCA
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [(axes[0], emb_tsne, 't-SNE'), (axes[1], emb_pca, 'PCA')]:
        for t in set(labels):
            mask = np.array([l == t for l in labels])
            ax.scatter(data[mask, 0], data[mask, 1],
                      c=colors_map.get(t, 'gray'), label=t, alpha=0.6, s=20)
        ax.set_title(f'DAWN Neuron Embeddings ({title})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path

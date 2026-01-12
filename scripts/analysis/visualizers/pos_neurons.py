"""
POS Neuron Visualizations
=========================
Visualization functions for POS-based neuron analysis.

Paper Figure 4: POS Neuron Specialization
- Specificity bar chart
- POS-specific neuron counts
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict

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
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Universal POS tags (UPOS)
UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X',
]


def plot_pos_heatmap(
    results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot POS x Neuron activation heatmap.

    Args:
        results: Results from POSNeuronAnalyzer.get_results()
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return None

    pos_neuron_freq = results.get('pos_neuron_freq', {})
    if not pos_neuron_freq:
        print("  Warning: No POS neuron frequency data, skipping heatmap")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Get all neurons and sort by total activation
    all_neurons = set()
    for freq in pos_neuron_freq.values():
        all_neurons.update(int(n) for n in freq.keys())

    if not all_neurons:
        print("  Warning: No neuron data found, skipping heatmap")
        return None

    # Filter to top neurons
    neuron_total = defaultdict(float)
    for freq in pos_neuron_freq.values():
        for n, f in freq.items():
            neuron_total[int(n)] += f

    top_neurons = sorted(neuron_total.keys(), key=lambda n: -neuron_total[n])[:100]

    if not top_neurons:
        print("  Warning: No top neurons found, skipping heatmap")
        return None

    # Build matrix
    pos_list = [p for p in UPOS_TAGS if p in pos_neuron_freq]

    if not pos_list:
        print("  Warning: No POS tags found, skipping heatmap")
        return None

    matrix = np.zeros((len(pos_list), len(top_neurons)))

    for i, pos in enumerate(pos_list):
        for j, neuron in enumerate(top_neurons):
            matrix[i, j] = pos_neuron_freq[pos].get(str(neuron), 0)

    # Check if matrix has any data
    if matrix.size == 0 or np.all(matrix == 0):
        print("  Warning: Empty matrix, skipping heatmap")
        return None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=[str(n) for n in top_neurons],
        yticklabels=pos_list,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Activation Frequency'}
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('POS Tag')
    ax.set_title('Neuron Activation by Part-of-Speech')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path


def plot_pos_clustering(
    results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot POS clustering based on neuron activation similarity.

    Args:
        results: Results from POSNeuronAnalyzer.get_results()
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN or not HAS_SCIPY:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    pos_neuron_freq = results['pos_neuron_freq']

    # Get all neurons
    all_neurons = set()
    for freq in pos_neuron_freq.values():
        all_neurons.update(int(n) for n in freq.keys())
    all_neurons = sorted(all_neurons)

    # Build feature vectors
    pos_list = [p for p in UPOS_TAGS if p in pos_neuron_freq]
    vectors = np.zeros((len(pos_list), len(all_neurons)))

    for i, pos in enumerate(pos_list):
        for j, neuron in enumerate(all_neurons):
            vectors[i, j] = pos_neuron_freq[pos].get(str(neuron), 0)

    if len(pos_list) < 2:
        return None

    # Filter out zero vectors (would cause NaN in cosine distance)
    nonzero_mask = np.any(vectors != 0, axis=1)
    if nonzero_mask.sum() < 2:
        print("  Warning: Not enough non-zero POS vectors for clustering")
        return None

    vectors = vectors[nonzero_mask]
    pos_list = [p for p, m in zip(pos_list, nonzero_mask) if m]

    distances = pdist(vectors, metric='cosine')

    # Replace NaN/Inf with max distance (1.0 for cosine)
    if not np.all(np.isfinite(distances)):
        print("  Warning: Replacing NaN/Inf in distance matrix")
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)

    linkage = hierarchy.linkage(distances, method='ward')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Dendrogram
    hierarchy.dendrogram(
        linkage,
        labels=pos_list,
        orientation='left',
        ax=axes[0]
    )
    axes[0].set_title('POS Clustering by Neuron Patterns')
    axes[0].set_xlabel('Distance')

    # Similarity heatmap
    similarity = 1 - pdist(vectors, metric='cosine')
    sim_matrix = np.zeros((len(pos_list), len(pos_list)))
    idx = 0
    for i in range(len(pos_list)):
        for j in range(i + 1, len(pos_list)):
            sim_matrix[i, j] = similarity[idx]
            sim_matrix[j, i] = similarity[idx]
            idx += 1
        sim_matrix[i, i] = 1.0

    sns.heatmap(
        sim_matrix,
        xticklabels=pos_list,
        yticklabels=pos_list,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        ax=axes[1]
    )
    axes[1].set_title('POS Similarity (Cosine)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path


def plot_top_neurons_by_pos(
    results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot top neurons for each POS.

    Args:
        results: Results from POSNeuronAnalyzer.get_results()
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    top_neurons = results.get('top_neurons_per_pos', {})
    pos_list = [p for p in UPOS_TAGS if p in top_neurons]

    if not pos_list:
        print("  Warning: No POS data to plot")
        return None

    n_cols = 4
    n_rows = max(1, (len(pos_list) + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, pos in enumerate(pos_list):
        neurons = top_neurons[pos][:10]
        if neurons:
            neuron_ids = [n[0] for n in neurons]
            freqs = [n[1] for n in neurons]

            axes[i].barh(range(len(neurons)), freqs, color='steelblue')
            axes[i].set_yticks(range(len(neurons)))
            axes[i].set_yticklabels([f'N{n}' for n in neuron_ids])
            axes[i].set_title(pos)
            axes[i].set_xlabel('Freq')
            axes[i].invert_yaxis()

    for i in range(len(pos_list), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Top 10 Neurons per POS Tag', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path


def plot_pos_specificity(
    results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot neuron specificity - neurons specialized for specific POS.

    Paper Figure 4: POS Neuron Specialization
    - Left: Top neurons by specificity score (horizontal bar)
    - Right: Specialized neurons per POS category (horizontal bar)

    Args:
        results: Results from POSNeuronAnalyzer.get_results()
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    specificity = results.get('neuron_specificity', {})
    if not specificity:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Top specific neurons
    items = list(specificity.items())[:20]
    neurons = [int(n) for n, _ in items]
    scores = [s['specificity'] for _, s in items]
    pos_tags = [s['top_pos'] for _, s in items]

    colors = plt.cm.tab20(np.linspace(0, 1, len(set(pos_tags))))
    pos_color_map = {pos: colors[i] for i, pos in enumerate(set(pos_tags))}

    axes[0].barh(
        range(len(neurons)),
        scores,
        color=[pos_color_map[p] for p in pos_tags]
    )
    axes[0].set_yticks(range(len(neurons)))
    axes[0].set_yticklabels([f'N{n} ({p})' for n, p in zip(neurons, pos_tags)])
    axes[0].set_xlabel('Specificity Score')
    axes[0].set_title('Most POS-Specific Neurons')
    axes[0].invert_yaxis()

    # POS-specific neuron counts
    pos_specific_counts = defaultdict(int)
    for _, s in specificity.items():
        pos_specific_counts[s['top_pos']] += 1

    pos_sorted = sorted(pos_specific_counts.items(), key=lambda x: -x[1])
    axes[1].barh(
        range(len(pos_sorted)),
        [c for _, c in pos_sorted],
        color='coral'
    )
    axes[1].set_yticks(range(len(pos_sorted)))
    axes[1].set_yticklabels([p for p, _ in pos_sorted])
    axes[1].set_xlabel('Number of Specific Neurons')
    axes[1].set_title('Specialized Neurons per POS')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path

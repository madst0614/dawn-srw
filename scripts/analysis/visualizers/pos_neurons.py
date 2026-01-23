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
            axes[i].set_yticklabels([str(n) for n in neuron_ids])
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
    axes[0].set_yticklabels([f'{n} ({p})' for n, p in zip(neurons, pos_tags)])
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


def plot_pos_specialization_from_features(
    neuron_features: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot POS neuron specialization from NeuronFeatureAnalyzer results.

    Paper Figure 4: POS Neuron Specialization
    - Left: Top specialized neurons by POS concentration %
    - Right: Number of specialized neurons per POS category

    Args:
        neuron_features: Results from NeuronFeatureAnalyzer.run_full_analysis()
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    # Extract specialized neurons for POS
    specialized = neuron_features.get('specialized_neurons', {})
    pos_neurons = specialized.get('pos', [])

    if not pos_neurons:
        print("  Warning: No POS-specialized neurons found")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by concentration percentage (descending)
    pos_neurons_sorted = sorted(pos_neurons, key=lambda x: -x.get('pct', 0))[:20]

    # Left: Top specialized neurons
    neurons = [n['neuron'] for n in pos_neurons_sorted]
    pcts = [n['pct'] for n in pos_neurons_sorted]
    pos_tags = [n['specialized_for'] for n in pos_neurons_sorted]

    # Color by POS
    unique_pos = list(set(pos_tags))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pos)))
    pos_color_map = {pos: colors[i] for i, pos in enumerate(unique_pos)}

    axes[0].barh(
        range(len(neurons)),
        pcts,
        color=[pos_color_map[p] for p in pos_tags]
    )
    axes[0].set_yticks(range(len(neurons)))
    axes[0].set_yticklabels([f'{n} ({p})' for n, p in zip(neurons, pos_tags)])
    axes[0].set_xlabel('POS Concentration (%)')
    axes[0].set_title('Top POS-Specialized Neurons (≥80%)')
    axes[0].invert_yaxis()
    axes[0].axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')

    # Right: Count per POS category
    pos_counts = defaultdict(int)
    for n in pos_neurons:
        pos_counts[n['specialized_for']] += 1

    pos_sorted = sorted(pos_counts.items(), key=lambda x: -x[1])
    axes[1].barh(
        range(len(pos_sorted)),
        [c for _, c in pos_sorted],
        color='coral'
    )
    axes[1].set_yticks(range(len(pos_sorted)))
    axes[1].set_yticklabels([p for p, _ in pos_sorted])
    axes[1].set_xlabel('Number of Specialized Neurons')
    axes[1].set_title('POS-Specialized Neurons by Category')
    axes[1].invert_yaxis()

    # Add summary text
    total_specialized = len(pos_neurons)
    n_profiled = neuron_features.get('n_neurons_profiled', 0)
    if n_profiled > 0:
        pct_specialized = total_specialized / n_profiled * 100
        fig.text(0.5, 0.02, f'Total: {total_specialized} specialized / {n_profiled} profiled ({pct_specialized:.1f}%)',
                 ha='center', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path


def plot_pos_selectivity_heatmap(
    selectivity_matrix: np.ndarray,
    active_indices: list,
    output_path: str,
    top_n: int = 50,
    figsize: Tuple[int, int] = (14, 12),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot POS selectivity heatmap for Fig 4.

    Selectivity score: P(neuron active | POS) / P(neuron active)
    - > 1 (red): neuron prefers this POS
    - = 1 (white): neuron is indifferent
    - < 1 (blue): neuron avoids this POS

    Args:
        selectivity_matrix: [n_neurons, n_pos] selectivity scores
        active_indices: List of active neuron indices
        output_path: Path to save the figure
        top_n: Number of top neurons to show
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        print("  Warning: matplotlib/seaborn not available, skipping selectivity heatmap")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Get top neurons by selectivity range (most variable = most interesting)
    if len(active_indices) == 0:
        print("  Warning: No active neurons for selectivity heatmap")
        return None

    # Select subset of neurons
    active_selectivity = selectivity_matrix[active_indices]
    selectivity_range = active_selectivity.max(axis=1) - active_selectivity.min(axis=1)
    top_idx = np.argsort(-selectivity_range)[:top_n]
    selected_indices = [active_indices[i] for i in top_idx]
    selected_selectivity = selectivity_matrix[selected_indices]

    # Log2 scale for visualization (centered at 0 = selectivity 1)
    log_selectivity = np.log2(np.clip(selected_selectivity, 0.01, 100))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[4, 1])

    # Main heatmap
    im = axes[0].imshow(
        log_selectivity,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-2, vmax=2,
        interpolation='nearest'
    )

    # Labels
    axes[0].set_xticks(range(len(UPOS_TAGS)))
    axes[0].set_xticklabels(UPOS_TAGS, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticks(range(0, len(selected_indices), max(1, len(selected_indices) // 20)))
    axes[0].set_yticklabels([str(selected_indices[i]) for i in range(0, len(selected_indices), max(1, len(selected_indices) // 20))], fontsize=8)
    axes[0].set_xlabel('POS Category', fontsize=11)
    axes[0].set_ylabel('Neuron (sorted by selectivity range)', fontsize=11)
    axes[0].set_title(f'Neuron POS Selectivity (Top {len(selected_indices)} neurons)', fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label('log₂(selectivity)\n>0: prefer, <0: avoid', fontsize=10)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(['0.25x', '0.5x', '1x', '2x', '4x'])

    # Mean selectivity per POS (bar chart)
    mean_sel = selectivity_matrix[active_indices].mean(axis=0)
    colors = ['coral' if s > 1.1 else 'steelblue' if s < 0.9 else 'gray' for s in mean_sel]
    axes[1].barh(range(len(UPOS_TAGS)), mean_sel, color=colors)
    axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_yticks(range(len(UPOS_TAGS)))
    axes[1].set_yticklabels(UPOS_TAGS, fontsize=9)
    axes[1].set_xlabel('Mean Selectivity', fontsize=10)
    axes[1].set_title('Population Mean', fontsize=11)
    axes[1].set_xlim(0, max(2.0, mean_sel.max() * 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_pos_selectivity_clustered(
    selectivity_matrix: np.ndarray,
    active_indices: list,
    output_path: str,
    top_n: int = 100,
    figsize: Tuple[int, int] = (16, 14),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot clustered POS selectivity heatmap with dendrogram.

    Uses hierarchical clustering to group neurons with similar POS preferences.

    Args:
        selectivity_matrix: [n_neurons, n_pos] selectivity scores
        active_indices: List of active neuron indices
        output_path: Path to save the figure
        top_n: Number of top neurons to cluster
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN or not HAS_SCIPY:
        print("  Warning: matplotlib/seaborn/scipy not available, skipping clustered heatmap")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if len(active_indices) == 0:
        print("  Warning: No active neurons for clustered heatmap")
        return None

    # Select top neurons by selectivity range
    active_selectivity = selectivity_matrix[active_indices]
    selectivity_range = active_selectivity.max(axis=1) - active_selectivity.min(axis=1)
    top_idx = np.argsort(-selectivity_range)[:top_n]
    selected_indices = [active_indices[i] for i in top_idx]
    selected_selectivity = selectivity_matrix[selected_indices]

    # Log2 scale
    log_selectivity = np.log2(np.clip(selected_selectivity, 0.01, 100))

    # Create clustermap
    g = sns.clustermap(
        log_selectivity,
        cmap='RdBu_r',
        center=0,
        vmin=-2, vmax=2,
        figsize=figsize,
        xticklabels=UPOS_TAGS,
        yticklabels=[str(i) for i in selected_indices],
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )

    g.ax_heatmap.set_xlabel('POS Category', fontsize=11)
    g.ax_heatmap.set_ylabel('Neuron', fontsize=11)
    g.fig.suptitle(f'Clustered Neuron POS Selectivity (Top {len(selected_indices)} neurons)', fontsize=13, y=1.02)

    # Rotate x labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

    g.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_pos_selectivity_from_json(
    selectivity_data: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150
) -> Optional[str]:
    """
    Plot POS selectivity from precomputed JSON data.

    Paper Figure 4: POS Neuron Selectivity
    - Left: Heatmap of top selective neurons per POS
    - Right: Mean selectivity bar chart

    Args:
        selectivity_data: Dict with 'top_selective_per_pos', 'mean_selectivity_by_pos'
        output_path: Path to save the figure
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("  Warning: matplotlib not available")
        return None

    top_selective = selectivity_data.get('top_selective_per_pos', {})
    mean_selectivity = selectivity_data.get('mean_selectivity_by_pos', {})

    if not top_selective:
        print("  Warning: No selectivity data found")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Get POS tags that have data
    pos_tags = [p for p in UPOS_TAGS if p in top_selective and top_selective[p]]
    if not pos_tags:
        print("  Warning: No POS tags with selectivity data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[3, 1])

    # Left: Top selective neurons per POS (grouped bar / heatmap style)
    n_show = 5  # Top N neurons per POS
    neurons_data = []
    labels = []

    for pos in pos_tags:
        neurons = top_selective[pos][:n_show]
        for n in neurons:
            neurons_data.append({
                'pos': pos,
                'neuron': n.get('neuron', str(n.get('neuron_idx', '?'))),
                'selectivity': n.get('selectivity', 0)
            })

    if not neurons_data:
        print("  Warning: No neuron selectivity data to plot")
        return None

    # Create heatmap-style visualization
    # Rows = POS, Cols = rank (1st, 2nd, ... top neuron)
    n_pos = len(pos_tags)
    matrix = np.zeros((n_pos, n_show))
    neuron_labels = [['' for _ in range(n_show)] for _ in range(n_pos)]

    for i, pos in enumerate(pos_tags):
        neurons = top_selective[pos][:n_show]
        for j, n in enumerate(neurons):
            matrix[i, j] = n.get('selectivity', 0)
            neuron_labels[i][j] = n.get('neuron', str(n.get('neuron_idx', '?')))

    # Plot heatmap
    im = axes[0].imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=1, vmax=min(10, matrix.max()))

    # Add neuron labels
    for i in range(n_pos):
        for j in range(n_show):
            if matrix[i, j] > 0:
                text_color = 'white' if matrix[i, j] > 5 else 'black'
                axes[0].text(j, i, f'{neuron_labels[i][j]}\n{matrix[i, j]:.1f}x',
                           ha='center', va='center', fontsize=7, color=text_color)

    axes[0].set_xticks(range(n_show))
    axes[0].set_xticklabels([f'Top {i+1}' for i in range(n_show)])
    axes[0].set_yticks(range(n_pos))
    axes[0].set_yticklabels(pos_tags)
    axes[0].set_xlabel('Rank', fontsize=11)
    axes[0].set_ylabel('POS Category', fontsize=11)
    axes[0].set_title('Most Selective Neurons per POS', fontsize=12)

    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label('Selectivity (×baseline)', fontsize=10)

    # Right: Mean selectivity bar chart
    mean_values = [mean_selectivity.get(pos, 1.0) for pos in pos_tags]
    colors = ['coral' if v > 1.2 else 'steelblue' if v < 0.8 else 'gray' for v in mean_values]

    axes[1].barh(range(n_pos), mean_values, color=colors)
    axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='baseline')
    axes[1].set_yticks(range(n_pos))
    axes[1].set_yticklabels(pos_tags)
    axes[1].set_xlabel('Mean Selectivity', fontsize=10)
    axes[1].set_title('Population Mean', fontsize=11)
    axes[1].set_xlim(0, max(2.0, max(mean_values) * 1.1) if mean_values else 2.0)

    # Add summary
    n_active = selectivity_data.get('n_active_neurons', 0)
    if n_active:
        fig.text(0.5, 0.02, f'Active neurons analyzed: {n_active}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path

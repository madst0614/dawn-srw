"""
Factual Knowledge Heatmap Visualizations
========================================
Visualization for factual knowledge neuron analysis.

Paper Figure 7: Factual Knowledge Neuron Heatmap
"""

import os
import numpy as np
from typing import Dict, List, Optional
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


def plot_factual_heatmap(
    factual_data: Dict,
    output_path: str,
    top_n_neurons: int = 20,
    dpi: int = 150
) -> Optional[str]:
    """
    Semantic-level factual knowledge neuron heatmap.

    Neurons are sorted by semantic category:
    - Category-shared neurons (e.g., all capitals share)
    - Category-specific neurons (e.g., only capitals, not colors)
    - Target-specific neurons

    This visualization shows that related outputs share neuron subsets.

    Args:
        factual_data: Results from BehavioralAnalyzer.analyze_factual_neurons()
        output_path: Path to save the figure
        top_n_neurons: Number of top neurons to show
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return None

    per_target = factual_data.get('per_target', {})
    if not per_target:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Collect all neurons and their frequencies
    all_neurons = defaultdict(lambda: defaultdict(float))

    for target, data in per_target.items():
        if 'error' in data:
            continue
        for nf in data.get('neuron_frequencies', []):
            if isinstance(nf, dict):
                neuron = nf['neuron']
                freq = nf['percentage'] / 100.0
            else:
                neuron, freq = nf
            all_neurons[target][neuron] = freq

    if not all_neurons:
        return None

    targets = list(all_neurons.keys())

    # Auto-detect semantic categories based on target names
    # Category 1: Capitals (cities)
    capital_keywords = ['Paris', 'Berlin', 'Tokyo', 'London', 'Rome', 'Madrid', 'Beijing', 'Seoul']
    capital_targets = [t for t in targets if any(k.lower() in t.lower() for k in capital_keywords)]

    # Category 2: Others (colors, etc.)
    other_targets = [t for t in targets if t not in capital_targets]

    # If no clear category separation, use all targets as one group
    if not capital_targets or not other_targets:
        capital_targets = targets
        other_targets = []

    # Collect all neuron IDs
    all_neuron_ids = set()
    for t in targets:
        all_neuron_ids.update(all_neurons[t].keys())

    if not all_neuron_ids:
        return None

    def get_neuron_category(neuron):
        """Classify neuron by activation pattern across categories."""
        capital_freqs = [all_neurons[t].get(neuron, 0) for t in capital_targets]
        other_freqs = [all_neurons[t].get(neuron, 0) for t in other_targets] if other_targets else []

        capital_avg = np.mean(capital_freqs) if capital_freqs else 0
        other_avg = np.mean(other_freqs) if other_freqs else 0
        capital_min = min(capital_freqs) if capital_freqs else 0

        # Category 0: Shared across all capitals (highest priority)
        if capital_min >= 0.8:
            return (0, -capital_avg, neuron)
        # Category 1: Capital-specific (high in capitals, low in others)
        elif capital_avg >= 0.5 and (not other_freqs or other_avg < 0.3):
            return (1, -capital_avg, neuron)
        # Category 2: Other-specific (high in others, low in capitals)
        elif other_freqs and other_avg >= 0.5 and capital_avg < 0.3:
            return (2, -other_avg, neuron)
        # Category 3: Mixed/weak
        else:
            return (3, -(capital_avg + other_avg), neuron)

    # Sort neurons by semantic category
    sorted_neurons = sorted(all_neuron_ids, key=get_neuron_category)[:top_n_neurons]

    # Order targets: capitals first, then others
    ordered_targets = capital_targets + other_targets

    # Build matrix
    matrix = np.zeros((len(ordered_targets), len(sorted_neurons)))
    for i, target in enumerate(ordered_targets):
        for j, neuron in enumerate(sorted_neurons):
            matrix[i, j] = all_neurons[target].get(neuron, 0)

    # Find category boundaries for vertical lines
    categories = [get_neuron_category(n)[0] for n in sorted_neurons]
    boundaries = []
    for i in range(1, len(categories)):
        if categories[i] != categories[i-1]:
            boundaries.append(i)

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, len(sorted_neurons) * 0.6), max(4, len(ordered_targets) * 0.8)))

    sns.heatmap(
        matrix,
        xticklabels=[f'N{n}' for n in sorted_neurons],
        yticklabels=ordered_targets,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Activation Frequency'},
        linewidths=0.5
    )

    # Add category boundary lines (vertical)
    for b in boundaries:
        ax.axvline(x=b, color='black', linewidth=2)

    # Add horizontal line between capitals and others
    if capital_targets and other_targets:
        ax.axhline(y=len(capital_targets), color='blue', linewidth=2, linestyle='--')

    # Category labels at top
    category_names = ['Shared', 'Capital-specific', 'Other-specific', 'Mixed']
    prev_boundary = 0
    for i, b in enumerate(boundaries + [len(sorted_neurons)]):
        if i < len(category_names) and b > prev_boundary:
            mid = (prev_boundary + b) / 2
            cat_idx = categories[prev_boundary] if prev_boundary < len(categories) else 3
            if cat_idx < len(category_names):
                ax.text(mid, -0.5, category_names[cat_idx], ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='darkblue')
        prev_boundary = b

    ax.set_xlabel('Neuron Index (grouped by semantic category)')
    ax.set_ylabel('Target Token')
    ax.set_title('Semantic-Level Neuron Activation: Related outputs share neuron subsets')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_factual_comparison(
    factual_data: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate comparison visualization for factual neurons.

    Shows:
    - Match rate per target
    - Neuron overlap between targets
    - Category-specific neurons

    Args:
        factual_data: Results from BehavioralAnalyzer.analyze_factual_neurons()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    per_target = factual_data.get('per_target', {})
    if not per_target:
        print("  Warning: No per_target data, skipping factual comparison")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    targets = list(per_target.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))

    # 1. Match rate bar chart
    ax = axes[0]
    match_rates = [per_target[t].get('match_rate', 0) * 100 for t in targets]

    bars = ax.barh(range(len(targets)), match_rates, color=colors)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Match Rate (%)')
    ax.set_title('Factual Recall Accuracy')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    for i, (bar, rate) in enumerate(zip(bars, match_rates)):
        ax.text(rate + 2, i, f'{rate:.0f}%', va='center', fontsize=9)

    # 2. Neuron count comparison (100%, 80%, 50%)
    ax = axes[1]

    x = np.arange(len(targets))
    width = 0.25

    counts_100 = [len(per_target[t].get('common_neurons_100', [])) for t in targets]
    counts_80 = [len(per_target[t].get('common_neurons_80', [])) for t in targets]
    counts_50 = [len(per_target[t].get('common_neurons_50', [])) for t in targets]

    ax.barh(x - width, counts_100, width, label='100%', color='darkred')
    ax.barh(x, counts_80, width, label='80%+', color='coral')
    ax.barh(x + width, counts_50, width, label='50%+', color='lightsalmon')

    ax.set_yticks(x)
    ax.set_yticklabels(targets)
    ax.set_xlabel('Number of Common Neurons')
    ax.set_title('Consistent Neurons by Threshold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # 3. Neuron overlap matrix
    ax = axes[2]

    # Calculate overlap between targets
    overlap_matrix = np.zeros((len(targets), len(targets)))
    for i, t1 in enumerate(targets):
        neurons_1 = set(per_target[t1].get('common_neurons_80', []))
        for j, t2 in enumerate(targets):
            neurons_2 = set(per_target[t2].get('common_neurons_80', []))
            if neurons_1 and neurons_2:
                overlap = len(neurons_1 & neurons_2) / len(neurons_1 | neurons_2)
                overlap_matrix[i, j] = overlap
            elif i == j:
                overlap_matrix[i, j] = 1.0

    if HAS_SEABORN:
        sns.heatmap(
            overlap_matrix,
            xticklabels=targets,
            yticklabels=targets,
            cmap='Blues',
            vmin=0, vmax=1,
            annot=True, fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Jaccard Similarity'}
        )
    else:
        im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(targets)))
        ax.set_yticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.set_yticklabels(targets)
        plt.colorbar(im, ax=ax, label='Jaccard Similarity')

    ax.set_title('Neuron Overlap (80%+ neurons)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path

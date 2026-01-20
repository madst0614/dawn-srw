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

    Uses common_neurons_80 (neurons that fire in 80%+ of successful runs)
    to identify reliable neurons for each target.

    Neurons are categorized by selectivity:
    - Shared: Fire for multiple targets
    - Target-specific: Fire mainly for one target

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

    # Collect neuron frequencies and contrastive scores for each target
    all_neurons = defaultdict(lambda: defaultdict(float))
    all_contrastive = defaultdict(lambda: defaultdict(float))  # contrastive scores
    common_neurons_per_target = {}

    for target, data in per_target.items():
        if 'error' in data:
            continue
        # Get common neurons (80%+ threshold)
        common_neurons_per_target[target] = set(data.get('common_neurons_80', []))
        # Get contrastive scores (target_freq - baseline_freq)
        contrastive_scores = data.get('contrastive_scores', {})
        for neuron, score in contrastive_scores.items():
            all_contrastive[target][neuron] = score
        # Get all frequencies
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
    capital_keywords = ['Paris', 'Berlin', 'Tokyo', 'London', 'Rome', 'Madrid', 'Beijing', 'Seoul']
    capital_targets = [t for t in targets if any(k.lower() in t.lower() for k in capital_keywords)]
    other_targets = [t for t in targets if t not in capital_targets]

    # If no clear category separation, use all targets as one group
    if not capital_targets or not other_targets:
        capital_targets = targets
        other_targets = []

    # Collect neuron IDs with:
    # 1. Meaningful activity (freq > 0.1)
    # 2. Target-specific activation (contrastive_score > 0.3)
    all_neuron_ids = set()
    for t in targets:
        for n, freq in all_neurons[t].items():
            contrastive = all_contrastive[t].get(n, 0)
            # Only include neurons that fire MORE for target than baseline
            if freq > 0.1 and contrastive > 0.3:
                all_neuron_ids.add(n)

    if not all_neuron_ids:
        # Fallback: relax contrastive threshold
        for t in targets:
            for n, freq in all_neurons[t].items():
                contrastive = all_contrastive[t].get(n, 0)
                if freq > 0.1 and contrastive > 0.1:
                    all_neuron_ids.add(n)

    if not all_neuron_ids:
        # Final fallback: use common neurons
        for neurons in common_neurons_per_target.values():
            all_neuron_ids.update(neurons)

    if not all_neuron_ids:
        return None

    def get_neuron_category(neuron):
        """Classify neuron by selectivity pattern across targets."""
        capital_freqs = [all_neurons[t].get(neuron, 0) for t in capital_targets]
        other_freqs = [all_neurons[t].get(neuron, 0) for t in other_targets] if other_targets else []

        capital_avg = np.mean(capital_freqs) if capital_freqs else 0
        other_avg = np.mean(other_freqs) if other_freqs else 0

        # Use percentile-based thresholds relative to the data
        # A neuron with avg freq > 0.3 for a group is considered "active" for that group
        capital_active = capital_avg > 0.3
        other_active = other_avg > 0.3 if other_freqs else False

        # Category 0: Shared - active in both capital and other categories
        if capital_active and other_active:
            return 0

        # Category 1: Capital-specific - active only in capitals
        if capital_active and not other_active:
            return 1

        # Category 2: Other-specific - active only in others
        if other_active and not capital_active:
            return 2

        # Category 3: Low/Mixed activity
        return 3

    def get_selectivity_score(neuron):
        """Compute selectivity: how much a neuron prefers one target over others."""
        freqs = [all_neurons[t].get(neuron, 0) for t in targets]
        if not freqs or max(freqs) == 0:
            return 0
        # Higher variance = more selective
        return float(np.std(freqs))

    def get_max_contrastive(neuron):
        """Get max contrastive score across all targets."""
        scores = [all_contrastive[t].get(neuron, 0) for t in targets]
        return max(scores) if scores else 0

    # Categorize and score all neurons
    categorized = {0: [], 1: [], 2: [], 3: []}
    for neuron in all_neuron_ids:
        cat = get_neuron_category(neuron)
        selectivity = get_selectivity_score(neuron)
        mean_freq = np.mean([all_neurons[t].get(neuron, 0) for t in targets])
        contrastive = get_max_contrastive(neuron)
        # Score: combine frequency, selectivity, and contrastive (target-specific)
        score = mean_freq + selectivity * 2 + contrastive * 3  # Weight contrastive highest
        categorized[cat].append((neuron, score))

    # Sort within each category by score (descending)
    for cat in categorized:
        categorized[cat].sort(key=lambda x: -x[1])

    # Select neurons: prioritize categories with interesting patterns
    # Category 0 (Shared) shows common knowledge, then specific categories
    neurons_per_cat = max(3, top_n_neurons // 4)
    sorted_neurons = []
    category_order = [0, 1, 2, 3]  # Shared first, then Capital-specific, Other-specific, Mixed

    for cat in category_order:
        neurons_in_cat = [n for n, _ in categorized[cat][:neurons_per_cat]]
        sorted_neurons.extend(neurons_in_cat)

    # Trim to top_n_neurons
    sorted_neurons = sorted_neurons[:top_n_neurons]

    if not sorted_neurons:
        # Fallback: just take top neurons by total activity
        all_neurons_scored = []
        for n in all_neuron_ids:
            total = sum(all_neurons[t].get(n, 0) for t in targets)
            all_neurons_scored.append((n, total))
        all_neurons_scored.sort(key=lambda x: -x[1])
        sorted_neurons = [n for n, _ in all_neurons_scored[:top_n_neurons]]

    if not sorted_neurons:
        return None

    # Order targets: capitals first, then others
    ordered_targets = capital_targets + other_targets

    # Build matrix
    matrix = np.zeros((len(ordered_targets), len(sorted_neurons)))
    for i, target in enumerate(ordered_targets):
        for j, neuron in enumerate(sorted_neurons):
            matrix[i, j] = all_neurons[target].get(neuron, 0)

    # Find category boundaries for vertical lines
    categories = [get_neuron_category(n) for n in sorted_neurons]
    boundaries = []
    for i in range(1, len(categories)):
        if categories[i] != categories[i-1]:
            boundaries.append(i)

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, len(sorted_neurons) * 0.6), max(4, len(ordered_targets) * 0.8)))

    # Capitalize target labels for display
    display_targets = [t.capitalize() for t in ordered_targets]

    sns.heatmap(
        matrix,
        xticklabels=[f'N{n}' for n in sorted_neurons],
        yticklabels=display_targets,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Activation Frequency', 'shrink': 0.8, 'pad': 0.02},
        linewidths=0.5
    )

    # Add category boundary lines (vertical)
    for b in boundaries:
        ax.axvline(x=b, color='black', linewidth=2)

    # Add horizontal line between capitals and others
    if capital_targets and other_targets:
        ax.axhline(y=len(capital_targets), color='blue', linewidth=2, linestyle='--')

    # Category labels - above heatmap using transAxes
    category_names = {0: 'Shared', 1: 'Capital-specific', 2: 'Other-specific', 3: 'Mixed'}
    prev_boundary = 0
    n_neurons = len(sorted_neurons)
    for i, b in enumerate(boundaries + [n_neurons]):
        if b > prev_boundary:
            mid = (prev_boundary + b) / 2 / n_neurons  # normalize to 0-1
            cat_idx = categories[prev_boundary] if prev_boundary < len(categories) else 3
            label = category_names.get(cat_idx, 'Mixed')
            ax.text(mid, 1.02, label, ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='darkblue',
                   transform=ax.transAxes)
        prev_boundary = b

    # Title - above category labels
    ax.set_title('Factual Knowledge Neurons: Related outputs share neuron subsets',
                fontsize=11, fontweight='bold', pad=20, y=1.06)

    ax.set_xlabel('Neuron Index (grouped by semantic category)')
    ax.set_ylabel('Target Token')

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

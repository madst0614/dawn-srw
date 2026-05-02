"""
Factual Knowledge Heatmap Visualizations
========================================
Visualization for factual knowledge neuron analysis.

Fig 5: Semantic Coherence of Knowledge Neurons
(Uses F-Know pool neurons only)
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

from .style import PAPER_STYLE
S = PAPER_STYLE


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

        # Handle both old structure (common_neurons_80 at top level)
        # and new multi-pool structure (per_pool: {pool: {common_80: [...]}})
        if 'common_neurons_80' in data:
            # Old structure
            common_neurons_per_target[target] = set(data.get('common_neurons_80', []))
            # Get contrastive scores
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
        elif 'per_pool' in data:
            # New multi-pool structure - only use fknow pool
            all_common = set()
            fknow_data = data.get('per_pool', {}).get('fknow', {})
            if isinstance(fknow_data, dict):
                common_80 = fknow_data.get('common_80', [])
                # Remove pool prefix (fknow_71 → 71)
                for n in common_80:
                    if '_' in str(n):
                        idx = str(n).split('_')[-1]
                        all_common.add(idx)
                    else:
                        all_common.add(str(n))

                # Use all_frequencies for complete freq data
                all_freqs = fknow_data.get('all_frequencies', {})
                for neuron, freq in all_freqs.items():
                    # Remove pool prefix if present
                    if '_' in str(neuron):
                        neuron = str(neuron).split('_')[-1]
                    all_neurons[target][neuron] = freq

                # Load contrastive scores (target_freq - baseline_freq)
                contrastive = fknow_data.get('contrastive_scores', {})
                for neuron, score in contrastive.items():
                    if '_' in str(neuron):
                        neuron = str(neuron).split('_')[-1]
                    all_contrastive[target][neuron] = score

            common_neurons_per_target[target] = all_common

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
        """Classify neuron by selectivity pattern - sequential filtering."""
        capital_freqs = [all_neurons[t].get(neuron, 0) for t in capital_targets]
        other_freqs = [all_neurons[t].get(neuron, 0) for t in other_targets] if other_targets else []
        all_freqs = capital_freqs + other_freqs

        # Step 1: Shared - 모든 타겟에서 0.7+
        if all_freqs and all(f >= 0.7 for f in all_freqs):
            return 0  # Shared

        # Step 2: Capital-specific - 모든 capital 0.7+ AND 모든 other < 0.3
        if capital_freqs and other_freqs:
            if (all(f >= 0.7 for f in capital_freqs) and
                all(f < 0.3 for f in other_freqs)):
                return 1  # Capital-specific

        # Step 3: Other-specific - 해당 other 0.7+ AND 모든 capital < 0.3
        if other_freqs and capital_freqs:
            if (any(f >= 0.7 for f in other_freqs) and
                all(f < 0.3 for f in capital_freqs)):
                return 2  # Other-specific

        return 3  # Mixed (exclude from plot)

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
    # Exclude Mixed category (3)
    # 각 카테고리별 최대 5개, mean_freq 기준으로 정렬
    max_per_category = 5
    sorted_neurons = []
    category_order = [0, 1, 2]  # Shared, Capital-specific, Other-specific (no Mixed)

    for cat in category_order:
        # Sort by mean_freq instead of composite score
        neurons_in_cat = sorted(
            categorized[cat],
            key=lambda x: -np.mean([all_neurons[t].get(x[0], 0) for t in targets])
        )[:max_per_category]
        sorted_neurons.extend([n for n, _ in neurons_in_cat])

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

    # Custom annotation: hide 0.00 values
    annot_matrix = np.where(matrix > 0, matrix, np.nan)
    annot_labels = [[f'{v:.2f}' if not np.isnan(v) else '' for v in row] for row in annot_matrix]

    sns.heatmap(
        matrix,
        xticklabels=[str(n) for n in sorted_neurons],  # Neuron index only
        yticklabels=display_targets,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=annot_labels, fmt='',
        annot_kws={'fontsize': S['font_size_annotation']},
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
                   fontsize=S['font_size_category'], fontweight='bold', color='darkblue',
                   transform=ax.transAxes)
        prev_boundary = b

    # No title - figure number added in paper

    ax.set_xlabel('Neuron Index')
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
        ax.text(rate + 2, i, f'{rate:.0f}%', va='center', fontsize=S['font_size_annotation'])

    # 2. Neuron count comparison (100%, 80%, 50%)
    ax = axes[1]

    x = np.arange(len(targets))
    width = 0.25

    # Helper to get neuron counts from both old and new structure
    def get_neuron_counts(target_data, threshold):
        # Old structure: common_neurons_XX at top level
        key = f'common_neurons_{threshold}'
        if key in target_data:
            return len(target_data[key])
        # New multi-pool structure: per_pool with common_XX
        if 'per_pool' in target_data:
            all_neurons = set()
            pool_key = f'common_{threshold}'
            for pool, pool_data in target_data.get('per_pool', {}).items():
                if isinstance(pool_data, dict):
                    all_neurons.update(pool_data.get(pool_key, []))
            return len(all_neurons)
        return 0

    def get_common_neurons(target_data, threshold=80):
        """Get set of common neurons from either data structure."""
        key = f'common_neurons_{threshold}'
        if key in target_data:
            return set(target_data[key])
        if 'per_pool' in target_data:
            all_neurons = set()
            pool_key = f'common_{threshold}'
            for pool, pool_data in target_data.get('per_pool', {}).items():
                if isinstance(pool_data, dict):
                    all_neurons.update(pool_data.get(pool_key, []))
            return all_neurons
        return set()

    counts_100 = [get_neuron_counts(per_target[t], 100) for t in targets]
    counts_80 = [get_neuron_counts(per_target[t], 80) for t in targets]
    counts_50 = [get_neuron_counts(per_target[t], 50) for t in targets]

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
        neurons_1 = get_common_neurons(per_target[t1], 80)
        for j, t2 in enumerate(targets):
            neurons_2 = get_common_neurons(per_target[t2], 80)
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

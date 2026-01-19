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
    top_n_neurons: int = 15,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate factual knowledge neuron heatmap.

    Paper Figure 7: Shows which neurons activate for different factual prompts.

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
    neuron_total = defaultdict(float)

    for target, data in per_target.items():
        if 'error' in data:
            continue
        for nf in data.get('neuron_frequencies', []):
            # Support both dict format and tuple format
            if isinstance(nf, dict):
                neuron = nf['neuron']
                freq = nf['percentage'] / 100.0  # Convert to 0-1 range
            else:
                neuron, freq = nf
            all_neurons[target][neuron] = freq
            neuron_total[neuron] += freq

    if not all_neurons:
        return None

    # Get top neurons by total frequency
    top_neurons = sorted(neuron_total.keys(), key=lambda n: -neuron_total[n])[:top_n_neurons]
    targets = list(all_neurons.keys())

    # Build matrix
    matrix = np.zeros((len(targets), len(top_neurons)))
    for i, target in enumerate(targets):
        for j, neuron in enumerate(top_neurons):
            matrix[i, j] = all_neurons[target].get(neuron, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(4, len(targets) * 0.6)))

    sns.heatmap(
        matrix,
        xticklabels=[f'N{n}' for n in top_neurons],
        yticklabels=targets,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Activation Frequency'}
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Target Token')
    ax.set_title('Factual Knowledge Neuron Activation Patterns')

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Match rate bar chart
    ax = axes[0]
    targets = list(per_target.keys())
    match_rates = [per_target[t].get('match_rate', 0) * 100 for t in targets]

    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
    bars = ax.barh(range(len(targets)), match_rates, color=colors)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Match Rate (%)')
    ax.set_title('Factual Recall Accuracy')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, match_rates)):
        ax.text(rate + 2, i, f'{rate:.0f}%', va='center', fontsize=9)

    # 2. Neuron count comparison
    ax = axes[1]
    common_counts = [len(per_target[t].get('common_neurons_80', [])) for t in targets]

    # Ensure non-negative values for proper display
    max_count = max(common_counts) if common_counts and max(common_counts) > 0 else 1

    bars = ax.barh(range(len(targets)), common_counts, color=colors)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Number of Common Neurons (>80%)')
    ax.set_title('Factual Knowledge Neurons')
    ax.set_xlim(0, max(max_count * 1.2, 1))  # Ensure positive xlim
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, common_counts)):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path

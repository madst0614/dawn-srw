"""
Neuron Health Visualizations (Forward-based)
=============================================
Visualization functions for neuron health analysis.

All visualizations based on forward pass activation data,
not EMA statistics.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_dead_neurons(
    results: Dict,
    pool_names: List[str],
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate dead neuron visualization (forward-based).

    Creates pie charts showing active/dead breakdown per pool,
    plus a summary bar chart.

    Args:
        results: Dead neuron analysis results (from forward-based analysis)
        pool_names: List of pool names to visualize
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    n_pools = len(pool_names)
    n_cols = 3
    n_rows = (n_pools + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_pools == 0 else list(axes)

    colors = ['green', 'red']
    labels = ['Active', 'Dead']

    for ax, name in zip(axes[:n_pools], pool_names):
        data = results.get(name, {})
        if not data or 'n_active' not in data:
            ax.axis('off')
            continue
        sizes = [data['n_active'], data['n_dead']]
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{name}\n(dead: {data["n_dead"]}/{data["n_total"]})')

    # Summary bar chart
    if n_pools < len(axes):
        ax = axes[n_pools]
        display_names = [n[:8] for n in pool_names]
        dead_counts = [results.get(n, {}).get('n_dead', 0) for n in pool_names]
        ax.bar(display_names, dead_counts, color='red', alpha=0.7)
        ax.set_title('Dead Neurons by Pool')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    for i in range(n_pools + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_activation_histogram(
    activation_data: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Create activation frequency histogram plots (forward-based).

    Args:
        activation_data: Dict mapping pool names to activation stats
            Each entry should have 'stats' with frequency statistics
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Filter to pools with data
    pools = [(name, data) for name, data in activation_data.items()
             if isinstance(data, dict) and 'active_ratio' in data]

    if not pools:
        return None

    n_pools = len(pools)
    n_cols = min(3, n_pools)
    n_rows = (n_pools + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_pools == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['red', 'orange', 'blue', 'green', 'purple', 'cyan']

    for i, (name, data) in enumerate(pools):
        ax = axes[i]
        color = colors[i % len(colors)]

        # Bar chart of active vs dead
        counts = [data['active'], data['dead']]
        ax.bar(['Active', 'Dead'], counts, color=[color, 'gray'], alpha=0.7)
        ax.set_title(f'{name}\n({data["active_ratio"]*100:.1f}% active)')
        ax.set_ylabel('Count')

    # Hide unused axes
    for i in range(len(pools), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_diversity_summary(
    diversity_data: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Create diversity summary visualization.

    Args:
        diversity_data: Results from diversity analysis
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Filter to pools with entropy data
    pools = [(name, data) for name, data in diversity_data.items()
             if isinstance(data, dict) and 'normalized_entropy' in data]

    if not pools:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normalized entropy bar chart
    ax = axes[0]
    names = [p[0][:8] for p in pools]
    entropies = [p[1]['normalized_entropy'] for p in pools]
    colors = ['green' if e > 0.7 else 'orange' if e > 0.4 else 'red' for e in entropies]
    ax.bar(names, entropies, color=colors, alpha=0.7)
    ax.set_title('Normalized Entropy by Pool')
    ax.set_ylabel('Normalized Entropy')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    # Coverage bar chart
    ax = axes[1]
    coverages = [p[1]['coverage'] * 100 for p in pools]
    ax.bar(names, coverages, color='steelblue', alpha=0.7)
    ax.set_title('Neuron Coverage by Pool')
    ax.set_ylabel('Coverage (%)')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


# Legacy alias for backward compatibility
def plot_usage_histogram(
    ema_data: List[Tuple[str, np.ndarray, str]],
    output_path: str,
    threshold: float = 0.01,
    dpi: int = 150
) -> Optional[str]:
    """
    Legacy function - creates activation histogram from raw data.

    Note: This is kept for backward compatibility. New code should use
    plot_activation_histogram with pre-computed activation_data.
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    n_plots = len(ema_data) + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else list(axes)

    for ax, (name, values, color) in zip(axes, ema_data):
        if not isinstance(values, np.ndarray):
            values = values.detach().cpu().numpy()
        ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'threshold={threshold}')
        ax.set_title(f'{name} Activation Frequency')
        ax.set_xlabel('Activation Frequency')
        ax.set_ylabel('Count')

        active = (values > threshold).sum()
        total = len(values)
        ax.text(0.95, 0.95, f'Active: {active}/{total}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10)

    # Summary bar chart
    if len(ema_data) < len(axes):
        ax = axes[len(ema_data)]
        names = [d[0] for d in ema_data]
        active_ratios = []
        for _, values, _ in ema_data:
            if not isinstance(values, np.ndarray):
                values = values.detach().cpu().numpy()
            active_ratios.append((values > threshold).mean())
        colors = [d[2] for d in ema_data]
        ax.bar(names, active_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Active Neuron Ratio by Pool')
        ax.set_ylabel('Active Ratio')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

    for i in range(len(ema_data) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path

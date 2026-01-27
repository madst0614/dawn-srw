"""
Q/K Specialization Visualizations
=================================
Visualization functions for Q/K neuron usage and specialization patterns.

Fig 3: Per-pool Q/K specialization (scatter + bar, 2×1 vertical)
- fig3_fqk_specialization.png (main paper)
- fig3_rqk_specialization.png (appendix)
"""

import os
import numpy as np
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Paper-quality style settings
from .style import PAPER_STYLE, apply_paper_style
if HAS_MATPLOTLIB:
    apply_paper_style(plt)
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
S = PAPER_STYLE

# Color palette
COLOR_Q = '#C0392B'      # Dark red for Q
COLOR_K = '#2471A3'      # Dark blue for K
COLOR_SHARED = '#50C878' # Green for shared
COLOR_INACTIVE = '#95A5A6'  # Gray for inactive
COLOR_BLACK = '#2C3E50'


def _format_millions(x, p):
    """Format axis values in millions (M)."""
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def plot_qk_specialization(
    qk_usage_data: Dict,
    output_path: str,
    pool_colors: Dict = None,
    figsize_per_pool: tuple = (12, 4.5),
    dpi: int = 300
) -> Optional[str]:
    """
    Generate Q/K specialization figure (Fig 3: Emergent Q/K Functional Separation).

    Creates 2-panel figure for each QK pool:
    - Left: Q vs K usage scatter plot
    - Right: Neuron specialization bar chart

    Args:
        qk_usage_data: Results from RoutingAnalyzer.analyze_qk_usage()
        output_path: Path to save the figure
        pool_colors: Optional dict mapping pool names to colors
        figsize_per_pool: Figure size per pool row
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Default colors per pool
    if pool_colors is None:
        pool_colors = {
            'feature_qk': COLOR_Q,
            'restore_qk': COLOR_K,
            'fqk': COLOR_Q,
            'rqk': COLOR_K,
        }

    # Filter out non-pool keys
    pools = {k: v for k, v in qk_usage_data.items() if isinstance(v, dict) and 'q_counts' in v}
    n_pools = len(pools)

    if n_pools == 0:
        return None

    fig, axes = plt.subplots(n_pools, 2, figsize=(figsize_per_pool[0], figsize_per_pool[1] * n_pools))
    if n_pools == 1:
        axes = axes.reshape(1, -1)

    for row, (pool_name, data) in enumerate(pools.items()):
        q_counts = np.array(data['q_counts'])
        k_counts = np.array(data['k_counts'])
        color = pool_colors.get(pool_name, COLOR_Q)
        display_name = data.get('display', pool_name).replace('_', '-').upper()

        # 1. Scatter: Q vs K usage
        ax = axes[row, 0]
        ax.scatter(q_counts, k_counts, alpha=0.6, s=25, c=color, edgecolors='white', linewidth=0.3)
        max_val = max(q_counts.max(), k_counts.max()) if len(q_counts) > 0 else 1
        ax.plot([0, max_val], [0, max_val], '--', color=COLOR_BLACK, alpha=0.5, linewidth=1, label='Q=K')
        ax.set_xlabel('Q Selection Count')
        ax.set_ylabel('K Selection Count')
        corr = data.get('correlation', 0)
        ax.set_title(f'{display_name}: Q vs K Usage (corr={corr:.3f})', fontsize=S['font_size_subtitle'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=S['font_size_legend'], framealpha=0.9)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        # Format axes in M/K notation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_millions))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_millions))

        # 2. Bar: Specialization categories
        ax = axes[row, 1]
        categories = ['Q-only', 'K-only', 'Shared']
        values = [
            data.get('q_specialized', 0),
            data.get('k_specialized', 0),
            data.get('shared', 0),
        ]
        colors = [COLOR_Q, COLOR_K, COLOR_SHARED]
        bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_ylabel('Neuron Count')
        ax.set_title(f'{display_name}: Neuron Specialization', fontsize=S['font_size_subtitle'], fontweight='bold')
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                       ha='center', va='bottom', fontsize=S['font_size_tick'], fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG only
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def plot_qk_pool(
    pool_data: Dict,
    pool_name: str,
    output_path: str,
    color: str = None,
    figsize: tuple = (5, 8),
    dpi: int = 300
) -> Optional[str]:
    """
    Generate per-pool Q/K specialization figure (scatter + bar, 2×1 vertical).

    Args:
        pool_data: Single pool data dict with q_counts, k_counts, etc.
        pool_name: Pool key name (e.g. 'fqk', 'rqk')
        output_path: Path to save the figure
        color: Scatter dot color
        figsize: Figure size
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    if color is None:
        color = {'fqk': COLOR_Q, 'feature_qk': COLOR_Q}.get(pool_name, COLOR_K)

    display_name = pool_data.get('display', pool_name).replace('_', '-').upper()
    q_counts = np.array(pool_data['q_counts'])
    k_counts = np.array(pool_data['k_counts'])
    corr = pool_data.get('correlation', 0)

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Top: Scatter
    ax = axes[0]
    ax.scatter(q_counts, k_counts, alpha=0.6, s=25, c=color, edgecolors='white', linewidth=0.3)
    max_val = max(q_counts.max(), k_counts.max()) if len(q_counts) > 0 else 1
    ax.plot([0, max_val], [0, max_val], '--', color=COLOR_BLACK, alpha=0.5, linewidth=1, label='Q=K')
    ax.set_xlabel('Q Selection Count')
    ax.set_ylabel('K Selection Count')
    ax.set_title(f'{display_name}: Q vs K Usage (corr={corr:.3f})', fontsize=S['font_size_subtitle'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=S['font_size_legend'], framealpha=0.9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_millions))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_millions))

    # Bottom: Bar
    ax = axes[1]
    categories = ['Q-only', 'K-only', 'Shared']
    values = [
        pool_data.get('q_specialized', 0),
        pool_data.get('k_specialized', 0),
        pool_data.get('shared', 0),
    ]
    colors = [COLOR_Q, COLOR_K, COLOR_SHARED]
    bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Neuron Count')
    ax.set_title(f'{display_name}: Neuron Specialization', fontsize=S['font_size_subtitle'], fontweight='bold')
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                   ha='center', va='bottom', fontsize=S['font_size_tick'], fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none')
    plt.close()
    return output_path


def plot_qk_usage(
    qk_usage_data: Dict,
    output_dir: str,
    filename: str = 'qk_usage_analysis.png'
) -> Optional[str]:
    """Legacy wrapper for plot_qk_specialization."""
    output_path = os.path.join(output_dir, filename)
    return plot_qk_specialization(qk_usage_data, output_path)

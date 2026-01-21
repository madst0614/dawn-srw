"""
Q/K Specialization Visualizations
=================================
Visualization functions for Q/K neuron usage and specialization patterns.

Paper Figure 3: Q/K Specialization
- Left: Q vs K scatter plot with correlation
- Middle: Q-only/K-only/Shared/Inactive bar chart
- Right: Q/(Q+K) ratio histogram
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
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

# Color palette
COLOR_Q = '#C0392B'      # Dark red for Q
COLOR_K = '#2471A3'      # Dark blue for K
COLOR_SHARED = '#50C878' # Green for shared
COLOR_INACTIVE = '#95A5A6'  # Gray for inactive
COLOR_PURPLE = '#9B59B6'    # Purple for histogram
COLOR_BLACK = '#2C3E50'


def plot_qk_specialization(
    qk_usage_data: Dict,
    output_path: str,
    pool_colors: Dict = None,
    figsize_per_pool: tuple = (15, 4.5),
    dpi: int = 300
) -> Optional[str]:
    """
    Generate Q/K specialization figure (Paper Figure 3).

    Creates 3-panel figure for each QK pool:
    - Left: Q vs K usage scatter plot
    - Middle: Neuron specialization bar chart
    - Right: Q/(Q+K) ratio distribution histogram

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

    fig, axes = plt.subplots(n_pools, 3, figsize=(figsize_per_pool[0], figsize_per_pool[1] * n_pools))
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
        ax.set_title(f'{display_name}: Q vs K Usage\n(corr={corr:.3f})', fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        # Disable scientific notation on axes
        ax.ticklabel_format(style='plain', axis='both')

        # 2. Bar: Specialization categories
        ax = axes[row, 1]
        categories = ['Q-only', 'K-only', 'Shared', 'Inactive']
        values = [
            data.get('q_specialized', 0),
            data.get('k_specialized', 0),
            data.get('shared', 0),
            data.get('inactive', 0)
        ]
        colors = [COLOR_Q, COLOR_K, COLOR_SHARED, COLOR_INACTIVE]
        bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_ylabel('Neuron Count')
        ax.set_title(f'{display_name}: Neuron Specialization', fontweight='bold')
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # 3. Histogram: Q/(Q+K) ratio distribution with specialization thresholds
        ax = axes[row, 2]
        # Use precomputed q_ratio if available (more accurate with inactive handling)
        if 'q_ratio_active' in data and len(data['q_ratio_active']) > 0:
            q_ratio_active = np.array(data['q_ratio_active'])
        else:
            total = q_counts + k_counts + 1e-8
            q_ratio = q_counts / total
            active_mask = (q_counts + k_counts) > 0
            q_ratio_active = q_ratio[active_mask] if active_mask.sum() > 0 else np.array([])

        if len(q_ratio_active) > 0:
            ax.hist(q_ratio_active, bins=20, alpha=0.75, color=COLOR_PURPLE,
                   edgecolor='white', linewidth=0.5)
        # Threshold lines for specialization
        thresholds = data.get('specialization_thresholds', {'q_specialized': 0.7, 'k_specialized': 0.3})
        ax.axvline(x=thresholds.get('k_specialized', 0.3), color=COLOR_K, linestyle='--',
                  linewidth=1.5, label=f'K-spec (<{thresholds.get("k_specialized", 0.3)})')
        ax.axvline(x=thresholds.get('q_specialized', 0.7), color=COLOR_Q, linestyle='--',
                  linewidth=1.5, label=f'Q-spec (>{thresholds.get("q_specialized", 0.7)})')
        ax.set_xlabel('Q Ratio (Q / (Q+K))')
        ax.set_ylabel('Neuron Count')
        ax.set_title(f'{display_name}: Q/K Balance Distribution', fontweight='bold')
        ax.legend(loc='upper center', fontsize=7, framealpha=0.9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG only
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def plot_qk_usage(
    qk_usage_data: Dict,
    output_dir: str,
    filename: str = 'qk_usage_analysis.png'
) -> Optional[str]:
    """
    Convenience wrapper for plot_qk_specialization.

    Args:
        qk_usage_data: Results from RoutingAnalyzer.analyze_qk_usage()
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved figure or None
    """
    output_path = os.path.join(output_dir, filename)
    return plot_qk_specialization(qk_usage_data, output_path)

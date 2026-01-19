"""
Layer Contribution Visualizer
============================

Paper-quality visualization for DAWN layer-wise circuit contribution.
Matches the style from figures/fig4_routing_stats.py.
"""

import os
import numpy as np
from typing import Dict, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Style settings (matching fig4_routing_stats.py)
if HAS_MATPLOTLIB:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

# Colors (matching fig4_routing_stats.py)
COLOR_ATTENTION = '#4A90D9'
COLOR_KNOWLEDGE = '#50C878'
COLOR_GRAY = '#7F8C8D'


def plot_layer_contribution(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate paper-quality layer contribution figure.

    Creates a single-panel figure showing Layer-wise Circuit Contribution
    (Attention vs Knowledge ratio per layer).

    Args:
        contribution_data: Dict with keys:
            - 'per_layer': Dict[str, dict] with layer data containing 'attention_ratio'
            - 'layer_contributions': Alternative list of attention ratios (0-100)
        output_path: Path for output PNG
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    # Extract layer stats
    layer_stats = contribution_data.get('layer_contributions', [])

    if not layer_stats:
        per_layer = contribution_data.get('per_layer', {})
        if per_layer:
            # Handle dict format (L0, L1, ...) or list format
            if isinstance(per_layer, dict):
                # Sort by layer index and extract attention ratios
                sorted_layers = sorted(per_layer.items(),
                                       key=lambda x: int(x[0].replace('L', '')) if x[0].startswith('L') else int(x[0]))
                layer_stats = []
                for _, layer_data in sorted_layers:
                    ratio = layer_data.get('attention_ratio', 0.5)
                    # Convert 0-1 to 0-100 if needed
                    if ratio <= 1.0:
                        ratio = ratio * 100
                    layer_stats.append(ratio)
            elif isinstance(per_layer, list):
                layer_stats = []
                for layer_data in per_layer:
                    ratio = layer_data.get('attention_ratio', 0.5)
                    if ratio <= 1.0:
                        ratio = ratio * 100
                    layer_stats.append(ratio)

    if not layer_stats:
        print("No layer contribution data available")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)

    n_layers = len(layer_stats)
    layers = list(range(1, n_layers + 1))

    # Plot line with markers
    ax.plot(layers, layer_stats, 'o-', color=COLOR_ATTENTION, linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)

    # Fill areas
    ax.fill_between(layers, layer_stats, 50,
                    where=[a >= 50 for a in layer_stats],
                    color=COLOR_ATTENTION, alpha=0.3)
    ax.fill_between(layers, layer_stats, 50,
                    where=[a < 50 for a in layer_stats],
                    color=COLOR_KNOWLEDGE, alpha=0.3)

    # Baseline
    ax.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

    # Axis settings
    ax.set_xlim(0.5, n_layers + 0.5)
    ax.set_ylim(35, 75)
    ax.set_xticks(layers)
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Attention Contribution (%)', fontsize=9)
    ax.set_title('Layer-wise Circuit Contribution', fontsize=10, fontweight='bold', pad=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
        mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
        plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG only
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path

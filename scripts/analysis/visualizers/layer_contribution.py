"""
Layer Contribution Visualizer
=============================

Fig 7: Attention-Knowledge Balance Across Layers

Paper-quality visualization for DAWN layer-wise circuit contribution.
Creates a single-panel figure showing attention vs knowledge contribution per layer.
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

# Style settings
from .style import PAPER_STYLE, apply_paper_style
if HAS_MATPLOTLIB:
    apply_paper_style(plt)
S = PAPER_STYLE

# Colors
COLOR_ATTENTION = '#4A90D9'
COLOR_KNOWLEDGE = '#50C878'
COLOR_GRAY = '#7F8C8D'


def extract_layer_stats_from_contribution(contribution_data: Dict) -> list:
    """
    Extract layer-wise attention contribution percentages.

    Args:
        contribution_data: Results from analyze_layer_contribution()

    Returns:
        List of attention contribution percentages (0-100) per layer
    """
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

    return layer_stats


def plot_routing_stats(
    routing_data: Dict,
    output_path: str,
    router=None,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate paper-quality layer-wise circuit contribution figure.

    Shows attention vs knowledge contribution per layer as a single panel.

    Args:
        routing_data: Dict with routing analysis results containing:
            - 'layer_stats': pre-computed layer stats list
            - 'layer_contribution': for layer-wise data extraction
        output_path: Path for output PNG
        router: NeuronRouter instance (unused, kept for API compatibility)
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    # Extract layer stats
    layer_stats = routing_data.get('layer_stats', [])
    if not layer_stats:
        contribution_data = routing_data.get('layer_contribution', {})
        if contribution_data:
            layer_stats = extract_layer_stats_from_contribution(contribution_data)

    if not layer_stats:
        print("No layer contribution data available")
        return None

    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    n_layers = len(layer_stats)
    layers = list(range(1, n_layers + 1))

    # Plot line with markers
    ax.plot(layers, layer_stats, 'o-', color=COLOR_ATTENTION, linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)

    # Fill areas above/below 50%
    ax.fill_between(layers, layer_stats, 50,
                    where=[a >= 50 for a in layer_stats],
                    color=COLOR_ATTENTION, alpha=0.3)
    ax.fill_between(layers, layer_stats, 50,
                    where=[a < 50 for a in layer_stats],
                    color=COLOR_KNOWLEDGE, alpha=0.3)

    # 50% baseline
    ax.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

    # Styling
    ax.set_xlim(0.5, n_layers + 0.5)
    ax.set_ylim(35, 75)
    ax.set_xticks(layers)
    ax.set_xlabel('Layer', fontsize=S['font_size_label'])
    ax.set_ylabel('Attention Contribution (%)', fontsize=S['font_size_label'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
        mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
        plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=S['font_size_legend'], framealpha=0.9)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path


# Backward compatible alias
def plot_layer_contribution(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 300
) -> Optional[str]:
    """Backward compatible wrapper for plot_routing_stats."""
    return plot_routing_stats(contribution_data, output_path, dpi=dpi)

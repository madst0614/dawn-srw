"""
Layer Contribution & Routing Stats Visualizer
==============================================

Paper-quality visualization for DAWN routing statistics.
Matches the style from figures/fig4_routing_stats.py.

Creates a 2-panel figure:
(a) Neuron Utilization - horizontal bar chart showing pool usage
(b) Layer-wise Circuit Contribution - line plot with fill
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
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'


def extract_utilization_from_routing(routing_data: Dict, router=None) -> Dict:
    """
    Extract neuron utilization from routing analysis data or router EMA.

    Priority:
    1. Router's usage_ema_* attributes (most accurate)
    2. Pre-computed 'utilization' in routing_data
    3. qk_usage counts (fallback)

    Args:
        routing_data: Results from RoutingAnalyzer
        router: NeuronRouter instance (optional, for EMA access)

    Returns:
        Dict with pool names and utilization percentages
    """
    utilization = {}

    # Method 1: Get EMA from router directly (most accurate, matches fig4)
    if router is not None:
        nr = router.neuron_router if hasattr(router, 'neuron_router') else router
        ema_attrs = {
            'Feature_Q': 'usage_ema_feature_q',
            'Feature_K': 'usage_ema_feature_k',
            'Feature_V': 'usage_ema_feature_v',
            'Restore_Q': 'usage_ema_restore_q',
            'Restore_K': 'usage_ema_restore_k',
            'Restore_V': 'usage_ema_restore_v',
            'Feature_Know': 'usage_ema_feature_know',
            'Restore_Know': 'usage_ema_restore_know',
        }

        for display, attr in ema_attrs.items():
            if hasattr(nr, attr):
                ema = getattr(nr, attr)
                if ema is not None:
                    # Match fig4: threshold > 0.01
                    active_ratio = (ema > 0.01).float().mean().item() * 100
                    utilization[display] = active_ratio

    # Method 2: Pre-computed utilization dict
    if not utilization:
        utilization = routing_data.get('utilization', {})

    # Method 3: Calculate from qk_usage counts (from actual forward passes)
    if not utilization:
        qk_usage = routing_data.get('qk_usage', {})
        for pool_name, pool_data in qk_usage.items():
            if pool_name == 'n_batches' or not isinstance(pool_data, dict):
                continue

            q_counts = pool_data.get('q_counts', [])
            k_counts = pool_data.get('k_counts', [])
            n_neurons = pool_data.get('n_neurons', len(q_counts) if q_counts else 0)

            if n_neurons > 0 and q_counts:
                q_counts = np.array(q_counts)
                k_counts = np.array(k_counts)

                # Utilization = neurons selected at least once / total neurons
                # Use count > 0 to match "ever selected during inference"
                q_active = (q_counts > 0).sum() / n_neurons * 100
                k_active = (k_counts > 0).sum() / n_neurons * 100

                # Map pool names: 'feature_qk' -> Feature_Q/K, 'restore_qk' -> Restore_Q/K
                if 'feature' in pool_name.lower():
                    utilization['Feature_Q'] = q_active
                    utilization['Feature_K'] = k_active
                elif 'restore' in pool_name.lower():
                    utilization['Restore_Q'] = q_active
                    utilization['Restore_K'] = k_active

    return utilization


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
    Generate paper-quality routing statistics figure with 2 panels.

    Creates:
    (a) Neuron Utilization - horizontal bar chart
    (b) Layer-wise Circuit Contribution - line plot

    Args:
        routing_data: Dict with routing analysis results containing:
            - 'utilization': pre-computed utilization dict
            - 'layer_stats': pre-computed layer stats list
            - 'layer_contribution': for layer-wise data extraction
            - 'qk_usage': fallback for utilization
        output_path: Path for output PNG
        router: NeuronRouter instance (optional, for EMA access)
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    # Extract utilization (try router EMA first, then routing_data)
    utilization = routing_data.get('utilization', {})
    if not utilization:
        utilization = extract_utilization_from_routing(routing_data, router)

    # Extract layer stats
    layer_stats = routing_data.get('layer_stats', [])
    if not layer_stats:
        contribution_data = routing_data.get('layer_contribution', {})
        if contribution_data:
            layer_stats = extract_layer_stats_from_contribution(contribution_data)

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=dpi)

    # === (a) Neuron Utilization ===
    if utilization:
        # Order pools for display
        pool_order = ['Feature_Q', 'Feature_K', 'Feature_V',
                     'Restore_Q', 'Restore_K', 'Restore_V',
                     'Feature_Know', 'Restore_Know']
        pools = [p for p in pool_order if p in utilization][::-1]  # Reverse for horizontal bar
        values = [utilization[p] for p in pools]

        # Colors based on type
        colors = []
        for p in pools:
            if 'Know' in p:
                colors.append(COLOR_KNOWLEDGE)
            elif 'Q' in p:
                colors.append('#C0392B')  # Darker red for Q
            elif 'K' in p:
                colors.append('#2471A3')  # Darker blue for K
            else:
                colors.append('#9B59B6')  # Purple for V

        y_pos = np.arange(len(pools))
        bars = ax1.barh(y_pos, values, height=0.7, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(val + 1, i, f'{val:.0f}%', va='center', fontsize=8, color=COLOR_BLACK)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pools, fontsize=8)
        ax1.set_xlim(0, 105)
        ax1.set_xlabel('Active Neurons (%)', fontsize=9)
        ax1.set_title('(a) Neuron Utilization', fontsize=10, fontweight='bold', pad=10)
        ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax1.set_axisbelow(True)
        ax1.axvline(x=50, color=COLOR_GRAY, linestyle=':', linewidth=1, alpha=0.7)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#C0392B', label='Q routing', alpha=0.85),
            mpatches.Patch(color='#2471A3', label='K routing', alpha=0.85),
            mpatches.Patch(color='#9B59B6', label='V routing', alpha=0.85),
            mpatches.Patch(color=COLOR_KNOWLEDGE, label='Knowledge', alpha=0.85),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax1.transAxes)

    # === (b) Layer-wise Circuit Contribution ===
    if layer_stats:
        n_layers = len(layer_stats)
        layers = list(range(1, n_layers + 1))

        ax2.plot(layers, layer_stats, 'o-', color=COLOR_ATTENTION, linewidth=2,
                markersize=6, markerfacecolor='white', markeredgewidth=1.5)

        # Fill areas
        ax2.fill_between(layers, layer_stats, 50,
                        where=[a >= 50 for a in layer_stats],
                        color=COLOR_ATTENTION, alpha=0.3)
        ax2.fill_between(layers, layer_stats, 50,
                        where=[a < 50 for a in layer_stats],
                        color=COLOR_KNOWLEDGE, alpha=0.3)

        ax2.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

        ax2.set_xlim(0.5, n_layers + 0.5)
        ax2.set_ylim(35, 75)
        ax2.set_xticks(layers)
        ax2.set_xlabel('Layer', fontsize=9)
        ax2.set_ylabel('Attention Contribution (%)', fontsize=9)
        ax2.set_title('(b) Layer-wise Circuit Contribution', fontsize=10, fontweight='bold', pad=10)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax2.set_axisbelow(True)

        # Legend
        legend_elements2 = [
            mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
            mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
            plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
        ]
        ax2.legend(handles=legend_elements2, loc='upper right', fontsize=7, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG only
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path


# Keep backward compatible alias
def plot_layer_contribution(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 300
) -> Optional[str]:
    """
    Backward compatible wrapper for plot_routing_stats.

    Now generates 2-panel figure if routing data is available.
    """
    return plot_routing_stats(contribution_data, output_path, dpi)

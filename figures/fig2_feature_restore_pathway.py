#!/usr/bin/env python3
"""
Figure 2: Feature-Restore Pathway
Paper-style figure showing the sparse routing mechanism.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

# Colors
COLOR_FEATURE = '#4A90D9'
COLOR_RESTORE = '#50C878'
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'
COLOR_LIGHT_BLUE = '#E3F2FD'
COLOR_LIGHT_GREEN = '#E8F5E9'

def draw_box(ax, x, y, w, h, text, facecolor='white', edgecolor='black',
             fontsize=9, textcolor='black', lw=1.5):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(box)
    if '\n' in text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, linespacing=1.2)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor)
    return box

def draw_arrow(ax, start, end, color='black', lw=1.5):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color,
                           linewidth=lw, mutation_scale=15)
    ax.add_patch(arrow)
    return arrow

def main():
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # === Top: Main Flow ===
    y_main = 4.0

    # Input x
    draw_box(ax, 0.3, y_main, 1.0, 0.7, 'x\n(d=384)',
             facecolor='#F5F5F5', edgecolor=COLOR_BLACK, fontsize=8)

    # Arrow
    draw_arrow(ax, (1.3, y_main + 0.35), (2.0, y_main + 0.35), color=COLOR_BLACK)

    # Feature Neurons
    draw_box(ax, 2.0, y_main - 0.1, 1.8, 0.9, 'Feature Neurons\n$\\Sigma w_i^F \\cdot F_i$\n(top-k=20)',
             facecolor=COLOR_LIGHT_BLUE, edgecolor=COLOR_FEATURE, fontsize=7)

    # Arrow
    draw_arrow(ax, (3.8, y_main + 0.35), (4.5, y_main + 0.35), color=COLOR_BLACK)

    # Hidden h
    draw_box(ax, 4.5, y_main, 1.0, 0.7, 'h\n(r=64)',
             facecolor='#FFF3E0', edgecolor='#F57C00', fontsize=8)

    # Arrow
    draw_arrow(ax, (5.5, y_main + 0.35), (6.2, y_main + 0.35), color=COLOR_BLACK)

    # Restore Neurons
    draw_box(ax, 6.2, y_main - 0.1, 1.8, 0.9, 'Restore Neurons\n$\\Sigma w_i^R \\cdot R_i$\n(top-k=20)',
             facecolor=COLOR_LIGHT_GREEN, edgecolor=COLOR_RESTORE, fontsize=7)

    # Arrow
    draw_arrow(ax, (8.0, y_main + 0.35), (8.7, y_main + 0.35), color=COLOR_BLACK)

    # Output
    draw_box(ax, 8.7, y_main, 1.0, 0.7, 'out\n(d=384)',
             facecolor='#F5F5F5', edgecolor=COLOR_BLACK, fontsize=8)

    # === Middle: Equations ===
    y_eq = 3.0
    ax.text(5.0, y_eq, r'$\mathbf{h} = \mathbf{x} \cdot \left(\sum_{i \in \text{top-k}} w_i^F \cdot \mathbf{F}_i\right)$',
            fontsize=10, ha='center', va='center', color=COLOR_FEATURE)
    ax.text(5.0, y_eq - 0.5, r'$\mathbf{out} = \mathbf{h} \cdot \left(\sum_{i \in \text{top-k}} w_i^R \cdot \mathbf{R}_i\right)$',
            fontsize=10, ha='center', va='center', color=COLOR_RESTORE)

    # === Bottom: Weight Visualization ===
    y_weights = 1.2

    # Router box
    draw_box(ax, 0.5, y_weights - 0.3, 1.5, 0.7, 'Router\n(from x)',
             facecolor='#F3E5F5', edgecolor='#9B59B6', fontsize=8)

    # Arrows to weights
    ax.annotate('', xy=(2.5, y_weights + 0.3), xytext=(2.0, y_weights + 0.05),
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.2))
    ax.annotate('', xy=(6.5, y_weights + 0.3), xytext=(2.0, y_weights + 0.05),
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.2))

    # Feature weights bar chart
    ax.text(3.5, y_weights + 1.0, '$w^F$ (Feature weights)', fontsize=8, ha='center', color=COLOR_FEATURE)

    # Sparse weights visualization
    np.random.seed(42)
    n_neurons = 25
    feature_weights = np.zeros(n_neurons)
    active_idx = [3, 7, 11, 15, 19]  # 5 active out of 25 (representing top-k sparsity)
    feature_weights[active_idx] = np.random.uniform(0.15, 0.25, len(active_idx))

    bar_width = 0.12
    bar_x = 2.3
    for i, w in enumerate(feature_weights):
        color = COLOR_FEATURE if w > 0 else '#E0E0E0'
        alpha = 1.0 if w > 0 else 0.3
        rect = Rectangle((bar_x + i * bar_width, y_weights + 0.35),
                         bar_width * 0.8, w * 2.5,
                         facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(rect)

    # Restore weights bar chart
    ax.text(7.5, y_weights + 1.0, '$w^R$ (Restore weights)', fontsize=8, ha='center', color=COLOR_RESTORE)

    restore_weights = np.zeros(n_neurons)
    active_idx_r = [2, 6, 10, 14, 22]  # Different active neurons
    restore_weights[active_idx_r] = np.random.uniform(0.15, 0.25, len(active_idx_r))

    bar_x = 6.3
    for i, w in enumerate(restore_weights):
        color = COLOR_RESTORE if w > 0 else '#E0E0E0'
        alpha = 1.0 if w > 0 else 0.3
        rect = Rectangle((bar_x + i * bar_width, y_weights + 0.35),
                         bar_width * 0.8, w * 2.5,
                         facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(rect)

    # Labels
    ax.text(3.5, y_weights + 0.15, 'neuron index', fontsize=6, ha='center', color=COLOR_GRAY)
    ax.text(7.5, y_weights + 0.15, 'neuron index', fontsize=6, ha='center', color=COLOR_GRAY)

    # Annotation
    ax.text(5.0, 0.3, 'Router independently selects top-k neurons for Feature and Restore paths\n(sparse: most weights = 0, only k non-zero)',
            fontsize=7, ha='center', va='center', color=COLOR_GRAY, style='italic')

    plt.tight_layout()
    plt.savefig('figures/fig2_feature_restore_pathway.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2_feature_restore_pathway.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig2_feature_restore_pathway.pdf")
    plt.close()

if __name__ == '__main__':
    main()

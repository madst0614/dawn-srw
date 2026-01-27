"""
Shared Style Configuration for Paper Figures
=============================================

Centralized font sizes and style settings for all visualizers.
Import PAPER_STYLE and use apply_paper_style() to set consistent rcParams.
"""

# Paper-mode font sizes (unified across all figures)
PAPER_STYLE = {
    # Base font
    'font_family': 'serif',
    'font_size_base': 10,

    # Axis labels (xlabel, ylabel)
    'font_size_label': 14,
    # Subplot / panel titles
    'font_size_subtitle': 14,
    # Tick labels
    'font_size_tick': 11,
    # Legend text
    'font_size_legend': 10,
    # Annotation / cell text (heatmap cell values, bar labels)
    'font_size_annotation': 12,
    # Category / group labels (e.g. "Shared", "Capital-specific" above heatmap)
    'font_size_category': 14,

    # Line widths
    'axes_linewidth': 0.8,
    'spines_top': False,
    'spines_right': False,
}


def apply_paper_style(plt_module=None):
    """
    Apply unified paper style to matplotlib rcParams.

    Args:
        plt_module: matplotlib.pyplot module. If None, imports it.
    """
    if plt_module is None:
        try:
            import matplotlib.pyplot as plt_module
        except ImportError:
            return

    s = PAPER_STYLE
    plt_module.rcParams.update({
        'font.family': s['font_family'],
        'font.size': s['font_size_base'],
        'axes.linewidth': s['axes_linewidth'],
        'axes.spines.top': s['spines_top'],
        'axes.spines.right': s['spines_right'],
        'axes.labelsize': s['font_size_label'],
        'xtick.labelsize': s['font_size_tick'],
        'ytick.labelsize': s['font_size_tick'],
    })

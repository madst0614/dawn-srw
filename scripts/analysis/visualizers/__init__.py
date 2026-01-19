"""
DAWN Visualizers
================
Visualization functions for DAWN analysis results.

Each module corresponds to a specific paper figure or analysis type.
"""

from .qk_specialization import plot_qk_specialization, plot_qk_usage
from .neuron_health import plot_dead_neurons, plot_usage_histogram, plot_qk_ema_overlap
from .embedding import plot_similarity_heatmap, plot_clustering, plot_embedding_space
from .pos_neurons import (
    plot_pos_heatmap, plot_pos_clustering,
    plot_top_neurons_by_pos, plot_pos_specificity,
    plot_pos_specialization_from_features
)
from .layer_contribution import plot_layer_contribution
from .factual_heatmap import plot_factual_heatmap, plot_factual_comparison

__all__ = [
    # Q/K Specialization (Figure 3)
    'plot_qk_specialization',
    'plot_qk_usage',
    # POS Neurons (Figure 4)
    'plot_pos_heatmap',
    'plot_pos_clustering',
    'plot_top_neurons_by_pos',
    'plot_pos_specificity',
    'plot_pos_specialization_from_features',
    # Neuron Health (Figure 6a)
    'plot_dead_neurons',
    'plot_usage_histogram',
    'plot_qk_ema_overlap',
    # Layer Contribution (Figure 7)
    'plot_layer_contribution',
    # Factual Knowledge (Figure 7)
    'plot_factual_heatmap',
    'plot_factual_comparison',
    # Embedding Structure
    'plot_similarity_heatmap',
    'plot_clustering',
    'plot_embedding_space',
]

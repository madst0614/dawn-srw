"""
Standalone Analysis Scripts
============================
CLI tools for specific analysis tasks.

Scripts:
    routing_analysis.py - Token-level routing analysis during generation
    compare_factual_neurons.py - Compare neurons across factual prompts

Usage:
    python scripts/analysis/standalone/routing_analysis.py \
        --checkpoint path/to/checkpoint \
        --prompt "The capital of France is" \
        --output routing_analysis/

    python scripts/analysis/standalone/compare_factual_neurons.py \
        --input factual_results/ \
        --output comparison/
"""

from .routing_analysis import (
    GenerationRoutingAnalyzer,
    analyze_common_neurons,
    analyze_token_neurons,
    plot_routing_heatmap,
    plot_routing_comparison,
)

__all__ = [
    'GenerationRoutingAnalyzer',
    'analyze_common_neurons',
    'analyze_token_neurons',
    'plot_routing_heatmap',
    'plot_routing_comparison',
]

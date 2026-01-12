"""
DAWN Analysis Package
======================
Comprehensive analysis toolkit for DAWN models.

This package provides modular analysis tools for:
- Neuron health and utilization
- Routing patterns and entropy
- Embedding analysis and visualization
- Weight matrix analysis
- Behavioral analysis
- Semantic analysis
- Co-selection pattern analysis
- POS neuron specialization
- Paper figure generation

Usage:
    # Complete one-touch analysis
    python scripts/analysis/analyze_all.py --checkpoint dawn.pt --val_data val.pt --output results/

    # Multi-model comparison
    python scripts/analysis/analyze_all.py --checkpoints dawn.pt vanilla.pt --val_data val.pt --output results/

    # Paper figures only
    from scripts.analysis import PaperFigureGenerator
    gen = PaperFigureGenerator('checkpoint.pt', dataloader)
    gen.generate('all', './paper_figures')

    # Individual analyzers
    from scripts.analysis import load_model, NeuronHealthAnalyzer
    model, tokenizer, config = load_model('checkpoint.pt')
    health = NeuronHealthAnalyzer(model)
    results = health.run_all('./health_output')

    # Programmatic complete analysis
    from scripts.analysis import ModelAnalyzer
    analyzer = ModelAnalyzer('checkpoint.pt', 'val.pt', 'output/')
    analyzer.run_all()
"""

from .utils import (
    # Model loading
    load_model,
    get_router,
    get_neurons,
    get_shared_neurons,  # Alias
    create_dataloader,

    # Constants
    NEURON_TYPES,
    NEURON_TYPES_V18,
    ROUTING_KEYS,
    KNOWLEDGE_ROUTING_KEYS,
    ALL_ROUTING_KEYS,
    NEURON_ATTRS,
    COSELECTION_PAIRS,
    QK_POOLS,

    # Utilities
    gini_coefficient,
    calc_entropy,
    calc_entropy_ratio,
    convert_to_serializable,
    save_results,
    simple_pos_tag,

    # Flags
    HAS_MATPLOTLIB,
    HAS_SKLEARN,
    HAS_TQDM,
)

# Base analyzer
from .base import BaseAnalyzer

# Analyzers
from .neuron_health import NeuronHealthAnalyzer
from .routing import RoutingAnalyzer
from .embedding import EmbeddingAnalyzer
from .weight import WeightAnalyzer
from .behavioral import BehavioralAnalyzer
from .semantic import SemanticAnalyzer
from .coselection import CoselectionAnalyzer
from .paper_figures import PaperFigureGenerator

# Visualizers
from . import visualizers
from .standalone.routing_analysis import (
    GenerationRoutingAnalyzer,
    analyze_common_neurons,
    analyze_token_neurons,
    plot_routing_heatmap,
    plot_routing_comparison,
)
# POS Neuron Analyzer (refactored)
from .pos_neuron import POSNeuronAnalyzer

# V18.x Specific Analyzer
from .v18 import V18Analyzer

# Complete analysis tool
from .analyze_all import ModelAnalyzer, MultiModelAnalyzer

# Legacy imports from visualizers for backward compatibility
from .visualizers.pos_neurons import (
    plot_pos_heatmap,
    plot_pos_clustering,
    plot_top_neurons_by_pos,
    plot_pos_specificity as plot_specificity,
)


__all__ = [
    # Model loading
    'load_model',
    'get_router',
    'get_neurons',
    'get_shared_neurons',
    'create_dataloader',

    # Constants
    'NEURON_TYPES',
    'ROUTING_KEYS',
    'KNOWLEDGE_ROUTING_KEYS',
    'ALL_ROUTING_KEYS',
    'NEURON_ATTRS',
    'COSELECTION_PAIRS',
    'QK_POOLS',

    # Utilities
    'gini_coefficient',
    'calc_entropy',
    'calc_entropy_ratio',
    'convert_to_serializable',
    'save_results',
    'simple_pos_tag',

    # Flags
    'HAS_MATPLOTLIB',
    'HAS_SKLEARN',
    'HAS_TQDM',

    # Base
    'BaseAnalyzer',
    'visualizers',

    # Analyzers
    'NeuronHealthAnalyzer',
    'RoutingAnalyzer',
    'EmbeddingAnalyzer',
    'WeightAnalyzer',
    'BehavioralAnalyzer',
    'SemanticAnalyzer',
    'CoselectionAnalyzer',
    'PaperFigureGenerator',

    # Routing analysis
    'GenerationRoutingAnalyzer',
    'analyze_common_neurons',
    'analyze_token_neurons',
    'plot_routing_heatmap',
    'plot_routing_comparison',

    # POS neuron analysis
    'POSNeuronAnalyzer',
    'plot_pos_heatmap',
    'plot_pos_clustering',
    'plot_top_neurons_by_pos',
    'plot_specificity',

    # V18.x analysis
    'V18Analyzer',

    # Complete analysis
    'ModelAnalyzer',
    'MultiModelAnalyzer',
]

__version__ = '1.0.0'

"""
Paper Figure Generator
=======================
Generate publication-ready figures for DAWN paper.

READ-ONLY: This module only reads precomputed data and generates visualizations.
All analysis must be run via analyze_all.py first.
"""

import os
import json
from typing import Dict, Optional, List

from .utils import HAS_MATPLOTLIB, convert_to_serializable

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt


class PaperFigureGenerator:
    """Generate paper-ready figures from precomputed DAWN analysis.

    IMPORTANT: This class does NOT run any analysis.
    All data must be precomputed via analyze_all.py and passed via precomputed dict.

    ICML 2026 Paper Figure Mapping:
        Fig 3: Q/K Pathway Separation
        Fig 4: POS Specialization
        Fig 5: Semantic Coherence (Factual Heatmap)
        Fig 6: Training Dynamics (Appendix)
        Fig 7: Neuron Utilization (Appendix)
    """

    FIGURE_MAP = {
        '3': 'generate_figure3',   # Q/K Specialization
        '4': 'generate_figure4',   # POS Neurons
        '5': 'generate_figure5',   # Semantic Coherence (Factual Heatmap)
        '6': 'generate_figure6',   # Training Dynamics
        '7': 'generate_figure7',   # Layer Contribution
    }

    def __init__(self, checkpoint_path: str = None, val_data_path: str = None,
                 device: str = 'cuda'):
        """
        Initialize generator.

        Args:
            checkpoint_path: Path to model checkpoint (for metadata only)
            val_data_path: Not used (kept for backward compatibility)
            device: Not used (kept for backward compatibility)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device

    def generate(self, figures: str, output_dir: str, n_batches: int = 50,
                 precomputed: Optional[Dict] = None, config: Optional[Dict] = None) -> Dict:
        """
        Generate specified figures from precomputed data.

        Args:
            figures: Comma-separated figure numbers (e.g., "3,4,5") or "all"
            output_dir: Directory for output figures
            n_batches: Not used (kept for backward compatibility)
            precomputed: Pre-computed analysis results (REQUIRED)
            config: Additional configuration options

        Returns:
            Dict with results for each figure
        """
        os.makedirs(output_dir, exist_ok=True)
        precomputed = precomputed or {}
        config = config or {}

        # Parse figure list
        if figures == 'all':
            fig_nums = list(self.FIGURE_MAP.keys())
        else:
            fig_nums = [f.strip() for f in figures.split(',')]

        results = {}
        for fig in fig_nums:
            if fig in self.FIGURE_MAP:
                method_name = self.FIGURE_MAP[fig]
                method = getattr(self, method_name)
                print(f"\n[Figure {fig}]", flush=True)
                try:
                    results[f'figure_{fig}'] = method(output_dir, precomputed, config)
                except Exception as e:
                    print(f"  Error: {e}", flush=True)
                    results[f'figure_{fig}'] = {'error': str(e)}

        return results

    def generate_figure3(self, output_dir: str, precomputed: Dict, config: Dict) -> Dict:
        """
        Figure 3: Q/K Specialization.

        Requires: precomputed['routing']['qk_usage']
        """
        from .visualizers import plot_qk_specialization

        routing = precomputed.get('routing', {})
        if 'qk_usage' not in routing:
            return {'error': 'Missing precomputed data: routing.qk_usage. Run analyze_all.py --only routing first.'}

        qk_data = routing['qk_usage']
        print("  Using pre-computed Q/K usage data...", flush=True)

        path = plot_qk_specialization(qk_data, os.path.join(output_dir, 'fig3_qk_specialization.png'))
        print(f"  Saved: {path}", flush=True)

        return {'qk_usage': qk_data, 'visualization': path}

    def generate_figure4(self, output_dir: str, precomputed: Dict, config: Dict) -> Dict:
        """
        Figure 4: POS Neuron Selectivity.

        Uses selectivity data (not 80% threshold specialization).
        Requires: selectivity_matrix.npy file and precomputed['neuron_features']['selectivity']
        """
        from .visualizers import plot_pos_selectivity_heatmap
        import numpy as np

        if 'neuron_features' not in precomputed:
            return {'error': 'Missing precomputed data: neuron_features. Run analyze_all.py --only neuron_features first.'}

        results = precomputed['neuron_features']
        selectivity = results.get('selectivity', {})

        if not selectivity:
            return {'error': 'Missing selectivity data in neuron_features. Re-run analyze_all.py --only neuron_features.'}

        print("  Using pre-computed selectivity data...", flush=True)

        # Find selectivity_matrix.npy (required)
        matrix_path = config.get('selectivity_matrix_path')

        if not matrix_path or not os.path.exists(matrix_path):
            analysis_output_dir = config.get('analysis_output_dir', '')
            possible_paths = [
                os.path.join(analysis_output_dir, 'neuron_features', 'selectivity_matrix.npy'),
                os.path.join(output_dir, '..', 'neuron_features', 'selectivity_matrix.npy'),
                os.path.join(output_dir, 'selectivity_matrix.npy'),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    matrix_path = p
                    break

        if not matrix_path or not os.path.exists(matrix_path):
            return {'error': 'selectivity_matrix.npy not found. Run analyze_all.py --only neuron_features first.'}

        print(f"  Loading selectivity matrix from {matrix_path}...", flush=True)
        selectivity_matrix = np.load(matrix_path)

        # Get active indices and pool_order from selectivity data
        active_indices = selectivity.get('active_neuron_indices', list(range(selectivity_matrix.shape[0])))
        pool_order = selectivity.get('pool_order', config.get('pool_order'))

        # Fallback: default pool_order for 40M DAWN model if not provided
        if not pool_order:
            pool_order = [
                ('fqk', 64),
                ('fv', 264),
                ('rqk', 64),
                ('rv', 264),
                ('fknow', 160),
                ('rknow', 160),
            ]

        path = plot_pos_selectivity_heatmap(
            selectivity_matrix,
            active_indices,
            os.path.join(output_dir, 'fig4_pos_selectivity.png'),
            pool_order=pool_order
        )

        if path:
            print(f"  Saved: {path}", flush=True)
        else:
            print("  Warning: Figure generation returned None", flush=True)

        return {'selectivity': selectivity, 'visualization': path}

    def generate_figure5(self, output_dir: str, precomputed: Dict, config: Dict) -> Dict:
        """
        Figure 5: Semantic Coherence (Factual Knowledge Heatmap).

        Requires: precomputed['factual']['per_target']
        """
        from .visualizers import plot_factual_heatmap

        factual = precomputed.get('factual', {})
        if 'per_target' not in factual:
            return {'error': 'Missing precomputed data: factual.per_target. Run analyze_all.py --only factual first.'}

        print("  Using pre-computed factual analysis...", flush=True)

        path = plot_factual_heatmap(factual, os.path.join(output_dir, 'fig5_semantic_coherence.png'))
        print(f"  Saved: {path}", flush=True)

        return {'factual_analysis': factual, 'visualization': path}

    def generate_figure6(self, output_dir: str, precomputed: Dict, config: Dict) -> Dict:
        """
        Figure 6: Training Dynamics (Appendix).

        Uses training logs from checkpoint paths in config.
        """
        from .visualizers import (
            plot_training_dynamics,
            find_training_log, parse_training_log
        )

        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib required'}

        print("  Generating training dynamics plot...", flush=True)

        data = {}

        # Method 1: Explicit training logs from config
        training_logs = config.get('training_logs', [])
        training_labels = config.get('training_labels', [])

        if training_logs:
            for i, log_path in enumerate(training_logs):
                if os.path.exists(log_path):
                    steps, losses, meta = parse_training_log(log_path, use_val_loss=True)
                    if steps:
                        label = training_labels[i] if i < len(training_labels) else f'Model-{i+1}'
                        data[label] = (steps, losses)
                        print(f"    Loaded {label}: {meta['n_points']} points")

        # Method 2: Checkpoint paths from config
        if not data:
            checkpoint_paths = config.get('checkpoint_paths', [])
            checkpoint_labels = config.get('checkpoint_labels', [])

            for i, ckpt_path in enumerate(checkpoint_paths):
                log_path = find_training_log(ckpt_path)
                if log_path:
                    steps, losses, meta = parse_training_log(log_path, use_val_loss=True)
                    if steps:
                        if i < len(checkpoint_labels):
                            label = checkpoint_labels[i]
                        else:
                            from pathlib import Path
                            p = Path(ckpt_path)
                            name = p.name if p.is_file() else p.name
                            if 'v17' in name.lower() or 'v18' in name.lower() or 'dawn' in name.lower():
                                label = 'DAWN'
                            elif 'baseline' in name.lower() or 'vanilla' in name.lower():
                                label = 'Vanilla'
                            else:
                                label = name[:20]
                        data[label] = (steps, losses)
                        print(f"    Loaded {label}: {meta['n_points']} points from {log_path}")

        # Method 3: Auto-find from self.checkpoint_path
        if not data and self.checkpoint_path:
            log_path = find_training_log(self.checkpoint_path)
            if log_path:
                steps, losses, meta = parse_training_log(log_path, use_val_loss=True)
                if steps:
                    data['DAWN'] = (steps, losses)
                    print(f"    Loaded DAWN: {meta['n_points']} points")

        if not data:
            return {'error': 'No training logs found. Check checkpoint paths.'}

        path = plot_training_dynamics(data, os.path.join(output_dir, 'fig6_training_dynamics.png'))
        print(f"  Saved: {path}", flush=True)

        return {'visualization': path, 'data': {k: len(v[0]) for k, v in data.items()}}

    def generate_figure7(self, output_dir: str, precomputed: Dict, config: Dict) -> Dict:
        """
        Figure 7: Routing Statistics (Appendix).

        Requires: precomputed['routing']['layer_contribution'] and precomputed['routing']['qk_usage']
        """
        from .visualizers import plot_routing_stats
        from .utils import get_router, load_model

        routing = precomputed.get('routing', {})

        if 'layer_contribution' not in routing:
            return {'error': 'Missing precomputed data: routing.layer_contribution. Run analyze_all.py --only routing first.'}

        contrib_data = routing['layer_contribution']
        qk_usage = routing.get('qk_usage', {})

        print("  Using pre-computed routing data...", flush=True)

        combined_data = {
            'layer_contribution': contrib_data,
            'qk_usage': qk_usage,
        }

        # Try to get router for pool sizes (optional)
        router = None
        if self.checkpoint_path:
            try:
                model, _, _ = load_model(self.checkpoint_path, self.device)
                router = get_router(model)
                del model
            except:
                pass

        path = plot_routing_stats(
            combined_data,
            os.path.join(output_dir, 'fig7_routing_stats.png'),
            router=router
        )
        print(f"  Saved: {path}", flush=True)

        return {'layer_contribution': contrib_data, 'visualization': path}


def main():
    """CLI interface - requires precomputed data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate DAWN paper figures from precomputed data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This tool requires precomputed analysis data.
Run analyze_all.py first to generate the required data.

Examples:
  # First run analysis
  python -m scripts.analysis.analyze_all --checkpoint model.pt --only routing,factual,neuron_features

  # Then generate figures
  python -m scripts.analysis.paper_figures --checkpoint model.pt --precomputed results/paper_data.json
"""
    )

    parser.add_argument('--checkpoint', help='Path to model checkpoint')
    parser.add_argument('--precomputed', required=True, help='Path to precomputed results JSON')
    parser.add_argument('--output', default='./paper_figures', help='Output directory')
    parser.add_argument('--figures', default='all',
                       help='Figures to generate: "all" or comma-separated (e.g., "3,5,7")')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load precomputed data
    with open(args.precomputed) as f:
        precomputed = json.load(f)

    gen = PaperFigureGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    results = gen.generate(
        args.figures,
        args.output,
        precomputed=precomputed
    )

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    for fig, result in results.items():
        status = "✓" if 'error' not in result else f"✗ {result['error']}"
        print(f"  {fig}: {status}")


if __name__ == '__main__':
    main()

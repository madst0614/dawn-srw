"""
V18.x Specific Analyzer
=======================
Analysis tools for DAWN v18.x specific features:
- Learnable tau parameters
- Gate distribution
- Q/K tau differentiation
- Confidence statistics (v18.3+)
- Runtime gate/tau analysis
"""

import os
import json
from typing import Dict, Optional, List
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseAnalyzer
from .utils import save_results, HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt


class V18Analyzer(BaseAnalyzer):
    """Analyzer for DAWN v18.x specific features."""

    def __init__(self, model, device: str = 'cuda'):
        super().__init__(model, device)
        self.model_version = getattr(model, '__version__', '18.2')

        # Detect v18 structure
        self.has_old_tau = hasattr(model, 'router') and hasattr(model.router, 'tau_proj')
        self.has_new_tau = hasattr(model, 'router') and hasattr(model.router, 'tau_proj_feature')
        self.is_v18 = self.has_old_tau or self.has_new_tau

    def analyze_tau_parameters(self) -> Dict:
        """Analyze static tau parameters (bias, weight norm, weight std)."""
        if not self.is_v18:
            return {'error': 'Not a v18.x model with learnable tau'}

        results = {'model_version': self.model_version}

        if self.has_new_tau:
            # v18.4/v18.5 structure
            tau_proj_feature = self.model.router.tau_proj_feature
            feature_names = ['fq', 'fk', 'fv', 'feature_know']

            weight_f = tau_proj_feature.weight.detach().cpu()
            bias_f = tau_proj_feature.bias.detach().cpu()

            tau_params = {
                'structure': 'v18.4+',
                'tau_bias': {name: bias_f[i].item() for i, name in enumerate(feature_names)},
                'tau_weight_norm': {name: weight_f[i].norm().item() for i, name in enumerate(feature_names)},
                'tau_weight_std': {name: weight_f[i].std().item() for i, name in enumerate(feature_names)},
            }

            # Q/K Differentiation (feature only)
            qk_diff = {
                'fqk_bias_diff': abs(bias_f[0].item() - bias_f[1].item()),
                'fqk_weight_cosine': F.cosine_similarity(weight_f[0:1], weight_f[1:2]).item(),
            }

            # v18.5: context-based restore tau
            if hasattr(self.model.router, 'tau_proj_restore_Q'):
                restore_projs = {
                    'restore_Q': self.model.router.tau_proj_restore_Q,
                    'restore_K': self.model.router.tau_proj_restore_K,
                    'restore_v': self.model.router.tau_proj_restore_v,
                    'restore_know': self.model.router.tau_proj_restore_know,
                }
                for name, proj in restore_projs.items():
                    w = proj.weight.detach().cpu()
                    b = proj.bias.detach().cpu().item()
                    tau_params['tau_bias'][name] = b
                    tau_params['tau_weight_norm'][name] = w.norm().item()
                    tau_params['tau_weight_std'][name] = w.std().item()

            tau_params['qk_differentiation'] = qk_diff

        else:
            # v18.0-18.3 structure
            tau_proj = self.model.router.tau_proj
            pool_names = ['fq', 'fk', 'fv', 'rq', 'rk', 'rv', 'feature_know', 'restore_know']

            weight = tau_proj.weight.detach().cpu()
            bias = tau_proj.bias.detach().cpu()

            tau_params = {
                'structure': 'v18.0-18.3',
                'tau_bias': {name: bias[i].item() for i, name in enumerate(pool_names)},
                'tau_weight_norm': {name: weight[i].norm().item() for i, name in enumerate(pool_names)},
                'tau_weight_std': {name: weight[i].std().item() for i, name in enumerate(pool_names)},
            }

            # Q/K Differentiation
            qk_diff = {
                'fqk_bias_diff': abs(bias[0].item() - bias[1].item()),
                'rqk_bias_diff': abs(bias[3].item() - bias[4].item()),
                'fqk_weight_cosine': F.cosine_similarity(weight[0:1], weight[1:2]).item(),
                'rqk_weight_cosine': F.cosine_similarity(weight[3:4], weight[4:5]).item(),
            }
            tau_params['qk_differentiation'] = qk_diff

        results['tau_parameters'] = tau_params
        return results

    def analyze_runtime(self, dataloader, n_batches: int = 50) -> Dict:
        """Analyze runtime gate/tau statistics from forward passes."""
        if not self.is_v18:
            return {'error': 'Not a v18.x model'}

        gate_stats = defaultdict(list)
        tau_runtime = defaultdict(list)
        qk_patterns = defaultdict(list)
        conf_stats = defaultdict(list)

        self.model.eval()

        # Enable debug_mode if available
        if hasattr(self.model.router, 'debug_mode'):
            self.model.router.debug_mode = True

        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= n_batches:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                outputs = self.model(input_ids, return_routing_info=True)

                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    routing_infos = outputs[-1]

                    for layer_idx, layer_info in enumerate(routing_infos):
                        attn = layer_info.get('attention', layer_info)
                        know = layer_info.get('knowledge', {})

                        # Gate statistics - v18.4/v18.5 use gstr_* keys
                        for pool in ['fq', 'fk', 'fv']:
                            gstr_key = f'gstr_{pool}'
                            old_key = f'gate_{pool}_mean'
                            if gstr_key in attn:
                                gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[gstr_key])
                            elif old_key in attn:
                                gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[old_key])

                        # v18.5 restore gate_strength
                        for key, pool in [('gstr_rqk_Q', 'rq'), ('gstr_rqk_K', 'rk'), ('gstr_rv', 'rv')]:
                            if key in attn:
                                gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[key])

                        # Old-style restore gates
                        for pool in ['rq', 'rk', 'rv']:
                            old_key = f'gate_{pool}_mean'
                            if old_key in attn and f'L{layer_idx}_{pool}_gate' not in gate_stats:
                                gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[old_key])

                        # Tau statistics
                        for pool in ['fq', 'fk', 'fv']:
                            new_key = f'tau_offset_{pool}'
                            old_key = f'tau_{pool}'
                            if new_key in attn:
                                tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[new_key])
                            elif old_key in attn:
                                tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[old_key])

                        # v18.5 restore tau_offset
                        for key, pool in [('tau_offset_rqk_Q', 'rq'), ('tau_offset_rqk_K', 'rk'), ('tau_offset_rv', 'rv')]:
                            if key in attn:
                                tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[key])

                        # Knowledge tau
                        if 'tau_offset_feature' in know:
                            tau_runtime[f'L{layer_idx}_kf_tau'].append(know['tau_offset_feature'])
                        if 'tau_offset_restore' in know:
                            tau_runtime[f'L{layer_idx}_kr_tau'].append(know['tau_offset_restore'])

                        # Knowledge gate_strength
                        if 'gstr_feature' in know:
                            gate_stats[f'L{layer_idx}_kf_gate'].append(know['gstr_feature'])
                        if 'gstr_restore' in know:
                            gate_stats[f'L{layer_idx}_kr_gate'].append(know['gstr_restore'])

                        # Q/K patterns
                        if 'overlap_fqk' in attn:
                            qk_patterns[f'L{layer_idx}_fqk_overlap'].append(attn['overlap_fqk'])
                        if 'overlap_rqk' in attn:
                            qk_patterns[f'L{layer_idx}_rqk_overlap'].append(attn['overlap_rqk'])

                        # Confidence statistics (v18.3+)
                        for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                            conf_key = f'conf_{pool}_mean'
                            if conf_key in attn:
                                conf_stats[f'L{layer_idx}_{pool}_conf'].append(attn[conf_key])

                batch_count += 1

        # Disable debug_mode
        if hasattr(self.model.router, 'debug_mode'):
            self.model.router.debug_mode = False

        # Aggregate statistics
        results = {
            'n_batches': batch_count,
            'gate_distribution': {},
            'tau_runtime': {},
            'qk_patterns': {},
            'confidence': {},
        }

        for key, values in gate_stats.items():
            if values:
                results['gate_distribution'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        for key, values in tau_runtime.items():
            if values:
                results['tau_runtime'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        for key, values in qk_patterns.items():
            if values:
                results['qk_patterns'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        for key, values in conf_stats.items():
            if values:
                results['confidence'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        return results

    def get_per_layer_summary(self, runtime_results: Dict) -> Dict:
        """Extract per-layer summary from runtime results."""
        n_layers = getattr(self.model, 'n_layers', 12)

        gate_summary = {}
        tau_summary = {}
        qk_summary = {}
        conf_summary = {}

        gate_dist = runtime_results.get('gate_distribution', {})
        tau_rt = runtime_results.get('tau_runtime', {})
        qk_pat = runtime_results.get('qk_patterns', {})
        conf = runtime_results.get('confidence', {})

        for layer_idx in range(n_layers):
            # Gate per layer
            gate_summary[f'L{layer_idx}'] = {}
            for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                key = f'L{layer_idx}_{pool}_gate'
                if key in gate_dist:
                    gate_summary[f'L{layer_idx}'][pool] = gate_dist[key]['mean']

            # Tau per layer
            tau_summary[f'L{layer_idx}'] = {}
            for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                key = f'L{layer_idx}_{pool}_tau'
                if key in tau_rt:
                    tau_summary[f'L{layer_idx}'][pool] = tau_rt[key]['mean']

            # Q/K overlap per layer
            fqk_key = f'L{layer_idx}_fqk_overlap'
            rqk_key = f'L{layer_idx}_rqk_overlap'
            qk_summary[f'L{layer_idx}'] = {
                'fqk': qk_pat.get(fqk_key, {}).get('mean', 0),
                'rqk': qk_pat.get(rqk_key, {}).get('mean', 0),
            }

            # Confidence per layer
            conf_summary[f'L{layer_idx}'] = {}
            for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                key = f'L{layer_idx}_{pool}_conf'
                if key in conf:
                    conf_summary[f'L{layer_idx}'][pool] = conf[key]['mean']

        return {
            'gate': gate_summary,
            'tau': tau_summary,
            'qk_overlap': qk_summary,
            'confidence': conf_summary,
        }

    def visualize_tau_params(self, results: Dict, output_path: str):
        """Visualize tau parameters."""
        if not HAS_MATPLOTLIB:
            return

        tau_params = results.get('tau_parameters', {})
        if not tau_params:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Tau bias
        bias = tau_params.get('tau_bias', {})
        pools = list(bias.keys())
        values = [bias[p] for p in pools]

        axes[0].bar(pools, values, color='steelblue')
        axes[0].set_title('Tau Bias per Pool')
        axes[0].set_ylabel('Bias')
        axes[0].tick_params(axis='x', rotation=45)

        # Tau weight norm
        norms = tau_params.get('tau_weight_norm', {})
        values = [norms.get(p, 0) for p in pools]

        axes[1].bar(pools, values, color='coral')
        axes[1].set_title('Tau Weight Norm per Pool')
        axes[1].set_ylabel('L2 Norm')
        axes[1].tick_params(axis='x', rotation=45)

        # Q/K differentiation
        qk_diff = tau_params.get('qk_differentiation', {})
        if qk_diff:
            diff_names = list(qk_diff.keys())
            diff_values = [qk_diff[k] for k in diff_names]

            axes[2].bar(diff_names, diff_values, color='mediumseagreen')
            axes[2].set_title('Q/K Differentiation')
            axes[2].set_ylabel('Value')
            axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_gate_heatmap(self, per_layer: Dict, output_path: str):
        """Visualize per-layer gate values as heatmap."""
        if not HAS_MATPLOTLIB:
            return

        gate_data = per_layer.get('gate', {})
        if not gate_data:
            return

        layers = sorted(gate_data.keys(), key=lambda x: int(x[1:]))
        pools = ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']

        matrix = []
        for layer in layers:
            row = [gate_data[layer].get(pool, 0) for pool in pools]
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(pools)))
        ax.set_xticklabels(pools)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)

        ax.set_xlabel('Pool')
        ax.set_ylabel('Layer')
        ax.set_title('Gate Strength per Layer/Pool')

        plt.colorbar(im, ax=ax, label='Gate Mean')

        # Add values
        for i in range(len(layers)):
            for j in range(len(pools)):
                val = matrix[i, j]
                if val > 0:
                    color = 'white' if val > matrix.max() * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_qk_overlap(self, per_layer: Dict, output_path: str):
        """Visualize Q/K overlap per layer."""
        if not HAS_MATPLOTLIB:
            return

        qk_data = per_layer.get('qk_overlap', {})
        if not qk_data:
            return

        layers = sorted(qk_data.keys(), key=lambda x: int(x[1:]))
        fqk_values = [qk_data[l]['fqk'] for l in layers]
        rqk_values = [qk_data[l]['rqk'] for l in layers]

        x = range(len(layers))

        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.35

        ax.bar([i - width/2 for i in x], fqk_values, width, label='Feature Q/K', color='steelblue')
        ax.bar([i + width/2 for i in x], rqk_values, width, label='Restore Q/K', color='coral')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Overlap')
        ax.set_title('Q/K Selection Overlap per Layer')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def run_all(self, dataloader=None, output_dir: str = None, n_batches: int = 50) -> Dict:
        """Run all v18.x analyses."""
        results = {}

        # Static tau analysis (no dataloader needed)
        tau_results = self.analyze_tau_parameters()
        results.update(tau_results)

        # Runtime analysis (requires dataloader)
        if dataloader is not None:
            runtime_results = self.analyze_runtime(dataloader, n_batches)
            results.update(runtime_results)

            # Per-layer summary
            per_layer = self.get_per_layer_summary(runtime_results)
            results['per_layer'] = per_layer

        # Save and visualize
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            save_results(results, os.path.join(output_dir, 'v18_analysis.json'))

            # Visualizations
            if HAS_MATPLOTLIB:
                self.visualize_tau_params(results, os.path.join(output_dir, 'tau_parameters.png'))

                if 'per_layer' in results:
                    self.visualize_gate_heatmap(
                        results['per_layer'],
                        os.path.join(output_dir, 'gate_heatmap.png')
                    )
                    self.visualize_qk_overlap(
                        results['per_layer'],
                        os.path.join(output_dir, 'qk_overlap.png')
                    )

        return results

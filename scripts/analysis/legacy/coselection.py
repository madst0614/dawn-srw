"""
Co-selection Pattern Analysis
==============================
Analyze neuron co-selection patterns in DAWN v17.1 models.

This module analyzes how neurons from different pools are
selected together, revealing learned pairings between
Feature and Restore neurons.

Includes:
- Co-selection matrix analysis
- Specialization analysis
- Subspace diversity analysis
- Cross-pool alignment analysis
"""

import os
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base import BaseAnalyzer
from .utils import (
    COSELECTION_PAIRS,
    get_batch_input_ids, get_routing_from_outputs,
    HAS_MATPLOTLIB, HAS_TQDM, tqdm, plt
)


class CoselectionAnalyzer(BaseAnalyzer):
    """Co-selection pattern analyzer for DAWN."""

    def __init__(self, model, router=None, shared_neurons=None, device='cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance (auto-detected if None)
            shared_neurons: SharedNeurons instance (auto-detected if None)
            device: Device for computation
        """
        super().__init__(model, router=router, shared_neurons=shared_neurons, device=device)
        # Alias for backward compatibility
        self.shared = self.shared_neurons

    def analyze_coselection(self, dataloader, pair_key: str, n_batches: int = 50, layer_idx: int = None) -> Dict:
        """
        Analyze co-selection patterns between two neuron pools.

        Args:
            dataloader: DataLoader for input data
            pair_key: Key from COSELECTION_PAIRS (e.g., 'fqk_rqk')
            n_batches: Number of batches to process
            layer_idx: Specific layer index to analyze (None = aggregate all layers)

        Returns:
            Dictionary with co-selection matrix and statistics (per-layer if layer_idx is None)
        """
        pair_info = COSELECTION_PAIRS.get(pair_key)
        if pair_info is None:
            return {}

        pool_a = pair_info['pool_a']
        pool_b = pair_info['pool_b']

        n_a = getattr(self.router, pool_a['n_attr'], 0)
        n_b = getattr(self.router, pool_b['n_attr'], 0)

        if n_a == 0 or n_b == 0:
            return {'error': f'Empty pools: n_a={n_a}, n_b={n_b}'}

        print(f"\n  {pair_info['name']}: {pool_a['display']}({n_a}) x {pool_b['display']}({n_b})", flush=True)

        # Per-layer co-selection matrices
        layer_matrices = {}  # {layer_idx: (co_matrix, a_counts, b_counts)}
        total_samples = 0

        self.model.eval()
        self.enable_pref_tensors()
        try:
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc=pair_key, total=n_batches)):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)
                    B = input_ids.shape[0]
                    total_samples += B

                    try:
                        outputs = self.model(input_ids, return_routing_info=True)
                        routing_infos = get_routing_from_outputs(outputs)
                        if routing_infos is None:
                            continue
                    except Exception:
                        continue

                    # Process all layers (or specific layer)
                    for lidx, layer_info in enumerate(routing_infos):
                        if layer_idx is not None and lidx != layer_idx:
                            continue

                        if lidx not in layer_matrices:
                            layer_matrices[lidx] = (
                                torch.zeros(n_a, n_b, device=self.device),
                                torch.zeros(n_a, device=self.device),
                                torch.zeros(n_b, device=self.device)
                            )

                        co_matrix, a_counts, b_counts = layer_matrices[lidx]

                        # Get preferences from appropriate source
                        source_a = layer_info.get(pool_a.get('source', 'attention'), {})
                        source_b = layer_info.get(pool_b.get('source', 'attention'), {})
                        pref_a = source_a.get(pool_a['pref_key'])
                        pref_b = source_b.get(pool_b['pref_key'])

                        if pref_a is None or pref_b is None:
                            continue

                        # Handle different shapes: [B, S, N] or [B, N]
                        if pref_a.dim() == 3:
                            pref_a = pref_a.mean(dim=1)
                        if pref_b.dim() == 3:
                            pref_b = pref_b.mean(dim=1)

                        # Binary selection (above uniform threshold)
                        thresh_a = 1.0 / n_a
                        thresh_b = 1.0 / n_b

                        selected_a = (pref_a > thresh_a).float()
                        selected_b = (pref_b > thresh_b).float()

                        # Count individual selections
                        a_counts += selected_a.sum(dim=0)
                        b_counts += selected_b.sum(dim=0)

                        # Co-occurrence: vectorized batch outer product sum
                        # einsum('bi,bj->ij') = sum over batch of outer products
                        co_matrix += torch.einsum('bi,bj->ij', selected_a, selected_b)

                        # Store back
                        layer_matrices[lidx] = (co_matrix, a_counts, b_counts)
        finally:
            self.disable_pref_tensors()

        # Build results per layer
        results = {
            'pair_name': pair_info['name'],
            'total_samples': total_samples,
            'per_layer': {},
        }

        for lidx, (co_matrix, a_counts, b_counts) in layer_matrices.items():
            layer_result = self._analyze_coselection_matrix(
                co_matrix, a_counts, b_counts, n_a, n_b,
                pool_a['display'], pool_b['display']
            )
            layer_result['co_matrix'] = co_matrix.cpu().numpy().tolist()
            results['per_layer'][f'L{lidx}'] = layer_result

        # Aggregate summary across layers
        if layer_matrices:
            total_co = sum(m[0] for m in layer_matrices.values())
            total_a = sum(m[1] for m in layer_matrices.values())
            total_b = sum(m[2] for m in layer_matrices.values())

            results['aggregated'] = self._analyze_coselection_matrix(
                total_co, total_a, total_b, n_a, n_b,
                pool_a['display'], pool_b['display']
            )
            results['aggregated']['n_layers'] = len(layer_matrices)

            # Backward compatibility: put aggregated at top level
            for k, v in results['aggregated'].items():
                if k not in results:
                    results[k] = v

        return results

    def _analyze_coselection_matrix(self, co_matrix, a_counts, b_counts, n_a, n_b, name_a, name_b):
        """Analyze co-selection matrix statistics."""
        results = {}

        total = co_matrix.sum()
        if total == 0:
            return {'error': 'No co-selections found'}

        co_prob = co_matrix / total

        # Top pairs
        flat_co = co_matrix.view(-1)
        top_k = min(20, flat_co.numel())
        top_values, top_indices = torch.topk(flat_co, top_k)

        top_pairs = []
        for i in range(top_k):
            idx = top_indices[i].item()
            a_idx = idx // n_b
            b_idx = idx % n_b
            count = top_values[i].item()
            pct = count / total.item() * 100

            top_pairs.append({
                'a_idx': a_idx,
                'b_idx': b_idx,
                'a_name': f'{name_a}_{a_idx}',
                'b_name': f'{name_b}_{b_idx}',
                'count': int(count),
                'pct': pct
            })

        results['top_pairs'] = top_pairs

        # Concentration analysis
        cumsum = torch.cumsum(torch.sort(flat_co, descending=True)[0], dim=0)

        top10_pct = (cumsum[9] / total * 100).item() if len(cumsum) > 9 else 0
        top50_pct = (cumsum[49] / total * 100).item() if len(cumsum) > 49 else 0
        top100_pct = (cumsum[99] / total * 100).item() if len(cumsum) > 99 else 0

        # Entropy
        co_prob_flat = co_prob.view(-1)
        co_prob_flat = co_prob_flat[co_prob_flat > 0]
        entropy = -(co_prob_flat * co_prob_flat.log()).sum().item()
        max_entropy = np.log(n_a * n_b)
        norm_entropy = entropy / max_entropy

        results['concentration'] = {
            'top10_pct': top10_pct,
            'top50_pct': top50_pct,
            'top100_pct': top100_pct,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': norm_entropy,
        }

        # Per-A specialization
        a_specialization = []
        for a_idx in range(n_a):
            row = co_matrix[a_idx]
            if row.sum() > 0:
                top_b = row.argmax().item()
                top_b_pct = (row[top_b] / row.sum() * 100).item()
                a_specialization.append({
                    'a_idx': a_idx,
                    'top_b': top_b,
                    'top_b_pct': top_b_pct,
                    'total_count': int(row.sum().item())
                })

        a_specialization.sort(key=lambda x: x['top_b_pct'], reverse=True)
        results['a_specialization'] = a_specialization[:20]

        # Per-B specialization
        b_specialization = []
        for b_idx in range(n_b):
            col = co_matrix[:, b_idx]
            if col.sum() > 0:
                top_a = col.argmax().item()
                top_a_pct = (col[top_a] / col.sum() * 100).item()
                b_specialization.append({
                    'b_idx': b_idx,
                    'top_a': top_a,
                    'top_a_pct': top_a_pct,
                    'total_count': int(col.sum().item())
                })

        b_specialization.sort(key=lambda x: x['top_a_pct'], reverse=True)
        results['b_specialization'] = b_specialization[:20]

        return results

    def analyze_subspace_diversity(self, pair_key: str, max_neurons: int = 4096) -> Dict:
        """
        Analyze neuron subspace diversity within a pool.

        Measures how different the neurons are from each other.
        Uses sampling for large pools to avoid O(n²) memory issues.

        Args:
            pair_key: Key from COSELECTION_PAIRS
            max_neurons: Maximum neurons for full pairwise analysis (samples if larger)

        Returns:
            Diversity statistics per pool
        """
        pair_info = COSELECTION_PAIRS.get(pair_key)
        if pair_info is None or self.shared is None:
            return {}

        results = {}

        for pool in ['pool_a', 'pool_b']:
            pool_info = pair_info[pool]
            neuron_attr = pool_info['neuron_attr']

            neurons = getattr(self.shared, neuron_attr, None)
            if neurons is None:
                continue

            n_neurons = neurons.shape[0]
            display = pool_info['display']

            # Ensure neurons are on the correct device
            neurons = neurons.to(self.device)

            # Sample if too large to avoid O(n²) memory issues
            sampled = False
            if n_neurons > max_neurons:
                print(f"\n  {display} Subspace Diversity (sampling {max_neurons}/{n_neurons} neurons)", flush=True)
                indices = torch.randperm(n_neurons, device=self.device)[:max_neurons]
                neurons_sample = neurons[indices]
                n_sample = max_neurons
                sampled = True
            else:
                print(f"\n  {display} Subspace Diversity ({n_neurons} neurons, shape={neurons.shape})", flush=True)
                neurons_sample = neurons
                n_sample = n_neurons

            # Flatten each neuron
            neurons_flat = neurons_sample.reshape(n_sample, -1)

            # Pairwise cosine similarity (all on GPU)
            neurons_norm = F.normalize(neurons_flat, dim=-1)
            sim_matrix = torch.mm(neurons_norm, neurons_norm.t())

            # Remove diagonal (mask on same device as sim_matrix)
            mask = ~torch.eye(n_sample, dtype=torch.bool, device=sim_matrix.device)
            sim_off_diag = sim_matrix[mask]

            avg_sim = sim_off_diag.mean().item()
            std_sim = sim_off_diag.std().item()
            min_sim = sim_off_diag.min().item()
            max_sim = sim_off_diag.max().item()

            # Find most similar pairs
            sim_flat = sim_matrix.clone().view(-1)
            for i in range(n_sample):
                sim_flat[i * n_sample + i] = -2  # Exclude diagonal

            top_k = min(10, n_sample * (n_sample - 1) // 2)
            top_vals, top_idx = torch.topk(sim_flat, top_k)

            top_similar = []
            for i in range(top_k):
                idx = top_idx[i].item()
                n_i = idx // n_sample
                n_j = idx % n_sample
                top_similar.append((n_i, n_j, top_vals[i].item()))

            # Interpretation
            if avg_sim < 0.3:
                interpretation = 'DIVERSE: Neurons use distinct subspaces (good!)'
            elif avg_sim < 0.6:
                interpretation = 'MODERATE: Some overlap in neuron subspaces'
            else:
                interpretation = 'COLLAPSED: Neurons converging to similar subspaces (bad!)'

            results[display] = {
                'n_neurons': n_neurons,
                'n_sampled': n_sample if sampled else n_neurons,
                'sampled': sampled,
                'mean_similarity': avg_sim,
                'std_similarity': std_sim,
                'min_similarity': min_sim,
                'max_similarity': max_sim,
                'top_similar_pairs': top_similar,
                'interpretation': interpretation,
            }

            print(f"    Mean pairwise similarity: {avg_sim:.4f} +/- {std_sim:.4f}", flush=True)
            print(f"    Interpretation: {interpretation}", flush=True)

        return results

    def visualize_coselection(self, results: Dict, output_dir: str):
        """
        Visualize co-selection patterns.

        Args:
            results: Results from analyze_coselection()
            output_dir: Directory for output
        """
        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        for pair_key, pair_results in results.items():
            if 'co_matrix' not in pair_results:
                continue

            co_matrix = np.array(pair_results['co_matrix'])
            pair_name = pair_results.get('pair_name', pair_key)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1. Heatmap
            ax = axes[0]
            im = ax.imshow(co_matrix, aspect='auto', cmap='hot')
            fig.colorbar(im, ax=ax, label='Co-selection count')
            ax.set_xlabel('Pool B neuron')
            ax.set_ylabel('Pool A neuron')
            ax.set_title(f'{pair_name}\nCo-selection Heatmap')

            # 2. Concentration bar chart
            ax = axes[1]
            conc = pair_results.get('concentration', {})
            labels = ['Top 10', 'Top 50', 'Top 100']
            values = [conc.get('top10_pct', 0), conc.get('top50_pct', 0), conc.get('top100_pct', 0)]
            bars = ax.bar(labels, values, color=['red', 'orange', 'green'], alpha=0.7)
            ax.set_ylabel('% of all co-selections')
            ax.set_title(f'{pair_name}\nPair Concentration')
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center')

            # 3. Top pairs
            ax = axes[2]
            top_pairs = pair_results.get('top_pairs', [])[:10]
            if top_pairs:
                labels = [f"{p['a_name']}\n{p['b_name']}" for p in top_pairs]
                values = [p['pct'] for p in top_pairs]
                ax.barh(range(len(values)), values, color='steelblue', alpha=0.7)
                ax.set_yticks(range(len(values)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel('% of co-selections')
                ax.set_title(f'{pair_name}\nTop 10 Pairs')
                ax.invert_yaxis()

            plt.tight_layout()
            path = os.path.join(output_dir, f'coselection_{pair_key}.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

    def run_all(self, dataloader, output_dir: str = './coselection_analysis',
                pairs: str = 'all', n_batches: int = 50) -> Dict:
        """
        Run all co-selection analyses.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            pairs: Which pairs to analyze ('all' or comma-separated list)
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Select pairs to analyze
        if pairs == 'all':
            pairs_to_analyze = list(COSELECTION_PAIRS.keys())
        else:
            pairs_to_analyze = [p.strip() for p in pairs.split(',')]

        all_results = {}

        print("\n--- Analyzing Co-selection Patterns ---", flush=True)
        for pair_key in pairs_to_analyze:
            if pair_key not in COSELECTION_PAIRS:
                print(f"  Skip unknown pair: {pair_key}", flush=True)
                continue

            try:
                # Co-selection analysis
                print(f"  Starting {pair_key} co-selection...", flush=True)
                cosel_results = self.analyze_coselection(dataloader, pair_key, n_batches)
                if cosel_results:
                    all_results[pair_key] = cosel_results
                print(f"  Completed {pair_key} co-selection.", flush=True)

                # Subspace diversity
                print(f"  Starting {pair_key} subspace diversity...", flush=True)
                div_results = self.analyze_subspace_diversity(pair_key)
                if div_results and pair_key in all_results:
                    all_results[pair_key]['subspace_diversity'] = div_results
                print(f"  Completed {pair_key} subspace diversity.", flush=True)

            except Exception as e:
                print(f"  ERROR in {pair_key}: {e}", flush=True)
                traceback.print_exc()
                all_results[pair_key] = {'error': str(e)}

        # Visualize
        if HAS_MATPLOTLIB:
            print("\n--- Generating Visualizations ---", flush=True)
            try:
                self.visualize_coselection(all_results, output_dir)
            except Exception as e:
                print(f"  ERROR in visualization: {e}", flush=True)
                traceback.print_exc()

        print("--- Co-selection Analysis Complete ---", flush=True)
        return all_results

#!/usr/bin/env python3
"""
DAWN Routing Analysis for Generation
=====================================
Analyze token-level routing patterns during text generation.

Features:
1. Generate text while collecting per-token routing indices
2. Analyze common neurons across multiple runs
3. Visualize routing patterns with heatmaps

Usage:
    python scripts/analysis/routing_analysis.py \
        --checkpoint path/to/checkpoint \
        --prompt "The capital of France is" \
        --output routing_analysis/
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

# Handle both module import and standalone execution
try:
    from ..utils import (
        load_model, get_router,
        ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
        HAS_MATPLOTLIB, plt
    )
except ImportError:
    # Standalone execution - use explicit path
    from scripts.analysis.utils import (
        load_model, get_router,
        ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
        HAS_MATPLOTLIB, plt
    )

if HAS_MATPLOTLIB:
    import seaborn as sns


class GenerationRoutingAnalyzer:
    """Analyze routing patterns during text generation."""

    def __init__(self, model, tokenizer, device='cuda', target_layer: int = None):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            tokenizer: Tokenizer for text encoding/decoding
            device: Device for computation
            target_layer: Specific layer to analyze (None = all layers)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_layer = target_layer
        self.model.eval()

        # Detect v18.2+ models (uses path_weights instead of per-pool weights)
        self.is_v18_2 = hasattr(model, '__version__') and model.__version__.startswith('18.')

    def _enable_pref_tensors(self):
        """Enable store_pref_tensors for detailed analysis (v18.2+)."""
        if hasattr(self.model, 'router') and hasattr(self.model.router, 'store_pref_tensors'):
            self.model.router.store_pref_tensors = True

    def _disable_pref_tensors(self):
        """Disable store_pref_tensors after analysis."""
        if hasattr(self.model, 'router') and hasattr(self.model.router, 'store_pref_tensors'):
            self.model.router.store_pref_tensors = False

    def generate_with_routing(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Dict:
        """
        Generate text while collecting routing information.

        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = greedy)

        Returns:
            Dictionary with tokens, routing indices, and generated text
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]

        # Collect routing info per generated token
        routing_logs = {
            'tokens': [],
            'token_ids': [],
            # Feature routing (per token)
            'fv_indices': [],      # Feature V indices
            'fqk_q_indices': [],   # Feature QK (Q) indices
            'fqk_k_indices': [],   # Feature QK (K) indices
            # Restore routing (per token)
            'rv_indices': [],      # Restore V indices
            'rqk_q_indices': [],   # Restore QK (Q) indices
            'rqk_k_indices': [],   # Restore QK (K) indices
            # Knowledge routing (per token)
            'fknow_indices': [],   # Feature Knowledge indices
            'rknow_indices': [],   # Restore Knowledge indices
            # Weights for analysis
            'fv_weights': [],
            'rv_weights': [],
        }

        self._enable_pref_tensors()

        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # Forward with routing info
                    # v18.2 needs return_path_weights=True to store actual weights
                    if self.is_v18_2:
                        outputs = self.model(generated, return_path_weights=True)
                    else:
                        outputs = self.model(generated, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        logits, routing_infos = outputs[0], outputs[1]
                    else:
                        logits = outputs
                        routing_infos = None

                    # Get next token
                    next_token_logits = logits[:, -1, :]

                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                    # Decode token
                    token_text = self.tokenizer.decode(next_token[0])
                    routing_logs['tokens'].append(token_text)
                    routing_logs['token_ids'].append(next_token.item())

                    # Extract routing indices from last position
                    if routing_infos is not None:
                        self._extract_routing_indices(routing_infos, routing_logs)

                    generated = torch.cat([generated, next_token], dim=1)
        finally:
            self._disable_pref_tensors()

        # Decode full generation
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        routing_logs['prompt'] = prompt
        routing_logs['generated_text'] = generated_text
        routing_logs['full_text'] = generated_text

        return routing_logs

    def _extract_routing_indices(self, routing_infos: List[Dict], routing_logs: Dict):
        """Extract routing indices from routing_infos for the last token."""
        # Aggregate across layers (or specific layer if target_layer is set)
        fv_indices_all = []
        rv_indices_all = []
        fqk_q_indices_all = []
        fqk_k_indices_all = []
        rqk_q_indices_all = []
        rqk_k_indices_all = []
        fknow_indices_all = []
        rknow_indices_all = []

        fv_weights_last = None
        rv_weights_last = None

        # Filter layers if target_layer is specified
        if self.target_layer is not None:
            if 0 <= self.target_layer < len(routing_infos):
                layers_to_process = [(self.target_layer, routing_infos[self.target_layer])]
            else:
                layers_to_process = []
        else:
            layers_to_process = enumerate(routing_infos)

        for layer_idx, layer_info in layers_to_process:
            # v18.2: path_weights stored in layer_info['path_weights']
            path_weights = layer_info.get('path_weights', {})
            if path_weights:
                # v18.2 format: path_weights contains fv, rv, fqk_Q, etc.
                # Each is a LIST of [B, S, N] tensors (one per path)
                def extract_indices_from_paths(path_list, indices_list):
                    """Extract active neuron indices from list of path weight tensors."""
                    if path_list is None:
                        return None
                    combined_weights = None
                    for path_w in path_list:
                        if torch.is_tensor(path_w):
                            w_last = path_w[0, -1]  # [N]
                            if combined_weights is None:
                                combined_weights = w_last.clone()
                            else:
                                combined_weights = combined_weights + w_last
                    if combined_weights is not None:
                        idx = (combined_weights > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                        indices_list.extend(idx)
                        return combined_weights.cpu()
                    return None

                fv_weights_last = extract_indices_from_paths(path_weights.get('fv'), fv_indices_all)
                rv_weights_last = extract_indices_from_paths(path_weights.get('rv'), rv_indices_all)
                extract_indices_from_paths(path_weights.get('fqk_Q'), fqk_q_indices_all)
                extract_indices_from_paths(path_weights.get('fqk_K'), fqk_k_indices_all)
                extract_indices_from_paths(path_weights.get('rqk_Q'), rqk_q_indices_all)
                extract_indices_from_paths(path_weights.get('rqk_K'), rqk_k_indices_all)
                extract_indices_from_paths(path_weights.get('feature_know'), fknow_indices_all)
                extract_indices_from_paths(path_weights.get('restore_know'), rknow_indices_all)
            else:
                # v17.1/v18.5 format: weights or masks in attention/knowledge dicts
                attn = layer_info.get('attention', layer_info)
                know = layer_info.get('knowledge', {})

                # v18.5: Try masks first (binary, cleaner signal)
                fv_m = attn.get('fv_mask')
                if fv_m is not None:
                    fv_last = fv_m[0, -1]  # [N_v] boolean mask
                    fv_idx = fv_last.nonzero(as_tuple=True)[0].cpu().tolist()
                    fv_indices_all.extend(fv_idx)
                    fv_weights_last = fv_last.float().cpu()

                rv_m = attn.get('rv_mask')
                if rv_m is not None:
                    rv_last = rv_m[0, -1]
                    rv_idx = rv_last.nonzero(as_tuple=True)[0].cpu().tolist()
                    rv_indices_all.extend(rv_idx)
                    rv_weights_last = rv_last.float().cpu()

                fqk_q_m = attn.get('fqk_mask_Q')
                if fqk_q_m is not None:
                    fqk_q_idx = fqk_q_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    fqk_q_indices_all.extend(fqk_q_idx)

                fqk_k_m = attn.get('fqk_mask_K')
                if fqk_k_m is not None:
                    fqk_k_idx = fqk_k_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    fqk_k_indices_all.extend(fqk_k_idx)

                rqk_q_m = attn.get('rqk_mask_Q')
                if rqk_q_m is not None:
                    rqk_q_idx = rqk_q_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    rqk_q_indices_all.extend(rqk_q_idx)

                rqk_k_m = attn.get('rqk_mask_K')
                if rqk_k_m is not None:
                    rqk_k_idx = rqk_k_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    rqk_k_indices_all.extend(rqk_k_idx)

                fknow_m = know.get('feature_know_mask')
                if fknow_m is not None:
                    fknow_idx = fknow_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    fknow_indices_all.extend(fknow_idx)

                rknow_m = know.get('restore_know_mask')
                if rknow_m is not None:
                    rknow_idx = rknow_m[0, -1].nonzero(as_tuple=True)[0].cpu().tolist()
                    rknow_indices_all.extend(rknow_idx)

                # v17.1 fallback: weights (if masks not available)
                # Feature V weights -> indices
                fv_w = attn.get('fv_weights')
                if fv_w is not None and fv_m is None:
                    # Get indices for last token position
                    fv_last = fv_w[0, -1]  # [N_v]
                    fv_idx = (fv_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    fv_indices_all.extend(fv_idx)
                    fv_weights_last = fv_last.cpu()

                # Restore V weights -> indices (fallback)
                rv_w = attn.get('rv_weights')
                if rv_w is not None and rv_m is None:
                    rv_last = rv_w[0, -1]
                    rv_idx = (rv_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    rv_indices_all.extend(rv_idx)
                    rv_weights_last = rv_last.cpu()

                # Feature QK (Q/K) fallback
                fqk_q_w = attn.get('fqk_weights_Q')
                if fqk_q_w is not None and fqk_q_m is None:
                    fqk_q_last = fqk_q_w[0, -1]
                    fqk_q_idx = (fqk_q_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    fqk_q_indices_all.extend(fqk_q_idx)

                fqk_k_w = attn.get('fqk_weights_K')
                if fqk_k_w is not None and fqk_k_m is None:
                    fqk_k_last = fqk_k_w[0, -1]
                    fqk_k_idx = (fqk_k_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    fqk_k_indices_all.extend(fqk_k_idx)

                # Restore QK (Q/K) fallback
                rqk_q_w = attn.get('rqk_weights_Q')
                if rqk_q_w is not None and rqk_q_m is None:
                    rqk_q_last = rqk_q_w[0, -1]
                    rqk_q_idx = (rqk_q_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    rqk_q_indices_all.extend(rqk_q_idx)

                rqk_k_w = attn.get('rqk_weights_K')
                if rqk_k_w is not None and rqk_k_m is None:
                    rqk_k_last = rqk_k_w[0, -1]
                    rqk_k_idx = (rqk_k_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    rqk_k_indices_all.extend(rqk_k_idx)

                # Knowledge fallback
                fknow_w = know.get('feature_know_w')
                if fknow_w is not None and fknow_m is None:
                    fknow_last = fknow_w[0, -1]
                    fknow_idx = (fknow_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    fknow_indices_all.extend(fknow_idx)

                rknow_w = know.get('restore_know_w')
                if rknow_w is not None and rknow_m is None:
                    rknow_last = rknow_w[0, -1]
                    rknow_idx = (rknow_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    rknow_indices_all.extend(rknow_idx)

        # Store unique indices per token
        routing_logs['fv_indices'].append(list(set(fv_indices_all)))
        routing_logs['rv_indices'].append(list(set(rv_indices_all)))
        routing_logs['fqk_q_indices'].append(list(set(fqk_q_indices_all)))
        routing_logs['fqk_k_indices'].append(list(set(fqk_k_indices_all)))
        routing_logs['rqk_q_indices'].append(list(set(rqk_q_indices_all)))
        routing_logs['rqk_k_indices'].append(list(set(rqk_k_indices_all)))
        routing_logs['fknow_indices'].append(list(set(fknow_indices_all)))
        routing_logs['rknow_indices'].append(list(set(rknow_indices_all)))

        if fv_weights_last is not None:
            routing_logs['fv_weights'].append(fv_weights_last)
        if rv_weights_last is not None:
            routing_logs['rv_weights'].append(rv_weights_last)


def analyze_common_neurons(
    routing_logs_list: List[Dict],
    target_token: str = None,
    pool_type: str = 'fv',
) -> Dict:
    """
    Analyze common neurons across multiple generation runs.

    Args:
        routing_logs_list: List of routing logs from generate_with_routing()
        target_token: Specific token to analyze (e.g., "paris")
        pool_type: 'fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k', 'fknow', 'rknow'

    Returns:
        Dictionary with common neuron analysis
    """
    indices_key = f'{pool_type}_indices'
    all_indices = []
    token_to_indices = defaultdict(list)

    for routing_log in routing_logs_list:
        tokens = routing_log.get('tokens', [])
        indices_list = routing_log.get(indices_key, [])

        for i, (token, indices) in enumerate(zip(tokens, indices_list)):
            token_lower = token.strip().lower()
            all_indices.extend(indices)
            token_to_indices[token_lower].append(indices)

    # Overall frequency
    counter = Counter(all_indices)
    total_tokens = sum(len(routing_log.get('tokens', [])) for routing_log in routing_logs_list)

    results = {
        'pool_type': pool_type,
        'total_tokens_analyzed': total_tokens,
        'unique_neurons_used': len(counter),
        'top_neurons': counter.most_common(20),
    }

    # Target token analysis
    if target_token:
        target_lower = target_token.strip().lower()
        if target_lower in token_to_indices:
            target_indices_list = token_to_indices[target_lower]
            target_counter = Counter()
            for indices in target_indices_list:
                target_counter.update(indices)

            # Find neurons that appear in ALL occurrences
            n_occurrences = len(target_indices_list)
            common_neurons = [
                neuron for neuron, count in target_counter.items()
                if count == n_occurrences
            ]

            results['target_token'] = target_token
            results['target_occurrences'] = n_occurrences
            results['target_top_neurons'] = target_counter.most_common(10)
            results['target_common_neurons'] = common_neurons

    return results


def analyze_token_neurons(
    analyzer: 'GenerationRoutingAnalyzer',
    prompt: str,
    target_token: str,
    iterations: int = 50,
    pool_type: str = 'fv',
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    top_k: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Run multiple generations and analyze neurons activated for a specific token.

    Args:
        analyzer: GenerationRoutingAnalyzer instance
        prompt: Input prompt
        target_token: Token to search for (e.g., "paris")
        iterations: Number of generation runs
        pool_type: Neuron pool to analyze
        max_new_tokens: Max tokens per generation
        temperature: Sampling temperature
        top_k: Top-k sampling
        verbose: Print progress

    Returns:
        Dictionary with analysis results
    """
    target_lower = target_token.strip().lower()
    indices_key = f'{pool_type}_indices'

    # Collect runs where target token appears
    matching_runs = []
    all_runs = []

    if verbose:
        print(f"\nRunning {iterations} generations...")
        print(f"Target token: '{target_token}'")
        print(f"Pool: {pool_type.upper()}")
        print("-" * 50)

    for i in range(iterations):
        routing_log = analyzer.generate_with_routing(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        all_runs.append(routing_log)

        # Check if target token appears
        tokens = routing_log.get('tokens', [])
        indices_list = routing_log.get(indices_key, [])

        for j, token in enumerate(tokens):
            if target_lower in token.strip().lower():
                # Found target token
                if j < len(indices_list):
                    matching_runs.append({
                        'run_idx': i,
                        'token_idx': j,
                        'token': token,
                        'indices': indices_list[j],
                        'generated_text': routing_log['full_text'],
                    })
                break

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{iterations}] Found '{target_token}' in {len(matching_runs)} runs")

    # Analyze common neurons
    n_matching = len(matching_runs)
    n_total = iterations

    if verbose:
        print(f"\n{'='*50}")
        print(f"RESULTS: '{target_token}' appeared in {n_matching}/{n_total} runs")
        print('='*50)

    if n_matching == 0:
        return {
            'target_token': target_token,
            'prompt': prompt,
            'pool_type': pool_type,
            'layer': analyzer.target_layer,
            'iterations': iterations,
            'matching_runs': 0,
            'total_runs': iterations,
            'match_rate': 0.0,
            'common_neurons_100': [],
            'common_neurons_80': [],
            'common_neurons_50': [],
            'neuron_frequencies': [],
            'matching_run_details': [],
            'unique_neurons_total': 0,
        }

    # Count neuron frequencies across matching runs
    neuron_counter = Counter()
    for run in matching_runs:
        neuron_counter.update(run['indices'])

    # Find neurons that appear in ALL matching runs (100%)
    common_neurons_100 = [
        neuron for neuron, count in neuron_counter.items()
        if count == n_matching
    ]

    # Neurons with frequency >= 80%
    common_neurons_80 = [
        neuron for neuron, count in neuron_counter.items()
        if count >= n_matching * 0.8
    ]

    # Neurons with frequency >= 50%
    common_neurons_50 = [
        neuron for neuron, count in neuron_counter.items()
        if count >= n_matching * 0.5
    ]

    # Build frequency list
    neuron_frequencies = [
        {'neuron': neuron, 'count': count, 'percentage': count / n_matching * 100}
        for neuron, count in neuron_counter.most_common()
    ]

    if verbose:
        print(f"\n100% common neurons ({len(common_neurons_100)}): {sorted(common_neurons_100)[:20]}")
        print(f" 80%+ common neurons ({len(common_neurons_80)}): {sorted(common_neurons_80)[:20]}")
        print(f" 50%+ common neurons ({len(common_neurons_50)}): {sorted(common_neurons_50)[:20]}")

        print(f"\nTop 15 neurons by frequency:")
        for nf in neuron_frequencies[:15]:
            bar = '█' * int(nf['percentage'] / 5)
            print(f"  Neuron {nf['neuron']:4d}: {nf['count']:3d}/{n_matching} ({nf['percentage']:5.1f}%) {bar}")

        # Show sample generations
        print(f"\nSample generations with '{target_token}':")
        for run in matching_runs[:3]:
            print(f"  Run {run['run_idx']}: {run['generated_text'][:80]}...")

    return {
        'target_token': target_token,
        'prompt': prompt,
        'pool_type': pool_type,
        'layer': analyzer.target_layer,
        'iterations': iterations,
        'matching_runs': n_matching,
        'total_runs': n_total,
        'match_rate': n_matching / n_total,
        'common_neurons_100': sorted(common_neurons_100),
        'common_neurons_80': sorted(common_neurons_80),
        'common_neurons_50': sorted(common_neurons_50),
        'neuron_frequencies': neuron_frequencies,
        'matching_run_details': matching_runs,
        'unique_neurons_total': len(neuron_counter),
    }


def plot_routing_heatmap(
    routing_log: Dict,
    pool_type: str = 'fv',
    output_path: str = None,
    max_neurons: int = 50,
    figsize: Tuple[int, int] = (14, 8),
) -> Optional[str]:
    """
    Plot token x neuron routing heatmap.

    Args:
        routing_log: Routing log from generate_with_routing()
        pool_type: 'fv', 'rv', etc.
        output_path: Path to save figure
        max_neurons: Maximum number of neurons to display
        figsize: Figure size

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    tokens = routing_log.get('tokens', [])
    weights_key = f'{pool_type}_weights'
    weights_list = routing_log.get(weights_key, [])

    if not weights_list:
        # Fall back to binary indices
        indices_key = f'{pool_type}_indices'
        indices_list = routing_log.get(indices_key, [])
        if not indices_list:
            print(f"No data for {pool_type}")
            return None

        # Find all unique neurons
        all_neurons = set()
        for indices in indices_list:
            all_neurons.update(indices)
        all_neurons = sorted(all_neurons)[:max_neurons]

        # Create binary matrix
        neuron_to_idx = {n: i for i, n in enumerate(all_neurons)}
        matrix = np.zeros((len(tokens), len(all_neurons)))
        for t_idx, indices in enumerate(indices_list):
            for n in indices:
                if n in neuron_to_idx:
                    matrix[t_idx, neuron_to_idx[n]] = 1

        neuron_labels = [str(n) for n in all_neurons]
    else:
        # Use actual weights
        weights_tensor = torch.stack(weights_list)  # [T, N]
        n_neurons = weights_tensor.shape[1]

        # Find most active neurons
        total_activation = weights_tensor.sum(dim=0)
        top_indices = total_activation.argsort(descending=True)[:max_neurons]

        matrix = weights_tensor[:, top_indices].numpy()
        neuron_labels = [str(idx.item()) for idx in top_indices]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Truncate token labels
    token_labels = [t[:15] if len(t) > 15 else t for t in tokens]

    sns.heatmap(
        matrix,
        xticklabels=neuron_labels,
        yticklabels=token_labels,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Routing Weight'}
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Generated Token')
    ax.set_title(f'Routing Heatmap ({pool_type.upper()})\n"{routing_log.get("prompt", "")[:50]}..."')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_routing_comparison(
    routing_logs_list: List[Dict],
    pool_type: str = 'fv',
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[str]:
    """
    Compare routing patterns across multiple runs.

    Args:
        routing_logs_list: List of routing logs
        pool_type: Pool type to analyze
        output_path: Path to save figure

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    indices_key = f'{pool_type}_indices'

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Neuron usage frequency
    all_neurons = Counter()
    for routing_log in routing_logs_list:
        for indices in routing_log.get(indices_key, []):
            all_neurons.update(indices)

    top_20 = all_neurons.most_common(20)
    if top_20:
        neurons, counts = zip(*top_20)
        axes[0].barh(range(len(neurons)), counts, color='steelblue')
        axes[0].set_yticks(range(len(neurons)))
        axes[0].set_yticklabels([f'N{n}' for n in neurons])
        axes[0].set_xlabel('Selection Count')
        axes[0].set_title(f'Top 20 Neurons ({pool_type.upper()})')
        axes[0].invert_yaxis()

    # 2. Neurons per token distribution
    neurons_per_token = []
    for routing_log in routing_logs_list:
        for indices in routing_log.get(indices_key, []):
            neurons_per_token.append(len(indices))

    if neurons_per_token:
        axes[1].hist(neurons_per_token, bins=30, color='coral', edgecolor='black')
        axes[1].axvline(np.mean(neurons_per_token), color='red', linestyle='--',
                        label=f'Mean: {np.mean(neurons_per_token):.1f}')
        axes[1].set_xlabel('Neurons per Token')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Routing Sparsity Distribution')
        axes[1].legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='DAWN Routing Analysis for Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt analysis
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt --prompt "The capital of France is"

  # Multiple prompts with comparison
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt --prompts "Hello world" "The sky is" "Once upon a time"

  # Analyze specific pool type
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt --pool fknow --prompt "The meaning of life is"

  # Full analysis with all pools
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt --all-pools

  # Analyze specific layer only (e.g., last layer = 11)
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt --layer 11

  # Token neuron analysis: run N iterations, find common neurons for target token
  python scripts/analysis/routing_analysis.py --checkpoint checkpoint.pt \\
      --prompt "The capital of France is" --target_token "paris" \\
      --iterations 50 --layer 11 --pool fv
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file or directory')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt for generation')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Multiple prompts for comparison analysis')
    parser.add_argument('--max_tokens', type=int, default=30,
                        help='Max tokens to generate (default: 30)')
    parser.add_argument('--output', type=str, default='routing_analysis',
                        help='Output directory (default: routing_analysis)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--pool', type=str, default='fv',
                        choices=['fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k', 'fknow', 'rknow'],
                        help='Pool type to analyze (default: fv)')
    parser.add_argument('--all-pools', action='store_true',
                        help='Analyze all pool types')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling, 0=greedy (default: 0)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (useful for headless environments)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to analyze (e.g., 11 for last layer). Default: all layers')
    parser.add_argument('--target_token', type=str, default=None,
                        help='Target token for iterative analysis (e.g., "paris")')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations for target_token analysis (default: 50)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 precision (faster on A100/H100)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster inference (PyTorch 2.0+)')
    args = parser.parse_args()

    # Default prompts if none provided
    if args.prompt is None and args.prompts is None:
        args.prompts = [
            "The capital of France is",
            "The meaning of life is",
            "Once upon a time, there was a",
            "def fibonacci(n):",
        ]
    elif args.prompt is not None:
        args.prompts = [args.prompt]

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Find checkpoint file
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        # Find best checkpoint in directory (search recursively)
        candidates = (
            list(ckpt_path.glob('*best*.pt')) +
            list(ckpt_path.glob('**/*best*.pt')) +
            list(ckpt_path.glob('*.pt')) +
            list(ckpt_path.glob('**/*.pt'))
        )
        # Filter out optimizer checkpoints
        candidates = [c for c in candidates if 'optimizer' not in c.name.lower()]
        if not candidates:
            print(f"No checkpoint found in {ckpt_path}")
            return
        # Prefer 'best' checkpoints
        best_candidates = [c for c in candidates if 'best' in c.name.lower()]
        ckpt_path = best_candidates[0] if best_candidates else candidates[0]
        print(f"Using checkpoint: {ckpt_path}")

    # Load model
    print(f"\n{'='*70}")
    print(f"Loading model from {ckpt_path}")
    print('='*70)
    model, tokenizer, config = load_model(str(ckpt_path), args.device)
    model = model.to(args.device)
    model.eval()

    # Apply optimizations for A100/H100
    if args.bf16:
        print("Using bfloat16 precision")
        model = model.to(torch.bfloat16)

    if args.compile:
        if hasattr(torch, 'compile'):
            print("Applying torch.compile optimization...")
            model = torch.compile(model, mode='reduce-overhead')
        else:
            print("Warning: torch.compile not available (requires PyTorch 2.0+)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create analyzer
    analyzer = GenerationRoutingAnalyzer(model, tokenizer, args.device, target_layer=args.layer)

    if args.layer is not None:
        print(f"Analyzing layer {args.layer} only")

    # TARGET TOKEN MODE: iterative analysis
    if args.target_token is not None:
        if args.prompt is None:
            print("ERROR: --prompt is required when using --target_token")
            return

        # Use top_k=50 by default for target_token mode (need sampling variance)
        top_k = args.top_k if args.top_k > 0 else 50

        print(f"\n{'='*70}")
        print("TARGET TOKEN ANALYSIS MODE")
        print('='*70)

        result = analyze_token_neurons(
            analyzer=analyzer,
            prompt=args.prompt,
            target_token=args.target_token,
            iterations=args.iterations,
            pool_type=args.pool,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
            verbose=True,
        )

        # Save results
        output_file = os.path.join(
            args.output,
            f'token_analysis_{args.target_token}_{args.pool}_layer{args.layer or "all"}.json'
        )
        os.makedirs(args.output, exist_ok=True)

        # Remove non-serializable items
        save_result = {k: v for k, v in result.items() if k != 'matching_run_details'}
        save_result['matching_run_samples'] = result['matching_run_details'][:10]  # Save first 10

        with open(output_file, 'w') as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"  Target token: '{args.target_token}'")
        print(f"  Match rate: {result['matching_runs']}/{result['total_runs']} ({result['match_rate']*100:.1f}%)")
        print(f"  Pool: {args.pool.upper()}")
        print(f"  Layer: {args.layer if args.layer is not None else 'all'}")
        print(f"  100% common neurons: {len(result['common_neurons_100'])}")
        print(f"   80%+ neurons: {len(result['common_neurons_80'])}")
        print(f"   50%+ neurons: {len(result['common_neurons_50'])}")
        return

    # NORMAL MODE: single/multiple prompt analysis
    # Determine pools to analyze
    if args.all_pools:
        pools = ['fv', 'rv', 'fqk_q', 'fqk_k', 'fknow', 'rknow']
    else:
        pools = [args.pool]

    # Collect all routing logs
    all_routing_logs = []

    print(f"\n{'='*70}")
    print("GENERATION WITH ROUTING ANALYSIS")
    print('='*70)

    for i, prompt in enumerate(args.prompts):
        print(f"\n[Prompt {i+1}/{len(args.prompts)}] '{prompt}'")
        print('-'*50)

        routing_log = analyzer.generate_with_routing(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        all_routing_logs.append(routing_log)

        # Print generated text
        print(f"Generated: {routing_log['full_text']}")

        # Print routing info for primary pool
        print(f"\nRouting ({args.pool.upper()}):")
        indices_key = f'{args.pool}_indices'
        for j, (token, indices) in enumerate(zip(routing_log['tokens'], routing_log.get(indices_key, []))):
            truncated = indices[:8]
            suffix = f'... (+{len(indices)-8})' if len(indices) > 8 else ''
            print(f"  [{j:2d}] '{token:12s}' -> {len(indices):2d} neurons: {truncated}{suffix}")

    # Comparison analysis
    if len(all_routing_logs) > 1:
        print(f"\n{'='*70}")
        print("COMMON NEURON ANALYSIS")
        print('='*70)

        for pool in pools:
            common = analyze_common_neurons(all_routing_logs, pool_type=pool)
            print(f"\n[{pool.upper()}] Unique neurons: {common['unique_neurons_used']}, "
                  f"Total tokens: {common['total_tokens_analyzed']}")
            print(f"  Top 10 neurons: {common['top_neurons'][:10]}")

    # Plotting
    if not args.no_plot and HAS_MATPLOTLIB:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print('='*70)

        for pool in pools:
            # Individual heatmaps
            for i, routing_log in enumerate(all_routing_logs):
                prompt_slug = routing_log['prompt'][:20].replace(' ', '_').replace('/', '_')
                heatmap_path = os.path.join(args.output, f'heatmap_{pool}_{i}_{prompt_slug}.png')
                plot_routing_heatmap(routing_log, pool, heatmap_path)
                print(f"  Saved: {heatmap_path}")

            # Comparison plot (if multiple prompts)
            if len(all_routing_logs) > 1:
                compare_path = os.path.join(args.output, f'comparison_{pool}.png')
                plot_routing_comparison(all_routing_logs, pool, compare_path)
                print(f"  Saved: {compare_path}")

    # Save routing logs
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print('='*70)

    for i, routing_log in enumerate(all_routing_logs):
        # Remove tensor weights for JSON serialization
        save_log = {k: v for k, v in routing_log.items() if not k.endswith('_weights')}
        prompt_slug = routing_log['prompt'][:20].replace(' ', '_').replace('/', '_')
        log_path = os.path.join(args.output, f'routing_log_{i}_{prompt_slug}.json')
        with open(log_path, 'w') as f:
            json.dump(save_log, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {log_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"  Prompts analyzed: {len(args.prompts)}")
    print(f"  Pools analyzed: {pools}")
    print(f"  Layer: {args.layer if args.layer is not None else 'all'}")
    print(f"  Output directory: {args.output}")
    print(f"  Max tokens per generation: {args.max_tokens}")


if __name__ == '__main__':
    main()

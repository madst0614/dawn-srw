"""
Behavioral Analysis
===================
Analyze token-level behavioral patterns in DAWN v17.1 models.

Includes:
- Single neuron analysis
- Token trajectory analysis (routing by position)
- Probing classifier for POS prediction
- Ablation studies
"""

import os
import numpy as np
import torch
from typing import Dict, Optional
from collections import defaultdict

from .base import BaseAnalyzer
from .utils import (
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
    calc_entropy_ratio, simple_pos_tag,
    get_batch_input_ids,
    RoutingDataExtractor,  # Schema layer for model-agnostic access
    HAS_MATPLOTLIB, HAS_SKLEARN, HAS_TQDM, tqdm, plt
)

if HAS_SKLEARN:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score


class BehavioralAnalyzer(BaseAnalyzer):
    """Token-level behavioral analyzer."""

    def __init__(self, model, router=None, tokenizer=None, device='cuda', extractor=None):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance (auto-detected if None)
            tokenizer: Tokenizer instance
            device: Device for computation
            extractor: RoutingDataExtractor instance (created if None)
        """
        super().__init__(model, router=router, tokenizer=tokenizer, device=device)
        self.extractor = extractor or RoutingDataExtractor(model, device=device)

    def analyze_single_neuron(self, neuron_id: int, neuron_type: str) -> Dict:
        """
        Analyze a single neuron's properties.

        Args:
            neuron_id: Index of the neuron
            neuron_type: Type of neuron (e.g., 'feature_qk')

        Returns:
            Dictionary with neuron properties
        """
        results = {
            'neuron_type': neuron_type,
            'neuron_id': neuron_id,
        }

        neuron_types = self.get_neuron_types()

        # Get EMA usage
        type_info = neuron_types.get(neuron_type)
        if type_info:
            ema_attr = type_info[1]
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                if neuron_id < len(ema):
                    results['usage_ema'] = float(ema[neuron_id])

        # Get embedding properties
        emb = self.router.neuron_emb.detach().cpu().numpy()

        offset = 0
        for name, (_, _, n_attr, _) in neuron_types.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                if name == neuron_type:
                    if neuron_id < n:
                        neuron_emb = emb[offset + neuron_id]
                        results['embedding_norm'] = float(np.linalg.norm(neuron_emb))
                        results['embedding_mean'] = float(neuron_emb.mean())
                        results['embedding_std'] = float(neuron_emb.std())
                    break
                offset += n

        return results

    def analyze_token_trajectory(self, dataloader, n_batches: int = 20, layer_idx: int = None) -> Dict:
        """
        Analyze how routing entropy changes across sequence positions for ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            layer_idx: Specific layer to analyze (None = aggregate all layers)

        Returns:
            Dictionary with position-wise entropy statistics per layer
        """
        # {layer: {routing_key: {position: [entropy_values]}}}
        layer_position_routing = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        self.model.eval()
        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Trajectory')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                # Process ALL layers using extractor
                for layer in routing:
                    lidx = layer.layer_idx
                    if layer_idx is not None and lidx != layer_idx:
                        continue

                    # Use standardized keys
                    for key in ROUTING_KEYS.keys():
                        weights = layer.get_weight(key)
                        if weights is None:
                            continue

                        if weights.dim() == 3:  # [B, S, N] token-level
                            for pos in range(min(weights.shape[1], 128)):
                                ent = calc_entropy_ratio(weights[:, pos, :])
                                layer_position_routing[lidx][key][pos].append(ent)
                        elif weights.dim() == 2:  # [B, N] batch-level - same for all positions
                            ent = calc_entropy_ratio(weights)
                            for pos in range(128):
                                layer_position_routing[lidx][key][pos].append(ent)

        # Build per-layer results
        results = {'per_layer': {}}

        # Aggregate data for overall results
        aggregated_routing = defaultdict(lambda: defaultdict(list))

        for lidx, position_routing in layer_position_routing.items():
            layer_results = {}
            for key in ROUTING_KEYS.keys():
                if position_routing[key]:
                    pos_avg = {}
                    for pos, values in position_routing[key].items():
                        pos_avg[pos] = float(np.mean(values))
                        aggregated_routing[key][pos].extend(values)

                    early_positions = [v for p, v in pos_avg.items() if p < 10]
                    late_positions = [v for p, v in pos_avg.items() if p >= 10]

                    layer_results[key] = {
                        'display': ROUTING_KEYS[key][0],
                        'position_entropy': pos_avg,
                        'early_avg': float(np.mean(early_positions)) if early_positions else 0,
                        'late_avg': float(np.mean(late_positions)) if late_positions else 0,
                    }

            if layer_results:
                results['per_layer'][f'L{lidx}'] = layer_results

        # Aggregated results (backward compatibility)
        for key in ROUTING_KEYS.keys():
            if aggregated_routing[key]:
                pos_avg = {}
                for pos, values in aggregated_routing[key].items():
                    pos_avg[pos] = float(np.mean(values))

                early_positions = [v for p, v in pos_avg.items() if p < 10]
                late_positions = [v for p, v in pos_avg.items() if p >= 10]

                results[key] = {
                    'display': ROUTING_KEYS[key][0],
                    'position_entropy': pos_avg,
                    'early_avg': float(np.mean(early_positions)) if early_positions else 0,
                    'late_avg': float(np.mean(late_positions)) if late_positions else 0,
                }

        results['n_layers'] = len(layer_position_routing)
        return results

    def run_probing(self, dataloader, max_batches: int = 50, layer_idx: int = None) -> Dict:
        """
        Run probing classifier for POS prediction across ALL layers.

        Uses routing weights to predict part-of-speech tags.
        Optimized with vectorized tensor operations.

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process
            layer_idx: Specific layer to analyze (None = all layers aggregated)

        Returns:
            Dictionary with probing accuracy per layer/routing key
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        # {layer_key: {routing_key: tensor list}}
        X_tensors = defaultdict(lambda: defaultdict(list))
        y_labels = defaultdict(list)

        self.model.eval()
        with self.extractor.analysis_context():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Probing', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                B, S = input_ids.shape

                # Get attention mask if available
                if isinstance(batch, dict):
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                else:
                    attention_mask = torch.ones_like(input_ids)

                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True)
                    routing = self.extractor.extract(outputs)
                    if not routing:
                        continue
                except Exception:
                    continue

                # Flatten mask for vectorized selection
                flat_mask = attention_mask.view(-1).bool()  # [B*S]
                valid_count = flat_mask.sum().item()

                if valid_count == 0:
                    continue

                # Get valid token ids and compute POS labels once per batch
                flat_input_ids = input_ids.view(-1)  # [B*S]
                valid_token_ids = flat_input_ids[flat_mask].cpu().tolist()
                valid_tokens = self.tokenizer.convert_ids_to_tokens(valid_token_ids)
                batch_pos_labels = [simple_pos_tag(t) for t in valid_tokens]

                # Process ALL layers using extractor
                for layer in routing:
                    lidx = layer.layer_idx
                    if layer_idx is not None and lidx != layer_idx:
                        continue

                    layer_key = f'L{lidx}'

                    # Collect attention routing weights - vectorized, using standardized keys
                    for key in ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            if w.dim() == 3:  # [B, S, N] token-level
                                flat_w = w.view(-1, w.shape[-1])  # [B*S, N]
                                valid_w = flat_w[flat_mask]  # [valid_count, N]
                            elif w.dim() == 2:  # [B, N] batch-level
                                # Expand and flatten
                                expanded = w.unsqueeze(1).expand(-1, S, -1)  # [B, S, N]
                                flat_w = expanded.reshape(-1, w.shape[-1])  # [B*S, N]
                                valid_w = flat_w[flat_mask]  # [valid_count, N]
                            else:
                                continue
                            X_tensors[layer_key][key].append(valid_w.cpu())

                    # Collect knowledge routing weights - vectorized, using standardized keys
                    for key in KNOWLEDGE_ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            if w.dim() == 3:  # [B, S, N] token-level
                                flat_w = w.view(-1, w.shape[-1])  # [B*S, N]
                                valid_w = flat_w[flat_mask]  # [valid_count, N]
                            elif w.dim() == 2:  # [B, N] batch-level
                                expanded = w.unsqueeze(1).expand(-1, S, -1)  # [B, S, N]
                                flat_w = expanded.reshape(-1, w.shape[-1])  # [B*S, N]
                                valid_w = flat_w[flat_mask]  # [valid_count, N]
                            else:
                                continue
                            X_tensors[layer_key][key].append(valid_w.cpu())

                    # Add POS labels for this layer
                    y_labels[layer_key].extend(batch_pos_labels)

        # Convert tensors to numpy arrays
        X_data = defaultdict(lambda: defaultdict(list))
        for layer_key, routing_data in X_tensors.items():
            for routing_key, tensor_list in routing_data.items():
                if tensor_list:
                    # Concatenate all tensors and convert to numpy
                    combined = torch.cat(tensor_list, dim=0).numpy()
                    X_data[layer_key][routing_key] = combined

        # Train and evaluate classifiers per layer/routing key
        results = {'per_layer': {}}

        for layer_key in X_data.keys():
            results['per_layer'][layer_key] = {}

            for routing_key in X_data[layer_key].keys():
                X = np.array(X_data[layer_key][routing_key])
                y = np.array(y_labels[layer_key][:len(X)])

                if len(X) < 100 or len(np.unique(y)) < 2:
                    results['per_layer'][layer_key][routing_key] = {'error': 'Not enough data'}
                    continue

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)

                    # Get display name
                    if routing_key in ROUTING_KEYS:
                        display = ROUTING_KEYS[routing_key][0]
                    elif routing_key in KNOWLEDGE_ROUTING_KEYS:
                        display = KNOWLEDGE_ROUTING_KEYS[routing_key][0]
                    else:
                        display = routing_key

                    results['per_layer'][layer_key][routing_key] = {
                        'display': f'{layer_key}/{display}',
                        'accuracy': float(accuracy),
                        'n_samples': len(X),
                        'n_classes': len(np.unique(y)),
                    }
                except Exception as e:
                    results['per_layer'][layer_key][routing_key] = {'error': str(e)}

        # Summary: best accuracy per routing type across layers
        results['summary'] = {}
        all_routing_keys = set()
        for layer_data in results['per_layer'].values():
            all_routing_keys.update(layer_data.keys())

        for rkey in all_routing_keys:
            accuracies = []
            for layer_key, layer_data in results['per_layer'].items():
                if rkey in layer_data and 'accuracy' in layer_data[rkey]:
                    accuracies.append(layer_data[rkey]['accuracy'])
            if accuracies:
                results['summary'][rkey] = {
                    'max_accuracy': max(accuracies),
                    'mean_accuracy': float(np.mean(accuracies)),
                }

        return results

    def run_ablation(self, dataloader, neuron_type: str, neuron_ids: list,
                     n_batches: int = 20) -> Dict:
        """
        Run ablation study by zeroing out specific neurons.

        Args:
            dataloader: DataLoader for input data
            neuron_type: Type of neurons to ablate
            neuron_ids: List of neuron IDs to ablate
            n_batches: Number of batches to process

        Returns:
            Dictionary with loss before and after ablation
        """
        import torch.nn.functional as F

        results = {
            'neuron_type': neuron_type,
            'ablated_neurons': neuron_ids,
        }

        # Compute baseline loss
        baseline_losses = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Baseline')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                outputs = self.model(input_ids)

                if isinstance(outputs, tuple):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    logits = outputs

                # Compute CLM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                baseline_losses.append(loss.item())

        results['baseline_loss'] = float(np.mean(baseline_losses))

        # Note: Actual ablation requires modifying the model forward pass
        # This is a placeholder for the ablation experiment
        results['note'] = 'Full ablation requires model modification. See model code for implementation.'

        return results

    def visualize_trajectory(self, trajectory_results: Dict, output_dir: str) -> Optional[str]:
        """
        Visualize token trajectory results.

        Args:
            trajectory_results: Results from analyze_token_trajectory()
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        if not HAS_MATPLOTLIB:
            return None

        os.makedirs(output_dir, exist_ok=True)

        n_keys = len([k for k in trajectory_results if k in ROUTING_KEYS])
        if n_keys == 0:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        ax_idx = 0
        for key, data in trajectory_results.items():
            if key not in ROUTING_KEYS or ax_idx >= len(axes):
                continue

            pos_ent = data.get('position_entropy', {})
            if not pos_ent:
                continue

            positions = sorted(pos_ent.keys())
            entropies = [pos_ent[p] for p in positions]

            axes[ax_idx].plot(positions, entropies, '-o', markersize=2)
            axes[ax_idx].set_xlabel('Position')
            axes[ax_idx].set_ylabel('Entropy (%)')
            axes[ax_idx].set_title(f'{data["display"]} Entropy by Position')
            axes[ax_idx].axhline(y=data['early_avg'], color='r', linestyle='--',
                                 alpha=0.5, label=f'Early avg: {data["early_avg"]:.1f}')
            axes[ax_idx].axhline(y=data['late_avg'], color='b', linestyle='--',
                                 alpha=0.5, label=f'Late avg: {data["late_avg"]:.1f}')
            axes[ax_idx].legend()
            ax_idx += 1

        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'trajectory.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def analyze_factual_neurons(
        self,
        prompts: list,
        targets: list,
        n_runs: int = 10,
        pool_type: str = 'fv',
        temperature: float = 1.0,
    ) -> Dict:
        """
        Analyze neuron activations for factual knowledge prompts.

        For each prompt, generates multiple times and tracks which neurons
        consistently activate when producing the target token.

        Paper Figure 7 data generation.

        Args:
            prompts: List of prompts (e.g., ["The capital of France is"])
            targets: List of expected target tokens (e.g., ["Paris"])
            n_runs: Number of generation runs per prompt
            pool_type: Pool to analyze ('fv', 'rv', etc.)
            temperature: Sampling temperature

        Returns:
            Dictionary with per-target neuron frequencies
        """
        from collections import Counter

        results = {
            'prompts': prompts,
            'targets': targets,
            'n_runs': n_runs,
            'pool_type': pool_type,
            'per_target': {},
        }

        # pool_type is already a standardized key (fv, rv, fqk_q, etc.)
        # The extractor will handle the mapping to raw keys

        self.model.eval()

        for prompt, target in zip(prompts, targets):
            neuron_counts = Counter()
            matching_runs = 0

            target_id = self.tokenizer.encode(target, add_special_tokens=False)
            if target_id:
                target_id = target_id[0]
            else:
                continue

            for run_idx in range(n_runs):
                # Encode prompt
                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors='pt'
                ).to(self.device)

                with self.extractor.analysis_context():
                    with torch.no_grad():
                        outputs = self.model(input_ids, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        logits = outputs[0]
                        routing = self.extractor.extract(outputs)
                    else:
                        logits = outputs
                        routing = None

                    # Get predicted token and top-k info
                    last_logits = logits[:, -1, :]
                    if temperature != 1.0:
                        last_logits = last_logits / temperature

                    next_token = last_logits.argmax(dim=-1).item()

                    # Check top-k accuracy
                    top_k_tokens = last_logits.topk(20, dim=-1).indices[0].tolist()
                    target_rank = top_k_tokens.index(target_id) + 1 if target_id in top_k_tokens else -1

                    # Store prediction info on first run
                    if run_idx == 0:
                        predicted_text = self.tokenizer.decode([next_token])
                        target_text = self.tokenizer.decode([target_id])

                    # Check if matches target
                    if next_token == target_id:
                        matching_runs += 1

                        # Extract active neurons from routing info using standardized key
                        if routing:
                            for layer in routing:
                                weights = layer.get_weight(pool_type)
                                if weights is not None:
                                    if weights.dim() == 3:
                                        w = weights[0, -1]
                                    else:
                                        w = weights[0]
                                    active = (w > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                                    for n in active:
                                        neuron_counts[n] += 1

            # Store results for this target
            base_result = {
                'prompt': prompt,
                'target_token': target_text if 'target_text' in dir() else target,
                'predicted_token': predicted_text if 'predicted_text' in dir() else 'N/A',
                'target_rank': target_rank if 'target_rank' in dir() else -1,
                'matching_runs': matching_runs,
                'total_runs': n_runs,
                'match_rate': matching_runs / n_runs,
            }

            if matching_runs > 0:
                # Calculate frequency (fraction of matching runs)
                neuron_freq = {
                    n: count / matching_runs
                    for n, count in neuron_counts.items()
                }
                # Filter to neurons appearing in >80% and 100% of runs
                common_neurons_80 = [
                    n for n, freq in neuron_freq.items()
                    if freq >= 0.8
                ]
                common_neurons_100 = [
                    n for n, freq in neuron_freq.items()
                    if freq >= 1.0
                ]

                base_result.update({
                    'neuron_frequencies': sorted(
                        [(n, f) for n, f in neuron_freq.items()],
                        key=lambda x: -x[1]
                    )[:50],
                    'common_neurons_80': common_neurons_80,
                    'common_neurons_100': common_neurons_100,
                })
            else:
                base_result['note'] = f'Model predicted "{predicted_text}" (rank={target_rank})' if 'predicted_text' in dir() else 'No matching runs'

            results['per_target'][target] = base_result

        return results

    def run_all(self, dataloader, output_dir: str = './behavioral_analysis', n_batches: int = 20) -> Dict:
        """
        Run all behavioral analyses.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        import traceback
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        try:
            print("  Running trajectory analysis...", flush=True)
            results['trajectory'] = self.analyze_token_trajectory(dataloader, n_batches)
            print("  Trajectory complete.", flush=True)
        except Exception as e:
            print(f"  ERROR in trajectory: {e}", flush=True)
            traceback.print_exc()
            results['trajectory'] = {'error': str(e)}

        try:
            print("  Running probing analysis...", flush=True)
            results['probing'] = self.run_probing(dataloader, n_batches * 2)
            print("  Probing complete.", flush=True)
        except Exception as e:
            print(f"  ERROR in probing: {e}", flush=True)
            traceback.print_exc()
            results['probing'] = {'error': str(e)}

        # Visualization
        try:
            print("  Generating visualization...", flush=True)
            viz_path = self.visualize_trajectory(results.get('trajectory', {}), output_dir)
            if viz_path:
                results['trajectory_visualization'] = viz_path
            print("  Visualization complete.", flush=True)
        except Exception as e:
            print(f"  ERROR in visualization: {e}", flush=True)
            traceback.print_exc()

        return results

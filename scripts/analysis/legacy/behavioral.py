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

        # Get embedding properties
        emb = self.router.neuron_emb.detach().cpu().numpy()

        offset = 0
        for name, (_, n_attr, _) in neuron_types.items():
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

    def run_probing(self, dataloader, max_batches: int = 50, layer_idx: int = None,
                     max_samples_per_layer: int = 20000) -> Dict:
        """
        Memory-efficient probing classifier for POS prediction.

        Optimizations:
        - Single dataloader pass with per-layer sample limits
        - Train and free memory per layer (not all at once)
        - Skip layers that already have enough samples

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process
            layer_idx: Specific layer to analyze (None = all layers)
            max_samples_per_layer: Max samples per layer (default 20k)

        Returns:
            Dictionary with probing accuracy per layer/routing key
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        import gc

        n_layers = getattr(self.model, 'n_layers', 12)
        layers_to_process = [layer_idx] if layer_idx is not None else list(range(n_layers))

        # Per-layer storage with sample count tracking
        layer_data = {
            lidx: {
                'X': defaultdict(list),
                'y': [],
                'count': 0
            } for lidx in layers_to_process
        }

        self.model.eval()
        with self.extractor.analysis_context():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Probing', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                # Check if all layers have enough samples
                if all(layer_data[lidx]['count'] >= max_samples_per_layer for lidx in layers_to_process):
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                B, S = input_ids.shape

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

                flat_mask = attention_mask.view(-1).bool()
                valid_count = flat_mask.sum().item()
                if valid_count == 0:
                    continue

                # Compute POS labels once per batch
                flat_input_ids = input_ids.view(-1)
                valid_token_ids = flat_input_ids[flat_mask].cpu().tolist()
                valid_tokens = self.tokenizer.convert_ids_to_tokens(valid_token_ids)
                batch_pos_labels = [simple_pos_tag(t) for t in valid_tokens]

                for layer in routing:
                    lidx = layer.layer_idx
                    if lidx not in layers_to_process:
                        continue

                    ld = layer_data[lidx]

                    # Skip if already have enough samples
                    if ld['count'] >= max_samples_per_layer:
                        continue

                    # Calculate remaining samples needed
                    remaining = max_samples_per_layer - ld['count']
                    sample_size = min(valid_count, remaining)

                    # Random sampling if needed
                    if sample_size < valid_count:
                        sample_indices = torch.randperm(valid_count)[:sample_size]
                        sampled_labels = [batch_pos_labels[i] for i in sample_indices.tolist()]
                    else:
                        sample_indices = None
                        sampled_labels = batch_pos_labels

                    # Collect routing weights
                    all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())
                    for key in all_keys:
                        w = layer.get_weight(key)
                        if w is None:
                            continue

                        if w.dim() == 3:
                            flat_w = w.view(-1, w.shape[-1])
                            valid_w = flat_w[flat_mask]
                        elif w.dim() == 2:
                            expanded = w.unsqueeze(1).expand(-1, S, -1)
                            flat_w = expanded.reshape(-1, w.shape[-1])
                            valid_w = flat_w[flat_mask]
                        else:
                            continue

                        # Apply sampling
                        if sample_indices is not None:
                            valid_w = valid_w[sample_indices]

                        ld['X'][key].append(valid_w.cpu())

                    ld['y'].extend(sampled_labels)
                    ld['count'] += sample_size

        # Train classifiers per layer and free memory immediately
        results = {'per_layer': {}}

        total_classifiers = sum(
            len(layer_data[lidx]['X']) for lidx in layers_to_process
            if layer_data[lidx]['count'] >= 100
        )
        print(f"  Training ~{total_classifiers} classifiers...", flush=True)

        for lidx in layers_to_process:
            layer_key = f'L{lidx}'
            results['per_layer'][layer_key] = {}

            ld = layer_data[lidx]

            for routing_key, tensor_list in ld['X'].items():
                if not tensor_list:
                    continue

                X = torch.cat(tensor_list, dim=0).numpy()
                y = np.array(ld['y'][:len(X)])

                if len(X) < 100 or len(np.unique(y)) < 2:
                    results['per_layer'][layer_key][routing_key] = {'error': 'Not enough data'}
                    continue

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    solver = 'saga' if len(X_train) > 10000 else 'lbfgs'
                    clf = LogisticRegression(max_iter=500, random_state=42, solver=solver, n_jobs=-1)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)

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

            # Free memory for this layer immediately
            del layer_data[lidx]
            gc.collect()

        # Summary statistics
        all_accuracies = []
        all_routing_keys = set()
        for layer_key, routing_data in results['per_layer'].items():
            for routing_key, data in routing_data.items():
                all_routing_keys.add(routing_key)
                if 'accuracy' in data:
                    all_accuracies.append(data['accuracy'])

        results['summary'] = {}
        for rkey in all_routing_keys:
            accuracies = []
            for layer_key, layer_data_result in results['per_layer'].items():
                if rkey in layer_data_result and 'accuracy' in layer_data_result[rkey]:
                    accuracies.append(layer_data_result[rkey]['accuracy'])
            if accuracies:
                results['summary'][rkey] = {
                    'max_accuracy': max(accuracies),
                    'mean_accuracy': float(np.mean(accuracies)),
                }

        if all_accuracies:
            results['overall'] = {
                'mean_accuracy': float(np.mean(all_accuracies)),
                'std_accuracy': float(np.std(all_accuracies)),
                'max_accuracy': float(np.max(all_accuracies)),
                'min_accuracy': float(np.min(all_accuracies)),
                'n_classifiers': len(all_accuracies),
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
        pools: list = None,
        pool_type: str = None,  # Deprecated, use pools instead
        min_target_count: int = 100,
        max_tokens_per_run: int = 200,
        max_runs: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Dict:
        """
        Analyze neuron activations for factual knowledge prompts.

        Efficiently analyzes ALL specified pools in a single generation pass.
        Each forward pass extracts activations from all pools simultaneously.

        Args:
            prompts: List of prompts (e.g., ["The capital of France is"])
            targets: List of expected target tokens (e.g., ["Paris"])
            pools: List of pools to analyze (default: ['fv', 'rv', 'fknow', 'rknow'])
            pool_type: DEPRECATED - single pool for backward compatibility
            min_target_count: Minimum target occurrences to collect (default: 100)
            max_tokens_per_run: Max tokens per run before giving up (default: 200)
            max_runs: Max runs to prevent infinite loop (default: 500)
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = greedy)

        Returns:
            Dictionary with per-pool, per-target neuron frequencies using unified naming
        """
        from collections import Counter
        import torch.nn.functional as F

        # Handle backward compatibility: pool_type -> pools
        if pools is None:
            if pool_type is not None:
                pools = [pool_type]  # Single pool for backward compatibility
            else:
                pools = ['fv', 'rv', 'fknow', 'rknow']  # Default: all V/Knowledge pools

        results = {
            'prompts': prompts,
            'targets': targets,
            'pools_analyzed': pools,
            'min_target_count': min_target_count,
            'per_pool': {pool: {} for pool in pools},
            'per_target': {},
        }

        self.model.eval()
        self.extractor.enable_weight_storage()

        print(f"    Analyzing {len(pools)} pools simultaneously: {pools}")

        # Validate that each target is a single token
        for target in targets:
            token_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if len(token_ids) != 1:
                print(f"Warning: '{target}' is not a single token (tokenizes to {len(token_ids)} tokens: {self.tokenizer.convert_ids_to_tokens(token_ids)})")

        # Add token validation info to results
        results['token_validation'] = {
            target: {
                'is_single_token': len(self.tokenizer.encode(target, add_special_tokens=False)) == 1,
                'token_ids': self.tokenizer.encode(target, add_special_tokens=False),
                'tokens': self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(target, add_special_tokens=False))
            }
            for target in targets
        }

        for prompt_idx, (prompt, target) in enumerate(zip(prompts, targets)):
            # Per-pool counters
            target_neuron_counts = {pool: Counter() for pool in pools}
            baseline_neuron_counts = {pool: Counter() for pool in pools}
            successful_runs = 0
            total_runs = 0
            total_baseline_steps = 0
            target_lower = target.strip().lower()
            sample_generations = []

            print(f"\n    [{prompt_idx+1}/{len(prompts)}] \"{prompt}\" → target: \"{target}\"")

            # Encode prompt once
            base_input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                while successful_runs < min_target_count and total_runs < max_runs:
                    total_runs += 1
                    print(f"\r      {successful_runs}/{min_target_count} targets (run {total_runs})", end='', flush=True)

                    generated = base_input_ids.clone()

                    for step in range(max_tokens_per_run):
                        outputs = self.model(generated, return_routing_info=True)

                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            logits = outputs[0]
                            routing = self.extractor.extract(outputs)
                        else:
                            logits = outputs
                            routing = None

                        # Sample next token
                        next_logits = logits[:, -1, :]
                        if temperature != 1.0:
                            next_logits = next_logits / temperature

                        if top_k > 0:
                            topk_vals, _ = next_logits.topk(top_k, dim=-1)
                            mask = next_logits < topk_vals[..., -1, None]
                            next_logits[mask] = float('-inf')
                            probs = F.softmax(next_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = next_logits.argmax(dim=-1, keepdim=True)

                        next_token_id = next_token.item()
                        token_text = self.tokenizer.decode([next_token_id])

                        # Extract active neurons from ALL pools in single pass
                        step_neurons_per_pool = {pool: set() for pool in pools}
                        if routing:
                            for layer_idx, layer in enumerate(routing):
                                for pool in pools:
                                    # Handle Q/K shared pools
                                    if pool in ('fqk', 'rqk'):
                                        m_q = layer.get_mask(f'{pool}_q')
                                        m_k = layer.get_mask(f'{pool}_k')
                                        if m_q is not None and m_k is not None:
                                            if m_q.dim() == 3:
                                                m_q, m_k = m_q[0, -1], m_k[0, -1]
                                            else:
                                                m_q, m_k = m_q[0], m_k[0]
                                            m = m_q | m_k  # OR combine
                                            active = m.nonzero(as_tuple=True)[0].cpu().tolist()
                                            step_neurons_per_pool[pool].update(active)
                                        else:
                                            # Fallback to weights
                                            w_q = layer.get_weight(f'{pool}_q')
                                            w_k = layer.get_weight(f'{pool}_k')
                                            if w_q is not None and w_k is not None:
                                                if w_q.dim() == 3:
                                                    w_q, w_k = w_q[0, -1], w_k[0, -1]
                                                else:
                                                    w_q, w_k = w_q[0], w_k[0]
                                                w = torch.max(w_q, w_k)
                                                active = (w > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                                                step_neurons_per_pool[pool].update(active)
                                    else:
                                        # Regular pools (fv, rv, fknow, rknow)
                                        m = layer.get_mask(pool)
                                        if m is not None:
                                            if m.dim() == 3:
                                                m = m[0, -1]
                                            else:
                                                m = m[0]
                                            active = m.nonzero(as_tuple=True)[0].cpu().tolist()
                                            step_neurons_per_pool[pool].update(active)
                                        else:
                                            w = layer.get_weight(pool)
                                            if w is not None:
                                                if w.dim() == 3:
                                                    w = w[0, -1]
                                                else:
                                                    w = w[0]
                                                active = (w > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                                                step_neurons_per_pool[pool].update(active)

                        # Check if target found (exact match)
                        if token_text.strip().lower() == target_lower:
                            successful_runs += 1

                            # Record neurons for ALL pools with unified naming
                            for pool in pools:
                                for n in step_neurons_per_pool[pool]:
                                    target_neuron_counts[pool][f'{pool}_{n}'] += 1

                            if len(sample_generations) < 3:
                                gen_text = self.tokenizer.decode(
                                    torch.cat([generated, next_token], dim=1)[0],
                                    skip_special_tokens=True
                                )
                                sample_generations.append(gen_text)

                            break
                        else:
                            # Baseline step
                            for pool in pools:
                                for n in step_neurons_per_pool[pool]:
                                    baseline_neuron_counts[pool][f'{pool}_{n}'] += 1
                            total_baseline_steps += 1

                        generated = torch.cat([generated, next_token], dim=1)

                        if next_token_id == self.tokenizer.eos_token_id:
                            break

            print(f"\r      {successful_runs}/{min_target_count} targets (run {total_runs}) - Done!          ")

            match_rate = successful_runs / total_runs if total_runs > 0 else 0
            print(f"      Match rate: {match_rate*100:.1f}% ({successful_runs}/{total_runs})")

            # Compute per-pool results
            target_result = {
                'prompt': prompt,
                'target_token': target,
                'successful_runs': successful_runs,
                'total_runs': total_runs,
                'match_rate': match_rate,
                'sample_generations': sample_generations,
                'per_pool': {},
            }

            if successful_runs > 0:
                for pool in pools:
                    pool_target_counts = target_neuron_counts[pool]
                    pool_baseline_counts = baseline_neuron_counts[pool]

                    target_freq = {n: c / successful_runs for n, c in pool_target_counts.items()}
                    # Calculate baseline frequency and contrastive scores
                    baseline_freq = {n: c / total_baseline_steps for n, c in pool_baseline_counts.items()} if total_baseline_steps > 0 else {}
                    # Contrastive = target_freq - baseline_freq (positive = target-specific)
                    all_neurons_in_pool = set(target_freq.keys()) | set(baseline_freq.keys())
                    contrastive_scores = {n: target_freq.get(n, 0) - baseline_freq.get(n, 0) for n in all_neurons_in_pool}

                    common_100 = [n for n, f in target_freq.items() if f >= 1.0]
                    common_80 = [n for n, f in target_freq.items() if f >= 0.8]

                    target_result['per_pool'][pool] = {
                        'common_100': sorted(common_100),
                        'common_80': sorted(common_80),
                        'n_unique': len(pool_target_counts),
                        'top_neurons': sorted(
                            [{'neuron': n, 'freq': f * 100} for n, f in target_freq.items()],
                            key=lambda x: -x['freq']
                        )[:20],
                        # Full frequency data for heatmap visualization
                        'all_frequencies': {n: f for n, f in target_freq.items()},
                        # Contrastive scores: positive = more active on target than baseline
                        'contrastive_scores': contrastive_scores,
                    }

                    print(f"        {pool}: {len(common_100)} neurons@100%, {len(common_80)} neurons@80%")
            else:
                target_result['note'] = f'Target "{target}" not found in {total_runs} runs'

            results['per_target'][target] = target_result

        # Aggregate per-pool summary
        for pool in pools:
            all_common_100 = set()
            all_common_80 = set()
            for target_data in results['per_target'].values():
                if 'per_pool' in target_data and pool in target_data['per_pool']:
                    all_common_100.update(target_data['per_pool'][pool].get('common_100', []))
                    all_common_80.update(target_data['per_pool'][pool].get('common_80', []))
            results['per_pool'][pool] = {
                'n_common_100': len(all_common_100),
                'n_common_80': len(all_common_80),
                'top_neurons': sorted(all_common_100)[:20],
            }

        self.extractor.disable_weight_storage()
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

        # Save results to JSON
        import json
        results_path = os.path.join(output_dir, 'behavioral_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Results saved to: {results_path}", flush=True)
        except Exception as e:
            print(f"  ERROR saving results: {e}", flush=True)

        return results

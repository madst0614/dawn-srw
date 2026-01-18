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
        n_runs: int = 10,
        pool_type: str = 'fv',
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Dict:
        """
        Analyze neuron activations for factual knowledge prompts.

        Generates tokens autoregressively and uses contrastive analysis:
        - Records neurons at EVERY step
        - When target appears, compares neurons at target step vs. other steps
        - Finds neurons that are significantly more active for target token

        Paper Figure 7 data generation.

        Args:
            prompts: List of prompts (e.g., ["The capital of France is"])
            targets: List of expected target tokens (e.g., ["Paris"])
            n_runs: Number of generation runs per prompt
            pool_type: Pool to analyze ('fv', 'rv', etc.)
            max_new_tokens: Maximum tokens to generate per run
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = greedy)

        Returns:
            Dictionary with per-target neuron frequencies and contrastive analysis
        """
        from collections import Counter, defaultdict
        import torch.nn.functional as F

        results = {
            'prompts': prompts,
            'targets': targets,
            'n_runs': n_runs,
            'pool_type': pool_type,
            'max_new_tokens': max_new_tokens,
            'per_target': {},
        }

        self.model.eval()

        # Progress tracking
        try:
            from tqdm import tqdm
            prompt_iter = tqdm(zip(prompts, targets), total=len(prompts), desc="Prompts")
        except ImportError:
            prompt_iter = zip(prompts, targets)

        for prompt_idx, (prompt, target) in enumerate(prompt_iter):
            target_neuron_counts = Counter()  # Neurons when target generated
            baseline_neuron_counts = Counter()  # Neurons at other steps
            matching_runs = 0
            total_baseline_steps = 0
            target_lower = target.strip().lower()

            first_predicted_text = None
            sample_generations = []

            # Show prompt being processed
            print(f"\n    [{prompt_idx+1}/{len(prompts)}] \"{prompt}\" → target: \"{target}\"")

            for run_idx in range(n_runs):
                # Encode prompt
                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors='pt'
                ).to(self.device)

                generated = input_ids.clone()
                found_target = False
                target_position = -1
                run_neurons_per_step = []  # Track neurons at each step

                with torch.no_grad():
                    for step in range(max_new_tokens):
                        # Forward pass with routing info
                        outputs = self.model(generated, return_routing_info=True)

                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            logits = outputs[0]
                            routing = self.extractor.extract(outputs)
                        else:
                            logits = outputs
                            routing = None

                        # Get next token logits
                        next_logits = logits[:, -1, :]
                        if temperature != 1.0:
                            next_logits = next_logits / temperature

                        # Sampling
                        if top_k > 0:
                            topk_vals, topk_indices = next_logits.topk(top_k, dim=-1)
                            mask = next_logits < topk_vals[..., -1, None]
                            next_logits[mask] = float('-inf')
                            probs = F.softmax(next_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = next_logits.argmax(dim=-1, keepdim=True)

                        next_token_id = next_token.item()
                        token_text = self.tokenizer.decode([next_token_id])

                        # Extract active neurons at this step
                        step_neurons = set()
                        if routing:
                            for layer in routing:
                                weights = layer.get_weight(pool_type)
                                if weights is not None:
                                    if weights.dim() == 3:
                                        w = weights[0, -1]
                                    else:
                                        w = weights[0]
                                    active = (w > 0.01).nonzero(as_tuple=True)[0].cpu().tolist()
                                    step_neurons.update(active)

                        run_neurons_per_step.append({
                            'step': step,
                            'token': token_text,
                            'neurons': step_neurons,
                            'is_target': False,
                        })

                        # Check if target found
                        if not found_target and target_lower in token_text.strip().lower():
                            found_target = True
                            target_position = step
                            matching_runs += 1
                            run_neurons_per_step[-1]['is_target'] = True

                        generated = torch.cat([generated, next_token], dim=1)

                        # Stop on EOS
                        if next_token_id == self.tokenizer.eos_token_id:
                            break

                        # Early stop: if target found, generate a few more tokens for context then stop
                        if found_target and step >= target_position + 3:
                            break

                # Aggregate neurons: target vs baseline
                for step_info in run_neurons_per_step:
                    if step_info['is_target']:
                        for n in step_info['neurons']:
                            target_neuron_counts[n] += 1
                    else:
                        for n in step_info['neurons']:
                            baseline_neuron_counts[n] += 1
                        total_baseline_steps += 1

                # Store generation for display
                gen_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                if run_idx == 0:
                    first_predicted_text = gen_text
                if found_target and len(sample_generations) < 3:
                    sample_generations.append({
                        'text': gen_text,
                        'position': target_position,
                    })

                # Progress update every 10 runs
                if (run_idx + 1) % 10 == 0 or run_idx == n_runs - 1:
                    match_pct = matching_runs / (run_idx + 1) * 100
                    status = f"✓{matching_runs}" if matching_runs > 0 else "✗"
                    print(f"      Run {run_idx+1:3d}/{n_runs}: {status} ({match_pct:.0f}% match rate)", end='\r')

            # Final summary for this prompt
            match_pct = matching_runs / n_runs * 100
            n_unique = len(target_neuron_counts)
            print(f"      Completed: {matching_runs}/{n_runs} found ({match_pct:.0f}%), {n_unique} unique neurons")

            # Compute contrastive scores
            # Score = (freq in target) - (freq in baseline)
            # High score = neuron is more active for target than average
            contrastive_scores = {}
            if matching_runs > 0 and total_baseline_steps > 0:
                for neuron in set(target_neuron_counts.keys()) | set(baseline_neuron_counts.keys()):
                    target_freq = target_neuron_counts[neuron] / matching_runs
                    baseline_freq = baseline_neuron_counts[neuron] / total_baseline_steps
                    contrastive_scores[neuron] = target_freq - baseline_freq

            # Store results for this target
            base_result = {
                'prompt': prompt,
                'target_token': target,
                'matching_runs': matching_runs,
                'total_runs': n_runs,
                'match_rate': matching_runs / n_runs,
                'total_baseline_steps': total_baseline_steps,
                'first_generation': first_predicted_text,
                'sample_successful_generations': sample_generations,
            }

            if matching_runs > 0:
                # Neurons consistently active for target (>80% of target occurrences)
                target_freq = {n: c / matching_runs for n, c in target_neuron_counts.items()}
                common_neurons_80 = [n for n, f in target_freq.items() if f >= 0.8]
                common_neurons_100 = [n for n, f in target_freq.items() if f >= 1.0]

                # Target-SPECIFIC neurons (high contrastive score)
                # These appear much more often for target than for other tokens
                target_specific = [
                    n for n, score in contrastive_scores.items()
                    if score >= 0.5 and target_freq.get(n, 0) >= 0.8
                ]

                # Sort by contrastive score
                sorted_contrastive = sorted(
                    contrastive_scores.items(),
                    key=lambda x: -x[1]
                )[:50]

                base_result.update({
                    'common_neurons_80': common_neurons_80,
                    'common_neurons_100': common_neurons_100,
                    'target_specific_neurons': target_specific,
                    'total_unique_neurons': len(target_neuron_counts),
                    'contrastive_top50': [
                        {
                            'neuron': n,
                            'target_freq': target_freq.get(n, 0) * 100,
                            'baseline_freq': baseline_neuron_counts[n] / total_baseline_steps * 100 if total_baseline_steps > 0 else 0,
                            'score': score,
                        }
                        for n, score in sorted_contrastive
                    ],
                })
            else:
                base_result['note'] = f'Target "{target}" not found in {n_runs} generations (max {max_new_tokens} tokens each)'

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

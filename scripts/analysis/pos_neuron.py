"""
POS-based Neuron Analysis
=========================
Analyze neuron specialization by Part-of-Speech tags.

Uses Universal Dependencies English Web Treebank to show that
different POS categories activate different neurons.

Key metric: Selectivity = E[weight|POS] / E[weight|all]
- > 1: neuron is selective for this POS
- = 1: neuron is uniform across POS
- < 1: neuron avoids this POS
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .base import BaseAnalyzer
from .utils import (
    HAS_TQDM, tqdm,
    RoutingDataExtractor,  # Schema layer for model-agnostic access
    resolve_pool_type,     # Resolve pool type aliases
)


# Universal POS tags (UPOS)
UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X',
]

# Simplified POS groups
POS_GROUPS = {
    'Content Words': ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'],
    'Function Words': ['DET', 'ADP', 'AUX', 'PRON', 'CCONJ', 'SCONJ', 'PART'],
    'Other': ['NUM', 'PUNCT', 'SYM', 'INTJ', 'X'],
}

# POS tag to index mapping
POS_TO_IDX = {pos: i for i, pos in enumerate(UPOS_TAGS)}


class POSNeuronAnalyzer(BaseAnalyzer):
    """Analyze neuron activations by POS tags using continuous weights."""

    def __init__(
        self,
        model,
        router=None,
        tokenizer=None,
        device: str = 'cuda',
        target_layer: int = None,
        extractor=None,
    ):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter (auto-detected if None)
            tokenizer: Tokenizer instance
            device: Device for computation
            target_layer: Specific layer to analyze (None = all)
            extractor: RoutingDataExtractor instance (created if None)
        """
        super().__init__(model, router=router, tokenizer=tokenizer, device=device)
        self.extractor = extractor or RoutingDataExtractor(model, device=device)
        self.target_layer = target_layer
        self.model.eval()

        # Get neuron count from router
        self.n_neurons = self._get_neuron_count()
        self.n_pos = len(UPOS_TAGS)

        # Storage for weight accumulation (will be initialized in reset_stats)
        self.weight_sum = None  # [n_pos, n_neurons]
        self.weight_count = None  # [n_pos, n_neurons]
        self.pos_token_counts = None  # [n_pos]

    def _get_neuron_count(self) -> int:
        """Get total neuron count from router."""
        if hasattr(self.router, 'neuron_emb'):
            return self.router.neuron_emb.shape[0]
        # Fallback: sum of pool sizes
        total = 0
        for attr in ['n_feature_qk', 'n_feature_v', 'n_restore_qk', 'n_restore_v',
                     'n_feature_know', 'n_restore_know']:
            total += getattr(self.router, attr, 0)
        return total if total > 0 else 4000  # Default fallback

    def reset_stats(self):
        """Reset analysis statistics."""
        # Use float32 numpy arrays for accumulation
        self.weight_sum = np.zeros((self.n_pos, self.n_neurons), dtype=np.float32)
        self.weight_count = np.zeros((self.n_pos, self.n_neurons), dtype=np.int32)
        self.pos_token_counts = np.zeros(self.n_pos, dtype=np.int32)

    def load_ud_dataset(
        self,
        split: str = 'train',
        max_sentences: int = None,
        data_path: str = None
    ) -> List[Dict]:
        """
        Load Universal Dependencies English Web Treebank.

        Args:
            split: 'train', 'dev', or 'test'
            max_sentences: Maximum sentences to load
            data_path: Path to local conllu file (optional)

        Returns:
            List of {'tokens': [...], 'upos': [...]}
        """
        ud_urls = {
            'train': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu',
            'dev': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu',
            'test': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu',
        }

        try:
            import conllu
        except ImportError:
            raise ImportError("Please install conllu: pip install conllu")

        if data_path and os.path.exists(data_path):
            print(f"Loading from local file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.read()
        else:
            import urllib.request
            url = ud_urls.get(split, ud_urls['train'])
            print(f"Downloading UD English EWT ({split})...")

            try:
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
            except Exception as e:
                print(f"Download failed: {e}")
                return self._load_nltk_treebank(max_sentences)

        print("Parsing conllu data...")
        sentences = conllu.parse(data)

        if max_sentences:
            sentences = sentences[:max_sentences]

        dataset = []
        for sent in sentences:
            tokens = [token['form'] for token in sent]
            upos = [token['upos'] for token in sent]
            dataset.append({'tokens': tokens, 'upos': upos})

        print(f"Loaded {len(dataset)} sentences")
        return dataset

    def _load_nltk_treebank(self, max_sentences: int = None) -> List[Dict]:
        """Fallback: Load NLTK treebank with universal tagset."""
        try:
            import nltk
            nltk.download('treebank', quiet=True)
            nltk.download('universal_tagset', quiet=True)
            from nltk.corpus import treebank
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")

        print("Loading NLTK treebank...")

        nltk_to_upos = {
            'NOUN': 'NOUN', 'VERB': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
            'ADP': 'ADP', 'DET': 'DET', 'PRON': 'PRON', 'NUM': 'NUM',
            'CONJ': 'CCONJ', 'PRT': 'PART', '.': 'PUNCT', 'X': 'X',
        }

        sentences = treebank.tagged_sents(tagset='universal')
        if max_sentences:
            sentences = sentences[:max_sentences]

        dataset = []
        for sent in sentences:
            tokens = [word for word, tag in sent]
            upos = [nltk_to_upos.get(tag, 'X') for word, tag in sent]
            dataset.append({'tokens': tokens, 'upos': upos})

        print(f"Loaded {len(dataset)} sentences from NLTK treebank")
        return dataset

    def get_pos_for_tokens(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
    ) -> Tuple[List[str], List[int]]:
        """
        Map DAWN tokenizer tokens to POS tags using character spans.

        Returns:
            (list of POS tags, list of token IDs)
        """
        text = ""
        ud_char_spans = []

        for ud_token, pos in zip(ud_tokens, ud_pos):
            start = len(text)
            text += ud_token
            end = len(text)
            ud_char_spans.append((start, end, pos))
            text += " "

        text = text.rstrip()

        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            token_ids = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']

            if not token_ids:
                return [], []

            dawn_pos_tags = []
            for start, end in offset_mapping:
                assigned_pos = 'X'
                for ud_start, ud_end, pos in ud_char_spans:
                    if start < ud_end and end > ud_start:
                        assigned_pos = pos
                        break
                dawn_pos_tags.append(assigned_pos)

            return dawn_pos_tags, token_ids

        except (TypeError, KeyError):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                return [], []

            dawn_pos_tags = []
            decoded_so_far = ""

            for tid in token_ids:
                token_text = self.tokenizer.decode([tid])
                decoded_so_far += token_text

                char_count = 0
                assigned_pos = 'X'
                for i, (ud_start, ud_end, pos) in enumerate(ud_char_spans):
                    char_count = ud_end + 1
                    if len(decoded_so_far.strip()) <= char_count:
                        assigned_pos = pos
                        break

                dawn_pos_tags.append(assigned_pos)

            return dawn_pos_tags, token_ids

    def extract_routing_weights(
        self,
        token_ids: List[int],
        pool_type: str = 'fv',
    ) -> Optional[np.ndarray]:
        """
        Get routing weights for each token position.

        Args:
            token_ids: List of token IDs
            pool_type: Pool to analyze ('fv', 'rv', 'fqk_q', etc.)

        Returns:
            weights: [seq_len, n_neurons] array of routing weights (averaged across layers)
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        seq_len = len(token_ids)

        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(input_ids, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return None

        # Accumulate weights across layers
        weight_sum = None
        layer_count = 0

        for layer in routing:
            if self.target_layer is not None and layer.layer_idx != self.target_layer:
                continue

            # Get weights using standardized key
            weights = layer.get_weight(pool_type)

            if weights is None:
                continue

            # weights: [B, T, N] or [B, N]
            if weights.dim() == 3:
                w = weights[0].cpu().numpy()  # [T, N]
            elif weights.dim() == 2:
                # Batch-level routing - expand to all positions
                w = weights[0].unsqueeze(0).expand(seq_len, -1).cpu().numpy()  # [T, N]
            else:
                continue

            # Truncate or pad to match sequence length
            if w.shape[0] > seq_len:
                w = w[:seq_len]
            elif w.shape[0] < seq_len:
                pad = np.zeros((seq_len - w.shape[0], w.shape[1]), dtype=np.float32)
                w = np.vstack([w, pad])

            # Check neuron dimension
            if w.shape[1] > self.n_neurons:
                w = w[:, :self.n_neurons]
            elif w.shape[1] < self.n_neurons:
                pad = np.zeros((w.shape[0], self.n_neurons - w.shape[1]), dtype=np.float32)
                w = np.hstack([w, pad])

            if weight_sum is None:
                weight_sum = w.astype(np.float32)
            else:
                weight_sum += w

            layer_count += 1

        if weight_sum is None or layer_count == 0:
            return None

        # Average across layers
        return weight_sum / layer_count

    def analyze_sentence(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
        pool_type: str = 'fv',
    ):
        """Analyze a single sentence and update statistics with continuous weights."""
        try:
            pos_tags, token_ids = self.get_pos_for_tokens(ud_tokens, ud_pos)
        except Exception:
            return

        if not token_ids:
            return

        # Get routing weights: [seq_len, n_neurons]
        weights = self.extract_routing_weights(token_ids, pool_type)

        if weights is None:
            return

        seq_len = min(len(pos_tags), weights.shape[0])

        # Vectorized accumulation per POS
        for pos_idx in range(seq_len):
            pos = pos_tags[pos_idx]
            if pos not in POS_TO_IDX:
                continue

            pos_i = POS_TO_IDX[pos]
            w = weights[pos_idx]  # [n_neurons]

            # Only count neurons with non-zero weight
            active_mask = w > 0
            if not active_mask.any():
                continue

            # Accumulate weights
            self.weight_sum[pos_i] += w
            self.weight_count[pos_i] += active_mask.astype(np.int32)
            self.pos_token_counts[pos_i] += 1

    def analyze_dataset(
        self,
        dataset: List[Dict],
        pool_type: str = 'fv',
        max_sentences: int = None,
    ) -> Dict:
        """
        Analyze full dataset.

        Args:
            dataset: List of {'tokens': [...], 'upos': [...]}
            pool_type: Pool to analyze (fv, fqk, fqk_q, fqk_k, rv, rqk, rqk_q, rqk_k, fknow, rknow)
            max_sentences: Maximum sentences to process

        Returns:
            Analysis results dictionary
        """
        # Resolve pool type alias (e.g., 'fqk' -> 'fqk_q')
        pool_type = resolve_pool_type(pool_type)

        self.reset_stats()
        n_sentences = min(len(dataset), max_sentences) if max_sentences else len(dataset)

        print(f"\nAnalyzing {n_sentences} sentences...")
        print(f"Pool: {pool_type.upper()}")
        print(f"Layer: {self.target_layer if self.target_layer is not None else 'all'}")

        for i in tqdm(range(n_sentences), desc="Processing"):
            example = dataset[i]
            try:
                self.analyze_sentence(example['tokens'], example['upos'], pool_type)
            except Exception:
                continue

        return self.get_results()

    def get_results(self) -> Dict:
        """Compile analysis results with selectivity metrics."""

        # Compute mean weight per (POS, neuron)
        # mean_weight[pos, neuron] = weight_sum / weight_count
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_weight = np.where(
                self.weight_count > 0,
                self.weight_sum / self.weight_count,
                0.0
            )

        # Compute global mean weight per neuron (across all POS)
        # global_mean[neuron] = sum(weight_sum[all_pos]) / sum(weight_count[all_pos])
        total_weight_sum = self.weight_sum.sum(axis=0)  # [n_neurons]
        total_weight_count = self.weight_count.sum(axis=0)  # [n_neurons]

        with np.errstate(divide='ignore', invalid='ignore'):
            global_mean = np.where(
                total_weight_count > 0,
                total_weight_sum / total_weight_count,
                0.0
            )

        # Compute selectivity: mean_weight[pos, neuron] / global_mean[neuron]
        # selectivity > 1: selective for this POS
        # selectivity = 1: uniform
        # selectivity < 1: avoids this POS
        with np.errstate(divide='ignore', invalid='ignore'):
            selectivity = np.where(
                global_mean > 0,
                mean_weight / global_mean,
                0.0
            )

        # Replace inf/nan with 0
        selectivity = np.nan_to_num(selectivity, nan=0.0, posinf=0.0, neginf=0.0)

        # Find highly selective neurons per POS (selectivity > threshold)
        threshold = 2.0
        selective_neurons_per_pos = {}
        for pos_i, pos in enumerate(UPOS_TAGS):
            if self.pos_token_counts[pos_i] == 0:
                continue

            sel = selectivity[pos_i]
            high_sel_mask = sel > threshold
            high_sel_indices = np.where(high_sel_mask)[0]

            if len(high_sel_indices) > 0:
                # Sort by selectivity descending
                sorted_idx = high_sel_indices[np.argsort(sel[high_sel_indices])[::-1]]
                top_neurons = [
                    {
                        'neuron': int(n),
                        'selectivity': float(sel[n]),
                        'mean_weight': float(mean_weight[pos_i, n]),
                        'occurrences': int(self.weight_count[pos_i, n]),
                    }
                    for n in sorted_idx[:20]  # Top 20
                ]
                selective_neurons_per_pos[pos] = top_neurons

        # Top neurons per POS by mean weight (for comparison with old method)
        top_neurons_per_pos = {}
        for pos_i, pos in enumerate(UPOS_TAGS):
            if self.pos_token_counts[pos_i] == 0:
                continue

            mw = mean_weight[pos_i]
            top_indices = np.argsort(mw)[::-1][:20]
            top_neurons_per_pos[pos] = [
                (int(n), float(mw[n])) for n in top_indices if mw[n] > 0
            ]

        # Neuron-level specificity analysis
        # For each neuron, find the POS with highest selectivity
        neuron_specificity = {}
        active_neurons = np.where(total_weight_count > 0)[0]

        for neuron in active_neurons:
            sel = selectivity[:, neuron]
            if sel.max() <= 1.0:
                continue  # No selectivity

            top_pos_i = int(np.argmax(sel))
            top_pos = UPOS_TAGS[top_pos_i]
            top_sel = float(sel[top_pos_i])

            if top_sel > 1.5:  # Minimum threshold for specificity
                neuron_specificity[int(neuron)] = {
                    'top_pos': top_pos,
                    'selectivity': top_sel,
                    'mean_weight': float(mean_weight[top_pos_i, neuron]),
                    'global_mean_weight': float(global_mean[neuron]),
                }

        # Sort by selectivity
        neuron_specificity = dict(
            sorted(neuron_specificity.items(), key=lambda x: -x[1]['selectivity'])[:100]
        )

        # POS similarity based on selectivity patterns
        # Cosine similarity between POS selectivity vectors
        pos_similarity = {}
        active_pos = [i for i in range(self.n_pos) if self.pos_token_counts[i] > 0]

        for i, pos_i in enumerate(active_pos):
            for pos_j in active_pos[i+1:]:
                sel_i = selectivity[pos_i]
                sel_j = selectivity[pos_j]

                # Cosine similarity
                norm_i = np.linalg.norm(sel_i)
                norm_j = np.linalg.norm(sel_j)

                if norm_i > 0 and norm_j > 0:
                    cos_sim = float(np.dot(sel_i, sel_j) / (norm_i * norm_j))
                    pos_similarity[f"{UPOS_TAGS[pos_i]}-{UPOS_TAGS[pos_j]}"] = cos_sim

        # Build selectivity matrix for output (only active POS and neurons with activity)
        active_neuron_mask = total_weight_count > 0
        n_active_neurons = active_neuron_mask.sum()

        selectivity_summary = {
            'shape': [len(active_pos), int(n_active_neurons)],
            'pos_labels': [UPOS_TAGS[i] for i in active_pos],
            'mean_selectivity_per_pos': {
                UPOS_TAGS[i]: float(selectivity[i, active_neuron_mask].mean())
                for i in active_pos
            },
            'max_selectivity_per_pos': {
                UPOS_TAGS[i]: float(selectivity[i].max())
                for i in active_pos
            },
        }

        return {
            'pos_token_counts': {
                UPOS_TAGS[i]: int(self.pos_token_counts[i])
                for i in range(self.n_pos) if self.pos_token_counts[i] > 0
            },
            'selectivity_summary': selectivity_summary,
            'selective_neurons_per_pos': selective_neurons_per_pos,
            'top_neurons_per_pos': top_neurons_per_pos,
            'neuron_specificity': {
                str(k): v for k, v in neuron_specificity.items()
            },
            'pos_similarity': pos_similarity,
            'total_neurons_analyzed': int(n_active_neurons),
            # Store raw data for visualization
            '_selectivity_matrix': selectivity,
            '_mean_weight_matrix': mean_weight,
            '_pos_indices': active_pos,
        }

    def visualize(self, results: Dict, output_dir: str) -> Dict[str, str]:
        """
        Generate all visualizations.

        Args:
            results: Results from get_results()
            output_dir: Output directory

        Returns:
            Dictionary of {plot_name: path}
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # Selectivity heatmap (new)
        path = self._plot_selectivity_heatmap(results, os.path.join(output_dir, 'selectivity_heatmap.png'))
        if path:
            paths['selectivity_heatmap'] = path

        # High-selectivity neurons per POS
        path = self._plot_selective_neurons(results, os.path.join(output_dir, 'selective_neurons.png'))
        if path:
            paths['selective_neurons'] = path

        # POS similarity matrix
        path = self._plot_pos_similarity(results, os.path.join(output_dir, 'pos_similarity.png'))
        if path:
            paths['pos_similarity'] = path

        # Legacy visualizations (adapted)
        try:
            from .visualizers.pos_neurons import (
                plot_pos_heatmap, plot_pos_clustering,
                plot_top_neurons_by_pos, plot_pos_specificity
            )

            path = plot_pos_heatmap(results, os.path.join(output_dir, 'pos_neuron_heatmap.png'))
            if path:
                paths['heatmap'] = path

            path = plot_pos_clustering(results, os.path.join(output_dir, 'pos_clustering.png'))
            if path:
                paths['clustering'] = path

            path = plot_top_neurons_by_pos(results, os.path.join(output_dir, 'top_neurons_by_pos.png'))
            if path:
                paths['top_neurons'] = path

            path = plot_pos_specificity(results, os.path.join(output_dir, 'neuron_specificity.png'))
            if path:
                paths['specificity'] = path
        except ImportError:
            pass

        return paths

    def _plot_selectivity_heatmap(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot selectivity heatmap: POS x Neuron clusters."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        selectivity = results.get('_selectivity_matrix')
        if selectivity is None:
            return None

        pos_indices = results.get('_pos_indices', list(range(self.n_pos)))
        pos_labels = [UPOS_TAGS[i] for i in pos_indices]

        # Filter to active neurons and sample if too many
        active_mask = selectivity.max(axis=0) > 0
        selectivity_active = selectivity[np.ix_(pos_indices, active_mask)]

        n_neurons = selectivity_active.shape[1]
        if n_neurons > 200:
            # Sample neurons with highest max selectivity
            max_sel = selectivity_active.max(axis=0)
            top_indices = np.argsort(max_sel)[-200:]
            selectivity_active = selectivity_active[:, top_indices]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Clip for better visualization
        sel_clipped = np.clip(selectivity_active, 0, 5)

        im = ax.imshow(sel_clipped, aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=5)
        ax.set_yticks(range(len(pos_labels)))
        ax.set_yticklabels(pos_labels)
        ax.set_xlabel('Neuron (top 200 by max selectivity)')
        ax.set_ylabel('POS')
        ax.set_title('POS Selectivity (E[weight|POS] / E[weight|all])\n'
                     'Red: selective (>1), Blue: avoids (<1), White: uniform (=1)')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Selectivity')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_selective_neurons(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot high-selectivity neurons per POS."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        selective = results.get('selective_neurons_per_pos', {})
        if not selective:
            return None

        # Prepare data
        pos_list = list(selective.keys())
        if not pos_list:
            return None

        fig, axes = plt.subplots(2, (len(pos_list) + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()

        for i, pos in enumerate(pos_list[:len(axes)]):
            ax = axes[i]
            neurons = selective[pos][:10]  # Top 10

            if not neurons:
                ax.set_visible(False)
                continue

            neuron_ids = [n['neuron'] for n in neurons]
            selectivities = [n['selectivity'] for n in neurons]

            ax.barh(range(len(neuron_ids)), selectivities, color='steelblue')
            ax.set_yticks(range(len(neuron_ids)))
            ax.set_yticklabels([f'N{n}' for n in neuron_ids])
            ax.set_xlabel('Selectivity')
            ax.set_title(f'{pos}')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=2, color='orange', linestyle='--', alpha=0.5)

        # Hide unused axes
        for i in range(len(pos_list), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('High-Selectivity Neurons per POS (selectivity > 2.0)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_pos_similarity(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot POS similarity matrix based on selectivity patterns."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        pos_similarity = results.get('pos_similarity', {})
        if not pos_similarity:
            return None

        # Build similarity matrix
        active_pos = list(results.get('pos_token_counts', {}).keys())
        n = len(active_pos)

        sim_matrix = np.eye(n)
        for key, val in pos_similarity.items():
            parts = key.split('-')
            if len(parts) == 2:
                try:
                    i = active_pos.index(parts[0])
                    j = active_pos.index(parts[1])
                    sim_matrix[i, j] = val
                    sim_matrix[j, i] = val
                except ValueError:
                    continue

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(active_pos, rotation=45, ha='right')
        ax.set_yticklabels(active_pos)
        ax.set_title('POS Similarity (based on selectivity patterns)')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def run_all(
        self,
        output_dir: str = './pos_analysis',
        pool_type: str = 'fv',
        max_sentences: int = 2000,
        split: str = 'train',
        data_path: str = None,
    ) -> Dict:
        """
        Run full POS neuron analysis.

        Args:
            output_dir: Output directory
            pool_type: Pool to analyze
            max_sentences: Maximum sentences
            split: Dataset split
            data_path: Path to local conllu file

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = self.load_ud_dataset(split, max_sentences, data_path)

        # Analyze
        results = self.analyze_dataset(dataset, pool_type, max_sentences)

        # Remove internal matrices before saving JSON
        results_for_json = {
            k: v for k, v in results.items()
            if not k.startswith('_')
        }

        # Visualize
        viz_paths = self.visualize(results, output_dir)
        results_for_json['visualizations'] = viz_paths

        # Save results
        import json
        results_path = os.path.join(output_dir, f'pos_analysis_{pool_type}.json')
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)

        # Keep internal matrices in returned results for further analysis
        results['visualizations'] = viz_paths
        return results

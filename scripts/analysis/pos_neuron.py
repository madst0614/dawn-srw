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
                np.nan  # Use NaN for missing values to exclude from mean
            )

        # Compute selectivity using average of per-POS means (treats each POS equally)
        # neuron_avg[neuron] = mean of mean_weight[:, neuron] across active POS
        # selectivity[pos, neuron] = mean_weight[pos, neuron] / neuron_avg[neuron]
        with np.errstate(divide='ignore', invalid='ignore'):
            # Use nanmean to ignore POS where neuron wasn't active
            neuron_avg = np.nanmean(mean_weight, axis=0)  # [n_neurons]

            selectivity = np.where(
                neuron_avg > 0,
                mean_weight / neuron_avg,
                0.0
            )

        # Replace NaN in mean_weight with 0 for output
        mean_weight = np.nan_to_num(mean_weight, nan=0.0)

        # Replace inf/nan in selectivity with 0
        selectivity = np.nan_to_num(selectivity, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute global stats for reference
        total_weight_count = self.weight_count.sum(axis=0)  # [n_neurons]
        active_neuron_mask = total_weight_count > 0

        # Find top neurons per POS with detailed statistics
        # Specialist = selectivity > 2.0 AND mean_weight > 0.1
        SPECIALIST_SEL_THRESHOLD = 2.0
        SPECIALIST_MW_THRESHOLD = 0.1

        top_neurons_per_pos = {}
        pos_specialist_stats = {}  # Summary stats per POS

        for pos_i, pos in enumerate(UPOS_TAGS):
            if self.pos_token_counts[pos_i] == 0:
                continue

            mw = mean_weight[pos_i]
            sel = selectivity[pos_i]

            # Get active neurons for this POS
            active_mask = mw > 0
            if not active_mask.any():
                continue

            # Count neurons at different thresholds
            n_sel_gt_1_5 = int(((sel > 1.5) & active_mask).sum())
            n_sel_gt_2 = int(((sel > SPECIALIST_SEL_THRESHOLD) & active_mask).sum())
            n_specialists = int(((sel > SPECIALIST_SEL_THRESHOLD) & (mw > SPECIALIST_MW_THRESHOLD)).sum())

            pos_specialist_stats[pos] = {
                'n_active': int(active_mask.sum()),
                'n_sel_gt_1_5': n_sel_gt_1_5,
                'n_sel_gt_2': n_sel_gt_2,
                'n_specialists': n_specialists,  # sel > 2.0 AND mw > 0.1
            }

            # Combined filter for top neurons: selectivity > 1.5 AND meaningful weight
            active_weights = mw[active_mask]
            weight_threshold = np.percentile(active_weights, 75) if len(active_weights) > 4 else 0
            combined_mask = (sel > 1.5) & (mw >= weight_threshold)
            high_neurons = np.where(combined_mask)[0]

            if len(high_neurons) > 0:
                # Sort by selectivity * mean_weight (balance both)
                scores = sel[high_neurons] * mw[high_neurons]
                sorted_idx = high_neurons[np.argsort(scores)[::-1]]

                # Mark specialists
                top_neurons_per_pos[pos] = [
                    {
                        'neuron': int(n),
                        'selectivity': float(sel[n]),
                        'mean_weight': float(mw[n]),
                        'score': float(sel[n] * mw[n]),
                        'occurrences': int(self.weight_count[pos_i, n]),
                        'is_specialist': bool(sel[n] > SPECIALIST_SEL_THRESHOLD and mw[n] > SPECIALIST_MW_THRESHOLD),
                    }
                    for n in sorted_idx[:20]  # Top 20
                ]

        # Neuron-level analysis: for each neuron, find its most selective POS
        neuron_specificity = {}
        active_neurons = np.where(active_neuron_mask)[0]

        for neuron in active_neurons:
            sel = selectivity[:, neuron]
            mw = mean_weight[:, neuron]

            # Skip if no selectivity
            if np.nanmax(sel) <= 1.0:
                continue

            top_pos_i = int(np.nanargmax(sel))
            top_pos = UPOS_TAGS[top_pos_i]
            top_sel = float(sel[top_pos_i])
            top_mw = float(mw[top_pos_i])

            if top_sel > 1.5 and top_mw > 0:
                neuron_specificity[int(neuron)] = {
                    'top_pos': top_pos,
                    'selectivity': top_sel,
                    'mean_weight': top_mw,
                    'neuron_avg': float(neuron_avg[neuron]),
                }

        # Sort by selectivity descending, keep top 100
        neuron_specificity = dict(
            sorted(neuron_specificity.items(), key=lambda x: -x[1]['selectivity'])[:100]
        )

        # POS similarity based on selectivity patterns
        pos_similarity = {}
        active_pos = [i for i in range(self.n_pos) if self.pos_token_counts[i] > 0]

        for i, pos_i in enumerate(active_pos):
            for pos_j in active_pos[i+1:]:
                sel_i = selectivity[pos_i]
                sel_j = selectivity[pos_j]

                norm_i = np.linalg.norm(sel_i)
                norm_j = np.linalg.norm(sel_j)

                if norm_i > 0 and norm_j > 0:
                    cos_sim = float(np.dot(sel_i, sel_j) / (norm_i * norm_j))
                    pos_similarity[f"{UPOS_TAGS[pos_i]}-{UPOS_TAGS[pos_j]}"] = cos_sim

        # Selectivity summary
        n_active_neurons = int(active_neuron_mask.sum())
        selectivity_summary = {
            'shape': [len(active_pos), n_active_neurons],
            'pos_labels': [UPOS_TAGS[i] for i in active_pos],
            'mean_selectivity_per_pos': {
                UPOS_TAGS[i]: float(np.nanmean(selectivity[i, active_neuron_mask]))
                for i in active_pos
            },
            'max_selectivity_per_pos': {
                UPOS_TAGS[i]: float(np.nanmax(selectivity[i]))
                for i in active_pos
            },
            'mean_weight_per_pos': {
                UPOS_TAGS[i]: float(np.nanmean(mean_weight[i, active_neuron_mask]))
                for i in active_pos
            },
        }

        return {
            'pos_token_counts': {
                UPOS_TAGS[i]: int(self.pos_token_counts[i])
                for i in range(self.n_pos) if self.pos_token_counts[i] > 0
            },
            'selectivity_summary': selectivity_summary,
            'pos_specialist_stats': pos_specialist_stats,
            'top_neurons_per_pos': top_neurons_per_pos,
            'neuron_specificity': {
                str(k): v for k, v in neuron_specificity.items()
            },
            'pos_similarity': pos_similarity,
            'total_neurons_analyzed': n_active_neurons,
            # Store raw matrices for visualization
            '_selectivity_matrix': selectivity,
            '_mean_weight_matrix': mean_weight,
            '_pos_indices': active_pos,
        }

    def print_summary(self, results: Dict):
        """Print detailed summary of POS neuron analysis."""
        print("\n" + "=" * 70)
        print("POS NEURON SELECTIVITY ANALYSIS")
        print("=" * 70)

        # 1. Summary statistics table
        stats = results.get('pos_specialist_stats', {})
        if stats:
            print("\n┌─ Summary: Specialist Neurons per POS ─────────────────────────────────┐")
            print("│  Specialist = selectivity > 2.0 AND mean_weight > 0.1")
            print("│")
            print(f"│  {'POS':<8} {'Tokens':>8} {'Active':>8} {'Sel>1.5':>8} {'Sel>2.0':>8} {'Specialist':>10}")
            print(f"│  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

            pos_counts = results.get('pos_token_counts', {})
            total_specialists = 0
            for pos in UPOS_TAGS:
                if pos in stats:
                    s = stats[pos]
                    n_tokens = pos_counts.get(pos, 0)
                    print(f"│  {pos:<8} {n_tokens:>8} {s['n_active']:>8} {s['n_sel_gt_1_5']:>8} {s['n_sel_gt_2']:>8} {s['n_specialists']:>10}")
                    total_specialists += s['n_specialists']

            print(f"│  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
            print(f"│  {'TOTAL':<8} {'':<8} {'':<8} {'':<8} {'':<8} {total_specialists:>10}")
            print("└───────────────────────────────────────────────────────────────────────┘")

        # 2. Top neurons per POS (detailed table)
        top_neurons = results.get('top_neurons_per_pos', {})
        if top_neurons:
            print("\n┌─ Top Neurons per POS (selectivity > 1.5, top 25% weight) ────────────┐")

            for pos in UPOS_TAGS:
                if pos not in top_neurons:
                    continue

                neurons = top_neurons[pos][:10]  # Show top 10
                if not neurons:
                    continue

                n_specialists = sum(1 for n in neurons if n.get('is_specialist', False))
                print(f"│")
                print(f"│  {pos} ({n_specialists} specialists in top 10)")
                print(f"│  {'Neuron':<10} {'Selectivity':>12} {'Mean Weight':>12} {'Occurrences':>12} {'Specialist':>10}")
                print(f"│  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

                for n in neurons:
                    specialist_mark = "★" if n.get('is_specialist', False) else ""
                    print(f"│  N{n['neuron']:<9} {n['selectivity']:>12.3f} {n['mean_weight']:>12.4f} {n['occurrences']:>12} {specialist_mark:>10}")

            print("└───────────────────────────────────────────────────────────────────────┘")

        # 3. Overall statistics
        summary = results.get('selectivity_summary', {})
        if summary:
            print("\n┌─ Mean Selectivity & Weight per POS ──────────────────────────────────┐")
            mean_sel = summary.get('mean_selectivity_per_pos', {})
            max_sel = summary.get('max_selectivity_per_pos', {})
            mean_mw = summary.get('mean_weight_per_pos', {})

            print(f"│  {'POS':<8} {'Mean Sel':>10} {'Max Sel':>10} {'Mean Weight':>12}")
            print(f"│  {'─'*8} {'─'*10} {'─'*10} {'─'*12}")

            for pos in UPOS_TAGS:
                if pos in mean_sel:
                    print(f"│  {pos:<8} {mean_sel[pos]:>10.3f} {max_sel.get(pos, 0):>10.3f} {mean_mw.get(pos, 0):>12.5f}")

            print("└───────────────────────────────────────────────────────────────────────┘")

        print("\n" + "=" * 70)

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

        # Legacy visualizations - only call plot_pos_specificity (compatible with new format)
        try:
            from .visualizers.pos_neurons import plot_pos_specificity

            path = plot_pos_specificity(results, os.path.join(output_dir, 'neuron_specificity.png'))
            if path:
                paths['specificity'] = path
        except (ImportError, KeyError, Exception):
            pass  # Legacy visualizers may fail with new data format

        return paths

    def _plot_selectivity_heatmap(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot selectivity heatmap: POS x Neuron."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        selectivity = results.get('_selectivity_matrix')
        mean_weight = results.get('_mean_weight_matrix')
        if selectivity is None:
            return None

        pos_indices = results.get('_pos_indices', list(range(self.n_pos)))
        pos_labels = [UPOS_TAGS[i] for i in pos_indices]

        # Filter to active neurons and sample if too many
        active_mask = selectivity.max(axis=0) > 0
        selectivity_active = selectivity[np.ix_(pos_indices, active_mask)]
        mean_weight_active = mean_weight[np.ix_(pos_indices, active_mask)] if mean_weight is not None else None

        n_neurons = selectivity_active.shape[1]
        if n_neurons > 200:
            # Sample neurons with highest max selectivity
            max_sel = selectivity_active.max(axis=0)
            top_indices = np.argsort(max_sel)[-200:]
            selectivity_active = selectivity_active[:, top_indices]
            if mean_weight_active is not None:
                mean_weight_active = mean_weight_active[:, top_indices]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: Selectivity heatmap
        ax = axes[0]
        sel_clipped = np.clip(selectivity_active, 0, 5)
        im1 = ax.imshow(sel_clipped, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=5)
        ax.set_yticks(range(len(pos_labels)))
        ax.set_yticklabels(pos_labels)
        ax.set_xlabel('Neuron (top 200 by max selectivity)')
        ax.set_ylabel('POS')
        ax.set_title('Selectivity = mean_weight[pos] / avg(mean_weight)\n'
                     'Red: selective (>1), Blue: avoids (<1)')
        plt.colorbar(im1, ax=ax, label='Selectivity')

        # Right: Mean weight heatmap
        ax = axes[1]
        if mean_weight_active is not None:
            im2 = ax.imshow(mean_weight_active, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(pos_labels)))
            ax.set_yticklabels(pos_labels)
            ax.set_xlabel('Neuron (top 200 by max selectivity)')
            ax.set_ylabel('POS')
            ax.set_title('Mean Weight (activation strength)')
            plt.colorbar(im2, ax=ax, label='Mean Weight')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_selective_neurons(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot top neurons per POS (high selectivity AND high mean_weight)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        top_neurons = results.get('top_neurons_per_pos', {})
        if not top_neurons:
            return None

        pos_list = list(top_neurons.keys())
        if not pos_list:
            return None

        n_cols = min(4, len(pos_list))
        n_rows = (len(pos_list) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, pos in enumerate(pos_list):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            neurons = top_neurons[pos][:10]  # Top 10

            if not neurons:
                ax.set_visible(False)
                continue

            neuron_ids = [n['neuron'] for n in neurons]
            selectivities = [n['selectivity'] for n in neurons]
            mean_weights = [n['mean_weight'] for n in neurons]

            # Bar chart with selectivity, color by mean_weight
            colors = plt.cm.viridis([mw / max(mean_weights) for mw in mean_weights])
            bars = ax.barh(range(len(neuron_ids)), selectivities, color=colors)
            ax.set_yticks(range(len(neuron_ids)))
            ax.set_yticklabels([f'N{n}' for n in neuron_ids], fontsize=8)
            ax.set_xlabel('Selectivity')
            ax.set_title(f'{pos}', fontsize=10)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)

        # Hide unused axes
        for i in range(len(pos_list), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle('Top Neurons per POS (selectivity > 1.5 AND top 25% weight)\nColor = mean_weight', fontsize=12)
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

        # Print detailed summary
        self.print_summary(results)

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

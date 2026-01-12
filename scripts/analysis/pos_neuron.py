"""
POS-based Neuron Analysis
=========================
Analyze neuron specialization by Part-of-Speech tags.

Uses Universal Dependencies English Web Treebank to show that
different POS categories activate different neurons.
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


class POSNeuronAnalyzer(BaseAnalyzer):
    """Analyze neuron activations by POS tags."""

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

        # Storage for analysis
        self.pos_neuron_counts = defaultdict(lambda: defaultdict(int))
        self.pos_total_tokens = defaultdict(int)
        self.layer_pos_neurons = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def reset_stats(self):
        """Reset analysis statistics."""
        self.pos_neuron_counts = defaultdict(lambda: defaultdict(int))
        self.pos_total_tokens = defaultdict(int)
        self.layer_pos_neurons = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

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

    def extract_routing_for_tokens(
        self,
        token_ids: List[int],
        pool_type: str = 'fv',
    ) -> Dict[int, List[int]]:
        """
        Get routing indices for each token position.

        Args:
            token_ids: List of token IDs
            pool_type: Pool to analyze ('fv', 'rv', 'fqk_q', etc.) - uses standardized keys

        Returns:
            {position: [neuron_indices]}
        """
        input_ids = torch.tensor([token_ids], device=self.device)

        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(input_ids, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return {}

        position_neurons = defaultdict(set)
        seq_len = input_ids.shape[1]

        # pool_type is already a standardized key, extractor handles mapping
        for layer in routing:
            if self.target_layer is not None and layer.layer_idx != self.target_layer:
                continue

            # Get weights using standardized key
            weights = layer.get_weight(pool_type)

            if weights is not None:
                # weights: [B, T, N] or [B, N]
                if weights.dim() == 3:
                    for pos in range(min(seq_len, weights.shape[1])):
                        w = weights[0, pos]  # [N]
                        active_neurons = (w > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                        position_neurons[pos].update(active_neurons)
                elif weights.dim() == 2:
                    # Batch-level routing - same for all positions
                    w = weights[0]  # [N]
                    active_neurons = (w > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    for pos in range(seq_len):
                        position_neurons[pos].update(active_neurons)

        return {pos: list(neurons) for pos, neurons in position_neurons.items()}

    def analyze_sentence(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
        pool_type: str = 'fv',
    ):
        """Analyze a single sentence and update statistics."""
        try:
            pos_tags, token_ids = self.get_pos_for_tokens(ud_tokens, ud_pos)
        except Exception:
            return

        if not token_ids:
            return

        position_neurons = self.extract_routing_for_tokens(token_ids, pool_type)

        for pos_idx, pos in enumerate(pos_tags):
            neurons = position_neurons.get(pos_idx, [])
            self.pos_total_tokens[pos] += 1
            for neuron in neurons:
                self.pos_neuron_counts[pos][neuron] += 1

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
            pool_type: Pool to analyze
            max_sentences: Maximum sentences to process

        Returns:
            Analysis results dictionary
        """
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
        """Compile analysis results."""
        # Per-POS neuron frequency
        pos_neuron_freq = {}
        for pos in UPOS_TAGS:
            if self.pos_total_tokens[pos] > 0:
                neuron_counts = self.pos_neuron_counts[pos]
                total = self.pos_total_tokens[pos]
                freq = {neuron: count / total for neuron, count in neuron_counts.items()}
                pos_neuron_freq[pos] = freq

        # Top neurons per POS
        top_neurons_per_pos = {}
        for pos, freq in pos_neuron_freq.items():
            sorted_neurons = sorted(freq.items(), key=lambda x: -x[1])[:20]
            top_neurons_per_pos[pos] = sorted_neurons

        # Find POS-specific neurons
        all_neurons = set()
        for freq in pos_neuron_freq.values():
            all_neurons.update(freq.keys())

        neuron_specificity = {}
        for neuron in all_neurons:
            scores = []
            for pos in UPOS_TAGS:
                if pos in pos_neuron_freq:
                    scores.append((pos, pos_neuron_freq[pos].get(neuron, 0)))

            if scores:
                scores.sort(key=lambda x: -x[1])
                top_pos, top_score = scores[0]
                if len(scores) > 1:
                    second_score = scores[1][1]
                    specificity = top_score / (second_score + 1e-6)
                else:
                    specificity = float('inf')

                if top_score > 0.1:
                    neuron_specificity[neuron] = {
                        'top_pos': top_pos,
                        'top_score': top_score,
                        'specificity': min(specificity, 100),
                    }

        # POS overlap matrix
        overlap_matrix = {}
        for pos1 in UPOS_TAGS:
            if pos1 not in pos_neuron_freq:
                continue
            neurons1 = set(n for n, f in pos_neuron_freq[pos1].items() if f > 0.1)

            for pos2 in UPOS_TAGS:
                if pos2 not in pos_neuron_freq:
                    continue
                neurons2 = set(n for n, f in pos_neuron_freq[pos2].items() if f > 0.1)

                if neurons1 and neurons2:
                    overlap = len(neurons1 & neurons2)
                    union = len(neurons1 | neurons2)
                    jaccard = overlap / union if union > 0 else 0
                    overlap_matrix[f"{pos1}-{pos2}"] = jaccard

        return {
            'pos_token_counts': dict(self.pos_total_tokens),
            'pos_neuron_freq': {
                pos: {str(k): v for k, v in freq.items()}
                for pos, freq in pos_neuron_freq.items()
            },
            'top_neurons_per_pos': {
                pos: [(int(n), f) for n, f in neurons]
                for pos, neurons in top_neurons_per_pos.items()
            },
            'neuron_specificity': {
                str(k): v for k, v in sorted(
                    neuron_specificity.items(),
                    key=lambda x: -x[1]['specificity']
                )[:50]
            },
            'overlap_matrix': overlap_matrix,
            'total_neurons_seen': len(all_neurons),
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
        from .visualizers.pos_neurons import (
            plot_pos_heatmap, plot_pos_clustering,
            plot_top_neurons_by_pos, plot_pos_specificity
        )

        os.makedirs(output_dir, exist_ok=True)
        paths = {}

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

        return paths

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

        # Visualize
        viz_paths = self.visualize(results, output_dir)
        results['visualizations'] = viz_paths

        # Save results
        import json
        results_path = os.path.join(output_dir, f'pos_analysis_{pool_type}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

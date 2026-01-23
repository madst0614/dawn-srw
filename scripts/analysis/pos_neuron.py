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
    POOL_DISPLAY_NAMES,    # Pool shorthand to display name mapping
    get_neuron_display_name,  # Unified neuron naming
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

# Universal Dependency Relations (deprel) - core relations for analysis
DEPREL_TAGS = [
    'nsubj',    # nominal subject
    'obj',      # object (direct)
    'iobj',     # indirect object
    'obl',      # oblique nominal
    'amod',     # adjectival modifier
    'advmod',   # adverbial modifier
    'det',      # determiner
    'case',     # case marking (preposition)
    'mark',     # subordinating conjunction
    'cc',       # coordinating conjunction
    'conj',     # conjunct
    'nmod',     # nominal modifier
    'root',     # root of sentence
    'aux',      # auxiliary
    'cop',      # copula
    'compound', # compound
    'flat',     # flat (names)
    'punct',    # punctuation
]

# Deprel to index mapping
DEPREL_TO_IDX = {dep: i for i, dep in enumerate(DEPREL_TAGS)}

# Deprel groups for analysis
DEPREL_GROUPS = {
    'Core Arguments': ['nsubj', 'obj', 'iobj'],
    'Modifiers': ['amod', 'advmod', 'nmod', 'obl'],
    'Function': ['det', 'case', 'mark', 'cc', 'aux', 'cop'],
    'Structure': ['root', 'conj', 'compound', 'flat'],
    'Other': ['punct'],
}


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

        # Co-activation storage (online covariance computation)
        self.coact_sum_xy = None  # [n_neurons, n_neurons] - sum of w_i * w_j
        self.coact_sum_x = None   # [n_neurons] - sum of w_i
        self.coact_sum_x2 = None  # [n_neurons] - sum of w_i^2
        self.coact_count = 0      # number of tokens

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

    def reset_stats(self, compute_coactivation: bool = False):
        """Reset analysis statistics.

        Args:
            compute_coactivation: Whether to compute co-activation statistics
        """
        # Use float32 numpy arrays for accumulation
        self.weight_sum = np.zeros((self.n_pos, self.n_neurons), dtype=np.float32)
        self.weight_count = np.zeros((self.n_pos, self.n_neurons), dtype=np.int32)
        self.pos_token_counts = np.zeros(self.n_pos, dtype=np.int32)

        # Co-activation storage (only allocate if needed - memory intensive)
        self.compute_coactivation = compute_coactivation
        if compute_coactivation:
            # Use float64 for numerical stability in covariance
            self.coact_sum_xy = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float64)
            self.coact_sum_x = np.zeros(self.n_neurons, dtype=np.float64)
            self.coact_sum_x2 = np.zeros(self.n_neurons, dtype=np.float64)
            self.coact_count = 0
        else:
            self.coact_sum_xy = None
            self.coact_sum_x = None
            self.coact_sum_x2 = None
            self.coact_count = 0

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
            # Extract dependency relations (deprel)
            deprel = [token['deprel'] for token in sent]
            dataset.append({'tokens': tokens, 'upos': upos, 'deprel': deprel})

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
            # NLTK treebank doesn't have deprel, use empty
            deprel = ['_'] * len(tokens)
            dataset.append({'tokens': tokens, 'upos': upos, 'deprel': deprel})

        print(f"Loaded {len(dataset)} sentences from NLTK treebank (no deprel)")
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

    def get_tags_for_tokens(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
        ud_deprel: List[str],
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Map DAWN tokenizer tokens to POS tags and dependency relations.

        Args:
            ud_tokens: UD tokens
            ud_pos: UD POS tags
            ud_deprel: UD dependency relations

        Returns:
            (list of POS tags, list of deprel tags, list of token IDs)
        """
        text = ""
        ud_char_spans = []

        for ud_token, pos, dep in zip(ud_tokens, ud_pos, ud_deprel):
            start = len(text)
            text += ud_token
            end = len(text)
            ud_char_spans.append((start, end, pos, dep))
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
                return [], [], []

            dawn_pos_tags = []
            dawn_deprel_tags = []
            for start, end in offset_mapping:
                assigned_pos = 'X'
                assigned_dep = '_'
                for ud_start, ud_end, pos, dep in ud_char_spans:
                    if start < ud_end and end > ud_start:
                        assigned_pos = pos
                        assigned_dep = dep
                        break
                dawn_pos_tags.append(assigned_pos)
                dawn_deprel_tags.append(assigned_dep)

            return dawn_pos_tags, dawn_deprel_tags, token_ids

        except (TypeError, KeyError):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                return [], [], []

            dawn_pos_tags = []
            dawn_deprel_tags = []
            decoded_so_far = ""

            for tid in token_ids:
                token_text = self.tokenizer.decode([tid])
                decoded_so_far += token_text

                char_count = 0
                assigned_pos = 'X'
                assigned_dep = '_'
                for i, (ud_start, ud_end, pos, dep) in enumerate(ud_char_spans):
                    char_count = ud_end + 1
                    if len(decoded_so_far.strip()) <= char_count:
                        assigned_pos = pos
                        assigned_dep = dep
                        break

                dawn_pos_tags.append(assigned_pos)
                dawn_deprel_tags.append(assigned_dep)

            return dawn_pos_tags, dawn_deprel_tags, token_ids

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
        weights = weights[:seq_len]

        # Build POS index array for vectorized accumulation
        pos_indices = np.array([
            POS_TO_IDX.get(pos_tags[i], -1) for i in range(seq_len)
        ], dtype=np.int32)

        # Vectorized POS weight accumulation using np.add.at
        valid_mask = pos_indices >= 0
        valid_pos = pos_indices[valid_mask]
        valid_weights = weights[valid_mask]

        if len(valid_pos) > 0:
            # Active mask per token: [n_valid, n_neurons]
            active_masks = (valid_weights > 0).astype(np.int32)

            # Accumulate using np.add.at for efficient scatter-add
            np.add.at(self.weight_sum, valid_pos, valid_weights)
            np.add.at(self.weight_count, valid_pos, active_masks)
            np.add.at(self.pos_token_counts, valid_pos, 1)

        # Co-activation: vectorized outer product accumulation
        if self.compute_coactivation and weights.shape[0] > 0:
            # Batch outer product: sum over all tokens in sentence
            # Using einsum for efficient batch outer product
            # weights: [T, N] -> sum of w_i * w_j for all T
            self.coact_sum_xy += np.einsum('ti,tj->ij', weights, weights)  # [N, N]
            self.coact_sum_x += weights.sum(axis=0)  # [N]
            self.coact_sum_x2 += (weights ** 2).sum(axis=0)  # [N]
            self.coact_count += weights.shape[0]

    def analyze_dataset(
        self,
        dataset: List[Dict],
        pool_type: str = 'fv',
        max_sentences: int = None,
        compute_coactivation: bool = False,
    ) -> Dict:
        """
        Analyze full dataset.

        Args:
            dataset: List of {'tokens': [...], 'upos': [...]}
            pool_type: Pool to analyze (fv, fqk, fqk_q, fqk_k, rv, rqk, rqk_q, rqk_k, fknow, rknow)
            max_sentences: Maximum sentences to process
            compute_coactivation: Whether to compute neuron co-activation correlation

        Returns:
            Analysis results dictionary
        """
        # Resolve pool type alias (e.g., 'fqk' -> 'fqk_q')
        pool_type = resolve_pool_type(pool_type)

        self.reset_stats(compute_coactivation=compute_coactivation)
        n_sentences = min(len(dataset), max_sentences) if max_sentences else len(dataset)

        print(f"\nAnalyzing {n_sentences} sentences...")
        print(f"Pool: {pool_type.upper()}")
        print(f"Layer: {self.target_layer if self.target_layer is not None else 'all'}")
        if compute_coactivation:
            print("Co-activation analysis: ENABLED")

        for i in tqdm(range(n_sentences), desc="Processing"):
            example = dataset[i]
            try:
                self.analyze_sentence(example['tokens'], example['upos'], pool_type)
            except Exception:
                continue

        return self.get_results()

    def get_coactivation_matrix(self, min_occurrences: int = 10) -> Optional[Dict]:
        """
        Compute neuron co-activation correlation matrix from accumulated statistics.

        Uses online covariance formula:
            Cov[X,Y] = E[XY] - E[X]E[Y]
            Corr[X,Y] = Cov[X,Y] / sqrt(Var[X] * Var[Y])

        Args:
            min_occurrences: Minimum token count to consider (for statistical significance)

        Returns:
            Dictionary with correlation matrix and highly correlated pairs, or None
        """
        if self.coact_sum_xy is None or self.coact_count < min_occurrences:
            return None

        n = self.coact_count

        # Vectorized computation of correlation matrix
        # E[X], E[Y]
        mean_x = self.coact_sum_x / n  # [N]

        # E[XY]
        mean_xy = self.coact_sum_xy / n  # [N, N]

        # E[X^2]
        mean_x2 = self.coact_sum_x2 / n  # [N]

        # Var[X] = E[X^2] - E[X]^2
        var_x = mean_x2 - mean_x ** 2  # [N]
        var_x = np.maximum(var_x, 1e-10)  # Avoid division by zero

        # Cov[X,Y] = E[XY] - E[X]E[Y]  (using outer product for E[X]E[Y])
        cov_xy = mean_xy - np.outer(mean_x, mean_x)  # [N, N]

        # Corr[X,Y] = Cov[X,Y] / sqrt(Var[X] * Var[Y])
        std_x = np.sqrt(var_x)  # [N]
        std_outer = np.outer(std_x, std_x)  # [N, N]

        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.where(
                std_outer > 1e-10,
                cov_xy / std_outer,
                0.0
            )

        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)

        # Replace NaN/Inf
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip to [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Find highly correlated pairs (excluding diagonal)
        # Use upper triangle to avoid duplicates
        upper_tri = np.triu(corr_matrix, k=1)

        # Get pairs with |correlation| > 0.5
        high_corr_mask = np.abs(upper_tri) > 0.5
        high_corr_indices = np.where(high_corr_mask)

        high_corr_pairs = []
        for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
            high_corr_pairs.append({
                'neuron_i': int(i),
                'neuron_j': int(j),
                'correlation': float(corr_matrix[i, j])
            })

        # Sort by absolute correlation descending
        high_corr_pairs.sort(key=lambda x: -abs(x['correlation']))

        # Summary statistics
        off_diag = upper_tri[upper_tri != 0]
        summary = {
            'n_tokens': int(n),
            'n_neurons': int(corr_matrix.shape[0]),
            'mean_correlation': float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
            'std_correlation': float(np.std(off_diag)) if len(off_diag) > 0 else 0.0,
            'n_high_corr_pairs': len(high_corr_pairs),
            'n_positive_corr': int(np.sum(upper_tri > 0.3)),
            'n_negative_corr': int(np.sum(upper_tri < -0.3)),
        }

        return {
            'summary': summary,
            'high_corr_pairs': high_corr_pairs[:100],  # Top 100
            '_correlation_matrix': corr_matrix,  # For visualization
        }

    def analyze_pos_profile_clustering(
        self,
        selectivity_matrix: np.ndarray,
        n_clusters: int = 8,
        min_selectivity: float = 0.5,
    ) -> Dict:
        """
        Cluster neurons by their POS selectivity profiles.

        Each neuron has a selectivity vector [sel_NOUN, sel_VERB, sel_ADJ, ...]
        that defines its "POS profile". Clustering finds groups with similar profiles.

        Args:
            selectivity_matrix: [n_pos, n_neurons] selectivity values
            n_clusters: Number of clusters for K-means
            min_selectivity: Minimum max-selectivity to include neuron

        Returns:
            Dictionary with cluster assignments and profiles
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            HAS_SKLEARN = True
        except ImportError:
            HAS_SKLEARN = False

        # Transpose to [n_neurons, n_pos] - each row is a neuron's profile
        profiles = selectivity_matrix.T  # [n_neurons, n_pos]

        # Filter neurons with meaningful selectivity
        max_sel = profiles.max(axis=1)  # [n_neurons]
        active_mask = max_sel >= min_selectivity
        active_indices = np.where(active_mask)[0]

        if len(active_indices) < n_clusters:
            print(f"  Warning: Only {len(active_indices)} active neurons, reducing clusters")
            n_clusters = max(2, len(active_indices) // 2)

        if len(active_indices) < 2:
            return {'error': 'Not enough active neurons for clustering'}

        active_profiles = profiles[active_mask]  # [n_active, n_pos]
        silhouette = None

        if HAS_SKLEARN:
            # Normalize profiles for better clustering
            scaler = StandardScaler()
            scaled_profiles = scaler.fit_transform(active_profiles)

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_profiles)

            # Compute silhouette score (clustering quality metric)
            if len(np.unique(cluster_labels)) > 1:
                silhouette = float(silhouette_score(scaled_profiles, cluster_labels))
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # [k, n_pos]
        else:
            # Simple fallback: assign by dominant POS
            cluster_labels = active_profiles.argmax(axis=1)  # Cluster by dominant POS
            n_clusters = len(UPOS_TAGS)
            cluster_centers = None

        # Analyze each cluster
        clusters = {}
        for c in range(n_clusters):
            mask = cluster_labels == c
            if not mask.any():
                continue

            cluster_neurons = active_indices[mask]
            cluster_profiles = active_profiles[mask]  # [n_cluster, n_pos]

            # Mean profile for this cluster
            mean_profile = cluster_profiles.mean(axis=0)  # [n_pos]

            # Find dominant POS for this cluster
            top_pos_indices = np.argsort(mean_profile)[::-1][:3]
            dominant_pos = [UPOS_TAGS[i] for i in top_pos_indices if mean_profile[i] > 1.0]

            # Label the cluster by its dominant POS pattern
            if dominant_pos:
                cluster_name = "/".join(dominant_pos[:2])
            else:
                cluster_name = f"Cluster_{c}"

            clusters[cluster_name] = {
                'cluster_id': int(c),
                'n_neurons': int(mask.sum()),
                'neuron_ids': [int(n) for n in cluster_neurons[:20]],  # Sample
                'mean_selectivity_profile': {
                    UPOS_TAGS[i]: float(mean_profile[i])
                    for i in range(len(UPOS_TAGS))
                },
                'dominant_pos': dominant_pos,
                'profile_variance': float(cluster_profiles.var()),
            }

        # Summary statistics
        summary = {
            'n_clusters': len(clusters),
            'n_neurons_clustered': len(active_indices),
            'silhouette_score': silhouette,  # Clustering quality: -1 to 1, higher is better
            'cluster_sizes': {
                name: data['n_neurons'] for name, data in clusters.items()
            },
        }

        return {
            'summary': summary,
            'clusters': clusters,
            '_cluster_labels': cluster_labels,
            '_cluster_centers': cluster_centers,
            '_active_indices': active_indices,
        }

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

        # POS profile clustering
        pos_profile_clustering = self.analyze_pos_profile_clustering(
            selectivity, n_clusters=8, min_selectivity=0.5
        )

        # Co-activation analysis (if enabled)
        coactivation_results = self.get_coactivation_matrix(min_occurrences=10)

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
            # New: POS profile clustering
            'pos_profile_clustering': {
                k: v for k, v in pos_profile_clustering.items() if not k.startswith('_')
            },
            # New: Co-activation analysis
            'coactivation': {
                k: v for k, v in coactivation_results.items() if not k.startswith('_')
            } if coactivation_results else None,
            # Store raw matrices for visualization
            '_selectivity_matrix': selectivity,
            '_mean_weight_matrix': mean_weight,
            '_pos_indices': active_pos,
            '_pos_profile_clustering': pos_profile_clustering,
            '_coactivation': coactivation_results,
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

        # 4. POS Profile Clustering
        clustering = results.get('pos_profile_clustering', {})
        if clustering and 'clusters' in clustering:
            print("\n┌─ POS Profile Clustering (neurons grouped by selectivity patterns) ───┐")
            summary = clustering.get('summary', {})
            sil_score = summary.get('silhouette_score')
            sil_str = f"{sil_score:.3f}" if sil_score is not None else "N/A"
            print(f"│  Clusters: {summary.get('n_clusters', 0)}, "
                  f"Neurons clustered: {summary.get('n_neurons_clustered', 0)}")
            print(f"│  Silhouette Score: {sil_str}  (range: -1 to 1, higher = better separation)")
            print(f"│")

            for name, data in clustering['clusters'].items():
                dominant = ", ".join(data.get('dominant_pos', [])[:3]) or "Mixed"
                n_neurons = data.get('n_neurons', 0)
                sample_ids = data.get('neuron_ids', [])[:5]
                sample_str = ", ".join(f"N{n}" for n in sample_ids)
                print(f"│  {name:<15} ({n_neurons:>4} neurons) - dominant: {dominant}")
                if sample_ids:
                    print(f"│     sample: {sample_str}")

            print("└───────────────────────────────────────────────────────────────────────┘")

        # 5. Co-activation Analysis
        coactivation = results.get('coactivation')
        if coactivation:
            print("\n┌─ Co-activation Analysis (neuron correlation) ────────────────────────┐")
            summary = coactivation.get('summary', {})
            print(f"│  Tokens analyzed: {summary.get('n_tokens', 0):,}")
            print(f"│  Mean correlation: {summary.get('mean_correlation', 0):.4f}")
            print(f"│  High correlation pairs (|r|>0.5): {summary.get('n_high_corr_pairs', 0)}")
            print(f"│  Positive corr (r>0.3): {summary.get('n_positive_corr', 0)}")
            print(f"│  Negative corr (r<-0.3): {summary.get('n_negative_corr', 0)}")
            print(f"│")

            # Show top correlated pairs
            pairs = coactivation.get('high_corr_pairs', [])[:10]
            if pairs:
                print(f"│  Top correlated pairs:")
                print(f"│  {'Neuron i':>10} {'Neuron j':>10} {'Correlation':>12}")
                print(f"│  {'─'*10} {'─'*10} {'─'*12}")
                for p in pairs:
                    print(f"│  N{p['neuron_i']:<9} N{p['neuron_j']:<9} {p['correlation']:>12.4f}")

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

        # POS profile clustering visualization
        path = self._plot_pos_profile_clustering(results, os.path.join(output_dir, 'pos_profile_clusters.png'))
        if path:
            paths['pos_profile_clusters'] = path

        # Co-activation heatmap
        path = self._plot_coactivation_heatmap(results, os.path.join(output_dir, 'coactivation_heatmap.png'))
        if path:
            paths['coactivation_heatmap'] = path

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

    def _plot_pos_profile_clustering(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot POS profile clustering results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        clustering = results.get('_pos_profile_clustering')
        if not clustering or 'error' in clustering:
            return None

        clusters = clustering.get('clusters', {})
        if not clusters:
            return None

        cluster_centers = clustering.get('_cluster_centers')

        # Create figure with subplots
        n_clusters = len(clusters)
        n_cols = min(4, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (name, data) in enumerate(clusters.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Get mean profile for this cluster
            profile = data.get('mean_selectivity_profile', {})
            pos_labels = [p for p in UPOS_TAGS if p in profile]
            values = [profile.get(p, 0) for p in pos_labels]

            # Bar chart of selectivity profile
            colors = ['coral' if v > 1.5 else 'steelblue' for v in values]
            ax.barh(range(len(pos_labels)), values, color=colors)
            ax.set_yticks(range(len(pos_labels)))
            ax.set_yticklabels(pos_labels, fontsize=8)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('Mean Selectivity')
            ax.set_title(f'{name}\n({data["n_neurons"]} neurons)', fontsize=10)

        # Hide unused axes
        for idx in range(len(clusters), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle('POS Profile Clusters\n(neurons grouped by selectivity patterns)', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_coactivation_heatmap(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot neuron co-activation correlation heatmap."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        coactivation = results.get('_coactivation')
        if not coactivation:
            return None

        corr_matrix = coactivation.get('_correlation_matrix')
        if corr_matrix is None:
            return None

        # Sample neurons if too many (for visualization)
        n_neurons = corr_matrix.shape[0]
        if n_neurons > 200:
            # Select neurons with highest variance in correlation
            corr_var = np.var(corr_matrix, axis=1)
            top_indices = np.argsort(corr_var)[-200:]
            corr_matrix = corr_matrix[np.ix_(top_indices, top_indices)]
            sampled = True
        else:
            sampled = False

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Full correlation heatmap
        ax = axes[0]
        im1 = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
        title = 'Neuron Co-activation Correlation'
        if sampled:
            title += '\n(top 200 by correlation variance)'
        ax.set_title(title)
        plt.colorbar(im1, ax=ax, label='Correlation')

        # Right: Histogram of correlations
        ax = axes[1]
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

        ax.hist(upper_tri, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='r=0.3')
        ax.axvline(x=-0.3, color='red', linestyle='--', alpha=0.7, label='r=-0.3')
        ax.axvline(x=0.5, color='green', linestyle='-', alpha=0.7, label='r=0.5')
        ax.axvline(x=-0.5, color='red', linestyle='-', alpha=0.7, label='r=-0.5')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Pairwise Correlations')
        ax.legend()

        # Add summary stats
        summary = coactivation.get('summary', {})
        stats_text = (
            f"Mean: {summary.get('mean_correlation', 0):.4f}\n"
            f"Std: {summary.get('std_correlation', 0):.4f}\n"
            f"|r|>0.5: {summary.get('n_high_corr_pairs', 0)}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
        compute_coactivation: bool = False,
    ) -> Dict:
        """
        Run full POS neuron analysis.

        Args:
            output_dir: Output directory
            pool_type: Pool to analyze
            max_sentences: Maximum sentences
            split: Dataset split
            data_path: Path to local conllu file
            compute_coactivation: Whether to compute neuron co-activation correlation

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = self.load_ud_dataset(split, max_sentences, data_path)

        # Analyze
        results = self.analyze_dataset(
            dataset, pool_type, max_sentences,
            compute_coactivation=compute_coactivation
        )

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


class TokenCombinationAnalyzer(BaseAnalyzer):
    """
    Token-based neuron combination analysis.

    Instead of analyzing which POS each neuron prefers (neuron -> POS),
    this analyzes which neuron combinations each token activates (token -> neurons).

    Goal: Higher silhouette score by clustering tokens based on their
    neuron activation patterns, expecting POS to emerge as natural clusters.

    Key differences from POSNeuronAnalyzer:
    - Collects per-token binary activation masks (not per-neuron weight sums)
    - Concatenates all pools into single vector per token
    - Uses Jaccard similarity for binary vectors
    - Analyzes layer-wise divergence for same tokens
    """

    # Class-level cache for GloVe embeddings (shared across all instances)
    _glove_cache = None
    _glove_cache_dim = None

    def __init__(
        self,
        model,
        router=None,
        tokenizer=None,
        device: str = 'cuda',
        target_layer: int = None,
        activation_threshold: float = 1e-6,
    ):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter (auto-detected if None)
            tokenizer: Tokenizer instance
            device: Device for computation
            target_layer: Specific layer to analyze (None = all layers averaged)
            activation_threshold: Threshold for binary activation (weight > threshold)
        """
        super().__init__(model, router=router, tokenizer=tokenizer, device=device)
        self.extractor = RoutingDataExtractor(model, device=device)
        self.target_layer = target_layer
        self.activation_threshold = activation_threshold
        self.model.eval()

        # Get pool sizes for concatenation
        self.pool_sizes = self._get_pool_sizes()
        self.total_neurons = sum(self.pool_sizes.values())

        # Precompute pool order for fast concatenation
        # NOTE: Use physical neuron pools, not Q/K separately
        # Q/K share the same neurons, so fqk (not fqk_q + fqk_k)
        self.pool_order = [
            ('fqk', self.pool_sizes.get('fqk', 0)),
            ('fv', self.pool_sizes.get('fv', 0)),
            ('rqk', self.pool_sizes.get('rqk', 0)),
            ('rv', self.pool_sizes.get('rv', 0)),
            ('fknow', self.pool_sizes.get('fknow', 0)),
            ('rknow', self.pool_sizes.get('rknow', 0)),
        ]
        self.pool_order = [(k, s) for k, s in self.pool_order if s > 0]

        # Storage for token activations
        self.token_data = []  # List of {token_str, pos, mask, ...}

        # Storage for layer divergence analysis
        self.layer_token_data = defaultdict(list)  # {token_str: [{layer_masks: {...}, ...}]}

        # Logging flag (only log mask mode once)
        self._logged_mask_mode = False
        self._mask_mode_stats = {'actual': 0, 'fallback': 0}

    def _get_pool_sizes(self) -> Dict[str, int]:
        """Get sizes of each pool for concatenation."""
        sizes = {}
        for attr, key in [
            ('n_feature_qk', 'fqk'),
            ('n_feature_v', 'fv'),
            ('n_restore_qk', 'rqk'),
            ('n_restore_v', 'rv'),
            ('n_feature_know', 'fknow'),
            ('n_restore_know', 'rknow'),
        ]:
            sizes[key] = getattr(self.router, attr, 0)
        return sizes

    def reset_stats(self):
        """Reset collected token data."""
        self.token_data = []
        self.layer_token_data = defaultdict(list)

    def load_ud_dataset(self, split: str = 'train', max_sentences: int = None,
                        data_path: str = None) -> List[Dict]:
        """Load UD dataset (reuse from POSNeuronAnalyzer)."""
        temp = POSNeuronAnalyzer(self.model, tokenizer=self.tokenizer, device=self.device)
        return temp.load_ud_dataset(split, max_sentences, data_path)

    def get_pos_for_tokens(self, ud_tokens: List[str], ud_pos: List[str]) -> Tuple[List[str], List[int]]:
        """Map tokenizer tokens to POS tags (reuse from POSNeuronAnalyzer)."""
        temp = POSNeuronAnalyzer(self.model, tokenizer=self.tokenizer, device=self.device)
        return temp.get_pos_for_tokens(ud_tokens, ud_pos)

    def get_tags_for_tokens(self, ud_tokens: List[str], ud_pos: List[str],
                            ud_deprel: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """Map tokenizer tokens to POS tags and deprel (reuse from POSNeuronAnalyzer)."""
        temp = POSNeuronAnalyzer(self.model, tokenizer=self.tokenizer, device=self.device)
        return temp.get_tags_for_tokens(ud_tokens, ud_pos, ud_deprel)

    def _is_whole_word(self, token_str: str, next_token_str: str = None) -> bool:
        """
        Check if token represents a complete whole word (not a subword).

        For BERT-style tokenizers:
        - "playing" → ["play", "##ing"]
        - "play" is NOT whole word (has ## continuation)
        - "##ing" is NOT whole word (is continuation)
        - "cat" alone IS whole word

        Args:
            token_str: Current token string
            next_token_str: Next token string (to check for ## continuation)
        """
        if not token_str:
            return False

        # BERT-style: ## prefix means this is a continuation piece
        if token_str.startswith('##'):
            return False

        # BERT-style: if NEXT token starts with ##, current is NOT whole word
        # e.g., "play" followed by "##ing" → "play" is not whole word
        if next_token_str and next_token_str.startswith('##'):
            return False

        # GPT/SentencePiece: Ġ or ▁ means word-initial (whole word)
        if token_str.startswith(('Ġ', '▁', ' ')):
            return True

        # Default: treat as whole word (single-token word in BERT)
        return True

    def _extract_layer_mask(self, layer, seq_len: int) -> np.ndarray:
        """
        Extract concatenated mask from a single layer.

        Uses actual binary masks from routing (scores > learnable tau) if available.
        Falls back to weight > threshold only if masks unavailable.
        """
        masks = []
        used_actual = 0
        used_fallback = 0

        failed_pools = []

        # Debug: log available keys once
        if not self._logged_mask_mode:
            available_keys = list(layer.raw.keys())
            attn_keys = list(layer.attention.keys()) if layer.attention else []
            know_keys = list(layer.knowledge.keys()) if layer.knowledge else []
            print(f"  [Debug] routing_info keys: {available_keys[:10]}...")
            print(f"  [Debug] attention keys: {[k for k in attn_keys if 'mask' in k.lower()]}")
            print(f"  [Debug] knowledge keys: {[k for k in know_keys if 'mask' in k.lower()]}")

        for pool_key, expected_size in self.pool_order:
            # For Q/K shared pools, combine Q and K masks with OR
            # A neuron is active if selected for Q OR K
            if pool_key in ('fqk', 'rqk'):
                mask_q = layer.get_mask(f'{pool_key}_q')
                mask_k = layer.get_mask(f'{pool_key}_k')

                if mask_q is not None or mask_k is not None:
                    used_actual += 1
                    m = np.zeros((seq_len, expected_size), dtype=np.bool_)

                    for sub_mask in [mask_q, mask_k]:
                        if sub_mask is not None:
                            if sub_mask.dim() == 3:
                                sub_m = sub_mask[0, :seq_len].cpu().numpy().astype(np.bool_)
                            else:
                                sub_m = np.broadcast_to(
                                    sub_mask[0].cpu().numpy().astype(np.bool_),
                                    (seq_len, sub_mask.shape[-1])
                                )
                            if sub_m.shape[0] < seq_len:
                                sub_m = np.pad(sub_m, ((0, seq_len - sub_m.shape[0]), (0, 0)))
                            m = m | sub_m  # OR combine

                    masks.append(m)
                else:
                    # Fallback: use weights > threshold
                    used_fallback += 1
                    failed_pools.append(pool_key)
                    weights_q = layer.get_weight(f'{pool_key}_q')
                    weights_k = layer.get_weight(f'{pool_key}_k')

                    m = np.zeros((seq_len, expected_size), dtype=np.bool_)
                    for weights in [weights_q, weights_k]:
                        if weights is not None:
                            if weights.dim() == 3:
                                w = weights[0, :seq_len].cpu().numpy()
                            else:
                                w = np.broadcast_to(
                                    weights[0].cpu().numpy(),
                                    (seq_len, weights.shape[-1])
                                )
                            if w.shape[0] < seq_len:
                                w = np.pad(w, ((0, seq_len - w.shape[0]), (0, 0)))
                            m = m | (w > self.activation_threshold)  # OR combine

                    masks.append(m)
            else:
                # Regular pools: fv, rv, fknow, rknow
                mask = layer.get_mask(pool_key)

                if mask is not None:
                    used_actual += 1
                    # Use actual model mask (cleaner signal)
                    if mask.dim() == 3:
                        m = mask[0, :seq_len].cpu().numpy().astype(np.bool_)
                    else:
                        m = np.broadcast_to(
                            mask[0].cpu().numpy().astype(np.bool_),
                            (seq_len, mask.shape[-1])
                        )

                    if m.shape[0] < seq_len:
                        m = np.pad(m, ((0, seq_len - m.shape[0]), (0, 0)))

                    masks.append(m)
                else:
                    used_fallback += 1
                    failed_pools.append(pool_key)
                    # Fallback: use weights > threshold
                    weights = layer.get_weight(pool_key)
                    if weights is None:
                        masks.append(np.zeros((seq_len, expected_size), dtype=np.bool_))
                    else:
                        if weights.dim() == 3:
                            w = weights[0, :seq_len].cpu().numpy()
                        else:
                            w = np.broadcast_to(
                                weights[0].cpu().numpy(),
                                (seq_len, weights.shape[-1])
                            )

                        if w.shape[0] < seq_len:
                            w = np.pad(w, ((0, seq_len - w.shape[0]), (0, 0)))

                        masks.append(w > self.activation_threshold)

        # Log mask mode once
        if not self._logged_mask_mode:
            self._logged_mask_mode = True
            if used_actual > 0 and used_fallback == 0:
                print(f"  [Mask mode] Using actual masks (tau-based) for all {used_actual} pools ✓")
            elif used_fallback > 0 and used_actual == 0:
                print(f"  [Mask mode] Fallback to weights > {self.activation_threshold} for all {used_fallback} pools")
                print(f"              (masks not available in routing_info)")
            else:
                print(f"  [Mask mode] Mixed: {used_actual} actual masks, {used_fallback} fallback")
                print(f"              Fallback pools: {failed_pools}")

        self._mask_mode_stats['actual'] += used_actual
        self._mask_mode_stats['fallback'] += used_fallback

        return np.concatenate(masks, axis=-1) if masks else np.zeros((seq_len, 0), dtype=np.bool_)

    def _extract_layer_weights(self, layer, seq_len: int) -> np.ndarray:
        """
        Extract concatenated continuous weights from a single layer.

        Returns continuous routing weights for cosine similarity analysis.
        For Q/K shared pools, takes max(Q, K) weights.
        """
        weights_list = []

        for pool_key, expected_size in self.pool_order:
            # For Q/K shared pools, take max of Q and K weights
            if pool_key in ('fqk', 'rqk'):
                weights_q = layer.get_weight(f'{pool_key}_q')
                weights_k = layer.get_weight(f'{pool_key}_k')

                w = np.zeros((seq_len, expected_size), dtype=np.float32)
                for weights in [weights_q, weights_k]:
                    if weights is not None:
                        if weights.dim() == 3:
                            sub_w = weights[0, :seq_len].cpu().numpy().astype(np.float32)
                        else:
                            sub_w = np.broadcast_to(
                                weights[0].cpu().numpy().astype(np.float32),
                                (seq_len, weights.shape[-1])
                            )
                        if sub_w.shape[0] < seq_len:
                            sub_w = np.pad(sub_w, ((0, seq_len - sub_w.shape[0]), (0, 0)))
                        w = np.maximum(w, sub_w)  # Take max

                weights_list.append(w)
            else:
                # Regular pools
                weights = layer.get_weight(pool_key)

                if weights is None:
                    weights_list.append(np.zeros((seq_len, expected_size), dtype=np.float32))
                else:
                    if weights.dim() == 3:
                        w = weights[0, :seq_len].cpu().numpy().astype(np.float32)
                    else:
                        w = np.broadcast_to(
                            weights[0].cpu().numpy().astype(np.float32),
                            (seq_len, weights.shape[-1])
                        )

                    if w.shape[0] < seq_len:
                        w = np.pad(w, ((0, seq_len - w.shape[0]), (0, 0)))

                    weights_list.append(w)

        return np.concatenate(weights_list, axis=-1) if weights_list else np.zeros((seq_len, 0), dtype=np.float32)

    def extract_token_masks(
        self,
        token_ids: List[int],
        pos_tags: List[str],
        store_layer_masks: bool = False,
        deprel_tags: List[str] = None,
    ) -> List[Dict]:
        """
        Extract binary activation masks for each token.

        Args:
            token_ids: List of token IDs
            pos_tags: List of POS tags for each token
            store_layer_masks: Whether to store per-layer masks for divergence analysis
            deprel_tags: List of dependency relation tags (optional)

        Returns:
            List of {token_id, token_str, pos, deprel, mask, is_whole_word, layer_masks?}
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        seq_len = len(token_ids)

        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(input_ids, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return []

        # Collect masks and weights per layer
        layer_masks = {}
        layer_weights = {}
        for layer in routing:
            layer_idx = layer.layer_idx
            if self.target_layer is not None and layer_idx != self.target_layer:
                continue
            layer_masks[layer_idx] = self._extract_layer_mask(layer, seq_len)
            layer_weights[layer_idx] = self._extract_layer_weights(layer, seq_len)

        if not layer_masks:
            return []

        # Combined mask (union across layers) and weights (mean across layers)
        # DAWN shared pool: same neuron index = same physical neuron across layers
        # Union semantics: if a neuron is selected in ANY layer, it contributed to this token
        if self.target_layer is not None:
            combined_mask = layer_masks.get(self.target_layer)
            combined_weights = layer_weights.get(self.target_layer)
            if combined_mask is None:
                return []
        else:
            all_masks = np.stack(list(layer_masks.values()), axis=0)
            combined_mask = all_masks.any(axis=0)  # Union: active if selected in ANY layer
            all_weights = np.stack(list(layer_weights.values()), axis=0)
            combined_weights = all_weights.mean(axis=0)

        # Build token data (vectorized token string decode)
        token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]

        results = []
        n_pos = min(seq_len, len(pos_tags))
        for i in range(n_pos):
            token_str = token_strs[i]
            # Get next token for BERT-style whole word detection
            next_token_str = token_strs[i + 1] if i + 1 < len(token_strs) else None

            entry = {
                'token_id': token_ids[i],
                'token_str': token_str,
                'pos': pos_tags[i],
                'deprel': deprel_tags[i] if deprel_tags and i < len(deprel_tags) else '_',
                'mask': combined_mask[i],
                'weights': combined_weights[i],  # continuous weights for cosine similarity
                'is_whole_word': self._is_whole_word(token_str, next_token_str),
                'n_active': int(combined_mask[i].sum()),
            }

            if store_layer_masks:
                entry['layer_masks'] = {k: v[i] for k, v in layer_masks.items()}

            results.append(entry)

        return results

    def analyze_sentence(self, ud_tokens: List[str], ud_pos: List[str],
                         store_layer_masks: bool = False, ud_deprel: List[str] = None):
        """Analyze a single sentence and collect token activations."""
        try:
            if ud_deprel is not None:
                pos_tags, deprel_tags, token_ids = self.get_tags_for_tokens(ud_tokens, ud_pos, ud_deprel)
            else:
                pos_tags, token_ids = self.get_pos_for_tokens(ud_tokens, ud_pos)
                deprel_tags = None
        except Exception:
            return

        if not token_ids:
            return

        token_masks = self.extract_token_masks(token_ids, pos_tags, store_layer_masks, deprel_tags)
        self.token_data.extend(token_masks)

        # Store for layer divergence analysis
        if store_layer_masks:
            for t in token_masks:
                if 'layer_masks' in t:
                    self.layer_token_data[t['token_str'].lower().strip()].append(t)

    def analyze_dataset(self, dataset: List[Dict], max_sentences: int = None,
                        analyze_layer_divergence: bool = True, batch_size: int = 16) -> Dict:
        """
        Analyze full dataset with batched processing (5-10x faster).

        Args:
            dataset: List of {'tokens': [...], 'upos': [...], 'deprel': [...]}
            max_sentences: Maximum sentences to process
            analyze_layer_divergence: Whether to store per-layer masks for divergence
            batch_size: Number of sentences to process in parallel

        Returns:
            Analysis results dictionary
        """
        self.reset_stats()
        n_sentences = min(len(dataset), max_sentences) if max_sentences else len(dataset)

        print(f"\nCollecting token activations from {n_sentences} sentences...")
        print(f"Activation threshold: {self.activation_threshold}")
        print(f"Layer: {self.target_layer if self.target_layer is not None else 'all (union)'}")
        print(f"Layer divergence analysis: {'ON' if analyze_layer_divergence else 'OFF'}")
        print(f"Batch size: {batch_size} (batched processing for efficiency)")

        # Pre-process all sentences to get token_ids and pos_tags
        preprocessed = []
        for i in range(n_sentences):
            example = dataset[i]
            try:
                ud_deprel = example.get('deprel')
                if ud_deprel is not None:
                    pos_tags, deprel_tags, token_ids = self.get_tags_for_tokens(
                        example['tokens'], example['upos'], ud_deprel
                    )
                else:
                    pos_tags, token_ids = self.get_pos_for_tokens(
                        example['tokens'], example['upos']
                    )
                    deprel_tags = None

                if token_ids:
                    preprocessed.append({
                        'token_ids': token_ids,
                        'pos_tags': pos_tags,
                        'deprel_tags': deprel_tags,
                    })
            except Exception:
                continue

        # Process in batches
        n_batches = (len(preprocessed) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc="Batched Processing"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(preprocessed))
            batch = preprocessed[start:end]

            if not batch:
                continue

            try:
                token_masks = self._extract_batch_token_masks(
                    batch, analyze_layer_divergence
                )
                self.token_data.extend(token_masks)

                # Store for layer divergence analysis
                if analyze_layer_divergence:
                    for t in token_masks:
                        if 'layer_masks' in t:
                            self.layer_token_data[t['token_str'].lower().strip()].append(t)
            except Exception:
                # Fallback to one-by-one processing for this batch
                for item in batch:
                    try:
                        masks = self.extract_token_masks(
                            item['token_ids'], item['pos_tags'],
                            analyze_layer_divergence, item['deprel_tags']
                        )
                        self.token_data.extend(masks)
                        if analyze_layer_divergence:
                            for t in masks:
                                if 'layer_masks' in t:
                                    self.layer_token_data[t['token_str'].lower().strip()].append(t)
                    except Exception:
                        continue

        print(f"Collected {len(self.token_data)} tokens")
        return self.compute_results(analyze_layer_divergence)

    def _extract_batch_token_masks(
        self,
        batch: List[Dict],
        store_layer_masks: bool = False,
    ) -> List[Dict]:
        """
        Extract token masks for a batch of sentences in one forward pass.

        Args:
            batch: List of {'token_ids': [...], 'pos_tags': [...], 'deprel_tags': [...]}
            store_layer_masks: Whether to store per-layer masks

        Returns:
            List of token data dicts for all tokens in the batch
        """
        # Find max length and pad sequences
        max_len = max(len(item['token_ids']) for item in batch)
        batch_size = len(batch)

        # Create padded input tensor
        # Use 0 as padding (assuming 0 is PAD token, will be masked anyway)
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
        padded_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        seq_lens = []

        for i, item in enumerate(batch):
            seq_len = len(item['token_ids'])
            padded_input[i, :seq_len] = torch.tensor(item['token_ids'], dtype=torch.long)
            seq_lens.append(seq_len)

        # Forward pass for entire batch
        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(padded_input, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return []

        # Collect masks and weights per layer for each batch item
        all_layer_masks = {i: {} for i in range(batch_size)}
        all_layer_weights = {i: {} for i in range(batch_size)}

        for layer in routing:
            layer_idx = layer.layer_idx
            if self.target_layer is not None and layer_idx != self.target_layer:
                continue

            # Extract masks/weights for this layer (full batch)
            batch_masks = self._extract_batch_layer_mask(layer, batch_size, max_len)
            batch_weights = self._extract_batch_layer_weights(layer, batch_size, max_len)

            for b_idx in range(batch_size):
                all_layer_masks[b_idx][layer_idx] = batch_masks[b_idx, :seq_lens[b_idx]]
                all_layer_weights[b_idx][layer_idx] = batch_weights[b_idx, :seq_lens[b_idx]]

        # Build results for each token
        results = []
        for b_idx, item in enumerate(batch):
            seq_len = seq_lens[b_idx]
            layer_masks = all_layer_masks[b_idx]
            layer_weights = all_layer_weights[b_idx]

            if not layer_masks:
                continue

            # Combined mask and weights
            if self.target_layer is not None:
                combined_mask = layer_masks.get(self.target_layer)
                combined_weights = layer_weights.get(self.target_layer)
                if combined_mask is None:
                    continue
            else:
                all_masks_arr = np.stack(list(layer_masks.values()), axis=0)
                combined_mask = all_masks_arr.any(axis=0)
                all_weights_arr = np.stack(list(layer_weights.values()), axis=0)
                combined_weights = all_weights_arr.mean(axis=0)

            # Build token strings
            token_strs = [self.tokenizer.decode([tid]) for tid in item['token_ids']]

            n_pos = min(seq_len, len(item['pos_tags']))
            for i in range(n_pos):
                token_str = token_strs[i]
                next_token_str = token_strs[i + 1] if i + 1 < len(token_strs) else None

                entry = {
                    'token_id': item['token_ids'][i],
                    'token_str': token_str,
                    'pos': item['pos_tags'][i],
                    'deprel': item['deprel_tags'][i] if item['deprel_tags'] and i < len(item['deprel_tags']) else '_',
                    'mask': combined_mask[i],
                    'weights': combined_weights[i],
                    'is_whole_word': self._is_whole_word(token_str, next_token_str),
                    'n_active': int(combined_mask[i].sum()),
                }

                if store_layer_masks:
                    entry['layer_masks'] = {k: v[i] for k, v in layer_masks.items()}

                results.append(entry)

        return results

    def _extract_batch_layer_mask(self, layer, batch_size: int, max_len: int) -> np.ndarray:
        """
        Extract concatenated masks from a layer for entire batch.

        Returns: [batch_size, max_len, n_neurons] boolean array
        """
        masks = []

        for pool_key, expected_size in self.pool_order:
            if pool_key in ('fqk', 'rqk'):
                mask_q = layer.get_mask(f'{pool_key}_q')
                mask_k = layer.get_mask(f'{pool_key}_k')

                if mask_q is not None or mask_k is not None:
                    m = np.zeros((batch_size, max_len, expected_size), dtype=np.bool_)
                    for sub_mask in [mask_q, mask_k]:
                        if sub_mask is not None:
                            if sub_mask.dim() == 3:
                                sub_m = sub_mask[:, :max_len].cpu().numpy().astype(np.bool_)
                            else:
                                sub_m = np.broadcast_to(
                                    sub_mask.cpu().numpy().astype(np.bool_)[:, None, :],
                                    (batch_size, max_len, sub_mask.shape[-1])
                                )
                            pad_len = max_len - sub_m.shape[1]
                            if pad_len > 0:
                                sub_m = np.pad(sub_m, ((0, 0), (0, pad_len), (0, 0)))
                            m = m | sub_m[:, :max_len]
                    masks.append(m)
                else:
                    # Fallback to weights
                    weights_q = layer.get_weight(f'{pool_key}_q')
                    weights_k = layer.get_weight(f'{pool_key}_k')
                    m = np.zeros((batch_size, max_len, expected_size), dtype=np.bool_)
                    for weights in [weights_q, weights_k]:
                        if weights is not None:
                            if weights.dim() == 3:
                                w = weights[:, :max_len].cpu().numpy()
                            else:
                                w = np.broadcast_to(
                                    weights.cpu().numpy()[:, None, :],
                                    (batch_size, max_len, weights.shape[-1])
                                )
                            pad_len = max_len - w.shape[1]
                            if pad_len > 0:
                                w = np.pad(w, ((0, 0), (0, pad_len), (0, 0)))
                            m = m | (w[:, :max_len] > self.activation_threshold)
                    masks.append(m)
            else:
                mask = layer.get_mask(pool_key)
                if mask is not None:
                    if mask.dim() == 3:
                        m = mask[:, :max_len].cpu().numpy().astype(np.bool_)
                    else:
                        m = np.broadcast_to(
                            mask.cpu().numpy().astype(np.bool_)[:, None, :],
                            (batch_size, max_len, mask.shape[-1])
                        )
                    pad_len = max_len - m.shape[1]
                    if pad_len > 0:
                        m = np.pad(m, ((0, 0), (0, pad_len), (0, 0)))
                    masks.append(m[:, :max_len])
                else:
                    weights = layer.get_weight(pool_key)
                    if weights is None:
                        masks.append(np.zeros((batch_size, max_len, expected_size), dtype=np.bool_))
                    else:
                        if weights.dim() == 3:
                            w = weights[:, :max_len].cpu().numpy()
                        else:
                            w = np.broadcast_to(
                                weights.cpu().numpy()[:, None, :],
                                (batch_size, max_len, weights.shape[-1])
                            )
                        pad_len = max_len - w.shape[1]
                        if pad_len > 0:
                            w = np.pad(w, ((0, 0), (0, pad_len), (0, 0)))
                        masks.append(w[:, :max_len] > self.activation_threshold)

        return np.concatenate(masks, axis=-1) if masks else np.zeros((batch_size, max_len, 0), dtype=np.bool_)

    def _extract_batch_layer_weights(self, layer, batch_size: int, max_len: int) -> np.ndarray:
        """
        Extract concatenated weights from a layer for entire batch.

        Returns: [batch_size, max_len, n_neurons] float array
        """
        weights_list = []

        for pool_key, expected_size in self.pool_order:
            if pool_key in ('fqk', 'rqk'):
                weights_q = layer.get_weight(f'{pool_key}_q')
                weights_k = layer.get_weight(f'{pool_key}_k')
                w = np.zeros((batch_size, max_len, expected_size), dtype=np.float32)
                for weights in [weights_q, weights_k]:
                    if weights is not None:
                        if weights.dim() == 3:
                            sub_w = weights[:, :max_len].cpu().numpy().astype(np.float32)
                        else:
                            sub_w = np.broadcast_to(
                                weights.cpu().numpy().astype(np.float32)[:, None, :],
                                (batch_size, max_len, weights.shape[-1])
                            )
                        pad_len = max_len - sub_w.shape[1]
                        if pad_len > 0:
                            sub_w = np.pad(sub_w, ((0, 0), (0, pad_len), (0, 0)))
                        w = np.maximum(w, sub_w[:, :max_len])
                weights_list.append(w)
            else:
                weights = layer.get_weight(pool_key)
                if weights is None:
                    weights_list.append(np.zeros((batch_size, max_len, expected_size), dtype=np.float32))
                else:
                    if weights.dim() == 3:
                        w = weights[:, :max_len].cpu().numpy().astype(np.float32)
                    else:
                        w = np.broadcast_to(
                            weights.cpu().numpy().astype(np.float32)[:, None, :],
                            (batch_size, max_len, weights.shape[-1])
                        )
                    pad_len = max_len - w.shape[1]
                    if pad_len > 0:
                        w = np.pad(w, ((0, 0), (0, pad_len), (0, 0)))
                    weights_list.append(w[:, :max_len])

        return np.concatenate(weights_list, axis=-1) if weights_list else np.zeros((batch_size, max_len, 0), dtype=np.float32)

    def extract_token_masks_all_layers(
        self,
        token_ids: List[int],
        pos_tags: List[str],
        n_layers: int,
        deprel_tags: List[str] = None,
    ) -> Dict[int, List[Dict]]:
        """
        Extract masks/weights for ALL layers in a single forward pass.

        Args:
            token_ids: List of token IDs
            pos_tags: List of POS tags for each token
            n_layers: Number of layers in model
            deprel_tags: List of dependency relation tags (optional)

        Returns:
            Dict mapping layer_idx -> List of token data dicts
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        seq_len = len(token_ids)

        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(input_ids, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return {}

        # Extract masks and weights for ALL layers
        layer_masks = {}
        layer_weights = {}
        for layer in routing:
            layer_idx = layer.layer_idx
            layer_masks[layer_idx] = self._extract_layer_mask(layer, seq_len)
            layer_weights[layer_idx] = self._extract_layer_weights(layer, seq_len)

        if not layer_masks:
            return {}

        # Build token strings once
        token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]

        # Organize data by layer
        results_by_layer = {i: [] for i in range(n_layers)}
        n_pos = min(seq_len, len(pos_tags))

        for layer_idx in range(n_layers):
            if layer_idx not in layer_masks:
                continue

            masks = layer_masks[layer_idx]
            weights = layer_weights[layer_idx]

            for i in range(n_pos):
                token_str = token_strs[i]
                next_token_str = token_strs[i + 1] if i + 1 < len(token_strs) else None

                entry = {
                    'token_id': token_ids[i],
                    'token_str': token_str,
                    'pos': pos_tags[i],
                    'deprel': deprel_tags[i] if deprel_tags and i < len(deprel_tags) else '_',
                    'mask': masks[i],
                    'weights': weights[i],
                    'is_whole_word': self._is_whole_word(token_str, next_token_str),
                    'n_active': int(masks[i].sum()),
                }
                results_by_layer[layer_idx].append(entry)

        return results_by_layer

    def analyze_dataset_all_layers(
        self,
        dataset: List[Dict],
        n_layers: int,
        max_sentences: int = None,
    ) -> Dict[int, List[Dict]]:
        """
        Process dataset ONCE and collect token data for ALL layers.

        This is much more efficient than running analyze_dataset separately
        for each layer, as it only does ONE forward pass per sentence.

        Args:
            dataset: List of {'tokens': [...], 'upos': [...], 'deprel': [...]}
            n_layers: Number of layers in model
            max_sentences: Maximum sentences to process

        Returns:
            Dict mapping layer_idx -> List of token data dicts
        """
        n_sentences = min(len(dataset), max_sentences) if max_sentences else len(dataset)

        # Initialize storage for all layers
        all_layer_data = {i: [] for i in range(n_layers)}

        print(f"\nCollecting token activations from {n_sentences} sentences (ALL {n_layers} layers)...")
        print("  Single forward pass per sentence - efficient batch processing")

        for i in tqdm(range(n_sentences), desc="Processing"):
            example = dataset[i]
            ud_deprel = example.get('deprel')

            try:
                if ud_deprel is not None:
                    pos_tags, deprel_tags, token_ids = self.get_tags_for_tokens(
                        example['tokens'], example['upos'], ud_deprel
                    )
                else:
                    pos_tags, token_ids = self.get_pos_for_tokens(
                        example['tokens'], example['upos']
                    )
                    deprel_tags = None
            except Exception:
                continue

            if not token_ids:
                continue

            # Extract all layers in one forward pass
            layer_data = self.extract_token_masks_all_layers(
                token_ids, pos_tags, n_layers, deprel_tags
            )

            # Append to respective layer collections
            for layer_idx, tokens in layer_data.items():
                all_layer_data[layer_idx].extend(tokens)

        for layer_idx in range(n_layers):
            print(f"  Layer {layer_idx}: {len(all_layer_data[layer_idx])} tokens")

        return all_layer_data

    def compute_layer_results(self, token_data: List[Dict], layer_idx: int) -> Dict:
        """
        Compute silhouette and semantic correlation for a single layer's data.

        This is a lightweight version of compute_results() focused on the
        metrics needed for layer-wise analysis.

        Args:
            token_data: List of token data dicts for this layer
            layer_idx: Layer index (for logging)

        Returns:
            Dict with silhouette, function_silhouette, deprel_silhouette, semantic_correlation
        """
        if not token_data:
            return {'error': 'No token data'}

        # Build arrays
        n_tokens = len(token_data)
        n_neurons = len(token_data[0]['mask']) if n_tokens > 0 else 0

        masks = np.empty((n_tokens, n_neurons), dtype=np.bool_)
        weights = np.empty((n_tokens, n_neurons), dtype=np.float32)
        pos_labels = np.empty(n_tokens, dtype=np.int32)
        deprel_labels = np.empty(n_tokens, dtype=np.int32)
        is_whole_word = np.empty(n_tokens, dtype=np.bool_)

        for i, t in enumerate(token_data):
            masks[i] = t['mask']
            weights[i] = t['weights']
            pos_labels[i] = POS_TO_IDX.get(t['pos'], -1)
            deprel_labels[i] = DEPREL_TO_IDX.get(t.get('deprel', '_'), -1)
            is_whole_word[i] = t['is_whole_word']

        # Filter valid POS
        valid_mask = pos_labels >= 0
        masks = masks[valid_mask]
        weights = weights[valid_mask]
        pos_labels = pos_labels[valid_mask]
        deprel_labels = deprel_labels[valid_mask]
        is_whole_word = is_whole_word[valid_mask]
        # Also filter token_data to keep indices aligned
        filtered_token_data = [t for i, t in enumerate(token_data) if valid_mask[i]]

        n_tokens = len(masks)
        if n_tokens < 10:
            return {'error': f'Too few tokens: {n_tokens}'}

        # Compute key metrics
        silhouette = self._compute_silhouette(masks, pos_labels)
        function_silhouette = self._compute_function_word_silhouette(masks, pos_labels)
        deprel_silhouette = self._compute_deprel_silhouette(masks, deprel_labels)
        semantic_correlation = self._compute_semantic_correlation(
            masks, weights, pos_labels, is_whole_word, token_data=filtered_token_data
        )

        return {
            'n_tokens': n_tokens,
            'silhouette_score': silhouette,
            'function_word_silhouette': function_silhouette,
            'deprel_silhouette': deprel_silhouette,
            'semantic_correlation': semantic_correlation,
        }

    def _compute_jaccard_matrix(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Jaccard similarity matrix (optimized).

        Args:
            masks: [n_samples, n_features] boolean array

        Returns:
            [n_samples, n_samples] Jaccard similarity matrix
        """
        # Use float32 for memory efficiency
        masks_float = masks.astype(np.float32)

        # Vectorized: intersection = A @ B.T
        intersection = masks_float @ masks_float.T

        # |A| for each sample
        counts = masks.sum(axis=1, keepdims=True).astype(np.float32)

        # Union = |A| + |B| - intersection
        union = counts + counts.T - intersection

        # Jaccard = intersection / union (avoid division by zero)
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    def _compute_jaccard_pairwise(self, masks_a: np.ndarray, masks_b: np.ndarray) -> float:
        """Compute mean Jaccard between two groups (optimized)."""
        # Use broadcasting for efficiency
        a_float = masks_a.astype(np.float32)
        b_float = masks_b.astype(np.float32)

        intersection = a_float @ b_float.T
        counts_a = masks_a.sum(axis=1, keepdims=True)
        counts_b = masks_b.sum(axis=1, keepdims=True)
        union = counts_a + counts_b.T - intersection

        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        return float(jaccard.mean())

    def compute_results(self, analyze_layer_divergence: bool = True) -> Dict:
        """Compute all analysis results."""
        if not self.token_data:
            return {'error': 'No token data collected'}

        # Build arrays efficiently
        n_tokens = len(self.token_data)
        n_neurons = len(self.token_data[0]['mask']) if n_tokens > 0 else 0

        masks = np.empty((n_tokens, n_neurons), dtype=np.bool_)
        weights = np.empty((n_tokens, n_neurons), dtype=np.float32)
        pos_labels = np.empty(n_tokens, dtype=np.int32)
        deprel_labels = np.empty(n_tokens, dtype=np.int32)
        is_whole_word = np.empty(n_tokens, dtype=np.bool_)

        for i, t in enumerate(self.token_data):
            masks[i] = t['mask']
            weights[i] = t['weights']
            pos_labels[i] = POS_TO_IDX.get(t['pos'], -1)
            deprel_labels[i] = DEPREL_TO_IDX.get(t.get('deprel', '_'), -1)
            is_whole_word[i] = t['is_whole_word']

        # Filter valid POS
        valid_mask = pos_labels >= 0
        masks = masks[valid_mask]
        weights = weights[valid_mask]
        pos_labels = pos_labels[valid_mask]
        deprel_labels = deprel_labels[valid_mask]
        is_whole_word = is_whole_word[valid_mask]
        # Also filter token_data to keep indices aligned
        filtered_token_data = [t for i, t in enumerate(self.token_data) if valid_mask[i]]

        n_tokens = len(masks)
        print(f"\nAnalyzing {n_tokens} tokens with valid POS tags...")

        # 1. POS-based similarity analysis
        pos_similarity = self._compute_pos_similarity(masks, pos_labels)

        # 2. Silhouette score (all POS)
        silhouette = self._compute_silhouette(masks, pos_labels)

        # 3. Function word only silhouette (should be higher)
        function_silhouette = self._compute_function_word_silhouette(masks, pos_labels)

        # 3.5. Deprel silhouette (syntactic role clustering)
        deprel_silhouette = self._compute_deprel_silhouette(masks, deprel_labels)

        # 4. Content vs Function word analysis
        content_analysis = self._analyze_content_vs_function(masks, pos_labels, is_whole_word)

        # 5. Token statistics
        token_stats = self._compute_token_stats()

        # 6. Layer divergence analysis
        layer_divergence = None
        if analyze_layer_divergence and self.layer_token_data:
            layer_divergence = self._compute_layer_divergence()

        # 7. Content word hierarchical clustering
        content_hierarchy = self._compute_content_hierarchy(masks, pos_labels, is_whole_word)

        # 8. Semantic correlation analysis (neuron sim vs GloVe sim)
        # Use continuous weights for cosine similarity
        semantic_correlation = self._compute_semantic_correlation(
            masks, weights, pos_labels, is_whole_word, token_data=filtered_token_data
        )

        return {
            'n_tokens': n_tokens,
            'n_neurons': n_neurons,
            'pos_similarity': pos_similarity,
            'silhouette_score': silhouette,
            'function_word_silhouette': function_silhouette,
            'deprel_silhouette': deprel_silhouette,
            'content_analysis': content_analysis,
            'token_stats': token_stats,
            'layer_divergence': layer_divergence,
            'content_hierarchy': content_hierarchy,
            'semantic_correlation': semantic_correlation,
            '_masks': masks,
            '_weights': weights,
            '_pos_labels': pos_labels,
            '_deprel_labels': deprel_labels,
            '_is_whole_word': is_whole_word,
        }

    def _compute_pos_similarity(self, masks: np.ndarray, pos_labels: np.ndarray) -> Dict:
        """Compute within-POS and between-POS Jaccard similarity (optimized)."""
        unique_pos = np.unique(pos_labels)

        # Sample for efficiency
        max_per_pos = 150
        sampled_indices = []
        sampled_labels = []

        for pos_idx in unique_pos:
            pos_indices = np.where(pos_labels == pos_idx)[0]
            if len(pos_indices) > max_per_pos:
                pos_indices = np.random.choice(pos_indices, max_per_pos, replace=False)
            sampled_indices.extend(pos_indices)
            sampled_labels.extend([pos_idx] * len(pos_indices))

        if not sampled_indices:
            return {}

        sampled_masks = masks[sampled_indices]
        sampled_labels = np.array(sampled_labels)

        # Compute Jaccard matrix once
        print("  Computing Jaccard similarity matrix...")
        jaccard = self._compute_jaccard_matrix(sampled_masks)

        # Within-POS similarity
        within_sim = {}
        for pos_idx in unique_pos:
            pos_name = UPOS_TAGS[pos_idx]
            mask = sampled_labels == pos_idx
            if mask.sum() < 2:
                continue

            pos_jaccard = jaccard[np.ix_(mask, mask)]
            triu_vals = pos_jaccard[np.triu_indices_from(pos_jaccard, k=1)]

            if len(triu_vals) > 0:
                within_sim[pos_name] = {
                    'mean': float(triu_vals.mean()),
                    'std': float(triu_vals.std()),
                    'n_pairs': len(triu_vals),
                }

        # Between-POS similarity (sample pairs for speed)
        between_sim = {}
        pos_list = list(unique_pos)
        for i, pos_i in enumerate(pos_list):
            for pos_j in pos_list[i+1:]:
                mask_i = sampled_labels == pos_i
                mask_j = sampled_labels == pos_j

                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    continue

                between_vals = jaccard[np.ix_(mask_i, mask_j)].flatten()
                if len(between_vals) > 0:
                    between_sim[f"{UPOS_TAGS[pos_i]}-{UPOS_TAGS[pos_j]}"] = {
                        'mean': float(between_vals.mean()),
                        'std': float(between_vals.std()),
                    }

        all_within = [v['mean'] for v in within_sim.values()]
        all_between = [v['mean'] for v in between_sim.values()]

        return {
            'within_pos': within_sim,
            'between_pos': between_sim,
            'mean_within': float(np.mean(all_within)) if all_within else 0.0,
            'mean_between': float(np.mean(all_between)) if all_between else 0.0,
            'separation': float(np.mean(all_within) - np.mean(all_between)) if all_within and all_between else 0.0,
        }

    def _compute_silhouette(self, masks: np.ndarray, pos_labels: np.ndarray) -> Dict:
        """Compute silhouette score using Jaccard distance."""
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples
        except ImportError:
            return {'score': None, 'error': 'sklearn not available'}

        unique_pos, counts = np.unique(pos_labels, return_counts=True)
        valid_pos = unique_pos[counts >= 5]

        if len(valid_pos) < 2:
            return {'score': None, 'error': 'Not enough POS categories'}

        valid_mask = np.isin(pos_labels, valid_pos)
        filtered_masks = masks[valid_mask]
        filtered_labels = pos_labels[valid_mask]

        # Sample for speed
        max_samples = 1500
        if len(filtered_masks) > max_samples:
            indices = np.random.choice(len(filtered_masks), max_samples, replace=False)
            filtered_masks = filtered_masks[indices]
            filtered_labels = filtered_labels[indices]

        print("  Computing silhouette score...")
        jaccard_dist = 1.0 - self._compute_jaccard_matrix(filtered_masks)

        try:
            score = silhouette_score(jaccard_dist, filtered_labels, metric='precomputed')
            samples = silhouette_samples(jaccard_dist, filtered_labels, metric='precomputed')

            per_pos = {}
            for pos_idx in valid_pos:
                pos_name = UPOS_TAGS[pos_idx]
                pos_mask = filtered_labels == pos_idx
                if pos_mask.sum() > 0:
                    per_pos[pos_name] = float(samples[pos_mask].mean())

            return {
                'score': float(score),
                'per_pos': per_pos,
                'n_samples': len(filtered_masks),
                'n_pos_categories': len(valid_pos),
            }
        except Exception as e:
            return {'score': None, 'error': str(e)}

    def _compute_function_word_silhouette(self, masks: np.ndarray, pos_labels: np.ndarray) -> Dict:
        """Compute silhouette score for function words only.

        Function words (DET, AUX, CCONJ, ADP, etc.) should cluster better
        than content words since they have more consistent activation patterns.
        """
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples
        except ImportError:
            return {'score': None, 'error': 'sklearn not available'}

        # Filter to function words only
        function_pos_indices = [POS_TO_IDX[p] for p in POS_GROUPS['Function Words'] if p in POS_TO_IDX]
        function_mask = np.isin(pos_labels, function_pos_indices)

        if function_mask.sum() < 50:
            return {'score': None, 'error': 'Not enough function words'}

        func_masks = masks[function_mask]
        func_labels = pos_labels[function_mask]

        # Filter POS with enough samples
        unique_pos, counts = np.unique(func_labels, return_counts=True)
        valid_pos = unique_pos[counts >= 5]

        if len(valid_pos) < 2:
            return {'score': None, 'error': 'Not enough function word POS categories'}

        valid_mask = np.isin(func_labels, valid_pos)
        filtered_masks = func_masks[valid_mask]
        filtered_labels = func_labels[valid_mask]

        # Sample for speed
        max_samples = 1000
        if len(filtered_masks) > max_samples:
            indices = np.random.choice(len(filtered_masks), max_samples, replace=False)
            filtered_masks = filtered_masks[indices]
            filtered_labels = filtered_labels[indices]

        print("  Computing function word silhouette...")
        jaccard_dist = 1.0 - self._compute_jaccard_matrix(filtered_masks)

        try:
            score = silhouette_score(jaccard_dist, filtered_labels, metric='precomputed')
            samples = silhouette_samples(jaccard_dist, filtered_labels, metric='precomputed')

            per_pos = {}
            for pos_idx in valid_pos:
                pos_name = UPOS_TAGS[pos_idx]
                pos_mask = filtered_labels == pos_idx
                if pos_mask.sum() > 0:
                    per_pos[pos_name] = float(samples[pos_mask].mean())

            return {
                'score': float(score),
                'per_pos': per_pos,
                'n_samples': len(filtered_masks),
                'n_pos_categories': len(valid_pos),
                'pos_included': [UPOS_TAGS[i] for i in valid_pos],
            }
        except Exception as e:
            return {'score': None, 'error': str(e)}

    def _compute_deprel_silhouette(self, masks: np.ndarray, deprel_labels: np.ndarray) -> Dict:
        """Compute silhouette score using dependency relation labels.

        Hypothesis: If deprel silhouette > POS silhouette, neurons encode
        syntactic roles (nsubj, obj, amod) rather than just part-of-speech.
        """
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples
        except ImportError:
            return {'score': None, 'error': 'sklearn not available'}

        # Filter out unknown deprel
        valid_deprel_mask = deprel_labels >= 0
        if valid_deprel_mask.sum() < 50:
            return {'score': None, 'error': 'Not enough valid deprel labels'}

        filtered_masks = masks[valid_deprel_mask]
        filtered_labels = deprel_labels[valid_deprel_mask]

        # Filter deprel with enough samples (at least 5)
        unique_dep, counts = np.unique(filtered_labels, return_counts=True)
        valid_dep = unique_dep[counts >= 5]

        if len(valid_dep) < 2:
            return {'score': None, 'error': 'Not enough deprel categories'}

        valid_mask = np.isin(filtered_labels, valid_dep)
        filtered_masks = filtered_masks[valid_mask]
        filtered_labels = filtered_labels[valid_mask]

        # Sample for speed
        max_samples = 1500
        if len(filtered_masks) > max_samples:
            indices = np.random.choice(len(filtered_masks), max_samples, replace=False)
            filtered_masks = filtered_masks[indices]
            filtered_labels = filtered_labels[indices]

        print("  Computing deprel silhouette score...")
        jaccard_dist = 1.0 - self._compute_jaccard_matrix(filtered_masks)

        try:
            score = silhouette_score(jaccard_dist, filtered_labels, metric='precomputed')
            samples = silhouette_samples(jaccard_dist, filtered_labels, metric='precomputed')

            per_deprel = {}
            for dep_idx in valid_dep:
                if dep_idx < len(DEPREL_TAGS):
                    dep_name = DEPREL_TAGS[dep_idx]
                    dep_mask = filtered_labels == dep_idx
                    if dep_mask.sum() > 0:
                        per_deprel[dep_name] = float(samples[dep_mask].mean())

            return {
                'score': float(score),
                'per_deprel': per_deprel,
                'n_samples': len(filtered_masks),
                'n_deprel_categories': len(valid_dep),
                'deprel_included': [DEPREL_TAGS[i] for i in valid_dep if i < len(DEPREL_TAGS)],
            }
        except Exception as e:
            return {'score': None, 'error': str(e)}

    def _analyze_content_vs_function(self, masks: np.ndarray, pos_labels: np.ndarray,
                                      is_whole_word: np.ndarray) -> Dict:
        """Analyze content words vs function words."""
        content_pos = [POS_TO_IDX[p] for p in POS_GROUPS['Content Words'] if p in POS_TO_IDX]
        function_pos = [POS_TO_IDX[p] for p in POS_GROUPS['Function Words'] if p in POS_TO_IDX]

        content_mask = np.isin(pos_labels, content_pos)
        function_mask = np.isin(pos_labels, function_pos)

        content_density = masks[content_mask].mean() if content_mask.sum() > 0 else 0
        function_density = masks[function_mask].mean() if function_mask.sum() > 0 else 0

        content_whole = content_mask & is_whole_word
        content_partial = content_mask & ~is_whole_word

        return {
            'content_word_density': float(content_density),
            'function_word_density': float(function_density),
            'n_content': int(content_mask.sum()),
            'n_function': int(function_mask.sum()),
            'whole_word_density': float(masks[content_whole].mean()) if content_whole.sum() > 0 else 0,
            'partial_word_density': float(masks[content_partial].mean()) if content_partial.sum() > 0 else 0,
            'n_whole_content': int(content_whole.sum()),
            'n_partial_content': int(content_partial.sum()),
        }

    def _compute_token_stats(self) -> Dict:
        """Compute basic token statistics."""
        pos_counts = defaultdict(int)
        activation_counts = []

        for t in self.token_data:
            pos_counts[t['pos']] += 1
            activation_counts.append(t['n_active'])

        return {
            'pos_distribution': dict(pos_counts),
            'mean_active_neurons': float(np.mean(activation_counts)) if activation_counts else 0,
            'std_active_neurons': float(np.std(activation_counts)) if activation_counts else 0,
            'total_neurons': self.total_neurons,
        }

    def _compute_layer_divergence(self, target_tokens: List[str] = None) -> Dict:
        """
        Compute layer-wise divergence for same tokens.

        Measures how activations of the same token diverge across layers.
        Early layers should show high similarity, later layers more divergence.

        Args:
            target_tokens: Specific tokens to analyze (default: common function words)

        Returns:
            Dict with divergence curves per token
        """
        if target_tokens is None:
            target_tokens = ['the', 'is', 'a', 'to', 'and', 'of', 'in', 'that', 'it', 'for']

        print("  Computing layer divergence...")

        results = {}
        for token in target_tokens:
            token_lower = token.lower().strip()
            occurrences = self.layer_token_data.get(token_lower, [])

            if len(occurrences) < 5:
                continue

            # Get all layer indices
            layer_indices = sorted(occurrences[0].get('layer_masks', {}).keys())
            if not layer_indices:
                continue

            # Compute within-occurrence similarity per layer
            layer_sims = []
            for layer_idx in layer_indices:
                # Collect masks for this layer across all occurrences
                layer_masks = []
                for occ in occurrences[:50]:  # Limit for speed
                    lm = occ.get('layer_masks', {}).get(layer_idx)
                    if lm is not None:
                        layer_masks.append(lm)

                if len(layer_masks) < 2:
                    layer_sims.append(None)
                    continue

                # Compute mean pairwise Jaccard
                masks_arr = np.array(layer_masks)
                jaccard = self._compute_jaccard_matrix(masks_arr)
                triu_vals = jaccard[np.triu_indices_from(jaccard, k=1)]
                layer_sims.append(float(triu_vals.mean()) if len(triu_vals) > 0 else None)

            results[token] = {
                'layer_indices': layer_indices,
                'similarities': layer_sims,
                'n_occurrences': len(occurrences),
                'divergence': layer_sims[0] - layer_sims[-1] if layer_sims[0] and layer_sims[-1] else None,
            }

        # Summary
        divergences = [r['divergence'] for r in results.values() if r['divergence'] is not None]

        return {
            'per_token': results,
            'mean_divergence': float(np.mean(divergences)) if divergences else None,
            'tokens_analyzed': list(results.keys()),
        }

    def _compute_content_hierarchy(self, masks: np.ndarray, pos_labels: np.ndarray,
                                    is_whole_word: np.ndarray) -> Dict:
        """
        Compute hierarchical clustering for content words.

        Uses scipy hierarchical clustering to find structure among
        NOUN, VERB, ADJ tokens (whole words only).

        Returns:
            Dict with linkage data and cluster quality metrics
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            return {'error': 'scipy not available'}

        print("  Computing content word hierarchy...")

        # Filter to content words (whole words only)
        content_pos_indices = [POS_TO_IDX[p] for p in ['NOUN', 'VERB', 'ADJ'] if p in POS_TO_IDX]
        content_mask = np.isin(pos_labels, content_pos_indices) & is_whole_word

        if content_mask.sum() < 20:
            return {'error': 'Not enough content words'}

        content_masks = masks[content_mask]
        content_labels = pos_labels[content_mask]

        # Sample if too large
        max_samples = 300
        if len(content_masks) > max_samples:
            indices = np.random.choice(len(content_masks), max_samples, replace=False)
            content_masks = content_masks[indices]
            content_labels = content_labels[indices]

        # Compute Jaccard distance matrix
        jaccard_sim = self._compute_jaccard_matrix(content_masks)
        jaccard_dist = 1.0 - jaccard_sim
        np.fill_diagonal(jaccard_dist, 0)  # Ensure diagonal is 0

        # Hierarchical clustering (Ward's method on distances)
        try:
            # Convert to condensed form for linkage
            condensed_dist = squareform(jaccard_dist, checks=False)
            Z = linkage(condensed_dist, method='average')

            # Cut at different levels and measure POS purity
            purities = {}
            for n_clusters in [3, 5, 8, 10]:
                if n_clusters >= len(content_masks):
                    continue
                cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

                # Compute POS purity per cluster
                cluster_purities = []
                for c in range(1, n_clusters + 1):
                    c_mask = cluster_labels == c
                    if c_mask.sum() == 0:
                        continue
                    c_labels = content_labels[c_mask]
                    _, counts = np.unique(c_labels, return_counts=True)
                    cluster_purities.append(counts.max() / counts.sum())

                purities[n_clusters] = float(np.mean(cluster_purities)) if cluster_purities else 0

            # POS separation quality
            pos_separation = {}
            for pos_idx in content_pos_indices:
                pos_name = UPOS_TAGS[pos_idx]
                pos_mask = content_labels == pos_idx
                if pos_mask.sum() < 3:
                    continue

                # Within vs between POS distance
                pos_dist = jaccard_dist[np.ix_(pos_mask, pos_mask)]
                other_dist = jaccard_dist[np.ix_(pos_mask, ~pos_mask)]

                within_mean = pos_dist[np.triu_indices_from(pos_dist, k=1)].mean()
                between_mean = other_dist.mean() if other_dist.size > 0 else 0

                pos_separation[pos_name] = {
                    'within_dist': float(within_mean),
                    'between_dist': float(between_mean),
                    'separation': float(between_mean - within_mean),
                    'n_samples': int(pos_mask.sum()),
                }

            return {
                'n_samples': len(content_masks),
                'cluster_purities': purities,
                'pos_separation': pos_separation,
                '_linkage': Z,  # For dendrogram plotting
                '_labels': content_labels,
            }
        except Exception as e:
            return {'error': str(e)}

    def _compute_semantic_correlation(self, masks: np.ndarray, weights: np.ndarray,
                                       pos_labels: np.ndarray, is_whole_word: np.ndarray,
                                       token_data: List[Dict] = None) -> Dict:
        """
        Compute correlation between neuron combination similarity and semantic similarity.

        Hypothesis: Neuron combinations encode semantic similarity, not just POS.
        If true, tokens with similar meanings should have similar neuron patterns.

        Uses:
        - GloVe embeddings for semantic similarity (cosine)
        - Continuous weights for neuron similarity (cosine)

        Args:
            masks: Binary masks array
            weights: Continuous weights array
            pos_labels: POS label indices
            is_whole_word: Whole word flags
            token_data: Token data list (if None, uses self.token_data)

        Returns:
            Dict with correlation, p-value, scatter data for plotting
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return {'error': 'scipy not available'}

        print("  Computing semantic correlation (using continuous weights)...")

        # Use provided token_data or fall back to self.token_data
        if token_data is None:
            token_data = self.token_data

        # Filter to content words (NOUN, VERB, ADJ) that are whole words
        content_pos_indices = [POS_TO_IDX[p] for p in ['NOUN', 'VERB', 'ADJ'] if p in POS_TO_IDX]
        content_mask = np.isin(pos_labels, content_pos_indices) & is_whole_word

        if content_mask.sum() < 50:
            return {'error': 'Not enough content whole words'}

        content_indices = np.where(content_mask)[0]

        # Get token strings for content words
        content_tokens = []
        for idx in content_indices:
            token_str = token_data[idx]['token_str'].lower().strip()
            # Clean token (remove ## prefix if any, strip whitespace markers)
            token_str = token_str.lstrip('##').lstrip('Ġ').lstrip('▁').strip()
            if len(token_str) >= 2:  # Skip single chars
                content_tokens.append((idx, token_str))

        if len(content_tokens) < 50:
            return {'error': 'Not enough valid content tokens'}

        # Try to load GloVe embeddings
        embeddings = self._load_glove_embeddings()
        if embeddings is None:
            return {'error': 'Could not load embeddings (GloVe not available)'}

        # Filter to tokens with embeddings
        tokens_with_emb = []
        for idx, token in content_tokens:
            if token in embeddings:
                tokens_with_emb.append((idx, token, embeddings[token]))

        if len(tokens_with_emb) < 30:
            return {'error': f'Only {len(tokens_with_emb)} tokens have embeddings'}

        print(f"    {len(tokens_with_emb)} tokens with embeddings")

        # Sample pairs for correlation (avoid O(n^2) explosion)
        n_pairs = min(2000, len(tokens_with_emb) * (len(tokens_with_emb) - 1) // 2)

        semantic_sims = []
        neuron_sims = []
        scatter_data = []

        # Generate random pairs
        pairs_sampled = set()

        while len(semantic_sims) < n_pairs and len(pairs_sampled) < n_pairs * 10:
            i, j = np.random.choice(len(tokens_with_emb), 2, replace=False)
            if i > j:
                i, j = j, i
            if (i, j) in pairs_sampled:
                continue
            pairs_sampled.add((i, j))

            idx1, token1, emb1 = tokens_with_emb[i]
            idx2, token2, emb2 = tokens_with_emb[j]

            # Semantic similarity (cosine on GloVe)
            sem_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

            # Neuron similarity (cosine on continuous weights)
            w1 = weights[idx1]
            w2 = weights[idx2]
            norm1 = np.linalg.norm(w1)
            norm2 = np.linalg.norm(w2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                neu_sim = float(np.dot(w1, w2) / (norm1 * norm2))
            else:
                neu_sim = 0.0

            semantic_sims.append(sem_sim)
            neuron_sims.append(neu_sim)
            scatter_data.append({
                'token1': token1,
                'token2': token2,
                'semantic_sim': sem_sim,
                'neuron_sim': neu_sim,
            })

        if len(semantic_sims) < 30:
            return {'error': 'Not enough pairs sampled'}

        # Compute Spearman correlation
        correlation, p_value = spearmanr(semantic_sims, neuron_sims)

        # Summary stats
        semantic_arr = np.array(semantic_sims)
        neuron_arr = np.array(neuron_sims)

        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'n_pairs': len(semantic_sims),
            'n_tokens_with_embeddings': len(tokens_with_emb),
            'semantic_sim_mean': float(semantic_arr.mean()),
            'semantic_sim_std': float(semantic_arr.std()),
            'neuron_sim_mean': float(neuron_arr.mean()),
            'neuron_sim_std': float(neuron_arr.std()),
            '_scatter_data': scatter_data[:500],  # Limit for JSON
            '_semantic_sims': semantic_arr,
            '_neuron_sims': neuron_arr,
        }

    def _load_glove_embeddings(self, dim: int = 300) -> Optional[Dict[str, np.ndarray]]:
        """
        Load GloVe embeddings via gensim (auto-downloads and caches).

        Uses class-level cache to avoid reloading across layers/instances.

        Args:
            dim: Embedding dimension (50, 100, 200, 300)

        Returns:
            Dict mapping words to embedding vectors, or None
        """
        # Check class-level cache first
        if (TokenCombinationAnalyzer._glove_cache is not None and
            TokenCombinationAnalyzer._glove_cache_dim == dim):
            print(f"    Using cached GloVe-{dim}d embeddings")
            return TokenCombinationAnalyzer._glove_cache

        try:
            import gensim.downloader as api
            print(f"    Loading GloVe-{dim}d via gensim (auto-download if needed)...")
            model = api.load(f'glove-wiki-gigaword-{dim}')

            # Convert to dict for fast lookup
            embeddings = {word: model[word] for word in model.key_to_index}
            print(f"    Loaded {len(embeddings):,} word embeddings")

            # Cache at class level
            TokenCombinationAnalyzer._glove_cache = embeddings
            TokenCombinationAnalyzer._glove_cache_dim = dim

            return embeddings
        except Exception as e:
            print(f"    GloVe loading failed: {e}")
            print("    Install gensim: pip install gensim")
            return None

    def print_summary(self, results: Dict):
        """Print analysis summary."""
        print("\n" + "=" * 70)
        print("TOKEN-BASED NEURON COMBINATION ANALYSIS")
        print("=" * 70)

        n_tokens = results.get('n_tokens', 0)
        n_neurons = results.get('n_neurons', 0)
        print(f"\nTokens analyzed: {n_tokens:,}")
        print(f"Total neurons: {n_neurons:,}")

        # Mask mode stats
        if self._mask_mode_stats['actual'] > 0 or self._mask_mode_stats['fallback'] > 0:
            actual = self._mask_mode_stats['actual']
            fallback = self._mask_mode_stats['fallback']
            total = actual + fallback
            if fallback == 0:
                print(f"Mask mode: actual tau-based masks ({actual}/{total} pools) ✓")
            elif actual == 0:
                print(f"Mask mode: fallback weights > threshold ({fallback}/{total} pools)")
            else:
                print(f"Mask mode: mixed ({actual} actual, {fallback} fallback)")

        # Active neurons per token
        token_stats = results.get('token_stats', {})
        if token_stats:
            mean_active = token_stats.get('mean_active_neurons', 0)
            std_active = token_stats.get('std_active_neurons', 0)
            print(f"Mean active neurons per token: {mean_active:.1f} ± {std_active:.1f} / {n_neurons}")

        # Silhouette score
        sil = results.get('silhouette_score', {})
        if sil.get('score') is not None:
            print(f"\n┌─ Silhouette Score (POS clustering quality) ───────────────────────────┐")
            print(f"│  Overall: {sil['score']:.4f}  (range: -1 to 1, higher = better)")
            print(f"│  Samples: {sil.get('n_samples', 0)}, POS categories: {sil.get('n_pos_categories', 0)}")

            per_pos = sil.get('per_pos', {})
            if per_pos:
                print(f"│")
                print(f"│  Per-POS silhouette:")
                sorted_pos = sorted(per_pos.items(), key=lambda x: -x[1])
                for pos, score in sorted_pos[:8]:
                    bar = "█" * int(max(0, score + 1) * 10)
                    print(f"│    {pos:<8} {score:>7.3f} {bar}")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # Function word silhouette (should be higher than overall)
        func_sil = results.get('function_word_silhouette', {})
        if func_sil.get('score') is not None:
            print(f"\n┌─ Function Word Silhouette (DET/AUX/CCONJ/ADP/etc.) ────────────────────┐")
            print(f"│  Score: {func_sil['score']:.4f}  (expected higher than overall)")
            print(f"│  Samples: {func_sil.get('n_samples', 0)}, Categories: {func_sil.get('n_pos_categories', 0)}")
            pos_list = func_sil.get('pos_included', [])
            if pos_list:
                print(f"│  POS: {', '.join(pos_list)}")

            per_pos = func_sil.get('per_pos', {})
            if per_pos:
                print(f"│")
                sorted_pos = sorted(per_pos.items(), key=lambda x: -x[1])
                for pos, score in sorted_pos:
                    bar = "█" * int(max(0, score + 1) * 10)
                    print(f"│    {pos:<8} {score:>7.3f} {bar}")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # POS similarity
        pos_sim = results.get('pos_similarity', {})
        if pos_sim:
            print(f"\n┌─ POS Jaccard Similarity ──────────────────────────────────────────────┐")
            print(f"│  Mean within-POS:  {pos_sim.get('mean_within', 0):.4f}")
            print(f"│  Mean between-POS: {pos_sim.get('mean_between', 0):.4f}")
            print(f"│  Separation:       {pos_sim.get('separation', 0):.4f}")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # Layer divergence
        layer_div = results.get('layer_divergence')
        if layer_div and layer_div.get('per_token'):
            print(f"\n┌─ Layer Divergence (same token across layers) ─────────────────────────┐")
            print(f"│  Mean divergence (L0 - Lmax): {layer_div.get('mean_divergence', 0):.4f}")
            print(f"│")
            print(f"│  {'Token':<10} {'L0 sim':>8} {'Lmax sim':>9} {'Divergence':>11}")
            print(f"│  {'─'*10} {'─'*8} {'─'*9} {'─'*11}")

            for token, data in list(layer_div['per_token'].items())[:8]:
                sims = data['similarities']
                l0 = sims[0] if sims[0] is not None else 0
                lmax = sims[-1] if sims[-1] is not None else 0
                div = data['divergence'] if data['divergence'] is not None else 0
                print(f"│  {token:<10} {l0:>8.3f} {lmax:>9.3f} {div:>+11.3f}")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # Content hierarchy
        hierarchy = results.get('content_hierarchy', {})
        if hierarchy and not hierarchy.get('error'):
            print(f"\n┌─ Content Word Hierarchy (NOUN/VERB/ADJ) ──────────────────────────────┐")
            print(f"│  Samples: {hierarchy.get('n_samples', 0)}")

            purities = hierarchy.get('cluster_purities', {})
            if purities:
                print(f"│  Cluster purity by k:")
                for k, p in sorted(purities.items()):
                    print(f"│    k={k}: {p:.3f}")

            pos_sep = hierarchy.get('pos_separation', {})
            if pos_sep:
                print(f"│")
                print(f"│  POS separation (between - within distance):")
                for pos, data in pos_sep.items():
                    print(f"│    {pos:<6} sep={data['separation']:>+.3f} (n={data['n_samples']})")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # Content vs Function
        content = results.get('content_analysis', {})
        if content:
            print(f"\n┌─ Content vs Function Words ───────────────────────────────────────────┐")
            print(f"│  Content word density:  {content.get('content_word_density', 0):.4f} ({content.get('n_content', 0)} tokens)")
            print(f"│  Function word density: {content.get('function_word_density', 0):.4f} ({content.get('n_function', 0)} tokens)")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        # Semantic correlation
        sem_corr = results.get('semantic_correlation', {})
        if sem_corr and not sem_corr.get('error'):
            corr = sem_corr.get('correlation', 0)
            p_val = sem_corr.get('p_value', 1)
            n_pairs = sem_corr.get('n_pairs', 0)
            n_tokens = sem_corr.get('n_tokens_with_embeddings', 0)

            print(f"\n┌─ Semantic Correlation (GloVe vs Neuron Similarity) ───────────────────┐")
            print(f"│  Spearman correlation: {corr:+.4f} (p={p_val:.2e})")
            print(f"│  {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
            print(f"│")
            print(f"│  Tokens with embeddings: {n_tokens}")
            print(f"│  Pairs sampled: {n_pairs}")
            print(f"│")
            print(f"│  Semantic sim:  mean={sem_corr.get('semantic_sim_mean', 0):.3f}, std={sem_corr.get('semantic_sim_std', 0):.3f}")
            print(f"│  Neuron sim:    mean={sem_corr.get('neuron_sim_mean', 0):.3f}, std={sem_corr.get('neuron_sim_std', 0):.3f}")
            print(f"│")
            if corr > 0.3:
                print(f"│  → Strong positive: similar meanings → similar neurons")
            elif corr > 0.1:
                print(f"│  → Weak positive: some semantic encoding in neuron patterns")
            elif corr > -0.1:
                print(f"│  → Near zero: neurons encode something else (syntax?)")
            else:
                print(f"│  → Negative: opposite semantic = similar neurons (?)")
            print(f"└───────────────────────────────────────────────────────────────────────┘")
        elif sem_corr and sem_corr.get('error'):
            print(f"\n┌─ Semantic Correlation ────────────────────────────────────────────────┐")
            print(f"│  Skipped: {sem_corr.get('error')}")
            print(f"└───────────────────────────────────────────────────────────────────────┘")

        print("\n" + "=" * 70)

    def visualize(self, results: Dict, output_dir: str) -> Dict[str, str]:
        """Generate visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        path = self._plot_pos_similarity_heatmap(results, os.path.join(output_dir, 'pos_jaccard_heatmap.png'))
        if path:
            paths['pos_similarity_heatmap'] = path

        path = self._plot_silhouette_per_pos(results, os.path.join(output_dir, 'silhouette_per_pos.png'))
        if path:
            paths['silhouette_per_pos'] = path

        path = self._plot_layer_divergence(results, os.path.join(output_dir, 'layer_divergence.png'))
        if path:
            paths['layer_divergence'] = path

        path = self._plot_content_dendrogram(results, os.path.join(output_dir, 'content_dendrogram.png'))
        if path:
            paths['content_dendrogram'] = path

        path = self._plot_semantic_correlation(results, os.path.join(output_dir, 'semantic_correlation.png'))
        if path:
            paths['semantic_correlation'] = path

        return paths

    def _plot_pos_similarity_heatmap(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot POS similarity matrix."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        pos_sim = results.get('pos_similarity', {})
        within = pos_sim.get('within_pos', {})
        between = pos_sim.get('between_pos', {})

        if not within:
            return None

        active_pos = list(within.keys())
        n = len(active_pos)

        sim_matrix = np.zeros((n, n))
        for i, pos_i in enumerate(active_pos):
            sim_matrix[i, i] = within[pos_i]['mean']
            for j, pos_j in enumerate(active_pos[i+1:], i+1):
                key = f"{pos_i}-{pos_j}"
                key_rev = f"{pos_j}-{pos_i}"
                val = between.get(key, between.get(key_rev, {})).get('mean', 0)
                sim_matrix[i, j] = val
                sim_matrix[j, i] = val

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(active_pos, rotation=45, ha='right')
        ax.set_yticklabels(active_pos)
        ax.set_title('POS Jaccard Similarity\n(diagonal=within, off-diagonal=between)')

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if sim_matrix[i, j] > 0.25 else 'black')

        plt.colorbar(im, ax=ax, label='Jaccard Similarity')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_silhouette_per_pos(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot silhouette score per POS."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        sil = results.get('silhouette_score', {})
        per_pos = sil.get('per_pos', {})

        if not per_pos:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_pos = sorted(per_pos.items(), key=lambda x: -x[1])
        pos_names = [p[0] for p in sorted_pos]
        scores = [p[1] for p in sorted_pos]

        colors = ['green' if s > 0 else 'red' for s in scores]
        ax.barh(range(len(pos_names)), scores, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=sil.get('score', 0), color='blue', linestyle='--', linewidth=2,
                   label=f"Overall: {sil.get('score', 0):.3f}")

        ax.set_yticks(range(len(pos_names)))
        ax.set_yticklabels(pos_names)
        ax.set_xlabel('Silhouette Score')
        ax.set_title('Silhouette Score per POS')
        ax.legend()
        ax.set_xlim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_layer_divergence(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot layer divergence curves."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        layer_div = results.get('layer_divergence')
        if not layer_div or not layer_div.get('per_token'):
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        for token, data in layer_div['per_token'].items():
            sims = data['similarities']
            layers = data['layer_indices']
            valid = [(l, s) for l, s in zip(layers, sims) if s is not None]
            if len(valid) >= 2:
                ls, ss = zip(*valid)
                ax.plot(ls, ss, 'o-', label=f"'{token}' (n={data['n_occurrences']})", alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Within-token Jaccard Similarity')
        ax.set_title('Layer Divergence: Same Token Similarity Across Layers\n(decreasing = context specialization)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_content_dendrogram(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot content word dendrogram."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            from scipy.cluster.hierarchy import dendrogram
        except ImportError:
            return None

        hierarchy = results.get('content_hierarchy', {})
        if not hierarchy or hierarchy.get('error') or '_linkage' not in hierarchy:
            return None

        Z = hierarchy['_linkage']
        labels = hierarchy['_labels']

        fig, ax = plt.subplots(figsize=(14, 6))

        # Color by POS
        label_colors = {
            POS_TO_IDX.get('NOUN', -1): 'blue',
            POS_TO_IDX.get('VERB', -1): 'red',
            POS_TO_IDX.get('ADJ', -1): 'green',
        }

        dendrogram(
            Z, ax=ax,
            leaf_rotation=90,
            leaf_font_size=6,
            color_threshold=0.7 * max(Z[:, 2]),
        )

        ax.set_xlabel('Token Index')
        ax.set_ylabel('Jaccard Distance')
        ax.set_title('Content Word Hierarchical Clustering (NOUN/VERB/ADJ)')

        # Add legend for POS
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='NOUN'),
            Patch(facecolor='red', label='VERB'),
            Patch(facecolor='green', label='ADJ'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_semantic_correlation(self, results: Dict, output_path: str) -> Optional[str]:
        """Plot semantic vs neuron similarity scatter plot with regression line."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            from scipy import stats
        except ImportError:
            return None

        sem_corr = results.get('semantic_correlation', {})
        if not sem_corr or sem_corr.get('error'):
            return None

        semantic_sims = sem_corr.get('_semantic_sims')
        neuron_sims = sem_corr.get('_neuron_sims')

        if semantic_sims is None or neuron_sims is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot with alpha for density
        ax.scatter(semantic_sims, neuron_sims, alpha=0.3, s=10, c='steelblue')

        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(semantic_sims, neuron_sims)
        x_line = np.linspace(semantic_sims.min(), semantic_sims.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression (r={r_value:.3f})')

        # Add Spearman correlation annotation
        corr = sem_corr.get('correlation', 0)
        p_val = sem_corr.get('p_value', 1)
        n_pairs = sem_corr.get('n_pairs', 0)

        ax.text(0.05, 0.95, f"Spearman ρ = {corr:.3f}\np = {p_val:.2e}\nn = {n_pairs} pairs",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Semantic Similarity (GloVe cosine)', fontsize=12)
        ax.set_ylabel('Neuron Similarity (Jaccard)', fontsize=12)
        ax.set_title('Semantic vs Neuron Combination Similarity\n(Do similar meanings activate similar neurons?)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def run_all(
        self,
        output_dir: str = './token_combination',
        max_sentences: int = 2000,
        split: str = 'train',
        data_path: str = None,
        analyze_layer_divergence: bool = True,
    ) -> Dict:
        """
        Run full token combination analysis.

        Args:
            output_dir: Output directory
            max_sentences: Maximum sentences to process
            split: Dataset split ('train', 'dev', 'test')
            data_path: Path to local conllu file
            analyze_layer_divergence: Whether to analyze layer divergence

        Returns:
            Analysis results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        dataset = self.load_ud_dataset(split, max_sentences, data_path)
        results = self.analyze_dataset(dataset, max_sentences, analyze_layer_divergence)

        self.print_summary(results)

        results_for_json = {
            k: v for k, v in results.items()
            if not k.startswith('_')
        }

        viz_paths = self.visualize(results, output_dir)
        results_for_json['visualizations'] = viz_paths

        import json
        results_path = os.path.join(output_dir, 'token_combination_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)

        results['visualizations'] = viz_paths
        return results

    @classmethod
    def run_layerwise_analysis(
        cls,
        model,
        tokenizer,
        n_layers: int,
        output_dir: str = './layerwise_results',
        max_sentences: int = 500,
        device: str = 'cuda',
    ) -> Dict:
        """
        Run layer-wise semantic emergence analysis (OPTIMIZED).

        Uses single forward pass per sentence to extract ALL layer data at once,
        instead of running separate forward passes for each layer.

        Analyzes how semantic information emerges across layers by measuring:
        1. Semantic correlation: GloVe similarity vs neuron weight similarity
        2. POS silhouette: How well neurons cluster by part-of-speech

        Hypothesis:
        - Early layers: Strong POS clustering (syntax)
        - Later layers: Strong semantic correlation (semantics)
        - Crossover point = syntax→semantics transition

        Args:
            model: DAWN model
            tokenizer: Tokenizer
            n_layers: Number of layers in the model
            output_dir: Output directory
            max_sentences: Max sentences to process
            device: Device for computation

        Returns:
            Dict with per-layer results and summary
        """
        import json as json_module
        import torch
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print("LAYER-WISE SEMANTIC EMERGENCE ANALYSIS (OPTIMIZED)")
        print("=" * 70)
        print(f"Layers: 0-{n_layers-1}")
        print(f"Max sentences: {max_sentences}")
        print(f"Forward passes: 1 per sentence (not {n_layers})")
        print()

        # Create single analyzer instance (no specific target layer)
        analyzer = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_layer=None,  # Will extract ALL layers
        )

        # Load dataset once
        dataset = analyzer.load_ud_dataset('train', max_sentences)
        print(f"Loaded {len(dataset)} sentences")

        # Process ALL sentences with ALL layers in ONE pass
        print("\n" + "=" * 50)
        print("PHASE 1: Extract all layers (single pass)")
        print("=" * 50)
        all_layer_data = analyzer.analyze_dataset_all_layers(
            dataset, n_layers, max_sentences
        )

        # Now compute metrics for each layer from stored data
        print("\n" + "=" * 50)
        print("PHASE 2: Compute per-layer metrics")
        print("=" * 50)

        layer_results_list = []
        for layer_idx in range(n_layers):
            print(f"\nLayer {layer_idx}:")
            token_data = all_layer_data.get(layer_idx, [])

            if not token_data:
                print(f"  No data collected")
                layer_data = {
                    'layer': layer_idx,
                    'n_tokens': 0,
                    'semantic_correlation': None,
                    'semantic_p_value': None,
                    'silhouette': None,
                    'function_silhouette': None,
                    'deprel_silhouette': None,
                }
            else:
                # Compute results for this layer
                layer_results = analyzer.compute_layer_results(token_data, layer_idx)

                sem_corr = layer_results.get('semantic_correlation', {})
                silhouette = layer_results.get('silhouette_score', {})
                func_sil = layer_results.get('function_word_silhouette', {})
                deprel_sil = layer_results.get('deprel_silhouette', {})

                layer_data = {
                    'layer': layer_idx,
                    'n_tokens': layer_results.get('n_tokens', 0),
                    'semantic_correlation': sem_corr.get('correlation', None),
                    'semantic_p_value': sem_corr.get('p_value', None),
                    'silhouette': silhouette.get('score', None),
                    'function_silhouette': func_sil.get('score', None),
                    'deprel_silhouette': deprel_sil.get('score', None),
                }

                # Print summary
                if layer_data['semantic_correlation'] is not None:
                    print(f"  Semantic correlation: {layer_data['semantic_correlation']:.4f}")
                if layer_data['silhouette'] is not None:
                    print(f"  POS silhouette: {layer_data['silhouette']:.4f}")
                if layer_data['deprel_silhouette'] is not None:
                    print(f"  Deprel silhouette: {layer_data['deprel_silhouette']:.4f}")
                if layer_data['function_silhouette'] is not None:
                    print(f"  Function silhouette: {layer_data['function_silhouette']:.4f}")

            layer_results_list.append(layer_data)

        # Clear memory
        del all_layer_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Summary table
        print("\n" + "=" * 85)
        print("SUMMARY: Layer-wise Metrics")
        print("=" * 85)
        print(f"{'Layer':>5} | {'Sem Corr':>10} | {'POS Sil':>10} | {'Deprel Sil':>10} | {'Func Sil':>10}")
        print("-" * 65)

        for r in layer_results_list:
            sem = f"{r['semantic_correlation']:.4f}" if r['semantic_correlation'] is not None else "N/A"
            sil = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "N/A"
            dsil = f"{r['deprel_silhouette']:.4f}" if r.get('deprel_silhouette') is not None else "N/A"
            fsil = f"{r['function_silhouette']:.4f}" if r['function_silhouette'] is not None else "N/A"
            print(f"{r['layer']:>5} | {sem:>10} | {sil:>10} | {dsil:>10} | {fsil:>10}")

        # Save results
        output = {
            'per_layer': layer_results_list,
            'n_layers': n_layers,
            'max_sentences': max_sentences,
        }

        results_path = os.path.join(output_dir, 'layerwise_results.json')
        with open(results_path, 'w') as f:
            json_module.dump(output, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Generate plot
        plot_path = cls._plot_layerwise_results(layer_results_list, output_dir)
        if plot_path:
            print(f"Plot saved to: {plot_path}")
            output['plot_path'] = plot_path

        return output

    @staticmethod
    def _plot_layerwise_results(results: List[Dict], output_dir: str) -> Optional[str]:
        """Generate layer-wise plot with dual y-axis."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        valid_results = [r for r in results if r['semantic_correlation'] is not None and r['silhouette'] is not None]
        if len(valid_results) < 2:
            return None

        layers = [r['layer'] for r in valid_results]
        sem_corrs = [r['semantic_correlation'] for r in valid_results]
        silhouettes = [r['silhouette'] for r in valid_results]
        func_sils = [r['function_silhouette'] for r in valid_results if r['function_silhouette'] is not None]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Semantic correlation (left y-axis)
        color1 = 'tab:blue'
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Semantic Correlation (Spearman ρ)', color=color1, fontsize=12)
        line1 = ax1.plot(layers, sem_corrs, 'o-', color=color1, linewidth=2, markersize=8, label='Semantic Corr')
        ax1.tick_params(axis='y', labelcolor=color1)

        # POS silhouette (right y-axis)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color2, fontsize=12)
        line2 = ax2.plot(layers, silhouettes, 's--', color=color2, linewidth=2, markersize=8, label='POS Silhouette')

        if len(func_sils) == len(layers):
            line3 = ax2.plot(layers, func_sils, '^:', color='tab:orange', linewidth=2, markersize=6, label='Func Silhouette')
            lines = line1 + line2 + line3
        else:
            lines = line1 + line2

        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Layer-wise Semantic Emergence\n(Syntax vs Semantics)', fontsize=14)
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
        ax1.set_xticks(layers)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'layerwise_semantic.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path


# =============================================================================
# NEURON FEATURE ANALYZER (Matrix-optimized)
# =============================================================================

class NeuronFeatureAnalyzer:
    """
    Analyze what features each neuron responds to.

    Uses matrix operations for efficient computation:
    - activation_matrix.T @ feature_matrix computes all neuron-feature counts at once
    - Sparse matrices for memory efficiency

    Inverts the token->neuron perspective to neuron->token features.
    For each neuron, analyzes:
    - POS distribution of activating tokens
    - Sentence position distribution
    - Token frequency characteristics
    - Subword position (word-initial vs continuation)
    - Next token POS patterns

    Usage:
        nfa = NeuronFeatureAnalyzer(token_data, tokenizer)
        results = nfa.run_full_analysis()
    """

    # Position bins for sentence position analysis
    POSITION_BINS = ['start', 'early', 'middle', 'late', 'end']
    FREQ_BINS = ['high', 'med', 'low']

    def __init__(
        self,
        token_data: List[Dict],
        tokenizer,
        n_neurons: int = None,
        pool_order: List[Tuple[str, int]] = None,
    ):
        """
        Initialize analyzer.

        Args:
            token_data: List of token dicts from TokenCombinationAnalyzer
                       Each dict has: token_str, pos, mask, weights, is_whole_word, etc.
            tokenizer: Tokenizer instance
            n_neurons: Total number of neurons (auto-detected if None)
            pool_order: List of (pool_name, size) tuples for pool-specific analysis
        """
        self.token_data = token_data
        self.tokenizer = tokenizer
        self.n_tokens = len(token_data)
        self.pool_order = pool_order or []

        if n_neurons is None and token_data:
            # Use max mask length across all tokens (handles layerwise data with varying sizes)
            n_neurons = max(len(token['mask']) for token in token_data)
        self.n_neurons = n_neurons

        # Matrices (built lazily)
        self._activation_matrix = None  # sparse [n_tokens, n_neurons]
        self._pos_matrix = None         # [n_tokens, n_pos]
        self._position_matrix = None    # [n_tokens, 5]
        self._subword_matrix = None     # [n_tokens, 2]
        self._freq_matrix = None        # [n_tokens, 3]

        # Computed results
        self._neuron_totals = None      # [n_neurons] activation counts
        self._neuron_pos_counts = None  # [n_neurons, n_pos]
        self._neuron_pos_pct = None
        self._neuron_position_counts = None
        self._neuron_position_pct = None
        self._neuron_subword_counts = None
        self._neuron_subword_pct = None
        self._neuron_freq_counts = None
        self._neuron_freq_pct = None
        self._neuron_next_pos_counts = None
        self._neuron_next_pos_pct = None

        self.neuron_profiles = None

    def _build_matrices(self):
        """Build all feature matrices (one-time cost)."""
        if self._activation_matrix is not None:
            return

        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy required: pip install scipy")

        print("Building feature matrices...")
        n_tokens = self.n_tokens
        n_neurons = self.n_neurons
        n_pos = len(UPOS_TAGS)

        # 1. Sparse activation matrix [n_tokens, n_neurons]
        print("  Building activation matrix (sparse)...")
        rows, cols = [], []
        for tok_idx, token in enumerate(self.token_data):
            mask = token['mask']
            active_neurons = np.where(mask)[0]
            for neuron_idx in active_neurons:
                rows.append(tok_idx)
                cols.append(neuron_idx)

        self._activation_matrix = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_tokens, n_neurons)
        )
        print(f"    Shape: {self._activation_matrix.shape}, nnz: {self._activation_matrix.nnz:,}")

        # 2. POS matrix (one-hot) [n_tokens, n_pos]
        print("  Building POS matrix...")
        pos_indices = np.array([
            POS_TO_IDX.get(token.get('pos', 'X'), POS_TO_IDX.get('X', 0))
            for token in self.token_data
        ])
        self._pos_matrix = np.zeros((n_tokens, n_pos), dtype=np.float32)
        self._pos_matrix[np.arange(n_tokens), pos_indices] = 1

        # 3. Position matrix (one-hot) [n_tokens, 5]
        print("  Building position matrix...")
        position_bins = []
        for idx, token in enumerate(self.token_data):
            pos_in_seq = token.get('position', idx % 50)
            seq_len = token.get('seq_len', 50)
            bin_idx = self._get_position_bin_idx(pos_in_seq, seq_len)
            position_bins.append(bin_idx)
        position_bins = np.array(position_bins)

        self._position_matrix = np.zeros((n_tokens, 5), dtype=np.float32)
        self._position_matrix[np.arange(n_tokens), position_bins] = 1

        # 4. Subword matrix [n_tokens, 2] (word_initial, continuation)
        print("  Building subword matrix...")
        self._subword_matrix = np.zeros((n_tokens, 2), dtype=np.float32)
        for idx, token in enumerate(self.token_data):
            if self._is_word_initial(token['token_str']):
                self._subword_matrix[idx, 0] = 1  # word_initial
            else:
                self._subword_matrix[idx, 1] = 1  # continuation

        # 5. Frequency matrix [n_tokens, 3] (high, med, low)
        print("  Building frequency matrix...")
        freq_counter = defaultdict(int)
        for token in self.token_data:
            freq_counter[token['token_str'].lower().strip()] += 1

        all_freqs = list(freq_counter.values())
        q33, q66 = np.percentile(all_freqs, [33, 66])

        self._freq_matrix = np.zeros((n_tokens, 3), dtype=np.float32)
        for idx, token in enumerate(self.token_data):
            freq = freq_counter[token['token_str'].lower().strip()]
            if freq > q66:
                self._freq_matrix[idx, 0] = 1  # high
            elif freq > q33:
                self._freq_matrix[idx, 1] = 1  # med
            else:
                self._freq_matrix[idx, 2] = 1  # low

        print("  Matrices built!")

    def _compute_neuron_features(self):
        """Compute all neuron feature distributions via matrix multiplication."""
        if self._neuron_totals is not None:
            return

        self._build_matrices()

        print("Computing neuron feature distributions (matrix ops)...")
        act = self._activation_matrix

        # Neuron totals [n_neurons]
        self._neuron_totals = np.array(act.sum(axis=0)).flatten()

        # Avoid division by zero
        totals_safe = np.maximum(self._neuron_totals, 1)[:, None]

        # POS: [n_neurons, n_pos]
        print("  Computing POS distributions...")
        self._neuron_pos_counts = np.asarray(act.T @ self._pos_matrix)
        self._neuron_pos_pct = self._neuron_pos_counts / totals_safe * 100

        # Position: [n_neurons, 5]
        print("  Computing position distributions...")
        self._neuron_position_counts = np.asarray(act.T @ self._position_matrix)
        self._neuron_position_pct = self._neuron_position_counts / totals_safe * 100

        # Subword: [n_neurons, 2]
        print("  Computing subword distributions...")
        self._neuron_subword_counts = np.asarray(act.T @ self._subword_matrix)
        self._neuron_subword_pct = self._neuron_subword_counts / totals_safe * 100

        # Frequency: [n_neurons, 3]
        print("  Computing frequency distributions...")
        self._neuron_freq_counts = np.asarray(act.T @ self._freq_matrix)
        self._neuron_freq_pct = self._neuron_freq_counts / totals_safe * 100

        # Next POS: shift pos_matrix by 1 and multiply
        print("  Computing next-token POS distributions...")
        next_pos_matrix = np.zeros_like(self._pos_matrix)
        next_pos_matrix[:-1] = self._pos_matrix[1:]  # shift
        self._neuron_next_pos_counts = np.asarray(act.T @ next_pos_matrix)
        # For next_pos, total is n_tokens - 1 for each neuron (last token has no next)
        # Approximate with same totals
        self._neuron_next_pos_pct = self._neuron_next_pos_counts / totals_safe * 100

        print("  Done!")

    def _get_position_bin_idx(self, position: int, seq_len: int) -> int:
        """Map position to bin index (0-4)."""
        if seq_len <= 1:
            return 2  # middle

        ratio = position / (seq_len - 1) if seq_len > 1 else 0.5

        if ratio < 0.1:
            return 0  # start
        elif ratio < 0.3:
            return 1  # early
        elif ratio < 0.7:
            return 2  # middle
        elif ratio < 0.9:
            return 3  # late
        else:
            return 4  # end

    def _is_word_initial(self, token_str: str) -> bool:
        """Check if token is word-initial (not a continuation)."""
        if token_str.startswith(('Ġ', '▁', ' ')):
            return True
        if token_str.startswith('##'):
            return False
        return True

    def build_neuron_profiles(self) -> Dict[int, Dict]:
        """
        Build feature profiles for all neurons using precomputed matrices.

        Returns:
            Dict mapping neuron_idx -> feature profile
        """
        self._compute_neuron_features()

        print(f"\nBuilding profiles for {self.n_neurons} neurons...")

        profiles = {}
        active_neurons = np.where(self._neuron_totals > 0)[0]

        for neuron_idx in active_neurons:
            n_act = int(self._neuron_totals[neuron_idx])

            # POS
            pos_pct = self._neuron_pos_pct[neuron_idx]
            top_pos_idx = np.argmax(pos_pct)
            top_pos = UPOS_TAGS[top_pos_idx] if top_pos_idx < len(UPOS_TAGS) else 'X'

            # Position
            pos_position_pct = self._neuron_position_pct[neuron_idx]
            top_position_idx = np.argmax(pos_position_pct)
            top_position = self.POSITION_BINS[top_position_idx]

            # Subword
            subword_pct = self._neuron_subword_pct[neuron_idx]

            # Frequency
            freq_pct = self._neuron_freq_pct[neuron_idx]

            # Next POS
            next_pos_pct = self._neuron_next_pos_pct[neuron_idx]
            top_next_pos_idx = np.argmax(next_pos_pct)
            top_next_pos = UPOS_TAGS[top_next_pos_idx] if top_next_pos_idx < len(UPOS_TAGS) else 'X'

            profiles[int(neuron_idx)] = {
                'neuron_idx': int(neuron_idx),
                'n_activations': n_act,
                'pos': {
                    'top_pos': top_pos,
                    'top_pos_pct': float(pos_pct[top_pos_idx]),
                    'distribution': {UPOS_TAGS[i]: float(pos_pct[i]) for i in range(len(UPOS_TAGS)) if pos_pct[i] > 0},
                },
                'position': {
                    'dominant_position': top_position,
                    'dominant_pct': float(pos_position_pct[top_position_idx]),
                    'distribution': {self.POSITION_BINS[i]: float(pos_position_pct[i]) for i in range(5)},
                },
                'subword': {
                    'word_initial_pct': float(subword_pct[0]),
                    'continuation_pct': float(subword_pct[1]),
                },
                'frequency': {
                    'high_freq_pct': float(freq_pct[0]),
                    'med_freq_pct': float(freq_pct[1]),
                    'low_freq_pct': float(freq_pct[2]),
                },
                'next_pos': {
                    'top_next_pos': top_next_pos,
                    'top_next_pos_pct': float(next_pos_pct[top_next_pos_idx]),
                },
            }

        self.neuron_profiles = profiles
        print(f"Built profiles for {len(profiles)} active neurons")
        return profiles

    def detect_specialized_neurons(
        self,
        threshold: float = 0.8,
        min_activations: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Detect neurons specialized for specific features.

        A neuron is "specialized" if it has >= threshold concentration
        on a single feature value.

        Args:
            threshold: Minimum percentage for specialization (0-1)
            min_activations: Minimum activations to consider

        Returns:
            Dict with specialized neurons by feature type
        """
        if self.neuron_profiles is None:
            self.build_neuron_profiles()

        threshold_pct = threshold * 100

        specialized = {
            'pos': [],           # Specialized for specific POS
            'position': [],      # Specialized for sentence position
            'subword': [],       # Specialized for word-initial/continuation
            'frequency': [],     # Specialized for high/low frequency
            'next_pos': [],      # Specialized for predicting next POS
        }

        for neuron_idx, profile in self.neuron_profiles.items():
            if profile['n_activations'] < min_activations:
                continue

            # POS specialization
            pos_info = profile['pos']
            if pos_info.get('top_pos_pct', 0) >= threshold_pct:
                specialized['pos'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': pos_info['top_pos'],
                    'pct': pos_info['top_pos_pct'],
                    'n_activations': profile['n_activations'],
                })

            # Position specialization
            pos_dist = profile['position']
            if pos_dist.get('dominant_pct', 0) >= threshold_pct:
                specialized['position'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': pos_dist['dominant_position'],
                    'pct': pos_dist['dominant_pct'],
                    'n_activations': profile['n_activations'],
                })

            # Subword specialization
            subword = profile['subword']
            word_init_pct = subword.get('word_initial_pct', 50)
            cont_pct = subword.get('continuation_pct', 50)
            if word_init_pct >= threshold_pct:
                specialized['subword'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': 'word_initial',
                    'pct': word_init_pct,
                    'n_activations': profile['n_activations'],
                })
            elif cont_pct >= threshold_pct:
                specialized['subword'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': 'continuation',
                    'pct': cont_pct,
                    'n_activations': profile['n_activations'],
                })

            # Frequency specialization
            freq = profile['frequency']
            if freq.get('high_freq_pct', 0) >= threshold_pct:
                specialized['frequency'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': 'high_frequency',
                    'pct': freq['high_freq_pct'],
                    'n_activations': profile['n_activations'],
                })
            elif freq.get('low_freq_pct', 0) >= threshold_pct:
                specialized['frequency'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': 'low_frequency',
                    'pct': freq['low_freq_pct'],
                    'n_activations': profile['n_activations'],
                })

            # Next POS specialization
            next_pos = profile['next_pos']
            if next_pos.get('top_next_pos_pct', 0) >= threshold_pct:
                specialized['next_pos'].append({
                    'neuron': self._get_neuron_name(neuron_idx),
                    'neuron_idx': neuron_idx,
                    'pool': self._get_pool_for_neuron(neuron_idx),
                    'specialized_for': next_pos['top_next_pos'],
                    'pct': next_pos['top_next_pos_pct'],
                    'n_activations': profile['n_activations'],
                })

        # Sort by percentage
        for key in specialized:
            specialized[key] = sorted(specialized[key], key=lambda x: -x['pct'])

        return specialized

    def compute_selectivity_matrix(self, min_activations: int = 10) -> Dict:
        """
        Compute selectivity scores for all neurons across POS categories.

        Selectivity score = P(neuron active | POS) / P(neuron active)
        - > 1: neuron prefers this POS (more active than baseline)
        - = 1: neuron is indifferent to this POS
        - < 1: neuron avoids this POS (less active than baseline)

        Args:
            min_activations: Minimum activations to include neuron

        Returns:
            Dict with selectivity matrix, per-pool analysis, unified naming
        """
        self._compute_neuron_features()

        n_neurons = self.n_neurons
        n_pos = len(UPOS_TAGS)
        total_tokens = len(self.token_data)

        # POS totals: how many tokens of each POS
        pos_totals = self._pos_matrix.sum(axis=0)  # [n_pos]

        # Compute selectivity matrix [n_neurons, n_pos]
        selectivity = np.ones((n_neurons, n_pos), dtype=np.float32)

        # Active neurons (with enough activations)
        active_mask = self._neuron_totals >= min_activations
        active_indices = np.where(active_mask)[0]

        for n in active_indices:
            p_neuron = self._neuron_totals[n] / total_tokens  # P(neuron active)
            for p in range(n_pos):
                if pos_totals[p] > 0 and p_neuron > 0:
                    # P(neuron active | POS)
                    p_neuron_given_pos = self._neuron_pos_counts[n, p] / pos_totals[p]
                    selectivity[n, p] = p_neuron_given_pos / p_neuron

        # Get pool ranges for per-pool analysis
        pool_ranges = self._get_pool_ranges()

        # Top selective neurons per POS (with unified naming)
        top_selective = {}
        for p, pos_tag in enumerate(UPOS_TAGS):
            sel_scores = selectivity[active_mask, p]
            sel_indices = active_indices[np.argsort(-sel_scores)][:10]  # Top 10
            top_selective[pos_tag] = [
                {
                    'neuron': self._get_neuron_name(int(idx)),
                    'neuron_idx': int(idx),
                    'pool': self._get_pool_for_neuron(int(idx)),
                    'selectivity': round(float(selectivity[idx, p]), 2)
                }
                for idx in sel_indices
                if selectivity[idx, p] > 1.0
            ]

        # Per-pool analysis
        per_pool = {}
        for pool_name, (start_idx, end_idx) in pool_ranges.items():
            pool_size = end_idx - start_idx
            pool_active = active_mask[start_idx:end_idx]
            pool_selectivity = selectivity[start_idx:end_idx]
            n_active = int(pool_active.sum())

            # Top selective neurons in this pool
            pool_top = []
            if n_active > 0:
                for p, pos_tag in enumerate(UPOS_TAGS):
                    pool_sel = pool_selectivity[pool_active, p]
                    pool_indices = np.where(pool_active)[0]
                    if len(pool_indices) > 0:
                        top_idx = pool_indices[np.argmax(pool_sel)]
                        global_idx = start_idx + top_idx
                        if pool_selectivity[top_idx, p] > 1.5:  # Only strong selectivity
                            pool_top.append({
                                'neuron': self._get_neuron_name(global_idx),
                                'pos': pos_tag,
                                'selectivity': round(float(pool_selectivity[top_idx, p]), 2)
                            })

            per_pool[pool_name] = {
                'n_neurons': pool_size,
                'n_active': n_active,
                'top_selective': sorted(pool_top, key=lambda x: -x['selectivity'])[:10],
            }

        # Mean selectivity per POS
        mean_selectivity = {
            pos: round(float(selectivity[active_mask, p].mean()), 3)
            for p, pos in enumerate(UPOS_TAGS)
        }

        # Selectivity range per neuron
        selectivity_range = selectivity[active_mask].max(axis=1) - selectivity[active_mask].min(axis=1)

        # By-POS summary with top neurons
        by_pos = {}
        for p, pos_tag in enumerate(UPOS_TAGS):
            by_pos[pos_tag] = {
                'top_neurons': [n['neuron'] for n in top_selective[pos_tag][:5]],
                'mean_selectivity': mean_selectivity[pos_tag],
            }

        return {
            'selectivity_matrix': selectivity,  # [n_neurons, n_pos]
            'active_neuron_indices': active_indices.tolist(),
            'pos_tags': UPOS_TAGS,
            'pools_analyzed': list(pool_ranges.keys()),
            'total_neurons': n_neurons,
            # Per-pool analysis
            'per_pool': per_pool,
            # By POS analysis
            'by_pos': by_pos,
            'top_selective_per_pos': top_selective,
            'mean_selectivity_by_pos': mean_selectivity,
            'selectivity_range': {
                'mean': round(float(selectivity_range.mean()), 3),
                'std': round(float(selectivity_range.std()), 3),
                'max': round(float(selectivity_range.max()), 3),
            },
            'n_active_neurons': int(active_mask.sum()),
            'total_tokens': total_tokens,
        }

    def build_feature_vectors(self) -> Tuple[np.ndarray, List[int]]:
        """
        Build feature vectors for neuron clustering using precomputed matrices.

        Returns:
            (feature_matrix, neuron_indices) where feature_matrix is [n_active_neurons, n_features]
        """
        self._compute_neuron_features()

        # Get active neurons (those with activations)
        active_mask = self._neuron_totals > 0
        neuron_indices = list(np.where(active_mask)[0])

        if not neuron_indices:
            return np.array([]), []

        # Build feature matrix by concatenating precomputed percentages
        # Normalize to [0, 1] range
        feature_parts = []

        # POS distribution (top 10 POS categories): [n_neurons, 10]
        feature_parts.append(self._neuron_pos_pct[active_mask, :10] / 100.0)

        # Position distribution: [n_neurons, 5]
        feature_parts.append(self._neuron_position_pct[active_mask] / 100.0)

        # Frequency distribution: [n_neurons, 3]
        feature_parts.append(self._neuron_freq_pct[active_mask] / 100.0)

        # Subword (word_initial only, continuation is redundant): [n_neurons, 1]
        feature_parts.append(self._neuron_subword_pct[active_mask, :1] / 100.0)

        # Next POS (top 5): [n_neurons, 5]
        feature_parts.append(self._neuron_next_pos_pct[active_mask, :5] / 100.0)

        # Concatenate all features: [n_neurons, 10+5+3+1+5=24]
        feature_matrix = np.concatenate(feature_parts, axis=1)

        return feature_matrix, neuron_indices

    def cluster_neurons(self, n_clusters: int = 10, method: str = 'kmeans') -> Dict:
        """
        Cluster neurons by feature profile similarity.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'hierarchical')

        Returns:
            Dict with cluster assignments and cluster profiles
        """
        print(f"\nClustering neurons into {n_clusters} clusters...")

        feature_matrix, neuron_indices = self.build_feature_vectors()

        if len(feature_matrix) < n_clusters:
            return {'error': f'Not enough neurons ({len(feature_matrix)}) for {n_clusters} clusters'}

        try:
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(feature_matrix)
            else:
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(feature_matrix)
        except ImportError:
            return {'error': 'sklearn not available'}

        # Build index mapping for O(1) lookup
        neuron_to_idx = {n: i for i, n in enumerate(neuron_indices)}

        # Build cluster info
        clusters = defaultdict(list)
        for neuron_idx, label in zip(neuron_indices, labels):
            clusters[int(label)].append(neuron_idx)

        # Compute cluster profiles (mean feature vector)
        cluster_profiles = {}
        for label in range(n_clusters):
            cluster_neurons = clusters[label]
            if cluster_neurons:
                # Use precomputed index mapping (O(1) lookup)
                indices = [neuron_to_idx[n] for n in cluster_neurons]
                mean_features = feature_matrix[indices].mean(axis=0)
                cluster_profiles[label] = {
                    'n_neurons': len(cluster_neurons),
                    'neurons': cluster_neurons[:20],  # Sample
                    'mean_features': mean_features.tolist(),
                }

        # Compute silhouette score
        try:
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(feature_matrix, labels)
        except:
            sil_score = None

        return {
            'n_clusters': n_clusters,
            'cluster_assignments': {int(n): int(l) for n, l in zip(neuron_indices, labels)},
            'cluster_sizes': {int(k): len(v) for k, v in clusters.items()},
            'cluster_profiles': cluster_profiles,
            'silhouette_score': sil_score,
        }

    def print_summary(self, specialized: Dict = None, clusters: Dict = None):
        """Print analysis summary."""
        print("\n" + "=" * 70)
        print("NEURON FEATURE ANALYSIS SUMMARY")
        print("=" * 70)

        # Use precomputed totals if available
        if self._neuron_totals is not None:
            active_mask = self._neuron_totals > 0
            activations = self._neuron_totals[active_mask]
            print(f"\nTotal neurons profiled: {int(active_mask.sum())}")
            print(f"Activations per neuron: mean={np.mean(activations):.1f}, "
                  f"median={np.median(activations):.1f}")
        elif self.neuron_profiles:
            print(f"\nTotal neurons profiled: {len(self.neuron_profiles)}")
            activations = [p['n_activations'] for p in self.neuron_profiles.values()]
            print(f"Activations per neuron: mean={np.mean(activations):.1f}, "
                  f"median={np.median(activations):.1f}")

        if specialized:
            print("\n" + "-" * 50)
            print("SPECIALIZED NEURONS (80%+ concentration)")
            print("-" * 50)

            for feature, neurons in specialized.items():
                if neurons:
                    print(f"\n{feature.upper()}: {len(neurons)} specialized neurons")
                    for info in neurons[:5]:  # Top 5
                        print(f"  Neuron {info['neuron']}: {info['specialized_for']:12s} "
                              f"({info['pct']:.1f}%, {info['n_activations']} activations)")

        if clusters:
            print("\n" + "-" * 50)
            print("NEURON CLUSTERS")
            print("-" * 50)
            print(f"Number of clusters: {clusters.get('n_clusters', 0)}")
            if clusters.get('silhouette_score'):
                print(f"Silhouette score: {clusters['silhouette_score']:.4f}")
            print("\nCluster sizes:")
            for label, size in sorted(clusters.get('cluster_sizes', {}).items()):
                print(f"  Cluster {label}: {size} neurons")

    @classmethod
    def from_token_combination_analyzer(
        cls,
        tca: 'TokenCombinationAnalyzer',
    ) -> 'NeuronFeatureAnalyzer':
        """
        Create NeuronFeatureAnalyzer from TokenCombinationAnalyzer.

        Args:
            tca: TokenCombinationAnalyzer with collected token_data

        Returns:
            NeuronFeatureAnalyzer instance
        """
        if not tca.token_data:
            raise ValueError("TokenCombinationAnalyzer has no token_data. Run analyze_dataset first.")

        # Don't pass n_neurons - let __init__ auto-detect from max mask length
        # This handles layerwise data where masks have varying sizes
        return cls(
            token_data=tca.token_data,
            tokenizer=tca.tokenizer,
            pool_order=getattr(tca, 'pool_order', None),
        )

    def _get_pool_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Get neuron index ranges for each pool.

        Returns:
            Dict mapping pool_name -> (start_idx, end_idx)
        """
        ranges = {}
        offset = 0
        for pool_name, size in self.pool_order:
            ranges[pool_name] = (offset, offset + size)
            offset += size
        return ranges

    def _get_neuron_name(self, global_idx: int) -> str:
        """
        Convert global neuron index to unified naming format: {Pool}_{local_idx}

        Uses centralized POOL_DISPLAY_NAMES from utils for consistency.

        Examples:
            0 -> 'F_QK_0'
            64 -> 'F_V_0'
            328 -> 'R_QK_0'

        Args:
            global_idx: Global neuron index (0 to total_neurons-1)

        Returns:
            String like 'F_V_45', 'F_Know_12', etc.
        """
        pool_ranges = self._get_pool_ranges()
        for pool_name, (start_idx, end_idx) in pool_ranges.items():
            if start_idx <= global_idx < end_idx:
                local_idx = global_idx - start_idx
                return get_neuron_display_name(pool_name, local_idx)
        return f'Unknown_{global_idx}'

    def _get_pool_for_neuron(self, global_idx: int) -> str:
        """Get pool name for a global neuron index."""
        pool_ranges = self._get_pool_ranges()
        for pool_name, (start_idx, end_idx) in pool_ranges.items():
            if start_idx <= global_idx < end_idx:
                return pool_name
        return 'unknown'

    def _compute_utilization_stats(self, active_threshold: int = 1) -> Dict:
        """
        Compute forward-pass based neuron utilization statistics.

        This replaces EMA-based utilization (Table 2) with actual inference-time usage.

        Args:
            active_threshold: Minimum activations to consider neuron "active" (default: 1)

        Returns:
            Dict with per-pool utilization stats
        """
        self._compute_neuron_features()

        pool_ranges = self._get_pool_ranges()
        total_tokens = len(self.token_data)
        utilization = {}

        for pool_name, (start_idx, end_idx) in pool_ranges.items():
            pool_size = end_idx - start_idx
            pool_totals = self._neuron_totals[start_idx:end_idx]

            # Active neurons (activated at least once)
            active_mask = pool_totals >= active_threshold
            n_active = int(active_mask.sum())
            n_dead = pool_size - n_active

            # Usage statistics
            active_totals = pool_totals[active_mask]
            mean_usage = float(active_totals.mean()) if n_active > 0 else 0
            std_usage = float(active_totals.std()) if n_active > 0 else 0

            # Gini coefficient for usage inequality
            if n_active > 1:
                sorted_usage = np.sort(active_totals)
                n = len(sorted_usage)
                cumsum = np.cumsum(sorted_usage)
                gini = (2 * np.sum((np.arange(1, n + 1) * sorted_usage)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
            else:
                gini = 0.0

            utilization[pool_name] = {
                'display': pool_name.upper(),
                'total': pool_size,
                'active': n_active,
                'dead': n_dead,
                'active_ratio': round(n_active / pool_size, 4) if pool_size > 0 else 0,
                'gini': round(float(gini), 4),
                'stats': {
                    'mean': round(mean_usage, 2),
                    'std': round(std_usage, 2),
                    'min': int(pool_totals.min()) if pool_size > 0 else 0,
                    'max': int(pool_totals.max()) if pool_size > 0 else 0,
                },
                'method': 'forward_pass',  # Mark as forward-pass based
            }

        # Overall stats
        total_active = sum(u['active'] for u in utilization.values())
        total_neurons = sum(u['total'] for u in utilization.values())
        utilization['_overall'] = {
            'total_neurons': total_neurons,
            'total_active': total_active,
            'total_dead': total_neurons - total_active,
            'active_ratio': round(total_active / total_neurons, 4) if total_neurons > 0 else 0,
            'total_tokens_analyzed': total_tokens,
        }

        return utilization

    def run_full_analysis(self, output_dir: str = None) -> Dict:
        """
        Run complete neuron feature analysis.

        Args:
            output_dir: Directory to save results (optional)

        Returns:
            Dict with all analysis results
        """
        print("=" * 70)
        print("NEURON FEATURE ANALYSIS")
        print("=" * 70)

        # Build profiles
        profiles = self.build_neuron_profiles()

        # Detect specialized neurons at default threshold (80%)
        specialized = self.detect_specialized_neurons(threshold=0.8)

        # Multi-threshold specialization analysis for paper
        thresholds = [0.6, 0.7, 0.8]
        multi_threshold_results = {}
        for t in thresholds:
            spec_t = self.detect_specialized_neurons(threshold=t)
            multi_threshold_results[f'{int(t*100)}%'] = {
                k: len(v) for k, v in spec_t.items()
            }

        # Pool-specific specialization analysis
        pool_ranges = self._get_pool_ranges()
        pool_specialization = {}

        for pool_name, (start_idx, end_idx) in pool_ranges.items():
            pool_size = end_idx - start_idx
            pool_spec = {'total': pool_size}

            # Count specialized neurons in this pool at each threshold
            for t in thresholds:
                spec_t = self.detect_specialized_neurons(threshold=t)
                # Count POS-specialized neurons in this pool by pool name
                count = sum(
                    1 for n in spec_t.get('pos', [])
                    if n.get('pool') == pool_name
                )
                pool_spec[f'specialized_{int(t*100)}'] = count

            pool_specialization[pool_name] = pool_spec

        # Build specialization summary for paper
        total_neurons = self.n_neurons
        specialization_summary = {
            'total_neurons': total_neurons,
            'specialized_count': multi_threshold_results,
            'specialized_ratio': {
                k: {feat: round(cnt / total_neurons, 4) if total_neurons > 0 else 0
                    for feat, cnt in counts.items()}
                for k, counts in multi_threshold_results.items()
            },
            'by_pool': pool_specialization,
        }

        # Compute selectivity matrix for Fig 4 heatmap
        print("\n  Computing selectivity matrix...")
        selectivity_data = self.compute_selectivity_matrix(min_activations=10)

        # Compute forward-pass based utilization (replaces EMA-based Table 2)
        print("\n  Computing forward-pass utilization...")
        utilization = self._compute_utilization_stats()

        # Cluster neurons
        clusters = self.cluster_neurons(n_clusters=10)

        # Print summary
        self.print_summary(specialized, clusters)

        # Print selectivity summary
        print(f"\n  ┌─ POS Selectivity Summary ─────────────────────────────────────────────")
        print(f"  │ Active neurons: {selectivity_data['n_active_neurons']}")
        print(f"  │ Selectivity range (max-min): mean={selectivity_data['selectivity_range']['mean']:.2f}, "
              f"max={selectivity_data['selectivity_range']['max']:.2f}")
        print(f"  │ Top selective POS:")
        for pos in ['NOUN', 'VERB', 'ADJ', 'PUNCT']:
            top = selectivity_data['top_selective_per_pos'].get(pos, [])[:3]
            if top:
                top_str = ', '.join([f"{t['neuron']}({t['selectivity']:.1f}x)" for t in top])
                print(f"  │   {pos}: {top_str}")
        print(f"  └─────────────────────────────────────────────────────────────────────────")

        # Print utilization summary (forward-pass based, replaces EMA Table 2)
        print(f"\n  ┌─ Neuron Utilization (Forward-Pass) ─────────────────────────────────────")
        print(f"  │ {'Pool':<10} {'Active':>8} {'Total':>8} {'Ratio':>10} {'Gini':>8}")
        print(f"  │ {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        for pool_name, data in utilization.items():
            if pool_name.startswith('_'):
                continue
            print(f"  │ {pool_name:<10} {data['active']:>8d} {data['total']:>8d} "
                  f"{data['active_ratio']*100:>9.1f}% {data['gini']:>8.3f}")
        overall = utilization.get('_overall', {})
        if overall:
            print(f"  │ {'─'*10} {'─'*8} {'─'*8} {'─'*10}")
            print(f"  │ {'TOTAL':<10} {overall['total_active']:>8d} {overall['total_neurons']:>8d} "
                  f"{overall['active_ratio']*100:>9.1f}%")
            print(f"  │")
            print(f"  │ Tokens analyzed: {overall['total_tokens_analyzed']:,}")
        print(f"  └─────────────────────────────────────────────────────────────────────────")

        results = {
            'n_neurons_profiled': len(profiles),
            'specialized_neurons': specialized,
            'specialization_summary': specialization_summary,
            'selectivity': {
                'top_selective_per_pos': selectivity_data['top_selective_per_pos'],
                'mean_selectivity_by_pos': selectivity_data['mean_selectivity_by_pos'],
                'selectivity_range': selectivity_data['selectivity_range'],
                'n_active_neurons': selectivity_data['n_active_neurons'],
                'active_neuron_indices': selectivity_data['active_neuron_indices'],
                'pool_order': self.pool_order,  # For neuron name conversion in visualization
                'pos_tags': selectivity_data['pos_tags'],
                'per_pool': selectivity_data.get('per_pool', {}),
                'by_pos': selectivity_data.get('by_pos', {}),
                'pools_analyzed': selectivity_data.get('pools_analyzed', []),
            },
            # Forward-pass based utilization (replaces EMA Table 2)
            'utilization': utilization,
            'clusters': clusters,
            'summary': {
                'total_tokens': len(self.token_data),
                'n_neurons': self.n_neurons,
                'n_specialized': {k: len(v) for k, v in specialized.items()},
                'pools_analyzed': list(pool_ranges.keys()),
            }
        }

        # Store full selectivity matrix for visualization (not in JSON)
        self._selectivity_matrix = selectivity_data['selectivity_matrix']
        self._selectivity_active_indices = selectivity_data['active_neuron_indices']

        # Save if output_dir provided
        if output_dir:
            import json
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, 'neuron_feature_analysis.json')

            # Custom encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            # Convert to JSON-serializable
            results_json = {
                'n_neurons_profiled': results['n_neurons_profiled'],
                'specialized_neurons': results['specialized_neurons'],
                'specialization_summary': results['specialization_summary'],
                'selectivity': results.get('selectivity', {}),
                'clusters': {
                    k: v for k, v in results['clusters'].items()
                    if k != 'cluster_profiles'  # Skip large arrays
                },
                'summary': results['summary'],
            }

            # Save selectivity matrix separately (for heatmap visualization)
            if hasattr(self, '_selectivity_matrix'):
                selectivity_path = os.path.join(output_dir, 'selectivity_matrix.npy')
                np.save(selectivity_path, self._selectivity_matrix)
                print(f"  Selectivity matrix saved to: {selectivity_path}")

            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2, cls=NumpyEncoder)
            print(f"\nResults saved to: {results_path}")

        return results

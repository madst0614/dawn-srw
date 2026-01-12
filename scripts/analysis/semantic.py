"""
Semantic Analysis for DAWN
===========================
Analyze semantic properties of DAWN v17.1 routing.

This module validates the core claims of the DAWN paper:
1. Semantically similar inputs -> similar neuron routing paths
2. Neurons learn linguistic/semantic properties
3. Context-dependent dynamic routing

Key improvements (v2):
- Collects routing from ALL layers (not just layer 0)
- Includes Knowledge routing (critical for v17.1)
- Sentence-level mean pooling for similarity (not position-wise truncate)
- Entropy-based filtering for collapsed routing
- Layer-wise and routing-type-wise reporting

Performance optimizations (v3):
- Batch processing for multiple texts (single forward pass)
- GPU-based similarity computation
- spaCy nlp.pipe() for batch POS tagging
- Minimal CPU transfers
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .base import BaseAnalyzer
from .utils import (
    ROUTING_KEYS,
    KNOWLEDGE_ROUTING_KEYS,
    calc_entropy,
    convert_to_serializable,
    get_batch_input_ids, get_routing_from_outputs,
    RoutingDataExtractor,  # Schema layer for model-agnostic access
    HAS_TQDM, tqdm
)

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


# Minimum entropy ratio to consider routing as "healthy" (not collapsed)
MIN_HEALTHY_ENTROPY = 10.0  # 10% of max entropy


class SemanticAnalyzer(BaseAnalyzer):
    """Semantic analysis for DAWN routing patterns."""

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

        # Load spaCy for POS tagging
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                pass

        # Cache for routing paths (optional)
        self._path_cache = {}
        self._cache_enabled = True

    def clear_cache(self):
        """Clear routing path cache."""
        self._path_cache = {}

    # ============================================================
    # Entropy Helpers
    # ============================================================

    def compute_routing_entropy(self, weights: torch.Tensor) -> float:
        """
        Compute entropy of routing weights as percentage of maximum.

        Args:
            weights: Routing weights tensor [*, N]

        Returns:
            Entropy ratio (0-100)
        """
        if weights.numel() == 0:
            return 0.0

        # Flatten to [batch, neurons] and average
        if weights.dim() == 1:
            probs = weights
        elif weights.dim() == 2:
            probs = weights.mean(dim=0)
        else:
            probs = weights.reshape(-1, weights.shape[-1]).mean(dim=0)

        # Normalize to probabilities
        probs = probs.clamp(min=0)
        total = probs.sum()
        if total <= 0:
            return 0.0
        probs = probs / total

        # Compute entropy
        ent = calc_entropy(probs, dim=-1)
        if ent.dim() > 0:
            ent = ent.mean()

        max_ent = math.log(weights.shape[-1])
        return float(ent / max_ent * 100) if max_ent > 0 else 0.0

    def get_healthy_routing_keys(self, routing_dict: Dict, min_entropy: float = MIN_HEALTHY_ENTROPY) -> List[str]:
        """
        Filter routing keys to only include non-collapsed ones.

        Args:
            routing_dict: Dictionary of routing weights
            min_entropy: Minimum entropy threshold

        Returns:
            List of healthy routing keys
        """
        healthy = []
        for key, weights in routing_dict.items():
            if weights is None:
                continue
            ent = self.compute_routing_entropy(weights)
            if ent >= min_entropy:
                healthy.append(key)
        return healthy

    # ============================================================
    # Routing Path Extraction - BATCH VERSION (OPTIMIZED)
    # ============================================================

    def get_routing_paths_batch(self, texts: List[str], keep_on_gpu: bool = True) -> List[Dict]:
        """
        Extract routing paths for multiple texts in a single forward pass.

        Args:
            texts: List of input texts
            keep_on_gpu: Keep tensors on GPU (faster for subsequent operations)

        Returns:
            List of routing path dicts, one per text
        """
        if not texts:
            return []

        # Check cache
        if self._cache_enabled:
            cached_results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                if text in self._path_cache:
                    cached_results.append((i, self._path_cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            if not uncached_texts:
                # All cached
                return [r for _, r in sorted(cached_results, key=lambda x: x[0])]
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_results = []

        # Tokenize batch
        enc = self.tokenizer(
            uncached_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

        # Single forward pass with analysis context
        self.model.eval()

        with self.extractor.analysis_context():
            with torch.no_grad():
                outputs = self.model(input_ids, return_routing_info=True)

            routing = self.extractor.extract(outputs)
            if not routing:
                return [{} for _ in texts]

        batch_size = input_ids.shape[0]
        results = []

        # Process each sample in batch
        for b in range(batch_size):
            result = {}
            seq_len = int(attention_mask[b].sum().item())

            for layer in routing:
                layer_key = f'layer_{layer.layer_idx}'
                result[layer_key] = {}

                # Attention routing using standardized keys
                for key in ROUTING_KEYS.keys():
                    weights = layer.get_weight(key)
                    if weights is not None:
                        if weights.dim() == 3:
                            w = weights[b, :seq_len]  # [B,S,N] → [S, N]
                        elif weights.dim() == 2:
                            # Batch-level routing: expand to token-level [N] → [S, N]
                            w = weights[b].unsqueeze(0).expand(seq_len, -1)
                        else:
                            continue
                        result[layer_key][key] = w if keep_on_gpu else w.cpu()

                # Knowledge routing using standardized keys
                for key in KNOWLEDGE_ROUTING_KEYS.keys():
                    weights = layer.get_weight(key)
                    if weights is not None:
                        if weights.dim() == 3:
                            w = weights[b, :seq_len]  # [B,S,N] → [S, N]
                        elif weights.dim() == 2:
                            # Batch-level routing: expand to token-level [N] → [S, N]
                            w = weights[b].unsqueeze(0).expand(seq_len, -1)
                        else:
                            continue
                        result[layer_key][key] = w if keep_on_gpu else w.cpu()

            # Aggregated representation (GPU)
            result['aggregated'] = self._aggregate_routing_path_gpu(result)
            result['entropy'] = self._compute_path_entropy(result)

            results.append(result)

            # Cache
            if self._cache_enabled and b < len(uncached_texts):
                self._path_cache[uncached_texts[b]] = result

        # Merge cached and new results
        if cached_results:
            final_results = [None] * len(texts)
            for orig_idx, cached_result in cached_results:
                final_results[orig_idx] = cached_result
            for i, result in enumerate(results):
                final_results[uncached_indices[i]] = result
            return final_results

        return results

    def get_routing_path(self, text: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract routing path for a single text.
        Uses batch version internally.
        """
        results = self.get_routing_paths_batch([text], keep_on_gpu=False)
        return results[0] if results else {}

    def _aggregate_routing_path_gpu(self, layer_routing: Dict) -> Dict[str, torch.Tensor]:
        """
        Aggregate routing across all layers into sentence-level representation.
        Keeps computation on GPU.
        """
        aggregated = defaultdict(list)

        for layer_key, routing_dict in layer_routing.items():
            if layer_key in ['aggregated', 'entropy']:
                continue
            for routing_key, weights in routing_dict.items():
                if weights.dim() >= 1:
                    pooled = weights.mean(dim=0) if weights.dim() > 1 else weights
                    aggregated[routing_key].append(pooled)

        result = {}
        for key, layer_weights in aggregated.items():
            if layer_weights:
                stacked = torch.stack(layer_weights)
                result[key] = stacked.mean(dim=0)

        return result

    def _compute_path_entropy(self, layer_routing: Dict) -> Dict[str, float]:
        """Compute entropy for each routing key across all layers."""
        entropy = {}

        for layer_key, routing_dict in layer_routing.items():
            if layer_key in ['aggregated', 'entropy']:
                continue
            for routing_key, weights in routing_dict.items():
                full_key = f'{layer_key}/{routing_key}'
                entropy[full_key] = self.compute_routing_entropy(weights)

        return entropy

    # ============================================================
    # Path Similarity - GPU OPTIMIZED
    # ============================================================

    def compute_path_similarity_gpu(self, path1: Dict, path2: Dict,
                                     use_entropy_weighting: bool = True) -> Dict:
        """
        Compute similarity between two routing paths on GPU.

        Args:
            path1: First routing path
            path2: Second routing path
            use_entropy_weighting: Weight by entropy

        Returns:
            Dictionary with similarity metrics
        """
        results = {
            'per_layer': {},
            'per_routing_type': {},
            'overall': {},
        }

        agg1 = path1.get('aggregated', {})
        agg2 = path2.get('aggregated', {})
        ent1 = path1.get('entropy', {})
        ent2 = path2.get('entropy', {})

        all_cosines = []
        all_jaccards = []
        all_weights = []

        for key in agg1.keys():
            if key not in agg2:
                continue

            p1, p2 = agg1[key], agg2[key]

            # Ensure on same device
            if p1.device != p2.device:
                p2 = p2.to(p1.device)

            # Cosine similarity (GPU)
            p1_norm = F.normalize(p1.unsqueeze(0), dim=-1)
            p2_norm = F.normalize(p2.unsqueeze(0), dim=-1)
            cosine = float((p1_norm * p2_norm).sum())

            # Jaccard similarity (top-8 neurons)
            k = min(8, p1.shape[-1])
            top1 = set(p1.topk(k)[1].cpu().tolist())
            top2 = set(p2.topk(k)[1].cpu().tolist())
            jaccard = len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0

            results['per_routing_type'][key] = {
                'cosine': cosine,
                'jaccard': jaccard,
            }

            # Entropy-based weighting
            if use_entropy_weighting:
                key_entropies = [v for k, v in {**ent1, **ent2}.items() if key in k]
                avg_ent = np.mean(key_entropies) if key_entropies else 50.0
                weight = min(avg_ent / 100.0, 1.0)
            else:
                weight = 1.0

            all_cosines.append(cosine)
            all_jaccards.append(jaccard)
            all_weights.append(weight)

        # Weighted average
        if all_cosines:
            weights_sum = sum(all_weights)
            if weights_sum > 0:
                results['overall']['cosine_weighted'] = sum(c * w for c, w in zip(all_cosines, all_weights)) / weights_sum
                results['overall']['jaccard_weighted'] = sum(j * w for j, w in zip(all_jaccards, all_weights)) / weights_sum
            results['overall']['cosine_mean'] = np.mean(all_cosines)
            results['overall']['jaccard_mean'] = np.mean(all_jaccards)

        # Per-layer similarity
        for layer_key in path1.keys():
            if layer_key in ['aggregated', 'entropy'] or layer_key not in path2:
                continue

            layer_cosines = []
            for rkey in path1[layer_key].keys():
                if rkey not in path2[layer_key]:
                    continue

                p1 = path1[layer_key][rkey].mean(dim=0)
                p2 = path2[layer_key][rkey].mean(dim=0)

                if p1.device != p2.device:
                    p2 = p2.to(p1.device)

                p1_norm = F.normalize(p1.unsqueeze(0), dim=-1)
                p2_norm = F.normalize(p2.unsqueeze(0), dim=-1)
                cosine = float((p1_norm * p2_norm).sum())
                layer_cosines.append(cosine)

            if layer_cosines:
                results['per_layer'][layer_key] = {
                    'cosine_mean': np.mean(layer_cosines),
                }

        return results

    # ============================================================
    # Semantic Path Similarity Analysis - BATCH OPTIMIZED
    # ============================================================

    def analyze_semantic_path_similarity(self, sentence_pairs: List[Tuple[str, str, str]]) -> Dict:
        """
        Compare routing similarity for semantically similar vs different sentence pairs.
        Uses batch processing for efficiency.

        Args:
            sentence_pairs: List of (sent1, sent2, label)

        Returns:
            Statistics for similar vs different pairs
        """
        # Collect all unique sentences
        all_sentences = []
        sent_to_idx = {}
        for sent1, sent2, _ in sentence_pairs:
            for s in [sent1, sent2]:
                if s not in sent_to_idx:
                    sent_to_idx[s] = len(all_sentences)
                    all_sentences.append(s)

        # Batch process all sentences at once
        print(f"  Processing {len(all_sentences)} unique sentences in batch...")
        all_paths = self.get_routing_paths_batch(all_sentences, keep_on_gpu=True)

        similar_sims = []
        different_sims = []
        layer_stats = defaultdict(lambda: {'similar': [], 'different': []})
        routing_type_stats = defaultdict(lambda: {'similar': [], 'different': []})

        for sent1, sent2, label in tqdm(sentence_pairs, desc='Computing similarities'):
            path1 = all_paths[sent_to_idx[sent1]]
            path2 = all_paths[sent_to_idx[sent2]]

            if not path1 or not path2:
                continue

            sim = self.compute_path_similarity_gpu(path1, path2)

            overall = sim.get('overall', {})
            cos_w = overall.get('cosine_weighted', overall.get('cosine_mean', 0))
            jac_w = overall.get('jaccard_weighted', overall.get('jaccard_mean', 0))

            entry = {'cosine': cos_w, 'jaccard': jac_w}

            if label == 'similar':
                similar_sims.append(entry)
            else:
                different_sims.append(entry)

            for layer_key, layer_data in sim.get('per_layer', {}).items():
                cos = layer_data.get('cosine_mean', 0)
                if label == 'similar':
                    layer_stats[layer_key]['similar'].append(cos)
                else:
                    layer_stats[layer_key]['different'].append(cos)

            for rtype, rdata in sim.get('per_routing_type', {}).items():
                cos = rdata.get('cosine', 0)
                if label == 'similar':
                    routing_type_stats[rtype]['similar'].append(cos)
                else:
                    routing_type_stats[rtype]['different'].append(cos)

        results = {
            'similar_pairs': {
                'count': len(similar_sims),
                'cosine_mean': np.mean([s['cosine'] for s in similar_sims]) if similar_sims else 0,
                'cosine_std': np.std([s['cosine'] for s in similar_sims]) if similar_sims else 0,
                'jaccard_mean': np.mean([s['jaccard'] for s in similar_sims]) if similar_sims else 0,
            },
            'different_pairs': {
                'count': len(different_sims),
                'cosine_mean': np.mean([s['cosine'] for s in different_sims]) if different_sims else 0,
                'cosine_std': np.std([s['cosine'] for s in different_sims]) if different_sims else 0,
                'jaccard_mean': np.mean([s['jaccard'] for s in different_sims]) if different_sims else 0,
            },
        }

        # Layer-wise analysis
        results['per_layer'] = {}
        for layer_key, stats in layer_stats.items():
            sim_cos = np.mean(stats['similar']) if stats['similar'] else 0
            diff_cos = np.mean(stats['different']) if stats['different'] else 0
            results['per_layer'][layer_key] = {
                'similar_cosine': sim_cos,
                'different_cosine': diff_cos,
                'gap': sim_cos - diff_cos,
            }

        # Routing-type analysis
        results['per_routing_type'] = {}
        attention_gaps = []
        knowledge_gaps = []

        for rtype, stats in routing_type_stats.items():
            sim_cos = np.mean(stats['similar']) if stats['similar'] else 0
            diff_cos = np.mean(stats['different']) if stats['different'] else 0
            gap = sim_cos - diff_cos

            results['per_routing_type'][rtype] = {
                'similar_cosine': sim_cos,
                'different_cosine': diff_cos,
                'gap': gap,
            }

            if rtype in ['fknow', 'rknow']:
                knowledge_gaps.append(gap)
            else:
                attention_gaps.append(gap)

        results['routing_type_summary'] = {
            'attention_avg_gap': np.mean(attention_gaps) if attention_gaps else 0,
            'knowledge_avg_gap': np.mean(knowledge_gaps) if knowledge_gaps else 0,
        }

        # Interpretation
        if similar_sims and different_sims:
            sim_cos = results['similar_pairs']['cosine_mean']
            diff_cos = results['different_pairs']['cosine_mean']
            gap = sim_cos - diff_cos

            if gap > 0.1:
                verdict = 'GOOD: Semantic similarity reflected in routing'
            elif gap > 0.05:
                verdict = 'MODERATE: Some semantic correlation in routing'
            elif gap > 0:
                verdict = 'WEAK: Routing has slight semantic correlation'
            else:
                verdict = 'BAD: Routing inversely correlated with semantics'

            results['interpretation'] = {
                'similarity_gap': gap,
                'verdict': verdict,
                'best_layer': max(results['per_layer'].items(),
                                   key=lambda x: x[1]['gap'])[0] if results['per_layer'] else None,
                'best_routing_type': max(results['per_routing_type'].items(),
                                          key=lambda x: x[1]['gap'])[0] if results['per_routing_type'] else None,
            }

        return results

    def get_default_sentence_pairs(self) -> List[Tuple[str, str, str]]:
        """Get default test sentence pairs."""
        return [
            # Similar pairs (paraphrases)
            ("The cat sat on the mat.", "A feline rested on the rug.", "similar"),
            ("She bought a new car.", "She purchased a new vehicle.", "similar"),
            ("The weather is beautiful today.", "It's a lovely day outside.", "similar"),
            ("He runs every morning.", "He jogs each day at dawn.", "similar"),
            ("The book was interesting.", "The novel was captivating.", "similar"),
            ("I need to go to the store.", "I have to visit the shop.", "similar"),
            ("The children played in the park.", "Kids were playing at the playground.", "similar"),
            ("She cooked dinner for the family.", "She prepared a meal for her relatives.", "similar"),

            # Different pairs (unrelated)
            ("The cat sat on the mat.", "Stock prices rose sharply.", "different"),
            ("She bought a new car.", "The experiment failed completely.", "different"),
            ("The weather is beautiful today.", "Binary search has O(log n) complexity.", "different"),
            ("He runs every morning.", "The painting was sold at auction.", "different"),
            ("The book was interesting.", "Photosynthesis requires sunlight.", "different"),
            ("I need to go to the store.", "The treaty was signed in 1945.", "different"),
            ("The children played in the park.", "The server crashed unexpectedly.", "different"),
            ("She cooked dinner for the family.", "Quantum entanglement is mysterious.", "different"),
        ]

    # ============================================================
    # POS Routing Analysis - Using UD Dataset (pre-tagged, fast)
    # ============================================================

    def _load_ud_dataset(self, max_sentences: int = 1000) -> list:
        """Load UD English EWT dataset (cached)."""
        if hasattr(self, '_ud_cache') and self._ud_cache:
            return self._ud_cache

        try:
            import conllu
        except ImportError:
            print("conllu not installed. Install with: pip install conllu")
            return []

        import urllib.request
        url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu'

        try:
            print("Downloading UD English EWT...")
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
        except Exception as e:
            print(f"UD download failed: {e}")
            return []

        sentences = conllu.parse(data)[:max_sentences]
        dataset = []
        for sent in sentences:
            tokens = [token['form'] for token in sent]
            upos = [token['upos'] for token in sent]
            dataset.append({'tokens': tokens, 'upos': upos})

        self._ud_cache = dataset
        print(f"Loaded {len(dataset)} UD sentences")
        return dataset

    def _align_tokens_to_pos(self, ud_tokens: list, ud_pos: list) -> tuple:
        """Align tokenizer tokens to UD POS tags using offset_mapping."""
        text = " ".join(ud_tokens)
        ud_char_spans = []
        pos = 0
        for token, upos in zip(ud_tokens, ud_pos):
            start = pos
            end = pos + len(token)
            ud_char_spans.append((start, end, upos))
            pos = end + 1  # +1 for space

        try:
            encoding = self.tokenizer(
                text, add_special_tokens=False,
                return_offsets_mapping=True, return_tensors=None
            )
            token_ids = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']

            pos_tags = []
            for start, end in offset_mapping:
                assigned = 'X'
                for ud_start, ud_end, upos in ud_char_spans:
                    if start < ud_end and end > ud_start:
                        assigned = upos
                        break
                pos_tags.append(assigned)

            return pos_tags, token_ids
        except:
            return [], []

    def analyze_pos_routing(self, dataloader, max_batches: int = 50, target_layers: list = None) -> Dict:
        """
        Analyze routing patterns by POS using UD dataset (pre-tagged, fast).

        Args:
            dataloader: Unused (uses UD dataset instead)
            max_batches: Max sentences to process
            target_layers: Layer indices to analyze (default: [0, n_layers//2, n_layers-1])

        Returns:
            POS-wise routing statistics with layer breakdown
        """
        # Load UD dataset
        ud_data = self._load_ud_dataset(max_sentences=max_batches * 32)
        if not ud_data:
            return {'error': 'Could not load UD dataset'}

        pos_weights = defaultdict(list)
        pos_counts = defaultdict(int)

        n_layers = getattr(self.model, 'n_layers', 12)
        if target_layers is None:
            target_layers = [0, n_layers // 2, n_layers - 1]
        target_layers_set = set(target_layers)

        self.model.eval()

        with self.extractor.analysis_context():
            for sent_data in tqdm(ud_data, desc='POS Analysis (UD)'):
                pos_tags, token_ids = self._align_tokens_to_pos(
                    sent_data['tokens'], sent_data['upos']
                )
                if not token_ids:
                    continue

                input_ids = torch.tensor([token_ids], device=self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                # Process selected layers using extractor
                for layer in routing:
                    layer_idx = layer.layer_idx
                    if layer_idx not in target_layers_set:
                        continue

                    # Collect weights using standardized keys
                    all_routing = {}
                    for key in ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            all_routing[f'L{layer_idx}/{key}'] = w

                    for key in KNOWLEDGE_ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            all_routing[f'L{layer_idx}/{key}'] = w

                    # Map tokens to POS
                    for s, upos in enumerate(pos_tags):
                        if upos == 'X':
                            continue
                        pos_counts[upos] += 1

                        for routing_key, w in all_routing.items():
                            if w is not None and w.dim() >= 2:
                                if w.dim() == 3 and s < w.shape[1]:
                                    pos_weights[f"{routing_key}/{upos}"].append(w[0, s].cpu())
                                elif w.dim() == 2:
                                    pos_weights[f"{routing_key}/{upos}"].append(w[0].cpu())

        # Analyze results
        results = {
            'pos_counts': dict(pos_counts),
            'routing_by_pos': {},
            'layer_summary': {},
        }

        layer_pos_patterns = defaultdict(lambda: defaultdict(list))

        for key_pos, weights in pos_weights.items():
            if len(weights) < 10:
                continue

            stacked = torch.stack(weights)
            mean_w = stacked.mean(dim=0)

            top_k = min(5, mean_w.shape[0])
            top_neurons = mean_w.topk(top_k)

            entropy = self.compute_routing_entropy(stacked)

            results['routing_by_pos'][key_pos] = {
                'count': len(weights),
                'mean_activation': float(mean_w.mean()),
                'entropy': entropy,
                'top_neurons': [
                    (int(idx), float(val))
                    for idx, val in zip(top_neurons.indices.tolist(), top_neurons.values.tolist())
                ],
            }

            parts = key_pos.split('/')
            if len(parts) >= 2:
                layer = parts[0]
                pos = parts[-1]
                layer_pos_patterns[layer][pos].append(entropy)

        for layer, pos_ents in layer_pos_patterns.items():
            results['layer_summary'][layer] = {
                pos: np.mean(ents) for pos, ents in pos_ents.items()
            }

        return results

    # ============================================================
    # Context-Dependent Routing - BATCH OPTIMIZED
    # ============================================================

    def analyze_context_dependent_routing(self, word_contexts: Dict[str, List[str]]) -> Dict:
        """
        Analyze if the same word has different routing in different contexts.
        Uses batch processing.

        Args:
            word_contexts: Dictionary mapping word to list of sentences

        Returns:
            Per-word routing variance statistics
        """
        # Collect all sentences
        all_sentences = []
        sent_to_word = {}  # sentence -> (word, position in word's list)

        for word, sentences in word_contexts.items():
            for i, sent in enumerate(sentences):
                if sent not in sent_to_word:
                    sent_to_word[sent] = []
                    all_sentences.append(sent)
                sent_to_word[sent].append((word, i))

        # Batch process
        if all_sentences:
            all_paths = self.get_routing_paths_batch(all_sentences, keep_on_gpu=False)
            sent_to_path = dict(zip(all_sentences, all_paths))
        else:
            sent_to_path = {}

        results = {}

        for word, sentences in word_contexts.items():
            if len(sentences) < 2:
                continue

            word_routings = []

            for sent in sentences:
                tokens = self.tokenizer.tokenize(sent.lower())
                word_lower = word.lower()

                word_positions = []
                for i, tok in enumerate(tokens):
                    if word_lower in tok or tok in word_lower:
                        word_positions.append(i + 1)

                if not word_positions:
                    continue

                path = sent_to_path.get(sent, {})
                if not path:
                    continue

                word_routing = {}
                for layer_key, layer_routing in path.items():
                    if layer_key in ['aggregated', 'entropy']:
                        continue

                    for rkey, weights in layer_routing.items():
                        if weights.dim() >= 1:
                            pos = word_positions[0]
                            if pos < weights.shape[0]:
                                full_key = f'{layer_key}/{rkey}'
                                if full_key not in word_routing:
                                    word_routing[full_key] = []
                                word_routing[full_key].append(weights[pos])

                collapsed = {}
                for full_key, weight_list in word_routing.items():
                    if weight_list:
                        stacked = torch.stack(weight_list)
                        collapsed[full_key] = stacked.mean(dim=0)

                if collapsed:
                    word_routings.append(collapsed)

            if len(word_routings) < 2:
                continue

            variances = {}
            for key in word_routings[0].keys():
                key_routings = [wr[key] for wr in word_routings if key in wr]
                if len(key_routings) >= 2:
                    stacked = torch.stack(key_routings)
                    variance = stacked.var(dim=0).mean().item()
                    variances[key] = variance

            attn_vars = [v for k, v in variances.items() if 'fknow' not in k and 'rknow' not in k]
            know_vars = [v for k, v in variances.items() if 'fknow' in k or 'rknow' in k]

            results[word] = {
                'n_contexts': len(word_routings),
                'routing_variance': variances,
                'avg_variance': np.mean(list(variances.values())) if variances else 0,
                'attention_variance': np.mean(attn_vars) if attn_vars else 0,
                'knowledge_variance': np.mean(know_vars) if know_vars else 0,
            }

        if results:
            avg_var = np.mean([r['avg_variance'] for r in results.values()])
            attn_var = np.mean([r['attention_variance'] for r in results.values()])
            know_var = np.mean([r['knowledge_variance'] for r in results.values()])

            if avg_var > 0.1:
                interpretation = 'HIGH: Strong context-dependent routing'
            elif avg_var > 0.01:
                interpretation = 'MODERATE: Some context sensitivity'
            else:
                interpretation = 'LOW: Routing mostly context-independent'

            results['summary'] = {
                'overall_context_variance': avg_var,
                'attention_context_variance': attn_var,
                'knowledge_context_variance': know_var,
                'interpretation': interpretation,
                'more_context_sensitive': 'knowledge' if know_var > attn_var else 'attention',
            }

        return results

    def get_default_word_contexts(self) -> Dict[str, List[str]]:
        """Get default polysemous word contexts."""
        return {
            "bank": [
                "I deposited money at the bank.",
                "The river bank was covered with flowers.",
                "You can bank on his promise.",
            ],
            "bat": [
                "He swung the baseball bat.",
                "A bat flew out of the cave.",
            ],
            "light": [
                "Turn on the light please.",
                "The bag is very light.",
                "Light colors are better for summer.",
            ],
            "run": [
                "I run every morning.",
                "The program will run automatically.",
                "There's a run in her stocking.",
            ],
            "play": [
                "Children love to play outside.",
                "She will play the piano.",
                "We watched a play at the theater.",
            ],
        }

    # ============================================================
    # Neuron-Token Heatmap
    # ============================================================

    def analyze_neuron_token_heatmap(self, dataloader, max_batches: int = 30,
                                      top_k_neurons: int = 20) -> Dict:
        """
        Generate neuron-token activation heatmap data.

        Optimized with GPU scatter_add for vectorized aggregation.

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process
            top_k_neurons: Number of top neurons to include

        Returns:
            Per-neuron top token activations
        """
        vocab_size = self.tokenizer.vocab_size

        # {routing_key: tensor[vocab_size, N]}
        token_neuron_sums = {}
        neuron_sizes = {}  # Track N for each routing_key

        self.model.eval()

        with self.extractor.analysis_context():
            for i, batch in enumerate(tqdm(dataloader, desc='Neuron-Token Heatmap', total=max_batches)):
                if i >= max_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                B, S = input_ids.shape

                with torch.no_grad():
                    outputs = self.model(input_ids, return_routing_info=True)

                routing = self.extractor.extract(outputs)
                if not routing:
                    continue

                for layer in routing:
                    layer_idx = layer.layer_idx
                    routing_weights = {}

                    # Collect attention weights using standardized keys
                    for key in ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            if w.dim() == 3:
                                routing_weights[f'L{layer_idx}/{key}'] = w
                            elif w.dim() == 2:
                                routing_weights[f'L{layer_idx}/{key}'] = w.unsqueeze(1).expand(-1, S, -1)

                    # Collect knowledge weights using standardized keys
                    for key in KNOWLEDGE_ROUTING_KEYS.keys():
                        w = layer.get_weight(key)
                        if w is not None:
                            if w.dim() == 3:
                                routing_weights[f'L{layer_idx}/{key}'] = w
                            elif w.dim() == 2:
                                routing_weights[f'L{layer_idx}/{key}'] = w.unsqueeze(1).expand(-1, S, -1)

                    # Vectorized aggregation using scatter_add
                    for routing_key, weights in routing_weights.items():
                        B, S, N = weights.shape

                        # Initialize accumulator if needed
                        if routing_key not in token_neuron_sums:
                            token_neuron_sums[routing_key] = torch.zeros(
                                vocab_size, N, device=self.device, dtype=torch.float32
                            )
                            neuron_sizes[routing_key] = N

                        # Flatten: [B, S] -> [B*S] (use reshape for non-contiguous tensors)
                        flat_token_ids = input_ids.reshape(-1)  # [B*S]
                        flat_weights = weights.reshape(-1, N)    # [B*S, N]

                        # Mask special tokens ([CLS], [SEP], [PAD], etc.)
                        # Token IDs 100-103 are typically special in BERT
                        valid_mask = (flat_token_ids >= 104)  # Skip [UNK], [CLS], [SEP], [MASK], [PAD]

                        if valid_mask.any():
                            valid_ids = flat_token_ids[valid_mask]      # [valid_count]
                            valid_weights = flat_weights[valid_mask]    # [valid_count, N]

                            # scatter_add: accumulate weights by token_id
                            # token_neuron_sums[routing_key][token_id, :] += valid_weights
                            token_neuron_sums[routing_key].scatter_add_(
                                0,
                                valid_ids.unsqueeze(1).expand(-1, N),
                                valid_weights
                            )

        # Convert to results format
        results = {}
        for routing_key, sums in token_neuron_sums.items():
            # Move to CPU for final processing
            sums_cpu = sums.cpu()
            N = neuron_sizes[routing_key]

            # Find top neurons by total activation
            neuron_totals = sums_cpu.sum(dim=0)  # [N]
            top_neuron_indices = neuron_totals.topk(min(top_k_neurons, N)).indices.tolist()

            neuron_results = {}
            for neuron_id in top_neuron_indices:
                # Get top tokens for this neuron
                neuron_activations = sums_cpu[:, neuron_id]  # [vocab_size]

                # Find non-zero tokens
                nonzero_mask = neuron_activations > 0
                if nonzero_mask.sum() == 0:
                    continue

                nonzero_indices = nonzero_mask.nonzero().squeeze(-1)
                nonzero_values = neuron_activations[nonzero_mask]

                # Get top 10 tokens
                top_k = min(10, len(nonzero_values))
                top_values, top_local_idx = nonzero_values.topk(top_k)
                top_token_ids = nonzero_indices[top_local_idx]

                # Convert to token strings
                top_tokens = {}
                for tid, val in zip(top_token_ids.tolist(), top_values.tolist()):
                    token = self.tokenizer.convert_ids_to_tokens([tid])[0]
                    top_tokens[token] = val

                neuron_results[neuron_id] = {
                    'total_activation': float(neuron_totals[neuron_id]),
                    'top_tokens': top_tokens,
                }

            results[routing_key] = neuron_results

        return results

    # ============================================================
    # Run All
    # ============================================================

    def run_all(self, dataloader=None, output_dir: str = './semantic_analysis', max_batches: int = 50) -> Dict:
        """
        Run all semantic analyses.

        Args:
            dataloader: DataLoader for input data (optional)
            output_dir: Directory for outputs
            max_batches: Maximum batches for POS and heatmap analysis

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 1. Semantic Path Similarity (batch optimized)
        print("\n[1/4] Analyzing Semantic Path Similarity (batch)...")
        pairs = self.get_default_sentence_pairs()
        results['path_similarity'] = self.analyze_semantic_path_similarity(pairs)

        # 2. Context-dependent Routing (batch optimized)
        print("\n[2/4] Analyzing Context-dependent Routing (batch)...")
        word_contexts = self.get_default_word_contexts()
        results['context_routing'] = self.analyze_context_dependent_routing(word_contexts)

        # 3. POS Routing (requires dataloader, uses nlp.pipe)
        if dataloader is not None:
            print(f"\n[3/4] Analyzing POS Routing Patterns (max_batches={max_batches})...")
            results['pos_routing'] = self.analyze_pos_routing(dataloader, max_batches=max_batches)

            print(f"\n[4/4] Generating Neuron-Token Heatmap (max_batches={max_batches})...")
            results['neuron_heatmap'] = self.analyze_neuron_token_heatmap(dataloader, max_batches=max_batches)
        else:
            print("\n[3/4] Skipping POS analysis (no dataloader)")
            print("\n[4/4] Skipping heatmap (no dataloader)")

        # Clear cache to free memory
        self.clear_cache()

        # Save results
        import json
        output_path = os.path.join(output_dir, 'semantic_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results

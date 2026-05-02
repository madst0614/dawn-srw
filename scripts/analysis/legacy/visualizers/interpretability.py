#!/usr/bin/env python3
"""
DAWN Interpretability Visualizations (JAX/TPU)
================================================
Three interpretability visualizations for DAWN v17.1-JAX:

1. Context-dependent routing: Heatmap comparing neuron selection for
   the same token in different contexts.
2. Neuron embedding space: UMAP 2D scatter of 6 neuron pools.
3. Layer-wise routing entropy: Entropy of routing weights at each layer depth.

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/interpretability.py \
        --checkpoint gs://bucket/checkpoint.flax \
        --output ./interp_results \
        --viz all --val_data gs://bucket/val.bin

    # Multi-token, multi-pool exploration (broad sweep)
    python scripts/analysis/visualizers/interpretability.py \
        --checkpoint gs://bucket/checkpoint.flax \
        --viz context_routing \
        --target_token the,is,in \
        --pool_key fv,fqk_q,rv

    # Custom contexts from JSON file (for paper figure curation)
    python scripts/analysis/visualizers/interpretability.py \
        --checkpoint gs://bucket/checkpoint.flax \
        --viz context_routing \
        --contexts_file my_contexts.json

    # Embedding space with UMAP
    python scripts/analysis/visualizers/interpretability.py \
        --checkpoint gs://bucket/checkpoint.flax \
        --viz embedding_space

    # Layer-wise routing entropy
    python scripts/analysis/visualizers/interpretability.py \
        --checkpoint gs://bucket/checkpoint.flax \
        --viz layer_entropy \
        --val_data gs://bucket/val.bin
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from scripts.analysis.utils_jax import (
    load_model_jax, get_neuron_embeddings_jax,
    JAXRoutingDataExtractor, create_batches, load_val_data_jax,
    calc_entropy, EMBEDDING_POOLS_V18,
)


# ============================================================
# Pool color scheme (consistent across all plots)
# ============================================================

POOL_COLORS = {
    'feature_qk':   '#E63946',  # red
    'feature_v':    '#F4A261',  # orange
    'restore_qk':   '#457B9D',  # steel blue
    'restore_v':    '#2A9D8F',  # teal
    'feature_know': '#9B5DE5',  # purple
    'restore_know': '#00BBF9',  # cyan
}

POOL_LABELS = {
    'feature_qk':   'Feature QK',
    'feature_v':    'Feature V',
    'restore_qk':   'Restore QK',
    'restore_v':    'Restore V',
    'feature_know': 'Feature Know',
    'restore_know': 'Restore Know',
}


# ============================================================
# Context Presets (broad coverage for initial exploration)
# ============================================================

# Diverse contexts designed to probe different routing behaviors.
# Start broad, then curate based on results for paper figures.
CONTEXT_PRESETS = {
    'the': [
        # Domain-specific contexts
        ("Science",       "The theory of relativity changed the understanding of physics"),
        ("Literature",    "The author wrote the most famous novel in exile"),
        ("Law",           "The court ruled that the contract was invalid"),
        ("Cooking",       "The chef prepared the signature dish with fresh herbs"),
        ("Music",         "The orchestra performed the symphony in the grand hall"),
        ("Sports",        "The team won the championship after a tough season"),
        ("Tech",          "The engineer designed the new processor architecture"),
        ("Medicine",      "The doctor examined the patient with great care"),
        # Syntactic role variation
        ("Subject-det",   "The cat sat on the mat"),
        ("Object-det",    "She opened the door slowly"),
        ("Possessive",    "He lost the key to the apartment"),
        # Semantic distance
        ("Abstract",      "The concept of freedom has evolved over the centuries"),
        ("Concrete",      "The hammer hit the nail on the head"),
        ("Temporal",      "The year two thousand marked the beginning of a new era"),
        ("Negation",      "The plan was not the best option available"),
        ("Question-like", "What was the reason for the sudden change"),
    ],
    'is': [
        ("Copula",        "The sky is blue today"),
        ("Existential",   "There is a problem with the system"),
        ("Identity",      "Paris is the capital of France"),
        ("Property",      "Water is essential for life"),
        ("Modal",         "The question is whether we should proceed"),
        ("Passive-aux",   "The cake is baked in the oven"),
        ("Progressive",   "The child is running in the park"),
        ("Negated",       "This is not what I expected"),
        ("Emphasis",      "What is important is the result"),
        ("Technical",     "The function is defined as follows"),
    ],
    'in': [
        ("Location",      "The book is in the library"),
        ("Temporal",      "In the morning the streets are quiet"),
        ("Abstract",      "She excels in mathematics"),
        ("Membership",    "He is in the committee"),
        ("Process",       "The project is in development"),
        ("Material",      "Written in pencil on the paper"),
        ("Idiomatic",     "She was in trouble after the incident"),
        ("Contrast",      "In theory it works but in practice it fails"),
    ],
}

# Default: broad coverage for any target token
DEFAULT_CONTEXTS_TEMPLATE = [
    ("Science",       "The theory of relativity changed {token} understanding of physics"),
    ("Literature",    "The author wrote {token} most famous novel in exile"),
    ("Law",           "The court ruled that {token} contract was invalid"),
    ("Cooking",       "The chef prepared {token} signature dish with fresh herbs"),
    ("Music",         "The orchestra performed {token} symphony in the grand hall"),
    ("Sports",        "The team won {token} championship after a tough season"),
    ("Tech",          "The engineer designed {token} new processor architecture"),
    ("Medicine",      "The doctor examined {token} patient with great care"),
    ("Abstract",      "The concept of {token} freedom has evolved over the centuries"),
    ("Concrete",      "The hammer hit {token} nail on the head"),
    ("Narrative",     "Once upon a time {token} kingdom was in danger"),
    ("Formal",        "According to {token} report the results are significant"),
    ("Casual",        "I think {token} idea is pretty good actually"),
    ("Question",      "What makes {token} approach different from the rest"),
    ("Negation",      "This is not {token} best option available"),
    ("Code-like",     "The function returns {token} value of the expression"),
]


def _get_default_contexts(target_token: str) -> List[Tuple[str, str]]:
    """Get context pairs for a target token.

    Uses curated presets if available, otherwise fills a generic template.
    """
    if target_token in CONTEXT_PRESETS:
        return CONTEXT_PRESETS[target_token]

    return [(label, sent.format(token=target_token))
            for label, sent in DEFAULT_CONTEXTS_TEMPLATE]


def load_contexts_from_file(path: str) -> List[Tuple[str, str]]:
    """Load context pairs from a JSON file.

    Expected format:
        [
            {"label": "Science", "sentence": "The theory of ..."},
            ...
        ]
    or:
        [["Science", "The theory of ..."], ...]
    """
    with open(path, 'r') as f:
        data = json.load(f)

    pairs = []
    for item in data:
        if isinstance(item, dict):
            pairs.append((item['label'], item['sentence']))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((item[0], item[1]))
    return pairs


# ============================================================
# 1. Context-Dependent Routing Heatmap
# ============================================================

def visualize_context_routing(
    model_cls, params, config: Dict,
    output_dir: str,
    tokenizer=None,
    target_token: str = "the",
    context_pairs: List[Tuple[str, str]] = None,
    pool_key: str = 'fv',
    top_n_neurons: int = 30,
) -> str:
    """
    Compare routing weights for the same token in different contexts.

    Generates a heatmap: rows = contexts, columns = top-N neurons,
    cell intensity = routing weight assigned to that neuron.

    Args:
        model_cls: DAWN JAX model class
        params: FrozenDict of model parameters
        config: Model config
        output_dir: Output directory
        tokenizer: HuggingFace tokenizer (loaded automatically if None)
        target_token: Token to analyze across contexts
        context_pairs: List of (label, sentence) pairs
        pool_key: Routing pool to visualize ('fv', 'fqk_q', 'rv', etc.)
        top_n_neurons: Number of top neurons to show

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("  matplotlib required for visualization")
        return ""

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if context_pairs is None:
        context_pairs = _get_default_contexts(target_token)

    extractor = JAXRoutingDataExtractor(
        _create_model_instance(config), params, config
    )

    # Find target token ID
    target_id = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_id) != 1:
        print(f"  Warning: '{target_token}' tokenizes to {len(target_id)} tokens, using first")
    target_id = target_id[0]

    # Collect routing weights for target token in each context
    labels = []
    weight_rows = []

    for label, sentence in context_pairs:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True,
                                     max_length=config.get('max_seq_len', 512),
                                     truncation=True)
        input_ids_np = np.array([input_ids])

        # Find position of target token
        positions = [i for i, tid in enumerate(input_ids) if tid == target_id]
        if not positions:
            print(f"  Skipping '{label}': target token '{target_token}' not found")
            continue

        pos = positions[0]

        routing_data = extractor.extract_routing(input_ids_np)

        # Get weights for specified pool
        weights = _get_routing_weight(routing_data, pool_key)
        if weights is None:
            print(f"  Skipping '{label}': pool '{pool_key}' not found in routing data")
            continue

        # weights shape: [B, S, N] or [B, N]
        if weights.ndim == 3:
            w = weights[0, pos, :]  # [N]
        else:
            w = weights[0, :]

        labels.append(label)
        weight_rows.append(w)

    if len(weight_rows) < 2:
        print("  Need at least 2 valid contexts for comparison")
        return ""

    weight_matrix = np.stack(weight_rows)  # [n_contexts, N_neurons]

    # Select top-N neurons by max weight across all contexts
    max_per_neuron = weight_matrix.max(axis=0)
    top_indices = np.argsort(max_per_neuron)[-top_n_neurons:][::-1]
    top_weights = weight_matrix[:, top_indices]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(12, top_n_neurons * 0.4), max(4, len(labels) * 0.6)))

    im = ax.imshow(top_weights, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Routing Weight', fontsize=10)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(len(top_indices)))
    ax.set_xticklabels([f'N{i}' for i in top_indices], fontsize=7, rotation=45, ha='right')
    ax.set_xlabel('Neuron Index', fontsize=11)
    ax.set_title(
        f'Context-Dependent Routing: token="{target_token}", pool={pool_key}\n'
        f'Same token, different contexts → different neuron selections',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'context_routing_{target_token}_{pool_key}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Save raw data
    data_path = os.path.join(output_dir, f'context_routing_{target_token}_{pool_key}.json')
    data = {
        'target_token': target_token,
        'pool_key': pool_key,
        'contexts': labels,
        'top_neuron_indices': top_indices.tolist(),
        'weights': top_weights.tolist(),
    }
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)

    return save_path


# ============================================================
# 2. Neuron Embedding Space (UMAP)
# ============================================================

def visualize_embedding_space(
    model_cls, params, config: Dict,
    output_dir: str,
    method: str = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> str:
    """
    2D scatter plot of neuron embeddings colored by pool type.

    Uses UMAP (preferred) or PCA as fallback.

    Args:
        model_cls: DAWN JAX model class
        params: FrozenDict of model parameters
        config: Model config
        output_dir: Output directory
        method: 'umap' or 'pca'
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("  matplotlib required for visualization")
        return ""

    # Get neuron embeddings
    emb = get_neuron_embeddings_jax(params)
    if emb is None:
        print("  No neuron embeddings found in checkpoint")
        return ""

    print(f"  Neuron embeddings shape: {emb.shape}")

    # Build pool boundaries
    pool_sizes = {
        'feature_qk':   config.get('n_feature_qk', 0),
        'feature_v':    config.get('n_feature_v', 0),
        'restore_qk':   config.get('n_restore_qk', 0),
        'restore_v':    config.get('n_restore_v', 0),
        'feature_know': config.get('n_feature_know', 0),
        'restore_know': config.get('n_restore_know', 0),
    }

    # Assign pool labels to each neuron
    pool_labels_arr = []
    pool_colors_arr = []
    offset = 0
    pool_order = ['feature_qk', 'feature_v', 'restore_qk', 'restore_v',
                  'feature_know', 'restore_know']

    for pool_name in pool_order:
        n = pool_sizes[pool_name]
        if n > 0 and offset + n <= len(emb):
            pool_labels_arr.extend([pool_name] * n)
            pool_colors_arr.extend([POOL_COLORS[pool_name]] * n)
            offset += n

    if offset == 0:
        print("  No valid pool boundaries found")
        return ""

    emb_used = emb[:offset]
    print(f"  Total neurons mapped: {offset}")

    # Dimensionality reduction
    if method == 'umap' and HAS_UMAP:
        print(f"  Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=42, metric='cosine')
        coords = reducer.fit_transform(emb_used)
        method_label = 'UMAP'
    elif HAS_SKLEARN:
        print("  Running PCA (UMAP not available)...")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_used)
        method_label = f'PCA (var: {pca.explained_variance_ratio_.sum():.1%})'
    else:
        print("  Neither UMAP nor sklearn available")
        return ""

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for pool_name in pool_order:
        mask = np.array([l == pool_name for l in pool_labels_arr])
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=POOL_COLORS[pool_name],
            label=f'{POOL_LABELS[pool_name]} ({mask.sum()})',
            s=15, alpha=0.7, edgecolors='none',
        )

    ax.legend(loc='best', fontsize=9, framealpha=0.9, markerscale=2)
    ax.set_xlabel(f'{method_label} 1', fontsize=11)
    ax.set_ylabel(f'{method_label} 2', fontsize=11)
    ax.set_title(
        f'Neuron Embedding Space ({method_label})\n'
        f'{offset} neurons across 6 pools, d_space={config.get("d_space", "?")}',
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'embedding_space_{method}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Save coordinates
    data_path = os.path.join(output_dir, f'embedding_space_{method}.json')
    data = {
        'method': method_label,
        'n_neurons': offset,
        'pool_sizes': pool_sizes,
        'pool_labels': pool_labels_arr,
    }
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)

    return save_path


# ============================================================
# 3. Layer-wise Routing Entropy
# ============================================================

def visualize_layer_entropy(
    model_cls, params, config: Dict,
    output_dir: str,
    val_data_path: str = None,
    val_tokens: np.ndarray = None,
    n_batches: int = 20,
    batch_size: int = 4,
    seq_len: int = 512,
) -> str:
    """
    Line plot of routing entropy at each layer depth.

    Runs partial forward passes to get intermediate representations
    at each layer, then computes routing weights and their entropy.

    Args:
        model_cls: DAWN JAX model class
        params: FrozenDict of model parameters
        config: Model config
        output_dir: Output directory
        val_data_path: Path to validation data (.bin or .pt)
        val_tokens: Pre-loaded validation tokens (alternative to val_data_path)
        n_batches: Number of batches to process
        batch_size: Batch size (small for TPU memory)
        seq_len: Sequence length

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("  matplotlib required for visualization")
        return ""
    if not HAS_JAX:
        print("  JAX required")
        return ""

    # Load validation data
    if val_tokens is None:
        if val_data_path is None:
            print("  Need val_data_path or val_tokens for layer entropy analysis")
            return ""
        max_tokens = n_batches * batch_size * seq_len * 2
        val_tokens = load_val_data_jax(val_data_path, max_tokens=max_tokens)

    batches = create_batches(val_tokens, batch_size, seq_len)
    if n_batches:
        batches = batches[:n_batches]

    n_layers = config.get('n_layers', 16)
    print(f"  Analyzing {n_layers} layers over {len(batches)} batches...")

    # Extract routing params
    all_params = params.get('params', params)
    router_params = all_params.get('router', {})
    sn_params = all_params.get('shared_neurons', {})

    # Import pure functions from model
    from models.model_v17_1_jax import (
        _layer_norm, _router_attn_forward, _router_know_forward,
        _attention_forward, _knowledge_forward,
    )

    # Routing keys to track entropy
    routing_pools = {
        'F-QK_Q': 'fqk_Q',
        'F-QK_K': 'fqk_K',
        'F-V': 'fv',
        'R-QK_Q': 'rqk_Q',
        'R-QK_K': 'rqk_K',
        'R-V': 'rv',
        'F-Know': 'fknow',
        'R-Know': 'rknow',
    }

    # Accumulate entropy per layer per pool
    # entropy_accum[pool_name][layer_idx] = list of entropy values
    entropy_accum = {name: [[] for _ in range(n_layers)] for name in routing_pools}

    # Stack per-block params
    block_params_list = [all_params[f'block_{i}'] for i in range(n_layers)]

    rng_key = jax.random.PRNGKey(42)

    for batch in tqdm(batches, desc='Layer Entropy'):
        input_ids = jnp.array(batch)
        B, S = input_ids.shape

        # Compute initial embeddings
        token_emb_table = all_params['token_emb']['embedding']
        pos_emb_table = all_params['pos_emb']['embedding']
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = token_emb_table[input_ids] + pos_emb_table[positions]

        rng_key, batch_rng = jax.random.split(rng_key)

        # Process each layer
        for layer_idx in range(n_layers):
            bp = block_params_list[layer_idx]
            rng_key, rng_ar, rng_kr, rng_a, rng_k = jax.random.split(rng_key, 5)

            # Attention sub-block
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])

            (fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
             attn_aux) = _router_attn_forward(
                normed, router_params,
                config.get('n_feature_qk', 88), config.get('n_feature_v', 352),
                config.get('n_restore_qk', 88), config.get('n_restore_v', 352),
                config.get('d_space', 256),
                config.get('top_k_feature_qk', 16), config.get('top_k_feature_v', 16),
                config.get('top_k_restore_qk', 16), config.get('top_k_restore_v', 16),
                0.0, None, True, rng_ar)  # no dropout for analysis

            # Compute entropy for attention routing weights
            attn_weights_dict = {
                'F-QK_Q': np.array(fqk_w_Q), 'F-QK_K': np.array(fqk_w_K),
                'F-V': np.array(fv_w),
                'R-QK_Q': np.array(rqk_w_Q), 'R-QK_K': np.array(rqk_w_K),
                'R-V': np.array(rv_w),
            }

            for name, w in attn_weights_dict.items():
                # w: [B, S, N] - compute per-token entropy, average
                ent = calc_entropy(w, axis=-1)  # [B, S]
                entropy_accum[name][layer_idx].append(float(ent.mean()))

            # Forward attention to get next x
            attn_out = _attention_forward(
                normed, sn_params,
                fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
                bp['attn']['expand_O']['kernel'],
                config.get('n_feature_qk', 88), config.get('n_restore_qk', 88),
                config.get('n_heads', 8), config.get('d_model', 768),
                0.0, True, rng_a)
            x = x + attn_out

            # Knowledge sub-block
            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])

            feat_know_w, rest_know_w, know_aux = _router_know_forward(
                normed, router_params,
                config.get('n_feature_qk', 88), config.get('n_feature_v', 352),
                config.get('n_restore_qk', 88), config.get('n_restore_v', 352),
                config.get('n_feature_know', 224), config.get('n_restore_know', 224),
                config.get('top_k_feature_know', 16), config.get('top_k_restore_know', 16),
                0.0, None, True, rng_kr)

            know_weights_dict = {
                'F-Know': np.array(feat_know_w),
                'R-Know': np.array(rest_know_w),
            }

            for name, w in know_weights_dict.items():
                ent = calc_entropy(w, axis=-1)
                entropy_accum[name][layer_idx].append(float(ent.mean()))

            know_out = _knowledge_forward(
                normed, sn_params,
                feat_know_w, rest_know_w,
                0.0, True, rng_k)
            x = x + know_out

    # Compute mean entropy per layer per pool
    entropy_means = {}
    entropy_stds = {}
    for name in routing_pools:
        means = []
        stds = []
        for layer_idx in range(n_layers):
            vals = entropy_accum[name][layer_idx]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0.0)
                stds.append(0.0)
        entropy_means[name] = np.array(means)
        entropy_stds[name] = np.array(stds)

    # Compute max possible entropy for each pool
    pool_sizes_for_max = {
        'F-QK_Q': config.get('top_k_feature_qk', 16),
        'F-QK_K': config.get('top_k_feature_qk', 16),
        'F-V': config.get('top_k_feature_v', 16),
        'R-QK_Q': config.get('top_k_restore_qk', 16),
        'R-QK_K': config.get('top_k_restore_qk', 16),
        'R-V': config.get('top_k_restore_v', 16),
        'F-Know': config.get('top_k_feature_know', 16),
        'R-Know': config.get('top_k_restore_know', 16),
    }

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    layers = np.arange(n_layers)

    # Left: Attention routing entropy
    ax = axes[0]
    attn_pools = ['F-QK_Q', 'F-QK_K', 'F-V', 'R-QK_Q', 'R-QK_K', 'R-V']
    attn_colors = ['#E63946', '#C1121F', '#F4A261', '#457B9D', '#1D3557', '#2A9D8F']

    for name, color in zip(attn_pools, attn_colors):
        means = entropy_means[name]
        stds = entropy_stds[name]
        ax.plot(layers, means, '-o', color=color, label=name,
                markersize=4, linewidth=1.5)
        ax.fill_between(layers, means - stds, means + stds,
                        color=color, alpha=0.1)

    # Max entropy reference line
    max_ent = np.log(pool_sizes_for_max.get('F-V', 16))
    ax.axhline(y=max_ent, color='gray', linestyle='--', alpha=0.5, label=f'Max (uniform top-k)')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Routing Entropy (nats)', fontsize=11)
    ax.set_title('Attention Routing Entropy by Layer', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)

    # Right: Knowledge routing entropy
    ax = axes[1]
    know_pools = ['F-Know', 'R-Know']
    know_colors = ['#9B5DE5', '#00BBF9']

    for name, color in zip(know_pools, know_colors):
        means = entropy_means[name]
        stds = entropy_stds[name]
        ax.plot(layers, means, '-o', color=color, label=name,
                markersize=4, linewidth=1.5)
        ax.fill_between(layers, means - stds, means + stds,
                        color=color, alpha=0.1)

    max_ent_know = np.log(pool_sizes_for_max.get('F-Know', 16))
    ax.axhline(y=max_ent_know, color='gray', linestyle='--', alpha=0.5, label=f'Max (uniform top-k)')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Routing Entropy (nats)', fontsize=11)
    ax.set_title('Knowledge Routing Entropy by Layer', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)

    fig.suptitle(
        f'Layer-wise Routing Entropy ({n_layers} layers, {len(batches)} batches)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'layer_routing_entropy.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Save raw data
    data_path = os.path.join(output_dir, 'layer_routing_entropy.json')
    data = {
        'n_layers': n_layers,
        'n_batches': len(batches),
        'entropy_means': {k: v.tolist() for k, v in entropy_means.items()},
        'entropy_stds': {k: v.tolist() for k, v in entropy_stds.items()},
        'max_entropy': {k: float(np.log(v)) for k, v in pool_sizes_for_max.items()},
    }
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)

    return save_path


# ============================================================
# Helpers
# ============================================================

def _create_model_instance(config):
    """Create model instance from config."""
    from scripts.analysis.utils_jax import create_model_from_config
    return create_model_from_config(config)


def _get_routing_weight(routing_data: Dict, pool_key: str) -> Optional[np.ndarray]:
    """Get routing weight array from routing data dict."""
    key_map = {
        'fqk_q': 'fqk_weights_Q',
        'fqk_k': 'fqk_weights_K',
        'fv':    'fv_weights',
        'rqk_q': 'rqk_weights_Q',
        'rqk_k': 'rqk_weights_K',
        'rv':    'rv_weights',
        'fknow': 'feature_know_w',
        'rknow': 'restore_know_w',
    }
    raw_key = key_map.get(pool_key, pool_key)

    attn = routing_data.get('attention', {})
    know = routing_data.get('knowledge', {})
    result = attn.get(raw_key)
    if result is None:
        result = know.get(raw_key)
    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Interpretability Visualizations')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint path (local or gs://)')
    parser.add_argument('--output', default='./interp_results',
                        help='Output directory')
    parser.add_argument('--viz', default='all',
                        choices=['all', 'context_routing', 'embedding_space', 'layer_entropy'],
                        help='Which visualization to generate')
    parser.add_argument('--val_data', default=None,
                        help='Validation data path (.bin or .pt) for layer entropy')

    # Context routing options
    parser.add_argument('--target_token', default='the',
                        help='Target token for context routing (comma-sep for multi: the,is,in)')
    parser.add_argument('--pool_key', default='fv',
                        help='Routing pool key (fv, fqk_q, rv, etc. Comma-sep for multi)')
    parser.add_argument('--top_n_neurons', type=int, default=30,
                        help='Number of top neurons to display')
    parser.add_argument('--contexts_file', default=None,
                        help='JSON file with custom context pairs (overrides presets)')

    # Embedding space options
    parser.add_argument('--embed_method', default='umap', choices=['umap', 'pca'],
                        help='Dimensionality reduction method')

    # Layer entropy options
    parser.add_argument('--n_batches', type=int, default=20,
                        help='Number of batches for layer entropy')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (keep small for TPU memory)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    print(f"  Model: {config.get('model_version', '?')}, "
          f"d_model={config.get('d_model', '?')}, "
          f"n_layers={config.get('n_layers', '?')}")

    os.makedirs(args.output, exist_ok=True)

    run_all = args.viz == 'all'

    if run_all or args.viz == 'context_routing':
        print("\n[1/3] Context-Dependent Routing Heatmap")
        # Load custom contexts if provided
        custom_contexts = None
        if args.contexts_file:
            custom_contexts = load_contexts_from_file(args.contexts_file)
            print(f"  Loaded {len(custom_contexts)} contexts from {args.contexts_file}")

        # Support comma-separated tokens and pools for exploration
        tokens = [t.strip() for t in args.target_token.split(',')]
        pools = [p.strip() for p in args.pool_key.split(',')]

        for token in tokens:
            for pool in pools:
                print(f"  Token='{token}', Pool={pool}")
                visualize_context_routing(
                    model_cls, params, config, args.output,
                    target_token=token,
                    context_pairs=custom_contexts,
                    pool_key=pool,
                    top_n_neurons=args.top_n_neurons,
                )

    if run_all or args.viz == 'embedding_space':
        print("\n[2/3] Neuron Embedding Space")
        visualize_embedding_space(
            model_cls, params, config, args.output,
            method=args.embed_method,
        )

    if run_all or args.viz == 'layer_entropy':
        print("\n[3/3] Layer-wise Routing Entropy")
        if args.val_data is None and run_all:
            print("  Skipping: --val_data required for layer entropy")
        else:
            visualize_layer_entropy(
                model_cls, params, config, args.output,
                val_data_path=args.val_data,
                n_batches=args.n_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )

    print(f"\nDone. Results in: {args.output}")


if __name__ == '__main__':
    main()

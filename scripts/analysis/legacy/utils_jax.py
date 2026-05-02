"""
DAWN JAX Analysis Utilities
============================
JAX/Flax-specific utility functions for DAWN model analysis on TPU.

This module provides:
- JAX model loading utilities (native .flax checkpoint loading)
- JAX-compatible routing data extraction
- Data loading for JAX (numpy-based)
- Mathematical utilities compatible with JAX arrays

Parallel to utils.py but for JAX/TPU environment.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax import serialization
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    nn = None

# Import shared config from utils.py — with torch-free fallback
# utils.py requires torch at import time. On TPU-only environments
# (no PyTorch), we define the constants locally.
try:
    from scripts.analysis.utils import (
        NEURON_TYPES, NEURON_TYPES_V18, EMBEDDING_POOLS_V18,
        ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS, ALL_ROUTING_KEYS,
        QK_POOL_SHORTHAND, POOL_TYPE_ALIASES, POOL_N_ATTR,
        POOL_DISPLAY_NAMES, POOL_SHORTHAND,
        WEIGHT_KEY_MAP, MASK_KEY_MAP,
        get_neuron_display_name, parse_neuron_name, resolve_pool_type,
        gini_coefficient as _torch_gini,
        convert_to_serializable, save_results,
    )
    _HAS_TORCH_UTILS = True
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH_UTILS = False

    # ---- Torch-free definitions of all constants ----
    NEURON_TYPES = {
        'feature_qk':   ('F-QK',   'n_feature_qk',   'red'),
        'feature_v':    ('F-V',    'n_feature_v',    'orange'),
        'restore_qk':   ('R-QK',   'n_restore_qk',   'blue'),
        'restore_v':    ('R-V',    'n_restore_v',    'green'),
        'feature_know': ('F-Know', 'n_feature_know', 'purple'),
        'restore_know': ('R-Know', 'n_restore_know', 'cyan'),
    }
    NEURON_TYPES_V18 = {
        'feature_q':    ('F-Q',    'n_feature_qk',   'red'),
        'feature_k':    ('F-K',    'n_feature_qk',   'darkred'),
        'feature_v':    ('F-V',    'n_feature_v',    'orange'),
        'restore_q':    ('R-Q',    'n_restore_qk',   'blue'),
        'restore_k':    ('R-K',    'n_restore_qk',   'darkblue'),
        'restore_v':    ('R-V',    'n_restore_v',    'green'),
        'feature_know': ('F-Know', 'n_feature_know', 'purple'),
        'restore_know': ('R-Know', 'n_restore_know', 'cyan'),
    }
    EMBEDDING_POOLS_V18 = {
        'feature_qk':   ('FQK',    'n_feature_qk',   'red'),
        'feature_v':    ('FV',     'n_feature_v',    'orange'),
        'restore_qk':   ('RQK',    'n_restore_qk',   'blue'),
        'restore_v':    ('RV',     'n_restore_v',    'green'),
        'feature_know': ('F-Know', 'n_feature_know', 'purple'),
        'restore_know': ('R-Know', 'n_restore_know', 'cyan'),
    }
    ROUTING_KEYS = {
        'fqk_q': ('F-QK_Q', 'fqk_q_pref', 'fqk_weights_Q', 'feature_qk'),
        'fqk_k': ('F-QK_K', 'fqk_k_pref', 'fqk_weights_K', 'feature_qk'),
        'fv':    ('F-V',    'fv_pref',    'fv_weights',    'feature_v'),
        'rqk_q': ('R-QK_Q', 'rqk_q_pref', 'rqk_weights_Q', 'restore_qk'),
        'rqk_k': ('R-QK_K', 'rqk_k_pref', 'rqk_weights_K', 'restore_qk'),
        'rv':    ('R-V',    'rv_pref',    'rv_weights',    'restore_v'),
    }
    KNOWLEDGE_ROUTING_KEYS = {
        'fknow': ('F-Know', 'feature_know_w', 'feature_know'),
        'rknow': ('R-Know', 'restore_know_w', 'restore_know'),
    }
    ALL_ROUTING_KEYS = {**ROUTING_KEYS}
    for _k, _v in KNOWLEDGE_ROUTING_KEYS.items():
        ALL_ROUTING_KEYS[_k] = (_v[0], None, _v[1], _v[2])
    QK_POOL_SHORTHAND = {
        'fqk': ('feature_qk', 'fqk_q', 'fqk_k'),
        'rqk': ('restore_qk', 'rqk_q', 'rqk_k'),
    }
    POOL_TYPE_ALIASES = {
        'fqk': 'fqk_q', 'rqk': 'rqk_q',
        'feature_qk': 'fqk_q', 'feature_v': 'fv',
        'restore_qk': 'rqk_q', 'restore_v': 'rv',
        'feature_know': 'fknow', 'restore_know': 'rknow',
    }
    POOL_N_ATTR = {
        'feature_qk': 'n_feature_qk', 'feature_v': 'n_feature_v',
        'restore_qk': 'n_restore_qk', 'restore_v': 'n_restore_v',
        'feature_know': 'n_feature_know', 'restore_know': 'n_restore_know',
    }
    POOL_DISPLAY_NAMES = {
        'fqk': 'F_QK', 'fv': 'F_V', 'rqk': 'R_QK',
        'rv': 'R_V', 'fknow': 'F_Know', 'rknow': 'R_Know',
    }
    POOL_SHORTHAND = {v: k for k, v in POOL_DISPLAY_NAMES.items()}
    WEIGHT_KEY_MAP = {
        'fqk_q': 'fqk_weights_Q', 'fqk_k': 'fqk_weights_K',
        'fv': 'fv_weights', 'rqk_q': 'rqk_weights_Q',
        'rqk_k': 'rqk_weights_K', 'rv': 'rv_weights',
        'fknow': 'feature_know_w', 'rknow': 'restore_know_w',
    }
    MASK_KEY_MAP = {
        'fqk_q': 'fqk_mask_Q', 'fqk_k': 'fqk_mask_K',
        'fv': 'fv_mask', 'rqk_q': 'rqk_mask_Q',
        'rqk_k': 'rqk_mask_K', 'rv': 'rv_mask',
        'fknow': 'feature_know_mask', 'rknow': 'restore_know_mask',
    }
    QK_POOLS = {
        'feature_qk': {
            'display': 'F-QK', 'q_pref': 'fqk_q_pref', 'k_pref': 'fqk_k_pref',
            'q_weight': 'fqk_weights_Q', 'k_weight': 'fqk_weights_K',
            'n_attr': 'n_feature_qk', 'color': 'red',
        },
        'restore_qk': {
            'display': 'R-QK', 'q_pref': 'rqk_q_pref', 'k_pref': 'rqk_k_pref',
            'q_weight': 'rqk_weights_Q', 'k_weight': 'rqk_weights_K',
            'n_attr': 'n_restore_qk', 'color': 'blue',
        },
    }

    def resolve_pool_type(pool_type: str) -> str:
        return POOL_TYPE_ALIASES.get(pool_type, pool_type)

    def get_neuron_display_name(pool_shorthand: str, local_idx: int) -> str:
        display = POOL_DISPLAY_NAMES.get(pool_shorthand, pool_shorthand.upper())
        return f'{display}_{local_idx}'

    def parse_neuron_name(neuron_name: str):
        parts = neuron_name.rsplit('_', 1)
        if len(parts) != 2:
            return 'unknown', -1
        display_prefix, idx_str = parts
        try:
            local_idx = int(idx_str)
        except ValueError:
            return 'unknown', -1
        pool_shorthand = POOL_SHORTHAND.get(display_prefix, display_prefix.lower())
        return pool_shorthand, local_idx

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    def save_results(results, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)


# ============================================================
# GCS Utilities (shared with utils.py)
# ============================================================

def _is_gcs_path(path):
    return str(path).startswith('gs://')


def _list_gcs_files(gcs_dir, pattern='*'):
    """List files in a GCS directory matching pattern."""
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        bucket_path = gcs_dir.replace('gs://', '').rstrip('/')
        entries = fs.ls(bucket_path)
        import fnmatch
        files = []
        for e in entries:
            name = e.split('/')[-1]
            if fnmatch.fnmatch(name, pattern):
                files.append('gs://' + e)
        return files
    except Exception as e:
        print(f"  Warning: GCS listing failed: {e}")
        return []


def _read_gcs_bytes(path: str) -> bytes:
    """Read bytes from GCS path."""
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, 'rb') as f:
            return f.read()
    except ImportError:
        import tensorflow as tf
        return tf.io.gfile.GFile(path, 'rb').read()


# ============================================================
# JAX Model Loading
# ============================================================

def load_flax_checkpoint(path: str) -> Dict:
    """Load a .flax checkpoint (local or GCS).

    Returns:
        Dict with 'params', 'config', 'epoch', 'step', 'best_val_loss'
    """
    path_str = str(path)
    if _is_gcs_path(path_str):
        data = _read_gcs_bytes(path_str)
    else:
        with open(path_str, 'rb') as f:
            data = f.read()

    # Deserialize without target (raw dict/array structure)
    ckpt = serialization.from_bytes(None, data)
    return ckpt


def load_model_jax(checkpoint_path: str, config_override: Dict = None):
    """
    Load DAWN v17.1 model from .flax checkpoint as native JAX model.

    Args:
        checkpoint_path: Path to checkpoint file or directory (local or gs://)
        config_override: Optional config overrides

    Returns:
        Tuple of (model_cls, params, config)
        - model_cls: DAWN class (not instance - use model.apply())
        - params: FrozenDict of model parameters
        - config: Model configuration dict
    """
    if not HAS_JAX:
        raise RuntimeError("JAX/Flax not available. Install with: pip install jax flax")

    checkpoint_path = str(checkpoint_path)

    # Check if it's a directory
    is_dir = False
    if _is_gcs_path(checkpoint_path):
        if not checkpoint_path.endswith('.flax') and not checkpoint_path.endswith('.pt'):
            is_dir = True
    else:
        path = Path(checkpoint_path)
        is_dir = path.is_dir()

    if is_dir:
        # Look for .flax checkpoint
        if _is_gcs_path(checkpoint_path):
            flax_files = _list_gcs_files(checkpoint_path, '*.flax')
        else:
            path = Path(checkpoint_path)
            flax_files = [str(f) for f in path.glob('*.flax')]

        # Find best checkpoint
        found = False
        for f in flax_files:
            fname = f.split('/')[-1].lower()
            if 'best' in fname:
                checkpoint_path = f
                found = True
                break

        if not found and flax_files:
            checkpoint_path = sorted(flax_files)[-1]  # Latest by name
        elif not flax_files:
            raise FileNotFoundError(f"No .flax files found in {checkpoint_path}")

    print(f"Loading: {checkpoint_path}")
    ckpt = load_flax_checkpoint(checkpoint_path)

    params = ckpt['params']
    config = ckpt.get('config', {})

    if config_override:
        config.update(config_override)

    # Detect model type
    version = config.get('model_version', '')
    if version in ('baseline', 'baseline-JAX'):
        from models.baseline_transformer_jax import VanillaTransformer
        model_cls = VanillaTransformer
        config['model_version'] = 'baseline'
    else:
        from models.model_v17_1_jax import DAWN
        model_cls = DAWN
        config['model_version'] = '17.1'

    print(f"  Model version: {config['model_version']}")

    # Convert params to FrozenDict if not already
    from flax.core import freeze
    if not isinstance(params, dict):
        params = dict(params)
    params = freeze({'params': params})

    return model_cls, params, config


def create_model_from_config(config: Dict):
    """Create JAX model instance from config.

    Args:
        config: Model configuration dict

    Returns:
        Model class (for use with model.apply(params, ...))
    """
    version = config.get('model_version', '17.1')

    if version in ('baseline', 'baseline-JAX'):
        from models.baseline_transformer_jax import VanillaTransformer
        return VanillaTransformer(
            vocab_size=config.get('vocab_size', 30522),
            d_model=config.get('d_model', 768),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', config.get('d_model', 768) * 4),
            max_seq_len=config.get('max_seq_len', 512),
            dropout_rate=config.get('dropout', 0.1),
        )
    else:
        from models.model_v17_1_jax import DAWN
        return DAWN(
            vocab_size=config.get('vocab_size', 30522),
            d_model=config.get('d_model', 768),
            n_layers=config.get('n_layers', 16),
            n_heads=config.get('n_heads', 8),
            rank=config.get('rank', 64),
            max_seq_len=config.get('max_seq_len', 512),
            d_space=config.get('d_space', 256),
            n_feature_qk=config.get('n_feature_qk', 88),
            n_feature_v=config.get('n_feature_v', 352),
            top_k_feature_qk=config.get('top_k_feature_qk', 16),
            top_k_feature_v=config.get('top_k_feature_v', 16),
            n_restore_qk=config.get('n_restore_qk', 88),
            n_restore_v=config.get('n_restore_v', 352),
            top_k_restore_qk=config.get('top_k_restore_qk', 16),
            top_k_restore_v=config.get('top_k_restore_v', 16),
            n_feature_know=config.get('n_feature_know', 224),
            n_restore_know=config.get('n_restore_know', 224),
            top_k_feature_know=config.get('top_k_feature_know', 16),
            top_k_restore_know=config.get('top_k_restore_know', 16),
            knowledge_rank=config.get('knowledge_rank', 64),
            dropout_rate=config.get('dropout', 0.1),
            router_dropout=config.get('router_dropout', 0.1),
            gradient_checkpointing=config.get('gradient_checkpointing', False),
        )


# ============================================================
# DataLoader Utilities (JAX/NumPy)
# ============================================================

def load_bin_data(data_path: str, max_tokens: int = None) -> np.ndarray:
    """Load pretokenized .bin data.

    Args:
        data_path: Path to .bin file (local or gs://)
        max_tokens: Maximum tokens to load (None = all)

    Returns:
        numpy array of token IDs
    """
    if _is_gcs_path(data_path):
        import gcsfs
        fs = gcsfs.GCSFileSystem()

        # Get file size
        info = fs.info(data_path)
        total_bytes = info['size']

        if max_tokens:
            bytes_needed = min(max_tokens * 2, total_bytes)  # int16 = 2 bytes
        else:
            bytes_needed = total_bytes

        with fs.open(data_path, 'rb') as f:
            data = np.frombuffer(f.read(bytes_needed), dtype=np.int16)
    else:
        if max_tokens:
            data = np.memmap(data_path, dtype=np.int16, mode='r')[:max_tokens]
        else:
            data = np.memmap(data_path, dtype=np.int16, mode='r')

    return data.astype(np.int32)


def create_batches(tokens: np.ndarray, batch_size: int, seq_len: int) -> List[np.ndarray]:
    """Create batches from flat token array.

    Args:
        tokens: Flat array of token IDs
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        List of batches, each [batch_size, seq_len]
    """
    n_seqs = len(tokens) // seq_len
    n_batches = n_seqs // batch_size

    tokens = tokens[:n_batches * batch_size * seq_len]
    tokens = tokens.reshape(n_batches, batch_size, seq_len)

    return list(tokens)


def load_val_data_jax(val_path: str, max_tokens: int = 10_000_000) -> np.ndarray:
    """Load validation data for JAX.

    Args:
        val_path: Path to validation data (.bin or .pt)
        max_tokens: Maximum tokens to load

    Returns:
        numpy array of token IDs
    """
    if val_path.endswith('.bin'):
        return load_bin_data(val_path, max_tokens)
    elif val_path.endswith('.pt'):
        import torch
        data = torch.load(val_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            tokens = data.get('input_ids', data.get('tokens'))
        else:
            tokens = data
        return np.array(tokens)[:max_tokens]
    else:
        raise ValueError(f"Unsupported format: {val_path}")


# ============================================================
# JAX Routing Data Extraction
# ============================================================

def topk_sparsify_np(weights: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k sparsification with renormalization (numpy version).

    Args:
        weights: [..., N] softmax probabilities
        k: number of top elements to keep

    Returns:
        sparse_weights: [..., N] with only top-k nonzero, renormalized
        topk_idx: [..., k] indices
    """
    # Get top-k indices and values
    topk_idx = np.argpartition(weights, -k, axis=-1)[..., -k:]

    # Gather values at top-k positions
    N = weights.shape[-1]
    batch_shape = weights.shape[:-1]

    # Create sparse output
    sparse = np.zeros_like(weights)

    # For each position in batch, set top-k values
    for idx in np.ndindex(batch_shape):
        top_indices = topk_idx[idx]
        top_values = weights[idx][top_indices]
        sparse[idx][top_indices] = top_values

    # Renormalize
    sparse = sparse / (sparse.sum(axis=-1, keepdims=True) + 1e-8)

    return sparse, topk_idx


class JAXRoutingDataExtractor:
    """
    Extract routing weights from JAX DAWN model.

    Unlike PyTorch which stores routing_info during forward, JAX requires
    explicit extraction by calling router methods.

    Usage:
        extractor = JAXRoutingDataExtractor(model, params, config)
        routing_data = extractor.extract_routing(input_ids)
    """

    def __init__(self, model, params, config: Dict):
        """
        Args:
            model: JAX DAWN model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        self.model = model
        self.params = params
        self.config = config

        # Extract routing config
        self.n_feature_qk = config.get('n_feature_qk', 88)
        self.n_feature_v = config.get('n_feature_v', 352)
        self.n_restore_qk = config.get('n_restore_qk', 88)
        self.n_restore_v = config.get('n_restore_v', 352)
        self.n_feature_know = config.get('n_feature_know', 224)
        self.n_restore_know = config.get('n_restore_know', 224)

        self.top_k_feature_qk = config.get('top_k_feature_qk', 16)
        self.top_k_feature_v = config.get('top_k_feature_v', 16)
        self.top_k_restore_qk = config.get('top_k_restore_qk', 16)
        self.top_k_restore_v = config.get('top_k_restore_v', 16)
        self.top_k_feature_know = config.get('top_k_feature_know', 16)
        self.top_k_restore_know = config.get('top_k_restore_know', 16)

        self.d_space = config.get('d_space', 256)

    def _get_router_params(self):
        """Extract router parameters from full params."""
        # params structure: {'params': {'router': {...}, 'shared_neurons': {...}, ...}}
        all_params = self.params.get('params', self.params)
        return all_params.get('router', {})

    def _get_neuron_embeddings(self):
        """Get normalized neuron embeddings."""
        router_params = self._get_router_params()
        nr_params = router_params.get('neuron_router', {})

        neuron_emb = nr_params.get('neuron_emb')
        if neuron_emb is None:
            return None

        # Normalize embeddings
        norm = np.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8
        return neuron_emb / norm

    def compute_attention_routing(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute attention routing weights for input embeddings.

        Args:
            x: Input embeddings [B, S, D]

        Returns:
            Dict with routing weights for each pool
        """
        router_params = self._get_router_params()
        nr_params = router_params.get('neuron_router', {})

        # Get neuron embeddings
        emb_norm = self._get_neuron_embeddings()
        if emb_norm is None:
            return {}

        # Pool boundaries
        fqk_end = self.n_feature_qk
        fv_end = fqk_end + self.n_feature_v
        rqk_end = fv_end + self.n_restore_qk
        rv_end = rqk_end + self.n_restore_v

        # Projection: x @ proj_all.kernel + proj_all.bias
        proj_all = nr_params.get('proj_all', {})
        kernel = proj_all.get('kernel')
        bias = proj_all.get('bias')

        if kernel is None:
            return {}

        all_proj = np.einsum('bsd,df->bsf', x, kernel)
        if bias is not None:
            all_proj = all_proj + bias

        # Split into 6 projections
        d_space = all_proj.shape[-1] // 6
        splits = np.split(all_proj, 6, axis=-1)
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = splits

        # Pool embeddings
        fqk_emb = emb_norm[:fqk_end]
        fv_emb = emb_norm[fqk_end:fv_end]
        rqk_emb = emb_norm[fv_end:rqk_end]
        rv_emb = emb_norm[rqk_end:rv_end]

        # Compute logits
        logits_fqk_Q = np.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = np.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = np.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = np.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = np.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = np.einsum('bsd,nd->bsn', h_rv, rv_emb)

        # Softmax
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-8)

        pref_fqk_Q = softmax(logits_fqk_Q)
        pref_fqk_K = softmax(logits_fqk_K)
        pref_fv = softmax(logits_fv)
        pref_rqk_Q = softmax(logits_rqk_Q)
        pref_rqk_K = softmax(logits_rqk_K)
        pref_rv = softmax(logits_rv)

        # Top-k sparsification
        fqk_w_Q, _ = topk_sparsify_np(pref_fqk_Q, self.top_k_feature_qk)
        fqk_w_K, _ = topk_sparsify_np(pref_fqk_K, self.top_k_feature_qk)
        fv_w, _ = topk_sparsify_np(pref_fv, self.top_k_feature_v)
        rqk_w_Q, _ = topk_sparsify_np(pref_rqk_Q, self.top_k_restore_qk)
        rqk_w_K, _ = topk_sparsify_np(pref_rqk_K, self.top_k_restore_qk)
        rv_w, _ = topk_sparsify_np(pref_rv, self.top_k_restore_v)

        return {
            'fqk_weights_Q': fqk_w_Q,
            'fqk_weights_K': fqk_w_K,
            'fv_weights': fv_w,
            'rqk_weights_Q': rqk_w_Q,
            'rqk_weights_K': rqk_w_K,
            'rv_weights': rv_w,
            # Raw preferences (before top-k)
            'fqk_pref_Q': pref_fqk_Q,
            'fqk_pref_K': pref_fqk_K,
            'fv_pref': pref_fv,
            'rqk_pref_Q': pref_rqk_Q,
            'rqk_pref_K': pref_rqk_K,
            'rv_pref': pref_rv,
        }

    def compute_knowledge_routing(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute knowledge routing weights for input embeddings.

        Args:
            x: Input embeddings [B, S, D]

        Returns:
            Dict with routing weights for knowledge pools
        """
        router_params = self._get_router_params()
        nr_params = router_params.get('neuron_router', {})

        emb_norm = self._get_neuron_embeddings()
        if emb_norm is None:
            return {}

        # Pool boundaries
        rv_end = self.n_feature_qk + self.n_feature_v + self.n_restore_qk + self.n_restore_v
        fk_end = rv_end + self.n_feature_know

        # Feature knowledge projection
        proj_fk = nr_params.get('proj_feature_know', {})
        kernel_fk = proj_fk.get('kernel')
        bias_fk = proj_fk.get('bias')

        if kernel_fk is None:
            return {}

        h_fk = np.einsum('bsd,df->bsf', x, kernel_fk)
        if bias_fk is not None:
            h_fk = h_fk + bias_fk

        # Restore knowledge projection
        proj_rk = nr_params.get('proj_restore_know', {})
        kernel_rk = proj_rk.get('kernel')
        bias_rk = proj_rk.get('bias')

        h_rk = np.einsum('bsd,df->bsf', x, kernel_rk)
        if bias_rk is not None:
            h_rk = h_rk + bias_rk

        # Pool embeddings
        emb_fk = emb_norm[rv_end:fk_end]
        emb_rk = emb_norm[fk_end:]

        # Compute logits
        logits_fk = np.einsum('bsd,nd->bsn', h_fk, emb_fk)
        logits_rk = np.einsum('bsd,nd->bsn', h_rk, emb_rk)

        # Softmax
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-8)

        pref_fk = softmax(logits_fk)
        pref_rk = softmax(logits_rk)

        # Top-k sparsification
        fk_w, _ = topk_sparsify_np(pref_fk, self.top_k_feature_know)
        rk_w, _ = topk_sparsify_np(pref_rk, self.top_k_restore_know)

        return {
            'feature_know_w': fk_w,
            'restore_know_w': rk_w,
            'feature_know_pref': pref_fk,
            'restore_know_pref': pref_rk,
        }

    def extract_routing(self, input_ids: np.ndarray, layer_idx: int = 0) -> Dict[str, np.ndarray]:
        """Extract all routing weights for given input.

        Note: This computes routing based on token embeddings + positional embeddings.
        For layer-specific routing (after transformations), you'd need to run
        partial forward passes.

        Args:
            input_ids: Token IDs [B, S]
            layer_idx: Layer index (for future layer-specific extraction)

        Returns:
            Dict with all routing weights
        """
        # Get token embeddings
        all_params = self.params.get('params', self.params)

        # Token embedding
        token_emb_table = all_params.get('token_emb', {}).get('embedding')
        if token_emb_table is None:
            return {}

        # Position embedding
        pos_emb_table = all_params.get('pos_emb', {}).get('embedding')

        # Compute embeddings
        B, S = input_ids.shape
        tok_emb = token_emb_table[input_ids]  # [B, S, D]

        if pos_emb_table is not None:
            positions = np.arange(S)[np.newaxis, :]  # [1, S]
            pos_emb = pos_emb_table[positions]  # [1, S, D]
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Compute routing weights
        attn_routing = self.compute_attention_routing(x)
        know_routing = self.compute_knowledge_routing(x)

        return {
            'attention': attn_routing,
            'knowledge': know_routing,
            'layer_idx': layer_idx,
        }


class JAXRoutingData:
    """
    Standardized routing data container for JAX (mirrors RoutingData from utils.py).
    """

    def __init__(self, routing_info: Dict):
        self.routing_info = routing_info
        self.attention = routing_info.get('attention', {})
        self.knowledge = routing_info.get('knowledge', {})

    def get_weight(self, key: str) -> Optional[np.ndarray]:
        """Get weight tensor by standardized key.

        Standard keys: 'fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow'
        """
        raw_key = WEIGHT_KEY_MAP.get(key, key)

        result = self.attention.get(raw_key)
        if result is None:
            result = self.knowledge.get(raw_key)

        return result

    def get_pref(self, key: str) -> Optional[np.ndarray]:
        """Get raw preference (before top-k) by standardized key."""
        key_map = {
            'fqk_q': 'fqk_pref_Q',
            'fqk_k': 'fqk_pref_K',
            'fv': 'fv_pref',
            'rqk_q': 'rqk_pref_Q',
            'rqk_k': 'rqk_pref_K',
            'rv': 'rv_pref',
            'fknow': 'feature_know_pref',
            'rknow': 'restore_know_pref',
        }
        raw_key = key_map.get(key, key)

        result = self.attention.get(raw_key)
        if result is None:
            result = self.knowledge.get(raw_key)

        return result

    def get_all_weights(self) -> Dict[str, np.ndarray]:
        """Get all available weights."""
        weights = {}
        for std_key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow']:
            w = self.get_weight(std_key)
            if w is not None:
                weights[std_key] = w
        return weights


# ============================================================
# Mathematical Utilities (NumPy versions)
# ============================================================

def gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient (numpy version)."""
    values = values.flatten().astype(np.float64)
    if values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return float(1 - 2 * cumsum.sum() / (n * sorted_vals.sum()) + 1/n)


def calc_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Calculate entropy along dimension (numpy version)."""
    probs = np.clip(probs, 1e-8, None)
    return -np.sum(probs * np.log(probs), axis=axis)


def calc_entropy_ratio(probs: np.ndarray) -> float:
    """Calculate entropy as percentage of maximum (numpy version)."""
    if probs.size == 0:
        return 0.0

    if probs.ndim == 1:
        avg_probs = probs
    elif probs.ndim == 2:
        avg_probs = probs.mean(axis=0)
    else:
        avg_probs = probs.reshape(-1, probs.shape[-1]).mean(axis=0)

    ent = calc_entropy(avg_probs, axis=-1)

    if isinstance(ent, np.ndarray) and ent.ndim > 0:
        ent = ent.mean()

    max_ent = np.log(probs.shape[-1])
    return float(ent / max_ent * 100) if max_ent > 0 else 0.0


# ============================================================
# JAX Forward Pass Utilities
# ============================================================

def forward_jax(model, params, input_ids: np.ndarray,
                deterministic: bool = True,
                rng_key = None) -> Dict:
    """Run forward pass with JAX model.

    Args:
        model: JAX DAWN model class
        params: FrozenDict of model parameters
        input_ids: Input token IDs [B, S] (numpy array)
        deterministic: Whether to disable dropout
        rng_key: Optional JAX PRNG key for dropout

    Returns:
        Dict with 'logits' and/or 'loss', 'aux_loss'
    """
    if not HAS_JAX:
        raise RuntimeError("JAX not available")

    input_ids = jnp.array(input_ids)

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    return model.apply(
        params,
        input_ids,
        deterministic=deterministic,
        rngs={'dropout': rng_key}
    )


def evaluate_jax(model, params, config: Dict,
                 val_tokens: np.ndarray,
                 batch_size: int = 32,
                 seq_len: int = 512) -> Dict:
    """Evaluate JAX model on validation data.

    Args:
        model: JAX DAWN model class
        params: FrozenDict of model parameters
        config: Model configuration
        val_tokens: Validation tokens (flat numpy array)
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dict with 'loss', 'perplexity', 'accuracy'
    """
    if not HAS_JAX:
        raise RuntimeError("JAX not available")

    batches = create_batches(val_tokens, batch_size, seq_len)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    rng_key = jax.random.PRNGKey(42)

    for batch in batches:
        batch_jax = jnp.array(batch)

        # Forward pass with labels for loss computation
        result = model.apply(
            params,
            batch_jax,
            labels=batch_jax,
            deterministic=True,
            rngs={'dropout': rng_key}
        )

        if 'loss' in result:
            total_loss += float(result['loss']) * (batch_size * (seq_len - 1))
            total_correct += int(result.get('correct', 0))
            total_tokens += int(result.get('valid_count', batch_size * (seq_len - 1)))

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
    }


# ============================================================
# Shared Neuron Extraction
# ============================================================

def get_shared_neurons_jax(params) -> Dict[str, np.ndarray]:
    """Extract shared neuron parameters from JAX model params.

    Args:
        params: FrozenDict of model parameters

    Returns:
        Dict with neuron arrays: f_neurons, r_neurons, feature_know, restore_know
    """
    all_params = params.get('params', params)
    sn = all_params.get('shared_neurons', {})

    return {
        'f_neurons': np.array(sn.get('f_neurons', [])),
        'r_neurons': np.array(sn.get('r_neurons', [])),
        'feature_know': np.array(sn.get('feature_know', [])),
        'restore_know': np.array(sn.get('restore_know', [])),
    }


def get_neuron_embeddings_jax(params) -> np.ndarray:
    """Extract neuron embeddings from JAX model params.

    Args:
        params: FrozenDict of model parameters

    Returns:
        Normalized neuron embeddings [N_total, d_space]
    """
    all_params = params.get('params', params)
    router = all_params.get('router', {})
    nr = router.get('neuron_router', {})

    emb = nr.get('neuron_emb')
    if emb is None:
        return None

    emb = np.array(emb)
    norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
    return emb / norm


# ============================================================
# FLOPs Estimation
# ============================================================

def estimate_flops_jax(config: Dict, seq_len: int = 512) -> int:
    """
    Estimate theoretical FLOPs for forward pass (per sequence) for JAX model.

    Reports theoretical FLOPs based on active neurons (top-k sparse).

    Args:
        config: Model configuration dict
        seq_len: Sequence length

    Returns:
        Estimated FLOPs as int
    """
    d_model = config.get('d_model', 384)
    n_layers = config.get('n_layers', 12)
    rank = config.get('rank', 64)
    knowledge_rank = config.get('knowledge_rank', 128)
    vocab_size = config.get('vocab_size', 30522)
    n_heads = config.get('n_heads', 8)
    d_space = config.get('d_space', 64)

    # Neuron counts
    n_feature_qk = config.get('n_feature_qk', 56)
    n_feature_v = config.get('n_feature_v', 24)
    n_restore_qk = config.get('n_restore_qk', 56)
    n_restore_v = config.get('n_restore_v', 24)
    n_feature_know = config.get('n_feature_know', 24)
    n_restore_know = config.get('n_restore_know', 24)

    # Top-k values
    top_k_feature_qk = config.get('top_k_feature_qk', 16)
    top_k_feature_v = config.get('top_k_feature_v', 6)
    top_k_restore_qk = config.get('top_k_restore_qk', 16)
    top_k_restore_v = config.get('top_k_restore_v', 6)
    top_k_feature_know = config.get('top_k_feature_know', 4)
    top_k_restore_know = config.get('top_k_restore_know', 4)

    S = seq_len
    d_head = d_model // n_heads

    # Check model type
    model_version = config.get('model_version', '')
    is_baseline = model_version in ('baseline', 'baseline-JAX')

    if is_baseline:
        # Standard transformer FLOPs
        d_ff = config.get('d_ff', d_model * 4)

        # Per layer
        # QKV projections: 3 * 2 * S * d_model * d_model
        qkv_flops = 3 * 2 * S * d_model * d_model

        # Attention: QK^T (S*S*d_model) + softmax + AV (S*S*d_model)
        attn_flops = 2 * 2 * S * S * d_model

        # Output projection: 2 * S * d_model * d_model
        o_proj_flops = 2 * S * d_model * d_model

        # FFN: up (2*S*d_model*d_ff) + down (2*S*d_ff*d_model)
        ffn_flops = 2 * 2 * S * d_model * d_ff

        layer_flops = qkv_flops + attn_flops + o_proj_flops + ffn_flops
        total_layers = n_layers * layer_flops

        # Embeddings: token lookup (negligible) + position lookup
        embed_flops = 2 * S * d_model  # LayerNorm

        # LM head: 2 * S * d_model * vocab_size
        lm_head_flops = 2 * S * d_model * vocab_size

        return embed_flops + total_layers + lm_head_flops

    # DAWN model FLOPs
    # Per layer
    # 1. Router projections: 6 attention + 2 knowledge
    router_attn_flops = 2 * S * d_model * (6 * d_space)  # proj_all
    router_know_flops = 2 * S * d_model * (2 * d_space)  # proj_feature_know + proj_restore_know

    # Router logits: einsum('bsd,nd->bsn') for each pool
    router_logits_flops = (
        2 * S * d_space * n_feature_qk * 2 +  # fqk Q/K
        2 * S * d_space * n_feature_v +
        2 * S * d_space * n_restore_qk * 2 +  # rqk Q/K
        2 * S * d_space * n_restore_v +
        2 * S * d_space * n_feature_know +
        2 * S * d_space * n_restore_know
    )

    router_flops = router_attn_flops + router_know_flops + router_logits_flops

    # 2. Attention circuit (sparse)
    # Feature: einsum('bsd,ndr->bsnr') then weighted sum -> [B,S,R]
    # Only top-k neurons active
    feature_qk_flops = 2 * S * d_model * rank * top_k_feature_qk * 2  # Q and K
    feature_v_flops = 2 * S * d_model * rank * top_k_feature_v

    # Restore: [B,S,R] -> [B,S,D] via sparse neurons
    restore_qk_flops = 2 * S * rank * d_model * top_k_restore_qk * 2  # Q and K
    restore_v_flops = 2 * S * rank * d_model * top_k_restore_v

    # Attention: QK^T (S*S*d_model) + softmax + AV
    attn_flops = 2 * 2 * S * S * d_model

    # Output projection
    o_proj_flops = 2 * S * d_model * d_model

    attn_circuit_flops = (feature_qk_flops + feature_v_flops +
                          restore_qk_flops + restore_v_flops +
                          attn_flops + o_proj_flops)

    # 3. Knowledge circuit (sparse)
    feature_know_flops = 2 * S * d_model * knowledge_rank * top_k_feature_know
    restore_know_flops = 2 * S * knowledge_rank * d_model * top_k_restore_know

    know_circuit_flops = feature_know_flops + restore_know_flops

    # LayerNorms (approx)
    layernorm_flops = 4 * S * d_model  # 2 per block

    layer_flops = router_flops + attn_circuit_flops + know_circuit_flops + layernorm_flops
    total_layers = n_layers * layer_flops

    # Embeddings
    embed_flops = 2 * S * d_model

    # LM head
    lm_head_flops = 2 * S * d_model * vocab_size

    return embed_flops + total_layers + lm_head_flops


def count_params_jax(params) -> int:
    """Count total parameters in JAX model.

    Args:
        params: FrozenDict of model parameters

    Returns:
        Total parameter count
    """
    def count_leaves(tree):
        return sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree))

    if HAS_JAX:
        return count_leaves(params)
    else:
        # Fallback: manual counting
        total = 0
        def _count(d):
            nonlocal total
            if isinstance(d, dict):
                for v in d.values():
                    _count(v)
            elif hasattr(d, 'shape'):
                total += np.prod(d.shape)
        _count(params)
        return total


def convert_to_serializable(obj):
    """
    Convert numpy/JAX types to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif HAS_JAX and hasattr(obj, 'device_buffer'):
        # JAX DeviceArray
        return np.asarray(obj).tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj

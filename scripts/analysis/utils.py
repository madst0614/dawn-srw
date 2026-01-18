"""
DAWN Analysis Utilities
========================
Common utility functions and constants for DAWN model analysis.

This module provides:
- Model loading utilities
- Router/neuron extraction helpers
- DataLoader creation
- Mathematical utilities (Gini, entropy)
- Serialization helpers
- Configuration constants for neuron types and routing keys
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

# Import analysis utilities from version_registry (centralized model handling)
from models.version_registry import (
    get_router as get_global_router,  # Gets GlobalRouters (for store_pref_tensors)
    enable_analysis_mode,
    disable_analysis_mode,
    analysis_context,
    forward_for_analysis,
    get_model_version,
    is_v18_plus,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# DAWN Neuron Types Configuration
# ============================================================

NEURON_TYPES = {
    # (display_name, ema_attr, n_attr, color)
    'feature_qk':   ('F-QK',   'usage_ema_feature_qk',   'n_feature_qk',   'red'),
    'feature_v':    ('F-V',    'usage_ema_feature_v',    'n_feature_v',    'orange'),
    'restore_qk':   ('R-QK',   'usage_ema_restore_qk',   'n_restore_qk',   'blue'),
    'restore_v':    ('R-V',    'usage_ema_restore_v',    'n_restore_v',    'green'),
    'feature_know': ('F-Know', 'usage_ema_feature_know', 'n_feature_know', 'purple'),
    'restore_know': ('R-Know', 'usage_ema_restore_know', 'n_restore_know', 'cyan'),
}

# v18.x: Separate Q/K EMA tracking
NEURON_TYPES_V18 = {
    # (display_name, ema_attr, n_attr, color)
    'feature_q':    ('F-Q',    'usage_ema_feature_q',    'n_feature_qk',   'red'),
    'feature_k':    ('F-K',    'usage_ema_feature_k',    'n_feature_qk',   'darkred'),
    'feature_v':    ('F-V',    'usage_ema_feature_v',    'n_feature_v',    'orange'),
    'restore_q':    ('R-Q',    'usage_ema_restore_q',    'n_restore_qk',   'blue'),
    'restore_k':    ('R-K',    'usage_ema_restore_k',    'n_restore_qk',   'darkblue'),
    'restore_v':    ('R-V',    'usage_ema_restore_v',    'n_restore_v',    'green'),
    'feature_know': ('F-Know', 'usage_ema_feature_know', 'n_feature_know', 'purple'),
    'restore_know': ('R-Know', 'usage_ema_restore_know', 'n_restore_know', 'cyan'),
}

# v18.x: Embedding pool boundaries (6 unique pools, not 8 types)
# Q/K share the same embedding pool
EMBEDDING_POOLS_V18 = {
    # (display_name, n_attr, color)
    'feature_qk':   ('FQK',    'n_feature_qk',   'red'),
    'feature_v':    ('FV',     'n_feature_v',    'orange'),
    'restore_qk':   ('RQK',    'n_restore_qk',   'blue'),
    'restore_v':    ('RV',     'n_restore_v',    'green'),
    'feature_know': ('F-Know', 'n_feature_know', 'purple'),
    'restore_know': ('R-Know', 'n_restore_know', 'cyan'),
}

ROUTING_KEYS = {
    # (display_name, pref_key, weight_key, pool_type)
    'fqk_q': ('F-QK_Q', 'fqk_q_pref', 'fqk_weights_Q', 'feature_qk'),
    'fqk_k': ('F-QK_K', 'fqk_k_pref', 'fqk_weights_K', 'feature_qk'),
    'fv':    ('F-V',    'fv_pref',    'fv_weights',    'feature_v'),
    'rqk_q': ('R-QK_Q', 'rqk_q_pref', 'rqk_weights_Q', 'restore_qk'),
    'rqk_k': ('R-QK_K', 'rqk_k_pref', 'rqk_weights_K', 'restore_qk'),
    'rv':    ('R-V',    'rv_pref',    'rv_weights',    'restore_v'),
}

# Knowledge routing keys (separate from attention)
KNOWLEDGE_ROUTING_KEYS = {
    # (display_name, weight_key, pool_type)
    'fknow': ('F-Know', 'feature_know_w', 'feature_know'),
    'rknow': ('R-Know', 'restore_know_w', 'restore_know'),
}

# All routing keys combined
ALL_ROUTING_KEYS = {**ROUTING_KEYS}
for k, v in KNOWLEDGE_ROUTING_KEYS.items():
    ALL_ROUTING_KEYS[k] = (v[0], None, v[1], v[2])  # (display, None, weight_key, pool_type)

# Q/K pool keys (for overlap analysis)
QK_POOLS = {
    'fqk': ('feature_qk', 'fqk_q', 'fqk_k'),
    'rqk': ('restore_qk', 'rqk_q', 'rqk_k'),
}

# Pool type aliases for CLI convenience
# Maps shortcut names to canonical standardized keys
POOL_TYPE_ALIASES = {
    'fqk': 'fqk_q',    # Feature QK (default to Q)
    'rqk': 'rqk_q',    # Restore QK (default to Q)
    'feature_qk': 'fqk_q',
    'feature_v': 'fv',
    'restore_qk': 'rqk_q',
    'restore_v': 'rv',
    'feature_know': 'fknow',
    'restore_know': 'rknow',
}


def resolve_pool_type(pool_type: str) -> str:
    """Resolve pool type alias to canonical key."""
    return POOL_TYPE_ALIASES.get(pool_type, pool_type)

# Neuron attribute names for weight analysis
NEURON_ATTRS = {
    'feature_qk': 'feature_qk_neurons',
    'feature_v': 'feature_v_neurons',
    'restore_qk': 'restore_qk_neurons',
    'restore_v': 'restore_v_neurons',
    'feature_know': 'feature_know',
    'restore_know': 'restore_know',
}

# Direct pool_type to n_attr mapping (consistent across all model versions)
POOL_N_ATTR = {
    'feature_qk': 'n_feature_qk',
    'feature_v': 'n_feature_v',
    'restore_qk': 'n_restore_qk',
    'restore_v': 'n_restore_v',
    'feature_know': 'n_feature_know',
    'restore_know': 'n_restore_know',
}

# Co-selection pairs for v17.1 analysis
COSELECTION_PAIRS = {
    'fqk_rqk': {
        'name': 'F-QK / R-QK (Q/K Processing)',
        'pool_a': {
            'type': 'feature_qk',
            'display': 'F-QK',
            'pref_key': 'fqk_q_pref',
            'source': 'attention',
            'n_attr': 'n_feature_qk',
            'neuron_attr': 'feature_qk_neurons',
        },
        'pool_b': {
            'type': 'restore_qk',
            'display': 'R-QK',
            'pref_key': 'rqk_q_pref',
            'source': 'attention',
            'n_attr': 'n_restore_qk',
            'neuron_attr': 'restore_qk_neurons',
        },
    },
    'fv_rv': {
        'name': 'F-V / R-V (Value Processing)',
        'pool_a': {
            'type': 'feature_v',
            'display': 'F-V',
            'pref_key': 'fv_pref',
            'source': 'attention',
            'n_attr': 'n_feature_v',
            'neuron_attr': 'feature_v_neurons',
        },
        'pool_b': {
            'type': 'restore_v',
            'display': 'R-V',
            'pref_key': 'rv_pref',
            'source': 'attention',
            'n_attr': 'n_restore_v',
            'neuron_attr': 'restore_v_neurons',
        },
    },
    'fk_rk': {
        'name': 'F-Know / R-Know (Knowledge Processing)',
        'pool_a': {
            'type': 'feature_know',
            'display': 'F-Know',
            'pref_key': 'feature_know_w',
            'source': 'knowledge',
            'n_attr': 'n_feature_know',
            'neuron_attr': 'feature_know',
        },
        'pool_b': {
            'type': 'restore_know',
            'display': 'R-Know',
            'pref_key': 'restore_know_w',
            'source': 'knowledge',
            'n_attr': 'n_restore_know',
            'neuron_attr': 'restore_know',
        },
    },
}

# Q/K routing pools for v17.1
QK_POOLS = {
    'feature_qk': {
        'display': 'F-QK',
        'q_pref': 'fqk_q_pref',
        'k_pref': 'fqk_k_pref',
        'q_weight': 'fqk_weights_Q',
        'k_weight': 'fqk_weights_K',
        'n_attr': 'n_feature_qk',
        'color': 'red',
    },
    'restore_qk': {
        'display': 'R-QK',
        'q_pref': 'rqk_q_pref',
        'k_pref': 'rqk_k_pref',
        'q_weight': 'rqk_weights_Q',
        'k_weight': 'rqk_weights_K',
        'n_attr': 'n_restore_qk',
        'color': 'blue',
    },
}


# ============================================================
# Model Loading Utilities
# ============================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load DAWN v17.1 model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    from models import create_model_by_version
    from transformers import BertTokenizerFast

    path = Path(checkpoint_path)
    if path.is_dir():
        pt_files = list(path.glob('*.pt'))
        for f in pt_files:
            if 'best' in f.name.lower() or 'final' in f.name.lower():
                checkpoint_path = str(f)
                break
        else:
            if pt_files:
                checkpoint_path = str(sorted(pt_files, key=os.path.getmtime)[-1])

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Auto-detect version from checkpoint or config
    # Note: CheckpointManager stores model_version at top-level, not inside config
    version = checkpoint.get('model_version') or config.get('model_version')
    if version is None:
        # Check for version-specific keys (most specific first)
        # v18.3 has confidence-related keys or specific config
        v18_2_keys = ['router.tau_proj.weight', 'router.neuron_router.norm_fqk_Q.weight']
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']

        if all(k in cleaned for k in v18_2_keys):
            # Could be 18.2, 18.3, 18.4 - check config for hints
            cfg_version = config.get('model_version', '')
            if cfg_version.startswith('18.'):
                version = cfg_version
            else:
                version = '18.2'  # Default to 18.2 for v18.x
        elif any(k in cleaned for k in dawn_keys):
            if config.get('learnable_tau', False) or config.get('max_paths'):
                version = '18.2'
            else:
                version = '17.1'
        else:
            version = 'baseline'

    from models import normalize_version
    version = normalize_version(version)
    print(f"Model version: {version}")

    model = create_model_by_version(version, config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer, config


def get_router(model):
    """Get neuron router from model."""
    if hasattr(model, 'router') and hasattr(model.router, 'neuron_router'):
        return model.router.neuron_router
    if hasattr(model, 'global_routers'):
        return model.global_routers.neuron_router
    if hasattr(model, '_orig_mod'):
        return get_router(model._orig_mod)
    return None


def get_neurons(model):
    """Get shared neurons from model."""
    if hasattr(model, 'shared_neurons'):
        return model.shared_neurons
    if hasattr(model, '_orig_mod'):
        return get_neurons(model._orig_mod)
    return None


# Alias for compatibility
get_shared_neurons = get_neurons


# ============================================================
# DataLoader Utilities
# ============================================================

def create_dataloader(data_path: str, tokenizer, batch_size: int = 32, max_samples: int = 10000):
    """
    Create a DataLoader from various data formats.

    Supports:
    - .parquet files
    - .json files
    - .pt pre-tokenized files

    Args:
        data_path: Path to data file
        tokenizer: Tokenizer instance
        batch_size: Batch size for DataLoader
        max_samples: Maximum number of samples to load

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    if data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(data_path)
        texts = df['text'].tolist()[:max_samples]
        dataset = TextDataset(texts, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    elif data_path.endswith('.json'):
        with open(data_path) as f:
            data = json.load(f)
        texts = [d['text'] for d in data[:max_samples]]
        dataset = TextDataset(texts, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
        if isinstance(data, dict):
            input_ids = data.get('input_ids', data.get('tokens'))
        else:
            input_ids = data

        # Handle 1D flat tensor
        if input_ids.dim() == 1:
            seq_len = 512
            n_seqs = input_ids.shape[0] // seq_len
            input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)

        dataset = TensorDataset(input_ids[:max_samples])

        # Wrapper to return dict format
        class TensorDictDataset(Dataset):
            def __init__(self, tensor_dataset):
                self.dataset = tensor_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                input_ids = self.dataset[idx][0]
                return {'input_ids': input_ids}

        return DataLoader(TensorDictDataset(dataset), batch_size=batch_size, shuffle=False)

    else:
        raise ValueError(f"Unsupported data format: {data_path}")


# ============================================================
# Mathematical Utilities
# ============================================================

def gini_coefficient(values: torch.Tensor) -> float:
    """
    Calculate Gini coefficient.

    Args:
        values: Tensor of values

    Returns:
        Gini coefficient (0=equal, 1=unequal)
    """
    values = values.flatten().float()
    if values.sum() == 0:
        return 0.0
    sorted_vals = torch.sort(values)[0]
    n = len(sorted_vals)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    return float(1 - 2 * cumsum.sum() / (n * sorted_vals.sum()) + 1/n)


def calc_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate entropy along dimension.

    Args:
        probs: Probability tensor
        dim: Dimension to calculate entropy over

    Returns:
        Entropy tensor
    """
    probs = probs.clamp(min=1e-8)
    return -torch.sum(probs * torch.log(probs), dim=dim)


def calc_entropy_ratio(probs: torch.Tensor) -> float:
    """
    Calculate entropy as percentage of maximum.

    Args:
        probs: Probability tensor (can be 1D, 2D, or 3D)

    Returns:
        Entropy ratio (0-100) as a scalar float
    """
    if probs.numel() == 0:
        return 0.0

    # Flatten to 2D: [batch, features] then average over batch
    if probs.dim() == 1:
        avg_probs = probs
    elif probs.dim() == 2:
        avg_probs = probs.mean(dim=0)
    else:
        # For 3D+ tensors, flatten all but last dim and average
        avg_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)

    # Calculate entropy of the averaged distribution
    ent = calc_entropy(avg_probs, dim=-1)

    # Ensure ent is a scalar
    if ent.dim() > 0:
        ent = ent.mean()

    max_ent = np.log(probs.shape[-1])
    return float(ent / max_ent * 100) if max_ent > 0 else 0.0


# ============================================================
# Serialization Utilities
# ============================================================

def convert_to_serializable(obj):
    """
    Convert numpy/torch types to JSON-serializable format.

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
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def save_results(results: Dict, output_path: str):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)


# ============================================================
# POS Tagging Utilities
# ============================================================

def simple_pos_tag(token: str) -> str:
    """
    Simple rule-based POS tagging for probing analysis.

    Args:
        token: Token string

    Returns:
        POS tag string
    """
    token = token.lower().strip()
    if token in ['the', 'a', 'an']:
        return 'DET'
    elif token in ['is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'has', 'have', 'had', 'do', 'does', 'did']:
        return 'VERB'
    elif token in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about']:
        return 'PREP'
    elif token in ['and', 'or', 'but', 'so', 'yet']:
        return 'CONJ'
    elif token in ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                   'me', 'him', 'her', 'us', 'them']:
        return 'PRON'
    elif token in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}']:
        return 'PUNCT'
    elif token.isdigit():
        return 'NUM'
    elif token.startswith('[') and token.endswith(']'):
        return 'SPECIAL'
    else:
        return 'OTHER'


# ============================================================
# Batch Processing Utilities
# ============================================================

def get_batch_input_ids(batch, device='cuda'):
    """
    Extract input_ids from various batch formats.

    Args:
        batch: Batch from DataLoader (dict, tuple, or tensor)
        device: Device to move tensor to

    Returns:
        input_ids tensor
    """
    if isinstance(batch, dict):
        return batch['input_ids'].to(device)
    elif isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    else:
        return batch.to(device)


def get_routing_from_outputs(outputs):
    """
    Extract routing info from model outputs.

    Args:
        outputs: Model outputs (tuple or single value)

    Returns:
        List of routing_info dicts or None
    """
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None
    routing_infos = outputs[-1]
    if not routing_infos:
        return None
    return routing_infos


# ============================================================
# Routing Data Extraction Layer (Model-Agnostic Schema)
# ============================================================

# Weight key mapping: standard_key -> raw_key in routing_info
# Aligned with ROUTING_KEYS and KNOWLEDGE_ROUTING_KEYS above
WEIGHT_KEY_MAP = {
    # Attention weights (from ROUTING_KEYS)
    'fqk_q': 'fqk_weights_Q',
    'fqk_k': 'fqk_weights_K',
    'fv': 'fv_weights',
    'rqk_q': 'rqk_weights_Q',
    'rqk_k': 'rqk_weights_K',
    'rv': 'rv_weights',
    # Knowledge weights (from KNOWLEDGE_ROUTING_KEYS)
    'fknow': 'feature_know_w',
    'rknow': 'restore_know_w',
}

# Mask key mapping: standard_key -> raw_key in routing_info
# Binary masks (scores > tau) - for clean neuron selection
MASK_KEY_MAP = {
    # Attention masks
    'fqk_q': 'fqk_mask_Q',
    'fqk_k': 'fqk_mask_K',
    'fv': 'fv_mask',
    'rqk_q': 'rqk_mask_Q',
    'rqk_k': 'rqk_mask_K',
    'rv': 'rv_mask',
    # Knowledge masks
    'fknow': 'feature_know_mask',
    'rknow': 'restore_know_mask',
}

# Inverse mapping for quick lookup
RAW_KEY_TO_STD = {v: k for k, v in WEIGHT_KEY_MAP.items()}


class RoutingDataExtractor:
    """
    Central routing data extraction layer.

    Abstracts model version differences and provides standardized
    routing weight access for all analyzers.

    Usage:
        extractor = RoutingDataExtractor(model)

        with extractor.analysis_context():
            outputs = model(input_ids, return_routing_info=True)
            for layer_data in extractor.iter_layers(outputs):
                weights = layer_data.get_weight('fv')  # Standardized access
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.router = self._get_router()
        self.model_version = self._detect_version()

    def _get_router(self):
        """Get router from model (handles different model structures)."""
        if hasattr(self.model, 'router'):
            router = self.model.router
            if hasattr(router, 'neuron_router'):
                return router.neuron_router
            return router
        return None

    def _detect_version(self) -> str:
        """Detect model version."""
        try:
            return get_model_version(self.model)
        except:
            # Fallback detection
            if hasattr(self.model, 'router') and hasattr(self.model.router, 'max_paths'):
                return '18.x'
            return '17.x'

    def enable_weight_storage(self):
        """Enable weight tensor storage in routing_info."""
        # v17.x style: model.router
        if hasattr(self.model, 'router'):
            router = self.model.router
            if hasattr(router, 'store_pref_tensors'):
                router.store_pref_tensors = True
            if hasattr(router, 'neuron_router') and hasattr(router.neuron_router, 'store_pref_tensors'):
                router.neuron_router.store_pref_tensors = True
        # v18.x style: model.global_routers
        if hasattr(self.model, 'global_routers'):
            global_routers = self.model.global_routers
            if hasattr(global_routers, 'store_pref_tensors'):
                global_routers.store_pref_tensors = True

    def disable_weight_storage(self):
        """Disable weight tensor storage."""
        # v17.x style: model.router
        if hasattr(self.model, 'router'):
            router = self.model.router
            if hasattr(router, 'store_pref_tensors'):
                router.store_pref_tensors = False
            if hasattr(router, 'neuron_router') and hasattr(router.neuron_router, 'store_pref_tensors'):
                router.neuron_router.store_pref_tensors = False
        # v18.x style: model.global_routers
        if hasattr(self.model, 'global_routers'):
            global_routers = self.model.global_routers
            if hasattr(global_routers, 'store_pref_tensors'):
                global_routers.store_pref_tensors = False

    def analysis_context(self):
        """Context manager for analysis (enables/disables weight storage)."""
        return _AnalysisContext(self)

    def extract(self, outputs) -> 'RoutingData':
        """
        Extract standardized routing data from model outputs.

        Args:
            outputs: Model outputs tuple (logits, routing_infos)

        Returns:
            RoutingData object with standardized access
        """
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return RoutingData([])

        routing_infos = outputs[1]
        if not routing_infos:
            return RoutingData([])

        return RoutingData(routing_infos)

    def forward_and_extract(self, input_ids, **kwargs) -> 'RoutingData':
        """
        Run forward pass and extract routing data.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for model forward

        Returns:
            RoutingData object
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, return_routing_info=True, **kwargs)
        return self.extract(outputs)


class _AnalysisContext:
    """Context manager for routing analysis."""

    def __init__(self, extractor: RoutingDataExtractor):
        self.extractor = extractor

    def __enter__(self):
        self.extractor.enable_weight_storage()
        return self.extractor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.extractor.disable_weight_storage()
        return False


class RoutingData:
    """
    Standardized routing data container.

    Provides consistent access to routing weights regardless of model version.
    """

    def __init__(self, routing_infos: list):
        self.routing_infos = routing_infos
        self.n_layers = len(routing_infos)

    def __len__(self):
        return self.n_layers

    def __bool__(self):
        return self.n_layers > 0

    def __iter__(self):
        """Iterate over layers, yielding LayerRoutingData."""
        for layer_idx, layer_info in enumerate(self.routing_infos):
            yield LayerRoutingData(layer_idx, layer_info)

    def get_layer(self, layer_idx: int) -> 'LayerRoutingData':
        """Get routing data for a specific layer."""
        if 0 <= layer_idx < self.n_layers:
            return LayerRoutingData(layer_idx, self.routing_infos[layer_idx])
        return LayerRoutingData(layer_idx, {})


class LayerRoutingData:
    """
    Single layer routing data with standardized access.
    """

    def __init__(self, layer_idx: int, layer_info: dict):
        self.layer_idx = layer_idx
        self.raw = layer_info

        # Extract attention and knowledge sub-dicts
        # Handle both {'attention': {...}} and flat structure
        self.attention = layer_info.get('attention', layer_info)
        self.knowledge = layer_info.get('knowledge', {})

    def get_weight(self, key: str) -> Optional[torch.Tensor]:
        """
        Get weight tensor by standardized key.

        Standard keys (from ROUTING_KEYS):
            'fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow'

        Returns:
            Weight tensor [B, S, N] or [B, N], or None if not found
        """
        # Map standard key to raw key
        raw_key = WEIGHT_KEY_MAP.get(key, key)

        # Try attention first, then knowledge
        result = self.attention.get(raw_key)
        if result is None:
            result = self.knowledge.get(raw_key)

        return result

    def get_mask(self, key: str) -> Optional[torch.Tensor]:
        """
        Get binary mask tensor by standardized key.

        Binary masks indicate which neurons passed the learnable tau threshold.
        These are cleaner than using weights > arbitrary_threshold.

        Standard keys: 'fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow'

        Returns:
            Boolean mask tensor [B, S, N] or [B, N], or None if not found
        """
        raw_key = MASK_KEY_MAP.get(key, key)

        # Try attention first, then knowledge
        result = self.attention.get(raw_key)
        if result is None:
            result = self.knowledge.get(raw_key)

        return result

    def get_all_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get all available attention weights (using ROUTING_KEYS)."""
        weights = {}
        for std_key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']:
            raw_key = WEIGHT_KEY_MAP.get(std_key)
            if raw_key:
                w = self.attention.get(raw_key)
                if w is not None:
                    weights[std_key] = w
        return weights

    def get_all_knowledge_weights(self) -> Dict[str, torch.Tensor]:
        """Get all available knowledge weights (using KNOWLEDGE_ROUTING_KEYS)."""
        weights = {}
        for std_key in ['fknow', 'rknow']:
            raw_key = WEIGHT_KEY_MAP.get(std_key)
            if raw_key:
                w = self.knowledge.get(raw_key)
                if w is not None:
                    weights[std_key] = w
        return weights

    def get_all_weights(self) -> Dict[str, torch.Tensor]:
        """Get all available weights."""
        return {**self.get_all_attention_weights(), **self.get_all_knowledge_weights()}

    def has_weights(self) -> bool:
        """Check if any weights are available."""
        return bool(self.get_all_weights())

    def get_all_attention_masks(self) -> Dict[str, torch.Tensor]:
        """Get all available attention masks (binary, scores > tau)."""
        masks = {}
        for std_key in ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv']:
            raw_key = MASK_KEY_MAP.get(std_key)
            if raw_key:
                m = self.attention.get(raw_key)
                if m is not None:
                    masks[std_key] = m
        return masks

    def get_all_knowledge_masks(self) -> Dict[str, torch.Tensor]:
        """Get all available knowledge masks (binary, scores > tau)."""
        masks = {}
        for std_key in ['fknow', 'rknow']:
            raw_key = MASK_KEY_MAP.get(std_key)
            if raw_key:
                m = self.knowledge.get(raw_key)
                if m is not None:
                    masks[std_key] = m
        return masks

    def get_all_masks(self) -> Dict[str, torch.Tensor]:
        """Get all available binary masks."""
        return {**self.get_all_attention_masks(), **self.get_all_knowledge_masks()}

    def has_masks(self) -> bool:
        """Check if any masks are available."""
        return bool(self.get_all_masks())


# Convenience function for simple extraction
def extract_routing_data(model, input_ids, device='cuda') -> RoutingData:
    """
    One-shot routing data extraction.

    Usage:
        routing = extract_routing_data(model, input_ids)
        for layer in routing:
            fv = layer.get_weight('fv')
    """
    extractor = RoutingDataExtractor(model, device)
    with extractor.analysis_context():
        return extractor.forward_and_extract(input_ids)

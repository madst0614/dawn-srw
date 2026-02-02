"""
DAWN Model Version Registry - Single Source of Truth

Supported Versions:
  v18.5: Context-Aware Restore Routing (restore routing on [h_proj, neuron_context])
  v18.4: Relative Confidence Scaling (confidence = gate / gate_sum)
  v18.3: Confidence-Scaled Soft Gating (confidence = gate / (gate + 1))
  v18.2: ReLU-Masked Learnable Tau (mask = ReLU(scores - tau) > 0, Q/K separated)
  v18.1: Soft Mask + Learnable Tau (use_soft_mask=True, learnable_tau=True)
  v18.0: Fixed Threshold Multi-Path Routing (rank=16, max_paths=4, fixed_tau=0.0, path_min_k=8, path_max_k=16)
  v17.1: Q/K Separate Pool + Knowledge Feature-Restore (default)
  v17.1-tpu: TPU-optimized v17.1 (SSM removed, TPU-friendly computation order)
  v17.2: Feature QK Unified + Restore Q/K Separate
  baseline: Vanilla Transformer for comparison

Analysis Utilities:
  get_router(model) - Get router from any DAWN model version
  enable_analysis_mode(model) - Enable full tensor storage for analysis
  disable_analysis_mode(model) - Disable tensor storage after analysis
  forward_for_analysis(model, input_ids) - Standardized forward call for analysis
  analysis_context(model) - Context manager for analysis mode
"""

from typing import Dict, Any, List
from contextlib import contextmanager
import torch


# =============================================================================
# Analysis Utilities - Centralized model handling for all analysis scripts
# =============================================================================

def get_router(model):
    """
    Get router from model, handling version differences.

    Works for v17.x, v18.x, and future versions.
    """
    # v18.x style: GlobalRouters at model.router
    if hasattr(model, 'router'):
        return model.router

    # v17.x style: router in each layer
    if hasattr(model, 'layers') and len(model.layers) > 0:
        layer = model.layers[0]
        if hasattr(layer, 'router'):
            return layer.router

    return None


def enable_analysis_mode(model):
    """
    Enable full tensor storage for analysis.

    Sets store_pref_tensors=True (for routing tensors) and
    optionally debug_mode=True (for scalar stats).
    """
    router = get_router(model)
    if router:
        if hasattr(router, 'store_pref_tensors'):
            router.store_pref_tensors = True
        if hasattr(router, 'debug_mode'):
            router.debug_mode = True


def disable_analysis_mode(model):
    """
    Disable tensor storage after analysis to avoid memory leaks.
    """
    router = get_router(model)
    if router:
        if hasattr(router, 'store_pref_tensors'):
            router.store_pref_tensors = False
        if hasattr(router, 'debug_mode'):
            router.debug_mode = False


@contextmanager
def analysis_context(model):
    """
    Context manager for analysis mode.

    Usage:
        with analysis_context(model):
            outputs = model(input_ids, return_path_weights=True)
            # ... process outputs
        # automatically disables analysis mode on exit
    """
    enable_analysis_mode(model)
    try:
        yield
    finally:
        disable_analysis_mode(model)


def forward_for_analysis(model, input_ids, **kwargs):
    """
    Standardized forward call for analysis that works across all versions.

    Automatically determines which kwargs to pass based on model version.
    Returns (logits, routing_info) tuple.

    Usage:
        with analysis_context(model):
            outputs = forward_for_analysis(model, input_ids)
    """
    version = getattr(model, '__version__', None)

    # v18.2+ uses return_path_weights=True
    if version in ('18.2', '18.3', '18.4'):
        return model(input_ids, return_path_weights=True, **kwargs)

    # v18.0/v18.1, v17.x, and v17.1-TPU use return_routing_info=True
    return model(input_ids, return_routing_info=True, **kwargs)


def get_model_version(model) -> str:
    """Get model version string."""
    if hasattr(model, '__version__'):
        return model.__version__
    return 'unknown'


def is_v18_plus(model) -> bool:
    """Check if model is v18.x or later."""
    version = get_model_version(model)
    return version.startswith('18.')


# =============================================================================
# Version Registry - Configuration for each model version
# =============================================================================

VERSION_REGISTRY = {
    "18.5": {
        "description": "Context-Aware Restore Routing",
        "aliases": ["185"],
        "module": "model_v18_5",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            "learnable_tau": True,
            "tau_reg_weight": 0.0,
        },
        "display_info": lambda args: [
            f"DAWN v18.5: Context-Aware Restore Routing",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}, path_max_k={args.get('path_max_k', 16)}",
            f"  Feature routing on x, Restore routing on [h_proj, neuron_ctx]",
            f"  F-QK: {args.get('n_feature_qk')} - tau on x",
            f"  F-V: {args.get('n_feature_v')} - tau on x",
            f"  R-QK: {args.get('n_restore_qk')} - context-based routing",
            f"  R-V: {args.get('n_restore_v')} - context-based routing",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')} (context-routed)",
        ],
    },

    "18.4": {
        "description": "Relative Confidence Scaling",
        "aliases": ["184"],
        "module": "model_v18_4",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            "learnable_tau": True,
            "tau_reg_weight": 0.0,
        },
        "display_info": lambda args: [
            f"DAWN v18.4: Relative Confidence Scaling",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}, path_max_k={args.get('path_max_k', 16)}",
            f"  confidence = gate / gate_sum (relative, sum=1)",
            f"  F-QK: {args.get('n_feature_qk')} - Q/K separated tau",
            f"  F-V: {args.get('n_feature_v')}",
            f"  R-QK: {args.get('n_restore_qk')} - Q/K separated tau",
            f"  R-V: {args.get('n_restore_v')}",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')}",
        ],
    },

    "18.3": {
        "description": "Confidence-Scaled Soft Gating",
        "aliases": ["183"],
        "module": "model_v18_3",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            "learnable_tau": True,
        },
        "display_info": lambda args: [
            f"DAWN v18.3: Confidence-Scaled Soft Gating",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}, path_max_k={args.get('path_max_k', 16)}",
            f"  confidence = gate / (gate + 1), smoother gradient flow",
            f"  F-QK: {args.get('n_feature_qk')} - Q/K separated tau",
            f"  F-V: {args.get('n_feature_v')}",
            f"  R-QK: {args.get('n_restore_qk')} - Q/K separated tau",
            f"  R-V: {args.get('n_restore_v')}",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')}",
        ],
    },

    "18.2": {
        "description": "ReLU-Masked Learnable Tau (Q/K separated)",
        "aliases": ["182"],
        "module": "model_v18_2",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            "learnable_tau": True,
        },
        "display_info": lambda args: [
            f"DAWN v18.2: ReLU-Masked Learnable Tau",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}, path_max_k={args.get('path_max_k', 16)}",
            f"  fixed_tau={args.get('fixed_tau', 0.0)}, learnable_tau={args.get('learnable_tau', True)}",
            f"  F-QK: {args.get('n_feature_qk')} - Q/K separated tau",
            f"  F-V: {args.get('n_feature_v')}",
            f"  R-QK: {args.get('n_restore_qk')} - Q/K separated tau",
            f"  R-V: {args.get('n_restore_v')}",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')}",
        ],
    },

    "18.1": {
        "description": "Soft Mask + Learnable Tau",
        "aliases": ["181"],
        "module": "model_v18_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_min_k": 8,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            # v18.1 specific
            "use_soft_mask": True,
            "learnable_tau": True,
            "soft_mask_temp": 1.0,
            "soft_mask_penalty": 10.0,
        },
        "display_info": lambda args: [
            f"DAWN v18.1: Soft Mask + Learnable Tau",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}",
            f"  fixed_tau={args.get('fixed_tau', 0.0)}, path_min_k={args.get('path_min_k', 8)}, path_max_k={args.get('path_max_k', 16)}",
            f"  soft_mask_temp={args.get('soft_mask_temp', 1.0)}, soft_mask_penalty={args.get('soft_mask_penalty', 100.0)}",
            f"  F-QK: {args.get('n_feature_qk')} - Q/K shared pool",
            f"  F-V: {args.get('n_feature_v')}",
            f"  R-QK: {args.get('n_restore_qk')} - Q/K shared pool",
            f"  R-V: {args.get('n_restore_v')}",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')}",
        ],
    },

    "18.0": {
        "description": "Fixed Threshold Multi-Path Routing",
        "aliases": ["18", "180"],
        "module": "model_v18",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "max_paths": 4,
            "fixed_tau": 0.0,
            "path_min_k": 8,
            "path_max_k": 16,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
        },
        "display_info": lambda args: [
            f"DAWN v18.0: Fixed Threshold Multi-Path Routing",
            f"  rank={args.get('rank', 16)}, max_paths={args.get('max_paths', 4)}",
            f"  fixed_tau={args.get('fixed_tau', 0.0)}, path_min_k={args.get('path_min_k', 8)}, path_max_k={args.get('path_max_k', 16)}",
            f"  F-QK: {args.get('n_feature_qk')} - Q/K shared pool",
            f"  F-V: {args.get('n_feature_v')}",
            f"  R-QK: {args.get('n_restore_qk')} - Q/K shared pool",
            f"  R-V: {args.get('n_restore_v')}",
            f"  F-Know: {args.get('n_feature_know')}, R-Know: {args.get('n_restore_know')}",
        ],
    },

    "17.2": {
        "description": "Feature QK Unified + Restore Q/K Separate",
        "aliases": ["172"],
        "module": "model_v17_2",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_qk": 8,
            "top_k_feature_v": 3,
            "top_k_restore_qk": 8,
            "top_k_restore_v": 3,
            "top_k_feature_know": 4,
            "top_k_restore_know": 4,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
        },
        "display_info": lambda args: [
            f"DAWN v17.2: Feature QK Unified + Restore Q/K Separate",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  F-QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - unified Q/K",
            f"  F-V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  R-QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - separate Q/K",
            f"  R-V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  F-Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), R-Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "17.1": {
        "description": "Q/K Shared Pool + Knowledge Feature-Restore",
        "aliases": ["17", "171", "17.0"],
        "module": "model_v17_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_qk": 8,
            "top_k_feature_v": 3,
            "top_k_restore_qk": 8,
            "top_k_restore_v": 3,
            "top_k_feature_know": 4,
            "top_k_restore_know": 4,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "attention_token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
        },
        "display_info": lambda args: [
            f"DAWN v17.1: Q/K Separate + Knowledge Feature-Restore",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  F-QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - Q/K shared pool",
            f"  F-V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  R-QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - Q/K shared pool",
            f"  R-V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  F-Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), R-Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "17.1-tpu": {
        "description": "Q/K Shared Pool + Knowledge Feature-Restore (TPU-optimized, SSM removed)",
        "aliases": ["171tpu", "17.1tpu", "tpu"],
        "module": "model_v17_1_tpu",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "top_k_feature_qk": 8,
            "top_k_feature_v": 3,
            "top_k_restore_qk": 8,
            "top_k_restore_v": 3,
            "top_k_feature_know": 4,
            "top_k_restore_know": 4,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
        },
        "display_info": lambda args: [
            f"DAWN v17.1-TPU: Q/K Separate + Knowledge Feature-Restore (TPU-optimized)",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  SSM removed, token-level routing only",
            f"  TPU-optimized: pass through all neurons first, then weighted sum",
            f"  F-QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - Q/K shared pool",
            f"  F-V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  R-QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - Q/K shared pool",
            f"  R-V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  F-Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), R-Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "baseline": {
        "description": "Vanilla Transformer Baseline",
        "aliases": ["vanilla", "base"],
        "module": "baseline_transformer",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
        ],
        "optional_params": {
            "d_ff": None,
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"Vanilla Transformer Baseline",
            f"  d_model={args.get('d_model')}, n_layers={args.get('n_layers')}, n_heads={args.get('n_heads')}",
            f"  d_ff={args.get('d_ff', args.get('d_model', 256) * 4)}",
        ],
    },
}


def normalize_version(version: str) -> str:
    """Normalize version string to canonical form."""
    version = str(version)

    if version in VERSION_REGISTRY:
        return version

    for canonical, info in VERSION_REGISTRY.items():
        if version in info.get('aliases', []):
            return canonical

    raise ValueError(f"Unknown version: {version}. Supported: {', '.join(VERSION_REGISTRY.keys())}")


def get_version_info(version: str) -> Dict[str, Any]:
    """Get version info from registry."""
    canonical = normalize_version(version)
    return VERSION_REGISTRY[canonical]


def get_required_params(version: str) -> List[str]:
    """Get list of required parameters for a version."""
    info = get_version_info(version)
    return info.get('required_params', [])


def get_optional_params(version: str) -> Dict[str, Any]:
    """Get optional parameters with defaults for a version."""
    info = get_version_info(version)
    return info.get('optional_params', {})


def build_model_kwargs(version: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build model kwargs from config."""
    version = normalize_version(version)
    info = VERSION_REGISTRY[version]

    kwargs = {}

    for param in info.get('required_params', []):
        if param in config:
            kwargs[param] = config[param]

    for param, default in info.get('optional_params', {}).items():
        kwargs[param] = config.get(param, default)

    return kwargs


def load_model_params_to_args(args, config: Dict[str, Any]) -> None:
    """Load model params from config dict to args object."""
    all_params = {}

    for version_info in VERSION_REGISTRY.values():
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    special_mappings = {
        'rank': 'basis_rank',
    }

    for param, default in all_params.items():
        if param in ('vocab_size',):
            continue

        args_attr = special_mappings.get(param, param)
        current_value = getattr(args, args_attr, default)
        value = config.get(param, current_value)
        setattr(args, args_attr, value)

        if args_attr == 'basis_rank':
            setattr(args, 'rank', value)


def build_args_config(args, vocab_size: int) -> Dict[str, Any]:
    """Build config dict from args object."""
    all_params = {}

    for version_info in VERSION_REGISTRY.values():
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    config = {'vocab_size': vocab_size}

    special_mappings = {
        'basis_rank': 'rank',
    }

    for param, default in all_params.items():
        args_attr = None
        for args_name, config_name in special_mappings.items():
            if config_name == param:
                args_attr = args_name
                break

        if args_attr is None:
            args_attr = param

        if hasattr(args, args_attr):
            value = getattr(args, args_attr)
            if value is None and default is not None:
                value = default
            config[param] = value
        elif hasattr(args, param):
            value = getattr(args, param)
            if value is None and default is not None:
                value = default
            config[param] = value
        elif default is not None:
            config[param] = default

    return config


def print_version_info(version: str, args: Dict[str, Any]) -> None:
    """Print version-specific architecture information."""
    version = normalize_version(version)
    info = VERSION_REGISTRY.get(version, {})

    print(f"Model version: {version}")
    print(f"Description: {info.get('description', 'N/A')}")

    display_fn = info.get('display_info')
    if display_fn:
        lines = display_fn(args)
        for line in lines:
            print(line)


def list_versions() -> List[str]:
    """Get list of all available versions."""
    return list(VERSION_REGISTRY.keys())


def get_all_versions_info() -> str:
    """Get formatted string of all versions and descriptions."""
    lines = ["Available DAWN versions:"]
    for version, info in VERSION_REGISTRY.items():
        aliases = info.get('aliases', [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"  v{version}: {info['description']}{alias_str}")
    return "\n".join(lines)


def get_routing_log_info(routing_infos, calc_entropy_fn, calc_var_fn) -> Dict[str, Any]:
    """
    Extract routing log info for v17.1/v17.2/v18.0/v18.1/v18.2/v18.3.
    Computes AVERAGE entropy across all layers.
    """
    if isinstance(routing_infos, dict):
        routing_infos = [routing_infos]

    attn0 = routing_infos[0].get('attention', routing_infos[0])

    # v18.0/v18.1: Fixed Threshold Multi-Path (has n_paths_* keys)
    # Note: v18 doesn't store _pref tensors to avoid memory leaks
    # Uses selected neuron counts and score distribution instead of entropy/variance
    if attn0.get('n_paths_fqk_Q') is not None:
        know0 = routing_infos[0].get('knowledge', {})

        # Selected neurons per token (more informative than activation ratio)
        sel_fqk_Q = attn0.get('selected_fqk_Q', 0)
        sel_fqk_K = attn0.get('selected_fqk_K', 0)
        sel_fv = attn0.get('selected_fv', 0)
        sel_rqk_Q = attn0.get('selected_rqk_Q', 0)
        sel_rqk_K = attn0.get('selected_rqk_K', 0)
        sel_rv = attn0.get('selected_rv', 0)
        sel_kf = know0.get('selected_feature', 0)
        sel_kr = know0.get('selected_restore', 0)

        # Format selected neurons string (includes knowledge)
        sel_str = f"Sel FQK:{sel_fqk_Q:.0f}/{sel_fqk_K:.0f} FV:{sel_fv:.0f} RQK:{sel_rqk_Q:.0f}/{sel_rqk_K:.0f} RV:{sel_rv:.0f} K:{sel_kf:.0f}/{sel_kr:.0f}"

        # Q/K overlap ratio
        overlap_fqk = attn0.get('overlap_fqk', 0)
        overlap_rqk = attn0.get('overlap_rqk', 0)
        overlap_str = f"Q/K Overlap FQK/RQK:{overlap_fqk:.2f}/{overlap_rqk:.2f}"

        # v18 specific: path counts
        n_paths = f"Paths FQK:{attn0.get('n_paths_fqk_Q', 0):.1f}/{attn0.get('n_paths_fqk_K', 0):.1f} FV:{attn0.get('n_paths_fv', 0):.1f} RQK:{attn0.get('n_paths_rqk_Q', 0):.1f}/{attn0.get('n_paths_rqk_K', 0):.1f} RV:{attn0.get('n_paths_rv', 0):.1f}"

        # Detect v18.1/v18.2/v18.3 by flag and key presence
        # v18.3: confidence-scaled soft gating (same routing_info format as v18.2)
        # v18.2: has adj_* keys (ReLU mask)
        # v18.1: has gate_* keys (sigmoid gate)
        # v18.0: neither (fixed tau)
        is_soft_mask = attn0.get('use_soft_mask', False) or attn0.get('learnable_tau', False)
        if is_soft_mask:
            # v18.2/v18.3 uses adj_*, v18.1 uses gate_*
            # Note: v18.3 has same routing_info format as v18.2
            version = '18.2+' if attn0.get('adj_fq') is not None else '18.1'
        else:
            version = '18.0'

        return {
            'ent_str': sel_str,  # Use selected neurons instead of entropy for v18
            'var_str': None,  # No variance tracking for v18
            'overlap_str': overlap_str,  # Q/K overlap
            'paths_str': n_paths,
            'tau_str': None,  # tau logged separately in train.py for v18.1
            'version': version
        }

    # v17.2: Feature QK Unified (fqk_pref exists, fqk_q_pref does not)
    elif attn0.get('fqk_pref') is not None:
        all_ents = {k: [] for k in ['FQK', 'FV', 'RQK_Q', 'RQK_K', 'RV']}
        all_vars = {k: [] for k in ['FQK', 'FV', 'RQK_Q', 'RQK_K', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQK': attn.get('fqk_pref'),
                'FV': attn.get('fv_pref'),
                'RQK_Q': attn.get('rqk_q_pref'),
                'RQK_K': attn.get('rqk_k_pref'),
                'RV': attn.get('rv_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent F-QK/F-V/R-QK_Q/K/R-V:{ents['FQK']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_RQK_Q = attn0.get('rqk_weights_Q')
        w_RQK_K = attn0.get('rqk_weights_K')
        overlap_RQK = calc_overlap(w_RQK_Q, w_RQK_K)
        overlap_str = f"Q/K Overlap R-QK:{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '17.2'}

    # v17.1: Shared Pool + Separate Routing (fqk_q_pref exists)
    elif attn0.get('fqk_q_pref') is not None:
        all_ents = {k: [] for k in ['FQK_Q', 'FQK_K', 'FV', 'RQK_Q', 'RQK_K', 'RV']}
        all_vars = {k: [] for k in ['FQK_Q', 'FQK_K', 'FV', 'RQK_Q', 'RQK_K', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQK_Q': attn.get('fqk_q_pref'),
                'FQK_K': attn.get('fqk_k_pref'),
                'FV': attn.get('fv_pref'),
                'RQK_Q': attn.get('rqk_q_pref'),
                'RQK_K': attn.get('rqk_k_pref'),
                'RV': attn.get('rv_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent F-QK_Q/K/F-V/R-QK_Q/K/R-V:{ents['FQK_Q']:.0f}/{ents['FQK_K']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK_Q']:.4f}/{vars_['FQK_K']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_FQK_Q = attn0.get('fqk_weights_Q')
        w_FQK_K = attn0.get('fqk_weights_K')
        w_RQK_Q = attn0.get('rqk_weights_Q')
        w_RQK_K = attn0.get('rqk_weights_K')
        overlap_FQK = calc_overlap(w_FQK_Q, w_FQK_K)
        overlap_RQK = calc_overlap(w_RQK_Q, w_RQK_K)
        overlap_str = f"Q/K Overlap F-QK/R-QK:{overlap_FQK:.2f}/{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '17.1'}

    else:
        return {'ent_str': "Ent: N/A", 'var_str': "TokVar: N/A", 'overlap_str': None, 'version': 'unknown'}

"""
DAWN Models Module

v18.5: Context-Aware Restore Routing
- Feature routing: done on input x (same as v18.4)
- Restore routing: done on [h_proj, neuron_context]
- h_proj: intermediate representation h projected to d_space
- neuron_context: weighted sum of feature neuron embeddings

v18.4: Relative Confidence Scaling
- Gate: gate = ReLU(scores - tau), can be 0
- Confidence: confidence = gate / gate_sum (relative, sums to 1)
- tau decoupled from lb_loss, healthy gradient flow

v18.3: Confidence-Scaled Soft Gating
- Gate: gate = ReLU(scores - tau), can be 0
- Confidence: confidence = gate / (gate + 1)
- Scaled weights: weights * confidence
- Smoother gradient flow through confidence

v18.2: ReLU-Masked Learnable Tau (Q/K separated)
- ReLU mask: mask = (ReLU(scores - tau) > 0)
- Learnable tau: token-level tau via nn.Linear(d_model, 8) - Q/K separated
- Hard masking with differentiable tau learning

v18.1: Soft Mask + Token-level Learnable Tau
- Soft mask: sigmoid((score - tau) / temp) instead of hard threshold
- Learnable tau: token-level tau via nn.Linear(d_model, 8)
- Gate-based scoring: weights = softmax(scores * gate)

v18.0: Fixed Threshold Multi-Path Routing
- Fixed threshold + masked softmax for neuron selection
- Minimum/maximum neuron guarantees (path_min_k, path_max_k * max_paths)
- Multi-path (1~max_paths) parallel processing with path-wise Q,K,V aggregation
- rank=16, max_paths=4, fixed_tau=0.0, path_min_k=8, path_max_k=16

v17.1: Q/K Separate Pool + Knowledge Feature-Restore (default)
- Attention: Q/K separate pools (Feature_Q/K/V, Restore_Q/K/V)
- Knowledge: Feature-Restore pattern (Feature_Know, Restore_Know)
- 8 neuron pools for fine-grained routing

v17.2: Feature QK Unified + Restore Q/K Separate
- Feature stage: Q/K share single routing
- Restore stage: Q/K have separate routing
- Knowledge: Feature-Restore pattern

baseline: Vanilla Transformer for fair comparison
"""

# Lazy imports — PyTorch models loaded only when accessed (allows torch-free JAX usage)
def __getattr__(name):
    _torch_models = {
        'DAWN_v18_5': ('.model_v18_5', 'DAWN'),
        'DAWN_v18_4': ('.model_v18_4', 'DAWN'),
        'DAWN_v18_3': ('.model_v18_3', 'DAWN'),
        'DAWN_v18_2': ('.model_v18_2', 'DAWN'),
        'DAWN_v18_1': ('.model_v18_1', 'DAWN'),
        'DAWN_v18': ('.model_v18', 'DAWN'),
        'DAWN_v17_1': ('.model_v17_1', 'DAWN'),
        'DAWN_v17_1_TPU': ('.model_v17_1_tpu', 'DAWN'),
        'DAWN_v17_1_TPU_MemOpt': ('.model_v17_1_tpu_memopt', 'DAWN'),
        'DAWN_v17_2': ('.model_v17_2', 'DAWN'),
        'DAWN_Spatial': ('.dawn_spatial', 'DAWN'),
        'DAWN': ('.model_v17_1', 'DAWN'),
        'VanillaTransformer': ('.baseline_transformer', 'VanillaTransformer'),
    }
    _registry_names = {
        'VERSION_REGISTRY', 'normalize_version', 'get_version_info',
        'get_required_params', 'get_optional_params', 'build_model_kwargs',
        'build_args_config', 'load_model_params_to_args', 'print_version_info',
        'list_versions', 'get_all_versions_info', 'get_routing_log_info',
        'get_router', 'enable_analysis_mode', 'disable_analysis_mode',
        'analysis_context', 'forward_for_analysis', 'get_model_version',
        'is_v18_plus',
    }

    if name in _torch_models:
        import importlib
        module_path, attr = _torch_models[name]
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)

    if name in _registry_names:
        from . import version_registry
        return getattr(version_registry, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Models
    'DAWN',
    'DAWN_v18_5',
    'DAWN_v18_4',
    'DAWN_v18_3',
    'DAWN_v18_2',
    'DAWN_v18_1',
    'DAWN_v18',
    'DAWN_v17_2',
    'DAWN_v17_1',
    'DAWN_v17_1_TPU',
    'DAWN_v17_1_TPU_MemOpt',
    'DAWN_Spatial',
    'VanillaTransformer',
    # Version utilities
    'VERSION_REGISTRY',
    'normalize_version',
    'get_version_info',
    'get_required_params',
    'get_optional_params',
    'build_model_kwargs',
    'build_args_config',
    'load_model_params_to_args',
    'print_version_info',
    'list_versions',
    'get_all_versions_info',
    'get_routing_log_info',
    'create_model_by_version',
    # Analysis utilities
    'get_router',
    'enable_analysis_mode',
    'disable_analysis_mode',
    'analysis_context',
    'forward_for_analysis',
    'get_model_version',
    'is_v18_plus',
]

__version__ = "17.1"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "18.5", "18.4", "18.3", "18.2", "18.1", "18.0", "17.2", "17.1", "17.1-tpu", "17.1-tpu-memopt", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "18.5":
        return DAWN_v18_5(**config)
    elif version == "18.4":
        return DAWN_v18_4(**config)
    elif version == "18.3":
        return DAWN_v18_3(**config)
    elif version == "18.2":
        return DAWN_v18_2(**config)
    elif version == "18.1":
        return DAWN_v18_1(**config)
    elif version == "18.0":
        return DAWN_v18(**config)
    elif version == "17.2":
        return DAWN_v17_2(**config)
    elif version == "17.1":
        return DAWN_v17_1(**config)
    elif version == "17.1-tpu":
        return DAWN_v17_1_TPU(**config)
    elif version == "17.1-tpu-memopt":
        return DAWN_v17_1_TPU_MemOpt(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 18.5, 18.4, 18.3, 18.2, 18.1, 18.0, 17.2, 17.1, 17.1-tpu, 17.1-tpu-memopt, baseline")

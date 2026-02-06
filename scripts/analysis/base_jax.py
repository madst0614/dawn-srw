"""
Base Analyzer - JAX Version
============================
Base class for all DAWN analysis modules (JAX/Flax).

JAX/Flax compatible version for TPU analysis.

Provides common functionality:
- Model/params access
- Version detection (v17/v18)
- Analysis loop abstraction
- Neuron pool configuration

NOTE: This is a JAX port of base.py. PyTorch-specific features
are replaced with JAX/numpy equivalents.
"""

import numpy as np
from typing import Dict, Callable, Optional, Any
from abc import ABC, abstractmethod

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from .utils_jax import (
    NEURON_TYPES, NEURON_TYPES_V18, EMBEDDING_POOLS_V18,
    POOL_N_ATTR,
    create_batches,
    create_model_from_config,
    get_neuron_embeddings_jax,
    JAXRoutingDataExtractor,
    HAS_JAX,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x


class BaseAnalyzerJAX(ABC):
    """
    Base class for DAWN analyzers (JAX version).

    Provides common infrastructure for:
    - Model and params access
    - Version detection
    - Analysis loop
    - Neuron type configuration
    """

    def __init__(
        self,
        model,
        params,
        config: Dict,
        tokenizer=None,
    ):
        """
        Initialize analyzer.

        Args:
            model: DAWN JAX model class (not instance)
            params: FrozenDict of model parameters
            config: Model configuration dict
            tokenizer: Tokenizer instance (optional)
        """
        self.model = model
        self.params = params
        self.config = config
        self.tokenizer = tokenizer

        # Create model instance for apply()
        self.model_instance = create_model_from_config(config)

        # Create routing extractor
        self.extractor = JAXRoutingDataExtractor(self.model_instance, params, config)

        # Extract neuron counts from config
        self.n_feature_qk = config.get('n_feature_qk', 0)
        self.n_feature_v = config.get('n_feature_v', 0)
        self.n_restore_qk = config.get('n_restore_qk', 0)
        self.n_restore_v = config.get('n_restore_v', 0)
        self.n_feature_know = config.get('n_feature_know', 0)
        self.n_restore_know = config.get('n_restore_know', 0)

    @property
    def is_v18(self) -> bool:
        """Check if model is v18.x (has max_paths for multi-path routing)."""
        return self.config.get('max_paths', 1) > 1

    def get_neuron_types(self) -> Dict:
        """Get neuron type configuration for current model version."""
        return NEURON_TYPES_V18 if self.is_v18 else NEURON_TYPES

    def get_embedding_pools(self) -> Dict:
        """Get embedding pool configuration (6 unique pools for v18)."""
        if self.is_v18:
            return EMBEDDING_POOLS_V18
        # For non-v18, use NEURON_TYPES format (display, n_attr, color)
        return {k: v for k, v in NEURON_TYPES.items()}

    def get_pool_n(self, pool_type: str) -> int:
        """Get number of neurons for a pool type.

        Args:
            pool_type: Pool type key (e.g., 'feature_qk', 'restore_v')

        Returns:
            Number of neurons in the pool
        """
        n_attr = POOL_N_ATTR.get(pool_type)
        if n_attr is None:
            return 0
        return self.config.get(n_attr.replace('n_', ''), 0)

    def get_neuron_embedding(self) -> Optional[np.ndarray]:
        """Get normalized neuron embeddings."""
        return get_neuron_embeddings_jax(self.params)

    def run_analysis_loop(
        self,
        val_tokens: np.ndarray,
        n_batches: int,
        process_batch_fn: Callable[[np.ndarray, Dict], None],
        batch_size: int = 32,
        seq_len: int = 512,
        desc: str = "Analyzing"
    ):
        """
        Run common analysis loop over data.

        Args:
            val_tokens: Flat array of validation tokens
            n_batches: Number of batches to process
            process_batch_fn: Function(input_ids, routing_data) to process each batch
            batch_size: Batch size
            seq_len: Sequence length
            desc: Progress bar description
        """
        if not HAS_JAX:
            raise RuntimeError("JAX not available")

        batches = create_batches(val_tokens, batch_size, seq_len)

        if n_batches:
            batches = batches[:n_batches]

        iterator = tqdm(batches, total=len(batches), desc=desc) if HAS_TQDM else batches

        rng_key = jax.random.PRNGKey(42)

        for batch in iterator:
            input_ids = jnp.array(batch)

            # Extract routing data
            routing_data = self.extractor.extract_routing(np.array(input_ids))

            # Process batch
            process_batch_fn(np.array(input_ids), routing_data)

    def forward(self, input_ids: np.ndarray, deterministic: bool = True) -> Dict:
        """
        Run forward pass on model.

        Args:
            input_ids: Input token IDs [B, S]
            deterministic: Whether to disable dropout

        Returns:
            Model output dictionary
        """
        if not HAS_JAX:
            raise RuntimeError("JAX not available")

        rng_key = jax.random.PRNGKey(0)

        return self.model_instance.apply(
            self.params,
            jnp.array(input_ids),
            deterministic=deterministic,
            rngs={'dropout': rng_key}
        )

    def get_neuron_count(self, neuron_type: str) -> int:
        """Get number of neurons for a given type."""
        neuron_types = self.get_neuron_types()
        if neuron_type not in neuron_types:
            return 0
        _, n_attr, _ = neuron_types[neuron_type]
        return self.config.get(n_attr.replace('n_', ''), 0)

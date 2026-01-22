"""
Base Analyzer
=============
Base class for all DAWN analysis modules.

Provides common functionality:
- Model/router access
- Version detection (v17/v18)
- Analysis loop abstraction
- Pref tensor management
"""

import torch
from typing import Dict, Callable, Optional, Any
from abc import ABC, abstractmethod

from .utils import (
    NEURON_TYPES, NEURON_TYPES_V18, EMBEDDING_POOLS_V18,
    get_router, get_neurons,
    get_batch_input_ids, get_routing_from_outputs,
    HAS_TQDM, tqdm,
)


class BaseAnalyzer(ABC):
    """
    Base class for DAWN analyzers.

    Provides common infrastructure for:
    - Model and router access
    - Version detection
    - Analysis loop with pref tensor management
    - Neuron type configuration
    """

    def __init__(
        self,
        model,
        router=None,
        shared_neurons=None,
        tokenizer=None,
        device: str = 'cuda'
    ):
        """
        Initialize analyzer.

        Args:
            model: DAWN model instance
            router: NeuronRouter (auto-detected if None)
            shared_neurons: SharedNeurons (auto-detected if None)
            tokenizer: Tokenizer instance (optional)
            device: Device for computation
        """
        self.model = model
        self.router = router or get_router(model)
        self.shared_neurons = shared_neurons or get_neurons(model)
        self.tokenizer = tokenizer
        self.device = device

        # Validate router
        if self.router is None:
            raise ValueError("Could not find router in model")

    @property
    def is_v18(self) -> bool:
        """Check if model is v18.x (has max_paths for multi-path routing)."""
        gr = self.global_routers
        return gr is not None and hasattr(gr, 'max_paths')

    @property
    def global_routers(self):
        """Get GlobalRouters instance for store_pref_tensors."""
        if hasattr(self.model, 'router'):
            return self.model.router
        if hasattr(self.model, 'global_routers'):
            return self.model.global_routers
        if hasattr(self.model, '_orig_mod'):
            if hasattr(self.model._orig_mod, 'router'):
                return self.model._orig_mod.router
            if hasattr(self.model._orig_mod, 'global_routers'):
                return self.model._orig_mod.global_routers
        return None

    def get_neuron_types(self) -> Dict:
        """Get neuron type configuration for current model version."""
        return NEURON_TYPES_V18 if self.is_v18 else NEURON_TYPES

    def get_embedding_pools(self) -> Dict:
        """Get embedding pool configuration (6 unique pools for v18)."""
        if self.is_v18:
            return EMBEDDING_POOLS_V18
        # For non-v18, use NEURON_TYPES format (display, n_attr, color)
        return {k: v for k, v in NEURON_TYPES.items()}

    def enable_pref_tensors(self):
        """Enable store_pref_tensors for detailed routing analysis."""
        gr = self.global_routers
        if gr is not None and hasattr(gr, 'store_pref_tensors'):
            gr.store_pref_tensors = True

    def disable_pref_tensors(self):
        """Disable store_pref_tensors."""
        gr = self.global_routers
        if gr is not None and hasattr(gr, 'store_pref_tensors'):
            gr.store_pref_tensors = False

    def run_analysis_loop(
        self,
        dataloader,
        n_batches: int,
        process_batch_fn: Callable[[torch.Tensor, Any], None],
        enable_pref: bool = True,
        desc: str = "Analyzing"
    ):
        """
        Run common analysis loop over data.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            process_batch_fn: Function(input_ids, routing_infos) to process each batch
            enable_pref: Whether to enable store_pref_tensors
            desc: Progress bar description
        """
        self.model.eval()

        if enable_pref:
            self.enable_pref_tensors()

        try:
            with torch.no_grad():
                iterator = tqdm(dataloader, total=n_batches, desc=desc) if HAS_TQDM else dataloader
                for i, batch in enumerate(iterator):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)
                    outputs = self.model(input_ids, return_routing_info=True)
                    routing_infos = get_routing_from_outputs(outputs)

                    if routing_infos is None:
                        continue

                    process_batch_fn(input_ids, routing_infos)
        finally:
            if enable_pref:
                self.disable_pref_tensors()

    def get_neuron_count(self, neuron_type: str) -> int:
        """Get number of neurons for a given type."""
        neuron_types = self.get_neuron_types()
        if neuron_type not in neuron_types:
            return 0
        _, n_attr, _ = neuron_types[neuron_type]
        return getattr(self.router, n_attr, 0)

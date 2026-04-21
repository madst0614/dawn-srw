"""
DAWN v17.1-JAX Training Script (TPU Multi-Device)

JAX/Flax native training for DAWN v17.1 model.
- Multi-device data parallelism via jax.pmap
- Pure numpy/JAX data pipeline (no PyTorch dependency)
- GCS checkpoint support for TPU spot instances
- optax optimizer with warmup + cosine decay
- jax.jit / jax.pmap compiled train/eval steps
- Auto-resume: automatically finds latest checkpoint in config's checkpoint_dir

Usage:
    # Just provide config - auto-resumes if checkpoint exists, otherwise starts fresh
    python scripts/train_jax.py --config configs/train_config_tpu.yaml

    # Force start from scratch (ignores existing checkpoints)
    python scripts/train_jax.py --config configs/train_config_tpu.yaml --from-scratch
"""

import sys
import os
import signal
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather
import optax
import numpy as np
import time
import random
import argparse
import yaml
import numpy as np
from datetime import datetime
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map

from models.model_v17_1_jax import DAWN
from models.dawn_spatial import DAWN as DAWN_Spatial
from models.dawn_spatial_v2 import DAWN as DAWN_SpatialV2
from models.dawn_spatial_v3 import DAWN as DAWN_SpatialV3
from models.dawn_spatial_v3_baseline import DAWN as DAWN_SpatialV3Baseline
from models.dawn_spatial_v3_exp import DAWN as DAWN_SpatialV3Exp
from models.dawn_spatial_v394_exp import DAWN as DAWN_SpatialV394Exp
from models.dawn_spatial_v395_exp import DAWN as DAWN_SpatialV395Exp
from models.dawn_spatial_v396_exp import DAWN as DAWN_SpatialV396Exp
from models.dawn_spatial_v397_exp import DAWN as DAWN_SpatialV397Exp
from models.dawn_spatial_v3971_exp import DAWN as DAWN_SpatialV3971Exp
from models.dawn_spatial_v398_exp import DAWN as DAWN_SpatialV398Exp
from models.dawn_spatial_v3981_exp import DAWN as DAWN_SpatialV3981Exp
from models.dawn_spatial_v399_exp import DAWN as DAWN_SpatialV399Exp
from models.dawn_spatial_v400_exp import DAWN as DAWN_SpatialV400Exp
from models.dawn_spatial_v401_exp import DAWN as DAWN_SpatialV401Exp
from models.dawn_spatial_v402_exp import DAWN as DAWN_RW_V402
from models.dawn_spatial_v403_exp import DAWN as DAWN_SpatialV403Exp
from models.dawn_spatial_v404_exp import DAWN as DAWN_SpatialV404Exp
from models.dawn_spatial_v405_exp import DAWN as DAWN_SpatialV405Exp
from models.dawn_spatial_v406_exp import DAWN as DAWN_SpatialV406Exp
from models.dawn_spatial_v41_exp import DAWN as DAWN_SpatialV41Exp
from models.baseline_transformer_jax import VanillaTransformer

# ============================================================
# Constants
# ============================================================

LOG_INTERVAL = 100


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Config
# ============================================================

def load_config(config_path):
    """Load config from local or GCS path."""
    path_str = str(config_path)
    if path_str.startswith("gs://"):
        with _open_file(path_str, "r") as f:
            return yaml.safe_load(f)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg):
    """Build model from config dict. Supports DAWN and baseline."""
    mcfg = cfg['model']
    version = mcfg.get('model_version', '17.1')

    if version == 'baseline':
        model = VanillaTransformer(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            d_ff=mcfg.get('d_ff', 1536),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            dropout_rate=mcfg.get('dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
        )
    elif version == 'spatial-r1':
        model = DAWN_Spatial(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_space=mcfg.get('d_space', 64),
            n_qk=mcfg.get('n_qk', 256),
            n_v=mcfg.get('n_v', 256),
            n_know=mcfg.get('n_know', 512),
            max_k=mcfg.get('max_k', 32),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            # Hierarchical routing
            n_clusters_qk=mcfg.get('n_clusters_qk', 64),
            n_clusters_v=mcfg.get('n_clusters_v', 64),
            n_clusters_know=mcfg.get('n_clusters_know', 128),
            k_cluster_qk=mcfg.get('k_cluster_qk', 8),
            k_cluster_v=mcfg.get('k_cluster_v', 8),
            k_cluster_know=mcfg.get('k_cluster_know', 8),
        )
    elif version == 'spatial-r1-v4.0.0':
        model = DAWN_SpatialV400Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version in ('spatial-r1-v3.9.9', 'spatial-r1-v4.0.1', 'spatial-r1-v4.0.3',
                      'spatial-r1-v4.0.4', 'spatial-r1-v4.0.5',
                      'spatial-r1-v4.0.6', 'spatial-r1-v4.1'):
        _cls = {
            'spatial-r1-v4.0.1': DAWN_SpatialV401Exp,
            'spatial-r1-v4.0.3': DAWN_SpatialV403Exp,
            'spatial-r1-v4.0.4': DAWN_SpatialV404Exp,
            'spatial-r1-v4.0.5': DAWN_SpatialV405Exp,
            'spatial-r1-v4.0.6': DAWN_SpatialV406Exp,
            'spatial-r1-v4.1': DAWN_SpatialV41Exp,
        }.get(version, DAWN_SpatialV399Exp)
        _extra = {}
        if version == 'spatial-r1-v4.0.4':
            _extra['reverse_p_max'] = cfg['training'].get('reverse_p_max', 0.0)
        elif version == 'spatial-r1-v4.0.5':
            _extra['gate_drop_rate'] = cfg['training'].get('gate_drop_rate', 0.0)
            _extra['gate_boost_rate'] = cfg['training'].get('gate_boost_rate', 0.0)
        elif version == 'spatial-r1-v4.0.6':
            _extra['tau_alpha_init'] = cfg['training'].get('tau_alpha_init', 0.1)
        elif version == 'spatial-r1-v4.1':
            # v4.1 removed dynamic tau / alpha entirely; no model-level kwargs.
            pass
        model = _cls(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
            **_extra,
        )
    elif version == 'rw-v4.0.2':
        model = DAWN_RW_V402(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            n_q=mcfg.get('n_q', 790),
            n_k=mcfg.get('n_k', 790),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_q=cfg['training'].get('n_chunks_q', 1),
            n_chunks_k=cfg['training'].get('n_chunks_k', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
        )
    elif version == 'spatial-r1-v3.9.8.1':
        model = DAWN_SpatialV3981Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.8':
        model = DAWN_SpatialV398Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.7.1':
        model = DAWN_SpatialV3971Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.7':
        model = DAWN_SpatialV397Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.6':
        model = DAWN_SpatialV396Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.5':
        model = DAWN_SpatialV395Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
            gate_norm_mode=mcfg.get('gate_norm_mode', 'sqrt_active'),
        )
    elif version == 'spatial-r1-v3.9.4':
        model = DAWN_SpatialV394Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.3':
        model = DAWN_SpatialV3Exp(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version == 'spatial-r1-v3.9.1':
        model = DAWN_SpatialV3Baseline(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version.startswith('spatial-r1-v3'):
        model = DAWN_SpatialV3(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_route=mcfg.get('d_route', mcfg.get('d_bottleneck', 128)),
            n_qk=mcfg.get('n_qk', 1580),
            n_v=mcfg.get('n_v', 2600),
            n_know=mcfg.get('n_know', 25200),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
            n_chunks_know=cfg['training'].get('n_chunks_know', 1),
            n_chunks_qk=cfg['training'].get('n_chunks_qk', 1),
            n_chunks_v=cfg['training'].get('n_chunks_v', 1),
        )
    elif version.startswith('spatial-r1-v2'):
        model = DAWN_SpatialV2(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            max_seq_len=mcfg.get('max_seq_len', 512),
            pos_dim=mcfg.get('pos_dim', 2),
            grid_size=mcfg.get('grid_size', 64),
            candidates_multiplier=mcfg.get('candidates_multiplier', 2),
            grid_rebuild_interval=mcfg.get('grid_rebuild_interval', 100),
            pos_loss_weight=mcfg.get('pos_loss_weight', 0.01),
            know_chunk_size=mcfg.get('know_chunk_size', 16),
            n_qk=mcfg.get('n_qk', 3140),
            n_v=mcfg.get('n_v', 5240),
            n_know=mcfg.get('n_know', 42000),
            max_k_qk=mcfg.get('max_k_qk', 157),
            max_k_v=mcfg.get('max_k_v', 262),
            max_k_know=mcfg.get('max_k_know', 1536),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
        )
    else:
        model = DAWN(
            vocab_size=mcfg.get('vocab_size', 30522),
            d_model=mcfg.get('d_model', 384),
            n_layers=mcfg.get('n_layers', 12),
            n_heads=mcfg.get('n_heads', 6),
            rank=mcfg.get('rank', 64),
            max_seq_len=mcfg.get('max_seq_len', 512),
            d_space=mcfg.get('d_space', 64),
            n_feature_qk=mcfg.get('n_feature_qk', 56),
            n_feature_v=mcfg.get('n_feature_v', 24),
            top_k_feature_qk=mcfg.get('top_k_feature_qk', 16),
            top_k_feature_v=mcfg.get('top_k_feature_v', 6),
            n_restore_qk=mcfg.get('n_restore_qk', 56),
            n_restore_v=mcfg.get('n_restore_v', 24),
            top_k_restore_qk=mcfg.get('top_k_restore_qk', 16),
            top_k_restore_v=mcfg.get('top_k_restore_v', 6),
            n_feature_know=mcfg.get('n_feature_know', 24),
            n_restore_know=mcfg.get('n_restore_know', 24),
            top_k_feature_know=mcfg.get('top_k_feature_know', 4),
            top_k_restore_know=mcfg.get('top_k_restore_know', 4),
            knowledge_rank=mcfg.get('knowledge_rank', 128),
            dropout_rate=mcfg.get('dropout', 0.1),
            router_dropout=mcfg.get('router_dropout', 0.1),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
        )
    return model


# ============================================================
# GCS / file I/O helpers
# ============================================================

def _is_gcs(path):
    return str(path).startswith("gs://")


def _open_file(path, mode="rb"):
    """Open a file for read/write, supporting GCS paths."""
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.open(path_str, mode)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
        except ImportError:
            raise ImportError(
                "GCS support requires 'gcsfs' or 'tensorflow'. "
                "Install with: pip install gcsfs"
            )
    else:
        p = Path(path_str)
        if "w" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)


def _file_exists(path):
    """Check if a file exists (local or GCS)."""
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.exists(path_str)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            return tf.io.gfile.exists(path_str)
        except ImportError:
            return False
    return Path(path_str).exists()


def _list_files(directory, pattern="*.flax"):
    """List files in a directory (local or GCS), sorted by step number."""
    import re
    dir_str = str(directory)

    def _sort_key(path):
        """Extract step number for numeric sort. best_model sorts last."""
        name = path.rsplit('/', 1)[-1] if '/' in path else path
        if 'best_model' in name:
            return float('inf')
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 0

    if _is_gcs(dir_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = fs.glob(dir_str + pattern)
            return sorted(["gs://" + f for f in files], key=_sort_key)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = tf.io.gfile.glob(dir_str + pattern)
            return sorted(files, key=_sort_key)
        except ImportError:
            return []
    return sorted((str(f) for f in Path(dir_str).glob(pattern)), key=_sort_key)


def _makedirs(path):
    """Create directory (local only; GCS doesn't need explicit mkdir)."""
    if not _is_gcs(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _delete_file(path):
    """Delete a single file (local or GCS)."""
    path_str = str(path)
    if _is_gcs(path_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            fs.rm(path_str)
            return
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.io.gfile.remove(path_str)
            return
        except ImportError:
            pass
    else:
        p = Path(path_str)
        if p.exists():
            p.unlink()


def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """Keep only the last N step checkpoints. best_model/epoch/emergency are never deleted."""
    all_files = _list_files(checkpoint_dir, "checkpoint_step*.flax")
    if len(all_files) <= keep_last:
        return
    import re
    def _step_num(path):
        m = re.search(r'checkpoint_step(\d+)\.flax', str(path))
        return int(m.group(1)) if m else 0
    all_files.sort(key=_step_num)
    to_delete = all_files[:-keep_last]
    for f in to_delete:
        _delete_file(f)


# ============================================================
# Parameter count
# ============================================================

def count_parameters(params):
    """Count total parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


# ============================================================
# Orthogonality + diversity loss (inline for jit)
# ============================================================

def compute_orthogonality_loss(params, rank, knowledge_rank, n_feature_qk, n_restore_qk):
    """Compute orthogonality loss from shared neuron params.

    Matches the model's 6-group computation:
      f_neurons = [feature_qk ; feature_v]  -> split at n_feature_qk
      r_neurons = [restore_qk ; restore_v]  -> split at n_restore_qk
      feature_know, restore_know             -> separate params
    """
    sn = params['shared_neurons']
    I_rank = jnp.eye(rank)[jnp.newaxis]
    I_know = jnp.eye(knowledge_rank)[jnp.newaxis]

    f_neurons = sn['f_neurons']
    r_neurons = sn['r_neurons']
    feature_know = sn['feature_know']
    restore_know = sn['restore_know']

    # Split f_neurons into feature_qk [N_fqk, D, R] and feature_v [N_fv, D, R]
    W_fqk = f_neurons[:n_feature_qk]
    W_fv = f_neurons[n_feature_qk:]
    WtW_fqk = jnp.matmul(W_fqk.transpose(0, 2, 1), W_fqk)
    loss_fqk = ((WtW_fqk - I_rank) ** 2).mean()
    WtW_fv = jnp.matmul(W_fv.transpose(0, 2, 1), W_fv)
    loss_fv = ((WtW_fv - I_rank) ** 2).mean()

    # Split r_neurons into restore_qk [N_rqk, R, D] and restore_v [N_rv, R, D]
    W_rqk = r_neurons[:n_restore_qk]
    W_rv = r_neurons[n_restore_qk:]
    WWt_rqk = jnp.matmul(W_rqk, W_rqk.transpose(0, 2, 1))
    loss_rqk = ((WWt_rqk - I_rank) ** 2).mean()
    WWt_rv = jnp.matmul(W_rv, W_rv.transpose(0, 2, 1))
    loss_rv = ((WWt_rv - I_rank) ** 2).mean()

    WtW_fk = jnp.matmul(feature_know.transpose(0, 2, 1), feature_know)
    loss_fk = ((WtW_fk - I_know) ** 2).mean()

    WWt_rk = jnp.matmul(restore_know, restore_know.transpose(0, 2, 1))
    loss_rk = ((WWt_rk - I_know) ** 2).mean()

    return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fk + loss_rk) / 6


def compute_knowledge_diversity_loss(params):
    """Compute knowledge diversity loss from shared neuron params."""
    sn = params['shared_neurons']

    feat_know = sn['feature_know']
    feat_flat = feat_know.reshape(feat_know.shape[0], -1)
    feat_norm = feat_flat / (jnp.linalg.norm(feat_flat, axis=-1, keepdims=True) + 1e-8)
    feat_sim = jnp.matmul(feat_norm, feat_norm.T)
    mask_f = ~jnp.eye(feat_sim.shape[0], dtype=jnp.bool_)
    feat_loss = jnp.abs(feat_sim * mask_f).sum() / mask_f.sum()

    rest_know = sn['restore_know']
    rest_flat = rest_know.reshape(rest_know.shape[0], -1)
    rest_norm = rest_flat / (jnp.linalg.norm(rest_flat, axis=-1, keepdims=True) + 1e-8)
    rest_sim = jnp.matmul(rest_norm, rest_norm.T)
    mask_r = ~jnp.eye(rest_sim.shape[0], dtype=jnp.bool_)
    rest_loss = jnp.abs(rest_sim * mask_r).sum() / mask_r.sum()

    return (feat_loss + rest_loss) / 2


def compute_spatial_diversity_loss(params):
    """Compute neuron diversity loss for rank-1 spatial neurons.

    Penalizes high cosine similarity between neurons in each pool.
    Replaces orthogonality + knowledge diversity for spatial-r1.
    For large pools (>4096), uses deterministic strided sampling to avoid O(N^2).
    """
    pool = params['neuron_pool']

    def _pool_div(neurons, max_sample=4096):
        N = neurons.shape[0]
        if N > max_sample:
            stride = N // max_sample
            neurons = neurons[::stride][:max_sample]
        n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.matmul(n, n.T)
        mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
        return jnp.abs(sim * mask).sum() / mask.sum()

    # Support both v2 (qk_neurons) and v3.2 (qk_emb/qk_w) param names
    def _get_pool_arrays(pool):
        """Return list of neuron arrays from pool params."""
        arrays = []
        # v4.0.2 (rw): separate q/k pools, read/write only
        for prefix in ('q', 'k', 'v', 'know'):
            if f'{prefix}_read' in pool:
                arrays.append(pool[f'{prefix}_read'])
                arrays.append(pool[f'{prefix}_write'])
        if arrays:
            return arrays
        # v3/v4.0.1: shared qk pool
        for prefix in ('qk', 'v', 'know'):
            if f'{prefix}_neurons' in pool:
                arrays.append(pool[f'{prefix}_neurons'])
            else:
                if f'{prefix}_emb' in pool:
                    arrays.append(pool[f'{prefix}_emb'])
                if f'{prefix}_w' in pool:
                    arrays.append(pool[f'{prefix}_w'])
                if f'{prefix}_read' in pool:
                    arrays.append(pool[f'{prefix}_read'])
                    arrays.append(pool[f'{prefix}_write'])
        return arrays

    pool_arrays = _get_pool_arrays(pool)
    return sum(_pool_div(a) for a in pool_arrays) / len(pool_arrays)


# ============================================================
# Train / eval steps (pmap for multi-device)
# ============================================================

def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      tau_reg_weight, dead_penalty_weight,
                      exploration_weight, exploration_asymmetry,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk,
                      is_baseline=False, is_spatial=False,
                      sharded_fns=None, mesh=None):
    """Create a jit-compiled training step. Mesh SPMD handles parallelism.

    v4.1 explore (redesigned): no EMA, no warmup. For every step compute
    a batch-global per-token CE mean, define
        deviation = per_token_ce - sg(global_mean_ce)
        signal    = where(deviation > 0, deviation, asymmetry · deviation)
    and add
        explore_loss = λ · valid_weighted_mean(signal · Σ tau_offset)
    to total_loss.  Positive deviations (surprising tokens) push
    tau_offset DOWN at full strength; negative deviations (easy tokens)
    push UP at `exploration_asymmetry` of the strength.  The global mean
    baseline makes the net push roughly zero-sum each batch, so tau does
    not accumulate monotonically.

    `mesh` is required when the v4.1 exploration loss is active — the
    per-batch global mean is computed via a small shard_map.
    """
    # Shard_map'd valid-weighted global-mean reducer.  Inputs are sharded
    # on 'data' (batch-parallel); psum aggregates across shards + hosts.
    _global_mean_reducer = None
    if mesh is not None:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P('data', None),       # per_token_ce [B, S-1]
                           P('data', None)),      # valid_mask    [B, S-1]
                 out_specs=P(),                    # scalar replicated
                 check_rep=False)
        def _mean_reducer_fn(pce, vmask):
            vm_f = vmask.astype(jnp.float32)
            local_sum = (pce * vm_f).sum()
            local_cnt = vm_f.sum()
            g_sum = jax.lax.psum(local_sum, 'data')
            g_cnt = jax.lax.psum(local_cnt, 'data')
            return g_sum / (g_cnt + 1e-8)
        _global_mean_reducer = _mean_reducer_fn

    _asym = jnp.float32(exploration_asymmetry)

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, dropout_key):
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            extra_kw = {}
            if sharded_fns is not None:
                extra_kw['sharded_fns'] = sharded_fns
            result = model.apply(
                {'params': params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={'dropout': dropout_key},
                **extra_kw,
            )
            ce_loss = result['loss']
            aux_loss = result['aux_loss']
            tau_reg = result.get('tau_reg', jnp.float32(0.0))
            dead_penalty = result.get('dead_penalty', jnp.float32(0.0))

            # v4.1 batch-global-mean exploration loss.
            per_token_ce = result.get('per_token_ce', None)
            attn_tau_off = result.get('attn_tau_offset', None)
            know_tau_off = result.get('know_tau_offset', None)
            valid_mask = result.get('valid_mask', None)
            have_explore = (per_token_ce is not None
                            and attn_tau_off is not None
                            and know_tau_off is not None
                            and valid_mask is not None
                            and _global_mean_reducer is not None)
            if have_explore:
                vmask_f = valid_mask.astype(jnp.float32)
                # Global mean CE across all valid tokens (multi-host safe).
                global_mean_ce = _global_mean_reducer(per_token_ce, valid_mask)
                # Signed deviation; gradient only flows through tau_offset.
                deviation = jax.lax.stop_gradient(
                    per_token_ce - global_mean_ce)                   # [B, S-1]
                # Asymmetric: full push on hard tokens, `asym`·push on easy ones.
                signal = jnp.where(deviation > 0, deviation, _asym * deviation)
                # Sum / mean over layers and route dim to match [B, S-1].
                a_sum = attn_tau_off[:, :, :-1, :].sum(axis=(0, -1))
                k_sum = know_tau_off[:, :, :-1, :].sum(axis=(0, -1))
                a_tau_mean = attn_tau_off[:, :, :-1, :].mean(axis=(0, -1))
                k_tau_mean = know_tau_off[:, :, :-1, :].mean(axis=(0, -1))
                # v4.1 tau_offset distribution diagnostics (sg to keep them
                # out of the backward pass — they're purely observational).
                _a_tau_flat = jax.lax.stop_gradient(attn_tau_off)
                _k_tau_flat = jax.lax.stop_gradient(know_tau_off)
                attn_tau_off_min = _a_tau_flat.min()
                attn_tau_off_max = _a_tau_flat.max()
                attn_tau_off_p99 = jnp.quantile(_a_tau_flat, 0.99)
                attn_tau_off_p01 = jnp.quantile(_a_tau_flat, 0.01)
                attn_tau_off_neg_frac = (_a_tau_flat < 0).astype(jnp.float32).mean()
                know_tau_off_min = _k_tau_flat.min()
                know_tau_off_max = _k_tau_flat.max()
                know_tau_off_p99 = jnp.quantile(_k_tau_flat, 0.99)
                know_tau_off_p01 = jnp.quantile(_k_tau_flat, 0.01)
                know_tau_off_neg_frac = (_k_tau_flat < 0).astype(jnp.float32).mean()
                # Downward cap: when signal > 0 (about to push tau DOWN) and
                # the per-pool mean tau_offset is already ≤ 0, zero the
                # contribution for that token. Up-push (signal < 0) always
                # allowed — CE self-correction can still recover pools that
                # over-pruned. Comparison op is non-differentiable so no
                # gradient flows through the mask itself.
                pos_signal = (signal > 0)
                a_block = pos_signal & (a_tau_mean <= 0.0)
                k_block = pos_signal & (k_tau_mean <= 0.0)
                a_mask = jnp.where(a_block, 0.0, 1.0)
                k_mask = jnp.where(k_block, 0.0, 1.0)
                vsum_eps = vmask_f.sum() + 1e-8
                explore_attn_raw = (signal * a_mask * a_sum * vmask_f).sum() / vsum_eps
                explore_know_raw = (signal * k_mask * k_sum * vmask_f).sum() / vsum_eps
                explore_loss_raw = explore_attn_raw + explore_know_raw
                # Log-only stats (all stop_gradient via pure reductions).
                pos_mask = (deviation > 0).astype(jnp.float32) * vmask_f
                neg_mask = (deviation < 0).astype(jnp.float32) * vmask_f
                pos_frac = pos_mask.sum() / vsum_eps
                pos_mean = (jnp.maximum(deviation, 0.0) * vmask_f).sum() / (
                    pos_mask.sum() + 1e-8)
                neg_mean = (jnp.maximum(-deviation, 0.0) * vmask_f).sum() / (
                    neg_mask.sum() + 1e-8)
                # Down-push block fractions (of valid tokens).
                block_frac_a = jax.lax.stop_gradient(
                    (a_block.astype(jnp.float32) * vmask_f).sum() / vsum_eps)
                block_frac_k = jax.lax.stop_gradient(
                    (k_block.astype(jnp.float32) * vmask_f).sum() / vsum_eps)
                # Signal magnitude extremes (diag).
                _sig_sg = jax.lax.stop_gradient(signal * vmask_f)
                sig_pos_max = _sig_sg.max()
                sig_neg_max = (-_sig_sg).max()
            else:
                global_mean_ce = jnp.float32(0.0)
                explore_loss_raw = jnp.float32(0.0)
                explore_attn_raw = jnp.float32(0.0)
                explore_know_raw = jnp.float32(0.0)
                pos_frac = jnp.float32(0.0)
                pos_mean = jnp.float32(0.0)
                neg_mean = jnp.float32(0.0)
                block_frac_a = jnp.float32(0.0)
                block_frac_k = jnp.float32(0.0)
                sig_pos_max = jnp.float32(0.0)
                sig_neg_max = jnp.float32(0.0)
                attn_tau_off_min = jnp.float32(0.0)
                attn_tau_off_max = jnp.float32(0.0)
                attn_tau_off_p99 = jnp.float32(0.0)
                attn_tau_off_p01 = jnp.float32(0.0)
                attn_tau_off_neg_frac = jnp.float32(0.0)
                know_tau_off_min = jnp.float32(0.0)
                know_tau_off_max = jnp.float32(0.0)
                know_tau_off_p99 = jnp.float32(0.0)
                know_tau_off_p01 = jnp.float32(0.0)
                know_tau_off_neg_frac = jnp.float32(0.0)
            explore_loss_weighted = exploration_weight * explore_loss_raw

            if is_baseline:
                orth_loss = jnp.float32(0.0)
                div_loss = jnp.float32(0.0)
                total_loss = ce_loss
            elif is_spatial:
                orth_loss = jnp.float32(0.0)
                div_loss = compute_spatial_diversity_loss(params)
                total_loss = (ce_loss
                              + lb_weight * aux_loss
                              + tau_reg_weight * tau_reg
                              + div_weight * div_loss
                              + dead_penalty_weight * dead_penalty
                              + explore_loss_weighted)
            else:
                orth_loss = compute_orthogonality_loss(
                    params, rank, knowledge_rank, n_feature_qk, n_restore_qk)
                div_loss = compute_knowledge_diversity_loss(params)
                total_loss = (ce_loss
                              + lb_weight * aux_loss
                              + tau_reg_weight * tau_reg
                              + orth_weight * orth_loss
                              + div_weight * div_loss
                              + dead_penalty_weight * dead_penalty
                              + explore_loss_weighted)

            explore_stats = dict(
                global_mean_ce=global_mean_ce,
                explore_loss_raw=explore_loss_raw,
                explore_attn_raw=explore_attn_raw,
                explore_know_raw=explore_know_raw,
                pos_frac=pos_frac, pos_mean=pos_mean, neg_mean=neg_mean,
                block_frac_a=block_frac_a, block_frac_k=block_frac_k,
                sig_pos_max=sig_pos_max, sig_neg_max=sig_neg_max,
                attn_tau_off_min=attn_tau_off_min, attn_tau_off_max=attn_tau_off_max,
                attn_tau_off_p99=attn_tau_off_p99, attn_tau_off_p01=attn_tau_off_p01,
                attn_tau_off_neg_frac=attn_tau_off_neg_frac,
                know_tau_off_min=know_tau_off_min, know_tau_off_max=know_tau_off_max,
                know_tau_off_p99=know_tau_off_p99, know_tau_off_p01=know_tau_off_p01,
                know_tau_off_neg_frac=know_tau_off_neg_frac,
            )
            return total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                                dead_penalty, explore_stats, result)

        (total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                      dead_penalty, explore_stats, result)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)

        # XLA SPMD handles gradient all-reduce automatically
        # (loss computed on sharded data → gradients consistent across shards)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Re-project neuron pool vectors to unit norm after optimizer step
        def normalize_pool_params(params):
            pool = params['neuron_pool']
            norm_keys = [
                'qk_read', 'v_read', 'know_read',
                'qk_write', 'v_write', 'know_write',
                'qk_emb', 'v_emb', 'know_emb',
            ]
            new_pool = dict(pool)
            for key in norm_keys:
                w = new_pool[key]
                new_pool[key] = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)
            return {**params, 'neuron_pool': new_pool}

        # v3.9.4: re-projection disabled — forward unit-norm handles normalization,
        # param norm freedom provides implicit gradient scaling regularization
        # new_params = normalize_pool_params(new_params)

        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))

        # Tau bias (read inside jit — safe, no cross-device issue)
        tau_know_b = params.get('router', {}).get('tau_know', {}).get(
            'bias', jnp.zeros(1))
        tau_attn_b = params.get('router', {}).get('tau_attn', {}).get(
            'bias', jnp.zeros(3))

        metrics = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': aux_loss,
            'tau_reg': tau_reg,
            'orth_loss': orth_loss,
            'div_loss': div_loss,
            'correct': result['correct'],
            'valid_count': result['valid_count'],
            'grad_norm': grad_norm,
            'attn_aux': result.get('attn_aux', jnp.float32(0.0)),
            'know_aux': result.get('know_aux', jnp.float32(0.0)),
            'know_active': result.get('know_active', jnp.float32(0.0)),
            'know_active_N': result.get('know_active_N', jnp.float32(0.0)),
            'know_score_std': result.get('know_score_std', jnp.float32(0.0)),
            'know_raw_gate_max': result.get('know_gate_max', result.get('know_raw_gate_max', jnp.float32(0.0))),
            'know_gate_sum': result.get('know_gate_sum', jnp.float32(0.0)),
            'know_gate_conc': result.get('know_gate_conc', jnp.float32(0.0)),
            'know_active_n_mean': result.get('know_active_n_mean', jnp.float32(0.0)),
            'know_strong': result.get('know_strong', result.get('know_pos', jnp.float32(0.0))),
            'know_strength_mean': result.get('know_strength_mean', jnp.float32(0.0)),
            'know_strength_std': result.get('know_strength_std', jnp.float32(0.0)),
            'know_strength_min': result.get('know_strength_min', jnp.float32(0.0)),
            'know_strength_max': result.get('know_strength_max', jnp.float32(0.0)),
            'know_logit_mean': result.get('know_logit_mean', jnp.float32(0.0)),
            'know_logit_std': result.get('know_logit_std', jnp.float32(0.0)),
            'attn_qk_active': result.get('attn_qk_active', jnp.float32(0.0)),
            'attn_v_active': result.get('attn_v_active', jnp.float32(0.0)),
            'attn_active_N': result.get('attn_active_N', jnp.float32(0.0)),
            'attn_score_std': result.get('attn_score_std', jnp.float32(0.0)),
            'attn_raw_gate_max': result.get('attn_gate_max', result.get('attn_raw_gate_max', jnp.float32(0.0))),
            'attn_gate_sum': result.get('attn_gate_sum', jnp.float32(0.0)),
            'attn_gate_conc': result.get('attn_gate_conc', jnp.float32(0.0)),
            'attn_active_n_mean': result.get('attn_active_n_mean', jnp.float32(0.0)),
            'attn_strong': result.get('attn_strong', result.get('attn_pos', jnp.float32(0.0))),
            'attn_qk_pos': result.get('attn_qk_pos', jnp.float32(0.0)),
            'attn_v_pos': result.get('attn_v_pos', jnp.float32(0.0)),
            'attn_v_strength_mean': result.get('attn_v_strength_mean', jnp.float32(0.0)),
            'attn_v_strength_std': result.get('attn_v_strength_std', jnp.float32(0.0)),
            'attn_v_strength_min': result.get('attn_v_strength_min', jnp.float32(0.0)),
            'attn_v_strength_max': result.get('attn_v_strength_max', jnp.float32(0.0)),
            'attn_v_logit_mean': result.get('attn_v_logit_mean', jnp.float32(0.0)),
            'attn_v_logit_std': result.get('attn_v_logit_std', jnp.float32(0.0)),
            'attn_out_norm': result.get('attn_out_norm', jnp.float32(0.0)),
            'attn_tau_mean': result.get('attn_tau_mean', jnp.float32(0.0)),
            'know_tau_mean': result.get('know_tau_mean', jnp.float32(0.0)),
            'attn_tau_std': result.get('attn_tau_std', jnp.zeros(3)),
            'know_tau_std': result.get('know_tau_std', jnp.float32(0.0)),
            'attn_tau_kernel_norm': result.get('attn_tau_kernel_norm', jnp.float32(0.0)),
            'know_tau_kernel_norm': result.get('know_tau_kernel_norm', jnp.float32(0.0)),
            'attn_tau_abs_mean': result.get('attn_tau_abs_mean', jnp.float32(0.0)),
            'know_tau_abs_mean': result.get('know_tau_abs_mean', jnp.float32(0.0)),
            'attn_z_lt_075': result.get('attn_z_lt_075', jnp.float32(0.0)),
            'know_z_lt_075': result.get('know_z_lt_075', jnp.float32(0.0)),
            'attn_z_lt_030': result.get('attn_z_lt_030', jnp.float32(0.0)),
            'know_z_lt_030': result.get('know_z_lt_030', jnp.float32(0.0)),
            'attn_score_skew': result.get('attn_score_skew', jnp.float32(0.0)),
            'know_score_skew': result.get('know_score_skew', jnp.float32(0.0)),
            'attn_active_per_token_std': result.get('attn_active_per_token_std', jnp.float32(0.0)),
            'know_active_per_token_std': result.get('know_active_per_token_std', jnp.float32(0.0)),
            'attn_gate_entropy': result.get('attn_gate_entropy', jnp.float32(0.0)),
            'know_gate_entropy': result.get('know_gate_entropy', jnp.float32(0.0)),
            'attn_z_sum': result.get('attn_z_sum', jnp.float32(0.0)),
            'know_z_sum': result.get('know_z_sum', jnp.float32(0.0)),
            'know_emb_norm': result.get('know_emb_norm', jnp.float32(0.0)),
            'know_emb_norm_max': result.get('know_emb_norm_max', jnp.float32(0.0)),
            'know_emb_norm_min': result.get('know_emb_norm_min', jnp.float32(0.0)),
            'know_emb_norm_std': result.get('know_emb_norm_std', jnp.float32(0.0)),
            'qk_emb_norm_mean': result.get('qk_emb_norm_mean', jnp.float32(0.0)),
            'qk_emb_norm_max': result.get('qk_emb_norm_max', jnp.float32(0.0)),
            'qk_emb_norm_min': result.get('qk_emb_norm_min', jnp.float32(0.0)),
            'qk_emb_norm_std': result.get('qk_emb_norm_std', jnp.float32(0.0)),
            'v_emb_norm_mean': result.get('v_emb_norm_mean', jnp.float32(0.0)),
            'v_emb_norm_max': result.get('v_emb_norm_max', jnp.float32(0.0)),
            'v_emb_norm_min': result.get('v_emb_norm_min', jnp.float32(0.0)),
            'v_emb_norm_std': result.get('v_emb_norm_std', jnp.float32(0.0)),
            'attn_score_kurt': result.get('attn_score_kurt', jnp.float32(0.0)),
            'know_score_kurt': result.get('know_score_kurt', jnp.float32(0.0)),
            'attn_drop_rate': result.get('attn_drop_rate', jnp.float32(0.0)),
            'attn_boost_rate': result.get('attn_boost_rate', jnp.float32(0.0)),
            'know_drop_rate': result.get('know_drop_rate', jnp.float32(0.0)),
            'know_boost_rate': result.get('know_boost_rate', jnp.float32(0.0)),
            'know_read_norm': result.get('know_read_norm', jnp.float32(0.0)),
            'know_write_norm': result.get('know_write_norm', jnp.float32(0.0)),
            'tau_know_bias': tau_know_b[0],
            'tau_attn_bias_0': tau_attn_b[0],
            'tau_attn_bias_1': tau_attn_b[1],
            'tau_attn_bias_2': tau_attn_b[2],
            'know_out_norm': result.get('know_out_norm', jnp.float32(0.0)),
            'attn_qk_raw_norm': result.get('attn_qk_raw_norm', jnp.float32(0.0)),
            'attn_v_raw_norm': result.get('attn_v_raw_norm', jnp.float32(0.0)),
            'know_raw_out_norm': result.get('know_raw_out_norm', jnp.float32(0.0)),
            'debug_residual_norm': result.get('debug_residual_norm', jnp.float32(0.0)),
            'debug_emb_norm': result.get('debug_emb_norm', jnp.float32(0.0)),
            'debug_o_proj_norm': result.get('debug_o_proj_norm', jnp.float32(0.0)),
            'debug_q_norm': result.get('debug_q_norm', jnp.float32(0.0)),
            'debug_k_norm': result.get('debug_k_norm', jnp.float32(0.0)),
            'debug_v_norm': result.get('debug_v_norm', jnp.float32(0.0)),
            'debug_logit_max': result.get('debug_logit_max', jnp.float32(0.0)),
            'debug_o_input_norm': result.get('debug_o_input_norm', jnp.float32(0.0)),
            'know_phi_binary': result.get('know_phi_binary', jnp.float32(0.0)),
            'know_z_mean_active': result.get('know_z_mean_active', jnp.float32(0.0)),
            'attn_qk_phi_binary': result.get('attn_qk_phi_binary', jnp.float32(0.0)),
            'attn_v_phi_binary': result.get('attn_v_phi_binary', jnp.float32(0.0)),
            'attn_qk_z_mean_active': result.get('attn_qk_z_mean_active', jnp.float32(0.0)),
            'attn_v_z_mean_active': result.get('attn_v_z_mean_active', jnp.float32(0.0)),
            'per_layer_attn_out_norm': result.get('per_layer_attn_out_norm', jnp.zeros(1)),
            'per_layer_know_out_norm': result.get('per_layer_know_out_norm', jnp.zeros(1)),
            # v4.0.6: dead-only penalty + alpha / tau_shift observations.
            'dead_penalty': dead_penalty,
            'attn_dead_penalty': result.get('attn_dead_penalty', jnp.float32(0.0)),
            'know_dead_penalty': result.get('know_dead_penalty', jnp.float32(0.0)),
            'attn_dead_count': result.get('attn_dead_count', jnp.float32(0.0)),
            'know_dead_count': result.get('know_dead_count', jnp.float32(0.0)),
            'attn_qk_tau_shift_mean': result.get('attn_qk_tau_shift_mean', jnp.float32(0.0)),
            'attn_v_tau_shift_mean': result.get('attn_v_tau_shift_mean', jnp.float32(0.0)),
            'know_tau_shift_mean': result.get('know_tau_shift_mean', jnp.float32(0.0)),
            'alpha_qk': result.get('alpha_qk', jnp.float32(0.0)),
            'alpha_v': result.get('alpha_v', jnp.float32(0.0)),
            'alpha_know': result.get('alpha_know', jnp.float32(0.0)),
            'attn_s_std_min': result.get('attn_s_std_min', jnp.float32(0.0)),
            'know_s_std_min': result.get('know_s_std_min', jnp.float32(0.0)),
            # v4.1 RPE exploration + diagnostics.
            'global_mean_ce': explore_stats['global_mean_ce'],
            'pos_frac': explore_stats['pos_frac'],
            'pos_mean': explore_stats['pos_mean'],
            'neg_mean': explore_stats['neg_mean'],
            'explore_loss_raw': explore_stats['explore_loss_raw'],
            'explore_attn_raw': explore_stats['explore_attn_raw'],
            'explore_know_raw': explore_stats['explore_know_raw'],
            'explore_loss_weighted': exploration_weight * explore_stats['explore_loss_raw'],
            'explore_block_frac_a': explore_stats['block_frac_a'],
            'explore_block_frac_k': explore_stats['block_frac_k'],
            'sig_pos_max': explore_stats['sig_pos_max'],
            'sig_neg_max': explore_stats['sig_neg_max'],
            'attn_tau_off_min': explore_stats['attn_tau_off_min'],
            'attn_tau_off_max': explore_stats['attn_tau_off_max'],
            'attn_tau_off_p99': explore_stats['attn_tau_off_p99'],
            'attn_tau_off_p01': explore_stats['attn_tau_off_p01'],
            'attn_tau_off_neg_frac': explore_stats['attn_tau_off_neg_frac'],
            'know_tau_off_min': explore_stats['know_tau_off_min'],
            'know_tau_off_max': explore_stats['know_tau_off_max'],
            'know_tau_off_p99': explore_stats['know_tau_off_p99'],
            'know_tau_off_p01': explore_stats['know_tau_off_p01'],
            'know_tau_off_neg_frac': explore_stats['know_tau_off_neg_frac'],
            # v4.1 intensity diagnostics.
            'attn_int_max': result.get('attn_int_max', jnp.float32(0.0)),
            'know_int_max': result.get('know_int_max', jnp.float32(0.0)),
            'attn_int_cap_frac': result.get('attn_int_cap_frac', jnp.float32(0.0)),
            'know_int_cap_frac': result.get('know_int_cap_frac', jnp.float32(0.0)),
        }

        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step(model, sharded_fns=None):
    """Create a jit-compiled evaluation step."""

    @jax.jit
    def eval_step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        eval_rng = jax.random.PRNGKey(0)
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
            **extra_kw,
        )
        return result['loss'], result['correct'], result['valid_count']

    return eval_step


# ============================================================
# Mesh-based sharding (model parallel + data parallel)
# ============================================================

def create_mesh(mesh_data, mesh_model):
    """Create 2D Mesh for data + model parallelism."""
    devices = jax.devices()
    n_devices = len(devices)
    assert n_devices == mesh_data * mesh_model, (
        f"mesh_data({mesh_data}) * mesh_model({mesh_model}) = "
        f"{mesh_data * mesh_model} != {n_devices} devices")
    device_array = np.array(devices).reshape(mesh_data, mesh_model)
    return Mesh(device_array, ('data', 'model'))


def get_param_shardings(params, mesh, is_baseline=False):
    """Create sharding specs for params: neuron_pool N-axis on 'model', rest replicated.
    For baseline models (is_baseline=True), 2D+ params are sharded on 'data' axis (FSDP-style).
    """
    replicated = NamedSharding(mesh, P())  # no sharding
    n_sharded = NamedSharding(mesh, P('model', None))  # N axis on model
    n_sharded_3d = NamedSharding(mesh, P('model', None, None))
    data_sharded = NamedSharding(mesh, P('data', None))  # FSDP: first axis on data

    def _get_sharding(path, value):
        path_str = '/'.join(str(p) for p in path)
        # NeuronPool params: shard N axis (first dim) on 'model'
        if 'neuron_pool' in path_str:
            if value.ndim == 2:
                return n_sharded       # [N, d_bn] or [N, D]
            elif value.ndim == 3:
                return n_sharded_3d    # [N, D, R] for v17.1
            else:
                return replicated
        # Baseline FSDP: shard 2D kernels on data axis (skip embeddings)
        if is_baseline and value.ndim >= 2:
            if 'token_emb' in path_str or 'pos_emb' in path_str:
                return replicated
            return data_sharded
        return replicated

    flat_params = jax.tree.leaves_with_path(params)
    shardings = {}
    for path, leaf in flat_params:
        key_path = tuple(
            p.key if hasattr(p, 'key') else str(p) for p in path)
        shardings[key_path] = _get_sharding(path, leaf)

    # Build matching pytree of shardings
    return jax.tree.map_with_path(
        lambda path, x: _get_sharding(path, x), params)


def shard_params_to_mesh(params, param_shardings):
    """Place params on mesh according to shardings."""
    return jax.tree.map(
        lambda p, s: jax.device_put(p, s),
        params, param_shardings)


def shard_to_mesh(data, sharding, global_shape):
    """Multi-host: create global array from host-local data.

    Uses make_array_from_callback which correctly maps mesh indices
    to data slices, regardless of how devices map to hosts.

    data: [per_host_batch, ...] — this host's data portion
    sharding: NamedSharding
    global_shape: (global_batch, ...)
    """
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    per_host = data.shape[0]

    def data_callback(index):
        # index is a tuple of slices for each dimension
        # The batch slice tells us which global rows this device needs
        batch_slice = index[0]
        start = batch_slice.start or 0
        stop = batch_slice.stop or global_shape[0]
        # Map global indices to this host's local data
        local_start = start - host_id * per_host
        local_stop = stop - host_id * per_host
        # If this slice belongs to our host, return it; else zeros (shouldn't happen)
        if 0 <= local_start < per_host:
            return np.array(data[local_start:local_stop])
        else:
            return np.zeros((stop - start,) + data.shape[1:], dtype=data.dtype)

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


# ============================================================
# Helpers
# ============================================================

def shard_batch(batch, n_devices):
    """Reshape a batch for legacy compatibility: (B, ...) -> (n_devices, B//n_devices, ...).

    If the batch is already sharded (leading dim == n_devices), return as-is.
    """
    if isinstance(batch, (tuple, list)):
        return type(batch)(shard_batch(x, n_devices) for x in batch)
    if batch.shape[0] == n_devices:
        return batch  # already sharded by data loader
    return batch.reshape(n_devices, batch.shape[0] // n_devices, *batch.shape[1:])


# ============================================================
# Evaluation loop
# ============================================================

def evaluate(eval_step_fn, params, val_loader, n_devices, max_batches=200,
             verbose=True, data_sharding_spec=None):
    """Run evaluation and return avg loss and accuracy.

    All hosts must call this (pmap requires it), but only verbose=True host prints.
    """
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    eval_total = min(max_batches, len(val_loader))
    eval_start = time.time()

    for batch_idx, (input_ids, attention_mask) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        if data_sharding_spec is not None:
            gb = input_ids.shape[0] * jax.process_count()
            gs = (gb, input_ids.shape[1])
            input_ids = shard_to_mesh(input_ids, data_sharding_spec, gs)
            attention_mask = shard_to_mesh(attention_mask, data_sharding_spec, gs)

        ce_loss, correct, valid_count = eval_step_fn(params, input_ids, attention_mask)

        n_valid = int(valid_count)
        total_loss += float(ce_loss) * n_valid
        total_correct += int(correct)
        total_valid += n_valid

    eval_elapsed = time.time() - eval_start
    done = min(batch_idx + 1, eval_total)
    if verbose:
        print(f"  Eval: {done}/{eval_total} batches, {eval_elapsed:.1f}s", flush=True)
    avg_loss = total_loss / total_valid if total_valid > 0 else 0.0
    avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
    return avg_loss, avg_acc


# ============================================================
# Checkpoint save / load (with GCS support)
# ============================================================

def save_checkpoint(path, params, opt_state, epoch, step, best_val_loss, model_config,
                    step_in_epoch=0, steps_per_epoch=0, training_config=None):
    """Save checkpoint using flax serialization. Supports local and GCS paths."""
    import flax.serialization as serialization
    ckpt = {
        'params': params,
        'opt_state': opt_state,
        'epoch': epoch,
        'step': step,
        'step_in_epoch': step_in_epoch,
        'steps_per_epoch': steps_per_epoch,
        'best_val_loss': best_val_loss,
        'config': model_config,
        'training_config': training_config or {},
    }
    bytes_data = serialization.to_bytes(ckpt)

    with _open_file(path, 'wb') as f:
        f.write(bytes_data)
    print(f"  Checkpoint saved: {path} ({len(bytes_data) / 1e6:.1f} MB)")


def load_checkpoint(path, target_params, target_opt_state):
    """Load checkpoint using flax serialization. Supports local and GCS paths."""
    import flax.serialization as serialization
    with _open_file(path, 'rb') as f:
        bytes_data = f.read()
    target = {
        'params': target_params,
        'opt_state': target_opt_state,
        'epoch': 0,
        'step': 0,
        'step_in_epoch': 0,
        'steps_per_epoch': 0,
        'best_val_loss': float('inf'),
        'config': {},
        'training_config': {},
    }
    ckpt = serialization.from_bytes(target, bytes_data)
    if jax.process_index() == 0:
        print(f"  Checkpoint loaded: {path}")
    return ckpt


# ============================================================
# Logging
# ============================================================

class GCSLogger:
    """Logger that writes to a local file and periodically syncs to GCS.

    GCS doesn't support true append — each open('a')/write/close overwrites.
    So we always append to a local file and upload the full file to GCS
    on every sync() call.
    """

    def __init__(self, gcs_path, local_path):
        self.gcs_path = gcs_path
        self.local_path = local_path
        self._dirty = False
        # Ensure local parent dir exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, text):
        with open(self.local_path, 'a') as f:
            f.write(text)
        self._dirty = True

    def sync(self):
        """Upload local file to GCS if there are new writes."""
        if not self._dirty or not self.gcs_path:
            return
        try:
            with open(self.local_path, 'rb') as f:
                data = f.read()
            with _open_file(self.gcs_path, 'wb') as f:
                f.write(data)
            self._dirty = False
        except Exception as e:
            print(f"  [warn] GCS sync failed: {e}")


# Module-level loggers — set up in main()
_train_logger = None
_jsonl_logger = None


def _setup_loggers(training_log_file, jsonl_log_file):
    """Create GCSLogger instances for training log and JSONL log."""
    global _train_logger, _jsonl_logger
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "dawn_logs"
    tmpdir.mkdir(parents=True, exist_ok=True)

    if _is_gcs(training_log_file):
        local_txt = str(tmpdir / Path(training_log_file).name)
        _train_logger = GCSLogger(training_log_file, local_txt)
    else:
        _train_logger = GCSLogger(None)

    if _is_gcs(jsonl_log_file):
        local_jsonl = str(tmpdir / Path(jsonl_log_file).name)
        _jsonl_logger = GCSLogger(jsonl_log_file, local_jsonl)
    else:
        _jsonl_logger = GCSLogger(None, jsonl_log_file)


def sync_logs():
    """Flush local logs to GCS. Call periodically (e.g. every LOG_INTERVAL)."""
    if _train_logger:
        _train_logger.sync()
    if _jsonl_logger:
        _jsonl_logger.sync()


def log_message(msg, log_file=None):
    """Print and write to training log file."""
    print(msg, flush=True)
    if _train_logger:
        try:
            _train_logger.write(msg + '\n')
        except Exception:
            pass


def format_time(seconds):
    """Format seconds to H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def log_jsonl(record):
    """Append a JSON-lines record to the JSONL log file."""
    if not _jsonl_logger:
        return
    try:
        line = json.dumps(record, default=str)
        _jsonl_logger.write(line + '\n')
    except Exception:
        pass


def check_nan_inf(metrics_dict, global_step, epoch):
    """Check for NaN/INF in loss metrics. Returns True if NaN/INF detected."""
    total = metrics_dict.get('total_loss', 0.0)
    if np.isnan(total) or np.isinf(total):
        if jax.process_index() == 0:
            print(f"\n[WARNING] NaN/INF detected at step {global_step}!")
            print(f"  total_loss: {total}")
            print(f"  ce_loss:    {metrics_dict.get('ce_loss', 'N/A')}")
            print(f"  aux_loss:   {metrics_dict.get('aux_loss', 'N/A')}")
            print(f"  tau_reg:    {metrics_dict.get('tau_reg', 'N/A')}")
            print(f"  orth_loss:  {metrics_dict.get('orth_loss', 'N/A')}")
            print(f"  div_loss:   {metrics_dict.get('div_loss', 'N/A')}")
        return True
    return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train DAWN v17.1 (JAX/Flax, Multi-Device)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config (global)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: log every step with detailed metrics')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from specific run folder path (e.g. gs://...../run_v...)')
    cli_args = parser.parse_args()

    # ----------------------------------------------------------
    # Load config
    # ----------------------------------------------------------
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        # Try as absolute path (or GCS)
        if _file_exists(cli_args.config):
            config_path = cli_args.config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_config(config_path)

    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Training params (from YAML first, may be overridden by checkpoint config below)
    debug_mode = cli_args.debug
    tcfg = cfg['training']
    batch_size = cli_args.batch_size or tcfg['batch_size']  # global batch size
    num_epochs = cli_args.epochs or tcfg['num_epochs']
    lr = cli_args.lr or tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4))
    weight_decay = tcfg.get('weight_decay', 0.1)
    warmup_ratio = tcfg.get('warmup_ratio', 0.06)
    orth_weight = tcfg.get('orthogonality_weight', 0.01)
    div_weight = tcfg.get('diversity_weight', 0.1)
    lb_weight = tcfg.get('load_balance_weight', 2e-5)
    tau_reg_weight = tcfg.get('tau_reg_weight', 0.0)
    dead_penalty_weight = tcfg.get('dead_penalty_weight', 0.0)  # v4.0.6
    # v4.1 RPE exploration loss (0 weight => off; no-op for earlier versions).
    exploration_weight = tcfg.get('exploration_weight', 0.0)
    exploration_asymmetry = tcfg.get('exploration_asymmetry', 0.15)

    max_seq_len = cfg['model'].get('max_seq_len', 512)

    base_checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints_jax')
    _makedirs(base_checkpoint_dir)

    # ----------------------------------------------------------
    # Run folder: base_checkpoint_dir/run_v{version}_{timestamp}_{rand}/
    # All checkpoints + logs go in the same run folder (like train.py).
    # ----------------------------------------------------------
    resume_path = None
    checkpoint_dir = None  # will be set to a run folder

    def _join(base, name):
        if _is_gcs(base):
            return base.rstrip('/') + '/' + name
        return str(Path(base) / name)

    def _list_run_folders(base):
        """List run_* subdirectories under base (local or GCS)."""
        if _is_gcs(base):
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                prefix = base.rstrip('/') + '/run_'
                # strip gs:// for gcsfs
                bucket_path = base.replace('gs://', '').rstrip('/')
                entries = fs.ls(bucket_path)
                runs = sorted([
                    'gs://' + e for e in entries
                    if '/run_' in e
                ])
                return runs
            except Exception:
                return []
        else:
            p = Path(base)
            if not p.exists():
                return []
            return sorted([
                str(d) for d in p.iterdir()
                if d.is_dir() and d.name.startswith('run_')
            ])

    # Auto-resume: find latest run folder with checkpoints (unless --from-scratch)
    # --resume-from takes priority: resume from a specific run folder
    if not cli_args.from_scratch:
        if cli_args.resume_from:
            folder = cli_args.resume_from.rstrip('/')
            candidates = _list_files(folder, "*.flax")
            if candidates:
                resume_path = candidates[-1]
                checkpoint_dir = folder
                if jax.process_index() == 0:
                    print(f"  Resume from specified folder: {checkpoint_dir}")
                    print(f"  Resuming from: {resume_path}")
            else:
                raise FileNotFoundError(f"No .flax checkpoint found in {folder}")
        else:
            run_folders = _list_run_folders(base_checkpoint_dir)
            for folder in reversed(run_folders):
                candidates = _list_files(folder, "*.flax")
                if candidates:
                    resume_path = candidates[-1]
                    checkpoint_dir = folder
                    if jax.process_index() == 0:
                        print(f"  Auto-resume: found checkpoint in {checkpoint_dir}")
                        print(f"  Resuming from: {resume_path}")
                    break

    # Create new run folder if not resuming
    if checkpoint_dir is None:
        import random as _random
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        ts = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        rand_suffix = _random.randint(1000, 9999)
        version = cfg['model'].get('model_version', 'v17.1')
        run_name = f"run_v{version}_{ts}_{rand_suffix}"
        checkpoint_dir = _join(base_checkpoint_dir, run_name)
        _makedirs(checkpoint_dir)
        if jax.process_index() == 0:
            if cli_args.from_scratch:
                print(f"  Starting from scratch (--from-scratch)")
            print(f"  Created new run folder: {checkpoint_dir}")

    log_dir = checkpoint_dir  # logs go in same run folder

    # ----------------------------------------------------------
    # Resume config override: load training config from checkpoint
    # ----------------------------------------------------------
    if resume_path and _file_exists(resume_path):
        # Try config.json in run folder
        config_json_path = _join(checkpoint_dir, 'config.json')

        saved_training_config = None
        if _file_exists(config_json_path):
            try:
                with _open_file(config_json_path, 'r') as f:
                    content = f.read()
                saved_cfg = json.loads(content)
                saved_training_config = saved_cfg.get('training')
                if jax.process_index() == 0:
                    print(f"  Loaded training config from {config_json_path}")
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"  Warning: Failed to read config.json: {e}")
                saved_cfg = None

        if saved_training_config:
            # Apply checkpoint training config (CLI args take precedence)
            if cli_args.batch_size is None:
                batch_size = saved_training_config.get('batch_size', batch_size)
            if cli_args.epochs is None:
                num_epochs = saved_training_config.get('num_epochs', num_epochs)
            if cli_args.lr is None:
                lr = saved_training_config.get('lr', lr)
            weight_decay = saved_training_config.get('weight_decay', weight_decay)
            warmup_ratio = saved_training_config.get('warmup_ratio', warmup_ratio)
            orth_weight = saved_training_config.get('orthogonality_weight', orth_weight)
            div_weight = saved_training_config.get('diversity_weight', div_weight)
            lb_weight = saved_training_config.get('load_balance_weight', lb_weight)
            tau_reg_weight = saved_training_config.get('tau_reg_weight', tau_reg_weight)
            dead_penalty_weight = saved_training_config.get('dead_penalty_weight', dead_penalty_weight)
            exploration_weight = saved_training_config.get(
                'exploration_weight', exploration_weight)
            exploration_asymmetry = saved_training_config.get(
                'exploration_asymmetry', exploration_asymmetry)
            if jax.process_index() == 0:
                print(f"  Training config restored from checkpoint (CLI overrides take precedence)")

    # Build training_config dict for saving in checkpoints
    training_config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio,
        'orthogonality_weight': orth_weight,
        'diversity_weight': div_weight,
        'load_balance_weight': lb_weight,
        'tau_reg_weight': tau_reg_weight,
        'dead_penalty_weight': dead_penalty_weight,
        'exploration_weight': exploration_weight,
        'exploration_asymmetry': exploration_asymmetry,
    }

    # ----------------------------------------------------------
    # Detect devices (multi-host aware)
    # ----------------------------------------------------------
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    is_host0 = (host_id == 0)
    n_local_devices = jax.local_device_count()
    local_devices = jax.local_devices()

    # ALL hosts print device info (for multi-host debugging)
    print(f"[Host {host_id}/{n_hosts}] "
          f"local_devices={n_local_devices} total_devices={jax.device_count()} "
          f"backend={jax.default_backend()} "
          f"devices={[str(d) for d in local_devices]}", flush=True)

    per_host_batch = batch_size // n_hosts
    per_device_batch = per_host_batch // n_local_devices

    assert batch_size % n_hosts == 0, (
        f"Global batch_size ({batch_size}) must be divisible by n_hosts ({n_hosts})"
    )
    assert per_host_batch % n_local_devices == 0, (
        f"per_host_batch ({per_host_batch}) must be divisible by "
        f"n_local_devices ({n_local_devices})"
    )

    if is_host0:
        print(f"\n{'='*60}")
        print(f"DAWN Training (Multi-Host Multi-Device) -- {cfg['model'].get('model_version', 'unknown')}")
        print(f"{'='*60}")
        print(f"JAX version: {jax.__version__}")
        print(f"Hosts: {n_hosts}, Host ID: {host_id}")
        print(f"Local devices: {local_devices}")
        print(f"Local device count: {n_local_devices}")
        print(f"Total device count: {jax.device_count()}")
        print(f"Backend: {jax.default_backend()}")
        print(f"Config: {config_path}")
        print(f"Run folder: {checkpoint_dir}")
        print(f"Seed: {seed}")
        print(f"Global batch size: {batch_size}")
        print(f"Per-host batch size: {per_host_batch}")
        print(f"Per-device batch size: {per_device_batch}")

    # ----------------------------------------------------------
    # Load data (multi-host: each host loads its own data slice)
    # ----------------------------------------------------------
    if is_host0:
        print(f"\n{'='*60}")
        print("Loading data...")
        print(f"{'='*60}")

    from utils.data_jax import load_data
    train_loader, val_loader, vocab_size = load_data(
        cfg['data'],
        max_length=max_seq_len,
        batch_size=batch_size,
        n_devices=1,  # flat (per_host_batch, seq_len) — shard_to_mesh handles splitting
        n_hosts=n_hosts,
        host_id=host_id,
    )
    if is_host0:
        print(f"Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    cfg['model']['vocab_size'] = vocab_size
    model = build_model_from_config(cfg)

    # Initialize
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones((1, max_seq_len), dtype=jnp.int32)

    if is_host0:
        print("=== Starting model.init ===", flush=True)
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        dummy_input,
        deterministic=True,
    )
    params = variables['params']
    if is_host0:
        print("=== model.init done ===", flush=True)

        n_params = count_parameters(params)
        print(f"\nModel parameters: {n_params:,}")
        for line in model.get_model_info():
            print(line)

    rank = cfg['model'].get('rank', 64)
    knowledge_rank = cfg['model'].get('knowledge_rank', 128)

    # ----------------------------------------------------------
    # Optimizer (warmup + cosine decay + optional gradient accumulation)
    # ----------------------------------------------------------
    grad_accum_steps = tcfg.get('gradient_accumulation_steps', 1)

    steps_per_epoch = len(train_loader)
    # Schedule counts optimizer steps (after accumulation), not micro-steps
    effective_steps_per_epoch = steps_per_epoch // grad_accum_steps
    total_steps = num_epochs * effective_steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    # WD mask: exclude bias, layernorm, and output_scale params
    # Only apply mask if model has learnable scale params (v3.9.7.1+)
    _has_scale = 'qk_scale' in params.get('neuron_pool', {}) or \
                 'know_scale' in params.get('neuron_pool', {})
    _wd_mask_fn = None
    if _has_scale:
        def _wd_mask(params):
            def _should_decay(path, _):
                path_str = '/'.join(str(p) for p in path)
                if 'bias' in path_str:
                    return False
                if 'scale' in path_str and 'norm' in path_str.lower():
                    return False  # LayerNorm scale
                if path_str.endswith('_scale') or path_str.endswith('/qk_scale') \
                   or path_str.endswith('/v_scale') or path_str.endswith('/know_scale'):
                    return False  # learnable output_scale
                return True
            return jax.tree.map_with_path(_should_decay, params)
        _wd_mask_fn = _wd_mask

    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, b2=0.95,
                    mask=_wd_mask_fn),
    )

    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum_steps)
    else:
        optimizer = base_optimizer
    opt_state = optimizer.init(params)

    if is_host0:
        print(f"\nTraining config:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Global batch size: {batch_size}")
        print(f"  Per-host batch size: {per_host_batch}")
        print(f"  Per-device batch size: {per_device_batch}")
        print(f"  Hosts: {n_hosts}")
        print(f"  Local devices: {n_local_devices}")
        print(f"  Total devices: {jax.device_count()}")
        print(f"  Grad accum steps: {grad_accum_steps}")
        print(f"  Effective batch size: {batch_size * grad_accum_steps}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total optimizer steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  LR: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Orth weight: {orth_weight}")
        print(f"  Div weight: {div_weight}")
        print(f"  LB weight: {lb_weight}")
        print(f"  Tau reg weight: {tau_reg_weight}")
        print(f"  Dead penalty weight: {dead_penalty_weight}")
        print(f"  Exploration weight: {exploration_weight} "
              f"(asymmetry={exploration_asymmetry})")

    # ----------------------------------------------------------
    # Resume from checkpoint (resume_path detected earlier for config override)
    # ----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    start_step_in_epoch = 0
    best_val_loss = float('inf')

    if resume_path and _file_exists(resume_path):
        if is_host0:
            print(f"\nResuming from: {resume_path}")
        ckpt = load_checkpoint(resume_path, params, opt_state)
        params = ckpt['params']
        opt_state = ckpt['opt_state']
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        # v4.1 redesign removed EMA state; silently ignore any ema_ce left
        # in older checkpoints.
        # Precise resume: use step_in_epoch if available
        saved_step_in_epoch = ckpt.get('step_in_epoch', 0)
        saved_steps_per_epoch = ckpt.get('steps_per_epoch', 0)
        if saved_step_in_epoch > 0 and saved_steps_per_epoch == steps_per_epoch:
            start_step_in_epoch = saved_step_in_epoch
        elif saved_step_in_epoch > 0:
            # steps_per_epoch changed (batch size or data changed) — fallback
            if is_host0:
                print(f"  Warning: steps_per_epoch changed ({saved_steps_per_epoch} -> {steps_per_epoch}), "
                      f"cannot use step_in_epoch for resume. Starting epoch from beginning.")
            start_step_in_epoch = 0
        if is_host0:
            print(f"  Resuming: epoch={start_epoch}, global_step={global_step}, "
                  f"step_in_epoch={start_step_in_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        if is_host0:
            if not cli_args.from_scratch:
                print("\nNo checkpoint found. Starting from scratch.")
            else:
                print("\nStarting from scratch (--from-scratch).")

    # Save config.json for this run (host 0 only)
    if is_host0:
        try:
            cj_path = _join(checkpoint_dir, 'config.json')
            full_cfg = {'model': cfg['model'], 'training': training_config}
            with _open_file(cj_path, 'w') as f:
                f.write(json.dumps(full_cfg, indent=2, default=str))
            print(f"  Saved config.json: {cj_path}")
        except Exception as e:
            print(f"  Warning: Failed to save config.json: {e}")

    # ----------------------------------------------------------
    # Create Mesh + shard params
    # ----------------------------------------------------------
    n_feature_qk = cfg['model'].get('n_feature_qk', 56)
    n_restore_qk = cfg['model'].get('n_restore_qk', 56)
    model_version = cfg['model'].get('model_version', '17.1')
    is_baseline = model_version == 'baseline'
    is_spatial = (model_version == 'spatial-r1'
                  or model_version.startswith('spatial-r1-v2')
                  or model_version.startswith('spatial-r1-v3')
                  or model_version.startswith('spatial-r1-v4')
                  or model_version.startswith('rw-v'))

    mesh_model = cfg['training'].get('mesh_model', 1)
    mesh_data = cfg['training'].get('mesh_data', 0)  # 0 = auto
    total_devices = jax.device_count()
    if mesh_data == 0:
        mesh_data = total_devices // mesh_model

    mesh = create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    per_device_batch = batch_size // total_devices

    # Auto n_chunks: target ~2GB per chunk (bf16)
    def auto_n_chunks(N, target_gb=2.0):
        full_gb = per_device_batch * max_seq_len * N * 2 / 1e9  # bf16
        nc = max(1, int(np.ceil(full_gb / target_gb)))
        while N % nc != 0 and nc < N:
            nc += 1
        return min(nc, N)

    target_chunk_gb = cfg['training'].get('target_chunk_gb', 2.0)
    n_know = cfg['model'].get('n_know', 25200)
    n_qk = cfg['model'].get('n_qk', 1580)
    n_v = cfg['model'].get('n_v', 2600)
    # N_local = N / mesh_model (each chip's share)
    nk_local = n_know // mesh_model
    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model

    n_chunks_know = cfg['training'].get('n_chunks_know',
                                         auto_n_chunks(nk_local, target_chunk_gb))
    n_chunks_qk = cfg['training'].get('n_chunks_qk',
                                       auto_n_chunks(nqk_local, target_chunk_gb))
    n_chunks_v = cfg['training'].get('n_chunks_v',
                                      auto_n_chunks(nv_local, target_chunk_gb))

    if is_host0:
        print(f"\n=== Mesh: ({mesh_data}, {mesh_model}) = "
              f"{total_devices} devices, per_device_batch={per_device_batch} ===")
        print(f"  Chunks: know={n_chunks_know} (cs={nk_local // max(n_chunks_know,1)}), "
              f"qk={n_chunks_qk}, v={n_chunks_v}")
        chunk_mem = per_device_batch * max_seq_len * (nk_local // max(n_chunks_know,1)) * 2 / 1e9
        print(f"  Est chunk mem (know): {chunk_mem:.2f}GB bf16")

    # Shard params: neuron_pool N-axis on 'model', rest replicated
    param_shardings = get_param_shardings(params, mesh, is_baseline=is_baseline)
    params = shard_params_to_mesh(params, param_shardings)

    _is_resuming = (resume_path is not None and _file_exists(resume_path))
    if _is_resuming:
        _opt_template = optimizer.init(params)
        def _restore_leaf(restored_val, template_val):
            # template_val is already correctly sharded across all devices.
            # Create a zero with the same sharding, then add restored value.
            # This forces the result to inherit template's sharding.
            return jnp.zeros_like(template_val) + jnp.asarray(restored_val, dtype=template_val.dtype).reshape(template_val.shape)
        opt_state = jax.tree.map(_restore_leaf, opt_state, _opt_template)
        del _opt_template
        if is_host0:
            print(f"  Optimizer state restored from checkpoint and sharded to mesh")
    else:
        opt_state = optimizer.init(params)

    # Create shard_map functions if mesh_model > 1 (or always for v4.0.4
    # which removed its non-sharded fallback and depends on the sharded path).
    _sharded_fns = None
    _force_sharded = model_version in ('spatial-r1-v4.0.4', 'spatial-r1-v4.0.5',
                                        'spatial-r1-v4.0.6', 'spatial-r1-v4.1')
    if mesh_model > 1 or _force_sharded:
        _v3_mod = {'spatial-r1-v3.9.1': 'models.dawn_spatial_v3_baseline', 'spatial-r1-v3.9.3': 'models.dawn_spatial_v3_exp', 'spatial-r1-v3.9.4': 'models.dawn_spatial_v394_exp', 'spatial-r1-v3.9.5': 'models.dawn_spatial_v395_exp', 'spatial-r1-v3.9.6': 'models.dawn_spatial_v396_exp', 'spatial-r1-v3.9.7': 'models.dawn_spatial_v397_exp', 'spatial-r1-v3.9.7.1': 'models.dawn_spatial_v3971_exp', 'spatial-r1-v3.9.8': 'models.dawn_spatial_v398_exp', 'spatial-r1-v3.9.8.1': 'models.dawn_spatial_v3981_exp', 'spatial-r1-v3.9.9': 'models.dawn_spatial_v399_exp', 'spatial-r1-v4.0.0': 'models.dawn_spatial_v400_exp', 'spatial-r1-v4.0.1': 'models.dawn_spatial_v401_exp', 'rw-v4.0.2': 'models.dawn_spatial_v402_exp', 'spatial-r1-v4.0.3': 'models.dawn_spatial_v403_exp', 'spatial-r1-v4.0.4': 'models.dawn_spatial_v404_exp', 'spatial-r1-v4.0.5': 'models.dawn_spatial_v405_exp', 'spatial-r1-v4.0.6': 'models.dawn_spatial_v406_exp', 'spatial-r1-v4.1': 'models.dawn_spatial_v41_exp'}.get(model_version, 'models.dawn_spatial_v3')
        _v3 = __import__(_v3_mod, fromlist=['make_sharded_srw'])
        make_sharded_srw = _v3.make_sharded_srw
        max_chunk = cfg['training'].get('max_chunk_size', 12500)
        _gnm = cfg['model'].get('gate_norm_mode', 'sqrt_active')
        _srw_kwargs = {'mesh': mesh, 'max_chunk_size': max_chunk}
        if model_version.startswith('spatial-r1-v3.9.5'):
            _srw_kwargs['gate_norm_mode'] = _gnm
        # v4.0.4: reverse_p_max is a runtime scalar passed through shard_map
        # at each call (see _attn_forward / _know_forward); not a builder kwarg.
        # v4.0.6: dead_threshold is a builder kwarg (closure constant).
        if model_version in ('spatial-r1-v4.0.6', 'spatial-r1-v4.1'):
            _srw_kwargs['dead_threshold'] = cfg['training'].get(
                'dead_penalty_threshold',
                0.01 if model_version == 'spatial-r1-v4.1' else 1e-4)
        # v4.1: sharpness / activation_threshold / activation_cutoff /
        # epsilon / max_intensity — all closure constants for the
        # two-stage gate.
        if model_version == 'spatial-r1-v4.1':
            _srw_kwargs['sharpness'] = cfg['training'].get('sharpness', 500.0)
            _srw_kwargs['activation_threshold'] = cfg['training'].get(
                'activation_threshold', 0.5)
            _srw_kwargs['activation_cutoff'] = cfg['training'].get(
                'activation_cutoff', 0.01)
            _srw_kwargs['epsilon'] = cfg['training'].get('epsilon', 1e-4)
            _srw_kwargs['max_intensity'] = cfg['training'].get(
                'max_intensity', 10.0)
        _sharded_single = make_sharded_srw(**_srw_kwargs)
        if hasattr(_v3, 'make_sharded_srw_paired'):
            _sharded_paired = _v3.make_sharded_srw_paired(**_srw_kwargs)
            _sharded_fns = (_sharded_single, _sharded_paired)
        else:
            _sharded_fns = _sharded_single  # v4.0.2: no paired (Q/K split)
        if is_host0:
            print(f"  shard_map enabled (mesh_model={mesh_model}, QK fused)")

    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        tau_reg_weight, dead_penalty_weight,
        exploration_weight, exploration_asymmetry,
        rank, knowledge_rank, n_feature_qk, n_restore_qk,
        is_baseline=is_baseline, is_spatial=is_spatial,
        sharded_fns=_sharded_fns, mesh=mesh)
    eval_step_fn = create_eval_step(model, sharded_fns=_sharded_fns)

    # ----------------------------------------------------------
    # OOM check + JIT pre-compile
    # ----------------------------------------------------------
    if is_host0:
        print(f"\n=== OOM check: real train_step (forward+backward) "
              f"per_device_batch={per_device_batch}, seq_len={max_seq_len} ===", flush=True)
    try:
        global_shape = (batch_size, max_seq_len)
        dummy_ids = shard_to_mesh(
            jnp.zeros((per_host_batch, max_seq_len), dtype=jnp.int32),
            data_sharding, global_shape)
        dummy_mask = shard_to_mesh(
            jnp.ones((per_host_batch, max_seq_len), dtype=jnp.int32),
            data_sharding, global_shape)
        rng, dummy_step_rng = jax.random.split(rng)

        # First call: JIT compilation (slow)
        jit_start = time.time()
        _dp, _do, dummy_metrics = train_step_fn(
            params, opt_state, dummy_ids, dummy_mask, dummy_step_rng)
        jax.block_until_ready(dummy_metrics['total_loss'])
        jit_time = time.time() - jit_start
        jit_loss = float(dummy_metrics['total_loss'])
        if is_host0:
            print(f"  JIT compile: {jit_time:.1f}s", flush=True)

        # Free first step outputs before second call
        del _dp, _do, dummy_metrics

        # Second call: measure actual step time (post-JIT)
        rng, dummy_step_rng2 = jax.random.split(rng)
        step_start = time.time()
        _dp2, _do2, dummy_metrics2 = train_step_fn(
            params, opt_state, dummy_ids, dummy_mask, dummy_step_rng2)
        jax.block_until_ready(dummy_metrics2['total_loss'])
        step_time = time.time() - step_start
        if is_host0:
            print(f"  train_step OK -- loss={jit_loss:.4f}", flush=True)
            print(f"  Step time: {step_time*1000:.1f}ms/batch", flush=True)

            # Show memory usage after JIT compilation
            try:
                mem = jax.local_devices()[0].memory_stats()
                if mem:
                    used = mem.get('bytes_in_use', 0) / 1e9
                    peak = mem.get('peak_bytes_in_use', 0) / 1e9
                    limit = mem.get('bytes_limit', 0) / 1e9
                    print(f"  HBM: {used:.2f}G / {limit:.2f}G (peak={peak:.2f}G, free={limit - used:.2f}G)", flush=True)
            except Exception:
                pass

        # === Step-time breakdown (sharded, 1 layer) ===
        # NOTE: runs on ALL hosts — shard_map/psum require collective participation.
        # Only print statements are guarded by is_host0.
        try:
            _is_sharded = _sharded_fns is not None
            if is_host0:
                print(f"\n  === Step-time breakdown (1 layer, "
                      f"{'sharded' if _is_sharded else 'single-device'}) ===",
                      flush=True)

            _v3_mod = {'spatial-r1-v3.9.1': 'models.dawn_spatial_v3_baseline', 'spatial-r1-v3.9.3': 'models.dawn_spatial_v3_exp', 'spatial-r1-v3.9.4': 'models.dawn_spatial_v394_exp', 'spatial-r1-v3.9.5': 'models.dawn_spatial_v395_exp', 'spatial-r1-v3.9.6': 'models.dawn_spatial_v396_exp', 'spatial-r1-v3.9.7': 'models.dawn_spatial_v397_exp', 'spatial-r1-v3.9.7.1': 'models.dawn_spatial_v3971_exp', 'spatial-r1-v3.9.8': 'models.dawn_spatial_v398_exp', 'spatial-r1-v3.9.8.1': 'models.dawn_spatial_v3981_exp', 'spatial-r1-v3.9.9': 'models.dawn_spatial_v399_exp', 'spatial-r1-v4.0.0': 'models.dawn_spatial_v400_exp', 'spatial-r1-v4.0.1': 'models.dawn_spatial_v401_exp', 'rw-v4.0.2': 'models.dawn_spatial_v402_exp', 'spatial-r1-v4.0.3': 'models.dawn_spatial_v403_exp', 'spatial-r1-v4.0.4': 'models.dawn_spatial_v404_exp', 'spatial-r1-v4.0.5': 'models.dawn_spatial_v405_exp', 'spatial-r1-v4.0.6': 'models.dawn_spatial_v406_exp', 'spatial-r1-v4.1': 'models.dawn_spatial_v41_exp'}.get(model_version, 'models.dawn_spatial_v3')
            _v3 = __import__(_v3_mod, fromlist=['_layer_norm', '_attn_forward', '_know_forward', '_srw_chunked'])
            _layer_norm, _attn_forward, _know_forward, _srw_chunked = _v3._layer_norm, _v3._attn_forward, _v3._know_forward, _v3._srw_chunked

            # Use actual sharded params (no device_get)
            pool_p = params['neuron_pool']
            router_p = params['router']
            block_p = params['block_0']
            d_model = cfg['model']['d_model']
            n_heads = cfg['model']['n_heads']
            n_qk_cfg = cfg['model'].get('n_qk', 1580)
            n_v_cfg = cfg['model'].get('n_v', 2620)
            rd = cfg['model'].get('router_dropout', 0.1)
            dd = cfg['model'].get('dropout', 0.1)
            prof_rng = jax.random.PRNGKey(42)

            # Create properly sharded dummy_x [B, S, D]
            dummy_x_local = jnp.zeros(
                (per_host_batch, max_seq_len, d_model), dtype=jnp.float32)
            x_sharding = NamedSharding(mesh, P('data', None, None))
            global_x_shape = (batch_size, max_seq_len, d_model)
            dummy_x = shard_to_mesh(dummy_x_local, x_sharding, global_x_shape)

            N_RUNS = 5

            def _hbm_gb():
                """Current HBM usage in GB (device 0)."""
                try:
                    mem = jax.local_devices()[0].memory_stats()
                    if mem:
                        return mem.get('bytes_in_use', 0) / 1e9
                except Exception:
                    pass
                return 0.0

            def _peak_hbm_gb():
                """Peak HBM usage in GB (device 0)."""
                try:
                    mem = jax.local_devices()[0].memory_stats()
                    if mem:
                        return mem.get('peak_bytes_in_use', 0) / 1e9
                except Exception:
                    pass
                return 0.0

            def _t(fn, n=N_RUNS):
                """Time a function + measure HBM delta.
                Returns (ms, delta_gb, peak_gb)."""
                r = fn(); jax.block_until_ready(jax.tree.leaves(r))
                del r
                hbm_before = _hbm_gb()
                t0 = time.time()
                for _ in range(n):
                    r = fn(); jax.block_until_ready(jax.tree.leaves(r))
                elapsed = (time.time() - t0) / n * 1000
                hbm_after = _hbm_gb()
                peak = _peak_hbm_gb()
                return (elapsed, hbm_after - hbm_before, peak)

            # --- Jit-compiled component functions for profiling ---

            # 1) LayerNorm
            @jax.jit
            def prof_layernorm(x, scale, bias):
                return _layer_norm(x, scale, bias)

            # 2) Attn router: proj + split + tau
            @jax.jit
            def prof_attn_router(x, router_p):
                h_all = (x @ router_p['proj_attn']['kernel']
                         + router_p['proj_attn']['bias'])
                h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
                h_Q = h_Q / (jnp.linalg.norm(h_Q, axis=-1, keepdims=True) + 1e-8)
                h_K = h_K / (jnp.linalg.norm(h_K, axis=-1, keepdims=True) + 1e-8)
                h_V = h_V / (jnp.linalg.norm(h_V, axis=-1, keepdims=True) + 1e-8)
                tau_all = (x @ router_p['tau_attn']['kernel']
                           + router_p['tau_attn']['bias'])
                return h_Q, h_K, h_V, tau_all

            # 3) QK fused shard_map (paired)
            @jax.jit
            def prof_qk_fused(x, h_Q, h_K, qk_norm, tau_all, qk_read, qk_write):
                fused_paired = _sharded_fns[1]
                h_QK = jnp.stack([h_Q, h_K], axis=2)
                tau_QK = jnp.stack(
                    [tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
                results = fused_paired(
                    x, h_QK, qk_norm, tau_QK, qk_read, qk_write)
                QK_out, act = results[0], results[1]
                return QK_out[:, :, 0, :], QK_out[:, :, 1, :], act

            # 3b) QK non-sharded fallback
            @jax.jit
            def prof_qk_chunked(x, h_Q, h_K, qk_norm, tau_all, qk_read, qk_write):
                Q, *_ = _srw_chunked(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                                       qk_read, qk_write, n_chunks_qk)
                K, *_ = _srw_chunked(x, h_K, qk_norm, tau_all[:, :, 1:2],
                                       qk_read, qk_write, n_chunks_qk)
                return Q, K

            # 4) V shard_map (single)
            @jax.jit
            def prof_v_sharded(x, h_V, v_norm, tau_v, v_read, v_write):
                fused_single = _sharded_fns[0]
                return fused_single(
                    x, h_V, v_norm, tau_v, v_read, v_write)

            # 4b) V non-sharded fallback
            @jax.jit
            def prof_v_chunked(x, h_V, v_norm, tau_v, v_read, v_write):
                return _srw_chunked(x, h_V, v_norm, tau_v,
                                    v_read, v_write, n_chunks_v)

            # 5) Self-attention (QK scores + softmax + wV + O_proj)
            @jax.jit
            def prof_self_attn(Q, K, V, Ok):
                B, S, D = Q.shape
                dh = D // n_heads
                Qr = Q.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                Kr = K.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                Vr = V.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                sc = jnp.sqrt(jnp.float32(dh))
                scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / sc
                causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
                scores = jnp.where(causal, scores,
                                   jnp.finfo(scores.dtype).min)
                attn_w = jax.nn.softmax(scores, axis=-1)
                out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
                out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
                return out @ Ok

            # 6) Know router
            @jax.jit
            def prof_know_router(x, router_p):
                h = (x @ router_p['proj_know']['kernel']
                     + router_p['proj_know']['bias'])
                h = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
                tau = (x @ router_p['tau_know']['kernel']
                       + router_p['tau_know']['bias'])
                return h, tau

            # 7) Know shard_map (single)
            @jax.jit
            def prof_know_sharded(x, h, know_norm, tau, know_read, know_write):
                fused_single = _sharded_fns[0]
                return fused_single(
                    x, h, know_norm, tau, know_read, know_write)

            # 7b) Know non-sharded fallback
            @jax.jit
            def prof_know_chunked(x, h, know_norm, tau, know_read, know_write):
                return _srw_chunked(x, h, know_norm, tau,
                                    know_read, know_write, n_chunks_know)

            # --- Prepare intermediate values ---
            qk_emb = pool_p['qk_emb']
            v_emb = pool_p['v_emb']
            know_emb = pool_p['know_emb']
            qk_norm = qk_emb / (jnp.linalg.norm(
                qk_emb, axis=-1, keepdims=True) + 1e-8)
            v_norm = v_emb / (jnp.linalg.norm(
                v_emb, axis=-1, keepdims=True) + 1e-8)
            know_norm = know_emb / (jnp.linalg.norm(
                know_emb, axis=-1, keepdims=True) + 1e-8)

            normed = prof_layernorm(
                dummy_x, block_p['norm1']['scale'],
                block_p['norm1']['bias'])
            jax.block_until_ready(normed)

            h_Q, h_K, h_V, tau_all = prof_attn_router(normed, router_p)
            jax.block_until_ready(tau_all)

            if _is_sharded:
                Q, K, *_ = prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, *_ = prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write'])
            else:
                Q, K = prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, *_ = prof_v_chunked(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write'])
            jax.block_until_ready((Q, K, V))

            h_know, tau_know = prof_know_router(normed, router_p)
            jax.block_until_ready(tau_know)
            if _is_sharded:
                _kout, _, _, _, _, _, _, _ = prof_know_sharded(
                    normed, h_know, know_norm, tau_know,
                    pool_p['know_read'], pool_p['know_write'])
            else:
                _kout, _, _, _, _, _, _, _ = prof_know_chunked(
                    normed, h_know, know_norm, tau_know,
                    pool_p['know_read'], pool_p['know_write'])
            jax.block_until_ready(_kout)

            # --- Timed + memory measurements ---
            # Each _t() returns (ms, delta_hbm_gb, peak_hbm_gb)
            hbm_baseline = _hbm_gb()
            items = []  # [(name, ms, delta_gb, peak_gb)]

            ms, dg, pk = _t(lambda: prof_layernorm(
                dummy_x, block_p['norm1']['scale'],
                block_p['norm1']['bias']))
            items.append(("LayerNorm", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_attn_router(normed, router_p))
            items.append(("A router(proj+tau)", ms, dg, pk))

            if _is_sharded:
                ms, dg, pk = _t(lambda: prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write']))
                items.append(("A QK fused shard", ms, dg, pk))
                ms, dg, pk = _t(lambda: prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write']))
                items.append(("A V shard", ms, dg, pk))
            else:
                ms, dg, pk = _t(lambda: prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write']))
                items.append(("A QK chunked(x2)", ms, dg, pk))
                ms, dg, pk = _t(lambda: prof_v_chunked(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write']))
                items.append(("A V chunked", ms, dg, pk))

            Ok = block_p['attn']['expand_O']['kernel']
            ms, dg, pk = _t(lambda: prof_self_attn(Q, K, V, Ok))
            items.append(("A self-attn(QKV)", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_layernorm(
                dummy_x, block_p['norm2']['scale'],
                block_p['norm2']['bias']))
            items.append(("LayerNorm (know)", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_know_router(normed, router_p))
            items.append(("K router(proj+tau)", ms, dg, pk))

            if _is_sharded:
                ms, dg, pk = _t(lambda: prof_know_sharded(
                    normed, h_know, know_norm, tau_know,
                    pool_p['know_read'], pool_p['know_write']))
                items.append(("K know shard", ms, dg, pk))
            else:
                ms, dg, pk = _t(lambda: prof_know_chunked(
                    normed, h_know, know_norm, tau_know,
                    pool_p['know_read'], pool_p['know_write']))
                items.append(("K know chunked", ms, dg, pk))

            # --- Print breakdown (time + memory) --- host0 only
            if is_host0:
                total_ms = sum(ms for _, ms, _, _ in items)
                max_peak = max(pk for _, _, _, pk in items)
                n_layers = cfg['model']['n_layers']

                try:
                    mem = jax.local_devices()[0].memory_stats()
                    hbm_limit = mem.get('bytes_limit', 0) / 1e9 if mem else 0
                except Exception:
                    hbm_limit = 0

                print(f"\n  === Op breakdown (1 layer fwd, {total_ms:.0f} ms, "
                      f"peak={max_peak:.2f}G) ===", flush=True)
                print(f"    {'Op':22s} {'Time':>8s} {'%':>5s}  "
                      f"{'HBM d':>7s}  {'Peak':>7s}  {''}",
                      flush=True)
                print(f"    {'-'*22} {'-'*8} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*20}",
                      flush=True)
                for name, ms_val, dg_val, pk_val in items:
                    pct = ms_val / total_ms * 100 if total_ms > 0 else 0
                    bar = '#' * int(pct / 2)
                    dg_str = f"{dg_val:+.3f}G" if abs(dg_val) > 0.001 else "     -"
                    print(f"    {name:22s} {ms_val:7.1f}ms {pct:4.0f}%  "
                          f"{dg_str:>7s}  {pk_val:5.2f}G  {bar}",
                          flush=True)

                # Group summaries
                attn_ms = sum(ms for n, ms, _, _ in items if n.startswith('A '))
                know_ms = sum(ms for n, ms, _, _ in items if n.startswith('K '))
                norm_ms = sum(ms for n, ms, _, _ in items if n.startswith('LayerNorm'))
                print(f"    {'-'*22} {'-'*8}", flush=True)
                print(f"    {'Attention total':22s} {attn_ms:7.1f}ms "
                      f"{attn_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'Knowledge total':22s} {know_ms:7.1f}ms "
                      f"{know_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'LayerNorm total':22s} {norm_ms:7.1f}ms "
                      f"{norm_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'Layer total':22s} {total_ms:7.1f}ms", flush=True)
                print(f"    Est. {n_layers}-layer fwd: "
                      f"{total_ms * n_layers:.0f} ms "
                      f"(actual step incl. grad+opt)", flush=True)

                # Overall HBM summary
                hbm_now = _hbm_gb()
                print(f"\n  === HBM Summary (per device) ===", flush=True)
                print(f"    Baseline (params+opt):  {hbm_baseline:.2f}G",
                      flush=True)
                print(f"    After profile:          {hbm_now:.2f}G",
                      flush=True)
                print(f"    Peak during profile:    {max_peak:.2f}G",
                      flush=True)
                if hbm_limit > 0:
                    print(f"    Device limit:           {hbm_limit:.2f}G",
                          flush=True)
                    print(f"    Headroom:               "
                          f"{hbm_limit - max_peak:.2f}G "
                          f"({(hbm_limit - max_peak)/hbm_limit*100:.0f}%)",
                          flush=True)

            del normed, h_Q, h_K, h_V, tau_all, Q, K, V
            del h_know, tau_know, _kout, dummy_x
        except Exception as e:
            if is_host0:
                import traceback
                print(f"  Breakdown failed: {e}", flush=True)
                traceback.print_exc()

        # Clear XLA compilation cache and free profiling memory
        import gc
        gc.collect()
        jax.clear_caches()

        if is_host0:
            # Estimate total training time
            total_steps = len(train_loader) * num_epochs
            remaining_steps = total_steps - global_step
            est_seconds = remaining_steps * step_time
            est_hours = est_seconds / 3600
            print(f"  Estimated time: {est_hours:.1f}h ({remaining_steps:,} steps @ {step_time*1000:.1f}ms)", flush=True)

        del dummy_ids, dummy_mask
        del _dp2, _do2, dummy_metrics2
        if is_host0:
            print("=== OOM check passed (JIT compiled) ===\n", flush=True)
    except Exception as e:
        if is_host0:
            print(f"\n  *** OOM check FAILED: {e}")
            print(f"  The model + gradients do not fit in device memory.")
            print(f"  Try: reduce batch_size, enable gradient_checkpointing, or use a smaller model.")
        raise

    # ----------------------------------------------------------
    # Training log file (host 0 only)
    # ----------------------------------------------------------
    if is_host0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_log_file = _join(log_dir, f'training_log_{timestamp}.txt')
        jsonl_log_file = _join(log_dir, f'metrics_{timestamp}.jsonl')

        # Set up loggers (local append + periodic GCS sync)
        _setup_loggers(training_log_file, jsonl_log_file)

        n_params = count_parameters(params)
        log_message(f"DAWN {model_version} Training Log (Multi-Host) - {timestamp}")
        log_message(f"Config: {config_path}")
        log_message(f"Parameters: {n_params:,}")
        log_message(f"Hosts: {n_hosts}, Local devices: {n_local_devices}, Total: {jax.device_count()}")
        log_message(f"Total steps: {total_steps}")
        log_message("")
        sync_logs()

    # ----------------------------------------------------------
    # Set data loader resume position
    # ----------------------------------------------------------
    if start_step_in_epoch > 0:
        if is_host0:
            print(f"  Resuming data loader at step_in_epoch={start_step_in_epoch}")
        train_loader.reset(start_step=start_step_in_epoch)

    # ----------------------------------------------------------
    # SIGTERM handler for spot TPU preemption
    # ----------------------------------------------------------
    preemption_requested = [False]  # mutable container for closure

    def _ckpt_path(name):
        return _join(checkpoint_dir, name)

    def handle_preemption(signum, frame):
        """Emergency checkpoint on SIGTERM (spot preemption). Host 0 only saves."""
        if preemption_requested[0]:
            return  # avoid double-save
        preemption_requested[0] = True
        print(f"\n!!! SIGTERM received (host {host_id}) -- saving emergency checkpoint (step={global_step}) !!!", flush=True)
        try:
            params_single = _gather_for_save(params)
            opt_state_single = _gather_for_save(opt_state)
            if is_host0:
                epath = _ckpt_path(f"emergency_step{global_step}.flax")
                save_checkpoint(
                    epath, params_single, opt_state_single,
                    start_epoch, global_step, best_val_loss,
                    cfg['model'],
                    step_in_epoch=epoch_step_counter,
                    steps_per_epoch=steps_per_epoch,
                    training_config=training_config,
                )
                print(f"!!! Emergency checkpoint saved: {epath} !!!", flush=True)
        except Exception as e:
            print(f"!!! Emergency save FAILED: {e} !!!", flush=True)

    def _gather_for_save(x):
        """Gather sharded params for checkpoint save. Only needed for baseline FSDP."""
        if is_baseline:
            return jax.device_get(process_allgather(x))
        return jax.device_get(x)

    signal.signal(signal.SIGTERM, handle_preemption)
    if is_host0:
        print("  SIGTERM handler registered (spot preemption safety)")

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    if is_host0:
        print(f"\n{'='*60}")
        print("=== Starting training loop ===", flush=True)
        print(f"{'='*60}")

    train_start_time = time.time()
    total_micro_steps = num_epochs * steps_per_epoch
    val_interval = cfg['training'].get('val_interval', 5000)
    ckpt_interval = cfg['training'].get('checkpoint_interval', 5000)
    epoch_step_counter = start_step_in_epoch  # tracks position within current epoch

    # Emb drift snapshot (sense vectors) — updated every LOG_INTERVAL on host 0.
    _prev_emb_snap = None

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_valid = 0
        epoch_steps = 0

        # Window accumulators for periodic logging
        win_loss = 0.0
        win_ce = 0.0
        win_aux = 0.0
        win_tau_reg = 0.0
        win_orth = 0.0
        win_div = 0.0
        win_correct = 0
        win_valid = 0
        win_count = 0
        win_start_time = time.time()

        for local_step, (input_ids, attention_mask) in enumerate(train_loader):

            if preemption_requested[0]:
                if is_host0:
                    print("Preemption requested -- exiting training loop.", flush=True)
                break

            # Shard data and run train step
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.fold_in(step_rng, host_id)  # per-host dropout
            input_ids = shard_to_mesh(
                input_ids, data_sharding, (batch_size, max_seq_len))
            attention_mask = shard_to_mesh(
                attention_mask, data_sharding, (batch_size, max_seq_len))

            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                input_ids, attention_mask, step_rng)

            # Extract metrics (scalars from jit, no [0] indexing)
            def _m(v):
                return float(v)
            m_total = _m(metrics['total_loss'])
            m_ce = _m(metrics['ce_loss'])
            m_aux = _m(metrics['aux_loss'])
            m_tau_reg = _m(metrics.get('tau_reg', 0.0))
            m_orth = _m(metrics['orth_loss'])
            m_div = _m(metrics['div_loss'])
            m_correct = int(_m(metrics['correct']))
            m_valid = int(_m(metrics['valid_count']))

            # NaN/INF detection
            if check_nan_inf({
                'total_loss': m_total, 'ce_loss': m_ce, 'aux_loss': m_aux,
                'tau_reg': m_tau_reg,
                'orth_loss': m_orth, 'div_loss': m_div,
            }, global_step + 1, epoch):
                raise ValueError(f"NaN/INF loss detected at epoch {epoch}, step {global_step + 1}")

            epoch_loss += m_ce * m_valid
            epoch_correct += m_correct
            epoch_valid += m_valid
            epoch_steps += 1

            win_loss += m_total
            win_ce += m_ce
            win_aux += m_aux
            win_tau_reg += m_tau_reg
            win_orth += m_orth
            win_div += m_div
            win_correct += m_correct
            win_valid += m_valid
            win_count += 1

            global_step += 1
            epoch_step_counter += 1

            # ---- Periodic logging (host 0 only) ----
            _early_debug = global_step in (1, 5, 10, 20, 50)
            if global_step % LOG_INTERVAL == 0 or _early_debug or debug_mode:
                if is_host0:
                    elapsed = time.time() - win_start_time
                    steps_per_sec = win_count / elapsed if elapsed > 0 else 0
                    avg_loss = win_loss / win_count
                    avg_ce = win_ce / win_count
                    avg_aux = win_aux / win_count
                    avg_tau_reg = win_tau_reg / win_count
                    avg_orth = win_orth / win_count
                    avg_div = win_div / win_count
                    avg_acc = win_correct / win_valid if win_valid > 0 else 0.0

                    # Current LR from schedule (indexed by optimizer step, not micro-step)
                    opt_step = global_step // grad_accum_steps
                    current_lr = float(schedule(opt_step))

                    total_elapsed = time.time() - train_start_time
                    epoch_elapsed = time.time() - epoch_start
                    progress = global_step / total_micro_steps * 100

                    # Timing: elapsed<remaining, s/it
                    s_per_it = epoch_elapsed / epoch_steps if epoch_steps > 0 else 0
                    remaining_steps = steps_per_epoch - epoch_steps
                    eta = s_per_it * remaining_steps

                    # Grad norm
                    m_grad = _m(metrics['grad_norm'])

                    msg = (
                        f"[Step {global_step}/{total_micro_steps} ({progress:.1f}%)] "
                        f"loss={avg_loss:.4f} ce={avg_ce:.4f} | "
                        f"reg(raw): aux={avg_aux:.4f} tau_reg={avg_tau_reg:.4f} "
                        f"orth={avg_orth:.2e} div={avg_div:.2e} | "
                        f"acc={avg_acc:.4f} lr={current_lr:.2e} "
                        f"{format_time(epoch_elapsed)}<{format_time(eta)}, {s_per_it:.2f}s/it"
                    )
                    log_message(msg)

                    # Defensive defaults for downstream JSONL so values exist
                    # even if the detailed-stats block below raises.
                    a_tau_s = np.zeros(3, dtype=np.float32)
                    k_tau_s = 0.0
                    a_kern = 0.0
                    k_kern = 0.0
                    k_tau_abs = 0.0
                    a_tau_abs = 0.0
                    k_z075 = 0.0
                    a_z075 = 0.0
                    k_z030 = 0.0
                    a_z030 = 0.0
                    k_skew = 0.0
                    a_skew = 0.0
                    k_apt = 0.0
                    a_apt = 0.0
                    k_ent = 0.0
                    a_ent = 0.0
                    k_zsum = 0.0
                    a_zsum = 0.0
                    k_emb_nmax = 0.0
                    k_emb_nmin = 0.0
                    k_emb_nstd = 0.0
                    qk_emb_nmean = 0.0
                    qk_emb_nmax = 0.0
                    qk_emb_nmin = 0.0
                    qk_emb_nstd = 0.0
                    v_emb_nmean = 0.0
                    v_emb_nmax = 0.0
                    v_emb_nmin = 0.0
                    v_emb_nstd = 0.0
                    k_kurt = 0.0
                    a_kurt = 0.0
                    k_drop = 0.0
                    a_drop = 0.0
                    k_boost = 0.0
                    a_boost = 0.0

                    # Detailed stats (all from metrics, no params access)
                    try:
                        tk_b = _m(metrics['tau_know_bias'])
                        ta_b = [_m(metrics['tau_attn_bias_0']),
                                _m(metrics['tau_attn_bias_1']),
                                _m(metrics['tau_attn_bias_2'])]
                        tau_s = (f"tau: know={tk_b:.2f} "
                                 f"attn=[{ta_b[0]:.2f},{ta_b[1]:.2f},{ta_b[2]:.2f}]")

                        m_attn_aux = _m(metrics.get('attn_aux', 0.0))
                        m_know_aux = _m(metrics.get('know_aux', 0.0))
                        k_emb_n = _m(metrics.get('know_emb_norm', 0.0))
                        k_read_n = _m(metrics.get('know_read_norm', 0.0))
                        k_write_n = _m(metrics.get('know_write_norm', 0.0))

                        n_know_cfg = cfg['model'].get('n_know', 27200)
                        n_qk_cfg = cfg['model'].get('n_qk', 0)
                        n_v_cfg = cfg['model'].get('n_v', 0)
                        k_act = _m(metrics['know_active'])
                        k_aN = _m(metrics.get('know_active_N', 0.0))
                        k_sstd = _m(metrics.get('know_score_std', 0.0))
                        k_raw_gmax = _m(metrics.get('know_raw_gate_max', metrics.get('know_gate_max', 0.0)))
                        k_gsum = _m(metrics.get('know_gate_sum', 0.0))
                        k_zsum = _m(metrics.get('know_z_sum', 0.0))
                        k_gconc = _m(metrics.get('know_gate_conc', 0.0))
                        k_anm = _m(metrics.get('know_active_n_mean', 0.0))
                        k_strong = _m(metrics.get('know_strong', 0.0))

                        a_qk_act = _m(metrics.get('attn_qk_active', 0.0))
                        a_v_act = _m(metrics.get('attn_v_active', 0.0))
                        a_aN = _m(metrics.get('attn_active_N', 0.0))
                        a_sstd = _m(metrics.get('attn_score_std', 0.0))
                        a_raw_gmax = _m(metrics.get('attn_raw_gate_max', metrics.get('attn_gate_max', 0.0)))
                        a_gsum = _m(metrics.get('attn_gate_sum', 0.0))
                        a_zsum = _m(metrics.get('attn_z_sum', 0.0))
                        a_gconc = _m(metrics.get('attn_gate_conc', 0.0))
                        a_anm = _m(metrics.get('attn_active_n_mean', 0.0))
                        a_strong = _m(metrics.get('attn_strong', 0.0))
                        a_out_n = _m(metrics.get('attn_out_norm', 0.0))

                        a_tau_m = _m(metrics.get('attn_tau_mean', 0.0))
                        k_tau_m = _m(metrics.get('know_tau_mean', 0.0))
                        k_out_n = _m(metrics.get('know_out_norm', 0.0))

                        # tau_offset per-token std + kernel Frobenius norm
                        # (distinguishes bias-only offset from active per-token routing)
                        try:
                            a_tau_s = np.asarray(jax.device_get(metrics.get(
                                'attn_tau_std', jnp.zeros(3))))
                            if a_tau_s.size < 3:
                                a_tau_s = np.zeros(3, dtype=np.float32)
                        except Exception:
                            a_tau_s = np.zeros(3, dtype=np.float32)
                        k_tau_s = _m(metrics.get('know_tau_std', 0.0))
                        a_kern = _m(metrics.get('attn_tau_kernel_norm', 0.0))
                        k_kern = _m(metrics.get('know_tau_kernel_norm', 0.0))

                        log_message(
                            f"      {tau_s} | tau_mean: attn={a_tau_m:.3f}"
                            f" know={k_tau_m:.3f} | grad_norm={m_grad:.3f}")
                        log_message(
                            f"      tau_struct: k_std={k_tau_s:.3f}"
                            f" a_std=[{float(a_tau_s[0]):.3f},{float(a_tau_s[1]):.3f},{float(a_tau_s[2]):.3f}]"
                            f" k_kern={k_kern:.2f} a_kern={a_kern:.2f}")

                        # Absolute gate threshold + z-margin distribution
                        k_tau_abs = _m(metrics.get('know_tau_abs_mean', 0.0))
                        a_tau_abs = _m(metrics.get('attn_tau_abs_mean', 0.0))
                        k_z075 = _m(metrics.get('know_z_lt_075', 0.0))
                        a_z075 = _m(metrics.get('attn_z_lt_075', 0.0))
                        k_z030 = _m(metrics.get('know_z_lt_030', 0.0))
                        a_z030 = _m(metrics.get('attn_z_lt_030', 0.0))
                        log_message(
                            f"      gate_margin: k[abs={k_tau_abs:+.3f}"
                            f" z<075={k_z075*100:.2f}% z<030={k_z030*100:.2f}%]"
                            f" a[abs={a_tau_abs:+.3f}"
                            f" z<075={a_z075*100:.2f}% z<030={a_z030*100:.2f}%]")

                        # Score-dist skewness + active-count std across tokens
                        k_skew = _m(metrics.get('know_score_skew', 0.0))
                        a_skew = _m(metrics.get('attn_score_skew', 0.0))
                        k_apt = _m(metrics.get('know_active_per_token_std', 0.0))
                        a_apt = _m(metrics.get('attn_active_per_token_std', 0.0))
                        k_ent = _m(metrics.get('know_gate_entropy', 0.0))
                        a_ent = _m(metrics.get('attn_gate_entropy', 0.0))
                        k_kurt = _m(metrics.get('know_score_kurt', 0.0))
                        a_kurt = _m(metrics.get('attn_score_kurt', 0.0))
                        # Dead count (v4.0.6+). alpha / tau_shift / s_std_min
                        # removed in v4.1 (dynamic tau gone).
                        k_dead = _m(metrics.get('know_dead_count', 0.0))
                        a_dead = _m(metrics.get('attn_dead_count', 0.0))
                        log_message(
                            f"      dist: k[skew={k_skew:+.2f} kurt={k_kurt:.2f}"
                            f" apt_std={k_apt:.1f} ent={k_ent:.2f} dead={int(k_dead)}]"
                            f" a[skew={a_skew:+.2f} kurt={a_kurt:.2f}"
                            f" apt_std={a_apt:.1f} ent={a_ent:.2f} dead={int(a_dead)}]")

                        # v4.1 tau_offset distribution — prime diagnostic
                        # for exploration-driven runaway.
                        a_tau_off_min = _m(metrics.get('attn_tau_off_min', 0.0))
                        a_tau_off_max = _m(metrics.get('attn_tau_off_max', 0.0))
                        a_tau_off_p99 = _m(metrics.get('attn_tau_off_p99', 0.0))
                        a_tau_off_neg = _m(metrics.get('attn_tau_off_neg_frac', 0.0))
                        k_tau_off_min = _m(metrics.get('know_tau_off_min', 0.0))
                        k_tau_off_max = _m(metrics.get('know_tau_off_max', 0.0))
                        k_tau_off_p99 = _m(metrics.get('know_tau_off_p99', 0.0))
                        k_tau_off_neg = _m(metrics.get('know_tau_off_neg_frac', 0.0))
                        log_message(
                            f"      tau_off k: min={k_tau_off_min:+.2f}"
                            f" max={k_tau_off_max:+.2f}"
                            f" p99={k_tau_off_p99:+.2f}"
                            f" neg={k_tau_off_neg*100:.1f}%")
                        log_message(
                            f"      tau_off a: min={a_tau_off_min:+.2f}"
                            f" max={a_tau_off_max:+.2f}"
                            f" p99={a_tau_off_p99:+.2f}"
                            f" neg={a_tau_off_neg*100:.1f}%")

                        # v4.1 RPE exploration (batch-global-mean baseline,
                        # asymmetric signal; no EMA / warmup).
                        m_global_ce = _m(metrics.get('global_mean_ce', 0.0))
                        m_pos_frac = _m(metrics.get('pos_frac', 0.0))
                        m_pos_mean = _m(metrics.get('pos_mean', 0.0))
                        m_neg_mean = _m(metrics.get('neg_mean', 0.0))
                        m_explore_a = _m(metrics.get('explore_attn_raw', 0.0))
                        m_explore_k = _m(metrics.get('explore_know_raw', 0.0))
                        m_explore_w = _m(metrics.get('explore_loss_weighted', 0.0))
                        m_block_a = _m(metrics.get('explore_block_frac_a', 0.0))
                        m_block_k = _m(metrics.get('explore_block_frac_k', 0.0))
                        m_sig_pmax = _m(metrics.get('sig_pos_max', 0.0))
                        m_sig_nmax = _m(metrics.get('sig_neg_max', 0.0))
                        log_message(
                            f"      rpe: mean_ce={m_global_ce:.3f}"
                            f" pos_frac={m_pos_frac*100:.1f}%"
                            f" pos_avg={m_pos_mean:.3f} neg_avg={m_neg_mean:.3f}"
                            f" sig[max+={m_sig_pmax:.2f} max-={m_sig_nmax:.2f}]"
                            f" expl[a={m_explore_a:+.3f} k={m_explore_k:+.3f}]"
                            f" w={m_explore_w:+.4f}"
                            f" block[a={m_block_a*100:.1f}% k={m_block_k*100:.1f}%]")

                        # Emb norm per-pool stats (mean / max / min / std)
                        k_emb_nmax = _m(metrics.get('know_emb_norm_max', 0.0))
                        k_emb_nmin = _m(metrics.get('know_emb_norm_min', 0.0))
                        k_emb_nstd = _m(metrics.get('know_emb_norm_std', 0.0))
                        qk_emb_nmean = _m(metrics.get('qk_emb_norm_mean', 0.0))
                        qk_emb_nmax = _m(metrics.get('qk_emb_norm_max', 0.0))
                        qk_emb_nmin = _m(metrics.get('qk_emb_norm_min', 0.0))
                        qk_emb_nstd = _m(metrics.get('qk_emb_norm_std', 0.0))
                        v_emb_nmean = _m(metrics.get('v_emb_norm_mean', 0.0))
                        v_emb_nmax = _m(metrics.get('v_emb_norm_max', 0.0))
                        v_emb_nmin = _m(metrics.get('v_emb_norm_min', 0.0))
                        v_emb_nstd = _m(metrics.get('v_emb_norm_std', 0.0))
                        log_message(
                            f"      emb_n: k[{k_emb_n:.3f}/{k_emb_nstd:.3f} min={k_emb_nmin:.3f}]"
                            f" qk[{qk_emb_nmean:.3f}/{qk_emb_nstd:.3f} min={qk_emb_nmin:.3f}]"
                            f" v[{v_emb_nmean:.3f}/{v_emb_nstd:.3f} min={v_emb_nmin:.3f}]")

                        # Emb drift (relative L2 change since last LOG_INTERVAL snapshot).
                        drift_qk = drift_v = drift_know = 0.0
                        try:
                            pool_now = params['neuron_pool']
                            cur_qk = pool_now['qk_emb']
                            cur_v = pool_now['v_emb']
                            cur_know = pool_now['know_emb']
                            if _prev_emb_snap is not None:
                                p_qk, p_v, p_know = _prev_emb_snap
                                drift_qk = float(
                                    jnp.linalg.norm(cur_qk - p_qk)
                                    / (jnp.linalg.norm(p_qk) + 1e-8))
                                drift_v = float(
                                    jnp.linalg.norm(cur_v - p_v)
                                    / (jnp.linalg.norm(p_v) + 1e-8))
                                drift_know = float(
                                    jnp.linalg.norm(cur_know - p_know)
                                    / (jnp.linalg.norm(p_know) + 1e-8))
                                log_message(
                                    f"      drift ({LOG_INTERVAL}step):"
                                    f" qk_emb={drift_qk:.4e}"
                                    f" v_emb={drift_v:.4e}"
                                    f" know_emb={drift_know:.4e}")
                            _prev_emb_snap = (cur_qk, cur_v, cur_know)
                        except Exception as _drift_err:
                            log_message(f"      drift: failed ({_drift_err})")
                        # v4.1: aux line removed (lb_weight=0 → no loss contribution).
                        k_raw_n = _m(metrics.get('know_raw_out_norm', 0.0))
                        a_qk_raw_n = _m(metrics.get('attn_qk_raw_norm', 0.0))
                        a_v_raw_n = _m(metrics.get('attn_v_raw_norm', 0.0))

                        # know line: show active_N or gate_sum/conc depending on version
                        k_extra = ""
                        if k_gsum > 0:  # v3.9.1+
                            k_extra = f" gate_max={k_raw_gmax:.4f} conc={k_gconc:.1f} gsum={k_gsum:.1f}"
                        if k_anm > 0:  # v3.9.5
                            k_extra = f" gate_max={k_raw_gmax:.4f} active_n={k_anm:.0f} gsum={k_gsum:.1f}"
                        if k_zsum > 0:  # v4.0.3 (Σz^+ denominator)
                            k_extra += f" z_sum={k_zsum:.1f}"
                        if k_aN > 0:    # v3.9.2
                            k_extra += f" active_N={k_aN:.0f}"

                        # v4.0.0 (symmetric gate): know_pos metric exists → show pos/neg split
                        # All others (ReLU/GELU gate): show active/total + optional strong
                        _has_pos = bool(metrics.get('know_pos', 0.0))
                        if _has_pos:
                            k_pos = _m(metrics.get('know_pos', k_strong))
                            k_neg = max(k_act - k_pos, 0.0)
                            k_pos_n = k_pos * n_know_cfg
                            k_neg_n = k_neg * n_know_cfg
                            k_total_n = k_act * n_know_cfg
                            log_message(
                                f"      know: pos={k_pos_n:.0f}({k_pos*100:.1f}%)"
                                f" neg={k_neg_n:.0f}({k_neg*100:.1f}%)"
                                f" active={k_total_n:.0f}({k_act*100:.1f}%){k_extra}"
                                f" s_std={k_sstd:.3f}"
                                f" raw_norm={k_raw_n:.6f} out_norm={k_out_n:.3f}")
                        else:
                            # v4.1 compressed know line.
                            k_phi = _m(metrics.get('know_phi_binary', 0.0))
                            k_z_act = _m(metrics.get('know_z_mean_active', 0.0))
                            k_z075 = _m(metrics.get('know_z_lt_075', 0.0))
                            k_z030 = _m(metrics.get('know_z_lt_030', 0.0))
                            k_int_max = _m(metrics.get('know_int_max', 0.0))
                            k_int_cap = _m(metrics.get('know_int_cap_frac', 0.0))
                            k_act_n = k_act * n_know_cfg
                            k_strong_n = k_strong * n_know_cfg
                            k_strong_of_act = k_strong_n / max(k_act_n, 1.0)
                            log_message(
                                f"      know: act={k_act_n:.0f}/{n_know_cfg}({k_act*100:.1f}%)"
                                f" strong={k_strong_n:.0f}/{k_act_n:.0f}({k_strong_of_act*100:.1f}%)"
                                f" gate_max={k_raw_gmax:.2f}"
                                f" int_avg={k_z_act:.2f} int_max={k_int_max:.2f}"
                                f" cap={k_int_cap*100:.1f}%"
                                f" s_std={k_sstd:.2f} raw={k_raw_n:.3f} out={k_out_n:.2f}"
                                f" phi_bin={k_phi*100:.1f}% bnd={k_z075*100:.1f}%"
                                f" mid={k_z030*100:.1f}%")

                        # attn line
                        a_extra = ""
                        if a_gsum > 0:  # v3.9.1+
                            a_extra = f" gate_max={a_raw_gmax:.4f} conc={a_gconc:.1f} gsum={a_gsum:.1f}"
                        if a_anm > 0:  # v3.9.5
                            a_extra = f" gate_max={a_raw_gmax:.4f} active_n={a_anm:.0f} gsum={a_gsum:.1f}"
                        if a_aN > 0:    # v3.9.2
                            a_extra += f" active_N={a_aN:.0f}"
                        if a_zsum > 0:  # v4.0.3 (Σz^+ denominator)
                            a_extra += f" z_sum={a_zsum:.1f}"

                        _has_attn_pos = bool(metrics.get('attn_qk_pos', metrics.get('attn_pos', 0.0)))
                        if _has_attn_pos:
                            a_qk_pos = _m(metrics.get('attn_qk_pos', metrics.get('attn_pos', metrics.get('attn_strong', 0.0))))
                            a_v_pos = _m(metrics.get('attn_v_pos', 0.0))
                            a_qk_neg = max(a_qk_act - a_qk_pos, 0.0)
                            a_v_neg = max(a_v_act - a_v_pos, 0.0)
                            log_message(
                                f"      attn: qk_pos={a_qk_pos*n_qk_cfg:.0f}({a_qk_pos*100:.1f}%)"
                                f" qk_neg={a_qk_neg*n_qk_cfg:.0f}({a_qk_neg*100:.1f}%)"
                                f" v_pos={a_v_pos*n_v_cfg:.0f}({a_v_pos*100:.1f}%)"
                                f" v_neg={a_v_neg*n_v_cfg:.0f}({a_v_neg*100:.1f}%){a_extra}"
                                f" s_std={a_sstd:.3f}"
                                f" qk_raw={a_qk_raw_n:.6f} v_raw={a_v_raw_n:.6f}"
                                f" out_norm={a_out_n:.3f}")
                        else:
                            # v4.1 compressed attn line.
                            a_qk_phi = _m(metrics.get('attn_qk_phi_binary', 0.0))
                            a_v_phi = _m(metrics.get('attn_v_phi_binary', 0.0))
                            a_qk_z = _m(metrics.get('attn_qk_z_mean_active', 0.0))
                            a_v_z = _m(metrics.get('attn_v_z_mean_active', 0.0))
                            a_z075 = _m(metrics.get('attn_z_lt_075', 0.0))
                            a_z030 = _m(metrics.get('attn_z_lt_030', 0.0))
                            a_int_max = _m(metrics.get('attn_int_max', 0.0))
                            a_int_cap = _m(metrics.get('attn_int_cap_frac', 0.0))
                            a_qk_act_n = a_qk_act * n_qk_cfg
                            a_v_act_n = a_v_act * n_v_cfg
                            # `attn_strong` is avg of qk/v strong fractions;
                            # combined active count = sum of both active counts.
                            # Express "strong among active" against the combined
                            # population so the %denom matches what act= shows.
                            a_total_act_n = a_qk_act_n + a_v_act_n
                            a_strong_n = a_strong * (n_qk_cfg + n_v_cfg)
                            a_strong_of_act = a_strong_n / max(a_total_act_n, 1.0)
                            log_message(
                                f"      attn: qk_act={a_qk_act_n:.0f}/{n_qk_cfg}({a_qk_act*100:.1f}%)"
                                f" v_act={a_v_act_n:.0f}/{n_v_cfg}({a_v_act*100:.1f}%)"
                                f" strong={a_strong_n:.0f}/{a_total_act_n:.0f}({a_strong_of_act*100:.1f}%)"
                                f" gate_max={a_raw_gmax:.2f}"
                                f" int_avg[qk={a_qk_z:.2f} v={a_v_z:.2f}]"
                                f" int_max={a_int_max:.2f} cap={a_int_cap*100:.1f}%"
                                f" s_std={a_sstd:.2f}"
                                f" qk_raw={a_qk_raw_n:.3f} v_raw={a_v_raw_n:.3f}"
                                f" out={a_out_n:.2f}"
                                f" phi_bin[qk={a_qk_phi*100:.1f}% v={a_v_phi*100:.1f}%]"
                                f" bnd={a_z075*100:.1f}% mid={a_z030*100:.1f}%")
                        # Strength (v3.9.3)
                        k_str_m = _m(metrics.get('know_strength_mean', 0.0))
                        if k_str_m > 0:
                            k_str_s = _m(metrics.get('know_strength_std', 0.0))
                            k_str_mn = _m(metrics.get('know_strength_min', 0.0))
                            k_str_mx = _m(metrics.get('know_strength_max', 0.0))
                            k_lg_m = _m(metrics.get('know_logit_mean', 0.0))
                            k_lg_s = _m(metrics.get('know_logit_std', 0.0))
                            log_message(
                                f"      know_str: mean={k_str_m:.2f} std={k_str_s:.2f}"
                                f" min={k_str_mn:.2f} max={k_str_mx:.2f}"
                                f" | logit: mean={k_lg_m:.3f} std={k_lg_s:.3f}")
                        a_v_str_m = _m(metrics.get('attn_v_strength_mean', 0.0))
                        if a_v_str_m > 0:
                            a_v_str_s = _m(metrics.get('attn_v_strength_std', 0.0))
                            a_v_str_mn = _m(metrics.get('attn_v_strength_min', 0.0))
                            a_v_str_mx = _m(metrics.get('attn_v_strength_max', 0.0))
                            a_v_lg_m = _m(metrics.get('attn_v_logit_mean', 0.0))
                            a_v_lg_s = _m(metrics.get('attn_v_logit_std', 0.0))
                            log_message(
                                f"      v_str: mean={a_v_str_m:.2f} std={a_v_str_s:.2f}"
                                f" min={a_v_str_mn:.2f} max={a_v_str_mx:.2f}"
                                f" | logit: mean={a_v_lg_m:.3f} std={a_v_lg_s:.3f}")
                        # Full DEBUG (per-layer stacks, attn q/k/v norms,
                        # residual/emb/o_proj) is verbose — emit every 500
                        # steps or when explicitly in debug mode / early-debug
                        # window. Logit_max + out norm are cheap signals that
                        # fit on the attn line, so skip the one-line summary.
                        _full_debug = (debug_mode or _early_debug
                                        or (global_step % 500 == 0))
                        if _full_debug:
                            d_res = _m(metrics.get('debug_residual_norm', 0.0))
                            d_emb = _m(metrics.get('debug_emb_norm', 0.0))
                            d_oproj = _m(metrics.get('debug_o_proj_norm', 0.0))
                            d_q = _m(metrics.get('debug_q_norm', 0.0))
                            d_k = _m(metrics.get('debug_k_norm', 0.0))
                            d_v = _m(metrics.get('debug_v_norm', 0.0))
                            d_lm = _m(metrics.get('debug_logit_max', 0.0))
                            d_oi = _m(metrics.get('debug_o_input_norm', 0.0))
                            log_message(
                                f"      [DEBUG] residual={d_res:.3f}"
                                f" emb={d_emb:.3f} o_proj={d_oproj:.3f}"
                                f" read={k_read_n:.3f}")
                            log_message(
                                f"      [DEBUG] attn_detail:"
                                f" q={d_q:.3f} k={d_k:.3f} v={d_v:.3f}"
                                f" logit_max={d_lm:.3f}"
                                f" o_in={d_oi:.3f} o_out={a_out_n:.3f}")
                            try:
                                pl_attn = jax.device_get(metrics['per_layer_attn_out_norm'])
                                pl_know = jax.device_get(metrics['per_layer_know_out_norm'])
                                attn_s = ', '.join(f'l{i}={v:.2f}' for i, v in enumerate(pl_attn))
                                know_s = ', '.join(f'l{i}={v:.2f}' for i, v in enumerate(pl_know))
                                log_message(f"      [DEBUG] per_layer_attn: [{attn_s}]")
                                log_message(f"      [DEBUG] per_layer_know: [{know_s}]")
                            except Exception:
                                pass
                    except Exception:
                        log_message(f"      grad_norm={m_grad:.3f}")

                    # JSONL structured log.
                    # *_loss / tau_reg: raw values (pre-weight).
                    # *_weighted: contribution to total_loss (= raw * weight).
                    log_jsonl({
                        'type': 'train',
                        'step': global_step,
                        'epoch': epoch,
                        'total_loss': avg_loss,
                        'ce_loss': avg_ce,
                        'aux_loss': avg_aux,
                        'tau_reg': avg_tau_reg,
                        'orth_loss': avg_orth,
                        'div_loss': avg_div,
                        'aux_weighted': lb_weight * avg_aux,
                        'tau_reg_weighted': tau_reg_weight * avg_tau_reg,
                        'orth_weighted': orth_weight * avg_orth,
                        'div_weighted': div_weight * avg_div,
                        'drift_qk_emb': drift_qk,
                        'drift_v_emb': drift_v,
                        'drift_know_emb': drift_know,
                        # Tau dynamics (Phase 1/2/3a/4 metrics; instantaneous, not window-averaged)
                        'attn_tau_std_q': float(a_tau_s[0]),
                        'attn_tau_std_k': float(a_tau_s[1]),
                        'attn_tau_std_v': float(a_tau_s[2]),
                        'know_tau_std': k_tau_s,
                        'attn_tau_kernel_norm': a_kern,
                        'know_tau_kernel_norm': k_kern,
                        'attn_tau_abs_mean': a_tau_abs,
                        'know_tau_abs_mean': k_tau_abs,
                        'attn_z_lt_075': a_z075,
                        'know_z_lt_075': k_z075,
                        'attn_z_lt_030': a_z030,
                        'know_z_lt_030': k_z030,
                        'attn_score_skew': a_skew,
                        'know_score_skew': k_skew,
                        'attn_active_per_token_std': a_apt,
                        'know_active_per_token_std': k_apt,
                        'attn_gate_entropy': a_ent,
                        'know_gate_entropy': k_ent,
                        'attn_z_sum': a_zsum,
                        'know_z_sum': k_zsum,
                        'know_emb_norm_max': k_emb_nmax,
                        'know_emb_norm_min': k_emb_nmin,
                        'know_emb_norm_std': k_emb_nstd,
                        'qk_emb_norm_mean': qk_emb_nmean,
                        'qk_emb_norm_max': qk_emb_nmax,
                        'qk_emb_norm_min': qk_emb_nmin,
                        'qk_emb_norm_std': qk_emb_nstd,
                        'v_emb_norm_mean': v_emb_nmean,
                        'v_emb_norm_max': v_emb_nmax,
                        'v_emb_norm_min': v_emb_nmin,
                        'v_emb_norm_std': v_emb_nstd,
                        'attn_score_kurt': a_kurt,
                        'know_score_kurt': k_kurt,
                        'attn_drop_rate': a_drop,
                        'attn_boost_rate': a_boost,
                        'know_drop_rate': k_drop,
                        'know_boost_rate': k_boost,
                        # Dead-only penalty (v4.0.6+).
                        'dead_penalty': float(metrics.get('dead_penalty', 0.0)),
                        'attn_dead_penalty': float(metrics.get('attn_dead_penalty', 0.0)),
                        'know_dead_penalty': float(metrics.get('know_dead_penalty', 0.0)),
                        'dead_penalty_weighted': dead_penalty_weight * float(metrics.get('dead_penalty', 0.0)),
                        'attn_dead_count': float(metrics.get('attn_dead_count', 0.0)),
                        'know_dead_count': float(metrics.get('know_dead_count', 0.0)),
                        # v4.1 RPE exploration (redesigned, asymmetric).
                        'global_mean_ce': float(metrics.get('global_mean_ce', 0.0)),
                        'pos_frac': float(metrics.get('pos_frac', 0.0)),
                        'pos_mean': float(metrics.get('pos_mean', 0.0)),
                        'neg_mean': float(metrics.get('neg_mean', 0.0)),
                        'explore_loss_raw': float(metrics.get('explore_loss_raw', 0.0)),
                        'explore_attn_raw': float(metrics.get('explore_attn_raw', 0.0)),
                        'explore_know_raw': float(metrics.get('explore_know_raw', 0.0)),
                        'explore_loss_weighted': float(metrics.get('explore_loss_weighted', 0.0)),
                        'explore_block_frac_a': float(metrics.get('explore_block_frac_a', 0.0)),
                        'explore_block_frac_k': float(metrics.get('explore_block_frac_k', 0.0)),
                        'accuracy': avg_acc,
                        'lr': current_lr,
                        'steps_per_sec': steps_per_sec,
                        'elapsed': total_elapsed,
                        'timestamp': datetime.now().isoformat(),
                    })

                    # TPU memory stats
                    try:
                        mem = jax.local_devices()[0].memory_stats()
                        if mem:
                            used = mem.get('bytes_in_use', 0) / 1e9
                            peak = mem.get('peak_bytes_in_use', 0) / 1e9
                            limit = mem.get('bytes_limit', 0) / 1e9
                            log_message(
                                f"      HBM: {used:.2f}G / {limit:.2f}G "
                                f"(peak={peak:.2f}G, free={limit - used:.2f}G)")
                    except Exception:
                        pass

                    # Sync logs to GCS
                    sync_logs()

                # Reset window (all hosts)
                win_loss = 0.0
                win_ce = 0.0
                win_aux = 0.0
                win_tau_reg = 0.0
                win_orth = 0.0
                win_div = 0.0
                win_correct = 0
                win_valid = 0
                win_count = 0
                win_start_time = time.time()

            # ---- Mid-epoch validation (all hosts run eval, host 0 saves/logs) ----
            if global_step % val_interval == 0 and global_step > 0:
                if is_host0:
                    log_message(f"\n  Mid-epoch validation at step {global_step}...")
                val_loader.reset()
                val_loss, val_acc = evaluate(
                    eval_step_fn, params, val_loader, n_local_devices,
                    verbose=is_host0, data_sharding_spec=data_sharding)
                if is_host0:
                    log_message(f"  Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")
                    log_jsonl({
                        'type': 'val',
                        'step': global_step,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'timestamp': datetime.now().isoformat(),
                    })

                # Best model save (device_get on ALL hosts)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    params_single = _gather_for_save(params)
                    opt_state_single = _gather_for_save(opt_state)
                    if is_host0:
                        save_checkpoint(
                            _ckpt_path("best_model.flax"),
                            params_single, opt_state_single,
                            epoch, global_step, best_val_loss,
                            cfg['model'],
                            step_in_epoch=epoch_step_counter,
                            steps_per_epoch=steps_per_epoch,
                            training_config=training_config,
                        )
                        log_message(f"  New best model saved! val_loss={best_val_loss:.4f}")
                    del params_single, opt_state_single

                # Dead neuron diagnosis (rw-v4.0.2)
                if model_version.startswith('rw-v') and is_host0:
                    try:
                        from models.dawn_spatial_v402_exp import diagnose_dead_neurons as _diag_dead
                        _diag_cfg = {
                            'd_model': cfg['model']['d_model'],
                            'n_layers': cfg['model']['n_layers'],
                            'n_heads': cfg['model']['n_heads'],
                            'max_seq_len': cfg['model']['max_seq_len'],
                            'n_q': cfg['model'].get('n_q', 790),
                            'n_k': cfg['model'].get('n_k', 790),
                            'n_v': cfg['model'].get('n_v', 2600),
                            'n_know': cfg['model'].get('n_know', 25200),
                        }
                        _params_cpu = _gather_for_save(params)
                        # Collect real val data for dead neuron diagnosis
                        val_loader.reset()
                        _diag_batches = []
                        for _di, (_dids, _dmask) in enumerate(val_loader):
                            _diag_batches.append(np.array(_dids))
                            if len(_diag_batches) * _dids.shape[0] >= 128:
                                break
                        _diag_tokens = jnp.array(np.concatenate(_diag_batches, axis=0)[:128])
                        _dead = jax.device_get(_diag_dead(
                            _params_cpu, _diag_cfg, _diag_tokens,
                            n_batches=4, batch_size=32))
                        del _diag_tokens
                        for _pn in ('Q', 'K', 'V', 'Know'):
                            _ps = _dead[_pn]
                            log_message(
                                f"      [DEAD] {_pn}: dead={int(_ps['n_dead'])}({float(_ps['dead_frac'])*100:.1f}%)"
                                f" rare(<1%)={int(_ps['n_rare'])}({float(_ps['rare_frac'])*100:.1f}%)")
                        del _params_cpu, _dead
                    except Exception as e:
                        log_message(f"      [DEAD] diagnosis failed: {e}")

            # ---- Mid-epoch checkpoint ----
            if global_step % ckpt_interval == 0 and global_step > 0:
                # device_get on ALL hosts (may be collective for sharded params)
                params_single = _gather_for_save(params)
                opt_state_single = _gather_for_save(opt_state)
                if is_host0:
                    save_checkpoint(
                        _ckpt_path(f"checkpoint_step{global_step}.flax"),
                        params_single, opt_state_single,
                        epoch, global_step, best_val_loss,
                        cfg['model'],
                        step_in_epoch=epoch_step_counter,
                        steps_per_epoch=steps_per_epoch,
                        training_config=training_config,
                    )
                del params_single, opt_state_single
                cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

        if preemption_requested[0]:
            break

        # ---- End of epoch ----
        epoch_elapsed = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / epoch_valid if epoch_valid > 0 else 0.0
        epoch_avg_acc = epoch_correct / epoch_valid if epoch_valid > 0 else 0.0

        if is_host0:
            log_message(
                f"\n{'='*60}\n"
                f"Epoch {epoch} complete in {format_time(epoch_elapsed)}\n"
                f"  Train loss={epoch_avg_loss:.4f}, Train acc={epoch_avg_acc:.4f}\n"
                f"{'='*60}"
            )

        # End-of-epoch validation (all hosts must participate in eval)
        if is_host0:
            log_message("  Running end-of-epoch validation...")
        val_loader.reset()
        val_loss, val_acc = evaluate(
            eval_step_fn, params, val_loader, n_local_devices,
            verbose=is_host0, data_sharding_spec=data_sharding)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if is_host0:
            log_message(f"  Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")
            log_jsonl({
                'type': 'val_epoch',
                'step': global_step,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': epoch_avg_loss,
                'train_acc': epoch_avg_acc,
                'epoch_time': epoch_elapsed,
                'timestamp': datetime.now().isoformat(),
            })

        # Save epoch checkpoint (device_get on ALL hosts)
        params_single = _gather_for_save(params)
        opt_state_single = _gather_for_save(opt_state)

        if is_host0:
            save_checkpoint(
                _ckpt_path(f"checkpoint_epoch{epoch}.flax"),
                params_single, opt_state_single,
                epoch + 1, global_step, best_val_loss,
                cfg['model'],
                step_in_epoch=0,  # start of next epoch
                steps_per_epoch=steps_per_epoch,
                training_config=training_config,
            )
            if is_best:
                save_checkpoint(
                    _ckpt_path("best_model.flax"),
                    params_single, opt_state_single,
                    epoch + 1, global_step, best_val_loss,
                    cfg['model'],
                    step_in_epoch=0,
                    steps_per_epoch=steps_per_epoch,
                    training_config=training_config,
                )
                log_message(f"  New best model! val_loss={best_val_loss:.4f}")

            del params_single, opt_state_single

            log_message(f"  Best val loss so far: {best_val_loss:.4f}")
            sync_logs()

        # Reset data loader for next epoch (no re-read, just reset position)
        if epoch < num_epochs - 1:
            train_loader.reset(start_step=0)
            epoch_step_counter = 0

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    total_time = time.time() - train_start_time
    if is_host0:
        log_message(
            f"\n{'='*60}\n"
            f"Training complete!\n"
            f"  Total time: {format_time(total_time)}\n"
            f"  Best val loss: {best_val_loss:.4f}\n"
            f"  Final step: {global_step}\n"
            f"{'='*60}"
        )
        sync_logs()


if __name__ == '__main__':
    main()

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

from models.model_v17_1_jax import DAWN
from models.dawn_spatial import DAWN as DAWN_Spatial
from models.dawn_spatial_v2 import DAWN as DAWN_SpatialV2
from models.dawn_spatial_v3 import DAWN as DAWN_SpatialV3
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
    """List files in a directory (local or GCS)."""
    dir_str = str(directory)
    if _is_gcs(dir_str):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = fs.glob(dir_str + pattern)
            return sorted(["gs://" + f for f in files])
        except ImportError:
            pass
        try:
            import tensorflow as tf
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = tf.io.gfile.glob(dir_str + pattern)
            return sorted(files)
        except ImportError:
            return []
    return sorted(str(f) for f in Path(dir_str).glob(pattern))


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
        for prefix in ('qk', 'v', 'know'):
            if f'{prefix}_neurons' in pool:
                arrays.append(pool[f'{prefix}_neurons'])
            else:
                # v3.2: emb + w
                if f'{prefix}_emb' in pool:
                    arrays.append(pool[f'{prefix}_emb'])
                if f'{prefix}_w' in pool:
                    arrays.append(pool[f'{prefix}_w'])
        return arrays

    pool_arrays = _get_pool_arrays(pool)
    return sum(_pool_div(a) for a in pool_arrays) / len(pool_arrays)


# ============================================================
# Train / eval steps (pmap for multi-device)
# ============================================================

def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk,
                      is_baseline=False, is_spatial=False,
                      sharded_fns=None):
    """Create a jit-compiled training step. Mesh SPMD handles parallelism."""

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

            if is_baseline:
                orth_loss = jnp.float32(0.0)
                div_loss = jnp.float32(0.0)
                total_loss = ce_loss
            elif is_spatial:
                orth_loss = jnp.float32(0.0)
                div_loss = compute_spatial_diversity_loss(params)
                total_loss = (ce_loss
                              + lb_weight * aux_loss
                              + div_weight * div_loss)
            else:
                orth_loss = compute_orthogonality_loss(
                    params, rank, knowledge_rank, n_feature_qk, n_restore_qk)
                div_loss = compute_knowledge_diversity_loss(params)
                total_loss = (ce_loss
                              + lb_weight * aux_loss
                              + orth_weight * orth_loss
                              + div_weight * div_loss)

            return total_loss, (ce_loss, aux_loss, orth_loss, div_loss, result)

        (total_loss, (ce_loss, aux_loss, orth_loss, div_loss, result)), grads = \
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
            'orth_loss': orth_loss,
            'div_loss': div_loss,
            'correct': result['correct'],
            'valid_count': result['valid_count'],
            'grad_norm': grad_norm,
            'attn_aux': result.get('attn_aux', jnp.float32(0.0)),
            'know_aux': result.get('know_aux', jnp.float32(0.0)),
            'know_active': result.get('know_active', jnp.float32(0.0)),
            'know_raw_gate_max': result.get('know_raw_gate_max', jnp.float32(0.0)),
            'know_gate_sum': result.get('know_gate_sum', jnp.float32(0.0)),
            'know_gate_conc': result.get('know_gate_conc', jnp.float32(0.0)),
            'attn_qk_active': result.get('attn_qk_active', jnp.float32(0.0)),
            'attn_v_active': result.get('attn_v_active', jnp.float32(0.0)),
            'attn_raw_gate_max': result.get('attn_raw_gate_max', jnp.float32(0.0)),
            'attn_gate_sum': result.get('attn_gate_sum', jnp.float32(0.0)),
            'attn_gate_conc': result.get('attn_gate_conc', jnp.float32(0.0)),
            'attn_out_norm': result.get('attn_out_norm', jnp.float32(0.0)),
            'attn_tau_mean': result.get('attn_tau_mean', jnp.float32(0.0)),
            'know_tau_mean': result.get('know_tau_mean', jnp.float32(0.0)),
            'know_emb_norm': result.get('know_emb_norm', jnp.float32(0.0)),
            'know_read_norm': result.get('know_read_norm', jnp.float32(0.0)),
            'know_write_norm': result.get('know_write_norm', jnp.float32(0.0)),
            'qk_output_scale': result.get('qk_output_scale', jnp.float32(1.0)),
            'v_output_scale': result.get('v_output_scale', jnp.float32(1.0)),
            'know_output_scale': result.get('know_output_scale', jnp.float32(1.0)),
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
            'per_layer_attn_out_norm': result.get('per_layer_attn_out_norm', jnp.zeros(1)),
            'per_layer_know_out_norm': result.get('per_layer_know_out_norm', jnp.zeros(1)),
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


def get_param_shardings(params, mesh):
    """Create sharding specs for params: neuron_pool N-axis on 'model', rest replicated."""
    replicated = NamedSharding(mesh, P())  # no sharding
    n_sharded = NamedSharding(mesh, P('model', None))  # N axis on model
    n_sharded_3d = NamedSharding(mesh, P('model', None, None))

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
    # All hosts detect the same checkpoint for consistency
    if not cli_args.from_scratch:
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
        print(f"DAWN Training (Multi-Host Multi-Device) — {cfg['model'].get('model_version', 'unknown')}")
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

    # Weight decay mask: disable for unit-norm re-projected params
    def create_wd_mask(params):
        def _mask(path, _):
            path_str = '/'.join(str(p) for p in path)
            # No WD for unit-norm re-projected params
            if 'neuron_pool' in path_str:
                for key in ['_emb', '_read', '_write']:
                    if key in path_str:
                        return False
                # No WD for learnable output_scale (like bias)
                if 'output_scale' in path_str:
                    return False
            return True
        return jax.tree.map_with_path(_mask, params)

    wd_mask = create_wd_mask(params)

    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.0, b2=0.95),
        optax.masked(
            optax.add_decayed_weights(weight_decay),
            wd_mask,
        ),
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
                  or model_version.startswith('spatial-r1-v3'))

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
    param_shardings = get_param_shardings(params, mesh)
    params = shard_params_to_mesh(params, param_shardings)
    opt_state = optimizer.init(params)  # reinit with sharded params

    # Create shard_map functions if mesh_model > 1
    _sharded_fns = None
    if mesh_model > 1:
        from models.dawn_spatial_v3 import make_sharded_srw, make_sharded_srw_paired
        max_chunk = cfg['training'].get('max_chunk_size', 12500)
        _sharded_single = make_sharded_srw(mesh, max_chunk_size=max_chunk)
        _sharded_paired = make_sharded_srw_paired(mesh, max_chunk_size=max_chunk)
        _sharded_fns = (_sharded_single, _sharded_paired)
        if is_host0:
            print(f"  shard_map enabled (mesh_model={mesh_model}, QK fused)")

    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        rank, knowledge_rank, n_feature_qk, n_restore_qk,
        is_baseline=is_baseline, is_spatial=is_spatial,
        sharded_fns=_sharded_fns)
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

            from models.dawn_spatial_v3 import (
                _layer_norm, _attn_forward, _know_forward, _srw_chunked)

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
                QK_out, act, gm, _lb, _ss, _gs, _gc, _sm = fused_paired(
                    x, h_QK, qk_norm, tau_QK, qk_read, qk_write)
                return QK_out[:, :, 0, :], QK_out[:, :, 1, :], act, gm

            # 3b) QK non-sharded fallback
            @jax.jit
            def prof_qk_chunked(x, h_Q, h_K, qk_norm, tau_all, qk_read, qk_write):
                Q, _, _, _, _, _, _, _ = _srw_chunked(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                                       qk_read, qk_write, n_chunks_qk)
                K, _, _, _, _, _, _, _ = _srw_chunked(x, h_K, qk_norm, tau_all[:, :, 1:2],
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
                Q, K, _, _ = prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, _, _, _, _, _, _, _ = prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write'])
            else:
                Q, K = prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, _, _, _, _, _, _, _ = prof_v_chunked(
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
                      f"{'HBM Δ':>7s}  {'Peak':>7s}  {''}",
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

        n_params = count_parameters(jax.device_get(params))
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
        print(f"\n!!! SIGTERM received (host {host_id}) — saving emergency checkpoint (step={global_step}) !!!", flush=True)
        try:
            params_single = jax.device_get(params)
            opt_state_single = jax.device_get(opt_state)
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
        win_orth = 0.0
        win_div = 0.0
        win_correct = 0
        win_valid = 0
        win_count = 0
        win_start_time = time.time()

        for local_step, (input_ids, attention_mask) in enumerate(train_loader):

            if preemption_requested[0]:
                if is_host0:
                    print("Preemption requested — exiting training loop.", flush=True)
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
            m_orth = _m(metrics['orth_loss'])
            m_div = _m(metrics['div_loss'])
            m_correct = int(_m(metrics['correct']))
            m_valid = int(_m(metrics['valid_count']))

            # NaN/INF detection
            if check_nan_inf({
                'total_loss': m_total, 'ce_loss': m_ce, 'aux_loss': m_aux,
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
                        f"loss={avg_loss:.4f} ce={avg_ce:.4f} aux={avg_aux:.4f} "
                        f"orth={avg_orth:.2e} div={avg_div:.2e} | "
                        f"acc={avg_acc:.4f} lr={current_lr:.2e} "
                        f"{format_time(epoch_elapsed)}<{format_time(eta)}, {s_per_it:.2f}s/it"
                    )
                    log_message(msg)

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
                        k_act = _m(metrics['know_active'])
                        k_raw_gmax = _m(metrics['know_raw_gate_max'])
                        k_gsum = _m(metrics.get('know_gate_sum', 0.0))
                        k_gconc = _m(metrics.get('know_gate_conc', 0.0))

                        a_qk_act = _m(metrics.get('attn_qk_active', 0.0))
                        a_v_act = _m(metrics.get('attn_v_active', 0.0))
                        a_raw_gmax = _m(metrics.get('attn_raw_gate_max', 0.0))
                        a_gsum = _m(metrics.get('attn_gate_sum', 0.0))
                        a_gconc = _m(metrics.get('attn_gate_conc', 0.0))
                        a_out_n = _m(metrics.get('attn_out_norm', 0.0))

                        a_tau_m = _m(metrics.get('attn_tau_mean', 0.0))
                        k_tau_m = _m(metrics.get('know_tau_mean', 0.0))
                        k_out_n = _m(metrics.get('know_out_norm', 0.0))

                        log_message(
                            f"      {tau_s} | tau_mean: attn={a_tau_m:.3f}"
                            f" know={k_tau_m:.3f} | grad_norm={m_grad:.3f}")
                        qk_os = _m(metrics.get('qk_output_scale', 1.0))
                        v_os = _m(metrics.get('v_output_scale', 1.0))
                        k_os = _m(metrics.get('know_output_scale', 1.0))
                        log_message(
                            f"      aux: attn={m_attn_aux:.4f} know={m_know_aux:.4f}"
                            f" | norms: emb={k_emb_n:.3f} read={k_read_n:.3f}"
                            f" write={k_write_n:.3f}"
                            f" | oscale: qk={qk_os:.3f} v={v_os:.3f} k={k_os:.3f}")
                        k_raw_n = _m(metrics.get('know_raw_out_norm', 0.0))
                        a_qk_raw_n = _m(metrics.get('attn_qk_raw_norm', 0.0))
                        a_v_raw_n = _m(metrics.get('attn_v_raw_norm', 0.0))

                        log_message(
                            f"      know: active={k_act * n_know_cfg:.0f}/{n_know_cfg}"
                            f"({k_act*100:.1f}%) raw_max={k_raw_gmax:.4f}"
                            f" conc={k_gconc:.1f}"
                            f" gsum={k_gsum:.1f}"
                            f" raw_norm={k_raw_n:.6f} out_norm={k_out_n:.3f}")
                        log_message(
                            f"      attn: qk_active={a_qk_act:.1%}"
                            f" v_active={a_v_act:.1%}"
                            f" raw_max={a_raw_gmax:.4f}"
                            f" conc={a_gconc:.1f}"
                            f" gsum={a_gsum:.1f}"
                            f" qk_raw={a_qk_raw_n:.6f} v_raw={a_v_raw_n:.6f}"
                            f" out_norm={a_out_n:.3f}")
                        if _early_debug or debug_mode:
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

                    # JSONL structured log
                    log_jsonl({
                        'type': 'train',
                        'step': global_step,
                        'epoch': epoch,
                        'total_loss': avg_loss,
                        'ce_loss': avg_ce,
                        'aux_loss': avg_aux,
                        'orth_loss': avg_orth,
                        'div_loss': avg_div,
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
                    params_single = jax.device_get(params)
                    opt_state_single = jax.device_get(opt_state)
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

            # ---- Mid-epoch checkpoint ----
            if global_step % ckpt_interval == 0 and global_step > 0:
                # device_get on ALL hosts (may be collective for sharded params)
                params_single = jax.device_get(params)
                opt_state_single = jax.device_get(opt_state)
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
        params_single = jax.device_get(params)
        opt_state_single = jax.device_get(opt_state)

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

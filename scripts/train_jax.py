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
from datetime import datetime
from functools import partial

from models.model_v17_1_jax import DAWN
from models.dawn_spatial import DAWN as DAWN_Spatial
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
    """
    pool = params['neuron_pool']

    def _pool_div(neurons):
        n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.matmul(n, n.T)
        mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
        return jnp.abs(sim * mask).sum() / mask.sum()

    return (_pool_div(pool['qk_neurons']) +
            _pool_div(pool['v_neurons']) +
            _pool_div(pool['know_neurons'])) / 3


# ============================================================
# Train / eval steps (pmap for multi-device)
# ============================================================

def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk,
                      n_devices=1, is_baseline=False, is_spatial=False):
    """Create a compiled training step function.

    Uses jax.pmap for multi-device data parallelism.
    When n_devices=1, pmap degenerates to single-device execution.
    """

    @partial(jax.pmap, axis_name='dp')
    def train_step(params, opt_state, input_ids, attention_mask, dropout_key):
        # Labels for CLM: input_ids shifted, padding masked
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            result = model.apply(
                {'params': params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={'dropout': dropout_key},
            )
            ce_loss = result['loss']
            aux_loss = result['aux_loss']

            if is_baseline:
                orth_loss = jnp.float32(0.0)
                div_loss = jnp.float32(0.0)
                total_loss = ce_loss
            elif is_spatial:
                # Spatial-r1: no orthogonality (rank-1 vectors), diversity only
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

        # All-reduce gradients across devices
        grads = jax.lax.pmean(grads, axis_name='dp')

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Aggregate metrics across devices
        metrics = {
            'total_loss': jax.lax.pmean(total_loss, axis_name='dp'),
            'ce_loss': jax.lax.pmean(ce_loss, axis_name='dp'),
            'aux_loss': jax.lax.pmean(aux_loss, axis_name='dp'),
            'orth_loss': jax.lax.pmean(orth_loss, axis_name='dp'),
            'div_loss': jax.lax.pmean(div_loss, axis_name='dp'),
            'correct': jax.lax.psum(result['correct'], axis_name='dp'),
            'valid_count': jax.lax.psum(result['valid_count'], axis_name='dp'),
        }

        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step(model, n_devices=1):
    """Create a compiled evaluation step function.

    Note: dropout RNG is required because the lax.scan forward path always
    calls make_rng('dropout') (safe_dropout neutralizes it via deterministic flag).
    """

    @partial(jax.pmap, axis_name='dp')
    def eval_step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        # deterministic=True -> dropout masks are all-ones, but RNG key is still
        # needed for tracing (safe_dropout always generates a mask).
        eval_rng = jax.random.PRNGKey(0)  # fixed key -- never used for real randomness
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
        )
        ce_loss = result['loss']
        correct = result['correct']
        valid_count = result['valid_count']

        return (
            jax.lax.pmean(ce_loss, axis_name='dp'),
            jax.lax.psum(correct, axis_name='dp'),
            jax.lax.psum(valid_count, axis_name='dp'),
        )

    return eval_step


# ============================================================
# Multi-device helpers
# ============================================================

def replicate(pytree, devices=None):
    """Replicate a pytree across devices."""
    if devices is None:
        devices = jax.devices()
    return jax.device_put_replicated(pytree, devices)


def unreplicate(pytree):
    """Extract first replica from a replicated pytree."""
    return jax.tree.map(lambda x: x[0], pytree)


def shard_batch(batch, n_devices):
    """Reshape a batch for pmap: (B, ...) -> (n_devices, B//n_devices, ...).

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

def evaluate(eval_step_fn, params, val_loader, n_devices, max_batches=200, verbose=True):
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

        # Ensure sharded for pmap
        input_ids = shard_batch(input_ids, n_devices)
        attention_mask = shard_batch(attention_mask, n_devices)

        ce_loss, correct, valid_count = eval_step_fn(params, input_ids, attention_mask)

        # Extract from first device (already aggregated via pmean/psum)
        n_valid = int(valid_count[0])
        total_loss += float(ce_loss[0]) * n_valid
        total_correct += int(correct[0])
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
        print(f"DAWN v17.1-JAX Training (Multi-Host Multi-Device)")
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
        n_devices=n_local_devices,
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

    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, b2=0.95),
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
    # Replicate params/opt_state across local devices
    # ----------------------------------------------------------
    params = replicate(params, local_devices)
    opt_state = replicate(opt_state, local_devices)

    # ----------------------------------------------------------
    # Create pmap-compiled step functions
    # ----------------------------------------------------------
    n_feature_qk = cfg['model'].get('n_feature_qk', 56)
    n_restore_qk = cfg['model'].get('n_restore_qk', 56)
    model_version = cfg['model'].get('model_version', '17.1')
    is_baseline = model_version == 'baseline'
    is_spatial = model_version == 'spatial-r1'
    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        rank, knowledge_rank, n_feature_qk, n_restore_qk,
        n_devices=n_local_devices, is_baseline=is_baseline,
        is_spatial=is_spatial)
    eval_step_fn = create_eval_step(model, n_devices=n_local_devices)

    # ----------------------------------------------------------
    # OOM check + JIT pre-compile: real train_step (forward + backward)
    # All hosts must participate (pmap requires it)
    # ----------------------------------------------------------
    if is_host0:
        print(f"\n=== OOM check: real train_step (forward+backward) "
              f"per_device_batch={per_device_batch}, seq_len={max_seq_len} ===", flush=True)
    try:
        dummy_ids = jnp.zeros((n_local_devices, per_device_batch, max_seq_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((n_local_devices, per_device_batch, max_seq_len), dtype=jnp.int32)
        rng, dummy_step_rng = jax.random.split(rng)
        dummy_dropout_keys = jax.random.split(dummy_step_rng, n_local_devices)

        # First call: JIT compilation (slow)
        jit_start = time.time()
        _dummy_params, _dummy_opt, dummy_metrics = train_step_fn(
            params, opt_state, dummy_ids, dummy_mask, dummy_dropout_keys,
        )
        jax.block_until_ready(dummy_metrics['total_loss'])
        jit_time = time.time() - jit_start
        if is_host0:
            print(f"  JIT compile: {jit_time:.1f}s", flush=True)

        # Second call: measure actual step time (post-JIT)
        rng, dummy_step_rng2 = jax.random.split(rng)
        dummy_dropout_keys2 = jax.random.split(dummy_step_rng2, n_local_devices)
        step_start = time.time()
        _dummy_params2, _dummy_opt2, dummy_metrics2 = train_step_fn(
            params, opt_state, dummy_ids, dummy_mask, dummy_dropout_keys2,
        )
        jax.block_until_ready(dummy_metrics2['total_loss'])
        step_time = time.time() - step_start
        if is_host0:
            print(f"  train_step OK -- loss={float(dummy_metrics['total_loss'][0]):.4f}", flush=True)
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

            # Estimate total training time
            total_steps = len(train_loader) * num_epochs
            remaining_steps = total_steps - global_step
            est_seconds = remaining_steps * step_time
            est_hours = est_seconds / 3600
            print(f"  Estimated time: {est_hours:.1f}h ({remaining_steps:,} steps @ {step_time*1000:.1f}ms)", flush=True)

        del _dummy_params, _dummy_opt, dummy_metrics, dummy_ids, dummy_mask, dummy_dropout_keys
        del _dummy_params2, _dummy_opt2, dummy_metrics2, dummy_dropout_keys2
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

        n_params = count_parameters(unreplicate(params))
        log_message(f"DAWN v17.1-JAX Training Log (Multi-Host) - {timestamp}")
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
        if is_host0:
            print(f"\n!!! SIGTERM received — saving emergency checkpoint (step={global_step}) !!!", flush=True)
            try:
                params_single = unreplicate(params)
                opt_state_single = unreplicate(opt_state)
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

            # Ensure sharded for pmap (local devices only)
            input_ids = shard_batch(input_ids, n_local_devices)
            attention_mask = shard_batch(attention_mask, n_local_devices)

            # Different dropout key per local device per step
            rng, step_rng = jax.random.split(rng)
            dropout_keys = jax.random.split(step_rng, n_local_devices)

            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                input_ids, attention_mask, dropout_keys,
            )

            # Extract metrics (take first device, already aggregated via pmean/psum)
            m_total = float(metrics['total_loss'][0])
            m_ce = float(metrics['ce_loss'][0])
            m_aux = float(metrics['aux_loss'][0])
            m_orth = float(metrics['orth_loss'][0])
            m_div = float(metrics['div_loss'][0])
            m_correct = int(metrics['correct'][0])
            m_valid = int(metrics['valid_count'][0])

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
            if global_step % LOG_INTERVAL == 0:
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

                    msg = (
                        f"[Step {global_step}/{total_micro_steps} ({progress:.1f}%)] "
                        f"loss={avg_loss:.4f} ce={avg_ce:.4f} aux={avg_aux:.4f} "
                        f"orth={avg_orth:.2e} div={avg_div:.2e} | "
                        f"acc={avg_acc:.4f} lr={current_lr:.2e} "
                        f"{format_time(epoch_elapsed)}<{format_time(eta)}, {s_per_it:.2f}s/it"
                    )
                    log_message(msg)

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
                val_loss, val_acc = evaluate(eval_step_fn, params, val_loader,
                                             n_local_devices, verbose=is_host0)
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

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        params_single = unreplicate(params)
                        opt_state_single = unreplicate(opt_state)
                        save_checkpoint(
                            _ckpt_path("best_model.flax"),
                            params_single, opt_state_single,
                            epoch, global_step, best_val_loss,
                            cfg['model'],
                            step_in_epoch=epoch_step_counter,
                            steps_per_epoch=steps_per_epoch,
                            training_config=training_config,
                        )
                        del params_single, opt_state_single
                        log_message(f"  New best model saved! val_loss={best_val_loss:.4f}")

            # ---- Mid-epoch checkpoint (host 0 only) ----
            if global_step % ckpt_interval == 0 and global_step > 0 and is_host0:
                params_single = unreplicate(params)
                opt_state_single = unreplicate(opt_state)
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
        val_loss, val_acc = evaluate(eval_step_fn, params, val_loader,
                                     n_local_devices, verbose=is_host0)

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

            # Save epoch checkpoint (host 0 only)
            params_single = unreplicate(params)
            opt_state_single = unreplicate(opt_state)

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

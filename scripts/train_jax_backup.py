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
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather
try:
    from jax.experimental.multihost_utils import broadcast_one_to_all as _bcast_one_to_all
    _HAVE_BROADCAST = True
except ImportError:
    _bcast_one_to_all = None
    _HAVE_BROADCAST = False
import optax
import numpy as np
import time
import random
import argparse
import yaml
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map

# Model registry imports. Legacy DAWN experiments live in models/legacy/;
# restore and re-register them only when reproducing an old run.
from models.baseline_transformer_jax import VanillaTransformer
from models.dawn_spatial_v394_exp import DAWN as DAWN_V394
from models.dawn_spatial_v415_operator_route_sig_exp import DAWN as DAWN_V415
from models.dawn_spatial_v4152_operator_route_free_emb_exp import DAWN as DAWN_V4152
from models.dawn_spatial_v4152_tensor_parallel import DAWN as DAWN_V4152_TP

# ============================================================
# Constants
# ============================================================

# Log cadence is config-driven: see log_interval / log_analysis_multiplier
# in `training:`. The legacy module-level LOG_INTERVAL constant was removed.



# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def print_xla_oom_diagnostics():
    """Print the newest XLA dump files likely to contain HBM allocation info."""
    dump_dir = Path(os.environ.get("XLA_DUMP_DIR", "/tmp/xla_dump_train"))
    if not dump_dir.exists():
        print(f"  XLA dump dir not found: {dump_dir}", flush=True)
        print("  Set XLA_DUMP_DIR and XLA_FLAGS=--xla_dump_to=$XLA_DUMP_DIR "
              "--xla_dump_hlo_as_text before launching Python.", flush=True)
        return

    patterns = (
        "*memory*",
        "*buffer*",
        "*after_optimizations.txt",
        "*.txt",
    )
    files = []
    seen = set()
    for pat in patterns:
        for path in dump_dir.rglob(pat):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[:12]

    if not files:
        print(f"  No XLA text dumps found under {dump_dir}", flush=True)
        return

    print(f"\n  === XLA OOM diagnostics ===", flush=True)
    print(f"  Dump dir: {dump_dir}", flush=True)
    print("  Newest relevant dump files:", flush=True)
    for path in files[:5]:
        try:
            size_mb = path.stat().st_size / 1e6
            print(f"    {path} ({size_mb:.1f} MB)", flush=True)
        except OSError:
            print(f"    {path}", flush=True)

    needles = (
        "Total hbm usage",
        "Program hbm requirement",
        "Largest program allocations",
        "Allocation type: HLO temp",
        "Size:",
        "source_file=",
        "Shape:",
    )
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        is_memory_report = "memory-usage-report" in path.name
        if not is_memory_report and not any(n in text for n in needles[:3]):
            continue
        print(f"\n  --- XLA memory excerpt: {path} ---", flush=True)
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines)
                if any(n in line for n in needles[:3])
                or "bytes used" in line.lower()
                or "allocation" in line.lower()]
        start = max(0, hits[0] - 2) if hits else 0
        end = min(len(lines), start + 120)
        printed = 0
        for line in lines[start:end]:
            interesting = (
                any(n in line for n in needles)
                or "Operator:" in line
                or "bytes used" in line.lower()
                or "allocation" in line.lower()
                or is_memory_report
            )
            if interesting:
                print(f"  {line[:240]}", flush=True)
                printed += 1
                if printed >= 60:
                    print("  ... excerpt truncated; inspect dump file above for full report.",
                          flush=True)
                    break
        return

    print("  No memory report excerpt found yet. Inspect latest dumps above.",
          flush=True)


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


# ============================================================
# Model Registry
# ============================================================

@dataclass
class ModelSpec:
    """Registered model entry.

    - build_kwargs: cfg -> kwargs dict for the model constructor.
    - sharded_kwargs: cfg -> extra kwargs for make_sharded_srw /
      make_sharded_srw_paired (v4.1 closure constants live here).
    - force_sharded: require shard_map path even when mesh_model==1
      (v4.1 removed non-sharded fallback).
    """
    name: str
    module_path: str
    cls: Any
    build_kwargs: Callable[[dict], dict]
    supports_sharded: bool = False
    force_sharded: bool = False
    sharded_kwargs: Optional[Callable[[dict], dict]] = None
    tensor_cls: Any = None
    tensor_module_path: Optional[str] = None


def _baseline_kwargs(cfg):
    m = cfg['model']
    return dict(
        vocab_size=m.get('vocab_size', 30522),
        d_model=m.get('d_model', 384),
        d_ff=m.get('d_ff', 1536),
        n_layers=m.get('n_layers', 12),
        n_heads=m.get('n_heads', 6),
        max_seq_len=m.get('max_seq_len', 512),
        dropout_rate=m.get('dropout', 0.1),
        gradient_checkpointing=m.get('gradient_checkpointing', False),
    )


def _dawn_shared_kwargs(cfg):
    """Init kwargs shared by active DAWN variants with the base signature."""
    m = cfg['model']
    t = cfg['training']
    return dict(
        vocab_size=m.get('vocab_size', 30522),
        d_model=m.get('d_model', 384),
        n_layers=m.get('n_layers', 12),
        n_heads=m.get('n_heads', 6),
        max_seq_len=m.get('max_seq_len', 512),
        d_route=m.get('d_route', m.get('d_bottleneck', 128)),
        n_qk=m.get('n_qk', 1580),
        n_v=m.get('n_v', 2600),
        n_know=m.get('n_know', 25200),
        dropout_rate=m.get('dropout', 0.1),
        router_dropout=m.get('router_dropout', 0.1),
        gradient_checkpointing=m.get('gradient_checkpointing', False),
        n_chunks_know=t.get('n_chunks_know', 1),
        n_chunks_qk=t.get('n_chunks_qk', 1),
        n_chunks_v=t.get('n_chunks_v', 1),
    )


def _dawn_v415_kwargs(cfg):
    kw = _dawn_shared_kwargs(cfg)
    m = cfg['model']
    tag_dim = m.get('tag_dim', 16)
    read_sig_dim = m.get('read_sig_dim', 24)
    write_sig_dim = m.get('write_sig_dim', 24)
    expected_route = tag_dim + read_sig_dim + write_sig_dim
    d_route = m.get('d_route', expected_route)
    if d_route != expected_route:
        raise ValueError(
            f"d_route must equal tag_dim + read_sig_dim + write_sig_dim, got "
            f"d_route={d_route}, tag_dim={tag_dim}, "
            f"read_sig_dim={read_sig_dim}, write_sig_dim={write_sig_dim}")
    kw['d_route'] = d_route
    kw['tag_dim'] = tag_dim
    kw['read_sig_dim'] = read_sig_dim
    kw['write_sig_dim'] = write_sig_dim
    return kw


def _v415_sharded_kwargs(cfg):
    """Gate constants for the active v4.1.5 sharded SRW path."""
    t = cfg['training']
    return dict(
        dead_threshold=t.get('dead_penalty_threshold', 0.01),
        sharpness=t.get('sharpness', 500.0),
        activation_threshold=t.get('activation_threshold', 0.5),
        activation_cutoff=t.get('activation_cutoff', 0.01),
        epsilon=t.get('epsilon', 1e-4),
        max_intensity=t.get('max_intensity', 10.0),
        scan_scale=t.get('scan_scale', 0.01),
        scan_std_floor=t.get('scan_std_floor', 0.5),
    )


MODEL_REGISTRY = {
    'baseline': ModelSpec(
        name='baseline',
        module_path='models.baseline_transformer_jax',
        cls=VanillaTransformer,
        build_kwargs=_baseline_kwargs,
    ),
    'spatial-r1-v3.9.4': ModelSpec(
        name='spatial-r1-v3.9.4',
        module_path='models.dawn_spatial_v394_exp',
        cls=DAWN_V394,
        build_kwargs=_dawn_shared_kwargs,
        supports_sharded=True,
    ),
    'spatial-r1-v4.1.5': ModelSpec(
        name='spatial-r1-v4.1.5',
        module_path='models.dawn_spatial_v415_operator_route_sig_exp',
        cls=DAWN_V415,
        build_kwargs=_dawn_v415_kwargs,
        supports_sharded=True,
        force_sharded=True,
        sharded_kwargs=_v415_sharded_kwargs,
    ),
    'spatial-r1-v4.1.5.2': ModelSpec(
        name='spatial-r1-v4.1.5.2',
        module_path='models.dawn_spatial_v4152_operator_route_free_emb_exp',
        cls=DAWN_V4152,
        build_kwargs=_dawn_v415_kwargs,
        supports_sharded=True,
        force_sharded=True,
        sharded_kwargs=_v415_sharded_kwargs,
        tensor_cls=DAWN_V4152_TP,
        tensor_module_path='models.dawn_spatial_v4152_tensor_parallel',
    ),
}


def build_model_from_config(cfg):
    """Build model from config via MODEL_REGISTRY.

    Unknown versions raise ValueError with restoration instructions:
    legacy versions live in models/legacy/ and can be re-registered in
    MODEL_REGISTRY when resuming old checkpoints or reproducing paper
    results. See models/legacy/README.md.
    """
    version = cfg['model'].get('model_version', 'spatial-r1-v4.1.5')
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_version: {version!r}. "
            f"Known: {sorted(MODEL_REGISTRY.keys())}. "
            f"Legacy versions live in models/legacy/; to resume an old "
            f"checkpoint, move the model file back to models/ and add a "
            f"ModelSpec entry to MODEL_REGISTRY. See models/legacy/README.md.")
    spec = MODEL_REGISTRY[version]
    kwargs = spec.build_kwargs(cfg)
    cls = spec.cls
    parallelism = cfg['model'].get('parallelism', '').lower()
    if parallelism in ('tensor', 'tp'):
        if spec.tensor_cls is None:
            raise ValueError(
                f"model_version={version!r} has no tensor-parallel backend")
        cls = spec.tensor_cls
        t = cfg['training']
        m = cfg['model']
        kwargs['attention_checkpoint'] = m.get(
            'attention_checkpoint', t.get('attention_checkpoint', True))
        kwargs['loss_checkpoint'] = m.get(
            'loss_checkpoint', t.get('loss_checkpoint', True))
    if version in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2'):
        print(
            "operator route signature: "
            f"tag_dim={kwargs['tag_dim']}, "
            f"read_sig_dim={kwargs['read_sig_dim']}, "
            f"write_sig_dim={kwargs['write_sig_dim']}, "
            f"d_route={kwargs['d_route']}"
            f"{' parallelism=tensor' if cls is spec.tensor_cls else ''}")
    return cls(**kwargs)


# ============================================================
# GCS / file I/O helpers
# ============================================================

_GCS_FS_CACHE = None


def _get_gcs_fs():
    """Cached gcsfs.GCSFileSystem singleton.

    Avoids per-call auth + init overhead. Returns None if gcsfs is not
    installed; callers are expected to fall back to tensorflow or raise.
    """
    global _GCS_FS_CACHE
    if _GCS_FS_CACHE is None:
        try:
            import gcsfs
            _GCS_FS_CACHE = gcsfs.GCSFileSystem()
        except ImportError:
            return None
    return _GCS_FS_CACHE


def _is_gcs(path):
    return str(path).startswith("gs://")


def _open_file(path, mode="rb"):
    """Open a file for read/write, supporting GCS paths."""
    path_str = str(path)
    if _is_gcs(path_str):
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.open(path_str, mode)
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
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.exists(path_str)
        try:
            import tensorflow as tf
            return tf.io.gfile.exists(path_str)
        except ImportError:
            raise ImportError(
                f"Cannot check GCS path {path_str}: "
                f"neither gcsfs nor tensorflow available.")
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
        fs = _get_gcs_fs()
        if fs is not None:
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = fs.glob(dir_str + pattern)
            return sorted(["gs://" + f for f in files], key=_sort_key)
        try:
            import tensorflow as tf
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = tf.io.gfile.glob(dir_str + pattern)
            return sorted(files, key=_sort_key)
        except ImportError:
            raise ImportError(
                f"Cannot list GCS path {dir_str}: "
                f"neither gcsfs nor tensorflow available.")
    return sorted((str(f) for f in Path(dir_str).glob(pattern)), key=_sort_key)


def _makedirs(path):
    """Create directory (local only; GCS doesn't need explicit mkdir)."""
    if not _is_gcs(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _delete_file(path):
    """Delete a single file (local or GCS). Warns on failure; never silent."""
    path_str = str(path)
    if _is_gcs(path_str):
        fs = _get_gcs_fs()
        if fs is not None:
            try:
                fs.rm(path_str)
            except FileNotFoundError:
                pass
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"Warning: _delete_file({path_str}) failed: {e}", flush=True)
            return
        try:
            import tensorflow as tf
            tf.io.gfile.remove(path_str)
            return
        except ImportError:
            raise ImportError(
                f"Cannot delete GCS path {path_str}: "
                f"neither gcsfs nor tensorflow available.")
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

def _model_accepts_analysis(model):
    """Return True if model.__call__ accepts an `analysis` kwarg.

    v4.1+ accepts it (routes the full-stats forward); older versions
    don't -passing it there raises, so we must gate it.
    """
    import inspect as _inspect
    try:
        return 'analysis' in _inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      tau_reg_weight, dead_penalty_weight,
                      exploration_weight, exploration_asymmetry,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk,
                      exploration_warmup_steps=5000,
                      exploration_lower_bound=-0.5,
                      exploration_upper_bound=2.0,
                      exploration_bound_eps=1.0e-3,
                      is_baseline=False, is_spatial=False,
                      sharded_fns=None, mesh=None):
    """Create a jit-compiled training step. Mesh SPMD handles parallelism.

    v4.1 explore (redesigned): no EMA, no warmup. For every step compute
    a batch-global per-token CE mean, define
        deviation = per_token_ce - sg(global_mean_ce)
        signal    = where(deviation > 0, deviation, asymmetry 쨌 deviation)
    and add
        explore_loss = 貫 쨌 valid_weighted_mean(signal 쨌 誇 tau_offset)
    to total_loss.  Positive deviations (surprising tokens) push
    tau_offset DOWN at full strength; negative deviations (easy tokens)
    push UP at `exploration_asymmetry` of the strength.  The global mean
    baseline makes the net push roughly zero-sum each batch, so tau does
    not accumulate monotonically.

    `mesh` is required when the v4.1 exploration loss is active -the
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
    _explore_lower = jnp.float32(exploration_lower_bound)
    _explore_upper = jnp.float32(exploration_upper_bound)
    _explore_eps = jnp.float32(exploration_bound_eps)
    _warmup_steps = jnp.int32(exploration_warmup_steps)
    _pass_analysis_kw = _model_accepts_analysis(model)

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, dropout_key,
                   prev_emb_snap, step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            extra_kw = {}
            if sharded_fns is not None:
                extra_kw['sharded_fns'] = sharded_fns
            if _pass_analysis_kw:
                extra_kw['analysis'] = False
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
                # Asymmetric: full push on hard tokens, `asym`쨌push on easy ones.
                signal = jnp.where(deviation > 0, deviation, _asym * deviation)

                # v4.1+ per-token/layer/route bounded explore. tau_offset is
                # NOT clipped (CE keeps full gradient); only the explore-loss
                # contribution is turned off per-element when further push in
                # that direction would breach [lower, upper].
                # attn_tau_off shape [L, B, S, 3]; know_tau_off [L, B, S, 1].
                a_tau_t = attn_tau_off[:, :, :-1, :]     # [L, B, S-1, 3]
                k_tau_t = know_tau_off[:, :, :-1, :]     # [L, B, S-1, 1]
                sig_b = signal[None, :, :, None]          # [1, B, S-1, 1]
                vmask_b = vmask_f[None, :, :, None]       # [1, B, S-1, 1]

                # Per-element bound-hit masks. Hard off, not soft decay.
                a_down_off = (sig_b > 0) & (a_tau_t <= _explore_lower + _explore_eps)
                a_up_off   = (sig_b < 0) & (a_tau_t >= _explore_upper - _explore_eps)
                a_off_mask = a_down_off | a_up_off
                k_down_off = (sig_b > 0) & (k_tau_t <= _explore_lower + _explore_eps)
                k_up_off   = (sig_b < 0) & (k_tau_t >= _explore_upper - _explore_eps)
                k_off_mask = k_down_off | k_up_off

                a_active = jnp.where(a_off_mask, 0.0, 1.0)
                k_active = jnp.where(k_off_mask, 0.0, 1.0)

                # tau_offset distribution diagnostics (stop_gradient, obs-only).
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

                # Per-element contribution -reduce. Gradient flows through
                # the tau_offset tensor only (signal is stop_gradient'd).
                vsum_eps = vmask_f.sum() + 1e-8
                a_contrib = sig_b * a_tau_t * a_active * vmask_b
                k_contrib = sig_b * k_tau_t * k_active * vmask_b
                explore_attn_raw = a_contrib.sum() / vsum_eps
                explore_know_raw = k_contrib.sum() / vsum_eps
                explore_loss_raw = explore_attn_raw + explore_know_raw

                # Observational stats (same interface as before).
                pos_mask = (deviation > 0).astype(jnp.float32) * vmask_f
                neg_mask = (deviation < 0).astype(jnp.float32) * vmask_f
                pos_frac = pos_mask.sum() / vsum_eps
                pos_mean = (jnp.maximum(deviation, 0.0) * vmask_f).sum() / (
                    pos_mask.sum() + 1e-8)
                neg_mean = (jnp.maximum(-deviation, 0.0) * vmask_f).sum() / (
                    neg_mask.sum() + 1e-8)

                # Off fractions replace pool-mean block fractions. Denominator
                # is total (layer 횞 batch 횞 valid-time 횞 route) slots.
                _a_tot = vmask_b.sum() * a_tau_t.shape[0] * a_tau_t.shape[-1]
                _k_tot = vmask_b.sum() * k_tau_t.shape[0] * k_tau_t.shape[-1]
                block_frac_a = jax.lax.stop_gradient(
                    (a_off_mask.astype(jnp.float32) * vmask_b).sum() / (_a_tot + 1e-8))
                block_frac_k = jax.lax.stop_gradient(
                    (k_off_mask.astype(jnp.float32) * vmask_b).sum() / (_k_tot + 1e-8))

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
            # Warmup gate: zero out explore loss until warmup_steps has passed.
            # W_sense needs time to settle before exploration signals become
            # meaningful; early CE-dominated learning keeps tau gradient clean.
            explore_active = (step >= _warmup_steps).astype(jnp.float32)
            explore_loss_weighted = exploration_weight * explore_loss_raw * explore_active

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
                explore_active=explore_active,
                step_in_train=step,
            )
            return total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                                dead_penalty, explore_stats, result)

        (total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                      dead_penalty, explore_stats, result)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)

        # XLA SPMD handles gradient all-reduce automatically
        # (loss computed on sharded data -gradients consistent across shards)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))

        def _tree_norm(tree):
            leaves = jax.tree.leaves(tree)
            if not leaves:
                return jnp.float32(0.0)
            return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                                for x in leaves) + 1e-12)

        def _child_norm(tree, key):
            return _tree_norm(tree[key]) if key in tree else jnp.float32(0.0)

        _grouter = grads.get('router', {})
        _gpool = grads.get('neuron_pool', {})
        grad_router_proj = (
            _child_norm(_grouter, 'proj_attn') + _child_norm(_grouter, 'proj_know'))
        grad_router_tau = (
            _child_norm(_grouter, 'tau_attn') + _child_norm(_grouter, 'tau_know')
            + _child_norm(_grouter, 'tau_q') + _child_norm(_grouter, 'tau_k')
            + _child_norm(_grouter, 'tau_v'))
        grad_router_scan = (
            _child_norm(_grouter, 'scan_bias_attn')
            + _child_norm(_grouter, 'scan_bias_know'))
        grad_pool_emb = (
            _child_norm(_gpool, 'qk_emb') + _child_norm(_gpool, 'v_emb')
            + _child_norm(_gpool, 'know_emb'))
        grad_pool_read = (
            _child_norm(_gpool, 'qk_read') + _child_norm(_gpool, 'v_read')
            + _child_norm(_gpool, 'know_read') + _child_norm(_gpool, 'q_read')
            + _child_norm(_gpool, 'k_read'))
        grad_pool_write = (
            _child_norm(_gpool, 'qk_write') + _child_norm(_gpool, 'v_write')
            + _child_norm(_gpool, 'know_write') + _child_norm(_gpool, 'q_write')
            + _child_norm(_gpool, 'k_write'))

        # Emb drift is computed inside jit so every host participates in the
        # norm collective. A host-0-only version halts the launch group on
        # multi-host meshes.
        _pool = new_params['neuron_pool']
        if 'qk_emb' in _pool:
            _cur_qk = _pool['qk_emb']
            _cur_v = _pool['v_emb']
            _cur_know = _pool['know_emb']
        else:
            # Some archived pool variants expose read tensors instead of emb
            # tensors; keep the diagnostic slots comparable when resuming them.
            _cur_qk = _pool['q_read']
            _cur_v = _pool['v_read']
            _cur_know = _pool['know_read']
        _prev_qk = prev_emb_snap['qk_emb']
        _prev_v = prev_emb_snap['v_emb']
        _prev_know = prev_emb_snap['know_emb']
        drift_qk_emb = (jnp.linalg.norm(_cur_qk - _prev_qk)
                        / (jnp.linalg.norm(_prev_qk) + 1e-8))
        drift_v_emb = (jnp.linalg.norm(_cur_v - _prev_v)
                       / (jnp.linalg.norm(_prev_v) + 1e-8))
        drift_know_emb = (jnp.linalg.norm(_cur_know - _prev_know)
                          / (jnp.linalg.norm(_prev_know) + 1e-8))

        # Tau / scan biases (read inside jit -safe, no cross-device issue)
        tau_know_b = params.get('router', {}).get('tau_know', {}).get(
            'bias', jnp.zeros(1))
        tau_attn_b = params.get('router', {}).get('tau_attn', {}).get(
            'bias', jnp.zeros(3))
        tau_q_b = params.get('router', {}).get('tau_q', {}).get(
            'bias', jnp.zeros(1))
        tau_k_b = params.get('router', {}).get('tau_k', {}).get(
            'bias', jnp.zeros(1))
        tau_v_b = params.get('router', {}).get('tau_v', {}).get(
            'bias', jnp.zeros(1))
        scan_know_b = params.get('router', {}).get('scan_bias_know', {}).get(
            'bias', jnp.zeros(1))
        scan_attn_b = params.get('router', {}).get('scan_bias_attn', {}).get(
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
            'grad_router_proj': grad_router_proj,
            'grad_router_tau': grad_router_tau,
            'grad_router_scan': grad_router_scan,
            'grad_pool_emb': grad_pool_emb,
            'grad_pool_read': grad_pool_read,
            'grad_pool_write': grad_pool_write,
            'attn_aux': result.get('attn_aux', jnp.float32(0.0)),
            'know_aux': result.get('know_aux', jnp.float32(0.0)),
            # Core activity (v4.1).
            'know_active': result.get('know_active', jnp.float32(0.0)),
            'know_strong': result.get('know_strong', jnp.float32(0.0)),
            'know_score_std': result.get('know_score_std', jnp.float32(0.0)),
            'know_raw_gate_max': result.get('know_raw_gate_max', jnp.float32(0.0)),
            'know_gate_sum': result.get('know_gate_sum', jnp.float32(0.0)),
            'know_active_n_mean': result.get('know_active_n_mean', jnp.float32(0.0)),
            'attn_qk_active': result.get('attn_qk_active', jnp.float32(0.0)),
            'attn_v_active': result.get('attn_v_active', jnp.float32(0.0)),
            'attn_strong': result.get('attn_strong', jnp.float32(0.0)),
            'attn_qk_strong': result.get(
                'attn_qk_strong',
                result.get('attn_strong', jnp.float32(0.0))),
            'attn_v_strong': result.get(
                'attn_v_strong',
                result.get('attn_strong', jnp.float32(0.0))),
            'attn_score_std': result.get('attn_score_std', jnp.float32(0.0)),
            'attn_score_mean': result.get('attn_score_mean', jnp.float32(0.0)),
            'attn_raw_gate_max': result.get('attn_raw_gate_max', jnp.float32(0.0)),
            'attn_gate_sum': result.get('attn_gate_sum', jnp.float32(0.0)),
            'attn_active_n_mean': result.get('attn_active_n_mean', jnp.float32(0.0)),
            'attn_out_norm': result.get('attn_out_norm', jnp.float32(0.0)),
            # tau structure.
            'attn_tau_mean': result.get('attn_tau_mean', jnp.float32(0.0)),
            'know_tau_mean': result.get('know_tau_mean', jnp.float32(0.0)),
            'know_score_mean': result.get('know_score_mean', jnp.float32(0.0)),
            'attn_tau_abs_mean': result.get('attn_tau_abs_mean', jnp.float32(0.0)),
            'know_tau_abs_mean': result.get('know_tau_abs_mean', jnp.float32(0.0)),
            # Emb norm stats (REGULAR subset; *_max moved to analysis_step).
            'know_emb_norm': result.get('know_emb_norm', jnp.float32(0.0)),
            'know_emb_norm_min': result.get('know_emb_norm_min', jnp.float32(0.0)),
            'know_emb_norm_std': result.get('know_emb_norm_std', jnp.float32(0.0)),
            'qk_emb_norm_mean': result.get('qk_emb_norm_mean', jnp.float32(0.0)),
            'qk_emb_norm_min': result.get('qk_emb_norm_min', jnp.float32(0.0)),
            'qk_emb_norm_std': result.get('qk_emb_norm_std', jnp.float32(0.0)),
            'v_emb_norm_mean': result.get('v_emb_norm_mean', jnp.float32(0.0)),
            'v_emb_norm_min': result.get('v_emb_norm_min', jnp.float32(0.0)),
            'v_emb_norm_std': result.get('v_emb_norm_std', jnp.float32(0.0)),
            'know_read_norm': result.get('know_read_norm', jnp.float32(0.0)),
            'know_write_norm': result.get('know_write_norm', jnp.float32(0.0)),
            # tau bias (scalar learned params).
            'tau_know_bias': tau_know_b[0],
            'tau_attn_bias_0': tau_attn_b[0],
            'tau_attn_bias_1': tau_attn_b[1],
            'tau_attn_bias_2': tau_attn_b[2],
            'tau_q_bias': tau_q_b[0],
            'tau_k_bias': tau_k_b[0],
            'tau_v_bias': tau_v_b[0],
            'scan_know_bias': scan_know_b[0],
            'scan_attn_bias_0': scan_attn_b[0],
            'scan_attn_bias_1': scan_attn_b[1],
            'scan_attn_bias_2': scan_attn_b[2],
            # Output norms (REGULAR subset).
            'know_out_norm': result.get('know_out_norm', jnp.float32(0.0)),
            'attn_qk_raw_norm': result.get('attn_qk_raw_norm', jnp.float32(0.0)),
            'attn_v_raw_norm': result.get('attn_v_raw_norm', jnp.float32(0.0)),
            'know_raw_out_norm': result.get('know_raw_out_norm', jnp.float32(0.0)),
            # z_mean_active (kept: cheap scalar).
            'know_z_mean_active': result.get('know_z_mean_active', jnp.float32(0.0)),
            'attn_qk_z_mean_active': result.get('attn_qk_z_mean_active', jnp.float32(0.0)),
            'attn_v_z_mean_active': result.get('attn_v_z_mean_active', jnp.float32(0.0)),
            # Per-layer diagnostics.
            'per_layer_attn_out_norm': result.get('per_layer_attn_out_norm', jnp.zeros(1)),
            'per_layer_know_out_norm': result.get('per_layer_know_out_norm', jnp.zeros(1)),
            # Dead-only penalty.
            'dead_penalty': dead_penalty,
            'attn_dead_penalty': result.get('attn_dead_penalty', jnp.float32(0.0)),
            'know_dead_penalty': result.get('know_dead_penalty', jnp.float32(0.0)),
            'attn_dead_count': result.get('attn_dead_count', jnp.float32(0.0)),
            'know_dead_count': result.get('know_dead_count', jnp.float32(0.0)),
            # v4.1 RPE exploration + diagnostics.
            'global_mean_ce': explore_stats['global_mean_ce'],
            'pos_frac': explore_stats['pos_frac'],
            'pos_mean': explore_stats['pos_mean'],
            'neg_mean': explore_stats['neg_mean'],
            'explore_loss_raw': explore_stats['explore_loss_raw'],
            'explore_attn_raw': explore_stats['explore_attn_raw'],
            'explore_know_raw': explore_stats['explore_know_raw'],
            'explore_loss_weighted': exploration_weight * explore_stats['explore_loss_raw'] * explore_stats['explore_active'],
            'explore_active': explore_stats['explore_active'],
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
            # v4.1 intensity / v4.1.5 gate-denominator diagnostics.
            'attn_int_max': result.get('attn_int_max', jnp.float32(0.0)),
            'know_int_max': result.get('know_int_max', jnp.float32(0.0)),
            'attn_intensity_sum_mean': result.get('attn_intensity_sum_mean', jnp.float32(0.0)),
            'know_intensity_sum_mean': result.get('know_intensity_sum_mean', jnp.float32(0.0)),
            'attn_gate_den_sum_mean': result.get('attn_gate_den_sum_mean', jnp.float32(0.0)),
            'know_gate_den_sum_mean': result.get('know_gate_den_sum_mean', jnp.float32(0.0)),
            'attn_den_cost_mean': result.get('attn_den_cost_mean', jnp.float32(0.0)),
            'know_den_cost_mean': result.get('know_den_cost_mean', jnp.float32(0.0)),
            'attn_act_cost_mean': result.get('attn_act_cost_mean', jnp.float32(0.0)),
            'know_act_cost_mean': result.get('know_act_cost_mean', jnp.float32(0.0)),
            'attn_current_cost_mean': result.get('attn_current_cost_mean', jnp.float32(0.0)),
            'know_current_cost_mean': result.get('know_current_cost_mean', jnp.float32(0.0)),
            'qk_read_sig_norm_mean': result.get('qk_read_sig_norm_mean', jnp.float32(0.0)),
            'qk_read_sig_norm_std': result.get('qk_read_sig_norm_std', jnp.float32(0.0)),
            'qk_write_sig_norm_mean': result.get('qk_write_sig_norm_mean', jnp.float32(0.0)),
            'qk_write_sig_norm_std': result.get('qk_write_sig_norm_std', jnp.float32(0.0)),
            'v_read_sig_norm_mean': result.get('v_read_sig_norm_mean', jnp.float32(0.0)),
            'v_read_sig_norm_std': result.get('v_read_sig_norm_std', jnp.float32(0.0)),
            'v_write_sig_norm_mean': result.get('v_write_sig_norm_mean', jnp.float32(0.0)),
            'v_write_sig_norm_std': result.get('v_write_sig_norm_std', jnp.float32(0.0)),
            'know_read_sig_norm_mean': result.get('know_read_sig_norm_mean', jnp.float32(0.0)),
            'know_read_sig_norm_std': result.get('know_read_sig_norm_std', jnp.float32(0.0)),
            'know_write_sig_norm_mean': result.get('know_write_sig_norm_mean', jnp.float32(0.0)),
            'know_write_sig_norm_std': result.get('know_write_sig_norm_std', jnp.float32(0.0)),
            # Emb drift (relative L2) since prev snapshot -see top of fn.
            'drift_qk_emb': drift_qk_emb,
            'drift_v_emb': drift_v_emb,
            'drift_know_emb': drift_know_emb,
        }

        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step(model, sharded_fns=None):
    """Create a jit-compiled evaluation step.

    Uses the SLIM forward (analysis=False) -eval only needs loss /
    correct / valid_count.
    """
    _pass_analysis_kw = _model_accepts_analysis(model)

    @jax.jit
    def eval_step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        eval_rng = jax.random.PRNGKey(0)
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        if _pass_analysis_kw:
            extra_kw['analysis'] = False
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


def create_analysis_step(model, sharded_fns=None):
    """Create a jit-compiled analysis step (FULL forward, observational).

    Runs the model with `analysis=True` and the ANALYSIS variant of
    sharded_fns. Returns a dict of distribution / boundary / debug
    stats that the 2-tier logger's ANALYSIS block consumes. Called
    once per val tick (val_interval), so the compile cost amortises.
    """
    _pass_analysis_kw = _model_accepts_analysis(model)

    @jax.jit
    def analysis_step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        eval_rng = jax.random.PRNGKey(0)
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        if _pass_analysis_kw:
            extra_kw['analysis'] = True
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
            **extra_kw,
        )
        return result

    return analysis_step


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


def get_param_shardings(params, mesh, is_baseline=False, tensor_parallel=False):
    """Create sharding specs for params: neuron_pool N-axis on 'model', rest replicated.
    For baseline models (is_baseline=True), 2D+ params are sharded on 'data' axis (FSDP-style).
    """
    replicated = NamedSharding(mesh, P())  # no sharding
    n_sharded = NamedSharding(mesh, P('model', None))  # N axis on model
    n_sharded_3d = NamedSharding(mesh, P('model', None, None))
    d_sharded = NamedSharding(mesh, P(None, 'model'))
    d_vector_sharded = NamedSharding(mesh, P('model'))
    din_sharded = NamedSharding(mesh, P('model', None))
    data_sharded = NamedSharding(mesh, P('data', None))  # FSDP: first axis on data

    def _get_sharding(path, value):
        path_str = '/'.join(str(p) for p in path)
        if tensor_parallel:
            if value.ndim == 1 and (
                'norm' in path_str or path_str.endswith('/bias')
            ) and value.shape[0] % mesh.shape['model'] == 0:
                return d_vector_sharded
            if 'token_emb' in path_str or 'pos_emb' in path_str:
                if value.ndim == 2:
                    return d_sharded
            if 'neuron_pool' in path_str:
                if '_read' in path_str or '_write' in path_str:
                    if '_sig_proj' in path_str:
                        return din_sharded
                    return d_sharded
                return replicated
            if 'router' in path_str and value.ndim == 2:
                if value.shape[0] % mesh.shape['model'] == 0:
                    return din_sharded
            if 'expand_O' in path_str and value.ndim == 2:
                if value.shape[0] % mesh.shape['model'] == 0:
                    return din_sharded
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

    data: [per_host_batch, ...] -this host's data portion
    sharding: NamedSharding
    global_shape: (global_batch, ...)
    """
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    per_host = data.shape[0]

    def data_callback(index):
        # index is a tuple of slices for each dimension.
        # The batch slice tells us which global rows this device needs.
        batch_slice = index[0]
        start = batch_slice.start or 0
        stop = batch_slice.stop or global_shape[0]
        local_start = start - host_id * per_host
        local_stop = stop - host_id * per_host
        if 0 <= local_start < per_host:
            return np.array(data[local_start:local_stop])
        # Previously returned silent zeros -that corrupts training with
        # a zero-batch whenever the mesh's host locality doesn't match
        # the data partition. Fail loud instead so the misconfiguration
        # is caught at setup rather than showing up as mysterious loss.
        raise RuntimeError(
            f"shard_to_mesh: device requests global index [{start}, {stop}) "
            f"but host {host_id} has local range [0, {per_host}) "
            f"(local_start={local_start}). Mesh layout likely doesn't match "
            f"host locality. Check create_mesh() device order.")

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


# ============================================================
# Helpers
# ============================================================

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

def evaluate(eval_step_fn, params, val_loader, n_devices, max_batches=200,
             verbose=True, data_sharding_spec=None, mesh=None):
    """Run evaluation and return avg loss and accuracy.

    All hosts must call this (pmap requires it), but only verbose=True host prints.
    Accumulates on device -one TPU-to-CPU sync at the end instead of three
    per batch -so eval stays fast on 1B-scale runs.
    """
    total_loss_jax = jnp.float32(0.0)
    total_correct_jax = jnp.int32(0)
    total_valid_jax = jnp.int32(0)

    eval_total = min(max_batches, len(val_loader))
    eval_start = time.time()
    batch_idx = -1

    for batch_idx, (input_ids, attention_mask) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        if data_sharding_spec is not None:
            gb = input_ids.shape[0] * jax.process_count()
            gs = (gb, input_ids.shape[1])
            input_ids = shard_to_mesh(input_ids, data_sharding_spec, gs)
            attention_mask = shard_to_mesh(attention_mask, data_sharding_spec, gs)

        if mesh is not None:
            with mesh:
                ce_loss, correct, valid_count = eval_step_fn(
                    params, input_ids, attention_mask)
        else:
            ce_loss, correct, valid_count = eval_step_fn(
                params, input_ids, attention_mask)

        total_loss_jax = total_loss_jax + ce_loss * valid_count.astype(jnp.float32)
        total_correct_jax = total_correct_jax + correct
        total_valid_jax = total_valid_jax + valid_count

    totals = jax.device_get({
        'loss': total_loss_jax,
        'correct': total_correct_jax,
        'valid': total_valid_jax,
    })
    total_loss = float(totals['loss'])
    total_correct = int(totals['correct'])
    total_valid = int(totals['valid'])

    eval_elapsed = time.time() - eval_start
    done = min(batch_idx + 1, eval_total) if batch_idx >= 0 else 0
    if verbose:
        print(f"  Eval: {done}/{eval_total} batches, {eval_elapsed:.1f}s", flush=True)
    avg_loss = total_loss / total_valid if total_valid > 0 else 0.0
    avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
    return avg_loss, avg_acc


# ============================================================
# Checkpoint save / load (with GCS support)
# ============================================================

def _serialize_checkpoint(params, opt_state, epoch, step, best_val_loss, model_config,
                          step_in_epoch=0, steps_per_epoch=0, training_config=None):
    """Serialize a checkpoint dict to bytes (no write).

    Split out so callers that write the same bytes to multiple paths
    (e.g. checkpoint_epochN.flax + best_model.flax in the same event)
    can reuse a single serialization pass.
    """
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
    return serialization.to_bytes(ckpt)


def _write_checkpoint_bytes(path, bytes_data):
    """Write pre-serialized checkpoint bytes to path."""
    with _open_file(path, 'wb') as f:
        f.write(bytes_data)
    print(f"  Checkpoint saved: {path} ({len(bytes_data) / 1e6:.1f} MB)")


def save_checkpoint(path, params, opt_state, epoch, step, best_val_loss, model_config,
                    step_in_epoch=0, steps_per_epoch=0, training_config=None):
    """Save checkpoint using flax serialization. Supports local and GCS paths."""
    bytes_data = _serialize_checkpoint(
        params, opt_state, epoch, step, best_val_loss, model_config,
        step_in_epoch, steps_per_epoch, training_config)
    _write_checkpoint_bytes(path, bytes_data)


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
    """Logger that writes to a local file and syncs to GCS on sync().

    GCS doesn't support true append -each open('a')/write/close overwrites.
    So we always append to a local file and upload the full file to GCS
    on every sync() call. Callers decide the sync cadence (training
    loop syncs once per FAST log boundary); the logger itself doesn't
    throttle. Uploading the whole file every FAST log is cheap in
    GCS API cost ($5 per 1M write ops) and in host-0 wall time
    (percent-of-a-percent over a multi-hour run), and the
    near-real-time visibility is worth it.
    """

    def __init__(self, gcs_path, local_path, resume=False):
        self.gcs_path = gcs_path
        self.local_path = local_path
        self._dirty = False
        if local_path:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if resume and gcs_path and local_path:
            # Seed the local file from existing GCS contents so the
            # subsequent open('a')/write path appends in-place; without
            # the seed the first sync() would overwrite GCS with only
            # this session's tail. If the GCS file doesn't exist yet we
            # silently continue (fresh-looking logger).
            try:
                with _open_file(gcs_path, 'rb') as f:
                    data = f.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            except FileNotFoundError:
                pass
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"  [warn] could not seed log from {gcs_path}: {e}", flush=True)

    def write(self, text):
        with open(self.local_path, 'a') as f:
            f.write(text)
        self._dirty = True

    def sync(self):
        """Upload local file to GCS if there are unflushed writes."""
        if not self._dirty or not self.gcs_path:
            return
        try:
            with open(self.local_path, 'rb') as f:
                data = f.read()
            with _open_file(self.gcs_path, 'wb') as f:
                f.write(data)
            self._dirty = False
        except Exception as e:
            if jax.process_index() == 0:
                print(f"  [warn] GCS sync failed: {e}", flush=True)


# Module-level loggers -set up in main()
_train_logger = None
_jsonl_logger = None


def _setup_loggers(training_log_file, jsonl_log_file, resume=False):
    """Create GCSLogger instances for training log and JSONL log.

    resume=True downloads existing GCS content to the local scratch file
    first so new lines append rather than overwrite.
    """
    global _train_logger, _jsonl_logger
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "dawn_logs"
    tmpdir.mkdir(parents=True, exist_ok=True)

    if _is_gcs(training_log_file):
        local_txt = str(tmpdir / Path(training_log_file).name)
        _train_logger = GCSLogger(training_log_file, local_txt, resume=resume)
    else:
        _train_logger = GCSLogger(None, training_log_file, resume=resume)

    if _is_gcs(jsonl_log_file):
        local_jsonl = str(tmpdir / Path(jsonl_log_file).name)
        _jsonl_logger = GCSLogger(jsonl_log_file, local_jsonl, resume=resume)
    else:
        _jsonl_logger = GCSLogger(None, jsonl_log_file, resume=resume)


def sync_logs():
    """Flush local logs to GCS. Call every FAST log for live visibility."""
    if _train_logger:
        _train_logger.sync()
    if _jsonl_logger:
        _jsonl_logger.sync()


def log_message(msg, log_file=None):
    """Print and write to training log file. Host 0 only."""
    if jax.process_index() != 0:
        return
    print(msg, flush=True)
    if _train_logger:
        try:
            _train_logger.write(msg + '\n')
        except Exception as e:
            print(f"  [warn] log_message write failed: {e}", flush=True)


def format_time(seconds):
    """Format seconds to H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def log_jsonl(record):
    """Append a JSON-lines record to the JSONL log file. Host 0 only."""
    if jax.process_index() != 0:
        return
    if not _jsonl_logger:
        return
    try:
        line = json.dumps(record, default=str)
        _jsonl_logger.write(line + '\n')
    except Exception as e:
        print(f"  [warn] log_jsonl write failed: {e}", flush=True)


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
# 2-tier periodic logging (REGULAR / ANALYSIS)
# ============================================================
#
# REGULAR  every log_interval steps                        (default 100)
# ANALYSIS every log_interval * log_analysis_multiplier steps
#
# v4.1: ANALYSIS is not emitted on every REGULAR tick. The distribution /
# boundary / saturation stats require a separate forward with the full-stats
# shard_map kernels (analysis_step), so the multiplier controls that cost.
#
# REGULAR carries the training-dynamics block (loss, activity, tau
# structure, emb norms, RPE, per-layer). ANALYSIS (`type='train_analysis'`)
# adds distribution-shape / boundary /
# saturation / debug diagnostics.
#
# _build_analysis_record accepts `base={}` on the new path -the
# `base`/`metrics` split is preserved for back-compat but the ANALYSIS
# record is now standalone.


def _fmt_act_count(frac, total):
    """Format 'XX.X%(N)' -active fraction with the implied count."""
    return f"{frac * 100:.1f}%({int(round(frac * total))})"


def _build_regular_record(metrics, win_avgs, ctx, global_step, epoch):
    """REGULAR tier: all training-dynamics fields needed for live monitoring.

    Equivalent to the former FAST + DEEP record merged; consumers of the
    old train_fast / train_deep JSONL types should switch to type='train'.
    """
    m = metrics
    is_v415 = ctx.get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2')
    rec = {
        'step': global_step,
        'epoch': epoch,
        # Loss components (window-averaged).
        'total_loss': win_avgs['loss'],
        'ce_loss': win_avgs['ce'],
        'aux_loss': win_avgs['aux'],
        'tau_reg': win_avgs['tau_reg'],
        'orth_loss': win_avgs['orth'],
        'div_loss': win_avgs['div'],
        'aux_weighted': ctx['lb_weight'] * win_avgs['aux'],
        'tau_reg_weighted': ctx['tau_reg_weight'] * win_avgs['tau_reg'],
        'orth_weighted': ctx['orth_weight'] * win_avgs['orth'],
        'div_weighted': ctx['div_weight'] * win_avgs['div'],
        # Dead-only penalty.
        'dead_penalty': float(m.get('dead_penalty', 0.0)),
        'attn_dead_penalty': float(m.get('attn_dead_penalty', 0.0)),
        'know_dead_penalty': float(m.get('know_dead_penalty', 0.0)),
        'dead_penalty_weighted': ctx['dead_penalty_weight'] * float(m.get('dead_penalty', 0.0)),
        # Explore loss (RPE).
        'explore_loss_raw': float(m.get('explore_loss_raw', 0.0)),
        'explore_loss_weighted': float(m.get('explore_loss_weighted', 0.0)),
        # Accuracy / training status.
        'accuracy': win_avgs['acc'],
        'grad_norm': float(m['grad_norm']),
        'grad_router_proj': float(m.get('grad_router_proj', 0.0)),
        'grad_router_tau': float(m.get('grad_router_tau', 0.0)),
        'grad_router_scan': float(m.get('grad_router_scan', 0.0)),
        'grad_pool_emb': float(m.get('grad_pool_emb', 0.0)),
        'grad_pool_read': float(m.get('grad_pool_read', 0.0)),
        'grad_pool_write': float(m.get('grad_pool_write', 0.0)),
        'lr': ctx['current_lr'],
        'steps_per_sec': ctx['steps_per_sec'],
        'elapsed': ctx['total_elapsed'],
        # Drift (reduced inside train_step).
        'drift_qk_emb': float(m.get('drift_qk_emb', 0.0)),
        'drift_v_emb': float(m.get('drift_v_emb', 0.0)),
        'drift_know_emb': float(m.get('drift_know_emb', 0.0)),
        # Core activity.
        'attn_qk_active': float(m.get('attn_qk_active', 0.0)),
        'attn_v_active': float(m.get('attn_v_active', 0.0)),
        'know_active': float(m.get('know_active', 0.0)),
        'attn_strong': float(m.get('attn_strong', 0.0)),
        'attn_qk_strong': float(m.get('attn_qk_strong', m.get('attn_strong', 0.0))),
        'attn_v_strong': float(m.get('attn_v_strong', m.get('attn_strong', 0.0))),
        'know_strong': float(m.get('know_strong', 0.0)),
        'attn_raw_gate_max': float(m.get('attn_raw_gate_max', 0.0)),
        'know_raw_gate_max': float(m.get('know_raw_gate_max', 0.0)),
        'attn_int_max': float(m.get('attn_int_max', 0.0)),
        'know_int_max': float(m.get('know_int_max', 0.0)),
        'attn_den_cost_mean': float(m.get('attn_den_cost_mean', 0.0)),
        'know_den_cost_mean': float(m.get('know_den_cost_mean', 0.0)),
        'attn_act_cost_mean': float(m.get('attn_act_cost_mean', 0.0)),
        'know_act_cost_mean': float(m.get('know_act_cost_mean', 0.0)),
        'attn_current_cost_mean': float(m.get('attn_current_cost_mean', 0.0)),
        'know_current_cost_mean': float(m.get('know_current_cost_mean', 0.0)),
        'qk_read_sig_norm_mean': float(m.get('qk_read_sig_norm_mean', 0.0)),
        'qk_read_sig_norm_std': float(m.get('qk_read_sig_norm_std', 0.0)),
        'qk_write_sig_norm_mean': float(m.get('qk_write_sig_norm_mean', 0.0)),
        'qk_write_sig_norm_std': float(m.get('qk_write_sig_norm_std', 0.0)),
        'v_read_sig_norm_mean': float(m.get('v_read_sig_norm_mean', 0.0)),
        'v_read_sig_norm_std': float(m.get('v_read_sig_norm_std', 0.0)),
        'v_write_sig_norm_mean': float(m.get('v_write_sig_norm_mean', 0.0)),
        'v_write_sig_norm_std': float(m.get('v_write_sig_norm_std', 0.0)),
        'know_read_sig_norm_mean': float(m.get('know_read_sig_norm_mean', 0.0)),
        'know_read_sig_norm_std': float(m.get('know_read_sig_norm_std', 0.0)),
        'know_write_sig_norm_mean': float(m.get('know_write_sig_norm_mean', 0.0)),
        'know_write_sig_norm_std': float(m.get('know_write_sig_norm_std', 0.0)),
        'attn_gate_sum': float(m.get('attn_gate_sum', 0.0)),
        'know_gate_sum': float(m.get('know_gate_sum', 0.0)),
        'attn_active_n_mean': float(m.get('attn_active_n_mean', 0.0)),
        'know_active_n_mean': float(m.get('know_active_n_mean', 0.0)),
        'attn_dead_count': float(m.get('attn_dead_count', 0.0)),
        'know_dead_count': float(m.get('know_dead_count', 0.0)),
        'attn_tau_mean': float(m.get('attn_tau_mean', 0.0)),
        'know_tau_mean': float(m.get('know_tau_mean', 0.0)),
        'attn_score_mean': float(m.get('attn_score_mean', 0.0)),
        'know_score_mean': float(m.get('know_score_mean', 0.0)),
        'attn_out_norm': float(m.get('attn_out_norm', 0.0)),
        'know_out_norm': float(m.get('know_out_norm', 0.0)),
        # tau structure (bias + offset distribution).
        'tau_know_bias': float(m.get('tau_know_bias', 0.0)),
        'tau_attn_bias_0': float(m.get('tau_attn_bias_0', 0.0)),
        'tau_attn_bias_1': float(m.get('tau_attn_bias_1', 0.0)),
        'tau_attn_bias_2': float(m.get('tau_attn_bias_2', 0.0)),
        'tau_q_bias': float(m.get('tau_q_bias', 0.0)),
        'tau_k_bias': float(m.get('tau_k_bias', 0.0)),
        'tau_v_bias': float(m.get('tau_v_bias', 0.0)),
        'scan_know_bias': float(m.get('scan_know_bias', 0.0)),
        'scan_attn_bias_0': float(m.get('scan_attn_bias_0', 0.0)),
        'scan_attn_bias_1': float(m.get('scan_attn_bias_1', 0.0)),
        'scan_attn_bias_2': float(m.get('scan_attn_bias_2', 0.0)),
        'attn_tau_abs_mean': float(m.get('attn_tau_abs_mean', 0.0)),
        'know_tau_abs_mean': float(m.get('know_tau_abs_mean', 0.0)),
        'attn_tau_off_min': float(m.get('attn_tau_off_min', 0.0)),
        'attn_tau_off_max': float(m.get('attn_tau_off_max', 0.0)),
        'attn_tau_off_p99': float(m.get('attn_tau_off_p99', 0.0)),
        'attn_tau_off_p01': float(m.get('attn_tau_off_p01', 0.0)),
        'attn_tau_off_neg_frac': float(m.get('attn_tau_off_neg_frac', 0.0)),
        'know_tau_off_min': float(m.get('know_tau_off_min', 0.0)),
        'know_tau_off_max': float(m.get('know_tau_off_max', 0.0)),
        'know_tau_off_p99': float(m.get('know_tau_off_p99', 0.0)),
        'know_tau_off_p01': float(m.get('know_tau_off_p01', 0.0)),
        'know_tau_off_neg_frac': float(m.get('know_tau_off_neg_frac', 0.0)),
        'attn_score_std': float(m.get('attn_score_std', 0.0)),
        'know_score_std': float(m.get('know_score_std', 0.0)),
        # Emb norm stats.
        'know_emb_norm': float(m.get('know_emb_norm', 0.0)),
        'know_emb_norm_min': float(m.get('know_emb_norm_min', 0.0)),
        'know_emb_norm_std': float(m.get('know_emb_norm_std', 0.0)),
        'qk_emb_norm_mean': float(m.get('qk_emb_norm_mean', 0.0)),
        'qk_emb_norm_min': float(m.get('qk_emb_norm_min', 0.0)),
        'qk_emb_norm_std': float(m.get('qk_emb_norm_std', 0.0)),
        'v_emb_norm_mean': float(m.get('v_emb_norm_mean', 0.0)),
        'v_emb_norm_min': float(m.get('v_emb_norm_min', 0.0)),
        'v_emb_norm_std': float(m.get('v_emb_norm_std', 0.0)),
        'know_read_norm': float(m.get('know_read_norm', 0.0)),
        'know_write_norm': float(m.get('know_write_norm', 0.0)),
        'attn_qk_raw_norm': float(m.get('attn_qk_raw_norm', 0.0)),
        'attn_v_raw_norm': float(m.get('attn_v_raw_norm', 0.0)),
        'know_raw_out_norm': float(m.get('know_raw_out_norm', 0.0)),
        'know_z_mean_active': float(m.get('know_z_mean_active', 0.0)),
        'attn_qk_z_mean_active': float(m.get('attn_qk_z_mean_active', 0.0)),
        'attn_v_z_mean_active': float(m.get('attn_v_z_mean_active', 0.0)),
        # RPE exploration diag.
        'global_mean_ce': float(m.get('global_mean_ce', 0.0)),
        'pos_frac': float(m.get('pos_frac', 0.0)),
        'pos_mean': float(m.get('pos_mean', 0.0)),
        'neg_mean': float(m.get('neg_mean', 0.0)),
        'explore_attn_raw': float(m.get('explore_attn_raw', 0.0)),
        'explore_know_raw': float(m.get('explore_know_raw', 0.0)),
        'explore_block_frac_a': float(m.get('explore_block_frac_a', 0.0)),
        'explore_block_frac_k': float(m.get('explore_block_frac_k', 0.0)),
        'sig_pos_max': float(m.get('sig_pos_max', 0.0)),
        'sig_neg_max': float(m.get('sig_neg_max', 0.0)),
        'timestamp': datetime.now().isoformat(),
    }
    if is_v415:
        rec.pop('attn_den_cost_mean', None)
        rec.pop('know_den_cost_mean', None)
        rec.update({
            'attn_gate_den_sum_mean': float(m.get(
                'attn_gate_den_sum_mean', m.get('attn_intensity_sum_mean', 0.0))),
            'know_gate_den_sum_mean': float(m.get(
                'know_gate_den_sum_mean', m.get('know_intensity_sum_mean', 0.0))),
        })
    else:
        rec.update({
            'attn_intensity_sum_mean': float(m.get('attn_intensity_sum_mean', 0.0)),
            'know_intensity_sum_mean': float(m.get('know_intensity_sum_mean', 0.0)),
        })
    # Per-layer norms (materialise lists).
    try:
        pl_a = jax.device_get(m['per_layer_attn_out_norm']).tolist()
        pl_k = jax.device_get(m['per_layer_know_out_norm']).tolist()
    except Exception:
        pl_a, pl_k = [], []
    rec['per_layer_attn_out_norm'] = pl_a
    rec['per_layer_know_out_norm'] = pl_k
    return rec


def _print_regular_block(rec, ctx):
    """Print REGULAR tier -~8 lines covering the live training dynamics."""
    log_message(
        f"[Step {rec['step']}/{ctx['total_micro_steps']} ({ctx['progress']:.1f}%)] "
        f"loss={rec['total_loss']:.4f} ce={rec['ce_loss']:.4f} aux={rec['aux_loss']:.4f} | "
        f"grad={rec['grad_norm']:.2f} | "
        f"acc={rec['accuracy']:.4f} lr={rec['lr']:.2e}"
    )
    log_message(
        f"  act: qk={_fmt_act_count(rec['attn_qk_active'], ctx['n_qk_cfg'])}"
        f" v={_fmt_act_count(rec['attn_v_active'], ctx['n_v_cfg'])}"
        f" k={_fmt_act_count(rec['know_active'], ctx['n_know_cfg'])}"
        f" | strong: qk={rec['attn_qk_strong']*100:.1f}%"
        f" v={rec['attn_v_strong']*100:.1f}%"
        f" k={rec['know_strong']*100:.1f}%"
    )
    log_message(
        f"  gate_max[a={rec['attn_raw_gate_max']:.1f}"
        f" k={rec['know_raw_gate_max']:.1f}]"
        f" int_max[a={rec['attn_int_max']:.1f} k={rec['know_int_max']:.1f}]"
        f" dead[a={int(rec['attn_dead_count'])} k={int(rec['know_dead_count'])}]"
        f" drift[qk={rec['drift_qk_emb']:.2e}"
        f" v={rec['drift_v_emb']:.2e}"
        f" k={rec['drift_know_emb']:.2e}]"
    )
    if ctx.get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2'):
        log_message(
            f"  gate_den_sum mean[a={rec['attn_gate_den_sum_mean']:.1f}"
            f" k={rec['know_gate_den_sum_mean']:.1f}]"
            f" | read_sig qk[m={rec['qk_read_sig_norm_mean']:.2f} s={rec['qk_read_sig_norm_std']:.2f}]"
            f" v[m={rec['v_read_sig_norm_mean']:.2f} s={rec['v_read_sig_norm_std']:.2f}]"
            f" k[m={rec['know_read_sig_norm_mean']:.2f} s={rec['know_read_sig_norm_std']:.2f}]"
        )
        log_message(
            f"  write_sig qk[m={rec['qk_write_sig_norm_mean']:.2f} s={rec['qk_write_sig_norm_std']:.2f}]"
            f" v[m={rec['v_write_sig_norm_mean']:.2f} s={rec['v_write_sig_norm_std']:.2f}]"
            f" k[m={rec['know_write_sig_norm_mean']:.2f} s={rec['know_write_sig_norm_std']:.2f}]"
        )
    log_message(
        f"  tau: know_b={rec['tau_know_bias']:+.2f}"
        f" attn_b=[{rec['tau_attn_bias_0']:+.2f} {rec['tau_attn_bias_1']:+.2f} {rec['tau_attn_bias_2']:+.2f}]"
        f" | tau_mean[a={rec['attn_tau_mean']:+.3f} k={rec['know_tau_mean']:+.3f}]"
        f" abs[a={rec['attn_tau_abs_mean']:.3f} k={rec['know_tau_abs_mean']:.3f}]"
    )
    if ctx.get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2'):
        log_message(
            f"  scan_bias: know={rec['scan_know_bias']:+.3f}"
            f" attn=[{rec['scan_attn_bias_0']:+.3f} {rec['scan_attn_bias_1']:+.3f} {rec['scan_attn_bias_2']:+.3f}]"
        )
    log_message(
        f"  tau_off k[min={rec['know_tau_off_min']:+.2f} p01={rec['know_tau_off_p01']:+.2f}"
        f" p99={rec['know_tau_off_p99']:+.2f} max={rec['know_tau_off_max']:+.2f}"
        f" neg={rec['know_tau_off_neg_frac']*100:.1f}%]"
        f" a[min={rec['attn_tau_off_min']:+.2f} p01={rec['attn_tau_off_p01']:+.2f}"
        f" p99={rec['attn_tau_off_p99']:+.2f} max={rec['attn_tau_off_max']:+.2f}"
        f" neg={rec['attn_tau_off_neg_frac']*100:.1f}%]"
    )
    log_message(
        f"  score_std[a={rec['attn_score_std']:.2f} k={rec['know_score_std']:.2f}]"
        f" | emb_n k[m={rec['know_emb_norm']:.2f} s={rec['know_emb_norm_std']:.2f}"
        f" min={rec['know_emb_norm_min']:.2f}]"
        f" qk[m={rec['qk_emb_norm_mean']:.2f} s={rec['qk_emb_norm_std']:.2f}"
        f" min={rec['qk_emb_norm_min']:.2f}]"
        f" v[m={rec['v_emb_norm_mean']:.2f} s={rec['v_emb_norm_std']:.2f}"
        f" min={rec['v_emb_norm_min']:.2f}]"
    )
    log_message(
        f"  rpe: mean_ce={rec['global_mean_ce']:.3f}"
        f" pos={rec['pos_frac']*100:.1f}%"
        f" pos_avg={rec['pos_mean']:.3f} neg_avg={rec['neg_mean']:.3f}"
        f" sig[+={rec['sig_pos_max']:.2f} -={rec['sig_neg_max']:.2f}]"
        f" expl[a={rec['explore_attn_raw']:+.3f} k={rec['explore_know_raw']:+.3f}]"
        f" w={rec['explore_loss_weighted']:+.4f}"
        f" block[a={rec['explore_block_frac_a']*100:.1f}%"
        f" k={rec['explore_block_frac_k']*100:.1f}%]"
    )
    _pl_a = rec.get('per_layer_attn_out_norm', []) or []
    _pl_k = rec.get('per_layer_know_out_norm', []) or []
    if _pl_a or _pl_k:
        log_message(
            f"  per_layer out: attn=[{' '.join(f'{v:.2f}' for v in _pl_a)}]"
            f" know=[{' '.join(f'{v:.2f}' for v in _pl_k)}]"
        )
    log_message(
        f"  time: {format_time(ctx['epoch_elapsed'])}<{format_time(ctx['eta'])},"
        f" {ctx['s_per_it']:.2f}s/it"
    )


def _build_analysis_record(base, metrics, ctx):
    """ANALYSIS tier: distribution shape, boundary, saturation, debug.

    In v4.1 this is fed by analysis_step (a separate full-stats forward
    run at val ticks), not by train_step. `base` is an empty dict on the
    new path -kept for back-compat. All ANALYSIS fields come from
    `metrics`, which is the dict returned by analysis_step. Needs
    `attn_out_norm` / `know_out_norm` for the raw_n print line, so
    those are pulled from analysis_result too.
    """
    m = metrics
    is_v415 = ctx.get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2')
    rec = dict(base)
    # tau per-route std (attn [3]) -materialise once.
    try:
        a_tau_s = np.asarray(jax.device_get(m.get('attn_tau_std', jnp.zeros(3))))
        if a_tau_s.size < 3:
            a_tau_s = np.zeros(3, dtype=np.float32)
    except Exception:
        a_tau_s = np.zeros(3, dtype=np.float32)
    rec.update({
        'attn_out_norm': float(m.get('attn_out_norm', 0.0)),
        'know_out_norm': float(m.get('know_out_norm', 0.0)),
        'attn_score_skew': float(m.get('attn_score_skew', 0.0)),
        'know_score_skew': float(m.get('know_score_skew', 0.0)),
        'attn_score_kurt': float(m.get('attn_score_kurt', 0.0)),
        'know_score_kurt': float(m.get('know_score_kurt', 0.0)),
        'attn_active_per_token_std': float(m.get('attn_active_per_token_std', 0.0)),
        'know_active_per_token_std': float(m.get('know_active_per_token_std', 0.0)),
        'attn_gate_entropy': float(m.get('attn_gate_entropy', 0.0)),
        'know_gate_entropy': float(m.get('know_gate_entropy', 0.0)),
        'attn_qk_phi_binary': float(m.get('attn_qk_phi_binary', 0.0)),
        'attn_v_phi_binary': float(m.get('attn_v_phi_binary', 0.0)),
        'know_phi_binary': float(m.get('know_phi_binary', 0.0)),
        'attn_z_lt_075': float(m.get('attn_z_lt_075', 0.0)),
        'know_z_lt_075': float(m.get('know_z_lt_075', 0.0)),
        'attn_z_lt_030': float(m.get('attn_z_lt_030', 0.0)),
        'know_z_lt_030': float(m.get('know_z_lt_030', 0.0)),
        'attn_int_cap_frac': float(m.get('attn_int_cap_frac', 0.0)),
        'know_int_cap_frac': float(m.get('know_int_cap_frac', 0.0)),
        'qk_emb_norm_max': float(m.get('qk_emb_norm_max', 0.0)),
        'v_emb_norm_max': float(m.get('v_emb_norm_max', 0.0)),
        'know_emb_norm_max': float(m.get('know_emb_norm_max', 0.0)),
        'attn_tau_std_q': float(a_tau_s[0]),
        'attn_tau_std_k': float(a_tau_s[1]),
        'attn_tau_std_v': float(a_tau_s[2]),
        'know_tau_std': float(m.get('know_tau_std', 0.0)),
        'attn_tau_kernel_norm': float(m.get('attn_tau_kernel_norm', 0.0)),
        'know_tau_kernel_norm': float(m.get('know_tau_kernel_norm', 0.0)),
        'attn_qk_raw_norm': float(m.get('attn_qk_raw_norm', 0.0)),
        'attn_v_raw_norm': float(m.get('attn_v_raw_norm', 0.0)),
        'know_raw_out_norm': float(m.get('know_raw_out_norm', 0.0)),
        'attn_z_sum': float(m.get('attn_z_sum', 0.0)),
        'know_z_sum': float(m.get('know_z_sum', 0.0)),
        'attn_den_cost': float(m.get('attn_den_cost', 0.0)),
        'know_den_cost': float(m.get('know_den_cost', 0.0)),
        'attn_activation_cost': float(m.get('attn_activation_cost', 0.0)),
        'know_activation_cost': float(m.get('know_activation_cost', 0.0)),
        'attn_current_cost': float(m.get('attn_current_cost', 0.0)),
        'know_current_cost': float(m.get('know_current_cost', 0.0)),
        'debug_residual_norm': float(m.get('debug_residual_norm', 0.0)),
        'debug_emb_norm': float(m.get('debug_emb_norm', 0.0)),
        'debug_o_proj_norm': float(m.get('debug_o_proj_norm', 0.0)),
        'debug_q_norm': float(m.get('debug_q_norm', 0.0)),
        'debug_k_norm': float(m.get('debug_k_norm', 0.0)),
        'debug_v_norm': float(m.get('debug_v_norm', 0.0)),
        'debug_logit_max': float(m.get('debug_logit_max', 0.0)),
        'debug_o_input_norm': float(m.get('debug_o_input_norm', 0.0)),
    })
    if is_v415:
        rec.pop('attn_den_cost', None)
        rec.pop('know_den_cost', None)
        rec.update({
            'attn_gate_den_sum': float(m.get(
                'attn_gate_den_sum', m.get('attn_den_cost', 0.0))),
            'know_gate_den_sum': float(m.get(
                'know_gate_den_sum', m.get('know_den_cost', 0.0))),
        })
    # HBM (host-0 local device 0 snapshot).
    try:
        mem = jax.local_devices()[0].memory_stats()
        if mem:
            used = mem.get('bytes_in_use', 0) / 1e9
            peak = mem.get('peak_bytes_in_use', 0) / 1e9
            limit = mem.get('bytes_limit', 0) / 1e9
            rec['hbm_used_gb'] = float(used)
            rec['hbm_peak_gb'] = float(peak)
            rec['hbm_limit_gb'] = float(limit)
    except Exception:
        pass
    return rec


def _print_analysis_block(rec, ctx):
    log_message(
        f"  dist k[skew={rec['know_score_skew']:+.2f} kurt={rec['know_score_kurt']:.2f}"
        f" apt_std={rec['know_active_per_token_std']:.1f} ent={rec['know_gate_entropy']:.2f}]"
        f" a[skew={rec['attn_score_skew']:+.2f} kurt={rec['attn_score_kurt']:.2f}"
        f" apt_std={rec['attn_active_per_token_std']:.1f} ent={rec['attn_gate_entropy']:.2f}]"
    )
    log_message(
        f"  boundary k[phi={rec['know_phi_binary']*100:.1f}%"
        f" z<075={rec['know_z_lt_075']*100:.1f}%"
        f" z<030={rec['know_z_lt_030']*100:.1f}%]"
        f" a_qk[phi={rec['attn_qk_phi_binary']*100:.1f}%]"
        f" a_v[phi={rec['attn_v_phi_binary']*100:.1f}%]"
        f" a[z<075={rec['attn_z_lt_075']*100:.1f}%"
        f" z<030={rec['attn_z_lt_030']*100:.1f}%]"
    )
    log_message(
        f"  saturation cap[a={rec['attn_int_cap_frac']*100:.1f}%"
        f" k={rec['know_int_cap_frac']*100:.1f}%]"
        f" | emb_max k={rec['know_emb_norm_max']:.2f}"
        f" qk={rec['qk_emb_norm_max']:.2f}"
        f" v={rec['v_emb_norm_max']:.2f}"
    )
    if ctx.get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2'):
        log_message(
            f"  gate_den_sum: a={rec['attn_gate_den_sum']:.1f}"
            f" k={rec['know_gate_den_sum']:.1f}"
        )
    log_message(
        f"  tau_struct k_std={rec['know_tau_std']:.2f}"
        f" a_std=[{rec['attn_tau_std_q']:.2f} {rec['attn_tau_std_k']:.2f} {rec['attn_tau_std_v']:.2f}]"
        f" k_kern={rec['know_tau_kernel_norm']:.1f}"
        f" a_kern={rec['attn_tau_kernel_norm']:.1f}"
    )
    log_message(
        f"  raw_n qk={rec['attn_qk_raw_norm']:.2f}"
        f" v={rec['attn_v_raw_norm']:.2f}"
        f" k={rec['know_raw_out_norm']:.2f}"
        f" | out_n a={rec['attn_out_norm']:.2f}"
        f" k={rec['know_out_norm']:.2f}"
    )
    log_message(
        f"  debug resid={rec['debug_residual_norm']:.2f}"
        f" emb={rec['debug_emb_norm']:.2f}"
        f" o_proj={rec['debug_o_proj_norm']:.2f}"
        f" q={rec['debug_q_norm']:.2f}"
        f" k={rec['debug_k_norm']:.2f}"
        f" v={rec['debug_v_norm']:.2f}"
        f" logit_max={rec['debug_logit_max']:.1f}"
        f" o_in={rec['debug_o_input_norm']:.2f}"
    )
    if 'hbm_used_gb' in rec:
        log_message(
            f"  HBM: {rec['hbm_used_gb']:.2f}G / {rec['hbm_limit_gb']:.2f}G"
            f" (peak={rec['hbm_peak_gb']:.2f}G,"
            f" free={rec['hbm_limit_gb'] - rec['hbm_used_gb']:.2f}G)"
        )


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
    # v4.1 free-norm: pool params (qk/v/know 횞 emb/read/write, 9 tensors)
    # get a lower WD than dense kernels. Bias / LayerNorm / *_scale
    # excluded from both groups.
    pool_weight_decay = tcfg.get('pool_weight_decay', 0.02)
    warmup_ratio = tcfg.get('warmup_ratio', 0.06)
    orth_weight = tcfg.get('orthogonality_weight', 0.01)
    div_weight = tcfg.get('diversity_weight', 0.1)
    lb_weight = tcfg.get('load_balance_weight', 2e-5)
    tau_reg_weight = tcfg.get('tau_reg_weight', 0.0)
    dead_penalty_weight = tcfg.get('dead_penalty_weight', 0.0)  # v4.0.6
    # v4.1 RPE exploration loss (0 weight => off; no-op for earlier versions).
    exploration_weight = tcfg.get('exploration_weight', 0.0)
    exploration_asymmetry = tcfg.get('exploration_asymmetry', 0.15)
    # v4.1+ bounded-explore: warmup + per-element bound-off cap.
    exploration_warmup_steps = tcfg.get('exploration_warmup_steps', 5000)
    exploration_lower_bound = tcfg.get('exploration_lower_bound', -0.5)
    exploration_upper_bound = tcfg.get('exploration_upper_bound', 2.0)
    exploration_bound_eps = tcfg.get('exploration_bound_eps', 1.0e-3)
    # 2-tier logging cadence.
    log_interval = int(tcfg.get('log_interval', 100))
    log_analysis_multiplier = int(tcfg.get('log_analysis_multiplier', 20))

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
        """List run_* subdirectories under base (local or GCS).

        FileNotFoundError on GCS is treated as "no prior runs yet" to
        match the local-path behavior (Path.exists() check below) -
        first training on a fresh checkpoint_dir shouldn't fail. Every
        other exception still propagates so credential / permission
        failures can't masquerade as "nothing to resume".
        """
        if _is_gcs(base):
            fs = _get_gcs_fs()
            if fs is None:
                raise ImportError(
                    f"Cannot list GCS path {base}: gcsfs not available.")
            bucket_path = base.replace('gs://', '').rstrip('/')
            try:
                entries = fs.ls(bucket_path)
            except FileNotFoundError:
                return []
            runs = sorted([
                'gs://' + e for e in entries
                if '/run_' in e
            ])
            return runs
        else:
            p = Path(base)
            if not p.exists():
                return []
            return sorted([
                str(d) for d in p.iterdir()
                if d.is_dir() and d.name.startswith('run_')
            ])

    def _broadcast_str_from_host0(s, max_len=512):
        """Broadcast a string (or None) from host 0 to all hosts.

        Must be called collectively on every host. Each host passes its
        local value; only host 0's value is adopted everywhere. Empty
        string and None both encode as all-zero padding and decode back
        to None. max_len caps the payload (GCS URLs usually fit well
        under 512 bytes).
        """
        if s is None:
            s = ''
        encoded = s.encode('utf-8')
        if len(encoded) > max_len:
            raise ValueError(
                f"Path too long for broadcast: {len(encoded)} > {max_len}")
        buf = np.zeros(max_len, dtype=np.uint8)
        if jax.process_index() == 0:
            buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

        if _HAVE_BROADCAST:
            broadcast_buf = np.asarray(_bcast_one_to_all(buf))
        else:
            gathered = np.asarray(process_allgather(buf))
            # Shape can be (n_hosts, max_len) or flat (n_hosts * max_len,)
            # depending on JAX version -pick host 0's slice either way.
            if gathered.ndim == 1:
                broadcast_buf = gathered[:max_len]
            else:
                broadcast_buf = gathered[0]
        result = bytes(broadcast_buf).rstrip(b'\x00').decode('utf-8')
        return result if result else None

    # Auto-resume: find latest run folder with checkpoints (unless --from-scratch)
    # --resume-from takes priority: resume from a specific run folder.
    #
    # Only host 0 lists GCS; the resulting (resume_path, checkpoint_dir)
    # is broadcast to all hosts. Independent per-host listing can diverge
    # under gcsfs caching, concurrent cleanup, or preemption-timing
    # races -a split resume mis-syncs global_step across the mesh and
    # later halts collectives inside train_step.
    if not cli_args.from_scratch:
        _host0_resume_path = None
        _host0_checkpoint_dir = None
        _host0_explicit_missing = False

        if jax.process_index() == 0:
            if cli_args.resume_from:
                folder = cli_args.resume_from.rstrip('/')
                candidates = _list_files(folder, "*.flax")
                if candidates:
                    _host0_resume_path = candidates[-1]
                    _host0_checkpoint_dir = folder
                    print(f"  Resume from specified folder: {_host0_checkpoint_dir}")
                    print(f"  Resuming from: {_host0_resume_path}")
                else:
                    _host0_explicit_missing = True
                    print(f"  No .flax checkpoint found in {folder}")
            else:
                run_folders = _list_run_folders(base_checkpoint_dir)
                for folder in reversed(run_folders):
                    candidates = _list_files(folder, "*.flax")
                    if candidates:
                        _host0_resume_path = candidates[-1]
                        _host0_checkpoint_dir = folder
                        print(f"  Auto-resume: found checkpoint in {_host0_checkpoint_dir}")
                        print(f"  Resuming from: {_host0_resume_path}")
                        break

        # Collective broadcast -all hosts must call.
        resume_path = _broadcast_str_from_host0(_host0_resume_path)
        checkpoint_dir = _broadcast_str_from_host0(_host0_checkpoint_dir)
        # Broadcast the explicit-missing signal as a single-byte string
        # so every host raises together.
        _missing_signal = _broadcast_str_from_host0(
            'MISSING' if _host0_explicit_missing else '')
        if _missing_signal == 'MISSING':
            raise FileNotFoundError(
                f"No .flax checkpoint found in {cli_args.resume_from}")

    # Create new run folder if not resuming
    if checkpoint_dir is None:
        import random as _random
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        ts = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        rand_suffix = _random.randint(1000, 9999)
        version = cfg['model'].get('model_version', 'spatial-r1-v4.1.5')
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
            pool_weight_decay = saved_training_config.get(
                'pool_weight_decay', pool_weight_decay)
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
            exploration_warmup_steps = saved_training_config.get(
                'exploration_warmup_steps', exploration_warmup_steps)
            exploration_lower_bound = saved_training_config.get(
                'exploration_lower_bound', exploration_lower_bound)
            exploration_upper_bound = saved_training_config.get(
                'exploration_upper_bound', exploration_upper_bound)
            exploration_bound_eps = saved_training_config.get(
                'exploration_bound_eps', exploration_bound_eps)
            log_interval = int(saved_training_config.get(
                'log_interval', log_interval))
            log_analysis_multiplier = int(saved_training_config.get(
                'log_analysis_multiplier', log_analysis_multiplier))
            if jax.process_index() == 0:
                print(f"  Training config restored from checkpoint (CLI overrides take precedence)")

    # Build training_config dict for saving in checkpoints
    training_config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'pool_weight_decay': pool_weight_decay,
        'warmup_ratio': warmup_ratio,
        'orthogonality_weight': orth_weight,
        'diversity_weight': div_weight,
        'load_balance_weight': lb_weight,
        'tau_reg_weight': tau_reg_weight,
        'dead_penalty_weight': dead_penalty_weight,
        'exploration_weight': exploration_weight,
        'exploration_asymmetry': exploration_asymmetry,
        'exploration_warmup_steps': exploration_warmup_steps,
        'exploration_lower_bound': exploration_lower_bound,
        'exploration_upper_bound': exploration_upper_bound,
        'exploration_bound_eps': exploration_bound_eps,
        'log_interval': log_interval,
        'log_analysis_multiplier': log_analysis_multiplier,
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
        n_devices=1,  # flat (per_host_batch, seq_len) -shard_to_mesh handles splitting
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

    # v4.1 per-group WD: pool tensors (qk/v/know 횞 emb/read/write) get
    # pool_weight_decay; dense kernels get weight_decay. Bias / LayerNorm /
    # learnable *_scale excluded from both groups.
    #
    # optax.adamw is chain(scale_by_adam, add_decayed_weights, scale_by_lr).
    # To apply two different WDs we decompose it: one scale_by_adam, then
    # two masked add_decayed_weights (base + pool -masks are disjoint so
    # each param is touched at most once), then a single scale_by_lr.

    _MODEL_VERSION = cfg['model'].get('model_version', 'spatial-r1-v4.1.5')

    _POOL_PARAM_NAMES = (
        'qk_emb', 'v_emb', 'know_emb',
        'q_read', 'k_read',
        'qk_read', 'v_read', 'know_read',
        'q_write', 'k_write',
        'qk_write', 'v_write', 'know_write',
    )
    _RW_PARAM_NAMES = (
        'q_read', 'k_read',
        'qk_read', 'v_read', 'know_read',
        'q_write', 'k_write',
        'qk_write', 'v_write', 'know_write',
    )

    def _path_str(path):
        return '/'.join(str(p) for p in path)

    def _is_pool_param(path_str):
        return any(name in path_str for name in _POOL_PARAM_NAMES)

    def _is_sig_proj_param(path_str):
        return 'read_sig_proj' in path_str or 'write_sig_proj' in path_str

    def _is_rw_param(path_str):
        return any(name in path_str for name in _RW_PARAM_NAMES)

    def _is_excluded(path_str):
        leaf = path_str.rsplit('/', 1)[-1]
        if leaf == 'bias':
            return True
        if 'scale' in path_str and 'norm' in path_str.lower():
            return True  # LayerNorm scale
        if path_str.endswith('_scale') or path_str.endswith('/qk_scale') \
           or path_str.endswith('/v_scale') or path_str.endswith('/know_scale'):
            return True  # learnable output_scale
        if _is_sig_proj_param(path_str):
            return True  # v4.1.5 fixed read/write signature projections
        if _MODEL_VERSION in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2') and _is_rw_param(path_str):
            return True  # v4.1.5 variants forward-normalize read/write directions
        return False

    def _freeze_mask_sig_proj(params):
        def _f(path, _):
            return _is_sig_proj_param(_path_str(path))
        return jax.tree.map_with_path(_f, params)

    def _wd_mask_base(params):
        def _f(path, _):
            ps = _path_str(path)
            if _is_excluded(ps):
                return False
            return not _is_pool_param(ps)
        return jax.tree.map_with_path(_f, params)

    def _wd_mask_pool(params):
        def _f(path, _):
            ps = _path_str(path)
            if _is_excluded(ps):
                return False
            return _is_pool_param(ps)
        return jax.tree.map_with_path(_f, params)

    base_optimizer = optax.chain(
        optax.masked(optax.set_to_zero(), mask=_freeze_mask_sig_proj),
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=0.95),
        optax.add_decayed_weights(weight_decay, mask=_wd_mask_base),
        optax.add_decayed_weights(pool_weight_decay, mask=_wd_mask_pool),
        optax.scale_by_learning_rate(schedule),
        optax.masked(optax.set_to_zero(), mask=_freeze_mask_sig_proj),
    )

    if is_host0:
        def _count_true(mask):
            n = [0]
            def _f(v):
                if v:
                    n[0] += 1
                return v
            jax.tree.map(_f, mask)
            return n[0]
        def _collect_pool_paths(mask):
            out = []
            def _f(path, v):
                if v:
                    out.append(_path_str(path))
                return v
            jax.tree.map_with_path(_f, mask)
            return out
        def _count_sig_proj(params):
            n = [0]
            def _f(path, _):
                if _is_sig_proj_param(_path_str(path)):
                    n[0] += 1
                return _
            jax.tree.map_with_path(_f, params)
            return n[0]
        _base_mask = _wd_mask_base(params)
        _pool_mask = _wd_mask_pool(params)
        print(f"  WD groups: base ({weight_decay}) = {_count_true(_base_mask)} tensors, "
              f"pool ({pool_weight_decay}) = {_count_true(_pool_mask)} tensors, "
              f"frozen_sig_proj_count = {_count_sig_proj(params)}")
        _pool_paths = _collect_pool_paths(_pool_mask)
        if _pool_paths:
            print(f"    pool params: {_pool_paths[:9]}")

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
        print(f"  Weight decay: {weight_decay} (pool: {pool_weight_decay})")
        print(f"  Orth weight: {orth_weight}")
        print(f"  Div weight: {div_weight}")
        print(f"  LB weight: {lb_weight}")
        print(f"  Tau reg weight: {tau_reg_weight}")
        print(f"  Dead penalty weight: {dead_penalty_weight}")
        print(f"  Exploration weight: {exploration_weight} "
              f"(asymmetry={exploration_asymmetry})")
        print(f"    warmup_steps={exploration_warmup_steps} "
              f"bounds=[{exploration_lower_bound}, {exploration_upper_bound}] "
              f"eps={exploration_bound_eps}")
        print(f"  Dropout: residual={cfg['model'].get('dropout', 0.0)} "
              f"router={cfg['model'].get('router_dropout', 0.0)}")
        # Active v4.1.5 gate closure constants.
        gate_msg = (
            f"  Gate (v4.1): sharpness={tcfg.get('sharpness', 500.0)} "
            f"act_thr={tcfg.get('activation_threshold', 0.5)} "
            f"act_cut={tcfg.get('activation_cutoff', 0.01)} "
            f"eps={tcfg.get('epsilon', 1.0e-4)} "
            f"max_int={tcfg.get('max_intensity', 10.0)}"
        )
        if cfg['model'].get('model_version') in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2'):
            gate_msg += (
                f" scan_scale={tcfg.get('scan_scale', 0.01)} "
                f"scan_std_floor={tcfg.get('scan_std_floor', 0.5)}"
            )
        print(gate_msg)

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
            # steps_per_epoch changed (batch size or data changed) -fallback
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

    # Fail-fast check: global_step must match across hosts after resume.
    # broadcast handles the common path but we still verify -if it ever
    # drifts, hang-debugging mid-training is painful; raise now instead.
    if n_hosts > 1:
        _gs_local = np.array([global_step], dtype=np.int64)
        _gs_all = np.asarray(process_allgather(_gs_local)).flatten()
        if not np.all(_gs_all == global_step):
            raise RuntimeError(
                f"global_step inconsistent across hosts after resume: "
                f"host {host_id} sees {global_step}, all hosts: {_gs_all.tolist()}. "
                f"Resume broadcast likely failed or checkpoint files diverged.")
        if is_host0:
            print(f"  [verified] global_step={global_step} consistent across {n_hosts} hosts")

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
    model_version = cfg['model'].get('model_version', 'spatial-r1-v4.1.5')
    is_baseline = model_version == 'baseline'
    is_spatial = model_version in ('spatial-r1-v3.9.4', 'spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2')

    mesh_model = cfg['training'].get('mesh_model', 1)
    mesh_data = cfg['training'].get('mesh_data', 0)  # 0 = auto
    total_devices = jax.device_count()
    if mesh_data == 0:
        mesh_data = total_devices // mesh_model

    mesh = create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    per_device_batch = batch_size // total_devices
    tensor_parallel = cfg['model'].get('parallelism', '').lower() in ('tensor', 'tp')
    if tensor_parallel:
        d_model = cfg['model']['d_model']
        n_heads = cfg['model']['n_heads']
        if d_model % mesh_model != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by mesh_model={mesh_model} "
                "for tensor-parallel residual sharding.")
        if n_heads % mesh_model != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by mesh_model={mesh_model} "
                "for tensor-parallel head sharding.")

    # Auto n_chunks: target ~2GB per chunk (bf16)
    def auto_n_chunks(N, target_gb=2.0):
        full_gb = per_device_batch * max_seq_len * N * 2 / 1e9  # bf16
        nc = max(1, int(np.ceil(full_gb / target_gb)))
        while N % nc != 0 and nc < N:
            nc += 1
        return min(nc, N)

    target_chunk_gb = cfg['training'].get('target_chunk_gb', 2.0)
    n_know = cfg['model'].get('n_know', 25200)
    n_qk = cfg['model'].get('n_qk', cfg['model'].get('n_q', 1580))
    n_v = cfg['model'].get('n_v', 2600)
    if tensor_parallel:
        # TP uses the model axis for D/head sharding; each model shard sees
        # the full neuron pool and chunks only to bound score/gate temps.
        nk_local = n_know
        nqk_local = n_qk
        nv_local = n_v
    else:
        for _name, _N in (('n_know', n_know), ('n_qk', n_qk), ('n_v', n_v)):
            if _N % mesh_model != 0:
                raise ValueError(
                    f"{_name}={_N} must be divisible by mesh_model={mesh_model} "
                    "for model-axis sharding.")
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

    def _chunk_size_from_count(name, n_local, n_chunks):
        n_chunks = int(n_chunks)
        if n_chunks < 1:
            raise ValueError(f"{name} chunks must be >= 1, got {n_chunks}")
        if n_chunks > n_local:
            raise ValueError(
                f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
        return max(1, int(np.ceil(n_local / n_chunks)))

    qk_max_chunk = _chunk_size_from_count('qk', nqk_local, n_chunks_qk)
    v_max_chunk = _chunk_size_from_count('v', nv_local, n_chunks_v)
    know_max_chunk = _chunk_size_from_count('know', nk_local, n_chunks_know)

    if is_host0:
        print(f"\n=== Mesh: ({mesh_data}, {mesh_model}) = "
              f"{total_devices} devices, per_device_batch={per_device_batch} ===")
        print(f"  Chunks: know={n_chunks_know} (cs={nk_local // max(n_chunks_know,1)}), "
              f"qk={n_chunks_qk}, v={n_chunks_v}")
        chunk_mem = per_device_batch * max_seq_len * know_max_chunk * 2 / 1e9
        print(f"  Est chunk mem (know): {chunk_mem:.2f}GB bf16")

    # Shard params: neuron_pool N-axis on 'model', rest replicated
    param_shardings = get_param_shardings(
        params, mesh, is_baseline=is_baseline,
        tensor_parallel=tensor_parallel)
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

    # Create shard_map functions if mesh_model > 1 or the model demands
    # the sharded path (v4.1 removed its non-sharded fallback).
    #
    # v4.1: two parallel kernel sets. `_sharded_fns` = slim (train path,
    # observational stats stripped). `_sharded_fns_analysis` = full
    # (all distribution/boundary/saturation stats; used only by
    # analysis_step at val time).
    _sharded_fns = None
    _sharded_fns_analysis = None
    _spec = MODEL_REGISTRY.get(model_version)
    _force_sharded = bool(_spec and _spec.force_sharded)
    if _spec is not None and (mesh_model > 1 or _force_sharded):
        if not _spec.supports_sharded:
            raise RuntimeError(
                f"model_version={model_version!r} is registered without "
                f"supports_sharded=True but mesh_model>1 or force_sharded=True.")
        _parallelism = cfg['model'].get('parallelism', '').lower()
        _module_path = _spec.module_path
        if _parallelism in ('tensor', 'tp') and _spec.tensor_module_path:
            _module_path = _spec.tensor_module_path
        _v3 = __import__(_module_path, fromlist=['make_sharded_srw'])
        make_sharded_srw = _v3.make_sharded_srw
        max_chunk = cfg['training'].get('max_chunk_size', None)
        if max_chunk is not None:
            qk_max_chunk = v_max_chunk = know_max_chunk = int(max_chunk)

        _srw_base_kwargs = {'mesh': mesh}
        if _spec.sharded_kwargs is not None:
            _srw_base_kwargs.update(_spec.sharded_kwargs(cfg))
        if _parallelism in ('tensor', 'tp'):
            _srw_base_kwargs.update({
                'stats_checkpoint': cfg['training'].get(
                    'srw_stats_checkpoint', True),
                'gate_checkpoint': cfg['training'].get(
                    'srw_gate_checkpoint', True),
            })
        # Slim (train) -kwargs don't set analysis, so defaults to False.
        _sharded_single_v = make_sharded_srw(
            max_chunk_size=v_max_chunk, **_srw_base_kwargs)
        _sharded_single_know = make_sharded_srw(
            max_chunk_size=know_max_chunk, **_srw_base_kwargs)
        if hasattr(_v3, 'make_sharded_srw_paired'):
            _sharded_paired_qk = _v3.make_sharded_srw_paired(
                max_chunk_size=qk_max_chunk, **_srw_base_kwargs)
            _sharded_fns = {
                'single': _sharded_single_v,
                'v_single': _sharded_single_v,
                'know_single': _sharded_single_know,
                'paired': _sharded_paired_qk,
                'qk_paired': _sharded_paired_qk,
            }
        else:
            _sharded_fns = _sharded_single_know
        # Analysis (observation only). Factory kwargs forward analysis=True
        # only to factories that accept it -v4.1 does, earlier versions
        # silently absorb it via **kwargs or raise; only probe when the
        # factory advertises the kwarg.
        import inspect as _inspect
        _supports_analysis = (
            'analysis' in _inspect.signature(make_sharded_srw).parameters
        )
        if _supports_analysis:
            _sharded_single_v_a = make_sharded_srw(
                analysis=True, max_chunk_size=v_max_chunk, **_srw_base_kwargs)
            _sharded_single_know_a = make_sharded_srw(
                analysis=True, max_chunk_size=know_max_chunk, **_srw_base_kwargs)
            if hasattr(_v3, 'make_sharded_srw_paired'):
                _sharded_paired_a = _v3.make_sharded_srw_paired(
                    analysis=True, max_chunk_size=qk_max_chunk,
                    **_srw_base_kwargs)
                _sharded_fns_analysis = {
                    'single': _sharded_single_v_a,
                    'v_single': _sharded_single_v_a,
                    'know_single': _sharded_single_know_a,
                    'paired': _sharded_paired_a,
                    'qk_paired': _sharded_paired_a,
                }
            else:
                _sharded_fns_analysis = _sharded_single_know_a
        if is_host0:
            print(f"  shard_map enabled (mesh_model={mesh_model}, QK fused"
                  f"; chunks qk/v/know={n_chunks_qk}/{n_chunks_v}/{n_chunks_know}"
                  f"; max_chunk qk/v/know={qk_max_chunk}/{v_max_chunk}/{know_max_chunk}"
                  f"; analysis kernels={'on' if _supports_analysis else 'off'})")

    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        tau_reg_weight, dead_penalty_weight,
        exploration_weight, exploration_asymmetry,
        rank, knowledge_rank, n_feature_qk, n_restore_qk,
        exploration_warmup_steps=exploration_warmup_steps,
        exploration_lower_bound=exploration_lower_bound,
        exploration_upper_bound=exploration_upper_bound,
        exploration_bound_eps=exploration_bound_eps,
        is_baseline=is_baseline, is_spatial=is_spatial,
        sharded_fns=_sharded_fns, mesh=mesh)
    eval_step_fn = create_eval_step(model, sharded_fns=_sharded_fns)
    # v4.1: analysis_step is only meaningful when the full analysis
    # kernels exist. Older model versions skip it -analysis logging
    # degrades to empty then.
    if _sharded_fns_analysis is not None:
        analysis_step_fn = create_analysis_step(
            model, sharded_fns=_sharded_fns_analysis)
    else:
        analysis_step_fn = None

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

        # Initial emb-drift snapshot: pytree of sharded refs matching
        # params['neuron_pool'][*_emb]. Identity here -drift=0 on first step.
        def _drift_snap(p):
            pool = p['neuron_pool']
            if 'qk_emb' in pool:
                return {
                    'qk_emb': pool['qk_emb'],
                    'v_emb': pool['v_emb'],
                    'know_emb': pool['know_emb'],
                }
            return {
                'qk_emb': pool['q_read'],
                'v_emb': pool['v_read'],
                'know_emb': pool['know_read'],
            }

        _dummy_emb_snap = _drift_snap(params)

        # First call: JIT compilation (slow)
        jit_start = time.time()
        with mesh:
            _dp, _do, dummy_metrics = train_step_fn(
                params, opt_state, dummy_ids, dummy_mask, dummy_step_rng,
                _dummy_emb_snap, jnp.asarray(0, jnp.int32))
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
        with mesh:
            _dp2, _do2, dummy_metrics2 = train_step_fn(
                params, opt_state, dummy_ids, dummy_mask, dummy_step_rng2,
                _dummy_emb_snap, jnp.asarray(0, jnp.int32))
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
        # NOTE: runs on ALL hosts -shard_map/psum require collective participation.
        # Only print statements are guarded by is_host0.
        try:
            _is_sharded = _sharded_fns is not None
            _uses_scan_bias = model_version in ('spatial-r1-v4.1.5', 'spatial-r1-v4.1.5.2')
            if is_host0:
                print(f"\n  === Step-time breakdown (1 layer, "
                      f"{'sharded' if _is_sharded else 'single-device'}) ===",
                      flush=True)

            _v3 = __import__(
                (MODEL_REGISTRY[model_version].tensor_module_path
                 if cfg['model'].get('parallelism', '').lower() in ('tensor', 'tp')
                 and MODEL_REGISTRY[model_version].tensor_module_path
                 else MODEL_REGISTRY[model_version].module_path),
                fromlist=['_layer_norm', '_attn_forward', '_know_forward', '_srw_chunked'])
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
                if _uses_scan_bias:
                    scan_bias_all = (x @ router_p['scan_bias_attn']['kernel']
                                     + router_p['scan_bias_attn']['bias'])
                else:
                    scan_bias_all = jnp.zeros_like(tau_all)
                return h_Q, h_K, h_V, tau_all, scan_bias_all

            # 3) QK fused shard_map (paired)
            @jax.jit
            def prof_qk_fused(x, h_Q, h_K, qk_norm, tau_all, scan_bias_all, qk_read, qk_write):
                fused_paired = (_sharded_fns.get('qk_paired', _sharded_fns['paired'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[1])
                h_QK = jnp.stack([h_Q, h_K], axis=2)
                tau_QK = jnp.stack(
                    [tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
                scan_bias_QK = jnp.stack(
                    [scan_bias_all[:, :, 0:1], scan_bias_all[:, :, 1:2]], axis=2)
                if _uses_scan_bias:
                    results = fused_paired(
                        x, h_QK, qk_norm, tau_QK, scan_bias_QK, qk_read, qk_write)
                else:
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
            def prof_v_sharded(x, h_V, v_norm, tau_v, scan_bias_v, v_read, v_write):
                fused_single = (_sharded_fns.get('v_single', _sharded_fns['single'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[0])
                if _uses_scan_bias:
                    return fused_single(
                        x, h_V, v_norm, tau_v, scan_bias_v, v_read, v_write)
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
                if _uses_scan_bias:
                    scan_bias = (x @ router_p['scan_bias_know']['kernel']
                                 + router_p['scan_bias_know']['bias'])
                else:
                    scan_bias = jnp.zeros_like(tau)
                return h, tau, scan_bias

            # 7) Know shard_map (single)
            @jax.jit
            def prof_know_sharded(x, h, know_norm, tau, scan_bias, know_read, know_write):
                fused_single = (_sharded_fns.get('know_single', _sharded_fns['single'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[0])
                if _uses_scan_bias:
                    return fused_single(
                        x, h, know_norm, tau, scan_bias, know_read, know_write)
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

            h_Q, h_K, h_V, tau_all, scan_bias_all = prof_attn_router(normed, router_p)
            jax.block_until_ready(tau_all)

            if _is_sharded:
                Q, K, *_ = prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all, scan_bias_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, *_ = prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write'])
            else:
                Q, K = prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    pool_p['qk_read'], pool_p['qk_write'])
                V, *_ = prof_v_chunked(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    pool_p['v_read'], pool_p['v_write'])
            jax.block_until_ready((Q, K, V))

            h_know, tau_know, scan_bias_know = prof_know_router(normed, router_p)
            jax.block_until_ready(tau_know)
            if _is_sharded:
                _kout = prof_know_sharded(
                    normed, h_know, know_norm, tau_know, scan_bias_know,
                    pool_p['know_read'], pool_p['know_write'])[0]
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
                    normed, h_Q, h_K, qk_norm, tau_all, scan_bias_all,
                    pool_p['qk_read'], pool_p['qk_write']))
                items.append(("A QK fused shard", ms, dg, pk))
                ms, dg, pk = _t(lambda: prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
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
                    normed, h_know, know_norm, tau_know, scan_bias_know,
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
            print_xla_oom_diagnostics()
        raise

    # ----------------------------------------------------------
    # Training log file (host 0 only)
    # ----------------------------------------------------------
    if is_host0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # On resume, reuse the existing log filenames from the run folder
        # so the session appends to the prior log instead of fragmenting
        # into training_log_<ts1>.txt + training_log_<ts2>.txt + ...
        _existing_logs = sorted(_list_files(log_dir, "training_log_*.txt"))
        _existing_jsonls = sorted(_list_files(log_dir, "metrics_*.jsonl"))
        _is_log_resume = (resume_path is not None) and bool(_existing_logs)
        if _is_log_resume:
            training_log_file = _existing_logs[-1]
            jsonl_log_file = (_existing_jsonls[-1] if _existing_jsonls
                              else _join(log_dir, f'metrics_{timestamp}.jsonl'))
        else:
            training_log_file = _join(log_dir, f'training_log_{timestamp}.txt')
            jsonl_log_file = _join(log_dir, f'metrics_{timestamp}.jsonl')

        # Set up loggers (local append + periodic GCS sync)
        _setup_loggers(training_log_file, jsonl_log_file, resume=_is_log_resume)

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
        """Flag-only SIGTERM handler (spot preemption).

        Saving from a signal handler is unsafe on multi-host: calling a
        collective here (_gather_for_save / process_allgather) requires
        every host to enter at the same point, but SIGTERM fires
        asynchronously per host. We just flag here; the main loop
        cooperatively saves after the inner-loop break.
        """
        if preemption_requested[0]:
            return
        preemption_requested[0] = True
        print(f"\n!!! SIGTERM received (host {host_id}) at step={global_step} -- flagging preemption !!!", flush=True)

    def _gather_for_save(x):
        """Gather sharded params to host-local full arrays for checkpoint save.

        Uses process_allgather with tiled=True so the output reconstructs
        the global shape: sharded axes get concatenated across processes
        and replicated arrays pass through unchanged.

        Must be called from ALL hosts simultaneously (collective).
        """
        return jax.device_get(process_allgather(x, tiled=True))

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

    # Logging cadence. REGULAR every log_interval steps. ANALYSIS every
    # log_interval * log_analysis_multiplier steps.
    LOG_REGULAR = log_interval
    LOG_ANALYSIS = max(1, log_interval * log_analysis_multiplier)
    if is_host0:
        print(f"  Log cadence: regular={LOG_REGULAR}"
              f" analysis={LOG_ANALYSIS}"
              f" val={val_interval}",
              flush=True)

    # Emb drift snapshot (sense vectors). Held on every host, refreshed at
    # each log event. Fed into train_step so the drift collective runs inside
    # jit on all hosts; the actual ||쨌|| reductions live there.
    _prev_emb_snap = _drift_snap(params)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        # Epoch accumulators on device -one device_get per epoch at the
        # end, rather than per-step float()/int() sync.
        _epoch_loss_jax = jnp.float32(0.0)
        # int64: valid_count sums to ~n_steps * tokens_per_step, which
        # exceeds int32 range (2.15e9) on any multi-billion-token epoch.
        _epoch_correct_jax = jnp.int64(0)
        _epoch_valid_jax = jnp.int64(0)
        epoch_steps = 0

        # Window accumulators on device -one device_get per log boundary.
        _win_loss_jax = jnp.float32(0.0)
        _win_ce_jax = jnp.float32(0.0)
        _win_aux_jax = jnp.float32(0.0)
        _win_tau_reg_jax = jnp.float32(0.0)
        _win_orth_jax = jnp.float32(0.0)
        _win_div_jax = jnp.float32(0.0)
        _win_correct_jax = jnp.int32(0)
        _win_valid_jax = jnp.int32(0)
        win_count = 0
        win_start_time = time.time()

        for local_step, (input_ids, attention_mask) in enumerate(train_loader):

            # Cross-host SIGTERM sync every 10 steps. Handles the case where
            # spot preemption fires on only some hosts first -without this,
            # a flagged host would break while unflagged hosts continue into
            # the next train_step collective and hang. Cost: one bool
            # all-gather per 10 steps (bytes).
            if local_step % 10 == 0:
                _preempt_any = bool(np.any(process_allgather(
                    np.array([preemption_requested[0]], dtype=np.bool_))))
                if _preempt_any and not preemption_requested[0]:
                    preemption_requested[0] = True
                    if is_host0:
                        print(
                            f"Preemption detected on another host"
                            f" (step={global_step}) -- cooperative break.",
                            flush=True,
                        )

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

            with mesh:
                params, opt_state, metrics = train_step_fn(
                    params, opt_state,
                    input_ids, attention_mask, step_rng, _prev_emb_snap,
                    jnp.asarray(global_step, jnp.int32))

            # Scalar helper kept for log-block use (m_grad etc.).
            def _m(v):
                return float(v)

            # Device-side accumulation -no per-step TPU-to-CPU sync on the
            # regression/metric scalars. Window + epoch values are
            # materialized only at log boundary and end of epoch.
            # Token-weighted accumulation: every window/epoch loss is summed
            # as (loss * valid_count) so the final avg divides by total
            # valid tokens, matching evaluate()'s token-level mean. Makes
            # train/val loss directly comparable.
            _valid_f = metrics['valid_count'].astype(jnp.float32)
            _win_loss_jax = _win_loss_jax + metrics['total_loss'] * _valid_f
            _win_ce_jax = _win_ce_jax + metrics['ce_loss'] * _valid_f
            _win_aux_jax = _win_aux_jax + metrics['aux_loss'] * _valid_f
            _win_tau_reg_jax = _win_tau_reg_jax + metrics.get('tau_reg', jnp.float32(0.0)) * _valid_f
            _win_orth_jax = _win_orth_jax + metrics['orth_loss'] * _valid_f
            _win_div_jax = _win_div_jax + metrics['div_loss'] * _valid_f
            _win_correct_jax = _win_correct_jax + metrics['correct']
            _win_valid_jax = _win_valid_jax + metrics['valid_count']

            _epoch_loss_jax = _epoch_loss_jax + metrics['ce_loss'] * _valid_f
            _epoch_correct_jax = _epoch_correct_jax + metrics['correct']
            _epoch_valid_jax = _epoch_valid_jax + metrics['valid_count']

            win_count += 1
            epoch_steps += 1

            # Per-step NaN check on total_loss only. A single scalar sync
            # catches loss explosions immediately; the full 6-key check runs
            # at log boundary on already-materialized window averages.
            _m_total_for_nan = float(metrics['total_loss'])
            if not np.isfinite(_m_total_for_nan):
                raise ValueError(
                    f"NaN/INF total_loss at epoch {epoch}, step {global_step + 1}")

            global_step += 1
            epoch_step_counter += 1

            # ---- REGULAR periodic logging ----
            # ANALYSIS is driven from the val path (below), not from here -
            # the ANALYSIS stats now require a separate forward with the
            # full-stats kernels and only run on val ticks.
            _is_early_debug = global_step in (1, 5, 10, 20, 50)
            is_regular = (global_step % LOG_REGULAR == 0) or _is_early_debug or debug_mode

            if is_regular:
                # Refresh emb-drift snapshot on every host (ref reassignment
                # only -no collective). Must run outside is_host0 so the
                # next jit'd train_step sees a consistent snap pytree.
                _prev_emb_snap = _drift_snap(params)
                # One TPU-to-CPU sync for the whole window.
                _win_vals = jax.device_get({
                    'loss': _win_loss_jax, 'ce': _win_ce_jax,
                    'aux': _win_aux_jax, 'tau_reg': _win_tau_reg_jax,
                    'orth': _win_orth_jax, 'div': _win_div_jax,
                    'correct': _win_correct_jax, 'valid': _win_valid_jax,
                })
                _win_correct_py = int(_win_vals['correct'])
                _win_valid_py = int(_win_vals['valid'])
                _vdiv = _win_valid_py if _win_valid_py > 0 else 1
                win_avgs = {
                    'loss':    float(_win_vals['loss'])    / _vdiv,
                    'ce':      float(_win_vals['ce'])      / _vdiv,
                    'aux':     float(_win_vals['aux'])     / _vdiv,
                    'tau_reg': float(_win_vals['tau_reg']) / _vdiv,
                    'orth':    float(_win_vals['orth'])    / _vdiv,
                    'div':     float(_win_vals['div'])     / _vdiv,
                    'acc':     _win_correct_py             / _vdiv,
                }
                # Full NaN/INF check on the materialized window averages.
                if check_nan_inf({
                    'total_loss': win_avgs['loss'], 'ce_loss': win_avgs['ce'],
                    'aux_loss': win_avgs['aux'], 'tau_reg': win_avgs['tau_reg'],
                    'orth_loss': win_avgs['orth'], 'div_loss': win_avgs['div'],
                }, global_step, epoch):
                    raise ValueError(
                        f"NaN/INF window averages at epoch {epoch}, step {global_step}")

                if is_host0:
                    _elapsed = time.time() - win_start_time
                    _steps_per_sec = (win_count / _elapsed) if _elapsed > 0 else 0.0
                    _opt_step = global_step // grad_accum_steps
                    _current_lr = float(schedule(_opt_step))
                    _total_elapsed = time.time() - train_start_time
                    _epoch_elapsed = time.time() - epoch_start
                    _progress = (global_step / total_micro_steps * 100
                                 if total_micro_steps > 0 else 0.0)
                    _s_per_it = _epoch_elapsed / epoch_steps if epoch_steps > 0 else 0.0
                    # ETA based on absolute epoch position so resume mid-epoch
                    # doesn't over-estimate (epoch_steps counts only this
                    # run's steps; epoch_step_counter starts from
                    # start_step_in_epoch).
                    _remaining = max(steps_per_epoch - epoch_step_counter, 0)
                    _eta = _s_per_it * _remaining
                    ctx = {
                        'lb_weight': lb_weight,
                        'tau_reg_weight': tau_reg_weight,
                        'orth_weight': orth_weight,
                        'div_weight': div_weight,
                        'dead_penalty_weight': dead_penalty_weight,
                        'n_qk_cfg': cfg['model'].get(
                            'n_qk', cfg['model'].get('n_q', 0)),
                        'n_v_cfg': cfg['model'].get('n_v', 0),
                        'n_know_cfg': cfg['model'].get('n_know', 0),
                        'current_lr': _current_lr,
                        'steps_per_sec': _steps_per_sec,
                        'total_elapsed': _total_elapsed,
                        'epoch_elapsed': _epoch_elapsed,
                        'eta': _eta,
                        's_per_it': _s_per_it,
                        'total_micro_steps': total_micro_steps,
                        'progress': _progress,
                        'model_version': model_version,
                    }
                    rec = _build_regular_record(metrics, win_avgs, ctx, global_step, epoch)
                    _print_regular_block(rec, ctx)
                    log_jsonl({'type': 'train', **rec})
                    sync_logs()

                # Reset window accumulators (all hosts)
                _win_loss_jax = jnp.float32(0.0)
                _win_ce_jax = jnp.float32(0.0)
                _win_aux_jax = jnp.float32(0.0)
                _win_tau_reg_jax = jnp.float32(0.0)
                _win_orth_jax = jnp.float32(0.0)
                _win_div_jax = jnp.float32(0.0)
                _win_correct_jax = jnp.int32(0)
                _win_valid_jax = jnp.int32(0)
                win_count = 0
                win_start_time = time.time()

            # ---- Mid-epoch validation (all hosts run eval, host 0 saves/logs) ----
            _do_val = (global_step % val_interval == 0 and global_step > 0)
            _do_analysis = (global_step % LOG_ANALYSIS == 0 and global_step > 0)
            _do_ckpt = (global_step % ckpt_interval == 0 and global_step > 0)
            _new_best = False

            if _do_val:
                if is_host0:
                    log_message(f"\n  Mid-epoch validation at step {global_step}...")
                val_loader.reset()
                val_loss, val_acc = evaluate(
                    eval_step_fn, params, val_loader, n_local_devices,
                    verbose=is_host0, data_sharding_spec=data_sharding,
                    mesh=mesh)
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
                    _new_best = True


            # ---- ANALYSIS: run full-stats forward on one val batch ----
            # Single analysis forward at the configured analysis cadence. Compiles
            # once on first call (extra HBM + time logged). Result dict
            # is released after the JSONL write so HBM snaps back.
            if _do_analysis and analysis_step_fn is not None:
                val_loader.reset()
                _analysis_batch = None
                for _ab_ids, _ab_mask in val_loader:
                    _analysis_batch = (_ab_ids, _ab_mask)
                    break
                if _analysis_batch is not None:
                    _a_ids, _a_mask = _analysis_batch
                    _a_gb = _a_ids.shape[0] * jax.process_count()
                    _a_gs = (_a_gb, _a_ids.shape[1])
                    _a_ids = shard_to_mesh(_a_ids, data_sharding, _a_gs)
                    _a_mask = shard_to_mesh(_a_mask, data_sharding, _a_gs)
                    try:
                        _a_compile_start = time.time()
                        with mesh:
                            analysis_result = analysis_step_fn(
                                params, _a_ids, _a_mask)
                        # Force the computation so HBM usage of the
                        # analysis kernels registers now, not on the
                        # next Python line.
                        jax.block_until_ready(
                            analysis_result.get('aux_loss',
                                                jnp.float32(0.0)))
                        _a_elapsed = time.time() - _a_compile_start
                        if is_host0:
                            _ctx_a = {
                                'n_qk_cfg': cfg['model'].get(
                                    'n_qk', cfg['model'].get('n_q', 0)),
                                'n_v_cfg': cfg['model'].get('n_v', 0),
                                'n_know_cfg': cfg['model'].get('n_know', 0),
                                'model_version': model_version,
                            }
                            a_rec = _build_analysis_record(
                                {}, analysis_result, _ctx_a)
                            a_rec['step'] = global_step
                            a_rec['epoch'] = epoch
                            a_rec['analysis_step_sec'] = float(_a_elapsed)
                            _print_analysis_block(a_rec, _ctx_a)
                            log_jsonl({'type': 'train_analysis', **a_rec})
                            sync_logs()
                    finally:
                        # Explicit release -jit-returned dict holds
                        # TPU buffers that outlive the val block
                        # otherwise.
                        try:
                            del analysis_result
                        except NameError:
                            pass
                        del _a_ids, _a_mask, _analysis_batch


            # ---- Unified save path ----
            # best_model + mid-epoch checkpoint share a single gather +
            # serialize when both fire on the same step (val_interval ==
            # ckpt_interval is the common case). Previously that meant two
            # independent _gather_for_save collectives and two full
            # re-serializations of the same params -expensive at 1B.
            if _new_best or _do_ckpt:
                params_single = _gather_for_save(params)
                opt_state_single = _gather_for_save(opt_state)
                if is_host0:
                    bytes_data = _serialize_checkpoint(
                        params_single, opt_state_single,
                        epoch, global_step, best_val_loss,
                        cfg['model'],
                        step_in_epoch=epoch_step_counter,
                        steps_per_epoch=steps_per_epoch,
                        training_config=training_config)
                    if _new_best:
                        _write_checkpoint_bytes(
                            _ckpt_path("best_model.flax"), bytes_data)
                        log_message(f"  New best model saved! val_loss={best_val_loss:.4f}")
                    if _do_ckpt:
                        _write_checkpoint_bytes(
                            _ckpt_path(f"checkpoint_step{global_step}.flax"), bytes_data)
                        # GCS list+delete only from host 0 -racing cleanups across
                        # hosts can drop the checkpoint that was just written.
                        cleanup_old_checkpoints(checkpoint_dir, keep_last=3)
                del params_single, opt_state_single

        if preemption_requested[0]:
            # Cooperative emergency save. All hosts participate in
            # _gather_for_save (collective); only host 0 writes the file.
            # Previously this ran from the SIGTERM signal handler, which
            # was unsafe because hosts enter the handler asynchronously.
            try:
                _emerg_params = _gather_for_save(params)
                _emerg_opt = _gather_for_save(opt_state)
                if is_host0:
                    epath = _ckpt_path(f"emergency_step{global_step}.flax")
                    save_checkpoint(
                        epath, _emerg_params, _emerg_opt,
                        epoch, global_step, best_val_loss,
                        cfg['model'],
                        step_in_epoch=epoch_step_counter,
                        steps_per_epoch=steps_per_epoch,
                        training_config=training_config,
                    )
                    print(f"!!! Emergency checkpoint saved: {epath} !!!", flush=True)
                del _emerg_params, _emerg_opt
            except Exception as e:
                if is_host0:
                    print(f"!!! Emergency save FAILED: {e} !!!", flush=True)
            break

        # ---- End of epoch ----
        epoch_elapsed = time.time() - epoch_start
        # Single TPU-to-CPU sync for the whole epoch totals.
        _ep = jax.device_get({
            'loss': _epoch_loss_jax,
            'correct': _epoch_correct_jax,
            'valid': _epoch_valid_jax,
        })
        epoch_loss = float(_ep['loss'])
        epoch_correct = int(_ep['correct'])
        epoch_valid = int(_ep['valid'])
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
            verbose=is_host0, data_sharding_spec=data_sharding,
            mesh=mesh)

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

        # Save epoch checkpoint (device_get on ALL hosts). best_model reuses
        # the same serialized bytes -no double serialize at 1B scale.
        params_single = _gather_for_save(params)
        opt_state_single = _gather_for_save(opt_state)

        if is_host0:
            bytes_data = _serialize_checkpoint(
                params_single, opt_state_single,
                epoch + 1, global_step, best_val_loss,
                cfg['model'],
                step_in_epoch=0,
                steps_per_epoch=steps_per_epoch,
                training_config=training_config)
            _write_checkpoint_bytes(
                _ckpt_path(f"checkpoint_epoch{epoch}.flax"), bytes_data)
            if is_best:
                _write_checkpoint_bytes(
                    _ckpt_path("best_model.flax"), bytes_data)
                log_message(f"  New best model! val_loss={best_val_loss:.4f}")

            log_message(f"  Best val loss so far: {best_val_loss:.4f}")
            sync_logs()

        # Release gathered copies on every host -all hosts hold them after
        # _gather_for_save; host-0-only del leaves multi-GB pinned elsewhere.
        del params_single, opt_state_single

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

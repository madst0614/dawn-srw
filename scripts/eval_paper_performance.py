#!/usr/bin/env python3
"""Manifest-driven paper evaluation for baseline Transformer and DAWN-SRW.

This runner evaluates C4 validation loss/accuracy for dense baseline
checkpoints and DAWN-SRW full/hard-pruned variants, then emits raw and
publication-ready CSV/Markdown/LaTeX tables.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    import numpy as np
    _NUMPY_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    np = None
    _NUMPY_IMPORT_ERROR = exc

try:
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import scripts.train_jax as tj
    from scripts.downstream_finetune_jax import _adapt_checkpoint_params_to_target
    from utils.data_jax import _build_dataset
    _JAX_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    # Allow --dry_run on lightweight machines. Real evaluation still requires
    # the same JAX/Flax environment used for training.
    jax = None
    jnp = None
    serialization = None
    Mesh = Any
    NamedSharding = None
    P = None
    tj = None
    _adapt_checkpoint_params_to_target = None
    _build_dataset = None
    _JAX_IMPORT_ERROR = exc


DAWN_SRW_VERSIONS = {
    "dawn_srw",
    "spatial-r1-v4.1.5.5",
    "spatial-r1-v4.1.5.2",
}

README_NOTE = (
    "The present implementation evaluates RW-neuron pools with dense/chunked "
    "computation. The reported active/strong RW FLOPs are estimated "
    "sparse-execution FLOPs. Hard-pruned validation variants are actual "
    "forward passes in which gates below the activation threshold are zeroed "
    "and RWCompose is renormalized over retained gates."
)

FLOPS_NOTE = (
    "All FLOPs use multiply-add = 2 FLOPs convention. full DAWN-SRW FLOPs "
    "means theoretical full-pool dense/chunked evaluation. sparse_exact_routing "
    "FLOPs assumes exact full-pool routing but hard-pruned RW execution. "
    "sparse_rw_only FLOPs is an idealized selected-RW execution estimate and "
    "does not represent current wall-clock cost."
)

FLOP_CONVENTION = "multiply_add_2_flops"

FLOPS_TABLE_COLUMNS = (
    "model_name", "variant", "val_loss", "delta_loss",
    "d_model", "n_layers", "seq_len", "implemented_type",
    "transformer_total_flops_per_token",
    "dawn_full_total_flops_per_token",
    "dawn_sparse_exact_total_flops_per_token",
    "dawn_sparse_rw_only_total_flops_per_token",
    "ratio_dawn_full_vs_transformer",
    "ratio_sparse_exact_vs_transformer",
    "ratio_sparse_exact_vs_dawn_full",
    "ratio_sparse_rw_only_vs_full_rw",
    "kept_qk", "kept_v", "kept_rst",
    "retained_gate_mass_qk",
    "retained_gate_mass_v",
    "retained_gate_mass_rst",
)

DAWN_METRIC_KEYS = (
    "attn_qk_active", "attn_v_active", "rst_active",
    "attn_qk_strong", "attn_v_strong", "rst_strong",
    "attn_active_n_mean", "rst_active_n_mean",
    "attn_gate_sum", "rst_gate_sum",
    "attn_gate_eff_n", "attn_gate_eff_ratio", "attn_top1_gate_frac",
    "rst_gate_eff_n", "rst_gate_eff_ratio", "rst_top1_gate_frac",
    "attn_int_max", "rst_int_max",
    "attn_score_std", "rst_score_std",
    "attn_tau_abs_mean", "rst_tau_abs_mean",
    "attn_dead_count", "rst_dead_count",
    "attn_dead_penalty", "rst_dead_penalty",
    "attn_qk_kept_count_mean", "attn_qk_kept_frac_mean",
    "attn_qk_full_gate_sum_mean", "attn_qk_kept_gate_sum_mean",
    "attn_qk_retained_gate_mass", "attn_qk_int_cap_frac",
    "attn_qk_gate_max_mean",
    "attn_v_kept_count_mean", "attn_v_kept_frac_mean",
    "attn_v_full_gate_sum_mean", "attn_v_kept_gate_sum_mean",
    "attn_v_retained_gate_mass", "attn_v_int_cap_frac",
    "attn_v_gate_max_mean",
    "rst_kept_count_mean", "rst_kept_frac_mean",
    "rst_full_gate_sum_mean", "rst_kept_gate_sum_mean",
    "rst_retained_gate_mass", "rst_int_cap_frac",
    "rst_gate_max_mean",
)


@dataclass(frozen=True)
class Variant:
    name: str
    prune_enabled: bool = False
    scope: str = "all"
    threshold: Optional[float] = None
    denominator: str = "pruned"
    notes: str = ""


MAIN_DAWN_VARIANTS = (
    Variant("full_soft", False, "all", None, notes="soft-gated full pool"),
    Variant("hard_all_t050", True, "all", 0.50),
    Variant("hard_all_t070", True, "all", 0.70),
    Variant("hard_all_t080", True, "all", 0.80),
    Variant("hard_all_t090", True, "all", 0.90),
    Variant("hard_all_t095", True, "all", 0.95),
    Variant("hard_all_t099", True, "all", 0.99),
    Variant("hard_rst_t090", True, "rst", 0.90),
    Variant("hard_attn_t090", True, "attn", 0.90),
    Variant("hard_qk_t090", True, "qk", 0.90),
    Variant("hard_v_t090", True, "v", 0.90),
)

SUB_DAWN_VARIANTS = (
    Variant("full_soft", False, "all", None, notes="soft-gated full pool"),
    Variant("sub_hard_all_t090", True, "all", 0.90),
)


def is_host0() -> bool:
    if jax is None:
        return True
    return int(jax.process_index()) == 0


def log(msg: str) -> None:
    if is_host0():
        print(msg, flush=True)


def is_gcs(path: str | os.PathLike[str]) -> bool:
    return str(path).startswith("gs://")


def join_path(base: str | os.PathLike[str], name: str) -> str:
    base_s = str(base)
    return base_s.rstrip("/") + "/" + name if is_gcs(base_s) else str(Path(base_s) / name)


def ensure_dir(path: str | os.PathLike[str]) -> str:
    path_s = str(path)
    if not is_gcs(path_s):
        Path(path_s).mkdir(parents=True, exist_ok=True)
    return path_s


def open_file(path: str | os.PathLike[str], mode: str = "r"):
    if tj is not None:
        return tj._open_file(str(path), mode)
    if is_gcs(path):
        raise RuntimeError("GCS paths require the JAX/TPU environment dependencies")
    p = Path(str(path))
    if "w" in mode or "a" in mode:
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def file_exists(path: str | os.PathLike[str]) -> bool:
    if tj is None:
        return Path(str(path)).exists()
    return tj._file_exists(str(path))


def load_manifest(path: str) -> Dict[str, Any]:
    with open_file(path, "r") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    return yaml.safe_load(text) or {}


def load_config(path: str) -> Dict[str, Any]:
    return tj.load_config(path)


def write_text(path: str, text: str) -> None:
    with open_file(path, "w") as f:
        f.write(text)


def json_safe(obj: Any) -> Any:
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    array_types = []
    if np is not None:
        array_types.append(np.ndarray)
    if jnp is not None:
        array_types.append(jnp.ndarray)
    if array_types and isinstance(obj, tuple(array_types)):
        return np.asarray(obj).tolist() if np is not None else obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    line = json.dumps(row, default=json_safe, sort_keys=True) + "\n"
    if is_gcs(path):
        old = ""
        if file_exists(path):
            with open_file(path, "r") as f:
                old = f.read()
        with open_file(path, "w") as f:
            f.write(old + line)
        return
    with open_file(path, "a") as f:
        f.write(line)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open_file(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=json_safe, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not file_exists(path):
        return []
    rows = []
    with open_file(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in columns})


def resolve_checkpoint(path: str) -> str:
    if path.endswith(".flax"):
        if not file_exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path
    best = join_path(path, "best_model.flax")
    if file_exists(best):
        return best
    step_files = tj._list_files(path, "checkpoint_step*.flax")
    if step_files:
        return step_files[-1]
    any_files = tj._list_files(path, "*.flax")
    if any_files:
        return any_files[-1]
    raise FileNotFoundError(f"No .flax checkpoint found in {path}")


def init_target_params(model, cfg: Dict[str, Any], seq_len: int):
    seed = int(cfg.get("seed", 1))
    dummy_len = min(int(cfg.get("model", {}).get("max_seq_len", seq_len)), seq_len)
    dummy = jnp.ones((1, dummy_len), dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(seed),
         "dropout": jax.random.PRNGKey(seed + 1)},
        dummy,
        labels=dummy,
        attention_mask=jnp.ones_like(dummy),
        deterministic=True,
    )
    return variables["params"]


def restore_params(checkpoint: str, target_params, model_version: str):
    ckpt_file = resolve_checkpoint(checkpoint)
    log(f"  loading checkpoint: {ckpt_file}")
    with open_file(ckpt_file, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    if model_version in DAWN_SRW_VERSIONS and isinstance(raw, dict):
        raw = tj.migrate_legacy_v4155_params(raw)
        if hasattr(tj, "_migrate_v4152_route_params"):
            raw = tj._migrate_v4152_route_params(raw, {"params": target_params})
    raw_params = raw.get("params", raw) if isinstance(raw, dict) else raw
    raw_params = _adapt_checkpoint_params_to_target(raw_params, target_params)
    params = serialization.from_state_dict(
        {"params": target_params}, {"params": raw_params})["params"]
    meta = {
        "checkpoint_file": ckpt_file,
        "step": int(raw.get("step", 0)) if isinstance(raw, dict) else 0,
        "epoch": int(raw.get("epoch", 0)) if isinstance(raw, dict) else 0,
        "best_val_loss": float(raw.get("best_val_loss", float("nan"))) if isinstance(raw, dict) else float("nan"),
    }
    return params, meta


def count_params(params) -> int:
    return int(tj.count_parameters(params))


def infer_mesh_shape(cfg: Dict[str, Any], is_baseline: bool, args: argparse.Namespace,
                     model_entry: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    if args.single_device:
        return 1, 1
    n_devices = int(jax.device_count())
    model_entry = model_entry or {}
    mesh_model = args.mesh_model if args.mesh_model is not None else model_entry.get("mesh_model")
    if mesh_model is None:
        mesh_model = 1 if is_baseline else int(cfg.get("training", {}).get("mesh_model", 1) or 1)
    mesh_data = args.mesh_data if args.mesh_data is not None else model_entry.get("mesh_data")
    if mesh_data is None:
        cfg_mesh_data = int(cfg.get("training", {}).get("mesh_data", 0) or 0)
        mesh_data = cfg_mesh_data if cfg_mesh_data > 0 else max(1, n_devices // int(mesh_model))
    if int(mesh_data) * int(mesh_model) != n_devices:
        raise ValueError(
            f"mesh_data({mesh_data}) * mesh_model({mesh_model}) must equal "
            f"jax.device_count()({n_devices})")
    return int(mesh_data), int(mesh_model)


def create_mesh(mesh_data: int, mesh_model: int, single_device: bool) -> Mesh:
    devices = jax.devices()
    if single_device:
        devices = devices[:1]
        mesh_data, mesh_model = 1, 1
    return Mesh(np.asarray(devices).reshape(mesh_data, mesh_model), ("data", "model"))


def chunk_sizes(cfg: Dict[str, Any], mesh_model: int, batch_size: int, seq_len: int) -> Tuple[int, int, int]:
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    n_qk = int(m.get("n_qk", 1580))
    n_v = int(m.get("n_v", 2600))
    n_rst = int(m.get("n_rst", m.get("n_know", 25200)))
    for name, n in (("n_qk", n_qk), ("n_v", n_v), ("n_rst", n_rst)):
        if n % mesh_model != 0:
            raise ValueError(f"{name}={n} must be divisible by mesh_model={mesh_model}")
    if t.get("max_chunk_size") is not None:
        forced = int(t["max_chunk_size"])
        return forced, forced, forced
    per_device_batch = max(1, batch_size // max(1, jax.device_count()))
    target_gb = float(t.get("target_chunk_gb", 2.0))

    def auto_chunks(n_local: int, configured: Optional[int]) -> int:
        if configured:
            return int(configured)
        full_gb = per_device_batch * seq_len * n_local * 2 / 1.0e9
        nc = max(1, int(math.ceil(full_gb / target_gb)))
        while n_local % nc != 0 and nc < n_local:
            nc += 1
        return min(nc, n_local)

    def max_chunk(n_local: int, n_chunks: int) -> int:
        n_chunks = max(1, int(n_chunks))
        if n_chunks > n_local:
            raise ValueError(f"n_chunks={n_chunks} exceeds local pool size {n_local}")
        return max(1, int(math.ceil(n_local / n_chunks)))

    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model
    nrst_local = n_rst // mesh_model
    return (
        max_chunk(nqk_local, auto_chunks(nqk_local, t.get("n_chunks_qk"))),
        max_chunk(nv_local, auto_chunks(nv_local, t.get("n_chunks_v"))),
        max_chunk(nrst_local, auto_chunks(nrst_local, t.get("n_chunks_rst", t.get("n_chunks_know")))),
    )


def pool_prunes(pool: str, variant: Variant) -> bool:
    if not variant.prune_enabled:
        return False
    if variant.scope == "all":
        return True
    if variant.scope == "attn":
        return pool in ("qk", "v")
    return variant.scope == pool


def build_sharded_fns(cfg: Dict[str, Any], mesh: Mesh, mesh_model: int,
                      batch_size: int, seq_len: int, variant: Variant):
    version = cfg.get("model", {}).get("model_version", "dawn_srw")
    spec = tj.MODEL_REGISTRY.get(version)
    if spec is None or not spec.supports_sharded:
        return None
    if mesh_model <= 1 and not getattr(spec, "force_sharded", False):
        return None

    module = __import__(spec.module_path, fromlist=["make_sharded_srw"])
    qk_chunk, v_chunk, rst_chunk = chunk_sizes(cfg, mesh_model, batch_size, seq_len)
    base = {"mesh": mesh}
    if spec.sharded_kwargs is not None:
        base.update(spec.sharded_kwargs(cfg))

    def kwargs_for(pool: str) -> Dict[str, Any]:
        kw = dict(base)
        enabled = pool_prunes(pool, variant)
        kw.update({
            "prune_enabled": enabled,
            "prune_activation_threshold": variant.threshold if enabled else None,
            "prune_scope": variant.scope,
            "prune_denominator": variant.denominator,
            "return_prune_stats": True,
        })
        return kw

    single_v = module.make_sharded_srw(max_chunk_size=v_chunk, **kwargs_for("v"))
    single_rst = module.make_sharded_srw(max_chunk_size=rst_chunk, **kwargs_for("rst"))
    paired = module.make_sharded_srw_paired(max_chunk_size=qk_chunk, **kwargs_for("qk"))
    log(
        "  shard_map: "
        f"mesh_model={mesh_model}, chunks qk/v/rst={qk_chunk}/{v_chunk}/{rst_chunk}, "
        f"variant={variant.name}")
    return {
        "single": single_v,
        "attn_v_single": single_v,
        "rst_single": single_rst,
        "paired": paired,
        "attn_qk_paired": paired,
    }


def make_eval_step(model, sharded_fns, return_prune_stats: bool):
    import inspect
    try:
        params = inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_analysis = "analysis" in params
    accepts_prune_stats = "return_prune_stats" in params

    metric_keys = DAWN_METRIC_KEYS if return_prune_stats else ()

    @jax.jit
    def eval_step(params_tree, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        kwargs = {}
        if sharded_fns is not None:
            kwargs["sharded_fns"] = sharded_fns
        if accepts_analysis:
            kwargs["analysis"] = False
        if accepts_prune_stats:
            kwargs["return_prune_stats"] = return_prune_stats
        result = model.apply(
            {"params": params_tree},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(0)},
            **kwargs,
        )
        metrics = {
            k: jnp.asarray(result.get(k, jnp.float32(0.0)), dtype=jnp.float32)
            for k in metric_keys
        }
        return result["loss"], result["correct"], result["valid_count"], metrics

    return eval_step


def shard_global_batch(global_batch: np.ndarray, sharding, global_shape):
    def callback(index):
        return np.asarray(global_batch[index])
    return jax.make_array_from_callback(global_shape, sharding, callback)


def iter_validation_batches(val_bin: str, seq_len: int, batch_size: int,
                            max_batches: Optional[int], local_cache_dir: Optional[str]):
    max_sequences = None
    if max_batches is not None and max_batches >= 0:
        max_sequences = max_batches * batch_size
    dataset = _build_dataset(val_bin, seq_len, max_sequences, local_cache_dir)
    n_batches = len(dataset) // batch_size
    if max_batches is not None and max_batches >= 0:
        n_batches = min(n_batches, max_batches)
    for b in range(n_batches):
        batch = dataset.get_batch(b * batch_size, batch_size)
        if batch is None or len(batch) < batch_size:
            break
        yield b, np.asarray(batch, dtype=np.int32), n_batches


def evaluate_variant(model, params, cfg: Dict[str, Any], val_bin: str,
                     batch_size: int, seq_len: int,
                     max_batches: Optional[int], local_cache_dir: Optional[str],
                     mesh: Optional[Mesh], data_sharding, sharded_fns,
                     variant: Variant, progress_every: int) -> Dict[str, Any]:
    eval_step = make_eval_step(model, sharded_fns, return_prune_stats=cfg["model"].get("model_version") in DAWN_SRW_VERSIONS)
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0.0
    stat_sums = {k: 0.0 for k in DAWN_METRIC_KEYS}
    n_batches_done = 0
    t0 = time.time()

    for b, batch_np, n_batches in iter_validation_batches(
            val_bin, seq_len, batch_size, max_batches, local_cache_dir):
        mask_np = np.ones_like(batch_np, dtype=np.int32)
        if mesh is not None:
            batch = shard_global_batch(batch_np, data_sharding, (batch_size, seq_len))
            mask = shard_global_batch(mask_np, data_sharding, (batch_size, seq_len))
        else:
            batch = jnp.asarray(batch_np)
            mask = jnp.asarray(mask_np)
        loss, correct, valid, metrics = eval_step(params, batch, mask)
        host = jax.device_get((loss, correct, valid, metrics))
        loss_f = float(host[0])
        correct_f = float(host[1])
        valid_f = float(host[2])
        total_loss += loss_f * valid_f
        total_correct += correct_f
        total_valid += valid_f
        for key, value in host[3].items():
            stat_sums[key] = stat_sums.get(key, 0.0) + float(value) * valid_f
        n_batches_done += 1
        if is_host0() and (b + 1 == n_batches or (b + 1) % progress_every == 0):
            print(f"    {variant.name}: {b + 1}/{n_batches} batches", flush=True)

    if total_valid <= 0:
        raise RuntimeError("No valid tokens evaluated")
    elapsed = time.time() - t0
    avg_stats = {k: v / total_valid for k, v in stat_sums.items()}
    loss = total_loss / total_valid
    return {
        "val_loss": loss,
        "ppl": math.exp(loss) if loss < 100 else float("inf"),
        "accuracy": total_correct / total_valid,
        "valid_tokens": int(total_valid),
        "num_batches": int(n_batches_done),
        "time_sec": elapsed,
        "tokens_per_sec": total_valid / elapsed if elapsed > 0 else 0.0,
        "metrics": avg_stats,
    }


def model_pool_sizes(cfg: Dict[str, Any]) -> Dict[str, int]:
    m = cfg.get("model", {})
    return {
        "qk": int(m.get("n_qk", 0) or 0),
        "v": int(m.get("n_v", 0) or 0),
        "rst": int(m.get("n_rst", m.get("n_know", 0)) or 0),
    }


def safe_div(num: Any, den: Any) -> Optional[float]:
    if num is None or den in (None, 0, 0.0, ""):
        return None
    return float(num) / float(den)


def model_dims(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg.get("model", {})
    d_model = int(m.get("d_model", 0) or 0)
    d_ff = m.get("d_ff")
    return {
        "d_model": d_model,
        "d_ff": int(d_ff) if d_ff is not None else None,
        "n_layers": int(m.get("n_layers", 0) or 0),
        "n_heads": int(m.get("n_heads", 0) or 0),
        "d_route": int(m.get("d_route", m.get("d_bottleneck", 0)) or 0),
        "vocab_size": int(m.get("vocab_size", 30522) or 30522),
    }


def add_common_flop_metadata(row: Dict[str, Any], cfg: Dict[str, Any],
                             include_lm_head_flops: bool) -> Dict[str, Any]:
    dims = model_dims(cfg)
    row.update({
        "flop_convention": FLOP_CONVENTION,
        "d_model": dims["d_model"],
        "d_ff": dims["d_ff"],
        "n_layers": dims["n_layers"],
        "n_heads": dims["n_heads"],
        "d_route": dims["d_route"] if row.get("model_type") == "dawn_srw" else None,
        "vocab_size": dims["vocab_size"],
        "include_lm_head_flops": bool(include_lm_head_flops),
    })
    return dims


def add_transformer_theoretical_flops(row: Dict[str, Any], cfg: Dict[str, Any],
                                      include_lm_head_flops: bool) -> None:
    dims = add_common_flop_metadata(row, cfg, include_lm_head_flops)
    d_model = int(dims["d_model"])
    d_ff = int(dims["d_ff"] or (4 * d_model))
    n_layers = int(dims["n_layers"])
    seq_len = int(row["seq_len"])
    vocab_size = int(dims["vocab_size"])

    qkv = 3 * 2 * d_model * d_model
    out_proj = 2 * d_model * d_model
    attn_proj_total = qkv + out_proj
    ffn = (2 * d_model * d_ff) + (2 * d_ff * d_model)
    attention_causal_avg = 2 * d_model * (seq_len + 1)
    layer = attn_proj_total + attention_causal_avg + ffn
    total_per_token = layer * n_layers
    lm_head = 2 * d_model * vocab_size
    row.update({
        "d_ff": d_ff,
        "implemented_type": row.get("implemented_type") or "dense_transformer",
        "transformer_qkv_proj_flops": qkv,
        "transformer_out_proj_flops": out_proj,
        "transformer_attn_proj_total_flops": attn_proj_total,
        "transformer_ffn_flops": ffn,
        "transformer_attention_causal_avg_flops": attention_causal_avg,
        "transformer_layer_flops_per_token": layer,
        "transformer_total_flops_per_token": total_per_token,
        "transformer_total_flops_per_sequence": total_per_token * seq_len,
        "lm_head_flops_per_token": lm_head,
        "total_with_lm_head_flops_per_token": (
            total_per_token + lm_head if include_lm_head_flops else total_per_token
        ),
    })


def add_dawn_theoretical_flops(row: Dict[str, Any], cfg: Dict[str, Any],
                               include_lm_head_flops: bool) -> None:
    dims = add_common_flop_metadata(row, cfg, include_lm_head_flops)
    sizes = model_pool_sizes(cfg)
    d_model = int(dims["d_model"])
    d_route = int(dims["d_route"])
    n_layers = int(dims["n_layers"])
    seq_len = int(row["seq_len"])
    vocab_size = int(dims["vocab_size"])
    n_qk, n_v, n_rst = sizes["qk"], sizes["v"], sizes["rst"]

    route_query = 2 * d_model * d_route
    q_score = 2 * n_qk * d_route
    k_score = 2 * n_qk * d_route
    v_score = 2 * n_v * d_route
    rst_score = 2 * n_rst * d_route
    rw_read_only = 2 * d_model
    rw_read_write = 4 * d_model

    q_rw_full = n_qk * rw_read_write
    k_rw_full = n_qk * rw_read_write
    v_rw_full = n_v * rw_read_write
    rst_rw_full = n_rst * rw_read_write
    full_rw_total = q_rw_full + k_rw_full + v_rw_full + rst_rw_full

    dawn_attention = 2 * d_model * (seq_len + 1)
    attn_out_proj = 2 * d_model * d_model
    full_qkv_route_score = (
        route_query + route_query + route_query
        + q_score + k_score + v_score
    )
    full_qkv_rw = q_rw_full + k_rw_full + v_rw_full
    full_rst_route_score = route_query + rst_score
    full_rst_rw = rst_rw_full
    full_layer = (
        full_qkv_route_score
        + full_qkv_rw
        + dawn_attention
        + attn_out_proj
        + full_rst_route_score
        + full_rst_rw
    )
    full_total = full_layer * n_layers

    kept_qk = float(row.get("estimated_executed_rw_ops_qk") or 0.0)
    kept_v = float(row.get("estimated_executed_rw_ops_v") or 0.0)
    kept_rst = float(row.get("estimated_executed_rw_ops_rst") or 0.0)
    q_rw_kept = kept_qk * rw_read_write
    k_rw_kept = kept_qk * rw_read_write
    v_rw_kept = kept_v * rw_read_write
    rst_rw_kept = kept_rst * rw_read_write
    sparse_exact_layer = (
        full_qkv_route_score
        + q_rw_kept + k_rw_kept + v_rw_kept
        + dawn_attention
        + attn_out_proj
        + full_rst_route_score
        + rst_rw_kept
    )
    sparse_exact_total = sparse_exact_layer * n_layers
    sparse_rw_only_layer = q_rw_kept + k_rw_kept + v_rw_kept + rst_rw_kept
    sparse_rw_only_total = sparse_rw_only_layer * n_layers
    lm_head = 2 * d_model * vocab_size

    row.update({
        "implemented_type": row.get("implemented_type") or "full_pool_dense_chunked",
        "n_qk": n_qk,
        "n_v": n_v,
        "n_rst": n_rst,
        "route_query_flops_per_route": route_query,
        "q_route_query_flops": route_query,
        "k_route_query_flops": route_query,
        "v_route_query_flops": route_query,
        "rst_route_query_flops": route_query,
        "q_score_flops_full": q_score,
        "k_score_flops_full": k_score,
        "v_score_flops_full": v_score,
        "rst_score_flops_full": rst_score,
        "rw_read_only_flops_per_neuron": rw_read_only,
        "rw_read_write_flops_per_neuron": rw_read_write,
        "q_rw_flops_full": q_rw_full,
        "k_rw_flops_full": k_rw_full,
        "v_rw_flops_full": v_rw_full,
        "rst_rw_flops_full": rst_rw_full,
        "dawn_attention_causal_avg_flops": dawn_attention,
        "dawn_attn_out_proj_flops": attn_out_proj,
        "dawn_full_qkv_route_score_flops": full_qkv_route_score,
        "dawn_full_qkv_rw_flops": full_qkv_rw,
        "dawn_full_rst_route_score_flops": full_rst_route_score,
        "dawn_full_rst_rw_flops": full_rst_rw,
        "dawn_full_layer_flops_per_token": full_layer,
        "dawn_full_total_flops_per_token": full_total,
        "dawn_full_total_flops_per_sequence": full_total * seq_len,
        "q_rw_flops_kept": q_rw_kept,
        "k_rw_flops_kept": k_rw_kept,
        "v_rw_flops_kept": v_rw_kept,
        "rst_rw_flops_kept": rst_rw_kept,
        "dawn_sparse_exact_layer_flops_per_token": sparse_exact_layer,
        "dawn_sparse_exact_total_flops_per_token": sparse_exact_total,
        "dawn_sparse_rw_only_layer_flops_per_token": sparse_rw_only_layer,
        "dawn_sparse_rw_only_total_flops_per_token": sparse_rw_only_total,
        "ratio_sparse_exact_vs_dawn_full": safe_div(sparse_exact_total, full_total),
        "ratio_sparse_rw_only_vs_full_rw": safe_div(sparse_rw_only_layer, full_rw_total),
        "sparse_exact_routing_flops_note": (
            "Full-pool route/score cost plus hard-pruned RW execution cost."
        ),
        "sparse_rw_only_flops_note": (
            "Idealized selected-RW execution cost; not current wall-clock cost."
        ),
        "lm_head_flops_per_token": lm_head,
        "dawn_full_total_with_lm_head_flops_per_token": (
            full_total + lm_head if include_lm_head_flops else full_total
        ),
        "dawn_sparse_exact_total_with_lm_head_flops_per_token": (
            sparse_exact_total + lm_head if include_lm_head_flops else sparse_exact_total
        ),
    })


def pool_stats(metrics: Dict[str, float], cfg: Dict[str, Any], variant: Variant) -> Dict[str, Dict[str, Optional[float]]]:
    sizes = model_pool_sizes(cfg)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    maps = {
        "qk": "attn_qk",
        "v": "attn_v",
        "rst": "rst",
    }
    for pool, prefix in maps.items():
        n = sizes[pool]
        active_frac = float(metrics.get(f"{prefix}_active", 0.0))
        strong_frac = float(metrics.get(f"{prefix}_strong", 0.0))
        kept = metrics.get(f"{prefix}_kept_count_mean")
        kept_frac = metrics.get(f"{prefix}_kept_frac_mean")
        if kept is None or kept == 0.0 and not variant.prune_enabled:
            kept = float(n)
            kept_frac = 1.0
        retained = metrics.get(f"{prefix}_retained_gate_mass")
        if retained is None or retained == 0.0 and not variant.prune_enabled:
            retained = 1.0
        out[pool] = {
            "pool_size": n,
            "active_count_mean": active_frac * n,
            "strong_count_mean": strong_frac * n,
            "kept_count_mean": float(kept),
            "active_frac_mean": active_frac,
            "strong_frac_mean": strong_frac,
            "kept_frac_mean": float(kept_frac if kept_frac is not None else 1.0),
            "full_gate_sum_mean": metrics.get(f"{prefix}_full_gate_sum_mean"),
            "kept_gate_sum_mean": metrics.get(f"{prefix}_kept_gate_sum_mean"),
            "retained_gate_mass": min(1.000001, max(-1.0e-6, float(retained))),
            "gate_eff_n_mean": metrics.get(f"{prefix}_gate_eff_n", metrics.get("attn_gate_eff_n" if pool in ("qk", "v") else "rst_gate_eff_n")),
            "gate_eff_ratio_mean": metrics.get(f"{prefix}_gate_eff_ratio", metrics.get("attn_gate_eff_ratio" if pool in ("qk", "v") else "rst_gate_eff_ratio")),
            "top1_gate_frac_mean": metrics.get(f"{prefix}_top1_gate_frac", metrics.get("attn_top1_gate_frac" if pool in ("qk", "v") else "rst_top1_gate_frac")),
            "gate_max_mean": metrics.get(f"{prefix}_gate_max_mean"),
            "int_max_mean": metrics.get("attn_int_max" if pool in ("qk", "v") else "rst_int_max"),
            "int_cap_frac": metrics.get(f"{prefix}_int_cap_frac"),
            "score_std_mean": metrics.get("attn_score_std" if pool in ("qk", "v") else "rst_score_std"),
            "tau_abs_mean": metrics.get("attn_tau_abs_mean" if pool in ("qk", "v") else "rst_tau_abs_mean"),
            "dead_count": metrics.get("attn_dead_count" if pool in ("qk", "v") else "rst_dead_count"),
            "dead_penalty": metrics.get("attn_dead_penalty" if pool in ("qk", "v") else "rst_dead_penalty"),
        }
    return out


def add_sparse_estimates(row: Dict[str, Any], cfg: Dict[str, Any],
                         stats: Dict[str, Dict[str, Optional[float]]]) -> None:
    sizes = model_pool_sizes(cfg)
    d_model = int(cfg.get("model", {}).get("d_model", 0) or 0)
    rw_read_only_flops_per_neuron = 2 * d_model
    rw_flops_per_neuron = 4 * d_model
    kept_qk = float(stats["qk"]["kept_count_mean"] or 0.0)
    kept_v = float(stats["v"]["kept_count_mean"] or 0.0)
    kept_rst = float(stats["rst"]["kept_count_mean"] or 0.0)
    full_total = sizes["qk"] + sizes["v"] + sizes["rst"]
    kept_total = kept_qk + kept_v + kept_rst
    row.update({
        "implemented_pool_eval": "full_pool_dense_chunked",
        "implemented_type": row.get("implemented_type") or "full_pool_dense_chunked",
        "sparse_flops_is_estimate": True,
        "rw_flops_per_neuron_convention": "4*d_model (read dot plus write accumulation)",
        "rw_read_only_flops_per_neuron": rw_read_only_flops_per_neuron,
        "rw_read_write_flops_per_neuron": rw_flops_per_neuron,
        "estimated_executed_rw_ops_qk": kept_qk,
        "estimated_executed_rw_ops_v": kept_v,
        "estimated_executed_rw_ops_rst": kept_rst,
        "estimated_executed_rw_ops_total_per_token_per_layer": kept_total,
        "estimated_pool_ratio_qk": kept_qk / sizes["qk"] if sizes["qk"] else None,
        "estimated_pool_ratio_v": kept_v / sizes["v"] if sizes["v"] else None,
        "estimated_pool_ratio_rst": kept_rst / sizes["rst"] if sizes["rst"] else None,
        "estimated_rw_flops_per_token_per_layer_qk": kept_qk * rw_flops_per_neuron,
        "estimated_rw_flops_per_token_per_layer_v": kept_v * rw_flops_per_neuron,
        "estimated_rw_flops_per_token_per_layer_rst": kept_rst * rw_flops_per_neuron,
        "estimated_rw_flops_per_token_per_layer_total": kept_total * rw_flops_per_neuron,
        "estimated_rw_flops_ratio_vs_full_pool": kept_total / full_total if full_total else None,
    })


def parse_models(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    exp = manifest.get("experiments", manifest)
    models = []
    for item in exp.get("baseline", []) or []:
        models.append({**item, "family": "baseline", "role": item.get("role", "main")})
    dawn = exp.get("dawn_srw", {}) or {}
    for item in dawn.get("main", []) or []:
        models.append({**item, "family": "dawn_srw", "role": "main"})
    for item in dawn.get("sub", []) or []:
        models.append({**item, "family": "dawn_srw", "role": "sub"})
    return models


def variants_for(model_entry: Dict[str, Any]) -> Tuple[Variant, ...]:
    if model_entry["family"] == "baseline":
        return (Variant("full_validation", False, notes="baseline full validation"),)
    if model_entry.get("role") == "sub":
        return SUB_DAWN_VARIANTS
    return MAIN_DAWN_VARIANTS


def selected_variants_for(model_entry: Dict[str, Any],
                          args: Optional[argparse.Namespace]) -> Tuple[Variant, ...]:
    variants = variants_for(model_entry)
    if args is None or not getattr(args, "variants", None):
        return variants
    if model_entry["family"] == "baseline":
        return variants
    wanted = {
        name.strip()
        for name in str(args.variants).split(",")
        if name.strip()
    }
    selected = tuple(v for v in variants if v.name in wanted)
    if not selected:
        raise ValueError(
            f"No variants selected for {model_entry.get('name')}; "
            f"requested={sorted(wanted)}, available={[v.name for v in variants]}")
    return selected


def resolve_run_config(manifest: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    exp = manifest.get("experiments", manifest)
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = exp.get("max_batches", None)
    if max_batches == -1:
        max_batches = None
    output_dir = args.output_dir or exp.get("output_dir") or "results/paper_eval"
    return {
        "val_bin": exp["val_bin"],
        "seq_len": int(args.seq_len or exp.get("seq_len", 512)),
        "batch_size": int(args.batch_size or exp.get("batch_size", 8)),
        "max_batches": max_batches,
        "output_dir": output_dir,
        "local_cache_dir": exp.get("local_cache_dir"),
    }


def completed_keys(raw_path: str) -> set[Tuple[str, str]]:
    done = set()
    for row in read_jsonl(raw_path):
        if row.get("status", "ok") == "ok":
            done.add((row.get("model_name"), row.get("variant")))
    return done


def threshold_label(v: Variant) -> Optional[float]:
    return v.threshold


def build_row(model_entry: Dict[str, Any], cfg: Dict[str, Any], params,
              ckpt_meta: Dict[str, Any], variant: Variant,
              eval_out: Dict[str, Any], run_cfg: Dict[str, Any],
              args: argparse.Namespace) -> Dict[str, Any]:
    model_version = cfg.get("model", {}).get("model_version", "")
    model_type = "baseline_transformer" if model_entry["family"] == "baseline" else "dawn_srw"
    row = {
        "status": "ok",
        "model_name": model_entry["name"],
        "model_type": model_type,
        "role": model_entry.get("role", ""),
        "variant": variant.name,
        "implemented_type": (
            "dense_transformer"
            if model_entry["family"] == "baseline"
            else (
                "full_pool_dense_chunked_hard_pruned_gates"
                if variant.prune_enabled
                else "full_pool_dense_chunked_soft_gated"
            )
        ),
        "prune_enabled": variant.prune_enabled,
        "prune_scope": variant.scope if variant.prune_enabled else "",
        "prune_activation_threshold": threshold_label(variant),
        "prune_denominator": variant.denominator if variant.prune_enabled else "",
        "params": count_params(params),
        "checkpoint": model_entry["checkpoint"],
        "checkpoint_file": ckpt_meta.get("checkpoint_file", ""),
        "checkpoint_step": ckpt_meta.get("step", 0),
        "config": model_entry["config"],
        "model_version": model_version,
        "val_loss": eval_out["val_loss"],
        "ppl": eval_out["ppl"],
        "accuracy": eval_out["accuracy"],
        "valid_tokens": eval_out["valid_tokens"],
        "num_batches": eval_out["num_batches"],
        "seq_len": run_cfg["seq_len"],
        "batch_size": run_cfg["batch_size"],
        "dtype": args.dtype,
        "notes": variant.notes,
    }
    if model_entry["family"] == "baseline":
        add_transformer_theoretical_flops(row, cfg, args.include_lm_head_flops)
    elif model_entry["family"] == "dawn_srw":
        stats = pool_stats(eval_out["metrics"], cfg, variant)
        row["pruning_stats"] = stats
        for pool in ("qk", "v", "rst"):
            for key, val in stats[pool].items():
                row[f"{pool}_{key}"] = val
        add_sparse_estimates(row, cfg, stats)
        add_dawn_theoretical_flops(row, cfg, args.include_lm_head_flops)
    return row


def add_deltas(rows: List[Dict[str, Any]]) -> None:
    full_by_model = {
        r["model_name"]: r for r in rows
        if r.get("model_type") == "dawn_srw" and r.get("variant") == "full_soft"
        and r.get("status", "ok") == "ok"
    }
    baseline_by_role = {}
    for r in rows:
        if r.get("model_type") == "baseline_transformer" and r.get("status", "ok") == "ok":
            baseline_by_role.setdefault(r.get("role", "main"), r)
    for r in rows:
        if r.get("status", "ok") != "ok":
            continue
        full = full_by_model.get(r.get("model_name"))
        if full and r.get("model_type") == "dawn_srw":
            r["delta_loss_vs_full_soft"] = r["val_loss"] - full["val_loss"]
            r["loss_delta_vs_full_soft"] = r["delta_loss_vs_full_soft"]
            r["delta_acc_vs_full_soft"] = r["accuracy"] - full["accuracy"]
            r["accuracy_delta_vs_full_soft"] = r["delta_acc_vs_full_soft"]
            r["kept_ratio_vs_full_pool"] = r.get("estimated_rw_flops_ratio_vs_full_pool")
            masses = [
                r.get("qk_retained_gate_mass"),
                r.get("v_retained_gate_mass"),
                r.get("rst_retained_gate_mass"),
            ]
            masses = [float(x) for x in masses if x is not None]
            r["retained_gate_mass"] = sum(masses) / len(masses) if masses else None
        base = baseline_by_role.get(r.get("role", "main")) or baseline_by_role.get("main")
        if base and r.get("model_type") == "dawn_srw":
            r["loss_delta_vs_baseline"] = r["val_loss"] - base["val_loss"]
            r["accuracy_delta_vs_baseline"] = r["accuracy"] - base["accuracy"]
            transformer_flops = base.get("transformer_total_flops_per_token")
            if transformer_flops is not None:
                r["transformer_total_flops_per_token"] = transformer_flops
                r["ratio_dawn_full_vs_transformer"] = safe_div(
                    r.get("dawn_full_total_flops_per_token"), transformer_flops)
                r["ratio_sparse_exact_vs_transformer"] = safe_div(
                    r.get("dawn_sparse_exact_total_flops_per_token"), transformer_flops)


def load_all_rows(raw_path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(raw_path)
    add_deltas(rows)
    return rows


def fnum(x: Any, digits: int = 4) -> Any:
    if x is None or x == "":
        return ""
    try:
        return round(float(x), digits)
    except Exception:
        return x


def flops_table_row(r: Dict[str, Any]) -> Dict[str, Any]:
    if r.get("model_type") == "dawn_srw":
        delta_loss = r.get("delta_loss_vs_full_soft")
    else:
        delta_loss = r.get("loss_delta_vs_baseline", "")
    return {
        "model_name": r.get("model_name", ""),
        "variant": r.get("variant", ""),
        "val_loss": fnum(r.get("val_loss"), 5),
        "delta_loss": fnum(delta_loss, 5),
        "d_model": r.get("d_model", ""),
        "n_layers": r.get("n_layers", ""),
        "seq_len": r.get("seq_len", ""),
        "implemented_type": r.get("implemented_type", ""),
        "transformer_total_flops_per_token": fnum(r.get("transformer_total_flops_per_token"), 2),
        "dawn_full_total_flops_per_token": fnum(r.get("dawn_full_total_flops_per_token"), 2),
        "dawn_sparse_exact_total_flops_per_token": fnum(r.get("dawn_sparse_exact_total_flops_per_token"), 2),
        "dawn_sparse_rw_only_total_flops_per_token": fnum(r.get("dawn_sparse_rw_only_total_flops_per_token"), 2),
        "ratio_dawn_full_vs_transformer": fnum(r.get("ratio_dawn_full_vs_transformer"), 5),
        "ratio_sparse_exact_vs_transformer": fnum(r.get("ratio_sparse_exact_vs_transformer"), 5),
        "ratio_sparse_exact_vs_dawn_full": fnum(r.get("ratio_sparse_exact_vs_dawn_full"), 5),
        "ratio_sparse_rw_only_vs_full_rw": fnum(r.get("ratio_sparse_rw_only_vs_full_rw"), 5),
        "kept_qk": fnum(r.get("estimated_executed_rw_ops_qk"), 2),
        "kept_v": fnum(r.get("estimated_executed_rw_ops_v"), 2),
        "kept_rst": fnum(r.get("estimated_executed_rw_ops_rst"), 2),
        "retained_gate_mass_qk": fnum(r.get("qk_retained_gate_mass"), 4),
        "retained_gate_mass_v": fnum(r.get("v_retained_gate_mass"), 4),
        "retained_gate_mass_rst": fnum(r.get("rst_retained_gate_mass"), 4),
    }


def summarize_tables(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    main_perf = []
    for r in ok:
        if r.get("model_type") == "baseline_transformer" or (
                r.get("model_type") == "dawn_srw"
                and r.get("role") == "main"
                and r.get("variant") in ("full_soft", "hard_all_t090")):
            main_perf.append({
                "model_name": r["model_name"] if r.get("variant") in ("full_validation", "full_soft") else f"{r['model_name']} ({r['variant']})",
                "model_type": r["model_type"],
                "params": r["params"],
                "tokens": r["valid_tokens"],
                "val_loss": fnum(r["val_loss"], 5),
                "ppl": fnum(r["ppl"], 3),
                "acc": fnum(r["accuracy"], 5),
                "notes": r.get("notes", ""),
            })

    hard_main = []
    for r in ok:
        if r.get("model_type") == "dawn_srw" and r.get("role") == "main" and (
                r.get("variant") == "full_soft" or r.get("variant", "").endswith("_t090")):
            hard_main.append({
                "model_name": r["model_name"],
                "variant": r["variant"],
                "threshold": r.get("prune_activation_threshold"),
                "val_loss": fnum(r["val_loss"], 5),
                "delta_loss_vs_full": fnum(r.get("delta_loss_vs_full_soft"), 5),
                "acc": fnum(r["accuracy"], 5),
                "delta_acc_vs_full": fnum(r.get("delta_acc_vs_full_soft"), 5),
                "kept_qk": fnum(r.get("estimated_executed_rw_ops_qk"), 2),
                "kept_v": fnum(r.get("estimated_executed_rw_ops_v"), 2),
                "kept_rst": fnum(r.get("estimated_executed_rw_ops_rst"), 2),
                "retained_gate_mass_qk": fnum(r.get("qk_retained_gate_mass"), 4),
                "retained_gate_mass_v": fnum(r.get("v_retained_gate_mass"), 4),
                "retained_gate_mass_rst": fnum(r.get("rst_retained_gate_mass"), 4),
                "estimated_rw_flops_ratio_vs_full_pool": fnum(r.get("estimated_rw_flops_ratio_vs_full_pool"), 5),
            })

    threshold_sweep = []
    for r in ok:
        if r.get("model_type") == "dawn_srw" and r.get("role") == "main" and (
                r.get("variant") == "full_soft" or r.get("variant", "").startswith("hard_all_t")):
            threshold_sweep.append(dict(hard_main[-1]) if False else {
                "model_name": r["model_name"],
                "variant": r["variant"],
                "threshold": r.get("prune_activation_threshold"),
                "val_loss": fnum(r["val_loss"], 5),
                "delta_loss_vs_full": fnum(r.get("delta_loss_vs_full_soft"), 5),
                "acc": fnum(r["accuracy"], 5),
                "delta_acc_vs_full": fnum(r.get("delta_acc_vs_full_soft"), 5),
                "kept_qk": fnum(r.get("estimated_executed_rw_ops_qk"), 2),
                "kept_v": fnum(r.get("estimated_executed_rw_ops_v"), 2),
                "kept_rst": fnum(r.get("estimated_executed_rw_ops_rst"), 2),
                "retained_gate_mass_qk": fnum(r.get("qk_retained_gate_mass"), 4),
                "retained_gate_mass_v": fnum(r.get("v_retained_gate_mass"), 4),
                "retained_gate_mass_rst": fnum(r.get("rst_retained_gate_mass"), 4),
                "estimated_rw_flops_ratio_vs_full_pool": fnum(r.get("estimated_rw_flops_ratio_vs_full_pool"), 5),
            })

    sub_rows = []
    by_model = {}
    for r in ok:
        if r.get("model_type") == "dawn_srw" and r.get("role") == "sub":
            by_model.setdefault(r["model_name"], {})[r["variant"]] = r
    for name, variants in by_model.items():
        full = variants.get("full_soft")
        strong = variants.get("sub_hard_all_t090")
        if not full or not strong:
            continue
        sub_rows.append({
            "model_name": name,
            "full_loss": fnum(full["val_loss"], 5),
            "strong_loss": fnum(strong["val_loss"], 5),
            "delta_loss": fnum(strong["val_loss"] - full["val_loss"], 5),
            "full_acc": fnum(full["accuracy"], 5),
            "strong_acc": fnum(strong["accuracy"], 5),
            "delta_acc": fnum(strong["accuracy"] - full["accuracy"], 5),
            "active_qk": fnum(full.get("qk_active_count_mean"), 2),
            "active_v": fnum(full.get("v_active_count_mean"), 2),
            "active_rst": fnum(full.get("rst_active_count_mean"), 2),
            "strong_qk": fnum(full.get("qk_strong_count_mean"), 2),
            "strong_v": fnum(full.get("v_strong_count_mean"), 2),
            "strong_rst": fnum(full.get("rst_strong_count_mean"), 2),
            "strong_ratio_qk": fnum(full.get("qk_strong_frac_mean"), 5),
            "strong_ratio_v": fnum(full.get("v_strong_frac_mean"), 5),
            "strong_ratio_rst": fnum(full.get("rst_strong_frac_mean"), 5),
            "retained_gate_mass_qk": fnum(strong.get("qk_retained_gate_mass"), 4),
            "retained_gate_mass_v": fnum(strong.get("v_retained_gate_mass"), 4),
            "retained_gate_mass_rst": fnum(strong.get("rst_retained_gate_mass"), 4),
            "estimated_rw_flops_ratio_vs_full_pool": fnum(strong.get("estimated_rw_flops_ratio_vs_full_pool"), 5),
        })

    explore = []
    for name, variants in by_model.items():
        full = variants.get("full_soft")
        strong = variants.get("sub_hard_all_t090")
        if not full or not strong:
            continue
        ew = full.get("exploration_weight")
        ea = full.get("exploration_asymmetry")
        if ew in (None, "") and ea in (None, ""):
            continue
        explore.append({
            "model_name": name,
            "exploration_weight": ew,
            "exploration_asymmetry": ea,
            "val_loss": fnum(full["val_loss"], 5),
            "strong_loss": fnum(strong["val_loss"], 5),
            "active_qk": fnum(full.get("qk_active_count_mean"), 2),
            "active_v": fnum(full.get("v_active_count_mean"), 2),
            "active_rst": fnum(full.get("rst_active_count_mean"), 2),
            "strong_qk": fnum(full.get("qk_strong_count_mean"), 2),
            "strong_v": fnum(full.get("v_strong_count_mean"), 2),
            "strong_rst": fnum(full.get("rst_strong_count_mean"), 2),
            "estimated_rw_flops_ratio": fnum(strong.get("estimated_rw_flops_ratio_vs_full_pool"), 5),
        })

    flops_main = []
    for r in ok:
        is_main_model = (
            r.get("model_type") == "baseline_transformer"
            or r.get("role") == "main"
        )
        is_main_variant = (
            r.get("variant") in ("full_validation", "full_soft", "hard_all_t090")
        )
        if is_main_model and is_main_variant:
            flops_main.append(flops_table_row(r))

    flops_appendix = [flops_table_row(r) for r in ok]
    flops_tradeoff = [
        flops_table_row(r) for r in ok
        if r.get("model_type") == "dawn_srw"
    ]

    return {
        "table_main_performance.csv": main_perf,
        "table_dawn_hard_pruning_main.csv": hard_main,
        "table_dawn_threshold_sweep_appendix.csv": threshold_sweep,
        "table_dawn_sub_strong_compare.csv": sub_rows,
        "table_explore_sweep_appendix.csv": explore,
        "table_theoretical_flops_main.csv": flops_main,
        "table_theoretical_flops_appendix.csv": flops_appendix,
        "table_flops_vs_loss_tradeoff.csv": flops_tradeoff,
    }


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "_No rows._\n"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def csv_table_text(rows: Sequence[Dict[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    cols = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json_safe(row.get(k, "")) for k in cols})
    return buf.getvalue().strip()


def latex_escape(x: Any) -> str:
    s = str(x)
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def latex_table(name: str, rows: Sequence[Dict[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return f"% {name}: no rows\n"
    cols = list(rows[0].keys())
    spec = "l" * len(cols)
    out = [f"% {name}", f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    out.append(" & ".join(latex_escape(c) for c in cols) + " \\\\")
    out.append("\\hline")
    for row in rows:
        out.append(" & ".join(latex_escape(row.get(c, "")) for c in cols) + " \\\\")
    out.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(out)


def write_publication_files(output_dir: str, tables: Dict[str, List[Dict[str, Any]]],
                            no_markdown: bool, no_latex: bool) -> None:
    if not no_markdown:
        parts = ["# Paper Evaluation Tables\n", README_NOTE + "\n", FLOPS_NOTE + "\n"]
        for filename, rows in tables.items():
            parts.append(f"\n## {filename[:-4]}\n")
            parts.append(markdown_table(rows))
        write_text(join_path(output_dir, "paper_tables.md"), "\n".join(parts))
    if not no_latex:
        parts = [latex_table(filename[:-4], rows) for filename, rows in tables.items()]
        write_text(join_path(output_dir, "paper_tables.tex"), "\n\n".join(parts))


def write_readme(output_dir: str) -> None:
    text = (
        "DAWN-SRW paper evaluation output\n\n"
        f"{README_NOTE}\n\n"
        "FLOPs convention:\n"
        f"{FLOPS_NOTE}\n\n"
        "Files:\n"
        "- results_raw.jsonl / results_raw.csv: one row per model variant.\n"
        "- table_main_performance.csv: compact paper main-table candidates.\n"
        "- table_dawn_hard_pruning_main.csv: main DAWN full vs hard-pruned rows.\n"
        "- table_dawn_threshold_sweep_appendix.csv: threshold sweep appendix rows.\n"
        "- table_dawn_sub_strong_compare.csv: sub-checkpoint full vs strong-only comparison.\n"
        "- table_explore_sweep_appendix.csv: exploration sweep rows when config fields exist.\n"
        "- table_theoretical_flops_main.csv: main FLOPs accounting rows.\n"
        "- table_theoretical_flops_appendix.csv: all model/variant FLOPs accounting rows.\n"
        "- table_flops_vs_loss_tradeoff.csv: DAWN FLOPs/loss tradeoff rows.\n\n"
        "The final console summary also prints full copyable CSV blocks between "
        "COPYABLE_PAPER_DATA_CSV_BEGIN and COPYABLE_PAPER_DATA_CSV_END.\n"
    )
    write_text(join_path(output_dir, "README.txt"), text)


def save_outputs(output_dir: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    raw_csv = join_path(output_dir, "results_raw.csv")
    write_csv(raw_csv, rows)
    tables = summarize_tables(rows)
    for filename, table_rows in tables.items():
        columns = FLOPS_TABLE_COLUMNS if filename in (
            "table_theoretical_flops_main.csv",
            "table_theoretical_flops_appendix.csv",
            "table_flops_vs_loss_tradeoff.csv",
        ) else None
        write_csv(join_path(output_dir, filename), table_rows, columns=columns)
    write_publication_files(output_dir, tables, args.no_markdown, args.no_latex)
    write_readme(output_dir)


def print_console_summary(tables: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
    if not is_host0():
        return
    summary_files = (
        ("Baseline/Main Performance", "table_main_performance.csv"),
        ("DAWN Main Full vs Strong/Hard", "table_dawn_hard_pruning_main.csv"),
        ("DAWN Threshold Sweep", "table_dawn_threshold_sweep_appendix.csv"),
        ("DAWN Sub Strong Compare", "table_dawn_sub_strong_compare.csv"),
        ("Theoretical FLOPs Main", "table_theoretical_flops_main.csv"),
        ("FLOPs vs Loss Tradeoff", "table_flops_vs_loss_tradeoff.csv"),
    )
    for title, filename in summary_files:
        print(f"\n=== {title} ===")
        print(markdown_table(tables.get(filename, [])))

    print("\n=== COPYABLE_PAPER_DATA_CSV_BEGIN ===")
    for title, filename in summary_files:
        rows = tables.get(filename, [])
        print(f"\n--- {filename} ---")
        text = csv_table_text(rows)
        print(text if text else "# no rows")
    print("\n=== COPYABLE_PAPER_DATA_CSV_END ===")

    print("\nOutput paths:")
    for fn in (
            "results_raw.jsonl", "results_raw.csv",
            "table_main_performance.csv",
            "table_dawn_hard_pruning_main.csv",
            "table_dawn_threshold_sweep_appendix.csv",
            "table_dawn_sub_strong_compare.csv",
            "table_explore_sweep_appendix.csv",
            "table_theoretical_flops_main.csv",
            "table_theoretical_flops_appendix.csv",
            "table_flops_vs_loss_tradeoff.csv",
            "paper_tables.md", "paper_tables.tex",
            "run_manifest_resolved.yaml", "README.txt"):
        print(f"  {join_path(output_dir, fn)}")


def sanity_checks(rows: List[Dict[str, Any]]) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status", "ok") == "ok" and r.get("model_type") == "dawn_srw":
            by_model.setdefault(r["model_name"], []).append(r)
            for pool in ("qk", "v", "rst"):
                mass = r.get(f"{pool}_retained_gate_mass")
                if mass is not None and not (-1e-5 <= float(mass) <= 1.00001):
                    raise ValueError(f"{r['model_name']} {r['variant']} {pool} retained_gate_mass={mass}")
    for name, rs in by_model.items():
        by_variant = {r["variant"]: r for r in rs}
        for variant_name in ("hard_all_t050",):
            row = by_variant.get(variant_name)
            if row:
                for pool in ("qk", "v", "rst"):
                    kept = float(row.get(f"{pool}_kept_count_mean") or 0.0)
                    active = float(row.get(f"{pool}_active_count_mean") or 0.0)
                    if kept + 1e-3 < active * 0.95:
                        raise ValueError(
                            f"{name}: {variant_name} {pool} kept_count "
                            f"{kept} is unexpectedly below active_count {active}")
        for variant_name in ("hard_all_t090", "sub_hard_all_t090"):
            row = by_variant.get(variant_name)
            if row:
                for pool in ("qk", "v", "rst"):
                    kept = float(row.get(f"{pool}_kept_count_mean") or 0.0)
                    strong = float(row.get(f"{pool}_strong_count_mean") or 0.0)
                    if abs(kept - strong) > max(1.0, 0.05 * max(strong, 1.0)):
                        raise ValueError(
                            f"{name}: {variant_name} {pool} kept_count "
                            f"{kept} does not match strong_count {strong}")
        if "hard_all_t090" in by_variant and "hard_all_t099" in by_variant:
            k90 = by_variant["hard_all_t090"].get("estimated_executed_rw_ops_total_per_token_per_layer")
            k99 = by_variant["hard_all_t099"].get("estimated_executed_rw_ops_total_per_token_per_layer")
            if k90 is not None and k99 is not None and float(k99) > float(k90) + 1e-3:
                raise ValueError(f"{name}: hard_all_t099 kept more neurons than hard_all_t090")


def parse_config_explore_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("training", {})
    return {
        "exploration_weight": t.get("exploration_weight", ""),
        "exploration_asymmetry": t.get("exploration_asymmetry", ""),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="-1 or omitted manifest null means full validation")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--mesh_auto", action="store_true",
                    help="Use config mesh_model/mesh_data when available.")
    ap.add_argument("--mesh_data", type=int, default=None)
    ap.add_argument("--mesh_model", type=int, default=None)
    ap.add_argument("--single_device", action="store_true")
    ap.add_argument("--variants", default=None,
                    help=("Comma-separated DAWN variants to run. Baseline "
                          "full_validation is always kept. Example: "
                          "full_soft,hard_all_t090"))
    ap.add_argument("--no_latex", action="store_true")
    ap.add_argument("--no_markdown", action="store_true")
    ap.add_argument("--exclude_lm_head_flops", dest="include_lm_head_flops",
                    action="store_false",
                    help="Keep LM-head FLOPs separate but exclude them from *_with_lm_head totals.")
    ap.set_defaults(include_lm_head_flops=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume_existing", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--progress_every", type=int, default=25)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    run_cfg = resolve_run_config(manifest, args)
    output_dir = ensure_dir(run_cfg["output_dir"])
    raw_path = join_path(output_dir, "results_raw.jsonl")

    models = parse_models(manifest)
    resolved = {
        "manifest": args.manifest,
        "run": run_cfg,
        "models": models,
        "variants": {
            m["name"]: [v.__dict__ for v in selected_variants_for(m, args)]
            for m in models
        },
    }
    if is_host0():
        write_text(join_path(output_dir, "run_manifest_resolved.yaml"),
                   yaml.safe_dump(resolved, sort_keys=False))

    log("=== Paper evaluation plan ===")
    log(f"output_dir={output_dir}")
    log(f"val_bin={run_cfg['val_bin']} seq_len={run_cfg['seq_len']} batch_size={run_cfg['batch_size']} max_batches={run_cfg['max_batches']}")
    for m in models:
        log(
            f"  {m['family']} {m.get('role', '')}: {m['name']} "
            f"({len(selected_variants_for(m, args))} variants, "
            f"mesh_model={m.get('mesh_model', 'config/default')})")

    if args.dry_run:
        log("dry_run=True: stopping before checkpoint/model evaluation.")
        return
    if _JAX_IMPORT_ERROR is not None or _NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "JAX/Flax/NumPy dependencies are required for evaluation. "
            f"Original import errors: jax={_JAX_IMPORT_ERROR!r}, "
            f"numpy={_NUMPY_IMPORT_ERROR!r}")

    done = completed_keys(raw_path) if args.resume_existing else set()

    for m in models:
        try:
            cfg = load_config(m["config"])
            cfg.setdefault("model", {})
            cfg.setdefault("training", {})
            if m["family"] == "baseline":
                cfg["model"]["model_version"] = cfg["model"].get("model_version", "baseline")
            model_version = cfg["model"].get("model_version", "")
            is_baseline = m["family"] == "baseline" or model_version == "baseline"
            model = tj.build_model_from_config(cfg)
            target_params = init_target_params(model, cfg, run_cfg["seq_len"])
            params, ckpt_meta = restore_params(m["checkpoint"], target_params, model_version)

            mesh_data, mesh_model = infer_mesh_shape(cfg, is_baseline, args, m)
            mesh = create_mesh(mesh_data, mesh_model, args.single_device)
            data_sharding = NamedSharding(mesh, P("data", None))
            param_shardings = tj.get_param_shardings(params, mesh, is_baseline=is_baseline)
            params = tj.shard_params_to_mesh(params, param_shardings)
            if run_cfg["batch_size"] % max(1, mesh_data) != 0:
                raise ValueError(
                    f"batch_size={run_cfg['batch_size']} must be divisible "
                    f"by mesh_data={mesh_data} for {m['name']} on this slice")
            log(f"\n=== {m['name']} mesh=({mesh_data},{mesh_model}) params={count_params(params):,} ===")

            for variant in selected_variants_for(m, args):
                if (m["name"], variant.name) in done:
                    log(f"  skip existing: {m['name']} {variant.name}")
                    continue
                try:
                    sharded_fns = None
                    if m["family"] == "dawn_srw":
                        sharded_fns = build_sharded_fns(
                            cfg, mesh, mesh_model,
                            run_cfg["batch_size"], run_cfg["seq_len"], variant)
                    with mesh:
                        eval_out = evaluate_variant(
                            model, params, cfg, run_cfg["val_bin"],
                            run_cfg["batch_size"], run_cfg["seq_len"],
                            run_cfg["max_batches"], run_cfg["local_cache_dir"],
                            mesh, data_sharding, sharded_fns,
                            variant, max(1, args.progress_every))
                    row = build_row(m, cfg, params, ckpt_meta, variant, eval_out, run_cfg, args)
                    row.update(parse_config_explore_fields(cfg))
                    append_jsonl(raw_path, row)
                    rows = load_all_rows(raw_path)
                    save_outputs(output_dir, rows, args)
                    log(
                        f"  result {m['name']} {variant.name}: "
                        f"loss={row['val_loss']:.5f} ppl={row['ppl']:.3f} "
                        f"acc={row['accuracy']:.5f} tokens={row['valid_tokens']}")
                except Exception as exc:
                    err = {
                        "status": "error",
                        "model_name": m.get("name"),
                        "model_type": "dawn_srw" if m.get("family") == "dawn_srw" else "baseline_transformer",
                        "role": m.get("role", ""),
                        "variant": variant.name,
                        "checkpoint": m.get("checkpoint"),
                        "config": m.get("config"),
                        "error": repr(exc),
                    }
                    append_jsonl(raw_path, err)
                    log(f"  ERROR {m.get('name')} {variant.name}: {exc!r}")
                    if args.fail_fast:
                        raise
        except Exception as exc:
            log(f"ERROR loading model {m.get('name')}: {exc!r}")
            if args.fail_fast:
                raise
            for variant in selected_variants_for(m, args):
                append_jsonl(raw_path, {
                    "status": "error",
                    "model_name": m.get("name"),
                    "model_type": "dawn_srw" if m.get("family") == "dawn_srw" else "baseline_transformer",
                    "role": m.get("role", ""),
                    "variant": variant.name,
                    "checkpoint": m.get("checkpoint"),
                    "config": m.get("config"),
                    "error": repr(exc),
                })

    rows = load_all_rows(raw_path)
    sanity_checks(rows)
    write_jsonl(raw_path, rows)
    save_outputs(output_dir, rows, args)
    print_console_summary(summarize_tables(rows), output_dir)


if __name__ == "__main__":
    main()

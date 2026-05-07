#!/usr/bin/env python3
"""Paper validation evaluator for DAWN-SRW and dense Transformer checkpoints.

This script is intentionally small and boring: it evaluates one checkpoint on a
tokenized validation .bin, then writes model_info.json/model_table.csv and
validation.json/validation_table.csv.  It supports local and gs:// paths and can
run either on a single device or through the same JAX mesh/shard_map path used by
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAWN_ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "analysis" / "dawn_srw"
for p in (PROJECT_ROOT, DAWN_ANALYSIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dawn_srw_common import (
    count_params,
    is_gcs,
    load_config,
    load_val_tokens,
    open_file,
    save_json,
    select_checkpoint,
)
import scripts.train_jax as tj
from scripts.downstream_finetune_jax import _adapt_checkpoint_params_to_target


DAWN_SRW_VERSIONS = {
    "dawn_srw",
    "spatial-r1-v4.1.5.5",
    "spatial-r1-v4.1.5.2",
}


def is_host0() -> bool:
    return int(jax.process_index()) == 0


def log(msg: str) -> None:
    if is_host0():
        print(msg, flush=True)


def join_path(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name if is_gcs(base) else str(Path(base) / name)


def write_csv(rows, output_dir: str, filename: str) -> str:
    path = join_path(output_dir, filename)
    if not rows:
        rows = [{}]
    fieldnames = list(rows[0].keys())
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def create_mesh(mesh_data: int, mesh_model: int, *, single_device: bool) -> Mesh:
    devices = jax.devices()
    if single_device:
        devices = devices[:1]
        mesh_data, mesh_model = 1, 1
    needed = mesh_data * mesh_model
    if needed != len(devices):
        raise ValueError(
            f"mesh_data({mesh_data}) * mesh_model({mesh_model}) = {needed}, "
            f"but selected JAX devices = {len(devices)}. "
            "Pass --mesh-data/--mesh-model or --single-device."
        )
    return Mesh(np.asarray(devices).reshape(mesh_data, mesh_model), ("data", "model"))


def infer_mesh_shape(cfg: Dict[str, Any], args: argparse.Namespace, is_baseline: bool) -> Tuple[int, int]:
    if args.single_device:
        return 1, 1
    n_devices = int(jax.device_count())
    if args.mesh_model is not None:
        mesh_model = int(args.mesh_model)
    elif is_baseline:
        mesh_model = 1
    else:
        mesh_model = int(cfg.get("training", {}).get("mesh_model", 1) or 1)
    if mesh_model < 1:
        mesh_model = 1
    if args.mesh_data is not None:
        mesh_data = int(args.mesh_data)
    else:
        cfg_mesh_data = int(cfg.get("training", {}).get("mesh_data", 0) or 0)
        mesh_data = cfg_mesh_data if cfg_mesh_data > 0 and not is_baseline else max(1, n_devices // mesh_model)
    if mesh_data * mesh_model != n_devices:
        raise ValueError(
            f"mesh_data({mesh_data}) * mesh_model({mesh_model}) must equal "
            f"jax.device_count()({n_devices})."
        )
    return mesh_data, mesh_model


def chunk_sizes(cfg: Dict[str, Any], mesh_model: int, batch_size: int, seq_len: int) -> Tuple[int, int, int]:
    m = cfg["model"]
    t = cfg.get("training", {})
    n_rst = int(m.get("n_rst", m.get("n_know", 25200)))
    n_qk = int(m.get("n_qk", 1580))
    n_v = int(m.get("n_v", 2600))
    for name, n in (("n_qk", n_qk), ("n_v", n_v), ("n_rst", n_rst)):
        if n % mesh_model != 0:
            raise ValueError(f"{name}={n} must be divisible by mesh_model={mesh_model}")
    per_device_batch = max(1, batch_size // max(1, jax.device_count()))
    target_gb = float(t.get("target_chunk_gb", 2.0))

    def auto_chunks(n_local: int) -> int:
        full_gb = per_device_batch * seq_len * n_local * 2 / 1.0e9
        nc = max(1, int(math.ceil(full_gb / target_gb)))
        while n_local % nc != 0 and nc < n_local:
            nc += 1
        return min(nc, n_local)

    def max_chunk(name: str, n_local: int, n_chunks: int) -> int:
        if n_chunks < 1:
            raise ValueError(f"{name} chunks must be >= 1")
        if n_chunks > n_local:
            raise ValueError(f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
        return max(1, int(math.ceil(n_local / n_chunks)))

    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model
    nrst_local = n_rst // mesh_model
    n_chunks_qk = int(t.get("n_chunks_qk", auto_chunks(nqk_local)))
    n_chunks_v = int(t.get("n_chunks_v", auto_chunks(nv_local)))
    n_chunks_rst = int(t.get("n_chunks_rst", t.get("n_chunks_know", auto_chunks(nrst_local))))
    if t.get("max_chunk_size") is not None:
        forced = int(t["max_chunk_size"])
        return forced, forced, forced
    return (
        max_chunk("attn_qk", nqk_local, n_chunks_qk),
        max_chunk("attn_v", nv_local, n_chunks_v),
        max_chunk("rst", nrst_local, n_chunks_rst),
    )


def build_sharded_fns_if_needed(
    cfg: Dict[str, Any],
    mesh: Optional[Mesh],
    mesh_model: int,
    batch_size: int,
    seq_len: int,
):
    if mesh is None:
        return None
    version = cfg.get("model", {}).get("model_version", "dawn_srw")
    spec = tj.MODEL_REGISTRY.get(version)
    if spec is None or not spec.supports_sharded:
        return None
    if mesh_model <= 1 and not getattr(spec, "force_sharded", False):
        return None

    module = __import__(spec.module_path, fromlist=["make_sharded_srw"])
    make_sharded_srw = module.make_sharded_srw
    qk_chunk, v_chunk, rst_chunk = chunk_sizes(cfg, mesh_model, batch_size, seq_len)
    base_kwargs = {"mesh": mesh}
    if spec.sharded_kwargs is not None:
        base_kwargs.update(spec.sharded_kwargs(cfg))
    single_v = make_sharded_srw(max_chunk_size=v_chunk, **base_kwargs)
    single_rst = make_sharded_srw(max_chunk_size=rst_chunk, **base_kwargs)
    if hasattr(module, "make_sharded_srw_paired"):
        paired = module.make_sharded_srw_paired(max_chunk_size=qk_chunk, **base_kwargs)
        log(
            "  shard_map enabled: "
            f"mesh_model={mesh_model}, chunks attn_qk/attn_v/rst={qk_chunk}/{v_chunk}/{rst_chunk}"
        )
        return {
            "single": single_v,
            "attn_v_single": single_v,
            "rst_single": single_rst,
            "paired": paired,
            "attn_qk_paired": paired,
        }
    log(f"  shard_map enabled: mesh_model={mesh_model}, chunk={max(v_chunk, rst_chunk)}")
    return (single_v, single_rst)


def init_target_params(model, cfg: Dict[str, Any]):
    seed = int(cfg.get("seed", 1))
    max_seq_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    dummy_len = min(max_seq_len, 32)
    dummy = jnp.ones((1, dummy_len), dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(seed), "dropout": jax.random.PRNGKey(seed + 1)},
        dummy,
        labels=dummy,
        attention_mask=jnp.ones_like(dummy),
        deterministic=True,
    )
    return variables["params"]


def restore_params(checkpoint: str, target_params) -> Tuple[Any, Dict[str, Any]]:
    ckpt_file = select_checkpoint(checkpoint)
    log(f"Loading checkpoint: {ckpt_file}")
    with open_file(ckpt_file, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    raw_params = raw.get("params", raw) if isinstance(raw, dict) else raw
    raw_params = _adapt_checkpoint_params_to_target(raw_params, target_params)
    try:
        restored = serialization.from_state_dict({"params": target_params}, {"params": raw_params})["params"]
    except ValueError as exc:
        raw_keys = sorted(raw_params.keys())[:30] if isinstance(raw_params, dict) else type(raw_params)
        tgt_keys = sorted(target_params.keys())[:30] if isinstance(target_params, dict) else type(target_params)
        raise ValueError(f"Parameter restore failed. target keys={tgt_keys}, raw keys={raw_keys}") from exc
    meta = {
        "checkpoint_file": ckpt_file,
        "step": int(raw.get("step", 0)) if isinstance(raw, dict) else 0,
        "epoch": int(raw.get("epoch", 0)) if isinstance(raw, dict) else 0,
        "best_val_loss": float(raw.get("best_val_loss", float("nan"))) if isinstance(raw, dict) else float("nan"),
    }
    return restored, meta


def host_local_batch(global_batch: np.ndarray) -> np.ndarray:
    n_hosts = int(jax.process_count())
    host = int(jax.process_index())
    if global_batch.shape[0] % n_hosts != 0:
        raise ValueError(
            f"batch_size={global_batch.shape[0]} must be divisible by JAX process_count={n_hosts}"
        )
    per_host = global_batch.shape[0] // n_hosts
    return global_batch[host * per_host:(host + 1) * per_host]


def shard_global_batch(global_batch: np.ndarray, sharding, global_shape):
    """Build a global batch array from a fully replicated host copy.

    The training helper slices by host.  For paper eval every host reads the
    same validation tokens, which is more robust for model-parallel meshes where
    the data axis can be smaller than process_count.
    """

    def callback(index):
        return np.asarray(global_batch[index])

    return jax.make_array_from_callback(global_shape, sharding, callback)


def load_validation_sequences(val_data: str, seq_len: int, max_tokens: int, batch_size: int, max_batches: int) -> np.ndarray:
    tokens = load_val_tokens(val_data, max_tokens=max_tokens if max_tokens > 0 else None)
    n_seqs = int(tokens.shape[0]) // seq_len
    if max_batches > 0:
        n_seqs = min(n_seqs, max_batches * batch_size)
    n_seqs = (n_seqs // batch_size) * batch_size
    if n_seqs <= 0:
        raise ValueError(f"Not enough validation tokens for seq_len={seq_len}, batch_size={batch_size}")
    return np.asarray(tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len), dtype=np.int32)


def summarize_model(model_name: str, checkpoint: str, checkpoint_meta: Dict[str, Any], cfg: Dict[str, Any], params) -> Dict[str, Any]:
    m = cfg.get("model", {})
    model_version = m.get("model_version", "dawn_srw")
    n_params = count_params(params)
    return {
        "model_name": model_name,
        "model_version": model_version,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_meta.get("checkpoint_file", ""),
        "checkpoint_step": checkpoint_meta.get("step", 0),
        "params": n_params,
        "params_M": n_params / 1.0e6,
        "d_model": m.get("d_model", ""),
        "layers": m.get("n_layers", ""),
        "heads": m.get("n_heads", ""),
        "d_ff": m.get("d_ff", ""),
        "d_route": m.get("d_route", m.get("d_bottleneck", "")),
        "n_qk": m.get("n_qk", ""),
        "n_v": m.get("n_v", ""),
        "n_rst": m.get("n_rst", m.get("n_know", "")),
        "max_seq_len": m.get("max_seq_len", ""),
    }


def evaluate(
    model,
    params,
    sequences: np.ndarray,
    batch_size: int,
    seq_len: int,
    mesh: Optional[Mesh],
    data_sharding,
    sharded_fns,
    *,
    single_device: bool,
) -> Dict[str, Any]:
    eval_step = tj.create_eval_step(model, sharded_fns=sharded_fns)
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0.0
    n_batches = sequences.shape[0] // batch_size
    t0 = time.time()
    for b in range(n_batches):
        start = b * batch_size
        batch_np = sequences[start:start + batch_size]
        mask_np = (batch_np != 0).astype(np.int32)
        if mesh is not None:
            if single_device:
                batch = jax.device_put(batch_np, data_sharding)
                mask = jax.device_put(mask_np, data_sharding)
            else:
                batch = shard_global_batch(batch_np, data_sharding, (batch_size, seq_len))
                mask = shard_global_batch(mask_np, data_sharding, (batch_size, seq_len))
        else:
            batch = jnp.asarray(batch_np)
            mask = jnp.asarray(mask_np)
        loss, correct, valid = eval_step(params, batch, mask)
        loss_f = float(jax.device_get(loss))
        correct_f = float(jax.device_get(correct))
        valid_f = float(jax.device_get(valid))
        total_loss += loss_f * valid_f
        total_correct += correct_f
        total_valid += valid_f
        if is_host0() and (b + 1 == n_batches or (b + 1) % max(1, n_batches // 10) == 0):
            print(f"  eval progress: {b + 1}/{n_batches} batches", flush=True)
    elapsed = time.time() - t0
    val_loss = total_loss / max(total_valid, 1.0)
    return {
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss) if val_loss < 100 else float("inf"),
        "accuracy": total_correct / max(total_valid, 1.0),
        "accuracy_pct": 100.0 * total_correct / max(total_valid, 1.0),
        "total_valid_tokens": int(total_valid),
        "n_sequences": int(sequences.shape[0]),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "time_sec": elapsed,
        "tokens_per_sec": total_valid / elapsed if elapsed > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--max-val-tokens", type=int, default=262144)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=0, help="Defaults to config model.max_seq_len.")
    ap.add_argument("--max-batches", type=int, default=0, help="0 means all batches available under max-val-tokens.")
    ap.add_argument("--mesh-data", type=int, default=None)
    ap.add_argument("--mesh-model", type=int, default=None)
    ap.add_argument("--single-device", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_device and jax.process_count() > 1 and not is_host0():
        print("single-device mode: non-primary host exits without work", flush=True)
        return

    cfg = load_config(args.config)
    model_version = cfg.get("model", {}).get("model_version", "dawn_srw")
    is_baseline = model_version == "baseline"
    seq_len = int(args.seq_len or cfg.get("model", {}).get("max_seq_len", 512))
    log("=== Paper JAX validation ===")
    log(f"model_name={args.model_name} version={model_version}")
    log(f"devices={jax.device_count()} local_devices={jax.local_device_count()} hosts={jax.process_count()}")

    model = tj.build_model_from_config(cfg)
    target_params = init_target_params(model, cfg)
    params, checkpoint_meta = restore_params(args.checkpoint, target_params)
    model_info = summarize_model(args.model_name, args.checkpoint, checkpoint_meta, cfg, params)

    mesh = None
    data_sharding = None
    sharded_fns = None
    use_mesh = args.single_device or jax.device_count() > 1 or model_version in DAWN_SRW_VERSIONS
    mesh_data, mesh_model = (1, 1)
    if use_mesh:
        mesh_data, mesh_model = infer_mesh_shape(cfg, args, is_baseline)
        mesh = create_mesh(mesh_data, mesh_model, single_device=args.single_device)
        data_sharding = NamedSharding(mesh, P("data", None))
        sharded_fns = build_sharded_fns_if_needed(cfg, mesh, mesh_model, args.batch_size, seq_len)
        param_shardings = tj.get_param_shardings(params, mesh, is_baseline=is_baseline)
        params = tj.shard_params_to_mesh(params, param_shardings)
        log(f"mesh=({mesh_data}, {mesh_model}) single_device={args.single_device}")
    else:
        params = jax.tree.map(jnp.asarray, params)

    sequences = load_validation_sequences(args.val_data, seq_len, args.max_val_tokens, args.batch_size, args.max_batches)
    log(f"validation sequences={sequences.shape[0]} batch_size={args.batch_size} seq_len={seq_len}")
    if mesh is not None:
        with mesh:
            validation = evaluate(
                model, params, sequences, args.batch_size, seq_len,
                mesh, data_sharding, sharded_fns, single_device=args.single_device,
            )
    else:
        validation = evaluate(
            model, params, sequences, args.batch_size, seq_len,
            None, None, None, single_device=args.single_device,
        )
    validation.update({
        "model_name": args.model_name,
        "model_version": model_version,
        "checkpoint": args.checkpoint,
        "checkpoint_file": checkpoint_meta.get("checkpoint_file", ""),
        "mesh_data": mesh_data,
        "mesh_model": mesh_model,
        "jax_device_count": int(jax.device_count()),
        "jax_process_count": int(jax.process_count()),
    })

    if is_host0():
        save_json(model_info, args.output, "model_info.json")
        save_json(validation, args.output, "validation.json")
        write_csv([model_info], args.output, "model_table.csv")
        write_csv([
            {
                "model_name": validation["model_name"],
                "checkpoint": validation["checkpoint"],
                "val_loss": validation["val_loss"],
                "perplexity": validation["perplexity"],
                "accuracy": validation["accuracy"],
                "accuracy_pct": validation["accuracy_pct"],
                "total_valid_tokens": validation["total_valid_tokens"],
                "seq_len": validation["seq_len"],
                "batch_size": validation["batch_size"],
            }
        ], args.output, "validation_table.csv")
        log(
            f"result: loss={validation['val_loss']:.5f} "
            f"ppl={validation['perplexity']:.3f} acc={validation['accuracy_pct']:.2f}% "
            f"tokens={validation['total_valid_tokens']}"
        )


if __name__ == "__main__":
    main()

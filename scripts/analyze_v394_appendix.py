#!/usr/bin/env python3
"""One-shot paper-output analysis for the 400M spatial-r1-v3.9.4 pilot.

Inputs:
  --tf-checkpoint       dense Transformer checkpoint file or run directory
  --dawn394-checkpoint  DAWN spatial-r1-v3.9.4 checkpoint file or run directory
  --results             directory containing baseline.txt, baseline_random.txt,
                        dawn.txt, dawn_random.txt

Default behavior:
  - Parse downstream logs.
  - Resolve/load checkpoint bytes and report checkpoint metadata/param counts.
  - Print copyable paper tables and appendix wording to stdout.
  - Do not write CSV/TeX/Markdown files.

Optional:
  --run-forward runs a small validation batch from the checkpoints and prints
  validation/utilization diagnostics. This is intentionally opt-in because 400M
  checkpoints are large and forward analysis should run on a capable JAX host.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DAWN_CONFIG = "configs/legacy/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32.yaml"
DEFAULT_TF_CONFIG = "configs/downstream/baseline/sst2.yaml"

DOWNSTREAM_LOGS = (
    "baseline.txt",
    "baseline_random.txt",
    "dawn.txt",
    "dawn_random.txt",
)

LIMITATION_TEXT = (
    "v3.9.4 is an earlier SRW-family implementation. We include it as evidence "
    "that SRW-style rank-1 operator composition can be trained at 400M scale "
    "and can produce transferable representations. We do not use this experiment "
    "as the main evidence for the final v4.1.5.5 sparse operator-selection mechanism."
)

UTIL_NOTE = (
    "Because v3.9.4 uses an earlier ReLU/gate-sum-normalized router, these "
    "utilization statistics are not used as the primary sparsity evidence for DAWN-SRW."
)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def pct(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}"


def pp(value: Optional[float]) -> str:
    if value is None:
        return "--"
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.2f}"


def sci(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3e}"


def num(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "--"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def task_label(task: str) -> str:
    return {
        "sst2": "SST-2",
        "rte": "RTE",
        "wic": "WiC",
        "boolq": "BoolQ",
        "mnli": "MNLI",
    }.get(task.lower(), task.upper())


def section(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def is_primary_host() -> bool:
    """Return True only on JAX host 0, falling back to True off TPU/JAX."""
    try:
        import jax

        return int(jax.process_index()) == 0
    except Exception:
        return True


def emit(*args: Any, **kwargs: Any) -> None:
    """Print paper-ready output once in multi-host TPU runs."""
    if is_primary_host():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config and parameter estimates
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def baseline_param_estimate(model_cfg: Dict[str, Any]) -> int:
    vocab = int(model_cfg.get("vocab_size", 30522))
    d = int(model_cfg.get("d_model", 1280))
    d_ff = int(model_cfg.get("d_ff", 4 * d))
    layers = int(model_cfg.get("n_layers", 18))
    seq = int(model_cfg.get("max_seq_len", 512))
    embeddings = vocab * d + seq * d
    attn = 4 * d * d + 3 * d
    ffn = 2 * d * d_ff + d_ff + d
    norms = 4 * d
    final_norm = 2 * d
    return embeddings + layers * (attn + ffn + norms) + final_norm


def dawn394_param_estimate(model_cfg: Dict[str, Any]) -> int:
    vocab = int(model_cfg.get("vocab_size", 30522))
    d = int(model_cfg.get("d_model", 1280))
    route = int(model_cfg.get("d_route", 256))
    layers = int(model_cfg.get("n_layers", 18))
    seq = int(model_cfg.get("max_seq_len", 512))
    n_qk = int(model_cfg.get("n_qk", 6400))
    n_v = int(model_cfg.get("n_v", 10400))
    n_know = int(model_cfg.get("n_know", 100000))
    embeddings = vocab * d + seq * d
    pool = (n_qk + n_v + n_know) * (route + 2 * d)
    router = d * (3 * route) + (3 * route) + d * route + route + d * 3 + 3 + d + 1
    blocks = layers * (d * d + 4 * d)
    final_norm = 2 * d
    return embeddings + pool + router + blocks + final_norm


def token_budget(cfg: Dict[str, Any]) -> Optional[int]:
    value = cfg.get("data", {}).get("max_train_tokens")
    try:
        return int(value)
    except Exception:
        return None


def infer_dataset_and_budget(cfg: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    text = " ".join(
        str(part)
        for part in (
            cfg.get("data", {}).get("bin_train", ""),
            meta.get("_resolved", ""),
            meta.get("checkpoint_file", ""),
        )
    ).lower()
    dataset = "C4" if "c4" in text else "unknown"
    budget = token_budget(cfg)
    if budget is None:
        match = re.search(r"c4[_-](\d+(?:\.\d+)?)([bmk])\b", text)
        if match:
            value = float(match.group(1))
            scale = {"b": 1_000_000_000, "m": 1_000_000, "k": 1_000}[match.group(2)]
            budget = int(value * scale)
    return dataset, budget


def total_steps_estimate(cfg: Dict[str, Any]) -> Optional[int]:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    train = cfg.get("training", {})
    try:
        max_tokens = int(data["max_train_tokens"])
        seq = int(model.get("max_seq_len", 512))
        batch = int(train["batch_size"])
        accum = int(train.get("gradient_accumulation_steps", 1))
        epochs = int(train.get("num_epochs", 1))
    except Exception:
        return None
    sequences = max_tokens // seq
    micro_steps = sequences // batch
    return (micro_steps // accum) * epochs


# ---------------------------------------------------------------------------
# Downstream parser
# ---------------------------------------------------------------------------


HEADER_RE = re.compile(r"Downstream fine-tune:\s+model=(?P<model>\S+)\s+task=(?P<task>\S+)")
ROWS_RE = re.compile(r"train_rows=(?P<train>\d+)\s+eval_rows=(?P<eval>\d+)\s+total_steps=(?P<steps>\d+)")
EVAL_RE = re.compile(r"\[eval\]\s+step=(?P<step>\d+)\s+(?:acc|accuracy)=(?P<acc>[0-9.]+)\s+total=(?P<total>\d+)")
TRAIN_TIME_RE = re.compile(r"\[train\]\s+step=(?P<step>\d+)/(?:\d+).*?\btime=(?P<time>[0-9.]+)s")


@dataclass
class EvalPoint:
    step: int
    acc: float
    total: int


def normalize_path(path: str) -> str:
    """Accept both gs://bucket/key and bucket/key for the DAWN GCS bucket."""
    p = str(path)
    if p.startswith("gs://"):
        return p
    if p.startswith("dawn-tpu-data-c4/"):
        return "gs://" + p
    return p


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def path_name(path: str) -> str:
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def gcs_open(path: str, mode: str = "r"):
    """Open a GCS file without importing project training code."""
    try:
        import gcsfs

        return gcsfs.GCSFileSystem().open(path, mode)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.io.gfile.GFile(path, mode)
    except ImportError as exc:
        raise ImportError("GCS paths require gcsfs or tensorflow installed.") from exc


def file_exists(path: str) -> bool:
    path = normalize_path(path)
    if is_gcs_path(path):
        try:
            import gcsfs

            return bool(gcsfs.GCSFileSystem().exists(path))
        except ImportError:
            pass
        try:
            import tensorflow as tf

            return bool(tf.io.gfile.exists(path))
        except ImportError as exc:
            raise ImportError("GCS paths require gcsfs or tensorflow installed.") from exc
    return Path(path).exists()


def read_text(path: str) -> str:
    path = normalize_path(path)
    if is_gcs_path(path):
        with gcs_open(path, "r") as f:
            return f.read()
    return Path(path).read_text(errors="replace")


def join_ref(base: str, name: str) -> str:
    base = normalize_path(base).rstrip("/")
    return f"{base}/{name}"


def model_label(raw_model: str, source_name: str) -> str:
    name = source_name.lower()
    raw = raw_model.lower()
    if "baseline" in name or "baseline" in raw:
        return "dense Transformer 400M"
    return "DAWN v3.9.4 400M"


def init_type(source_name: str, init_from: str) -> str:
    if "random" in source_name.lower():
        return "random"
    if not init_from or init_from == "<none>":
        return "random"
    return "pretrained"


def finish_run(rows: List[Dict[str, Any]], current: Dict[str, Any], evals: List[EvalPoint]) -> None:
    if not current:
        return
    best = max(evals, key=lambda e: e.acc) if evals else None
    final = evals[-1] if evals else None
    step0 = next((e for e in evals if e.step == 0), None)
    rows.append(
        {
            **current,
            "step0_acc": step0.acc if step0 else None,
            "final_acc": final.acc if final else None,
            "best_acc": best.acc if best else None,
            "best_step": best.step if best else None,
        }
    )


def parse_downstream_log(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    evals: List[EvalPoint] = []
    init_from = ""
    source_name = path_name(path)
    for line in read_text(path).splitlines():
        line = line.strip()
        if match := HEADER_RE.search(line):
            finish_run(rows, current, evals)
            init_from = ""
            evals = []
            current = {
                "task": match.group("task").lower(),
                "model": model_label(match.group("model"), source_name),
                "source_log": source_name,
            }
            continue
        if not current:
            continue
        if match := ROWS_RE.search(line):
            current["train_rows"] = int(match.group("train"))
            current["eval_rows"] = int(match.group("eval"))
            current["total_steps"] = int(match.group("steps"))
            continue
        if line.startswith("init_from="):
            init_from = line.split("=", 1)[1].strip()
            current["init_type"] = init_type(source_name, init_from)
            continue
        if line.startswith("run_dir="):
            current["run_dir"] = line.split("=", 1)[1].strip()
            continue
        if match := EVAL_RE.search(line):
            evals.append(EvalPoint(int(match.group("step")), float(match.group("acc")), int(match.group("total"))))
            continue
        if match := TRAIN_TIME_RE.search(line):
            current["wall_clock_final_s"] = float(match.group("time"))
    finish_run(rows, current, evals)
    for row in rows:
        row.setdefault("init_type", init_type(source_name, init_from))
    return rows


def parse_downstream_results(results_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    results_dir = normalize_path(results_dir)
    for name in DOWNSTREAM_LOGS:
        path = join_ref(results_dir, name)
        if file_exists(path):
            rows.extend(parse_downstream_log(path))
    if not rows:
        raise FileNotFoundError(f"No downstream logs found in {results_dir}; expected {', '.join(DOWNSTREAM_LOGS)}")
    return sorted(rows, key=lambda r: (r["task"], r["model"], r["init_type"]))


def downstream_lookup(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("best_acc") is not None:
            out[(row["task"], row["model"], row["init_type"])] = float(row["best_acc"])
    return out


def main_downstream_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = downstream_lookup(rows)
    task_order = ["sst2", "rte", "wic", "boolq", "mnli"]
    present = {row["task"] for row in rows}
    tasks = [t for t in task_order if t in present] + sorted(present.difference(task_order))
    out = []
    dawn = "DAWN v3.9.4 400M"
    tf = "dense Transformer 400M"
    for task in tasks:
        d_pre = lookup.get((task, dawn, "pretrained"))
        d_rand = lookup.get((task, dawn, "random"))
        t_pre = lookup.get((task, tf, "pretrained"))
        t_rand = lookup.get((task, tf, "random"))
        out.append(
            {
                "task": task,
                "dawn_pre": d_pre,
                "dawn_random": d_rand,
                "dawn_delta": None if d_pre is None or d_rand is None else d_pre - d_rand,
                "tf_pre": t_pre,
                "tf_random": t_rand,
                "tf_delta": None if t_pre is None or t_rand is None else t_pre - t_rand,
                "dawn_minus_tf": None if d_pre is None or t_pre is None else d_pre - t_pre,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Checkpoint metadata and optional forward analysis
# ---------------------------------------------------------------------------


def resolve_checkpoint(path: str) -> str:
    import scripts.train_jax as tj

    p = normalize_path(str(path))
    if p.endswith(".flax"):
        return p
    best = p.rstrip("/") + "/best_model.flax"
    if tj._file_exists(best):
        return best
    step_files = tj._list_files(p, "checkpoint_step*.flax")
    if step_files:
        def step_num(item: str) -> int:
            match = re.search(r"checkpoint_step(\d+)\.flax", item)
            return int(match.group(1)) if match else -1
        return sorted(step_files, key=step_num)[-1]
    any_files = tj._list_files(p, "*.flax")
    if any_files:
        return any_files[-1]
    raise FileNotFoundError(f"No .flax checkpoint found in {path}")


def raw_param_count(params: Any) -> int:
    if isinstance(params, dict):
        return sum(raw_param_count(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(raw_param_count(v) for v in params)
    if hasattr(params, "shape"):
        return int(math.prod(params.shape))
    return 0


def load_checkpoint_metadata(label: str, checkpoint: str, estimated_params: int, show_paths: bool) -> Dict[str, Any]:
    try:
        from flax import serialization
        import scripts.train_jax as tj

        resolved = resolve_checkpoint(normalize_path(checkpoint))
        with tj._open_file(resolved, "rb") as f:
            raw = serialization.msgpack_restore(f.read())
        params = raw.get("params", {}) if isinstance(raw, dict) else {}
        count = raw_param_count(params) or estimated_params
        return {
            "model": label,
            "loaded": True,
            "checkpoint_file": resolved if show_paths else "<hidden>",
            "step": raw.get("step", "") if isinstance(raw, dict) else "",
            "epoch": raw.get("epoch", "") if isinstance(raw, dict) else "",
            "best_val_loss": raw.get("best_val_loss", "") if isinstance(raw, dict) else "",
            "parameter_count": count,
            "_raw": raw,
            "_resolved": resolved,
        }
    except Exception as exc:
        return {
            "model": label,
            "loaded": False,
            "checkpoint_file": "<failed>",
            "step": "",
            "epoch": "",
            "best_val_loss": "",
            "parameter_count": estimated_params,
            "error": f"{type(exc).__name__}: {exc}",
        }


def read_validation_batch(path: str, batch_size: int, seq_len: int) -> Any:
    import numpy as np
    import scripts.train_jax as tj

    need = batch_size * seq_len
    path = normalize_path(path)
    if path.startswith("gs://"):
        with tj._open_file(path, "rb") as f:
            data = f.read(need * np.dtype(np.uint16).itemsize)
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.int32)
    else:
        arr = np.memmap(path, dtype=np.uint16, mode="r")[:need].astype(np.int32)
    if arr.size < need:
        raise ValueError(f"validation data has {arr.size} tokens, need {need}: {path}")
    return np.asarray(arr.reshape(batch_size, seq_len), dtype=np.int32)


def restore_params_for_forward(raw: Dict[str, Any], target_params: Dict[str, Any]) -> Dict[str, Any]:
    from flax import serialization
    from scripts.downstream_finetune_jax import _adapt_checkpoint_params_to_target

    raw_params = raw.get("params", {})
    raw_params = _adapt_checkpoint_params_to_target(raw_params, target_params)
    return serialization.from_state_dict({"params": target_params}, {"params": raw_params})["params"]


def infer_mesh_shape(cfg: Dict[str, Any], args: argparse.Namespace, is_dawn: bool) -> Tuple[int, int]:
    import jax

    n_devices = jax.device_count()
    if n_devices <= 1:
        return 1, 1

    if args.mesh_model is not None:
        mesh_model = int(args.mesh_model)
    else:
        mesh_model = int(cfg.get("training", {}).get("mesh_model", 1) or 1)
        if not is_dawn:
            mesh_model = 1

    if args.mesh_data is not None:
        mesh_data = int(args.mesh_data)
    else:
        cfg_mesh_data = int(cfg.get("training", {}).get("mesh_data", 0) or 0)
        mesh_data = cfg_mesh_data if cfg_mesh_data > 0 and is_dawn else max(1, n_devices // mesh_model)

    if mesh_data * mesh_model != n_devices:
        raise ValueError(
            f"mesh_data({mesh_data}) * mesh_model({mesh_model}) must equal "
            f"jax.device_count()({n_devices}). Pass --mesh-data/--mesh-model explicitly."
        )
    return mesh_data, mesh_model


def host_local_batch(global_batch: Any) -> Any:
    import jax

    n_hosts = jax.process_count()
    host_id = jax.process_index()
    if global_batch.shape[0] % n_hosts != 0:
        raise ValueError(
            f"--batch-size must be divisible by the number of hosts. "
            f"batch_size={global_batch.shape[0]}, hosts={n_hosts}"
        )
    per_host = global_batch.shape[0] // n_hosts
    return global_batch[host_id * per_host:(host_id + 1) * per_host]


def run_forward_analysis(
    label: str,
    checkpoint_meta: Dict[str, Any],
    cfg: Dict[str, Any],
    val_data: str,
    batch_size: int,
    seq_len: int,
    is_dawn: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not checkpoint_meta.get("loaded"):
        return {"forward_error": "checkpoint not loaded"}

    import jax
    import jax.numpy as jnp
    import scripts.train_jax as tj

    model = tj.build_model_from_config(cfg)
    seed = int(cfg.get("seed", 1))
    init_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    global_batch_np = read_validation_batch(val_data, batch_size, seq_len)

    sharded_fns = None
    data_sharding = None
    mesh = None
    use_mesh = (not args.single_device_forward) and jax.device_count() > 1

    if use_mesh:
        mesh_data, mesh_model = infer_mesh_shape(cfg, args, is_dawn)
        mesh = tj.create_mesh(mesh_data, mesh_model)
        from jax.sharding import NamedSharding, PartitionSpec as P

        data_sharding = NamedSharding(mesh, P("data", None))
        if is_dawn:
            from scripts.downstream_finetune_jax import build_sharded_fns_if_needed

            sharded_fns = build_sharded_fns_if_needed(cfg, mesh)

    def _init_restore_and_apply():
        variables = model.init(
            {"params": jax.random.PRNGKey(seed), "dropout": jax.random.PRNGKey(seed + 1)},
            jnp.ones((1, init_len), dtype=jnp.int32),
            deterministic=True,
        )
        params = restore_params_for_forward(checkpoint_meta["_raw"], variables["params"])

        if use_mesh:
            param_shardings = tj.get_param_shardings(params, mesh, is_baseline=not is_dawn)
            params = tj.shard_params_to_mesh(params, param_shardings)
            local = host_local_batch(global_batch_np)
            batch = tj.shard_to_mesh(local, data_sharding, (batch_size, seq_len))
        else:
            batch = jnp.asarray(global_batch_np)

        kwargs = {"sharded_fns": sharded_fns} if sharded_fns is not None else {}
        return model.apply(
            {"params": params},
            batch,
            labels=batch,
            attention_mask=jnp.ones_like(batch),
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(seed + 2)},
            **kwargs,
        )

    if mesh is not None:
        with mesh:
            out = _init_restore_and_apply()
    else:
        out = _init_restore_and_apply()

    valid = float(jax.device_get(out["valid_count"]))
    loss = float(jax.device_get(out["loss"]))
    correct = float(jax.device_get(out["correct"]))
    result = {
        "validation_loss": loss,
        "validation_accuracy": correct / max(valid, 1.0),
        "validation_perplexity": math.exp(loss) if loss < 100 else float("inf"),
    }
    if is_dawn:
        m = cfg["model"]
        n_qk = int(m.get("n_qk", 0))
        n_v = int(m.get("n_v", 0))
        n_know = int(m.get("n_know", 0))
        qk_active = float(jax.device_get(out.get("attn_qk_active", 0.0)))
        v_active = float(jax.device_get(out.get("attn_v_active", 0.0)))
        know_active = float(jax.device_get(out.get("know_active", 0.0)))
        result.update(
            {
                "qk_active_fraction": qk_active,
                "qk_active_neurons_per_token": qk_active * n_qk,
                "v_active_fraction": v_active,
                "v_active_neurons_per_token": v_active * n_v,
                "know_active_fraction": know_active,
                "know_active_neurons_per_token": know_active * n_know,
                "gate_sum_mean": (
                    float(jax.device_get(out.get("attn_gate_sum", 0.0)))
                    + float(jax.device_get(out.get("know_gate_sum", 0.0)))
                )
                / 2.0,
                "gate_max_mean": (
                    float(jax.device_get(out.get("attn_raw_gate_max", 0.0)))
                    + float(jax.device_get(out.get("know_raw_gate_max", 0.0)))
                )
                / 2.0,
                "gate_conc_mean": (
                    float(jax.device_get(out.get("attn_gate_conc", 0.0)))
                    + float(jax.device_get(out.get("know_gate_conc", 0.0)))
                )
                / 2.0,
                "score_std_mean": (
                    float(jax.device_get(out.get("attn_score_std", 0.0)))
                    + float(jax.device_get(out.get("know_score_std", 0.0)))
                )
                / 2.0,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Compute estimate
# ---------------------------------------------------------------------------


def baseline_core_flops(cfg: Dict[str, Any], seq_len: int) -> float:
    m = cfg["model"]
    d = int(m.get("d_model", 1280))
    d_ff = int(m.get("d_ff", 4 * d))
    layers = int(m.get("n_layers", 18))
    return float(layers * (8 * d * d + 4 * d * d_ff + 4 * seq_len * d))


def dawn_impl_flops(cfg: Dict[str, Any], seq_len: int) -> float:
    m = cfg["model"]
    d = int(m.get("d_model", 1280))
    r = int(m.get("d_route", 256))
    layers = int(m.get("n_layers", 18))
    n_qk = int(m.get("n_qk", 6400))
    n_v = int(m.get("n_v", 10400))
    n_know = int(m.get("n_know", 100000))
    attn_router = 2 * d * (3 * r) + 2 * d * 3
    know_router = 2 * d * r + 2 * d
    qk_pair_full = 8 * r * n_qk + 6 * d * n_qk
    v_full = 4 * r * n_v + 4 * d * n_v
    know_full = 4 * r * n_know + 4 * d * n_know
    attention = 4 * seq_len * d
    out_proj = 2 * d * d
    return float(layers * (attn_router + qk_pair_full + v_full + attention + out_proj + know_router + know_full))


def dawn_ideal_active_flops(cfg: Dict[str, Any], seq_len: int, util: Dict[str, Any]) -> Optional[float]:
    needed = ("qk_active_neurons_per_token", "v_active_neurons_per_token", "know_active_neurons_per_token")
    if any(key not in util for key in needed):
        return None
    m = cfg["model"]
    d = int(m.get("d_model", 1280))
    r = int(m.get("d_route", 256))
    layers = int(m.get("n_layers", 18))
    n_qk = int(m.get("n_qk", 6400))
    n_v = int(m.get("n_v", 10400))
    n_know = int(m.get("n_know", 100000))
    aq = float(util["qk_active_neurons_per_token"])
    av = float(util["v_active_neurons_per_token"])
    ak = float(util["know_active_neurons_per_token"])
    attn_router = 2 * d * (3 * r) + 2 * d * 3
    know_router = 2 * d * r + 2 * d
    full_scores = 8 * r * n_qk + 4 * r * n_v + 4 * r * n_know
    active_ops = 8 * d * aq + 4 * d * av + 4 * d * ak
    attention = 4 * seq_len * d
    out_proj = 2 * d * d
    return float(layers * (attn_router + know_router + full_scores + active_ops + attention + out_proj))


def lm_head_flops(cfg: Dict[str, Any]) -> float:
    m = cfg["model"]
    return float(2 * int(m.get("d_model", 1280)) * int(m.get("vocab_size", 30522)))


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def latex_downstream_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Downstream transfer for the 400M v3.9.4 pilot. Accuracy entries are best validation accuracies in percent. Delta columns are percentage-point differences.}",
        r"\label{tab:v394_downstream}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Task & DAWN pre. & DAWN rand. & DAWN $\Delta$ & Dense pre. & Dense rand. & Dense $\Delta$ & DAWN-Dense \\",
        r"\midrule",
    ]
    for row in main_downstream_rows(rows):
        lines.append(
            f"{tex_escape(task_label(row['task']))} & {pct(row['dawn_pre'])} & {pct(row['dawn_random'])} & "
            f"{pp(row['dawn_delta'])} & {pct(row['tf_pre'])} & {pct(row['tf_random'])} & "
            f"{pp(row['tf_delta'])} & {pp(row['dawn_minus_tf'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def markdown_downstream_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| task | DAWN pre | DAWN random | DAWN pre-rand pp | dense pre | dense random | dense pre-rand pp | DAWN pre-dense pre pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_downstream_rows(rows):
        lines.append(
            f"| {task_label(row['task'])} | {pct(row['dawn_pre'])} | {pct(row['dawn_random'])} | "
            f"{pp(row['dawn_delta'])} | {pct(row['tf_pre'])} | {pct(row['tf_random'])} | "
            f"{pp(row['tf_delta'])} | {pp(row['dawn_minus_tf'])} |"
        )
    return "\n".join(lines)


def checkpoint_table(metas: List[Dict[str, Any]], forward: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| model | checkpoint loaded | params | ckpt step | ckpt best val loss | val loss | val ppl |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for meta in metas:
        fwd = forward.get(meta["model"], {})
        lines.append(
            f"| {meta['model']} | {meta.get('loaded')} | {num(meta.get('parameter_count'), 0)} | "
            f"{num(meta.get('step'), 0)} | {num(meta.get('best_val_loss'), 4)} | "
            f"{num(fwd.get('validation_loss'), 4)} | {num(fwd.get('validation_perplexity'), 2)} |"
        )
    return "\n".join(lines)


def pretraining_setup_table(dawn_cfg: Dict[str, Any], tf_cfg: Dict[str, Any], dawn_meta: Dict[str, Any], tf_meta: Dict[str, Any]) -> str:
    rows = [
        ("DAWN v3.9.4 400M", dawn_cfg, dawn_meta),
        ("dense Transformer 400M", tf_cfg, tf_meta),
    ]
    lines = [
        "| model | params | dataset/token budget | seq len | estimated steps | checkpoint best val loss |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for label, cfg, meta in rows:
        dataset, budget = infer_dataset_and_budget(cfg, meta)
        budget_text = f"{dataset}, {budget:,} tokens" if budget else dataset
        lines.append(
            f"| {label} | {num(meta.get('parameter_count'), 0)} | {budget_text} | "
            f"{cfg.get('model', {}).get('max_seq_len', '--')} | {num(total_steps_estimate(cfg), 0)} | "
            f"{num(meta.get('best_val_loss'), 4)} |"
        )
    return "\n".join(lines)


def compute_table(dawn_cfg: Dict[str, Any], tf_cfg: Dict[str, Any], seq_len: int, dawn_forward: Dict[str, Any]) -> str:
    tf_core = baseline_core_flops(tf_cfg, seq_len)
    tf_lm = lm_head_flops(tf_cfg)
    dawn_core = dawn_impl_flops(dawn_cfg, seq_len)
    dawn_lm = lm_head_flops(dawn_cfg)
    ideal = dawn_ideal_active_flops(dawn_cfg, seq_len, dawn_forward)
    lines = [
        "| compute view | core FLOPs/token | total incl. LM head | note |",
        "|---|---:|---:|---|",
        f"| dense Transformer baseline | {sci(tf_core)} | {sci(tf_core + tf_lm)} | approximate dense Transformer forward compute |",
        f"| implemented v3.9.4 full routing | {sci(dawn_core)} | {sci(dawn_core + dawn_lm)} | includes full routing over qk, v, know pools |",
        f"| idealized active-operator v3.9.4 | {sci(ideal)} | {sci(None if ideal is None else ideal + dawn_lm)} | diagnostic only; do not claim wall-clock reduction from this row |",
    ]
    return "\n".join(lines)


def utilization_block(dawn_forward: Dict[str, Any]) -> str:
    if "qk_active_fraction" not in dawn_forward:
        return (
            "Utilization diagnostics were not run. To populate them, rerun with --run-forward "
            "and a validation .bin path if the config default is not accessible.\n\n"
            + UTIL_NOTE
        )
    return "\n".join(
        [
            "| pool | active fraction | active neurons/token |",
            "|---|---:|---:|",
            f"| qk | {num(dawn_forward.get('qk_active_fraction'), 4)} | {num(dawn_forward.get('qk_active_neurons_per_token'), 2)} |",
            f"| v | {num(dawn_forward.get('v_active_fraction'), 4)} | {num(dawn_forward.get('v_active_neurons_per_token'), 2)} |",
            f"| know | {num(dawn_forward.get('know_active_fraction'), 4)} | {num(dawn_forward.get('know_active_neurons_per_token'), 2)} |",
            "",
            f"gate_sum mean: {num(dawn_forward.get('gate_sum_mean'), 4)}",
            f"gate_max mean: {num(dawn_forward.get('gate_max_mean'), 4)}",
            f"gate_conc mean: {num(dawn_forward.get('gate_conc_mean'), 4)}",
            f"score_std mean: {num(dawn_forward.get('score_std_mean'), 4)}",
            "",
            UTIL_NOTE,
        ]
    )


def appendix_text() -> str:
    return textwrap.dedent(
        f"""
        \\subsection{{400M Pilot with an Earlier SRW Variant}}
        This appendix reports the 400M spatial-r1-v3.9.4 experiment as an earlier SRW-family scaling and downstream-transfer pilot. The comparison uses the v3.9.4 DAWN run, a dense Transformer baseline at the same nominal scale, and matched downstream fine-tuning runs from pretrained and random initialization.

        \\paragraph{{Language-modeling scaling result.}}
        The checkpoint-derived summary reports the available model scale, token-budget, step, and validation metadata. Missing final validation-loss or throughput entries should be filled from the original pretraining logs rather than inferred from downstream transfer.

        \\paragraph{{Downstream transfer.}}
        v3.9.4 shows clear transfer over random initialization and is competitive with the dense Transformer baseline, but task-level results vary.

        \\paragraph{{Utilization diagnostics.}}
        {UTIL_NOTE}

        \\paragraph{{Compute discussion.}}
        The compute estimate separates the dense Transformer baseline, the implemented v3.9.4 computation with full routing over the operator pools, and an idealized active-operator view using measured active counts when available. We do not claim wall-clock FLOPs reduction unless the implemented computation supports it.

        \\paragraph{{Limitations.}}
        {LIMITATION_TEXT}
        """
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf-checkpoint", required=True, help="Dense Transformer checkpoint file or run directory.")
    parser.add_argument("--dawn394-checkpoint", required=True, help="DAWN spatial-r1-v3.9.4 checkpoint file or run directory.")
    parser.add_argument("--results", default="results", help="Directory with baseline/dawn downstream .txt logs.")
    parser.add_argument("--tf-config", default=DEFAULT_TF_CONFIG, help="Dense Transformer model config used for shape/compute.")
    parser.add_argument("--dawn394-config", default=DEFAULT_DAWN_CONFIG, help="DAWN v3.9.4 model/pretraining config.")
    parser.add_argument("--run-forward", action="store_true", help="Run validation forward pass and DAWN utilization diagnostics.")
    parser.add_argument("--val-data", default=None, help="Validation .bin path. Defaults to each config's data.bin_val when present.")
    parser.add_argument("--batch-size", type=int, default=16, help="Global forward-analysis batch size. On multi-host TPU this must be divisible by host count.")
    parser.add_argument("--seq-len", type=int, default=512, help="Forward-analysis sequence length and compute-estimate context length.")
    parser.add_argument("--mesh-data", type=int, default=None, help="Forward-analysis data mesh size. Defaults to config for DAWN and inferred data-only mesh for baseline.")
    parser.add_argument("--mesh-model", type=int, default=None, help="Forward-analysis model mesh size. Defaults to config for DAWN and 1 for baseline.")
    parser.add_argument("--single-device-forward", action="store_true", help="Disable mesh sharding for forward analysis even when multiple devices are visible.")
    parser.add_argument("--show-paths", action="store_true", help="Print checkpoint/log paths in the output. Off by default to avoid leaking internal paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = normalize_path(args.results)
    dawn_cfg = load_yaml(args.dawn394_config)
    tf_cfg = load_yaml(args.tf_config)
    downstream = parse_downstream_results(results_dir)

    dawn_est = dawn394_param_estimate(dawn_cfg["model"])
    tf_est = baseline_param_estimate(tf_cfg["model"])
    dawn_checkpoint = normalize_path(args.dawn394_checkpoint)
    tf_checkpoint = normalize_path(args.tf_checkpoint)
    dawn_meta = load_checkpoint_metadata("DAWN v3.9.4 400M", dawn_checkpoint, dawn_est, args.show_paths)
    tf_meta = load_checkpoint_metadata("dense Transformer 400M", tf_checkpoint, tf_est, args.show_paths)

    forward: Dict[str, Dict[str, Any]] = {}
    if args.run_forward:
        dawn_val = normalize_path(args.val_data) if args.val_data else dawn_cfg.get("data", {}).get("bin_val")
        tf_val = normalize_path(args.val_data) if args.val_data else tf_cfg.get("data", {}).get("bin_val")
        if dawn_val:
            forward["DAWN v3.9.4 400M"] = run_forward_analysis(
                "DAWN v3.9.4 400M", dawn_meta, dawn_cfg, str(dawn_val), args.batch_size, args.seq_len, is_dawn=True, args=args
            )
        else:
            forward["DAWN v3.9.4 400M"] = {"forward_error": "no validation data path"}
        if tf_val:
            forward["dense Transformer 400M"] = run_forward_analysis(
                "dense Transformer 400M", tf_meta, tf_cfg, str(tf_val), args.batch_size, args.seq_len, is_dawn=False, args=args
            )
        else:
            forward["dense Transformer 400M"] = {"forward_error": "no validation data path"}

    emit(section("COPYABLE PAPER OUTPUT: v3.9.4 400M PILOT"))
    emit("Use this as an earlier SRW-family 400M scaling/downstream-transfer pilot only.")
    emit("Do not present v3.9.4 as evidence for the final v4.1.5.5 sparsity mechanism.")

    emit(section("1. CLAIM SENTENCE"))
    emit("v3.9.4 shows clear transfer over random initialization and is competitive with the dense Transformer baseline, but task-level results vary.")

    emit(section("2. CHECKPOINT / PRETRAINING SETUP TABLE"))
    emit(pretraining_setup_table(dawn_cfg, tf_cfg, dawn_meta, tf_meta))
    failed = [m for m in (dawn_meta, tf_meta) if not m.get("loaded")]
    if failed:
        emit("\nCheckpoint load warnings:")
        for meta in failed:
            emit(f"- {meta['model']}: {meta.get('error')}")

    emit(section("3. CHECKPOINT-DERIVED METRICS TABLE"))
    emit(checkpoint_table([dawn_meta, tf_meta], forward))
    if not args.run_forward:
        emit("\nForward metrics are blank because --run-forward was not used.")

    emit(section("4. DOWNSTREAM MAIN TABLE (MARKDOWN)"))
    emit(markdown_downstream_table(downstream))

    emit(section("5. DOWNSTREAM MAIN TABLE (LATEX)"))
    emit(latex_downstream_table(downstream))

    emit(section("6. UTILIZATION DIAGNOSTICS"))
    emit(utilization_block(forward.get("DAWN v3.9.4 400M", {})))

    emit(section("7. FLOPS / COMPUTE ESTIMATE"))
    emit(compute_table(dawn_cfg, tf_cfg, args.seq_len, forward.get("DAWN v3.9.4 400M", {})))
    emit("\nCompute note: implemented compute and ideal active-operator compute are separate. Do not claim wall-clock FLOPs reduction from the idealized row.")

    emit(section("8. APPENDIX DRAFT TEXT"))
    emit(appendix_text())

    emit(section("9. LIMITATION TO INCLUDE VERBATIM"))
    emit(LIMITATION_TEXT)

    if args.show_paths:
        emit(section("10. PATHS USED"))
        emit(f"results: {results_dir}")
        emit(f"dawn394 checkpoint: {dawn_meta.get('checkpoint_file')}")
        emit(f"tf checkpoint: {tf_meta.get('checkpoint_file')}")


if __name__ == "__main__":
    main()

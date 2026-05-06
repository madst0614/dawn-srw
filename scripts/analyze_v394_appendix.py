#!/usr/bin/env python3
"""One-shot data extraction for the 400M spatial-r1-v3.9.4 pilot.

Inputs:
  --tf-checkpoint       dense Transformer checkpoint file or run directory
  --dawn394-checkpoint  DAWN spatial-r1-v3.9.4 checkpoint file or run directory
  --results             directory containing baseline.txt, baseline_random.txt,
                        dawn.txt, dawn_random.txt

Default behavior:
  - Parse downstream logs.
  - Resolve/load checkpoint bytes and report checkpoint metadata/param counts.
  - Print copyable data tables and diagnostics to stdout.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DAWN_CONFIG = "configs/legacy/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32.yaml"
DEFAULT_TF_CONFIG = "configs/downstream/baseline/sst2.yaml"
DEFAULT_DOWNSTREAM_CONFIG_ROOT = "configs/downstream"

DOWNSTREAM_LOGS = (
    "baseline.txt",
    "baseline_random.txt",
    "dawn.txt",
    "dawn_random.txt",
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


def is_primary_host() -> bool:
    """Return True only on JAX host 0, falling back to True off TPU/JAX."""
    try:
        import jax

        return int(jax.process_index()) == 0
    except Exception:
        return True


def emit(*args: Any, **kwargs: Any) -> None:
    """Print final data output once in multi-host TPU runs."""
    if is_primary_host():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressTracker:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, int(total_units))
        self.done_units = 0
        self.start_time = time.time()
        self.last_percent = -1.0

    def update(self, message: str, units: int = 0) -> None:
        if not is_primary_host():
            return
        self.done_units = min(self.total_units, self.done_units + max(0, int(units)))
        percent = 100.0 * self.done_units / self.total_units
        elapsed = time.time() - self.start_time
        eta = None
        if self.done_units > 0:
            eta = elapsed * (self.total_units - self.done_units) / self.done_units
        line = f"\r[progress] {percent:6.2f}% ETA {format_seconds(eta)} elapsed {format_seconds(elapsed)}  {progress_stage(message):<18}"
        sys.stderr.write(line)
        sys.stderr.flush()
        if self.done_units >= self.total_units:
            sys.stderr.write("\n")
            sys.stderr.flush()


_PROGRESS: Optional[ProgressTracker] = None


def set_progress_total(total_units: int) -> None:
    global _PROGRESS
    _PROGRESS = ProgressTracker(total_units)


def progress(message: str, units: int = 0) -> None:
    if not is_primary_host():
        return
    if _PROGRESS is not None:
        _PROGRESS.update(message, units=units)
    else:
        sys.stderr.write(f"\r[progress]   0.00% ETA --:--:-- elapsed 00:00:00  {progress_stage(message):<18}")
        sys.stderr.flush()


def progress_stage(message: str) -> str:
    text = message.lower()
    if "final" in text or "done" in text:
        return "done"
    if "dense transformer" in text or "dense" in text:
        return "dense forward"
    if "dawn" in text and "forward" in text:
        return "dawn forward"
    if "forward" in text:
        return "forward"
    if "checkpoint" in text:
        return "checkpoints"
    if "downstream" in text:
        return "downstream"
    if "config" in text:
        return "config"
    if "analysis" in text:
        return "analysis"
    return "running"


# ---------------------------------------------------------------------------
# Config and parameter estimates
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def fmt_config_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def unique_config_value(values: Iterable[Any]) -> str:
    clean = sorted({fmt_config_value(v) for v in values if v is not None})
    if not clean:
        return "--"
    return clean[0] if len(clean) == 1 else "/".join(clean)


def downstream_config_paths(root: str) -> List[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.yaml") if p.is_file())


def downstream_finetune_hparams(root: str) -> Dict[str, Any]:
    configs: List[Dict[str, Any]] = []
    for path in downstream_config_paths(root):
        cfg = load_yaml(str(path))
        if cfg.get("training") and cfg.get("downstream"):
            configs.append(cfg)

    seeds = [cfg.get("seed") for cfg in configs]
    training = [cfg.get("training", {}) for cfg in configs]
    downstream = [cfg.get("downstream", {}) for cfg in configs]

    eval_by_interval: Dict[str, set] = {}
    for cfg in configs:
        task = task_label(str(cfg.get("downstream", {}).get("task", "")))
        interval = fmt_config_value(cfg.get("training", {}).get("eval_interval"))
        if task and interval != "--":
            eval_by_interval.setdefault(interval, set()).add(task)
    eval_interval = "; ".join(
        f"{interval}: {'/'.join(sorted(tasks))}"
        for interval, tasks in sorted(eval_by_interval.items(), key=lambda item: float(item[0]))
    ) or "--"

    return {
        "optimizer": "AdamW",
        "learning_rate": unique_config_value(t.get("lr", t.get("learning_rate")) for t in training),
        "warmup": "warmup_ratio=" + unique_config_value(t.get("warmup_ratio", t.get("warmup_steps")) for t in training),
        "weight_decay": unique_config_value(t.get("weight_decay") for t in training),
        "batch_size": unique_config_value(t.get("batch_size") for t in training),
        "max_seq_len": unique_config_value(d.get("max_seq_len") for d in downstream),
        "eval_interval": eval_interval,
        "number_of_seeds": str(len({s for s in seeds if s is not None}) or 0),
    }


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
    progress(f"downstream: scanning {results_dir}")
    for name in DOWNSTREAM_LOGS:
        path = join_ref(results_dir, name)
        progress(f"downstream: checking {name}")
        if file_exists(path):
            parsed = parse_downstream_log(path)
            rows.extend(parsed)
            progress(f"downstream: parsed {len(parsed)} run(s) from {name}", units=1)
        else:
            progress(f"downstream: missing {name}", units=1)
    if not rows:
        raise FileNotFoundError(f"No downstream logs found in {results_dir}; expected {', '.join(DOWNSTREAM_LOGS)}")
    progress(f"downstream: total parsed runs={len(rows)}")
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

        progress(f"checkpoint: resolving {label}")
        resolved = resolve_checkpoint(normalize_path(checkpoint))
        progress(f"checkpoint: loading {label}")
        with tj._open_file(resolved, "rb") as f:
            raw = serialization.msgpack_restore(f.read())
        params = raw.get("params", {}) if isinstance(raw, dict) else {}
        count = raw_param_count(params) or estimated_params
        progress(f"checkpoint: loaded {label}; params={count:,}", units=1)
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
        progress(f"checkpoint: failed {label}: {type(exc).__name__}: {exc}", units=1)
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
        progress(f"forward: skipped {label}; checkpoint not loaded", units=6)
        return {"forward_error": "checkpoint not loaded"}

    import jax
    import jax.numpy as jnp
    import scripts.train_jax as tj

    progress(f"forward: building model {label}", units=1)
    model = tj.build_model_from_config(cfg)
    seed = int(cfg.get("seed", 1))
    init_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    progress(f"forward: reading validation batch {label}; batch={batch_size} seq_len={seq_len}", units=1)
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
        progress(f"forward: mesh ready {label}; mesh_data={mesh_data} mesh_model={mesh_model}", units=1)
    else:
        progress(f"forward: single-device path ready {label}", units=1)

    def _init_restore_and_apply():
        progress(f"forward: initializing/restoring params {label}", units=1)
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
        progress(f"forward: running apply {label}", units=1)
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
    progress(f"forward: metrics ready {label}", units=1)
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


def finetune_hparams_table(hparams: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "| optimizer | learning_rate | warmup | weight_decay | batch_size | max_seq_len | eval_interval | number_of_seeds |",
            "|---|---:|---|---:|---:|---:|---|---:|",
            f"| {hparams.get('optimizer', '--')} | {hparams.get('learning_rate', '--')} | {hparams.get('warmup', '--')} | "
            f"{hparams.get('weight_decay', '--')} | {hparams.get('batch_size', '--')} | {hparams.get('max_seq_len', '--')} | "
            f"{hparams.get('eval_interval', '--')} | {hparams.get('number_of_seeds', '--')} |",
        ]
    )


def latex_downstream_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
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
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def raw_downstream_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| task | model | init | train_rows | eval_rows | total_steps | step0_acc | final_acc | best_acc | best_step | wall_clock_final_s | run_dir |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {task_label(row.get('task', ''))} | {row.get('model', '--')} | {row.get('init_type', '--')} | "
            f"{num(row.get('train_rows'), 0)} | {num(row.get('eval_rows'), 0)} | {num(row.get('total_steps'), 0)} | "
            f"{pct(row.get('step0_acc'))} | {pct(row.get('final_acc'))} | {pct(row.get('best_acc'))} | "
            f"{num(row.get('best_step'), 0)} | {num(row.get('wall_clock_final_s'), 2)} | {row.get('run_dir', '--')} |"
        )
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
        "| compute_view | core_flops_per_token | total_flops_per_token_incl_lm_head |",
        "|---|---:|---:|",
        f"| dense_transformer_baseline | {sci(tf_core)} | {sci(tf_core + tf_lm)} |",
        f"| implemented_v394_full_routing | {sci(dawn_core)} | {sci(dawn_core + dawn_lm)} |",
        f"| idealized_v394_active_operator | {sci(ideal)} | {sci(None if ideal is None else ideal + dawn_lm)} |",
    ]
    return "\n".join(lines)


def utilization_block(dawn_forward: Dict[str, Any]) -> str:
    if "qk_active_fraction" not in dawn_forward:
        return "| metric | value |\n|---|---:|\n| status | not_run |"
    return "\n".join(
        [
            "| pool | active_fraction | active_neurons_per_token |",
            "|---|---:|---:|",
            f"| qk | {num(dawn_forward.get('qk_active_fraction'), 4)} | {num(dawn_forward.get('qk_active_neurons_per_token'), 2)} |",
            f"| v | {num(dawn_forward.get('v_active_fraction'), 4)} | {num(dawn_forward.get('v_active_neurons_per_token'), 2)} |",
            f"| know | {num(dawn_forward.get('know_active_fraction'), 4)} | {num(dawn_forward.get('know_active_neurons_per_token'), 2)} |",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| gate_sum_mean | {num(dawn_forward.get('gate_sum_mean'), 4)} |",
            f"| gate_max_mean | {num(dawn_forward.get('gate_max_mean'), 4)} |",
            f"| gate_conc_mean | {num(dawn_forward.get('gate_conc_mean'), 4)} |",
            f"| score_std_mean | {num(dawn_forward.get('score_std_mean'), 4)} |",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf-checkpoint", required=True, help="Dense Transformer checkpoint file or run directory.")
    parser.add_argument("--dawn394-checkpoint", required=True, help="DAWN spatial-r1-v3.9.4 checkpoint file or run directory.")
    parser.add_argument("--results", default="results", help="Directory with baseline/dawn downstream .txt logs.")
    parser.add_argument("--downstream-config-root", default=DEFAULT_DOWNSTREAM_CONFIG_ROOT, help="Directory with downstream fine-tuning YAML configs.")
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
    total_units = 1 + len(DOWNSTREAM_LOGS) + 2 + (12 if args.run_forward else 0) + 1
    set_progress_total(total_units)
    progress("analysis: start")
    results_dir = normalize_path(args.results)
    progress("config: loading yaml files")
    dawn_cfg = load_yaml(args.dawn394_config)
    tf_cfg = load_yaml(args.tf_config)
    finetune_hparams = downstream_finetune_hparams(args.downstream_config_root)
    progress("config: loaded yaml files", units=1)
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
            progress("forward: skipped DAWN v3.9.4 400M; no validation data path", units=6)
        if tf_val:
            forward["dense Transformer 400M"] = run_forward_analysis(
                "dense Transformer 400M", tf_meta, tf_cfg, str(tf_val), args.batch_size, args.seq_len, is_dawn=False, args=args
            )
        else:
            forward["dense Transformer 400M"] = {"forward_error": "no validation data path"}
            progress("forward: skipped dense Transformer 400M; no validation data path", units=6)

    emit("downstream_finetuning_hyperparameters_markdown")
    emit(finetune_hparams_table(finetune_hparams))
    if finetune_hparams.get("number_of_seeds") == "1":
        emit("All downstream results are single-run evaluations.")

    emit("")
    emit("checkpoint_pretraining_setup_markdown")
    emit(pretraining_setup_table(dawn_cfg, tf_cfg, dawn_meta, tf_meta))
    failed = [m for m in (dawn_meta, tf_meta) if not m.get("loaded")]
    if failed:
        emit("")
        emit("checkpoint_load_warnings")
        for meta in failed:
            emit(f"- {meta['model']}: {meta.get('error')}")

    emit("")
    emit("checkpoint_metrics_markdown")
    emit(checkpoint_table([dawn_meta, tf_meta], forward))
    if not args.run_forward:
        emit("")
        emit("forward_status: not_run")

    emit("")
    emit("downstream_raw_runs_markdown")
    emit(raw_downstream_table(downstream))

    emit("")
    emit("downstream_best_accuracy_markdown")
    emit(markdown_downstream_table(downstream))

    emit("")
    emit("downstream_best_accuracy_latex_tabular")
    emit(latex_downstream_table(downstream))

    emit("")
    emit("utilization_diagnostics")
    emit(utilization_block(forward.get("DAWN v3.9.4 400M", {})))

    emit("")
    emit("compute_estimate_markdown")
    emit(compute_table(dawn_cfg, tf_cfg, args.seq_len, forward.get("DAWN v3.9.4 400M", {})))

    if args.show_paths:
        emit("")
        emit("paths_used")
        emit(f"results: {results_dir}")
        emit(f"dawn394 checkpoint: {dawn_meta.get('checkpoint_file')}")
        emit(f"tf checkpoint: {tf_meta.get('checkpoint_file')}")
    progress("analysis: final data output emitted", units=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Common utilities for DAWN-SRW analysis and experiments.

This module assumes the official model implementation lives at:
    models.dawn_srw

It supports local and GCS checkpoints, legacy v4.1.5.5 parameter migration,
BERT tokenizer loading, JSON saving, and a few pure-JAX helpers used by the
experiment scripts.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

def _find_project_root() -> Path:
    """Find repo root containing models/dawn_srw.py.

    This lets the analysis scripts run from scripts/analysis/dawn_srw/
    without requiring PYTHONPATH to be set manually.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "models" / "dawn_srw.py").exists():
            return parent
    # Standard repo layout fallback:
    # scripts/analysis/dawn_srw/dawn_srw_common.py -> repo root is parents[3].
    if len(here.parents) > 3:
        return here.parents[3]
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import flax.serialization as serialization
import numpy as np
import yaml


def import_dawn_srw():
    import importlib
    return importlib.import_module("models.dawn_srw")


def is_gcs(path: str | os.PathLike[str]) -> bool:
    return str(path).startswith("gs://")


def open_file(path: str | os.PathLike[str], mode: str = "rb"):
    path_str = str(path)
    if is_gcs(path_str):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().open(path_str, mode)
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
    p = Path(path_str)
    if "w" in mode or "a" in mode:
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def list_dir(path: str | os.PathLike[str]) -> List[str]:
    path_str = str(path)
    if is_gcs(path_str):
        try:
            import gcsfs
            return ["gs://" + f for f in gcsfs.GCSFileSystem().ls(path_str)]
        except ImportError:
            import tensorflow as tf
            return [path_str.rstrip("/") + "/" + f for f in tf.io.gfile.listdir(path_str)]
    return [str(p) for p in Path(path_str).iterdir()]


def load_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open_file(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, output_dir: str | os.PathLike[str], filename: str) -> str:
    out = str(output_dir)
    if not is_gcs(out):
        Path(out).mkdir(parents=True, exist_ok=True)
    filepath = out.rstrip("/") + "/" + filename

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            val = float(obj)
            return str(val) if math.isnan(val) or math.isinf(val) else val
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return np.asarray(obj).tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return obj

    with open_file(filepath, "w") as f:
        json.dump(data, f, indent=2, default=convert, ensure_ascii=False)
    return filepath


def select_checkpoint(checkpoint: str | os.PathLike[str]) -> str:
    ckpt = str(checkpoint)
    if ckpt.endswith(".flax"):
        return ckpt
    files = list_dir(ckpt)
    flax_files = sorted([f for f in files if f.endswith(".flax")])
    best = [f for f in flax_files if "best_model" in os.path.basename(f)]
    if best:
        return best[0]
    if flax_files:
        return flax_files[-1]
    raise FileNotFoundError(f"No .flax checkpoint found in {ckpt}. Files: {files[:20]}")


def build_model(cfg: Dict[str, Any]):
    mod = import_dawn_srw()
    m = cfg["model"]
    t = cfg.get("training", {})
    return mod.DAWN(
        vocab_size=m.get("vocab_size", 30522),
        d_model=m.get("d_model", 384),
        n_layers=m.get("n_layers", 12),
        n_heads=m.get("n_heads", 6),
        max_seq_len=m.get("max_seq_len", 512),
        d_route=m.get("d_route", m.get("d_bottleneck", 128)),
        n_qk=m.get("n_qk", 1580),
        n_v=m.get("n_v", 2600),
        n_rst=m.get("n_rst", m.get("n_know", 25200)),
        n_know=m.get("n_know", None),
        dropout_rate=m.get("dropout", 0.1),
        router_dropout=m.get("router_dropout", 0.1),
        gradient_checkpointing=False,
        n_chunks_rst=t.get("n_chunks_rst", t.get("n_chunks_know", 1)),
        n_chunks_know=t.get("n_chunks_know", 1),
        n_chunks_qk=t.get("n_chunks_qk", 1),
        n_chunks_v=t.get("n_chunks_v", 1),
    )


def model_cfg_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg["model"]
    t = cfg.get("training", {})
    return {
        "vocab_size": m.get("vocab_size", 30522),
        "d_model": m.get("d_model", 384),
        "n_layers": m.get("n_layers", 12),
        "n_heads": m.get("n_heads", 6),
        "max_seq_len": m.get("max_seq_len", 512),
        "d_route": m.get("d_route", m.get("d_bottleneck", 128)),
        "n_qk": m.get("n_qk", 1580),
        "n_v": m.get("n_v", 2600),
        "n_rst": m.get("n_rst", m.get("n_know", 25200)),
        "n_know": m.get("n_know", m.get("n_rst", 25200)),
        "n_chunks_qk": t.get("n_chunks_qk", 1),
        "n_chunks_v": t.get("n_chunks_v", 1),
        "n_chunks_rst": t.get("n_chunks_rst", t.get("n_chunks_know", 1)),
    }


def squeeze_replicated_params(params):
    def _sq(x):
        if hasattr(x, "ndim") and x.ndim >= 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x
    return jax.tree.map(_sq, params)


def init_target_params(model, cfg: Dict[str, Any]):
    rng = jax.random.PRNGKey(0)
    max_seq = cfg["model"].get("max_seq_len", 512)
    dummy = jnp.ones((1, max_seq), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return variables["params"]


def load_checkpoint_params(checkpoint: str | os.PathLike[str], cfg: Dict[str, Any], model=None):
    """Load params from a DAWN-SRW checkpoint and apply legacy migration.

    Returns:
        params, meta
    """
    mod = import_dawn_srw()
    if model is None:
        model = build_model(cfg)
    target_params = init_target_params(model, cfg)
    ckpt_path = select_checkpoint(checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    with open_file(ckpt_path, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    raw = mod.migrate_legacy_v4155_params(raw)
    raw_params = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
    params = serialization.from_state_dict(target_params, raw_params)
    params = squeeze_replicated_params(params)
    meta = {
        "checkpoint": ckpt_path,
        "step": int(raw.get("step", 0)) if isinstance(raw, dict) else 0,
        "epoch": int(raw.get("epoch", 0)) if isinstance(raw, dict) else 0,
    }
    return params, meta


def load_tokenizer(name: str = "bert-base-uncased"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def load_val_tokens(val_data_path: str, max_tokens: Optional[int] = None) -> np.ndarray:
    print(f"Loading validation tokens: {val_data_path}")
    if is_gcs(val_data_path):
        try:
            import gcsfs
            with gcsfs.GCSFileSystem().open(val_data_path, "rb") as f:
                raw = f.read()
        except ImportError:
            import tensorflow as tf
            with tf.io.gfile.GFile(val_data_path, "rb") as f:
                raw = f.read()
        tokens = np.frombuffer(raw, dtype=np.uint16).copy()
    else:
        tokens = np.memmap(val_data_path, dtype=np.uint16, mode="r")
        tokens = np.array(tokens)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params) if hasattr(x, "size")))


def encode_prompt(tokenizer, prompt: str, max_seq_len: int):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
    return ids


def find_token_index(tokens: Sequence[str], target: Optional[str], default: str = "last") -> int:
    """Find a token index by token text. Handles BERT ## subwords lightly."""
    if target is None or target == "":
        return len(tokens) - 1 if default == "last" else 0
    target_l = target.lower()
    stripped = [t.lower().replace("##", "") for t in tokens]
    for i, t in enumerate(stripped):
        if t == target_l:
            return i
    for i, t in enumerate(stripped):
        if target_l in t:
            return i
    raise ValueError(f"Could not find target token {target!r} in tokens: {tokens}")


def np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def token_level_ce(logits: np.ndarray, input_ids: Sequence[int]) -> Tuple[np.ndarray, float]:
    """Return next-token CE per position and mean CE for a single sequence."""
    ids = np.asarray(input_ids, dtype=np.int64)
    if len(ids) < 2:
        return np.array([], dtype=np.float32), 0.0
    shift_logits = logits[0, :-1, :]
    labels = ids[1:]
    probs = np_softmax(shift_logits, axis=-1)
    ce = -np.log(np.take_along_axis(probs, labels[:, None], axis=-1).squeeze(-1) + 1e-12)
    return ce, float(np.mean(ce))


def ensure_dir(path: str | os.PathLike[str]) -> str:
    p = str(path)
    if not is_gcs(p):
        Path(p).mkdir(parents=True, exist_ok=True)
    return p

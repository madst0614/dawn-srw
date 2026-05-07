#!/usr/bin/env python3
"""Aggregate paper runner outputs into copyable CSV/JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def open_file(path: str, mode: str = "r"):
    if is_gcs(path):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().open(path, mode)
        except ImportError:
            import tensorflow as tf
            return tf.io.gfile.GFile(path, mode)
    p = Path(path)
    if "w" in mode or "a" in mode:
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def save_json(data: Any, output_dir: str, filename: str) -> str:
    path = join_path(output_dir, filename)
    with open_file(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def join_path(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name if is_gcs(base) else str(Path(base) / name)


def exists(path: str) -> bool:
    if is_gcs(path):
        try:
            with open_file(path, "r"):
                return True
        except Exception:
            return False
    return Path(path).exists()


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not exists(path):
        return None
    with open_file(path, "r") as f:
        return json.load(f)


def write_csv(rows: List[Dict[str, Any]], output_dir: str, filename: str) -> str:
    path = join_path(output_dir, filename)
    if not rows:
        rows = [{}]
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def num(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "--"
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return "--"
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def model_row(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": info.get("model_name", ""),
        "model_version": info.get("model_version", ""),
        "checkpoint": info.get("checkpoint", ""),
        "checkpoint_file": info.get("checkpoint_file", ""),
        "checkpoint_step": info.get("checkpoint_step", ""),
        "params": info.get("params", info.get("n_params", "")),
        "params_M": info.get("params_M", info.get("n_params_M", "")),
        "d_model": info.get("d_model", ""),
        "layers": info.get("layers", info.get("n_layers", "")),
        "heads": info.get("heads", info.get("n_heads", "")),
        "d_ff": info.get("d_ff", ""),
        "d_route": info.get("d_route", ""),
        "n_qk": info.get("n_qk", ""),
        "n_v": info.get("n_v", ""),
        "n_rst": info.get("n_rst", ""),
    }


def validation_row(validation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": validation.get("model_name", ""),
        "checkpoint": validation.get("checkpoint", ""),
        "val_loss": validation.get("val_loss", validation.get("loss", "")),
        "perplexity": validation.get("perplexity", ""),
        "accuracy": validation.get("accuracy", ""),
        "accuracy_pct": validation.get("accuracy_pct", ""),
        "total_valid_tokens": validation.get("total_valid_tokens", ""),
        "seq_len": validation.get("seq_len", ""),
        "batch_size": validation.get("batch_size", ""),
    }


def dense_ffn_flops_per_token(info: Optional[Dict[str, Any]]) -> Any:
    if not info:
        return ""
    try:
        d_model = int(info.get("d_model", 0))
        d_ff = int(info.get("d_ff", 0) or 4 * d_model)
        return float(4 * d_model * d_ff)
    except Exception:
        return ""


def dense_core_flops_per_token(info: Optional[Dict[str, Any]], validation: Optional[Dict[str, Any]]) -> Any:
    if not info:
        return ""
    try:
        d_model = int(info.get("d_model", 0))
        d_ff = int(info.get("d_ff", 0) or 4 * d_model)
        layers = int(info.get("layers", info.get("n_layers", 0)))
        seq_len = int((validation or {}).get("seq_len", info.get("max_seq_len", 512)) or 512)
        return float(layers * (8 * d_model * d_model + 4 * d_model * d_ff + 4 * seq_len * d_model))
    except Exception:
        return ""


def print_validation_table(rows: List[Dict[str, Any]]) -> None:
    print("")
    print("paper_validation_table")
    print("| model | val_loss | perplexity | accuracy_pct | valid_tokens |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r.get('model_name', '')} | {num(r.get('val_loss'), 5)} | "
            f"{num(r.get('perplexity'), 3)} | {num(r.get('accuracy_pct'), 2)} | "
            f"{r.get('total_valid_tokens', '--')} |"
        )


def print_compute_table(rows: List[Dict[str, Any]]) -> None:
    print("")
    print("paper_active_compute_table")
    print("| model | val_loss | active_rw_flops/token | dense_ffn_flops/token | dense_core_flops/token | dense_ffn_ratio |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r.get('model_name', '')} | {num(r.get('val_loss'), 5)} | "
            f"{num(r.get('active_rw_flops_per_token'), 3)} | "
            f"{num(r.get('dense_ffn_flops_per_token'), 3)} | "
            f"{num(r.get('dense_tf_core_flops_per_token_est'), 3)} | "
            f"{num(r.get('dense_ffn_equiv_ratio'), 4)} |"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf-dir", required=True, help="Directory containing TF model_info.json and validation.json.")
    ap.add_argument("--dawn-dir", required=True, help="Directory containing DAWN model_info.json and validation.json.")
    ap.add_argument("--active-json", required=True, help="dawn_active_compute.json path.")
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tf_info = read_json(join_path(args.tf_dir, "model_info.json"))
    dawn_info = read_json(join_path(args.dawn_dir, "model_info.json"))
    tf_val = read_json(join_path(args.tf_dir, "validation.json"))
    dawn_val = read_json(join_path(args.dawn_dir, "validation.json"))
    active = read_json(args.active_json)

    model_rows = [model_row(x) for x in (tf_info, dawn_info) if x]
    validation_rows = [validation_row(x) for x in (tf_val, dawn_val) if x]

    compute_rows: List[Dict[str, Any]] = []
    if tf_val:
        compute_rows.append(
            {
                "model_name": tf_val.get("model_name", "tf_baseline"),
                "val_loss": tf_val.get("val_loss", ""),
                "active_rw_flops_per_token": "",
                "dense_ffn_flops_per_token": dense_ffn_flops_per_token(tf_info),
                "dense_tf_core_flops_per_token_est": dense_core_flops_per_token(tf_info, tf_val),
                "dense_ffn_equiv_ratio": 1.0,
                "note": "dense Transformer FFN baseline",
            }
        )
    if dawn_val or active:
        active_layer_rows = (active or {}).get("layer_rows", [])
        dense_ffn = ""
        if active_layer_rows:
            vals = [r.get("dense_ffn_equiv_flops_per_token", "") for r in active_layer_rows]
            vals = [float(v) for v in vals if v not in ("", None)]
            dense_ffn = sum(vals) / len(vals) if vals else ""
        compute_rows.append(
            {
                "model_name": (dawn_val or {}).get("model_name", "dawn_srw"),
                "val_loss": (dawn_val or {}).get("val_loss", ""),
                "active_rw_flops_per_token": (active or {}).get("mean_estimated_rw_active_flops_per_token", ""),
                "dense_ffn_flops_per_token": dense_ffn or dense_ffn_flops_per_token(dawn_info),
                "dense_tf_core_flops_per_token_est": dense_core_flops_per_token(tf_info, tf_val),
                "dense_ffn_equiv_ratio": (active or {}).get("mean_dense_ffn_equiv_ratio", ""),
                "active_positions": (active or {}).get("total_positions", ""),
                "note": "validation-set DAWN-SRW active RW estimate",
            }
        )

    write_csv(model_rows, args.output, "model_table.csv")
    write_csv(validation_rows, args.output, "validation_table.csv")
    write_csv(compute_rows, args.output, "loss_vs_active_compute_table.csv")
    save_json(
        {
            "model_table": model_rows,
            "validation_table": validation_rows,
            "loss_vs_active_compute_table": compute_rows,
            "active_json": args.active_json,
        },
        args.output,
        "paper_tables.json",
    )

    print("=== Aggregated Paper Tables ===")
    print(f"output: {args.output}")
    print_validation_table(validation_rows)
    print_compute_table(compute_rows)


if __name__ == "__main__":
    main()

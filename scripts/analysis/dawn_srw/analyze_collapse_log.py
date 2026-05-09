#!/usr/bin/env python3
"""Summarize DAWN-SRW collapse diagnostics from text or JSONL logs.

The parser is intentionally tolerant: it understands the new debug_log_*.txt
diagnostic lines, the existing training_log_*.txt human lines, and JSONL
records emitted by train_jax.py.  Missing values are left blank in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


CSV_COLUMNS = [
    "step", "loss", "ce", "loss_minus_ce", "grad", "lr",
    "active_qk", "active_v", "active_rst",
    "dead_a", "dead_rst",
    "int_max_a", "int_max_rst",
    "gate_eff_a", "gate_eff_rst",
    "top1_a", "top1_rst",
    "gate_den_sum_a", "gate_den_sum_rst",
    "tau_mean_a", "tau_mean_rst", "tau_abs_a", "tau_abs_rst",
    "tau_off_min_a", "tau_off_p01_a", "tau_off_p99_a", "tau_off_max_a",
    "tau_off_min_rst", "tau_off_p01_rst", "tau_off_p99_rst", "tau_off_max_rst",
    "score_std_a", "score_std_rst",
    "emb_max_qk", "emb_max_v", "emb_max_rst",
    "rw_qk_read_max", "rw_qk_write_max",
    "rw_v_read_max", "rw_v_write_max",
    "rw_rst_read_max", "rw_rst_write_max",
    "op_gain_qk_max", "op_gain_v_max", "op_gain_rst_max",
    "per_layer_last_attn", "per_layer_last_rst",
    "rpe_expl", "rpe_w",
]


ALIASES = {
    "total_loss": "loss",
    "ce_loss": "ce",
    "total_loss_minus_ce": "loss_minus_ce",
    "grad_norm": "grad",
    "grad_global_preclip": "grad",
    "attn_qk_active": "active_qk",
    "attn_v_active": "active_v",
    "rst_active": "active_rst",
    "attn_dead_count": "dead_a",
    "rst_dead_count": "dead_rst",
    "attn_int_max": "int_max_a",
    "rst_int_max": "int_max_rst",
    "attn_gate_eff_n": "gate_eff_a",
    "rst_gate_eff_n": "gate_eff_rst",
    "attn_top1_gate_frac": "top1_a",
    "rst_top1_gate_frac": "top1_rst",
    "attn_gate_den_sum_mean": "gate_den_sum_a",
    "rst_gate_den_sum_mean": "gate_den_sum_rst",
    "attn_tau_mean": "tau_mean_a",
    "rst_tau_mean": "tau_mean_rst",
    "attn_tau_abs_mean": "tau_abs_a",
    "rst_tau_abs_mean": "tau_abs_rst",
    "attn_tau_off_min": "tau_off_min_a",
    "attn_tau_off_p01": "tau_off_p01_a",
    "attn_tau_off_p99": "tau_off_p99_a",
    "attn_tau_off_max": "tau_off_max_a",
    "rst_tau_off_min": "tau_off_min_rst",
    "rst_tau_off_p01": "tau_off_p01_rst",
    "rst_tau_off_p99": "tau_off_p99_rst",
    "rst_tau_off_max": "tau_off_max_rst",
    "attn_score_std": "score_std_a",
    "rst_score_std": "score_std_rst",
    "attn_qk_emb_norm_max": "emb_max_qk",
    "attn_v_emb_norm_max": "emb_max_v",
    "rst_emb_norm_max": "emb_max_rst",
    "attn_qk_read_norm_max": "rw_qk_read_max",
    "attn_qk_write_norm_max": "rw_qk_write_max",
    "attn_v_read_norm_max": "rw_v_read_max",
    "attn_v_write_norm_max": "rw_v_write_max",
    "rst_read_norm_max": "rw_rst_read_max",
    "rst_write_norm_max": "rw_rst_write_max",
    "attn_qk_op_gain_max": "op_gain_qk_max",
    "attn_v_op_gain_max": "op_gain_v_max",
    "rst_op_gain_max": "op_gain_rst_max",
    "exploration_loss_raw_total": "rpe_expl",
    "exploration_loss_weighted_total": "rpe_w",
    "explore_loss_raw": "rpe_expl",
    "explore_loss_weighted": "rpe_w",
}


def _open_text(path: str) -> Iterable[str]:
    if path.startswith("gs://"):
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:
            raise SystemExit("Reading gs:// paths requires gcsfs.") from exc
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rt") as fh:
            yield from fh
        return
    if path.startswith("https://storage.cloud.google.com/"):
        gs_path = "gs://" + path.split("https://storage.cloud.google.com/", 1)[1]
        yield from _open_text(gs_path)
        return
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def _num(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _put(rec: Dict[str, float], key: str, value: object) -> None:
    val = _num(value)
    if val is not None:
        rec[ALIASES.get(key, key)] = val


def _grab(line: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, line)
    if not m:
        return None
    return _num(m.group(1))


def _update_pairs(rec: Dict[str, float], line: str) -> None:
    for key, value in re.findall(rf"\b([A-Za-z_][A-Za-z0-9_]*)=({FLOAT})", line):
        _put(rec, key, value)


def _parse_json(line: str) -> Optional[Tuple[int, Dict[str, float]]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    step = _num(obj.get("step"))
    if step is None:
        return None
    rec: Dict[str, float] = {}
    for key, value in obj.items():
        if key == "step":
            continue
        if key == "per_layer_attn_out_norm" and isinstance(value, list) and value:
            _put(rec, "per_layer_last_attn", value[-1])
        elif key == "per_layer_rst_out_norm" and isinstance(value, list) and value:
            _put(rec, "per_layer_last_rst", value[-1])
        else:
            _put(rec, key, value)
    return int(step), rec


def _parse_line(line: str, current_step: Optional[int]) -> Tuple[Optional[int], Dict[str, float]]:
    parsed_json = _parse_json(line)
    if parsed_json:
        return parsed_json

    rec: Dict[str, float] = {}
    m_step = re.search(
        r"(?:\[Step\s+|\[DEBUG_DIAG\]\s*step=|analysis_diag:\s*step=)(\d+)",
        line,
    )
    if m_step:
        current_step = int(m_step.group(1))
    if current_step is None:
        return None, rec

    _update_pairs(rec, line)

    if "[Step" in line or "[DEBUG_DIAG]" in line:
        for key, pat in {
            "loss": rf"loss=({FLOAT})",
            "ce": rf"ce=({FLOAT})",
            "grad": rf"grad(?:ient)?(?:_pre)?=({FLOAT})",
            "lr": rf"lr=({FLOAT})",
        }.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("loss_terms:"):
        for key, pat in {
            "ce": rf"\bce=({FLOAT})",
            "loss_minus_ce": rf"total_minus_ce=({FLOAT})",
            "rpe_expl": rf"expl_raw=({FLOAT})",
            "rpe_w": rf"expl_w=({FLOAT})",
        }.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("dead_diag:"):
        for key, pat in {
            "dead_a": rf"a_count=({FLOAT})",
            "dead_rst": rf"rst_count=({FLOAT})",
        }.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("route_diag:"):
        patterns = {
            "active_qk": rf"active_frac\[qk=({FLOAT})",
            "active_v": rf"active_frac\[qk={FLOAT}\s+v=({FLOAT})",
            "active_rst": rf"active_frac\[qk={FLOAT}\s+v={FLOAT}\s+rst=({FLOAT})",
            "gate_den_sum_a": rf"den\[attn=({FLOAT})",
            "gate_den_sum_rst": rf"den\[attn={FLOAT}\s+rst=({FLOAT})",
            "gate_eff_a": rf"eff\[attn=({FLOAT})/",
            "gate_eff_rst": rf"rst=({FLOAT})/",
            "top1_a": rf"top1\[attn_m=({FLOAT})",
            "top1_rst": rf"rst_m=({FLOAT})",
        }
        for key, pat in patterns.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("tau_diag:"):
        patterns = {
            "score_std_a": rf"score_std\[attn=({FLOAT})",
            "score_std_rst": rf"score_std\[attn={FLOAT}\s+rst=({FLOAT})",
            "tau_abs_a": rf"tau_abs\[attn=({FLOAT})",
            "tau_abs_rst": rf"tau_abs\[attn={FLOAT}\s+rst=({FLOAT})",
            "tau_off_min_a": rf"tau_offset_attn\[min=({FLOAT})",
            "tau_off_p01_a": rf"tau_offset_attn\[min={FLOAT}\s+p01=({FLOAT})",
            "tau_off_p99_a": rf"p99=({FLOAT})",
            "tau_off_max_a": rf"tau_offset_attn\[.*?max=({FLOAT})",
            "tau_off_min_rst": rf"tau_offset_rst\[min=({FLOAT})",
            "tau_off_p01_rst": rf"tau_offset_rst\[min={FLOAT}\s+p01=({FLOAT})",
            "tau_off_p99_rst": rf"tau_offset_rst\[.*?p99=({FLOAT})",
            "tau_off_max_rst": rf"tau_offset_rst\[.*?max=({FLOAT})",
        }
        for key, pat in patterns.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("amp_diag:"):
        patterns = {
            "int_max_a": rf"int_max\[attn=({FLOAT})",
            "int_max_rst": rf"int_max\[attn={FLOAT}\s+rst=({FLOAT})",
            "op_gain_qk_max": rf"op_gain_max\[qk=({FLOAT})",
            "op_gain_v_max": rf"op_gain_max\[qk={FLOAT}\s+v=({FLOAT})",
            "op_gain_rst_max": rf"op_gain_max\[qk={FLOAT}\s+v={FLOAT}\s+rst=({FLOAT})",
            "rw_qk_read_max": rf"qk_r=({FLOAT})",
            "rw_qk_write_max": rf"qk_w=({FLOAT})",
            "rw_v_read_max": rf"v_r=({FLOAT})",
            "rw_v_write_max": rf"v_w=({FLOAT})",
            "rw_rst_read_max": rf"rst_r=({FLOAT})",
            "rw_rst_write_max": rf"rst_w=({FLOAT})",
        }
        for key, pat in patterns.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("out_diag:"):
        for key, pat in {
            "per_layer_last_attn": rf"layer_attn_max=({FLOAT})",
            "per_layer_last_rst": rf"layer_rst_max=({FLOAT})",
        }.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    if line.startswith("expl_diag:"):
        for key, pat in {
            "rpe_expl": rf"total=({FLOAT})\]",
            "rpe_w": rf"weighted\[.*?total=({FLOAT})",
        }.items():
            val = _grab(line, pat)
            if val is not None:
                rec[key] = val

    return current_step, rec


def parse_log(path: str) -> List[Dict[str, float]]:
    by_step: Dict[int, Dict[str, float]] = {}
    current_step: Optional[int] = None
    for raw in _open_text(path):
        line = raw.strip()
        if not line:
            continue
        step, rec = _parse_line(line, current_step)
        if step is not None:
            current_step = step
        if step is None or not rec:
            continue
        dst = by_step.setdefault(step, {"step": float(step)})
        dst.update(rec)
    return [by_step[k] for k in sorted(by_step)]


def _series(rows: List[Dict[str, float]], key: str) -> List[float]:
    return [r[key] for r in rows if key in r and math.isfinite(r[key])]


def _first_abnormal(rows: List[Dict[str, float]]) -> Optional[int]:
    baselines = {
        key: median(vals) if vals else 0.0
        for key, vals in (
            ("grad", _series(rows[: max(5, len(rows) // 5)], "grad")),
            ("loss", _series(rows[: max(5, len(rows) // 5)], "loss")),
            ("dead_a", _series(rows[: max(5, len(rows) // 5)], "dead_a")),
            ("dead_rst", _series(rows[: max(5, len(rows) // 5)], "dead_rst")),
        )
    }
    for row in rows:
        dead_total = row.get("dead_a", 0.0) + row.get("dead_rst", 0.0)
        if row.get("grad", 0.0) > 50.0 or row.get("grad", 0.0) > baselines["grad"] * 3.0:
            return int(row["step"])
        if row.get("loss_minus_ce", 0.0) > 5.0:
            return int(row["step"])
        if row.get("loss", 0.0) > 20.0 or row.get("loss", 0.0) > baselines["loss"] * 2.0:
            return int(row["step"])
        if (dead_total > 100.0
                or dead_total > baselines["dead_a"] + baselines["dead_rst"] + 50.0):
            return int(row["step"])
    return None


def _detect_collapse(rows: List[Dict[str, float]]) -> Optional[int]:
    return _first_abnormal(rows)


def _slope(rows: List[Dict[str, float]], key: str) -> float:
    pts = [(r["step"], r[key]) for r in rows if key in r]
    if len(pts) < 2:
        return 0.0
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return (y1 - y0) / max(x1 - x0, 1.0)


def _top_cause(window: List[Dict[str, float]]) -> str:
    if not window:
        return "unknown"
    last = window[-1]
    dead_total = last.get("dead_a", 0.0) + last.get("dead_rst", 0.0)
    op_gain = max(last.get("op_gain_qk_max", 0.0),
                  last.get("op_gain_v_max", 0.0),
                  last.get("op_gain_rst_max", 0.0))
    top1 = max(last.get("top1_a", 0.0), last.get("top1_rst", 0.0))
    int_max = max(last.get("int_max_a", 0.0), last.get("int_max_rst", 0.0))
    if dead_total > 100 or _slope(window, "dead_a") + _slope(window, "dead_rst") > 0:
        return "dead penalty trigger"
    if op_gain > 20:
        return "operator norm amplification"
    if top1 > 0.5 or min(last.get("gate_eff_a", 1.0), last.get("gate_eff_rst", 1.0)) < 1.0:
        return "routing threshold bifurcation"
    if last.get("ce", 0.0) > 20 or last.get("loss", 0.0) > 20:
        return "CE/logit explosion"
    if int_max > 90:
        return "intensity saturation"
    return "unknown"


def write_csv(rows: List[Dict[str, float]], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (int(row[key]) if key == "step" and key in row else row.get(key, ""))
                for key in CSV_COLUMNS
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = parse_log(args.log_path)
    if not rows:
        raise SystemExit("No scalar rows parsed.")

    collapse_step = _detect_collapse(rows)
    if collapse_step is None:
        collapse_step = int(rows[-1]["step"])
    first_abnormal = _first_abnormal(rows)
    window_rows = [
        row for row in rows
        if collapse_step - args.window <= row["step"] <= collapse_step
    ]
    write_csv(window_rows, args.out_csv)

    watched = [
        "loss", "ce", "loss_minus_ce", "grad", "dead_a", "dead_rst",
        "op_gain_qk_max", "op_gain_v_max", "op_gain_rst_max",
        "top1_a", "top1_rst", "int_max_a", "int_max_rst", "rpe_w",
    ]
    slopes = {key: _slope(window_rows, key) for key in watched}
    top_slopes = sorted(slopes.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

    print(f"collapse step: {collapse_step}")
    print(f"first abnormal signal step: {first_abnormal or 'not detected'}")
    print(f"top suspected cause: {_top_cause(window_rows)}")
    print("largest pre-collapse slopes:")
    for key, val in top_slopes:
        print(f"  {key}: {val:+.6g}/step")
    print(f"csv: {args.out_csv}")


if __name__ == "__main__":
    main()

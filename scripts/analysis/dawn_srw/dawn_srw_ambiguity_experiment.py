#!/usr/bin/env python3
"""DAWN-SRW ambiguity / context-dependent RST model-decision experiment.

This experiment compares the RST Layer model decision for the same surface
string under two different contexts, e.g.:

    "I deposited money in the bank"
    "He sat by the river bank"

The main output is a layer-wise Jaccard trajectory:

    early layers  -> shared / token-level transition candidates
    deeper layers -> context-specific RST model decisions

For each layer and context pair, the script computes three overlaps:

1. active_set_jaccard
   Jaccard overlap over all SRW neurons with activation > active_threshold.
   This measures the broad selected candidate set.

2. topk_jaccard
   Jaccard overlap over the top-k neurons ranked by the requested transition
   contribution axis. Default is projection, i.e. absolute directional
   contribution to the final RST transition output.

3. effective_mass_jaccard
   Jaccard overlap over the minimal neuron set whose cumulative contribution
   mass reaches --mass-threshold. Default mass metric is abs_projection.
   This approximates the "actually utilized" operator set without fixing k.

It also saves plots:
  - <target>_jaccard_by_layer.png
  - <target>_rst_norm_by_layer.png
  - <target>_active_count_by_layer.png
  - mean_jaccard_by_layer.png  (when multiple cases share layers)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model,
    encode_prompt,
    ensure_dir,
    find_token_index,
    import_dawn_srw,
    load_checkpoint_params,
    load_config,
    load_tokenizer,
    model_cfg_from_config,
    save_json,
)
from dawn_srw_decision_probe import forward_to_rst_input, compute_rst_decision


DEFAULT_PAIRS = [
    ("bank", "I deposited money in the bank", "He sat by the river bank"),
    ("charge", "The phone needs a charge", "The police filed a charge"),
    ("light", "The room was filled with bright light", "This bag is very light"),
    ("apple", "She ate an apple after lunch", "Apple released a new product"),
]


RANKING_CHOICES = [
    "gate",
    "score",
    "intensity",
    "contribution",
    "contribution_norm",
    "projection",
    "abs_projection",
    "positive_projection",
    "alignment",
]

MASS_METRIC_CHOICES = [
    "abs_projection",
    "positive_projection",
    "contribution",
    "contribution_norm",
    "gate",
]


def _dense(x, dense_params):
    return x @ dense_params["kernel"] + dense_params["bias"]


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    return s.strip("_") or "case"


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / (len(sa | sb) + 1e-12))


def _rst_scale_scalar(pool_params) -> jnp.ndarray:
    val = pool_params.get("rst_scale", jnp.asarray([1.0]))
    arr = jnp.asarray(val, dtype=jnp.float32)
    return arr.reshape(-1)[0]


def _effective_mass_ids(metric: np.ndarray, mass_threshold: float) -> List[int]:
    """Return smallest descending set whose cumulative positive mass >= threshold."""
    metric = np.asarray(metric, dtype=np.float64)
    metric = np.where(np.isfinite(metric), metric, 0.0)
    metric = np.maximum(metric, 0.0)
    total = float(metric.sum())
    if total <= 1e-12:
        return []
    order = np.argsort(-metric)
    cumsum = np.cumsum(metric[order]) / total
    n = int(np.searchsorted(cumsum, mass_threshold, side="left") + 1)
    n = max(1, min(n, len(order)))
    return [int(i) for i in order[:n]]


def _top_ids(metric: np.ndarray, k: int) -> List[int]:
    metric = np.asarray(metric, dtype=np.float64)
    metric = np.where(np.isfinite(metric), metric, -np.inf)
    if metric.size == 0:
        return []
    k = min(int(k), metric.size)
    return [int(i) for i in np.argsort(-metric)[:k]]


def compute_rst_decision_vectors(
    mod,
    params,
    state: Dict[str, jnp.ndarray],
    token_index: int,
    *,
    top_k: int,
    sort_by: str,
    active_threshold: float,
    mass_threshold: float,
    mass_metric: str,
) -> Dict[str, object]:
    """Compute full RST decision vectors for set/Jaccard analysis.

    This mirrors DAWN-SRW's RST implementation:

        scores = h @ rst_emb.T
        tau = mean(scores) + tau_offset * std(scores) - scan_offset/std
        activation = sigmoid(sharpness * ((score - tau) - threshold))
        intensity = eps + min(max(margin - cutoff, 0), max_intensity)
        gate = activation * intensity
        contribution_i = (rst_scale / den) * gate_i * <x, r_i> * w_i

    The returned sets distinguish:
      - selected candidates: activation > active_threshold
      - top-k contributors under sort_by
      - effective-mass contributors under mass_metric
    """
    pp = params["neuron_pool"]

    x = state["rst_input_normed"][0, token_index, :].astype(jnp.float32)
    h = state["rst_route_query"][0, token_index, :].astype(jnp.float32)
    tau_offset = state["rst_tau_offset"][0, token_index, 0].astype(jnp.float32)
    raw_scan_offset = state["rst_raw_scan_offset"][0, token_index, 0].astype(jnp.float32)

    emb = pp["rst_emb"].astype(jnp.float32)
    read = pp["rst_read"].astype(jnp.float32)
    write = pp["rst_write"].astype(jnp.float32)
    rst_scale = _rst_scale_scalar(pp)

    scores = h @ emb.T
    s_mean = scores.mean()
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores - s_mean))) + 1e-8
    scan_offset = mod.SCAN_SCALE * jnp.tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, mod.SCAN_STD_FLOOR)

    raw = scores - tau
    margin = raw - mod.ACTIVATION_THRESHOLD
    activation = jax.nn.sigmoid(mod.SHARPNESS * margin)
    active_margin = jnp.maximum(margin - mod.ACTIVATION_CUTOFF, 0.0)
    intensity = mod.EPSILON + jnp.minimum(active_margin, mod.MAX_INTENSITY)
    gate = activation * intensity

    read_value = read @ x
    coeff = gate * read_value
    den = jnp.maximum(gate.sum(), 1.0)
    scaled_coeff = rst_scale * coeff / den
    contribution_vectors = scaled_coeff[:, None] * write
    contribution_norm = jnp.linalg.norm(contribution_vectors, axis=-1)

    rst_output_vec = state["rst_output"][0, token_index, :].astype(jnp.float32)
    rst_output_norm = jnp.linalg.norm(rst_output_vec) + 1e-8
    rst_unit = rst_output_vec / rst_output_norm
    signed_projection = contribution_vectors @ rst_unit
    abs_projection = jnp.abs(signed_projection)
    positive_projection = jnp.maximum(signed_projection, 0.0)
    alignment_cosine = signed_projection / (contribution_norm + 1e-8)

    metrics = jax.device_get({
        "scores": scores,
        "activation": activation,
        "intensity": intensity,
        "gate": gate,
        "read_value": read_value,
        "contribution_norm": contribution_norm,
        "signed_projection": signed_projection,
        "abs_projection": abs_projection,
        "positive_projection": positive_projection,
        "alignment_cosine": alignment_cosine,
        "score_mean": s_mean,
        "score_std": s_std,
        "tau_offset": tau_offset,
        "raw_scan_offset": raw_scan_offset,
        "scan_offset": scan_offset,
        "tau": tau,
        "gate_sum": gate.sum(),
        "denominator": den,
        "rst_scale": rst_scale,
        "active_count": (activation > active_threshold).sum(),
        "intensity_cap_count": (intensity >= (mod.MAX_INTENSITY + mod.EPSILON - 1e-3)).sum(),
        "gate_entropy": -(gate / (gate.sum() + 1e-8) * jnp.log(gate / (gate.sum() + 1e-8) + 1e-8)).sum(),
        "sum_contribution_norm": contribution_norm.sum(),
        "sum_abs_projection": abs_projection.sum(),
        "sum_positive_projection": positive_projection.sum(),
        "rst_output_norm": jnp.linalg.norm(state["rst_output"][0, token_index, :]),
        "attention_output_norm": jnp.linalg.norm(state["attention_output"][0, token_index, :]),
    })

    def metric_for(name: str) -> np.ndarray:
        if name in ("contribution", "contribution_norm"):
            return np.asarray(metrics["contribution_norm"])
        if name == "score":
            return np.asarray(metrics["scores"])
        if name == "intensity":
            return np.asarray(metrics["intensity"])
        if name == "positive_projection":
            return np.asarray(metrics["positive_projection"])
        if name in ("projection", "abs_projection"):
            return np.asarray(metrics["abs_projection"])
        if name == "alignment":
            return np.asarray(metrics["alignment_cosine"])
        return np.asarray(metrics["gate"])

    active_ids = np.where(np.asarray(metrics["activation"]) > active_threshold)[0].astype(np.int64).tolist()
    topk_ids = _top_ids(metric_for(sort_by), top_k)
    effective_ids = _effective_mass_ids(metric_for(mass_metric), mass_threshold)

    summary = {
        "score_mean": float(metrics["score_mean"]),
        "score_std": float(metrics["score_std"]),
        "tau_offset": float(metrics["tau_offset"]),
        "raw_scan_offset": float(metrics["raw_scan_offset"]),
        "scan_offset": float(metrics["scan_offset"]),
        "tau": float(metrics["tau"]),
        "gate_sum": float(metrics["gate_sum"]),
        "denominator": float(metrics["denominator"]),
        "rst_scale": float(metrics["rst_scale"]),
        "active_count": int(metrics["active_count"]),
        "intensity_cap_count": int(metrics["intensity_cap_count"]),
        "gate_entropy": float(metrics["gate_entropy"]),
        "sum_contribution_norm": float(metrics["sum_contribution_norm"]),
        "sum_abs_projection": float(metrics["sum_abs_projection"]),
        "sum_positive_projection": float(metrics["sum_positive_projection"]),
        "attention_output_norm_at_token": float(metrics["attention_output_norm"]),
        "rst_output_norm_at_token": float(metrics["rst_output_norm"]),
        "active_threshold": float(active_threshold),
        "mass_threshold": float(mass_threshold),
        "mass_metric": mass_metric,
        "topk_metric": sort_by,
        "active_set_size": int(len(active_ids)),
        "topk_set_size": int(len(topk_ids)),
        "effective_mass_set_size": int(len(effective_ids)),
    }

    return {
        "summary": summary,
        "active_ids": active_ids,
        "topk_ids": topk_ids,
        "effective_ids": effective_ids,
    }


def run_pair(
    mod,
    params,
    model_cfg,
    tokenizer,
    target: str,
    prompt_a: str,
    prompt_b: str,
    layers: Sequence[int],
    top_k: int,
    sort_by: str,
    active_threshold: float,
    mass_threshold: float,
    mass_metric: str,
    include_top_rows: bool,
):
    out = {
        "target": target,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "layers": [],
    }

    encoded = []
    for prompt in [prompt_a, prompt_b]:
        ids = encode_prompt(tokenizer, prompt, model_cfg["max_seq_len"])
        tokens = tokenizer.convert_ids_to_tokens(ids)
        idx = find_token_index(tokens, target, default="last")
        encoded.append((ids, tokens, idx))

    out["tokens_a"] = encoded[0][1]
    out["tokens_b"] = encoded[1][1]
    out["token_index_a"] = encoded[0][2]
    out["token_index_b"] = encoded[1][2]

    for layer in layers:
        decision_items = []
        for ids, tokens, idx in encoded:
            state = forward_to_rst_input(
                mod,
                params,
                model_cfg,
                jnp.asarray([ids], dtype=jnp.int32),
                layer,
            )
            dec = compute_rst_decision_vectors(
                mod,
                params,
                state,
                idx,
                top_k=top_k,
                sort_by=sort_by,
                active_threshold=active_threshold,
                mass_threshold=mass_threshold,
                mass_metric=mass_metric,
            )
            item = {
                "token_index": int(idx),
                "token_text": tokens[idx],
                "summary": dec["summary"],
                "active_ids": dec["active_ids"],
                "topk_ids": dec["topk_ids"],
                "effective_mass_ids": dec["effective_ids"],
            }
            if include_top_rows:
                probe_summary, top_rows = compute_rst_decision(
                    mod, params, state, idx, top_k=top_k, sort_by=sort_by
                )
                item["top_selected_srw_neurons"] = top_rows
            decision_items.append(item)

        a, b = decision_items
        active_j = jaccard(a["active_ids"], b["active_ids"])
        topk_j = jaccard(a["topk_ids"], b["topk_ids"])
        eff_j = jaccard(a["effective_mass_ids"], b["effective_mass_ids"])

        row = {
            "layer": int(layer),
            "top_k": int(top_k),
            "topk_metric": sort_by,
            "mass_metric": mass_metric,
            "mass_threshold": float(mass_threshold),
            "active_threshold": float(active_threshold),
            "active_set_jaccard": active_j,
            "topk_jaccard": topk_j,
            "effective_mass_jaccard": eff_j,
            "active_overlap_count": len(set(a["active_ids"]) & set(b["active_ids"])),
            "topk_overlap_count": len(set(a["topk_ids"]) & set(b["topk_ids"])),
            "effective_mass_overlap_count": len(set(a["effective_mass_ids"]) & set(b["effective_mass_ids"])),
            "active_union_count": len(set(a["active_ids"]) | set(b["active_ids"])),
            "topk_union_count": len(set(a["topk_ids"]) | set(b["topk_ids"])),
            "effective_mass_union_count": len(set(a["effective_mass_ids"]) | set(b["effective_mass_ids"])),
            "rst_output_norm_delta_abs": abs(
                a["summary"]["rst_output_norm_at_token"]
                - b["summary"]["rst_output_norm_at_token"]
            ),
            "gate_sum_delta_abs": abs(a["summary"]["gate_sum"] - b["summary"]["gate_sum"]),
            "a": a,
            "b": b,
        }
        out["layers"].append(row)

    return out


def _write_case_csv(case: Dict[str, object], output_dir: str) -> str:
    path = Path(output_dir) / f"{_safe_name(case['target'])}_jaccard_by_layer.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "layer",
        "active_set_jaccard",
        "topk_jaccard",
        "effective_mass_jaccard",
        "active_overlap_count",
        "topk_overlap_count",
        "effective_mass_overlap_count",
        "a_active_count",
        "b_active_count",
        "a_gate_entropy",
        "b_gate_entropy",
        "a_rst_norm",
        "b_rst_norm",
        "a_attn_norm",
        "b_attn_norm",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in case["layers"]:
            a = row["a"]["summary"]
            b = row["b"]["summary"]
            w.writerow({
                "target": case["target"],
                "layer": row["layer"],
                "active_set_jaccard": row["active_set_jaccard"],
                "topk_jaccard": row["topk_jaccard"],
                "effective_mass_jaccard": row["effective_mass_jaccard"],
                "active_overlap_count": row["active_overlap_count"],
                "topk_overlap_count": row["topk_overlap_count"],
                "effective_mass_overlap_count": row["effective_mass_overlap_count"],
                "a_active_count": a["active_count"],
                "b_active_count": b["active_count"],
                "a_gate_entropy": a["gate_entropy"],
                "b_gate_entropy": b["gate_entropy"],
                "a_rst_norm": a["rst_output_norm_at_token"],
                "b_rst_norm": b["rst_output_norm_at_token"],
                "a_attn_norm": a["attention_output_norm_at_token"],
                "b_attn_norm": b["attention_output_norm_at_token"],
            })
    return str(path)


def _plot_case(case: Dict[str, object], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped: matplotlib unavailable ({exc})")
        return []

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = str(case["target"])
    stem = _safe_name(target)
    layers = np.asarray([r["layer"] for r in case["layers"]], dtype=np.int32)

    paths = []

    # Jaccard trajectory.
    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, [r["active_set_jaccard"] for r in case["layers"]], marker="o", label="active set")
    plt.plot(layers, [r["topk_jaccard"] for r in case["layers"]], marker="o", label="top-k projected contributors")
    plt.plot(layers, [r["effective_mass_jaccard"] for r in case["layers"]], marker="o", label="effective-mass contributors")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Layer")
    plt.ylabel("Jaccard overlap")
    plt.title(f"RST model-decision overlap for '{target}'")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = outdir / f"{stem}_jaccard_by_layer.png"
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(str(p))

    # RST norm by layer.
    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, [r["a"]["summary"]["rst_output_norm_at_token"] for r in case["layers"]], marker="o", label="prompt A RST norm")
    plt.plot(layers, [r["b"]["summary"]["rst_output_norm_at_token"] for r in case["layers"]], marker="o", label="prompt B RST norm")
    plt.plot(layers, [r["a"]["summary"]["attention_output_norm_at_token"] for r in case["layers"]], linestyle="--", alpha=0.7, label="prompt A attention norm")
    plt.plot(layers, [r["b"]["summary"]["attention_output_norm_at_token"] for r in case["layers"]], linestyle="--", alpha=0.7, label="prompt B attention norm")
    plt.xlabel("Layer")
    plt.ylabel("Output norm at target token")
    plt.title(f"Attention vs RST transition norm for '{target}'")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = outdir / f"{stem}_rst_norm_by_layer.png"
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(str(p))

    # Active counts by layer.
    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, [r["a"]["summary"]["active_count"] for r in case["layers"]], marker="o", label="prompt A active")
    plt.plot(layers, [r["b"]["summary"]["active_count"] for r in case["layers"]], marker="o", label="prompt B active")
    plt.xlabel("Layer")
    plt.ylabel("Active SRW neurons")
    plt.title(f"RST active set size for '{target}'")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = outdir / f"{stem}_active_count_by_layer.png"
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(str(p))

    return paths


def _plot_mean(results: Dict[str, object], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  mean plot skipped: matplotlib unavailable ({exc})")
        return None

    cases = results.get("cases", [])
    if not cases:
        return None
    layer_lists = [[r["layer"] for r in c["layers"]] for c in cases]
    if any(ll != layer_lists[0] for ll in layer_lists):
        return None
    layers = np.asarray(layer_lists[0], dtype=np.int32)

    def mean_line(key: str):
        arr = np.asarray([[r[key] for r in c["layers"]] for c in cases], dtype=np.float64)
        return arr.mean(axis=0), arr.std(axis=0)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for key, label in [
        ("active_set_jaccard", "active set"),
        ("topk_jaccard", "top-k contributors"),
        ("effective_mass_jaccard", "effective-mass contributors"),
    ]:
        mean, std = mean_line(key)
        plt.plot(layers, mean, marker="o", label=label)
        plt.fill_between(layers, mean - std, mean + std, alpha=0.15)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Layer")
    plt.ylabel("Mean Jaccard overlap")
    plt.title("Mean context-dependent RST decision overlap")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = outdir / "mean_jaccard_by_layer.png"
    plt.savefig(p, dpi=180)
    plt.close()
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="results/dawn_srw_ambiguity")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sort-by", choices=RANKING_CHOICES, default="projection")
    ap.add_argument("--mass-metric", choices=MASS_METRIC_CHOICES, default="abs_projection")
    ap.add_argument("--mass-threshold", type=float, default=0.80)
    ap.add_argument("--active-threshold", type=float, default=0.5)
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices, e.g. 0,5,11")
    ap.add_argument("--case", action="append", default=[], help="Custom case: target|||prompt A|||prompt B")
    ap.add_argument("--include-top-rows", action="store_true", help="Also save full top-k rows per layer/context. Larger JSON.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    tokenizer = load_tokenizer(args.tokenizer)

    if args.layers == "all":
        layers = list(range(model_cfg["n_layers"]))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    cases = []
    for c in args.case:
        parts = c.split("|||")
        if len(parts) != 3:
            raise ValueError("--case must be: target|||prompt A|||prompt B")
        cases.append(tuple(parts))
    if not cases:
        cases = DEFAULT_PAIRS

    ensure_dir(args.output)

    results = {
        "checkpoint": meta,
        "top_k": args.top_k,
        "sort_by": args.sort_by,
        "mass_metric": args.mass_metric,
        "mass_threshold": args.mass_threshold,
        "active_threshold": args.active_threshold,
        "layers": layers,
        "interpretation": {
            "active_set_jaccard": "overlap over neurons with activation > active_threshold",
            "topk_jaccard": "overlap over top-k neurons by --sort-by metric",
            "effective_mass_jaccard": "overlap over minimal neuron set covering --mass-threshold of --mass-metric mass",
        },
        "cases": [],
        "plots": [],
        "csv": [],
    }

    for target, a, b in cases:
        print(f"\n=== target={target!r} ===")
        print(f"A: {a}")
        print(f"B: {b}")
        item = run_pair(
            mod,
            params,
            model_cfg,
            tokenizer,
            target,
            a,
            b,
            layers,
            args.top_k,
            args.sort_by,
            args.active_threshold,
            args.mass_threshold,
            args.mass_metric,
            args.include_top_rows,
        )
        results["cases"].append(item)
        for row in item["layers"]:
            print(
                f"  L{row['layer']:02d}: "
                f"J_active={row['active_set_jaccard']:.3f} "
                f"J_topk={row['topk_jaccard']:.3f} "
                f"J_eff={row['effective_mass_jaccard']:.3f} | "
                f"active=({row['a']['summary']['active_count']}, {row['b']['summary']['active_count']}) "
                f"rst=({row['a']['summary']['rst_output_norm_at_token']:.3f}, "
                f"{row['b']['summary']['rst_output_norm_at_token']:.3f})"
            )
        csv_path = _write_case_csv(item, args.output)
        results["csv"].append(csv_path)
        print(f"  csv: {csv_path}")
        if not args.no_plots:
            plot_paths = _plot_case(item, args.output)
            results["plots"].extend(plot_paths)
            for p in plot_paths:
                print(f"  plot: {p}")

    if not args.no_plots:
        mean_plot = _plot_mean(results, args.output)
        if mean_plot:
            results["plots"].append(mean_plot)
            print(f"\n  plot: {mean_plot}")

    path = save_json(results, args.output, "ambiguity_comparison.json")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

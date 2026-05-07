#!/usr/bin/env python3
"""DAWN-SRW causal RST top-contributor intervention experiment.

This experiment asks whether the RST operators ranked as top contributors to
an executed transition are locally causal for that transition.  For each
prompt/layer/token it computes the full per-operator RST contribution vectors,
then compares ablations of:

  - top contributors under --sort-by
  - random active operators
  - low-contribution active operators
  - optionally, random operators from the full RST pool

The primary ablation is ``no_renorm``:

    z_ablated = z_baseline - sum_{i in selected} contribution_i

This directly removes the measured operator contribution under the original
gate denominator.  ``renorm`` is also available and recomputes the RST output
after setting selected gates to zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model,
    encode_prompt,
    ensure_dir,
    find_token_index,
    import_dawn_srw,
    is_gcs,
    load_checkpoint_params,
    load_config,
    load_tokenizer,
    model_cfg_from_config,
    open_file,
    save_json,
)
from dawn_srw_decision_probe import forward_to_rst_input


DEFAULT_CASES = [
    {
        "target": "bank",
        "prompt": "I deposited money in the bank",
        "prompt_label": "bank_financial",
    },
    {
        "target": "bank",
        "prompt": "He sat by the river bank",
        "prompt_label": "bank_river",
    },
    {
        "target": "charge",
        "prompt": "The phone needs a charge",
        "prompt_label": "charge_device",
    },
    {
        "target": "charge",
        "prompt": "The police filed a charge",
        "prompt_label": "charge_legal",
    },
    {
        "target": "light",
        "prompt": "The room was filled with bright light",
        "prompt_label": "light_illumination",
    },
    {
        "target": "light",
        "prompt": "This bag is very light",
        "prompt_label": "light_weight",
    },
    {
        "target": "apple",
        "prompt": "She ate an apple after lunch",
        "prompt_label": "apple_fruit",
    },
    {
        "target": "apple",
        "prompt": "Apple released a new product",
        "prompt_label": "apple_company",
    },
]


RANKING_CHOICES = [
    "gate",
    "score",
    "intensity",
    "contribution_norm",
    "abs_projection",
    "positive_projection",
    "alignment",
]

CSV_FIELDS = [
    "target",
    "prompt",
    "prompt_label",
    "layer",
    "intervention_type",
    "repeat_id",
    "ablation_mode",
    "set_size",
    "baseline_rst_norm",
    "ablated_rst_norm",
    "removed_contribution_norm",
    "delta_norm",
    "relative_delta_norm",
    "cosine_baseline_ablated",
    "removed_signed_projection_sum",
    "removed_abs_projection_sum",
    "removed_positive_projection_sum",
    "removed_mass_fraction_abs",
    "removed_mass_fraction_positive",
    "active_count",
    "gate_sum",
    "total_abs_projection",
    "total_positive_projection",
]

AGG_METRICS = [
    "delta_norm",
    "relative_delta_norm",
    "cosine_baseline_ablated",
    "removed_mass_fraction_abs",
    "removed_mass_fraction_positive",
]

EPS = 1.0e-8


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    return s.strip("_") or "case"


def _out_path(output_dir: str, filename: str) -> str:
    """Join an output directory and filename without corrupting gs:// paths."""
    return str(output_dir).rstrip("/") + "/" + filename


def _write_matplotlib_figure(plt, output_dir: str, filename: str, dpi: int = 180) -> str:
    """Save the current matplotlib figure to local or GCS output_dir."""
    dest = _out_path(output_dir, filename)
    if is_gcs(dest):
        suffix = Path(filename).suffix or ".png"
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_name = tmp.name
            plt.savefig(tmp_name, dpi=dpi)
            with open(tmp_name, "rb") as src, open_file(dest, "wb") as dst:
                dst.write(src.read())
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, dpi=dpi)
    return dest


def _rst_scale_scalar(pool_params) -> jnp.ndarray:
    val = pool_params.get("rst_scale", jnp.asarray([1.0]))
    arr = jnp.asarray(val, dtype=jnp.float32)
    return arr.reshape(-1)[0]


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    if spec == "all":
        return list(range(n_layers))
    layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} is outside valid range [0, {n_layers - 1}]")
    return layers


def _parse_cases(raw_cases: Sequence[str]) -> List[Dict[str, str]]:
    cases = []
    for i, raw in enumerate(raw_cases):
        parts = [p.strip() for p in raw.split("|||")]
        if len(parts) == 2:
            target, prompt = parts
            label = f"custom_{i}_{_safe_name(target)}"
        elif len(parts) == 3:
            target, prompt, label = parts
        else:
            raise ValueError("--case must be: target|||prompt or target|||prompt|||label")
        cases.append({"target": target, "prompt": prompt, "prompt_label": label})
    return cases or list(DEFAULT_CASES)


def _metric_np(decision: Dict[str, Any], sort_by: str) -> np.ndarray:
    arrays = decision["arrays"]
    if sort_by == "score":
        arr = arrays["scores"]
    elif sort_by == "intensity":
        arr = arrays["intensity"]
    elif sort_by == "contribution_norm":
        arr = arrays["contribution_norm"]
    elif sort_by == "abs_projection":
        arr = arrays["abs_projection"]
    elif sort_by == "positive_projection":
        arr = arrays["positive_projection"]
    elif sort_by == "alignment":
        arr = arrays["alignment_cosine"]
    else:
        arr = arrays["gate"]
    return np.asarray(jax.device_get(arr), dtype=np.float64)


def _ranked_from_pool(
    metric: np.ndarray,
    pool_ids: Sequence[int],
    k: int,
    *,
    descending: bool,
) -> List[int]:
    if k <= 0 or not pool_ids:
        return []
    ids = np.asarray(pool_ids, dtype=np.int64)
    vals = np.asarray(metric[ids], dtype=np.float64)
    if descending:
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        order = np.argsort(-vals, kind="mergesort")
    else:
        vals = np.where(np.isfinite(vals), vals, np.inf)
        order = np.argsort(vals, kind="mergesort")
    return [int(x) for x in ids[order[: min(k, ids.size)]]]


def _sample_without_replacement(
    rng: np.random.Generator,
    pool_ids: Sequence[int] | np.ndarray,
    k: int,
) -> List[int]:
    if k <= 0:
        return []
    ids = np.asarray(pool_ids, dtype=np.int64)
    if ids.size == 0:
        return []
    k = min(k, ids.size)
    return [int(x) for x in rng.choice(ids, size=k, replace=False).tolist()]


def compute_rst_contribution_vectors(
    mod,
    params,
    state: Dict[str, jnp.ndarray],
    token_index: int,
    *,
    active_threshold: float,
) -> Dict[str, Any]:
    """Compute full per-operator RST contribution vectors.

    This intentionally mirrors dawn_srw_ambiguity_experiment.py and the
    implementation equation:

        contribution_i = rst_scale * gate_i * <x, read_i> / den * write_i
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
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores - s_mean))) + EPS
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
    rst_output_norm = jnp.linalg.norm(rst_output_vec)
    rst_unit = rst_output_vec / (rst_output_norm + EPS)
    signed_projection = contribution_vectors @ rst_unit
    abs_projection = jnp.abs(signed_projection)
    positive_projection = jnp.maximum(signed_projection, 0.0)
    alignment_cosine = signed_projection / (contribution_norm + EPS)

    gate_sum = gate.sum()
    gate_dist = gate / (gate_sum + EPS)
    reconstructed = contribution_vectors.sum(axis=0)
    reconstruction_error_norm = jnp.linalg.norm(reconstructed - rst_output_vec)

    summary_raw = jax.device_get({
        "score_mean": s_mean,
        "score_std": s_std,
        "tau_offset": tau_offset,
        "raw_scan_offset": raw_scan_offset,
        "scan_offset": scan_offset,
        "tau": tau,
        "gate_sum": gate_sum,
        "denominator": den,
        "rst_scale": rst_scale,
        "active_count": (activation > active_threshold).sum(),
        "intensity_cap_count": (
            intensity >= (mod.MAX_INTENSITY + mod.EPSILON - 1e-3)
        ).sum(),
        "gate_entropy": -(gate_dist * jnp.log(gate_dist + EPS)).sum(),
        "total_contribution_norm": contribution_norm.sum(),
        "total_abs_projection": abs_projection.sum(),
        "total_positive_projection": positive_projection.sum(),
        "baseline_rst_norm": rst_output_norm,
        "attention_output_norm_at_token": jnp.linalg.norm(
            state["attention_output"][0, token_index, :]
        ),
        "reconstruction_error_norm": reconstruction_error_norm,
    })
    summary = {
        key: int(value) if key.endswith("_count") or key == "active_count" else float(value)
        for key, value in summary_raw.items()
    }
    summary["active_threshold"] = float(active_threshold)

    return {
        "summary": summary,
        "arrays": {
            "scores": scores,
            "activation": activation,
            "intensity": intensity,
            "gate": gate,
            "read_value": read_value,
            "write": write,
            "rst_scale": rst_scale,
            "rst_output": rst_output_vec,
            "contribution_vectors": contribution_vectors,
            "contribution_norm": contribution_norm,
            "signed_projection": signed_projection,
            "abs_projection": abs_projection,
            "positive_projection": positive_projection,
            "alignment_cosine": alignment_cosine,
            "total_abs_projection": abs_projection.sum(),
            "total_positive_projection": positive_projection.sum(),
        },
    }


def evaluate_intervention(
    decision: Dict[str, Any],
    selected_ids: Sequence[int],
    *,
    intervention_type: str,
    repeat_id: int,
    ablation_mode: str,
) -> Dict[str, Any]:
    """Compute local causal metrics for one intervention set."""
    arrays = decision["arrays"]
    selected = [int(x) for x in selected_ids]
    idx = jnp.asarray(selected, dtype=jnp.int32)

    baseline = arrays["rst_output"]
    selected_contrib = jnp.take(arrays["contribution_vectors"], idx, axis=0)
    selected_contrib_sum = selected_contrib.sum(axis=0)

    if ablation_mode == "no_renorm":
        ablated = baseline - selected_contrib_sum
    elif ablation_mode == "renorm":
        selected_mask = jnp.zeros_like(arrays["gate"], dtype=jnp.bool_).at[idx].set(True)
        gate_ablated = jnp.where(selected_mask, 0.0, arrays["gate"])
        coeff_ablated = gate_ablated * arrays["read_value"]
        den_ablated = jnp.maximum(gate_ablated.sum(), 1.0)
        ablated = arrays["rst_scale"] * (coeff_ablated @ arrays["write"]) / den_ablated
    else:
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")

    delta_vec = baseline - ablated
    baseline_norm = jnp.linalg.norm(baseline)
    ablated_norm = jnp.linalg.norm(ablated)
    delta_norm = jnp.linalg.norm(delta_vec)
    removed_contribution_norm = jnp.linalg.norm(selected_contrib_sum)
    cosine = jnp.vdot(baseline, ablated) / (baseline_norm * ablated_norm + EPS)

    selected_signed = jnp.take(arrays["signed_projection"], idx, axis=0)
    selected_abs = jnp.take(arrays["abs_projection"], idx, axis=0)
    selected_positive = jnp.take(arrays["positive_projection"], idx, axis=0)
    removed_signed_sum = selected_signed.sum()
    removed_abs_sum = selected_abs.sum()
    removed_positive_sum = selected_positive.sum()

    metrics_raw = jax.device_get({
        "baseline_rst_norm": baseline_norm,
        "ablated_rst_norm": ablated_norm,
        "removed_contribution_norm": removed_contribution_norm,
        "delta_norm": delta_norm,
        "relative_delta_norm": delta_norm / (baseline_norm + EPS),
        "cosine_baseline_ablated": cosine,
        "removed_signed_projection_sum": removed_signed_sum,
        "removed_abs_projection_sum": removed_abs_sum,
        "removed_positive_projection_sum": removed_positive_sum,
        "removed_mass_fraction_abs": removed_abs_sum / (arrays["total_abs_projection"] + EPS),
        "removed_mass_fraction_positive": (
            removed_positive_sum / (arrays["total_positive_projection"] + EPS)
        ),
    })

    out = {
        "intervention_type": intervention_type,
        "repeat_id": int(repeat_id),
        "ablation_mode": ablation_mode,
        "set_size": int(len(selected)),
        "selected_operator_ids": selected,
        **{key: float(value) for key, value in metrics_raw.items()},
    }
    return out


def _intervention_sets_for_layer(
    decision: Dict[str, Any],
    *,
    top_k: int,
    sort_by: str,
    active_threshold: float,
    random_repeats: int,
    include_random_pool: bool,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    activation = np.asarray(jax.device_get(decision["arrays"]["activation"]), dtype=np.float64)
    active_ids = np.where(activation > active_threshold)[0].astype(np.int64).tolist()
    metric = _metric_np(decision, sort_by)

    # Keep all comparison sets the same size.  If no operators are active, all
    # interventions intentionally become empty but still produce finite metrics.
    set_size = min(int(top_k), len(active_ids))
    top_ids = _ranked_from_pool(metric, active_ids, set_size, descending=True)
    low_ids = _ranked_from_pool(metric, active_ids, set_size, descending=False)

    items = [
        {
            "intervention_type": "top_contributors",
            "repeat_id": 0,
            "selected_operator_ids": top_ids,
        },
    ]
    for repeat_id in range(max(0, int(random_repeats))):
        items.append({
            "intervention_type": "random_active",
            "repeat_id": repeat_id,
            "selected_operator_ids": _sample_without_replacement(rng, active_ids, set_size),
        })
    items.append({
        "intervention_type": "low_contributors",
        "repeat_id": 0,
        "selected_operator_ids": low_ids,
    })
    if include_random_pool:
        n_rst = int(decision["arrays"]["gate"].shape[0])
        full_pool = np.arange(n_rst, dtype=np.int64)
        for repeat_id in range(max(0, int(random_repeats))):
            items.append({
                "intervention_type": "random_pool",
                "repeat_id": repeat_id,
                "selected_operator_ids": _sample_without_replacement(rng, full_pool, set_size),
            })
    return items, active_ids


def run_case(
    mod,
    params,
    model_cfg: Dict[str, Any],
    tokenizer,
    case: Dict[str, str],
    *,
    layers: Sequence[int],
    top_k: int,
    sort_by: str,
    active_threshold: float,
    random_repeats: int,
    ablation_modes: Sequence[str],
    include_operator_ids: bool,
    include_random_pool: bool,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = case["prompt"]
    target = case["target"]
    prompt_label = case["prompt_label"]

    ids = encode_prompt(tokenizer, prompt, model_cfg["max_seq_len"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    token_index = find_token_index(tokens, target, default="last")

    out = {
        "target": target,
        "prompt": prompt,
        "prompt_label": prompt_label,
        "tokens": tokens,
        "token_index": int(token_index),
        "token_text": tokens[token_index],
        "layers": [],
    }
    summary_rows = []

    for layer in layers:
        state = forward_to_rst_input(
            mod,
            params,
            model_cfg,
            jnp.asarray([ids], dtype=jnp.int32),
            int(layer),
        )
        decision = compute_rst_contribution_vectors(
            mod,
            params,
            state,
            token_index,
            active_threshold=active_threshold,
        )
        intervention_sets, active_ids = _intervention_sets_for_layer(
            decision,
            top_k=top_k,
            sort_by=sort_by,
            active_threshold=active_threshold,
            random_repeats=random_repeats,
            include_random_pool=include_random_pool,
            rng=rng,
        )

        layer_item = {
            "layer": int(layer),
            "decision_summary": decision["summary"],
            "interventions": [],
        }
        if include_operator_ids:
            layer_item["active_ids"] = active_ids

        for item in intervention_sets:
            for mode in ablation_modes:
                metrics = evaluate_intervention(
                    decision,
                    item["selected_operator_ids"],
                    intervention_type=item["intervention_type"],
                    repeat_id=item["repeat_id"],
                    ablation_mode=mode,
                )
                layer_item["interventions"].append(metrics)
                row = {
                    "target": target,
                    "prompt": prompt,
                    "prompt_label": prompt_label,
                    "layer": int(layer),
                    **{k: metrics[k] for k in CSV_FIELDS if k in metrics},
                    "active_count": int(decision["summary"]["active_count"]),
                    "gate_sum": float(decision["summary"]["gate_sum"]),
                    "total_abs_projection": float(decision["summary"]["total_abs_projection"]),
                    "total_positive_projection": float(
                        decision["summary"]["total_positive_projection"]
                    ),
                }
                summary_rows.append(row)

        out["layers"].append(layer_item)

    return out, summary_rows


# TODO: Add an optional end-to-end continuation hook here: replace the RST
# output at (layer, token_index), add it to x_after_attention, then run the
# remaining blocks plus final LM head to compare next-token logits.


def _write_summary_csv(rows: Sequence[Dict[str, Any]], output_dir: str) -> str:
    path = _out_path(str(output_dir), "rst_intervention_summary.csv")
    if not is_gcs(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return path


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["intervention_type"]), str(row["ablation_mode"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for intervention_type, ablation_mode in sorted(grouped):
        group_rows = grouped[(intervention_type, ablation_mode)]
        agg = {
            "intervention_type": intervention_type,
            "ablation_mode": ablation_mode,
            "n_rows": int(len(group_rows)),
        }
        for metric in AGG_METRICS:
            vals = np.asarray([float(r[metric]) for r in group_rows], dtype=np.float64)
            agg[f"mean_{metric}"] = float(np.nanmean(vals)) if vals.size else math.nan
            agg[f"std_{metric}"] = float(np.nanstd(vals)) if vals.size else math.nan
        out.append(agg)
    return out


def _write_aggregate_csv(rows: Sequence[Dict[str, Any]], output_dir: str) -> str:
    path = _out_path(str(output_dir), "rst_intervention_aggregate.csv")
    if not is_gcs(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["intervention_type", "ablation_mode", "n_rows"]
    for metric in AGG_METRICS:
        fields.extend([f"mean_{metric}", f"std_{metric}"])
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _plot_metric_by_layer(
    rows: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    metric: str,
    ylabel: str,
    filename: str,
) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped: matplotlib unavailable ({exc})")
        return None

    if not rows:
        return None

    layers = sorted({int(r["layer"]) for r in rows})
    groups = sorted({(str(r["intervention_type"]), str(r["ablation_mode"])) for r in rows})

    plt.figure(figsize=(8.5, 5.0))
    for intervention_type, ablation_mode in groups:
        means = []
        stds = []
        for layer in layers:
            vals = [
                float(r[metric])
                for r in rows
                if int(r["layer"]) == layer
                and str(r["intervention_type"]) == intervention_type
                and str(r["ablation_mode"]) == ablation_mode
            ]
            arr = np.asarray(vals, dtype=np.float64)
            means.append(float(np.nanmean(arr)) if arr.size else math.nan)
            stds.append(float(np.nanstd(arr)) if arr.size else 0.0)
        x = np.asarray(layers, dtype=np.int32)
        y = np.asarray(means, dtype=np.float64)
        e = np.asarray(stds, dtype=np.float64)
        label = intervention_type if len({r["ablation_mode"] for r in rows}) == 1 else f"{intervention_type} / {ablation_mode}"
        plt.plot(x, y, marker="o", label=label)
        if np.isfinite(y).any():
            plt.fill_between(x, y - e, y + e, alpha=0.12)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(ylabel + " by RST intervention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = _write_matplotlib_figure(plt, output_dir, filename)
    plt.close()
    return path


def _write_plots(rows: Sequence[Dict[str, Any]], output_dir: str) -> List[str]:
    plots = []
    for metric, ylabel, filename in [
        (
            "relative_delta_norm",
            "Mean relative RST delta norm",
            "intervention_relative_delta_by_layer.png",
        ),
        (
            "cosine_baseline_ablated",
            "Mean cosine(baseline, ablated)",
            "intervention_cosine_by_layer.png",
        ),
        (
            "removed_mass_fraction_abs",
            "Mean removed abs projection mass",
            "intervention_removed_mass_by_layer.png",
        ),
    ]:
        path = _plot_metric_by_layer(
            rows,
            output_dir,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
        )
        if path:
            plots.append(path)
    return plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="results/dawn_srw_rst_intervention")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sort-by", choices=RANKING_CHOICES, default="abs_projection")
    ap.add_argument("--active-threshold", type=float, default=0.5)
    ap.add_argument("--random-repeats", type=int, default=20)
    ap.add_argument(
        "--ablation-mode",
        choices=["no_renorm", "renorm", "both"],
        default="no_renorm",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--include-operator-ids",
        action="store_true",
        help="Also save full active_ids per layer. Selected intervention ids are always saved.",
    )
    ap.add_argument("--include-random-pool", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Custom case: target|||prompt or target|||prompt|||label",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    tokenizer = load_tokenizer(args.tokenizer)

    layers = _parse_layers(args.layers, model_cfg["n_layers"])
    cases = _parse_cases(args.case)
    ablation_modes = (
        ["no_renorm", "renorm"] if args.ablation_mode == "both" else [args.ablation_mode]
    )
    rng = np.random.default_rng(args.seed)
    ensure_dir(args.output)

    results = {
        "config": args.config,
        "checkpoint": meta,
        "tokenizer": args.tokenizer,
        "layers": layers,
        "top_k": int(args.top_k),
        "sort_by": args.sort_by,
        "active_threshold": float(args.active_threshold),
        "random_repeats": int(args.random_repeats),
        "ablation_mode": args.ablation_mode,
        "ablation_modes_evaluated": ablation_modes,
        "seed": int(args.seed),
        "include_operator_ids": bool(args.include_operator_ids),
        "include_random_pool": bool(args.include_random_pool),
        "interpretation": {
            "top_contributors": "active RST operators with largest --sort-by metric",
            "random_active": "same-size random samples from activation > active_threshold",
            "low_contributors": "active RST operators with smallest --sort-by metric",
            "random_pool": "same-size random samples from the full RST pool when --include-random-pool is set",
            "no_renorm": "subtracts selected baseline contribution vectors under the original denominator",
            "renorm": "zeros selected gates and recomputes the RST output denominator over remaining gates",
        },
        "end_to_end_intervention": {
            "implemented": False,
            "todo": (
                "Add a token/layer-local continuation hook that replaces the RST "
                "output at (layer, token_index), adds it to x_after_attention, "
                "then runs remaining blocks and final LM head to compare logits."
            ),
        },
        "cases": [],
        "csv": [],
        "plots": [],
        "aggregate": [],
    }

    all_rows: List[Dict[str, Any]] = []
    for case in cases:
        print(f"\n=== target={case['target']!r} label={case['prompt_label']!r} ===")
        print(f"Prompt: {case['prompt']}")
        case_result, rows = run_case(
            mod,
            params,
            model_cfg,
            tokenizer,
            case,
            layers=layers,
            top_k=args.top_k,
            sort_by=args.sort_by,
            active_threshold=args.active_threshold,
            random_repeats=args.random_repeats,
            ablation_modes=ablation_modes,
            include_operator_ids=args.include_operator_ids,
            include_random_pool=args.include_random_pool,
            rng=rng,
        )
        results["cases"].append(case_result)
        all_rows.extend(rows)
        for layer_item in case_result["layers"]:
            summary = layer_item["decision_summary"]
            top_rows = [
                r
                for r in layer_item["interventions"]
                if r["intervention_type"] == "top_contributors"
                and r["ablation_mode"] == ablation_modes[0]
            ]
            top_delta = top_rows[0]["relative_delta_norm"] if top_rows else math.nan
            print(
                f"  L{layer_item['layer']:02d}: active={summary['active_count']} "
                f"gate_sum={summary['gate_sum']:.3f} "
                f"rst_norm={summary['baseline_rst_norm']:.5f} "
                f"top_rel_delta={top_delta:.5f}"
            )

    summary_csv = _write_summary_csv(all_rows, args.output)
    aggregate_rows = _aggregate_rows(all_rows)
    aggregate_csv = _write_aggregate_csv(aggregate_rows, args.output)
    results["csv"].extend([summary_csv, aggregate_csv])
    results["aggregate"] = aggregate_rows
    print(f"\n  csv: {summary_csv}")
    print(f"  csv: {aggregate_csv}")

    if not args.no_plots:
        plot_paths = _write_plots(all_rows, args.output)
        results["plots"].extend(plot_paths)
        for path in plot_paths:
            print(f"  plot: {path}")

    json_path = save_json(results, args.output, "rst_intervention_results.json")
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()

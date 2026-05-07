#!/usr/bin/env python3
"""Validation-set DAWN-SRW active/operator-compute analysis.

Unlike prompt diagnostics, this samples the validation .bin directly and
averages per-layer Q/K/V/RST active-count diagnostics.  The implementation
computes gate statistics inside a layer scan and returns only small per-layer
arrays, so it avoids materializing all layer gates on the host.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from dawn_srw_common import (
    build_model,
    ensure_dir,
    import_dawn_srw,
    is_gcs,
    load_checkpoint_params,
    load_config,
    load_val_tokens,
    model_cfg_from_config,
    open_file,
    save_json,
)


POOLS = ("Q", "K", "V", "RST")
METRICS = (
    "active_count_sum",
    "effective_selected_count_sum",
    "gate_entropy_sum",
    "top1_gate_frac_sum",
)


def is_host0() -> bool:
    return int(jax.process_index()) == 0


def log(msg: str) -> None:
    if is_host0():
        print(msg, flush=True)


def out_path(output_dir: str, filename: str) -> str:
    return output_dir.rstrip("/") + "/" + filename if is_gcs(output_dir) else str(Path(output_dir) / filename)


def write_csv(rows: List[Dict[str, Any]], output_dir: str, filename: str) -> str:
    path = out_path(output_dir, filename)
    fieldnames = list(rows[0].keys()) if rows else []
    with open_file(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_plot(rows: List[Dict[str, Any]], output_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"plot skipped: {type(exc).__name__}: {exc}")
        return None

    layers = [int(r["layer"]) for r in rows]
    plt.figure(figsize=(9, 5))
    for pool in POOLS:
        key = f"{pool.lower()}_active_count_mean"
        plt.plot(layers, [float(r[key]) for r in rows], marker="o", label=pool)
    plt.xlabel("Layer")
    plt.ylabel("Mean active operators / token")
    plt.title("DAWN-SRW validation active count by layer")
    plt.legend()
    plt.tight_layout()

    dest = out_path(output_dir, "active_count_by_layer.png")
    if is_gcs(dest):
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_name = tmp.name
            plt.savefig(tmp_name, dpi=180)
            with open(tmp_name, "rb") as src, open_file(dest, "wb") as dst:
                dst.write(src.read())
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, dpi=180)
    plt.close()
    return dest


def make_batch_stats_fn(mod, params, model_cfg: Dict[str, Any], active_eps: float):
    params = mod._squeeze_params(jax.tree.map(jnp.asarray, params))
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    active_eps_j = jnp.float32(active_eps)

    pool_params = params["neuron_pool"]
    router_params = params["router"]
    norm_params = params["norm"]
    emb_matrix = jnp.asarray(params["token_emb"]["embedding"])
    pos_matrix = jnp.asarray(params["pos_emb"]["embedding"])
    qk_route = pool_params["attn_qk_emb"]
    v_route = pool_params["attn_v_emb"]
    rst_route = pool_params["rst_emb"]
    block_params = [params[f"block_{i}"] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params)

    def pool_stats(raw_gate, norm_gate):
        active_count = (raw_gate > active_eps_j).sum(axis=-1, dtype=jnp.float32)
        eff_n = 1.0 / (jnp.square(norm_gate).sum(axis=-1) + 1.0e-12)
        entropy = -(norm_gate * jnp.log(norm_gate + 1.0e-12)).sum(axis=-1)
        top1 = norm_gate.max(axis=-1)
        return jnp.stack(
            [
                active_count.sum(),
                eff_n.sum(),
                entropy.sum(),
                top1.sum(),
            ]
        )

    @jax.jit
    def batch_stats(input_ids):
        input_ids = input_ids.astype(jnp.int32)
        bsz, seq_len = input_ids.shape
        positions = jnp.arange(seq_len)[jnp.newaxis, :]
        x = emb_matrix[input_ids] + pos_matrix[positions]

        def layer_fn(x, bp):
            normed = mod._layer_norm(x, bp["norm1"]["scale"], bp["norm1"]["bias"])
            h_all = normed @ router_params["proj_attn"]["kernel"] + router_params["proj_attn"]["bias"]
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params["tau_attn"]["kernel"] + router_params["tau_attn"]["bias"]
            scan_all = (
                normed @ router_params["raw_scan_offset_attn"]["kernel"]
                + router_params["raw_scan_offset_attn"]["bias"]
            )

            Q, gate_Q_raw, gate_Q = mod._srw_inference_with_gates(
                normed,
                h_Q,
                qk_route,
                tau_all[:, :, 0:1],
                scan_all[:, :, 0:1],
                pool_params["attn_qk_read"],
                pool_params["attn_qk_write"],
            )
            K, gate_K_raw, gate_K = mod._srw_inference_with_gates(
                normed,
                h_K,
                qk_route,
                tau_all[:, :, 1:2],
                scan_all[:, :, 1:2],
                pool_params["attn_qk_read"],
                pool_params["attn_qk_write"],
            )
            V, gate_V_raw, gate_V = mod._srw_inference_with_gates(
                normed,
                h_V,
                v_route,
                tau_all[:, :, 2:3],
                scan_all[:, :, 2:3],
                pool_params["attn_v_read"],
                pool_params["attn_v_write"],
            )
            Q = Q * pool_params["attn_qk_scale"]
            K = K * pool_params["attn_qk_scale"]
            V = V * pool_params["attn_v_scale"]

            d_head = d_model // n_heads
            Qr = Q.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            scores = jnp.einsum("bhsd,bhtd->bhst", Qr, Kr) / jnp.sqrt(jnp.float32(d_head))
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum("bhst,bhtd->bhsd", attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
            attn_out = attn_out @ bp["attn"]["expand_O"]["kernel"]
            attn_out_norm_sum = jnp.linalg.norm(attn_out, axis=-1).sum()
            x = x + attn_out

            normed = mod._layer_norm(x, bp["norm2"]["scale"], bp["norm2"]["bias"])
            h_rst = normed @ router_params["proj_rst"]["kernel"] + router_params["proj_rst"]["bias"]
            tau_rst = normed @ router_params["tau_rst"]["kernel"] + router_params["tau_rst"]["bias"]
            scan_rst = (
                normed @ router_params["raw_scan_offset_rst"]["kernel"]
                + router_params["raw_scan_offset_rst"]["bias"]
            )
            rst_out, gate_RST_raw, gate_RST = mod._srw_inference_with_gates(
                normed,
                h_rst,
                rst_route,
                tau_rst,
                scan_rst,
                pool_params["rst_read"],
                pool_params["rst_write"],
            )
            rst_out = rst_out * pool_params["rst_scale"]
            rst_out_norm_sum = jnp.linalg.norm(rst_out, axis=-1).sum()
            x = x + rst_out

            layer_stats = jnp.stack(
                [
                    pool_stats(gate_Q_raw, gate_Q),
                    pool_stats(gate_K_raw, gate_K),
                    pool_stats(gate_V_raw, gate_V),
                    pool_stats(gate_RST_raw, gate_RST),
                ],
                axis=0,
            )
            norms = jnp.stack([attn_out_norm_sum, rst_out_norm_sum])
            return x, {"pool_stats": layer_stats, "norm_sums": norms}

        x, scan_stats = jax.lax.scan(layer_fn, x, stacked)
        _ = mod._layer_norm(x, norm_params["scale"], norm_params["bias"])
        token_count = jnp.asarray(input_ids.size, dtype=jnp.float32)
        return {
            "pool_stats": scan_stats["pool_stats"],  # [L, 4 pools, 4 metrics]
            "norm_sums": scan_stats["norm_sums"],    # [L, 2]
            "token_count": token_count,
        }

    return batch_stats


def flatten_accumulator(acc: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray([acc["token_count"]], dtype=np.float64),
            np.asarray(acc["pool_stats"], dtype=np.float64).ravel(),
            np.asarray(acc["norm_sums"], dtype=np.float64).ravel(),
        ]
    )


def unflatten_accumulator(vec: np.ndarray, n_layers: int) -> Dict[str, np.ndarray]:
    offset = 0
    token_count = float(vec[offset])
    offset += 1
    pool_size = n_layers * len(POOLS) * len(METRICS)
    pool_stats = vec[offset:offset + pool_size].reshape(n_layers, len(POOLS), len(METRICS))
    offset += pool_size
    norm_sums = vec[offset:offset + n_layers * 2].reshape(n_layers, 2)
    return {"token_count": token_count, "pool_stats": pool_stats, "norm_sums": norm_sums}


def load_sequences(val_data: str, seq_len: int, max_tokens: int, batch_size: int, max_batches: int) -> np.ndarray:
    tokens = load_val_tokens(val_data, max_tokens=max_tokens if max_tokens > 0 else None)
    n_seqs = int(tokens.shape[0]) // seq_len
    if max_batches > 0:
        n_seqs = min(n_seqs, max_batches * batch_size)
    n_seqs = (n_seqs // batch_size) * batch_size
    if n_seqs <= 0:
        raise ValueError(f"Not enough validation tokens for seq_len={seq_len}, batch_size={batch_size}")
    return np.asarray(tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len), dtype=np.int32)


def compute_rows(acc: Dict[str, np.ndarray], cfg: Dict[str, Any], dense_d_ff: int | None) -> List[Dict[str, Any]]:
    m = cfg.get("model", {})
    d_model = int(m.get("d_model", 384))
    d_ff = int(dense_d_ff or m.get("d_ff", 4 * d_model))
    denom = max(float(acc["token_count"]), 1.0)
    pool_means = acc["pool_stats"] / denom
    norm_means = acc["norm_sums"] / denom
    rows: List[Dict[str, Any]] = []
    for layer in range(pool_means.shape[0]):
        row: Dict[str, Any] = {"layer": layer}
        for p_idx, pool in enumerate(POOLS):
            prefix = pool.lower()
            for m_idx, metric in enumerate(METRICS):
                out_metric = metric.replace("_sum", "_mean")
                row[f"{prefix}_{out_metric}"] = float(pool_means[layer, p_idx, m_idx])
        q_active = float(pool_means[layer, 0, 0])
        k_active = float(pool_means[layer, 1, 0])
        v_active = float(pool_means[layer, 2, 0])
        rst_active = float(pool_means[layer, 3, 0])
        active_rw_flops = 4.0 * d_model * (q_active + k_active + v_active + rst_active)
        dense_ffn_flops = 4.0 * d_model * d_ff
        row.update(
            {
                "attn_out_norm": float(norm_means[layer, 0]),
                "rst_out_norm": float(norm_means[layer, 1]),
                "estimated_rw_active_flops_per_token": active_rw_flops,
                "dense_ffn_equiv_flops_per_token": dense_ffn_flops,
                "dense_ffn_equiv_ratio": active_rw_flops / dense_ffn_flops if dense_ffn_flops > 0 else "",
            }
        )
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]], acc: Dict[str, np.ndarray], cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    mean_ratio = float(np.mean([float(r["dense_ffn_equiv_ratio"]) for r in rows])) if rows else 0.0
    mean_rw = float(np.mean([float(r["estimated_rw_active_flops_per_token"]) for r in rows])) if rows else 0.0
    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "val_data": args.val_data,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_val_tokens": args.max_val_tokens,
        "max_batches": args.max_batches,
        "active_eps": args.active_eps,
        "jax_process_count": int(jax.process_count()),
        "jax_process_index": int(jax.process_index()),
        "total_positions": int(acc["token_count"]),
        "n_layers": int(cfg.get("model", {}).get("n_layers", len(rows))),
        "mean_estimated_rw_active_flops_per_token": mean_rw,
        "mean_dense_ffn_equiv_ratio": mean_ratio,
        "layer_rows": rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output", default="results/dawn_srw_val_active")
    ap.add_argument("--max-val-tokens", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=0, help="0 means all under max-val-tokens.")
    ap.add_argument("--active-eps", type=float, default=1.0e-8)
    ap.add_argument("--dense-d-ff", type=int, default=None, help="Dense FFN width for ratio denominator; default 4*d_model.")
    ap.add_argument("--single-device", action="store_true", help="Run only on process 0 when launched on multiple hosts.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_device and jax.process_count() > 1 and not is_host0():
        print("single-device active analysis: non-primary host exits without work", flush=True)
        return

    log("=== DAWN-SRW validation active/compute analysis ===")
    log(f"devices={jax.device_count()} local_devices={jax.local_device_count()} hosts={jax.process_count()}")
    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    seq_len = min(int(args.seq_len), int(model_cfg["max_seq_len"]))
    sequences = load_sequences(args.val_data, seq_len, args.max_val_tokens, args.batch_size, args.max_batches)

    n_layers = int(model_cfg["n_layers"])
    acc = {
        "token_count": 0.0,
        "pool_stats": np.zeros((n_layers, len(POOLS), len(METRICS)), dtype=np.float64),
        "norm_sums": np.zeros((n_layers, 2), dtype=np.float64),
    }
    batch_stats = make_batch_stats_fn(mod, params, model_cfg, args.active_eps)
    n_batches = sequences.shape[0] // args.batch_size
    proc_count = 1 if args.single_device else int(jax.process_count())
    proc_index = 0 if args.single_device else int(jax.process_index())
    local_batch_ids = list(range(proc_index, n_batches, proc_count))
    log(
        f"validation sequences={sequences.shape[0]} batch_size={args.batch_size} "
        f"seq_len={seq_len} batches={n_batches}"
    )
    t0 = time.time()
    for done, b in enumerate(local_batch_ids, start=1):
        batch = jnp.asarray(sequences[b * args.batch_size:(b + 1) * args.batch_size], dtype=jnp.int32)
        out = batch_stats(batch)
        pool_stats = np.asarray(jax.device_get(out["pool_stats"]), dtype=np.float64)
        norm_sums = np.asarray(jax.device_get(out["norm_sums"]), dtype=np.float64)
        token_count = float(jax.device_get(out["token_count"]))
        acc["pool_stats"] += pool_stats
        acc["norm_sums"] += norm_sums
        acc["token_count"] += token_count
        if is_host0() and (done == len(local_batch_ids) or done % max(1, len(local_batch_ids) // 10) == 0):
            print(f"  active progress: host0 {done}/{len(local_batch_ids)} local batches", flush=True)

    if proc_count > 1:
        gathered = np.asarray(process_allgather(flatten_accumulator(acc)))
        acc = unflatten_accumulator(gathered.reshape(proc_count, -1).sum(axis=0), n_layers)

    if is_host0():
        rows = compute_rows(acc, cfg, args.dense_d_ff)
        summary = summarize(rows, acc, cfg, args)
        summary["checkpoint_meta"] = meta
        summary["time_sec"] = time.time() - t0
        ensure_dir(args.output)
        save_json(summary, args.output, "dawn_active_compute.json")
        write_csv(rows, args.output, "dawn_active_compute_table.csv")
        plot_path = write_plot(rows, args.output)
        if plot_path:
            summary["plot"] = plot_path
            save_json(summary, args.output, "dawn_active_compute.json")
        log(
            f"result: positions={summary['total_positions']} "
            f"mean_rw_flops/token={summary['mean_estimated_rw_active_flops_per_token']:.3e} "
            f"dense_ffn_ratio={summary['mean_dense_ffn_equiv_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()

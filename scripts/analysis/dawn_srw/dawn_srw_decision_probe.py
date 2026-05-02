#!/usr/bin/env python3
"""DAWN-SRW RST model-decision trace probe.

This is the first mechanism experiment. It loads a DAWN-SRW checkpoint,
forwards a prompt to a chosen layer, and extracts the RST Layer model decision
for a chosen token:

    post-attention residual state
    -> route query
    -> signature matching scores
    -> tau / scan offset
    -> activation / intensity / gate
    -> selected SRW neurons
    -> RW operator contribution

Example:
  python scripts/analysis/dawn_srw/dawn_srw_decision_probe.py \
    --config configs/train_config_dawn_srw.yaml \
    --checkpoint checkpoints/run_xxx \
    --prompt "I deposited money in the bank" \
    --target-token bank --layer 5 --top-k 20 \
    --output results/decision_probe
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model, encode_prompt, ensure_dir, find_token_index, import_dawn_srw,
    load_checkpoint_params, load_config, load_tokenizer, model_cfg_from_config,
    save_json,
)


def _dense(x, dense_params):
    return x @ dense_params["kernel"] + dense_params["bias"]


def _attention_step(mod, params, model_cfg, x, layer_index: int):
    """Run only the Attention Layer of block layer_index."""
    pp = params["neuron_pool"]
    rp = params["router"]
    bp = params[f"block_{layer_index}"]
    B, S, D = x.shape
    n_heads = model_cfg["n_heads"]
    d_model = model_cfg["d_model"]
    d_head = d_model // n_heads

    normed = mod._layer_norm(x, bp["norm1"]["scale"], bp["norm1"]["bias"])
    h_all = _dense(normed, rp["proj_attn"])
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    tau_all = _dense(normed, rp["tau_attn"])
    raw_scan_all = _dense(normed, rp["raw_scan_offset_attn"])

    Q = mod._srw_inference(normed, h_Q, pp["attn_qk_emb"], tau_all[:, :, 0:1], raw_scan_all[:, :, 0:1], pp["attn_qk_read"], pp["attn_qk_write"]) * pp["attn_qk_scale"]
    K = mod._srw_inference(normed, h_K, pp["attn_qk_emb"], tau_all[:, :, 1:2], raw_scan_all[:, :, 1:2], pp["attn_qk_read"], pp["attn_qk_write"]) * pp["attn_qk_scale"]
    V = mod._srw_inference(normed, h_V, pp["attn_v_emb"], tau_all[:, :, 2:3], raw_scan_all[:, :, 2:3], pp["attn_v_read"], pp["attn_v_write"]) * pp["attn_v_scale"]

    Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum("bhsd,bhtd->bhst", Qr, Kr) / scale
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn_scores = jnp.where(causal, attn_scores, jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)
    attn_out = jnp.einsum("bhst,bhtd->bhsd", attn_w, Vr)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
    attn_out = attn_out @ bp["attn"]["expand_O"]["kernel"]
    return x + attn_out, attn_out


def _rst_step(mod, params, x_after_attn, layer_index: int):
    """Run the RST Layer of block layer_index."""
    pp = params["neuron_pool"]
    rp = params["router"]
    bp = params[f"block_{layer_index}"]
    normed = mod._layer_norm(x_after_attn, bp["norm2"]["scale"], bp["norm2"]["bias"])
    h = _dense(normed, rp["proj_rst"])
    tau = _dense(normed, rp["tau_rst"])
    raw_scan = _dense(normed, rp["raw_scan_offset_rst"])
    rst_out = mod._srw_inference(normed, h, pp["rst_emb"], tau, raw_scan, pp["rst_read"], pp["rst_write"]) * pp["rst_scale"]
    return x_after_attn + rst_out, rst_out, normed, h, tau, raw_scan


def forward_to_rst_input(mod, params, model_cfg, input_ids, layer_index: int):
    """Return RST input state for a given layer after running previous layers."""
    B, S = input_ids.shape
    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params["token_emb"]["embedding"][input_ids] + params["pos_emb"]["embedding"][positions]
    for i in range(layer_index):
        x_after_attn, _ = _attention_step(mod, params, model_cfg, x, i)
        x, _, _, _, _, _ = _rst_step(mod, params, x_after_attn, i)
    x_after_attn, attn_out = _attention_step(mod, params, model_cfg, x, layer_index)
    x_next, rst_out, normed, h, tau_offset, raw_scan_offset = _rst_step(mod, params, x_after_attn, layer_index)
    return {
        "x_before_attention": x,
        "x_after_attention": x_after_attn,
        "attention_output": attn_out,
        "rst_input_normed": normed,
        "rst_route_query": h,
        "rst_tau_offset": tau_offset,
        "rst_raw_scan_offset": raw_scan_offset,
        "rst_output": rst_out,
        "x_next": x_next,
    }


def compute_rst_decision(mod, params, state, token_index: int, top_k: int = 20, sort_by: str = "gate"):
    pp = params["neuron_pool"]
    x = state["rst_input_normed"][0, token_index, :].astype(jnp.float32)
    h = state["rst_route_query"][0, token_index, :].astype(jnp.float32)
    tau_offset = state["rst_tau_offset"][0, token_index, 0].astype(jnp.float32)
    raw_scan_offset = state["rst_raw_scan_offset"][0, token_index, 0].astype(jnp.float32)
    emb = pp["rst_emb"].astype(jnp.float32)
    read = pp["rst_read"].astype(jnp.float32)
    write = pp["rst_write"].astype(jnp.float32)

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
    write_norm = jnp.linalg.norm(write, axis=-1)
    contribution_norm_raw = jnp.abs(coeff) * write_norm
    den = jnp.maximum(gate.sum(), 1.0)
    contribution_norm = contribution_norm_raw / den

    if sort_by == "contribution":
        order_metric = contribution_norm
    elif sort_by == "score":
        order_metric = scores
    else:
        order_metric = gate
    k = min(top_k, int(gate.shape[0]))
    top_vals, top_idx = jax.lax.top_k(order_metric, k)

    arrays = jax.device_get({
        "scores": scores[top_idx],
        "raw": raw[top_idx],
        "margin": margin[top_idx],
        "activation": activation[top_idx],
        "intensity": intensity[top_idx],
        "gate": gate[top_idx],
        "read_value": read_value[top_idx],
        "coeff": coeff[top_idx],
        "write_norm": write_norm[top_idx],
        "contribution_norm": contribution_norm[top_idx],
        "top_idx": top_idx,
        "top_metric": top_vals,
        "s_mean": s_mean,
        "s_std": s_std,
        "tau_offset": tau_offset,
        "raw_scan_offset": raw_scan_offset,
        "scan_offset": scan_offset,
        "tau": tau,
        "gate_sum": gate.sum(),
        "active_count": (activation > 0.5).sum(),
        "gate_entropy": -(gate / (gate.sum() + 1e-8) * jnp.log(gate / (gate.sum() + 1e-8) + 1e-8)).sum(),
        "rst_output_norm": jnp.linalg.norm(state["rst_output"][0, token_index, :]),
        "attn_output_norm": jnp.linalg.norm(state["attention_output"][0, token_index, :]),
    })

    top = []
    for rank, idx in enumerate(np.asarray(arrays["top_idx"]).tolist()):
        top.append({
            "rank": rank + 1,
            "neuron_id": int(idx),
            "score": float(arrays["scores"][rank]),
            "raw": float(arrays["raw"][rank]),
            "margin": float(arrays["margin"][rank]),
            "activation": float(arrays["activation"][rank]),
            "intensity": float(arrays["intensity"][rank]),
            "gate": float(arrays["gate"][rank]),
            "read_value": float(arrays["read_value"][rank]),
            "coefficient_gate_times_read": float(arrays["coeff"][rank]),
            "write_norm": float(arrays["write_norm"][rank]),
            "contribution_norm": float(arrays["contribution_norm"][rank]),
            "sort_metric": float(arrays["top_metric"][rank]),
        })

    summary = {
        "score_mean": float(arrays["s_mean"]),
        "score_std": float(arrays["s_std"]),
        "tau_offset": float(arrays["tau_offset"]),
        "raw_scan_offset": float(arrays["raw_scan_offset"]),
        "scan_offset": float(arrays["scan_offset"]),
        "tau": float(arrays["tau"]),
        "gate_sum": float(arrays["gate_sum"]),
        "active_count_activation_gt_0_5": int(arrays["active_count"]),
        "gate_entropy": float(arrays["gate_entropy"]),
        "attention_output_norm_at_token": float(arrays["attn_output_norm"]),
        "rst_output_norm_at_token": float(arrays["rst_output_norm"]),
    }
    return summary, top


def run_probe(config_path, checkpoint, prompt, target_token, layer, top_k, sort_by, tokenizer_name, output_dir):
    cfg = load_config(config_path)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    tokenizer = load_tokenizer(tokenizer_name)
    ids = encode_prompt(tokenizer, prompt, model_cfg["max_seq_len"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    token_index = find_token_index(tokens, target_token, default="last")
    input_ids = jnp.asarray([ids], dtype=jnp.int32)

    layers = list(range(model_cfg["n_layers"])) if layer == "all" else [int(layer)]
    results = []
    for li in layers:
        state = forward_to_rst_input(mod, params, model_cfg, input_ids, li)
        summary, top = compute_rst_decision(mod, params, state, token_index, top_k, sort_by)
        item = {
            "prompt": prompt,
            "tokens": tokens,
            "target_token": target_token,
            "token_index": token_index,
            "token_text": tokens[token_index],
            "layer": li,
            "pool": "rst",
            "checkpoint": meta,
            "summary": summary,
            "top_selected_srw_neurons": top,
        }
        results.append(item)
        print(f"\nLayer {li} token[{token_index}]={tokens[token_index]!r}")
        print(f"  active={summary['active_count_activation_gt_0_5']} gate_sum={summary['gate_sum']:.4f} entropy={summary['gate_entropy']:.4f}")
        print(f"  attn_norm={summary['attention_output_norm_at_token']:.4f} rst_norm={summary['rst_output_norm_at_token']:.4f}")
        for row in top[:min(8, len(top))]:
            print(f"  #{row['rank']:02d} neuron={row['neuron_id']:5d} gate={row['gate']:.5f} contrib={row['contribution_norm']:.5f} score={row['score']:.4f}")

    ensure_dir(output_dir)
    path = save_json(results, output_dir, "rst_decision_trace.json")
    print(f"\nSaved: {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target-token", default=None)
    ap.add_argument("--layer", default="all", help="Layer index or 'all'")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sort-by", choices=["gate", "contribution", "score"], default="gate")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--output", default="results/dawn_srw_decision_probe")
    args = ap.parse_args()
    run_probe(args.config, args.checkpoint, args.prompt, args.target_token, args.layer, args.top_k, args.sort_by, args.tokenizer, args.output)


if __name__ == "__main__":
    main()

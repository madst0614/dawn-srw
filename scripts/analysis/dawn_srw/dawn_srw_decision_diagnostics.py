#!/usr/bin/env python3
"""DAWN-SRW model-decision diagnostics.

Computes layer-wise aggregate statistics from analysis_forward:
  - active count from raw gates
  - normalized-gate entropy
  - effective selected count
  - top-1 gate fraction
  - Attention output norm vs RST output norm

This is a lightweight bridge between basic sanity checks and detailed
per-token decision traces.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model, encode_prompt, ensure_dir, import_dawn_srw,
    load_checkpoint_params, load_config, load_tokenizer, model_cfg_from_config,
    save_json,
)


def gate_stats(raw_gate, norm_gate):
    # raw_gate/norm_gate: [L, B, S, N]
    active = (raw_gate > 1e-8).astype(np.float32)
    active_count = active.sum(axis=-1).mean(axis=(1, 2))
    top1_frac = norm_gate.max(axis=-1).mean(axis=(1, 2))
    eff_n = (1.0 / (np.square(norm_gate).sum(axis=-1) + 1e-12)).mean(axis=(1, 2))
    entropy = (-(norm_gate * np.log(norm_gate + 1e-12)).sum(axis=-1)).mean(axis=(1, 2))
    return {
        "active_count_mean": active_count.tolist(),
        "top1_gate_frac_mean": top1_frac.tolist(),
        "effective_selected_count_mean": eff_n.tolist(),
        "gate_entropy_mean": entropy.tolist(),
    }


def run_batch(mod, params, model_cfg, ids_batch):
    logits, info = mod.analysis_forward(params, model_cfg, jnp.asarray(ids_batch, dtype=jnp.int32), mode="full")
    info = jax.device_get(info)
    pools = {
        "Q": (np.asarray(info["gate_Q_raw"]), np.asarray(info["gate_Q"])),
        "K": (np.asarray(info["gate_K_raw"]), np.asarray(info["gate_K"])),
        "V": (np.asarray(info["gate_V_raw"]), np.asarray(info["gate_V"])),
        "RST": (np.asarray(info["gate_RST_raw"]), np.asarray(info["gate_RST"])),
    }
    stats = {name: gate_stats(raw, norm) for name, (raw, norm) in pools.items()}
    stats["attn_out_norm"] = np.asarray(info["attn_out_norm"]).tolist()
    stats["rst_out_norm"] = np.asarray(info["rst_out_norm"]).tolist()
    return stats


def merge_stats(items):
    if len(items) == 1:
        return items[0]
    out = {}
    for key in items[0].keys():
        if isinstance(items[0][key], dict):
            out[key] = {}
            for sub in items[0][key].keys():
                arr = np.asarray([it[key][sub] for it in items], dtype=np.float64)
                out[key][sub] = arr.mean(axis=0).tolist()
        else:
            arr = np.asarray([it[key] for it in items], dtype=np.float64)
            out[key] = arr.mean(axis=0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="results/dawn_srw_decision_diagnostics")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--max-length", type=int, default=128, help="Prompt truncation/padding length for stable JIT shape")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    tokenizer = load_tokenizer(args.tokenizer)

    prompts = args.prompt or [
        "I deposited money in the bank",
        "He sat by the river bank",
        "The phone needs a charge",
        "The police filed a charge",
        "The room was filled with bright light",
        "This bag is very light",
    ]
    max_len = min(args.max_length, model_cfg["max_seq_len"])
    batches = []
    for p in prompts:
        ids = encode_prompt(tokenizer, p, max_len)
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        batches.append([ids])  # batch size 1 keeps gate tensors manageable

    print(f"Running decision diagnostics on {len(batches)} prompts, length={max_len}")
    per_prompt = []
    for prompt, batch in zip(prompts, batches):
        print(f"  {prompt!r}")
        stats = run_batch(mod, params, model_cfg, np.asarray(batch, dtype=np.int32))
        stats["prompt"] = prompt
        per_prompt.append(stats)

    merged = merge_stats([{k: v for k, v in x.items() if k != "prompt"} for x in per_prompt])
    result = {"checkpoint": meta, "prompts": prompts, "per_prompt": per_prompt, "mean": merged}

    print("\nLayer-wise mean RST diagnostics:")
    rst = merged["RST"]
    for i, (act, ent, top1, eff) in enumerate(zip(rst["active_count_mean"], rst["gate_entropy_mean"], rst["top1_gate_frac_mean"], rst["effective_selected_count_mean"])):
        print(f"  L{i:02d}: active={act:.1f} entropy={ent:.3f} top1={top1:.4f} eff_n={eff:.1f} attn_norm={merged['attn_out_norm'][i]:.4f} rst_norm={merged['rst_out_norm'][i]:.4f}")

    ensure_dir(args.output)
    path = save_json(result, args.output, "decision_diagnostics.json")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""DAWN-SRW ambiguity / context-dependent model decision experiment.

Compares RST Layer model decisions for the same surface token in different
contexts, e.g.:
  - "I deposited money in the bank" vs "He sat by the river bank"

Output includes layer-wise top-neuron overlap and decision summaries.
"""

from __future__ import annotations

import argparse
from typing import List

import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model, encode_prompt, ensure_dir, find_token_index, import_dawn_srw,
    load_checkpoint_params, load_config, load_tokenizer, model_cfg_from_config,
    save_json,
)
from dawn_srw_decision_probe import forward_to_rst_input, compute_rst_decision


DEFAULT_PAIRS = [
    ("bank", "I deposited money in the bank", "He sat by the river bank"),
    ("charge", "The phone needs a charge", "The police filed a charge"),
    ("light", "The room was filled with bright light", "This bag is very light"),
    ("apple", "She ate an apple after lunch", "Apple released a new product"),
]


def top_ids(top_rows, k):
    return [int(r["neuron_id"]) for r in top_rows[:k]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / (len(sa | sb) + 1e-12)


def run_pair(mod, params, model_cfg, tokenizer, target, prompt_a, prompt_b, layers, top_k, sort_by):
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
        layer_results = []
        top_sets = []
        summaries = []
        for ids, tokens, idx in encoded:
            state = forward_to_rst_input(mod, params, model_cfg, jnp.asarray([ids], dtype=jnp.int32), layer)
            summary, top = compute_rst_decision(mod, params, state, idx, top_k=top_k, sort_by=sort_by)
            layer_results.append({
                "token_index": idx,
                "token_text": tokens[idx],
                "summary": summary,
                "top_selected_srw_neurons": top,
            })
            top_sets.append(top_ids(top, top_k))
            summaries.append(summary)
        overlap = jaccard(top_sets[0], top_sets[1])
        out["layers"].append({
            "layer": layer,
            "top_k": top_k,
            "top_neuron_jaccard": overlap,
            "gate_sum_delta_abs": abs(summaries[0]["gate_sum"] - summaries[1]["gate_sum"]),
            "rst_output_norm_delta_abs": abs(summaries[0]["rst_output_norm_at_token"] - summaries[1]["rst_output_norm_at_token"]),
            "a": layer_results[0],
            "b": layer_results[1],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="results/dawn_srw_ambiguity")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sort-by", choices=["gate", "contribution", "score"], default="gate")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices, e.g. 0,5,11")
    ap.add_argument("--case", action="append", default=[], help="Custom case: target|||prompt A|||prompt B")
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

    results = {"checkpoint": meta, "top_k": args.top_k, "sort_by": args.sort_by, "cases": []}
    for target, a, b in cases:
        print(f"\n=== target={target!r} ===")
        print(f"A: {a}")
        print(f"B: {b}")
        item = run_pair(mod, params, model_cfg, tokenizer, target, a, b, layers, args.top_k, args.sort_by)
        results["cases"].append(item)
        for row in item["layers"]:
            print(f"  L{row['layer']:02d}: top-{args.top_k} Jaccard={row['top_neuron_jaccard']:.3f} | rst_norm_delta={row['rst_output_norm_delta_abs']:.4f}")

    ensure_dir(args.output)
    path = save_json(results, args.output, "ambiguity_comparison.json")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

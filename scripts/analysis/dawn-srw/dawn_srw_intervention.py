#!/usr/bin/env python3
"""DAWN-SRW RW-operator intervention experiment.

Selects top RST SRW neurons for a token/layer, suppresses those neurons, and
measures the effect on loss/logits. This tests whether the observed model
decision is causally relevant, not merely diagnostic.

Suppression is global over the RST pool for the forward pass because the
current model helper accepts pool-level masks. This is intentionally simple
and conservative for a first causal probe.
"""

from __future__ import annotations

import argparse
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from dawn_srw_common import (
    build_model, encode_prompt, ensure_dir, find_token_index, import_dawn_srw,
    load_checkpoint_params, load_config, load_tokenizer, model_cfg_from_config,
    np_softmax, save_json, token_level_ce,
)
from dawn_srw_decision_probe import forward_to_rst_input, compute_rst_decision


def forward_logits(mod, params, model_cfg, input_ids, rst_mask=None):
    if rst_mask is None:
        f = mod.build_suppressed_forward(params, model_cfg, {})
    else:
        f = mod.build_suppressed_forward(params, model_cfg, {"rst": jnp.asarray(rst_mask, dtype=bool)})
    return f(jnp.asarray([input_ids], dtype=jnp.int32))


def summarize_effect(tokenizer, input_ids, token_index, base_logits, suppressed_logits):
    base_np = np.asarray(jax.device_get(base_logits))
    sup_np = np.asarray(jax.device_get(suppressed_logits))
    base_ce, base_mean = token_level_ce(base_np, input_ids)
    sup_ce, sup_mean = token_level_ce(sup_np, input_ids)
    out = {
        "mean_ce_base": base_mean,
        "mean_ce_suppressed": sup_mean,
        "mean_ce_delta": sup_mean - base_mean,
    }
    if token_index < len(input_ids) - 1:
        next_id = int(input_ids[token_index + 1])
        probs_base = np_softmax(base_np[0, token_index, :])
        probs_sup = np_softmax(sup_np[0, token_index, :])
        out.update({
            "position": token_index,
            "next_token_id": next_id,
            "next_token_text": tokenizer.convert_ids_to_tokens([next_id])[0],
            "next_token_prob_base": float(probs_base[next_id]),
            "next_token_prob_suppressed": float(probs_sup[next_id]),
            "next_token_prob_delta": float(probs_sup[next_id] - probs_base[next_id]),
            "next_token_logit_base": float(base_np[0, token_index, next_id]),
            "next_token_logit_suppressed": float(sup_np[0, token_index, next_id]),
            "next_token_logit_delta": float(sup_np[0, token_index, next_id] - base_np[0, token_index, next_id]),
            "ce_at_position_base": float(base_ce[token_index]) if token_index < len(base_ce) else None,
            "ce_at_position_suppressed": float(sup_ce[token_index]) if token_index < len(sup_ce) else None,
            "ce_at_position_delta": float(sup_ce[token_index] - base_ce[token_index]) if token_index < len(base_ce) else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target-token", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sort-by", choices=["gate", "contribution", "score"], default="gate")
    ap.add_argument("--random-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--output", default="results/dawn_srw_intervention")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    params, meta = load_checkpoint_params(args.checkpoint, cfg, model=model)
    mod = import_dawn_srw()
    model_cfg = model_cfg_from_config(cfg)
    tokenizer = load_tokenizer(args.tokenizer)

    ids = encode_prompt(tokenizer, args.prompt, model_cfg["max_seq_len"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    token_index = find_token_index(tokens, args.target_token, default="last")

    state = forward_to_rst_input(mod, params, model_cfg, jnp.asarray([ids], dtype=jnp.int32), args.layer)
    summary, top = compute_rst_decision(mod, params, state, token_index, top_k=args.top_k, sort_by=args.sort_by)
    selected = [int(r["neuron_id"]) for r in top[:args.top_k]]
    n_rst = int(params["neuron_pool"]["rst_emb"].shape[0])
    mask = np.zeros(n_rst, dtype=bool)
    mask[selected] = True

    print(f"Prompt: {args.prompt}")
    print(f"Target token: index={token_index}, text={tokens[token_index]!r}")
    print(f"Suppressing top-{args.top_k} RST neurons from layer {args.layer}: {selected[:10]}{'...' if len(selected)>10 else ''}")

    base_logits = forward_logits(mod, params, model_cfg, ids, rst_mask=None)
    sup_logits = forward_logits(mod, params, model_cfg, ids, rst_mask=mask)
    effect = summarize_effect(tokenizer, ids, token_index, base_logits, sup_logits)

    result = {
        "checkpoint": meta,
        "prompt": args.prompt,
        "tokens": tokens,
        "target_token": args.target_token,
        "token_index": token_index,
        "token_text": tokens[token_index],
        "layer": args.layer,
        "top_k": args.top_k,
        "sort_by": args.sort_by,
        "selected_rst_neurons": selected,
        "decision_summary": summary,
        "intervention_effect": effect,
    }
    print(f"  CE mean: base={effect['mean_ce_base']:.4f} suppressed={effect['mean_ce_suppressed']:.4f} delta={effect['mean_ce_delta']:.4f}")
    if "next_token_prob_base" in effect:
        print(f"  next token {effect['next_token_text']!r}: p base={effect['next_token_prob_base']:.5f} p suppressed={effect['next_token_prob_suppressed']:.5f} delta={effect['next_token_prob_delta']:.5f}")
        print(f"  next logit delta={effect['next_token_logit_delta']:.4f} CE delta={effect['ce_at_position_delta']:.4f}")

    if args.random_baseline:
        rng = np.random.default_rng(args.seed)
        random_ids = rng.choice(n_rst, size=args.top_k, replace=False).tolist()
        random_mask = np.zeros(n_rst, dtype=bool)
        random_mask[random_ids] = True
        random_logits = forward_logits(mod, params, model_cfg, ids, rst_mask=random_mask)
        random_effect = summarize_effect(tokenizer, ids, token_index, base_logits, random_logits)
        result["random_baseline"] = {"selected_rst_neurons": random_ids, "effect": random_effect}
        print(f"  random baseline CE delta={random_effect['mean_ce_delta']:.4f}")

    ensure_dir(args.output)
    path = save_json(result, args.output, "rst_operator_intervention.json")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

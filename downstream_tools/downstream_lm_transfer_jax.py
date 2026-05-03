#!/usr/bin/env python3
"""
Prompt-style downstream fine-tuning/evaluation for DAWN and VanillaTransformer JAX/Flax LMs.

Why prompt-style?
- Uses the existing causal-LM interface: model(input_ids, labels=...) -> loss.
- No classification head required, so DAWN and Transformer are compared fairly.
- Evaluation uses candidate answer log-likelihood, not generation heuristics.

Supported tasks:
  GLUE:       sst2, qqp, mrpc, rte, mnli
  SuperGLUE: boolq, wic

Example:
  python downstream_lm_transfer_jax.py \
    --model_type dawn \
    --model_module /path/to/dawn_spatial_v394_exp.py \
    --config_json /path/to/model_config.json \
    --checkpoint /path/to/pretrain_ckpt \
    --tokenizer /path/to/tokenizer \
    --task rte \
    --output_dir runs/dawn_rte \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import checkpoints, train_state

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install datasets: pip install datasets") from exc

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install transformers: pip install transformers sentencepiece") from exc


# -----------------------------
# Model loading
# -----------------------------


def import_module_from_path(path: str):
    path = str(Path(path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("Install pyyaml or use JSON config") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if raw is None:
        return {}
    # Allow train configs with nested model/model_config blocks.
    for key in ("model", "model_config", "architecture"):
        if isinstance(raw, dict) and isinstance(raw.get(key), dict):
            merged = dict(raw)
            merged.update(raw[key])
            return merged
    return raw


def filter_model_kwargs(model_cls: Any, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Flax Module dataclass fields are available through __dataclass_fields__.
    allowed = set(getattr(model_cls, "__dataclass_fields__", {}).keys())
    cfg = dict(cfg)
    # Common aliases from training configs.
    aliases = {
        "num_layers": "n_layers",
        "num_heads": "n_heads",
        "seq_len": "max_seq_len",
        "sequence_length": "max_seq_len",
    }
    for src, dst in aliases.items():
        if src in cfg and dst not in cfg:
            cfg[dst] = cfg[src]

    overrides = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_seq_len": args.max_seq_len,
        "dropout_rate": args.dropout_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "d_route": args.d_route,
        "n_qk": args.n_qk,
        "n_v": args.n_v,
        "n_know": args.n_know,
        "router_dropout": args.router_dropout,
        "n_chunks_qk": args.n_chunks_qk,
        "n_chunks_v": args.n_chunks_v,
        "n_chunks_know": args.n_chunks_know,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    # Avoid passing trainer-only keys into the Module constructor.
    return {k: v for k, v in cfg.items() if k in allowed}


def create_model(args: argparse.Namespace):
    module = import_module_from_path(args.model_module)
    if args.model_type == "dawn":
        if not hasattr(module, "DAWN"):
            raise AttributeError("model_module has no DAWN class")
        model_cls = module.DAWN
    elif args.model_type == "transformer":
        if not hasattr(module, "VanillaTransformer"):
            raise AttributeError("model_module has no VanillaTransformer class")
        model_cls = module.VanillaTransformer
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    cfg = load_config(args.config_json)
    kwargs = filter_model_kwargs(model_cls, cfg, args)
    return model_cls(**kwargs), kwargs


def unwrap_params(tree: Any) -> Any:
    """Best-effort extraction of a params pytree from common checkpoint shapes."""
    if isinstance(tree, dict):
        if "params" in tree:
            return tree["params"]
        if "target" in tree:
            return unwrap_params(tree["target"])
        if "state" in tree:
            return unwrap_params(tree["state"])
        if "model" in tree:
            return unwrap_params(tree["model"])
    # flax.struct dataclass / TrainState-like
    if hasattr(tree, "params"):
        return tree.params
    return tree


def load_checkpoint_params(path: Optional[str], init_params: Any) -> Any:
    """Loads only params. If path is absent, returns init_params."""
    if not path:
        print("[ckpt] No checkpoint given: using random initialization.")
        return init_params

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    # 1) Flax legacy directory checkpoints.
    if p.is_dir():
        try:
            restored = checkpoints.restore_checkpoint(str(p), target={"params": init_params})
            params = unwrap_params(restored)
            print(f"[ckpt] Loaded Flax checkpoint dir: {p}")
            return params
        except Exception as e:
            print(f"[ckpt] Flax restore_checkpoint failed: {e}")
        # 2) Orbax directory checkpoints.
        try:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(str(p))
            params = unwrap_params(restored)
            print(f"[ckpt] Loaded Orbax checkpoint dir: {p}")
            return params
        except Exception as e:
            print(f"[ckpt] Orbax restore failed: {e}")

    # 3) Pickle / msgpack / raw flax serialization file.
    raw = p.read_bytes()
    if p.suffix.lower() in {".pkl", ".pickle"}:
        obj = pickle.loads(raw)
        params = unwrap_params(obj)
        print(f"[ckpt] Loaded pickle checkpoint: {p}")
        return params

    for target in ({"params": init_params}, init_params):
        try:
            restored = serialization.from_bytes(target, raw)
            params = unwrap_params(restored)
            print(f"[ckpt] Loaded Flax serialization file: {p}")
            return params
        except Exception:
            pass

    raise RuntimeError(
        f"Could not load checkpoint {p}. Supported: Flax checkpoint dir, Orbax dir, pickle, msgpack/from_bytes."
    )


# -----------------------------
# Task registry
# -----------------------------

Candidate = Tuple[str, int]  # text answer, gold label id


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    candidates: Tuple[Candidate, ...]
    gold_index: int


@dataclass(frozen=True)
class TaskSpec:
    hf_name: str
    hf_config: Optional[str]
    train_split: str
    eval_split: str
    build: Callable[[Dict[str, Any]], PromptExample]
    metric_name: str = "accuracy"


def _gold_index(candidates: Sequence[Candidate], label: int) -> int:
    for i, (_, y) in enumerate(candidates):
        if int(y) == int(label):
            return i
    raise ValueError(f"label {label} not found in candidates {candidates}")


def build_sst2(ex: Dict[str, Any]) -> PromptExample:
    cands = ((" negative", 0), (" positive", 1))
    prompt = f"Sentence: {ex['sentence']}\nSentiment:"
    return PromptExample(prompt, cands, _gold_index(cands, ex["label"]))


def build_qqp(ex: Dict[str, Any]) -> PromptExample:
    cands = ((" no", 0), (" yes", 1))
    prompt = f"Question 1: {ex['question1']}\nQuestion 2: {ex['question2']}\nAre these questions duplicates? Answer:"
    return PromptExample(prompt, cands, _gold_index(cands, ex["label"]))


def build_mrpc(ex: Dict[str, Any]) -> PromptExample:
    cands = ((" no", 0), (" yes", 1))
    prompt = f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nAre these sentences paraphrases? Answer:"
    return PromptExample(prompt, cands, _gold_index(cands, ex["label"]))


def build_rte(ex: Dict[str, Any]) -> PromptExample:
    # GLUE RTE: 0=entailment, 1=not_entailment
    cands = ((" yes", 0), (" no", 1))
    prompt = f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}\nDoes the premise entail the hypothesis? Answer:"
    return PromptExample(prompt, cands, _gold_index(cands, ex["label"]))


def build_mnli(ex: Dict[str, Any]) -> PromptExample:
    # GLUE MNLI: 0=entailment, 1=neutral, 2=contradiction
    cands = ((" yes", 0), (" maybe", 1), (" no", 2))
    prompt = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nDoes the premise entail the hypothesis? Answer:"
    return PromptExample(prompt, cands, _gold_index(cands, ex["label"]))


def build_boolq(ex: Dict[str, Any]) -> PromptExample:
    # SuperGLUE BoolQ label is bool.
    cands = ((" no", 0), (" yes", 1))
    label = int(bool(ex["label"]))
    prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
    return PromptExample(prompt, cands, _gold_index(cands, label))


def build_wic(ex: Dict[str, Any]) -> PromptExample:
    # label: whether target word has same meaning in both sentences.
    cands = ((" no", 0), (" yes", 1))
    label = int(bool(ex["label"]))
    prompt = (
        f"Word: {ex['word']}\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}\n"
        "Does the word have the same meaning in both sentences? Answer:"
    )
    return PromptExample(prompt, cands, _gold_index(cands, label))


TASKS: Dict[str, TaskSpec] = {
    "sst2": TaskSpec("glue", "sst2", "train", "validation", build_sst2),
    "qqp": TaskSpec("glue", "qqp", "train", "validation", build_qqp),
    "mrpc": TaskSpec("glue", "mrpc", "train", "validation", build_mrpc),
    "rte": TaskSpec("glue", "rte", "train", "validation", build_rte),
    "mnli": TaskSpec("glue", "mnli", "train", "validation_matched", build_mnli),
    "boolq": TaskSpec("super_glue", "boolq", "train", "validation", build_boolq),
    "wic": TaskSpec("super_glue", "wic", "train", "validation", build_wic),
}


# -----------------------------
# Tokenization / batching
# -----------------------------


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer.pad_token_id


def encode_prompt_answer(
    tokenizer,
    prompt: str,
    answer: str,
    max_seq_len: int,
    add_eos: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    if len(answer_ids) == 0:
        return None
    ids = prompt_ids + answer_ids
    if len(ids) > max_seq_len:
        # Keep the answer and the tail of the prompt.
        overflow = len(ids) - max_seq_len
        prompt_ids = prompt_ids[overflow:]
        ids = prompt_ids + answer_ids
    if len(ids) < 2 or len(ids) > max_seq_len:
        return None

    labels = [-100] * len(prompt_ids) + list(answer_ids)
    score_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int32),
        "score_mask": np.asarray(score_mask, dtype=np.int32),
    }


def pad_1d(xs: Sequence[np.ndarray], pad_value: int, max_len: Optional[int] = None) -> np.ndarray:
    if max_len is None:
        max_len = max(len(x) for x in xs)
    out = np.full((len(xs), max_len), pad_value, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        n = min(len(x), max_len)
        out[i, :n] = x[:n]
    return out


def make_train_rows(
    raw_dataset,
    spec: TaskSpec,
    tokenizer,
    max_seq_len: int,
    max_samples: Optional[int],
    seed: int,
    add_eos: bool,
) -> List[Dict[str, np.ndarray]]:
    indices = list(range(len(raw_dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples:
        indices = indices[:max_samples]
    rows: List[Dict[str, np.ndarray]] = []
    for idx in indices:
        ex = raw_dataset[idx]
        if "label" in ex and int(ex["label"]) < 0:
            continue
        pe = spec.build(ex)
        answer = pe.candidates[pe.gold_index][0]
        row = encode_prompt_answer(tokenizer, pe.prompt, answer, max_seq_len, add_eos=add_eos)
        if row is not None:
            rows.append(row)
    return rows


def make_eval_rows(
    raw_dataset,
    spec: TaskSpec,
    tokenizer,
    max_seq_len: int,
    max_samples: Optional[int],
    add_eos: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    limit = len(raw_dataset) if max_samples is None else min(max_samples, len(raw_dataset))
    for idx in range(limit):
        ex = raw_dataset[idx]
        if "label" in ex and int(ex["label"]) < 0:
            continue
        pe = spec.build(ex)
        encoded_candidates = []
        for cand_text, _ in pe.candidates:
            enc = encode_prompt_answer(tokenizer, pe.prompt, cand_text, max_seq_len, add_eos=add_eos)
            if enc is None:
                break
            encoded_candidates.append(enc)
        if len(encoded_candidates) == len(pe.candidates):
            rows.append({"candidates": encoded_candidates, "gold_index": pe.gold_index})
    return rows


def train_batches(rows: List[Dict[str, np.ndarray]], batch_size: int, pad_token_id: int, seed: int):
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) < batch_size:
            continue
        batch_rows = [rows[i] for i in batch_idx]
        input_ids = pad_1d([r["input_ids"] for r in batch_rows], pad_token_id)
        labels = pad_1d([r["labels"] for r in batch_rows], -100, input_ids.shape[1])
        yield {
            "input_ids": jnp.asarray(input_ids),
            "labels": jnp.asarray(labels),
        }


def eval_candidate_batches(rows: List[Dict[str, Any]], batch_size: int, pad_token_id: int):
    flat = []
    meta = []
    for ex_id, row in enumerate(rows):
        for cand_id, cand in enumerate(row["candidates"]):
            flat.append(cand)
            meta.append((ex_id, cand_id, row["gold_index"]))
    for start in range(0, len(flat), batch_size):
        batch_rows = flat[start:start + batch_size]
        batch_meta = meta[start:start + batch_size]
        input_ids = pad_1d([r["input_ids"] for r in batch_rows], pad_token_id)
        score_mask = pad_1d([r["score_mask"] for r in batch_rows], 0, input_ids.shape[1])
        yield {
            "input_ids": jnp.asarray(input_ids),
            "score_mask": jnp.asarray(score_mask),
            "meta": batch_meta,
        }


# -----------------------------
# Train / eval
# -----------------------------


class LMTrainState(train_state.TrainState):
    dropout_rng: Any


def create_optimizer(args: argparse.Namespace):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=max(args.total_train_steps or 1, 1),
        end_value=args.learning_rate * args.min_lr_ratio,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay),
    )
    return tx


def make_train_step(model, aux_weight: float, tau_weight: float):
    @jax.jit
    def train_step(state: LMTrainState, batch: Dict[str, jnp.ndarray]):
        rng, step_rng = jax.random.split(state.dropout_rng)

        def loss_fn(params):
            out = model.apply(
                {"params": params},
                batch["input_ids"],
                labels=batch["labels"],
                deterministic=False,
                rngs={"dropout": step_rng},
            )
            lm_loss = out["loss"]
            aux_loss = out.get("aux_loss", jnp.float32(0.0))
            tau_reg = out.get("tau_reg", jnp.float32(0.0))
            loss = lm_loss + aux_weight * aux_loss + tau_weight * tau_reg
            metrics = {
                "loss": loss,
                "lm_loss": lm_loss,
                "aux_loss": aux_loss,
                "tau_reg": tau_reg,
                "accuracy": out["correct"] / jnp.maximum(out["valid_count"], 1),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads, dropout_rng=rng)
        metrics["grad_norm"] = grad_norm
        return state, metrics

    return train_step


def make_score_step(model):
    @jax.jit
    def score_step(params: Any, input_ids: jnp.ndarray, score_mask: jnp.ndarray):
        out = model.apply(
            {"params": params},
            input_ids,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        logits = out["logits"][:, :-1, :]
        target = input_ids[:, 1:]
        mask = score_mask[:, 1:].astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        tok_lp = jnp.take_along_axis(log_probs, target[..., None], axis=-1).squeeze(-1)
        summed = (tok_lp * mask).sum(axis=-1)
        denom = jnp.maximum(mask.sum(axis=-1), 1.0)
        # Mean log-prob avoids length bias. For yes/no this is equivalent to sum.
        return summed / denom

    return score_step


def evaluate(model, params: Any, eval_rows: List[Dict[str, Any]], batch_size: int, pad_token_id: int) -> Dict[str, float]:
    score_step = make_score_step(model)
    scores_by_ex: Dict[int, List[Tuple[int, float]]] = {}
    gold_by_ex: Dict[int, int] = {}
    for batch in eval_candidate_batches(eval_rows, batch_size, pad_token_id):
        scores = np.asarray(score_step(params, batch["input_ids"], batch["score_mask"]))
        for s, (ex_id, cand_id, gold) in zip(scores, batch["meta"]):
            scores_by_ex.setdefault(ex_id, []).append((cand_id, float(s)))
            gold_by_ex[ex_id] = gold

    correct = 0
    total = 0
    for ex_id, cand_scores in scores_by_ex.items():
        cand_scores.sort(key=lambda x: x[0])
        pred = max(cand_scores, key=lambda x: x[1])[0]
        gold = gold_by_ex[ex_id]
        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / max(total, 1), "total": float(total)}


def save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["dawn", "transformer"], required=True)
    parser.add_argument("--model_module", required=True, help="Path to dawn_spatial_*.py or baseline_transformer_jax.py")
    parser.add_argument("--config_json", default=None, help="JSON/YAML config with model constructor kwargs")
    parser.add_argument("--checkpoint", default=None, help="Pretrained checkpoint path. Omit for random-init control.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name/path used for pretraining")
    parser.add_argument("--task", choices=sorted(TASKS.keys()), required=True)
    parser.add_argument("--output_dir", required=True)

    # Model overrides.
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--d_route", type=int, default=None)
    parser.add_argument("--n_qk", type=int, default=None)
    parser.add_argument("--n_v", type=int, default=None)
    parser.add_argument("--n_know", type=int, default=None)
    parser.add_argument("--router_dropout", type=float, default=None)
    parser.add_argument("--n_chunks_qk", type=int, default=None)
    parser.add_argument("--n_chunks_v", type=int, default=None)
    parser.add_argument("--n_chunks_know", type=int, default=None)

    # Training.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Usually 0.0 for downstream transfer.")
    parser.add_argument("--tau_weight", type=float, default=0.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--total_train_steps", type=int, default=None, help="Overrides num_epochs-derived steps for LR schedule")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    pad_token_id = ensure_pad_token(tokenizer)

    model, model_kwargs = create_model(args)
    max_seq_len = int(model_kwargs.get("max_seq_len", args.max_seq_len or 512))
    vocab_size = int(model_kwargs.get("vocab_size", args.vocab_size or getattr(tokenizer, "vocab_size", 0)))
    if tokenizer.vocab_size is not None and vocab_size != tokenizer.vocab_size:
        print(f"[warn] model vocab_size={vocab_size}, tokenizer.vocab_size={tokenizer.vocab_size}. Make sure this matches your pretraining tokenizer.")

    spec = TASKS[args.task]
    print(f"[data] Loading {spec.hf_name}/{spec.hf_config} task={args.task}")
    ds_train = load_dataset(spec.hf_name, spec.hf_config, split=spec.train_split)
    ds_eval = load_dataset(spec.hf_name, spec.hf_config, split=spec.eval_split)
    train_rows = make_train_rows(ds_train, spec, tokenizer, max_seq_len, args.max_train_samples, args.seed, args.add_eos)
    eval_rows = make_eval_rows(ds_eval, spec, tokenizer, max_seq_len, args.max_eval_samples, args.add_eos)
    if not train_rows:
        raise RuntimeError("No train rows after tokenization/truncation")
    if not eval_rows:
        raise RuntimeError("No eval rows after tokenization/truncation")
    print(f"[data] train_rows={len(train_rows)}, eval_rows={len(eval_rows)}, max_seq_len={max_seq_len}")

    # Init model.
    key = jax.random.PRNGKey(args.seed)
    init_len = min(max_seq_len, max(8, min(64, max(len(r["input_ids"]) for r in train_rows[:128]))))
    dummy = jnp.ones((1, init_len), dtype=jnp.int32)
    variables = model.init({"params": key, "dropout": key}, dummy, labels=dummy, deterministic=True)
    init_params = variables["params"]
    params = load_checkpoint_params(args.checkpoint, init_params)

    steps_per_epoch = len(train_rows) // args.batch_size
    total_steps = args.total_train_steps or max(steps_per_epoch * args.num_epochs, 1)
    args.total_train_steps = total_steps
    tx = create_optimizer(args)
    state = LMTrainState.create(apply_fn=model.apply, params=params, tx=tx, dropout_rng=jax.random.PRNGKey(args.seed + 1))
    train_step = make_train_step(model, args.aux_weight, args.tau_weight)

    metadata = {
        "model_type": args.model_type,
        "model_module": args.model_module,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "model_kwargs": model_kwargs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    save_json(out_dir / "run_config.json", metadata)

    print("[eval] before fine-tune")
    best = evaluate(model, state.params, eval_rows, args.eval_batch_size, pad_token_id)
    best["step"] = 0
    print(f"[eval] step=0 accuracy={best['accuracy']:.4f} total={int(best['total'])}")
    save_json(out_dir / "eval_step_0.json", best)

    global_step = 0
    t0 = time.time()
    for epoch in range(args.num_epochs):
        losses = []
        accs = []
        for batch in train_batches(train_rows, args.batch_size, pad_token_id, args.seed + epoch):
            global_step += 1
            state, metrics = train_step(state, batch)
            m = jax.device_get(metrics)
            losses.append(float(m["loss"]))
            accs.append(float(m["accuracy"]))
            if global_step % 20 == 0:
                print(
                    f"[train] step={global_step}/{total_steps} epoch={epoch+1} "
                    f"loss={np.mean(losses[-20:]):.4f} acc={np.mean(accs[-20:]):.4f} "
                    f"grad={float(m['grad_norm']):.3f}"
                )
            if global_step % args.eval_every_steps == 0:
                ev = evaluate(model, state.params, eval_rows, args.eval_batch_size, pad_token_id)
                ev["step"] = global_step
                print(f"[eval] step={global_step} accuracy={ev['accuracy']:.4f} total={int(ev['total'])}")
                save_json(out_dir / f"eval_step_{global_step}.json", ev)
                if ev["accuracy"] > best["accuracy"]:
                    best = ev
                    checkpoints.save_checkpoint(str(out_dir / "best"), target={"params": state.params}, step=global_step, overwrite=True)
                    save_json(out_dir / "best_eval.json", best)
            if global_step % args.save_every_steps == 0:
                checkpoints.save_checkpoint(str(out_dir / "checkpoints"), target={"params": state.params}, step=global_step, keep=2, overwrite=True)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    final = evaluate(model, state.params, eval_rows, args.eval_batch_size, pad_token_id)
    final["step"] = global_step
    final["seconds"] = time.time() - t0
    save_json(out_dir / "final_eval.json", final)
    if final["accuracy"] > best["accuracy"]:
        best = final
        checkpoints.save_checkpoint(str(out_dir / "best"), target={"params": state.params}, step=global_step, overwrite=True)
        save_json(out_dir / "best_eval.json", best)

    print(f"[done] final_acc={final['accuracy']:.4f}, best_acc={best['accuracy']:.4f} at step={best['step']}")


if __name__ == "__main__":
    main()

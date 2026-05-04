#!/usr/bin/env python3
"""Downstream prompt-style fine-tuning/eval for DAWN/Transformer causal LMs on TPU pods.

Does not modify or depend on C4 loaders. Uses task examples -> prompt + answer-token labels.

Modes:
  --init-from  : start each task from pretrained params only; optimizer fresh; step=0
  --resume-from: resume a downstream run from checkpoint; params+optimizer+step restored

Path resolution for both init/resume:
  directory -> best_model.flax first -> latest checkpoint_step*.flax -> latest *.flax
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

# Reuse the verified model registry, GCS I/O, sharding and checkpoint helpers.
import scripts.train_jax as tj

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# -----------------------------
# Small utilities
# -----------------------------

def is_host0() -> bool:
    return jax.process_index() == 0


def log(msg: str):
    if is_host0():
        print(msg, flush=True)


def join_path(base: str, name: str) -> str:
    return base.rstrip('/') + '/' + name


def path_basename(path: str) -> str:
    return path.rstrip('/').rsplit('/', 1)[-1]


def step_num(path: str) -> int:
    m = re.search(r'checkpoint_step(\d+)\.flax', path_basename(path))
    return int(m.group(1)) if m else -1


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Resolve file/dir checkpoint. Directories prefer best_model.flax."""
    if not path:
        return None
    p = str(path)
    if p.endswith('.flax'):
        if not tj._file_exists(p):
            raise FileNotFoundError(f'Checkpoint file not found: {p}')
        return p

    best = join_path(p, 'best_model.flax')
    if tj._file_exists(best):
        return best

    step_files = tj._list_files(p, 'checkpoint_step*.flax')
    if step_files:
        step_files = sorted(step_files, key=step_num)
        return step_files[-1]

    any_files = tj._list_files(p, '*.flax')
    if any_files:
        return any_files[-1]

    raise FileNotFoundError(f'No .flax checkpoint found in: {p}')


def load_yaml(path: str) -> Dict[str, Any]:
    with tj._open_file(path, 'r') as f:
        return yaml.safe_load(f) or {}


def write_text(path: str, text: str):
    with tj._open_file(path, 'w') as f:
        f.write(text)


def append_text(path: str, text: str):
    # GCS does not support normal append reliably; rewrite small log file.
    old = ''
    if tj._file_exists(path):
        with tj._open_file(path, 'r') as f:
            old = f.read()
    write_text(path, old + text)


def write_json(path: str, obj: Any):
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def append_csv(path: str, row: Dict[str, Any], header: Sequence[str]):
    # GCS append is awkward; write per-run CSV from stored rows instead.
    exists = tj._file_exists(path)
    if str(path).startswith('gs://'):
        rows = []
        if exists:
            with tj._open_file(path, 'r') as f:
                rows = list(csv.DictReader(f.read().splitlines()))
        rows.append({k: row.get(k, '') for k in header})
        out = []
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(header))
        w.writeheader(); w.writerows(rows)
        write_text(path, buf.getvalue())
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(header))
            if not exists:
                w.writeheader()
            w.writerow(row)


# -----------------------------
# Task prompts
# -----------------------------

@dataclass(frozen=True)
class PromptExample:
    prompt: str
    candidates: Tuple[Tuple[str, int], ...]
    gold_index: int


def _gold(cands: Sequence[Tuple[str, int]], label: int) -> int:
    for i, (_, y) in enumerate(cands):
        if int(y) == int(label):
            return i
    raise ValueError(f'label={label} not in {cands}')


def build_prompt(task: str, ex: Dict[str, Any]) -> PromptExample:
    task = task.lower()
    if task == 'sst2':
        c = ((" negative", 0), (" positive", 1))
        return PromptExample(f"Sentence: {ex['sentence']}\nSentiment:", c, _gold(c, ex['label']))
    if task == 'rte':
        c = ((" yes", 0), (" no", 1))  # GLUE RTE: 0 entailment, 1 not_entailment
        return PromptExample(
            f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}\nDoes the premise entail the hypothesis? Answer:",
            c, _gold(c, ex['label']))
    if task == 'mnli':
        c = ((" yes", 0), (" maybe", 1), (" no", 2))
        return PromptExample(
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nDoes the premise entail the hypothesis? Answer:",
            c, _gold(c, ex['label']))
    if task == 'qqp':
        c = ((" no", 0), (" yes", 1))
        return PromptExample(
            f"Question 1: {ex['question1']}\nQuestion 2: {ex['question2']}\nAre these questions duplicates? Answer:",
            c, _gold(c, ex['label']))
    if task == 'mrpc':
        c = ((" no", 0), (" yes", 1))
        return PromptExample(
            f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nAre these sentences paraphrases? Answer:",
            c, _gold(c, ex['label']))
    if task == 'boolq':
        c = ((" no", 0), (" yes", 1))
        lab = int(bool(ex['label']))
        return PromptExample(f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:", c, _gold(c, lab))
    if task == 'wic':
        c = ((" no", 0), (" yes", 1))
        lab = int(bool(ex['label']))
        return PromptExample(
            f"Word: {ex['word']}\nSentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nDoes the word have the same meaning in both sentences? Answer:",
            c, _gold(c, lab))
    raise ValueError(f'Unsupported task: {task}')


def hf_spec(task: str) -> Tuple[str, Optional[str], str, str]:
    task = task.lower()
    if task in ('sst2', 'rte', 'mnli', 'qqp', 'mrpc'):
        eval_split = 'validation_matched' if task == 'mnli' else 'validation'
        return 'glue', task, 'train', eval_split
    if task in ('boolq', 'wic'):
        return 'super_glue', task, 'train', 'validation'
    raise ValueError(f'No HF spec for task={task}')


# -----------------------------
# Data loading
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with tj._open_file(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tsv(path: str) -> List[Dict[str, Any]]:
    with tj._open_file(path, 'r') as f:
        text = f.read().splitlines()
    return list(csv.DictReader(text, delimiter='\t'))


def load_raw_splits(data_cfg: Dict[str, Any], task: str):
    source = data_cfg.get('source', 'hf')
    if source == 'hf':
        if load_dataset is None:
            raise RuntimeError('datasets is not installed. Install datasets or use data.source=jsonl/tsv.')
        name, subset, train_split, eval_split = hf_spec(task)
        name = data_cfg.get('hf_name', name)
        subset = data_cfg.get('hf_config', subset)
        train_split = data_cfg.get('train_split', train_split)
        eval_split = data_cfg.get('eval_split', eval_split)
        log(f'[data] HF load_dataset({name!r}, {subset!r}) train={train_split} eval={eval_split}')
        train = load_dataset(name, subset, split=train_split)
        evals = load_dataset(name, subset, split=eval_split)
        return train, evals
    if source == 'jsonl':
        return load_jsonl(data_cfg['train_path']), load_jsonl(data_cfg['eval_path'])
    if source == 'tsv':
        return load_tsv(data_cfg['train_path']), load_tsv(data_cfg['eval_path'])
    raise ValueError(f'Unknown data.source: {source}')


def ensure_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return int(tokenizer.pad_token_id)


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_seq_len: int, add_eos=False) -> Optional[Dict[str, np.ndarray]]:
    pids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    aids = tokenizer(answer, add_special_tokens=False)['input_ids']
    if add_eos and tokenizer.eos_token_id is not None:
        aids = aids + [tokenizer.eos_token_id]
    if not aids:
        return None
    if len(pids) + len(aids) > max_seq_len:
        keep = max_seq_len - len(aids)
        if keep <= 0:
            return None
        pids = pids[-keep:]
    ids = pids + aids
    labels = [-100] * len(pids) + aids
    score_mask = [0] * len(pids) + [1] * len(aids)
    return {
        'input_ids': np.asarray(ids, dtype=np.int32),
        'labels': np.asarray(labels, dtype=np.int32),
        'score_mask': np.asarray(score_mask, dtype=np.int32),
    }


def make_train_rows(raw, task: str, tokenizer, max_seq_len: int, max_samples: Optional[int], seed: int, add_eos=False):
    idxs = list(range(len(raw)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if max_samples is not None:
        idxs = idxs[:int(max_samples)]
    rows = []
    for i in idxs:
        ex = raw[i]
        if 'label' in ex:
            try:
                if int(ex['label']) < 0:
                    continue
            except Exception:
                pass
        pe = build_prompt(task, ex)
        answer = pe.candidates[pe.gold_index][0]
        row = encode_prompt_answer(tokenizer, pe.prompt, answer, max_seq_len, add_eos=add_eos)
        if row is not None:
            rows.append(row)
    return rows


def make_eval_rows(raw, task: str, tokenizer, max_seq_len: int, max_samples: Optional[int], add_eos=False):
    lim = len(raw) if max_samples is None else min(int(max_samples), len(raw))
    rows = []
    for i in range(lim):
        ex = raw[i]
        if 'label' in ex:
            try:
                if int(ex['label']) < 0:
                    continue
            except Exception:
                pass
        pe = build_prompt(task, ex)
        cands = []
        for txt, _ in pe.candidates:
            row = encode_prompt_answer(tokenizer, pe.prompt, txt, max_seq_len, add_eos=add_eos)
            if row is None:
                cands = []
                break
            cands.append(row)
        if cands:
            rows.append({'candidates': cands, 'gold_index': pe.gold_index})
    return rows


def pad_fixed(rows: Sequence[np.ndarray], max_seq_len: int, pad_value: int, dtype=np.int32) -> np.ndarray:
    out = np.full((len(rows), max_seq_len), pad_value, dtype=dtype)
    for i, x in enumerate(rows):
        n = min(len(x), max_seq_len)
        out[i, :n] = x[:n]
    return out


def build_global_train_batch(rows: List[Dict[str, np.ndarray]], indices: List[int], max_seq_len: int, pad_token_id: int):
    br = [rows[i] for i in indices]
    input_ids = pad_fixed([r['input_ids'] for r in br], max_seq_len, pad_token_id)
    labels = pad_fixed([r['labels'] for r in br], max_seq_len, -100)
    attention_mask = (input_ids != pad_token_id).astype(np.int32)
    return input_ids, labels, attention_mask


def build_candidate_global_batch(flat_rows: List[Dict[str, np.ndarray]], indices: List[int], max_seq_len: int, pad_token_id: int):
    br = [flat_rows[i] for i in indices]
    input_ids = pad_fixed([r['input_ids'] for r in br], max_seq_len, pad_token_id)
    score_mask = pad_fixed([r['score_mask'] for r in br], max_seq_len, 0)
    attention_mask = (input_ids != pad_token_id).astype(np.int32)
    return input_ids, score_mask, attention_mask


def local_slice(global_arr: np.ndarray) -> np.ndarray:
    n_hosts = jax.process_count()
    host = jax.process_index()
    assert global_arr.shape[0] % n_hosts == 0, (global_arr.shape, n_hosts)
    per_host = global_arr.shape[0] // n_hosts
    return global_arr[host * per_host:(host + 1) * per_host]


# -----------------------------
# Checkpoints
# -----------------------------

def restore_params_only(path: str, params):
    import flax.serialization as serialization
    with tj._open_file(path, 'rb') as f:
        data = f.read()
    raw = serialization.msgpack_restore(data)
    if hasattr(tj, 'migrate_legacy_v4155_params'):
        raw = tj.migrate_legacy_v4155_params(raw)
    if not isinstance(raw, dict) or 'params' not in raw:
        raise ValueError(f'Checkpoint does not contain params: {path}')
    restored = serialization.from_state_dict({'params': params}, {'params': raw['params']})
    log(f'[ckpt] params-only loaded: {path}')
    return restored['params']


def save_downstream_checkpoint(path: str, params, opt_state, step: int, best_metric: float, cfg: Dict[str, Any], extra=None):
    if not is_host0():
        return
    model_cfg = cfg.get('model', {})
    train_cfg = dict(cfg.get('training', {}))
    train_cfg['downstream'] = cfg.get('downstream', {})
    if extra:
        train_cfg['extra'] = extra
    tj.save_checkpoint(path, params, opt_state, epoch=0, step=step, best_val_loss=-best_metric,
                       model_config=model_cfg, step_in_epoch=0, steps_per_epoch=0,
                       training_config=train_cfg)


# -----------------------------
# Model/optimizer/mesh
# -----------------------------

def make_optimizer(cfg: Dict[str, Any], total_steps: int):
    t = cfg.get('training', {})
    lr = float(t.get('lr', t.get('learning_rate', 2e-5)))
    warmup_steps = int(t.get('warmup_steps', max(1, int(total_steps * float(t.get('warmup_ratio', 0.06))))))
    wd = float(t.get('weight_decay', 0.01))
    end_ratio = float(t.get('min_lr_ratio', 0.1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps), end_value=lr * end_ratio)
    return optax.chain(
        optax.clip_by_global_norm(float(t.get('max_grad_norm', 1.0))),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=wd),
    )


def build_sharded_fns_if_needed(cfg: Dict[str, Any], mesh):
    version = cfg.get('model', {}).get('model_version', '')
    mesh_model = int(cfg.get('training', {}).get('mesh_model', 1))
    if version != 'spatial-r1-v3.9.4' or mesh_model <= 1:
        return None
    import models.legacy.dawn_spatial_v394_exp as dawn394
    m = cfg['model']; tr = cfg.get('training', {})
    n_know = int(m.get('n_know', 25200)) // mesh_model
    n_qk = int(m.get('n_qk', 1580)) // mesh_model
    n_v = int(m.get('n_v', 2600)) // mesh_model
    ck = max(1, math.ceil(n_know / max(1, int(tr.get('n_chunks_know', 1)))))
    cq = max(1, math.ceil(n_qk / max(1, int(tr.get('n_chunks_qk', 1)))))
    cv = max(1, math.ceil(n_v / max(1, int(tr.get('n_chunks_v', 1)))))
    single_chunk = max(ck, cv)
    paired_chunk = cq
    return (dawn394.make_sharded_srw(mesh, max_chunk_size=single_chunk),
            dawn394.make_sharded_srw_paired(mesh, max_chunk_size=paired_chunk))


def model_apply_train(model, params, input_ids, labels, attention_mask, dropout_key, sharded_fns):
    kwargs = {}
    if sharded_fns is not None:
        kwargs['sharded_fns'] = sharded_fns
    return model.apply({'params': params}, input_ids, labels=labels,
                       attention_mask=attention_mask, deterministic=False,
                       rngs={'dropout': dropout_key}, **kwargs)


def model_apply_logits(model, params, input_ids, attention_mask, sharded_fns):
    kwargs = {}
    if sharded_fns is not None:
        kwargs['sharded_fns'] = sharded_fns
    return model.apply({'params': params}, input_ids, labels=None,
                       attention_mask=attention_mask, deterministic=True,
                       rngs={'dropout': jax.random.PRNGKey(0)}, **kwargs)['logits']


def make_train_step(model, optimizer, sharded_fns, aux_weight: float, tau_weight: float):
    @jax.jit
    def train_step(params, opt_state, input_ids, labels, attention_mask, dropout_key):
        def loss_fn(p):
            out = model_apply_train(model, p, input_ids, labels, attention_mask, dropout_key, sharded_fns)
            lm_loss = out['loss']
            aux_loss = out.get('aux_loss', jnp.float32(0.0))
            tau_reg = out.get('tau_reg', jnp.float32(0.0))
            total = lm_loss + aux_weight * aux_loss + tau_weight * tau_reg
            acc = out['correct'] / jnp.maximum(out['valid_count'], 1)
            return total, {'loss': total, 'lm_loss': lm_loss, 'aux_loss': aux_loss, 'tau_reg': tau_reg, 'acc': acc,
                           'valid_count': out['valid_count']}
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        metrics['grad_norm'] = optax.global_norm(grads)
        return new_params, new_opt, metrics
    return train_step


def make_score_step(model, sharded_fns):
    @jax.jit
    def score_step(params, input_ids, score_mask, attention_mask):
        logits = model_apply_logits(model, params, input_ids, attention_mask, sharded_fns)
        logits = logits[:, :-1, :]
        target = input_ids[:, 1:]
        mask = score_mask[:, 1:].astype(jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        tok = jnp.take_along_axis(logp, target[..., None], axis=-1).squeeze(-1)
        summed = (tok * mask).sum(axis=-1)
        denom = jnp.maximum(mask.sum(axis=-1), 1.0)
        return summed / denom
    return score_step


# -----------------------------
# Evaluation
# -----------------------------

def flatten_eval(eval_rows):
    flat, meta = [], []
    for ex_id, row in enumerate(eval_rows):
        for cand_id, cand in enumerate(row['candidates']):
            flat.append(cand)
            meta.append((ex_id, cand_id, row['gold_index']))
    return flat, meta


def evaluate(params, score_step, eval_rows, batch_size: int, max_seq_len: int, pad_token_id: int, data_sharding):
    flat, meta = flatten_eval(eval_rows)
    # Pad flat candidates to full global batches for static shape.
    n = len(flat)
    n_batches = math.ceil(n / batch_size)
    scores_by_ex: Dict[int, List[Tuple[int, float]]] = {}
    gold_by_ex: Dict[int, int] = {}
    for b in range(n_batches):
        idxs = list(range(b * batch_size, min((b + 1) * batch_size, n)))
        valid_n = len(idxs)
        if valid_n < batch_size:
            idxs = idxs + [idxs[-1]] * (batch_size - valid_n)
        g_ids, g_mask, g_attn = build_candidate_global_batch(flat, idxs, max_seq_len, pad_token_id)
        l_ids, l_mask, l_attn = local_slice(g_ids), local_slice(g_mask), local_slice(g_attn)
        gs = (batch_size, max_seq_len)
        ids = tj.shard_to_mesh(l_ids, data_sharding, gs)
        sm = tj.shard_to_mesh(l_mask, data_sharding, gs)
        am = tj.shard_to_mesh(l_attn, data_sharding, gs)
        sc_arr = score_step(params, ids, sm, am)
        # Multi-host SPMD arrays can span non-addressable devices; do not
        # jax.device_get() them directly. Gather the global candidate scores
        # across hosts, then only host 0 materializes/records metrics.
        sc_global = np.asarray(process_allgather(sc_arr, tiled=True)).reshape(-1)[:valid_n]
        if is_host0():
            for s, mi in zip(sc_global, idxs[:valid_n]):
                ex_id, cand_id, gold = meta[mi]
                scores_by_ex.setdefault(ex_id, []).append((cand_id, float(s)))
                gold_by_ex[ex_id] = gold
    if not is_host0():
        return {'accuracy': 0.0, 'total': 0}
    correct = 0
    total = 0
    for ex_id, xs in scores_by_ex.items():
        pred = max(xs, key=lambda x: x[1])[0]
        gold = gold_by_ex[ex_id]
        correct += int(pred == gold)
        total += 1
    return {'accuracy': correct / max(total, 1), 'total': total}


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--init-from', default=None)
    ap.add_argument('--resume-from', default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ds_cfg = cfg.get('downstream', cfg.get('data', {}))
    task = ds_cfg.get('task') or cfg.get('task')
    if not task:
        raise ValueError('Config must set downstream.task')

    if args.init_from and args.resume_from:
        raise ValueError('Use only one of --init-from or --resume-from')

    cfg_resume = cfg.get('resume_from') or ds_cfg.get('resume_from')
    resume_from = args.resume_from or cfg_resume
    init_from = None if resume_from else (args.init_from or cfg.get('init_from') or ds_cfg.get('init_from'))

    seed = int(cfg.get('seed', 1))
    random.seed(seed + jax.process_index())
    np.random.seed(seed + jax.process_index())

    if AutoTokenizer is None:
        raise RuntimeError('transformers is not installed. Install transformers.')
    tok_name = cfg.get('tokenizer', ds_cfg.get('tokenizer', 'bert-base-uncased'))
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    pad_id = ensure_pad_token(tokenizer)

    # Build model from the existing train_jax registry. No train_jax/model code is modified here.
    model = tj.build_model_from_config(cfg)
    model_version = cfg.get('model', {}).get('model_version', 'baseline')
    is_baseline = model_version == 'baseline'
    is_spatial = model_version in ('spatial-r1-v3.9.4', 'spatial-r1-v4.1.5.2', 'spatial-r1-v4.1.5.5', 'dawn_srw')

    mcfg, tcfg = cfg.get('model', {}), cfg.get('training', {})
    max_seq_len = int(ds_cfg.get('max_seq_len', mcfg.get('max_seq_len', 512)))
    batch_size = int(tcfg.get('batch_size', 64))
    eval_batch_size = int(tcfg.get('eval_batch_size', batch_size))

    total_devices = jax.device_count()
    n_hosts = jax.process_count()
    host = jax.process_index()
    if batch_size % total_devices != 0:
        raise ValueError(f'training.batch_size={batch_size} must be divisible by device_count={total_devices}')
    if eval_batch_size % total_devices != 0:
        raise ValueError(f'eval_batch_size={eval_batch_size} must be divisible by device_count={total_devices}')

    # Load downstream data, not C4.
    raw_train, raw_eval = load_raw_splits(ds_cfg, task)
    train_rows = make_train_rows(raw_train, task, tokenizer, max_seq_len, ds_cfg.get('max_train_samples'), seed, bool(ds_cfg.get('add_eos', False)))
    eval_rows = make_eval_rows(raw_eval, task, tokenizer, max_seq_len, ds_cfg.get('max_eval_samples'), bool(ds_cfg.get('add_eos', False)))
    if not train_rows or not eval_rows:
        raise RuntimeError(f'Empty downstream rows: train={len(train_rows)}, eval={len(eval_rows)}')

    num_epochs = int(tcfg.get('num_epochs', 3))
    max_steps_cfg = tcfg.get('max_steps')
    steps_per_epoch = max(1, len(train_rows) // batch_size)
    total_steps = int(max_steps_cfg) if max_steps_cfg else steps_per_epoch * num_epochs
    log_every = int(tcfg.get('log_interval', 20))
    eval_every = int(tcfg.get('eval_interval', 200))
    save_every = int(tcfg.get('checkpoint_interval', 500))

    # Output/run dir.
    ckpt_root = cfg.get('checkpoint_dir') or ds_cfg.get('checkpoint_dir')
    if not ckpt_root:
        raise ValueError('Config must set checkpoint_dir')
    # train_jax-style run directory: never write directly into a fixed run_name.
    # Treat cfg.run_name as a human-readable prefix only, then append timestamp+pid.
    # This prevents reruns from overwriting the same downstream folder.
    if resume_from:
        run_dir = str(resume_from).rstrip('/')
        run_name = run_dir.rstrip('/').rsplit('/', 1)[-1]
    else:
        run_prefix = cfg.get('run_name') or f"{model_version}_{task}"
        if not str(run_prefix).startswith('run_'):
            run_name = f"run_v{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        else:
            run_name = f"{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        run_dir = join_path(ckpt_root, run_name)
    train_log_path = join_path(run_dir, 'training_log.txt')
    metrics_csv_path = join_path(run_dir, 'metrics.csv')
    summary_path = join_path(run_dir, 'results_summary.txt')

    def record(msg: str):
        if is_host0():
            print(msg, flush=True)
            append_text(train_log_path, msg + '\n')

    if is_host0():
        record('=' * 60)
        record(f'Downstream fine-tune: model={model_version} task={task}')
        record(f'Hosts={n_hosts} devices={total_devices} local_devices={jax.local_device_count()} host_id={host}')
        record(f'batch={batch_size} eval_batch={eval_batch_size} max_seq_len={max_seq_len}')
        record(f'train_rows={len(train_rows)} eval_rows={len(eval_rows)} total_steps={total_steps}')
        record(f'init_from={init_from or "<none>"}')
        record(f'resume_from={resume_from or "<none>"}')
        record(f'run_dir={run_dir}')
        record('=' * 60)

    # Initialize params.
    key = jax.random.PRNGKey(seed)
    dummy_ids = jnp.ones((1, min(max_seq_len, 32)), dtype=jnp.int32)
    dummy_labels = jnp.ones_like(dummy_ids)
    variables = model.init({'params': key, 'dropout': key}, dummy_ids, labels=dummy_labels, deterministic=True)
    params = variables['params']

    # Mesh/shard params.
    mesh_model = int(tcfg.get('mesh_model', 1))
    mesh_data = int(tcfg.get('mesh_data', 0)) or (total_devices // mesh_model)
    mesh = tj.create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    param_shardings = tj.get_param_shardings(params, mesh, is_baseline=is_baseline)
    params = tj.shard_params_to_mesh(params, param_shardings)

    sharded_fns = build_sharded_fns_if_needed(cfg, mesh)

    optimizer = make_optimizer(cfg, total_steps)
    opt_state = optimizer.init(params)

    # Load params-only or resume full state.
    global_step = 0
    best_acc = -1.0
    if resume_from:
        rp = resolve_checkpoint_path(resume_from)
        ckpt = tj.load_checkpoint(rp, params, opt_state)
        params = ckpt['params']; opt_state = ckpt['opt_state']
        global_step = int(ckpt.get('step', 0))
        # save_checkpoint stores best as -best_metric in best_val_loss.
        best_acc = float(-ckpt.get('best_val_loss', 1.0))
        log(f'[resume] loaded params+optimizer+step from: {rp} step={global_step}')
    elif init_from:
        ip = resolve_checkpoint_path(init_from)
        params = restore_params_only(ip, params)
        opt_state = optimizer.init(params)
        global_step = 0
        best_acc = -1.0
        log(f'[init] loaded params only from: {ip}; optimizer=fresh; step=0')
    else:
        log('[init] no init-from: random-init downstream control')

    # Verify step consistency.
    if n_hosts > 1:
        local = np.array([global_step], dtype=np.int32)
        all_steps = np.asarray(process_allgather(local)).flatten()
        if not np.all(all_steps == global_step):
            raise RuntimeError(f'global_step mismatch across hosts: {all_steps.tolist()}')
        log(f'[verified] global_step={global_step} consistent across {n_hosts} hosts')

    # TrainJAX-style run metadata: config.json plus append-only text/csv logs.
    # Do not write per-eval JSON files.
    if is_host0():
        write_json(join_path(run_dir, 'config.json'), cfg)
        write_text(metrics_csv_path, 'phase,step,loss,lm_loss,acc,grad_norm,tokens,eval_acc,total,best_acc,elapsed_sec\n')

    train_step = make_train_step(model, optimizer, sharded_fns,
                                 float(tcfg.get('aux_weight', 0.0)), float(tcfg.get('tau_weight', 0.0)))
    score_step = make_score_step(model, sharded_fns)

    # Initial eval.
    ev = evaluate(params, score_step, eval_rows, eval_batch_size, max_seq_len, pad_id, data_sharding)
    if is_host0():
        ev['step'] = global_step
        record(f"[eval] step={global_step} acc={ev['accuracy']:.4f} total={ev['total']}")
        append_text(metrics_csv_path, f"eval,{global_step},,,,,,{ev['accuracy']},{ev['total']},{max(best_acc, ev['accuracy']):.6f},0.0\n")
        if ev['accuracy'] > best_acc:
            best_acc = ev['accuracy']
            save_downstream_checkpoint(join_path(run_dir, 'best_model.flax'), params, opt_state, global_step, best_acc, cfg, {'task': task})

    # Training.
    rng = jax.random.PRNGKey(seed + 1000 + global_step)
    t0 = time.time()
    epoch = global_step // max(steps_per_epoch, 1)
    while global_step < total_steps:
        # Deterministic shuffle per epoch; all hosts compute same global batch ids.
        epoch = global_step // max(steps_per_epoch, 1)
        rng_py = random.Random(seed + epoch)
        order = list(range(len(train_rows)))
        rng_py.shuffle(order)
        pos = (global_step % max(steps_per_epoch, 1)) * batch_size
        if pos + batch_size <= len(order):
            idxs = order[pos:pos + batch_size]
        else:
            idxs = (order[pos:] + order[:batch_size - (len(order) - pos)])
        if len(idxs) < batch_size:
            idxs = (idxs * ((batch_size // max(len(idxs), 1)) + 1))[:batch_size]

        g_ids, g_labels, g_attn = build_global_train_batch(train_rows, idxs, max_seq_len, pad_id)
        l_ids, l_labels, l_attn = local_slice(g_ids), local_slice(g_labels), local_slice(g_attn)
        gs = (batch_size, max_seq_len)
        ids = tj.shard_to_mesh(l_ids, data_sharding, gs)
        labels = tj.shard_to_mesh(l_labels, data_sharding, gs)
        attn = tj.shard_to_mesh(l_attn, data_sharding, gs)
        rng, step_rng = jax.random.split(rng)
        params, opt_state, metrics = train_step(params, opt_state, ids, labels, attn, step_rng)
        global_step += 1

        if global_step % log_every == 0 or global_step == 1:
            # Metrics may be global sharded arrays in multi-host mode. Gather
            # each scalar and print only on host 0.
            m = jax.tree.map(lambda x: np.asarray(process_allgather(x)).reshape(-1)[0], metrics)
            if is_host0():
                elapsed = time.time() - t0
                tok = global_step * batch_size * max_seq_len
                record(f"[train] step={global_step}/{total_steps} loss={float(m['lm_loss']):.4f} acc={float(m['acc']):.4f} grad={float(m['grad_norm']):.3f} tokens={tok} time={elapsed:.1f}s")
                append_text(metrics_csv_path, f"train,{global_step},{float(m['loss']):.6f},{float(m['lm_loss']):.6f},{float(m['acc']):.6f},{float(m['grad_norm']):.6f},{tok},,,{best_acc:.6f},{elapsed:.3f}\n")

        if global_step % eval_every == 0 or global_step == total_steps:
            ev = evaluate(params, score_step, eval_rows, eval_batch_size, max_seq_len, pad_id, data_sharding)
            if is_host0():
                ev['step'] = global_step
                record(f"[eval] step={global_step} acc={ev['accuracy']:.4f} total={ev['total']}")
                append_text(metrics_csv_path, f"eval,{global_step},,,,,,{ev['accuracy']},{ev['total']},{max(best_acc, ev['accuracy']):.6f},{time.time() - t0:.3f}\n")
                if ev['accuracy'] > best_acc:
                    best_acc = ev['accuracy']
                    save_downstream_checkpoint(join_path(run_dir, 'best_model.flax'), params, opt_state, global_step, best_acc, cfg, {'task': task})

        # Save periodic resume checkpoints only during training.
        # Do NOT force a final full checkpoint at task end: on multi-host TPU
        # that can leave non-host0 workers exiting/starting the next task while
        # host0 is still writing a 4.7GB checkpoint, which stalls the sequence.
        # best_model.flax is still saved whenever eval improves.
        if save_every and save_every > 0 and (global_step % save_every == 0) and (global_step < total_steps):
            save_downstream_checkpoint(join_path(run_dir, f'checkpoint_step{global_step}.flax'), params, opt_state, global_step, best_acc, cfg, {'task': task})

    if is_host0():
        final_msg = f'[done] task={task} best_acc={best_acc:.4f} step={global_step} run_dir={run_dir}'
        record(final_msg)
        write_text(summary_path, final_msg + '\n')

    # Explicit end-of-task barrier.  The outer sequence script starts the next
    # Python process independently on each worker, so all hosts must leave this
    # task together.  Without this, fast non-host0 workers can start the next
    # config while host0 is still writing logs/checkpoints, causing hangs.
    if n_hosts > 1:
        done = np.array([global_step], dtype=np.int32)
        gathered_done = np.asarray(process_allgather(done)).reshape(-1)
        if not np.all(gathered_done == global_step):
            raise RuntimeError(f'end-of-task step mismatch across hosts: {gathered_done.tolist()}')


if __name__ == '__main__':
    main()

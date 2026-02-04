"""
DAWN v17.1-JAX Training Script

JAX/Flax native training for DAWN v17.1 model.
- optax optimizer with warmup + cosine decay
- jax.jit compiled train/eval steps
- Neuron usage EMA monitoring
- orbax checkpointing
- bf16 mixed precision support

Usage:
    python scripts/train_jax.py --config configs/train_config_v17_1_tpu_memopt_40M_c4_5B_seed1_token_routing_v_expand6.yaml
    python scripts/train_jax.py --config configs/train_config.yaml --from-scratch
    python scripts/train_jax.py --config configs/train_config.yaml --resume checkpoints/run_xxx
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import random
import argparse
import yaml
from datetime import datetime
from functools import partial

from models.model_v17_1_jax import DAWN

# ============================================================
# Constants
# ============================================================

LOG_INTERVAL = 100


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Config
# ============================================================

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg):
    """Build DAWN model from config dict."""
    mcfg = cfg['model']
    # Resolve vocab_size from tokenizer later; use default for now
    model = DAWN(
        vocab_size=mcfg.get('vocab_size', 30522),
        d_model=mcfg.get('d_model', 384),
        n_layers=mcfg.get('n_layers', 12),
        n_heads=mcfg.get('n_heads', 6),
        rank=mcfg.get('rank', 64),
        max_seq_len=mcfg.get('max_seq_len', 512),
        d_space=mcfg.get('d_space', 64),
        n_feature_qk=mcfg.get('n_feature_qk', 56),
        n_feature_v=mcfg.get('n_feature_v', 24),
        top_k_feature_qk=mcfg.get('top_k_feature_qk', 16),
        top_k_feature_v=mcfg.get('top_k_feature_v', 6),
        n_restore_qk=mcfg.get('n_restore_qk', 56),
        n_restore_v=mcfg.get('n_restore_v', 24),
        top_k_restore_qk=mcfg.get('top_k_restore_qk', 16),
        top_k_restore_v=mcfg.get('top_k_restore_v', 6),
        n_feature_know=mcfg.get('n_feature_know', 24),
        n_restore_know=mcfg.get('n_restore_know', 24),
        top_k_feature_know=mcfg.get('top_k_feature_know', 4),
        top_k_restore_know=mcfg.get('top_k_restore_know', 4),
        knowledge_rank=mcfg.get('knowledge_rank', 128),
        dropout_rate=mcfg.get('dropout', 0.1),
        router_dropout=mcfg.get('router_dropout', 0.1),
        gradient_checkpointing=mcfg.get('gradient_checkpointing', False),
    )
    return model


# ============================================================
# Data loading (reuse PyTorch DataLoader, convert to JAX)
# ============================================================

def pytorch_loader_to_jax_batches(dataloader):
    """Yield JAX arrays from a PyTorch DataLoader."""
    for batch in dataloader:
        input_ids = jnp.array(batch['input_ids'].numpy())
        attention_mask = jnp.array(batch['attention_mask'].numpy())
        yield input_ids, attention_mask




# ============================================================
# Parameter count
# ============================================================

def count_parameters(params):
    """Count total parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


# ============================================================
# Orthogonality + diversity loss (inline for jit)
# ============================================================

def compute_orthogonality_loss(params, rank, knowledge_rank, n_feature_qk, n_restore_qk):
    """Compute orthogonality loss from shared neuron params.

    Matches the model's 6-group computation:
      f_neurons = [feature_qk ; feature_v]  -> split at n_feature_qk
      r_neurons = [restore_qk ; restore_v]  -> split at n_restore_qk
      feature_know, restore_know             -> separate params
    """
    sn = params['shared_neurons']
    I_rank = jnp.eye(rank)[jnp.newaxis]
    I_know = jnp.eye(knowledge_rank)[jnp.newaxis]

    f_neurons = sn['f_neurons']
    r_neurons = sn['r_neurons']
    feature_know = sn['feature_know']
    restore_know = sn['restore_know']

    # Split f_neurons into feature_qk [N_fqk, D, R] and feature_v [N_fv, D, R]
    W_fqk = f_neurons[:n_feature_qk]
    W_fv = f_neurons[n_feature_qk:]
    WtW_fqk = jnp.matmul(W_fqk.transpose(0, 2, 1), W_fqk)
    loss_fqk = ((WtW_fqk - I_rank) ** 2).mean()
    WtW_fv = jnp.matmul(W_fv.transpose(0, 2, 1), W_fv)
    loss_fv = ((WtW_fv - I_rank) ** 2).mean()

    # Split r_neurons into restore_qk [N_rqk, R, D] and restore_v [N_rv, R, D]
    W_rqk = r_neurons[:n_restore_qk]
    W_rv = r_neurons[n_restore_qk:]
    WWt_rqk = jnp.matmul(W_rqk, W_rqk.transpose(0, 2, 1))
    loss_rqk = ((WWt_rqk - I_rank) ** 2).mean()
    WWt_rv = jnp.matmul(W_rv, W_rv.transpose(0, 2, 1))
    loss_rv = ((WWt_rv - I_rank) ** 2).mean()

    WtW_fk = jnp.matmul(feature_know.transpose(0, 2, 1), feature_know)
    loss_fk = ((WtW_fk - I_know) ** 2).mean()

    WWt_rk = jnp.matmul(restore_know, restore_know.transpose(0, 2, 1))
    loss_rk = ((WWt_rk - I_know) ** 2).mean()

    return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fk + loss_rk) / 6


def compute_knowledge_diversity_loss(params):
    """Compute knowledge diversity loss from shared neuron params."""
    sn = params['shared_neurons']

    feat_know = sn['feature_know']
    feat_flat = feat_know.reshape(feat_know.shape[0], -1)
    feat_norm = feat_flat / (jnp.linalg.norm(feat_flat, axis=-1, keepdims=True) + 1e-8)
    feat_sim = jnp.matmul(feat_norm, feat_norm.T)
    mask_f = ~jnp.eye(feat_sim.shape[0], dtype=jnp.bool_)
    feat_loss = jnp.abs(feat_sim * mask_f).sum() / mask_f.sum()

    rest_know = sn['restore_know']
    rest_flat = rest_know.reshape(rest_know.shape[0], -1)
    rest_norm = rest_flat / (jnp.linalg.norm(rest_flat, axis=-1, keepdims=True) + 1e-8)
    rest_sim = jnp.matmul(rest_norm, rest_norm.T)
    mask_r = ~jnp.eye(rest_sim.shape[0], dtype=jnp.bool_)
    rest_loss = jnp.abs(rest_sim * mask_r).sum() / mask_r.sum()

    return (feat_loss + rest_loss) / 2


# ============================================================
# Train / eval steps
# ============================================================

def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk):
    """Create a jit-compiled training step function."""

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, dropout_key):
        # Labels for CLM: input_ids shifted, padding masked
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            result = model.apply(
                {'params': params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={'dropout': dropout_key},
            )
            ce_loss = result['loss']
            aux_loss = result['aux_loss']

            # Auxiliary losses computed from params (6-group orthogonality)
            orth_loss = compute_orthogonality_loss(
                params, rank, knowledge_rank, n_feature_qk, n_restore_qk)
            div_loss = compute_knowledge_diversity_loss(params)

            total_loss = (ce_loss
                          + lb_weight * aux_loss
                          + orth_weight * orth_loss
                          + div_weight * div_loss)

            return total_loss, (ce_loss, aux_loss, orth_loss, div_loss, result)

        (total_loss, (ce_loss, aux_loss, orth_loss, div_loss, result)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Accuracy: computed inside model via _compute_lm_loss (no full logits)
        metrics = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': aux_loss,
            'orth_loss': orth_loss,
            'div_loss': div_loss,
            'correct': result['correct'],
            'valid_count': result['valid_count'],
        }

        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step(model):
    """Create a jit-compiled evaluation step function.

    Note: dropout RNG is required because the lax.scan forward path always
    calls make_rng('dropout') (safe_dropout neutralizes it via deterministic flag).
    """

    @jax.jit
    def eval_step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        # deterministic=True → dropout masks are all-ones, but RNG key is still
        # needed for tracing (safe_dropout always generates a mask).
        eval_rng = jax.random.PRNGKey(0)  # fixed key — never used for real randomness
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
        )
        ce_loss = result['loss']
        correct = result['correct']
        valid_count = result['valid_count']

        return ce_loss, correct, valid_count

    return eval_step


# ============================================================
# Evaluation loop
# ============================================================

def evaluate(eval_step_fn, params, val_loader, max_batches=200):
    """Run evaluation and return avg loss and accuracy."""
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    for batch_idx, (input_ids, attention_mask) in enumerate(pytorch_loader_to_jax_batches(val_loader)):
        if batch_idx >= max_batches:
            break
        ce_loss, correct, valid_count = eval_step_fn(params, input_ids, attention_mask)
        n_valid = int(valid_count)
        total_loss += float(ce_loss) * n_valid
        total_correct += int(correct)
        total_valid += n_valid

    avg_loss = total_loss / total_valid if total_valid > 0 else 0.0
    avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
    return avg_loss, avg_acc


# ============================================================
# Checkpoint save / load
# ============================================================

def save_checkpoint(path, params, opt_state, epoch, step, best_val_loss, model_config):
    """Save checkpoint using flax serialization."""
    import flax.serialization as serialization
    ckpt = {
        'params': params,
        'opt_state': opt_state,
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'config': model_config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_data = serialization.to_bytes(ckpt)
    with open(path, 'wb') as f:
        f.write(bytes_data)
    print(f"  Checkpoint saved: {path} ({len(bytes_data) / 1e6:.1f} MB)")


def load_checkpoint(path, target_params, target_opt_state):
    """Load checkpoint using flax serialization."""
    import flax.serialization as serialization
    with open(path, 'rb') as f:
        bytes_data = f.read()
    target = {
        'params': target_params,
        'opt_state': target_opt_state,
        'epoch': 0,
        'step': 0,
        'best_val_loss': float('inf'),
        'config': {},
    }
    ckpt = serialization.from_bytes(target, bytes_data)
    print(f"  Checkpoint loaded: {path}")
    return ckpt


# ============================================================
# Logging
# ============================================================

def log_message(msg, log_file=None):
    """Print and optionally write to log file."""
    print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')


def format_time(seconds):
    """Format seconds to H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train DAWN v17.1 (JAX/Flax)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file or directory to resume from')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    cli_args = parser.parse_args()

    # ----------------------------------------------------------
    # Load config
    # ----------------------------------------------------------
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_config(config_path)

    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Training params
    tcfg = cfg['training']
    batch_size = cli_args.batch_size or tcfg['batch_size']
    num_epochs = cli_args.epochs or tcfg['num_epochs']
    lr = cli_args.lr or tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4))
    weight_decay = tcfg.get('weight_decay', 0.1)
    warmup_ratio = tcfg.get('warmup_ratio', 0.06)
    orth_weight = tcfg.get('orthogonality_weight', 0.01)
    div_weight = tcfg.get('diversity_weight', 0.1)
    lb_weight = tcfg.get('load_balance_weight', 2e-5)

    max_seq_len = cfg['model'].get('max_seq_len', 512)

    checkpoint_dir = Path(cfg.get('checkpoint_dir', 'checkpoints_jax'))
    log_dir = Path(cfg.get('log_dir', 'logs_jax'))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Print JAX device info
    # ----------------------------------------------------------
    devices = jax.devices()
    print(f"\n{'='*60}")
    print(f"DAWN v17.1-JAX Training")
    print(f"{'='*60}")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {devices}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Config: {config_path}")
    print(f"Seed: {seed}")

    # jax_log_compiles is too noisy (hundreds of warnings during init).
    # Use stage-based print markers instead to pinpoint where it dies.

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")

    from utils.data import load_data
    train_loader, val_loader, tokenizer = load_data(
        cfg['data'],
        max_length=max_seq_len,
        batch_size=batch_size,
    )
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    cfg['model']['vocab_size'] = vocab_size
    model = build_model_from_config(cfg)

    # Initialize
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones((1, max_seq_len), dtype=jnp.int32)

    print("=== Starting model.init ===", flush=True)
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        dummy_input,
        deterministic=True,
    )
    params = variables['params']
    print("=== model.init done ===", flush=True)

    n_params = count_parameters(params)
    print(f"\nModel parameters: {n_params:,}")
    for line in model.get_model_info():
        print(line)

    rank = cfg['model'].get('rank', 64)
    knowledge_rank = cfg['model'].get('knowledge_rank', 128)

    # ----------------------------------------------------------
    # OOM check: dummy forward pass with actual batch shape
    # ----------------------------------------------------------
    print(f"\n=== OOM check: dummy forward with batch_size={batch_size}, seq_len={max_seq_len} ===", flush=True)
    try:
        dummy_batch = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
        dummy_labels = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((batch_size, max_seq_len), dtype=jnp.int32)
        rng, dummy_dropout_rng = jax.random.split(rng)

        # Training forward (deterministic=False uses dropout → more memory)
        dummy_result = model.apply(
            {'params': params},
            dummy_batch,
            labels=dummy_labels,
            attention_mask=dummy_mask,
            deterministic=False,
            rngs={'dropout': dummy_dropout_rng},
        )
        jax.block_until_ready(dummy_result['loss'])
        print(f"  Forward OK — loss={float(dummy_result['loss']):.4f}", flush=True)

        # Eval forward (deterministic=True)
        dummy_eval_result = model.apply(
            {'params': params},
            dummy_batch,
            labels=dummy_labels,
            attention_mask=dummy_mask,
            deterministic=True,
            rngs={'dropout': dummy_dropout_rng},
        )
        jax.block_until_ready(dummy_eval_result['loss'])
        print(f"  Eval forward OK — loss={float(dummy_eval_result['loss']):.4f}", flush=True)

        del dummy_batch, dummy_labels, dummy_mask, dummy_result, dummy_eval_result
        print("=== OOM check passed ===\n", flush=True)
    except Exception as e:
        print(f"  *** OOM check FAILED: {e}")
        print(f"  This likely means the model is too large for the device memory.")
        print(f"  Try reducing batch_size or enabling gradient_checkpointing.")
        raise

    # ----------------------------------------------------------
    # Optimizer (warmup + cosine decay + optional gradient accumulation)
    # ----------------------------------------------------------
    grad_accum_steps = tcfg.get('gradient_accumulation_steps', 1)

    steps_per_epoch = len(train_loader)
    # Schedule counts optimizer steps (after accumulation), not micro-steps
    effective_steps_per_epoch = steps_per_epoch // grad_accum_steps
    total_steps = num_epochs * effective_steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum_steps)
    else:
        optimizer = base_optimizer
    opt_state = optimizer.init(params)

    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum steps: {grad_accum_steps}")
    print(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print(f"  Steps/epoch (micro): {steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Orth weight: {orth_weight}")
    print(f"  Div weight: {div_weight}")
    print(f"  LB weight: {lb_weight}")

    # ----------------------------------------------------------
    # Resume from checkpoint
    # ----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    resume_path = None
    if cli_args.resume:
        p = Path(cli_args.resume)
        if p.is_file():
            resume_path = p
        elif p.is_dir():
            # Find best_model.flax or latest checkpoint
            candidates = sorted(p.glob('*.flax'))
            if candidates:
                resume_path = candidates[-1]
    elif not cli_args.from_scratch:
        # Auto-search for latest checkpoint
        candidates = sorted(checkpoint_dir.glob('*.flax'))
        if candidates:
            resume_path = candidates[-1]

    if resume_path and resume_path.exists():
        print(f"\nResuming from: {resume_path}")
        ckpt = load_checkpoint(resume_path, params, opt_state)
        params = ckpt['params']
        opt_state = ckpt['opt_state']
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, step {global_step}, best_val_loss={best_val_loss:.4f}")
    else:
        if not cli_args.from_scratch:
            print("\nNo checkpoint found. Starting from scratch.")
        else:
            print("\nStarting from scratch (--from-scratch).")

    # ----------------------------------------------------------
    # Create jit-compiled step functions
    # ----------------------------------------------------------
    n_feature_qk = cfg['model'].get('n_feature_qk', 56)
    n_restore_qk = cfg['model'].get('n_restore_qk', 56)
    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        rank, knowledge_rank, n_feature_qk, n_restore_qk)
    eval_step_fn = create_eval_step(model)

    # ----------------------------------------------------------
    # Pre-compile train_step and check HBM requirements
    # ----------------------------------------------------------
    print("\n=== Lowering train_step (no execution) ===", flush=True)
    try:
        dummy_ids = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((batch_size, max_seq_len), dtype=jnp.int32)
        dummy_drng = jax.random.PRNGKey(0)

        lowered = train_step_fn.lower(
            params, opt_state,
            dummy_ids, dummy_mask, dummy_drng,
        )
        print("=== Lowering done, compiling XLA ===", flush=True)

        compiled = lowered.compile()
        print("=== Compile done ===", flush=True)

        # Memory analysis (may not be available on all backends)
        try:
            mem = compiled.memory_analysis()
            if mem is not None:
                temp_gb = getattr(mem, 'temp_size_in_bytes', 0) / 1e9
                arg_gb = getattr(mem, 'argument_size_in_bytes', 0) / 1e9
                out_gb = getattr(mem, 'output_size_in_bytes', 0) / 1e9
                alias_gb = getattr(mem, 'alias_size_in_bytes', 0) / 1e9
                print(f"  HBM temp (activations): {temp_gb:.2f} GB", flush=True)
                print(f"  HBM args (params+opt): {arg_gb:.2f} GB", flush=True)
                print(f"  HBM output: {out_gb:.2f} GB", flush=True)
                print(f"  HBM alias: {alias_gb:.2f} GB", flush=True)
                print(f"  HBM total: {temp_gb + arg_gb + out_gb:.2f} GB", flush=True)
            else:
                print("  memory_analysis() returned None", flush=True)
        except Exception as me:
            print(f"  memory_analysis not available: {me}", flush=True)

        del dummy_ids, dummy_mask, dummy_drng
        print("=== train_step compile check passed ===\n", flush=True)

    except Exception as e:
        print(f"  *** train_step compile FAILED: {e}", flush=True)
        print(f"  Backward graph likely exceeds HBM. Try reducing batch_size.", flush=True)
        raise

    # ----------------------------------------------------------
    # Training log file
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_log_file = str(log_dir / f'training_log_{timestamp}.txt')
    log_message(f"DAWN v17.1-JAX Training Log - {timestamp}", training_log_file)
    log_message(f"Config: {config_path}", training_log_file)
    log_message(f"Parameters: {n_params:,}", training_log_file)
    log_message(f"Total steps: {total_steps}", training_log_file)
    log_message("", training_log_file)

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("=== Starting training loop ===", flush=True)
    print(f"{'='*60}")

    train_start_time = time.time()
    total_micro_steps = num_epochs * steps_per_epoch
    val_interval = cfg['training'].get('val_interval', 5000)
    ckpt_interval = cfg['training'].get('checkpoint_interval', 5000)
    first_step_done = False

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_valid = 0
        epoch_steps = 0

        # Window accumulators for periodic logging
        win_loss = 0.0
        win_ce = 0.0
        win_aux = 0.0
        win_orth = 0.0
        win_div = 0.0
        win_correct = 0
        win_valid = 0
        win_count = 0
        win_start_time = time.time()

        for local_step, (input_ids, attention_mask) in enumerate(
                pytorch_loader_to_jax_batches(train_loader)):

            # Skip steps if resuming mid-epoch
            step_in_epoch = epoch * steps_per_epoch + local_step
            if step_in_epoch < global_step:
                continue

            if not first_step_done:
                print(f"=== First train_step: compiling JIT (this may take minutes) ===", flush=True)

            # New dropout key each step
            rng, dropout_key = jax.random.split(rng)

            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                input_ids, attention_mask, dropout_key,
            )

            if not first_step_done:
                # Force sync to catch OOM here, not later
                jax.block_until_ready(metrics['total_loss'])
                print(f"=== First train_step done — loss={float(metrics['total_loss']):.4f} ===", flush=True)
                first_step_done = True

            # Accumulate
            m_total = float(metrics['total_loss'])
            m_ce = float(metrics['ce_loss'])
            m_aux = float(metrics['aux_loss'])
            m_orth = float(metrics['orth_loss'])
            m_div = float(metrics['div_loss'])
            m_correct = int(metrics['correct'])
            m_valid = int(metrics['valid_count'])

            epoch_loss += m_ce * m_valid
            epoch_correct += m_correct
            epoch_valid += m_valid
            epoch_steps += 1

            win_loss += m_total
            win_ce += m_ce
            win_aux += m_aux
            win_orth += m_orth
            win_div += m_div
            win_correct += m_correct
            win_valid += m_valid
            win_count += 1

            global_step += 1

            # ---- Periodic logging ----
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - win_start_time
                steps_per_sec = win_count / elapsed if elapsed > 0 else 0
                avg_loss = win_loss / win_count
                avg_ce = win_ce / win_count
                avg_aux = win_aux / win_count
                avg_orth = win_orth / win_count
                avg_div = win_div / win_count
                avg_acc = win_correct / win_valid if win_valid > 0 else 0.0

                # Current LR from schedule (indexed by optimizer step, not micro-step)
                opt_step = global_step // grad_accum_steps
                current_lr = float(schedule(opt_step))

                total_elapsed = time.time() - train_start_time
                progress = global_step / total_micro_steps * 100

                msg = (
                    f"[Step {global_step}/{total_micro_steps} ({progress:.1f}%)] "
                    f"loss={avg_loss:.4f} ce={avg_ce:.4f} aux={avg_aux:.4f} "
                    f"orth={avg_orth:.2e} div={avg_div:.2e} | "
                    f"acc={avg_acc:.4f} lr={current_lr:.2e} "
                    f"step/s={steps_per_sec:.1f} "
                    f"elapsed={format_time(total_elapsed)}"
                )
                log_message(msg, training_log_file)

                # TPU memory stats
                try:
                    mem = jax.devices()[0].memory_stats()
                    if mem:
                        used = mem.get('bytes_in_use', 0) / 1e9
                        peak = mem.get('peak_bytes_in_use', 0) / 1e9
                        limit = mem.get('bytes_limit', 0) / 1e9
                        log_message(
                            f"      HBM: {used:.2f}G / {limit:.2f}G "
                            f"(peak={peak:.2f}G, free={limit - used:.2f}G)",
                            training_log_file)
                except Exception:
                    pass

                # Reset window
                win_loss = 0.0
                win_ce = 0.0
                win_aux = 0.0
                win_orth = 0.0
                win_div = 0.0
                win_correct = 0
                win_valid = 0
                win_count = 0
                win_start_time = time.time()

            # ---- Mid-epoch validation ----
            if global_step % val_interval == 0 and global_step > 0:
                log_message(f"\n  Mid-epoch validation at step {global_step}...", training_log_file)
                val_loss, val_acc = evaluate(eval_step_fn, params, val_loader)
                log_message(
                    f"  Val loss={val_loss:.4f}, Val acc={val_acc:.4f}",
                    training_log_file,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        checkpoint_dir / 'best_model.flax',
                        params, opt_state,
                        epoch, global_step, best_val_loss,
                        cfg['model'],
                    )
                    log_message(f"  New best model saved! val_loss={best_val_loss:.4f}", training_log_file)

            # ---- Mid-epoch checkpoint ----
            if global_step % ckpt_interval == 0 and global_step > 0:
                save_checkpoint(
                    checkpoint_dir / f'checkpoint_step{global_step}.flax',
                    params, opt_state,
                    epoch, global_step, best_val_loss,
                    cfg['model'],
                )

        # ---- End of epoch ----
        epoch_elapsed = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / epoch_valid if epoch_valid > 0 else 0.0
        epoch_avg_acc = epoch_correct / epoch_valid if epoch_valid > 0 else 0.0

        log_message(
            f"\n{'='*60}\n"
            f"Epoch {epoch} complete in {format_time(epoch_elapsed)}\n"
            f"  Train loss={epoch_avg_loss:.4f}, Train acc={epoch_avg_acc:.4f}\n"
            f"{'='*60}",
            training_log_file,
        )

        # End-of-epoch validation
        log_message("  Running end-of-epoch validation...", training_log_file)
        val_loss, val_acc = evaluate(eval_step_fn, params, val_loader)
        log_message(f"  Val loss={val_loss:.4f}, Val acc={val_acc:.4f}", training_log_file)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save epoch checkpoint
        save_checkpoint(
            checkpoint_dir / f'checkpoint_epoch{epoch}.flax',
            params, opt_state,
            epoch + 1, global_step, best_val_loss,
            cfg['model'],
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / 'best_model.flax',
                params, opt_state,
                epoch + 1, global_step, best_val_loss,
                cfg['model'],
            )
            log_message(f"  New best model! val_loss={best_val_loss:.4f}", training_log_file)

        log_message(
            f"  Best val loss so far: {best_val_loss:.4f}",
            training_log_file,
        )

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    total_time = time.time() - train_start_time
    log_message(
        f"\n{'='*60}\n"
        f"Training complete!\n"
        f"  Total time: {format_time(total_time)}\n"
        f"  Best val loss: {best_val_loss:.4f}\n"
        f"  Final step: {global_step}\n"
        f"{'='*60}",
        training_log_file,
    )


if __name__ == '__main__':
    main()

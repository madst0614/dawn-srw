"""
DAWN (Dynamic Architecture With Neurons) Training Script

Usage:
    # Default training (auto-resume from latest checkpoint)
    python scripts/train.py

    # Start from scratch
    python scripts/train.py --from-scratch

    # Resume from specific checkpoint folder
    python scripts/train.py --resume checkpoints/run_20240101_120000_1234

    # Resume from specific .pt file
    python scripts/train.py --resume /path/to/checkpoint_epoch1_step5000.pt

    # Use custom config file
    python scripts/train.py --config configs/my_config.yaml

Checkpoint Options:
    (default)        - Auto-search for latest best_model.pt and resume
    --from-scratch   - Disable auto-search, start from scratch
    --resume <folder> - Resume from best_model.pt in specified folder
    --resume <file>   - Resume directly from specified .pt file
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy torch inductor warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')
warnings.filterwarnings('ignore', message='.*online softmax.*')

# CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TPU/XLA support (optional)
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time
import numpy as np
import math
import random


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Speed optimization: TF32 and cuDNN settings for Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

from models import create_model_by_version, print_version_info, normalize_version, build_model_kwargs, build_args_config, load_model_params_to_args, get_routing_log_info
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.checkpoint import load_checkpoint_smart, load_optimizer_state, strip_compile_prefix
from utils.data import MLM_CONFIG, apply_mlm_masking, TextDataset, collate_fn_dynamic_padding, load_data, compute_mlm_accuracy


# ============================================================
# CONSTANTS
# ============================================================

# Dead neuron threshold: neurons with usage EMA below this are considered "dead"
# A value of 0.01 means the neuron is selected less than 1% of the time
DEAD_NEURON_THRESHOLD = 0.01

# Weight concentration warning threshold
# If max routing weight > 0.8, the model is over-relying on a single neuron
WEIGHT_CONCENTRATION_WARNING = 0.8

# Logging interval: how often to log training metrics (every N steps)
# Both console and file logging use this interval
LOG_INTERVAL = 100


# ============================================================
# DEBUG LOGGING FUNCTIONS
# ============================================================

class DebugLogger:
    """Debug logger for basis_up analysis, gradients, and orthogonality loss"""

    # Epochs to log detailed stats
    LOG_EPOCHS = {0, 1, 5, 10, 20, 50, 100}

    def __init__(self, log_file_path):
        self.log_file = log_file_path
        self.epoch_logged = set()

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DAWN Debug Log - Basis_up Analysis\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def should_log_epoch(self, epoch):
        """Check if this epoch should be logged"""
        return epoch in self.LOG_EPOCHS or epoch % 10 == 0

    def log(self, message):
        """Write message to debug log file"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def log_section(self, title):
        """Log a section header"""
        self.log(f"\n{'='*60}")
        self.log(f"{title}")
        self.log(f"{'='*60}")

    def log_basis_stats(self, model, epoch, step=None):
        """
        Log basis statistics at specific epochs

        Supports:
        - v7.8: NeuronBank W_Q/K/V/O (no basis)
        - v7.7: basis_qk / basis_vo
        - v7.6: basis_down / basis_up

        Tracks:
        - Singular value distribution per basis/neuron
        - Overall condition number
        - Potential collapse detection
        """
        base_model = get_underlying_model(model)

        # Only for models with shared_basis (aliased as neuron_bank for v7.8)
        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Basis/Neuron Stats - Epoch {epoch}{step_str}")

        sb = base_model.shared_basis

        # Detect model version
        is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')  # NeuronBank
        is_v77 = hasattr(sb, 'basis_qk')

        with torch.no_grad():
            if is_v78:
                # v7.8: NeuronBank with independent W_Q/K/V/O per neuron
                self.log(f"\n[v7.8 NeuronBank - Independent Neuron W Matrices]")
                n_neurons = sb.n_neurons

                for W_name, W in [('W_Q', sb.W_Q), ('W_K', sb.W_K), ('W_V', sb.W_V), ('W_O', sb.W_O)]:
                    # Per-neuron condition numbers
                    cond_nums = []
                    for i in range(n_neurons):
                        _, S, _ = torch.linalg.svd(W[i])
                        cond = (S[0] / (S[-1] + 1e-10)).item()
                        cond_nums.append(cond)

                    avg_cond = sum(cond_nums) / len(cond_nums)
                    max_cond = max(cond_nums)
                    min_cond = min(cond_nums)

                    self.log(f"\n  {W_name} condition numbers (across {n_neurons} neurons):")
                    self.log(f"    mean = {avg_cond:.2f}, max = {max_cond:.2f}, min = {min_cond:.2f}")

                    if max_cond > 100:
                        self.log(f"    ⚠️  WARNING: High condition in some neurons!")
                    elif max_cond > 10:
                        self.log(f"    ⚠️  CAUTION: Condition numbers drifting from init")
                    else:
                        self.log(f"    ✓ Orthogonality well maintained")

            elif is_v77:
                # v7.7: basis_vo is the "output" basis (O uses its transpose)
                basis_o = sb.basis_vo.detach()  # [n_basis, D, rank]
                basis_o_name = "Basis_VO"
                basis_qk = sb.basis_qk.detach()  # [n_basis, D, rank]
                basis_qk_name = "Basis_QK"

                n_basis = basis_o.shape[0]

                # Per-basis singular values for O projection basis
                self.log(f"\n[{basis_o_name} Per-Basis Singular Values (top 5)]")
                all_singular_values = []
                for i in range(n_basis):
                    _, S, _ = torch.linalg.svd(basis_o[i])
                    all_singular_values.extend(S.cpu().tolist())
                    self.log(f"  {basis_o_name}[{i}]: {S[:5].cpu().numpy()}")

                # Overall condition number for O basis
                B_o_flat = basis_o.view(n_basis, -1)
                _, S_all, _ = torch.linalg.svd(B_o_flat)
                sigma_max = S_all[0].item()
                sigma_min = S_all[-1].item()
                condition_number = sigma_max / (sigma_min + 1e-10)

                self.log(f"\n[{basis_o_name} Overall Condition Number]")
                self.log(f"  σ_max = {sigma_max:.6f}")
                self.log(f"  σ_min = {sigma_min:.10f}")
                self.log(f"  Condition number = {condition_number:.2e}")

                # Collapse detection
                if condition_number > 1e6:
                    self.log(f"  ⚠️  WARNING: High condition number indicates potential collapse!")
                elif condition_number > 1e4:
                    self.log(f"  ⚠️  CAUTION: Condition number getting high")
                else:
                    self.log(f"  ✓ Condition number is healthy")

                # Also log QK basis for comparison
                B_qk_flat = basis_qk.view(n_basis, -1)
                _, S_qk, _ = torch.linalg.svd(B_qk_flat)
                cond_qk = S_qk[0].item() / (S_qk[-1].item() + 1e-10)

                self.log(f"\n[{basis_qk_name} Condition Number (for comparison)]")
                self.log(f"  σ_max = {S_qk[0].item():.6f}")
                self.log(f"  σ_min = {S_qk[-1].item():.10f}")
                self.log(f"  Condition number = {cond_qk:.2e}")

            elif hasattr(sb, 'basis_up') and hasattr(sb, 'basis_down'):
                # v7.6: basis_up is the "output" basis
                basis_o = sb.basis_up.detach()  # [n_basis, rank, D]
                basis_o_name = "Basis_up"
                basis_qk = sb.basis_down.detach()  # [n_basis, D, rank]
                basis_qk_name = "Basis_down"

                n_basis = basis_o.shape[0]

                # Per-basis singular values for O projection basis
                self.log(f"\n[{basis_o_name} Per-Basis Singular Values (top 5)]")
                all_singular_values = []
                for i in range(n_basis):
                    _, S, _ = torch.linalg.svd(basis_o[i])
                    all_singular_values.extend(S.cpu().tolist())
                    self.log(f"  {basis_o_name}[{i}]: {S[:5].cpu().numpy()}")

                # Overall condition number for O basis
                B_o_flat = basis_o.view(n_basis, -1)
                _, S_all, _ = torch.linalg.svd(B_o_flat)
                sigma_max = S_all[0].item()
                sigma_min = S_all[-1].item()
                condition_number = sigma_max / (sigma_min + 1e-10)

                self.log(f"\n[{basis_o_name} Overall Condition Number]")
                self.log(f"  σ_max = {sigma_max:.6f}")
                self.log(f"  σ_min = {sigma_min:.10f}")
                self.log(f"  Condition number = {condition_number:.2e}")

                # Collapse detection
                if condition_number > 1e6:
                    self.log(f"  ⚠️  WARNING: High condition number indicates potential collapse!")
                elif condition_number > 1e4:
                    self.log(f"  ⚠️  CAUTION: Condition number getting high")
                else:
                    self.log(f"  ✓ Condition number is healthy")

                # Also log QK basis for comparison
                B_qk_flat = basis_qk.view(n_basis, -1)
                _, S_qk, _ = torch.linalg.svd(B_qk_flat)
                cond_qk = S_qk[0].item() / (S_qk[-1].item() + 1e-10)

                self.log(f"\n[{basis_qk_name} Condition Number (for comparison)]")
                self.log(f"  σ_max = {S_qk[0].item():.6f}")
                self.log(f"  σ_min = {S_qk[-1].item():.10f}")
                self.log(f"  Condition number = {cond_qk:.2e}")
            else:
                # Other versions (v7.5, v7.1, etc.) - skip detailed basis logging
                self.log(f"\n[Note: Detailed basis stats not available for this model version]")

    def log_gradient_flow(self, model, epoch, step=None):
        """
        Log gradient flow for basis/neuron parameters

        Supports:
        - v7.8: neuron_bank.W_Q/K/V/O
        - v7.6/v7.7: basis_down/up, basis_qk/vo

        Tracks:
        - Gradient norms for basis/neuron W parameters
        - Parameter norms
        - Gradient/parameter ratio
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Gradient Flow - Epoch {epoch}{step_str}")

        self.log("\n[Basis/Neuron Gradient Analysis]")
        for name, param in base_model.named_parameters():
            # Match basis or neuron_bank parameters
            if ('basis' in name or 'neuron_bank' in name) and param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                ratio = grad_norm / (param_norm + 1e-10)

                self.log(f"  {name}:")
                self.log(f"    grad_norm = {grad_norm:.8f}")
                self.log(f"    param_norm = {param_norm:.6f}")
                self.log(f"    ratio = {ratio:.8f}")

                # Gradient health check
                if grad_norm < 1e-8:
                    self.log(f"    ⚠️  WARNING: Vanishing gradient!")
                elif grad_norm > 100:
                    self.log(f"    ⚠️  WARNING: Exploding gradient!")

    def log_orthogonality_breakdown(self, model, epoch, step=None):
        """
        Log orthogonality loss breakdown

        Supports:
        - v7.8: neuron_bank W_Q/K/V/O (per-neuron orthogonality)
        - v7.7: basis_qk / basis_vo
        - v7.6: basis_down / basis_up

        Tracks:
        - ortho loss for each basis/neuron
        - Which direction dominates
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Orthogonality Loss Breakdown - Epoch {epoch}{step_str}")

        with torch.no_grad():
            sb = base_model.shared_basis

            # Detect model version
            is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')
            is_v77 = hasattr(sb, 'basis_qk')

            if is_v78:
                # v7.8: Per-neuron W orthogonality
                self.log(f"\n[Orthogonality Loss Components (v7.8 - Per Neuron)]")
                n_neurons = sb.n_neurons

                for W_name, W in [('W_Q', sb.W_Q), ('W_K', sb.W_K), ('W_V', sb.W_V), ('W_O', sb.W_O)]:
                    # Compute per-neuron orthogonality error
                    ortho_errors = []
                    for i in range(n_neurons):
                        W_i = W[i]  # [D, rank] or [rank, D]
                        if W_i.shape[0] > W_i.shape[1]:
                            gram = W_i.T @ W_i  # [rank, rank]
                        else:
                            gram = W_i @ W_i.T  # [rank, rank]
                        I = torch.eye(gram.shape[0], device=gram.device)
                        error = ((gram - I) ** 2).mean().item()
                        ortho_errors.append(error)

                    avg_error = sum(ortho_errors) / len(ortho_errors)
                    max_error = max(ortho_errors)
                    self.log(f"  {W_name}: mean_error={avg_error:.8f}, max_error={max_error:.8f}")

                self.log(f"\n  (Lower is better, 0 = perfect orthogonality)")

            elif is_v77:
                # v7.7: basis_qk and basis_vo (both QR initialized)
                n_basis = sb.n_basis
                I = torch.eye(n_basis, device=sb.basis_qk.device)
                B_qk = sb.basis_qk.view(n_basis, -1)
                B_vo = sb.basis_vo.view(n_basis, -1)
                gram_qk = B_qk @ B_qk.T
                gram_vo = B_vo @ B_vo.T
                ortho_qk = ((gram_qk - I) ** 2).mean().item()
                ortho_vo = ((gram_vo - I) ** 2).mean().item()
                off_diagonal_mask = ~I.bool()

                self.log(f"\n[Orthogonality Loss Components (v7.7)]")
                self.log(f"  ortho_qk = {ortho_qk:.8f}")
                self.log(f"  ortho_vo = {ortho_vo:.8f}")
                self.log(f"  total (avg) = {(ortho_qk + ortho_vo) / 2:.8f}")

                self.log(f"\n[Gram Matrix Diagnostics]")
                self.log(f"  gram_qk diagonal mean: {gram_qk.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_qk off-diag mean: {gram_qk[off_diagonal_mask].mean().item():.6f} (target: 0.0)")
                self.log(f"  gram_vo diagonal mean: {gram_vo.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_vo off-diag mean: {gram_vo[off_diagonal_mask].mean().item():.6f} (target: 0.0)")

                if ortho_vo > ortho_qk * 10:
                    self.log(f"\n  ⚠️  ortho_vo >> ortho_qk: V/O projection learning may be unstable")
                elif ortho_qk > ortho_vo * 10:
                    self.log(f"\n  ⚠️  ortho_qk >> ortho_vo: Q/K projection learning may be unstable")
            elif hasattr(sb, 'basis_up') and hasattr(sb, 'basis_down'):
                # v7.6: basis_down and basis_up
                n_basis = sb.n_basis
                I = torch.eye(n_basis, device=sb.basis_down.device)
                B_down = sb.basis_down.view(n_basis, -1)
                gram_down = B_down @ B_down.T
                ortho_down = ((gram_down - I) ** 2).mean().item()

                # basis_up orthogonality (with normalization for v7.6)
                B_up = sb.basis_up.view(n_basis, -1)
                B_up_norm = F.normalize(B_up, dim=-1)
                gram_up = B_up_norm @ B_up_norm.T
                off_diagonal_mask = ~I.bool()
                ortho_up = (gram_up[off_diagonal_mask] ** 2).mean().item()

                self.log(f"\n[Orthogonality Loss Components (v7.6)]")
                self.log(f"  ortho_down = {ortho_down:.8f}")
                self.log(f"  ortho_up   = {ortho_up:.8f}")
                self.log(f"  total (avg) = {(ortho_down + ortho_up) / 2:.8f}")

                self.log(f"\n[Gram Matrix Diagnostics]")
                self.log(f"  gram_down diagonal mean: {gram_down.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_down off-diag mean: {gram_down[off_diagonal_mask].mean().item():.6f} (target: 0.0)")
                self.log(f"  gram_up diagonal mean: {gram_up.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_up off-diag mean: {gram_up[off_diagonal_mask].mean().item():.6f} (target: 0.0)")

                if ortho_up > ortho_down * 10:
                    self.log(f"\n  ⚠️  ortho_up >> ortho_down: O projection learning may be unstable")
                elif ortho_down > ortho_up * 10:
                    self.log(f"\n  ⚠️  ortho_down >> ortho_up: Q/K/V projection learning may be unstable")
            else:
                # Other versions - skip
                self.log(f"\n[Note: Orthogonality breakdown not available for this model version]")

    def log_recipe_analysis(self, model, sample_input, epoch, step=None):
        """
        Log Recipe → W_O analysis

        Supports:
        - v7.8: NeuronBank W_O (no recipe, direct neuron mixing)
        - v7.7: get_basis_o() (basis_vo.T)
        - v7.6: get_basis_up()

        Tracks:
        - Recipe/neuron weight entropy (diversity)
        - Max weight concentration
        - W_O singular values
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Recipe/Neuron → W_O Analysis - Epoch {epoch}{step_str}")

        sb = base_model.shared_basis
        is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')
        is_v77 = hasattr(sb, 'basis_qk')

        base_model.eval()
        with torch.no_grad():
            B, S = sample_input.shape
            device = sample_input.device

            # Get first layer's qkv_dynamic
            layer = base_model.layers[0]
            qkv = layer.qkv_dynamic

            # Routing
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = base_model.token_emb(sample_input) + base_model.pos_emb(pos)

            scores = qkv.W_router(x)
            topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
            weights = F.softmax(topk_scores, dim=-1)

            if is_v78:
                # v7.8: No recipe, analyze neuron weight distribution
                self.log(f"\n[v7.8 Neuron Weight Statistics (no recipe)]")

                # Neuron weight entropy
                entropy = (-weights * torch.log(weights + 1e-10)).sum(-1).mean()
                max_entropy = math.log(qkv.k)  # Maximum entropy for k neurons

                # Max weight
                max_weight = weights.max(-1)[0].mean()

                self.log(f"  Neuron weight entropy: {entropy.item():.4f} (max possible: {max_entropy:.4f})")
                self.log(f"  Normalized entropy: {entropy.item() / max_entropy:.4f}")
                self.log(f"  Max weight (mean): {max_weight.item():.4f}")

                if max_weight.item() > WEIGHT_CONCENTRATION_WARNING:
                    self.log(f"  ⚠️  WARNING: Neuron weights too concentrated on single neuron!")

                # W_O construction via neuron mixing
                self.log(f"\n[W_O Construction (v7.8): weighted_avg(neuron_W_O)]")
                nb = qkv.neuron_bank
                W_O_neurons = nb.get_W_O(topk_idx)  # [B, S, k, rank, D]
                weights_exp = weights.unsqueeze(-1).unsqueeze(-1)
                W_O = (W_O_neurons * weights_exp).sum(dim=2)  # [B, S, rank, D]

                # Analyze sample W_O
                W_O_sample = W_O[0, 0]  # [rank, D]
                _, S_wo, _ = torch.linalg.svd(W_O_sample)

                self.log(f"\n[Mixed W_O Singular Values (sample token)]")
                self.log(f"  Top 5: {S_wo[:5].cpu().numpy()}")
                self.log(f"  σ_max/σ_min: {(S_wo[0] / (S_wo[-1] + 1e-10)).item():.2e}")

                if S_wo[-1].item() < 1e-6:
                    self.log(f"  ⚠️  WARNING: W_O has near-zero singular values!")

            elif hasattr(qkv, 'neuron_recipe_O'):
                # v7.5/v7.6/v7.7: Recipe-based analysis
                # Recipe O
                recipe_O = qkv.neuron_recipe_O[topk_idx]
                token_recipe_O = (recipe_O * weights.unsqueeze(-1)).sum(dim=2)
                token_recipe_O = F.softmax(token_recipe_O, dim=-1)  # [B, S, n_basis]

                # Recipe entropy
                entropy = (-token_recipe_O * torch.log(token_recipe_O + 1e-10)).sum(-1).mean()
                max_entropy = math.log(qkv.n_basis)  # Maximum possible entropy

                # Max weight
                max_weight = token_recipe_O.max(-1)[0].mean()

                self.log(f"\n[Recipe_O Statistics]")
                self.log(f"  Entropy: {entropy.item():.4f} (max possible: {max_entropy:.4f})")
                self.log(f"  Normalized entropy: {entropy.item() / max_entropy:.4f}")
                self.log(f"  Max weight (mean): {max_weight.item():.4f}")

                if max_weight.item() > WEIGHT_CONCENTRATION_WARNING:
                    self.log(f"  ⚠️  WARNING: Recipe too concentrated on single basis!")

                # W_O singular values - handle v7.6 vs v7.7
                if is_v77:
                    basis_o = sb.get_basis_o()  # [n_basis, rank, D] = basis_vo.T
                    self.log(f"\n[W_O Construction (v7.7): recipe_O @ basis_vo.T]")
                else:
                    basis_o = sb.get_basis_up()  # [n_basis, rank, D]
                    self.log(f"\n[W_O Construction (v7.6): recipe_O @ basis_up]")

                W_O = torch.einsum('bsn,nrd->bsrd', token_recipe_O, basis_o)  # [B, S, rank, D]

                # Analyze first token's W_O
                W_O_sample = W_O[0, 0]  # [rank, D]
                _, S_wo, _ = torch.linalg.svd(W_O_sample)

                self.log(f"\n[W_O Singular Values (sample token)]")
                self.log(f"  Top 5: {S_wo[:5].cpu().numpy()}")
                self.log(f"  σ_max/σ_min: {(S_wo[0] / (S_wo[-1] + 1e-10)).item():.2e}")

                # Check if W_O is degenerating
                if S_wo[-1].item() < 1e-6:
                    self.log(f"  ⚠️  WARNING: W_O has near-zero singular values!")

            else:
                # Other versions - skip recipe analysis
                self.log(f"\n[Note: Recipe analysis not available for this model version]")

        base_model.train()

    def log_epoch_summary(self, model, sample_input, epoch, step=None):
        """Log all debug info for an epoch"""
        self.log_basis_stats(model, epoch, step)
        self.log_orthogonality_breakdown(model, epoch, step)
        self.log_recipe_analysis(model, sample_input, epoch, step)

    def log_post_backward(self, model, epoch, step=None):
        """Log gradient info after backward pass"""
        self.log_gradient_flow(model, epoch, step)


def get_underlying_model(model):
    """Get the underlying model from a potentially torch.compile() wrapped model"""
    # torch.compile() wraps models in OptimizedModule with _orig_mod attribute
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def set_v18_debug_mode(model, enabled: bool):
    """Toggle debug_mode for v18.x models (controls routing stats computation)"""
    base_model = get_underlying_model(model)
    # v18.x: debug_mode is in GlobalRouters (base_model.router)
    if hasattr(base_model, 'router') and hasattr(base_model.router, 'debug_mode'):
        base_model.router.debug_mode = enabled


def format_v18_routing_stats(routing_infos, model_version, prefix="  "):
    """
    Format v18 routing stats from routing_infos.
    Returns list of formatted strings (ready for print or file.write).
    """
    if not routing_infos:
        return []

    lines = []
    n = len(routing_infos)

    # Helper for mean±std across layers
    def get_mean_std(key, sub='attention'):
        vals = [ri.get(sub, ri).get(key, 0) for ri in routing_infos]
        if not vals:
            return 0, 0
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return mean, std

    # Paths (mean±std across layers) - skip for v18.4/v18.5 (simplified output)
    if not model_version.startswith('18.4') and not model_version.startswith('18.5'):
        p_fqk_Q, s_fqk_Q = get_mean_std('n_paths_fqk_Q')
        p_fqk_K, s_fqk_K = get_mean_std('n_paths_fqk_K')
        p_fv, s_fv = get_mean_std('n_paths_fv')
        p_rqk_Q, s_rqk_Q = get_mean_std('n_paths_rqk_Q')
        p_rqk_K, s_rqk_K = get_mean_std('n_paths_rqk_K')
        p_rv, s_rv = get_mean_std('n_paths_rv')
        p_kf, s_kf = get_mean_std('n_paths_feature', 'knowledge')
        p_kr, s_kr = get_mean_std('n_paths_restore', 'knowledge')
        lines.append(f"{prefix}[v18] Paths: fqk_Q={p_fqk_Q:.1f}±{s_fqk_Q:.1f} fqk_K={p_fqk_K:.1f}±{s_fqk_K:.1f} fv={p_fv:.1f}±{s_fv:.1f} rqk_Q={p_rqk_Q:.1f}±{s_rqk_Q:.1f} rqk_K={p_rqk_K:.1f}±{s_rqk_K:.1f} rv={p_rv:.1f}±{s_rv:.1f} kf={p_kf:.1f}±{s_kf:.1f} kr={p_kr:.1f}±{s_kr:.1f}")

    # Selected neurons per token (mean±std across layers)
    sel_fqk_Q, ss_fqk_Q = get_mean_std('selected_fqk_Q')
    sel_fqk_K, ss_fqk_K = get_mean_std('selected_fqk_K')
    sel_fv, ss_fv = get_mean_std('selected_fv')
    sel_rqk_Q, ss_rqk_Q = get_mean_std('selected_rqk_Q')
    sel_rqk_K, ss_rqk_K = get_mean_std('selected_rqk_K')
    sel_rv, ss_rv = get_mean_std('selected_rv')
    sel_kf, ss_kf = get_mean_std('selected_feature', 'knowledge')
    sel_kr, ss_kr = get_mean_std('selected_restore', 'knowledge')

    # v18.4/v18.5: Show pass rate % (selected / top_k)
    if model_version.startswith('18.4') or model_version.startswith('18.5'):
        # Get top_k from routing_info (computed in debug_mode only)
        top_k = routing_infos[0].get('attention', {}).get('top_k', 32) if routing_infos else 32
        def pct(sel): return sel / top_k * 100 if top_k > 0 else 0
        lines.append(f"{prefix}Selected: fqk_Q={sel_fqk_Q:.0f}({pct(sel_fqk_Q):.0f}%) fqk_K={sel_fqk_K:.0f}({pct(sel_fqk_K):.0f}%) fv={sel_fv:.0f}({pct(sel_fv):.0f}%) rqk_Q={sel_rqk_Q:.0f}({pct(sel_rqk_Q):.0f}%) rqk_K={sel_rqk_K:.0f}({pct(sel_rqk_K):.0f}%) rv={sel_rv:.0f}({pct(sel_rv):.0f}%) kf={sel_kf:.0f}({pct(sel_kf):.0f}%) kr={sel_kr:.0f}({pct(sel_kr):.0f}%)")
    else:
        lines.append(f"{prefix}Selected: fqk_Q={sel_fqk_Q:.0f}±{ss_fqk_Q:.0f} fqk_K={sel_fqk_K:.0f}±{ss_fqk_K:.0f} fv={sel_fv:.0f}±{ss_fv:.0f} rqk_Q={sel_rqk_Q:.0f}±{ss_rqk_Q:.0f} rqk_K={sel_rqk_K:.0f}±{ss_rqk_K:.0f} rv={sel_rv:.0f}±{ss_rv:.0f} kf={sel_kf:.0f}±{ss_kf:.0f} kr={sel_kr:.0f}±{ss_kr:.0f}")

    # Score distribution (mean±std, average across layers) - skip for v18.4/v18.5 (simplified output)
    if not model_version.startswith('18.4') and not model_version.startswith('18.5'):
        score_fqk_Q_mean = sum(ri.get('attention', ri).get('score_fqk_Q_mean', 0) for ri in routing_infos) / n
        score_fqk_Q_std = sum(ri.get('attention', ri).get('score_fqk_Q_std', 0) for ri in routing_infos) / n
        score_fqk_K_mean = sum(ri.get('attention', ri).get('score_fqk_K_mean', 0) for ri in routing_infos) / n
        score_fqk_K_std = sum(ri.get('attention', ri).get('score_fqk_K_std', 0) for ri in routing_infos) / n
        score_fv_mean = sum(ri.get('attention', ri).get('score_fv_mean', 0) for ri in routing_infos) / n
        score_fv_std = sum(ri.get('attention', ri).get('score_fv_std', 0) for ri in routing_infos) / n
        score_rqk_Q_mean = sum(ri.get('attention', ri).get('score_rqk_Q_mean', 0) for ri in routing_infos) / n
        score_rqk_Q_std = sum(ri.get('attention', ri).get('score_rqk_Q_std', 0) for ri in routing_infos) / n
        score_rqk_K_mean = sum(ri.get('attention', ri).get('score_rqk_K_mean', 0) for ri in routing_infos) / n
        score_rqk_K_std = sum(ri.get('attention', ri).get('score_rqk_K_std', 0) for ri in routing_infos) / n
        score_rv_mean = sum(ri.get('attention', ri).get('score_rv_mean', 0) for ri in routing_infos) / n
        score_rv_std = sum(ri.get('attention', ri).get('score_rv_std', 0) for ri in routing_infos) / n
        # Knowledge scores
        score_kf_mean = sum(ri.get('knowledge', {}).get('score_feature_mean', 0) for ri in routing_infos) / n
        score_kf_std = sum(ri.get('knowledge', {}).get('score_feature_std', 0) for ri in routing_infos) / n
        score_kr_mean = sum(ri.get('knowledge', {}).get('score_restore_mean', 0) for ri in routing_infos) / n
        score_kr_std = sum(ri.get('knowledge', {}).get('score_restore_std', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Score: fqk_Q={score_fqk_Q_mean:.2f}±{score_fqk_Q_std:.2f} fqk_K={score_fqk_K_mean:.2f}±{score_fqk_K_std:.2f} fv={score_fv_mean:.2f}±{score_fv_std:.2f} rqk_Q={score_rqk_Q_mean:.2f}±{score_rqk_Q_std:.2f} rqk_K={score_rqk_K_mean:.2f}±{score_rqk_K_std:.2f} rv={score_rv_mean:.2f}±{score_rv_std:.2f} kf={score_kf_mean:.2f}±{score_kf_std:.2f} kr={score_kr_mean:.2f}±{score_kr_std:.2f}")

    # v18.1+ specific: Tau and Gate with mean±std (18.1, 18.2, 18.3)
    # v18.4/v18.5 use relative tau, so absolute Tau values are not meaningful
    if model_version.startswith('18.') and not model_version.startswith('18.0') and not model_version.startswith('18.4') and not model_version.startswith('18.5'):
        # Tau with std (absolute values, meaningful for v18.1-18.3)
        tau_fq = sum(ri.get('attention', {}).get('tau_fq', 0) for ri in routing_infos) / n
        tau_fq_std = sum(ri.get('attention', {}).get('tau_fq_std', 0) for ri in routing_infos) / n
        tau_fk = sum(ri.get('attention', {}).get('tau_fk', 0) for ri in routing_infos) / n
        tau_fk_std = sum(ri.get('attention', {}).get('tau_fk_std', 0) for ri in routing_infos) / n
        tau_fv = sum(ri.get('attention', {}).get('tau_fv', 0) for ri in routing_infos) / n
        tau_fv_std = sum(ri.get('attention', {}).get('tau_fv_std', 0) for ri in routing_infos) / n
        tau_rq = sum(ri.get('attention', {}).get('tau_rq', 0) for ri in routing_infos) / n
        tau_rq_std = sum(ri.get('attention', {}).get('tau_rq_std', 0) for ri in routing_infos) / n
        tau_rk = sum(ri.get('attention', {}).get('tau_rk', 0) for ri in routing_infos) / n
        tau_rk_std = sum(ri.get('attention', {}).get('tau_rk_std', 0) for ri in routing_infos) / n
        tau_rv = sum(ri.get('attention', {}).get('tau_rv', 0) for ri in routing_infos) / n
        tau_rv_std = sum(ri.get('attention', {}).get('tau_rv_std', 0) for ri in routing_infos) / n
        tau_kf = sum(ri.get('knowledge', {}).get('tau_feature', 0) for ri in routing_infos) / n
        tau_kf_std = sum(ri.get('knowledge', {}).get('tau_feature_std', 0) for ri in routing_infos) / n
        tau_kr = sum(ri.get('knowledge', {}).get('tau_restore', 0) for ri in routing_infos) / n
        tau_kr_std = sum(ri.get('knowledge', {}).get('tau_restore_std', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Tau: fq={tau_fq:.2f}±{tau_fq_std:.2f} fk={tau_fk:.2f}±{tau_fk_std:.2f} fv={tau_fv:.2f}±{tau_fv_std:.2f} rq={tau_rq:.2f}±{tau_rq_std:.2f} rk={tau_rk:.2f}±{tau_rk_std:.2f} rv={tau_rv:.2f}±{tau_rv_std:.2f} kf={tau_kf:.2f}±{tau_kf_std:.2f} kr={tau_kr:.2f}±{tau_kr_std:.2f}")

    # v18.4: tau_offset (learned parameter, in std units from mean) + gate_strength
    if model_version.startswith('18.4'):
        # tau_offset with +/- sign format
        off_fq = sum(ri.get('attention', {}).get('tau_offset_fq', 0) for ri in routing_infos) / n
        off_fk = sum(ri.get('attention', {}).get('tau_offset_fk', 0) for ri in routing_infos) / n
        off_fv = sum(ri.get('attention', {}).get('tau_offset_fv', 0) for ri in routing_infos) / n
        off_rq = sum(ri.get('attention', {}).get('tau_offset_rq', 0) for ri in routing_infos) / n
        off_rk = sum(ri.get('attention', {}).get('tau_offset_rk', 0) for ri in routing_infos) / n
        off_rv = sum(ri.get('attention', {}).get('tau_offset_rv', 0) for ri in routing_infos) / n
        off_kf = sum(ri.get('knowledge', {}).get('tau_offset_feature', 0) for ri in routing_infos) / n
        off_kr = sum(ri.get('knowledge', {}).get('tau_offset_restore', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}TauOff: fq={off_fq:+.2f} fk={off_fk:+.2f} fv={off_fv:+.2f} rq={off_rq:+.2f} rk={off_rk:+.2f} rv={off_rv:+.2f} kf={off_kf:+.2f} kr={off_kr:+.2f}")

        # gate_strength (tanh of max exp_gate, 0~1, low = more pass-through)
        gstr_fq = sum(ri.get('attention', {}).get('gstr_fq', 0) for ri in routing_infos) / n
        gstr_fk = sum(ri.get('attention', {}).get('gstr_fk', 0) for ri in routing_infos) / n
        gstr_fv = sum(ri.get('attention', {}).get('gstr_fv', 0) for ri in routing_infos) / n
        gstr_rq = sum(ri.get('attention', {}).get('gstr_rq', 0) for ri in routing_infos) / n
        gstr_rk = sum(ri.get('attention', {}).get('gstr_rk', 0) for ri in routing_infos) / n
        gstr_rv = sum(ri.get('attention', {}).get('gstr_rv', 0) for ri in routing_infos) / n
        gstr_kf = sum(ri.get('knowledge', {}).get('gstr_feature', 0) for ri in routing_infos) / n
        gstr_kr = sum(ri.get('knowledge', {}).get('gstr_restore', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}GateStr: fq={gstr_fq:.2f} fk={gstr_fk:.2f} fv={gstr_fv:.2f} rq={gstr_rq:.2f} rk={gstr_rk:.2f} rv={gstr_rv:.2f} kf={gstr_kf:.2f} kr={gstr_kr:.2f}")

        # Overlap (Q/K mask overlap ratio)
        ovlp_fqk = sum(ri.get('attention', {}).get('overlap_fqk', 0) for ri in routing_infos) / n
        ovlp_rqk = sum(ri.get('attention', {}).get('overlap_rqk', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Overlap: fqk={ovlp_fqk:.0%} rqk={ovlp_rqk:.0%}")

    # v18.5: tau_offset for both feature and restore (context-based learnable tau)
    if model_version.startswith('18.5'):
        # Feature tau_offset
        off_fq = sum(ri.get('attention', {}).get('tau_offset_fq', 0) for ri in routing_infos) / n
        off_fk = sum(ri.get('attention', {}).get('tau_offset_fk', 0) for ri in routing_infos) / n
        off_fv = sum(ri.get('attention', {}).get('tau_offset_fv', 0) for ri in routing_infos) / n
        off_kf = sum(ri.get('knowledge', {}).get('tau_offset_feature', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}TauOff(feat): fq={off_fq:+.2f} fk={off_fk:+.2f} fv={off_fv:+.2f} kf={off_kf:+.2f}")

        # Restore tau_offset (context-based learnable, Q/K separated)
        off_rq = sum(ri.get('attention', {}).get('tau_offset_rqk_Q', 0) for ri in routing_infos) / n
        off_rk = sum(ri.get('attention', {}).get('tau_offset_rqk_K', 0) for ri in routing_infos) / n
        off_rv = sum(ri.get('attention', {}).get('tau_offset_rv', 0) for ri in routing_infos) / n
        off_kr = sum(ri.get('knowledge', {}).get('tau_offset_restore', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}TauOff(rest): rq={off_rq:+.2f} rk={off_rk:+.2f} rv={off_rv:+.2f} kr={off_kr:+.2f}")

        # gate_strength (feature only from routing_info)
        gstr_fq = sum(ri.get('attention', {}).get('gstr_fq', 0) for ri in routing_infos) / n
        gstr_fk = sum(ri.get('attention', {}).get('gstr_fk', 0) for ri in routing_infos) / n
        gstr_fv = sum(ri.get('attention', {}).get('gstr_fv', 0) for ri in routing_infos) / n
        gstr_kf = sum(ri.get('knowledge', {}).get('gstr_feature', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}GateStr(feat): fq={gstr_fq:.2f} fk={gstr_fk:.2f} fv={gstr_fv:.2f} kf={gstr_kf:.2f}")

        # gate_strength for restore (context-based)
        gstr_rq = sum(ri.get('attention', {}).get('gstr_rqk_Q', 0) for ri in routing_infos) / n
        gstr_rk = sum(ri.get('attention', {}).get('gstr_rqk_K', 0) for ri in routing_infos) / n
        gstr_rv = sum(ri.get('attention', {}).get('gstr_rv', 0) for ri in routing_infos) / n
        gstr_kr = sum(ri.get('knowledge', {}).get('gstr_restore', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}GateStr(rest): rq={gstr_rq:.2f} rk={gstr_rk:.2f} rv={gstr_rv:.2f} kr={gstr_kr:.2f}")

        # Overlap (Q/K mask overlap ratio)
        ovlp_fqk = sum(ri.get('attention', {}).get('overlap_fqk', 0) for ri in routing_infos) / n
        ovlp_rqk = sum(ri.get('attention', {}).get('overlap_rqk', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Overlap: fqk={ovlp_fqk:.0%} rqk={ovlp_rqk:.0%}")

    # Gate with std (v18.1-18.3, skip for v18.4/v18.5)
    if model_version.startswith('18.') and not model_version.startswith('18.0') and not model_version.startswith('18.4') and not model_version.startswith('18.5'):
        gate_fq = sum(ri.get('attention', {}).get('gate_fq_mean', 0) for ri in routing_infos) / n
        gate_fq_std = sum(ri.get('attention', {}).get('gate_fq_std', 0) for ri in routing_infos) / n
        gate_fk = sum(ri.get('attention', {}).get('gate_fk_mean', 0) for ri in routing_infos) / n
        gate_fk_std = sum(ri.get('attention', {}).get('gate_fk_std', 0) for ri in routing_infos) / n
        gate_fv = sum(ri.get('attention', {}).get('gate_fv_mean', 0) for ri in routing_infos) / n
        gate_fv_std = sum(ri.get('attention', {}).get('gate_fv_std', 0) for ri in routing_infos) / n
        gate_rq = sum(ri.get('attention', {}).get('gate_rq_mean', 0) for ri in routing_infos) / n
        gate_rq_std = sum(ri.get('attention', {}).get('gate_rq_std', 0) for ri in routing_infos) / n
        gate_rk = sum(ri.get('attention', {}).get('gate_rk_mean', 0) for ri in routing_infos) / n
        gate_rk_std = sum(ri.get('attention', {}).get('gate_rk_std', 0) for ri in routing_infos) / n
        gate_rv = sum(ri.get('attention', {}).get('gate_rv_mean', 0) for ri in routing_infos) / n
        gate_rv_std = sum(ri.get('attention', {}).get('gate_rv_std', 0) for ri in routing_infos) / n
        gate_kf = sum(ri.get('knowledge', {}).get('gate_feature_mean', 0) for ri in routing_infos) / n
        gate_kf_std = sum(ri.get('knowledge', {}).get('gate_feature_std', 0) for ri in routing_infos) / n
        gate_kr = sum(ri.get('knowledge', {}).get('gate_restore_mean', 0) for ri in routing_infos) / n
        gate_kr_std = sum(ri.get('knowledge', {}).get('gate_restore_std', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Gate: fq={gate_fq:.2f}±{gate_fq_std:.2f} fk={gate_fk:.2f}±{gate_fk_std:.2f} fv={gate_fv:.2f}±{gate_fv_std:.2f} rq={gate_rq:.2f}±{gate_rq_std:.2f} rk={gate_rk:.2f}±{gate_rk_std:.2f} rv={gate_rv:.2f}±{gate_rv_std:.2f} kf={gate_kf:.2f}±{gate_kf_std:.2f} kr={gate_kr:.2f}±{gate_kr_std:.2f}")

    # v18.3 specific: Confidence stats (skip for v18.4)
    if model_version.startswith('18.3'):
        conf_fq = sum(ri.get('attention', {}).get('conf_fq_mean', 0) for ri in routing_infos) / n
        conf_fq_std = sum(ri.get('attention', {}).get('conf_fq_std', 0) for ri in routing_infos) / n
        conf_fk = sum(ri.get('attention', {}).get('conf_fk_mean', 0) for ri in routing_infos) / n
        conf_fk_std = sum(ri.get('attention', {}).get('conf_fk_std', 0) for ri in routing_infos) / n
        conf_fv = sum(ri.get('attention', {}).get('conf_fv_mean', 0) for ri in routing_infos) / n
        conf_fv_std = sum(ri.get('attention', {}).get('conf_fv_std', 0) for ri in routing_infos) / n
        conf_rq = sum(ri.get('attention', {}).get('conf_rq_mean', 0) for ri in routing_infos) / n
        conf_rq_std = sum(ri.get('attention', {}).get('conf_rq_std', 0) for ri in routing_infos) / n
        conf_rk = sum(ri.get('attention', {}).get('conf_rk_mean', 0) for ri in routing_infos) / n
        conf_rk_std = sum(ri.get('attention', {}).get('conf_rk_std', 0) for ri in routing_infos) / n
        conf_rv = sum(ri.get('attention', {}).get('conf_rv_mean', 0) for ri in routing_infos) / n
        conf_rv_std = sum(ri.get('attention', {}).get('conf_rv_std', 0) for ri in routing_infos) / n
        conf_kf = sum(ri.get('knowledge', {}).get('conf_feature_mean', 0) for ri in routing_infos) / n
        conf_kf_std = sum(ri.get('knowledge', {}).get('conf_feature_std', 0) for ri in routing_infos) / n
        conf_kr = sum(ri.get('knowledge', {}).get('conf_restore_mean', 0) for ri in routing_infos) / n
        conf_kr_std = sum(ri.get('knowledge', {}).get('conf_restore_std', 0) for ri in routing_infos) / n
        lines.append(f"{prefix}Conf: fq={conf_fq:.2f}±{conf_fq_std:.2f} fk={conf_fk:.2f}±{conf_fk_std:.2f} fv={conf_fv:.2f}±{conf_fv_std:.2f} rq={conf_rq:.2f}±{conf_rq_std:.2f} rk={conf_rk:.2f}±{conf_rk_std:.2f} rv={conf_rv:.2f}±{conf_rv_std:.2f} kf={conf_kf:.2f}±{conf_kf_std:.2f} kr={conf_kr:.2f}±{conf_kr_std:.2f}")

    return lines


def format_v18_neuron_usage(router, model_version, prefix="      "):
    """
    Format v18 neuron usage stats from router.
    Returns list of formatted strings.

    Optimized: Move EMA tensors to CPU first (8 GPU syncs), then compute on CPU.
    """
    lines = []

    # Move all EMA tensors to CPU first (8 GPU syncs total)
    with torch.no_grad():
        ema_fq = router.usage_ema_feature_q.cpu()
        ema_fk = router.usage_ema_feature_k.cpu()
        ema_fv = router.usage_ema_feature_v.cpu()
        ema_rq = router.usage_ema_restore_q.cpu()
        ema_rk = router.usage_ema_restore_k.cpu()
        ema_rv = router.usage_ema_restore_v.cpu()
        ema_FK = router.usage_ema_feature_know.cpu()
        ema_RK = router.usage_ema_restore_know.cpu()

    # Calculate all stats on CPU (no GPU sync for .item())
    def calc_stats(ema):
        n = ema.numel()
        active = (ema > 0.01).sum().item()
        dead = (ema < DEAD_NEURON_THRESHOLD).float().mean().item()

        # Gini (CPU) - inline to avoid function call overhead
        x_sorted = torch.sort(ema)[0]
        idx = torch.arange(1, n + 1, dtype=ema.dtype)
        gini = (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()

        # Entropy (CPU)
        p = ema / (ema.sum() + 1e-8)
        log_p = torch.log(p + 1e-8)
        entropy = -(p * log_p).sum()
        max_ent = math.log(n)
        ent = (entropy / max_ent * 100).item() if max_ent > 0 else 0.0

        return active, n, dead, gini, ent

    fq = calc_stats(ema_fq)
    fk = calc_stats(ema_fk)
    fv = calc_stats(ema_fv)
    rq = calc_stats(ema_rq)
    rk = calc_stats(ema_rk)
    rv = calc_stats(ema_rv)
    FK = calc_stats(ema_FK)
    RK = calc_stats(ema_RK)

    lines.append(f"{prefix}Usage: FQ={fq[0]}/{fq[1]} FK={fk[0]}/{fk[1]} FV={fv[0]}/{fv[1]} | RQ={rq[0]}/{rq[1]} RK={rk[0]}/{rk[1]} RV={rv[0]}/{rv[1]} | Know: F={FK[0]}/{FK[1]} R={RK[0]}/{RK[1]}")
    lines.append(f"{prefix}Ent: FQ={fq[4]:.0f} FK={fk[4]:.0f} FV={fv[4]:.0f} RQ={rq[4]:.0f} RK={rk[4]:.0f} RV={rv[4]:.0f} FKnow={FK[4]:.0f} RKnow={RK[4]:.0f}")
    lines.append(f"{prefix}Dead: FQ={fq[2]:.1%} FK={fk[2]:.1%} FV={fv[2]:.1%} RQ={rq[2]:.1%} RK={rk[2]:.1%} RV={rv[2]:.1%} FKnow={FK[2]:.1%} RKnow={RK[2]:.1%}")
    lines.append(f"{prefix}Gini: FQ={fq[3]:.2f} FK={fk[3]:.2f} FV={fv[3]:.2f} RQ={rq[3]:.2f} RK={rk[3]:.2f} RV={rv[3]:.2f} FKnow={FK[3]:.2f} RKnow={RK[3]:.2f}")

    return lines


def format_v18_full_log(routing_infos, router, model_version, step, avg_loss, avg_acc,
                        ent_str, overlap_str, attn_str, tau_grad_info=None, prefix=""):
    """
    v18 콘솔/파일 공용 로그 포맷터.

    Args:
        routing_infos: list of routing info dicts from model forward
        router: UnifiedNeuronRouter instance (base_model.router.neuron_router)
        model_version: str like "18.2"
        step: current step (0-indexed, will display as step+1)
        avg_loss: average loss for window
        avg_acc: average accuracy for window
        ent_str: entropy string from get_routing_log_info
        overlap_str: Q/K overlap string
        attn_str: attention ratio string (e.g., "62/68/74...")
        tau_grad_info: optional dict with 'weight' and 'bias' gradient norms
        prefix: line prefix for indentation

    Returns: list of strings (each line)
    """
    lines = []

    # Header line
    lines.append(f"{prefix}[{step+1}] Loss:{avg_loss:.4f} Acc:{avg_acc:.4f} | {ent_str} | {overlap_str} | Attn:{attn_str}")

    # Routing stats (Paths, Selected, Score, Tau, Gate)
    inner_prefix = prefix + "      "
    lines.extend(format_v18_routing_stats(routing_infos, model_version, prefix=inner_prefix))

    # Neuron usage stats
    if router is not None:
        lines.extend(format_v18_neuron_usage(router, model_version, prefix=inner_prefix))

    # Tau gradient (only if available)
    if tau_grad_info is not None:
        lines.append(f"{inner_prefix}Tau grad: weight={tau_grad_info['weight']:.6f} bias={tau_grad_info['bias']:.6f}")

    return lines


def is_modern_dawn_model(model):
    """Check if model is DAWN v16.0+"""
    base_model = get_underlying_model(model)

    # Check for v16+ structure
    if hasattr(base_model, '__version__') and base_model.__version__ in ["16.0", "16.1"]:
        return True

    # Structure check: v16+ has layers with .attn and .memory
    if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
        if hasattr(base_model.layers[0], 'attn') and hasattr(base_model.layers[0], 'memory'):
            return True

    return False


def is_v16_model(model):
    """Check if model is DAWN v16.x by version string"""
    base_model = get_underlying_model(model)
    version = getattr(base_model, '__version__', '')
    return version.startswith('16.')


def is_v17_model(model):
    """Check if model is DAWN v17.x by version string"""
    base_model = get_underlying_model(model)
    version = getattr(base_model, '__version__', '')
    return version.startswith('17')


def is_v18_model(model):
    """Check if model is DAWN v18.x by version string"""
    base_model = get_underlying_model(model)
    version = getattr(base_model, '__version__', '')
    return version.startswith('18')


def is_tpu_model(model):
    """Check if model is TPU-optimized version (skip routing_info during training)"""
    base_model = get_underlying_model(model)
    version = getattr(base_model, '__version__', '')
    return 'TPU' in version.upper()


def needs_routing_info(model):
    """Check if model needs routing_info for usage logging.
    TPU models skip routing_info collection during training for performance.
    """
    if is_tpu_model(model):
        return False
    return is_v16_model(model) or is_v17_model(model) or is_v18_model(model)


def _gini(x):
    """Gini coefficient (0=equal, 1=one neuron dominates)"""
    x_sorted = torch.sort(x)[0]
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()


def _entropy(x):
    """Normalized entropy (0=concentrated, 100=uniform distribution)"""
    # Normalize to probability distribution
    p = x / (x.sum() + 1e-8)
    # Entropy: -sum(p * log(p)), avoiding log(0)
    log_p = torch.log(p + 1e-8)
    entropy = -(p * log_p).sum()
    # Normalize by max entropy (log(N))
    max_entropy = math.log(x.numel())
    return (entropy / max_entropy * 100).item() if max_entropy > 0 else 0.0


def _get_router_log_lines(router, global_step, total_steps, global_routers=None):
    """Generate router log lines for both console and file output.

    Returns list of log lines (without trailing newlines).
    """
    lines = []

    # v18.0: Adaptive Threshold Multi-Path Routing
    # Detected by: global_routers has max_paths attribute (v18 specific)
    # v18 uses soft weights (sigmoid 0~1), EMA tracks average weight intensity
    if global_routers is not None and hasattr(global_routers, 'max_paths'):
        ema_fq = router.usage_ema_feature_q
        ema_fk = router.usage_ema_feature_k
        ema_fv = router.usage_ema_feature_v
        ema_rq = router.usage_ema_restore_q
        ema_rk = router.usage_ema_restore_k
        ema_rv = router.usage_ema_restore_v
        ema_FK = router.usage_ema_feature_know
        ema_RK = router.usage_ema_restore_know

        # For v18 soft weights: "active" = average weight > 0.01 (not fully suppressed)
        active_fq = (ema_fq > 0.01).sum().item()
        active_fk = (ema_fk > 0.01).sum().item()
        active_fv = (ema_fv > 0.01).sum().item()
        active_rq = (ema_rq > 0.01).sum().item()
        active_rk = (ema_rk > 0.01).sum().item()
        active_rv = (ema_rv > 0.01).sum().item()
        active_FK = (ema_FK > 0.01).sum().item()
        active_RK = (ema_RK > 0.01).sum().item()
        n_fq, n_fk, n_fv = ema_fq.numel(), ema_fk.numel(), ema_fv.numel()
        n_rq, n_rk, n_rv = ema_rq.numel(), ema_rk.numel(), ema_rv.numel()
        n_FK, n_RK = ema_FK.numel(), ema_RK.numel()

        # Gini coefficients
        gini_fq, gini_fk, gini_fv = _gini(ema_fq), _gini(ema_fk), _gini(ema_fv)
        gini_rq, gini_rk, gini_rv = _gini(ema_rq), _gini(ema_rk), _gini(ema_rv)
        gini_FK, gini_RK = _gini(ema_FK), _gini(ema_RK)

        # Dead neuron ratios (average weight < 0.01)
        dead_fq = (ema_fq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fk = (ema_fk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fv = (ema_fv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rq = (ema_rq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rk = (ema_rk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rv = (ema_rv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_FK = (ema_FK < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_RK = (ema_RK < DEAD_NEURON_THRESHOLD).float().mean().item()

        # Entropy (normalized, 0=concentrated, 100=uniform)
        ent_fq, ent_fk, ent_fv = _entropy(ema_fq), _entropy(ema_fk), _entropy(ema_fv)
        ent_rq, ent_rk, ent_rv = _entropy(ema_rq), _entropy(ema_rk), _entropy(ema_rv)
        ent_FK, ent_RK = _entropy(ema_FK), _entropy(ema_RK)

        # v18.0 format (multi-path, masked softmax)
        lines.append(f"         [v18.0] Neuron Usage")
        lines.append(f"             Feature_Q: {int(active_fq)}/{n_fq} | Feature_K: {int(active_fk)}/{n_fk} | Feature_V: {int(active_fv)}/{n_fv}")
        lines.append(f"             Restore_Q: {int(active_rq)}/{n_rq} | Restore_K: {int(active_rk)}/{n_rk} | Restore_V: {int(active_rv)}/{n_rv}")
        lines.append(f"             Feature_Know: {int(active_FK)}/{n_FK} | Restore_Know: {int(active_RK)}/{n_RK}")
        lines.append(f"             Ent: FQ={ent_fq:.0f} FK={ent_fk:.0f} FV={ent_fv:.0f} RQ={ent_rq:.0f} RK={ent_rk:.0f} RV={ent_rv:.0f} FKnow={ent_FK:.0f} RKnow={ent_RK:.0f}")
        lines.append(f"             Dead: FQ={dead_fq:.1%} FK={dead_fk:.1%} FV={dead_fv:.1%} RQ={dead_rq:.1%} RK={dead_rk:.1%} RV={dead_rv:.1%} FKnow={dead_FK:.1%} RKnow={dead_RK:.1%}")
        lines.append(f"             Gini: FQ={gini_fq:.2f} FK={gini_fk:.2f} FV={gini_fv:.2f} RQ={gini_rq:.2f} RK={gini_rk:.2f} RV={gini_rv:.2f} FKnow={gini_FK:.2f} RKnow={gini_RK:.2f}")

    # v17.2: Feature QK Unified + Restore Q/K Separate
    # (has usage_ema_feature_qk AND usage_ema_restore_q AND usage_ema_feature_know)
    elif hasattr(router, 'usage_ema_feature_qk') and hasattr(router, 'usage_ema_restore_q') and hasattr(router, 'usage_ema_feature_know'):
        ema_fqk = router.usage_ema_feature_qk
        ema_fv = router.usage_ema_feature_v
        ema_rq = router.usage_ema_restore_q
        ema_rk = router.usage_ema_restore_k
        ema_rv = router.usage_ema_restore_v
        ema_FK = router.usage_ema_feature_know
        ema_RK = router.usage_ema_restore_know

        # Active neuron counts
        active_fqk = (ema_fqk > 0.01).sum().item()
        active_fv = (ema_fv > 0.01).sum().item()
        active_rq = (ema_rq > 0.01).sum().item()
        active_rk = (ema_rk > 0.01).sum().item()
        active_rv = (ema_rv > 0.01).sum().item()
        active_FK = (ema_FK > 0.01).sum().item()
        active_RK = (ema_RK > 0.01).sum().item()
        n_fqk, n_fv = ema_fqk.numel(), ema_fv.numel()
        n_rq, n_rk, n_rv = ema_rq.numel(), ema_rk.numel(), ema_rv.numel()
        n_FK, n_RK = ema_FK.numel(), ema_RK.numel()

        # Gini coefficients
        gini_fqk, gini_fv = _gini(ema_fqk), _gini(ema_fv)
        gini_rq, gini_rk, gini_rv = _gini(ema_rq), _gini(ema_rk), _gini(ema_rv)
        gini_FK, gini_RK = _gini(ema_FK), _gini(ema_RK)

        # Dead neuron ratios
        dead_fqk = (ema_fqk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fv = (ema_fv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rq = (ema_rq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rk = (ema_rk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rv = (ema_rv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_FK = (ema_FK < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_RK = (ema_RK < DEAD_NEURON_THRESHOLD).float().mean().item()

        # v17.2 format
        lines.append(f"         [v17.2] Neuron Usage")
        lines.append(f"             Feature_QK: {int(active_fqk)}/{n_fqk} | Feature_V: {int(active_fv)}/{n_fv}")
        lines.append(f"             Restore_Q: {int(active_rq)}/{n_rq} | Restore_K: {int(active_rk)}/{n_rk} | Restore_V: {int(active_rv)}/{n_rv}")
        lines.append(f"             Feature_Know: {int(active_FK)}/{n_FK} | Restore_Know: {int(active_RK)}/{n_RK}")
        lines.append(f"             Dead: FQK={dead_fqk:.1%} FV={dead_fv:.1%} RQ={dead_rq:.1%} RK={dead_rk:.1%} RV={dead_rv:.1%} FK={dead_FK:.1%} RK={dead_RK:.1%}")
        lines.append(f"             Gini: FQK={gini_fqk:.2f} FV={gini_fv:.2f} RQ={gini_rq:.2f} RK={gini_rk:.2f} RV={gini_rv:.2f} FK={gini_FK:.2f} RK={gini_RK:.2f}")

    # v17.1: Q/K Separate (has usage_ema_feature_q, not usage_ema_feature_qk)
    elif hasattr(router, 'usage_ema_feature_q') and hasattr(router, 'usage_ema_feature_know'):
        ema_fq = router.usage_ema_feature_q
        ema_fk = router.usage_ema_feature_k
        ema_fv = router.usage_ema_feature_v
        ema_rq = router.usage_ema_restore_q
        ema_rk = router.usage_ema_restore_k
        ema_rv = router.usage_ema_restore_v
        ema_FK = router.usage_ema_feature_know
        ema_RK = router.usage_ema_restore_know

        # Active neuron counts
        active_fq = (ema_fq > 0.01).sum().item()
        active_fk = (ema_fk > 0.01).sum().item()
        active_fv = (ema_fv > 0.01).sum().item()
        active_rq = (ema_rq > 0.01).sum().item()
        active_rk = (ema_rk > 0.01).sum().item()
        active_rv = (ema_rv > 0.01).sum().item()
        active_FK = (ema_FK > 0.01).sum().item()
        active_RK = (ema_RK > 0.01).sum().item()
        n_fq, n_fk, n_fv = ema_fq.numel(), ema_fk.numel(), ema_fv.numel()
        n_rq, n_rk, n_rv = ema_rq.numel(), ema_rk.numel(), ema_rv.numel()
        n_FK, n_RK = ema_FK.numel(), ema_RK.numel()

        # Gini coefficients
        gini_fq, gini_fk, gini_fv = _gini(ema_fq), _gini(ema_fk), _gini(ema_fv)
        gini_rq, gini_rk, gini_rv = _gini(ema_rq), _gini(ema_rk), _gini(ema_rv)
        gini_FK, gini_RK = _gini(ema_FK), _gini(ema_RK)

        # Dead neuron ratios
        dead_fq = (ema_fq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fk = (ema_fk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fv = (ema_fv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rq = (ema_rq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rk = (ema_rk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rv = (ema_rv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_FK = (ema_FK < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_RK = (ema_RK < DEAD_NEURON_THRESHOLD).float().mean().item()

        # v17.1 format (Q/K separate)
        lines.append(f"         [v17.1] Neuron Usage")
        lines.append(f"             Feature_Q: {int(active_fq)}/{n_fq} | Feature_K: {int(active_fk)}/{n_fk} | Feature_V: {int(active_fv)}/{n_fv}")
        lines.append(f"             Restore_Q: {int(active_rq)}/{n_rq} | Restore_K: {int(active_rk)}/{n_rk} | Restore_V: {int(active_rv)}/{n_rv}")
        lines.append(f"             Feature_Know: {int(active_FK)}/{n_FK} | Restore_Know: {int(active_RK)}/{n_RK}")
        lines.append(f"             Dead: FQ={dead_fq:.1%} FK={dead_fk:.1%} FV={dead_fv:.1%} RQ={dead_rq:.1%} RK={dead_rk:.1%} RV={dead_rv:.1%} FKnow={dead_FK:.1%} RKnow={dead_RK:.1%}")
        lines.append(f"             Gini: FQ={gini_fq:.2f} FK={gini_fk:.2f} FV={gini_fv:.2f} RQ={gini_rq:.2f} RK={gini_rk:.2f} RV={gini_rv:.2f} FKnow={gini_FK:.2f} RKnow={gini_RK:.2f}")

    # v17 fallback: Q/K/V separate without knowledge tracking
    elif hasattr(router, 'usage_ema_feature_q'):
        ema_fq = router.usage_ema_feature_q
        ema_fk = router.usage_ema_feature_k
        ema_fv = router.usage_ema_feature_v
        ema_rq = router.usage_ema_restore_q
        ema_rk = router.usage_ema_restore_k
        ema_rv = router.usage_ema_restore_v

        # Active neuron counts
        active_fq = (ema_fq > 0.01).sum().item()
        active_fk = (ema_fk > 0.01).sum().item()
        active_fv = (ema_fv > 0.01).sum().item()
        active_rq = (ema_rq > 0.01).sum().item()
        active_rk = (ema_rk > 0.01).sum().item()
        active_rv = (ema_rv > 0.01).sum().item()
        n_fq, n_fk, n_fv = ema_fq.numel(), ema_fk.numel(), ema_fv.numel()
        n_rq, n_rk, n_rv = ema_rq.numel(), ema_rk.numel(), ema_rv.numel()

        # Gini coefficients
        gini_fq, gini_fk, gini_fv = _gini(ema_fq), _gini(ema_fk), _gini(ema_fv)
        gini_rq, gini_rk, gini_rv = _gini(ema_rq), _gini(ema_rk), _gini(ema_rv)

        # Dead neuron ratios
        dead_fq = (ema_fq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fk = (ema_fk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fv = (ema_fv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rq = (ema_rq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rk = (ema_rk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rv = (ema_rv < DEAD_NEURON_THRESHOLD).float().mean().item()

        lines.append(f"         Neuron Usage | Active FQ/FK/FV:{int(active_fq)}/{n_fq},{int(active_fk)}/{n_fk},{int(active_fv)}/{n_fv}")
        lines.append(f"             Active RQ/RK/RV:{int(active_rq)}/{n_rq},{int(active_rk)}/{n_rk},{int(active_rv)}/{n_rv}")
        lines.append(f"             Gini FQ/FK/FV: {gini_fq:.2f}/{gini_fk:.2f}/{gini_fv:.2f} | RQ/RK/RV: {gini_rq:.2f}/{gini_rk:.2f}/{gini_rv:.2f}")
        lines.append(f"             Dead FQ/FK/FV: {dead_fq:.1%}/{dead_fk:.1%}/{dead_fv:.1%} | RQ/RK/RV: {dead_rq:.1%}/{dead_rk:.1%}/{dead_rv:.1%}")

        # Knowledge neurons (Feature/Restore separated in v17)
        if hasattr(router, 'usage_ema_feature_know'):
            ema_FK = router.usage_ema_feature_know
            ema_RK = router.usage_ema_restore_know
            active_FK = (ema_FK > 0.01).sum().item()
            active_RK = (ema_RK > 0.01).sum().item()
            n_FK, n_RK = ema_FK.numel(), ema_RK.numel()
            gini_FK, gini_RK = _gini(ema_FK), _gini(ema_RK)
            dead_FK = (ema_FK < DEAD_NEURON_THRESHOLD).float().mean().item()
            dead_RK = (ema_RK < DEAD_NEURON_THRESHOLD).float().mean().item()
            lines.append(f"             Knowledge F/R: Active {int(active_FK)}/{n_FK},{int(active_RK)}/{n_RK} | Dead:{dead_FK:.1%}/{dead_RK:.1%} | Gini:{gini_FK:.2f}/{gini_RK:.2f}")

    # v16.3: Complete Q/K/V Pool Separation
    elif hasattr(router, 'usage_ema_fq'):
        ema_fq = router.usage_ema_fq
        ema_fk = router.usage_ema_fk
        ema_fv = router.usage_ema_fv
        ema_rq = router.usage_ema_rq
        ema_rk = router.usage_ema_rk
        ema_rv = router.usage_ema_rv

        # Active neuron counts
        active_fq = (ema_fq > 0.01).sum().item()
        active_fk = (ema_fk > 0.01).sum().item()
        active_fv = (ema_fv > 0.01).sum().item()
        active_rq = (ema_rq > 0.01).sum().item()
        active_rk = (ema_rk > 0.01).sum().item()
        active_rv = (ema_rv > 0.01).sum().item()
        n_fq, n_fk, n_fv = ema_fq.numel(), ema_fk.numel(), ema_fv.numel()
        n_rq, n_rk, n_rv = ema_rq.numel(), ema_rk.numel(), ema_rv.numel()

        # Gini coefficients
        gini_fq, gini_fk, gini_fv = _gini(ema_fq), _gini(ema_fk), _gini(ema_fv)
        gini_rq, gini_rk, gini_rv = _gini(ema_rq), _gini(ema_rk), _gini(ema_rv)

        # Dead neuron ratios
        dead_fq = (ema_fq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fk = (ema_fk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_fv = (ema_fv < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rq = (ema_rq < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rk = (ema_rk < DEAD_NEURON_THRESHOLD).float().mean().item()
        dead_rv = (ema_rv < DEAD_NEURON_THRESHOLD).float().mean().item()

        lines.append(f"         Neuron Usage | Active FQ/FK/FV:{int(active_fq)}/{n_fq},{int(active_fk)}/{n_fk},{int(active_fv)}/{n_fv}")
        lines.append(f"             Active RQ/RK/RV:{int(active_rq)}/{n_rq},{int(active_rk)}/{n_rk},{int(active_rv)}/{n_rv}")
        lines.append(f"             Gini FQ/FK/FV: {gini_fq:.2f}/{gini_fk:.2f}/{gini_fv:.2f} | RQ/RK/RV: {gini_rq:.2f}/{gini_rk:.2f}/{gini_rv:.2f}")
        lines.append(f"             Dead FQ/FK/FV: {dead_fq:.1%}/{dead_fk:.1%}/{dead_fv:.1%} | RQ/RK/RV: {dead_rq:.1%}/{dead_rk:.1%}/{dead_rv:.1%}")

        # Knowledge neurons (if available)
        if hasattr(router, 'usage_ema_knowledge'):
            ema_K = router.usage_ema_knowledge
            active_K = (ema_K > 0.01).sum().item()
            n_K = ema_K.numel()
            gini_K = _gini(ema_K)
            dead_K = (ema_K < DEAD_NEURON_THRESHOLD).float().mean().item()
            lines.append(f"             Knowledge: Active {int(active_K)}/{n_K} | Dead:{dead_K:.1%} | Gini:{gini_K:.2f}")

    return lines


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None, log_file=None,
                orthogonality_weight=0.0, diversity_weight=0.0, load_balance_weight=0.0, entropy_weight=0.0, tau_reg_weight=0.0,
                debug_logger=None, ckpt_manager=None, model_config=None, start_step=0, global_step=0, total_steps=1,
                total_epoch_steps=None, val_loader=None, val_interval=5000, is_tpu=False):
    """Train for one epoch

    Args:
        start_step: Step offset for this epoch (for logging/checkpointing when resuming)
        total_epoch_steps: Total steps in full epoch (for progress bar when using subset)
        global_step: Global training step counter
        total_steps: Total training steps
        is_tpu: Whether running on TPU/XLA device
    """
    model.train()

    # Debug: Log at start of key epochs
    debug_log_steps = {0, 100, 500}  # Steps to log gradient info

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0
    num_batches = 0

    # Window accumulators for logging every LOG_INTERVAL steps
    log_interval = LOG_INTERVAL
    window_loss = 0.0
    window_acc_correct = 0
    window_acc_valid = 0
    window_count = 0

    # Last neuron metrics (for epoch summary)
    last_neuron_metrics = None

    # Note: start_step is used to calculate correct global step, but we don't skip
    # batches here - the caller provides a subset dataloader when resuming

    # Use total_epoch_steps for progress bar if provided (shows full epoch progress when resuming)
    pbar_total = total_epoch_steps if total_epoch_steps else len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", initial=start_step, total=pbar_total)
    # Gradient accumulation
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    for local_step, batch in enumerate(pbar):
        # Calculate actual step (for logging and checkpointing)
        step = local_step + start_step

        if is_tpu:
            # MpDeviceLoader already moves data to TPU
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

        # CLM: labels = input_ids (model does shift internally)
        # Set padding positions to -100 so they're ignored in loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Zero grad only at the start of accumulation
        if local_step % accumulation_steps == 0:
            optimizer.zero_grad()

        # Enable v18 debug_mode on log steps (for routing stats)
        is_log_step = (step + 1) % log_interval == 0
        if is_log_step:
            set_v18_debug_mode(model, True)

        # TPU training (bf16 native, no GradScaler)
        if is_tpu:
            base_model = get_underlying_model(model)

            # DAWN model forward
            if load_balance_weight > 0 or entropy_weight > 0 or needs_routing_info(model):
                ce_loss, logits, routing_infos = model(input_ids, labels, attention_mask=attention_mask, return_routing_info=True)
            else:
                ce_loss, logits = model(input_ids, labels, attention_mask=attention_mask)
                routing_infos = None

            # Orthogonality loss
            orth_loss = 0.0
            if orthogonality_weight > 0 and hasattr(base_model, 'orthogonality_loss'):
                orth_loss = base_model.orthogonality_loss()

            # Knowledge diversity loss
            diversity_loss = 0.0
            if diversity_weight > 0 and hasattr(base_model, 'knowledge_diversity_loss'):
                diversity_loss = base_model.knowledge_diversity_loss()

            # Tau regularization loss
            tau_reg_loss = 0.0
            if tau_reg_weight > 0 and hasattr(base_model, 'tau_regularization_loss'):
                tau_reg_loss = base_model.tau_regularization_loss()

            # Load balance loss
            lb_loss = 0.0
            if load_balance_weight > 0:
                if hasattr(base_model, 'aux_loss') and base_model.aux_loss != 0.0:
                    lb_loss = base_model.aux_loss
                elif routing_infos is not None and hasattr(base_model, 'load_balance_loss'):
                    lb_loss = base_model.load_balance_loss(routing_infos)

            # Total loss
            loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + tau_reg_weight * tau_reg_loss + load_balance_weight * lb_loss

            # Scale loss for accumulation and backward
            (loss / accumulation_steps).backward()

            # Disable v18 debug_mode after backward
            if is_log_step:
                set_v18_debug_mode(model, False)

            # Only update optimizer on accumulation boundary
            if (local_step + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Debug: Log gradient flow at specific steps
                if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                    debug_logger.log_post_backward(model, epoch, step)

                xm.optimizer_step(optimizer)
            else:
                xm.mark_step()

        # Mixed precision training (CUDA)
        elif scaler is not None:
            with torch.amp.autocast('cuda', enabled=True):
                # Get underlying model for attribute checks (handles torch.compile)
                base_model = get_underlying_model(model)

                # DAWN model forward
                if load_balance_weight > 0 or entropy_weight > 0 or needs_routing_info(model):
                    ce_loss, logits, routing_infos = model(input_ids, labels, attention_mask=attention_mask, return_routing_info=True)
                else:
                    ce_loss, logits = model(input_ids, labels, attention_mask=attention_mask)
                    routing_infos = None

                # Orthogonality loss
                orth_loss = 0.0
                if orthogonality_weight > 0 and hasattr(base_model, 'orthogonality_loss'):
                    orth_loss = base_model.orthogonality_loss()

                # Knowledge diversity loss
                diversity_loss = 0.0
                if diversity_weight > 0 and hasattr(base_model, 'knowledge_diversity_loss'):
                    diversity_loss = base_model.knowledge_diversity_loss()

                # Tau regularization loss (v18.0)
                tau_reg_loss = 0.0
                if tau_reg_weight > 0 and hasattr(base_model, 'tau_regularization_loss'):
                    tau_reg_loss = base_model.tau_regularization_loss()

                # Load balance loss (from model.aux_loss)
                lb_loss = 0.0
                if load_balance_weight > 0:
                    if hasattr(base_model, 'aux_loss') and base_model.aux_loss != 0.0:
                        lb_loss = base_model.aux_loss
                    elif routing_infos is not None and hasattr(base_model, 'load_balance_loss'):
                        lb_loss = base_model.load_balance_loss(routing_infos)

                # Total loss
                loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + tau_reg_weight * tau_reg_loss + load_balance_weight * lb_loss

                # NaN/INF detection
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARNING] NaN/INF detected at step {step}!")
                    print(f"  ce_loss: {ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss}")
                    print(f"  orth_loss: {orth_loss.item() if torch.is_tensor(orth_loss) else orth_loss}")
                    print(f"  diversity_loss: {diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss}")
                    print(f"  lb_loss: {lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss}")
                    raise ValueError(f"NaN/INF loss detected at epoch {epoch}, step {step}")

            # Scale loss for accumulation and backward
            scaler.scale(loss / accumulation_steps).backward()

            # Disable v18 debug_mode after backward (must be after for gradient checkpointing)
            if is_log_step:
                set_v18_debug_mode(model, False)

            # Only update optimizer on accumulation boundary
            if (local_step + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Debug: Log gradient flow at specific steps
                if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                    debug_logger.log_post_backward(model, epoch, step)

                scaler.step(optimizer)
                scaler.update()
        else:
            # Non-AMP training (CPU or no CUDA)
            base_model = get_underlying_model(model)

            # DAWN model forward
            if load_balance_weight > 0 or entropy_weight > 0 or needs_routing_info(model):
                ce_loss, logits, routing_infos = model(input_ids, labels, attention_mask=attention_mask, return_routing_info=True)
            else:
                ce_loss, logits = model(input_ids, labels, attention_mask=attention_mask)
                routing_infos = None

            # Orthogonality loss
            orth_loss = 0.0
            if orthogonality_weight > 0 and hasattr(base_model, 'orthogonality_loss'):
                orth_loss = base_model.orthogonality_loss()

            # Knowledge diversity loss
            diversity_loss = 0.0
            if diversity_weight > 0 and hasattr(base_model, 'knowledge_diversity_loss'):
                diversity_loss = base_model.knowledge_diversity_loss()

            # Tau regularization loss (v18.0)
            tau_reg_loss = 0.0
            if tau_reg_weight > 0 and hasattr(base_model, 'tau_regularization_loss'):
                tau_reg_loss = base_model.tau_regularization_loss()

            # Load balance loss (from model.aux_loss)
            lb_loss = 0.0
            if load_balance_weight > 0:
                if hasattr(base_model, 'aux_loss') and base_model.aux_loss != 0.0:
                    lb_loss = base_model.aux_loss
                elif routing_infos is not None and hasattr(base_model, 'load_balance_loss'):
                    lb_loss = base_model.load_balance_loss(routing_infos)

            # Total loss
            loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + tau_reg_weight * tau_reg_loss + load_balance_weight * lb_loss

            # NaN/INF detection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/INF detected at step {step}!")
                print(f"  ce_loss: {ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss}")
                print(f"  orth_loss: {orth_loss.item() if torch.is_tensor(orth_loss) else orth_loss}")
                print(f"  diversity_loss: {diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss}")
                print(f"  tau_reg_loss: {tau_reg_loss.item() if torch.is_tensor(tau_reg_loss) else tau_reg_loss}")
                print(f"  lb_loss: {lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss}")
                raise ValueError(f"NaN/INF loss detected at epoch {epoch}, step {step}")

            # Scale loss for accumulation and backward
            (loss / accumulation_steps).backward()

            # Disable v18 debug_mode after backward (must be after for gradient checkpointing)
            if is_log_step:
                set_v18_debug_mode(model, False)

            # Only update optimizer on accumulation boundary
            if (local_step + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Debug: Log gradient flow at specific steps
                if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                    debug_logger.log_post_backward(model, epoch, step)

                optimizer.step()

        # Scheduler and global step increment only on accumulation boundary
        if (local_step + 1) % accumulation_steps == 0:
            if scheduler is not None:
                scheduler.step()

            # Increment global step counter (optimizer step basis)
            global_step += 1

        # Accuracy calculation (CLM: shifted)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        valid_mask = (shift_labels != -100)
        correct_predictions = (predictions == shift_labels) & valid_mask

        # TPU: minimize .item() calls (each triggers host sync)
        # Only sync on log steps to avoid TPU pipeline stalls
        if is_tpu:
            correct = correct_predictions.sum()
            valid_tokens = valid_mask.sum()
            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len

            # Accumulate as tensors (no .item() sync)
            total_correct += correct.item() if is_log_step else 0
            total_valid_tokens += valid_tokens.item() if is_log_step else 0
            total_loss += ce_loss.item() * num_tokens if is_log_step else 0
            total_tokens += num_tokens

            num_batches += 1

            if is_log_step:
                step_acc = correct.item() / (valid_tokens.item() + 1e-8)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{step_acc:.4f}"
                })
                window_loss += loss.item()
                window_acc_correct += correct.item()
                window_acc_valid += valid_tokens.item()
                window_count += 1
            else:
                pbar.update(1)
        else:
            correct = correct_predictions.sum().item()
            valid_tokens = valid_mask.sum().item()

            total_correct += correct
            total_valid_tokens += valid_tokens

            # Track total loss (ce_loss only, for fair comparison with validation)
            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += ce_loss.item() * num_tokens
            total_tokens += num_tokens

            num_batches += 1
            step_acc = correct / valid_tokens if valid_tokens > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{step_acc:.4f}"
            })

            # Accumulate for window logging
            window_loss += loss.item()
            window_acc_correct += correct
            window_acc_valid += valid_tokens
            window_count += 1

        # Real-time entropy monitoring (every log_interval steps)
        if (step + 1) % log_interval == 0 and routing_infos is not None:
            with torch.no_grad():
                try:
                    # Helper functions for entropy/variance calculation
                    def calc_entropy_ratio(pref):
                        if pref is None:
                            return 0.0
                        ent = -(pref * (pref + 1e-8).log()).sum(-1).mean()
                        return (ent / math.log(pref.shape[-1]) * 100).item()

                    def calc_token_var(pref):
                        if pref is None:
                            return 0.0
                        return pref.var(dim=1).mean().item()

                    # Use version_registry for routing log info (auto-detects version)
                    # Pass all layers for average entropy calculation
                    log_info = get_routing_log_info(routing_infos, calc_entropy_ratio, calc_token_var)
                    ent_str = log_info['ent_str']
                    var_str = log_info['var_str']
                    overlap_str = log_info['overlap_str']
                    paths_str = log_info.get('paths_str')  # v18 only
                    tau_str = log_info.get('tau_str')  # v18 only

                    # Attention ratio (attn_out_norm vs mem/know_out_norm per layer)
                    attn_ratios = []
                    for layer_info in routing_infos:
                        attn_norm = layer_info.get('attn_out_norm')
                        # v16: mem_out_norm, v17: know_out_norm
                        mem_norm = layer_info.get('mem_out_norm') or layer_info.get('know_out_norm')
                        if attn_norm is not None and mem_norm is not None:
                            # attn_norm/mem_norm are already Python floats from .item() in model
                            ratio = attn_norm / (attn_norm + mem_norm + 1e-8) * 100
                            attn_ratios.append(f"{ratio:.0f}")
                        else:
                            attn_ratios.append("-")
                    attn_str = "/".join(attn_ratios)

                    # Compute window average for display
                    avg_loss = window_loss / window_count if window_count > 0 else 0.0
                    avg_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

                    # Compact output with loss/acc
                    model_version = getattr(base_model, '__version__', '')
                    if model_version.startswith('18'):
                        # Get tau gradient info if available (v18.1, v18.2, v18.3, v18.4, v18.5)
                        tau_grad_info = None
                        if hasattr(base_model, 'router'):
                            # v18.5 uses tau_proj_feature + tau_proj_restore_*, v18.1-18.4 use tau_proj
                            tau_proj = getattr(base_model.router, 'tau_proj_feature', None) or getattr(base_model.router, 'tau_proj', None)
                            if tau_proj is not None and tau_proj.weight.grad is not None:
                                tau_grad_info = {
                                    'weight': tau_proj.weight.grad.norm().item(),
                                    'bias': tau_proj.bias.grad.norm().item() if tau_proj.bias.grad is not None else 0
                                }
                                # v18.5: also log restore tau gradients (Q/K separated)
                                restore_tau_projs = [
                                    ('rQ', getattr(base_model.router, 'tau_proj_restore_Q', None)),
                                    ('rK', getattr(base_model.router, 'tau_proj_restore_K', None)),
                                    ('rv', getattr(base_model.router, 'tau_proj_restore_v', None)),
                                    ('rknow', getattr(base_model.router, 'tau_proj_restore_know', None)),
                                ]
                                for name, proj in restore_tau_projs:
                                    if proj is not None and proj.weight.grad is not None:
                                        tau_grad_info[f'weight_{name}'] = proj.weight.grad.norm().item()
                                        tau_grad_info[f'bias_{name}'] = proj.bias.grad.norm().item() if proj.bias.grad is not None else 0

                        # Use unified formatter for v18 logging
                        router = base_model.router.neuron_router
                        log_lines = list(format_v18_full_log(
                            routing_infos, router, model_version, step, avg_loss, avg_acc,
                            ent_str, overlap_str, attn_str, tau_grad_info=tau_grad_info
                        ))

                        # Console output
                        for line in log_lines:
                            print(line)

                        # File output (same format)
                        if log_file:
                            with open(log_file, 'a') as f:
                                for line in log_lines:
                                    f.write(line + "\n")

                    else:
                        # Non-v18 models: collect log lines for console + file
                        non_v18_lines = []
                        if overlap_str:
                            non_v18_lines.append(f"[{step+1}] Loss:{avg_loss:.4f} Acc:{avg_acc:.4f} | {ent_str} | {overlap_str} | Attn:{attn_str}")
                        else:
                            non_v18_lines.append(f"[{step+1}] Loss:{avg_loss:.4f} Acc:{avg_acc:.4f} | {ent_str} | {var_str} | Attn:{attn_str}")

                        # Usage EMA logging (v17/v16)
                        router = None
                        global_routers = None
                        if hasattr(base_model, 'router') and hasattr(base_model.router, 'neuron_router'):  # v17/v17.1
                            router = base_model.router.neuron_router
                            global_routers = base_model.router
                        elif hasattr(base_model, 'global_routers') and hasattr(base_model.global_routers, 'neuron_router'):  # v16
                            router = base_model.global_routers.neuron_router
                        if router is not None:
                            non_v18_lines.extend(_get_router_log_lines(router, global_step, total_steps, global_routers))

                        # Console output
                        for line in non_v18_lines:
                            print(line)

                        # File output (same format)
                        if log_file:
                            with open(log_file, 'a') as f:
                                for line in non_v18_lines:
                                    f.write(line + "\n")

                    # Importance entropy logging (from routing preferences)
                    try:
                        imp_entropies = []
                        for layer_info in routing_infos:
                            attn_info = layer_info.get('attention', {})
                            # Try importance tensor first, fallback to relational_q_pref
                            importance = attn_info.get('importance')
                            if importance is None:
                                importance = attn_info.get('relational_q_pref')
                            if importance is not None and importance.numel() > 0:
                                # Normalize to probability distribution
                                p = torch.softmax(importance.float(), dim=-1)
                                # Entropy: -sum(p * log(p))
                                ent = -(p * (p + 1e-8).log()).sum(-1).mean()
                                imp_entropies.append(ent.item())
                        if imp_entropies:
                            avg_imp_entropy = sum(imp_entropies) / len(imp_entropies)
                            print(f"         Importance Entropy: {avg_imp_entropy:.4f} (uniform@100={math.log(100):.2f})")
                    except Exception:
                        pass

                    # Knowledge neuron usage stats
                    try:
                        # Collect knowledge indices from all layers
                        all_k_idx = []
                        all_coarse_idx = []
                        is_v17_style = False  # v17 uses feature_know/restore_know separation
                        for layer_info in routing_infos:
                            # v17: knowledge -> feature_know_w, restore_know_w (Feature-Restore separation)
                            know = layer_info.get('knowledge', {})
                            feature_know_w = know.get('feature_know_w')
                            restore_know_w = know.get('restore_know_w')
                            if feature_know_w is not None:
                                is_v17_style = True
                            topk_idx = know.get('topk_indices')
                            if topk_idx is not None:
                                all_k_idx.append(topk_idx.flatten())
                            # v16.x: memory -> fine_indices, coarse_indices (2-stage retrieval)
                            mem = layer_info.get('memory', {})
                            fine_idx = mem.get('fine_indices')
                            coarse_idx = mem.get('coarse_indices')
                            if fine_idx is not None:
                                all_k_idx.append(fine_idx.flatten())
                            if coarse_idx is not None:
                                all_coarse_idx.append(coarse_idx.flatten())
                            # Backward compat: knowledge_indices
                            k_idx = mem.get('knowledge_indices')
                            if k_idx is not None:
                                all_k_idx.append(k_idx.flatten())

                        # v17: Feature-Restore separation - usage stats in _get_router_log_lines()
                        if all_k_idx and not is_v17_style:
                            all_idx = torch.cat(all_k_idx)
                            n_knowledge = base_model.n_knowledge if hasattr(base_model, 'n_knowledge') else 80

                            # Count usage per knowledge neuron
                            usage_counts = torch.bincount(all_idx.long(), minlength=n_knowledge).float()
                            usage_freq = usage_counts / (usage_counts.sum() + 1e-8)

                            # Active: neurons used at least once
                            active_K = (usage_counts > 0).sum().item()

                            # Entropy (normalized)
                            ent_K = -(usage_freq * (usage_freq + 1e-8).log()).sum()
                            max_ent = math.log(n_knowledge)
                            ent_ratio_K = (ent_K / max_ent * 100).item()

                            # Gini
                            gini_K = _gini(usage_freq)

                            # v16.x: coarse→fine ratio (2-stage retrieval)
                            if all_coarse_idx:
                                coarse_all = torch.cat(all_coarse_idx)
                                coarse_unique = len(torch.unique(coarse_all))
                                fine_unique = len(torch.unique(all_idx))
                                # fine_k / coarse_k ratio (how many survive from coarse to fine)
                                coarse_k = base_model.coarse_k if hasattr(base_model, 'coarse_k') else 20
                                fine_k = base_model.fine_k if hasattr(base_model, 'fine_k') else 10
                                print(f"         Knowledge: Active {int(active_K)}/{n_knowledge} | Ent:{ent_ratio_K:.0f}% | Gini:{gini_K:.2f} | coarse→fine: {coarse_k}→{fine_k} (unique: {coarse_unique}→{fine_unique})")
                            else:
                                print(f"         Knowledge: Active {int(active_K)}/{n_knowledge} | Ent:{ent_ratio_K:.0f}% | Gini:{gini_K:.2f}")
                    except Exception:
                        pass

                except Exception as e:
                    print(f"[v18 logging error] {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

        # Collect neuron metrics every log_interval steps
        if log_file and (step + 1) % log_interval == 0:
            # Neuron metrics collection
            model.eval()
            with torch.no_grad():
                try:
                    _, neuron_indices = model(input_ids, return_routing_info=True)
                    neuron_metrics = compute_training_metrics(model, neuron_indices, device)
                    last_neuron_metrics = neuron_metrics

                    # Log neuron metrics to file
                    with open(log_file, 'a') as f:
                        f.write(f"METRICS,{epoch},{step+1},"
                               f"avg_usage={neuron_metrics['avg_usage']:.4f},"
                               f"avg_gini={neuron_metrics['avg_gini']:.4f},"
                               f"avg_entropy={neuron_metrics['avg_entropy']:.4f},"
                               f"avg_top10={neuron_metrics['avg_top10']:.4f},"
                               f"avg_top50={neuron_metrics['avg_top50']:.4f}")
                        # Add per-layer details
                        for i in range(len(neuron_indices)):
                            f.write(f",L{i}_usage={neuron_metrics[f'L{i}_usage']:.4f},"
                                   f"L{i}_gini={neuron_metrics[f'L{i}_gini']:.4f},"
                                   f"L{i}_entropy={neuron_metrics[f'L{i}_entropy']:.4f},"
                                   f"L{i}_top10={neuron_metrics[f'L{i}_top10']:.4f},"
                                   f"L{i}_top50={neuron_metrics[f'L{i}_top50']:.4f}")
                        f.write("\n")
                except Exception as e:
                    # If metrics collection fails, continue training
                    pass
            model.train()

            # Reset window
            window_loss = 0.0
            window_acc_correct = 0
            window_acc_valid = 0
            window_count = 0

        # Save checkpoint every 1000 steps
        if ckpt_manager is not None and (step + 1) % 1000 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
            step_metrics = {
                'train_loss': avg_loss,
                'train_acc': avg_acc,
                'step': step + 1,
            }
            ckpt_manager.save_checkpoint(
                model, optimizer, epoch, avg_loss, step_metrics, is_best=False,
                scheduler=scheduler, scaler=scaler, model_config=model_config,
                filename=f'checkpoint_epoch{epoch}_step{step+1}.pt',
                epoch_completed=False  # Mid-epoch checkpoint
            )
            if not is_tpu:
                torch.cuda.empty_cache()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{step_acc:.4f}",
                "ckpt": f"step{step+1}"
            })

        # Mid-epoch validation every val_interval steps
        if val_loader is not None and (step + 1) % val_interval == 0:
            val_dl = pl.MpDeviceLoader(val_loader, device) if is_tpu else val_loader
            val_loss, val_acc = evaluate(model, val_dl, device, args, tokenizer, is_tpu=is_tpu)
            print(f"\n[Step {step+1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"epoch={epoch},step={step+1},val_loss={val_loss:.6f},val_acc={val_acc:.6f}\n")
            model.train()  # Back to training mode
            # Clear CUDA cache after validation
            if not is_tpu and (device.type == 'cuda' if hasattr(device, 'type') else 'cuda' in str(device)):
                torch.cuda.empty_cache()

    # Log remaining steps at end of epoch
    if log_file and window_count > 0:
        avg_window_loss = window_loss / window_count
        avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

        with open(log_file, 'a') as f:
            f.write(f"epoch={epoch},step={num_batches},loss={avg_window_loss:.6f},"
                   f"acc={avg_window_acc:.6f}\n")

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

    return avg_loss, avg_acc, last_neuron_metrics, global_step


def evaluate(model, dataloader, device, args, tokenizer=None, max_batches=200, is_tpu=False):
    """Evaluate model with MLM masking

    Args:
        max_batches: Maximum number of batches to evaluate (default 200 for faster eval)
        is_tpu: Whether running on TPU/XLA device
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0

    # Clear CUDA cache before evaluation (helps with torch.compile memory)
    if not is_tpu and (device.type == 'cuda' if hasattr(device, 'type') else 'cuda' in str(device)):
        torch.cuda.empty_cache()

    # Use original model if torch.compiled (avoids CUDA graph memory issues)
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False, position=0, dynamic_ncols=True, total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
            if is_tpu:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

            # CLM evaluation
            logits = eval_model(input_ids, attention_mask=attention_mask)

            # Create labels with padding masked
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Shift for autoregressive loss
            B, S, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()  # Ensure Long type for cross_entropy

            # Count valid (non-padding) tokens for proper averaging
            valid_mask = (shift_labels != -100)
            valid_tokens = valid_mask.sum().item()

            loss = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Accuracy calculation
            predictions = shift_logits.argmax(dim=-1)
            correct_predictions = (predictions == shift_labels) & valid_mask

            correct = correct_predictions.sum().item()

            total_correct += correct
            total_valid_tokens += valid_tokens

            # Use valid tokens for loss averaging
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return avg_loss, avg_acc


def compute_training_metrics(model, neuron_indices, device):
    """Compute detailed neuron usage metrics during training

    Args:
        model: DAWN model
        neuron_indices: List of [B, S, k] tensors (one per layer)
        device: torch device

    Returns:
        metrics: Dict with per-layer and aggregate metrics
    """
    metrics = {}

    # Get n_neurons from model
    if hasattr(model, '_orig_mod'):
        n_neurons = model._orig_mod.n_neurons
    else:
        n_neurons = model.n_neurons

    layer_usage = []
    layer_gini = []
    layer_top10 = []
    layer_top50 = []
    layer_entropy = []

    for layer_idx, selected_idx in enumerate(neuron_indices):
        # selected_idx: [B, S, k]
        flat_idx = selected_idx.reshape(-1)
        total_selections = flat_idx.numel()

        # 1. Usage rate (unique neurons used / total neurons)
        unique_neurons = torch.unique(flat_idx).numel()
        usage_rate = unique_neurons / n_neurons
        layer_usage.append(usage_rate)

        # 2. Selection frequency distribution
        counts = torch.bincount(flat_idx, minlength=n_neurons).float()
        freq = counts / (counts.sum() + 1e-10)

        # 3. Gini coefficient (0 = perfect equality, 1 = maximum inequality)
        sorted_counts = torch.sort(counts)[0]
        n = len(sorted_counts)
        index = torch.arange(1, n + 1, device=device, dtype=torch.float32)
        gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * sorted_counts.sum() + 1e-10)
        layer_gini.append(gini.item())

        # 4. Top-K concentration
        if n_neurons >= 10:
            top10 = torch.topk(counts, 10).values.sum() / (counts.sum() + 1e-10)
            layer_top10.append(top10.item())
        else:
            layer_top10.append(1.0)

        if n_neurons >= 50:
            top50 = torch.topk(counts, 50).values.sum() / (counts.sum() + 1e-10)
            layer_top50.append(top50.item())
        else:
            layer_top50.append(1.0)

        # 5. Entropy (higher = more uniform distribution)
        # Normalize to [0, 1] by dividing by log(n_neurons)
        entropy = -(freq * torch.log(freq + 1e-10)).sum()
        normalized_entropy = entropy / (torch.log(torch.tensor(n_neurons, dtype=torch.float32)) + 1e-10)
        layer_entropy.append(normalized_entropy.item())

        # Per-layer metrics
        metrics[f'L{layer_idx}_usage'] = usage_rate
        metrics[f'L{layer_idx}_gini'] = gini.item()
        metrics[f'L{layer_idx}_top10'] = layer_top10[-1]
        metrics[f'L{layer_idx}_top50'] = layer_top50[-1]
        metrics[f'L{layer_idx}_entropy'] = normalized_entropy.item()

    # Aggregate metrics
    metrics['avg_usage'] = sum(layer_usage) / len(layer_usage)
    metrics['avg_gini'] = sum(layer_gini) / len(layer_gini)
    metrics['avg_top10'] = sum(layer_top10) / len(layer_top10)
    metrics['avg_top50'] = sum(layer_top50) / len(layer_top50)
    metrics['avg_entropy'] = sum(layer_entropy) / len(layer_entropy)

    return metrics


def load_config(config_path):
    """Load config from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DAWN (Dynamic Architecture With Neurons)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint folder to resume from (e.g., checkpoints/run_20240101_120000_1234)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (disable auto-resume)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging to debug.txt (basis_up analysis, gradients, etc.)')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for faster training (may cause issues with variable seq lengths)')
    # Training parameter overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    # Ablation options
    parser.add_argument('--skip-householder', action='store_true',
                        help='Ablation: skip Householder transforms (v8 only)')
    parser.add_argument('--gelu-only', action='store_true',
                        help='Ablation: add GELU after compress (v8 only)')
    cli_args = parser.parse_args()

    # Load config
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # Set random seed if specified
    seed = cfg.get('seed', None)
    if seed is not None:
        set_seed(seed)
        print(f"🎲 Random seed set to {seed}")

    # Create args namespace from config
    class Args:
        pass
    args = Args()

    # Model (Dynamic Neuron Transformer)
    args.model_version = cfg['model'].get('model_version', '17.1')  # Default to v17.1
    args.d_model = cfg['model'].get('d_model', 512)
    args.n_layers = cfg['model'].get('n_layers', 6)
    args.n_heads = cfg['model'].get('n_heads', 8)
    args.n_neurons = cfg['model'].get('n_neurons', 1024)
    args.n_patterns = cfg['model'].get('n_patterns', 512)

    # Backward compatibility: neuron_k (new) vs k (old)
    args.neuron_k = cfg['model'].get('neuron_k', cfg['model'].get('k', 8))
    args.k = args.neuron_k  # Keep k for backward compatibility
    args.pattern_k = cfg['model'].get('pattern_k', 16)

    args.d_ff = cfg['model'].get('d_ff', None)  # Auto-calculate if None
    args.max_seq_len = cfg['model'].get('max_seq_len', 2048)
    args.dropout = cfg['model'].get('dropout', 0.1)
    args.pattern_dropout = cfg['model'].get('pattern_dropout', 0.0)

    # v6.0: Basis FFN parameters
    args.neuron_rank = cfg['model'].get('neuron_rank', None)  # v6.0: not used anymore (backward compat)
    args.n_basis = cfg['model'].get('n_basis', 8)
    # v10+ uses 'rank' key, older versions use 'basis_rank'
    args.basis_rank = cfg['model'].get('basis_rank', cfg['model'].get('rank', 64))
    args.mod_rank = cfg['model'].get('mod_rank', None)  # v5.0 compatibility (ignored)
    args.router_temperature = cfg['model'].get('router_temperature', None)  # v6.0 only (v7.0 ignores)

    # Knowledge neurons parameters
    args.n_knowledge = cfg['model'].get('n_knowledge', 64)
    args.knowledge_rank = cfg['model'].get('knowledge_rank', None)  # None = use rank

    # 2-stage retrieval parameters (v15+)
    args.coarse_k = cfg['model'].get('coarse_k', 20)
    args.fine_k = cfg['model'].get('fine_k', 10)

    # SSM parameters
    args.state_dim = cfg['model'].get('state_dim', 64)

    # Gradient checkpointing
    args.gradient_checkpointing = cfg['model'].get('gradient_checkpointing', False)

    # Unified router parameters
    args.d_space = cfg['model'].get('d_space', 64)
    args.router_dropout = cfg['model'].get('router_dropout', 0.1)

    # Load all version-specific model params from VERSION_REGISTRY
    load_model_params_to_args(args, cfg['model'])

    # Training
    args.batch_size = cfg['training']['batch_size']
    args.num_epochs = cfg['training']['num_epochs']
    args.lr = cfg['training'].get('lr', cfg['training'].get('learning_rate'))
    args.weight_decay = cfg['training']['weight_decay']

    # CLI overrides (takes precedence over config)
    if cli_args.epochs is not None:
        args.num_epochs = cli_args.epochs
        print(f"📌 CLI override: epochs={args.num_epochs}")
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
        print(f"📌 CLI override: batch_size={args.batch_size}")
    if cli_args.lr is not None:
        args.lr = cli_args.lr
        print(f"📌 CLI override: lr={args.lr}")
    if cli_args.skip_householder:
        args.skip_householder = True
        print(f"📌 CLI override: skip_householder=True (ablation mode)")
    if cli_args.gelu_only:
        args.compress_gelu = True
        print(f"📌 CLI override: compress_gelu=True (GELU after compress)")
    args.warmup_epochs = cfg['training'].get('warmup_epochs', None)
    args.warmup_ratio = cfg['training'].get('warmup_ratio', None)  # Alternative to warmup_epochs

    # Regularization weights
    args.orthogonality_weight = cfg['training'].get('orthogonality_weight', 0.0)  # v6.0 compat
    args.diversity_weight = cfg['training'].get('diversity_weight', 0.0)  # v7.0: recipe diversity
    args.load_balance_weight = cfg['training'].get('load_balance_weight', 0.0)  # v7.0: load balance
    args.entropy_weight = cfg['training'].get('entropy_weight', 0.0)  # router entropy loss
    args.tau_reg_weight = cfg['training'].get('tau_reg_weight', 0.0)  # v18.0: tau regularization
    args.gradient_accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)  # gradient accumulation

    # Other
    args.use_amp = cfg.get('use_amp', True)
    args.checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')
    args.log_dir = cfg.get('log_dir', 'logs')

    # Device
    is_tpu = False
    if HAS_XLA:
        device = xm.xla_device()
        is_tpu = True
        print(f"Using device: TPU ({device})")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # Create directories
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint loading logic
    latest_best_checkpoint = None
    checkpoint_dir = None

    if cli_args.resume:
        # Explicit resume - can be either a .pt file or a folder
        resume_path = Path(cli_args.resume)

        if resume_path.suffix == '.pt':
            # Direct .pt file path
            if resume_path.exists():
                latest_best_checkpoint = resume_path
                checkpoint_dir = resume_path.parent  # Use the folder containing the checkpoint
                print(f"\n✓ Resuming from checkpoint file: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")
            else:
                print(f"\n⚠️  Warning: Checkpoint file not found at {resume_path}")
                print(f"    Starting from scratch instead.")
        else:
            # Folder path - look for best_model.pt or latest checkpoint inside
            resume_folder = resume_path
            if not resume_folder.is_absolute():
                resume_folder = Path(args.checkpoint_dir) / resume_folder.name

            best_ckpt = resume_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                checkpoint_dir = resume_folder  # Use existing folder
                print(f"\n✓ Resuming from: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")
            else:
                # best_model.pt doesn't exist, look for checkpoint_epoch*_step*.pt
                checkpoint_files = sorted(
                    resume_folder.glob('checkpoint_epoch*_step*.pt'),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if checkpoint_files:
                    latest_best_checkpoint = checkpoint_files[0]
                    checkpoint_dir = resume_folder
                    print(f"\n✓ No best_model.pt found, using latest intermediate checkpoint:")
                    print(f"  {latest_best_checkpoint}")
                    print(f"✓ Continuing in same folder: {checkpoint_dir}")
                else:
                    print(f"\n⚠️  Warning: No checkpoints found in {resume_folder}")
                    print(f"    Starting from scratch instead.")

    elif not cli_args.from_scratch:
        # Auto-resume: find latest checkpoint and use its folder
        run_folders = sorted([
            d for d in base_checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ], reverse=True)

        if run_folders:
            latest_folder = run_folders[0]
            best_ckpt = latest_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                checkpoint_dir = latest_folder  # Use existing folder
                print(f"\n✓ Auto-resume: Found latest checkpoint: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")

    if cli_args.from_scratch:
        print(f"\n✓ Starting from scratch (--from-scratch)")

    # Create new run folder only if not resuming
    if checkpoint_dir is None:
        import random
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        version = cfg['model'].get('model_version', '9.0')
        run_name = f"run_v{version}_{timestamp}_{random_suffix}"
        checkpoint_dir = base_checkpoint_dir / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created new run folder: {checkpoint_dir}")

        # Save config for new runs (add model version if not present)
        if 'model_version' not in cfg['model']:
            cfg['model']['model_version'] = '9.0'
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    log_dir = checkpoint_dir
    print(f"Run folder: {checkpoint_dir}")

    # ============================================================
    # STEP 1: Load checkpoint config FIRST (before logging)
    # ============================================================
    resume_checkpoint = None
    if latest_best_checkpoint:
        resume_checkpoint = latest_best_checkpoint

    checkpoint_config = None
    checkpoint_training_config = None
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\n📌 Resuming from checkpoint: {resume_checkpoint}")

        # Try config.json first, then checkpoint file
        config_json_path = resume_checkpoint.parent / 'config.json'
        if config_json_path.exists():
            with open(config_json_path, 'r') as f:
                saved_cfg = json.load(f)
                checkpoint_config = saved_cfg.get('model')
                checkpoint_training_config = saved_cfg.get('training')
                # Also load top-level settings
                if 'use_amp' in saved_cfg:
                    args.use_amp = saved_cfg['use_amp']
                print(f"✅ Loaded config.json from checkpoint folder")
        else:
            temp_checkpoint = torch.load(resume_checkpoint, map_location='cpu')
            if 'config' in temp_checkpoint:
                checkpoint_config = temp_checkpoint['config']
                print(f"✅ Loaded model config from checkpoint file")
            del temp_checkpoint

    # Update args from checkpoint config (if resuming)
    if checkpoint_config:
        args.model_version = checkpoint_config.get('model_version', args.model_version)
        args.d_model = checkpoint_config.get('d_model', args.d_model)
        args.n_layers = checkpoint_config.get('n_layers', args.n_layers)
        args.n_heads = checkpoint_config.get('n_heads', args.n_heads)
        args.n_neurons = checkpoint_config.get('n_neurons', args.n_neurons)
        args.k = checkpoint_config.get('neuron_k', args.k)
        args.n_basis = checkpoint_config.get('n_basis', args.n_basis)
        args.basis_rank = checkpoint_config.get('basis_rank', args.basis_rank)
        args.d_ff = checkpoint_config.get('d_ff', args.d_ff)
        args.max_seq_len = checkpoint_config.get('max_seq_len', args.max_seq_len)
        args.dropout = checkpoint_config.get('dropout', args.dropout)

        # Shared parameters
        args.rank = checkpoint_config.get('rank', getattr(args, 'rank', 64))
        args.basis_rank = args.rank  # Sync basis_rank with rank for model creation
        args.n_knowledge = checkpoint_config.get('n_knowledge', getattr(args, 'n_knowledge', 64))

        # Load all version-specific architecture params from checkpoint (must match)
        load_model_params_to_args(args, checkpoint_config)

        if checkpoint_training_config:
            # Training hyperparameters (only if not overridden by CLI)
            if cli_args.batch_size is None:
                args.batch_size = checkpoint_training_config.get('batch_size', args.batch_size)
            if cli_args.epochs is None:
                args.num_epochs = checkpoint_training_config.get('num_epochs', args.num_epochs)
            if cli_args.lr is None:
                args.lr = checkpoint_training_config.get('lr', args.lr)
            args.warmup_ratio = checkpoint_training_config.get('warmup_ratio', args.warmup_ratio)
            args.weight_decay = checkpoint_training_config.get('weight_decay', args.weight_decay)
            # Loss weights
            args.orthogonality_weight = checkpoint_training_config.get('orthogonality_weight', args.orthogonality_weight)
            args.diversity_weight = checkpoint_training_config.get('diversity_weight', args.diversity_weight)
            args.load_balance_weight = checkpoint_training_config.get('load_balance_weight', args.load_balance_weight)
            args.entropy_weight = checkpoint_training_config.get('entropy_weight', args.entropy_weight)
            args.tau_reg_weight = checkpoint_training_config.get('tau_reg_weight', args.tau_reg_weight)
            print(f"   → Training params: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

        print(f"   → Updated args from checkpoint config (v{args.model_version})")

    # ============================================================
    # STEP 2: Print configuration summary (using updated args)
    # ============================================================
    print(f"\n{'='*60}")
    model_version = getattr(args, 'model_version', '17.1')
    if model_version == 'baseline':
        print(f"Vanilla Transformer Baseline Training")
    else:
        print(f"DAWN (Dynamic Neuron Transformer) Training")
    print(f"{'='*60}")
    print(f"\nModel version: {model_version}")
    print(f"\nModel: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")

    if model_version != 'baseline':
        # Use version_registry for version info (detailed info after model creation)
        try:
            normalized = normalize_version(model_version)
            print(f"DAWN v{normalized} (detailed info after model creation)")
        except ValueError:
            print(f"⚠️  Unknown version: {model_version}")
    else:
        print(f"Standard FFN: d_ff={args.d_ff}")

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if args.gradient_accumulation_steps > 1:
        print(f"Training: batch={args.batch_size} x {args.gradient_accumulation_steps} (effective={effective_batch_size}), epochs={args.num_epochs}, lr={args.lr}")
    else:
        print(f"Training: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

    # Regularization summary
    reg_parts = []
    if args.orthogonality_weight > 0:
        reg_parts.append(f"orth={args.orthogonality_weight}")
    if args.diversity_weight > 0:
        reg_parts.append(f"div={args.diversity_weight}")
    if args.load_balance_weight > 0:
        reg_parts.append(f"lb={args.load_balance_weight}")
    if args.entropy_weight > 0:
        reg_parts.append(f"ent={args.entropy_weight}")
    if args.tau_reg_weight > 0:
        reg_parts.append(f"tau={args.tau_reg_weight}")
    if reg_parts:
        print(f"Regularization: {', '.join(reg_parts)}")

    # ============================================================
    # STEP 3: Load data
    # ============================================================
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_data(
        data_config=cfg['data'],
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ============================================================
    # STEP 4: Create model (using args)
    # ============================================================
    print(f"\n{'='*60}")
    print("Creating DAWN model...")
    print(f"{'='*60}")

    # Build model kwargs from args using version_registry
    model_version = getattr(args, 'model_version', '17.1')

    # Build config dict dynamically from VERSION_REGISTRY (single source of truth)
    args_config = build_args_config(args, vocab_size)

    # Use version_registry to build model_kwargs with correct params for version
    model_kwargs = build_model_kwargs(model_version, args_config)

    # Create model
    model = create_model_by_version(model_version, model_kwargs)

    if is_tpu:
        model = model.to(dtype=torch.bfloat16, device=device)
        # Re-tie lm_head weights after dtype conversion (.to() breaks weight tying)
        if hasattr(model, 'lm_head') and hasattr(model, 'token_emb'):
            model.lm_head.weight = model.token_emb.weight
    else:
        model = model.to(device)
    print(f"✅ Model created: v{getattr(model, '__version__', model_version)}")

    # Print detailed model info
    if hasattr(model, 'get_model_info'):
        print()
        for line in model.get_model_info():
            print(line)

    # NOTE: torch.compile() is applied AFTER checkpoint loading to avoid _orig_mod. prefix issues

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of layers: {args.n_layers}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Warmup + Cosine scheduler
    total_steps = args.num_epochs * len(train_loader)

    # Support both warmup_ratio and warmup_epochs
    if args.warmup_ratio is not None:
        warmup_steps = int(total_steps * args.warmup_ratio)
    elif args.warmup_epochs is not None:
        warmup_steps = args.warmup_epochs * len(train_loader)
    else:
        warmup_steps = len(train_loader)  # Default: 1 epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Mixed precision scaler
    if is_tpu:
        scaler = None  # TPU uses bf16 natively, no GradScaler needed
        print(f"\nUsing bfloat16 (TPU native)")
    elif args.use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"\nUsing Automatic Mixed Precision (AMP)")
    else:
        scaler = None

    # Resume from checkpoint (weights loading)
    start_epoch = 1
    best_val_loss = float('inf')
    start_step = 0  # Step within epoch to resume from

    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\n{'='*60}")
        print("Loading checkpoint weights...")
        print(f"{'='*60}")

        # Use smart checkpoint loading with version awareness
        use_strict = checkpoint_config is not None
        checkpoint, load_info = load_checkpoint_smart(
            model, str(resume_checkpoint), device=device,
            strict=use_strict, verbose=True
        )

        # Load optimizer, scheduler, and scaler states
        start_epoch, best_val_loss, start_step = load_optimizer_state(
            optimizer, checkpoint, scheduler=scheduler, scaler=scaler, verbose=True
        )
    else:
        print(f"\n🆕 Starting fresh training (no checkpoint found)")

    # PyTorch 2.0+ compilation for speed boost (optional)
    # Applied AFTER checkpoint loading to avoid _orig_mod. prefix issues
    if cli_args.compile and hasattr(torch, 'compile'):
        print(f"\nCompiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print(f"  Model compiled successfully!")

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training log file (append mode if resuming)
    training_log_file = log_dir / "training_log.txt"

    # Open in append mode if resuming, write mode if new
    log_mode = 'a' if latest_best_checkpoint else 'w'
    if log_mode == 'w':
        with open(training_log_file, 'w') as f:
            f.write("# Training Log\n")
            f.write("# Step logs: epoch,step,loss,acc\n")
            f.write("# Epoch summaries: EPOCH,epoch,train_loss,train_acc,val_loss,val_acc,lr,time\n")
            f.write("\n")
    else:
        # Append separator for resumed training
        with open(training_log_file, 'a') as f:
            f.write(f"\n# === Resumed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    # Debug logger (if --debug flag is set)
    debug_logger = None
    if cli_args.debug:
        debug_log_file = checkpoint_dir / "debug.txt"
        debug_logger = DebugLogger(str(debug_log_file))
        print(f"\n🔍 Debug mode enabled")
        print(f"  Debug log: {debug_log_file}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Training log: {training_log_file}")
    print(f"{'='*60}")

    # Get sample batch for debug logging
    sample_batch_for_debug = None
    if debug_logger:
        sample_batch_for_debug = next(iter(train_loader))['input_ids'][:1].to(device)
        # Log initial state (before any training)
        debug_logger.log_section(f"Initial State (Before Training)")
        debug_logger.log_epoch_summary(model, sample_batch_for_debug, epoch=0)

    # Calculate total_steps for decay
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    global_step = (start_epoch - 1) * steps_per_epoch + start_step  # Resume from correct step

    for epoch in range(start_epoch, args.num_epochs + 1):
        # Clear CUDA cache at start of each epoch (helps with torch.compile memory)
        if not is_tpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        # Determine start_step for this epoch (only non-zero for first epoch when resuming)
        epoch_start_step = start_step if epoch == start_epoch else 0

        # Create subset loader if resuming mid-epoch (skip batches efficiently)
        if epoch_start_step > 0:
            from torch.utils.data import DataLoader, Subset
            from functools import partial
            # Get original dataset and create subset starting from resume point
            original_dataset = train_loader.dataset
            start_idx = epoch_start_step * args.batch_size
            subset_indices = list(range(start_idx, len(original_dataset)))
            subset_dataset = Subset(original_dataset, subset_indices)
            # Create new loader without shuffle (preserve order for this partial epoch)
            epoch_loader = DataLoader(
                subset_dataset,
                batch_size=args.batch_size,
                shuffle=False,  # No shuffle for resume epoch
                num_workers=2,
                collate_fn=train_loader.collate_fn
            )
            print(f"  Resuming from step {epoch_start_step}/{steps_per_epoch} ({len(subset_dataset)} samples)")
        else:
            epoch_loader = train_loader

        # Wrap DataLoader for TPU (host→device transfer pipeline)
        if is_tpu:
            epoch_loader = pl.MpDeviceLoader(epoch_loader, device)

        # Train
        train_loss, train_acc, neuron_metrics, global_step = train_epoch(
            model, epoch_loader, optimizer, scheduler, device, epoch, args,
            scaler, tokenizer, log_file=str(training_log_file),
            orthogonality_weight=args.orthogonality_weight,
            diversity_weight=args.diversity_weight,
            load_balance_weight=args.load_balance_weight,
            entropy_weight=args.entropy_weight,
            tau_reg_weight=args.tau_reg_weight,
            total_epoch_steps=steps_per_epoch,  # Full epoch steps for progress bar
            debug_logger=debug_logger,
            ckpt_manager=ckpt_manager,
            model_config=model_kwargs,
            start_step=epoch_start_step,
            global_step=global_step,
            total_steps=total_steps,
            val_loader=val_loader,
            val_interval=5000,  # Validation every 5000 steps
            is_tpu=is_tpu,
        )

        # Evaluate
        if is_tpu:
            val_loader_wrapped = pl.MpDeviceLoader(val_loader, device)
            val_loss, val_acc = evaluate(model, val_loader_wrapped, device, args, tokenizer, is_tpu=is_tpu)
        else:
            val_loss, val_acc = evaluate(model, val_loader, device, args, tokenizer)
        if not is_tpu:
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {format_time(epoch_time)}")

        # Print neuron metrics if available
        if neuron_metrics is not None:
            print(f"  Neuron Metrics:")
            print(f"    Usage: {neuron_metrics['avg_usage']:.1%} | "
                  f"Gini: {neuron_metrics['avg_gini']:.3f} | "
                  f"Entropy: {neuron_metrics['avg_entropy']:.3f}")
            print(f"    Top-10: {neuron_metrics['avg_top10']:.2%} | "
                  f"Top-50: {neuron_metrics['avg_top50']:.2%}")
            # Per-layer breakdown
            n_layers = sum(1 for k in neuron_metrics.keys() if k.startswith('L') and k.endswith('_usage'))
            layer_strs = []
            for i in range(n_layers):
                layer_strs.append(
                    f"L{i}: U={neuron_metrics[f'L{i}_usage']:.0%} "
                    f"G={neuron_metrics[f'L{i}_gini']:.2f} "
                    f"E={neuron_metrics[f'L{i}_entropy']:.2f}"
                )
            print(f"    Per-layer usage: {' | '.join(layer_strs)}")

        # Write epoch summary to log
        with open(training_log_file, 'a') as f:
            f.write(f"EPOCH,{epoch},{train_loss:.6f},{train_acc:.6f},"
                   f"{val_loss:.6f},{val_acc:.6f},"
                   f"{optimizer.param_groups[0]['lr']:.6e},{epoch_time:.2f}\n")

        # Debug: Log epoch summary for specific epochs
        if debug_logger and debug_logger.should_log_epoch(epoch):
            debug_logger.log_section(f"End of Epoch {epoch}")
            debug_logger.log_epoch_summary(model, sample_batch_for_debug, epoch)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model! (val_loss: {best_val_loss:.4f})")

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best,
            scheduler=scheduler, scaler=scaler, model_config=model_kwargs
        )
        if not is_tpu:
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()

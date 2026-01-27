#!/usr/bin/env python3
"""
Checkpoint Comparison Analysis Script

Compare multiple model checkpoints (Vanilla / DAWN) with:
- Validation loss, perplexity, accuracy
- Parameter count, FLOPs estimation
- Model architecture info

Usage:
    python -m scripts.evaluation.evaluate \
        --checkpoints path/to/ckpt1 path/to/ckpt2 ... \
        --val_data path/to/val.pt \
        --output results.csv

Output:
    - Table format (paper-ready)
    - CSV file with detailed metrics
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn.functional as F
import csv
import math

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from models import create_model_by_version, normalize_version
from utils.checkpoint import load_checkpoint_smart


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, config=None, seq_len=512):
    """
    Estimate theoretical FLOPs for forward pass (per sequence).

    Reports theoretical FLOPs based on active neurons (top-k sparse).
    Includes both sparse matmul and einsum operations.

    FLOPs formula: matmul (m,k) @ (k,n) = 2 * m * k * n

    Args:
        model: The model instance
        config: Checkpoint config dict (preferred source for architecture params)
        seq_len: Sequence length
    """
    # Helper: read from config first, then model attribute, then default
    def _get(key, default):
        if config and key in config and config[key] is not None:
            return config[key]
        val = getattr(model, key, None)
        if val is not None:
            return val
        # Check model.config dict if available
        model_cfg = getattr(model, 'config', {})
        if isinstance(model_cfg, dict) and key in model_cfg and model_cfg[key] is not None:
            return model_cfg[key]
        return default

    d_model = _get('d_model', 384)
    n_layers = _get('n_layers', 12)
    rank = _get('rank', 64)
    knowledge_rank = _get('knowledge_rank', 128)

    if hasattr(model, 'shared_neurons'):
        # DAWN - Top-k values
        top_k_fqk = _get('top_k_feature_qk', 16)
        top_k_fv = _get('top_k_feature_v', 6)
        top_k_rqk = _get('top_k_restore_qk', 16)
        top_k_rv = _get('top_k_restore_v', 6)
        top_k_fknow = _get('top_k_feature_know', 4)
        top_k_rknow = _get('top_k_restore_know', 4)

        # === Attention Circuit ===
        # Feature sparse matmul: Q, K (각 top_k_fqk), V (top_k_fv)
        attn_feat_matmul = 2 * (top_k_fqk * 2 + top_k_fv) * d_model * rank * seq_len

        # Feature einsum (x3 for Q, K, V)
        attn_feat_einsum = 2 * 3 * d_model * rank * seq_len

        # Restore sparse matmul: Q, K (각 top_k_rqk), V (top_k_rv)
        attn_rest_matmul = 2 * (top_k_rqk * 2 + top_k_rv) * rank * d_model * seq_len

        # Restore einsum (x3 for Q, K, V)
        attn_rest_einsum = 2 * 3 * rank * d_model * seq_len

        # Attention scores: QK^T + scores@V
        attn_scores = 2 * 2 * seq_len * seq_len * d_model

        # expand_O: Linear(d_model, d_model)
        expand_o = 2 * d_model * d_model * seq_len

        # === Knowledge Circuit ===
        # Feature sparse matmul + einsum
        know_feat = 2 * top_k_fknow * d_model * knowledge_rank * seq_len
        know_feat_ein = 2 * d_model * knowledge_rank * seq_len

        # Restore sparse matmul + einsum
        know_rest = 2 * top_k_rknow * knowledge_rank * d_model * seq_len
        know_rest_ein = 2 * knowledge_rank * d_model * seq_len

        per_layer = (attn_feat_matmul + attn_feat_einsum +
                     attn_rest_matmul + attn_rest_einsum +
                     attn_scores + expand_o +
                     know_feat + know_feat_ein +
                     know_rest + know_rest_ein)
    else:
        # Vanilla transformer
        d_ff = _get('d_ff', 4 * d_model)

        # QKV + O projections: 4 * d_model^2
        qkvo = 2 * 4 * d_model * d_model * seq_len

        # Attention scores: QK^T + scores@V
        attn_scores = 2 * 2 * seq_len * seq_len * d_model

        # FFN: up + down
        ffn = 2 * 2 * d_model * d_ff * seq_len

        per_layer = qkvo + attn_scores + ffn

    return n_layers * per_layer


def format_flops(flops):
    """Format FLOPs"""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    return f"{flops:.0f}"


def format_params(params):
    """Format parameter count"""
    if params >= 1e9:
        return f"{params/1e9:.2f}B"
    elif params >= 1e6:
        return f"{params/1e6:.2f}M"
    elif params >= 1e3:
        return f"{params/1e3:.2f}K"
    return f"{params}"


def load_val_data(val_path, max_tokens=None):
    """Load validation data from .pt file"""
    print(f"Loading validation data: {val_path}")
    data = torch.load(val_path)

    if isinstance(data, dict):
        tokens = data.get('input_ids', data.get('tokens', data.get('data', None)))
    else:
        tokens = data

    if tokens is None:
        raise ValueError(f"Could not find tokens in {val_path}")

    # Flatten if needed
    if tokens.dim() > 1:
        tokens = tokens.view(-1)

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    print(f"  Loaded {len(tokens):,} tokens")
    return tokens


def evaluate_model(model, val_tokens, batch_size=32, seq_len=512, device='cuda'):
    """Run full validation"""
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    # Create sequences
    n_tokens = len(val_tokens)
    n_seqs = n_tokens // seq_len
    if n_seqs == 0:
        raise ValueError(f"Not enough tokens ({n_tokens}) for seq_len={seq_len}")

    val_tokens = val_tokens[:n_seqs * seq_len].view(n_seqs, seq_len)
    n_batches = (n_seqs + batch_size - 1) // batch_size

    iterator = range(n_batches)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)

    with torch.no_grad():
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_seqs)
            batch = val_tokens[start_idx:end_idx].to(device)

            # Forward
            output = model(batch)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Loss & accuracy
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous().long()  # Ensure Long type for cross_entropy

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens * 100
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens
    }


def load_model_from_checkpoint(ckpt_path, device='cuda'):
    """Load model from checkpoint"""
    ckpt_path = Path(ckpt_path)

    # Find checkpoint file
    if ckpt_path.is_dir():
        candidates = ['best_model.pt', 'checkpoint_best.pt', 'model.pt']
        for c in candidates:
            if (ckpt_path / c).exists():
                ckpt_path = ckpt_path / c
                break
        else:
            pt_files = list(ckpt_path.glob('*.pt'))
            if pt_files:
                ckpt_path = sorted(pt_files)[-1]
            else:
                raise FileNotFoundError(f"No checkpoint in {ckpt_path}")

    print(f"  Loading: {ckpt_path.name}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Get config
    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    if not config:
        raise ValueError(f"No model config in checkpoint")

    # Get state_dict first (needed for version detection)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Create model
    version = config.get('model_version', config.get('version', None))

    # Auto-detect version from state_dict keys if not in config
    if version is None:
        # Check for version-specific keys (most specific first)
        v18_2_keys = ['router.tau_proj.weight', 'router.neuron_router.norm_fqk_Q.weight']
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']
        dawn_config_keys = ['n_feature_qk', 'n_feature_v', 'n_restore_qk', 'rank']

        if all(k in state_dict for k in v18_2_keys):
            version = '18.2'
        elif any(k in state_dict for k in dawn_keys):
            if config.get('learnable_tau', False) or config.get('max_paths'):
                version = '18.2'
            else:
                version = '17.1'
        elif any(k in config for k in dawn_config_keys):
            version = '17.1'
        else:
            version = 'baseline'

    version = normalize_version(version)

    model = create_model_by_version(version, config)

    # Remove compiled prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, config, ckpt_path


def get_model_info(model, ckpt_path, config):
    """Extract model information"""
    # Determine model type
    if hasattr(model, 'shared_neurons'):
        model_type = 'DAWN'
        version = getattr(model, '__version__', config.get('model_version', 'N/A'))
    else:
        model_type = 'Vanilla'
        version = 'N/A'

    # Get name from path
    name = ckpt_path.parent.name if ckpt_path.parent.name != 'checkpoints' else ckpt_path.stem

    return {
        'name': name,
        'type': model_type,
        'version': version,
        'd_model': getattr(model, 'd_model', config.get('d_model', 'N/A')),
        'n_layers': getattr(model, 'n_layers', config.get('n_layers', 'N/A')),
        'n_heads': getattr(model, 'n_heads', config.get('n_heads', 'N/A')),
    }


def compare_v18_hard_mask(model, val_tokens, batch_size=32, seq_len=512, device='cuda'):
    """
    Compare v18.x model performance with soft mask (training) vs hard mask (inference).

    Returns:
        Dict with soft_loss, hard_loss, and difference
    """
    # Check if model is v18.x with inference_hard_mask support
    if not hasattr(model, 'router') or not hasattr(model.router, 'inference_hard_mask'):
        return {'error': 'Not a v18.x model with inference_hard_mask support'}

    print("\n--- v18.x Hard Mask Comparison ---")

    # Test with soft mask (default training mode)
    model.router.inference_hard_mask = False
    model.eval()
    print("  Evaluating with soft mask (log-gated)...")
    soft_metrics = evaluate_model(model, val_tokens, batch_size, seq_len, device)

    # Test with hard mask
    model.router.inference_hard_mask = True
    model.eval()
    print("  Evaluating with hard mask (clean threshold)...")
    hard_metrics = evaluate_model(model, val_tokens, batch_size, seq_len, device)

    # Reset to default
    model.router.inference_hard_mask = False

    # Calculate difference
    loss_diff = hard_metrics['loss'] - soft_metrics['loss']
    ppl_ratio = hard_metrics['perplexity'] / soft_metrics['perplexity']

    result = {
        'soft_loss': soft_metrics['loss'],
        'soft_ppl': soft_metrics['perplexity'],
        'hard_loss': hard_metrics['loss'],
        'hard_ppl': hard_metrics['perplexity'],
        'loss_diff': loss_diff,
        'ppl_ratio': ppl_ratio,
    }

    print(f"\n  Results:")
    print(f"    Soft mask:  Loss={soft_metrics['loss']:.4f}, PPL={soft_metrics['perplexity']:.2f}")
    print(f"    Hard mask:  Loss={hard_metrics['loss']:.4f}, PPL={hard_metrics['perplexity']:.2f}")
    print(f"    Difference: ΔLoss={loss_diff:+.4f}, PPL ratio={ppl_ratio:.4f}x")

    if abs(loss_diff) < 0.01:
        print(f"    → Hard mask is equivalent (diff < 0.01)")
    elif loss_diff > 0:
        print(f"    → Hard mask is slightly worse")
    else:
        print(f"    → Hard mask is slightly better")

    return result


def print_table(results, headers):
    """Print markdown table"""
    # Calculate column widths
    col_widths = []
    for h in headers:
        max_len = len(h)
        for r in results:
            max_len = max(max_len, len(str(r.get(h, ''))))
        col_widths.append(max_len + 2)

    # Print
    header_line = '|' + '|'.join(h.center(w) for h, w in zip(headers, col_widths)) + '|'
    separator = '|' + '|'.join('-' * w for w in col_widths) + '|'

    print(header_line)
    print(separator)

    for r in results:
        values = [str(r.get(h, '')) for h in headers]
        print('|' + '|'.join(v.center(w) for v, w in zip(values, col_widths)) + '|')


def main():
    parser = argparse.ArgumentParser(description='Checkpoint Comparison Analysis')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to checkpoint files/directories')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Validation data path (.pt file)')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--max_val_tokens', type=int, default=2000000,
                        help='Max validation tokens')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation, show model info only')

    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load validation data
    val_tokens = None
    if not args.skip_eval:
        val_tokens = load_val_data(args.val_data, args.max_val_tokens)

    # Analyze checkpoints
    results = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")
        print('='*60)

        try:
            model, config, actual_path = load_model_from_checkpoint(ckpt_path, args.device)
            info = get_model_info(model, actual_path, config)

            total_params, _ = count_parameters(model)
            flops = estimate_flops(model, args.seq_len)

            if not args.skip_eval and val_tokens is not None:
                metrics = evaluate_model(
                    model, val_tokens,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=args.device
                )
            else:
                metrics = {'loss': None, 'perplexity': None, 'accuracy': None}

            result = {
                'Model': info['name'],
                'Type': info['type'],
                'Params': format_params(total_params),
                'Params_raw': total_params,
                'FLOPs': format_flops(flops),
                'FLOPs_raw': flops,
                'd_model': info['d_model'],
                'n_layers': info['n_layers'],
                'Val Loss': f"{metrics['loss']:.4f}" if metrics['loss'] else 'N/A',
                'PPL': f"{metrics['perplexity']:.1f}" if metrics['perplexity'] else 'N/A',
                'Acc': f"{metrics['accuracy']:.1f}%" if metrics['accuracy'] else 'N/A',
            }
            results.append(result)

            # Print summary
            print(f"  Type: {info['type']} | d={info['d_model']}, L={info['n_layers']}")
            print(f"  Params: {format_params(total_params)} | FLOPs: {format_flops(flops)}")
            if metrics['loss']:
                print(f"  Loss: {metrics['loss']:.4f} | PPL: {metrics['perplexity']:.1f} | Acc: {metrics['accuracy']:.1f}%")

            # v18.x hard mask comparison (auto-detect)
            if val_tokens is not None and hasattr(model, 'router') and hasattr(model.router, 'inference_hard_mask'):
                hard_mask_result = compare_v18_hard_mask(
                    model, val_tokens,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=args.device
                )
                if 'error' not in hard_mask_result:
                    result['Hard Loss'] = f"{hard_mask_result['hard_loss']:.4f}"
                    result['Hard PPL'] = f"{hard_mask_result['hard_ppl']:.1f}"
                    result['ΔLoss'] = f"{hard_mask_result['loss_diff']:+.4f}"

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'Model': Path(ckpt_path).name,
                'Type': 'ERROR',
                'Params': 'N/A',
                'FLOPs': 'N/A',
                'Val Loss': 'ERROR',
                'PPL': 'ERROR',
                'Acc': 'ERROR',
            })

    # Print final table
    print(f"\n{'='*80}")
    print("RESULTS (Paper-ready)")
    print('='*80 + '\n')

    headers = ['Model', 'Params', 'FLOPs', 'Val Loss', 'PPL', 'Acc']
    print_table(results, headers)

    # Save CSV
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()

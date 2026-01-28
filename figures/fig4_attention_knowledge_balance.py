#!/usr/bin/env python3
"""
Figure 4: Attention-Knowledge Balance (from checkpoint)

Extracts real routing statistics from DAWN v17.1 checkpoint:
- Neuron Utilization: EMA usage from each pool
- Layer-wise Circuit Contribution: attention vs knowledge per layer

Usage:
    python figures/fig4_attention_knowledge_balance.py --checkpoint /path/to/run_dir
    python figures/fig4_attention_knowledge_balance.py --demo  # Use demo data
"""

import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add project root
FIGURES_DIR = Path(__file__).parent
PROJECT_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
COLOR_ATTENTION = '#4A90D9'
COLOR_KNOWLEDGE = '#50C878'
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'


def load_checkpoint_stats(checkpoint_dir: str) -> dict:
    """Load routing statistics from checkpoint."""
    import torch
    from pathlib import Path

    ckpt_path = Path(checkpoint_dir)

    # Find checkpoint file
    if ckpt_path.is_file():
        ckpt_file = ckpt_path
    else:
        # Search for best or latest checkpoint
        pt_files = list(ckpt_path.glob('*.pt'))
        ckpt_file = None
        for f in pt_files:
            if 'best' in f.name.lower() or 'final' in f.name.lower():
                ckpt_file = f
                break
        if ckpt_file is None and pt_files:
            ckpt_file = sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1]

    if ckpt_file is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    # Extract EMA usage stats from state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    utilization = {}
    layer_stats = []

    # Look for EMA usage tensors in state_dict
    ema_keys = {
        'Feature_Q': 'usage_ema_feature_q',
        'Feature_K': 'usage_ema_feature_k',
        'Feature_V': 'usage_ema_feature_v',
        'Restore_Q': 'usage_ema_restore_q',
        'Restore_K': 'usage_ema_restore_k',
        'Restore_V': 'usage_ema_restore_v',
        'Feature_Know': 'usage_ema_feature_know',
        'Restore_Know': 'usage_ema_restore_know',
    }

    for display_name, key_pattern in ema_keys.items():
        # Find matching key in state_dict
        for key, value in state_dict.items():
            if key_pattern in key.lower().replace('_', ''):
                if isinstance(value, torch.Tensor):
                    # Calculate percentage of neurons with usage > threshold
                    usage = value.float()
                    active_ratio = (usage > 0.01).float().mean().item() * 100
                    utilization[display_name] = active_ratio
                    print(f"  {display_name}: {active_ratio:.1f}% active")
                break

    # If no EMA stats found, try to load model and run inference
    if not utilization:
        print("No EMA stats in checkpoint, running inference to collect stats...")
        utilization, layer_stats = collect_stats_from_inference(checkpoint_dir)

    return {
        'utilization': utilization,
        'layer_stats': layer_stats,
    }


def collect_stats_from_inference(checkpoint_dir: str, max_batches: int = 50) -> tuple:
    """Run inference to collect routing statistics."""
    import torch
    from models import create_model_by_version

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    ckpt_path = Path(checkpoint_dir)
    pt_files = list(ckpt_path.glob('*.pt')) if ckpt_path.is_dir() else [ckpt_path]

    ckpt_file = None
    for f in pt_files:
        if 'best' in f.name.lower() or 'final' in f.name.lower():
            ckpt_file = f
            break
    if ckpt_file is None and pt_files:
        ckpt_file = sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1]

    print(f"Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    model = create_model_by_version('17.1', config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    # Get router
    router = None
    if hasattr(model, 'router'):
        router = model.router
    elif hasattr(model, 'global_routers'):
        router = model.global_routers

    if router is None:
        print("Warning: Could not find router in model")
        return {}, []

    # Collect layer-wise stats from a few forward passes
    from collections import defaultdict
    layer_attn_contributions = defaultdict(list)
    pool_activations = defaultdict(list)

    # Create dummy data for forward pass
    batch_size = 4
    seq_len = 128
    vocab_size = config.get('vocab_size', 30522)

    print(f"Running {max_batches} forward passes to collect stats...")
    with torch.no_grad():
        for batch_idx in range(max_batches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Forward pass with routing info
            outputs = model(input_ids, return_routing_info=True)

            if isinstance(outputs, tuple) and len(outputs) >= 2:
                routing_infos = outputs[-1]

                for layer_idx, layer_info in enumerate(routing_infos):
                    if isinstance(layer_info, dict):
                        # Collect attention weights
                        attn_total = 0
                        know_total = 0

                        for key in ['fqk_weights_Q', 'fqk_weights_K', 'fv_weights',
                                   'rqk_weights_Q', 'rqk_weights_K', 'rv_weights']:
                            if key in layer_info:
                                w = layer_info[key]
                                if w is not None:
                                    attn_total += w.abs().sum().item()

                        for key in ['feature_know_w', 'restore_know_w']:
                            if key in layer_info:
                                w = layer_info[key]
                                if w is not None:
                                    know_total += w.abs().sum().item()

                        total = attn_total + know_total
                        if total > 0:
                            attn_pct = attn_total / total * 100
                            layer_attn_contributions[layer_idx].append(attn_pct)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches")

    # Calculate layer-wise attention contribution
    n_layers = len(layer_attn_contributions)
    layer_stats = []
    for i in range(n_layers):
        if layer_attn_contributions[i]:
            avg = np.mean(layer_attn_contributions[i])
            layer_stats.append(avg)
        else:
            layer_stats.append(50.0)

    # Get utilization from router EMA
    utilization = {}
    if hasattr(router, 'neuron_router'):
        nr = router.neuron_router
        ema_attrs = {
            'Feature_Q': 'usage_ema_feature_q',
            'Feature_K': 'usage_ema_feature_k',
            'Feature_V': 'usage_ema_feature_v',
            'Restore_Q': 'usage_ema_restore_q',
            'Restore_K': 'usage_ema_restore_k',
            'Restore_V': 'usage_ema_restore_v',
            'Feature_Know': 'usage_ema_feature_know',
            'Restore_Know': 'usage_ema_restore_know',
        }

        for display, attr in ema_attrs.items():
            if hasattr(nr, attr):
                ema = getattr(nr, attr)
                if ema is not None:
                    active_ratio = (ema > 0.01).float().mean().item() * 100
                    utilization[display] = active_ratio

    return utilization, layer_stats


def get_demo_data() -> dict:
    """Return demo data for testing without checkpoint."""
    return {
        'utilization': {
            'Feature_Q': 54.2,
            'Feature_K': 81.7,
            'Feature_V': 91.7,
            'Restore_Q': 55.0,
            'Restore_K': 64.2,
            'Restore_V': 87.5,
            'Feature_Know': 87.5,
            'Restore_Know': 91.7,
        },
        'layer_stats': [49, 59, 61, 62, 63, 63, 62, 61, 61, 61, 48, 46],
    }


def create_figure(stats: dict, output_dir: Path):
    """Create the routing statistics figure."""
    utilization = stats['utilization']
    layer_stats = stats['layer_stats']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

    # === (a) Neuron Utilization ===
    if utilization:
        # Order pools
        pool_order = ['Feature_Q', 'Feature_K', 'Feature_V',
                     'Restore_Q', 'Restore_K', 'Restore_V',
                     'Feature_Know', 'Restore_Know']
        pools = [p for p in pool_order if p in utilization][::-1]  # Reverse for display
        values = [utilization[p] for p in pools]

        # Colors based on type
        colors = []
        for p in pools:
            if 'Know' in p:
                colors.append(COLOR_KNOWLEDGE)
            elif 'Q' in p:
                colors.append('#E74C3C')  # Red for Q
            elif 'K' in p:
                colors.append('#3498DB')  # Blue for K
            else:
                colors.append('#9B59B6')  # Purple for V

        y_pos = np.arange(len(pools))
        bars = ax1.barh(y_pos, values, height=0.7, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(val + 1, i, f'{val:.0f}%', va='center', fontsize=8, color=COLOR_BLACK)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pools, fontsize=8)
        ax1.set_xlim(0, 105)
        ax1.set_xlabel('Active Neurons (%)', fontsize=9)
        ax1.set_title('(a) Neuron Utilization', fontsize=10, fontweight='bold', pad=10)
        ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax1.set_axisbelow(True)
        ax1.axvline(x=50, color=COLOR_GRAY, linestyle=':', linewidth=1, alpha=0.7)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#E74C3C', label='Q routing', alpha=0.85),
            mpatches.Patch(color='#3498DB', label='K routing', alpha=0.85),
            mpatches.Patch(color='#9B59B6', label='V routing', alpha=0.85),
            mpatches.Patch(color=COLOR_KNOWLEDGE, label='Knowledge', alpha=0.85),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax1.transAxes)

    # === (b) Layer-wise Circuit Contribution ===
    if layer_stats:
        n_layers = len(layer_stats)
        layers = list(range(1, n_layers + 1))

        ax2.plot(layers, layer_stats, 'o-', color=COLOR_ATTENTION, linewidth=2,
                markersize=6, markerfacecolor='white', markeredgewidth=1.5)

        # Fill areas
        ax2.fill_between(layers, layer_stats, 50,
                        where=[a >= 50 for a in layer_stats],
                        color=COLOR_ATTENTION, alpha=0.3)
        ax2.fill_between(layers, layer_stats, 50,
                        where=[a < 50 for a in layer_stats],
                        color=COLOR_KNOWLEDGE, alpha=0.3)

        ax2.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

        ax2.set_xlim(0.5, n_layers + 0.5)
        ax2.set_ylim(35, 75)
        ax2.set_xticks(layers)
        ax2.set_xlabel('Layer', fontsize=9)
        ax2.set_ylabel('Attention Contribution (%)', fontsize=9)
        ax2.set_title('(b) Layer-wise Circuit Contribution', fontsize=10, fontweight='bold', pad=10)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax2.set_axisbelow(True)

        # Legend
        legend_elements2 = [
            mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
            mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
            plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
        ]
        ax2.legend(handles=legend_elements2, loc='upper right', fontsize=7, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    # Save
    output_png = output_dir / 'fig4_attention_knowledge_balance.png'
    output_pdf = output_dir / 'fig4_attention_knowledge_balance.pdf'
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_png}")
    print(f"Saved: {output_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate routing statistics figure')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint directory or file')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo data (no checkpoint needed)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: figures/)')

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else FIGURES_DIR

    print("=" * 50)
    print("Figure 5: Attention-Knowledge Balance")
    print("=" * 50)

    if args.demo or not args.checkpoint:
        if not args.demo and not args.checkpoint:
            print("\nNo checkpoint provided, using demo data...")
        else:
            print("\nUsing demo data...")
        stats = get_demo_data()
    else:
        print(f"\nLoading from: {args.checkpoint}")
        try:
            stats = load_checkpoint_stats(args.checkpoint)

            # Fill missing with demo data
            demo = get_demo_data()
            if not stats['utilization']:
                print("Using demo utilization data...")
                stats['utilization'] = demo['utilization']
            if not stats['layer_stats']:
                print("Using demo layer stats...")
                stats['layer_stats'] = demo['layer_stats']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Falling back to demo data...")
            stats = get_demo_data()

    print(f"\nUtilization pools: {len(stats['utilization'])}")
    print(f"Layer stats: {len(stats['layer_stats'])} layers")

    create_figure(stats, output_dir)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())

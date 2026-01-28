#!/usr/bin/env python3
"""
Figure 6 (Appendix): Convergence Comparison
Line plot comparing DAWN vs Vanilla training dynamics.

Usage:
    # From checkpoint directories (will find training_log.txt):
    python figures/fig6_convergence_comparison.py \\
        --checkpoints path/to/dawn_ckpt path/to/vanilla_22m path/to/vanilla_108m \\
        --labels "DAWN-24M" "Vanilla-22M" "Vanilla-108M"

    # Example with actual run directories:
    python figures/fig6_convergence_comparison.py \\
        --checkpoints \\
            logs_v17.1_20M_c4_5B/run_v17.1_20251217_172040_8948 \\
            logs_baseline_22M_c4_5B/run_vbaseline_20251210_134902_4447 \\
            logs_baseline_108M_c4_5B/run_vbaseline_20251216_220530_1907 \\
        --labels "DAWN-24M" "Vanilla-22M" "Vanilla-108M"

    # From log files directly:
    python figures/fig6_convergence_comparison.py \\
        --logs path/to/training_log.txt \\
        --labels "DAWN-24M"

    # With demo data (for testing):
    python figures/fig6_convergence_comparison.py --demo

Output:
    figures/fig6_convergence_comparison.pdf
    figures/fig6_convergence_comparison.png

Key message:
    DAWN-24M converges faster than Vanilla-22M and achieves lower loss
    than even Vanilla-108M (4.5x larger model).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from pathlib import Path

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors and styles for specific models
MODEL_STYLES = {
    'DAWN-24M': {'color': '#4A90D9', 'linestyle': '-', 'linewidth': 2.5},
    'Vanilla-22M': {'color': '#888888', 'linestyle': '--', 'linewidth': 2.0},
    'Vanilla-108M': {'color': '#444444', 'linestyle': '-.', 'linewidth': 2.0},
}

# Default colors for unknown models
DEFAULT_COLORS = ['#4A90D9', '#888888', '#444444', '#E74C3C', '#50C878', '#9B59B6']
DEFAULT_STYLES = ['-', '--', '-.', ':', '-', '--']


def find_training_log(checkpoint_path: str, verbose: bool = True) -> str:
    """
    Find training_log.txt from checkpoint path.

    Searches in order:
    1. Same directory as checkpoint
    2. Parent directory
    3. Recursive search in directory
    4. Common log file patterns
    """
    ckpt_path = Path(checkpoint_path)

    # If it's a file, use its directory
    if ckpt_path.is_file():
        ckpt_dir = ckpt_path.parent
    else:
        ckpt_dir = ckpt_path

    # Search patterns - prioritized list
    search_paths = [
        ckpt_dir / 'training_log.txt',
        ckpt_dir / 'training.log',
        ckpt_dir / 'train_log.txt',
        ckpt_dir / 'log.txt',
        ckpt_dir.parent / 'training_log.txt',
        ckpt_dir.parent / 'training.log',
    ]

    # Also try finding by replacing 'checkpoints' with 'logs' in path
    ckpt_str = str(ckpt_dir)
    if 'checkpoint' in ckpt_str.lower():
        log_path = ckpt_str.replace('checkpoint', 'log').replace('Checkpoint', 'Log')
        search_paths.append(Path(log_path) / 'training_log.txt')

    # Check explicit paths first
    for path in search_paths:
        if path.exists():
            return str(path)

    # Try glob search in directory
    if ckpt_dir.exists():
        for pattern in ['**/training_log.txt', '**/training*.log', '**/*log*.txt']:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                return str(matches[0])

    # Debug: print what we searched
    if verbose:
        print(f"  Searched paths for {ckpt_path.name}:")
        for p in search_paths[:4]:
            exists = "✓" if p.exists() else "✗"
            print(f"    {exists} {p}")
        if ckpt_dir.exists():
            files = list(ckpt_dir.iterdir())[:10]
            print(f"  Files in directory: {[f.name for f in files]}")

    return None


def parse_training_log(log_path: str, use_val_loss: bool = True):
    """
    Parse training log file to extract step and loss data.

    Log format (from train.py):
        # Step logs: epoch,step,loss,acc
        # Epoch summaries: EPOCH,epoch,train_loss,train_acc,val_loss,val_acc,lr,time

        epoch=1,step=100,loss=5.123456,acc=0.1234
        epoch=1,step=5000,val_loss=4.567890,val_acc=0.2345
        EPOCH,1,4.12,0.15,3.98,0.18,0.00055,1234.5

    Args:
        log_path: Path to training_log.txt
        use_val_loss: If True, prefer validation loss; else use training loss

    Returns:
        steps: list of step numbers (cumulative)
        losses: list of losses
        metadata: dict with additional info
    """
    steps = []
    losses = []
    epochs_data = []

    current_epoch = 0
    steps_per_epoch = None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Match step-level validation logs: epoch=X,step=Y,val_loss=Z,val_acc=W
            val_match = re.search(r'epoch=(\d+),step=(\d+),val_loss=([\d.]+)', line)
            if val_match and use_val_loss:
                epoch = int(val_match.group(1))
                step = int(val_match.group(2))
                loss = float(val_match.group(3))

                # Calculate cumulative step
                if steps_per_epoch and epoch > 1:
                    cumulative_step = (epoch - 1) * steps_per_epoch + step
                else:
                    cumulative_step = step

                steps.append(cumulative_step)
                losses.append(loss)
                continue

            # Match step-level training logs: epoch=X,step=Y,loss=Z,acc=W
            train_match = re.search(r'epoch=(\d+),step=(\d+),loss=([\d.]+)', line)
            if train_match and not use_val_loss:
                epoch = int(train_match.group(1))
                step = int(train_match.group(2))
                loss = float(train_match.group(3))

                if steps_per_epoch and epoch > 1:
                    cumulative_step = (epoch - 1) * steps_per_epoch + step
                else:
                    cumulative_step = step

                steps.append(cumulative_step)
                losses.append(loss)
                continue

            # Match epoch summary: EPOCH,epoch,train_loss,train_acc,val_loss,val_acc,lr,time
            if line.startswith('EPOCH,'):
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        epoch = int(parts[1])
                        train_loss = float(parts[2])
                        val_loss = float(parts[4])

                        # Estimate steps per epoch from last step log
                        if steps and epoch == 1:
                            steps_per_epoch = max(steps)

                        epochs_data.append({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                        })
                    except (ValueError, IndexError):
                        pass

    # If no step-level data, fall back to epoch-level
    if not steps and epochs_data:
        estimated_steps_per_epoch = 100000  # Default estimate
        for ed in epochs_data:
            steps.append(ed['epoch'] * estimated_steps_per_epoch)
            losses.append(ed['val_loss'] if use_val_loss else ed['train_loss'])

    metadata = {
        'log_path': log_path,
        'n_points': len(steps),
        'steps_per_epoch': steps_per_epoch,
        'n_epochs': len(epochs_data),
    }

    return steps, losses, metadata


def generate_demo_data():
    """Generate synthetic training curves for demonstration."""
    np.random.seed(42)
    steps = np.arange(0, 102000, 2000)

    data = {}

    # DAWN: faster convergence, lower final loss
    dawn_loss = 4.5 * np.exp(-steps / 30000) + 2.1 + 0.03 * np.random.randn(len(steps))
    dawn_loss = np.maximum(dawn_loss, 2.1)
    data['DAWN-24M'] = (steps.tolist(), dawn_loss.tolist())

    # Vanilla-22M: slower convergence, higher final loss
    vanilla_22m_loss = 5.0 * np.exp(-steps / 40000) + 3.95 + 0.03 * np.random.randn(len(steps))
    vanilla_22m_loss = np.maximum(vanilla_22m_loss, 3.95)
    data['Vanilla-22M'] = (steps.tolist(), vanilla_22m_loss.tolist())

    # Vanilla-108M: medium convergence
    vanilla_108m_loss = 4.8 * np.exp(-steps / 35000) + 3.45 + 0.03 * np.random.randn(len(steps))
    vanilla_108m_loss = np.maximum(vanilla_108m_loss, 3.45)
    data['Vanilla-108M'] = (steps.tolist(), vanilla_108m_loss.tolist())

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate training loss curve figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From checkpoint directories:
    python figures/fig6_convergence_comparison.py \\
        --checkpoints ckpts/dawn_v17.1 ckpts/vanilla_22m \\
        --labels "DAWN-24M" "Vanilla-22M"

    # From log files:
    python figures/fig6_convergence_comparison.py \\
        --logs logs/dawn/training_log.txt logs/vanilla/training_log.txt

    # Demo mode:
    python figures/fig6_convergence_comparison.py --demo
        """
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--checkpoints', nargs='+',
                            help='Checkpoint directories (will find training_log.txt)')
    input_group.add_argument('--logs', nargs='+',
                            help='Training log files directly')
    input_group.add_argument('--demo', action='store_true',
                            help='Use demo data for testing')

    # Labels and output
    parser.add_argument('--labels', nargs='+',
                       help='Labels for each model (default: extracted from path)')
    parser.add_argument('--output', type=str, default='figures/fig6_convergence_comparison',
                       help='Output path without extension (default: figures/fig6_convergence_comparison)')

    # Plot options
    parser.add_argument('--use_train_loss', action='store_true',
                       help='Use training loss instead of validation loss')
    parser.add_argument('--log_scale', action='store_true',
                       help='Use log scale for y-axis')
    parser.add_argument('--title', type=str, default=None,
                       help='Figure title (optional)')
    parser.add_argument('--show_annotations', action='store_true',
                       help='Show final loss values at line endpoints')

    args = parser.parse_args()

    # Collect data
    data = {}

    if args.demo:
        print("Using demo data...")
        data = generate_demo_data()

    elif args.checkpoints:
        for i, ckpt_path in enumerate(args.checkpoints):
            # Find training log
            log_path = find_training_log(ckpt_path)
            if log_path is None:
                print(f"Warning: Could not find training_log.txt for {ckpt_path}")
                continue

            # Parse log
            steps, losses, metadata = parse_training_log(
                log_path, use_val_loss=not args.use_train_loss
            )

            if not steps:
                print(f"Warning: No data found in {log_path}")
                continue

            # Determine label
            if args.labels and i < len(args.labels):
                label = args.labels[i]
            else:
                # Extract from path
                label = Path(ckpt_path).name
                # Clean up common patterns
                label = re.sub(r'checkpoint[s]?_?', '', label, flags=re.IGNORECASE)
                label = re.sub(r'_c4_\d+[BMK]', '', label)
                label = label.strip('_')

            data[label] = (steps, losses)
            print(f"Loaded {label}: {metadata['n_points']} points from {log_path}")

    elif args.logs:
        for i, log_path in enumerate(args.logs):
            if not Path(log_path).exists():
                print(f"Warning: Log file not found: {log_path}")
                continue

            steps, losses, metadata = parse_training_log(
                log_path, use_val_loss=not args.use_train_loss
            )

            if not steps:
                print(f"Warning: No data found in {log_path}")
                continue

            if args.labels and i < len(args.labels):
                label = args.labels[i]
            else:
                label = Path(log_path).parent.name

            data[label] = (steps, losses)
            print(f"Loaded {label}: {metadata['n_points']} points")

    if not data:
        print("Error: No data available. Use --demo for demonstration or provide valid paths.")
        return 1

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot each model
    for i, (label, (steps, losses)) in enumerate(data.items()):
        # Use predefined style if available, otherwise fallback to defaults
        if label in MODEL_STYLES:
            style = MODEL_STYLES[label]
            color = style['color']
            linestyle = style['linestyle']
            linewidth = style['linewidth']
        else:
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            linestyle = DEFAULT_STYLES[i % len(DEFAULT_STYLES)]
            linewidth = 2.5 if i == 0 else 2.0

        ax.plot(steps, losses, linestyle=linestyle, color=color,
                linewidth=linewidth, label=label)

    # Formatting
    loss_type = 'Training' if args.use_train_loss else 'Validation'
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel(f'{loss_type} Loss', fontsize=11)

    if args.title:
        ax.set_title(args.title, fontsize=12, fontweight='bold')

    # Log scale if requested or if range is large
    if args.log_scale:
        ax.set_yscale('log')
    else:
        all_losses = []
        for _, (_, losses) in data.items():
            all_losses.extend(losses)
        if max(all_losses) / min(all_losses) > 5:
            ax.set_yscale('log')

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Format x-axis with K notation
    def format_steps(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        return f'{x:.0f}'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_steps))

    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    # Clean annotation: show final loss values at line endpoints (optional)
    if getattr(args, 'show_annotations', False):
        for label, (steps, losses) in data.items():
            if steps and losses:
                final_step = steps[-1]
                final_loss = losses[-1]
                style = MODEL_STYLES.get(label, {})
                color = style.get('color', DEFAULT_COLORS[0])

                # Small text showing final loss at end of line
                ax.annotate(f'{final_loss:.2f}',
                           xy=(final_step, final_loss),
                           textcoords='offset points',
                           xytext=(5, 0),
                           fontsize=7, color=color, va='center')

    plt.tight_layout()

    # Save
    output_base = args.output
    os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)

    plt.savefig(f'{output_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_base}.pdf")
    print(f"Saved: {output_base}.png")
    plt.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())

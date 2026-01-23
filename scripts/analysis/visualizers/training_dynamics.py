"""
Training Dynamics Visualizer
============================

Fig 6: Convergence Comparison

Paper-quality visualization for training loss curves.
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Style settings
if HAS_MATPLOTLIB:
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

DEFAULT_COLORS = ['#4A90D9', '#888888', '#444444', '#E74C3C', '#50C878', '#9B59B6']
DEFAULT_STYLES = ['-', '--', '-.', ':', '-', '--']


def find_training_log(checkpoint_path: str) -> Optional[str]:
    """
    Find training_log.txt from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        Path to training log file or None
    """
    ckpt_path = Path(checkpoint_path)

    if ckpt_path.is_file():
        ckpt_dir = ckpt_path.parent
    else:
        ckpt_dir = ckpt_path

    # Search patterns
    search_paths = [
        ckpt_dir / 'training_log.txt',
        ckpt_dir / 'training.log',
        ckpt_dir / 'train_log.txt',
        ckpt_dir / 'log.txt',
        ckpt_dir.parent / 'training_log.txt',
        ckpt_dir.parent / 'training.log',
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    # Glob search
    if ckpt_dir.exists():
        for pattern in ['**/training_log.txt', '**/training*.log', '**/*log*.txt']:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                return str(matches[0])

    return None


def parse_training_log(log_path: str, use_val_loss: bool = True) -> Tuple[List, List, Dict]:
    """
    Parse training log file to extract step and loss data.

    Log format:
        epoch=1,step=100,loss=5.123456,acc=0.1234
        epoch=1,step=5000,val_loss=4.567890,val_acc=0.2345
        EPOCH,1,4.12,0.15,3.98,0.18,0.00055,1234.5

    Args:
        log_path: Path to training_log.txt
        use_val_loss: If True, prefer validation loss

    Returns:
        steps: list of step numbers
        losses: list of losses
        metadata: dict with additional info
    """
    steps = []
    losses = []
    epochs_data = []

    steps_per_epoch = None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Match step-level validation logs
            val_match = re.search(r'epoch=(\d+),step=(\d+),val_loss=([\d.]+)', line)
            if val_match and use_val_loss:
                epoch = int(val_match.group(1))
                step = int(val_match.group(2))
                loss = float(val_match.group(3))

                if steps_per_epoch and epoch > 1:
                    cumulative_step = (epoch - 1) * steps_per_epoch + step
                else:
                    cumulative_step = step

                steps.append(cumulative_step)
                losses.append(loss)
                continue

            # Match step-level training logs
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

            # Match epoch summary
            if line.startswith('EPOCH,'):
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        epoch = int(parts[1])
                        train_loss = float(parts[2])
                        val_loss = float(parts[4])

                        if steps and epoch == 1:
                            steps_per_epoch = max(steps)

                        epochs_data.append({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                        })
                    except (ValueError, IndexError):
                        pass

    # Fall back to epoch-level data
    if not steps and epochs_data:
        estimated_steps_per_epoch = 100000
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


def plot_training_dynamics(
    data: Dict[str, Tuple[List, List]],
    output_path: str,
    use_log_scale: bool = False,
    title: Optional[str] = 'Fig 6: Convergence Comparison',
    dpi: int = 300
) -> Optional[str]:
    """
    Generate paper-quality training loss curve figure.

    Args:
        data: Dict mapping model names to (steps, losses) tuples
        output_path: Path for output PNG
        use_log_scale: Use log scale for y-axis
        title: Optional figure title
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    if not data:
        print("No training data to plot")
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    # Plot each model
    for i, (label, (steps, losses)) in enumerate(data.items()):
        if not steps or not losses:
            continue

        # Use predefined style if available
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
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Auto log scale if range is large
    if use_log_scale:
        ax.set_yscale('log')
    else:
        all_losses = []
        for _, (_, losses) in data.items():
            all_losses.extend(losses)
        if all_losses and max(all_losses) / min(all_losses) > 5:
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

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PNG only
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path


def plot_training_from_logs(
    log_paths: List[str],
    labels: Optional[List[str]],
    output_path: str,
    use_val_loss: bool = True,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate training loss curve from log files.

    Args:
        log_paths: List of paths to training log files
        labels: Labels for each model (optional)
        output_path: Path for output PNG
        use_val_loss: Use validation loss instead of training loss
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    data = {}

    for i, log_path in enumerate(log_paths):
        if not Path(log_path).exists():
            print(f"Warning: Log file not found: {log_path}")
            continue

        steps, losses, metadata = parse_training_log(log_path, use_val_loss)

        if not steps:
            print(f"Warning: No data found in {log_path}")
            continue

        if labels and i < len(labels):
            label = labels[i]
        else:
            label = Path(log_path).parent.name

        data[label] = (steps, losses)
        print(f"Loaded {label}: {metadata['n_points']} points")

    return plot_training_dynamics(data, output_path, dpi=dpi)


def plot_training_from_checkpoints(
    checkpoint_paths: List[str],
    labels: Optional[List[str]],
    output_path: str,
    use_val_loss: bool = True,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate training loss curve from checkpoint directories.

    Args:
        checkpoint_paths: List of paths to checkpoint directories
        labels: Labels for each model (optional)
        output_path: Path for output PNG
        use_val_loss: Use validation loss instead of training loss
        dpi: Output resolution

    Returns:
        Path to saved figure
    """
    log_paths = []

    for ckpt_path in checkpoint_paths:
        log_path = find_training_log(ckpt_path)
        if log_path:
            log_paths.append(log_path)
            print(f"Found log: {log_path}")
        else:
            print(f"Warning: No training log found for {ckpt_path}")

    if not log_paths:
        print("No training logs found")
        return None

    return plot_training_from_logs(log_paths, labels, output_path, use_val_loss, dpi)

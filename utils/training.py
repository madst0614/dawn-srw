"""
Training utilities for SPROUT.

Checkpoint management, monitoring, and training helpers.
"""

import torch
import os
import json
from datetime import datetime
from typing import Dict, Optional


class CheckpointManager:
    """
    Manage model checkpoints with best model tracking.
    """

    def __init__(self, checkpoint_dir: str, keep_best_n: int = 3):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_best_n = keep_best_n
        self.checkpoints = []  # List of (loss, path) tuples

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metrics: dict,
        is_best: bool = False,
        scheduler=None,
        scaler=None,
        model_config: dict = None,
        filename: str = None,
        epoch_completed: bool = True
    ) -> str:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            metrics: Training metrics
            is_best: Whether this is the best model so far
            scheduler: Optional scheduler state
            scaler: Optional AMP scaler state
            model_config: Optional model configuration dict
            filename: Optional custom filename (default: checkpoint_epoch{epoch}.pt)
            epoch_completed: Whether the epoch was fully completed (False for mid-epoch saves)

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"checkpoint_epoch{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Get model version (handle torch.compile wrapper)
        if hasattr(model, '_orig_mod'):
            model_version = getattr(model._orig_mod, '__version__', 'unknown')
        else:
            model_version = getattr(model, '__version__', 'unknown')

        # Get state dict and strip _orig_mod. prefix from torch.compile() wrapped models
        state_dict = model.state_dict()
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_version': model_version,
            'epoch_completed': epoch_completed,
            'step': metrics.get('step', 0)  # Save step for mid-epoch resume
        }

        # Save model config if provided
        if model_config is not None:
            checkpoint['config'] = model_config

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Use xm.save() on TPU, torch.save() otherwise
        try:
            import torch_xla.core.xla_model as xm
            xm.save(checkpoint, checkpoint_path)
        except ImportError:
            torch.save(checkpoint, checkpoint_path)

        # Also save as best_model.pt if this is the best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            try:
                import torch_xla.core.xla_model as xm
                xm.save(checkpoint, best_path)
            except ImportError:
                torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: best_model.pt")

        # Track checkpoints
        self.checkpoints.append((loss, checkpoint_path))
        self.checkpoints.sort(key=lambda x: x[0])  # Sort by loss

        # Remove old checkpoints if we have too many
        if len(self.checkpoints) > self.keep_best_n:
            _, old_path = self.checkpoints.pop()
            if os.path.exists(old_path) and not old_path.endswith('best_'):
                os.remove(old_path)

        print(f"💾 Saved checkpoint: {filename}")
        print(f"   Loss: {loss:.4f} | Best: {is_best}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load state

        Returns:
            Checkpoint dict
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✅ Loaded checkpoint: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']} | Loss: {checkpoint['loss']:.4f}")

        return checkpoint

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if self.checkpoints:
            return self.checkpoints[0][1]  # Lowest loss
        return None


class TrainingMonitor:
    """
    Monitor training progress and log metrics.
    """

    def __init__(self, log_dir: str):
        """
        Initialize training monitor.

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        self.metrics_history = []
        self.current_epoch = 0

        os.makedirs(log_dir, exist_ok=True)

        # Create log file
        self.log_file = os.path.join(log_dir, "training_log.jsonl")

    def log_epoch(self, epoch: int, metrics: Dict):
        """
        Log epoch metrics.

        Args:
            epoch: Epoch number
            metrics: Metrics dictionary
        """
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        self.metrics_history.append(log_entry)
        self.current_epoch = epoch

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Print summary
        self._print_epoch_summary(log_entry)

    def log_step(self, step: int, metrics: Dict):
        """
        Log training step metrics.

        Args:
            step: Step number
            metrics: Metrics dictionary
        """
        log_entry = {
            'step': step,
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _print_epoch_summary(self, log_entry: Dict):
        """Print formatted epoch summary."""
        print(f"\n{'='*70}")
        print(f"EPOCH {log_entry['epoch']} SUMMARY")
        print(f"{'='*70}")

        for key, value in log_entry.items():
            if key in ['epoch', 'timestamp']:
                continue

            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k:18s}: {v}")
            else:
                print(f"  {key:20s}: {value}")

        print(f"{'='*70}\n")

    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        if not self.metrics_history:
            return {}

        # Extract specific metrics
        losses = [m.get('loss', 0) for m in self.metrics_history]
        accuracies = [m.get('accuracy', 0) for m in self.metrics_history]

        return {
            'total_epochs': len(self.metrics_history),
            'best_loss': min(losses) if losses else 0,
            'final_loss': losses[-1] if losses else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'final_accuracy': accuracies[-1] if accuracies else 0
        }


def format_time(seconds: float) -> str:
    """
    Format seconds to readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_pct': 100.0 * trainable_params / total_params if total_params > 0 else 0
    }

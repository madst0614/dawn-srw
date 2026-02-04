"""
Pure numpy/JAX data loader for DAWN training on TPU.

Reads pretokenized .bin files (uint16, packed sequences) produced by
pretokenize_c4.py. No PyTorch dependency.

Supports:
- Local files and GCS paths (gs://...)
- Checkpoint resume via global_step offset
- Multi-device batching: (n_devices, per_device_batch, seq_len)
"""

import json
import math
import os
import numpy as np
import jax.numpy as jnp


# ============================================================
# GCS / file I/O helpers
# ============================================================

def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def _get_gcs_fs():
    """Get a GCS filesystem handle (lazy import)."""
    try:
        import gcsfs
        return gcsfs.GCSFileSystem()
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.io.gfile
    except ImportError:
        raise ImportError(
            "GCS support requires 'gcsfs' or 'tensorflow'. "
            "Install with: pip install gcsfs  OR  pip install tensorflow"
        )


def _read_bin_file(path: str) -> np.ndarray:
    """Read a flat uint16 binary file into a 1D numpy array."""
    if is_gcs_path(path):
        fs = _get_gcs_fs()
        # gcsfs
        if hasattr(fs, 'open'):
            with fs.open(path, 'rb') as f:
                data = f.read()
        # tf.io.gfile
        elif hasattr(fs, 'GFile'):
            with fs.GFile(path, 'rb') as f:
                data = f.read()
        else:
            raise RuntimeError(f"Cannot read GCS path: {path}")
        return np.frombuffer(data, dtype=np.uint16)
    else:
        return np.memmap(path, dtype=np.uint16, mode='r')


def _read_json(path: str) -> dict:
    """Read a JSON file from local or GCS."""
    if is_gcs_path(path):
        fs = _get_gcs_fs()
        if hasattr(fs, 'open'):
            with fs.open(path, 'r') as f:
                return json.load(f)
        elif hasattr(fs, 'GFile'):
            with fs.GFile(path, 'r') as f:
                return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


# ============================================================
# BinDataset: numpy-based dataset from .bin file
# ============================================================

class BinDataset:
    """Memory-mapped (or loaded) dataset from a pretokenized .bin file.

    The .bin file contains uint16 tokens packed as (N, seq_len) where
    N = total_tokens // seq_len. We reshape the flat buffer accordingly.
    """

    def __init__(self, bin_path: str, seq_len: int = 512, max_sequences: int = None):
        """
        Args:
            bin_path: Path to .bin file (local or gs://).
            seq_len: Sequence length (must match pretokenization).
            max_sequences: Optional limit on number of sequences.
        """
        self.bin_path = bin_path
        self.seq_len = seq_len

        raw = _read_bin_file(bin_path)
        total_tokens = len(raw)
        n_seqs = total_tokens // seq_len
        if n_seqs == 0:
            raise ValueError(
                f"Not enough tokens in {bin_path}: {total_tokens} < seq_len={seq_len}"
            )

        # Trim tokens that don't fill a complete sequence
        used_tokens = n_seqs * seq_len
        self.data = raw[:used_tokens].reshape(n_seqs, seq_len)

        if max_sequences is not None and len(self.data) > max_sequences:
            self.data = self.data[:max_sequences]

        print(f"  BinDataset: {bin_path}")
        print(f"    {len(self.data):,} sequences x {seq_len} = {len(self.data) * seq_len:,} tokens")

    def __len__(self):
        return len(self.data)

    def get_batch(self, start_idx: int, batch_size: int) -> np.ndarray:
        """Get a batch of sequences as numpy int32 array.

        Args:
            start_idx: Starting sequence index.
            batch_size: Number of sequences.

        Returns:
            np.ndarray of shape (batch_size, seq_len), dtype int32.
        """
        end_idx = min(start_idx + batch_size, len(self.data))
        actual_size = end_idx - start_idx
        if actual_size <= 0:
            return None
        batch = self.data[start_idx:end_idx].astype(np.int32)
        return batch


# ============================================================
# DataLoader: sequential batch iterator
# ============================================================

class BinDataLoader:
    """Sequential batch iterator over a BinDataset.

    Yields (input_ids, attention_mask) as jnp.arrays.
    Supports resume from a given global_step.
    Supports multi-device reshaping.
    """

    def __init__(
        self,
        dataset: BinDataset,
        batch_size: int,
        n_devices: int = 1,
        start_step: int = 0,
    ):
        """
        Args:
            dataset: BinDataset instance.
            batch_size: Global batch size.
            n_devices: Number of devices (for reshaping).
            start_step: Skip this many batches (for checkpoint resume).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_devices = n_devices
        self.start_step = start_step
        self.seq_len = dataset.seq_len

        assert batch_size % n_devices == 0, (
            f"batch_size ({batch_size}) must be divisible by n_devices ({n_devices})"
        )
        self.per_device_batch = batch_size // n_devices

        self._num_batches = len(dataset) // batch_size

    def __len__(self):
        return max(0, self._num_batches - self.start_step)

    def __iter__(self):
        """Yields (input_ids, attention_mask) tuples.

        Shapes:
          - n_devices == 1: (batch_size, seq_len)
          - n_devices > 1:  (n_devices, per_device_batch, seq_len)
        """
        for batch_idx in range(self._num_batches):
            # Skip batches for checkpoint resume
            if batch_idx < self.start_step:
                continue

            start = batch_idx * self.batch_size
            batch = self.dataset.get_batch(start, self.batch_size)
            if batch is None or len(batch) < self.batch_size:
                return  # end of data

            input_ids = jnp.array(batch)
            # Pretokenized + packed → no padding → all ones
            attention_mask = jnp.ones_like(input_ids)

            if self.n_devices > 1:
                input_ids = input_ids.reshape(
                    self.n_devices, self.per_device_batch, self.seq_len
                )
                attention_mask = attention_mask.reshape(
                    self.n_devices, self.per_device_batch, self.seq_len
                )

            yield input_ids, attention_mask


# ============================================================
# load_data: drop-in replacement for utils.data.load_data
# ============================================================

def load_data(data_config, max_length=512, batch_size=128, n_devices=1, start_step=0):
    """Load pretokenized .bin data for JAX training.

    Drop-in compatible interface for train_jax.py.

    Args:
        data_config: Dict with keys:
            - bin_train: Path to training .bin file (local or gs://)
            - bin_val: Path to validation .bin file (local or gs://)
            - max_train_tokens: Optional token limit for training.
            - max_val_tokens: Optional token limit for validation (default 10M).
            ---- Legacy support (ignored if bin_train/bin_val present) ----
            - base_dir, train_files, val_files: Old .pt-based config (not supported)
        max_length: Sequence length (default 512).
        batch_size: Global batch size.
        n_devices: Number of devices for multi-device reshaping.
        start_step: Resume from this step (skip batches).

    Returns:
        (train_loader, val_loader, vocab_size)
        - train_loader: BinDataLoader yielding (input_ids, attention_mask)
        - val_loader: BinDataLoader yielding (input_ids, attention_mask)
        - vocab_size: int (30522 for bert-base-uncased)
    """
    seq_len = max_length

    # ---- Resolve paths ----
    train_path = data_config.get("bin_train")
    val_path = data_config.get("bin_val")

    if train_path is None or val_path is None:
        raise ValueError(
            "data_config must have 'bin_train' and 'bin_val' keys "
            "pointing to pretokenized .bin files. "
            "Run scripts/pretokenize_c4.py first."
        )

    # ---- Token limits ----
    max_train_tokens = data_config.get("max_train_tokens", None)
    max_val_tokens = data_config.get("max_val_tokens", 10_000_000)

    max_train_seqs = None
    if max_train_tokens is not None:
        max_train_seqs = max_train_tokens // seq_len

    max_val_seqs = max_val_tokens // seq_len if max_val_tokens else None

    # ---- Load datasets ----
    print(f"\nLoading pretokenized data (seq_len={seq_len})...")
    train_dataset = BinDataset(train_path, seq_len, max_sequences=max_train_seqs)
    val_dataset = BinDataset(val_path, seq_len, max_sequences=max_val_seqs)

    # ---- Create loaders ----
    train_loader = BinDataLoader(
        train_dataset,
        batch_size=batch_size,
        n_devices=n_devices,
        start_step=start_step,
    )
    val_loader = BinDataLoader(
        val_dataset,
        batch_size=batch_size,
        n_devices=n_devices,
        start_step=0,
    )

    # Vocab size: bert-base-uncased = 30522. Read from metadata if available.
    vocab_size = 30522
    meta_path = train_path.replace(".bin", "_meta.json")
    try:
        meta = _read_json(meta_path)
        vocab_size = meta.get("vocab_size", 30522)
    except Exception:
        pass

    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Devices: {n_devices}")
    if n_devices > 1:
        print(f"  Per-device batch: {batch_size // n_devices}")

    return train_loader, val_loader, vocab_size

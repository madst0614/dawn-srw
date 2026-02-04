"""
Pure numpy/JAX data loader for DAWN training on TPU.

Reads pretokenized .bin files (uint16, packed sequences) produced by
pretokenize_c4.py. No PyTorch dependency.

Supports:
- Single .bin file or sharded .bin files (via _meta.json)
- Local files and GCS paths (gs://...)
- GCS → local SSD cache for memmap (local_cache_dir)
- Checkpoint resume via global_step offset
- Multi-device batching: (n_devices, per_device_batch, seq_len)
- reset() for epoch boundary without re-reading files
"""

import json
import os
import shutil
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


def _read_bin_local(path: str) -> np.ndarray:
    """Memory-map a local uint16 binary file."""
    return np.memmap(path, dtype=np.uint16, mode='r')


def _read_bin_gcs(path: str) -> np.ndarray:
    """Read a GCS uint16 binary file fully into memory."""
    fs = _get_gcs_fs()
    if hasattr(fs, 'open'):
        with fs.open(path, 'rb') as f:
            data = f.read()
    elif hasattr(fs, 'GFile'):
        with fs.GFile(path, 'rb') as f:
            data = f.read()
    else:
        raise RuntimeError(f"Cannot read GCS path: {path}")
    return np.frombuffer(data, dtype=np.uint16).copy()


def _copy_gcs_to_local(gcs_path: str, local_path: str):
    """Download a GCS file to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        # Already cached
        gcs_size = _gcs_file_size(gcs_path)
        local_size = os.path.getsize(local_path)
        if gcs_size is not None and local_size == gcs_size:
            print(f"    Cache hit: {local_path} ({local_size / 1e9:.2f} GB)")
            return
        print(f"    Cache size mismatch, re-downloading: {local_path}")

    print(f"    Downloading: {gcs_path} -> {local_path}")
    fs = _get_gcs_fs()
    if hasattr(fs, 'get'):
        fs.get(gcs_path, local_path)
    elif hasattr(fs, 'GFile'):
        import tensorflow as tf
        tf.io.gfile.copy(gcs_path, local_path, overwrite=True)
    else:
        raise RuntimeError(f"Cannot copy from GCS: {gcs_path}")
    print(f"    Downloaded: {os.path.getsize(local_path) / 1e9:.2f} GB")


def _gcs_file_size(path: str):
    """Get file size on GCS, or None on failure."""
    try:
        fs = _get_gcs_fs()
        if hasattr(fs, 'info'):
            return fs.info(path).get('size', None)
        elif hasattr(fs, 'GFile'):
            import tensorflow as tf
            return tf.io.gfile.stat(path).length
    except Exception:
        return None


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


def _gcs_path_to_local(gcs_path: str, cache_dir: str) -> str:
    """Convert gs://bucket/path/file.bin -> cache_dir/bucket/path/file.bin"""
    stripped = gcs_path.replace("gs://", "")
    return os.path.join(cache_dir, stripped)


# ============================================================
# BinDataset: single .bin file
# ============================================================

class BinDataset:
    """Dataset from a single pretokenized .bin file.

    Uses memmap for local files, full load for GCS (unless cached locally).
    """

    def __init__(self, bin_path: str, seq_len: int = 512,
                 max_sequences: int = None, local_cache_dir: str = None):
        self.bin_path = bin_path
        self.seq_len = seq_len

        # Resolve path: GCS with local cache -> download then memmap
        if is_gcs_path(bin_path) and local_cache_dir:
            local_path = _gcs_path_to_local(bin_path, local_cache_dir)
            _copy_gcs_to_local(bin_path, local_path)
            raw = _read_bin_local(local_path)
        elif is_gcs_path(bin_path):
            raw = _read_bin_gcs(bin_path)
        else:
            raw = _read_bin_local(bin_path)

        total_tokens = len(raw)
        n_seqs = total_tokens // seq_len
        if n_seqs == 0:
            raise ValueError(
                f"Not enough tokens in {bin_path}: {total_tokens} < seq_len={seq_len}"
            )

        used_tokens = n_seqs * seq_len
        self.data = raw[:used_tokens].reshape(n_seqs, seq_len)

        if max_sequences is not None and len(self.data) > max_sequences:
            self.data = self.data[:max_sequences]

        print(f"  BinDataset: {bin_path}")
        print(f"    {len(self.data):,} sequences x {seq_len} = {len(self.data) * seq_len:,} tokens")

    def __len__(self):
        return len(self.data)

    def get_batch(self, start_idx: int, batch_size: int) -> np.ndarray:
        end_idx = min(start_idx + batch_size, len(self.data))
        actual_size = end_idx - start_idx
        if actual_size <= 0:
            return None
        return self.data[start_idx:end_idx].astype(np.int32)


# ============================================================
# ShardedBinDataset: multiple .bin shard files
# ============================================================

class ShardedBinDataset:
    """Dataset from multiple sharded .bin files.

    Only loads one shard at a time to limit memory usage.
    Shard info is read from the _meta.json produced by pretokenize_c4.py.
    """

    def __init__(self, meta_path: str, seq_len: int = 512,
                 max_sequences: int = None, local_cache_dir: str = None):
        self.seq_len = seq_len
        self.local_cache_dir = local_cache_dir

        meta = _read_json(meta_path)
        self.shard_infos = meta['shards']  # list of {path, sequences, tokens}
        self.vocab_size = meta.get('vocab_size', 30522)

        # Build cumulative sequence counts for shard lookup
        self._shard_seq_counts = [s['sequences'] for s in self.shard_infos]
        self._shard_cum_seqs = []
        cum = 0
        for count in self._shard_seq_counts:
            self._shard_cum_seqs.append(cum)
            cum += count
        self._total_seqs = cum

        if max_sequences is not None and self._total_seqs > max_sequences:
            self._total_seqs = max_sequences

        # Current shard state
        self._current_shard_idx = -1
        self._current_data = None

        print(f"  ShardedBinDataset: {meta_path}")
        print(f"    {len(self.shard_infos)} shards, {self._total_seqs:,} sequences, "
              f"{self._total_seqs * seq_len:,} tokens")

    def _load_shard(self, shard_idx: int):
        """Load a specific shard into memory."""
        if shard_idx == self._current_shard_idx:
            return  # already loaded

        info = self.shard_infos[shard_idx]
        path = info['path']

        if is_gcs_path(path) and self.local_cache_dir:
            local_path = _gcs_path_to_local(path, self.local_cache_dir)
            _copy_gcs_to_local(path, local_path)
            raw = _read_bin_local(local_path)
        elif is_gcs_path(path):
            raw = _read_bin_gcs(path)
        else:
            raw = _read_bin_local(path)

        n_seqs = len(raw) // self.seq_len
        self._current_data = raw[:n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)
        self._current_shard_idx = shard_idx

    def _find_shard(self, global_seq_idx: int):
        """Find which shard contains a given global sequence index."""
        for i in range(len(self.shard_infos)):
            shard_start = self._shard_cum_seqs[i]
            shard_end = shard_start + self._shard_seq_counts[i]
            if shard_start <= global_seq_idx < shard_end:
                return i, global_seq_idx - shard_start
        return None, None

    def __len__(self):
        return self._total_seqs

    def get_batch(self, start_idx: int, batch_size: int) -> np.ndarray:
        """Get a batch, potentially spanning shard boundaries."""
        if start_idx >= self._total_seqs:
            return None
        end_idx = min(start_idx + batch_size, self._total_seqs)
        actual_size = end_idx - start_idx
        if actual_size <= 0:
            return None

        result_parts = []
        remaining = actual_size
        current_global = start_idx

        while remaining > 0:
            shard_idx, local_idx = self._find_shard(current_global)
            if shard_idx is None:
                break

            self._load_shard(shard_idx)
            shard_len = len(self._current_data)
            available = shard_len - local_idx
            take = min(remaining, available)

            result_parts.append(
                self._current_data[local_idx:local_idx + take].astype(np.int32)
            )
            remaining -= take
            current_global += take

        if not result_parts:
            return None

        return np.concatenate(result_parts, axis=0)


# ============================================================
# BinDataLoader: sequential batch iterator with reset()
# ============================================================

class BinDataLoader:
    """Sequential batch iterator over a BinDataset or ShardedBinDataset.

    Yields (input_ids, attention_mask) as jnp.arrays.
    Supports resume from a given step and reset() for epoch boundaries.
    """

    def __init__(self, dataset, batch_size: int,
                 n_devices: int = 1, start_step: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_devices = n_devices
        self._start_step = start_step
        self.seq_len = dataset.seq_len

        assert batch_size % n_devices == 0, (
            f"batch_size ({batch_size}) must be divisible by n_devices ({n_devices})"
        )
        self.per_device_batch = batch_size // n_devices
        self._num_batches = len(dataset) // batch_size

    def reset(self, start_step: int = 0):
        """Reset iterator position without re-reading data files."""
        self._start_step = start_step

    def __len__(self):
        return max(0, self._num_batches - self._start_step)

    def __iter__(self):
        """Yields (input_ids, attention_mask) tuples.

        Shapes:
          - n_devices == 1: (batch_size, seq_len)
          - n_devices > 1:  (n_devices, per_device_batch, seq_len)
        """
        for batch_idx in range(self._num_batches):
            if batch_idx < self._start_step:
                continue

            start = batch_idx * self.batch_size
            batch = self.dataset.get_batch(start, self.batch_size)
            if batch is None or len(batch) < self.batch_size:
                return

            input_ids = jnp.array(batch)
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
# load_data: main entry point
# ============================================================

def load_data(data_config, max_length=512, batch_size=128,
              n_devices=1, start_step=0):
    """Load pretokenized .bin data for JAX training.

    Supports three modes via data_config:

    1. Single .bin file:
       data:
         bin_train: path/to/c4_train.bin
         bin_val: path/to/c4_val.bin

    2. Sharded .bin files (auto-detected from _meta.json):
       data:
         bin_train: path/to/c4_train   # base name — reads c4_train_meta.json
         bin_val: path/to/c4_val.bin

    3. With local cache (GCS → local SSD for memmap):
       data:
         bin_train: gs://bucket/data/c4_train
         bin_val: gs://bucket/data/c4_val.bin
         local_cache_dir: /tmp/data

    Args:
        data_config: Dict with bin_train, bin_val, and optional fields.
        max_length: Sequence length (default 512).
        batch_size: Global batch size.
        n_devices: Number of devices for multi-device reshaping.
        start_step: Resume from this step (skip batches).

    Returns:
        (train_loader, val_loader, vocab_size)
    """
    seq_len = max_length

    train_path = data_config.get("bin_train")
    val_path = data_config.get("bin_val")
    local_cache_dir = data_config.get("local_cache_dir", None)

    if train_path is None or val_path is None:
        raise ValueError(
            "data_config must have 'bin_train' and 'bin_val' keys. "
            "Run scripts/pretokenize_c4.py first."
        )

    max_train_tokens = data_config.get("max_train_tokens", None)
    max_val_tokens = data_config.get("max_val_tokens", 10_000_000)
    max_train_seqs = max_train_tokens // seq_len if max_train_tokens else None
    max_val_seqs = max_val_tokens // seq_len if max_val_tokens else None

    print(f"\nLoading pretokenized data (seq_len={seq_len})...")

    # ---- Build datasets ----
    train_dataset = _build_dataset(
        train_path, seq_len, max_train_seqs, local_cache_dir)
    val_dataset = _build_dataset(
        val_path, seq_len, max_val_seqs, local_cache_dir)

    # ---- Create loaders ----
    train_loader = BinDataLoader(
        train_dataset, batch_size=batch_size,
        n_devices=n_devices, start_step=start_step)
    val_loader = BinDataLoader(
        val_dataset, batch_size=batch_size,
        n_devices=n_devices, start_step=0)

    # Vocab size
    vocab_size = 30522
    if isinstance(train_dataset, ShardedBinDataset):
        vocab_size = train_dataset.vocab_size
    else:
        meta_path = _meta_path_for(train_path)
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
    if local_cache_dir:
        print(f"  Local cache: {local_cache_dir}")

    return train_loader, val_loader, vocab_size


def _meta_path_for(path: str) -> str:
    """Derive meta.json path from a .bin path or base name."""
    base = path
    if base.endswith(".bin"):
        base = base[:-4]
    return base + "_meta.json"


def _meta_exists(path: str) -> bool:
    """Check if a _meta.json exists for the given path."""
    meta_path = _meta_path_for(path)
    if is_gcs_path(meta_path):
        try:
            fs = _get_gcs_fs()
            if hasattr(fs, 'exists'):
                return fs.exists(meta_path)
            elif hasattr(fs, 'GFile'):
                import tensorflow as tf
                return tf.io.gfile.exists(meta_path)
        except Exception:
            return False
    return os.path.exists(meta_path)


def _is_sharded(path: str) -> bool:
    """Detect if a path refers to sharded data (has _meta.json with num_shards > 1)."""
    meta_path = _meta_path_for(path)
    try:
        meta = _read_json(meta_path)
        return meta.get("num_shards", 1) > 1
    except Exception:
        return False


def _build_dataset(path, seq_len, max_sequences, local_cache_dir):
    """Build the appropriate dataset type based on path and metadata."""
    if _is_sharded(path):
        meta_path = _meta_path_for(path)
        return ShardedBinDataset(
            meta_path, seq_len,
            max_sequences=max_sequences,
            local_cache_dir=local_cache_dir)
    else:
        # Single .bin file
        bin_path = path if path.endswith(".bin") else path + ".bin"
        return BinDataset(
            bin_path, seq_len,
            max_sequences=max_sequences,
            local_cache_dir=local_cache_dir)

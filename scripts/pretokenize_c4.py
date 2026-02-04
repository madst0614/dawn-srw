"""
Pretokenize C4 dataset for DAWN TPU training.

Streams C4 from HuggingFace, tokenizes with bert-base-uncased,
packs into fixed-length sequences (no padding), and saves as
flat numpy binary (.bin) with uint16 dtype.

Designed for TPU training pipeline — outputs are consumed by
utils/data_jax.py (pure numpy/JAX data loader, no PyTorch).

Usage:
    python scripts/pretokenize_c4.py --split train --num_tokens 20_000_000_000 --output gs://my-bucket/data/c4_train.bin
    python scripts/pretokenize_c4.py --split validation --num_tokens 10_000_000 --output ./data/c4_val.bin
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def open_output(path: str, mode: str = "wb"):
    """Open a file for writing, supporting both local and GCS paths."""
    if is_gcs_path(path):
        try:
            from google.cloud import storage
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            return fs.open(path, mode)
        except ImportError:
            try:
                import tensorflow as tf
                return tf.io.gfile.GFile(path, mode)
            except ImportError:
                raise ImportError(
                    "GCS output requires 'gcsfs' or 'tensorflow'. "
                    "Install with: pip install gcsfs  OR  pip install tensorflow"
                )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return open(path, mode)


def write_json(path: str, data: dict):
    """Write JSON metadata, supporting both local and GCS paths."""
    content = json.dumps(data, indent=2)
    if is_gcs_path(path):
        with open_output(path, "w") as f:
            f.write(content)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Pretokenize C4 for DAWN training")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation"],
                        help="Dataset split to process")
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Target number of tokens (e.g. 20_000_000_000 for 20B)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for .bin file (local or gs://...)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length for packing (default: 512)")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="HuggingFace tokenizer name (default: bert-base-uncased)")
    parser.add_argument("--flush_every", type=int, default=100_000,
                        help="Flush buffer to disk every N sequences (default: 100000)")
    args = parser.parse_args()

    seq_len = args.seq_len
    target_tokens = args.num_tokens
    target_seqs = target_tokens // seq_len

    print(f"{'='*60}")
    print(f"Pretokenize C4")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Target sequences: {target_seqs:,} (seq_len={seq_len})")
    print(f"Output: {args.output}")
    print(f"Tokenizer: {args.tokenizer}")

    # ---- Load tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    assert vocab_size <= 65535, f"Vocab size {vocab_size} exceeds uint16 max (65535)"

    # ---- Stream C4 dataset ----
    from datasets import load_dataset
    print(f"\nLoading C4 ({args.split}) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split=args.split, streaming=True)

    # ---- Tokenize and pack ----
    print("Starting tokenization and packing...")

    output_path = args.output
    if not output_path.endswith(".bin"):
        output_path += ".bin"
    meta_path = output_path.replace(".bin", "_meta.json")

    # Token buffer for packing: accumulate tokens, then slice into seq_len chunks
    token_buffer = []
    total_tokens_written = 0
    total_seqs_written = 0
    docs_processed = 0
    start_time = time.time()
    last_log_tokens = 0

    # Write sequences in chunks to avoid holding everything in memory
    flush_every = args.flush_every  # sequences
    pending_seqs = []

    f_out = open_output(output_path, "wb")

    try:
        for example in dataset:
            text = example.get("text", "")
            if not text or len(text.strip()) == 0:
                continue

            # Tokenize without special tokens (we're packing, no [CLS]/[SEP])
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            token_buffer.extend(token_ids)
            docs_processed += 1

            # Slice complete sequences from buffer
            while len(token_buffer) >= seq_len:
                seq = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]

                pending_seqs.append(np.array(seq, dtype=np.uint16))
                total_seqs_written += 1
                total_tokens_written += seq_len

                # Flush to disk periodically
                if len(pending_seqs) >= flush_every:
                    chunk = np.stack(pending_seqs)
                    f_out.write(chunk.tobytes())
                    pending_seqs = []

                # Progress logging every 1M tokens
                if total_tokens_written - last_log_tokens >= 1_000_000:
                    elapsed = time.time() - start_time
                    tok_per_sec = total_tokens_written / elapsed if elapsed > 0 else 0
                    progress = total_tokens_written / target_tokens * 100
                    print(
                        f"  [{progress:6.2f}%] {total_tokens_written/1e6:.1f}M tokens, "
                        f"{total_seqs_written:,} seqs, {docs_processed:,} docs, "
                        f"{tok_per_sec/1e6:.2f}M tok/s, "
                        f"elapsed={elapsed:.0f}s"
                    )
                    last_log_tokens = total_tokens_written

                # Check if we've reached the target
                if total_tokens_written >= target_tokens:
                    break

            if total_tokens_written >= target_tokens:
                break

        # Flush remaining sequences
        if pending_seqs:
            chunk = np.stack(pending_seqs)
            f_out.write(chunk.tobytes())
            pending_seqs = []

    finally:
        f_out.close()

    elapsed = time.time() - start_time

    # ---- Write metadata ----
    metadata = {
        "total_tokens": total_tokens_written,
        "total_sequences": total_seqs_written,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer": args.tokenizer,
        "dtype": "uint16",
        "split": args.split,
        "docs_processed": docs_processed,
        "discarded_tail_tokens": len(token_buffer),
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(total_tokens_written / elapsed, 1) if elapsed > 0 else 0,
    }
    write_json(meta_path, metadata)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    print(f"  Tokens written: {total_tokens_written:,}")
    print(f"  Sequences: {total_seqs_written:,}")
    print(f"  Docs processed: {docs_processed:,}")
    print(f"  Discarded tail: {len(token_buffer)} tokens")
    print(f"  Output: {output_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  File size: ~{total_seqs_written * seq_len * 2 / 1e9:.2f} GB")
    print(f"  Elapsed: {elapsed:.0f}s ({total_tokens_written / elapsed / 1e6:.2f}M tok/s)")


if __name__ == "__main__":
    main()

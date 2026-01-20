"""
Convert LMDB embeddings from pickled numpy format to raw float32 bytes.

This creates a Java-compatible LMDB database where values are raw float32 arrays.

Usage:
    python -m delft.utilities.convert_lmdb_embeddings \
        --input data/db/glove-840B \
        --output data/db/glove-840B-raw
"""

import argparse
import os
import pickle

import lmdb
import numpy as np
from tqdm import tqdm


def convert_lmdb(input_path: str, output_path: str, batch_size: int = 10000):
    """
    Convert LMDB from pickled numpy arrays to raw float32 bytes.

    Args:
        input_path: Path to source LMDB (with pickled numpy)
        output_path: Path to destination LMDB (with raw float32)
        batch_size: Number of entries to write per transaction
    """
    # Open source database
    src_env = lmdb.open(input_path, readonly=True, max_dbs=1, lock=False)

    # Get stats
    with src_env.begin() as txn:
        stats = txn.stat()
        total_entries = stats["entries"]

    print(f"Source database: {input_path}")
    print(f"  Total entries: {total_entries}")

    # Check one entry to get embedding size
    with src_env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        key, value = cursor.item()
        arr = pickle.loads(value)
        embed_size = arr.shape[0]
        print(f"  Embedding size: {embed_size}")
        print(f"  Source format: pickled numpy {arr.dtype}")

    # Create destination database
    os.makedirs(output_path, exist_ok=True)

    # Calculate map size (word bytes + embed floats) per entry, with overhead
    map_size = total_entries * (100 + embed_size * 4) * 2  # 2x overhead
    map_size = max(map_size, 10 * 1024 * 1024 * 1024)  # At least 10GB

    dst_env = lmdb.open(output_path, map_size=map_size, max_dbs=1)

    print(f"\nDestination database: {output_path}")
    print(f"  Map size: {map_size / (1024**3):.1f} GB")
    print(f"  Target format: raw float32 ({embed_size * 4} bytes per vector)")

    # Convert entries
    print(f"\nConverting {total_entries} entries...")

    count = 0
    batch = []

    with src_env.begin() as src_txn:
        cursor = src_txn.cursor()

        for key, value in tqdm(cursor, total=total_entries, desc="Converting"):
            # Unpickle numpy array
            arr = pickle.loads(value)

            # Convert to float32 bytes
            raw_bytes = arr.astype(np.float32).tobytes()

            batch.append((key, raw_bytes))

            if len(batch) >= batch_size:
                # Write batch
                with dst_env.begin(write=True) as dst_txn:
                    for k, v in batch:
                        dst_txn.put(k, v)
                count += len(batch)
                batch = []

    # Write remaining
    if batch:
        with dst_env.begin(write=True) as dst_txn:
            for k, v in batch:
                dst_txn.put(k, v)
        count += len(batch)

    # Close databases
    src_env.close()
    dst_env.close()

    print("\nConversion complete!")
    print(f"  Entries converted: {count}")

    # Verify
    print("\nVerifying...")
    verify_env = lmdb.open(output_path, readonly=True)
    with verify_env.begin() as txn:
        # Check a known word
        test_words = [b"the", b"April", b"2001", b"January"]
        for word in test_words:
            value = txn.get(word)
            if value:
                arr = np.frombuffer(value, dtype=np.float32)
                print(
                    f"  '{word.decode()}': shape={arr.shape}, first 3=[{arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}]"
                )
    verify_env.close()

    print("\nDone! Java can now read this database as raw float32 bytes.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LMDB embeddings from pickled numpy to raw float32"
    )
    parser.add_argument("--input", required=True, help="Path to source LMDB database")
    parser.add_argument(
        "--output", required=True, help="Path to destination LMDB database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for writing (default: 10000)",
    )

    args = parser.parse_args()

    convert_lmdb(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()

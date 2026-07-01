"""Dataset access shared by the training and inference apps.

Keeping split_holdout in one shared function is what makes the contamination
guarantee real; train and eval must not compute the boundary independently.
"""

import glob
import os

from ttt_config import DATASET_SOURCE, HOLDOUT_LAST_N


def open_dataset():
    """Load the parquet dataset from the HF Hub; a local dir path also works."""
    from datasets import load_dataset, load_from_disk

    src = DATASET_SOURCE
    if "CHANGE_ME" in src:
        raise RuntimeError(
            "Set DATASET_SOURCE in ttt_config.py to your HF repo id."
        )
    if os.path.isdir(src):
        shards = sorted(glob.glob(os.path.join(src, "**", "*.parquet"),
                                  recursive=True))
        if shards:
            print(f"loading {len(shards)} parquet shard(s) from {src}")
            return load_dataset("parquet", data_files=shards, split="train")
        print(f"loading arrow dataset from {src}")
        return load_from_disk(src)
    print(f"loading from HF Hub repo {src}")
    return load_dataset(src, split="train")


def split_holdout(ds):
    """Train = everything except the newest HOLDOUT_LAST_N papers (reserved for eval).

    Slow weights must never train on the papers used to measure fast weight memory.
    """
    cut = len(ds) - HOLDOUT_LAST_N
    return ds.select(range(cut)), ds.select(range(cut, len(ds)))

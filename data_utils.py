"""
Dataset access shared by the training and inference apps (DRY).

open_dataset() auto-detects the storage layout, split_holdout() defines
the single train/eval boundary used everywhere. Keeping the holdout
split in one shared function is what makes the contamination guarantee
real; if train and eval computed it independently they could drift.
"""

import glob
import os

from ttt_config import DATASET_SOURCE, HOLDOUT_LAST_N


def open_dataset():
    """Load the parquet dataset from the HF Hub (downloads cache to
    HF_HOME on the cache volume, so the Hub is only hit once). A local
    directory path still works as a fallback for ad-hoc experiments."""
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
    """Train = everything except the newest HOLDOUT_LAST_N papers, which
    are reserved for contamination-free session evaluation. The slow
    weights (LoRA, W_down, W_target, Conv1D) must never have trained on
    the papers used to measure fast weight memory."""
    cut = len(ds) - HOLDOUT_LAST_N
    return ds.select(range(cut)), ds.select(range(cut, len(ds)))

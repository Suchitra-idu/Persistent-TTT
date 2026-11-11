"""HfDataSource — the Hub, or a local directory of parquet/arrow shards."""

from __future__ import annotations

import glob
import os
import sys
import time

from ttt.adapters.hf_table import HfTable
from ttt.core.config.dataset import DatasetSpec
from ttt.ports.table import SOURCE_COLUMN


class HfDataSource:
    """Caches per instance: a run loads the same spec twice, for training and the holdout."""

    def __init__(self) -> None:
        self._cache: dict[DatasetSpec, HfTable] = {}

    def load(self, spec: DatasetSpec) -> HfTable:
        cached = self._cache.get(spec)
        if cached is not None:
            return cached
        print(f"loading {spec.source!r}...")
        t0 = time.time()
        dataset = _load(spec)
        print(f"  loaded {len(dataset):,d} rows in {time.time() - t0:.1f}s")
        if spec.source_meta_column not in dataset.column_names:
            raise ValueError(
                f"dataset {spec.source!r} has no {spec.source_meta_column!r} "
                f"column to read a source label from (columns: "
                f"{dataset.column_names})"
            )
        t0 = time.time()
        labels = source_labels(dataset, spec)
        print(f"  labeled sources in {time.time() - t0:.1f}s")
        table = HfTable(dataset.add_column(SOURCE_COLUMN, labels))
        self._cache[spec] = table
        return table


def source_labels(dataset, spec: DatasetSpec) -> list[str]:
    """Read a chunk at a time and appended as one column: `dataset.map` would
    rewrite every row, tens of GB on SlimPajama-6B."""
    labels: list[str] = []
    for chunk in dataset.data.column(spec.source_meta_column).chunks:
        labels.extend(
            sys.intern(spec.source_of({spec.source_meta_column: value}))
            for value in chunk.to_pylist()
        )
    return labels


def _load(spec: DatasetSpec):
    from datasets import load_dataset, load_from_disk

    if not os.path.isdir(spec.source):
        return load_dataset(spec.source, split="train")
    shards = sorted(glob.glob(os.path.join(spec.source, "**", "*.parquet"), recursive=True))
    if shards:
        return load_dataset("parquet", data_files=shards, split="train")
    return load_from_disk(spec.source)

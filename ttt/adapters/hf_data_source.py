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
    """`load` is called twice per run — the training pool, then `holdout`'s own
    pool — on the same spec; caching per instance turns the second call from a
    full re-load-and-relabel of the raw corpus into a dict lookup. Scoped to
    the instance (not module-level) so it dies with the entrypoint that made
    it, not leak across separate Modal invocations."""

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
        missing = spec.source_meta_column and (
            spec.source_meta_column not in dataset.column_names
        )
        if missing:
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
    """The `source` column's values, read a chunk at a time.

    Deliberately not `dataset.map`: that rewrites every column of every row to
    append one string, which on SlimPajama-6B is tens of GB of writes to a
    network-backed volume. Appending one array leaves the rest memory-mapped.
    Interned because a 6M-row corpus holds about six distinct labels.
    """
    if spec.source_meta_column is None:
        return [spec.constant_source] * len(dataset)
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

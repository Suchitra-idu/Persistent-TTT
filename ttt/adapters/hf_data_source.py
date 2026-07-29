"""HfDataSource — the Hub, or a local directory of parquet/arrow shards."""

from __future__ import annotations

import glob
import os

from ttt.adapters.hf_table import HfTable
from ttt.core.config.dataset import DatasetSpec
from ttt.ports.table import SOURCE_COLUMN


class HfDataSource:
    def load(self, spec: DatasetSpec) -> HfTable:
        dataset = _load(spec)
        missing = spec.source_meta_column and (
            spec.source_meta_column not in dataset.column_names
        )
        if missing:
            raise ValueError(
                f"dataset {spec.source!r} has no {spec.source_meta_column!r} "
                f"column to read a source label from (columns: "
                f"{dataset.column_names})"
            )
        # `map`, not a Python list: SlimPajama-6B does not fit in memory.
        dataset = dataset.map(
            lambda row: {SOURCE_COLUMN: spec.source_of(row)},
            desc="labelling source",
        )
        return HfTable(dataset)


def _load(spec: DatasetSpec):
    from datasets import load_dataset, load_from_disk

    if not os.path.isdir(spec.source):
        return load_dataset(spec.source, split="train")
    shards = sorted(glob.glob(os.path.join(spec.source, "**", "*.parquet"), recursive=True))
    if shards:
        return load_dataset("parquet", data_files=shards, split="train")
    return load_from_disk(spec.source)

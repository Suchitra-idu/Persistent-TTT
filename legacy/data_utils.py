"""Dataset access shared by the training and inference apps.

Everything goes through DATASET_SPEC (chosen by TTT_DATASET). Callers may
pass an explicit spec for tests or ablations; the default is the active
spec so day-to-day code stays terse.

Keeping split_holdout in one shared function is what makes the
contamination guarantee real; train and eval must not compute the
boundary independently.
"""

import glob
import os

from ttt_config import DATASET_SPEC, DatasetSpec


def open_dataset(spec: DatasetSpec | None = None):
    """Load the dataset for `spec` (default DATASET_SPEC), then attach a
    top-level `source` column if the spec declares one. A local dir of
    parquet/arrow shards also works as `spec.source`."""
    from datasets import load_dataset, load_from_disk

    spec = spec or DATASET_SPEC
    src = spec.source
    if os.path.isdir(src):
        shards = sorted(glob.glob(os.path.join(src, "**", "*.parquet"),
                                  recursive=True))
        if shards:
            print(f"loading {len(shards)} parquet shard(s) from {src}")
            ds = load_dataset("parquet", data_files=shards, split="train")
        else:
            print(f"loading arrow dataset from {src}")
            ds = load_from_disk(src)
    else:
        print(f"loading from HF Hub repo {src}")
        ds = load_dataset(src, split="train")
    return _annotate_source(ds, spec)


def extract_source_from_row(row: dict, spec: DatasetSpec) -> str:
    """Pure extractor: pull the source label out of one row given a spec.
    Handles both struct-typed and JSON-string meta fields. Returns "" if
    the spec has no source config or the row can't be parsed."""
    if not (spec.source_meta_column and spec.source_meta_key):
        return ""
    v = row.get(spec.source_meta_column)
    key = spec.source_meta_key
    if isinstance(v, dict):
        return str(v.get(key, ""))
    if isinstance(v, str):
        import json
        try:
            return str(json.loads(v).get(key, ""))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return ""
    return ""


def _annotate_source(ds, spec: DatasetSpec):
    """Add a `source` column by extracting `spec.source_meta_key` from
    `spec.source_meta_column`. No-op when the spec doesn't declare a
    source field."""
    if not (spec.source_meta_column and spec.source_meta_key):
        return ds
    if spec.source_meta_column not in ds.column_names:
        raise ValueError(
            f"dataset {spec.source!r} is missing declared "
            f"source_meta_column {spec.source_meta_column!r} "
            f"(columns: {ds.column_names})"
        )
    return ds.map(lambda ex: {"source": extract_source_from_row(ex, spec)},
                  desc="extracting source label")


def apply_source_filter(ds, spec: DatasetSpec | None = None):
    """Keep only rows whose `source` is in `spec.include_sources`. No-op
    if the spec has no include list. Requires a prior _annotate_source
    pass (open_dataset does this automatically)."""
    spec = spec or DATASET_SPEC
    if not spec.include_sources:
        return ds
    if "source" not in ds.column_names:
        raise ValueError(
            "apply_source_filter needs a 'source' column; the spec "
            "declares include_sources but no source_meta_column/key was "
            "extracted -- check open_dataset was called with this spec"
        )
    keep = frozenset(spec.include_sources)
    return ds.filter(lambda ex: ex["source"] in keep,
                     desc="filtering by include_sources")


def split_holdout(ds, spec: DatasetSpec | None = None):
    """Train = everything except the newest `spec.holdout_last_n` rows.
    Slow weights must never train on the eval-reserved suffix."""
    spec = spec or DATASET_SPEC
    cut = len(ds) - spec.holdout_last_n
    return ds.select(range(cut)), ds.select(range(cut, len(ds)))

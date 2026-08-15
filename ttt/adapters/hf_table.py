"""HfTable — the Table port over a `datasets.Dataset`, Arrow-backed and lazy."""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

MAX_FILTER_PROCS = 8
# Benchmarked against this pipeline's real table sizes, batched=True already
# in effect: single-process-batched beats num_proc=8 up to ~500k rows (process
# spawn cost dominates a filter that now finishes in a few hundred ms) and
# loses past ~1M. The pipeline's own tables cluster clearly on either side —
# ~15k-220k for drop_short/the holdout pool, ~3.7M-8M for filter_sources and
# prefilter — so this doesn't need to be precisely tuned, just on the right
# side of both clusters.
MIN_ROWS_TO_PARALLELIZE = 1_000_000
# batched=True amortises the Python<->Arrow row crossing that dominates a
# cheap predicate (e.g. a length check) over a big table; the predicate is
# still called once per row, just out of one Python list instead of one
# `.filter()` example callback per row.
FILTER_BATCH_SIZE = 1000


def filter_num_proc(n_rows: int) -> int | None:
    if n_rows < MIN_ROWS_TO_PARALLELIZE:
        return None
    return min(MAX_FILTER_PROCS, os.cpu_count() or 1)


class HfTable:
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._dataset.column_names)

    def column(self, name: str) -> list[Any]:
        if name not in self._dataset.column_names:
            raise KeyError(f"no column {name!r}; have {self._dataset.column_names}")
        # Safe because select/filter below guarantee `self._dataset` never
        # carries a pending indices mapping: going through `.data` directly
        # (skipping HF's own __getitem__ formatting) is the same result
        # ~15-60x faster on a multi-million-row column, but silently wrong —
        # returns the table in its *pre-selection* order/membership — the
        # moment that guarantee doesn't hold.
        return self._dataset.data.column(name).to_pylist()

    def row(self, index: int) -> dict[str, Any]:
        return dict(self._dataset[index])

    def select(self, indices: Sequence[int]) -> "HfTable":
        return HfTable(self._dataset.select(list(indices)).flatten_indices())

    def filter(self, name: str, predicate: Callable[[Any], bool]) -> "HfTable":
        if name not in self._dataset.column_names:
            raise KeyError(f"no column {name!r}; have {self._dataset.column_names}")
        return HfTable(
            self._dataset.filter(
                lambda batch: [predicate(value) for value in batch[name]],
                batched=True,
                batch_size=FILTER_BATCH_SIZE,
                desc=f"filter {name}",
                num_proc=filter_num_proc(len(self._dataset)),
            ).flatten_indices()
        )

    def with_column(self, name: str, values: Sequence[Any]) -> "HfTable":
        if len(values) != len(self._dataset):
            raise ValueError(
                f"column {name!r} has {len(values)} values for "
                f"{len(self._dataset)} rows"
            )
        dataset = self._dataset
        if name in dataset.column_names:
            dataset = dataset.remove_columns([name])
        return HfTable(dataset.add_column(name, list(values)))

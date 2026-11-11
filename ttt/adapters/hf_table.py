"""HfTable — the Table port over a `datasets.Dataset`, Arrow-backed and lazy."""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

MAX_FILTER_PROCS = 8
# Measured: below about 1M rows, process spawn costs more than it saves.
MIN_ROWS_TO_PARALLELIZE = 1_000_000
FILTER_BATCH_SIZE = 1000


def filter_num_proc(n_rows: int) -> int | None:
    if n_rows < MIN_ROWS_TO_PARALLELIZE:
        return None
    return min(MAX_FILTER_PROCS, os.cpu_count() or 1)


class HfTable:
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._dataset.column_names)

    def column(self, name: str) -> list[Any]:
        if name not in self._dataset.column_names:
            raise KeyError(f"no column {name!r}; have {self._dataset.column_names}")
        # Fast path; correct only because select/filter flatten their indices,
        # otherwise `.data` would return the pre-selection rows.
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

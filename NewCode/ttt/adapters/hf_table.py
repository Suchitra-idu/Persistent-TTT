"""HfTable — the Table port over a `datasets.Dataset`, Arrow-backed and lazy."""

from __future__ import annotations

from typing import Any, Sequence


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
        return list(self._dataset[name])

    def row(self, index: int) -> dict[str, Any]:
        return dict(self._dataset[index])

    def select(self, indices: Sequence[int]) -> "HfTable":
        return HfTable(self._dataset.select(list(indices)))

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

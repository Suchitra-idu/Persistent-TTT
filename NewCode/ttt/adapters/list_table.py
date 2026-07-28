"""ListTable — rows of dicts in memory. The whole pipeline runs on this."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class ListTable:
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        column_names: Sequence[str] | None = None,
    ) -> None:
        self._rows = tuple(dict(row) for row in rows)
        if column_names is None:
            column_names = tuple(self._rows[0]) if self._rows else ()
        self._column_names = tuple(column_names)

        expected = set(self._column_names)
        bad = [i for i, row in enumerate(self._rows) if set(row) != expected]
        if bad:
            raise ValueError(
                f"row {bad[0]} has columns {sorted(self._rows[bad[0]])}, table "
                f"declares {sorted(expected)}; a ragged table makes `column` lie"
            )

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._column_names

    def column(self, name: str) -> list[Any]:
        if name not in self._column_names:
            raise KeyError(f"no column {name!r}; have {list(self._column_names)}")
        return [row[name] for row in self._rows]

    def row(self, index: int) -> dict[str, Any]:
        return dict(self._rows[index])

    def select(self, indices: Sequence[int]) -> "ListTable":
        return ListTable([self._rows[i] for i in indices], self._column_names)

    def with_column(self, name: str, values: Sequence[Any]) -> "ListTable":
        if len(values) != len(self._rows):
            raise ValueError(
                f"column {name!r} has {len(values)} values for {len(self._rows)} rows"
            )
        rows = [{**row, name: value} for row, value in zip(self._rows, values)]
        names = self._column_names
        if name not in names:
            names = names + (name,)
        return ListTable(rows, names)

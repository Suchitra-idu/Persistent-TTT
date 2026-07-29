"""Table — a columnar row store. Every method returns a new table."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, runtime_checkable

SOURCE_COLUMN = "source"


@runtime_checkable
class Table(Protocol):
    def __len__(self) -> int: ...

    @property
    def column_names(self) -> tuple[str, ...]: ...

    def column(self, name: str) -> list[Any]: ...

    def row(self, index: int) -> dict[str, Any]: ...

    def select(self, indices: Sequence[int]) -> "Table":
        """Reorders as well as filters: `select` is how shuffling happens."""

    def filter(self, name: str, predicate: Callable[[Any], bool]) -> "Table":
        """Row-wise, so a corpus larger than memory never materialises."""

    def with_column(self, name: str, values: Sequence[Any]) -> "Table":
        """Replaces an existing column of the same name."""

"""Table — a columnar row store. The pipeline's only view of a corpus.

Every method returns a new table; a stage never mutates its input, which is
what lets the pipeline log what each stage kept without re-reading anything.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

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

    def with_column(self, name: str, values: Sequence[Any]) -> "Table":
        """Replaces an existing column of the same name."""

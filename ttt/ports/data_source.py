"""DataSource — where a corpus comes from, and the one place D9 is honoured."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ttt.core.config.dataset import DatasetSpec
from ttt.ports.table import Table


@runtime_checkable
class DataSource(Protocol):
    def load(self, spec: DatasetSpec) -> Table:
        """Rows in corpus order, carrying `spec.text_column` and a non-empty
        `source` column on every row (D9)."""

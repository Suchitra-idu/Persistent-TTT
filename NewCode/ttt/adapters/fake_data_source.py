"""FakeDataSource — corpora written out in the test, labelled through the spec."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ttt.adapters.list_table import ListTable
from ttt.core.config.dataset import DatasetSpec
from ttt.ports.table import SOURCE_COLUMN


class FakeDataSource:
    def __init__(self, rows_by_spec: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._rows_by_spec = {name: list(rows) for name, rows in rows_by_spec.items()}

    def load(self, spec: DatasetSpec) -> ListTable:
        if spec.name not in self._rows_by_spec:
            raise KeyError(
                f"no rows scripted for dataset {spec.name!r}; "
                f"have {sorted(self._rows_by_spec)}"
            )
        rows = self._rows_by_spec[spec.name]
        labelled = [{**row, SOURCE_COLUMN: spec.source_of(row)} for row in rows]
        return ListTable(labelled)

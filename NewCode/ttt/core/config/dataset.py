"""DatasetSpec — how to pull rows out of one corpus. Frozen (D3).

Enforces D9: every row carries a source label, extracted or stamped. No
tokens_est_column — token counts are always estimated (`core.tokens`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    text_column: str = "text"
    source_meta_column: str | None = None
    source_meta_key: str | None = None
    constant_source: str | None = None
    include_sources: tuple[str, ...] | None = None
    default_source_preset: str | None = None
    holdout_last_n: int = 5000

    def __post_init__(self) -> None:
        extracts = bool(self.source_meta_column and self.source_meta_key)
        stamps = bool(self.constant_source)
        if extracts == stamps:
            raise ValueError(
                f"dataset spec {self.name!r} must label every row exactly one "
                "way: either source_meta_column + source_meta_key, or "
                "constant_source (D9)"
            )
        if self.holdout_last_n < 0:
            raise ValueError(f"holdout_last_n must be >= 0, got {self.holdout_last_n}")
        if self.include_sources is not None:
            object.__setattr__(self, "include_sources", tuple(self.include_sources))
            if not self.include_sources:
                raise ValueError(
                    f"dataset spec {self.name!r} has an empty include_sources; "
                    "use None to keep every source"
                )

    @property
    def is_multi_source(self) -> bool:
        return self.constant_source is None

    def source_of(self, row: Mapping[str, Any]) -> str:
        """Never empty. Handles struct and JSON-string meta columns, because
        parquet shards and the Hub disagree about which they return."""
        if self.constant_source is not None:
            return self.constant_source

        value = row.get(self.source_meta_column)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"dataset spec {self.name!r}: column "
                    f"{self.source_meta_column!r} is a string that is not JSON"
                ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(
                f"dataset spec {self.name!r}: row has no usable "
                f"{self.source_meta_column!r} field to read "
                f"{self.source_meta_key!r} from (got {type(value).__name__})"
            )
        label = value.get(self.source_meta_key)
        if not label:
            raise ValueError(
                f"dataset spec {self.name!r}: row's "
                f"{self.source_meta_column}.{self.source_meta_key} is empty; "
                "every row must carry a source label (D9)"
            )
        return str(label)

    def keeps(self, source_label: str) -> bool:
        return self.include_sources is None or source_label in self.include_sources

    def holdout_boundary(self, n_rows: int) -> int:
        """Shared by train and eval; computing it twice would break the
        contamination guarantee."""
        if n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {n_rows}")
        return max(0, n_rows - self.holdout_last_n)

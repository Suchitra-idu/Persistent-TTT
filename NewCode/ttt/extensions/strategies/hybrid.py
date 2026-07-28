"""Length-gated one-doc sessions: short docs whole, long docs sliced.

The short tail is what teaches the S_0 = 0 case; the sliced long docs are what
teach carry across a TBPTT boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ttt.core.report import CompositionRow
from ttt.core.schedule import derive_slice_count, slice_doc
from ttt.core.types import Session, WorkItem
from ttt.extensions.strategies._registry import SESSION, register


@dataclass(frozen=True)
class Hybrid:
    name: str = "hybrid"
    carry_scope: str = SESSION

    carry_min_tokens: int = 2100
    slice_min_tokens: int = 1000
    slices_min: int = 2
    slices_max: int = 6

    def __post_init__(self) -> None:
        if self.carry_min_tokens < self.slices_min * self.slice_min_tokens:
            raise ValueError(
                f"carry_min_tokens={self.carry_min_tokens} is below "
                f"slices_min * slice_min_tokens = "
                f"{self.slices_min * self.slice_min_tokens}; docs at the boundary "
                "would take the multi-slice path and silently collapse back to one"
            )

    def slices_for(self, doc_length: int) -> int:
        if doc_length < 1:
            raise ValueError(
                f"doc length must be >= 1, got {doc_length}; empty docs are "
                "dropped by the pipeline, not scheduled"
            )
        if doc_length < self.carry_min_tokens:
            return 1
        return derive_slice_count(
            doc_length, self.slice_min_tokens, self.slices_min, self.slices_max
        )

    def build(self, doc_lengths: Sequence[int], rng) -> tuple[Session, ...]:
        return tuple(
            Session(
                items=tuple(
                    WorkItem(doc_idx=int(doc_idx), start=start, end=end)
                    for start, end in slice_doc(
                        int(doc_lengths[doc_idx]),
                        self.slices_for(int(doc_lengths[doc_idx])),
                        self.slice_min_tokens,
                        rng,
                    )
                )
            )
            for doc_idx in rng.permutation(len(doc_lengths))
        )

    def count(self, doc_lengths: Sequence[int]) -> int:
        return sum(self.slices_for(int(length)) for length in doc_lengths)

    def compose(
        self, doc_lengths: Sequence[int], sources: Sequence[str]
    ) -> tuple[CompositionRow, ...]:
        no_carry: Counter[str] = Counter()
        carry: Counter[str] = Counter()
        items: Counter[str] = Counter()
        for length, source in zip(doc_lengths, sources, strict=True):
            k = self.slices_for(int(length))
            (no_carry if k == 1 else carry)[source] += 1
            items[source] += k
        return tuple(
            CompositionRow(
                source=source,
                no_carry_docs=no_carry[source],
                carry_docs=carry[source],
                items=items[source],
            )
            for source in sorted(items)
        )

    def describe(self) -> str:
        return (
            f"hybrid: one doc per session; < {self.carry_min_tokens} tokens whole, "
            f"else {self.slices_min}-{self.slices_max} slices of "
            f">= {self.slice_min_tokens} tokens"
        )


HYBRID = register(Hybrid())

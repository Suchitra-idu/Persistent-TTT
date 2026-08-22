"""Whole-doc sessions with a per-source carry that outlives them all.

This is what produces the trained per-source seeds the 5-regime eval measures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ttt.core.report import CompositionRow
from ttt.core.types import Session, WorkItem
from ttt.extensions.strategies._registry import SOURCE, register


@dataclass(frozen=True)
class Everlasting:
    name: str = "everlasting"
    carry_scope: str = SOURCE

    def build(
        self, doc_lengths: Sequence[int], sources: Sequence[str], rng
    ) -> tuple[Session, ...]:
        """Shuffled, so each source's carrier sees interleaved updates rather
        than one long same-source run. `sources` unused — the carrier is
        looked up per item at train time (train_loop._seed), not scheduled."""
        return tuple(
            Session(
                items=(
                    WorkItem(
                        doc_idx=int(doc_idx),
                        start=0,
                        end=int(doc_lengths[doc_idx]),
                    ),
                )
            )
            for doc_idx in rng.permutation(len(doc_lengths))
        )

    def count(self, doc_lengths: Sequence[int]) -> int:
        return len(doc_lengths)

    def compose(
        self, doc_lengths: Sequence[int], sources: Sequence[str]
    ) -> tuple[CompositionRow, ...]:
        counts: Counter[str] = Counter()
        for _, source in zip(doc_lengths, sources, strict=True):
            counts[source] += 1
        return tuple(
            CompositionRow(
                source=source, no_carry_docs=0, carry_docs=n, items=n
            )
            for source, n in sorted(counts.items())
        )

    def describe(self) -> str:
        return "everlasting: whole-doc sessions, one persistent carry per source"


EVERLASTING = register(Everlasting())

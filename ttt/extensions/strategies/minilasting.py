"""Multi-doc sessions, bounded per source: a carry that outlives one document
but not the epoch.

Hybrid carries within one (possibly sliced) document, never across a
document boundary. Everlasting never resets at all. This is the window
between them — `docs_per_session` documents of one source chained into one
session, same as single_doc_eval_v1.run_chained measures, then reset — the
carry a user gets from leaving it on for a session of use, not forever.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from ttt.core.report import CompositionRow
from ttt.core.schedule import derive_slice_count, slice_doc
from ttt.core.types import Session, WorkItem
from ttt.extensions.strategies._registry import SESSION, register


@dataclass(frozen=True)
class Minilasting:
    name: str = "minilasting"
    carry_scope: str = SESSION

    docs_per_session: int = 5
    carry_min_tokens: int = 2100
    slice_min_tokens: int = 1000
    slices_min: int = 2
    slices_max: int = 6

    def __post_init__(self) -> None:
        if self.docs_per_session < 1:
            raise ValueError(
                f"docs_per_session must be >= 1, got {self.docs_per_session}"
            )
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

    def build(
        self, doc_lengths: Sequence[int], sources: Sequence[str], rng
    ) -> tuple[Session, ...]:
        """One shuffle orders every document; each source's share of that
        order is then cut into `docs_per_session`-sized runs — never mixing
        sources within a run — and the runs themselves are shuffled again so
        training doesn't grind through one source's runs before the next."""
        by_source: dict[str, list[int]] = defaultdict(list)
        for doc_idx in rng.permutation(len(doc_lengths)):
            by_source[sources[doc_idx]].append(int(doc_idx))

        groups = [
            docs[start : start + self.docs_per_session]
            for docs in (by_source[source] for source in sorted(by_source))
            for start in range(0, len(docs), self.docs_per_session)
        ]
        return tuple(
            Session(
                items=tuple(
                    WorkItem(doc_idx=doc_idx, start=start, end=end)
                    for doc_idx in groups[g]
                    for start, end in slice_doc(
                        int(doc_lengths[doc_idx]),
                        self.slices_for(int(doc_lengths[doc_idx])),
                        self.slice_min_tokens,
                        rng,
                    )
                )
            )
            for g in rng.permutation(len(groups))
        )

    def count(self, doc_lengths: Sequence[int]) -> int:
        return sum(self.slices_for(int(length)) for length in doc_lengths)

    def compose(
        self, doc_lengths: Sequence[int], sources: Sequence[str]
    ) -> tuple[CompositionRow, ...]:
        """`no_carry_docs` is each group's first document — the one that
        starts from a reset carry, same meaning as hybrid's untouched docs."""
        by_source: dict[str, list[int]] = defaultdict(list)
        for length, source in zip(doc_lengths, sources, strict=True):
            by_source[source].append(int(length))

        no_carry: Counter[str] = Counter()
        carry: Counter[str] = Counter()
        items: Counter[str] = Counter()
        for source, lengths in by_source.items():
            for position, length in enumerate(lengths):
                (no_carry if position % self.docs_per_session == 0 else carry)[
                    source
                ] += 1
                items[source] += self.slices_for(length)
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
            f"minilasting: {self.docs_per_session} docs per source per session, "
            f"carry reset between sessions; docs < {self.carry_min_tokens} tokens "
            f"whole, else {self.slices_min}-{self.slices_max} slices"
        )


MINILASTING = register(Minilasting())

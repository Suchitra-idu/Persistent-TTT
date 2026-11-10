"""Multi-doc sessions that cross both same-source and other-source boundaries.

Chains `docs_per_session` documents, each next one keeping the source with
probability `same_source_share`, so the carry is trained on both kinds of
boundary. Slicing matches hybrid exactly, so the LR schedule is the same length.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ttt.core.schedule import derive_slice_count, slice_doc
from ttt.core.types import Session, WorkItem
from ttt.extensions.strategies._registry import register


@dataclass(frozen=True)
class Mixed:
    name: str = "mixed"

    docs_per_session: int = 5
    same_source_share: float = 0.5
    carry_min_tokens: int = 2100
    slice_min_tokens: int = 1000
    slices_min: int = 2
    slices_max: int = 6

    def __post_init__(self) -> None:
        if self.docs_per_session < 1:
            raise ValueError(
                f"docs_per_session must be >= 1, got {self.docs_per_session}"
            )
        if not 0.0 <= self.same_source_share <= 1.0:
            raise ValueError(
                f"same_source_share must be in [0, 1], got {self.same_source_share}"
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
        """Every document exactly once. A switch picks the new source in
        proportion to its remaining documents; when the wanted kind of next
        document has run out, the other kind is used, and when both have, the
        session ends short."""
        order = [int(i) for i in rng.permutation(len(doc_lengths))]
        queues: dict[str, deque[int]] = {}
        for doc_idx in order:
            queues.setdefault(sources[doc_idx], deque()).append(doc_idx)
        used: set[int] = set()

        sessions = []
        for first in order:
            if first in used:
                continue
            chain = [self._take(queues, sources[first], used)]
            while len(chain) < self.docs_per_session:
                current = sources[chain[-1]]
                same = rng.random() < self.same_source_share
                picked = self._next(queues, current, same, used, rng)
                if picked is None:
                    picked = self._next(queues, current, not same, used, rng)
                if picked is None:
                    break
                chain.append(picked)
            sessions.append(
                Session(
                    items=tuple(
                        WorkItem(doc_idx=doc_idx, start=start, end=end)
                        for doc_idx in chain
                        for start, end in slice_doc(
                            int(doc_lengths[doc_idx]),
                            self.slices_for(int(doc_lengths[doc_idx])),
                            self.slice_min_tokens,
                            rng,
                        )
                    )
                )
            )
        return tuple(sessions)

    @staticmethod
    def _take(queues: dict[str, deque[int]], source: str, used: set[int]) -> int:
        queue = queues[source]
        while queue[0] in used:
            queue.popleft()
        doc_idx = queue.popleft()
        used.add(doc_idx)
        return doc_idx

    def _next(
        self,
        queues: dict[str, deque[int]],
        current: str,
        same: bool,
        used: set[int],
        rng,
    ) -> int | None:
        for queue in queues.values():
            while queue and queue[0] in used:
                queue.popleft()
        if same:
            return self._take(queues, current, used) if queues[current] else None
        others = sorted(s for s, q in queues.items() if s != current and q)
        if not others:
            return None
        draw = rng.integers(0, sum(len(queues[s]) for s in others))
        for source in others:
            if draw < len(queues[source]):
                return self._take(queues, source, used)
            draw -= len(queues[source])
        raise AssertionError("weighted draw fell outside the remaining documents")

    def count(self, doc_lengths: Sequence[int]) -> int:
        return sum(self.slices_for(int(length)) for length in doc_lengths)

    def describe(self) -> str:
        return (
            f"mixed: {self.docs_per_session} docs per session, next doc same source "
            f"with p={self.same_source_share}, carry reset between sessions; docs < "
            f"{self.carry_min_tokens} tokens whole, else "
            f"{self.slices_min}-{self.slices_max} slices"
        )


MIXED = register(Mixed())

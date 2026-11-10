"""Rng — every draw in the system, injected (RULES.md rule 3)."""

from __future__ import annotations

from typing import MutableSequence, Protocol, Sequence, runtime_checkable



@runtime_checkable
class Rng(Protocol):
    def integers(self, low: int, high: int) -> int:
        """Uniform over [low, high). Empty ranges are the caller's bug."""

    def permutation(self, n: int) -> list[int]: ...

    def shuffle(self, items: MutableSequence) -> None: ...

    def random(self) -> float: ...

def is_permutation(values: Sequence[int], n: int) -> bool:
    return sorted(values) == list(range(n))

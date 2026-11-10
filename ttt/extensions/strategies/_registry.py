"""The Strategy protocol and the STRATEGIES registry (D4)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ttt.core.report import CompositionRow
from ttt.core.types import Session

@runtime_checkable
class Strategy(Protocol):
    """One epoch of sessions; the carry is reset between them."""

    name: str

    def build(
        self, doc_lengths: Sequence[int], sources: Sequence[str], rng
    ) -> tuple[Session, ...]:
        """The epoch's schedule. Deterministic given `rng`'s seed."""

    def count(self, doc_lengths: Sequence[int]) -> int:
        """Total work items, without drawing cuts — this sizes the LR schedule."""

    def compose(
        self, doc_lengths: Sequence[int], sources: Sequence[str]
    ) -> tuple[CompositionRow, ...]:
        """Per-source split of carrying vs single-item docs, for the boot log."""

    def describe(self) -> str:
        """One line, for the boot log."""


STRATEGIES: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    if strategy.name in STRATEGIES:
        raise ValueError(
            f"strategy {strategy.name!r} is already registered; a name collision "
            "would make the resolved config ambiguous"
        )
    STRATEGIES[strategy.name] = strategy
    return strategy


def get(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}. Known: {sorted(STRATEGIES)}")
    return STRATEGIES[name]

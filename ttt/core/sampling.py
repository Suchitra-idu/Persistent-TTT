"""Choosing which documents to look at, given an injected rng.

The rng needs only `random()`, half-open `integers(low, high)`, in-place
`shuffle(list)` and `permutation(n)` — the shape Ring 2's Rng port formalises.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SourceShortfall:
    """A source that could not supply as many documents as were asked for."""

    source: str
    got: int
    wanted: int


def permutation(n: int, rng) -> list[int]:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return list(rng.permutation(n))


def uniform_indices(n_available: int, n_wanted: int, rng) -> list[int]:
    if n_available < 0:
        raise ValueError(f"n_available must be >= 0, got {n_available}")
    pool = list(range(n_available))
    rng.shuffle(pool)
    return pool[: max(0, n_wanted)]


def n_per_source_indices(
    source_labels: Sequence[str], n_per_source: int, rng
) -> tuple[list[int], list[SourceShortfall]]:
    """Up to n per source, grouped by source in sorted order.

    A scarce source contributes everything it has and reports a shortfall,
    which is a caveat on every number in that eval rather than a log line.
    """
    if n_per_source < 0:
        raise ValueError(f"n_per_source must be >= 0, got {n_per_source}")
    by_source = _group_indices(source_labels)

    picked: list[int] = []
    shortfalls: list[SourceShortfall] = []
    for source in sorted(by_source):
        pool = by_source[source]
        rng.shuffle(pool)
        take = min(n_per_source, len(pool))
        picked.extend(pool[:take])
        if take < n_per_source:
            shortfalls.append(
                SourceShortfall(source=source, got=take, wanted=n_per_source)
            )
    return picked, shortfalls


def stratified_indices(source_labels: Sequence[str], n_target: int, rng) -> list[int]:
    """Round-robin per source to n_target, then top up randomly.

    Unreachable at the default config, where n_per_source is the policy
    (PLAN §9.5); kept because cutting it is a research decision.
    """
    if n_target < 0:
        raise ValueError(f"n_target must be >= 0, got {n_target}")
    by_source = _group_indices(source_labels)
    for pool in by_source.values():
        rng.shuffle(pool)

    order = sorted(by_source)
    picked: list[int] = []
    while len(picked) < n_target and any(by_source[k] for k in order):
        for key in order:
            if not by_source[key]:
                continue
            picked.append(by_source[key].pop())
            if len(picked) >= n_target:
                break

    if len(picked) < n_target:
        taken = set(picked)
        remaining = [i for i in range(len(source_labels)) if i not in taken]
        rng.shuffle(remaining)
        picked.extend(remaining[: n_target - len(picked)])
    return picked


def _group_indices(source_labels: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(source_labels):
        grouped[label].append(index)
    return dict(grouped)

"""Rebalancing a source-skewed pool to a preset's ratios.

Callers must shuffle first: this takes the head of each per-source index list,
and the pre-shuffle is what makes that head a random sample.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SourceQuota:
    source: str
    took: int
    target: int
    available: int

    @property
    def is_short(self) -> bool:
        return self.took < self.target


@dataclass(frozen=True)
class BalanceResult:
    indices: tuple[int, ...]
    quotas: tuple[SourceQuota, ...]

    @property
    def total(self) -> int:
        return len(self.indices)

    @property
    def short_sources(self) -> tuple[SourceQuota, ...]:
        return tuple(q for q in self.quotas if q.is_short)


def balanced_indices(
    source_labels: Sequence[str],
    weights: Mapping[str, int],
    target_total: int,
) -> BalanceResult:
    """Pick indices so per-source counts follow `weights`, aiming at target_total.

    Weights are unnormalised; sources absent from them are dropped, which is
    how --only-sources is expressed as data. The result can be smaller than
    target_total but never larger than the pool, and stays in pool order so
    domains remain interleaved.
    """
    if target_total < 0:
        raise ValueError(f"target_total must be >= 0, got {target_total}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"preset weights must be non-negative, got {dict(weights)}")

    by_source: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(source_labels):
        by_source[label].append(index)

    present = [s for s in weights if s in by_source]
    if not present:
        raise ValueError(
            f"none of the preset's sources {sorted(weights)} are in the pool, "
            f"which has {sorted(by_source)}"
        )
    total_weight = sum(weights[s] for s in present)
    if total_weight <= 0:
        raise ValueError(
            f"preset weights for the present sources {sorted(present)} sum to "
            f"{total_weight}; at least one must be positive"
        )

    keep: list[int] = []
    quotas: list[SourceQuota] = []
    for source in sorted(present):
        target = int(target_total * weights[source] / total_weight)
        pool = by_source[source]
        take = min(target, len(pool))
        keep.extend(pool[:take])
        quotas.append(
            SourceQuota(source=source, took=take, target=target, available=len(pool))
        )

    keep.sort()
    return BalanceResult(indices=tuple(keep), quotas=tuple(quotas))


def source_counts(source_labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for label in source_labels:
        counts[label] += 1
    return dict(counts)

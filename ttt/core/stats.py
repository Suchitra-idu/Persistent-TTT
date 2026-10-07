"""Uncertainty over held-out documents: percentile bootstrap, ranks, quantiles.

The unit of resampling is the document. Slices and repeats of one document
are correlated, so resampling them would report a falsely tight interval.
Resample draws are injected, never generated here (RULES.md rule 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ttt.core.metrics import geometric_mean_ppl

DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True)
class Estimate:
    """A gap in perplexity with its interval. Positive means `treated` helped.
    `reference` is the reference arm's geometric-mean ppl, the scale that
    makes gaps comparable across sources."""

    measure: str
    source: str
    n_docs: int
    interval: Interval
    win_rate: float
    reference: float = float("nan")

    @property
    def relative(self) -> float:
        return self.interval.point / self.reference


def quantile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if not values:
        return float("nan")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    fraction = position - below
    return ordered[below] + (ordered[above] - ordered[below]) * fraction


def bootstrap(
    statistic: Callable[[Sequence[int]], float],
    n: int,
    resamples: Sequence[Sequence[int]],
    *,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """`statistic` takes the document indices of one resample."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    point = statistic(range(n))
    replicates = [statistic(indices) for indices in resamples]
    finite = [value for value in replicates if math.isfinite(value)]
    if not finite:
        return Interval(point=point, low=float("nan"), high=float("nan"))
    tail = (1.0 - level) / 2.0
    return Interval(
        point=point, low=quantile(finite, tail), high=quantile(finite, 1.0 - tail)
    )


def ppl_gap(reference: Sequence[float], treated: Sequence[float]) -> float:
    """Geometric-mean ppl of `reference` minus that of `treated`, as `Gaps` does."""
    return geometric_mean_ppl(reference) - geometric_mean_ppl(treated)


def paired_gap(
    reference: Sequence[float],
    treated: Sequence[float],
    resamples: Sequence[Sequence[int]],
    *,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """Both arms resampled with the same document indices."""
    if len(reference) != len(treated):
        raise ValueError(
            f"paired arms differ in length: {len(reference)} vs {len(treated)}"
        )

    def statistic(indices: Sequence[int]) -> float:
        return ppl_gap([reference[i] for i in indices], [treated[i] for i in indices])

    return bootstrap(statistic, len(reference), resamples, level=level)


def win_rate(reference: Sequence[float], treated: Sequence[float]) -> float:
    """Fraction of documents where `treated` is strictly lower; ties count half."""
    if not reference:
        return float("nan")
    score = sum(
        1.0 if t < r else 0.5 if t == r else 0.0
        for r, t in zip(reference, treated, strict=True)
    )
    return score / len(reference)


def ranks(values: Sequence[float]) -> list[float]:
    """1-based, ties share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        shared = (start + end) / 2.0 + 1.0
        for k in range(start, end + 1):
            result[order[k]] = shared
        start = end + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """NaN below three points or when either side is constant."""
    if len(xs) != len(ys):
        raise ValueError(f"lengths differ: {len(xs)} vs {len(ys)}")
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / math.sqrt(vx * vy)

"""Perplexity aggregation and the gap decomposition.

Perplexity only ever combines in log space weighted by tokens. The five-regime
decomposition is additive, so each delta attributes to one mechanism:

    within  = fresh          - cold_carry_off
    between = cold_carry_off - cold_carry
    seed    = cold_carry     - carry
    total   = fresh          - carry
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH, EvalRow, PplRow


@dataclass(frozen=True)
class Gaps:
    """Positive means the mechanism helped."""

    within: float
    between: float
    seed: float
    total: float

    @property
    def is_additive(self) -> bool:
        return math.isclose(
            self.within + self.between + self.seed,
            self.total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


@dataclass(frozen=True)
class SourceSummary:
    source: str
    n_docs: int
    n_tokens: int
    ppl_by_regime: Mapping[str, float]
    gaps: Gaps


def token_weighted_ppl(rows: Sequence[PplRow]) -> float:
    """NaN over zero tokens."""
    total_log = sum(math.log(row.ppl) * row.n_tokens for row in rows)
    total_tokens = sum(row.n_tokens for row in rows)
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_log / total_tokens)


def geometric_mean_ppl(ppls: Sequence[float]) -> float:
    """Documents weigh equally here; slices within a document weigh by tokens."""
    if not ppls:
        return float("nan")
    if any(p <= 0.0 for p in ppls):
        raise ValueError("perplexities must be positive")
    return math.exp(sum(math.log(p) for p in ppls) / len(ppls))


def gap_decomposition(
    *,
    fresh: float,
    cold_carry_off: float,
    cold_carry: float,
    carry: float | None = None,
) -> Gaps:
    """carry=None is the zero-seed eval: seed is 0 and total is fresh - cold_carry."""
    seeded = cold_carry if carry is None else carry
    return Gaps(
        within=fresh - cold_carry_off,
        between=cold_carry_off - cold_carry,
        seed=cold_carry - seeded,
        total=fresh - seeded,
    )


def seed_gap(cold_carry: float, seeded_carry: float) -> float:
    """Positive => the trained seed beats cold-start accumulation."""
    return cold_carry - seeded_carry


def summarise_by_source(rows: Sequence[EvalRow]) -> tuple[SourceSummary, ...]:
    by_source: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source].append(row)

    summaries: list[SourceSummary] = []
    for source in sorted(by_source):
        source_rows = by_source[source]
        by_regime: dict[str, list[PplRow]] = defaultdict(list)
        for row in source_rows:
            by_regime[row.regime].append(PplRow(n_tokens=row.n_tokens, ppl=row.ppl))
        ppls = {
            regime: token_weighted_ppl(regime_rows)
            for regime, regime_rows in by_regime.items()
        }
        reference = by_regime.get(COLD_CARRY) or next(iter(by_regime.values()))
        summaries.append(
            SourceSummary(
                source=source,
                n_docs=len({row.doc_idx for row in source_rows}),
                n_tokens=sum(row.n_tokens for row in reference),
                ppl_by_regime=ppls,
                gaps=gap_decomposition(
                    fresh=ppls.get(FRESH, float("nan")),
                    cold_carry_off=ppls.get(COLD_CARRY_OFF, float("nan")),
                    cold_carry=ppls.get(COLD_CARRY, float("nan")),
                    carry=ppls.get(CARRY),
                ),
            )
        )
    return tuple(summaries)


def clip_ratio(total_norm: float, max_grad_norm: float) -> float:
    """1.0 means the step was not clipped."""
    if max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
    return min(1.0, max_grad_norm / (total_norm + 1e-12))

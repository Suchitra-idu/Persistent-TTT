"""Perplexity aggregation and the gap decomposition.

Perplexity only ever combines in log space weighted by tokens. The six-regime
decomposition is additive, so each delta attributes to one mechanism:

    lora    = fresh          - lora_only
    within  = lora_only      - cold_carry_off
    between = cold_carry_off - cold_carry
    seed    = cold_carry     - carry
    total   = fresh          - carry
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.core.types import (
    CARRY,
    COLD_CARRY,
    COLD_CARRY_OFF,
    FRESH,
    LORA_ONLY,
    EvalRow,
    PplRow,
    RepeatRow,
    SliceRow,
)

ALL_SOURCES = "ALL"


@dataclass(frozen=True)
class Gaps:
    """Positive means the mechanism helped."""

    lora: float
    within: float
    between: float
    seed: float
    total: float

    @property
    def is_additive(self) -> bool:
        return math.isclose(
            self.lora + self.within + self.between + self.seed,
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


@dataclass(frozen=True)
class SliceSummary:
    """Same shape as `SourceSummary`, grouped by distance into the document
    instead of by source."""

    slice_index: int
    n_docs: int
    n_tokens: int
    ppl_by_regime: Mapping[str, float]
    gaps: Gaps


@dataclass(frozen=True)
class RepeatSummary:
    """Same shape as `SourceSummary`, grouped by (source, repeat) instead of
    by source alone. `source=ALL_SOURCES` rolls every source up at that
    repeat. `state_ratio` and `gate_mean` are `cold_carry`'s only — the
    regime that evolves."""

    source: str
    repeat: int
    n_docs: int
    n_tokens: int
    ppl_by_regime: Mapping[str, float]
    gaps: Gaps
    state_ratio: float
    gate_mean: float = 1.0


def perplexity(nll: float) -> float:
    """exp(nll), saturating at inf: a diverged eval is a number, not a crash."""
    try:
        return math.exp(nll)
    except OverflowError:
        return float("inf")


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
    lora_only: float,
    cold_carry_off: float,
    cold_carry: float,
    carry: float | None = None,
) -> Gaps:
    """carry=None is the zero-seed eval: seed is 0 and total is fresh - cold_carry."""
    seeded = cold_carry if carry is None else carry
    return Gaps(
        lora=fresh - lora_only,
        within=lora_only - cold_carry_off,
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
                    lora_only=ppls.get(LORA_ONLY, float("nan")),
                    cold_carry_off=ppls.get(COLD_CARRY_OFF, float("nan")),
                    cold_carry=ppls.get(COLD_CARRY, float("nan")),
                    carry=ppls.get(CARRY),
                ),
            )
        )
    return tuple(summaries)


def summarise_by_slice_index(slices: Sequence[SliceRow]) -> tuple[SliceSummary, ...]:
    """A win concentrated in early slices vs. one that holds up in later ones
    is exactly what a flat aggregate perplexity number can't show (arXiv
    2410.23771) — each slice is a fixed *fraction* of its own document
    (`schedule.equal_token_slices`), so index 0 is always "earliest 1/n"."""
    by_index: dict[int, list[SliceRow]] = defaultdict(list)
    for row in slices:
        by_index[row.slice_index].append(row)

    summaries: list[SliceSummary] = []
    for index in sorted(by_index):
        index_rows = by_index[index]
        by_regime: dict[str, list[PplRow]] = defaultdict(list)
        for row in index_rows:
            by_regime[row.regime].append(PplRow(n_tokens=row.n_tokens, ppl=row.ppl))
        ppls = {
            regime: token_weighted_ppl(regime_rows)
            for regime, regime_rows in by_regime.items()
        }
        reference = by_regime.get(COLD_CARRY) or next(iter(by_regime.values()))
        summaries.append(
            SliceSummary(
                slice_index=index,
                n_docs=len({row.doc_idx for row in index_rows}),
                n_tokens=sum(row.n_tokens for row in reference),
                ppl_by_regime=ppls,
                gaps=gap_decomposition(
                    fresh=ppls.get(FRESH, float("nan")),
                    lora_only=ppls.get(LORA_ONLY, float("nan")),
                    cold_carry_off=ppls.get(COLD_CARRY_OFF, float("nan")),
                    cold_carry=ppls.get(COLD_CARRY, float("nan")),
                    carry=ppls.get(CARRY),
                ),
            )
        )
    return tuple(summaries)


def summarise_by_repeat(rows: Sequence[RepeatRow]) -> tuple[RepeatSummary, ...]:
    """Per (source, repeat), plus an `ALL_SOURCES` rollup — whether a doc's
    own carry pays off across replays is a within-source question first."""
    return _repeat_summaries(rows, key=lambda row: row.source) + _repeat_summaries(
        rows, key=lambda row: ALL_SOURCES
    )


def _repeat_summaries(rows: Sequence[RepeatRow], *, key) -> tuple[RepeatSummary, ...]:
    by_cell: dict[tuple[str, int], list[RepeatRow]] = defaultdict(list)
    for row in rows:
        by_cell[(key(row), row.repeat)].append(row)

    summaries = []
    for source, repeat in sorted(by_cell):
        cell = by_cell[(source, repeat)]
        by_regime: dict[str, list[RepeatRow]] = defaultdict(list)
        for row in cell:
            by_regime[row.regime].append(row)
        ppls = {
            regime: token_weighted_ppl(
                [PplRow(n_tokens=r.n_tokens, ppl=r.ppl) for r in regime_rows]
            )
            for regime, regime_rows in by_regime.items()
        }
        reference = by_regime.get(COLD_CARRY) or next(iter(by_regime.values()))
        summaries.append(
            RepeatSummary(
                source=source,
                repeat=repeat,
                n_docs=len({r.doc_idx for r in reference}),
                n_tokens=sum(r.n_tokens for r in reference),
                ppl_by_regime=ppls,
                gaps=gap_decomposition(
                    fresh=ppls.get(FRESH, float("nan")),
                    lora_only=ppls.get(LORA_ONLY, float("nan")),
                    cold_carry_off=ppls.get(COLD_CARRY_OFF, float("nan")),
                    cold_carry=ppls.get(COLD_CARRY, float("nan")),
                    carry=ppls.get(CARRY),
                ),
                state_ratio=_mean_state_ratio(by_regime.get(COLD_CARRY, ())),
                gate_mean=_mean_gate(by_regime.get(COLD_CARRY, ())),
            )
        )
    return tuple(summaries)


def _mean_state_ratio(rows: Sequence[RepeatRow]) -> float:
    if not rows:
        return 0.0
    return sum(row.state_ratio for row in rows) / len(rows)


def _mean_gate(rows: Sequence[RepeatRow]) -> float:
    if not rows:
        return 1.0
    return sum(row.gate_mean for row in rows) / len(rows)


def clip_ratio(total_norm: float, max_grad_norm: float) -> float:
    """1.0 means the step was not clipped."""
    if max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
    return min(1.0, max_grad_norm / (total_norm + 1e-12))

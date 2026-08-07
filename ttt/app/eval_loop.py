"""Held-out perplexity under six regimes, and the gap decomposition over them.

Snapshots the carry it found and puts it back, so firing this mid-training is a
no-op against the loop that called it (PLAN §7 defect 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.app.data_pipeline import Doc
from ttt.core import metrics, schedule
from ttt.core.metrics import SourceSummary
from ttt.core.types import (
    CARRY,
    CARRY_OFF,
    COLD_CARRY,
    COLD_CARRY_OFF,
    FRESH,
    LORA_ONLY,
    Carry,
    EvalRow,
    PplRow,
    SliceRow,
)
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import CARRY as CARRY_FAMILY
from ttt.ports.fast_weights import FastWeights

# (regime, evolve, reset between slices, seeded, lora). `fresh` and
# `lora_only` have TTT off, so the seed cannot reach either — which is why the
# seeded pass does not repeat them. `fresh` is the only regime with LoRA off:
# every other regime, `lora_only` included, is the trained model as-is.
PASSES = (
    (COLD_CARRY, True, False, False, True),
    (COLD_CARRY_OFF, True, True, False, True),
    (LORA_ONLY, False, False, False, True),
    (FRESH, False, False, False, False),
    (CARRY, True, False, True, True),
    (CARRY_OFF, True, True, True, True),
)


@dataclass(frozen=True)
class EvalReport:
    rows: tuple[EvalRow, ...]
    metrics: Mapping[str, float]
    summaries: tuple[SourceSummary, ...]
    slices: tuple[SliceRow, ...] = ()


def evaluate(
    *,
    docs: Sequence[Doc],
    compute: Compute,
    fast_weights: FastWeights,
    n_slices: int,
    carries: Mapping[str, Carry] | None = None,
    session_training: bool = True,
) -> EvalReport:
    if not docs:
        return EvalReport(rows=(), metrics={}, summaries=())

    restore = fast_weights.snapshot()
    try:
        rows, slices = _measure_all(docs, compute, fast_weights, n_slices, carries or {})
    finally:
        fast_weights.set_mode(evolve=True, stream=False, session=session_training)
        fast_weights.reset_carry()
        fast_weights.install(restore)

    return EvalReport(
        rows=rows,
        metrics=_aggregate(rows),
        summaries=metrics.summarise_by_source(rows),
        slices=slices,
    )


def _measure_all(
    docs, compute, fast_weights, n_slices, carries
) -> tuple[tuple[EvalRow, ...], tuple[SliceRow, ...]]:
    rows: list[EvalRow] = []
    slices: list[SliceRow] = []
    for doc in docs:
        seed = carries.get(doc.source)
        for regime, evolve, reset_between, seeded, lora in PASSES:
            if seeded and seed is None:
                continue
            measured = _measure(
                doc,
                compute=compute,
                fast_weights=fast_weights,
                n_slices=n_slices,
                evolve=evolve,
                reset_between=reset_between,
                seed=seed if seeded else None,
                lora=lora,
            )
            rows.append(_row(doc, regime, measured))
            slices.extend(_slice_rows(doc, regime, measured))
    return tuple(rows), tuple(slices)


def _measure(
    doc: Doc, *, compute, fast_weights, n_slices, evolve, reset_between, seed, lora
) -> tuple[PplRow, ...]:
    fast_weights.set_mode(evolve=evolve, stream=False, session=True)
    _restart(fast_weights, seed)

    rows: list[PplRow] = []
    for start, end in schedule.equal_token_slices(doc.n_tokens, n_slices):
        loss = compute.eval_loss(doc.token_ids[start:end], lora=lora)
        fast_weights.advance_carry()
        rows.append(
            PplRow(
                n_tokens=end - start,
                ppl=metrics.perplexity(loss),
                state_ratio=fast_weights.state_ratio(family=CARRY_FAMILY),
            )
        )
        if reset_between:
            _restart(fast_weights, seed)
    return tuple(rows)


def _restart(fast_weights, seed: Carry | None) -> None:
    fast_weights.reset_carry()
    if seed is not None:
        fast_weights.install(seed)


def _row(doc: Doc, regime: str, measured: Sequence[PplRow]) -> EvalRow:
    return EvalRow(
        doc_idx=doc.index,
        source=doc.source,
        regime=regime,
        n_tokens=sum(row.n_tokens for row in measured),
        ppl=metrics.token_weighted_ppl(measured),
        state_ratio_final=measured[-1].state_ratio if measured else 0.0,
    )


def _slice_rows(doc: Doc, regime: str, measured: Sequence[PplRow]) -> tuple[SliceRow, ...]:
    return tuple(
        SliceRow(
            doc_idx=doc.index,
            source=doc.source,
            regime=regime,
            slice_index=index,
            n_tokens=row.n_tokens,
            ppl=row.ppl,
        )
        for index, row in enumerate(measured)
    )


def _aggregate(rows: Sequence[EvalRow]) -> dict[str, float]:
    by_regime = _by_regime(rows)
    cold_carry = _mean(by_regime[COLD_CARRY])
    cold_carry_off = _mean(by_regime[COLD_CARRY_OFF])
    lora_only = _mean(by_regime[LORA_ONLY])
    fresh = _mean(by_regime[FRESH])
    gaps = metrics.gap_decomposition(
        fresh=fresh,
        lora_only=lora_only,
        cold_carry_off=cold_carry_off,
        cold_carry=cold_carry,
    )

    measured = {
        "eval/carry_ppl": cold_carry,
        "eval/carry_off_ppl": cold_carry_off,
        "eval/lora_only_ppl": lora_only,
        "eval/fresh_ppl": fresh,
        "eval/gap_lora": gaps.lora,
        "eval/gap_within": gaps.within,
        "eval/gap_between": gaps.between,
        "eval/gap_total": gaps.total,
        "eval/state_ratio_final": _mean_state_ratio(by_regime[COLD_CARRY]),
    }
    measured.update(_seeded(by_regime))
    return measured


def _seeded(by_regime: Mapping[str, list[EvalRow]]) -> dict[str, float]:
    """Δseed compares the seeded docs against themselves cold, never against
    the whole pool — the seeded subset is not a random sample of it."""
    seeded = by_regime[CARRY]
    if not seeded:
        return {}
    docs = {row.doc_idx for row in seeded}
    carry = _mean(seeded)
    return {
        "eval_seed/carry_ppl": carry,
        "eval_seed/carry_off_ppl": _mean(by_regime[CARRY_OFF]),
        "eval_seed/gap_between": _mean(by_regime[CARRY_OFF]) - carry,
        "eval_seed/gap_seed": _mean(
            [row for row in by_regime[COLD_CARRY] if row.doc_idx in docs]
        )
        - carry,
        "eval_seed/n_seeded_docs": float(len(seeded)),
    }


def _by_regime(rows: Sequence[EvalRow]) -> dict[str, list[EvalRow]]:
    grouped: dict[str, list[EvalRow]] = {regime: [] for regime, *_ in PASSES}
    for row in rows:
        grouped[row.regime].append(row)
    return grouped


def _mean(rows: Sequence[EvalRow]) -> float:
    """Documents weigh equally in the aggregate; slices weigh by tokens."""
    return metrics.geometric_mean_ppl([row.ppl for row in rows])


def _mean_state_ratio(rows: Sequence[EvalRow]) -> float:
    if not rows:
        return 0.0
    return sum(row.state_ratio_final for row in rows) / len(rows)

"""Cross-checkpoint estimates for the paper, from paper_eval bundles' raw rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ttt.app import evidence
from ttt.core.metrics import ALL_SOURCES
from ttt.core.stats import Estimate
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY, EvalRow, SliceRow
from ttt.ports.rng import Rng

VS_MATCHED = "vs_matched_lora"
POSITION_GAPS = (
    ("within", LORA_ONLY, COLD_CARRY_OFF),
    ("between_slices", COLD_CARRY_OFF, COLD_CARRY),
)


@dataclass(frozen=True)
class Saturation:
    threshold: float
    n_rows: int
    fraction_clipped: float
    min_factor: float


@dataclass(frozen=True)
class StatePoint:
    group: str
    position: int
    n_rows: int
    state_ratio: evidence.Band
    gate_mean: evidence.Band


def rebased(rows: Sequence[EvalRow], base: Sequence[EvalRow]) -> tuple[EvalRow, ...]:
    """`rows` with `fresh` taken from `base`, a checkpoint with no TTT layers.

    A TTT bundle written before the pretrained-W_down fix measured `fresh` with
    the trained W_down still in place; only a no-TTT checkpoint's is the base.
    """
    docs = {(r.source, r.doc_idx) for r in rows}
    fresh = tuple(r for r in base if r.regime == FRESH and (r.source, r.doc_idx) in docs)
    if len(fresh) != len(docs):
        raise ValueError(
            f"base measured {len(fresh)} of the {len(docs)} documents; "
            "the bundles do not share a holdout"
        )
    return tuple(r for r in rows if r.regime != FRESH) + fresh


def matched_estimates(
    control: Sequence[EvalRow], candidate: Sequence[EvalRow], *, rng: Rng, n_boot: int
) -> tuple[Estimate, ...]:
    """The candidate's full TTT path against a LoRA-only control's LoRA path."""
    return evidence.paired_estimates(
        VS_MATCHED,
        evidence.per_doc(control, where=lambda r: r.regime == LORA_ONLY),
        evidence.per_doc(candidate, where=lambda r: r.regime == COLD_CARRY),
        rng=rng,
        n_boot=n_boot,
    )


def position_estimates(
    rows: Sequence[SliceRow], *, rng: Rng, n_boot: int
) -> tuple[tuple[int, Estimate], ...]:
    """Pooled gaps per slice index. Slice 0 has nothing carried in, so its
    between-slices gap is zero by construction."""
    out: list[tuple[int, Estimate]] = []
    for index in sorted({r.slice_index for r in rows}):
        at = [r for r in rows if r.slice_index == index]
        for measure, reference, treated in POSITION_GAPS:
            out += [
                (index, e)
                for e in evidence.paired_estimates(
                    measure,
                    evidence.per_doc(at, where=lambda r, g=reference: r.regime == g),
                    evidence.per_doc(at, where=lambda r, g=treated: r.regime == g),
                    rng=rng,
                    n_boot=n_boot,
                )
                if e.source == ALL_SOURCES
            ]
    return tuple(out)


def clip_saturation(
    state_ratios: Sequence[float], *, clip_tau: float, w0_norm: float
) -> Saturation:
    """The read-time clip binds once ||eta*S|| > tau, i.e. a state ratio above tau/||W0||."""
    if clip_tau <= 0.0 or w0_norm <= 0.0:
        raise ValueError(f"clip_tau and w0_norm must be > 0, got {clip_tau}, {w0_norm}")
    if not state_ratios:
        raise ValueError("no state ratios to test against the clip")
    threshold = clip_tau / w0_norm
    return Saturation(
        threshold=threshold,
        n_rows=len(state_ratios),
        fraction_clipped=sum(r > threshold for r in state_ratios) / len(state_ratios),
        min_factor=min(state_ratios) / threshold,
    )


def state_points(
    rows: Iterable, *, group: Callable, position: Callable
) -> tuple[StatePoint, ...]:
    """State ratio and gate quartiles per (group, position)."""
    cells: dict[tuple[str, int], list] = defaultdict(list)
    for row in rows:
        cells[(group(row), position(row))].append(row)
    return tuple(
        StatePoint(
            group=key[0],
            position=key[1],
            n_rows=len(cell),
            state_ratio=evidence.band([r.state_ratio for r in cell]),
            gate_mean=evidence.band([r.gate_mean for r in cell]),
        )
        for key, cell in sorted(cells.items())
    )

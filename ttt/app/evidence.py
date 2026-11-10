"""Per-document rows in, estimates with bootstrap intervals out.

Every estimate is paired on (source, doc_idx) and reported per source plus an
`ALL_SOURCES` pool. Rows are persisted as JSON through the Storage port so an
analysis can be rerun without a GPU.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ttt.app.boundary_eval import DOC_RESET, PERSIST, SLICE_RESET, ArmRow
from ttt.app.repeat_eval import RepeatItem
from ttt.core import metrics, stats
from ttt.core.metrics import ALL_SOURCES
from ttt.core.stats import Estimate, Interval
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY, EvalRow, PplRow
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage

Key = tuple[str, int]

# (measure, reference regime, treated regime)
REGIME_GAPS = (
    ("lora", FRESH, LORA_ONLY),
    ("within", LORA_ONLY, COLD_CARRY_OFF),
    ("between_slices", COLD_CARRY_OFF, COLD_CARRY),
    ("total", FRESH, COLD_CARRY),
)

CROSS_DOC = "cross_doc"
CROSS_SLICE = "cross_slice"
DOMAIN_MEASURES = ("within", CROSS_DOC, CROSS_SLICE)

# Baseline checkpoint regime, candidate checkpoint regime. Positive = candidate wins.
CHECKPOINT_GAPS = (
    ("vs_cold_carry", LORA_ONLY, COLD_CARRY),
    ("vs_cold_carry_off", LORA_ONLY, COLD_CARRY_OFF),
    ("vs_lora_path", LORA_ONLY, LORA_ONLY),
    ("fresh_sanity", FRESH, FRESH),
)


@dataclass(frozen=True)
class Band:
    low: float
    median: float
    high: float


@dataclass(frozen=True)
class RepeatPoint:
    """Repeat r against the same documents' repeat 0."""

    source: str
    repeat: int
    n_docs: int
    gap: Interval
    state_ratio: Band
    gate_mean: Band


@dataclass(frozen=True)
class Correlate:
    measure: str
    n_sources: int
    rho: float


def resamples(n: int, n_boot: int, rng: Rng) -> tuple[tuple[int, ...], ...]:
    if n == 0:
        return ()
    return tuple(tuple(rng.integers(0, n) for _ in range(n)) for _ in range(n_boot))


def paired_estimates(
    measure: str,
    reference: Mapping[Key, float],
    treated: Mapping[Key, float],
    *,
    rng: Rng,
    n_boot: int,
    level: float = stats.DEFAULT_LEVEL,
) -> tuple[Estimate, ...]:
    """Joined on the documents both arms measured; sources first, pool last."""
    keys = sorted(set(reference) & set(treated))
    groups = [(s, [k for k in keys if k[0] == s]) for s in sorted({k[0] for k in keys})]
    if keys:
        groups.append((ALL_SOURCES, keys))
    out = []
    for source, group in groups:
        ref = [reference[k] for k in group]
        tr = [treated[k] for k in group]
        out.append(
            Estimate(
                measure=measure,
                source=source,
                n_docs=len(group),
                interval=stats.paired_gap(ref, tr, resamples(len(group), n_boot, rng), level=level),
                win_rate=stats.win_rate(ref, tr),
                reference=metrics.geometric_mean_ppl(ref),
            )
        )
    return tuple(out)


def gain_change(
    measure: str,
    candidate: tuple[Mapping[Key, float], Mapping[Key, float]],
    baseline: tuple[Mapping[Key, float], Mapping[Key, float]],
    *,
    rng: Rng,
    n_boot: int,
    level: float = stats.DEFAULT_LEVEL,
) -> Estimate:
    """How much more the candidate model gains from its treated arm than the
    baseline model does, pooled over the documents all four arms measured.
    Win rate is the share of documents with a larger per-document gain."""
    keys = sorted(set(candidate[0]) & set(candidate[1]) & set(baseline[0]) & set(baseline[1]))
    cand = ([candidate[0][k] for k in keys], [candidate[1][k] for k in keys])
    base = ([baseline[0][k] for k in keys], [baseline[1][k] for k in keys])
    diffs = [(cr - ct) - (br - bt) for cr, ct, br, bt in zip(*cand, *base)]
    return Estimate(
        measure=measure,
        source=ALL_SOURCES,
        n_docs=len(keys),
        interval=stats.gain_difference(cand, base, resamples(len(keys), n_boot, rng), level=level),
        win_rate=sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs) / len(diffs) if diffs else float("nan"),
        reference=metrics.geometric_mean_ppl(base[0]),
    )


def per_doc(rows: Iterable, *, where=lambda row: True) -> dict[Key, float]:
    """Token-weighted ppl per document over whichever rows pass `where`."""
    grouped: dict[Key, list[PplRow]] = defaultdict(list)
    for row in rows:
        if where(row):
            grouped[(row.source, row.doc_idx)].append(
                PplRow(n_tokens=row.n_tokens, ppl=row.ppl)
            )
    return {key: metrics.token_weighted_ppl(group) for key, group in grouped.items()}


def regime_estimates(
    rows: Sequence[EvalRow], *, rng: Rng, n_boot: int
) -> tuple[Estimate, ...]:
    out: tuple[Estimate, ...] = ()
    for measure, reference, treated in REGIME_GAPS:
        out += paired_estimates(
            measure,
            per_doc(rows, where=lambda r, regime=reference: r.regime == regime),
            per_doc(rows, where=lambda r, regime=treated: r.regime == regime),
            rng=rng,
            n_boot=n_boot,
        )
    return out


def boundary_estimates(
    rows: Sequence[ArmRow], *, rng: Rng, n_boot: int
) -> tuple[Estimate, ...]:
    def arm(name: str, *, skip_first: bool):
        return per_doc(
            rows,
            where=lambda r: r.arm == name and not (skip_first and r.chain_position == 0),
        )

    return paired_estimates(
        CROSS_DOC,
        arm(DOC_RESET, skip_first=True),
        arm(PERSIST, skip_first=True),
        rng=rng,
        n_boot=n_boot,
    ) + paired_estimates(
        CROSS_SLICE,
        arm(SLICE_RESET, skip_first=False),
        arm(DOC_RESET, skip_first=False),
        rng=rng,
        n_boot=n_boot,
    )


def repeat_curve(
    rows: Sequence[RepeatItem], *, rng: Rng, n_boot: int
) -> tuple[RepeatPoint, ...]:
    by_repeat: dict[int, list[RepeatItem]] = defaultdict(list)
    for row in rows:
        by_repeat[row.repeat].append(row)
    cold = per_doc(by_repeat.get(0, ()))
    points = []
    for repeat in sorted(r for r in by_repeat if r > 0):
        cell = by_repeat[repeat]
        treated = per_doc(cell)
        for estimate in paired_estimates(
            "repeat", cold, treated, rng=rng, n_boot=n_boot
        ):
            members = [
                r for r in cell if estimate.source in (ALL_SOURCES, r.source)
            ]
            points.append(
                RepeatPoint(
                    source=estimate.source,
                    repeat=repeat,
                    n_docs=estimate.n_docs,
                    gap=estimate.interval,
                    state_ratio=band([r.state_ratio for r in members]),
                    gate_mean=band([r.gate_mean for r in members]),
                )
            )
    return tuple(points)


def repeat_trend(
    rows: Sequence[RepeatItem], *, rng: Rng, n_boot: int
) -> tuple[Estimate, ...]:
    """Last repeat against the first: positive means the carry still compounds
    after one exposure; zero is the flat shape."""
    repeats = sorted({r.repeat for r in rows if r.repeat > 0})
    if len(repeats) < 2:
        return ()
    first, last = repeats[0], repeats[-1]
    return paired_estimates(
        f"repeat_{last}_vs_{first}",
        per_doc(rows, where=lambda r: r.repeat == first),
        per_doc(rows, where=lambda r: r.repeat == last),
        rng=rng,
        n_boot=n_boot,
    )


def fresh_by_source(rows: Sequence[EvalRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.regime == FRESH:
            grouped[row.source].append(row.ppl)
    return {s: metrics.geometric_mean_ppl(p) for s, p in sorted(grouped.items())}


def domain_correlates(
    fresh: Mapping[str, float], estimates: Sequence[Estimate]
) -> tuple[Correlate, ...]:
    """Spearman over sources of each measure's *relative* gap against fresh
    ppl — an absolute Δppl scales with the base ppl and correlates with it
    trivially. With six sources this is a description, not a test."""
    by_measure: dict[str, dict[str, float]] = defaultdict(dict)
    for e in estimates:
        if e.source != ALL_SOURCES and e.source in fresh and e.measure in DOMAIN_MEASURES:
            by_measure[e.measure][e.source] = e.relative
    out = []
    for measure in sorted(by_measure):
        sources = sorted(by_measure[measure])
        out.append(
            Correlate(
                measure=measure,
                n_sources=len(sources),
                rho=stats.spearman(
                    [fresh[s] for s in sources], [by_measure[measure][s] for s in sources]
                ),
            )
        )
    return tuple(out)


def checkpoint_estimates(
    baseline: Sequence[EvalRow],
    candidate: Sequence[EvalRow],
    *,
    rng: Rng,
    n_boot: int,
) -> tuple[Estimate, ...]:
    out: tuple[Estimate, ...] = ()
    for measure, base_regime, cand_regime in CHECKPOINT_GAPS:
        out += paired_estimates(
            measure,
            per_doc(baseline, where=lambda r, g=base_regime: r.regime == g),
            per_doc(candidate, where=lambda r, g=cand_regime: r.regime == g),
            rng=rng,
            n_boot=n_boot,
        )
    return out


def unmatched_docs(baseline: Sequence[EvalRow], candidate: Sequence[EvalRow]) -> int:
    """Documents measured by one checkpoint and not the other — nonzero means
    the two runs did not share a holdout and the pairing is partial."""
    return len({(r.source, r.doc_idx) for r in baseline} ^ {(r.source, r.doc_idx) for r in candidate})


def band(values: Sequence[float]) -> Band:
    finite = [v for v in values if math.isfinite(v)]
    return Band(
        low=stats.quantile(finite, 0.25),
        median=stats.quantile(finite, 0.5),
        high=stats.quantile(finite, 0.75),
    )


def records(rows: Iterable) -> list[dict[str, Any]]:
    return [dataclasses.asdict(row) for row in rows]


def rows_from(row_type, items: Iterable[Mapping[str, Any]]) -> tuple:
    return tuple(row_type(**item) for item in items)


def save_json(storage: Storage, path: str, payload: Any) -> None:
    storage.write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def load_json(storage: Storage, path: str) -> Any:
    return json.loads(storage.read_bytes(path).decode())

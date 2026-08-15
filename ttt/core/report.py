"""Table builders: rows in, strings out. Nothing here prints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.core.metrics import ALL_SOURCES, RepeatSummary, SliceSummary, SourceSummary
from ttt.core.ruler_types import RulerResult
from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY

LEFT = "left"
RIGHT = "right"


@dataclass(frozen=True)
class Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    aligns: tuple[str, ...]

    def __post_init__(self) -> None:
        width = len(self.headers)
        if len(self.aligns) != width:
            raise ValueError(f"{len(self.aligns)} alignments for {width} columns")
        bad = [row for row in self.rows if len(row) != width]
        if bad:
            raise ValueError(
                f"{len(bad)} row(s) do not have {width} cells; first is {bad[0]!r}"
            )


def ppl(value: float) -> str:
    return f"{value:.3f}"


def delta(value: float) -> str:
    return f"{value:+.3f}"


def ratio(value: float) -> str:
    return f"{value:.2e}"


def render(table: Table) -> str:
    columns = list(zip(table.headers, *table.rows)) if table.rows else [
        (h,) for h in table.headers
    ]
    widths = [max(len(cell) for cell in column) for column in columns]
    lines = [_render_row(table.headers, widths, table.aligns)]
    lines.extend(_render_row(row, widths, table.aligns) for row in table.rows)
    return "\n".join(lines)


def _render_row(
    cells: Sequence[str], widths: Sequence[int], aligns: Sequence[str]
) -> str:
    padded = [
        cell.ljust(width) if align == LEFT else cell.rjust(width)
        for cell, width, align in zip(cells, widths, aligns)
    ]
    return "  ".join(padded).rstrip()


def per_source_table(summaries: Sequence[SourceSummary]) -> Table:
    """3-mode or 5-mode depending on whether a seeded regime was measured."""
    seeded = any(CARRY in s.ppl_by_regime for s in summaries)

    headers = ["source", "n_docs", "n_tok"]
    aligns = [LEFT, RIGHT, RIGHT]
    if seeded:
        headers += [
            "carry", "cold-c", "cold-co", "lora", "fresh",
            "Δlora", "Δwithin", "Δbetween", "Δseed",
        ]
        aligns += [RIGHT] * 9
    else:
        headers += ["cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween"]
        aligns += [RIGHT] * 7

    rows = []
    for summary in summaries:
        by_regime = summary.ppl_by_regime
        cells = [summary.source, str(summary.n_docs), f"{summary.n_tokens:,d}"]
        if seeded:
            cells.append(ppl(by_regime.get(CARRY, float("nan"))))
        cells += [
            ppl(by_regime.get(COLD_CARRY, float("nan"))),
            ppl(by_regime.get(COLD_CARRY_OFF, float("nan"))),
            ppl(by_regime.get(LORA_ONLY, float("nan"))),
            ppl(by_regime.get(FRESH, float("nan"))),
            delta(summary.gaps.lora),
            delta(summary.gaps.within),
            delta(summary.gaps.between),
        ]
        if seeded:
            cells.append(delta(summary.gaps.seed))
        rows.append(tuple(cells))

    return Table(headers=tuple(headers), rows=tuple(rows), aligns=tuple(aligns))


def slice_gap_table(summaries: Sequence[SliceSummary]) -> Table:
    """Same columns as `per_source_table`, keyed by distance into the
    document instead of source — a win concentrated at low slice indices and
    fading at high ones is a local-adaptation win, not a long-range one."""
    seeded = any(CARRY in s.ppl_by_regime for s in summaries)

    headers = ["slice", "n_docs", "n_tok"]
    aligns = [RIGHT, RIGHT, RIGHT]
    if seeded:
        headers += [
            "carry", "cold-c", "cold-co", "lora", "fresh",
            "Δlora", "Δwithin", "Δbetween", "Δseed",
        ]
        aligns += [RIGHT] * 9
    else:
        headers += ["cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween"]
        aligns += [RIGHT] * 7

    rows = []
    for summary in sorted(summaries, key=lambda s: s.slice_index):
        by_regime = summary.ppl_by_regime
        cells = [str(summary.slice_index), str(summary.n_docs), f"{summary.n_tokens:,d}"]
        if seeded:
            cells.append(ppl(by_regime.get(CARRY, float("nan"))))
        cells += [
            ppl(by_regime.get(COLD_CARRY, float("nan"))),
            ppl(by_regime.get(COLD_CARRY_OFF, float("nan"))),
            ppl(by_regime.get(LORA_ONLY, float("nan"))),
            ppl(by_regime.get(FRESH, float("nan"))),
            delta(summary.gaps.lora),
            delta(summary.gaps.within),
            delta(summary.gaps.between),
        ]
        if seeded:
            cells.append(delta(summary.gaps.seed))
        rows.append(tuple(cells))

    return Table(headers=tuple(headers), rows=tuple(rows), aligns=tuple(aligns))


def repeat_table(summaries: Sequence[RepeatSummary]) -> Table:
    """Same columns as `per_source_table`, one row per (source, repeat),
    `ALL_SOURCES` last within each block. Only `cold-c` moves across a
    source's own repeats; the rest are that source's flat reference line."""
    seeded = any(CARRY in s.ppl_by_regime for s in summaries)

    headers = ["source", "repeat", "n_docs", "n_tok"]
    aligns = [LEFT, RIGHT, RIGHT, RIGHT]
    if seeded:
        headers += [
            "carry", "cold-c", "cold-co", "lora", "fresh",
            "Δlora", "Δwithin", "Δbetween", "Δseed",
        ]
        aligns += [RIGHT] * 9
    else:
        headers += ["cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween"]
        aligns += [RIGHT] * 7
    headers.append("state/W0")
    aligns.append(RIGHT)

    ordering = sorted(
        summaries, key=lambda s: (s.source == ALL_SOURCES, s.source, s.repeat)
    )
    rows = []
    for s in ordering:
        by_regime = s.ppl_by_regime
        cells = [s.source, str(s.repeat), str(s.n_docs), f"{s.n_tokens:,d}"]
        if seeded:
            cells.append(ppl(by_regime.get(CARRY, float("nan"))))
        cells += [
            ppl(by_regime.get(COLD_CARRY, float("nan"))),
            ppl(by_regime.get(COLD_CARRY_OFF, float("nan"))),
            ppl(by_regime.get(LORA_ONLY, float("nan"))),
            ppl(by_regime.get(FRESH, float("nan"))),
            delta(s.gaps.lora),
            delta(s.gaps.within),
            delta(s.gaps.between),
        ]
        if seeded:
            cells.append(delta(s.gaps.seed))
        cells.append(ratio(s.state_ratio))
        rows.append(tuple(cells))

    return Table(headers=tuple(headers), rows=tuple(rows), aligns=tuple(aligns))


def ruler_table(results: Sequence[RulerResult]) -> Table:
    """One row per task x length, one column per regime, mean score."""
    regimes = sorted({r.regime for r in results})
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for r in results:
        grouped.setdefault((r.task, r.length_bucket), {}).setdefault(
            r.regime, []
        ).append(r.score)

    rows = []
    for (task, length), by_regime in sorted(grouped.items()):
        cells = [task, str(length)]
        cells += [
            f"{sum(scores) / len(scores):.3f}"
            if (scores := by_regime.get(regime))
            else "n/a"
            for regime in regimes
        ]
        rows.append(tuple(cells))

    return Table(
        headers=("task", "length", *regimes),
        rows=tuple(rows),
        aligns=(LEFT, RIGHT, *([RIGHT] * len(regimes))),
    )


@dataclass(frozen=True)
class CompositionRow:
    """`no_carry` docs become single-item sessions (S_0 = 0, no adaptation)."""

    source: str
    no_carry_docs: int
    carry_docs: int
    items: int


def composition_table(rows: Sequence[CompositionRow]) -> Table:
    body = [
        (row.source, str(row.no_carry_docs), str(row.carry_docs), str(row.items))
        for row in sorted(rows, key=lambda r: r.source)
    ]
    body.append(
        (
            "TOTAL",
            str(sum(r.no_carry_docs for r in rows)),
            str(sum(r.carry_docs for r in rows)),
            str(sum(r.items for r in rows)),
        )
    )
    return Table(
        headers=("source", "no-carry docs", "carry docs", "total items"),
        rows=tuple(body),
        aligns=(LEFT, RIGHT, RIGHT, RIGHT),
    )

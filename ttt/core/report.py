"""Table builders: rows in, strings out. Nothing here prints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.core.metrics import SourceSummary
from ttt.core.stats import Estimate
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY

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
    headers = ("source", "n_docs", "n_tok", "cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween")
    aligns = (LEFT, RIGHT, RIGHT) + (RIGHT,) * 7
    rows = []
    for summary in summaries:
        by_regime = summary.ppl_by_regime
        rows.append((
            summary.source,
            str(summary.n_docs),
            f"{summary.n_tokens:,d}",
            ppl(by_regime.get(COLD_CARRY, float("nan"))),
            ppl(by_regime.get(COLD_CARRY_OFF, float("nan"))),
            ppl(by_regime.get(LORA_ONLY, float("nan"))),
            ppl(by_regime.get(FRESH, float("nan"))),
            delta(summary.gaps.lora),
            delta(summary.gaps.within),
            delta(summary.gaps.between),
        ))
    return Table(headers=headers, rows=tuple(rows), aligns=aligns)


def estimate_table(estimates: Sequence[Estimate]) -> Table:
    """`*` marks an interval that excludes zero."""
    rows = tuple(
        (
            e.measure,
            e.source,
            str(e.n_docs),
            delta(e.interval.point),
            f"[{delta(e.interval.low)}, {delta(e.interval.high)}]",
            "*" if e.interval.excludes_zero else "",
            f"{100 * e.relative:+.2f}%",
            f"{e.win_rate:.2f}",
        )
        for e in estimates
    )
    return Table(
        headers=("measure", "source", "n_docs", "Δppl", "95% CI", "", "rel", "win"),
        rows=rows,
        aligns=(LEFT, LEFT, RIGHT, RIGHT, RIGHT, LEFT, RIGHT, RIGHT),
    )

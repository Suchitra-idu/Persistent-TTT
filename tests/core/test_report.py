"""Table builders. The point of these being pure is that this file exists."""

from __future__ import annotations


from tests.core import _builders as build
from ttt.core import metrics, report
from ttt.core.stats import Estimate, Interval
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY


def _summaries():
    rows = []
    for doc_idx, source in enumerate(("RedPajamaC4", "RedPajamaBook")):
        base = 20.0 + doc_idx
        rows += [
            build.eval_row(doc_idx=doc_idx, source=source, regime=FRESH, ppl=base),
            build.eval_row(
                doc_idx=doc_idx, source=source, regime=LORA_ONLY, ppl=base - 0.5
            ),
            build.eval_row(
                doc_idx=doc_idx, source=source, regime=COLD_CARRY_OFF, ppl=base - 1
            ),
            build.eval_row(
                doc_idx=doc_idx, source=source, regime=COLD_CARRY, ppl=base - 2
            ),
        ]
    return metrics.summarise_by_source(rows)


def test_the_table_has_a_column_for_every_regime_and_gap():
    table = report.per_source_table(_summaries())

    assert {"cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween"} <= set(table.headers)


def test_gaps_are_rendered_with_an_explicit_sign():
    table = report.per_source_table(_summaries())
    within = table.headers.index("Δwithin")

    assert all(row[within].startswith(("+", "-")) for row in table.rows)


def test_an_estimate_table_stars_an_interval_that_excludes_zero():
    estimate = Estimate(measure="m", source="s", n_docs=3, interval=Interval(1.0, 0.5, 1.5), win_rate=0.8)

    assert report.estimate_table([estimate]).rows[0][5] == "*"



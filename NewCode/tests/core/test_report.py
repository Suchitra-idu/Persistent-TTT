"""Table builders. The point of these being pure is that this file exists."""

from __future__ import annotations

import pytest

from tests.core import _builders as build
from ttt.core import metrics, report
from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH


def _summaries(*, seeded: bool):
    rows = []
    for doc_idx, source in enumerate(("RedPajamaC4", "RedPajamaBook")):
        base = 20.0 + doc_idx
        rows += [
            build.eval_row(doc_idx=doc_idx, source=source, regime=FRESH, ppl=base),
            build.eval_row(
                doc_idx=doc_idx, source=source, regime=COLD_CARRY_OFF, ppl=base - 1
            ),
            build.eval_row(
                doc_idx=doc_idx, source=source, regime=COLD_CARRY, ppl=base - 2
            ),
        ]
        if seeded:
            rows.append(
                build.eval_row(
                    doc_idx=doc_idx, source=source, regime=CARRY, ppl=base - 3
                )
            )
    return metrics.summarise_by_source(rows)


def test_a_table_rejects_a_row_of_the_wrong_width():
    with pytest.raises(ValueError, match="do not have 2 cells"):
        report.Table(
            headers=("a", "b"), rows=(("only-one",),), aligns=(report.LEFT,) * 2
        )


def test_a_table_rejects_a_missing_alignment():
    with pytest.raises(ValueError, match="alignments for 2 columns"):
        report.Table(headers=("a", "b"), rows=(), aligns=(report.LEFT,))


def test_a_zero_seed_eval_gets_the_three_mode_table():
    table = report.per_source_table(_summaries(seeded=False))

    assert "carry" not in table.headers
    assert "Δseed" not in table.headers


def test_a_seeded_eval_gets_the_five_mode_table():
    table = report.per_source_table(_summaries(seeded=True))

    assert "carry" in table.headers
    assert "Δseed" in table.headers


def test_one_row_per_source():
    table = report.per_source_table(_summaries(seeded=True))

    assert [row[0] for row in table.rows] == ["RedPajamaBook", "RedPajamaC4"]


def test_every_row_matches_the_header_width():
    table = report.per_source_table(_summaries(seeded=True))

    assert all(len(row) == len(table.headers) for row in table.rows)


def test_gaps_are_rendered_with_an_explicit_sign():
    table = report.per_source_table(_summaries(seeded=False))
    within = table.headers.index("Δwithin")

    assert all(row[within].startswith(("+", "-")) for row in table.rows)


def test_an_empty_summary_list_still_produces_headers():
    table = report.per_source_table([])

    assert table.rows == () and table.headers[0] == "source"


def test_the_composition_table_totals_every_column():
    rows = [
        report.CompositionRow("c4", no_carry_docs=10, carry_docs=5, items=25),
        report.CompositionRow("books", no_carry_docs=0, carry_docs=4, items=20),
    ]

    table = report.composition_table(rows)

    assert table.rows[-1] == ("TOTAL", "10", "9", "45")


def test_the_composition_table_sorts_sources():
    rows = [
        report.CompositionRow("c4", 1, 1, 2),
        report.CompositionRow("arxiv", 1, 1, 2),
    ]

    table = report.composition_table(rows)

    assert [row[0] for row in table.rows] == ["arxiv", "c4", "TOTAL"]


def test_rendering_aligns_every_column_to_its_widest_cell():
    table = report.Table(
        headers=("source", "n"),
        rows=(("a-very-long-source", "1"), ("c4", "22")),
        aligns=(report.LEFT, report.RIGHT),
    )

    lines = report.render(table).splitlines()

    # Column widths are 18 ("a-very-long-source") and 2 ("22"), joined by two
    # spaces; the left column pads right, the right column pads left.
    assert lines[0] == "source".ljust(18) + "  " + "n".rjust(2)
    assert lines[1] == "a-very-long-source" + "  " + " 1"
    assert lines[2] == "c4".ljust(18) + "  " + "22"


def test_rendering_a_headers_only_table_does_not_crash():
    table = report.Table(headers=("a", "bb"), rows=(), aligns=(report.LEFT,) * 2)

    assert report.render(table) == "a  bb"


def test_formatters_pin_their_precision():
    assert (report.ppl(12.3456), report.delta(-0.5), report.ratio(0.00123)) == (
        "12.346",
        "-0.500",
        "1.23e-03",
    )

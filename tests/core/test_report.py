"""Table builders. The point of these being pure is that this file exists."""

from __future__ import annotations

import pytest

from tests.core import _builders as build
from ttt.core import metrics, report
from ttt.core.ruler_types import RulerResult
from ttt.core.types import (
    CARRY,
    COLD_CARRY,
    COLD_CARRY_OFF,
    FRESH,
    LORA_ONLY,
    LanguageScan,
)


def _summaries(*, seeded: bool):
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


def test_a_zero_seed_eval_gets_the_four_mode_table():
    table = report.per_source_table(_summaries(seeded=False))

    assert "carry" not in table.headers
    assert "Δseed" not in table.headers
    assert "lora" in table.headers


def test_a_seeded_eval_gets_the_six_mode_table():
    table = report.per_source_table(_summaries(seeded=True))

    assert "carry" in table.headers
    assert "Δseed" in table.headers
    assert "lora" in table.headers


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


def test_the_source_table_reports_bits_per_byte():
    table = report.per_source_table(_summaries(seeded=False))

    assert "bpb" in table.headers and "Δbpb" in table.headers


def test_bpb_gap_is_positive_when_carry_costs_fewer_bits_per_byte():
    rows = [
        build.eval_row(doc_idx=0, source="c4", regime=FRESH, ppl=20.0, n_bytes=1000),
        build.eval_row(doc_idx=0, source="c4", regime=LORA_ONLY, ppl=19.0, n_bytes=1000),
        build.eval_row(
            doc_idx=0, source="c4", regime=COLD_CARRY_OFF, ppl=15.0, n_bytes=1000
        ),
        build.eval_row(doc_idx=0, source="c4", regime=COLD_CARRY, ppl=10.0, n_bytes=1000),
    ]
    table = report.per_source_table(metrics.summarise_by_source(rows))
    gap = float(table.rows[0][table.headers.index("Δbpb")])

    assert gap > 0


def _slice_summaries(*, seeded: bool):
    rows = []
    for index in (0, 1):
        base = 20.0 + index
        rows += [
            build.slice_row(slice_index=index, regime=FRESH, ppl=base),
            build.slice_row(slice_index=index, regime=LORA_ONLY, ppl=base - 0.5),
            build.slice_row(slice_index=index, regime=COLD_CARRY_OFF, ppl=base - 1),
            build.slice_row(slice_index=index, regime=COLD_CARRY, ppl=base - 2),
        ]
        if seeded:
            rows.append(build.slice_row(slice_index=index, regime=CARRY, ppl=base - 3))
    return metrics.summarise_by_slice_index(rows)


def test_the_slice_table_is_seeded_or_not_the_same_way_as_per_source():
    assert "carry" not in report.slice_gap_table(_slice_summaries(seeded=False)).headers
    assert "carry" in report.slice_gap_table(_slice_summaries(seeded=True)).headers


def test_the_slice_table_is_sorted_by_index_ascending():
    summaries = _slice_summaries(seeded=True)[::-1]

    table = report.slice_gap_table(summaries)

    assert [row[0] for row in table.rows] == ["0", "1"]


def test_an_empty_slice_summary_list_still_produces_headers():
    table = report.slice_gap_table([])

    assert table.rows == () and table.headers[0] == "slice"


def test_the_slice_table_reports_bits_per_byte_too():
    table = report.slice_gap_table(_slice_summaries(seeded=False))

    assert "bpb" in table.headers and "Δbpb" in table.headers


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


def test_the_ruler_table_has_one_row_per_task_and_length():
    results = [
        RulerResult(task="niah_single", length_bucket=4096, regime=FRESH, score=0.5, n_new_tokens=1),
        RulerResult(task="niah_single", length_bucket=4096, regime=COLD_CARRY, score=1.0, n_new_tokens=1),
        RulerResult(task="vt", length_bucket=4096, regime=FRESH, score=0.0, n_new_tokens=1),
    ]

    table = report.ruler_table(results)

    assert [row[0] for row in table.rows] == ["niah_single", "vt"]


def test_the_ruler_table_averages_repeated_examples():
    results = [
        RulerResult(task="vt", length_bucket=4096, regime=FRESH, score=1.0, n_new_tokens=1),
        RulerResult(task="vt", length_bucket=4096, regime=FRESH, score=0.0, n_new_tokens=1),
    ]

    table = report.ruler_table(results)

    assert table.rows[0] == ("vt", "4096", "0.500")


def test_the_ruler_table_marks_a_regime_a_task_was_not_run_under():
    results = [
        RulerResult(task="niah_single", length_bucket=4096, regime=FRESH, score=1.0, n_new_tokens=1),
        RulerResult(task="vt", length_bucket=4096, regime=COLD_CARRY, score=1.0, n_new_tokens=1),
    ]

    table = report.ruler_table(results)

    assert table.rows[0] == ("niah_single", "4096", "n/a", "1.000")


def test_the_language_scan_table_sorts_worst_bpb_first():
    scans = [
        LanguageScan(code="aa", name="A", n_docs=5, n_tokens=100, n_bytes=100, ppl=10.0, bpb=1.0),
        LanguageScan(code="bb", name="B", n_docs=5, n_tokens=100, n_bytes=100, ppl=50.0, bpb=3.0),
    ]

    table = report.language_scan_table(scans)

    assert [row[0] for row in table.rows] == ["bb", "aa"]


def test_formatters_pin_their_precision():
    assert (report.ppl(12.3456), report.delta(-0.5), report.ratio(0.00123)) == (
        "12.346",
        "-0.500",
        "1.23e-03",
    )

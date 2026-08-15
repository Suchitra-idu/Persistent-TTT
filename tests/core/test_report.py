"""Table builders. The point of these being pure is that this file exists."""

from __future__ import annotations

import pytest

from tests.core import _builders as build
from ttt.core import metrics, report
from ttt.core.ruler_types import RulerResult
from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY


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


def _repeat_summaries(*, seeded: bool = False):
    rows = [
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=FRESH, repeat=0, ppl=25.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=FRESH, repeat=1, ppl=25.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=LORA_ONLY, repeat=0, ppl=22.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=LORA_ONLY, repeat=1, ppl=22.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=COLD_CARRY_OFF, repeat=0, ppl=20.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=COLD_CARRY_OFF, repeat=1, ppl=20.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=COLD_CARRY, repeat=0, ppl=10.0),
        build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=COLD_CARRY, repeat=1, ppl=8.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=FRESH, repeat=0, ppl=30.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=FRESH, repeat=1, ppl=30.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=LORA_ONLY, repeat=0, ppl=26.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=LORA_ONLY, repeat=1, ppl=26.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=COLD_CARRY_OFF, repeat=0, ppl=22.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=COLD_CARRY_OFF, repeat=1, ppl=22.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=COLD_CARRY, repeat=0, ppl=20.0),
        build.repeat_row(doc_idx=1, source="RedPajamaBook", regime=COLD_CARRY, repeat=1, ppl=16.0),
    ]
    if seeded:
        rows += [
            build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=CARRY, repeat=0, ppl=9.0),
            build.repeat_row(doc_idx=0, source="RedPajamaC4", regime=CARRY, repeat=1, ppl=7.0),
        ]
    return metrics.summarise_by_repeat(rows)


def test_the_repeat_table_has_one_row_per_source_and_repeat_plus_all():
    table = report.repeat_table(_repeat_summaries())

    assert {row[0] for row in table.rows} == {"RedPajamaC4", "RedPajamaBook", "ALL"}
    assert len(table.rows) == 6


def test_the_repeat_table_puts_all_last_within_each_source_block():
    table = report.repeat_table(_repeat_summaries())

    assert [row[0] for row in table.rows][-2:] == ["ALL", "ALL"]


def test_the_repeat_table_orders_repeats_ascending_within_a_source():
    table = report.repeat_table(_repeat_summaries())

    c4_repeats = [row[1] for row in table.rows if row[0] == "RedPajamaC4"]
    assert c4_repeats == ["0", "1"]


def test_an_unseeded_repeat_table_has_no_carry_column():
    table = report.repeat_table(_repeat_summaries(seeded=False))

    assert "carry" not in table.headers
    assert "Δseed" not in table.headers


def test_a_seeded_repeat_table_gets_the_carry_and_seed_columns():
    table = report.repeat_table(_repeat_summaries(seeded=True))

    assert "carry" in table.headers
    assert "Δseed" in table.headers


def test_the_repeat_table_carries_the_full_regime_columns():
    table = report.repeat_table(_repeat_summaries())

    for header in ("cold-c", "cold-co", "lora", "fresh", "Δlora", "Δwithin", "Δbetween"):
        assert header in table.headers


def test_the_repeat_gaps_carry_an_explicit_sign():
    table = report.repeat_table(_repeat_summaries())
    delta_col = table.headers.index("Δwithin")

    assert all(row[delta_col].startswith(("+", "-")) for row in table.rows)


def test_the_repeat_table_reports_the_carry_state_ratio():
    table = report.repeat_table(_repeat_summaries())

    assert "state/W0" in table.headers


def test_an_empty_repeat_summary_list_still_produces_headers():
    table = report.repeat_table([])

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


def test_formatters_pin_their_precision():
    assert (report.ppl(12.3456), report.delta(-0.5), report.ratio(0.00123)) == (
        "12.346",
        "-0.500",
        "1.23e-03",
    )

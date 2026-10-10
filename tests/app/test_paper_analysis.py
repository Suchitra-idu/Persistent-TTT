from __future__ import annotations

import pytest

from tests.app.test_evidence import REGIME_PPL, by, eval_rows, kw, repeat_rows
from ttt.app import paper_analysis
from ttt.app.paper_analysis import VS_MATCHED
from ttt.core.metrics import ALL_SOURCES
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY, SliceRow

BASE_PPL = {**REGIME_PPL, FRESH: 25.0}


SLICE_PPL = {LORA_ONLY: 16.0, COLD_CARRY_OFF: 15.0, COLD_CARRY: 14.0}


def slice_rows(n_docs=2, n_slices=3):
    def ppl(regime, index):
        return SLICE_PPL[COLD_CARRY_OFF] if regime == COLD_CARRY and index == 0 else SLICE_PPL[regime]

    return tuple(
        SliceRow(doc_idx=d, source="alpha", regime=regime, slice_index=i, n_tokens=10,
                 ppl=ppl(regime, i))
        for d in range(n_docs)
        for i in range(n_slices)
        for regime in SLICE_PPL
    )


def test_rebasing_takes_fresh_from_the_base():
    rebased = paper_analysis.rebased(eval_rows(), eval_rows(ppl=BASE_PPL))

    assert {r.ppl for r in rebased if r.regime == FRESH} == {25.0}


def test_rebasing_keeps_every_other_regime():
    rows = eval_rows()

    rebased = paper_analysis.rebased(rows, eval_rows(ppl=BASE_PPL))

    assert sorted((r.regime, r.ppl) for r in rebased if r.regime != FRESH) == sorted(
        (r.regime, r.ppl) for r in rows if r.regime != FRESH
    )


def test_rebasing_against_a_different_holdout_is_rejected():
    with pytest.raises(ValueError, match="do not share a holdout"):
        paper_analysis.rebased(eval_rows(n_docs=3), eval_rows(n_docs=2, ppl=BASE_PPL))


def test_matched_compares_the_ttt_path_against_the_controls_lora_path():
    control = eval_rows(ppl={**REGIME_PPL, LORA_ONLY: 15.0})

    estimates = paper_analysis.matched_estimates(control, eval_rows(), **kw())

    assert by(estimates, VS_MATCHED).interval.point == pytest.approx(1.0)


def test_position_estimates_report_each_slice_pooled():
    estimates = paper_analysis.position_estimates(slice_rows(), **kw())

    assert {(i, e.measure) for i, e in estimates} == {
        (i, m) for i in range(3) for m in ("within", "between_slices")
    }
    assert {e.source for _, e in estimates} == {ALL_SOURCES}


def test_the_first_slice_has_no_between_slices_gap():
    estimates = dict(
        ((i, e.measure), e) for i, e in paper_analysis.position_estimates(slice_rows(), **kw())
    )

    assert estimates[(0, "between_slices")].interval.point == 0.0
    assert estimates[(1, "between_slices")].interval.point == pytest.approx(1.0)


def test_saturation_counts_rows_above_tau_over_w0():
    saturation = paper_analysis.clip_saturation([0.5, 4.0, 8.0], clip_tau=20.0, w0_norm=20.0)

    assert (saturation.threshold, saturation.fraction_clipped) == (1.0, pytest.approx(2 / 3))
    assert saturation.min_factor == 0.5


def test_saturation_needs_rows():
    with pytest.raises(ValueError, match="no state ratios"):
        paper_analysis.clip_saturation([], clip_tau=20.0, w0_norm=20.0)


def test_state_points_group_rows_by_key():
    points = paper_analysis.state_points(
        repeat_rows(), group=lambda r: r.source, position=lambda r: r.repeat
    )

    assert [(p.group, p.position, p.n_rows) for p in points] == [
        ("alpha", 0, 2), ("alpha", 1, 2), ("alpha", 2, 2)
    ]
    assert points[2].state_ratio.median == pytest.approx(0.3)

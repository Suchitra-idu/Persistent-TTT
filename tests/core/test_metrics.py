"""Perplexity aggregation and the gap decomposition."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _builders as build
from tests.core._oracles import token_weighted_ppl as oracle_ppl
from ttt.core import metrics
from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY

ppl_values = st.floats(min_value=1.01, max_value=1e4, allow_nan=False)
token_counts = st.integers(min_value=1, max_value=20_000)
rows_strategy = st.lists(st.tuples(token_counts, ppl_values), min_size=1, max_size=20)


@given(pairs=rows_strategy)
@settings(max_examples=60, deadline=None)
def test_token_weighted_ppl_matches_the_oracle(pairs):
    result = metrics.token_weighted_ppl(build.ppl_rows(*pairs))

    assert result == pytest.approx(oracle_ppl(list(pairs)), rel=1e-12)


@given(pairs=rows_strategy)
@settings(max_examples=60, deadline=None)
def test_the_aggregate_lies_between_the_best_and_worst_slice(pairs):
    result = metrics.token_weighted_ppl(build.ppl_rows(*pairs))

    assert min(p for _, p in pairs) - 1e-9 <= result <= max(p for _, p in pairs) + 1e-9


@given(value=ppl_values, n=st.integers(min_value=1, max_value=10))
def test_identical_slices_aggregate_to_their_common_value(value, n):
    rows = build.ppl_rows(*[(100, value)] * n)

    assert metrics.token_weighted_ppl(rows) == pytest.approx(value, rel=1e-12)


def test_a_long_slice_dominates_a_short_one():
    """The mistake this guards against is averaging perplexities directly."""
    rows = build.ppl_rows((10_000, 10.0), (1, 1000.0))

    assert metrics.token_weighted_ppl(rows) < 11.0


def test_zero_tokens_aggregate_to_nan_rather_than_dividing_by_zero():
    assert math.isnan(metrics.token_weighted_ppl(build.ppl_rows((0, 5.0))))


def test_bits_per_byte_matches_a_hand_computed_value():
    from ttt.core.types import PplRow

    nll_per_token = math.log(10.0)
    row = PplRow(n_tokens=100, ppl=10.0, n_bytes=250)

    expected = (nll_per_token * 100) / math.log(2) / 250
    assert metrics.bits_per_byte([row]) == pytest.approx(expected)


def test_bits_per_byte_is_lower_for_a_more_efficient_tokenizer():
    """Same ppl, same tokens, more bytes per token => fewer bits per byte."""
    from ttt.core.types import PplRow

    efficient = PplRow(n_tokens=100, ppl=10.0, n_bytes=400)
    fragmented = PplRow(n_tokens=100, ppl=10.0, n_bytes=100)

    assert metrics.bits_per_byte([efficient]) < metrics.bits_per_byte([fragmented])


def test_zero_bytes_aggregate_to_nan_rather_than_dividing_by_zero():
    assert math.isnan(metrics.bits_per_byte(build.ppl_rows((100, 5.0))))


@given(values=st.lists(ppl_values, min_size=1, max_size=12))
@settings(max_examples=40, deadline=None)
def test_the_document_aggregate_is_the_geometric_mean(values):
    result = metrics.geometric_mean_ppl(values)

    expected = math.exp(sum(math.log(v) for v in values) / len(values))
    assert result == pytest.approx(expected, rel=1e-12)


def test_the_document_aggregate_weights_documents_equally():
    assert metrics.geometric_mean_ppl([4.0, 16.0]) == pytest.approx(8.0)


@given(
    fresh=ppl_values,
    lora_only=ppl_values,
    cold_carry_off=ppl_values,
    cold_carry=ppl_values,
    carry=ppl_values,
)
@settings(max_examples=100, deadline=None)
def test_six_mode_gaps_decompose_additively(
    fresh, lora_only, cold_carry_off, cold_carry, carry
):
    gaps = metrics.gap_decomposition(
        fresh=fresh,
        lora_only=lora_only,
        cold_carry_off=cold_carry_off,
        cold_carry=cold_carry,
        carry=carry,
    )

    assert gaps.is_additive


@given(
    fresh=ppl_values, lora_only=ppl_values, cold_carry_off=ppl_values, cold_carry=ppl_values
)
@settings(max_examples=100, deadline=None)
def test_four_mode_gaps_decompose_additively(fresh, lora_only, cold_carry_off, cold_carry):
    gaps = metrics.gap_decomposition(
        fresh=fresh, lora_only=lora_only, cold_carry_off=cold_carry_off, cold_carry=cold_carry
    )

    assert gaps.is_additive


def test_a_zero_seed_eval_reports_no_seed_benefit():
    gaps = metrics.gap_decomposition(
        fresh=20.0, lora_only=19.0, cold_carry_off=18.0, cold_carry=16.0
    )

    assert gaps.seed == 0.0


def test_each_gap_measures_exactly_one_mechanism():
    gaps = metrics.gap_decomposition(
        fresh=20.0, lora_only=19.0, cold_carry_off=18.0, cold_carry=16.0, carry=15.0
    )

    assert (gaps.lora, gaps.within, gaps.between, gaps.seed, gaps.total) == (
        1.0,
        1.0,
        2.0,
        1.0,
        5.0,
    )


def test_a_mechanism_that_hurts_shows_a_negative_gap():
    gaps = metrics.gap_decomposition(
        fresh=20.0, lora_only=19.0, cold_carry_off=22.0, cold_carry=16.0
    )

    assert gaps.within < 0.0 and gaps.is_additive


def test_the_seed_gap_is_positive_when_the_trained_carrier_helps():
    assert metrics.seed_gap(cold_carry=16.0, seeded_carry=15.0) == pytest.approx(1.0)


def _five_regime_rows(source: str, doc_idx: int, base: float):
    return [
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
        build.eval_row(doc_idx=doc_idx, source=source, regime=CARRY, ppl=base - 3),
    ]


def test_summaries_are_one_row_per_source_in_sorted_order():
    rows = _five_regime_rows("github", 0, 20.0) + _five_regime_rows("c4", 1, 30.0)

    summaries = metrics.summarise_by_source(rows)

    assert [s.source for s in summaries] == ["c4", "github"]


def test_a_summary_counts_distinct_documents():
    rows = _five_regime_rows("c4", 0, 20.0) + _five_regime_rows("c4", 1, 22.0)

    (summary,) = metrics.summarise_by_source(rows)

    assert summary.n_docs == 2


def test_a_summary_decomposes_that_sources_gaps():
    (summary,) = metrics.summarise_by_source(_five_regime_rows("c4", 0, 20.0))

    assert summary.gaps.is_additive
    assert summary.gaps.total == pytest.approx(3.0)


def test_a_summary_aggregates_each_regime_by_tokens():
    rows = [
        build.eval_row(doc_idx=0, source="c4", regime=COLD_CARRY, n_tokens=3000, ppl=8.0),
        build.eval_row(doc_idx=1, source="c4", regime=COLD_CARRY, n_tokens=1000, ppl=32.0),
    ]

    (summary,) = metrics.summarise_by_source(rows)

    expected = math.exp((math.log(8.0) * 3000 + math.log(32.0) * 1000) / 4000)
    assert summary.ppl_by_regime[COLD_CARRY] == pytest.approx(expected)
    assert summary.n_tokens == 4000


def test_a_summary_aggregates_bytes_the_same_way_as_tokens():
    rows = [
        build.eval_row(doc_idx=0, source="c4", regime=COLD_CARRY, n_bytes=3000),
        build.eval_row(doc_idx=1, source="c4", regime=COLD_CARRY, n_bytes=1000),
    ]

    (summary,) = metrics.summarise_by_source(rows)

    assert summary.n_bytes == 4000


def test_a_summary_carries_bits_per_byte_by_regime():
    rows = _five_regime_rows("c4", 0, 20.0)
    rows = [
        build.eval_row(
            doc_idx=r.doc_idx, source=r.source, regime=r.regime,
            n_tokens=r.n_tokens, ppl=r.ppl, n_bytes=4000,
        )
        for r in rows
    ]

    (summary,) = metrics.summarise_by_source(rows)

    assert set(summary.bpb_by_regime) == {FRESH, LORA_ONLY, COLD_CARRY_OFF, COLD_CARRY, CARRY}


def test_summarising_nothing_yields_nothing():
    assert metrics.summarise_by_source([]) == ()


def _five_regime_slices(slice_index: int, base: float):
    return [
        build.slice_row(slice_index=slice_index, regime=FRESH, ppl=base),
        build.slice_row(slice_index=slice_index, regime=LORA_ONLY, ppl=base - 0.5),
        build.slice_row(slice_index=slice_index, regime=COLD_CARRY_OFF, ppl=base - 1),
        build.slice_row(slice_index=slice_index, regime=COLD_CARRY, ppl=base - 2),
        build.slice_row(slice_index=slice_index, regime=CARRY, ppl=base - 3),
    ]


def test_slice_summaries_are_one_row_per_index_in_ascending_order():
    slices = _five_regime_slices(2, 20.0) + _five_regime_slices(0, 30.0)

    summaries = metrics.summarise_by_slice_index(slices)

    assert [s.slice_index for s in summaries] == [0, 2]


def test_a_slice_summary_decomposes_that_slices_gaps():
    (summary,) = metrics.summarise_by_slice_index(_five_regime_slices(0, 20.0))

    assert summary.gaps.is_additive
    assert summary.gaps.total == pytest.approx(3.0)


def test_a_slice_summary_counts_distinct_documents():
    slices = [
        build.slice_row(doc_idx=0, slice_index=0, regime=COLD_CARRY),
        build.slice_row(doc_idx=1, slice_index=0, regime=COLD_CARRY),
    ]

    (summary,) = metrics.summarise_by_slice_index(slices)

    assert summary.n_docs == 2


def test_summarising_no_slices_yields_nothing():
    assert metrics.summarise_by_slice_index([]) == ()


def test_an_unclipped_step_reports_a_ratio_of_one():
    assert metrics.clip_ratio(total_norm=1.0, max_grad_norm=10.0) == 1.0


def test_a_clipped_step_reports_how_far_it_was_scaled_down():
    assert metrics.clip_ratio(total_norm=20.0, max_grad_norm=10.0) == pytest.approx(0.5)


def test_a_zero_gradient_does_not_divide_by_zero():
    assert metrics.clip_ratio(total_norm=0.0, max_grad_norm=10.0) == 1.0


@given(nll=st.floats(min_value=-50.0, max_value=50.0))
def test_perplexity_is_the_exponentiated_loss(nll):
    assert metrics.perplexity(nll) == pytest.approx(math.exp(nll))


def test_a_diverged_loss_saturates_rather_than_overflowing():
    assert math.isinf(metrics.perplexity(1e6))


def test_a_nonfinite_loss_stays_nonfinite():
    assert math.isnan(metrics.perplexity(float("nan")))

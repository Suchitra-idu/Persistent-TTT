"""Bootstrap intervals, quantiles and rank correlation over documents."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _oracles
from ttt.core import stats

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
ppls = st.lists(st.floats(min_value=1.01, max_value=1e4), min_size=2, max_size=30)
levels = st.floats(min_value=0.0, max_value=1.0)


def identity_resamples(n: int, count: int = 5) -> list[list[int]]:
    return [list(range(n))] * count


def every_single_index(n: int) -> list[list[int]]:
    return [[i] * n for i in range(n)]


@given(values=st.lists(finite, min_size=1, max_size=40), q=levels)
@settings(max_examples=80, deadline=None)
def test_quantile_matches_numpy(values, q):
    assert stats.quantile(values, q) == pytest.approx(_oracles.quantile(values, q), abs=1e-6)


def test_quantile_of_nothing_is_nan():
    assert math.isnan(stats.quantile([], 0.5))


def test_quantile_rejects_a_q_outside_the_unit_interval():
    with pytest.raises(ValueError):
        stats.quantile([1.0], 1.5)


@given(ref=ppls)
@settings(max_examples=40, deadline=None)
def test_identical_arms_have_a_zero_gap_and_a_zero_width_interval(ref):
    interval = stats.paired_gap(ref, ref, every_single_index(len(ref)))

    assert (interval.point, interval.low, interval.high) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


@given(ref=ppls, factor=st.floats(min_value=0.5, max_value=0.95))
@settings(max_examples=40, deadline=None)
def test_a_uniformly_lower_treated_arm_has_an_interval_above_zero(ref, factor):
    treated = [p * factor for p in ref]

    interval = stats.paired_gap(ref, treated, every_single_index(len(ref)))

    assert interval.low > 0.0


@given(ref=ppls)
@settings(max_examples=40, deadline=None)
def test_resampling_every_document_once_reproduces_the_point_estimate(ref):
    treated = [p * 0.9 for p in ref]

    interval = stats.paired_gap(ref, treated, identity_resamples(len(ref)))

    assert interval.low == pytest.approx(interval.point) == pytest.approx(interval.high)


@given(ref=ppls)
@settings(max_examples=40, deadline=None)
def test_the_point_is_the_gap_between_geometric_means(ref):
    treated = list(reversed(ref))

    interval = stats.paired_gap(ref, treated, identity_resamples(len(ref)))

    assert interval.point == pytest.approx(stats.ppl_gap(ref, treated))


@given(ref=ppls, level=st.floats(min_value=0.5, max_value=0.99))
@settings(max_examples=40, deadline=None)
def test_the_interval_is_ordered(ref, level):
    treated = [p * (1.0 + (i % 3 - 1) * 0.1) for i, p in enumerate(ref)]

    interval = stats.paired_gap(ref, treated, every_single_index(len(ref)), level=level)

    assert interval.low <= interval.high


def test_paired_arms_of_different_length_are_rejected():
    with pytest.raises(ValueError):
        stats.paired_gap([2.0, 3.0], [2.0], [[0, 1]])


def test_bootstrap_rejects_a_level_outside_the_open_unit_interval():
    with pytest.raises(ValueError):
        stats.bootstrap(lambda idx: 0.0, 2, [[0, 1]], level=1.0)


def test_bootstrap_with_no_resamples_has_a_nan_interval():
    interval = stats.bootstrap(lambda idx: 1.0, 2, [])

    assert math.isnan(interval.low) and math.isnan(interval.high)


@pytest.mark.parametrize(
    ("low", "high", "excludes"),
    [(0.1, 0.5, True), (-0.5, -0.1, True), (-0.1, 0.1, False), (0.0, 0.2, False)],
)
def test_an_interval_excludes_zero_only_when_strictly_one_sided(low, high, excludes):
    assert stats.Interval(point=0.0, low=low, high=high).excludes_zero is excludes


def test_win_rate_counts_treated_wins_and_halves_ties():
    assert stats.win_rate([2.0, 2.0, 2.0, 2.0], [1.0, 3.0, 2.0, 1.5]) == pytest.approx(0.625)


def test_win_rate_of_nothing_is_nan():
    assert math.isnan(stats.win_rate([], []))


def test_tied_values_share_their_average_rank():
    assert stats.ranks([10.0, 20.0, 20.0, 5.0]) == [2.0, 3.5, 3.5, 1.0]


@given(
    pairs=st.lists(
        st.tuples(st.integers(-1000, 1000), st.integers(-1000, 1000)),
        min_size=3, max_size=20, unique_by=(lambda p: p[0], lambda p: p[1]),
    )
)
@settings(max_examples=60, deadline=None)
def test_spearman_matches_the_closed_form_without_ties(pairs):
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]

    assert stats.spearman(xs, ys) == pytest.approx(_oracles.spearman_without_ties(xs, ys), abs=1e-9)


@given(xs=st.lists(st.integers(-100, 100), min_size=3, max_size=20, unique=True))
@settings(max_examples=40, deadline=None)
def test_spearman_is_one_under_any_monotone_transform(xs):
    values = [float(x) for x in xs]

    assert stats.spearman(values, [math.exp(v / 50.0) for v in values]) == pytest.approx(1.0)


def test_spearman_below_three_points_is_nan():
    assert math.isnan(stats.spearman([1.0, 2.0], [2.0, 1.0]))


def test_spearman_against_a_constant_is_nan():
    assert math.isnan(stats.spearman([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]))

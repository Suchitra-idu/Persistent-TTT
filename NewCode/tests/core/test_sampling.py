"""Sampling policies: every source represented, nothing invented, seed-stable."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core._builders import FakeRng
from ttt.core import sampling

SOURCES = ["c4"] * 20 + ["github"] * 5 + ["books"] * 2

labels = st.lists(
    st.sampled_from(["c4", "github", "books", "arxiv"]),
    min_size=1,
    max_size=60,
)


@given(n=st.integers(min_value=0, max_value=200), seed=st.integers(0, 10_000))
def test_a_permutation_is_a_rearrangement_of_every_index(n, seed):
    order = sampling.permutation(n, FakeRng(seed))

    assert sorted(order) == list(range(n))


def test_a_permutation_is_seed_stable():
    assert sampling.permutation(50, FakeRng(3)) == sampling.permutation(50, FakeRng(3))


@given(
    n_available=st.integers(min_value=0, max_value=100),
    n_wanted=st.integers(min_value=0, max_value=100),
    seed=st.integers(0, 10_000),
)
def test_uniform_sampling_returns_distinct_in_range_indices(
    n_available, n_wanted, seed
):
    picked = sampling.uniform_indices(n_available, n_wanted, FakeRng(seed))

    assert len(set(picked)) == len(picked)
    assert all(0 <= i < n_available for i in picked)


@given(
    n_available=st.integers(min_value=0, max_value=100),
    n_wanted=st.integers(min_value=0, max_value=100),
    seed=st.integers(0, 10_000),
)
def test_uniform_sampling_never_exceeds_the_pool(n_available, n_wanted, seed):
    picked = sampling.uniform_indices(n_available, n_wanted, FakeRng(seed))

    assert len(picked) == min(n_wanted, n_available)


@given(source_labels=labels, n=st.integers(min_value=0, max_value=10))
@settings(max_examples=50, deadline=None)
def test_n_per_source_never_takes_more_than_n_from_any_source(source_labels, n):
    picked, _ = sampling.n_per_source_indices(source_labels, n, FakeRng(1))

    counts: dict[str, int] = {}
    for index in picked:
        counts[source_labels[index]] = counts.get(source_labels[index], 0) + 1
    assert all(c <= n for c in counts.values())


@given(source_labels=labels, n=st.integers(min_value=1, max_value=10))
@settings(max_examples=50, deadline=None)
def test_every_source_present_in_the_pool_is_represented(source_labels, n):
    picked, _ = sampling.n_per_source_indices(source_labels, n, FakeRng(1))

    assert {source_labels[i] for i in picked} == set(source_labels)


@given(source_labels=labels, n=st.integers(min_value=0, max_value=10))
@settings(max_examples=50, deadline=None)
def test_n_per_source_picks_are_distinct(source_labels, n):
    picked, _ = sampling.n_per_source_indices(source_labels, n, FakeRng(2))

    assert len(set(picked)) == len(picked)


def test_a_scarce_source_reports_a_shortfall_instead_of_failing():
    picked, shortfalls = sampling.n_per_source_indices(SOURCES, 5, FakeRng(0))

    assert [(s.source, s.got, s.wanted) for s in shortfalls] == [("books", 2, 5)]
    assert len(picked) == 5 + 5 + 2


def test_no_shortfall_is_reported_when_every_source_can_supply():
    _, shortfalls = sampling.n_per_source_indices(SOURCES, 2, FakeRng(0))

    assert shortfalls == []


def test_n_per_source_is_seed_stable():
    first, _ = sampling.n_per_source_indices(SOURCES, 3, FakeRng(9))
    second, _ = sampling.n_per_source_indices(SOURCES, 3, FakeRng(9))

    assert first == second


@given(source_labels=labels, n_target=st.integers(min_value=0, max_value=40))
@settings(max_examples=50, deadline=None)
def test_stratified_sampling_returns_distinct_indices(source_labels, n_target):
    picked = sampling.stratified_indices(source_labels, n_target, FakeRng(4))

    assert len(set(picked)) == len(picked)


@given(source_labels=labels, n_target=st.integers(min_value=0, max_value=40))
@settings(max_examples=50, deadline=None)
def test_stratified_sampling_gives_what_was_asked_for_or_the_whole_pool(
    source_labels, n_target
):
    picked = sampling.stratified_indices(source_labels, n_target, FakeRng(4))

    assert len(picked) == min(n_target, len(source_labels))


def test_stratified_sampling_spreads_across_sources_before_repeating_one():
    picked = sampling.stratified_indices(SOURCES, 3, FakeRng(0))

    assert {SOURCES[i] for i in picked} == {"c4", "github", "books"}


def test_sampling_rejects_a_negative_request():
    with pytest.raises(ValueError, match="n_per_source must be >= 0"):
        sampling.n_per_source_indices(SOURCES, -1, FakeRng(0))

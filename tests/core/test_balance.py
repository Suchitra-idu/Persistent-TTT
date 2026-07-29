"""Balancing invariants: never invent rows, never lose the interleaving."""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ttt.core import balance

SLIM_RESEARCH = {
    "RedPajamaC4": 25,
    "RedPajamaGithub": 20,
    "RedPajamaBook": 15,
    "RedPajamaArXiv": 20,
    "RedPajamaWikipedia": 10,
    "RedPajamaStackExchange": 10,
}

labels = st.lists(
    st.sampled_from(["c4", "github", "books"]), min_size=1, max_size=120
)
weights = st.dictionaries(
    keys=st.sampled_from(["c4", "github", "books"]),
    values=st.integers(min_value=1, max_value=100),
    min_size=1,
)


@given(source_labels=labels, w=weights, target=st.integers(0, 200))
@settings(max_examples=60, deadline=None)
def test_balancing_never_returns_more_rows_than_the_pool_holds(
    source_labels, w, target
):
    result = _balance_or_skip(source_labels, w, target)

    assert result.total <= len(source_labels)


@given(source_labels=labels, w=weights, target=st.integers(0, 200))
@settings(max_examples=60, deadline=None)
def test_balanced_indices_are_distinct_and_in_range(source_labels, w, target):
    result = _balance_or_skip(source_labels, w, target)

    assert len(set(result.indices)) == len(result.indices)
    assert all(0 <= i < len(source_labels) for i in result.indices)


@given(source_labels=labels, w=weights, target=st.integers(0, 200))
@settings(max_examples=60, deadline=None)
def test_balancing_preserves_pool_order(source_labels, w, target):
    """Interleaving is the point: a sorted result keeps domains mixed."""
    result = _balance_or_skip(source_labels, w, target)

    assert list(result.indices) == sorted(result.indices)


@given(source_labels=labels, w=weights, target=st.integers(0, 200))
@settings(max_examples=60, deadline=None)
def test_sources_outside_the_preset_are_dropped_entirely(source_labels, w, target):
    result = _balance_or_skip(source_labels, w, target)

    assert all(source_labels[i] in w for i in result.indices)


@given(source_labels=labels, w=weights, target=st.integers(0, 200))
@settings(max_examples=60, deadline=None)
def test_no_source_exceeds_its_quota(source_labels, w, target):
    result = _balance_or_skip(source_labels, w, target)

    for quota in result.quotas:
        assert quota.took <= quota.target
        assert quota.took <= quota.available


def test_a_source_short_of_its_target_takes_everything_it_has():
    pool = ["c4"] * 100 + ["books"] * 3

    result = balance.balanced_indices(pool, {"c4": 50, "books": 50}, 100)

    books = next(q for q in result.quotas if q.source == "books")
    assert (books.took, books.target, books.available) == (3, 50, 3)
    assert books.is_short


def test_ratios_are_honoured_when_every_source_has_enough():
    pool = (["c4"] * 200) + (["github"] * 200)

    result = balance.balanced_indices(pool, {"c4": 75, "github": 25}, 100)

    took = {q.source: q.took for q in result.quotas}
    assert took == {"c4": 75, "github": 25}


def test_weights_are_unnormalised_so_only_ratios_matter():
    pool = (["c4"] * 200) + (["github"] * 200)

    tens = balance.balanced_indices(pool, {"c4": 30, "github": 10}, 100)
    thousands = balance.balanced_indices(pool, {"c4": 3000, "github": 1000}, 100)

    assert tens.indices == thousands.indices


def test_the_research_preset_downweights_web_text():
    pool = (["RedPajamaC4"] * 1000) + (["RedPajamaGithub"] * 1000)

    result = balance.balanced_indices(pool, SLIM_RESEARCH, 90)

    took = {q.source: q.took for q in result.quotas}
    assert took["RedPajamaC4"] == 50 and took["RedPajamaGithub"] == 40


def test_balancing_fails_loudly_when_no_preset_source_is_in_the_pool():
    with pytest.raises(ValueError, match="none of the preset's sources"):
        balance.balanced_indices(["c4"] * 10, {"books": 1}, 5)


def test_balancing_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        balance.balanced_indices(["c4"] * 10, {"c4": -1}, 5)


def test_source_counts_report_the_raw_mix():
    assert balance.source_counts(["c4", "books", "c4"]) == {"c4": 2, "books": 1}


def _balance_or_skip(source_labels, w, target):
    """Property-test helper: discard inputs where no preset source is present.

    `assume`, not `pytest.skip` — a skip inside a hypothesis test aborts the
    whole property at the first bad example rather than discarding that one
    example, which would leave these five properties silently unrun. That
    empty-intersection case has its own example test above.
    """
    assume(any(s in w for s in source_labels))
    return balance.balanced_indices(source_labels, w, target)
